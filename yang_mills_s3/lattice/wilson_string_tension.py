"""
Wilson loop measurement and string tension extraction on the 600-cell.

Computes Wilson loops W(R,T) for various sizes and extracts the
string tension sigma from area-law decay:

    <W(R,T)> ~ exp(-sigma * R * T)

On the 600-cell (120 vertices, 720 edges, 1200 triangular faces):
  - The lattice is COARSE: only 120 sites with valence 12
  - "Rectangular" loops are approximated by graph paths of specified
    spatial and temporal extent
  - Quantitative sigma extraction is limited by lattice coarseness
  - Qualitative area law is the primary target

STATUS: NUMERICAL
Honest caveats:
  - 600-cell has lattice spacing a ~ 0.63 (unit S3), very coarse
  - Only ~120 vertices, so R,T are limited to a few lattice units
  - Finite-volume effects dominate at this lattice size
  - String tension values are QUALITATIVE, not competitive with lattice QCD
  - Short MC runs (100-1000 configs) for proof-of-concept
"""

import numpy as np
from collections import defaultdict
from .s3_lattice import S3Lattice
from .mc_engine import MCEngine


class WilsonStringTension:
    """
    Wilson loop measurement and string tension extraction on the 600-cell.

    Strategy for "rectangular" W(R,T) on a triangular lattice embedded in S3:
      1. Pick a "time" direction via one of the 4D coordinates (w-axis)
      2. Classify edges as "temporal" (mostly along w) or "spatial" (mostly
         transverse to w)
      3. Build rectangular loops as paths with R spatial hops and T temporal
         hops
      4. Measure <W(R,T)> over MC configurations and extract sigma

    Since the 600-cell is highly symmetric, all directions are equivalent --
    the "time" choice is a gauge choice for the measurement.
    """

    def __init__(self, lattice, beta=4.0, rng=None):
        """
        Parameters
        ----------
        lattice : S3Lattice
            The 600-cell lattice.
        beta : float
            Coupling beta = 4/g^2 for SU(2).
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        self.lattice = lattice
        self.beta = beta
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.engine = MCEngine(lattice, beta=beta, rng=self.rng)

        # Precompute geometric structures
        self._verts = lattice.vertices / lattice.R  # unit sphere
        self._adj = self._build_adjacency()
        self._n_verts = lattice.vertex_count()
        self._lattice_spacing = lattice.lattice_spacing()

        # Classify edges by direction
        self._temporal_neighbors, self._spatial_neighbors = \
            self._classify_edges()

    def _build_adjacency(self):
        """Build adjacency dict from edges."""
        adj = defaultdict(set)
        for (i, j) in self.lattice.edges():
            adj[i].add(j)
            adj[j].add(i)
        return dict(adj)

    def _classify_edges(self, time_coord=0):
        """
        Classify each edge as 'temporal' or 'spatial' based on the
        change in the time coordinate.

        An edge (i,j) is temporal if |dw| > |dw_perp|, where w is the
        time coordinate and dw_perp is the transverse displacement.

        Every vertex is guaranteed at least one temporal and one spatial
        neighbor: if a vertex has none in one category, its neighbor
        with the largest projection in that direction is reassigned.

        Returns
        -------
        temporal_neighbors : dict {vertex: set of temporal neighbors}
        spatial_neighbors : dict {vertex: set of spatial neighbors}
        """
        temporal = defaultdict(set)
        spatial = defaultdict(set)

        # Score each edge by its temporal component
        edge_scores = {}
        for (i, j) in self.lattice.edges():
            dv = self._verts[j] - self._verts[i]
            dw = abs(dv[time_coord])
            dw_perp = np.sqrt(sum(dv[k]**2 for k in range(4)
                                  if k != time_coord))
            edge_scores[(i, j)] = (dw, dw_perp)

            if dw >= dw_perp:
                temporal[i].add(j)
                temporal[j].add(i)
            else:
                spatial[i].add(j)
                spatial[j].add(i)

        # Ensure every vertex has at least one temporal and one spatial
        # neighbor. If not, reassign the best candidate edge.
        for v in range(self._n_verts):
            if v not in temporal or len(temporal[v]) == 0:
                # Find the neighbor with the largest temporal component
                best_nb = None
                best_dw = -1.0
                for nb in self._adj.get(v, set()):
                    edge = (min(v, nb), max(v, nb))
                    dw, _ = edge_scores.get(edge, (0, 0))
                    if dw > best_dw:
                        best_dw = dw
                        best_nb = nb
                if best_nb is not None:
                    temporal[v].add(best_nb)
                    temporal[best_nb].add(v)
                    # Remove from spatial if it was there
                    spatial[v].discard(best_nb)
                    spatial[best_nb].discard(v)

            if v not in spatial or len(spatial[v]) == 0:
                # Find the neighbor with the largest spatial component
                best_nb = None
                best_ds = -1.0
                for nb in self._adj.get(v, set()):
                    edge = (min(v, nb), max(v, nb))
                    _, ds = edge_scores.get(edge, (0, 0))
                    if ds > best_ds:
                        best_ds = ds
                        best_nb = nb
                if best_nb is not None:
                    spatial[v].add(best_nb)
                    spatial[best_nb].add(v)
                    # Remove from temporal if it was there
                    temporal[v].discard(best_nb)
                    temporal[best_nb].discard(v)

        return dict(temporal), dict(spatial)

    # ==================================================================
    # Loop construction
    # ==================================================================

    def find_rectangular_loops(self, R_max=3, T_max=3, max_per_size=30):
        """
        Find approximately rectangular Wilson loops W(R,T) on the graph.

        A "rectangular" loop of size (R,T) consists of:
          - R spatial hops in one direction
          - T temporal hops
          - R spatial hops back
          - T temporal hops back

        On the 600-cell's graph, these are approximate -- the paths
        are not perfectly rectangular, but they have the correct
        spatial and temporal extent.

        Parameters
        ----------
        R_max : int
            Maximum spatial extent in lattice hops.
        T_max : int
            Maximum temporal extent in lattice hops.
        max_per_size : int
            Maximum number of loops per (R,T) size.

        Returns
        -------
        dict : {(R,T): list of vertex paths}
        """
        loops = {}

        for R in range(1, R_max + 1):
            for T in range(1, T_max + 1):
                found = self._find_RT_loops(R, T, max_per_size)
                if found:
                    loops[(R, T)] = found

        return loops

    def _find_RT_loops(self, R, T, max_count):
        """
        Find loops with R spatial hops and T temporal hops.

        Strategy: BFS/DFS to build paths
          start -> R spatial hops -> T temporal hops ->
          R spatial hops (back) -> T temporal hops (back) -> start

        Returns list of vertex paths (closed loops).
        """
        found = []
        n_attempts = min(self._n_verts, 40)

        for start in range(n_attempts):
            if len(found) >= max_count:
                break

            # Phase 1: R spatial hops from start
            spatial_ends = self._walk_direction(
                start, R, self._spatial_neighbors, 'positive')

            for corner1 in spatial_ends:
                if len(found) >= max_count:
                    break

                path_spatial = corner1['path']

                # Phase 2: T temporal hops from corner1
                temporal_ends = self._walk_direction(
                    corner1['vertex'], T, self._temporal_neighbors, 'positive')

                for corner2 in temporal_ends:
                    if len(found) >= max_count:
                        break

                    path_temporal = corner2['path']

                    # Phase 3: R spatial hops back (try to get closer to
                    # the temporal column of start)
                    return_spatial = self._walk_direction(
                        corner2['vertex'], R, self._spatial_neighbors,
                        'negative')

                    for corner3 in return_spatial:
                        if len(found) >= max_count:
                            break

                        path_return_spatial = corner3['path']

                        # Phase 4: T temporal hops back to start
                        path_return_temporal = self._walk_toward(
                            corner3['vertex'], start, T,
                            self._temporal_neighbors)

                        if path_return_temporal is not None:
                            # Build full loop
                            full_path = (
                                path_spatial[:-1] +
                                path_temporal[:-1] +
                                path_return_spatial[:-1] +
                                path_return_temporal[:-1]
                            )

                            # Check: path is a valid closed loop
                            if self._is_valid_closed_loop(full_path):
                                canon = self._canonicalize(full_path)
                                if canon not in [
                                    self._canonicalize(f) for f in found
                                ]:
                                    found.append(full_path)

        return found

    def _walk_direction(self, start, n_steps, neighbor_dict, direction):
        """
        Walk n_steps from start using the given neighbor set,
        preferring edges that go in the specified direction
        (+w for 'positive', -w for 'negative', or maximum spatial
        displacement for 'spatial').

        Returns list of {vertex, path} dicts.
        """
        if n_steps == 0:
            return [{'vertex': start, 'path': [start]}]

        results = []
        self._walk_dfs(start, [start], n_steps, neighbor_dict,
                       direction, results, max_results=5)
        return results

    def _walk_dfs(self, current, path, remaining, neighbor_dict,
                  direction, results, max_results):
        """DFS helper for directional walking."""
        if len(results) >= max_results:
            return

        if remaining == 0:
            results.append({'vertex': current, 'path': list(path)})
            return

        neighbors = neighbor_dict.get(current, set())
        if not neighbors:
            return

        # Sort neighbors by preference
        scored = []
        for nb in neighbors:
            if nb in path:
                continue
            dv = self._verts[nb] - self._verts[current]
            if direction == 'positive':
                score = dv[0]  # prefer increasing w
            elif direction == 'negative':
                score = -dv[0]  # prefer decreasing w
            else:
                # For spatial: prefer maximum transverse displacement
                score = np.sqrt(dv[1]**2 + dv[2]**2 + dv[3]**2)
            scored.append((score, nb))

        scored.sort(reverse=True)

        for _, nb in scored[:3]:  # try top 3 candidates
            path.append(nb)
            self._walk_dfs(nb, path, remaining - 1, neighbor_dict,
                           direction, results, max_results)
            path.pop()

    def _walk_toward(self, start, target, n_steps, neighbor_dict):
        """
        Walk n_steps from start trying to reach target using the
        given neighbor set.

        Returns a path [start, ..., target] or None if impossible.
        """
        if n_steps == 0:
            return [start] if start == target else None

        # BFS with depth limit
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if len(path) > n_steps + 1:
                continue

            if len(path) == n_steps + 1:
                if current == target:
                    return path
                continue

            for nb in neighbor_dict.get(current, set()):
                if nb in visited and nb != target:
                    continue
                if nb == target and len(path) == n_steps:
                    return path + [nb]
                if nb not in visited and len(path) < n_steps:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))

        return None

    def _is_valid_closed_loop(self, path):
        """Check that a path forms a valid closed loop on the graph."""
        if len(path) < 3:
            return False

        n = len(path)
        for step in range(n):
            i = path[step]
            j = path[(step + 1) % n]
            if j not in self._adj.get(i, set()):
                return False

        # Check no repeated vertices (simple loop)
        if len(set(path)) != len(path):
            return False

        return True

    def _canonicalize(self, path):
        """Canonical form: rotate to start at minimum vertex."""
        if not path:
            return tuple()
        min_idx = path.index(min(path))
        rotated = path[min_idx:] + path[:min_idx]
        return tuple(rotated)

    # ==================================================================
    # Alternative: general loops by perimeter length with area estimate
    # ==================================================================

    def find_loops_with_area(self, max_length=8, max_per_length=40):
        """
        Find loops of various perimeter lengths and estimate their
        enclosed area on S3.

        For the 600-cell, the minimal plaquette (triangle) has a
        well-defined area. Larger loops enclose more area.

        Returns
        -------
        dict : {length: list of (path, estimated_area)}
        """
        loops_dict = self.engine.find_loops_by_length(
            max_length=max_length, max_per_length=max_per_length)

        result = {}
        a = self._lattice_spacing  # edge length

        for length, paths in loops_dict.items():
            result[length] = []
            for path in paths:
                area = self._estimate_loop_area(path, a)
                result[length].append((path, area))

        return result

    def _estimate_loop_area(self, path, a):
        """
        Estimate the area enclosed by a loop on S3.

        For a loop of L edges with edge length a, the enclosed area
        of a regular polygon is approximately:
          A = (L * a^2 / 4) * cot(pi/L)

        For triangular plaquettes (L=3): A = a^2 * sqrt(3)/4
        This is exact for equilateral triangles.

        On curved S3, this is approximate; corrections are O(a^2/R^2).
        """
        L = len(path)
        if L < 3:
            return 0.0

        # Solid angle subtended, computed from vertex positions
        # For small loops, planar area approximation works
        if L == 3:
            # Exact for equilateral triangle
            return a**2 * np.sqrt(3) / 4

        # General polygon: sum of triangle areas using fan decomposition
        # Use the centroid as the fan center
        center = np.mean(self._verts[path], axis=0)
        total_area = 0.0

        for k in range(L):
            v1 = self._verts[path[k]]
            v2 = self._verts[path[(k + 1) % L]]
            # Triangle: center, v1, v2
            # Area = 0.5 * |cross product| (in R4, use the generalized
            # cross product = sqrt(det(G)) where G is the Gram matrix)
            e1 = v1 - center
            e2 = v2 - center
            # In R4, area = sqrt(|e1|^2 |e2|^2 - (e1.e2)^2) / 2
            dot11 = np.dot(e1, e1)
            dot22 = np.dot(e2, e2)
            dot12 = np.dot(e1, e2)
            area_sq = dot11 * dot22 - dot12**2
            if area_sq > 0:
                total_area += 0.5 * np.sqrt(area_sq)

        return total_area

    # ==================================================================
    # MC simulation and measurement
    # ==================================================================

    def thermalize(self, n_therm=20, method='heatbath'):
        """
        Thermalize the gauge field configuration.

        Parameters
        ----------
        n_therm : int
            Number of thermalization sweeps.
        method : str
            'heatbath', 'metropolis', or 'compound'.
        """
        self.engine.set_hot_start()
        for _ in range(n_therm):
            if method == 'heatbath':
                self.engine.heatbath_sweep()
            elif method == 'metropolis':
                self.engine.metropolis_sweep(epsilon=0.3)
            elif method == 'compound':
                self.engine.compound_sweep(n_heatbath=1, n_overrelax=3)

    def measure_wilson_loops_mc(self, loops, n_configs=100, n_skip=2,
                                method='heatbath'):
        """
        Measure Wilson loop expectation values over MC configurations.

        Parameters
        ----------
        loops : dict
            Either {(R,T): [paths]} or {length: [paths]} or
            {length: [(path, area)]}
        n_configs : int
            Number of configurations to measure.
        n_skip : int
            Sweeps between measurements for decorrelation.
        method : str
            Update method: 'heatbath', 'metropolis', 'compound'.

        Returns
        -------
        dict : keyed by loop label, values are measurement arrays
        """
        # Flatten loops into a list of (label, path) pairs
        loop_list = []
        for key, items in loops.items():
            for item in items:
                if isinstance(item, tuple) and len(item) == 2:
                    path, area = item
                    loop_list.append((key, path, area))
                else:
                    path = item
                    area = self._estimate_loop_area(
                        path, self._lattice_spacing)
                    loop_list.append((key, path, area))

        # Group loop_list by key for per-config averaging
        loops_by_key = defaultdict(list)
        for key, path, area in loop_list:
            loops_by_key[key].append((path, area))

        # Storage: one mean value per config per key
        measurements = {key: [] for key in loops_by_key}

        for config in range(n_configs):
            # Update configuration
            for _ in range(n_skip):
                if method == 'heatbath':
                    self.engine.heatbath_sweep()
                elif method == 'metropolis':
                    self.engine.metropolis_sweep(epsilon=0.3)
                elif method == 'compound':
                    self.engine.compound_sweep(n_heatbath=1, n_overrelax=3)

            # Measure all loops, average over loops within each key
            for key, items in loops_by_key.items():
                vals = []
                for path, area in items:
                    w = np.real(self.engine.wilson_loop_path(path))
                    vals.append(w)
                # Store the mean over all loops of this type for this config
                measurements[key].append(float(np.mean(vals)))

        # Convert to arrays (one entry per config)
        return {key: np.array(vals) for key, vals in measurements.items()}

    # ==================================================================
    # String tension extraction
    # ==================================================================

    def extract_string_tension_from_area(self, measurements, loops):
        """
        Extract string tension from Wilson loop measurements using
        the area law: <W(C)> ~ exp(-sigma * Area(C)).

        Fits -ln<W> vs Area for each loop category.

        Parameters
        ----------
        measurements : dict from measure_wilson_loops_mc
        loops : dict with area information

        Returns
        -------
        dict with string tension estimates and diagnostics
        """
        # Collect (area, -ln<W>) pairs
        data_points = []

        for key, items in loops.items():
            vals = measurements.get(key, None)
            if vals is None:
                continue

            mean_W = np.mean(vals)
            std_W = np.std(vals) / np.sqrt(len(vals))

            if mean_W > 1e-10:
                # Average area for this loop category
                areas = []
                for item in items:
                    if isinstance(item, tuple):
                        _, area = item
                        areas.append(area)
                    else:
                        areas.append(self._estimate_loop_area(
                            item, self._lattice_spacing))

                avg_area = np.mean(areas)
                neg_ln_W = -np.log(mean_W)

                # Error propagation: d(-ln W) = dW / W
                if mean_W > 0:
                    err = std_W / mean_W
                else:
                    err = np.inf

                data_points.append({
                    'label': key,
                    'area': avg_area,
                    'neg_ln_W': neg_ln_W,
                    'mean_W': mean_W,
                    'std_W': std_W,
                    'err_neg_ln_W': err,
                    'n_loops': len(items),
                })

        if len(data_points) < 2:
            return {
                'sigma': 0.0,
                'sigma_err': np.inf,
                'data_points': data_points,
                'fit_quality': 'insufficient_data',
            }

        # Sort by area
        data_points.sort(key=lambda d: d['area'])

        # Linear fit: -ln<W> = sigma * Area + c
        areas = np.array([d['area'] for d in data_points])
        neg_ln_W = np.array([d['neg_ln_W'] for d in data_points])
        weights = np.array([1.0 / max(d['err_neg_ln_W'], 0.01)
                            for d in data_points])

        # Weighted least squares
        if len(areas) >= 2:
            W_mat = np.diag(weights)
            A_mat = np.column_stack([areas, np.ones(len(areas))])
            try:
                AtwA = A_mat.T @ W_mat @ A_mat
                Atwy = A_mat.T @ W_mat @ neg_ln_W
                params = np.linalg.solve(AtwA, Atwy)
                sigma = params[0]
                intercept = params[1]

                # Residuals for fit quality
                pred = sigma * areas + intercept
                residuals = neg_ln_W - pred
                chi2 = np.sum(weights * residuals**2) / max(len(areas) - 2, 1)

                # Error on sigma from covariance
                try:
                    cov = np.linalg.inv(AtwA)
                    sigma_err = np.sqrt(abs(cov[0, 0]))
                except np.linalg.LinAlgError:
                    sigma_err = np.inf
            except np.linalg.LinAlgError:
                sigma = 0.0
                intercept = 0.0
                chi2 = np.inf
                sigma_err = np.inf
        else:
            sigma = 0.0
            intercept = 0.0
            chi2 = np.inf
            sigma_err = np.inf

        return {
            'sigma': float(sigma),
            'sigma_err': float(sigma_err),
            'intercept': float(intercept),
            'chi2_per_dof': float(chi2),
            'data_points': data_points,
            'fit_quality': 'good' if chi2 < 5 else 'poor',
        }

    def extract_creutz_ratios(self, measurements, loops):
        """
        Extract Creutz ratios from rectangular Wilson loops.

        The Creutz ratio for rectangular R x T loops:
          chi(R,T) = -ln[W(R,T) W(R-1,T-1) / (W(R-1,T) W(R,T-1))]

        converges to the string tension sigma * a^2 for large R,T.

        Parameters
        ----------
        measurements : dict {(R,T): array of W values}
        loops : dict {(R,T): list of paths}

        Returns
        -------
        dict with Creutz ratio estimates
        """
        # Compute <W(R,T)> for each (R,T)
        W_mean = {}
        for key, vals in measurements.items():
            if isinstance(key, tuple) and len(key) == 2:
                W_mean[key] = np.mean(vals)

        creutz = []
        for (R, T), W_RT in W_mean.items():
            if R < 2 or T < 2:
                continue
            W_Rm1_T = W_mean.get((R - 1, T))
            W_R_Tm1 = W_mean.get((R, T - 1))
            W_Rm1_Tm1 = W_mean.get((R - 1, T - 1))

            if (W_Rm1_T is not None and W_R_Tm1 is not None and
                W_Rm1_Tm1 is not None and
                all(w > 1e-15 for w in [W_RT, W_Rm1_T, W_R_Tm1,
                                         W_Rm1_Tm1])):

                ratio = (W_RT * W_Rm1_Tm1) / (W_Rm1_T * W_R_Tm1)
                if ratio > 0:
                    chi = -np.log(ratio)
                    # sigma * a^2 = chi
                    a = self._lattice_spacing
                    sigma_lat = chi / a**2

                    creutz.append({
                        'R': R, 'T': T,
                        'chi': float(chi),
                        'sigma_lat': float(sigma_lat),
                        'W_values': {
                            'W_RT': float(W_RT),
                            'W_Rm1_T': float(W_Rm1_T),
                            'W_R_Tm1': float(W_R_Tm1),
                            'W_Rm1_Tm1': float(W_Rm1_Tm1),
                        },
                    })

        return {
            'creutz_ratios': creutz,
            'sigma_mean': float(np.mean([c['sigma_lat'] for c in creutz]))
                          if creutz else 0.0,
        }

    # ==================================================================
    # Physical string tension conversion
    # ==================================================================

    @staticmethod
    def sigma_to_physical(sigma_lattice, a_fm):
        """
        Convert lattice string tension to physical units.

        Parameters
        ----------
        sigma_lattice : float
            String tension in lattice units (1/a^2).
        a_fm : float
            Lattice spacing in fm.

        Returns
        -------
        dict with sqrt_sigma_MeV and sigma_GeV2
        """
        hbar_c = 0.197327  # GeV * fm

        sigma_phys = sigma_lattice / a_fm**2  # fm^{-2}
        sigma_GeV2 = sigma_phys * hbar_c**2   # GeV^2

        sqrt_sigma_MeV = np.sqrt(abs(sigma_GeV2)) * 1000  # MeV

        return {
            'sigma_lattice': sigma_lattice,
            'sigma_phys_fm2': sigma_phys,
            'sigma_GeV2': sigma_GeV2,
            'sqrt_sigma_MeV': sqrt_sigma_MeV,
            'known_value_MeV': 440.0,
            'ratio_to_known': sqrt_sigma_MeV / 440.0,
        }

    # ==================================================================
    # Full analysis pipeline
    # ==================================================================

    def run_analysis(self, n_therm=20, n_configs=100, n_skip=2,
                     R_max=3, T_max=3, max_length=6,
                     method='heatbath', verbose=False):
        """
        Full Wilson loop analysis pipeline.

        Steps:
          1. Thermalize
          2. Find rectangular loops W(R,T)
          3. Find general loops by perimeter length
          4. Measure over MC configurations
          5. Extract string tension from both methods
          6. Convert to physical units

        Parameters
        ----------
        n_therm : int
            Thermalization sweeps.
        n_configs : int
            Measurement configurations.
        n_skip : int
            Sweeps between measurements.
        R_max, T_max : int
            Max spatial/temporal extent for rectangular loops.
        max_length : int
            Max perimeter for general loops.
        method : str
            MC update method.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with complete analysis results
        """
        if verbose:
            print(f"Wilson loop analysis at beta={self.beta}")
            print(f"  Lattice: 600-cell, {self._n_verts} vertices, "
                  f"a={self._lattice_spacing:.4f}")

        # Step 1: Thermalize
        if verbose:
            print(f"  Thermalizing ({n_therm} sweeps)...")
        self.thermalize(n_therm=n_therm, method=method)

        plaq_therm = self.engine.plaquette_average()
        if verbose:
            print(f"  <P> after therm = {plaq_therm:.4f}")

        # Step 2: Find rectangular loops
        if verbose:
            print(f"  Finding rectangular loops (R<={R_max}, T<={T_max})...")
        rect_loops = self.find_rectangular_loops(
            R_max=R_max, T_max=T_max, max_per_size=20)
        if verbose:
            for key, paths in rect_loops.items():
                print(f"    W{key}: {len(paths)} loops found")

        # Step 3: Find general loops by perimeter
        if verbose:
            print(f"  Finding general loops (L<={max_length})...")
        general_loops = self.find_loops_with_area(
            max_length=max_length, max_per_length=30)
        if verbose:
            for length, items in general_loops.items():
                print(f"    L={length}: {len(items)} loops")

        # Step 4: Measure over MC
        if verbose:
            print(f"  Measuring ({n_configs} configs, "
                  f"skip={n_skip})...")

        # Measure rectangular loops
        rect_meas = {}
        if rect_loops:
            rect_meas = self.measure_wilson_loops_mc(
                rect_loops, n_configs=n_configs, n_skip=n_skip,
                method=method)

        # Measure general loops
        gen_meas = self.measure_wilson_loops_mc(
            general_loops, n_configs=n_configs, n_skip=n_skip,
            method=method)

        # Step 5: Extract string tension
        # Method A: from area law fit to general loops
        if verbose:
            print("  Extracting string tension...")
        area_result = self.extract_string_tension_from_area(
            gen_meas, general_loops)

        # Method B: Creutz ratios from rectangular loops
        creutz_result = self.extract_creutz_ratios(rect_meas, rect_loops)

        # Step 6: Physical units
        # The 600-cell on S3(R=2.2 fm) has lattice spacing:
        #   a = R * edge_length_on_unit_S3
        R_phys = self.lattice.R  # fm if R was set in fm
        a_phys = self._lattice_spacing  # in same units as R
        # For unit S3 (R=1): a ~ 0.63
        # For R=2.2 fm: a ~ 1.39 fm

        # Use the area-law sigma
        sigma_lat = area_result.get('sigma', 0.0)
        if sigma_lat > 0 and a_phys > 0:
            physical = self.sigma_to_physical(sigma_lat, a_phys)
        else:
            physical = None

        # Compile Wilson loop summary: <W> by size
        wl_summary = {}
        for key, vals in gen_meas.items():
            wl_summary[key] = {
                'mean_W': float(np.mean(vals)),
                'std_W': float(np.std(vals)),
                'n_configs': len(vals),
            }
        for key, vals in rect_meas.items():
            wl_summary[f'rect_{key}'] = {
                'mean_W': float(np.mean(vals)),
                'std_W': float(np.std(vals)),
                'n_configs': len(vals),
            }

        # Check area law qualitatively
        lengths = sorted([k for k in wl_summary if isinstance(k, int)])
        area_law_check = 'inconclusive'
        if len(lengths) >= 3:
            means = [wl_summary[l]['mean_W'] for l in lengths]
            if all(means[i] > means[i + 1] for i in range(len(means) - 1)):
                area_law_check = 'qualitative_area_law'
            elif means[0] > means[-1]:
                area_law_check = 'partial_decay'
            else:
                area_law_check = 'no_area_law'

        result = {
            'beta': self.beta,
            'lattice_spacing': float(self._lattice_spacing),
            'n_vertices': self._n_verts,
            'plaquette': float(plaq_therm),
            'wilson_loops': wl_summary,
            'area_law_sigma': area_result,
            'creutz_ratios': creutz_result,
            'physical_sigma': physical,
            'area_law_check': area_law_check,
            'n_configs': n_configs,
            'caveats': [
                'NUMERICAL: 600-cell has only 120 vertices (very coarse)',
                f'Lattice spacing a = {self._lattice_spacing:.3f} '
                f'(unit S3)',
                f'Short MC run: {n_configs} configs',
                'String tension is QUALITATIVE, not competitive with '
                'lattice QCD',
                'Finite-volume effects dominate at this lattice size',
            ],
        }

        if verbose:
            print("\n=== Results ===")
            print(f"  Plaquette: {plaq_therm:.4f}")
            print(f"  Area law check: {area_law_check}")
            if area_result.get('sigma', 0) > 0:
                print(f"  sigma (area fit): {area_result['sigma']:.4f} "
                      f"(lattice units)")
            if creutz_result.get('sigma_mean', 0) > 0:
                print(f"  sigma (Creutz): {creutz_result['sigma_mean']:.4f}")
            if physical:
                print(f"  sqrt(sigma): {physical['sqrt_sigma_MeV']:.0f} MeV "
                      f"(known: 440 MeV)")
                print(f"  Ratio to known: "
                      f"{physical['ratio_to_known']:.2f}")

        return result


def run_multi_beta_analysis(betas=None, n_therm=20, n_configs=100,
                            n_skip=2, R_fm=1.0, verbose=True):
    """
    Run Wilson loop analysis at multiple beta values.

    This is the main entry point for the string tension study.

    Parameters
    ----------
    betas : list of float
        Beta values to scan. Default: [0.637, 2.0, 4.0, 8.0]
    n_therm : int
        Thermalization sweeps per beta.
    n_configs : int
        Configs per beta.
    n_skip : int
        Sweeps between measurements.
    R_fm : float
        S3 radius in fm (for physical unit conversion).
    verbose : bool
        Print progress.

    Returns
    -------
    dict : {beta: analysis_result}
    """
    if betas is None:
        # Physical coupling: beta = 4/g^2 = 4/6.28 ~ 0.637 for SU(2)
        # Plus weak-coupling comparison points
        betas = [0.637, 2.0, 4.0, 8.0]

    lattice = S3Lattice(R=R_fm)
    results = {}

    for beta in betas:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Beta = {beta:.3f} (g^2 = {4.0/beta:.3f})")
            print(f"{'='*60}")

        wst = WilsonStringTension(
            lattice, beta=beta,
            rng=np.random.default_rng(42 + int(beta * 1000)))

        result = wst.run_analysis(
            n_therm=n_therm,
            n_configs=n_configs,
            n_skip=n_skip,
            R_max=2,  # keep small for speed on coarse lattice
            T_max=2,
            max_length=6,
            method='heatbath',
            verbose=verbose,
        )

        results[beta] = result

    # Summary comparison
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY: Wilson loops and string tension vs beta")
        print(f"{'='*60}")
        print(f"{'beta':>8} {'<P>':>8} {'W(3)':>8} {'W(4)':>8} "
              f"{'W(5)':>8} {'sigma':>10} {'area_law':>20}")
        print("-" * 80)
        for beta in sorted(results.keys()):
            r = results[beta]
            wl = r['wilson_loops']
            P = r['plaquette']
            W3 = wl.get(3, {}).get('mean_W', np.nan)
            W4 = wl.get(4, {}).get('mean_W', np.nan)
            W5 = wl.get(5, {}).get('mean_W', np.nan)
            sig = r['area_law_sigma'].get('sigma', 0.0)
            area = r['area_law_check']
            print(f"{beta:8.3f} {P:8.4f} {W3:8.4f} {W4:8.4f} "
                  f"{W5:8.4f} {sig:10.4f} {area:>20}")

    return results
