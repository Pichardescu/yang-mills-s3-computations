"""
Optimized Monte Carlo engine for SU(2) Yang-Mills on the 600-cell.

Key optimizations over the base LatticeYM:
  1. Precomputed link-to-plaquette map for O(1) staple computation
  2. SU(2) Cabibbo-Marinari heat bath (exact sampling, no accept/reject)
  3. Overrelaxation sweeps for faster decorrelation
  4. Vectorized observable measurements

Wilson action: S = beta * Sum_plaq (1 - (1/2) Re Tr U_plaq)
  where U_plaq = U_{ij} U_{jk} U_{ki} for triangular plaquettes.

On the 600-cell (120 vertices, 720 edges, 1200 triangular faces):
  - Each link participates in exactly 10 plaquettes
  - Valence = 12 (each vertex has 12 neighbors)
  - Triangular plaquettes (not square like hypercubic lattice)

STATUS: NUMERICAL
"""

import numpy as np
from .s3_lattice import S3Lattice


class MCEngine:
    """
    Fast Monte Carlo engine for SU(2) lattice gauge theory on S3 (600-cell).

    Uses SU(2)-specific heat bath algorithm (Kennedy-Pendleton, 1985)
    and microcanonical overrelaxation for efficient sampling.
    """

    def __init__(self, lattice, beta=4.0, rng=None):
        """
        Parameters
        ----------
        lattice : S3Lattice
            The 600-cell discretization of S3.
        beta : float
            Coupling: beta = 4/g^2 for SU(2).
        rng : numpy.random.Generator, optional
            Random number generator. Created from seed 42 if not given.
        """
        self.lattice = lattice
        self.beta = beta
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Lattice data
        self._edges = lattice.edges()
        self._faces = lattice.faces()
        self._n_links = len(self._edges)
        self._n_plaq = len(self._faces)

        # Edge index map: (i,j) -> link index
        self._edge_idx = {}
        for idx, (i, j) in enumerate(self._edges):
            self._edge_idx[(i, j)] = idx
            self._edge_idx[(j, i)] = idx

        # Link variables: array of shape (n_links, 2, 2), complex
        self._links = np.zeros((self._n_links, 2, 2), dtype=complex)
        self._set_identity()

        # Orientation: for each edge stored as (i,j) with i<j,
        # _edge_fwd[(a,b)] is True if (a,b) = stored order
        self._edge_fwd = {}
        for idx, (i, j) in enumerate(self._edges):
            self._edge_fwd[(i, j)] = True
            self._edge_fwd[(j, i)] = False

        # Precompute plaquette structure: for each plaquette, store
        # [(link_idx, forward), ...] triples
        self._plaq_links = self._build_plaquette_links()

        # Precompute link-to-plaquette map: for each link, which plaquettes
        # contain it and what is the "staple" (product of other links in plaq)
        self._link_to_plaqs = self._build_link_plaquette_map()

        # Adjacency for great circle / Wilson loop finding
        self._adj = self._build_adjacency()

    # ==================================================================
    # Initialization
    # ==================================================================

    def _set_identity(self):
        """Set all links to identity (trivial vacuum)."""
        for idx in range(self._n_links):
            self._links[idx] = np.eye(2, dtype=complex)

    def set_cold_start(self):
        """Cold start: all links = identity."""
        self._set_identity()

    def set_hot_start(self):
        """Hot start: all links = random SU(2)."""
        for idx in range(self._n_links):
            self._links[idx] = self._random_su2()

    # ==================================================================
    # SU(2) matrix utilities
    # ==================================================================

    def _random_su2(self):
        """
        Generate a uniformly distributed (Haar) random SU(2) matrix.

        SU(2) ~ S3, so we sample a uniform point on S3 and convert to matrix:
        U = a0*I + i*(a1*sigma1 + a2*sigma2 + a3*sigma3)
        with a0^2 + a1^2 + a2^2 + a3^2 = 1.
        """
        # Marsaglia method for uniform S3
        while True:
            x = self.rng.uniform(-1, 1, 2)
            s1 = x[0]**2 + x[1]**2
            if s1 < 1.0:
                break
        while True:
            y = self.rng.uniform(-1, 1, 2)
            s2 = y[0]**2 + y[1]**2
            if s2 < 1.0:
                break
        factor = np.sqrt((1 - s1) / s2)
        a = np.array([x[0], x[1], y[0] * factor, y[1] * factor])
        return self._quaternion_to_su2(a)

    @staticmethod
    def _quaternion_to_su2(a):
        """
        Convert quaternion (a0, a1, a2, a3) with |a|=1 to SU(2) matrix.

        U = a0*I + i*(a1*sigma1 + a2*sigma2 + a3*sigma3)
          = [[a0 + i*a3, a2 + i*a1],
             [-a2 + i*a1, a0 - i*a3]]
        """
        return np.array([
            [a[0] + 1j * a[3],  a[2] + 1j * a[1]],
            [-a[2] + 1j * a[1], a[0] - 1j * a[3]]
        ], dtype=complex)

    @staticmethod
    def _project_su2(M):
        """
        Project a 2x2 matrix to the nearest SU(2) element.

        Uses the quaternion representation: extract a0..a3, normalize.
        This is essential for preventing numerical drift.
        """
        a0 = 0.5 * np.real(M[0, 0] + M[1, 1])
        a3 = 0.5 * np.imag(M[0, 0] - M[1, 1])
        a1 = 0.5 * np.imag(M[0, 1] + M[1, 0])
        a2 = 0.5 * np.real(M[0, 1] - M[1, 0])
        norm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2)
        if norm < 1e-15 or not np.isfinite(norm):
            return np.eye(2, dtype=complex)
        a0, a1, a2, a3 = a0/norm, a1/norm, a2/norm, a3/norm
        return np.array([
            [a0 + 1j * a3,  a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3]
        ], dtype=complex)

    @staticmethod
    def _su2_to_quaternion(U):
        """
        Extract quaternion (a0, a1, a2, a3) from SU(2) matrix.
        """
        a0 = 0.5 * np.real(U[0, 0] + U[1, 1])
        a3 = 0.5 * np.imag(U[0, 0] - U[1, 1])
        a1 = 0.5 * np.imag(U[0, 1] + U[1, 0])
        a2 = 0.5 * np.real(U[0, 1] - U[1, 0])
        return np.array([a0, a1, a2, a3])

    def _su2_near_identity(self, epsilon):
        """
        Generate SU(2) matrix near identity with spread epsilon.

        Uses the fact that SU(2) ~ S3 and sampling near the north pole.
        """
        # Sample a3-vector uniformly in ball of radius epsilon
        v = self.rng.standard_normal(3) * epsilon
        norm_sq = np.dot(v, v)
        if norm_sq > 1.0:
            v = v / np.sqrt(norm_sq)
            norm_sq = 1.0
        a0 = np.sqrt(max(1.0 - norm_sq, 0.0))
        a = np.array([a0, v[0], v[1], v[2]])
        a = a / np.linalg.norm(a)  # ensure unit quaternion
        return self._quaternion_to_su2(a)

    # ==================================================================
    # Plaquette / staple precomputation
    # ==================================================================

    def _build_plaquette_links(self):
        """
        For each face (i,j,k), compute the ordered link indices and orientations.

        Returns list of [(link_idx, forward), ...] for each plaquette.
        """
        plaq_links = []
        for (i, j, k) in self._faces:
            links = []
            for (a, b) in [(i, j), (j, k), (k, i)]:
                idx = self._edge_idx[(a, b)]
                fwd = self._edge_fwd[(a, b)]
                links.append((idx, fwd))
            plaq_links.append(links)
        return plaq_links

    def _build_link_plaquette_map(self):
        """
        For each link, store which plaquettes contain it and the indices
        of the other two links (the "staple").

        Returns dict: link_idx -> list of (plaq_idx, staple_info)
        where staple_info = [(link2_idx, fwd2), (link3_idx, fwd3)]
        and the staple matrix is U2 * U3 (in the appropriate direction).
        """
        link_map = {idx: [] for idx in range(self._n_links)}

        for p_idx, plaq in enumerate(self._plaq_links):
            # plaq = [(l0, f0), (l1, f1), (l2, f2)]
            # For link l0, staple is l1 * l2
            # For link l1, staple is l2 * l0
            # For link l2, staple is l0 * l1
            for pos in range(3):
                l_idx = plaq[pos][0]
                l_fwd = plaq[pos][1]
                # The staple for this link in this plaquette:
                # If the plaquette trace is Tr(U0 U1 U2), and we want the
                # derivative w.r.t. U_pos, the staple is (U_{pos+1} U_{pos+2})^dag
                # Actually: Tr(U_pos * staple^dag) where staple = U_{pos+2}^dag U_{pos+1}^dag
                # More carefully:
                # U_plaq = U0 U1 U2 (with appropriate daggers for direction)
                # For updating link pos: staple is the product of the OTHER links
                # in the order that makes: plaq_trace = Tr(U_pos * staple)
                # where U_pos might be dag'd depending on orientation.

                # The two "other" link positions
                p1 = (pos + 1) % 3
                p2 = (pos + 2) % 3

                link_map[l_idx].append({
                    'plaq_idx': p_idx,
                    'link_fwd': l_fwd,
                    'staple_links': [(plaq[p1][0], plaq[p1][1]),
                                     (plaq[p2][0], plaq[p2][1])],
                })

        return link_map

    def _build_adjacency(self):
        """Build vertex adjacency dict."""
        adj = {i: set() for i in range(self.lattice.vertex_count())}
        for (i, j) in self._edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj

    # ==================================================================
    # Link access
    # ==================================================================

    def get_link(self, idx, forward=True):
        """Get link matrix. If not forward, return U^dag."""
        if forward:
            return self._links[idx]
        else:
            return self._links[idx].conj().T

    def _compute_staple(self, link_idx):
        """
        Compute the sum of staples for a given link.

        The staple V for link U_l is defined so that:
        delta S / delta U_l = -(beta/2) * V

        For each plaquette containing link l:
        V += product_of_other_links_in_plaquette

        Returns (2,2) complex matrix (not necessarily SU(2)).
        """
        V = np.zeros((2, 2), dtype=complex)

        for info in self._link_to_plaqs[link_idx]:
            link_fwd = info['link_fwd']
            (l1_idx, l1_fwd), (l2_idx, l2_fwd) = info['staple_links']

            U1 = self.get_link(l1_idx, l1_fwd)
            U2 = self.get_link(l2_idx, l2_fwd)

            # Staple = U1 * U2, but we need the orientation correct.
            # The plaquette trace is Tr(U_link * U1 * U2)
            # where each U might be dag'd based on forward flag.
            # So V_link = U1 * U2 (the staple that combines with U_link)
            staple = U1 @ U2

            if not link_fwd:
                # If the link is traversed backward in this plaquette,
                # the plaquette reads ...U_link^dag... so the contribution
                # to the action variation involves the conjugated staple
                staple = staple.conj().T

            V += staple

        return V

    # ==================================================================
    # Update algorithms
    # ==================================================================

    def metropolis_sweep(self, epsilon=0.3):
        """
        One Metropolis sweep over all links.

        Returns acceptance rate.
        """
        accepted = 0
        for link_idx in range(self._n_links):
            # Compute staple
            V = self._compute_staple(link_idx)

            # Current link
            U_old = self._links[link_idx].copy()

            # Local action: S_local = -beta/2 * Re Tr(U * V)
            # (up to constant; only the part that depends on this link)
            s_old = -0.5 * self.beta * np.real(np.trace(U_old @ V))

            # Propose new link
            dU = self._su2_near_identity(epsilon)
            U_new = dU @ U_old

            s_new = -0.5 * self.beta * np.real(np.trace(U_new @ V))

            # Accept/reject
            dS = s_new - s_old
            if dS < 0 or self.rng.random() < np.exp(-dS):
                self._links[link_idx] = U_new
                accepted += 1
            # else: keep old

        return accepted / self._n_links

    def heatbath_sweep(self):
        """
        One heat bath sweep over all links using Kennedy-Pendleton algorithm.

        For SU(2), the conditional distribution of a link given the staple
        can be sampled exactly (no accept/reject needed).

        The distribution is: P(U) ~ exp(beta/2 * Re Tr(U * V))
        where V is the staple sum. This is equivalent to sampling on S3
        with a distribution peaked at V/|V|.

        Returns acceptance rate (always 1.0 for heat bath).
        """
        for link_idx in range(self._n_links):
            V = self._compute_staple(link_idx)

            # Decompose V = k * W where W is SU(2) and k = sqrt(det(V†V))/2
            # For V a 2x2 matrix: V = k * W with k = sqrt(|det V|), W in SU(2)
            # Actually for sum of SU(2) staples, V is proportional to SU(2).
            # V = alpha * W where alpha = sqrt(det(V†V))^{1/2}
            # But simpler: V^dag V = alpha^2 * I for SU(2), so alpha = sqrt(Tr(V^dag V)/2)

            VdV = V.conj().T @ V
            alpha_sq = 0.5 * np.real(np.trace(VdV))

            if not np.isfinite(alpha_sq) or alpha_sq < 1e-28:
                # Staple is zero or NaN -> random link
                self._links[link_idx] = self._random_su2()
                continue

            alpha = np.sqrt(alpha_sq)
            W = V / alpha  # approximately SU(2)
            # Project to exact SU(2)
            W = self._project_su2(W)

            # The action term for this link is:
            #   S_local = -beta/2 * Re Tr(U * V)
            #           = -beta/2 * alpha * Re Tr(U * W)
            # This is MINIMIZED (Boltzmann weight maximized) when U = W^dag.
            #
            # Substituting U = U' * W^dag where U' in SU(2):
            #   Re Tr(U * V) = alpha * Re Tr(U' * W^dag * W) = alpha * Re Tr(U')
            #                = alpha * 2 * a0
            # where a0 is the identity component of U'.
            #
            # So P(U') ~ exp(beta * alpha * a0) * sqrt(1 - a0^2)
            # and the final link is U = U' * W^dag.

            # Kennedy-Pendleton algorithm for sampling a0
            a0 = self._sample_su2_a0(self.beta * alpha)

            # Sample the remaining 3-vector uniformly on S2 * sqrt(1-a0^2)
            r = np.sqrt(1.0 - a0 * a0)
            # Uniform direction on S2
            phi = self.rng.uniform(0, 2 * np.pi)
            cos_theta = self.rng.uniform(-1, 1)
            sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

            a1 = r * sin_theta * np.cos(phi)
            a2 = r * sin_theta * np.sin(phi)
            a3 = r * cos_theta

            U_prime = self._quaternion_to_su2(np.array([a0, a1, a2, a3]))
            U_new = U_prime @ W.conj().T
            # Project back to SU(2) to prevent numerical drift
            self._links[link_idx] = self._project_su2(U_new)

        return 1.0  # heat bath always accepts

    def _sample_su2_a0(self, k):
        """
        Sample a0 from P(a0) ~ sqrt(1-a0^2) * exp(k * a0) on [-1, 1].

        Uses Kennedy-Pendleton algorithm (efficient rejection method).

        Parameters
        ----------
        k : float
            The coupling parameter (beta * alpha).

        Returns
        -------
        float : sampled a0 in [-1, 1].
        """
        if k < 1e-10:
            # Very weak coupling: nearly uniform
            # Sample uniformly on S3 and return a0
            while True:
                x = self.rng.uniform(-1, 1, 4)
                norm = np.linalg.norm(x)
                if norm > 0:
                    return x[0] / norm

        # Kennedy-Pendleton (1985): efficient for SU(2) heat bath
        # Generate x = exp(-2*k) < lambda < 1, with lambda = 1 - exp(-2k*u^2)
        # then a0 = 1 - 2*u^2 = 1 + log(lambda)/(k)
        while True:
            r1 = self.rng.random()
            r2 = self.rng.random()
            r3 = self.rng.random()
            r4 = self.rng.random()

            # lambda in the KP algorithm
            lam = -(np.log(r1) + np.log(r3) * np.cos(np.pi * r2) ** 2) / k

            if r4 * r4 <= 1.0 - 0.5 * lam:
                return 1.0 - lam

    def overrelaxation_sweep(self):
        """
        One microcanonical overrelaxation sweep.

        For each link, reflect it through the staple direction:
        U_new = W^dag * U_old^dag * W^dag
        where V = alpha * W is the staple, W in SU(2).

        This preserves the local action:
        Re Tr(U_new * V) = Re Tr(W^dag U^dag W^dag * alpha * W)
                         = alpha * Re Tr(W^dag U^dag) = alpha * Re Tr(U W)^*
                         = alpha * Re Tr(U W) = Re Tr(U * V)

        This is a microcanonical (energy-preserving) update that
        speeds up decorrelation without changing the energy.
        """
        for link_idx in range(self._n_links):
            V = self._compute_staple(link_idx)

            VdV = V.conj().T @ V
            alpha_sq = 0.5 * np.real(np.trace(VdV))

            if not np.isfinite(alpha_sq) or alpha_sq < 1e-28:
                continue

            alpha = np.sqrt(alpha_sq)
            W = self._project_su2(V / alpha)
            Wd = W.conj().T
            U_old = self._links[link_idx]
            # Overrelaxation: U_new = W^dag * U_old^dag * W^dag
            U_new = Wd @ U_old.conj().T @ Wd
            # Project back to SU(2) to prevent numerical drift
            self._links[link_idx] = self._project_su2(U_new)

    def compound_sweep(self, n_heatbath=1, n_overrelax=4):
        """
        One compound sweep: n_heatbath heat bath + n_overrelax overrelaxation.

        Standard combination for efficient lattice MC:
        - Heat bath for ergodicity (samples new configs)
        - Overrelaxation for decorrelation (moves through config space faster)

        Returns dict with acceptance info.
        """
        for _ in range(n_heatbath):
            self.heatbath_sweep()
        for _ in range(n_overrelax):
            self.overrelaxation_sweep()
        return {'acceptance_rate': 1.0}

    # ==================================================================
    # Observables
    # ==================================================================

    def plaquette_average(self):
        """
        Average plaquette: <(1/2) Re Tr U_plaq> over all 1200 plaquettes.

        For trivial vacuum: 1.0
        For random config: ~0 (SU(2))
        For thermalized: between 0 and 1, increasing with beta.
        """
        total = 0.0
        for plaq in self._plaq_links:
            U = np.eye(2, dtype=complex)
            for (l_idx, fwd) in plaq:
                U = U @ self.get_link(l_idx, fwd)
            total += 0.5 * np.real(np.trace(U))
        return total / self._n_plaq

    def wilson_action(self):
        """Total Wilson action: S = beta * Sum_plaq (1 - (1/2) Re Tr U_plaq)."""
        return self.beta * self._n_plaq * (1.0 - self.plaquette_average())

    def wilson_loop_path(self, vertex_path):
        """
        Compute Wilson loop along a given path of vertices.

        Parameters
        ----------
        vertex_path : list of int
            Ordered list of vertex indices forming a closed loop.

        Returns
        -------
        complex : (1/2) Tr(product of links along path)
        """
        U = np.eye(2, dtype=complex)
        n = len(vertex_path)
        for step in range(n):
            i = vertex_path[step]
            j = vertex_path[(step + 1) % n]
            if (i, j) in self._edge_idx:
                l_idx = self._edge_idx[(i, j)]
                fwd = self._edge_fwd[(i, j)]
                U = U @ self.get_link(l_idx, fwd)
            else:
                # Vertices not connected by an edge -- invalid path
                return 0.0
        return 0.5 * np.trace(U)

    def find_loops_by_length(self, max_length=6, max_per_length=50):
        """
        Find closed loops of various lengths on the 600-cell graph.

        Returns dict: length -> list of vertex paths.

        For triangular lattice:
          length 3: triangles (1200 of them = the faces)
          length 4: squares (exist on 600-cell since it's not bipartite)
          length 5: pentagons
          length 6: hexagons
          etc.
        """
        loops = {}

        # Length 3: the faces themselves
        loops[3] = [list(f) for f in self._faces[:max_per_length]]

        # Length 4+: BFS-based loop finder
        n_verts = self.lattice.vertex_count()

        for length in range(4, max_length + 1):
            found = []
            visited_set = set()

            for start in range(min(n_verts, 30)):  # start from first 30 vertices
                if len(found) >= max_per_length:
                    break
                # DFS to find loops of exact length
                self._find_loops_dfs(
                    start, start, [start], length, found,
                    visited_set, max_per_length
                )

            loops[length] = found

        return loops

    def _find_loops_dfs(self, start, current, path, target_len, found,
                        visited_set, max_count):
        """DFS helper to find loops of exact length."""
        if len(found) >= max_count:
            return

        if len(path) == target_len:
            # Check if we can close the loop
            if start in self._adj[current]:
                # Canonicalize: rotate so smallest index is first
                canon = tuple(path)
                min_idx = path.index(min(path))
                canon = tuple(path[min_idx:] + path[:min_idx])
                if canon not in visited_set:
                    visited_set.add(canon)
                    found.append(list(path))
            return

        for nb in sorted(self._adj[current]):
            if nb == start and len(path) < target_len:
                continue  # Don't close too early
            if nb in path[1:]:
                continue  # No repeated vertices (except closing)
            path.append(nb)
            self._find_loops_dfs(start, nb, path, target_len, found,
                                 visited_set, max_count)
            path.pop()

    def measure_wilson_loops(self, loops_dict):
        """
        Measure Wilson loops organized by length.

        Parameters
        ----------
        loops_dict : dict of length -> list of vertex paths

        Returns
        -------
        dict of length -> (mean_W, std_W, n_loops)
        """
        results = {}
        for length, paths in loops_dict.items():
            if not paths:
                continue
            vals = []
            for path in paths:
                w = self.wilson_loop_path(path)
                vals.append(np.real(w))
            vals = np.array(vals)
            results[length] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'n_loops': len(vals),
            }
        return results

    def plaquette_field(self):
        """
        Compute plaquette trace for every face.

        Returns (n_plaq,) array of (1/2) Re Tr U_plaq values.
        """
        field = np.zeros(self._n_plaq)
        for p_idx, plaq in enumerate(self._plaq_links):
            U = np.eye(2, dtype=complex)
            for (l_idx, fwd) in plaq:
                U = U @ self.get_link(l_idx, fwd)
            field[p_idx] = 0.5 * np.real(np.trace(U))
        return field

    def plaquette_correlator_by_distance(self, n_distance_bins=15):
        """
        Compute plaquette-plaquette connected correlator binned by geodesic distance.

        C(d) = <P(x) P(y)>_conn = <P(x)P(y)> - <P>^2

        averaged over all face pairs (x,y) at geodesic distance d.

        Returns
        -------
        dict with 'distances', 'correlator', 'n_pairs'
        """
        # Get face centers and distance matrix
        verts = self.lattice.vertices / self.lattice.R
        faces = self._faces
        n_f = len(faces)

        # Face centers on unit sphere
        centers = np.zeros((n_f, 4))
        for f_idx, (i, j, k) in enumerate(faces):
            c = (verts[i] + verts[j] + verts[k]) / 3.0
            norm = np.linalg.norm(c)
            if norm > 1e-12:
                centers[f_idx] = c / norm

        # Geodesic distances
        dots = np.clip(centers @ centers.T, -1.0, 1.0)
        dist_matrix = np.arccos(dots)

        # Measure plaquette field
        P = self.plaquette_field()
        P_mean = np.mean(P)
        P_fluct = P - P_mean

        # Bin distances
        max_dist = np.max(dist_matrix)
        bin_edges = np.linspace(0, max_dist + 1e-10, n_distance_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Compute correlator in each bin using upper triangle only
        corr = np.zeros(n_distance_bins)
        n_pairs = np.zeros(n_distance_bins, dtype=int)

        for i in range(n_f):
            for j in range(i + 1, n_f):
                d = dist_matrix[i, j]
                b = np.searchsorted(bin_edges, d) - 1
                b = max(0, min(b, n_distance_bins - 1))
                corr[b] += P_fluct[i] * P_fluct[j]
                n_pairs[b] += 1

        # Average
        mask = n_pairs > 0
        corr[mask] /= n_pairs[mask]

        return {
            'distances': bin_centers,
            'correlator': corr,
            'n_pairs': n_pairs,
        }

    def great_circle_paths(self, n_paths=10):
        """
        Find great circle paths through the lattice graph.

        On the 600-cell, great circles pass through 10 vertices each
        (the decagonal great circles).

        Returns list of vertex-index lists.
        """
        return self.lattice.great_circles(max_circles=n_paths)

    def polyakov_loops(self, paths=None):
        """
        Measure Polyakov-like loops (Wilson loops along great circles).

        Parameters
        ----------
        paths : list of vertex paths, or None (auto-detect).

        Returns
        -------
        array of complex Polyakov loop values.
        """
        if paths is None:
            paths = self.great_circle_paths(n_paths=10)

        vals = []
        for path in paths:
            w = self.wilson_loop_path(path)
            vals.append(w)
        return np.array(vals)

    # ==================================================================
    # Coordinate-based correlator (for mass gap extraction)
    # ==================================================================

    def vertex_coordinate_bins(self, coord=0, n_bins=None):
        """
        Bin vertices by one of the 4D coordinates (for "time" slicing).

        On S3, we pick the w-coordinate (coord=0) as "time".
        This gives a foliation of S3 into 3-spheres of decreasing radius.

        Returns
        -------
        bin_edges, bin_indices : bin boundaries and vertex bin assignments
        """
        verts = self.lattice.vertices
        w = verts[:, coord]

        if n_bins is None:
            n_bins = 10  # about 12 vertices per bin

        bin_edges = np.linspace(w.min() - 1e-10, w.max() + 1e-10, n_bins + 1)
        bin_indices = np.digitize(w, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        return bin_edges, bin_indices

    def time_slice_correlator(self, coord=0, n_bins=10):
        """
        Compute plaquette correlator binned by "time" coordinate.

        We average the plaquette observable over spatial slices at each
        "time" value, then compute the time-time correlator.

        C(t) = <O(t) O(0)> - <O>^2

        The mass gap is extracted from: C(t) ~ exp(-m*t).

        Returns
        -------
        dict with 'separations', 'correlator', 'observable_by_slice'
        """
        # Bin faces by their average w-coordinate
        verts = self.lattice.vertices
        faces = self._faces
        n_f = len(faces)

        w_faces = np.zeros(n_f)
        for f_idx, (i, j, k) in enumerate(faces):
            w_faces[f_idx] = (verts[i, coord] + verts[j, coord] + verts[k, coord]) / 3.0

        bin_edges = np.linspace(w_faces.min() - 1e-10, w_faces.max() + 1e-10, n_bins + 1)
        bin_indices = np.digitize(w_faces, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Measure plaquette field
        P = self.plaquette_field()

        # Average observable in each time slice
        O = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        for f_idx in range(n_f):
            b = bin_indices[f_idx]
            O[b] += P[f_idx]
            counts[b] += 1

        mask = counts > 0
        O[mask] /= counts[mask]

        # Connected correlator
        O_mean = np.mean(O[mask])
        O_fluct = np.zeros(n_bins)
        O_fluct[mask] = O[mask] - O_mean

        # C(dt) = (1/N_t) sum_t O(t)*O(t+dt)
        max_dt = n_bins // 2
        separations = np.arange(max_dt + 1, dtype=float)
        corr = np.zeros(max_dt + 1)

        for dt in range(max_dt + 1):
            c_sum = 0.0
            n_valid = 0
            for t in range(n_bins):
                t2 = (t + dt) % n_bins
                if mask[t] and mask[t2]:
                    c_sum += O_fluct[t] * O_fluct[t2]
                    n_valid += 1
            if n_valid > 0:
                corr[dt] = c_sum / n_valid

        # Convert bin index to geodesic distance
        # Each bin spans roughly pi/n_bins in geodesic distance on S3
        bin_width = np.pi * self.lattice.R / n_bins
        separations = separations * bin_width

        return {
            'separations': separations,
            'correlator': corr,
            'observable_by_slice': O,
            'bin_counts': counts,
        }

    # ==================================================================
    # String tension via Creutz ratios (adapted for triangular lattice)
    # ==================================================================

    def creutz_ratio_estimate(self, loops_dict):
        """
        Estimate string tension from Wilson loop area law.

        On a triangular lattice, Wilson loops of length L enclose
        area ~ L^2 / (4*sqrt(3)) (roughly). For area law confinement:
        <W(C)> ~ exp(-sigma * Area(C))

        The Creutz ratio is:
        chi(I,J) = -ln[W(I,J)*W(I-1,J-1) / (W(I-1,J)*W(I,J-1))]

        For our triangular lattice, we use loops of different lengths
        and extract the effective string tension from the length dependence.

        Parameters
        ----------
        loops_dict : dict from find_loops_by_length

        Returns
        -------
        dict with 'sigma_eff' (effective string tension estimates)
        """
        # Collect <W> by loop length
        wl = self.measure_wilson_loops(loops_dict)
        lengths = sorted(wl.keys())

        if len(lengths) < 2:
            return {'sigma_eff': [], 'sigma_mean': 0.0}

        # Effective string tension from consecutive lengths:
        # <W(L)> ~ exp(-sigma * A(L)) where A ~ L^2
        # so sigma_eff ~ -[ln W(L+1) - ln W(L)] / [A(L+1) - A(L)]
        sigma_effs = []
        for i in range(len(lengths) - 1):
            L1 = lengths[i]
            L2 = lengths[i + 1]
            W1 = wl[L1]['mean']
            W2 = wl[L2]['mean']

            if W1 > 1e-10 and W2 > 1e-10 and W1 > W2:
                # Area for a regular polygon with L edges on the lattice
                # Approximate: area ~ L * a^2 * sin(2*pi/L) / 2
                a = self.lattice.lattice_spacing()
                A1 = 0.5 * L1 * a ** 2 * np.sin(2 * np.pi / max(L1, 3))
                A2 = 0.5 * L2 * a ** 2 * np.sin(2 * np.pi / max(L2, 3))
                dA = A2 - A1
                if dA > 1e-10:
                    sigma = -(np.log(W2) - np.log(W1)) / dA
                    sigma_effs.append({
                        'lengths': (L1, L2),
                        'sigma': float(sigma),
                        'W_values': (W1, W2),
                    })

        sigma_mean = np.mean([s['sigma'] for s in sigma_effs]) if sigma_effs else 0.0

        return {
            'sigma_eff': sigma_effs,
            'sigma_mean': float(sigma_mean),
        }
