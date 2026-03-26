"""
Lattice Yang-Mills -- Lattice gauge theory on the 600-cell discretization of S^3.

Link variables U_ij in SU(N) live on each edge (i, j) of the 600-cell.
The Wilson action is:

    S = beta * Sum_plaq (1 - Re Tr U_plaq / N)

where U_plaq = U_ij * U_jk * U_ki for each triangular plaquette (i, j, k).

Key features on S^3:
    - Finite volume: no thermodynamic limit needed
    - Compact spatial manifold: no boundary conditions
    - Single limit: a -> 0 (lattice spacing to zero)
    - The flat vacuum (all U_ij = 1) is the Maurer-Cartan connection

Status labels:
    - Wilson action and Haar measure: THEOREM (standard lattice gauge theory)
    - Transfer matrix gap: NUMERICAL (extracted from correlator fits)
    - Continuum limit: OPEN (requires beta -> infinity analysis)
"""

import numpy as np
from .s3_lattice import S3Lattice


class LatticeYM:
    """
    Lattice Yang-Mills on the 600-cell discretization of S^3.

    Link variables U_ij in SU(N) on each edge (i, j).
    Wilson action: S = beta * Sum_plaq (1 - Re Tr U_plaq / N)

    The lattice provides a non-perturbative regularization of YM on S^3.
    The mass gap can be extracted from the transfer matrix or correlator
    decay.
    """

    def __init__(self, lattice, N=2, beta=1.0):
        """
        Initialize lattice YM with link variables.

        Parameters
        ----------
        lattice : S3Lattice
            The 600-cell lattice on S^3.
        N : int
            Gauge group SU(N). Default 2.
        beta : float
            Lattice coupling beta = 2N/g^2. Default 1.0.
        """
        self.lattice = lattice
        self.N = N
        self.beta = beta

        # Store edges as list for indexing
        self._edges = lattice.edges()
        self._n_links = len(self._edges)

        # Build edge-to-index map for fast lookup
        self._edge_index = {}
        for idx, (i, j) in enumerate(self._edges):
            self._edge_index[(i, j)] = idx
            self._edge_index[(j, i)] = idx  # Also store reverse

        # Initialize link variables: SU(N) matrices on each edge
        # Start with identity (flat vacuum = Maurer-Cartan)
        self._links = np.array([np.eye(N, dtype=complex) for _ in range(self._n_links)])

        # Precompute plaquettes as index triples with orientation
        self._plaquettes = self._build_plaquettes()

    # ------------------------------------------------------------------
    # Plaquette construction
    # ------------------------------------------------------------------
    def _build_plaquettes(self):
        """
        Build plaquettes with proper link orientations.

        Each triangular face (i, j, k) gives a plaquette:
            U_plaq = U_{ij} * U_{jk} * U_{ki}

        where U_{ji} = U_{ij}^dagger (reverse orientation).

        Returns
        -------
        list of (edge_idx, is_forward) tuples for each plaquette
        """
        faces = self.lattice.faces()
        plaquettes = []

        for (i, j, k) in faces:
            # Oriented links: i->j, j->k, k->i
            links_in_plaq = []
            for (a, b) in [(i, j), (j, k), (k, i)]:
                if (a, b) in self._edge_index:
                    idx = self._edge_index[(a, b)]
                    # Check if (a, b) is stored as (a, b) or (b, a)
                    stored_edge = self._edges[idx]
                    forward = (stored_edge[0] == a)
                    links_in_plaq.append((idx, forward))
                else:
                    # This should not happen if edges are built correctly
                    links_in_plaq.append((0, True))

            plaquettes.append(links_in_plaq)

        return plaquettes

    # ------------------------------------------------------------------
    # Link variable access
    # ------------------------------------------------------------------
    def get_link(self, edge_idx, forward=True):
        """
        Get the link variable for a given edge.

        Parameters
        ----------
        edge_idx : int
            Index into the edge list.
        forward : bool
            If True, return U_{ij}. If False, return U_{ji} = U_{ij}^dagger.

        Returns
        -------
        numpy array : N x N complex matrix in SU(N)
        """
        U = self._links[edge_idx]
        if forward:
            return U
        else:
            return U.conj().T

    def set_link(self, edge_idx, U):
        """Set the link variable for a given edge."""
        self._links[edge_idx] = U.copy()

    # ------------------------------------------------------------------
    # Wilson action
    # ------------------------------------------------------------------
    def plaquette_trace(self, plaq):
        """
        Compute (1/N) Re Tr U_plaq for a single plaquette.

        Parameters
        ----------
        plaq : list of (edge_idx, forward) tuples

        Returns
        -------
        float : (1/N) Re Tr(U_plaq)
        """
        U_product = np.eye(self.N, dtype=complex)
        for (idx, forward) in plaq:
            U = self.get_link(idx, forward)
            U_product = U_product @ U

        return np.real(np.trace(U_product)) / self.N

    def wilson_action(self):
        """
        Compute the Wilson plaquette action.

        S = beta * Sum_plaq (1 - (1/N) Re Tr U_plaq)

        Returns
        -------
        float : total Wilson action
        """
        total = 0.0
        for plaq in self._plaquettes:
            trace_val = self.plaquette_trace(plaq)
            total += 1.0 - trace_val
        return self.beta * total

    def plaquette_average(self):
        """
        Average plaquette: <(1/N) Re Tr U_plaq> over all plaquettes.

        For the trivial vacuum (all links = identity): plaq_avg = 1.0
        For random SU(N) links: plaq_avg -> 0 as N -> infinity
        For thermalized configs: 0 < plaq_avg < 1

        Returns
        -------
        float : average plaquette value
        """
        if not self._plaquettes:
            return 1.0

        total = sum(self.plaquette_trace(p) for p in self._plaquettes)
        return total / len(self._plaquettes)

    # ------------------------------------------------------------------
    # SU(N) utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_su_n(N, rng=None):
        """
        Generate a random SU(N) matrix, approximately Haar distributed.

        Method: take a random complex N x N matrix, compute its QR
        decomposition, and adjust to make det = 1.

        Parameters
        ----------
        N : int
            Size of the matrix.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        numpy array : N x N complex matrix in SU(N)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random complex matrix
        Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        Q, R_qr = np.linalg.qr(Z)

        # Make the decomposition unique: ensure diagonal of R is positive
        d = np.diag(R_qr)
        ph = d / np.abs(d)
        Q = Q @ np.diag(ph.conj())

        # Project to SU(N): det(Q) should be 1
        det = np.linalg.det(Q)
        Q = Q / (det ** (1.0 / N))

        return Q

    @staticmethod
    def su_n_near_identity(N, epsilon=0.1, rng=None):
        """
        Generate an SU(N) matrix close to the identity.

        Used for Metropolis updates: propose U' = V * U where V ~ 1.

        Parameters
        ----------
        N : int
        epsilon : float
            Controls how far from identity (smaller = closer).
        rng : numpy.random.Generator, optional

        Returns
        -------
        numpy array : N x N SU(N) matrix near identity
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random anti-Hermitian matrix (Lie algebra element)
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        A = epsilon * (A - A.conj().T) / 2  # anti-Hermitian

        # Make traceless
        A = A - np.trace(A) / N * np.eye(N)

        # Exponentiate to get SU(N) element near identity
        from scipy.linalg import expm
        U = expm(A)

        # Ensure det = 1
        det = np.linalg.det(U)
        U = U / (det ** (1.0 / N))

        return U

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def randomize_links(self, rng=None):
        """Set all link variables to random SU(N) matrices."""
        if rng is None:
            rng = np.random.default_rng()
        for idx in range(self._n_links):
            self._links[idx] = self.random_su_n(self.N, rng)

    def thermalize(self, n_sweeps=10, epsilon=0.3, rng=None):
        """
        Thermalize the configuration using Metropolis algorithm.

        Each sweep visits every link once.

        Parameters
        ----------
        n_sweeps : int
            Number of full sweeps.
        epsilon : float
            Step size for proposals.
        rng : numpy.random.Generator, optional

        Returns
        -------
        dict with 'acceptance_rate' and 'final_action'
        """
        if rng is None:
            rng = np.random.default_rng()

        accepted = 0
        total = 0

        for sweep in range(n_sweeps):
            for link_idx in range(self._n_links):
                # Compute local action contribution (staples)
                S_old = self._local_action(link_idx)

                # Propose new link
                U_old = self._links[link_idx].copy()
                V = self.su_n_near_identity(self.N, epsilon, rng)
                self._links[link_idx] = V @ U_old

                # Compute new local action
                S_new = self._local_action(link_idx)

                # Metropolis accept/reject
                dS = S_new - S_old
                if dS < 0 or rng.random() < np.exp(-dS):
                    accepted += 1  # Keep new config
                else:
                    self._links[link_idx] = U_old  # Revert

                total += 1

        return {
            'acceptance_rate': accepted / total if total > 0 else 0.0,
            'final_action': self.wilson_action(),
        }

    def _local_action(self, link_idx):
        """
        Compute the part of the action involving a specific link.

        Only plaquettes containing this link contribute.

        Parameters
        ----------
        link_idx : int

        Returns
        -------
        float : local action contribution
        """
        total = 0.0
        for plaq in self._plaquettes:
            # Check if this link is in the plaquette
            link_indices = [idx for (idx, _) in plaq]
            if link_idx in link_indices:
                trace_val = self.plaquette_trace(plaq)
                total += self.beta * (1.0 - trace_val)
        return total

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def polyakov_loop(self, path=None):
        """
        Polyakov loop: trace of product of links along a closed path.

        On S^3, there is no thermal circle, so this computes the
        spatial Wilson loop along a great circle of S^3.

        <P> = 0 -> confined phase
        <P> != 0 -> deconfined phase

        Parameters
        ----------
        path : list of vertex indices forming a closed loop, optional.
               If None, uses the first great circle found.

        Returns
        -------
        complex : (1/N) Tr(Product of links along path)
        """
        if path is None:
            circles = self.lattice.great_circles(max_circles=1)
            if not circles:
                return 0.0
            path = circles[0]

        # Compute product of links along the path
        U_product = np.eye(self.N, dtype=complex)
        for step in range(len(path)):
            i = path[step]
            j = path[(step + 1) % len(path)]

            if (i, j) in self._edge_index:
                idx = self._edge_index[(i, j)]
                stored = self._edges[idx]
                forward = (stored[0] == i)
                U = self.get_link(idx, forward)
                U_product = U_product @ U

        return np.trace(U_product) / self.N

    def correlator_at_separation(self, t_index, observable='plaquette'):
        """
        Compute correlator at a given "time" separation.

        On the 600-cell, we define "time" along one of the 4D coordinates
        (say w-coordinate). Vertices are binned by their w-value.

        Parameters
        ----------
        t_index : int
            Time separation in lattice units (bin index difference).
        observable : str
            Type of observable. Currently supports 'plaquette'.

        Returns
        -------
        float : correlator value <O(t) O(0)>
        """
        # Bin vertices by w-coordinate (first coordinate)
        vertices = self.lattice.vertices
        w_values = vertices[:, 0]

        # Create bins
        n_bins = max(6, int(np.sqrt(len(vertices))))
        bin_edges = np.linspace(w_values.min() - 1e-10,
                                w_values.max() + 1e-10, n_bins + 1)
        bin_indices = np.digitize(w_values, bin_edges) - 1

        # Compute observable in each bin (average plaquette in that slice)
        bin_obs = {}
        for plaq_idx, plaq in enumerate(self._plaquettes):
            # Get vertices of this plaquette
            face = self.lattice.faces()[plaq_idx]
            avg_bin = int(np.mean([bin_indices[v] for v in face]))
            if avg_bin not in bin_obs:
                bin_obs[avg_bin] = []
            bin_obs[avg_bin].append(self.plaquette_trace(plaq))

        # Average observable in each bin
        bin_means = {}
        for b, vals in bin_obs.items():
            bin_means[b] = np.mean(vals)

        # Compute correlator: <O(t) O(0)> = average over all bins b of O(b+t)*O(b)
        valid_pairs = 0
        corr = 0.0
        for b in bin_means:
            b2 = (b + t_index) % n_bins
            if b2 in bin_means:
                corr += bin_means[b] * bin_means[b2]
                valid_pairs += 1

        if valid_pairs == 0:
            return 0.0
        return corr / valid_pairs

    def transfer_matrix_gap(self, n_configs=50, n_therm=5, epsilon=0.3):
        """
        Estimate the mass gap from the transfer matrix approach.

        Method: generate thermalized configurations, compute correlator
        C(t) = <O(t) O(0)> for increasing t. Fit to C(t) ~ exp(-m*t)
        to extract mass gap m.

        On the 600-cell, "time" is discretized into bins along one coordinate.

        Parameters
        ----------
        n_configs : int
            Number of independent configurations to average over.
        n_therm : int
            Number of thermalization sweeps between measurements.
        epsilon : float
            Metropolis step size.

        Returns
        -------
        dict with:
            'gap_estimate': float (mass gap in lattice units)
            'correlators': list of (t, C(t)) values
            'gap_positive': bool
        """
        rng = np.random.default_rng(42)

        # Time separations
        max_t = 4
        t_values = list(range(max_t + 1))

        # Accumulate correlators
        corr_sum = {t: 0.0 for t in t_values}

        self.randomize_links(rng)
        # Initial thermalization
        self.thermalize(n_sweeps=n_therm * 2, epsilon=epsilon, rng=rng)

        for config in range(n_configs):
            # Thermalize between measurements
            self.thermalize(n_sweeps=n_therm, epsilon=epsilon, rng=rng)

            # Measure correlators
            for t in t_values:
                corr_sum[t] += self.correlator_at_separation(t)

        # Average
        correlators = [(t, corr_sum[t] / n_configs) for t in t_values]

        # Extract gap from log of correlator ratio
        # C(t) ~ exp(-m*t) => m ~ -log(C(t+1)/C(t))
        gap_estimate = 0.0
        if len(correlators) >= 3:
            c0 = correlators[0][1]
            c1 = correlators[1][1]
            if c0 > 0 and c1 > 0 and c1 < c0:
                gap_estimate = -np.log(c1 / c0)

        return {
            'gap_estimate': gap_estimate,
            'correlators': correlators,
            'gap_positive': gap_estimate > 0,
        }
