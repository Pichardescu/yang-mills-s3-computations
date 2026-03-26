"""
Poincare Lattice -- S^3/I* spectral sparsification on the 600-cell.

The 600-cell is a regular polytope with 120 vertices on S^3. These 120 vertices
correspond EXACTLY to the 120 elements of the binary icosahedral group I*
(viewed as unit quaternions in S^3 ~ SU(2)). Right multiplication by h in I*
permutes the vertices.

KEY INSIGHT: On S^3/I*, only I*-invariant modes survive. We verify this by:
  1. Building the I* permutation action on the 600-cell
  2. Computing the I*-invariant projector
  3. Applying it to eigenspaces of the discrete Laplacian
  4. Checking which eigenvalues survive

SCALAR SPECTRUM (THEOREM):
  On S^3: eigenvalues at k(k+2)/R^2 with multiplicity (k+1)^2
  On S^3/I*: m(k) > 0 only for k = 0, 12, 20, 24, ...
  600-cell resolves k=0..5; I*-invariant scalar spectrum has ONLY k=0 (constant mode).

COEXACT 1-FORM SPECTRUM (THEOREM):
  On S^3: coexact eigenvalues at (k+1)^2/R^2 with multiplicity 2k(k+2) for k>=1
  On S^3/I*: k=1 survives (3 modes), k=2..10 absent, k=11 first to return
  On the 600-cell: lowest coexact eigenvalue ~4/R^2 has I*-invariant modes (GAP PRESERVED)
  k=2 (eigenvalue ~9/R^2) has ZERO I*-invariant modes (SPARSIFICATION)

References:
  - Coxeter (1973): Regular Polytopes (600-cell construction)
  - Ikeda & Taniguchi (1978): Spectra on spherical space forms
  - Luminet et al. Nature 425, 593 (2003): Dodecahedral space topology
"""

import numpy as np
from scipy import linalg as la
from yang_mills_s3.lattice.s3_lattice import S3Lattice


def _quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 * q2.

    Quaternion convention: q = (w, x, y, z) = w + xi + yj + zk

    Parameters
    ----------
    q1, q2 : array-like of shape (4,)

    Returns
    -------
    ndarray of shape (4,): the product q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quaternion_multiply_batch(q1, Q2):
    """
    Multiply quaternion q1 by each row of Q2 (batch right-multiply).

    Parameters
    ----------
    q1 : array of shape (4,)
    Q2 : array of shape (N, 4)

    Returns
    -------
    ndarray of shape (N, 4)
    """
    w1, x1, y1, z1 = q1
    w2 = Q2[:, 0]
    x2 = Q2[:, 1]
    y2 = Q2[:, 2]
    z2 = Q2[:, 3]
    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class PoincareLattice:
    """
    S^3/I* spectral analysis on the 600-cell lattice.

    The 120 vertices of the 600-cell are the 120 elements of I* as unit
    quaternions. Right multiplication by any h in I* permutes these vertices,
    giving a faithful group action.

    This class builds:
      - The I* permutation action on vertices and edges
      - The I*-invariant projectors
      - The scalar and 1-form Laplacians
      - The projected (I*-invariant) spectra

    THEOREM: The Yang-Mills mass gap on S^3/I* equals that on S^3.
    The lowest coexact eigenvalue 4/R^2 survives the I* projection
    with multiplicity 3 (self-dual right-invariant forms).
    """

    def __init__(self, R=1.0):
        """
        Construct the Poincare lattice from the 600-cell.

        Parameters
        ----------
        R : float
            Radius of S^3. Default 1.0.
        """
        self.R = R
        self.lattice = S3Lattice(R)
        self._unit_verts = self.lattice.vertices / R
        self._n_vertices = self.lattice.vertex_count()
        self._n_edges = self.lattice.edge_count()

        # Canonical edge list with orientation i < j
        self._edge_list = self.lattice.edges()
        # Map (i,j) -> edge index for fast lookup
        self._edge_index = {}
        for idx, (i, j) in enumerate(self._edge_list):
            self._edge_index[(i, j)] = idx

        # Face list
        self._face_list = self.lattice.faces()
        self._n_faces = self.lattice.face_count()

        # Build the I* group action
        self._perms = None  # shape (120, 120): _perms[h, g] = index of g*h
        self._build_istar_action()

    # ==================================================================
    # I* Group Action
    # ==================================================================

    def _build_istar_action(self):
        """
        Build the I* right-multiplication action on the 600-cell vertices.

        For each group element h (vertex h_idx), compute the permutation
        sigma_h: g -> g*h for all vertices g.

        The result is stored as self._perms[h_idx, g_idx] = index of g*h.
        """
        V = self._unit_verts  # (120, 4) unit quaternions
        n = self._n_vertices
        perms = np.zeros((n, n), dtype=int)

        for h_idx in range(n):
            h = V[h_idx]
            # Compute g * h for all g
            products = _quaternion_multiply_batch(h, V)
            # Actually we want g * h, not h * g. Let's do it properly.
            # _quaternion_multiply_batch(q1, Q2) computes q1 * Q2[i] for each i
            # We want g * h for each g, i.e. V[g] * h for each g
            # So we need to loop or use a different batch function.
            # Let's compute V * h by doing each product.
            products = np.zeros((n, 4))
            for g_idx in range(n):
                products[g_idx] = _quaternion_multiply(V[g_idx], h)

            # Match each product to the nearest vertex
            # Use dot products for speed
            dots = products @ V.T  # (n, n)
            perms[h_idx] = np.argmax(dots, axis=1)

        self._perms = perms

    def vertex_permutation(self, h_idx):
        """
        Return the permutation of vertices induced by right-multiplication by vertex h_idx.

        sigma_h(g) = g * h

        Parameters
        ----------
        h_idx : int
            Index of the group element h.

        Returns
        -------
        ndarray of shape (120,): sigma[g] = index of g*h
        """
        return self._perms[h_idx].copy()

    def _find_identity_index(self):
        """
        Find the vertex index corresponding to the identity quaternion (1,0,0,0).

        Returns
        -------
        int : index of the identity vertex
        """
        V = self._unit_verts
        dots = V @ np.array([1.0, 0.0, 0.0, 0.0])
        return int(np.argmax(dots))

    # ==================================================================
    # Projectors
    # ==================================================================

    def istar_projector_vertices(self):
        """
        Build the I*-invariant projector on vertex (scalar) space.

        Pi = (1/|I*|) * sum_{h in I*} P_h

        where P_h is the permutation matrix for right-multiplication by h.

        Returns
        -------
        ndarray of shape (120, 120): the projector matrix
        """
        n = self._n_vertices
        Pi = np.zeros((n, n))

        for h_idx in range(n):
            perm = self._perms[h_idx]
            P_h = np.zeros((n, n))
            for g in range(n):
                P_h[g, perm[g]] = 1.0
            Pi += P_h

        Pi /= n  # divide by |I*| = 120
        return Pi

    def _edge_permutation(self, h_idx):
        """
        Compute the permutation of edges induced by vertex permutation h_idx.

        When vertex g maps to sigma_h(g), edge (i,j) maps to
        (sigma_h(i), sigma_h(j)) reoriented to canonical form.

        Returns
        -------
        perm_edges : ndarray of shape (720,), the edge permutation
        signs : ndarray of shape (720,), +1 or -1 for orientation
        """
        sigma = self._perms[h_idx]
        n_edges = self._n_edges
        perm_edges = np.zeros(n_edges, dtype=int)
        signs = np.ones(n_edges)

        for e_idx, (i, j) in enumerate(self._edge_list):
            new_i = sigma[i]
            new_j = sigma[j]
            # Canonical orientation: smaller index first
            if new_i < new_j:
                key = (new_i, new_j)
                sign = 1.0
            else:
                key = (new_j, new_i)
                sign = -1.0
            perm_edges[e_idx] = self._edge_index[key]
            signs[e_idx] = sign

        return perm_edges, signs

    def istar_projector_edges(self):
        """
        Build the I*-invariant projector on edge (1-form) space.

        Pi_edge = (1/|I*|) * sum_{h in I*} P_h^{edge}

        where P_h^{edge} is the signed permutation matrix on edges.

        Returns
        -------
        ndarray of shape (720, 720): the projector matrix
        """
        n_e = self._n_edges
        n_v = self._n_vertices
        Pi = np.zeros((n_e, n_e))

        for h_idx in range(n_v):
            perm_edges, signs = self._edge_permutation(h_idx)
            P_h = np.zeros((n_e, n_e))
            for e in range(n_e):
                P_h[e, perm_edges[e]] = signs[e]
            Pi += P_h

        Pi /= n_v  # divide by |I*| = 120
        return Pi

    # ==================================================================
    # Simplicial operators
    # ==================================================================

    def incidence_matrix(self):
        """
        Build d_0: C^0 -> C^1 (coboundary/incidence matrix).

        For oriented edge e_{ij} (i < j):
            (d_0 f)(e_{ij}) = f(j) - f(i)

        Returns
        -------
        ndarray of shape (n_edges, n_vertices) = (720, 120)
        """
        n_v = self._n_vertices
        n_e = self._n_edges
        d0 = np.zeros((n_e, n_v))

        for e_idx, (i, j) in enumerate(self._edge_list):
            d0[e_idx, i] = -1.0
            d0[e_idx, j] = 1.0

        return d0

    def face_boundary_matrix(self):
        """
        Build d_1: C^1 -> C^2 (coboundary from edges to faces).

        For oriented face f_{ijk} with i < j < k:
            (d_1 omega)(f_{ijk}) = omega_{ij} + omega_{jk} - omega_{ik}

        where omega_{ij} is the value on the oriented edge from i to j.

        Returns
        -------
        ndarray of shape (n_faces, n_edges) = (1200, 720)
        """
        n_e = self._n_edges
        n_f = self._n_faces
        d1 = np.zeros((n_f, n_e))

        for f_idx, (i, j, k) in enumerate(self._face_list):
            # Face (i, j, k) with i < j < k
            # Boundary: edge ij + edge jk - edge ik
            # (or equivalently: ij + jk + ki where ki = -ik)

            # Edge (i, j) — canonical orientation i < j
            e_ij = self._edge_index[(i, j)]
            d1[f_idx, e_ij] = 1.0

            # Edge (j, k) — canonical orientation j < k
            e_jk = self._edge_index[(j, k)]
            d1[f_idx, e_jk] = 1.0

            # Edge (i, k) — canonical orientation i < k, but we need ki = -ik
            e_ik = self._edge_index[(i, k)]
            d1[f_idx, e_ik] = -1.0

        return d1

    # ==================================================================
    # Laplacians
    # ==================================================================

    def scalar_laplacian(self):
        """
        Build the graph Laplacian on vertices (scalars).

        L = D - A where D is the degree matrix and A is the adjacency matrix.
        Equivalently, L = d_0^T d_0 (the Hodge Laplacian on 0-forms).

        Returns
        -------
        ndarray of shape (120, 120): symmetric positive semi-definite
        """
        d0 = self.incidence_matrix()
        return d0.T @ d0

    def hodge_laplacian_1(self):
        """
        Hodge Laplacian on 1-forms (combinatorial).

        L_1 = L_down + L_up = d_0 d_0^T + d_1^T d_1

        where:
          d_0: C^0 -> C^1, shape (n_e, n_v) = (720, 120)
          d_1: C^1 -> C^2, shape (n_f, n_e) = (1200, 720)
          L_down = d_0 d_0^T: acts on exact forms, shape (720, 720)
          L_up = d_1^T d_1: acts on coexact forms, shape (720, 720)

        These two parts have orthogonal images (by d_1 d_0 = 0), so the spectrum
        decomposes cleanly into exact, coexact, and harmonic parts.

        THEOREM: On S^3, b_1 = 0, so there are no harmonic 1-forms.
        dim(exact) = rank(d_0) = n_v - 1 = 119
        dim(coexact) = rank(d_1) = n_e - rank(d_0) = 601

        Returns
        -------
        ndarray of shape (720, 720): symmetric positive semi-definite
        """
        d0 = self.incidence_matrix()   # (720, 120)
        d1 = self.face_boundary_matrix()  # (1200, 720)

        L_down = d0 @ d0.T   # exact part, shape (720, 720)
        L_up = d1.T @ d1     # coexact part, shape (720, 720)

        return L_down + L_up

    # ==================================================================
    # Full spectra
    # ==================================================================

    def scalar_spectrum_full(self):
        """
        Full eigenvalues and eigenvectors of the scalar Laplacian on the 600-cell.

        Returns
        -------
        eigenvalues : ndarray of shape (120,), sorted ascending
        eigenvectors : ndarray of shape (120, 120), columns are eigenvectors
        """
        L = self.scalar_laplacian()
        evals, evecs = la.eigh(L)
        return evals, evecs

    def oneform_spectrum_full(self):
        """
        Full eigenvalues and eigenvectors of the Hodge Laplacian on 1-forms.

        Returns
        -------
        eigenvalues : ndarray of shape (720,), sorted ascending
        eigenvectors : ndarray of shape (720, 720), columns are eigenvectors
        """
        L1 = self.hodge_laplacian_1()
        evals, evecs = la.eigh(L1)
        return evals, evecs

    # ==================================================================
    # Coexact decomposition
    # ==================================================================

    def _decompose_oneform_spectrum(self):
        """
        Decompose the 1-form spectrum into exact, coexact, and harmonic parts.

        Hodge decomposition: C^1 = image(d_0) + image(d_1^T) + harmonic
          - Exact: image(d_0), dim = rank(d_0) = n_v - 1 = 119
          - Coexact: image(d_1^T), dim = rank(d_1) = 601
          - Harmonic: ker(L_1), dim = b_1(S^3) = 0

        Strategy: classify each eigenvector of L_1 by checking which part
        of the Laplacian (L_down or L_up) contributes to its eigenvalue.

        Returns
        -------
        dict with:
          'exact_evals': eigenvalues of exact modes
          'exact_evecs': corresponding eigenvectors
          'coexact_evals': eigenvalues of coexact modes
          'coexact_evecs': corresponding eigenvectors
          'harmonic_evecs': harmonic eigenvectors (empty for S^3)
        """
        d0 = self.incidence_matrix()   # (720, 120)
        d1 = self.face_boundary_matrix()  # (1200, 720)

        L_down = d0 @ d0.T  # exact part
        L_up = d1.T @ d1    # coexact part

        # Get full spectrum
        evals, evecs = self.oneform_spectrum_full()

        # Classify each eigenvector
        exact_indices = []
        coexact_indices = []
        harmonic_indices = []
        tol = 1e-8

        for i in range(len(evals)):
            v = evecs[:, i]
            down_norm = np.linalg.norm(L_down @ v)
            up_norm = np.linalg.norm(L_up @ v)

            if evals[i] < tol:
                harmonic_indices.append(i)
            elif down_norm > tol and up_norm < tol:
                exact_indices.append(i)
            elif up_norm > tol and down_norm < tol:
                coexact_indices.append(i)
            else:
                # Numerically mixed -- assign by dominant component
                if down_norm > up_norm:
                    exact_indices.append(i)
                else:
                    coexact_indices.append(i)

        return {
            'exact_evals': evals[exact_indices],
            'exact_evecs': evecs[:, exact_indices],
            'coexact_evals': evals[coexact_indices],
            'coexact_evecs': evecs[:, coexact_indices],
            'harmonic_evecs': evecs[:, harmonic_indices],
        }

    def oneform_spectrum_coexact(self):
        """
        Coexact part of the 1-form spectrum on the 600-cell.

        The coexact eigenvalues approximate (k+1)^2/R^2 on S^3.

        Returns
        -------
        eigenvalues : ndarray, sorted ascending
        eigenvectors : ndarray, columns are eigenvectors
        """
        decomp = self._decompose_oneform_spectrum()
        return decomp['coexact_evals'], decomp['coexact_evecs']

    # ==================================================================
    # I*-invariant spectra
    # ==================================================================

    def scalar_spectrum_istar(self):
        """
        I*-invariant scalar spectrum.

        Computes eigenvalues of Pi L Pi restricted to image(Pi).

        THEOREM: Only eigenvalue 0 (the constant mode) survives for k=0..5
        on the 600-cell, since m(k)=0 for k=1..11 (Molien series of I*).

        Returns
        -------
        eigenvalues : ndarray of surviving eigenvalues
        rank : int, rank of the projector (should be 1)
        """
        L = self.scalar_laplacian()
        Pi = self.istar_projector_vertices()

        # Projected Laplacian
        PLP = Pi @ L @ Pi

        # Diagonalize the projector to find its image
        pi_evals, pi_evecs = la.eigh(Pi)
        # Image of Pi: eigenvectors with eigenvalue ~1
        image_mask = pi_evals > 0.5
        rank = int(np.sum(image_mask))
        V_inv = pi_evecs[:, image_mask]  # (120, rank)

        # Restrict Laplacian to the invariant subspace
        L_restricted = V_inv.T @ L @ V_inv  # (rank, rank)
        evals_inv = la.eigh(L_restricted, eigvals_only=True)

        return evals_inv, rank

    def oneform_spectrum_istar(self):
        """
        I*-invariant 1-form spectrum (all: exact + coexact + harmonic).

        Returns
        -------
        eigenvalues : ndarray of surviving eigenvalues
        rank : int, rank of edge projector
        """
        L1 = self.hodge_laplacian_1()
        Pi_e = self.istar_projector_edges()

        # Find image of Pi_e
        pi_evals, pi_evecs = la.eigh(Pi_e)
        image_mask = pi_evals > 0.5
        rank = int(np.sum(image_mask))
        V_inv = pi_evecs[:, image_mask]

        # Restrict Laplacian to invariant subspace
        L_restricted = V_inv.T @ L1 @ V_inv
        evals_inv = la.eigh(L_restricted, eigvals_only=True)

        return evals_inv, rank

    def coexact_spectrum_istar(self):
        """
        I*-invariant COEXACT 1-form spectrum only.
        This is the physically relevant spectrum for Yang-Mills.

        Strategy:
          1. Find the I*-invariant subspace V_inv (from edge projector)
          2. Restrict L1 to V_inv to get I*-invariant eigenvalues
          3. Classify each as exact or coexact using L_down and L_up

        THEOREM: All I*-invariant 1-forms on S^3 are coexact.
        This is because the only I*-invariant scalar is the constant (k=0),
        whose gradient is zero, so there are no I*-invariant exact 1-forms.

        Returns
        -------
        eigenvalues : ndarray of surviving coexact eigenvalues
        multiplicities : list of (eigenvalue, multiplicity) clusters
        """
        L1 = self.hodge_laplacian_1()
        Pi_e = self.istar_projector_edges()
        d0 = self.incidence_matrix()
        d1 = self.face_boundary_matrix()
        L_down = d0 @ d0.T   # exact part
        L_up = d1.T @ d1     # coexact part

        # Find I*-invariant subspace
        pi_evals, pi_evecs = la.eigh(Pi_e)
        image_mask = pi_evals > 0.5
        V_inv = pi_evecs[:, image_mask]  # (720, rank)

        # Restrict L1 to invariant subspace
        L_restricted = V_inv.T @ L1 @ V_inv
        evals_r, evecs_r = la.eigh(L_restricted)

        # Classify each invariant eigenvector as exact or coexact
        coexact_evals = []
        tol = 1e-8
        for i in range(len(evals_r)):
            v = V_inv @ evecs_r[:, i]  # back to full edge space
            down_comp = np.linalg.norm(L_down @ v)
            up_comp = np.linalg.norm(L_up @ v)
            if up_comp > tol or (down_comp < tol and up_comp < tol):
                # Coexact (or harmonic, which is also physical)
                coexact_evals.append(evals_r[i])

        coexact_evals = np.array(coexact_evals)
        multiplicities = _cluster_eigenvalues(coexact_evals, tol=0.5)

        return coexact_evals, multiplicities

    # ==================================================================
    # Verification report
    # ==================================================================

    def verification_report(self):
        """
        Generate a complete verification report.

        Returns
        -------
        dict with:
          scalar_spectrum: {eigenvalues, istar_eigenvalues, istar_rank}
          coexact_spectrum: {eigenvalues, clusters, istar_eigenvalues, istar_clusters}
          gap_preserved: bool
          sparsification_verified: bool
          lattice_info: {n_vertices, n_edges, n_faces, R}
        """
        R = self.R

        # --- Scalar ---
        scalar_evals, _ = self.scalar_spectrum_full()
        scalar_istar_evals, scalar_rank = self.scalar_spectrum_istar()
        scalar_clusters = _cluster_eigenvalues(scalar_evals, tol=0.5)

        # --- Coexact 1-forms ---
        coexact_evals, _ = self.oneform_spectrum_coexact()
        coexact_clusters = _cluster_eigenvalues(coexact_evals, tol=0.5)
        coexact_istar_evals, coexact_istar_clusters = self.coexact_spectrum_istar()

        # --- Gap preservation ---
        # The lowest coexact eigenvalue should be ~4/R^2
        # Check that it survives in the I*-invariant spectrum
        gap_preserved = False
        if len(coexact_istar_evals) > 0:
            lowest_istar = np.min(coexact_istar_evals)
            lowest_full = np.min(coexact_evals)
            # They should be approximately equal (lattice corrections)
            gap_preserved = abs(lowest_istar - lowest_full) / (lowest_full + 1e-15) < 0.1

        # --- Sparsification ---
        # Check that the second coexact cluster from the FULL spectrum
        # (k=2 analog) does NOT appear in the I*-invariant spectrum.
        # The second full coexact cluster is typically at ~1.146 for R=1.
        sparsification_verified = True
        if len(coexact_clusters) >= 2:
            target_k2_lattice = coexact_clusters[1][0]
            for ev in coexact_istar_evals:
                if abs(ev - target_k2_lattice) / (target_k2_lattice + 1e-15) < 0.3:
                    sparsification_verified = False
                    break

        return {
            'scalar_spectrum': {
                'eigenvalues': scalar_evals,
                'clusters': scalar_clusters,
                'istar_eigenvalues': scalar_istar_evals,
                'istar_rank': scalar_rank,
            },
            'coexact_spectrum': {
                'eigenvalues': coexact_evals,
                'clusters': coexact_clusters,
                'istar_eigenvalues': coexact_istar_evals,
                'istar_clusters': coexact_istar_clusters,
            },
            'gap_preserved': gap_preserved,
            'sparsification_verified': sparsification_verified,
            'lattice_info': {
                'n_vertices': self._n_vertices,
                'n_edges': self._n_edges,
                'n_faces': self._n_faces,
                'R': R,
            },
        }


# ======================================================================
# Module-level helpers
# ======================================================================

def _cluster_eigenvalues(evals, tol=0.5):
    """
    Cluster eigenvalues that are within tol of each other.

    Returns
    -------
    list of (mean_eigenvalue, count) pairs
    """
    if len(evals) == 0:
        return []

    sorted_evals = np.sort(evals)
    clusters = []
    current_cluster = [sorted_evals[0]]

    for i in range(1, len(sorted_evals)):
        if sorted_evals[i] - sorted_evals[i - 1] < tol:
            current_cluster.append(sorted_evals[i])
        else:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
            current_cluster = [sorted_evals[i]]

    clusters.append((np.mean(current_cluster), len(current_cluster)))
    return clusters
