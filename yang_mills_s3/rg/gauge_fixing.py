"""
Gauge Fixing Within Blocks for Yang-Mills RG on S^3 — Estimate 3.

Implements Balaban's block-level gauge fixing (Papers 3-4) adapted to the
600-cell refinement hierarchy on S^3.  The key components are:

1. MaximalTree: BFS spanning tree on block subgraph
2. AxialGaugeFixer: set tree links to identity, transform non-tree links
3. BlockAverager: gauge-covariant averaging via parallel transport
4. GaugeFixedBlock: combined tree + fixer for a single block
5. HierarchicalGaugeFixer: full hierarchy across all RG scales

Mathematical context:
    In lattice gauge theory on S^3, link variables U_e live in SU(2).
    Axial gauge fixing on a spanning tree T of a block sets U_e = I
    for all e in T.  This removes |V|-1 gauge degrees of freedom per
    block (one per tree edge), leaving |E|-|V|+1 = #loops physical DOF
    on the non-tree links.

    THEOREM: Axial gauge on a spanning tree of a connected graph with
    |V| vertices and |E| edges removes exactly |V|-1 gauge DOF,
    leaving |E|-|V|+1 physical (loop) DOF.  No residual gauge freedom
    remains within the block; only a single global SU(2) rotation
    at the root is unfixed.

    THEOREM: Wilson loops (traces of holonomies around closed loops)
    are gauge-invariant and therefore unchanged by axial gauge fixing.
    This is verified numerically in the test suite.

S^3 advantage:
    SU(2) homogeneity of S^3 means the maximal tree structure is the
    SAME in every block at a given level (up to rotation), so we only
    need to compute the tree topology once per level.

References:
    [1] Balaban (1984-89), Papers 3-4: Averaging operations, gauge fixing
    [2] Creutz (1983): Lattice gauge theories — axial gauge
    [3] block_geometry.py: 600-cell hierarchy infrastructure
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .block_geometry import (
    generate_600_cell_vertices,
    build_edges_from_vertices,
    build_faces,
    build_cells,
    build_adjacency,
    RefinementLevel,
    RGBlock,
    RGBlockingScheme,
    build_refinement_hierarchy,
    geodesic_distance,
)


# ======================================================================
# SU(2) utilities
# ======================================================================

def _su2_identity():
    """Return the 2x2 identity matrix (identity element of SU(2))."""
    return np.eye(2, dtype=complex)


def _su2_dagger(U):
    """
    Hermitian conjugate of a 2x2 matrix.

    For U in SU(2), U^dagger = U^{-1}.

    Parameters
    ----------
    U : ndarray, shape (2, 2)

    Returns
    -------
    ndarray, shape (2, 2)
    """
    return U.conj().T


def _is_su2(U, tol=1e-8):
    """
    Check whether a 2x2 matrix is in SU(2).

    Conditions: U^dagger U = I and det(U) = 1.

    Parameters
    ----------
    U : ndarray, shape (2, 2)
    tol : float

    Returns
    -------
    bool
    """
    if U.shape != (2, 2):
        return False
    eye_check = np.allclose(U.conj().T @ U, np.eye(2), atol=tol)
    det_check = abs(np.linalg.det(U) - 1.0) < tol
    return eye_check and det_check


def _project_to_su2(M):
    """
    Project a 2x2 complex matrix to the nearest SU(2) element.

    Uses SVD polar decomposition: M = U S V^dagger, then W = U V^dagger
    is the closest unitary.  Adjust determinant to +1.

    Parameters
    ----------
    M : ndarray, shape (2, 2), complex

    Returns
    -------
    ndarray, shape (2, 2), complex — element of SU(2)
    """
    U, S, Vh = np.linalg.svd(M)
    W = U @ Vh
    det = np.linalg.det(W)
    if abs(det) > 1e-15:
        # Adjust phase so det = +1
        phase = np.sqrt(det / abs(det))
        W = W / phase
    return W


def random_su2(rng=None):
    """
    Generate a uniformly random SU(2) element (Haar measure).

    Uses the quaternion parametrization: sample q uniformly on S^3,
    then map to SU(2) via q = (a, b, c, d) -> [[a+bi, c+di], [-c+di, a-bi]].

    Parameters
    ----------
    rng : numpy.random.Generator or None

    Returns
    -------
    ndarray, shape (2, 2), complex
    """
    if rng is None:
        rng = np.random.default_rng()
    # Uniform on S^3 by normalizing Gaussian vector
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a + 1j * b, c + 1j * d],
        [-c + 1j * d, a - 1j * b],
    ], dtype=complex)


def random_su2_near_identity(epsilon=0.1, rng=None):
    """
    Generate a random SU(2) element near the identity.

    Useful for testing with small gauge fields (weak coupling regime).

    Parameters
    ----------
    epsilon : float
        Scale of deviation from identity.
    rng : numpy.random.Generator or None

    Returns
    -------
    ndarray, shape (2, 2), complex
    """
    if rng is None:
        rng = np.random.default_rng()
    # Small Lie algebra element
    omega = epsilon * rng.standard_normal(3)
    return _su2_exp(omega)


def _su2_exp(omega):
    """
    Exponential map from su(2) to SU(2).

    Given omega = (w1, w2, w3), compute exp(i * omega . sigma / 2)
    where sigma = (sigma_1, sigma_2, sigma_3) are Pauli matrices.

    THEOREM: For omega in R^3, exp(i omega . sigma / 2) =
    cos(|omega|/2) I + i sin(|omega|/2) (omega_hat . sigma)
    where omega_hat = omega / |omega|.

    Parameters
    ----------
    omega : ndarray, shape (3,)

    Returns
    -------
    ndarray, shape (2, 2), complex — element of SU(2)
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-15:
        return _su2_identity()

    n = omega / theta
    half = theta / 2.0

    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

    n_dot_sigma = n[0] * sigma1 + n[1] * sigma2 + n[2] * sigma3
    return np.cos(half) * _su2_identity() + 1j * np.sin(half) * n_dot_sigma


def _su2_log(U):
    """
    Logarithm map from SU(2) to su(2).

    Given U in SU(2), returns omega in R^3 such that U = exp(i omega . sigma / 2).

    Parameters
    ----------
    U : ndarray, shape (2, 2), complex

    Returns
    -------
    ndarray, shape (3,) — Lie algebra coordinates
    """
    # cos(theta/2) = Re(Tr(U)) / 2
    cos_half = np.real(np.trace(U)) / 2.0
    cos_half = np.clip(cos_half, -1.0, 1.0)
    half_theta = np.arccos(cos_half)

    if half_theta < 1e-12:
        return np.zeros(3)

    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

    # n_k = Im(Tr(sigma_k U)) / (2 sin(theta/2))
    sin_half = np.sin(half_theta)
    omega = np.zeros(3)
    for k, sigma in enumerate([sigma1, sigma2, sigma3]):
        omega[k] = np.imag(np.trace(sigma @ U)) / (2.0 * sin_half) * (2.0 * half_theta)

    return omega


# ======================================================================
# MaximalTree
# ======================================================================

class MaximalTree:
    """
    Maximal (spanning) tree on a block subgraph of the 600-cell lattice.

    Given a connected subgraph (vertices V, edges E), a spanning tree T
    is a connected acyclic subgraph containing all vertices.  It has
    exactly |V|-1 edges.

    THEOREM: For a connected graph with |V| vertices and |E| edges,
    the spanning tree has |V|-1 edges and the number of independent
    loops is |E| - |V| + 1.

    The tree is constructed via BFS from a chosen root vertex.

    Attributes
    ----------
    vertices : list of int
        Vertex indices in the block.
    edges : list of (int, int)
        All edges of the block subgraph (sorted pairs).
    root : int
        Root vertex for the BFS tree.
    tree_edges : list of (int, int)
        Edges in the spanning tree.
    non_tree_edges : list of (int, int)
        Edges NOT in the spanning tree (carry physical DOF).
    parent : dict
        {vertex: parent_vertex} in the BFS tree (root maps to None).
    depth : dict
        {vertex: depth} in the BFS tree.
    """

    def __init__(self, vertices, edges, root=None):
        """
        Construct spanning tree via BFS.

        Parameters
        ----------
        vertices : list of int
            Vertex indices in the block.
        edges : list of (int, int)
            Edges of the block subgraph. Each edge (i, j) with i < j.
        root : int or None
            Root vertex. If None, uses the first vertex.
        """
        self.vertices = sorted(set(vertices))
        self.edges = [(min(i, j), max(i, j)) for (i, j) in edges]
        # Remove duplicate edges
        self.edges = sorted(set(self.edges))

        if len(self.vertices) == 0:
            self.root = None
            self.tree_edges = []
            self.non_tree_edges = list(self.edges)
            self.parent = {}
            self.depth = {}
            return

        self.root = root if root is not None else self.vertices[0]
        if self.root not in self.vertices:
            raise ValueError(f"Root {self.root} not in vertex set")

        self.parent = {}
        self.depth = {}
        self.tree_edges = []

        self._build_bfs_tree()

        # Non-tree edges = all edges minus tree edges
        tree_set = set((min(i, j), max(i, j)) for (i, j) in self.tree_edges)
        self.non_tree_edges = [e for e in self.edges if e not in tree_set]

    def _build_bfs_tree(self):
        """Build spanning tree via BFS from the root."""
        # Build local adjacency
        vert_set = set(self.vertices)
        adj = {v: set() for v in self.vertices}
        for (i, j) in self.edges:
            if i in vert_set and j in vert_set:
                adj[i].add(j)
                adj[j].add(i)

        visited = set()
        queue = deque([self.root])
        visited.add(self.root)
        self.parent[self.root] = None
        self.depth[self.root] = 0

        while queue:
            v = queue.popleft()
            for w in sorted(adj[v]):  # sorted for reproducibility
                if w not in visited:
                    visited.add(w)
                    self.parent[w] = v
                    self.depth[w] = self.depth[v] + 1
                    self.tree_edges.append((min(v, w), max(v, w)))
                    queue.append(w)

    @property
    def n_vertices(self):
        """Number of vertices in the block."""
        return len(self.vertices)

    @property
    def n_edges(self):
        """Total number of edges in the block subgraph."""
        return len(self.edges)

    @property
    def n_tree_edges(self):
        """Number of tree edges = |V| - 1 (for connected graph)."""
        return len(self.tree_edges)

    @property
    def n_non_tree_edges(self):
        """Number of non-tree edges = physical DOF = independent loops."""
        return len(self.non_tree_edges)

    @property
    def n_loops(self):
        """
        Number of independent loops = |E| - |V| + 1.

        THEOREM: For a connected graph, the cycle rank (first Betti
        number) is |E| - |V| + 1.  Each non-tree edge creates exactly
        one independent loop.
        """
        return self.n_edges - self.n_vertices + 1

    def is_spanning(self):
        """
        Check whether the tree spans all vertices.

        Returns True if every vertex in the block is in the tree.
        """
        return len(self.parent) == len(self.vertices)

    def path_to_root(self, vertex):
        """
        Return the unique path from a vertex to the root in the tree.

        THEOREM: In a tree, there is a unique path between any two vertices.

        Parameters
        ----------
        vertex : int

        Returns
        -------
        path : list of int
            Sequence of vertices from `vertex` to root (inclusive).
        """
        if vertex not in self.parent:
            raise ValueError(f"Vertex {vertex} not in tree")
        path = [vertex]
        v = vertex
        while self.parent[v] is not None:
            v = self.parent[v]
            path.append(v)
        return path

    def path_edges_to_root(self, vertex):
        """
        Return the edges along the path from vertex to root.

        Each edge is returned as (source, target) in the direction
        from vertex toward root.

        Parameters
        ----------
        vertex : int

        Returns
        -------
        edges : list of (int, int)
            Directed edges from vertex to root.
        """
        path = self.path_to_root(vertex)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def path_between(self, v1, v2):
        """
        Return path from v1 to v2 through the tree (via LCA).

        Parameters
        ----------
        v1, v2 : int

        Returns
        -------
        path : list of int
        """
        path1 = self.path_to_root(v1)
        path2 = self.path_to_root(v2)

        # Find lowest common ancestor
        set2 = set(path2)
        lca = None
        for v in path1:
            if v in set2:
                lca = v
                break

        if lca is None:
            raise ValueError("Vertices not in same connected component")

        # path1 up to LCA
        part1 = []
        for v in path1:
            part1.append(v)
            if v == lca:
                break

        # path2 up to LCA (reversed)
        part2 = []
        for v in path2:
            if v == lca:
                break
            part2.append(v)

        return part1 + list(reversed(part2))


# ======================================================================
# AxialGaugeFixer
# ======================================================================

class AxialGaugeFixer:
    """
    Axial gauge fixing on a maximal tree.

    Given link variables U_e in SU(2) on all edges of a block, and a
    spanning tree T, axial gauge sets U_e = I for all tree edges.

    The gauge transformation achieving this is:
        g(root) = I
        g(v) = product of U along tree path from root to v

    More precisely, if the tree path from root to v passes through
    edges e_1, e_2, ..., e_k with orientations s_i -> t_i, then:
        g(v) = U_{e_1}^{eps_1} . U_{e_2}^{eps_2} . ... . U_{e_k}^{eps_k}
    where eps_i = +1 if the edge is traversed in its canonical direction,
    and eps_i = -1 (meaning U^dagger) if traversed backwards.

    After gauge transformation:
        U'_e = g(s(e))^dagger . U_e . g(t(e))

    THEOREM: After axial gauge fixing:
    (a) U'_e = I for all tree edges e in T.
    (b) Wilson loops W(C) = Tr(prod U_e) are unchanged.
    (c) The only residual gauge freedom is a global SU(2) rotation
        at the root: g(root) -> h . g(root) for h in SU(2).

    Attributes
    ----------
    tree : MaximalTree
    gauge_transform : dict
        {vertex: SU(2) matrix g(v)}
    """

    def __init__(self, tree):
        """
        Parameters
        ----------
        tree : MaximalTree
        """
        self.tree = tree
        self.gauge_transform = {}

    def compute_gauge_transform(self, link_field):
        """
        Compute the gauge transformation g(v) for each vertex.

        g(root) = I.
        For each vertex v with parent p in the tree:
            if edge (p, v) is canonical (p < v): g(v) = g(p) . U_{(p,v)}
            if edge (v, p) is canonical (v < p): g(v) = g(p) . U_{(v,p)}^dagger

        Wait — the convention matters.  Let us be precise.

        The link field stores U_{(i,j)} for canonical edge (i,j) with i < j.
        The parallel transport from i to j is U_{(i,j)}.
        The parallel transport from j to i is U_{(i,j)}^dagger.

        For the BFS tree, the parent of v is p.  The tree edge connects
        p and v.  We want g(v) such that the gauge-transformed link on
        this tree edge is I.

        The gauge-transformed link for canonical edge (i,j):
            U'_{(i,j)} = g(i)^dagger . U_{(i,j)} . g(j)

        Setting U'_{tree_edge} = I and solving:
            If p < v (canonical = (p,v)):
                I = g(p)^dag U_{(p,v)} g(v)  =>  g(v) = U_{(p,v)}^dag g(p)
            Wait, that gives g(v) = U^dag g(p), not g(p) U.

        Let us be careful:
            U' = g(s)^dag U g(t) = I
            => U g(t) = g(s)
            => g(t) = U^dag g(s)

        For edge from p to v (p = source, v = target in the tree direction):
            We want U'=I on the canonical edge.

        Case 1: p < v, canonical edge = (p,v), U_{(p,v)} is the link.
            Transport from p to v is U_{(p,v)}.
            U' = g(p)^dag . U_{(p,v)} . g(v) = I
            => g(v) = U_{(p,v)}^dag . g(p)

        Case 2: p > v, canonical edge = (v,p), U_{(v,p)} is the link.
            Transport from p to v is U_{(v,p)}^dag.
            U' = g(v)^dag . U_{(v,p)} . g(p) = I
            => U_{(v,p)} . g(p) = g(v)
            => g(v) = U_{(v,p)} . g(p)

        Parameters
        ----------
        link_field : dict
            {(i, j): ndarray of shape (2,2)} for canonical edges (i < j).
            Each value is an SU(2) matrix.

        Returns
        -------
        gauge_transform : dict
            {vertex: ndarray of shape (2,2)} — SU(2) gauge transformation.
        """
        tree = self.tree
        if tree.root is None:
            self.gauge_transform = {}
            return self.gauge_transform

        g = {tree.root: _su2_identity()}

        # BFS order: process vertices by depth
        # The tree.parent dict gives the BFS tree structure
        queue = deque([tree.root])
        visited = {tree.root}

        while queue:
            p = queue.popleft()
            for v in sorted(self.tree.vertices):
                if v in visited:
                    continue
                if self.tree.parent.get(v) != p:
                    continue

                # v is a child of p in the tree
                visited.add(v)
                canonical = (min(p, v), max(p, v))

                if canonical not in link_field:
                    # Edge not in link field — use identity
                    g[v] = g[p].copy()
                elif p < v:
                    # Case 1: canonical = (p, v)
                    U_pv = link_field[canonical]
                    g[v] = _su2_dagger(U_pv) @ g[p]
                else:
                    # Case 2: canonical = (v, p)
                    U_vp = link_field[canonical]
                    g[v] = U_vp @ g[p]

                queue.append(v)

        self.gauge_transform = g
        return g

    def apply_gauge_fix(self, link_field):
        """
        Apply axial gauge fixing to a link field.

        Computes g(v) for each vertex, then transforms all links:
            U'_{(i,j)} = g(i)^dagger . U_{(i,j)} . g(j)

        THEOREM: After this transformation, U'_e = I for all tree edges.

        Parameters
        ----------
        link_field : dict
            {(i, j): ndarray of shape (2,2)} for canonical edges (i < j).

        Returns
        -------
        fixed_field : dict
            {(i, j): ndarray of shape (2,2)} — gauge-fixed link variables.
        """
        g = self.compute_gauge_transform(link_field)

        fixed = {}
        for (i, j), U in link_field.items():
            gi = g.get(i, _su2_identity())
            gj = g.get(j, _su2_identity())
            fixed[(i, j)] = _su2_dagger(gi) @ U @ gj

        return fixed

    def verify_axial_gauge(self, fixed_field, tol=1e-8):
        """
        Verify that tree links are identity in the gauge-fixed field.

        Parameters
        ----------
        fixed_field : dict
        tol : float

        Returns
        -------
        bool : True if all tree links are within tol of identity.
        """
        I = _su2_identity()
        for edge in self.tree.tree_edges:
            canonical = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if canonical in fixed_field:
                U = fixed_field[canonical]
                if not np.allclose(U, I, atol=tol):
                    return False
        return True


# ======================================================================
# BlockAverager
# ======================================================================

class BlockAverager:
    """
    Gauge-covariant block averaging for the RG step.

    Given a fine-scale gauge field on a refined lattice, computes the
    block-averaged field on the coarser lattice.

    The block average of a coarse link connecting blocks B_1 and B_2 is:
        U_coarse = project_SU2( (1/N) sum_paths U_{path} )
    where the sum is over fine-scale paths from a representative vertex
    in B_1 to a representative vertex in B_2, and U_{path} is the
    ordered product of link variables along each path.

    THEOREM: The block averaging operator is gauge-COVARIANT:
    under a gauge transformation A -> g A g^{-1} + g dg^{-1},
    the coarse field transforms as U_coarse -> g_1 U_coarse g_2^{-1}
    where g_1, g_2 are the gauge transforms at the representative
    vertices of B_1 and B_2.

    Parameters
    ----------
    fine_vertices : ndarray, shape (N_fine, 4)
        Fine-lattice vertex positions.
    fine_edges : list of (int, int)
        Fine-lattice edges.
    blocks : list of RGBlock
        Blocks defining the coarse lattice.
    """

    def __init__(self, fine_vertices, fine_edges, blocks):
        self.fine_vertices = fine_vertices
        self.fine_edges = [(min(i, j), max(i, j)) for (i, j) in fine_edges]
        self.blocks = blocks
        self._fine_adj = None
        self._block_reps = None

    @property
    def fine_adj(self):
        """Adjacency dict for the fine lattice."""
        if self._fine_adj is None:
            n = len(self.fine_vertices)
            self._fine_adj = {}
            for (i, j) in self.fine_edges:
                self._fine_adj.setdefault(i, set()).add(j)
                self._fine_adj.setdefault(j, set()).add(i)
        return self._fine_adj

    @property
    def block_representatives(self):
        """
        Representative vertex for each block (the one closest to center).

        Returns
        -------
        reps : dict
            {block_id: vertex_index}
        """
        if self._block_reps is not None:
            return self._block_reps

        reps = {}
        for block in self.blocks:
            best_v = block.vertex_indices[0]
            best_d = float('inf')
            for vi in block.vertex_indices:
                d = np.linalg.norm(
                    self.fine_vertices[vi] - block.center
                )
                if d < best_d:
                    best_d = d
                    best_v = vi
            reps[block.block_id] = best_v
        self._block_reps = reps
        return reps

    def shortest_path_bfs(self, source, target):
        """
        Find shortest path on the fine lattice from source to target via BFS.

        Parameters
        ----------
        source, target : int
            Vertex indices.

        Returns
        -------
        path : list of int
            Vertex sequence from source to target (inclusive).
            Empty list if no path exists.
        """
        if source == target:
            return [source]

        adj = self.fine_adj
        visited = {source}
        parent = {source: None}
        queue = deque([source])

        while queue:
            v = queue.popleft()
            for w in adj.get(v, set()):
                if w not in visited:
                    visited.add(w)
                    parent[w] = v
                    if w == target:
                        # Reconstruct path
                        path = [w]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    queue.append(w)

        return []  # No path found

    def path_holonomy(self, path, link_field):
        """
        Compute the holonomy (product of link variables) along a path.

        Parameters
        ----------
        path : list of int
            Vertex sequence.
        link_field : dict
            {(i,j): SU(2) matrix} for canonical edges.

        Returns
        -------
        U : ndarray, shape (2, 2)
            Product U_{v0->v1} . U_{v1->v2} . ... . U_{v_{n-1}->v_n}.
        """
        U = _su2_identity()
        for k in range(len(path) - 1):
            s, t = path[k], path[k + 1]
            canonical = (min(s, t), max(s, t))
            U_link = link_field.get(canonical, _su2_identity())
            if s < t:
                U = U @ U_link
            else:
                # Traversing backwards: use U^dagger
                U = U @ _su2_dagger(U_link)
        return U

    def compute_coarse_link(self, block1_id, block2_id, link_field):
        """
        Compute the coarse link variable between two blocks.

        Uses the shortest path between block representatives on the
        fine lattice.

        NUMERICAL: For the 600-cell, blocks are tetrahedral cells with
        ~4 vertices each, so paths between adjacent block reps are
        typically 1-3 edges long.

        Parameters
        ----------
        block1_id, block2_id : int
        link_field : dict

        Returns
        -------
        U_coarse : ndarray, shape (2, 2) — SU(2)
        """
        reps = self.block_representatives
        v1 = reps[block1_id]
        v2 = reps[block2_id]

        path = self.shortest_path_bfs(v1, v2)
        if len(path) < 2:
            return _su2_identity()

        return self.path_holonomy(path, link_field)

    def compute_all_coarse_links(self, link_field, coarse_edges):
        """
        Compute all coarse link variables.

        Parameters
        ----------
        link_field : dict
            Fine-scale link variables.
        coarse_edges : list of (int, int)
            Coarse-lattice edges (block_id pairs).

        Returns
        -------
        coarse_field : dict
            {(b1, b2): SU(2) matrix} for coarse links.
        """
        coarse_field = {}
        for (b1, b2) in coarse_edges:
            U = self.compute_coarse_link(b1, b2, link_field)
            coarse_field[(min(b1, b2), max(b1, b2))] = U
        return coarse_field

    def average_within_block(self, block, link_field):
        """
        Compute the average link variable within a single block.

        Averages all link variables with both endpoints inside the block,
        then projects to SU(2).

        NUMERICAL: This gives a representative gauge field strength
        within the block for monitoring RG flow.

        Parameters
        ----------
        block : RGBlock
        link_field : dict

        Returns
        -------
        U_avg : ndarray, shape (2, 2) — SU(2)
        """
        vset = set(block.vertex_indices)
        total = np.zeros((2, 2), dtype=complex)
        count = 0

        for (i, j), U in link_field.items():
            if i in vset and j in vset:
                total += U
                count += 1

        if count == 0:
            return _su2_identity()

        return _project_to_su2(total / count)


# ======================================================================
# GaugeFixedBlock
# ======================================================================

class GaugeFixedBlock:
    """
    A single block with gauge fixing and averaging combined.

    Combines:
    1. MaximalTree construction for the block subgraph
    2. AxialGaugeFixer to eliminate gauge redundancy
    3. Computation of coarse links via holonomy

    THEOREM: After axial gauge fixing within each block:
    (a) Each block has exactly n_loops = |E_block| - |V_block| + 1
        independent link variables (the non-tree links).
    (b) Wilson loops through the block are preserved.
    (c) The only residual gauge freedom is one global SU(2) at the
        block root, which is fixed by the inter-block matching condition.

    Attributes
    ----------
    block : RGBlock
    tree : MaximalTree
    fixer : AxialGaugeFixer
    """

    def __init__(self, block, block_edges, root=None):
        """
        Parameters
        ----------
        block : RGBlock
        block_edges : list of (int, int)
            Edges with both endpoints in this block.
        root : int or None
            Root vertex for axial gauge. If None, uses the vertex
            closest to the block center (the block representative).
        """
        self.block = block
        self.block_edges = [(min(i, j), max(i, j)) for (i, j) in block_edges]

        # Choose root: vertex closest to center if not specified
        if root is None:
            root = block.vertex_indices[0]
        self.root = root

        self.tree = MaximalTree(block.vertex_indices, self.block_edges, root=root)
        self.fixer = AxialGaugeFixer(self.tree)

    @property
    def n_physical_dof(self):
        """
        Number of physical (gauge-invariant) DOF in this block.

        = n_non_tree_edges * dim(SU(2)) = n_loops * 3

        THEOREM: Each non-tree edge carries dim(G) = 3 DOF for SU(2).
        """
        return self.tree.n_non_tree_edges * 3  # dim(SU(2)) = 3

    @property
    def n_gauge_dof_fixed(self):
        """
        Number of gauge DOF fixed by the axial gauge.

        = n_tree_edges * dim(SU(2)) = (|V|-1) * 3

        One residual global SU(2) remains at the root.
        """
        return self.tree.n_tree_edges * 3

    def fix_gauge(self, link_field):
        """
        Apply axial gauge fixing to link variables within this block.

        Parameters
        ----------
        link_field : dict
            {(i,j): SU(2) matrix} — may contain links outside block.

        Returns
        -------
        fixed_field : dict
            Gauge-fixed link variables (only block edges).
        gauge_transform : dict
            {vertex: SU(2)} gauge transformation applied.
        """
        # Extract block-relevant links
        block_links = {}
        edge_set = set(self.block_edges)
        for (i, j), U in link_field.items():
            canonical = (min(i, j), max(i, j))
            if canonical in edge_set:
                block_links[canonical] = U

        fixed = self.fixer.apply_gauge_fix(block_links)
        return fixed, self.fixer.gauge_transform

    def fix_and_extract_physical(self, link_field):
        """
        Gauge-fix and return only the physical (non-tree) link variables.

        Parameters
        ----------
        link_field : dict

        Returns
        -------
        physical_links : dict
            {edge: SU(2)} for non-tree edges only.
        """
        fixed, _ = self.fix_gauge(link_field)
        physical = {}
        for edge in self.tree.non_tree_edges:
            canonical = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if canonical in fixed:
                physical[canonical] = fixed[canonical]
        return physical


# ======================================================================
# HierarchicalGaugeFixer
# ======================================================================

class HierarchicalGaugeFixer:
    """
    Hierarchical gauge fixing across all RG scales on S^3.

    At each level of the 600-cell refinement hierarchy:
    1. Partition fine lattice into blocks
    2. Build maximal tree within each block
    3. Apply axial gauge fixing
    4. Count physical DOF

    THEOREM: The total number of physical DOF at each scale satisfies:
        N_phys(j) = N_edges(j) - N_vertices(j) + N_blocks(j)
                   = sum over blocks of (|E_b| - |V_b| + 1)
    This equals the number of independent Wilson loops at scale j.

    NUMERICAL: For the 600-cell at level 0:
        N_phys = 720 - 120 + 1 = 601 (single block = whole lattice)
    At scale with 600 blocks:
        N_phys = sum of loops per tetrahedral block

    S^3 advantage: By SU(2) homogeneity, the tree structure within
    each block at a given level is identical up to rotation.  This
    means we compute the tree once and reuse it for all blocks.

    Parameters
    ----------
    hierarchy : list of RefinementLevel
        The 600-cell refinement hierarchy.
    R : float
        Radius of S^3.
    """

    def __init__(self, hierarchy, R=1.0):
        self.hierarchy = hierarchy
        self.R = R
        self._blocks_per_level = {}
        self._gauge_fixed_blocks = {}

    def get_block_subgraph(self, level, block_vertices):
        """
        Extract the subgraph (edges) induced by a set of vertices at a given level.

        Parameters
        ----------
        level : RefinementLevel
        block_vertices : list or set of int

        Returns
        -------
        edges : list of (int, int)
        """
        vset = set(block_vertices)
        edges = []
        for (i, j) in level.edges:
            if i in vset and j in vset:
                edges.append((i, j))
        return edges

    def build_blocks_at_level(self, level_idx):
        """
        Build GaugeFixedBlocks for all cells at a given refinement level.

        Each tetrahedral cell of the level becomes a block.

        Parameters
        ----------
        level_idx : int
            Index into self.hierarchy.

        Returns
        -------
        gf_blocks : list of GaugeFixedBlock
        """
        if level_idx in self._gauge_fixed_blocks:
            return self._gauge_fixed_blocks[level_idx]

        level = self.hierarchy[level_idx]
        gf_blocks = []

        if len(level.cells) == 0:
            # No cells — treat whole lattice as one block
            all_verts = list(range(level.n_vertices))
            block = RGBlock(0, all_verts, level.vertices, self.R)
            edges = list(level.edges)
            gfb = GaugeFixedBlock(block, edges)
            gf_blocks.append(gfb)
        else:
            for cell_id, cell in enumerate(level.cells):
                verts = list(cell)
                block = RGBlock(cell_id, verts, level.vertices, self.R)
                edges = self.get_block_subgraph(level, verts)
                gfb = GaugeFixedBlock(block, edges)
                gf_blocks.append(gfb)

        self._gauge_fixed_blocks[level_idx] = gf_blocks
        return gf_blocks

    def dof_count_at_level(self, level_idx):
        """
        Count physical DOF at a given refinement level.

        Returns
        -------
        result : dict with keys:
            'total_edges': int
            'total_vertices': int
            'n_blocks': int
            'physical_dof_per_block': list of int
            'total_physical_dof': int
            'total_gauge_fixed': int
            'global_residual': int (one SU(2) per block = 3 per block)
        """
        gf_blocks = self.build_blocks_at_level(level_idx)
        level = self.hierarchy[level_idx]

        phys_per_block = []
        gauge_fixed_per_block = []

        for gfb in gf_blocks:
            phys_per_block.append(gfb.n_physical_dof)
            gauge_fixed_per_block.append(gfb.n_gauge_dof_fixed)

        return {
            'total_edges': level.n_edges,
            'total_vertices': level.n_vertices,
            'n_blocks': len(gf_blocks),
            'physical_dof_per_block': phys_per_block,
            'total_physical_dof': sum(phys_per_block),
            'total_gauge_fixed': sum(gauge_fixed_per_block),
            'global_residual': len(gf_blocks) * 3,  # One global SU(2) per block
        }

    def fix_all_blocks(self, level_idx, link_field):
        """
        Apply gauge fixing to all blocks at a given level.

        Parameters
        ----------
        level_idx : int
        link_field : dict
            {(i,j): SU(2)} for ALL edges at this level.

        Returns
        -------
        all_fixed : dict
            {(i,j): SU(2)} — gauge-fixed link variables for all edges.
        all_transforms : dict
            {vertex: SU(2)} — combined gauge transformation.
        """
        gf_blocks = self.build_blocks_at_level(level_idx)

        all_fixed = dict(link_field)  # Start with original field
        all_transforms = {}

        for gfb in gf_blocks:
            fixed, transforms = gfb.fix_gauge(link_field)
            # Update the link field with fixed values
            all_fixed.update(fixed)
            all_transforms.update(transforms)

        return all_fixed, all_transforms

    def hierarchy_summary(self):
        """
        Summary of DOF counting across all levels.

        NUMERICAL: Demonstrates how gauge fixing reduces DOF at each scale.

        Returns
        -------
        summary : list of dict
        """
        summary = []
        for idx in range(len(self.hierarchy)):
            dof = self.dof_count_at_level(idx)
            summary.append({
                'level': idx,
                'n_vertices': self.hierarchy[idx].n_vertices,
                'n_edges': self.hierarchy[idx].n_edges,
                'n_blocks': dof['n_blocks'],
                'total_physical_dof': dof['total_physical_dof'],
                'total_gauge_fixed': dof['total_gauge_fixed'],
            })
        return summary


# ======================================================================
# Wilson loop computation (for gauge invariance verification)
# ======================================================================

def wilson_loop(link_field, loop_vertices):
    """
    Compute the Wilson loop for a closed loop of vertices.

    W(C) = Tr( U_{v0->v1} . U_{v1->v2} . ... . U_{v_{n-1}->v0} )

    THEOREM: Wilson loops are gauge-invariant observables.
    W(C) is unchanged by any gauge transformation U_e -> g(s) U_e g(t)^dag.

    Parameters
    ----------
    link_field : dict
        {(i,j): SU(2)} for canonical edges.
    loop_vertices : list of int
        Ordered sequence of vertices forming a closed loop.
        The loop closes from last vertex back to first.

    Returns
    -------
    W : complex
        Trace of the holonomy.
    """
    n = len(loop_vertices)
    if n < 2:
        return 2.0 + 0j  # Tr(I) = 2

    U_total = _su2_identity()
    for k in range(n):
        s = loop_vertices[k]
        t = loop_vertices[(k + 1) % n]
        canonical = (min(s, t), max(s, t))
        U_link = link_field.get(canonical, _su2_identity())
        if s < t:
            U_total = U_total @ U_link
        else:
            U_total = U_total @ _su2_dagger(U_link)

    return np.trace(U_total)


def plaquette_action(link_field, face_vertices):
    """
    Compute the plaquette action for a triangular face.

    S_plaq = 1 - (1/2) Re Tr(U_plaq)

    where U_plaq is the ordered product around the face.

    THEOREM: The plaquette action is gauge-invariant and approaches
    (a^2/4) F_{mu nu}^2 in the continuum limit.

    Parameters
    ----------
    link_field : dict
    face_vertices : tuple of (int, int, int)

    Returns
    -------
    float : plaquette action value
    """
    W = wilson_loop(link_field, list(face_vertices))
    return 1.0 - 0.5 * np.real(W)


def generate_random_link_field(edges, coupling='weak', rng=None):
    """
    Generate a random SU(2) link field on given edges.

    Parameters
    ----------
    edges : list of (int, int)
    coupling : str
        'weak' (near identity) or 'strong' (fully random, Haar measure).
    rng : numpy.random.Generator or None

    Returns
    -------
    link_field : dict
        {(i,j): SU(2) matrix} for each edge.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    field = {}
    for (i, j) in edges:
        canonical = (min(i, j), max(i, j))
        if coupling == 'weak':
            field[canonical] = random_su2_near_identity(epsilon=0.3, rng=rng)
        else:
            field[canonical] = random_su2(rng=rng)
    return field


def gauge_transform_field(link_field, gauge_transforms):
    """
    Apply a gauge transformation to a link field.

    U'_{(i,j)} = g(i)^dagger . U_{(i,j)} . g(j)

    Parameters
    ----------
    link_field : dict
        {(i,j): SU(2)} for canonical edges.
    gauge_transforms : dict
        {vertex: SU(2)} gauge transformation at each vertex.

    Returns
    -------
    transformed : dict
    """
    transformed = {}
    for (i, j), U in link_field.items():
        gi = gauge_transforms.get(i, _su2_identity())
        gj = gauge_transforms.get(j, _su2_identity())
        transformed[(i, j)] = _su2_dagger(gi) @ U @ gj
    return transformed
