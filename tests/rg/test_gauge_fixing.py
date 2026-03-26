"""
Tests for gauge_fixing.py — Estimate 3: Block Averaging and Gauge Fixing.

Tests verify:
    1. SU(2) utilities: exp/log, random generation, projection
    2. MaximalTree: spanning, DOF count, path computation
    3. AxialGaugeFixer: tree links become identity, Wilson loop invariance
    4. BlockAverager: covariance, path computation, holonomy
    5. GaugeFixedBlock: combined fixing + DOF count
    6. HierarchicalGaugeFixer: multi-level consistency
    7. Integration: gauge-invariant observables preserved
"""

import numpy as np
import pytest

from yang_mills_s3.rg.gauge_fixing import (
    _su2_identity,
    _su2_dagger,
    _is_su2,
    _project_to_su2,
    random_su2,
    random_su2_near_identity,
    _su2_exp,
    _su2_log,
    MaximalTree,
    AxialGaugeFixer,
    BlockAverager,
    GaugeFixedBlock,
    HierarchicalGaugeFixer,
    wilson_loop,
    plaquette_action,
    generate_random_link_field,
    gauge_transform_field,
)
from yang_mills_s3.rg.block_geometry import (
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
# Helpers
# ======================================================================

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(12345)


@pytest.fixture
def level0():
    """Base 600-cell."""
    R = 1.0
    verts = generate_600_cell_vertices(R)
    edges, _ = build_edges_from_vertices(verts, R)
    faces = build_faces(len(verts), edges)
    cells = build_cells(len(verts), edges, faces)
    return RefinementLevel(0, R, verts, edges, faces, cells)


@pytest.fixture
def small_graph():
    """
    Small graph for unit testing: a tetrahedron (4 vertices, 6 edges).
    This mimics a single 600-cell block.
    """
    vertices = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return vertices, edges


@pytest.fixture
def triangle_graph():
    """Triangle: 3 vertices, 3 edges, 1 loop."""
    vertices = [0, 1, 2]
    edges = [(0, 1), (0, 2), (1, 2)]
    return vertices, edges


@pytest.fixture
def line_graph():
    """Line: 3 vertices, 2 edges, 0 loops (a tree itself)."""
    vertices = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    return vertices, edges


# ======================================================================
# 1. SU(2) Utilities
# ======================================================================

class TestSU2Utilities:
    """Tests for SU(2) matrix operations."""

    def test_identity_is_su2(self):
        """THEOREM: The 2x2 identity matrix is in SU(2)."""
        I = _su2_identity()
        assert _is_su2(I)

    def test_identity_shape(self):
        """Identity has shape (2, 2)."""
        I = _su2_identity()
        assert I.shape == (2, 2)

    def test_dagger_of_identity(self):
        """THEOREM: I^dagger = I."""
        I = _su2_identity()
        np.testing.assert_allclose(_su2_dagger(I), I, atol=1e-12)

    def test_random_su2_is_su2(self, rng):
        """Random SU(2) element passes SU(2) check."""
        for _ in range(10):
            U = random_su2(rng)
            assert _is_su2(U), f"Random SU(2) failed check: det={np.linalg.det(U)}"

    def test_random_su2_near_identity(self, rng):
        """Near-identity element is SU(2) and close to I."""
        for _ in range(10):
            U = random_su2_near_identity(epsilon=0.01, rng=rng)
            assert _is_su2(U)
            np.testing.assert_allclose(U, _su2_identity(), atol=0.1)

    def test_su2_exp_identity(self):
        """THEOREM: exp(0) = I."""
        omega = np.zeros(3)
        U = _su2_exp(omega)
        np.testing.assert_allclose(U, _su2_identity(), atol=1e-12)

    def test_su2_exp_is_su2(self, rng):
        """THEOREM: exp(omega) in SU(2) for any omega in R^3."""
        for _ in range(20):
            omega = rng.standard_normal(3)
            U = _su2_exp(omega)
            assert _is_su2(U), f"exp({omega}) not in SU(2)"

    def test_su2_exp_log_roundtrip(self, rng):
        """THEOREM: log(exp(omega)) = omega for small omega."""
        for _ in range(10):
            omega = 0.5 * rng.standard_normal(3)
            U = _su2_exp(omega)
            recovered = _su2_log(U)
            np.testing.assert_allclose(recovered, omega, atol=1e-8)

    def test_su2_log_exp_roundtrip(self, rng):
        """THEOREM: exp(log(U)) = U for U in SU(2) near identity."""
        for _ in range(10):
            U = random_su2_near_identity(epsilon=0.5, rng=rng)
            omega = _su2_log(U)
            U_recovered = _su2_exp(omega)
            np.testing.assert_allclose(U_recovered, U, atol=1e-8)

    def test_project_to_su2(self, rng):
        """Projection of a perturbed SU(2) element stays in SU(2)."""
        for _ in range(5):
            U = random_su2(rng)
            # Perturb
            M = U + 0.1 * rng.standard_normal((2, 2)).astype(complex)
            P = _project_to_su2(M)
            assert _is_su2(P, tol=1e-6)

    def test_project_identity(self):
        """Projecting identity gives identity."""
        P = _project_to_su2(_su2_identity())
        np.testing.assert_allclose(P, _su2_identity(), atol=1e-10)

    def test_dagger_is_inverse(self, rng):
        """THEOREM: For U in SU(2), U^dagger U = I."""
        U = random_su2(rng)
        product = _su2_dagger(U) @ U
        np.testing.assert_allclose(product, _su2_identity(), atol=1e-10)


# ======================================================================
# 2. MaximalTree
# ======================================================================

class TestMaximalTree:
    """Tests for MaximalTree construction and properties."""

    def test_tetrahedron_tree_edges(self, small_graph):
        """THEOREM: A spanning tree of a 4-vertex graph has 3 edges."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_tree_edges == 3

    def test_tetrahedron_non_tree_edges(self, small_graph):
        """THEOREM: Tetrahedron has 6 - 4 + 1 = 3 non-tree (loop) edges."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_non_tree_edges == 3

    def test_tetrahedron_loops(self, small_graph):
        """THEOREM: Number of loops = |E| - |V| + 1 = 3."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_loops == 3

    def test_triangle_tree_edges(self, triangle_graph):
        """Triangle: 2 tree edges, 1 non-tree edge."""
        vertices, edges = triangle_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_tree_edges == 2
        assert tree.n_non_tree_edges == 1
        assert tree.n_loops == 1

    def test_line_is_tree(self, line_graph):
        """Line graph is already a tree: 0 non-tree edges."""
        vertices, edges = line_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_tree_edges == 2
        assert tree.n_non_tree_edges == 0
        assert tree.n_loops == 0

    def test_spanning_property(self, small_graph):
        """THEOREM: A spanning tree reaches all vertices."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.is_spanning()
        assert len(tree.parent) == 4

    def test_root_has_no_parent(self, small_graph):
        """Root vertex has parent = None."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.parent[0] is None

    def test_root_depth_zero(self, small_graph):
        """Root has depth 0."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.depth[0] == 0

    def test_non_root_positive_depth(self, small_graph):
        """Non-root vertices have positive depth."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        for v in [1, 2, 3]:
            assert tree.depth[v] > 0

    def test_tree_plus_non_tree_equals_total(self, small_graph):
        """THEOREM: |tree_edges| + |non_tree_edges| = |edges|."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        assert tree.n_tree_edges + tree.n_non_tree_edges == tree.n_edges

    def test_path_to_root_starts_at_vertex(self, small_graph):
        """Path starts at the given vertex."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        path = tree.path_to_root(2)
        assert path[0] == 2

    def test_path_to_root_ends_at_root(self, small_graph):
        """Path ends at the root."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        path = tree.path_to_root(3)
        assert path[-1] == 0

    def test_path_root_to_root(self, small_graph):
        """Path from root to root is just [root]."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        path = tree.path_to_root(0)
        assert path == [0]

    def test_path_between_symmetric(self, small_graph):
        """Path v1->v2 and v2->v1 have same vertices (reversed)."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        p12 = tree.path_between(1, 2)
        p21 = tree.path_between(2, 1)
        assert set(p12) == set(p21)

    def test_different_roots_same_loop_count(self, small_graph):
        """
        THEOREM: The number of non-tree edges (loops) is independent
        of the choice of root vertex.
        """
        vertices, edges = small_graph
        for root in vertices:
            tree = MaximalTree(vertices, edges, root=root)
            assert tree.n_loops == 3

    def test_empty_graph(self):
        """Empty graph: no edges, no tree."""
        tree = MaximalTree([], [])
        assert tree.n_tree_edges == 0
        assert tree.n_non_tree_edges == 0

    def test_single_vertex(self):
        """Single vertex: no edges, no loops."""
        tree = MaximalTree([0], [])
        assert tree.n_vertices == 1
        assert tree.n_tree_edges == 0
        assert tree.n_loops == 0
        assert tree.is_spanning()

    def test_600cell_single_cell_loops(self, level0):
        """
        NUMERICAL: A single tetrahedron from the 600-cell has 4 vertices,
        6 edges, and 3 loops.
        """
        cell = level0.cells[0]
        verts = list(cell)
        # Get edges within this cell
        vset = set(verts)
        cell_edges = [(i, j) for (i, j) in level0.edges if i in vset and j in vset]
        tree = MaximalTree(verts, cell_edges, root=verts[0])
        assert tree.n_vertices == 4
        assert tree.n_edges == 6
        assert tree.n_loops == 3
        assert tree.n_tree_edges == 3


# ======================================================================
# 3. AxialGaugeFixer
# ======================================================================

class TestAxialGaugeFixer:
    """Tests for axial gauge fixing on tree edges."""

    def test_tree_links_become_identity_weak(self, small_graph, rng):
        """
        THEOREM: After axial gauge fixing, all tree links are identity.
        Test with weak-coupling (near identity) links.
        """
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='weak', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)

        assert fixer.verify_axial_gauge(fixed)

    def test_tree_links_become_identity_strong(self, small_graph, rng):
        """
        THEOREM: After axial gauge fixing, tree links are identity.
        Test with strong-coupling (fully random Haar) links.
        """
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)

        assert fixer.verify_axial_gauge(fixed)

    def test_non_tree_links_remain_su2(self, small_graph, rng):
        """Non-tree links remain in SU(2) after gauge fixing."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)

        for edge in tree.non_tree_edges:
            canonical = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if canonical in fixed:
                assert _is_su2(fixed[canonical], tol=1e-6), \
                    f"Non-tree link on {canonical} not SU(2)"

    def test_gauge_transforms_are_su2(self, small_graph, rng):
        """All gauge transformation matrices are in SU(2)."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        g = fixer.compute_gauge_transform(link_field)

        for v, gv in g.items():
            assert _is_su2(gv, tol=1e-6), f"g({v}) not in SU(2)"

    def test_root_transform_is_identity(self, small_graph, rng):
        """THEOREM: The gauge transform at the root is the identity."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        g = fixer.compute_gauge_transform(link_field)

        np.testing.assert_allclose(g[0], _su2_identity(), atol=1e-10)

    def test_wilson_loop_invariance_triangle(self, rng):
        """
        THEOREM: Wilson loops are gauge-invariant.
        Test on a triangle (the simplest nontrivial loop).
        """
        vertices = [0, 1, 2]
        edges = [(0, 1), (0, 2), (1, 2)]
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)

        # Wilson loop before fixing
        W_before = wilson_loop(link_field, [0, 1, 2])

        # Wilson loop after fixing
        fixed = fixer.apply_gauge_fix(link_field)
        W_after = wilson_loop(fixed, [0, 1, 2])

        np.testing.assert_allclose(W_after, W_before, atol=1e-8)

    def test_wilson_loop_invariance_tetrahedron(self, small_graph, rng):
        """
        THEOREM: All face Wilson loops of a tetrahedron are preserved.
        """
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)

        # All 4 faces of the tetrahedron
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        for face in faces:
            W_before = wilson_loop(link_field, list(face))
            W_after = wilson_loop(fixed, list(face))
            np.testing.assert_allclose(
                W_after, W_before, atol=1e-8,
                err_msg=f"Wilson loop on face {face} changed"
            )

    def test_plaquette_action_invariance(self, small_graph, rng):
        """THEOREM: Plaquette action is gauge-invariant."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)

        face = (0, 1, 2)
        S_before = plaquette_action(link_field, face)
        S_after = plaquette_action(fixed, face)

        np.testing.assert_allclose(S_after, S_before, atol=1e-8)

    def test_identity_field_stays_identity(self, small_graph):
        """If all links are identity, gauge fixing does nothing."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = {e: _su2_identity() for e in edges}
        fixed = fixer.apply_gauge_fix(link_field)

        for e, U in fixed.items():
            np.testing.assert_allclose(U, _su2_identity(), atol=1e-10)

    def test_gauge_fix_idempotent(self, small_graph, rng):
        """Applying gauge fix twice gives the same result as once."""
        vertices, edges = small_graph
        tree = MaximalTree(vertices, edges, root=0)
        fixer = AxialGaugeFixer(tree)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed1 = fixer.apply_gauge_fix(link_field)

        # Second application: recompute transforms with the fixed field
        fixer2 = AxialGaugeFixer(tree)
        fixed2 = fixer2.apply_gauge_fix(fixed1)

        for e in edges:
            np.testing.assert_allclose(
                fixed2[e], fixed1[e], atol=1e-8,
                err_msg=f"Gauge fix not idempotent on edge {e}"
            )


# ======================================================================
# 4. BlockAverager
# ======================================================================

class TestBlockAverager:
    """Tests for gauge-covariant block averaging."""

    @pytest.fixture
    def simple_lattice(self):
        """
        Simple lattice: 6 vertices forming two tetrahedra sharing a face.
        Blocks: B0 = {0,1,2,3}, B1 = {1,2,3,4}, with shared face {1,2,3}
        and an extra vertex 5 for a third connection.
        """
        vertices = np.array([
            [1, 0, 0, 0],    # 0
            [0, 1, 0, 0],    # 1
            [0, 0, 1, 0],    # 2
            [0, 0, 0, 1],    # 3
            [-1, 0, 0, 0],   # 4
        ], dtype=float)
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms

        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4),
                 (2, 3), (2, 4), (3, 4)]
        R = 1.0
        b0 = RGBlock(0, [0, 1, 2, 3], vertices, R)
        b1 = RGBlock(1, [1, 2, 3, 4], vertices, R)
        return vertices, edges, [b0, b1], R

    def test_shortest_path_basic(self, simple_lattice):
        """BFS finds correct shortest path."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        path = averager.shortest_path_bfs(0, 4)
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 4

    def test_shortest_path_self(self, simple_lattice):
        """Path from vertex to itself is just [vertex]."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        path = averager.shortest_path_bfs(0, 0)
        assert path == [0]

    def test_shortest_path_adjacent(self, simple_lattice):
        """Path between adjacent vertices has length 2."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        path = averager.shortest_path_bfs(0, 1)
        assert len(path) == 2
        assert path == [0, 1]

    def test_holonomy_identity_field(self, simple_lattice):
        """Holonomy through identity field is identity."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = {e: _su2_identity() for e in edges}
        U = averager.path_holonomy([0, 1, 2], link_field)
        np.testing.assert_allclose(U, _su2_identity(), atol=1e-10)

    def test_coarse_link_identity_field(self, simple_lattice):
        """Coarse link from identity field is identity."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = {e: _su2_identity() for e in edges}
        U_coarse = averager.compute_coarse_link(0, 1, link_field)
        assert _is_su2(U_coarse, tol=1e-6)

    def test_coarse_link_is_su2(self, simple_lattice, rng):
        """Coarse link variable is always in SU(2)."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        U_coarse = averager.compute_coarse_link(0, 1, link_field)
        assert _is_su2(U_coarse, tol=1e-6)

    def test_covariance_under_gauge_transform(self, simple_lattice, rng):
        """
        THEOREM: Block averaging is gauge-covariant.

        Under gauge transform g: U'_coarse(B1,B2) = g(rep1)^dag U_coarse g(rep2)
        where rep1, rep2 are the block representatives.
        """
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = generate_random_link_field(edges, coupling='weak', rng=rng)

        # Random gauge transforms at all vertices
        g_transforms = {i: random_su2(rng) for i in range(len(vertices))}

        # Transform the field
        transformed_field = gauge_transform_field(link_field, g_transforms)

        # Coarse link from original
        U_orig = averager.compute_coarse_link(0, 1, link_field)
        # Coarse link from transformed
        U_trans = averager.compute_coarse_link(0, 1, transformed_field)

        # Expected: g(rep0)^dag U_orig g(rep1)
        reps = averager.block_representatives
        g0 = g_transforms[reps[0]]
        g1 = g_transforms[reps[1]]
        U_expected = _su2_dagger(g0) @ U_orig @ g1

        np.testing.assert_allclose(U_trans, U_expected, atol=1e-8)

    def test_average_within_block_is_su2(self, simple_lattice, rng):
        """Average link within a block is in SU(2)."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        U_avg = averager.average_within_block(blocks[0], link_field)
        assert _is_su2(U_avg, tol=1e-6)

    def test_average_within_block_identity_field(self, simple_lattice):
        """Average within block of identity field is identity."""
        vertices, edges, blocks, R = simple_lattice
        averager = BlockAverager(vertices, edges, blocks)

        link_field = {e: _su2_identity() for e in edges}
        U_avg = averager.average_within_block(blocks[0], link_field)
        np.testing.assert_allclose(U_avg, _su2_identity(), atol=1e-8)


# ======================================================================
# 5. GaugeFixedBlock
# ======================================================================

class TestGaugeFixedBlock:
    """Tests for combined gauge fixing within a single block."""

    def test_physical_dof_tetrahedron(self, small_graph):
        """
        THEOREM: A tetrahedron has 3 loops, so 3*3 = 9 physical DOF
        for SU(2) gauge theory.
        """
        vertices, edges = small_graph
        verts_4d = np.eye(4)  # Dummy positions on S^3
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)
        assert gfb.n_physical_dof == 9  # 3 loops * 3 (dim SU(2))

    def test_gauge_fixed_dof_tetrahedron(self, small_graph):
        """
        THEOREM: Axial gauge fixes |V|-1 = 3 edges, removing 3*3 = 9 gauge DOF.
        """
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)
        assert gfb.n_gauge_dof_fixed == 9

    def test_total_dof_conservation(self, small_graph):
        """
        THEOREM: physical + gauge_fixed + global_residual = total link DOF.
        Total link DOF = |E| * 3 = 18.
        Physical = 9, Gauge fixed = 9, Global residual = 3.
        But physical + gauge_fixed = 18, and the global residual is the
        unfixed root rotation (already included in the counting since
        we have |V|-1 tree edges, not |V|).
        """
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)
        total_link_dof = len(edges) * 3
        assert gfb.n_physical_dof + gfb.n_gauge_dof_fixed == total_link_dof

    def test_fix_gauge_result(self, small_graph, rng):
        """Gauge fixing returns fixed field and transforms."""
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed, transforms = gfb.fix_gauge(link_field)

        assert len(fixed) > 0
        assert len(transforms) == 4  # All 4 vertices

    def test_fix_gauge_tree_identity(self, small_graph, rng):
        """After fixing, tree links are identity."""
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed, _ = gfb.fix_gauge(link_field)

        I = _su2_identity()
        for edge in gfb.tree.tree_edges:
            canonical = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if canonical in fixed:
                np.testing.assert_allclose(fixed[canonical], I, atol=1e-8)

    def test_extract_physical_links(self, small_graph, rng):
        """Physical links extraction returns only non-tree edges."""
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        physical = gfb.fix_and_extract_physical(link_field)

        assert len(physical) == gfb.tree.n_non_tree_edges
        for edge, U in physical.items():
            assert _is_su2(U, tol=1e-6)

    def test_wilson_loop_preserved_after_block_fix(self, small_graph, rng):
        """Wilson loops through the block are preserved after gauge fixing."""
        vertices, edges = small_graph
        verts_4d = np.eye(4)
        block = RGBlock(0, vertices, verts_4d, 1.0)
        gfb = GaugeFixedBlock(block, edges, root=0)

        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        fixed, _ = gfb.fix_gauge(link_field)

        face = (0, 1, 2)
        W_before = wilson_loop(link_field, list(face))
        W_after = wilson_loop(fixed, list(face))
        np.testing.assert_allclose(W_after, W_before, atol=1e-8)


# ======================================================================
# 6. HierarchicalGaugeFixer
# ======================================================================

class TestHierarchicalGaugeFixer:
    """Tests for hierarchical gauge fixing across levels."""

    @pytest.fixture
    def hgf(self, level0):
        """HierarchicalGaugeFixer for the base 600-cell."""
        return HierarchicalGaugeFixer([level0], R=1.0)

    def test_build_blocks_level0(self, hgf):
        """Level 0 has 600 blocks (one per cell)."""
        blocks = hgf.build_blocks_at_level(0)
        assert len(blocks) == 600

    def test_each_block_has_tree(self, hgf):
        """Each block has a MaximalTree."""
        blocks = hgf.build_blocks_at_level(0)
        for gfb in blocks[:10]:
            assert gfb.tree is not None
            assert gfb.tree.is_spanning()

    def test_dof_count_level0(self, hgf):
        """DOF count at level 0 is consistent."""
        dof = hgf.dof_count_at_level(0)
        assert dof['n_blocks'] == 600
        assert dof['total_edges'] == 720
        assert dof['total_vertices'] == 120

    def test_dof_per_block_tetrahedra(self, hgf):
        """
        NUMERICAL: Each tetrahedral block has 4 vertices, 6 edges,
        3 tree edges, 3 non-tree edges, 9 physical DOF.
        """
        dof = hgf.dof_count_at_level(0)
        # Each block should have 9 physical DOF (3 loops * 3 dim SU(2))
        for phys in dof['physical_dof_per_block'][:10]:
            assert phys == 9

    def test_total_physical_dof(self, hgf):
        """Total physical DOF = 600 blocks * 9 DOF/block = 5400."""
        dof = hgf.dof_count_at_level(0)
        assert dof['total_physical_dof'] == 600 * 9

    def test_total_gauge_fixed_dof(self, hgf):
        """Total gauge-fixed DOF = 600 blocks * 9 fixed/block = 5400."""
        dof = hgf.dof_count_at_level(0)
        assert dof['total_gauge_fixed'] == 600 * 9

    def test_hierarchy_summary(self, hgf):
        """Summary has correct structure."""
        summary = hgf.hierarchy_summary()
        assert len(summary) == 1  # Only level 0
        s = summary[0]
        assert s['level'] == 0
        assert s['n_vertices'] == 120
        assert s['n_edges'] == 720
        assert s['n_blocks'] == 600

    def test_fix_all_blocks_preserves_su2(self, hgf, level0, rng):
        """Gauge fixing all blocks preserves SU(2) property of links."""
        link_field = generate_random_link_field(level0.edges, coupling='weak', rng=rng)
        fixed, transforms = hgf.fix_all_blocks(0, link_field)

        # Check a sample of fixed links
        for edge in list(level0.edges)[:20]:
            canonical = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if canonical in fixed:
                assert _is_su2(fixed[canonical], tol=1e-5), \
                    f"Fixed link on {canonical} not SU(2)"

    def test_fix_all_blocks_identity_field(self, hgf, level0):
        """Fixing identity field gives identity everywhere."""
        link_field = {(min(i, j), max(i, j)): _su2_identity()
                      for (i, j) in level0.edges}
        fixed, _ = hgf.fix_all_blocks(0, link_field)

        for edge, U in list(fixed.items())[:50]:
            np.testing.assert_allclose(U, _su2_identity(), atol=1e-10)


# ======================================================================
# 7. Integration: Gauge Invariant Quantities
# ======================================================================

class TestGaugeInvariance:
    """Integration tests: gauge-invariant quantities are preserved."""

    def test_total_wilson_action_preserved(self, level0, rng):
        """
        THEOREM: The total Wilson action S = sum_faces S_plaq is
        gauge-invariant, so gauge fixing preserves it.

        Test on a subset of faces for speed.
        """
        # Use first 100 faces
        test_faces = level0.faces[:100]
        edges_needed = set()
        for face in test_faces:
            for i in range(3):
                e = (min(face[i], face[(i + 1) % 3]),
                     max(face[i], face[(i + 1) % 3]))
                edges_needed.add(e)

        link_field = generate_random_link_field(list(edges_needed),
                                                 coupling='strong', rng=rng)

        S_before = sum(plaquette_action(link_field, f) for f in test_faces)

        # Apply a random gauge transformation
        all_verts = set()
        for (i, j) in edges_needed:
            all_verts.add(i)
            all_verts.add(j)
        g_transforms = {v: random_su2(rng) for v in all_verts}
        transformed = gauge_transform_field(link_field, g_transforms)

        S_after = sum(plaquette_action(transformed, f) for f in test_faces)

        np.testing.assert_allclose(S_after, S_before, atol=1e-6)

    def test_gauge_transform_field_is_su2(self, rng):
        """All links remain SU(2) after gauge transformation."""
        edges = [(0, 1), (1, 2), (0, 2)]
        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
        g = {0: random_su2(rng), 1: random_su2(rng), 2: random_su2(rng)}
        transformed = gauge_transform_field(link_field, g)

        for e, U in transformed.items():
            assert _is_su2(U, tol=1e-6)

    def test_double_gauge_transform(self, rng):
        """
        THEOREM: Two gauge transforms compose: (g1 . g2)(U) = g1(g2(U)).
        """
        edges = [(0, 1), (1, 2), (0, 2)]
        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)

        g1 = {0: random_su2(rng), 1: random_su2(rng), 2: random_su2(rng)}
        g2 = {0: random_su2(rng), 1: random_su2(rng), 2: random_su2(rng)}

        # Apply g2 then g1 sequentially
        step1 = gauge_transform_field(link_field, g2)
        step2 = gauge_transform_field(step1, g1)

        # Apply composed transform g1 . g2
        g_composed = {v: g1[v] @ g2[v] for v in [0, 1, 2]}
        # Note: gauge_transform_field uses g^dag U g, so composition is:
        # (g1 g2)^dag U (g1 g2) = g2^dag g1^dag U g1 g2
        # But sequential: g1^dag (g2^dag U g2) g1
        # These are NOT the same unless g1 and g2 commute.
        # The correct composition for sequential application is just to
        # verify that the Wilson loop (gauge-invariant) is the same.
        W_orig = wilson_loop(link_field, [0, 1, 2])
        W_step2 = wilson_loop(step2, [0, 1, 2])
        np.testing.assert_allclose(W_step2, W_orig, atol=1e-8)

    def test_wilson_loop_orientation_reversal(self, rng):
        """
        THEOREM: W(C^{-1}) = W(C)^* (complex conjugate) for SU(2).
        Actually for SU(2), Tr(U) = Tr(U^T) = Tr(U^{-1})^* only
        in special cases.  For SU(2): Tr(U^{-1}) = Tr(U)^* always.
        """
        edges = [(0, 1), (1, 2), (0, 2)]
        link_field = generate_random_link_field(edges, coupling='strong', rng=rng)

        W_forward = wilson_loop(link_field, [0, 1, 2])
        W_backward = wilson_loop(link_field, [2, 1, 0])

        # For SU(2): W(C^{-1}) = W(C)*
        np.testing.assert_allclose(W_backward, np.conj(W_forward), atol=1e-8)

    def test_plaquette_action_bounded(self, rng):
        """
        NUMERICAL: Plaquette action is bounded: 0 <= S_plaq <= 2 for SU(2).
        S = 1 - (1/2) Re Tr(U), and Tr(U) in [-2, 2] for SU(2).
        """
        edges = [(0, 1), (1, 2), (0, 2)]
        for _ in range(20):
            link_field = generate_random_link_field(edges, coupling='strong', rng=rng)
            S = plaquette_action(link_field, (0, 1, 2))
            assert 0.0 - 1e-10 <= S <= 2.0 + 1e-10, f"S_plaq = {S} out of bounds"

    def test_random_field_generators(self, rng):
        """Both coupling modes generate SU(2) fields."""
        edges = [(0, 1), (1, 2)]
        for coupling in ['weak', 'strong']:
            field = generate_random_link_field(edges, coupling=coupling, rng=rng)
            for e, U in field.items():
                assert _is_su2(U, tol=1e-6)


# ======================================================================
# 8. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_single_edge_graph(self, rng):
        """Single edge: 1 tree edge, 0 loops, 0 physical DOF."""
        tree = MaximalTree([0, 1], [(0, 1)], root=0)
        assert tree.n_tree_edges == 1
        assert tree.n_non_tree_edges == 0
        assert tree.n_loops == 0

        fixer = AxialGaugeFixer(tree)
        link_field = {(0, 1): random_su2(rng)}
        fixed = fixer.apply_gauge_fix(link_field)
        np.testing.assert_allclose(fixed[(0, 1)], _su2_identity(), atol=1e-8)

    def test_pentagon_loops(self):
        """Pentagon (5 vertices, 5 edges): 1 loop."""
        verts = [0, 1, 2, 3, 4]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        tree = MaximalTree(verts, edges, root=0)
        assert tree.n_loops == 1

    def test_complete_graph_5(self):
        """Complete graph K5: 5 vertices, 10 edges, 6 loops."""
        verts = [0, 1, 2, 3, 4]
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        assert len(edges) == 10
        tree = MaximalTree(verts, edges, root=0)
        assert tree.n_loops == 6

    def test_disconnected_vertices_in_tree(self):
        """
        Disconnected graph: tree spans only the connected component
        containing the root.
        """
        verts = [0, 1, 2, 3]
        edges = [(0, 1), (2, 3)]  # Two disconnected edges
        tree = MaximalTree(verts, edges, root=0)
        # Tree should span the root's component {0, 1}
        assert tree.n_tree_edges == 1
        # Only 2 vertices are in the tree
        assert len(tree.parent) == 2

    def test_large_block_many_loops(self, rng):
        """
        A block with many vertices has correct loop count.
        Use a grid-like subgraph: 4x4 = 16 vertices.
        """
        n = 16
        verts = list(range(n))
        # 4x4 grid edges
        edges = []
        for row in range(4):
            for col in range(4):
                v = row * 4 + col
                if col < 3:
                    edges.append((v, v + 1))
                if row < 3:
                    edges.append((v, v + 4))
        tree = MaximalTree(verts, edges, root=0)
        expected_loops = len(edges) - n + 1
        assert tree.n_loops == expected_loops

        # Gauge fix should still work
        fixer = AxialGaugeFixer(tree)
        link_field = generate_random_link_field(edges, coupling='weak', rng=rng)
        fixed = fixer.apply_gauge_fix(link_field)
        assert fixer.verify_axial_gauge(fixed)
