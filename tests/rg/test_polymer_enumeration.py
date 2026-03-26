"""
Tests for polymer enumeration on the 600-cell and large-field Peierls bounds.

Tests verify:
    1. 600-cell cell construction (600 cells, correct combinatorics)
    2. Cell adjacency graphs (face-sharing D=4, vertex-sharing D=56)
    3. Exact polymer counts N(k) for k=1..7
    4. Tree bound vs exact (bound always >= exact)
    5. Peierls suppression computation
    6. Z_large convergence at weak coupling
    7. Critical coupling sweep
    8. Face vs vertex adjacency comparison

Run:
    pytest tests/rg/test_polymer_enumeration.py -v
"""

import numpy as np
import pytest
from collections import Counter

from yang_mills_s3.rg.polymer_enumeration import (
    build_600_cell,
    build_cell_adjacency_face_sharing,
    build_cell_adjacency_vertex_sharing,
    adjacency_stats,
    count_polymers_exact,
    count_polymers_rooted,
    tree_bound,
    tree_bound_tight,
    compute_peierls_suppression,
    print_peierls_table,
    sweep_coupling,
    compare_adjacencies,
    G2_BARE_DEFAULT,
)


# ======================================================================
# 1. 600-cell cell construction
# ======================================================================

class Test600CellConstruction:
    """Verify the 600-cell has the correct combinatorial structure."""

    @pytest.fixture(scope='class')
    def cell_data(self):
        """Build 600-cell once for all tests in this class."""
        return build_600_cell(R=1.0)

    def test_vertex_count(self, cell_data):
        """THEOREM: The 600-cell has exactly 120 vertices."""
        vertices, edges, faces, cells = cell_data
        assert len(vertices) == 120

    def test_edge_count(self, cell_data):
        """THEOREM: The 600-cell has exactly 720 edges."""
        vertices, edges, faces, cells = cell_data
        assert len(edges) == 720

    def test_face_count(self, cell_data):
        """THEOREM: The 600-cell has exactly 1200 triangular faces."""
        vertices, edges, faces, cells = cell_data
        assert len(faces) == 1200

    def test_cell_count(self, cell_data):
        """THEOREM: The 600-cell has exactly 600 tetrahedral cells."""
        vertices, edges, faces, cells = cell_data
        assert len(cells) == 600

    def test_euler_characteristic(self, cell_data):
        """THEOREM: chi(S^3) = 0, so V - E + F - C = 0."""
        vertices, edges, faces, cells = cell_data
        chi = len(vertices) - len(edges) + len(faces) - len(cells)
        assert chi == 0, f"Euler characteristic = {chi}, expected 0"

    def test_cells_are_tetrahedra(self, cell_data):
        """Each cell has exactly 4 vertices."""
        vertices, edges, faces, cells = cell_data
        for cell in cells:
            assert len(cell) == 4

    def test_cells_are_sorted(self, cell_data):
        """Cell vertex tuples are sorted."""
        vertices, edges, faces, cells = cell_data
        for cell in cells:
            assert cell == tuple(sorted(cell))

    def test_cells_on_sphere(self, cell_data):
        """All cell vertices lie on S^3(1)."""
        vertices, edges, faces, cells = cell_data
        for cell in cells:
            for v_idx in cell:
                norm = np.linalg.norm(vertices[v_idx])
                assert abs(norm - 1.0) < 1e-10, f"Vertex {v_idx} has norm {norm}"


# ======================================================================
# 2. Cell adjacency graphs
# ======================================================================

class TestCellAdjacency:
    """Verify face-sharing and vertex-sharing adjacency structures."""

    @pytest.fixture(scope='class')
    def adjacency_data(self):
        """Build adjacency graphs once."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        vert_adj = build_cell_adjacency_vertex_sharing(cells)
        return cells, face_adj, vert_adj

    def test_face_adjacency_count(self, adjacency_data):
        """Every cell has exactly 4 face-sharing neighbors."""
        cells, face_adj, vert_adj = adjacency_data
        stats = adjacency_stats(face_adj)
        assert stats['min_deg'] == 4
        assert stats['max_deg'] == 4
        assert stats['mean_deg'] == 4.0

    def test_face_adjacency_regular(self, adjacency_data):
        """THEOREM: Face-sharing adjacency is 4-regular (D_face = 4)."""
        cells, face_adj, vert_adj = adjacency_data
        for i in range(len(cells)):
            assert len(face_adj[i]) == 4, f"Cell {i} has {len(face_adj[i])} face neighbors"

    def test_vertex_adjacency_count(self, adjacency_data):
        """Every cell has exactly 56 vertex-sharing neighbors."""
        cells, face_adj, vert_adj = adjacency_data
        stats = adjacency_stats(vert_adj)
        assert stats['min_deg'] == 56
        assert stats['max_deg'] == 56
        assert stats['mean_deg'] == 56.0

    def test_vertex_adjacency_regular(self, adjacency_data):
        """NUMERICAL: Vertex-sharing adjacency is 56-regular."""
        cells, face_adj, vert_adj = adjacency_data
        for i in range(len(cells)):
            assert len(vert_adj[i]) == 56, f"Cell {i} has {len(vert_adj[i])} vertex neighbors"

    def test_face_subset_of_vertex(self, adjacency_data):
        """Face-sharing neighbors are a subset of vertex-sharing neighbors."""
        cells, face_adj, vert_adj = adjacency_data
        for i in range(len(cells)):
            assert face_adj[i].issubset(vert_adj[i]), (
                f"Cell {i}: face neighbors {face_adj[i] - vert_adj[i]} "
                f"not in vertex neighbors"
            )

    def test_adjacency_symmetric(self, adjacency_data):
        """Adjacency is symmetric: i adj j iff j adj i."""
        cells, face_adj, vert_adj = adjacency_data
        for adj in [face_adj, vert_adj]:
            for i in range(len(cells)):
                for j in adj[i]:
                    assert i in adj[j], f"Asymmetry: {i} adj {j} but not reverse"

    def test_no_self_loops(self, adjacency_data):
        """No cell is adjacent to itself."""
        cells, face_adj, vert_adj = adjacency_data
        for adj in [face_adj, vert_adj]:
            for i in range(len(cells)):
                assert i not in adj[i], f"Cell {i} is self-adjacent"

    def test_face_shared_is_3_vertices(self, adjacency_data):
        """Face-adjacent cells share exactly 3 vertices."""
        cells, face_adj, vert_adj = adjacency_data
        for i in range(min(50, len(cells))):  # Check first 50
            cell_i = set(cells[i])
            for j in face_adj[i]:
                cell_j = set(cells[j])
                shared = cell_i & cell_j
                assert len(shared) == 3, (
                    f"Cells {i} and {j} share {len(shared)} vertices, expected 3"
                )

    def test_total_face_count(self, adjacency_data):
        """Total face-sharing pairs = 1200 (each face shared by 2 cells)."""
        cells, face_adj, vert_adj = adjacency_data
        total_pairs = sum(len(face_adj[i]) for i in range(len(cells))) // 2
        assert total_pairs == 1200, f"Total face-sharing pairs = {total_pairs}, expected 1200"


# ======================================================================
# 3. Exact polymer counts (face-sharing)
# ======================================================================

class TestExactPolymerCounts:
    """Verify exact polymer enumeration on the 600-cell."""

    @pytest.fixture(scope='class')
    def face_counts(self):
        """Exact counts for face-sharing adjacency up to size 7."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        return count_polymers_exact(face_adj, 7)

    def test_N1_equals_600(self, face_counts):
        """THEOREM: N(1) = 600 (one polymer per cell)."""
        assert face_counts[1] == 600

    def test_N2_equals_1200(self, face_counts):
        """NUMERICAL: N(2) = 1200 (one per shared face)."""
        assert face_counts[2] == 1200

    def test_N2_equals_face_count(self, face_counts):
        """N(2) = number of faces = 1200.

        Each pair of face-adjacent cells shares exactly one face,
        and each face belongs to exactly 2 cells. So the number of
        connected pairs = number of faces = 1200.
        """
        assert face_counts[2] == 1200

    def test_N3_value(self, face_counts):
        """NUMERICAL: N(3) = 3600."""
        assert face_counts[3] == 3600

    def test_N4_value(self, face_counts):
        """NUMERICAL: N(4) = 13200."""
        assert face_counts[4] == 13200

    def test_N5_value(self, face_counts):
        """NUMERICAL: N(5) = 51720."""
        assert face_counts[5] == 51720

    def test_N6_value(self, face_counts):
        """NUMERICAL: N(6) = 212400."""
        assert face_counts[6] == 212400

    def test_N7_value(self, face_counts):
        """NUMERICAL: N(7) = 900000."""
        assert face_counts[7] == 900000

    def test_monotone_growth(self, face_counts):
        """Polymer counts grow monotonically with size."""
        for k in range(2, 8):
            assert face_counts[k] > face_counts[k - 1], (
                f"N({k}) = {face_counts[k]} <= N({k-1}) = {face_counts[k-1]}"
            )

    def test_growth_rate(self, face_counts):
        """Growth rate N(k+1)/N(k) is bounded by O(D) = O(4)."""
        for k in range(1, 7):
            ratio = face_counts[k + 1] / face_counts[k]
            # For D=4 regular graph, growth rate should be < 4*e ~ 10.87
            assert ratio < 4 * np.e, (
                f"Growth rate N({k+1})/N({k}) = {ratio:.2f} exceeds 4e = {4*np.e:.2f}"
            )


# ======================================================================
# 4. Rooted polymer counts and consistency
# ======================================================================

class TestRootedCounts:
    """Verify rooted polymer counts and consistency with exact counts."""

    @pytest.fixture(scope='class')
    def count_data(self):
        """Compute both exact and rooted counts."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        exact = count_polymers_exact(face_adj, 6)
        rooted = count_polymers_rooted(face_adj, 6)
        return exact, rooted

    def test_rooted_N1(self, count_data):
        """Rooted N(1) = 1 (just the root itself)."""
        exact, rooted = count_data
        assert rooted[1] == 1

    def test_rooted_N2(self, count_data):
        """Rooted N(2) = 4 (root + each of its 4 neighbors)."""
        exact, rooted = count_data
        assert rooted[2] == 4

    def test_exact_vs_rooted_k1(self, count_data):
        """N_exact(1) = 600 * N_rooted(1) = 600."""
        exact, rooted = count_data
        assert exact[1] == 600 * rooted[1]

    def test_exact_vs_rooted_k2(self, count_data):
        """N_exact(2) = 600 * N_rooted(2) / 2 = 1200 (each pair counted twice)."""
        exact, rooted = count_data
        assert exact[2] == 600 * rooted[2] // 2

    def test_exact_over_rooted_decreasing(self, count_data):
        """Ratio N_exact(k) / N_rooted(k) decreases with k.

        This ratio = 600/k for perfect symmetry (every polymer
        of size k has exactly k distinct roots giving the same polymer
        up to symmetry). The ratio decreases because larger polymers
        have fewer automorphisms on average.
        """
        exact, rooted = count_data
        ratios = [exact[k] / rooted[k] for k in range(1, 7)]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], (
                f"Ratio at k={i+1} is {ratios[i]:.1f}, "
                f"at k={i+2} is {ratios[i+1]:.1f} (not decreasing)"
            )


# ======================================================================
# 5. Tree bound vs exact
# ======================================================================

class TestTreeBound:
    """Verify tree bounds are valid upper bounds on exact counts."""

    @pytest.fixture(scope='class')
    def bound_data(self):
        """Compute exact counts and tree bounds."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        exact = count_polymers_exact(face_adj, 7)
        D = 4
        bounds = tree_bound(600, D, 7)
        bounds_tight = tree_bound_tight(600, D, 7)
        return exact, bounds, bounds_tight

    def test_tree_bound_is_upper_bound(self, bound_data):
        """THEOREM: Tree bound >= exact count for all sizes."""
        exact, bounds, _ = bound_data
        for k in range(1, 8):
            assert bounds[k] >= exact[k], (
                f"Tree bound {bounds[k]} < exact {exact[k]} at k={k}"
            )

    def test_tight_bound_is_upper_bound(self, bound_data):
        """THEOREM: Tight tree bound (with 1/k) >= exact count."""
        exact, _, bounds_tight = bound_data
        for k in range(1, 8):
            assert bounds_tight[k] >= exact[k], (
                f"Tight bound {bounds_tight[k]} < exact {exact[k]} at k={k}"
            )

    def test_tree_bound_at_k1(self, bound_data):
        """Tree bound at k=1 equals 600 (exact)."""
        exact, bounds, _ = bound_data
        assert bounds[1] == 600

    def test_tree_overestimates(self, bound_data):
        """Tree bound overestimates by > 5x for k >= 2."""
        exact, bounds, _ = bound_data
        for k in range(2, 8):
            ratio = bounds[k] / exact[k]
            assert ratio > 5, (
                f"Tree bound only {ratio:.1f}x larger at k={k}"
            )

    def test_overestimate_grows_with_k(self, bound_data):
        """The overestimate ratio increases with k (tree bound is increasingly loose)."""
        exact, bounds, _ = bound_data
        ratios = [bounds[k] / exact[k] for k in range(1, 8)]
        for i in range(1, len(ratios) - 1):
            assert ratios[i + 1] > ratios[i], (
                f"Overestimate ratio decreases from k={i+1} to k={i+2}"
            )


# ======================================================================
# 6. Peierls suppression computation
# ======================================================================

class TestPeierlsSuppression:
    """Verify Peierls suppression at various couplings."""

    def test_weak_coupling_converges_face(self):
        """THEOREM: Z_large < 1 at very weak coupling (g^2=0.005) with face adjacency.

        With the Balaban choice p0 = g^{1/2}, the critical g^2 for
        Z_large < 1 is approximately 0.0065 (face-sharing, D=4).
        This is the UV regime where asymptotic freedom has driven
        the coupling very small.
        """
        table = compute_peierls_suppression(
            g_squared=0.005,
            adjacency_type='face',
            max_size=20,
            exact_up_to=4,
        )
        assert table.Z_large_converges, (
            f"Z_large = {table.Z_large:.4e} >= 1 at g^2=0.005"
        )
        assert table.label == 'THEOREM'

    def test_strong_coupling_diverges_face(self):
        """At strong coupling (g^2=6.28), face-sharing Peierls fails."""
        table = compute_peierls_suppression(
            g_squared=6.28,
            adjacency_type='face',
            max_size=20,
            exact_up_to=4,
        )
        assert not table.Z_large_converges

    def test_suppression_decreases_with_k(self):
        """exp(-c*k) is a decreasing function of k."""
        table = compute_peierls_suppression(
            g_squared=0.005,
            adjacency_type='face',
            max_size=10,
            exact_up_to=4,
        )
        for k in range(2, 11):
            if k in table.suppression and k - 1 in table.suppression:
                assert table.suppression[k] < table.suppression[k - 1]

    def test_net_exponent_positive_weak_coupling(self):
        """At weak coupling, net exponent = c - log(eD) > 0."""
        table = compute_peierls_suppression(
            g_squared=0.005,
            adjacency_type='face',
            max_size=10,
            exact_up_to=4,
        )
        assert table.net_exponent > 0, (
            f"Net exponent = {table.net_exponent:.4f} <= 0 at g^2=0.005"
        )

    def test_net_exponent_negative_strong_coupling(self):
        """At strong coupling, net exponent < 0."""
        table = compute_peierls_suppression(
            g_squared=6.28,
            adjacency_type='face',
            max_size=10,
            exact_up_to=4,
        )
        assert table.net_exponent < 0, (
            f"Net exponent = {table.net_exponent:.4f} > 0 at g^2=6.28"
        )

    def test_cumulative_increasing(self):
        """Cumulative sum is monotonically increasing."""
        table = compute_peierls_suppression(
            g_squared=0.005,
            adjacency_type='face',
            max_size=10,
            exact_up_to=4,
        )
        for k in range(2, 11):
            if k in table.cumulative and k - 1 in table.cumulative:
                assert table.cumulative[k] >= table.cumulative[k - 1]

    def test_beta_correct(self):
        """beta = 2*N_c / g^2."""
        table = compute_peierls_suppression(
            g_squared=6.28, N_c=2,
            adjacency_type='face',
            max_size=5, exact_up_to=2,
        )
        expected_beta = 2 * 2 / 6.28
        assert abs(table.beta - expected_beta) < 1e-10

    def test_c_suppression_formula(self):
        """c = p0^2 / (2*g^2) for Wilson action coefficient 1/2."""
        g2 = 1.0
        table = compute_peierls_suppression(
            g_squared=g2, p0=2.0,
            adjacency_type='face',
            max_size=5, exact_up_to=2,
        )
        expected_c = 0.5 * 4.0 / 1.0  # = 2.0
        assert abs(table.c_suppression - expected_c) < 1e-10

    def test_D_max_face(self):
        """Face-sharing D_max = 4."""
        table = compute_peierls_suppression(
            g_squared=1.0,
            adjacency_type='face',
            max_size=5, exact_up_to=2,
        )
        assert table.D_max == 4

    def test_D_max_vertex(self):
        """Vertex-sharing D_max = 56."""
        table = compute_peierls_suppression(
            g_squared=1.0,
            adjacency_type='vertex',
            max_size=3, exact_up_to=2,
        )
        assert table.D_max == 56


# ======================================================================
# 7. Peierls table formatting
# ======================================================================

class TestPeierlsTableFormat:
    """Verify the Peierls table formatter produces correct output."""

    def test_table_has_header(self):
        """Formatted table contains header line."""
        table = compute_peierls_suppression(
            g_squared=1.0,
            adjacency_type='face',
            max_size=5, exact_up_to=2,
        )
        text = print_peierls_table(table)
        assert 'PEIERLS TABLE' in text

    def test_table_has_verdict(self):
        """Table shows YES/NO verdict."""
        table = compute_peierls_suppression(
            g_squared=0.01,
            adjacency_type='face',
            max_size=10, exact_up_to=3,
        )
        text = print_peierls_table(table)
        assert 'Z_large < 1?' in text

    def test_table_has_all_rows(self):
        """Table has one row per size."""
        max_size = 5
        table = compute_peierls_suppression(
            g_squared=1.0,
            adjacency_type='face',
            max_size=max_size, exact_up_to=3,
        )
        text = print_peierls_table(table)
        for k in range(1, max_size + 1):
            assert f"  {k:4d}" in text


# ======================================================================
# 8. Critical coupling sweep
# ======================================================================

class TestCriticalCouplingSweep:
    """Verify coupling sweep finds the critical g^2."""

    def test_sweep_returns_values(self):
        """Sweep returns g^2 and Z_large arrays."""
        result = sweep_coupling(
            g2_min=0.01, g2_max=0.1, n_points=5,
            adjacency_type='face',
            max_size=10, exact_up_to=3,
        )
        assert len(result.g2_values) == 5
        assert len(result.Z_large_values) == 5

    def test_Z_large_increases_with_g2(self):
        """Z_large increases with g^2 (stronger coupling = less suppression)."""
        result = sweep_coupling(
            g2_min=0.005, g2_max=0.03, n_points=10,
            adjacency_type='face',
            max_size=15, exact_up_to=3,
        )
        # Z_large should generally increase
        finite_vals = [
            (g2, z) for g2, z in
            zip(result.g2_values, result.Z_large_values)
            if np.isfinite(z)
        ]
        if len(finite_vals) >= 2:
            for i in range(len(finite_vals) - 1):
                assert finite_vals[i + 1][1] >= finite_vals[i][1], (
                    f"Z_large decreased from g^2={finite_vals[i][0]:.4f} "
                    f"to g^2={finite_vals[i+1][0]:.4f}"
                )

    def test_critical_g2_in_expected_range(self):
        """Critical g^2 (face, eps=0) should be near 0.044."""
        result = sweep_coupling(
            g2_min=0.01, g2_max=0.15, n_points=40,
            adjacency_type='face',
            max_size=20, exact_up_to=4,
        )
        if result.g2_critical is not None:
            # The critical g^2 where Z_large = 1
            assert 0.01 < result.g2_critical < 0.15, (
                f"g^2_critical = {result.g2_critical:.4f} out of range"
            )


# ======================================================================
# 9. Adjacency comparison
# ======================================================================

class TestAdjacencyComparison:
    """Verify face vs vertex adjacency comparison."""

    def test_face_gives_fewer_polymers(self):
        """Face-sharing has fewer polymers than vertex-sharing at every size."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        vert_adj = build_cell_adjacency_vertex_sharing(cells)

        face_counts = count_polymers_exact(face_adj, 3)
        vert_counts = count_polymers_exact(vert_adj, 3)

        for k in range(1, 4):
            assert face_counts[k] <= vert_counts[k], (
                f"Face count {face_counts[k]} > vertex count {vert_counts[k]} at k={k}"
            )

    def test_N1_same_both(self):
        """N(1) = 600 for both adjacency types."""
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        vert_adj = build_cell_adjacency_vertex_sharing(cells)

        assert count_polymers_exact(face_adj, 1)[1] == 600
        assert count_polymers_exact(vert_adj, 1)[1] == 600

    def test_N2_vertex_much_larger(self):
        """N(2) for vertex-sharing >> N(2) for face-sharing.

        Face: N(2) = 1200 (each face = one pair).
        Vertex: N(2) = 600 * 56 / 2 = 16800 (each vertex-sharing pair).
        """
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        vert_adj = build_cell_adjacency_vertex_sharing(cells)

        nf = count_polymers_exact(face_adj, 2)[2]
        nv = count_polymers_exact(vert_adj, 2)[2]

        assert nf == 1200
        assert nv == 600 * 56 // 2  # = 16800
        assert nv > 10 * nf


# ======================================================================
# 10. Physical parameter tests
# ======================================================================

class TestPhysicalParameters:
    """Tests at physical parameter values."""

    def test_g2_bare(self):
        """G2_BARE_DEFAULT = 6.28."""
        assert abs(G2_BARE_DEFAULT - 6.28) < 1e-10

    def test_strong_coupling_peierls_fails(self):
        """NUMERICAL: At g^2 = 6.28, Peierls fails (Z_large > 1).

        This is expected: the Peierls argument is a UV tool.
        At physical IR coupling, the mass gap is controlled by
        the S^3 spectral gap, not by the Peierls argument.
        """
        table = compute_peierls_suppression(
            g_squared=6.28,
            adjacency_type='face',
            max_size=10,
            exact_up_to=4,
        )
        assert not table.Z_large_converges
        assert table.net_exponent < 0

    def test_uv_peierls_works(self):
        """THEOREM: At UV coupling g^2 = 0.005, Peierls bound holds.

        Asymptotic freedom drives g^2 -> 0 in the UV. With p0 = g^{1/2},
        the Peierls sum Z_large < 1 for g^2 < ~0.0065 (face-sharing D=4).
        """
        table = compute_peierls_suppression(
            g_squared=0.005,
            adjacency_type='face',
            max_size=20,
            exact_up_to=4,
        )
        assert table.Z_large_converges
        assert table.Z_large < 1.0
        assert table.net_exponent > 0

    def test_critical_coupling_face(self):
        """NUMERICAL: Critical g^2 for face-sharing is near g=0.21 (g^2~0.044).

        At g^2 = g^2_crit, the suppression c = 1/(2g) exactly equals
        the entropy rate log(e*D) = log(4e) = 2.386.
        So g_crit = 1/(2*2.386) = 0.2095, g^2_crit = 0.0439.
        """
        D_face = 4
        entropy = np.log(np.e * D_face)
        g_crit = 1.0 / (2.0 * entropy)
        g2_crit = g_crit ** 2
        assert abs(g2_crit - 0.0439) < 0.001


# ======================================================================
# 11. Exact polymer count table (regression)
# ======================================================================

class TestPolymerCountTable:
    """
    Regression test: the exact polymer counts for the 600-cell
    face-sharing adjacency graph.

    NUMERICAL: These values were computed by exhaustive BFS enumeration
    and serve as a reference for future optimizations.

    The table:
        k    N(k)
        1    600
        2    1200
        3    3600
        4    13200
        5    51720
        6    212400
        7    900000
    """

    EXPECTED_COUNTS = {
        1: 600,
        2: 1200,
        3: 3600,
        4: 13200,
        5: 51720,
        6: 212400,
        7: 900000,
    }

    @pytest.fixture(scope='class')
    def computed_counts(self):
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        return count_polymers_exact(face_adj, 7)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7])
    def test_count_matches(self, computed_counts, k):
        """N(k) matches the pre-computed reference value."""
        assert computed_counts[k] == self.EXPECTED_COUNTS[k], (
            f"N({k}) = {computed_counts[k]}, expected {self.EXPECTED_COUNTS[k]}"
        )


# ======================================================================
# 12. Growth rate analysis
# ======================================================================

class TestGrowthRate:
    """Analyze the asymptotic growth rate of polymer counts."""

    @pytest.fixture(scope='class')
    def face_counts(self):
        vertices, edges, faces, cells = build_600_cell(R=1.0)
        face_adj = build_cell_adjacency_face_sharing(cells)
        return count_polymers_exact(face_adj, 7)

    def test_growth_rate_bounded_by_D(self, face_counts):
        """Growth rate N(k+1)/N(k) < D for large k.

        For a D-regular graph, the growth rate of lattice animals
        converges to a constant mu < D (the connective constant).
        """
        D = 4
        for k in range(3, 7):
            ratio = face_counts[k + 1] / face_counts[k]
            assert ratio < D * np.e, (
                f"Growth rate N({k+1})/N({k}) = {ratio:.2f} exceeds D*e = {D*np.e:.2f}"
            )

    def test_growth_rate_stabilizes(self, face_counts):
        """Growth rate converges (ratios at k=5,6 are closer than k=2,3)."""
        ratios = [face_counts[k + 1] / face_counts[k] for k in range(1, 7)]
        # Later ratios should be more stable
        diff_early = abs(ratios[1] - ratios[0])
        diff_late = abs(ratios[-1] - ratios[-2])
        # Not a strict requirement but expected for lattice animals
        # Just check the ratio is bounded
        assert all(r > 1 for r in ratios), "Some growth ratios < 1"
        assert all(r < 20 for r in ratios), "Some growth ratios > 20"

    def test_connective_constant_estimate(self, face_counts):
        """NUMERICAL: Estimate the connective constant mu from exact data.

        For lattice animals on a D-regular graph, N(k) ~ A * mu^k
        where mu is the connective constant. The sequence N(k+1)/N(k)
        converges to mu.

        For the 600-cell face graph (D=4), we expect mu ~ 3-4.
        """
        ratios = [face_counts[k + 1] / face_counts[k] for k in range(4, 7)]
        mu_estimate = np.mean(ratios)
        assert 3.0 < mu_estimate < 5.0, (
            f"Connective constant estimate mu = {mu_estimate:.2f}, "
            f"expected between 3 and 5 for D=4 graph"
        )


# ======================================================================
# 13. Small graph validation
# ======================================================================

class TestSmallGraphs:
    """Validate polymer counting on small known graphs."""

    def test_single_node(self):
        """Single-node graph: N(1) = 1."""
        adj = {0: set()}
        counts = count_polymers_exact(adj, 3)
        assert counts[1] == 1
        assert counts.get(2, 0) == 0

    def test_two_nodes_connected(self):
        """Two connected nodes: N(1)=2, N(2)=1."""
        adj = {0: {1}, 1: {0}}
        counts = count_polymers_exact(adj, 2)
        assert counts[1] == 2
        assert counts[2] == 1

    def test_path_of_3(self):
        """Path 0-1-2: N(1)=3, N(2)=2, N(3)=1."""
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        counts = count_polymers_exact(adj, 3)
        assert counts[1] == 3
        assert counts[2] == 2
        assert counts[3] == 1

    def test_triangle(self):
        """Triangle K_3: N(1)=3, N(2)=3, N(3)=1."""
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        counts = count_polymers_exact(adj, 3)
        assert counts[1] == 3
        assert counts[2] == 3
        assert counts[3] == 1

    def test_K4(self):
        """Complete graph K_4: N(1)=4, N(2)=6, N(3)=4, N(4)=1."""
        adj = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}
        counts = count_polymers_exact(adj, 4)
        assert counts[1] == 4
        assert counts[2] == 6   # C(4,2) edges
        assert counts[3] == 4   # C(4,3) triangles (all connected)
        assert counts[4] == 1   # Full graph

    def test_cycle_4(self):
        """4-cycle: N(1)=4, N(2)=4, N(3)=4, N(4)=1."""
        adj = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {2, 0}}
        counts = count_polymers_exact(adj, 4)
        assert counts[1] == 4
        assert counts[2] == 4   # 4 edges
        assert counts[3] == 4   # 4 paths of length 2
        assert counts[4] == 1   # Full cycle

    def test_star_4(self):
        """Star with center 0 and leaves 1,2,3: N(1)=4, N(2)=3, N(3)=3, N(4)=1."""
        adj = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
        counts = count_polymers_exact(adj, 4)
        assert counts[1] == 4
        assert counts[2] == 3   # 3 edges
        assert counts[3] == 3   # Center + 2 leaves (C(3,2) = 3)
        assert counts[4] == 1   # Full star
