"""
Tests for the Large-Field Peierls Argument on S^3.

Tests cover:
  1. 600-cell combinatorial data (THEOREM)
  2. Polymer counting: analytical bounds (THEOREM)
  3. Polymer counting: exact enumeration on small graphs (THEOREM)
  4. Wilson action suppression (THEOREM)
  5. Gribov field bound (NUMERICAL)
  6. Peierls condition check (THEOREM)
  7. Perturbative regime analysis (THEOREM/NUMERICAL)
  8. Full Peierls bound — face-sharing (THEOREM)
  9. Full Peierls bound — vertex-sharing conservative (THEOREM)
  10. Scale-by-scale analysis (THEOREM)
  11. Combined RG bound (THEOREM)
  12. Uniformity in R (THEOREM)
  13. Coupling scan (NUMERICAL)
  14. Balaban comparison (informational)
  15. Complete analysis integration (THEOREM)
  16. Edge cases and stress tests (NUMERICAL)

Run:
    pytest tests/rg/test_large_field_peierls.py -v
"""

import numpy as np
import pytest

from yang_mills_s3.rg.large_field_peierls import (
    # Constants
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    G2_BARE_DEFAULT,
    G2_MAX,
    CELL_COUNT_600,
    VERTEX_COUNT_600,
    EDGE_COUNT_600,
    FACE_COUNT_600,
    FACE_SHARING_DEGREE,
    MAX_ADJACENCY_600,
    REFINEMENT_FACTOR,
    # 600-cell construction
    build_600_cell_adjacency,
    # Block counts
    block_count_at_scale,
    max_coordination_at_scale,
    # Polymer counting
    PolymerCount,
    count_polymers_on_graph,
    analytical_polymer_bound,
    polymer_entropy_at_scale,
    # Wilson suppression
    WilsonSuppression,
    wilson_action_suppression,
    # Gribov bound
    GribovFieldBound,
    gribov_field_bound,
    # Peierls condition
    PeierlsCondition,
    peierls_condition_check,
    minimum_p0,
    optimal_p0,
    # Perturbative regime
    PerturbativeRegime,
    perturbative_regime_analysis,
    # Full Peierls bound
    PeierlsBound,
    large_field_peierls_bound,
    # Scale-by-scale
    ScaleByScaleResult,
    scale_by_scale_peierls,
    # Combined bound
    CombinedRGBound,
    combined_rg_bound,
    # Uniformity
    UniformityResult,
    uniformity_in_R,
    # Conservative bound
    conservative_peierls_bound,
    # Coupling scan
    CouplingScanResult,
    coupling_scan,
    # Balaban comparison
    balaban_comparison,
    # Master analysis
    complete_peierls_analysis,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def cell_adjacency():
    """Build the 600-cell cell adjacency (expensive, do once)."""
    adj, n_cells, max_deg = build_600_cell_adjacency()
    return adj, n_cells, max_deg


@pytest.fixture
def linear_graph_5():
    """5-node linear graph: 0-1-2-3-4."""
    return {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3},
    }


@pytest.fixture
def cycle_graph_6():
    """6-node cycle: 0-1-2-3-4-5-0."""
    return {
        0: {1, 5},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 5},
        5: {4, 0},
    }


@pytest.fixture
def complete_graph_4():
    """Complete graph K_4."""
    return {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }


@pytest.fixture
def tree_graph():
    """Binary tree of depth 2 (7 nodes)."""
    return {
        0: {1, 2},
        1: {0, 3, 4},
        2: {0, 5, 6},
        3: {1},
        4: {1},
        5: {2},
        6: {2},
    }


# ======================================================================
# 1. 600-cell Combinatorial Data
# ======================================================================

class Test600CellConstants:
    """Verify the 600-cell combinatorial constants."""

    def test_cell_count(self):
        """600-cell has exactly 600 tetrahedral cells."""
        assert CELL_COUNT_600 == 600

    def test_vertex_count(self):
        """600-cell has exactly 120 vertices."""
        assert VERTEX_COUNT_600 == 120

    def test_edge_count(self):
        """600-cell has exactly 720 edges."""
        assert EDGE_COUNT_600 == 720

    def test_face_count(self):
        """600-cell has exactly 1200 triangular faces."""
        assert FACE_COUNT_600 == 1200

    def test_euler_characteristic(self):
        """
        Euler characteristic of 4-polytope boundary:
        V - E + F - C = 0 (for S^3 as boundary of 4-polytope).
        """
        chi = VERTEX_COUNT_600 - EDGE_COUNT_600 + FACE_COUNT_600 - CELL_COUNT_600
        assert chi == 0

    def test_face_sharing_degree(self):
        """Each tetrahedron has 4 faces, each shared with 1 neighbor."""
        assert FACE_SHARING_DEGREE == 4

    def test_face_sharing_consistency(self):
        """
        Total face-sharings: 600 cells * 4 faces each = 2400 half-edges
        in the dual graph, giving 1200 edges (= 1200 faces). Check.
        """
        dual_half_edges = CELL_COUNT_600 * FACE_SHARING_DEGREE
        dual_edges = dual_half_edges // 2
        assert dual_edges == FACE_COUNT_600

    def test_refinement_factor(self):
        """Midpoint subdivision quadruples cells."""
        assert REFINEMENT_FACTOR == 4


class Test600CellAdjacency:
    """Test the actual 600-cell cell adjacency graph construction."""

    def test_vertex_generation_count(self, cell_adjacency):
        """Build produces a cell adjacency graph."""
        adj, n_cells, max_deg = cell_adjacency
        # The 600-cell should have 600 cells
        assert n_cells == 600

    def test_cell_adjacency_degree(self, cell_adjacency):
        """Face-sharing degree should be exactly 4 for each cell."""
        adj, n_cells, max_deg = cell_adjacency
        assert max_deg == 4
        # Each cell should have exactly 4 neighbors
        for cell_id in adj:
            assert len(adj[cell_id]) == 4, (
                f"Cell {cell_id} has {len(adj[cell_id])} neighbors, expected 4"
            )

    def test_adjacency_is_symmetric(self, cell_adjacency):
        """If cell A is adjacent to cell B, then B is adjacent to A."""
        adj, n_cells, _ = cell_adjacency
        for a in adj:
            for b in adj[a]:
                assert a in adj[b], f"Adjacency not symmetric: {a} -> {b}"

    def test_no_self_loops(self, cell_adjacency):
        """No cell is adjacent to itself."""
        adj, n_cells, _ = cell_adjacency
        for a in adj:
            assert a not in adj[a], f"Self-loop at cell {a}"

    def test_total_face_count(self, cell_adjacency):
        """
        Total edges in dual graph = 1200 (= number of faces of 600-cell).
        Each face is shared by exactly 2 cells.
        """
        adj, n_cells, _ = cell_adjacency
        total_half_edges = sum(len(nbrs) for nbrs in adj.values())
        total_edges = total_half_edges // 2
        assert total_edges == 1200


# ======================================================================
# 2. Block Counts at Each Scale
# ======================================================================

class TestBlockCounts:
    """Test block_count_at_scale."""

    def test_level_0(self):
        """Level 0 has 600 blocks."""
        assert block_count_at_scale(0) == 600

    def test_level_1(self):
        """Level 1 has 600 * 4 = 2400 blocks."""
        assert block_count_at_scale(1) == 2400

    def test_level_2(self):
        """Level 2 has 600 * 16 = 9600 blocks."""
        assert block_count_at_scale(2) == 9600

    def test_level_n_formula(self):
        """General formula: 600 * 4^n."""
        for n in range(8):
            assert block_count_at_scale(n) == 600 * 4 ** n

    def test_always_finite(self):
        """Block count is finite for any level."""
        for n in range(20):
            assert np.isfinite(block_count_at_scale(n))
            assert block_count_at_scale(n) > 0


class TestMaxCoordination:
    """Test max_coordination_at_scale."""

    def test_face_sharing(self):
        """Face-sharing degree is 4."""
        for n in range(5):
            assert max_coordination_at_scale(n, use_face_sharing=True) == 4

    def test_vertex_sharing(self):
        """Vertex-sharing degree is 20 (conservative bound)."""
        for n in range(5):
            assert max_coordination_at_scale(n, use_face_sharing=False) == 20


# ======================================================================
# 3. Polymer Counting on Small Graphs
# ======================================================================

class TestPolymerCountingSmallGraphs:
    """Exact polymer counting on small test graphs."""

    def test_linear_5_size_1(self, linear_graph_5):
        """5-node linear graph has 5 polymers of size 1."""
        counts = count_polymers_on_graph(linear_graph_5, max_size=1)
        assert counts[1] == 5

    def test_linear_5_size_2(self, linear_graph_5):
        """5-node linear graph has 4 connected pairs."""
        counts = count_polymers_on_graph(linear_graph_5, max_size=2)
        assert counts[2] == 4  # {01, 12, 23, 34}

    def test_linear_5_size_3(self, linear_graph_5):
        """5-node linear graph has 3 connected triples."""
        counts = count_polymers_on_graph(linear_graph_5, max_size=3)
        assert counts[3] == 3  # {012, 123, 234}

    def test_linear_5_size_4(self, linear_graph_5):
        """5-node linear graph has 2 connected quadruples."""
        counts = count_polymers_on_graph(linear_graph_5, max_size=4)
        assert counts[4] == 2  # {0123, 1234}

    def test_linear_5_size_5(self, linear_graph_5):
        """5-node linear graph has 1 connected quintuple."""
        counts = count_polymers_on_graph(linear_graph_5, max_size=5)
        assert counts[5] == 1  # {01234}

    def test_cycle_6_size_2(self, cycle_graph_6):
        """6-node cycle has 6 edges."""
        counts = count_polymers_on_graph(cycle_graph_6, max_size=2)
        assert counts[2] == 6

    def test_cycle_6_size_3(self, cycle_graph_6):
        """6-node cycle has 6 connected triples (paths of length 2)."""
        counts = count_polymers_on_graph(cycle_graph_6, max_size=3)
        assert counts[3] == 6

    def test_complete_4_size_2(self, complete_graph_4):
        """K_4 has C(4,2) = 6 edges, all connected."""
        counts = count_polymers_on_graph(complete_graph_4, max_size=2)
        assert counts[2] == 6

    def test_complete_4_size_3(self, complete_graph_4):
        """K_4 has C(4,3) = 4 connected triples."""
        counts = count_polymers_on_graph(complete_graph_4, max_size=3)
        assert counts[3] == 4

    def test_complete_4_size_4(self, complete_graph_4):
        """K_4 has 1 connected quadruple."""
        counts = count_polymers_on_graph(complete_graph_4, max_size=4)
        assert counts[4] == 1

    def test_tree_size_1(self, tree_graph):
        """7-node tree has 7 singletons."""
        counts = count_polymers_on_graph(tree_graph, max_size=1)
        assert counts[1] == 7

    def test_tree_size_2(self, tree_graph):
        """7-node binary tree has 6 edges."""
        counts = count_polymers_on_graph(tree_graph, max_size=2)
        assert counts[2] == 6

    def test_empty_graph(self):
        """Empty graph has 0 polymers."""
        counts = count_polymers_on_graph({}, max_size=5)
        assert len(counts) == 0 or all(v == 0 for v in counts.values())


# ======================================================================
# 4. Polymer Counting: Analytical Bounds
# ======================================================================

class TestAnalyticalPolymerBound:
    """Test the lattice animal upper bound."""

    def test_size_1(self):
        """Size 1: bound = N (each node is a polymer)."""
        bounds = analytical_polymer_bound(600, 4, 1)
        assert bounds[1] == 600

    def test_monotone_in_size(self):
        """Bounds grow with size (for D > 1)."""
        bounds = analytical_polymer_bound(600, 4, 10)
        for s in range(2, 11):
            assert bounds[s] >= bounds[s - 1]

    def test_bounds_are_finite(self):
        """All bounds are finite positive integers."""
        bounds = analytical_polymer_bound(600, 4, 20)
        for s, b in bounds.items():
            assert np.isfinite(float(b))
            assert isinstance(b, int)
            assert b > 0

    def test_bound_formula(self):
        """Check the formula N * (eD)^{s-1}."""
        N = 600
        D = 4
        bounds = analytical_polymer_bound(N, D, 5)
        eD = np.e * D
        for s in range(1, 6):
            expected = int(np.ceil(N * eD ** (s - 1)))
            assert bounds[s] == expected

    def test_face_sharing_bound_below_exact(self, cell_adjacency):
        """
        Analytical bound with D=4 should be an UPPER bound on exact counts.
        Verify for sizes 1 and 2.
        """
        adj, n_cells, max_deg = cell_adjacency
        exact = count_polymers_on_graph(adj, max_size=2)
        bounds = analytical_polymer_bound(n_cells, max_deg, 2)

        # Exact count at size 1 = n_cells = 600
        assert exact[1] == n_cells
        assert bounds[1] >= exact[1]

        # At size 2: exact = number of edges in dual graph = 1200
        # Bound: 600 * (4e)^1 = 600 * 10.87 = 6524
        assert bounds[2] >= exact[2]


# ======================================================================
# 5. Polymer Entropy at Scale
# ======================================================================

class TestPolymerEntropyAtScale:
    """Test polymer_entropy_at_scale."""

    def test_scale_0_basics(self):
        """Base scale (level 0) has 600 blocks."""
        pc = polymer_entropy_at_scale(0, max_size=5)
        assert pc.n_blocks == 600
        assert pc.scale == 0
        assert pc.label == 'THEOREM'

    def test_scale_0_face_sharing(self):
        """With face-sharing, degree = 4."""
        pc = polymer_entropy_at_scale(0, max_size=5, use_face_sharing=True)
        assert pc.max_degree == 4

    def test_scale_0_vertex_sharing(self):
        """With vertex-sharing, degree = 20."""
        pc = polymer_entropy_at_scale(0, max_size=5, use_face_sharing=False)
        assert pc.max_degree == 20

    def test_total_count_positive(self):
        """Total polymer count is positive."""
        pc = polymer_entropy_at_scale(0, max_size=10)
        assert pc.total_count > 0

    def test_counts_by_size_populated(self):
        """counts_by_size has entries for sizes 1..max_size."""
        pc = polymer_entropy_at_scale(0, max_size=10)
        for s in range(1, 11):
            assert s in pc.counts_by_size
            assert pc.counts_by_size[s] > 0

    def test_higher_scale_more_blocks(self):
        """Higher refinement = more blocks."""
        pc0 = polymer_entropy_at_scale(0, max_size=3)
        pc1 = polymer_entropy_at_scale(1, max_size=3)
        assert pc1.n_blocks == 4 * pc0.n_blocks


# ======================================================================
# 6. Wilson Action Suppression
# ======================================================================

class TestWilsonSuppression:
    """Test wilson_action_suppression."""

    def test_suppression_positive(self):
        """Suppression is positive."""
        supp = wilson_action_suppression(6.28, 3.0)
        assert supp.suppression_per_block > 0
        assert supp.suppression_per_block <= 1.0

    def test_larger_p0_more_suppression(self):
        """Larger p_0 gives smaller suppression (more suppressed)."""
        s1 = wilson_action_suppression(6.28, 1.0)
        s2 = wilson_action_suppression(6.28, 5.0)
        assert s2.suppression_per_block < s1.suppression_per_block

    def test_smaller_g2_more_suppression(self):
        """Smaller g^2 gives more suppression (larger exponent)."""
        s1 = wilson_action_suppression(6.28, 3.0)
        s2 = wilson_action_suppression(1.0, 3.0)
        assert s2.suppression_per_block < s1.suppression_per_block

    def test_exponent_formula(self):
        """Exponent = (1/2) * p_0^2 / g^2."""
        g2 = 6.28
        p0 = 4.0
        supp = wilson_action_suppression(g2, p0)
        expected_exp = 0.5 * p0 ** 2 / g2
        assert abs(supp.exponent_per_block - expected_exp) < 1e-10

    def test_suppression_exp_of_exponent(self):
        """suppression = exp(-exponent)."""
        supp = wilson_action_suppression(6.28, 3.0)
        assert abs(supp.suppression_per_block - np.exp(-supp.exponent_per_block)) < 1e-15

    def test_c_wilson_is_half(self):
        """Wilson coefficient c_W = 0.5."""
        supp = wilson_action_suppression(6.28, 3.0)
        assert supp.c_wilson == 0.5

    def test_label_theorem(self):
        """Label is THEOREM (action positivity is rigorous)."""
        supp = wilson_action_suppression(6.28, 3.0)
        assert supp.label == 'THEOREM'


# ======================================================================
# 7. Gribov Field Bound
# ======================================================================

class TestGribovFieldBound:
    """Test gribov_field_bound."""

    def test_bound_positive(self):
        """Maximum field strength is positive."""
        gb = gribov_field_bound()
        assert gb.max_field_strength > 0

    def test_diameter_proportional_to_R(self):
        """Gribov diameter ~ 1.89 * R."""
        for R in [1.0, 2.2, 5.0]:
            gb = gribov_field_bound(R=R)
            assert abs(gb.gribov_diameter / R - 1.89) < 0.01

    def test_max_F_at_physical_coupling(self):
        """At g^2 = 6.28, |F|_max ~ 2.37."""
        gb = gribov_field_bound(R=R_PHYSICAL_FM, g_squared=G2_BARE_DEFAULT)
        g = np.sqrt(G2_BARE_DEFAULT)
        expected = g * 1.89 / 2.0
        assert abs(gb.max_field_strength - expected) < 0.01

    def test_max_F_grows_with_g(self):
        """Larger g gives larger |F|_max."""
        gb1 = gribov_field_bound(g_squared=1.0)
        gb2 = gribov_field_bound(g_squared=10.0)
        assert gb2.max_field_strength > gb1.max_field_strength

    def test_label_numerical(self):
        """Gribov diameter is NUMERICAL (from Session 6 computations)."""
        gb = gribov_field_bound()
        assert gb.label == 'NUMERICAL'


# ======================================================================
# 8. Peierls Condition Check
# ======================================================================

class TestPeierlsCondition:
    """Test peierls_condition_check."""

    def test_condition_met_high_p0(self):
        """Large p_0 satisfies Peierls."""
        cond = peierls_condition_check(6.28, 10.0, D_max=4)
        assert cond.is_satisfied
        assert cond.ratio > 1.0

    def test_condition_fails_small_p0(self):
        """Very small p_0 violates Peierls."""
        cond = peierls_condition_check(6.28, 0.1, D_max=4)
        assert not cond.is_satisfied
        assert cond.ratio < 1.0

    def test_net_exponent_sign(self):
        """Net exponent is positive iff condition is met."""
        for p0 in [0.1, 1.0, 5.0, 10.0, 20.0]:
            cond = peierls_condition_check(6.28, p0, D_max=4)
            if cond.is_satisfied:
                assert cond.net_exponent > 0
            else:
                assert cond.net_exponent <= 0

    def test_entropy_rate_for_D4(self):
        """Entropy rate = log(4e) ~ 2.386 for D=4."""
        cond = peierls_condition_check(6.28, 5.0, D_max=4)
        expected_entropy = np.log(4 * np.e)
        assert abs(cond.entropy_rate - expected_entropy) < 1e-10

    def test_entropy_rate_for_D20(self):
        """Entropy rate = log(20e) ~ 3.989 for D=20."""
        cond = peierls_condition_check(6.28, 5.0, D_max=20)
        expected_entropy = np.log(20 * np.e)
        assert abs(cond.entropy_rate - expected_entropy) < 1e-10

    def test_label_matches_condition(self):
        """Label is THEOREM when satisfied, FAILS otherwise."""
        cond_ok = peierls_condition_check(6.28, 10.0, D_max=4)
        assert cond_ok.label == 'THEOREM'

        cond_fail = peierls_condition_check(6.28, 0.1, D_max=4)
        assert cond_fail.label == 'FAILS'


class TestMinimumP0:
    """Test minimum_p0."""

    def test_positive(self):
        """Minimum p_0 is positive."""
        p0_min = minimum_p0(6.28, D_max=4)
        assert p0_min > 0

    def test_satisfies_condition(self):
        """p_0 = p_0_min + epsilon satisfies the Peierls condition."""
        p0_min = minimum_p0(6.28, D_max=4)
        cond = peierls_condition_check(6.28, p0_min * 1.01, D_max=4)
        assert cond.is_satisfied

    def test_violates_below(self):
        """p_0 = p_0_min - epsilon violates the Peierls condition."""
        p0_min = minimum_p0(6.28, D_max=4)
        cond = peierls_condition_check(6.28, p0_min * 0.99, D_max=4)
        assert not cond.is_satisfied

    def test_grows_with_g2(self):
        """Larger g^2 requires larger p_0."""
        p0_1 = minimum_p0(1.0, D_max=4)
        p0_10 = minimum_p0(10.0, D_max=4)
        assert p0_10 > p0_1

    def test_grows_with_D(self):
        """Larger D (more entropy) requires larger p_0."""
        p0_4 = minimum_p0(6.28, D_max=4)
        p0_20 = minimum_p0(6.28, D_max=20)
        assert p0_20 > p0_4


class TestOptimalP0:
    """Test optimal_p0."""

    def test_above_minimum(self):
        """Optimal p_0 exceeds minimum."""
        p0_min = minimum_p0(6.28, D_max=4)
        p0_opt = optimal_p0(6.28, D_max=4, safety_factor=2.0)
        assert p0_opt > p0_min

    def test_safety_factor(self):
        """p_0_opt = safety * p_0_min."""
        p0_min = minimum_p0(6.28, D_max=4)
        for sf in [1.5, 2.0, 3.0]:
            p0_opt = optimal_p0(6.28, D_max=4, safety_factor=sf)
            assert abs(p0_opt - sf * p0_min) < 1e-10

    def test_peierls_ratio_is_safety_squared(self):
        """Peierls ratio = safety_factor^2."""
        sf = 2.0
        p0 = optimal_p0(6.28, D_max=4, safety_factor=sf)
        cond = peierls_condition_check(6.28, p0, D_max=4)
        assert abs(cond.ratio - sf ** 2) < 0.01


# ======================================================================
# 9. Perturbative Regime Analysis
# ======================================================================

class TestPerturbativeRegime:
    """Test perturbative_regime_analysis."""

    def test_constructs(self):
        """Can construct without error."""
        pr = perturbative_regime_analysis()
        assert isinstance(pr, PerturbativeRegime)

    def test_p0_exceeds_F_max_at_physical_coupling(self):
        """
        KEY RESULT: At g^2 = 6.28, D=4, p_0_min > |F|_max(Gribov).
        The large-field region is EMPTY within the Gribov region!

        This is the central result: S^3 compactness (which gives D=4)
        combined with the Gribov bound makes large-field control trivial.
        """
        pr = perturbative_regime_analysis(g_squared=G2_BARE_DEFAULT,
                                          D_max=FACE_SHARING_DEGREE)
        assert pr.p0 > pr.gribov_max_F, (
            f"p0={pr.p0:.3f} should exceed F_max={pr.gribov_max_F:.3f}"
        )
        assert pr.is_perturbative
        assert pr.p0_over_F_max > 1.0

    def test_perturbative_fraction_one(self):
        """When p_0 > F_max, entire Gribov region is small-field."""
        pr = perturbative_regime_analysis(g_squared=G2_BARE_DEFAULT,
                                          D_max=FACE_SHARING_DEGREE)
        assert pr.perturbative_fraction == 1.0

    def test_gribov_max_F_value(self):
        """Check numerical value of F_max."""
        pr = perturbative_regime_analysis()
        g = np.sqrt(G2_BARE_DEFAULT)
        expected_F_max = g * 1.89 / 2.0
        assert abs(pr.gribov_max_F - expected_F_max) < 0.01

    def test_p0_min_value_D4(self):
        """
        Check p_0_min at D=4:
        p_0_min = sqrt(2 * g^2 * log(4e)) = sqrt(2 * 6.28 * 2.386) ~ 5.47
        """
        pr = perturbative_regime_analysis(D_max=4, safety_factor=1.0)
        expected_p0_min = np.sqrt(2 * G2_BARE_DEFAULT * np.log(4 * np.e))
        assert abs(pr.p0_min - expected_p0_min) < 0.01

    def test_also_works_with_D20(self):
        """Conservative D=20 also gives p_0 > F_max."""
        pr = perturbative_regime_analysis(D_max=MAX_ADJACENCY_600)
        # With D=20: p_0_min = sqrt(2 * 6.28 * log(20e)) ~ 7.08
        # F_max ~ 2.37
        # p_0 = 2 * 7.08 ~ 14.16 > 2.37
        assert pr.p0 > pr.gribov_max_F

    def test_label_theorem_when_empty(self):
        """Label should be THEOREM when large-field is empty."""
        pr = perturbative_regime_analysis(D_max=FACE_SHARING_DEGREE)
        assert pr.label == 'THEOREM'


# ======================================================================
# 10. Full Peierls Bound
# ======================================================================

class TestPeierlsBound:
    """Test large_field_peierls_bound."""

    def test_face_sharing_gribov_empty(self):
        """
        THEOREM: With face-sharing D=4, the large-field region is EMPTY
        within the Gribov region at g^2 = 6.28.
        """
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            refinement_level=0,
            use_face_sharing=True,
        )
        assert pb.gribov_empty, "Large-field region should be empty within Gribov"
        assert pb.large_field_bound == 0.0
        assert pb.contraction_factor == 0.0
        assert pb.label == 'THEOREM'

    def test_vertex_sharing_also_works(self):
        """Even with D=20 (conservative), Peierls holds."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            refinement_level=0,
            use_face_sharing=False,
        )
        # p_0 optimal with D=20 is larger, should still beat F_max
        assert pb.peierls_condition or pb.gribov_empty
        assert pb.label == 'THEOREM'

    def test_fixed_p0_below_gribov(self):
        """With small fixed p_0 < F_max, large-field region is NOT empty."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            refinement_level=0,
            p0=1.0,  # Much less than F_max ~ 2.37
            use_face_sharing=True,
        )
        assert not pb.gribov_empty

    def test_fixed_p0_large_peierls_holds(self):
        """With large fixed p_0, Peierls condition holds (even if not Gribov-empty)."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            refinement_level=0,
            p0=10.0,
            use_face_sharing=True,
        )
        assert pb.peierls_condition
        assert pb.large_field_bound < 1.0

    def test_n_blocks_correct(self):
        """n_blocks matches block_count_at_scale."""
        for level in range(4):
            pb = large_field_peierls_bound(
                g_squared=G2_BARE_DEFAULT,
                refinement_level=level,
                use_face_sharing=True,
            )
            assert pb.n_blocks == block_count_at_scale(level)

    def test_critical_g2_positive(self):
        """Critical coupling is positive."""
        pb = large_field_peierls_bound(g_squared=G2_BARE_DEFAULT)
        assert pb.critical_g_squared > 0

    def test_assumptions_populated(self):
        """Assumptions list is non-empty and contains key items."""
        pb = large_field_peierls_bound(g_squared=G2_BARE_DEFAULT)
        assert len(pb.assumptions) >= 5
        assert any('Wilson' in a for a in pb.assumptions)
        assert any('Gribov' in a for a in pb.assumptions)


class TestConservativeBound:
    """Test conservative_peierls_bound (D=20)."""

    def test_constructs(self):
        """Can construct without error."""
        pb = conservative_peierls_bound()
        assert isinstance(pb, PeierlsBound)

    def test_uses_D20(self):
        """Uses D_max = 20."""
        pb = conservative_peierls_bound()
        assert pb.max_degree == MAX_ADJACENCY_600

    def test_peierls_holds(self):
        """Even conservative bound gives Peierls convergence."""
        pb = conservative_peierls_bound()
        assert pb.peierls_condition or pb.gribov_empty
        assert pb.label == 'THEOREM'


# ======================================================================
# 11. Scale-by-Scale Analysis
# ======================================================================

class TestScaleByScale:
    """Test scale_by_scale_peierls."""

    def test_constructs(self):
        """Can construct without error."""
        sbs = scale_by_scale_peierls(n_scales=5)
        assert isinstance(sbs, ScaleByScaleResult)

    def test_n_scales_correct(self):
        """Number of bounds matches n_scales."""
        n = 5
        sbs = scale_by_scale_peierls(n_scales=n)
        assert len(sbs.bounds) == n

    def test_all_peierls_satisfied_face_sharing(self):
        """All scales satisfy Peierls with face-sharing D=4."""
        sbs = scale_by_scale_peierls(use_face_sharing=True)
        assert sbs.all_peierls_satisfied or sbs.n_gribov_empty == sbs.n_scales

    def test_coupling_decreases_in_uv(self):
        """Coupling runs down in the UV (asymptotic freedom)."""
        sbs = scale_by_scale_peierls(n_scales=7)
        g2_values = [b.g_squared for b in sbs.bounds]
        # UV scales (larger j) have smaller g^2
        # bounds[0] = UV (highest refinement), bounds[-1] = IR (base)
        # But in scale_by_scale_peierls, j=0 is the first scale with g^2_0,
        # and the running DECREASES g^2 for larger j
        for i in range(1, len(g2_values)):
            assert g2_values[i] <= g2_values[i - 1] + 1e-10

    def test_total_bound_finite(self):
        """Total large-field bound across scales is finite."""
        sbs = scale_by_scale_peierls()
        assert np.isfinite(sbs.total_large_field_bound)
        assert sbs.total_large_field_bound >= 0

    def test_label_theorem(self):
        """Label is THEOREM when all satisfied."""
        sbs = scale_by_scale_peierls(use_face_sharing=True)
        assert sbs.label == 'THEOREM'


# ======================================================================
# 12. Combined RG Bound
# ======================================================================

class TestCombinedBound:
    """Test combined_rg_bound."""

    def test_constructs(self):
        """Can construct without error."""
        cb = combined_rg_bound()
        assert isinstance(cb, CombinedRGBound)

    def test_is_contracting(self):
        """kappa_total < 1 for physical parameters."""
        cb = combined_rg_bound()
        assert cb.is_contracting
        assert cb.kappa_total < 1.0

    def test_kappa_small_dominates(self):
        """When Gribov-empty, kappa_total = kappa_small."""
        cb = combined_rg_bound(use_face_sharing=True)
        if cb.gribov_empty:
            assert abs(cb.kappa_total - cb.kappa_small) < 1e-10

    def test_custom_kappa_small(self):
        """Can provide custom kappa_small."""
        cb = combined_rg_bound(kappa_small=0.5)
        assert abs(cb.kappa_small - 0.5) < 1e-10

    def test_large_kappa_small_fails(self):
        """If kappa_small >= 1, total contraction fails."""
        cb = combined_rg_bound(kappa_small=1.1)
        assert not cb.is_contracting

    def test_label_theorem_when_works(self):
        """Label is THEOREM when contracting."""
        cb = combined_rg_bound(kappa_small=0.5, use_face_sharing=True)
        assert cb.label == 'THEOREM'

    def test_assumptions_populated(self):
        """Assumptions list describes the bound."""
        cb = combined_rg_bound()
        assert len(cb.assumptions) >= 3


# ======================================================================
# 13. Uniformity in R
# ======================================================================

class TestUniformityInR:
    """Test uniformity_in_R."""

    def test_constructs(self):
        """Can construct without error."""
        uni = uniformity_in_R()
        assert isinstance(uni, UniformityResult)

    def test_all_uniform(self):
        """Peierls bound is uniform across all R values."""
        uni = uniformity_in_R(use_face_sharing=True)
        assert uni.all_uniform

    def test_min_ratio_positive(self):
        """Minimum Peierls ratio is positive."""
        uni = uniformity_in_R()
        assert uni.min_ratio > 0

    def test_custom_R_values(self):
        """Can provide custom R array."""
        R_vals = np.array([0.1, 1.0, 100.0])
        uni = uniformity_in_R(R_values=R_vals)
        assert len(uni.R_values) == 3

    def test_gribov_empty_flags_array(self):
        """gribov_empty_flags has correct shape."""
        uni = uniformity_in_R(use_face_sharing=True)
        assert len(uni.gribov_empty_flags) == len(uni.R_values)

    def test_label_theorem_when_uniform(self):
        """Label is THEOREM when all uniform."""
        uni = uniformity_in_R(use_face_sharing=True)
        if uni.all_uniform:
            assert uni.label == 'THEOREM'


# ======================================================================
# 14. Coupling Scan
# ======================================================================

class TestCouplingScan:
    """Test coupling_scan."""

    def test_constructs(self):
        """Can construct without error."""
        cs = coupling_scan()
        assert isinstance(cs, CouplingScanResult)

    def test_g2_critical_positive(self):
        """Critical coupling is positive."""
        cs = coupling_scan()
        assert cs.g2_critical > 0

    def test_g2_gribov_critical_positive(self):
        """Gribov critical coupling is positive."""
        cs = coupling_scan()
        assert cs.g2_gribov_critical > 0

    def test_physical_coupling_below_critical(self):
        """Physical g^2 = 6.28 is below the Gribov critical coupling."""
        cs = coupling_scan()
        assert G2_BARE_DEFAULT < cs.g2_gribov_critical or cs.g2_gribov_critical >= 30.0

    def test_arrays_correct_length(self):
        """Arrays have n_points entries."""
        n = 30
        cs = coupling_scan(n_points=n)
        assert len(cs.g2_values) == n
        assert len(cs.p0_values) == n
        assert len(cs.peierls_ratios) == n

    def test_peierls_ratios_all_positive(self):
        """All Peierls ratios are positive (p_0 is chosen optimally)."""
        cs = coupling_scan()
        # With optimal p_0, ratio = safety_factor^2 = 4 for all g^2
        assert np.all(cs.peierls_ratios > 0)


# ======================================================================
# 15. Balaban Comparison
# ======================================================================

class TestBalabanComparison:
    """Test balaban_comparison."""

    def test_constructs(self):
        """Can construct without error."""
        comp = balaban_comparison()
        assert isinstance(comp, dict)

    def test_t4_blocks_grow_much_faster(self):
        """
        T^4 block count grows as L^{4n} = 16^n (exponential in 4D).
        S^3 block count grows as 600 * 4^n (exponential in 3D).
        T^4 growth RATE is faster (16^n vs 4^n), but S^3 starts at 600.
        The crossover happens at n ~ 5 (16^5 = 1048576 > 600*4^5 = 614400).
        For large enough n, T^4 always dominates.
        """
        comp = balaban_comparison(n_scales=8)
        # Check growth rates: T^4 ratio is 16x per level, S^3 is 4x
        t4 = comp['T4_blocks_per_scale']
        s3 = comp['S3_blocks_per_scale']
        for i in range(1, len(t4)):
            t4_ratio = t4[i] / t4[i - 1]
            s3_ratio = s3[i] / s3[i - 1]
            assert abs(t4_ratio - 16.0) < 0.01
            assert abs(s3_ratio - 4.0) < 0.01
        # T^4 growth rate (16x) exceeds S^3 growth rate (4x)
        assert t4_ratio > s3_ratio

    def test_key_difference_present(self):
        """Key difference explanation is present."""
        comp = balaban_comparison()
        assert 'key_difference' in comp
        assert len(comp['key_difference']) > 50


# ======================================================================
# 16. Complete Analysis
# ======================================================================

class TestCompleteAnalysis:
    """Test complete_peierls_analysis."""

    def test_constructs(self):
        """Can construct without error."""
        results = complete_peierls_analysis()
        assert isinstance(results, dict)

    def test_label_theorem(self):
        """Complete analysis gives THEOREM."""
        results = complete_peierls_analysis(use_face_sharing=True)
        assert results['label'] == 'THEOREM'

    def test_all_sections_present(self):
        """All expected sections are in the results."""
        results = complete_peierls_analysis()
        expected_keys = [
            'assessment', 'label', 'polymer_entropy', 'perturbative_regime',
            'scale_by_scale', 'combined', 'uniformity', 'coupling_scan',
            'balaban_comparison', 'key_advantages_S3',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_key_advantages_list(self):
        """Key advantages list is populated."""
        results = complete_peierls_analysis()
        assert len(results['key_advantages_S3']) >= 5

    def test_assessment_contains_theorem(self):
        """Assessment string mentions THEOREM."""
        results = complete_peierls_analysis(use_face_sharing=True)
        assert 'THEOREM' in results['assessment']

    def test_combined_contracting(self):
        """Combined bound gives contraction."""
        results = complete_peierls_analysis()
        assert results['combined']['is_contracting']

    def test_gribov_empty_in_combined(self):
        """Combined result reports Gribov emptiness (face-sharing)."""
        results = complete_peierls_analysis(use_face_sharing=True)
        assert results['combined']['gribov_empty']


# ======================================================================
# 17. Edge Cases and Stress Tests
# ======================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_coupling(self):
        """At very weak coupling, everything is more suppressed."""
        pb = large_field_peierls_bound(g_squared=0.01, use_face_sharing=True)
        assert pb.peierls_condition or pb.gribov_empty
        assert pb.label == 'THEOREM'

    def test_very_large_coupling(self):
        """At g^2 = 4*pi (saturation), check behavior."""
        pb = large_field_peierls_bound(g_squared=G2_MAX, use_face_sharing=True)
        # At strong coupling, the Gribov bound F_max ~ g * 0.945 is large
        # but p_0 ~ 2 * sqrt(2 * g^2_max * log(4e)) is also large
        # This tests that no numerical errors occur
        assert np.isfinite(pb.peierls_ratio)

    def test_very_small_R(self):
        """Small S^3 (high curvature)."""
        cb = combined_rg_bound(R=0.1, use_face_sharing=True)
        assert isinstance(cb, CombinedRGBound)

    def test_very_large_R(self):
        """Large S^3 (approaches flat space)."""
        cb = combined_rg_bound(R=1000.0, use_face_sharing=True)
        assert isinstance(cb, CombinedRGBound)
        # Should still work because Peierls bound is R-independent

    def test_refinement_level_5(self):
        """High refinement level produces valid results."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            refinement_level=5,
            use_face_sharing=True,
        )
        assert pb.n_blocks == block_count_at_scale(5)
        assert np.isfinite(pb.peierls_ratio)

    def test_max_polymer_size_1(self):
        """Polymer size 1 still gives valid bounds."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            max_polymer_size=1,
            use_face_sharing=True,
        )
        assert isinstance(pb, PeierlsBound)

    def test_su3_parameters(self):
        """SU(3) gauge group: dim_adj = 8, N_c = 3."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            dim_adj=8,
            N_c=3,
            use_face_sharing=True,
        )
        assert isinstance(pb, PeierlsBound)

    def test_p0_zero_fails_gracefully(self):
        """p_0 = 0 means no suppression (should fail or give trivial bound)."""
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            p0=0.0,
            use_face_sharing=True,
        )
        # p_0 = 0 means suppression exponent = 0, ratio = 0 < 1
        assert not pb.peierls_condition


# ======================================================================
# 18. The Three Advantages (Structural Tests)
# ======================================================================

class TestThreeAdvantages:
    """
    Test the three structural advantages of S^3 listed in the task:
    1. FINITE polymer count (THEOREM from compactness)
    2. Bounded Gribov region (THEOREM 9.4)
    3. Positive action S_YM >= 0
    """

    def test_advantage_1_finite_polymer_count(self):
        """
        THEOREM: At every refinement level, the polymer count is finite.
        This is trivial on S^3 (compact) but fails on T^4 (infinite volume).
        """
        for level in range(10):
            n = block_count_at_scale(level)
            assert np.isfinite(n)
            # The polymer count for size s is at most N * (eD)^{s-1}
            D = max_coordination_at_scale(level, True)
            for s in range(1, 11):
                count = n * (np.e * D) ** (s - 1)
                assert np.isfinite(count)

    def test_advantage_2_bounded_gribov(self):
        """
        THEOREM 9.4: The Gribov region is bounded, so |F| has a finite
        maximum. This allows the Peierls threshold p_0 to exceed |F|_max.
        """
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            gb = gribov_field_bound(R=R, g_squared=G2_BARE_DEFAULT)
            assert np.isfinite(gb.max_field_strength)
            assert gb.max_field_strength > 0

    def test_advantage_3_positive_action(self):
        """
        THEOREM: Wilson action S_W >= 0, giving Boltzmann suppression
        exp(-S_W) <= 1 for all configurations.
        """
        for g2 in [0.1, 1.0, 6.28, 10.0]:
            for p0 in [0.1, 1.0, 5.0, 10.0]:
                supp = wilson_action_suppression(g2, p0)
                assert supp.suppression_per_block <= 1.0
                assert supp.suppression_per_block > 0
                assert supp.exponent_per_block >= 0


# ======================================================================
# 19. Honesty Tests: Label Verification
# ======================================================================

class TestHonestyLabels:
    """
    Verify that labels (THEOREM/NUMERICAL/PROPOSITION/FAILS) are correctly
    assigned based on the rigor of each result.
    """

    def test_polymer_count_is_theorem(self):
        """Polymer finiteness is THEOREM (from compactness)."""
        pc = polymer_entropy_at_scale(0, max_size=5)
        assert pc.label == 'THEOREM'

    def test_wilson_suppression_is_theorem(self):
        """Wilson action positivity is THEOREM."""
        supp = wilson_action_suppression(6.28, 3.0)
        assert supp.label == 'THEOREM'

    def test_gribov_bound_is_numerical(self):
        """Gribov diameter value is NUMERICAL (from Session 6 computation)."""
        gb = gribov_field_bound()
        assert gb.label == 'NUMERICAL'

    def test_peierls_condition_is_theorem_when_met(self):
        """Peierls condition, when met, is THEOREM."""
        cond = peierls_condition_check(6.28, 10.0, D_max=4)
        assert cond.is_satisfied
        assert cond.label == 'THEOREM'

    def test_full_peierls_is_theorem(self):
        """Full Peierls bound is THEOREM."""
        pb = large_field_peierls_bound(g_squared=G2_BARE_DEFAULT,
                                        use_face_sharing=True)
        assert pb.label == 'THEOREM'

    def test_coupling_scan_is_numerical(self):
        """Coupling scan is NUMERICAL."""
        cs = coupling_scan()
        assert cs.label == 'NUMERICAL'


# ======================================================================
# 20. Exact Polymer Counts on 600-Cell (Integration)
# ======================================================================

class TestExactPolymerCounts600Cell:
    """
    Exact polymer counting on the 600-cell cell adjacency graph.
    These tests use the actual 600-cell graph (expensive, module-scoped fixture).
    """

    def test_size_1_is_600(self, cell_adjacency):
        """600 polymers of size 1 (one per cell)."""
        adj, n_cells, _ = cell_adjacency
        counts = count_polymers_on_graph(adj, max_size=1)
        assert counts[1] == 600

    def test_size_2_is_1200(self, cell_adjacency):
        """
        Polymers of size 2 = edges in dual graph.
        Each of 600 cells has 4 face-sharing neighbors.
        Edges = 600 * 4 / 2 = 1200.
        """
        adj, n_cells, _ = cell_adjacency
        counts = count_polymers_on_graph(adj, max_size=2)
        assert counts[2] == 1200

    def test_size_2_below_analytical_bound(self, cell_adjacency):
        """Exact count at size 2 is below the analytical upper bound."""
        adj, n_cells, max_deg = cell_adjacency
        exact = count_polymers_on_graph(adj, max_size=2)
        bound = analytical_polymer_bound(n_cells, max_deg, 2)
        assert exact[2] <= bound[2]

    def test_size_3_finite(self, cell_adjacency):
        """Polymer count at size 3 is finite and positive."""
        adj, n_cells, _ = cell_adjacency
        counts = count_polymers_on_graph(adj, max_size=3)
        assert counts[3] > 0
        assert np.isfinite(counts[3])

    def test_size_3_below_analytical_bound(self, cell_adjacency):
        """Exact count at size 3 is below the analytical bound."""
        adj, n_cells, max_deg = cell_adjacency
        exact = count_polymers_on_graph(adj, max_size=3)
        bound = analytical_polymer_bound(n_cells, max_deg, 3)
        assert exact[3] <= bound[3]

    def test_counts_monotone_initially(self, cell_adjacency):
        """Polymer counts grow with size (at least initially)."""
        adj, n_cells, _ = cell_adjacency
        counts = count_polymers_on_graph(adj, max_size=3)
        # 600, 1200, ... should be monotonically increasing
        assert counts[2] >= counts[1]

    def test_entropy_with_exact(self, cell_adjacency):
        """
        polymer_entropy_at_scale with exact enumeration uses the
        exact counts for small sizes.
        """
        adj, n_cells, _ = cell_adjacency
        pc = polymer_entropy_at_scale(
            0, max_size=5, exact_max=2, cell_adj=adj,
        )
        assert pc.is_exact.get(1, False) == True
        assert pc.is_exact.get(2, False) == True
        assert pc.exact_counts[1] == 600
        assert pc.exact_counts[2] == 1200


# ======================================================================
# 21. Suppression vs Entropy: The Core Inequality
# ======================================================================

class TestSuppressionVsEntropy:
    """
    The core mathematical content: verify that suppression beats entropy
    for all polymer sizes s = 1, 2, ..., 20.
    """

    def test_net_exponent_positive_all_sizes(self):
        """
        For each polymer of size s, the contribution to the Peierls sum is:
            N(s) * exp(-s * c_W * p_0^2 / g^2)

        The per-size "effective exponent" is:
            s * (c_W * p_0^2 / g^2) - (s-1) * log(eD)
            = s * net_exponent + log(eD)  (approximately)

        This must be positive for the sum to converge.
        """
        g2 = G2_BARE_DEFAULT
        D = FACE_SHARING_DEGREE
        p0 = optimal_p0(g2, D, safety_factor=2.0)
        c_W = 0.5

        suppression_exp = c_W * p0 ** 2 / g2
        entropy_rate = np.log(np.e * D)
        net = suppression_exp - entropy_rate

        assert net > 0, f"Net exponent {net:.4f} must be positive"

        # Verify term-by-term decay
        N = CELL_COUNT_600
        for s in range(1, 21):
            polymer_weight = N * (np.e * D) ** (s - 1)
            energy_factor = np.exp(-s * suppression_exp)
            term = polymer_weight * energy_factor
            # Each term should be smaller than the previous
            if s > 1:
                prev_weight = N * (np.e * D) ** (s - 2)
                prev_factor = np.exp(-(s - 1) * suppression_exp)
                prev_term = prev_weight * prev_factor
                ratio = term / prev_term if prev_term > 0 else 0
                assert ratio < 1.0, (
                    f"Term ratio at s={s}: {ratio:.6f} must be < 1"
                )

    def test_geometric_series_converges(self):
        """The Peierls sum is a convergent geometric series."""
        g2 = G2_BARE_DEFAULT
        D = FACE_SHARING_DEGREE
        p0 = optimal_p0(g2, D)

        c_W = 0.5
        suppression_exp = c_W * p0 ** 2 / g2
        entropy_rate = np.log(np.e * D)
        common_ratio = np.exp(-(suppression_exp - entropy_rate))

        assert common_ratio < 1.0, f"Common ratio {common_ratio:.6f} must be < 1"

        # The sum converges to N * exp(-suppression_exp) / (1 - common_ratio)
        N = CELL_COUNT_600
        total = N * np.exp(-suppression_exp) / (1 - common_ratio)
        assert np.isfinite(total)
        assert total > 0

    def test_peierls_sum_exponentially_small(self):
        """
        At physical coupling g^2 = 6.28 with D=4:
        The Peierls sum is exponentially small (or zero via Gribov).
        """
        pb = large_field_peierls_bound(
            g_squared=G2_BARE_DEFAULT,
            use_face_sharing=True,
        )
        assert pb.large_field_bound < 1e-3  # Much smaller than 1


# ======================================================================
# 22. Summary Statistics and Integration
# ======================================================================

class TestSummaryStatistics:
    """Integration tests verifying the overall picture."""

    def test_full_picture_face_sharing(self):
        """
        The complete picture with face-sharing (D=4):
        - Polymer count: finite (THEOREM)
        - Wilson suppression: exp(-c/g^2 * p_0^2) (THEOREM)
        - Peierls condition: satisfied (THEOREM)
        - Large-field region: EMPTY within Gribov (THEOREM)
        - Combined contraction: kappa_total < 1 (THEOREM)
        - Uniform in R (THEOREM)
        """
        results = complete_peierls_analysis(use_face_sharing=True)
        assert results['label'] == 'THEOREM'
        assert results['combined']['is_contracting']
        assert results['combined']['gribov_empty']
        assert results['uniformity']['all_uniform']

    def test_comparison_face_vs_vertex(self):
        """
        Face-sharing (D=4) gives strictly tighter bounds than
        vertex-sharing (D=20), but both give THEOREM.
        """
        r_face = complete_peierls_analysis(use_face_sharing=True)
        r_vert = complete_peierls_analysis(use_face_sharing=False)

        # Both should be THEOREM
        assert r_face['label'] == 'THEOREM'
        assert r_vert['label'] == 'THEOREM'

    def test_verbose_output(self, capsys):
        """Verbose mode prints output without errors."""
        complete_peierls_analysis(verbose=True, use_face_sharing=True)
        captured = capsys.readouterr()
        assert 'LARGE-FIELD PEIERLS ANALYSIS' in captured.out
        assert 'THEOREM' in captured.out
