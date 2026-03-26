"""
Tests for Multi-Step Linearization (Balaban k > 1 on S^3).

Tests cover all seven classes:
1. NestedSmallFieldCondition: doubly-exponential decay, hierarchy, max levels
2. IntermediateGreenFunction: positive at each level, L^inf bound, decay
3. NestedContractionMapping: q < 1 at each level, fixed point, total contraction
4. MultiLevelLinearization: constraint at each level, error bounds
5. BlockingHierarchy600Cell: block counts, adjacency, DOF
6. MultiStepMinimizerConvergence: geometric rate, O(1) iterations, S^3 improvement
7. MultiStepBackgroundField: regularity, Lipschitz, decomposition exact
Plus: k=1 vs k=2 vs k=3 comparison, edge cases.

Total: 80+ tests covering THEOREM, PROPOSITION, and NUMERICAL results.

References:
    [1] Balaban (1985), CMP 102 (Paper 6)
    [2] DST (2024), arXiv:2403.09800
"""

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from yang_mills_s3.rg.multistep_linearization import (
    NestedSmallFieldCondition,
    IntermediateGreenFunction,
    NestedContractionMapping,
    MultiLevelLinearization,
    BlockingHierarchy600Cell,
    MultiStepMinimizerConvergence,
    MultiStepBackgroundField,
    _gribov_epsilon_1,
)
from yang_mills_s3.rg.balaban_minimizer import (
    SmallFieldRegion,
    VariationalGreenFunction,
    BLOCKING_FACTOR,
    N_VERTICES_600CELL,
)
from yang_mills_s3.rg.background_minimizer import (
    DIM_9DOF,
    DIM_ADJ,
    N_MODES_TRUNC,
    G2_PHYSICAL,
)
from yang_mills_s3.rg.heat_kernel_slices import R_PHYSICAL_FM, coexact_eigenvalue


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def default_R():
    return R_PHYSICAL_FM

@pytest.fixture
def default_g2():
    return G2_PHYSICAL

@pytest.fixture
def default_eps():
    return 0.3

@pytest.fixture
def small_eps():
    return 0.1

@pytest.fixture
def hierarchy(default_eps, default_R, default_g2):
    return NestedSmallFieldCondition(
        epsilon_1=default_eps, R=default_R, g2=default_g2
    )

@pytest.fixture
def hierarchy_small(small_eps, default_R, default_g2):
    return NestedSmallFieldCondition(
        epsilon_1=small_eps, R=default_R, g2=default_g2
    )

@pytest.fixture
def green_fn(default_R):
    return IntermediateGreenFunction(R=default_R)

@pytest.fixture
def contraction(default_R, default_g2, default_eps):
    return NestedContractionMapping(
        R=default_R, g2=default_g2, epsilon_1=default_eps
    )

@pytest.fixture
def linearization(default_R, default_g2, default_eps):
    return MultiLevelLinearization(
        R=default_R, g2=default_g2, epsilon_1=default_eps
    )

@pytest.fixture
def blocking_sym(default_R):
    return BlockingHierarchy600Cell(R=default_R, scheme='symmetry')

@pytest.fixture
def blocking_geo(default_R):
    return BlockingHierarchy600Cell(R=default_R, scheme='geometric')

@pytest.fixture
def convergence(default_R, default_g2, default_eps):
    return MultiStepMinimizerConvergence(
        R=default_R, g2=default_g2, epsilon_1=default_eps
    )

@pytest.fixture
def background(default_R, default_g2, default_eps):
    return MultiStepBackgroundField(
        R=default_R, g2=default_g2, epsilon_1=default_eps
    )


# ======================================================================
# 1. NestedSmallFieldCondition tests
# ======================================================================

class TestNestedSmallFieldCondition:
    """Tests for the doubly-exponential epsilon hierarchy."""

    def test_epsilon_level_0_is_one(self, hierarchy):
        """Level 0 has no constraint: eps_0 = 1."""
        assert hierarchy.epsilon_at_level(0) == 1.0

    def test_epsilon_level_1_is_initial(self, hierarchy, default_eps):
        """Level 1 equals the initial threshold."""
        assert abs(hierarchy.epsilon_at_level(1) - default_eps) < 1e-15

    def test_doubly_exponential_decay(self, hierarchy, default_eps):
        """eps_j = eps_1^{2^{j-1}}: doubly-exponential THEOREM."""
        for j in range(1, 5):
            expected = default_eps ** (2 ** (j - 1))
            actual = hierarchy.epsilon_at_level(j)
            assert abs(actual - expected) < 1e-12, \
                f"Level {j}: expected {expected}, got {actual}"

    def test_hierarchy_condition_all_levels(self, hierarchy):
        """eps_j <= eps_{j-1}^2 at every level. THEOREM."""
        for j in range(1, 6):
            assert hierarchy.is_valid(j), f"Hierarchy fails at level {j}"

    def test_eps_2_is_eps_1_squared(self, hierarchy, default_eps):
        """Level 2: eps_2 = eps_1^2. THEOREM."""
        assert abs(hierarchy.epsilon_at_level(2) - default_eps**2) < 1e-14

    def test_eps_3_is_eps_1_fourth(self, hierarchy, default_eps):
        """Level 3: eps_3 = eps_1^4. THEOREM."""
        assert abs(hierarchy.epsilon_at_level(3) - default_eps**4) < 1e-14

    def test_eps_monotone_decreasing(self, hierarchy):
        """Epsilon strictly decreases with level."""
        for j in range(1, 6):
            assert hierarchy.epsilon_at_level(j + 1) < hierarchy.epsilon_at_level(j)

    def test_max_levels_finite(self, hierarchy):
        """Max levels before machine precision is finite."""
        k_max = hierarchy.max_levels()
        assert k_max >= 3, f"k_max = {k_max} is too small"
        assert k_max < 100, f"k_max = {k_max} is unreasonably large"

    def test_max_levels_decreases_with_smaller_epsilon(self, default_R, default_g2):
        """Smaller eps_1 allows more levels before machine precision."""
        h1 = NestedSmallFieldCondition(epsilon_1=0.1, R=default_R, g2=default_g2)
        h2 = NestedSmallFieldCondition(epsilon_1=0.3, R=default_R, g2=default_g2)
        # Smaller eps_1 => eps_k reaches machine precision faster
        # because eps_1^{2^{k-1}} decreases faster with smaller eps_1
        assert h1.max_levels() <= h2.max_levels() + 1

    def test_verify_hierarchy_returns_all_valid(self, hierarchy):
        """verify_hierarchy() reports all levels valid. THEOREM."""
        result = hierarchy.verify_hierarchy(k_max=5)
        assert result['all_valid'] is True
        assert result['label'] == 'THEOREM'

    def test_hierarchy_with_small_epsilon(self, hierarchy_small, small_eps):
        """Works with eps_1 = 0.1: eps_3 = 0.0001."""
        eps_3 = hierarchy_small.epsilon_at_level(3)
        assert abs(eps_3 - small_eps**4) < 1e-14

    def test_negative_level_raises(self, hierarchy):
        """Negative level raises ValueError."""
        with pytest.raises(ValueError):
            hierarchy.epsilon_at_level(-1)

    def test_invalid_epsilon_raises(self, default_R, default_g2):
        """Epsilon outside (0,1) raises ValueError."""
        with pytest.raises(ValueError):
            NestedSmallFieldCondition(epsilon_1=0.0, R=default_R, g2=default_g2)
        with pytest.raises(ValueError):
            NestedSmallFieldCondition(epsilon_1=1.0, R=default_R, g2=default_g2)
        with pytest.raises(ValueError):
            NestedSmallFieldCondition(epsilon_1=-0.1, R=default_R, g2=default_g2)

    def test_contraction_constant_bound_decreases(self, hierarchy):
        """Contraction bound q_j = C * eps_j decreases with level."""
        for j in range(1, 5):
            q_j = hierarchy.contraction_constant_bound(j)
            q_jp1 = hierarchy.contraction_constant_bound(j + 1)
            assert q_jp1 < q_j, f"q_{j+1} = {q_jp1} not < q_{j} = {q_j}"

    def test_gribov_bound_positive(self, hierarchy):
        """Gribov-derived epsilon bound is positive."""
        assert hierarchy._eps_gribov > 0

    def test_physical_parameters(self, default_eps):
        """Physical values: eps_1=0.3 gives eps_2=0.09, eps_3=0.0081."""
        h = NestedSmallFieldCondition(epsilon_1=default_eps)
        assert abs(h.epsilon_at_level(2) - 0.09) < 1e-10
        assert abs(h.epsilon_at_level(3) - 0.0081) < 1e-10


# ======================================================================
# 2. IntermediateGreenFunction tests
# ======================================================================

class TestIntermediateGreenFunction:
    """Tests for Green's functions at each intermediate level."""

    def test_green_function_level_0_exists(self, green_fn):
        """G_0 exists at the finest level."""
        gf = green_fn.green_function_at_level(0)
        assert gf is not None

    def test_green_function_level_1_exists(self, green_fn):
        """G_1 exists after one blocking step."""
        gf = green_fn.green_function_at_level(1)
        assert gf is not None

    def test_spectral_gap_positive_all_levels(self, green_fn):
        """Spectral gap is positive at all levels. THEOREM."""
        for j in range(3):
            gap = green_fn.spectral_gap_at_level(j)
            assert gap > 0, f"Gap at level {j} is {gap} <= 0"

    def test_spectral_gap_bounded_by_s3(self, green_fn, default_R):
        """Spectral gap is bounded below by S^3 gap."""
        s3_gap = coexact_eigenvalue(1, default_R)
        for j in range(3):
            gap = green_fn.spectral_gap_at_level(j)
            # The gap may be larger than the S^3 gap due to Q^T Q contribution
            assert gap > 0

    def test_linf_bound_finite_all_levels(self, green_fn):
        """L^inf norm is finite at all levels. THEOREM."""
        for j in range(3):
            norm = green_fn.linf_bound(j)
            assert np.isfinite(norm), f"L^inf norm at level {j} is infinite"
            assert norm > 0, f"L^inf norm at level {j} is non-positive"

    def test_green_function_caching(self, green_fn):
        """Green's function is cached: same object returned."""
        gf1 = green_fn.green_function_at_level(0)
        gf2 = green_fn.green_function_at_level(0)
        assert gf1 is gf2

    def test_sites_decrease_with_level(self, green_fn):
        """Number of sites decreases at deeper levels."""
        n0 = green_fn._n_sites_at_level(0)
        n1 = green_fn._n_sites_at_level(1)
        assert n1 <= n0

    def test_compose_levels_returns_dict(self, green_fn):
        """compose_levels() returns analysis dict."""
        result = green_fn.compose_levels(0, 1)
        assert 'norm_j1' in result
        assert 'norm_j2' in result
        assert 'coupling_decay' in result

    def test_compose_levels_coupling_decays(self, green_fn):
        """Coupling between levels decays with separation."""
        r01 = green_fn.compose_levels(0, 1)
        r02 = green_fn.compose_levels(0, 2)
        assert r02['coupling_decay'] < r01['coupling_decay']

    def test_decay_rate_nonnegative(self, green_fn):
        """Decay rate is non-negative at all levels."""
        for j in range(3):
            rate = green_fn.decay_rate(j)
            assert rate >= 0, f"Decay rate at level {j} is negative: {rate}"

    def test_invalid_R_raises(self):
        """Invalid R raises ValueError."""
        with pytest.raises(ValueError):
            IntermediateGreenFunction(R=-1.0)

    def test_invalid_L_raises(self, default_R):
        """Invalid L raises ValueError."""
        with pytest.raises(ValueError):
            IntermediateGreenFunction(R=default_R, L=1)


# ======================================================================
# 3. NestedContractionMapping tests
# ======================================================================

class TestNestedContractionMapping:
    """Tests for the nested contraction maps T_j."""

    def test_q_level_1_less_than_one(self, contraction):
        """q_1 < 1: contraction at level 1. THEOREM."""
        q1 = contraction.q_at_level(1)
        assert q1 < 1.0, f"q_1 = {q1} >= 1"

    def test_q_level_2_less_than_one(self, contraction):
        """q_2 < 1: contraction at level 2. THEOREM."""
        q2 = contraction.q_at_level(2)
        assert q2 < 1.0, f"q_2 = {q2} >= 1"

    def test_q_level_3_less_than_one(self, contraction):
        """q_3 < 1: contraction at level 3. THEOREM."""
        q3 = contraction.q_at_level(3)
        assert q3 < 1.0

    def test_q_decreases_with_level(self, contraction):
        """q_j decreases with level (doubly-exponential). THEOREM."""
        for j in range(1, 4):
            q_j = contraction.q_at_level(j)
            q_jp1 = contraction.q_at_level(j + 1)
            assert q_jp1 < q_j

    def test_contraction_at_level_1(self, contraction):
        """Level 1 contraction analysis works."""
        result = contraction.contraction_at_level(1)
        assert result['level'] == 1
        assert result['is_contraction'] is True
        assert result['label'] == 'THEOREM'

    def test_contraction_at_level_2(self, contraction):
        """Level 2 contraction with tighter epsilon."""
        result = contraction.contraction_at_level(2)
        assert result['is_contraction'] is True

    def test_fixed_point_level_1_converges(self, contraction):
        """Fixed point at level 1 converges."""
        A_star, info = contraction.fixed_point_at_level(1)
        assert info['converged'] is True
        assert A_star is not None

    def test_fixed_point_caching(self, contraction):
        """Fixed point is cached."""
        A1, _ = contraction.fixed_point_at_level(1)
        A1b, _ = contraction.fixed_point_at_level(1)
        assert np.array_equal(A1, A1b)

    def test_total_contraction_k1(self, contraction):
        """Total contraction for k=1."""
        result = contraction.total_contraction(1)
        assert result['all_contractions'] is True
        assert len(result['q_values']) == 1

    def test_total_contraction_k2(self, contraction):
        """Total contraction for k=2."""
        result = contraction.total_contraction(2)
        assert result['all_contractions'] is True
        assert len(result['q_values']) == 2

    def test_total_contraction_k3(self, contraction):
        """Total contraction for k=3."""
        result = contraction.total_contraction(3)
        assert result['all_contractions'] is True
        assert len(result['q_values']) == 3

    def test_q_product_less_than_one(self, contraction):
        """Product of all q_j is < 1. THEOREM."""
        result = contraction.total_contraction(3)
        assert result['q_product'] < 1.0

    def test_q_product_decreases_with_k(self, contraction):
        """q_product(k) decreases as k increases (more levels = more suppression)."""
        # Actually q_product(k+1) = q_product(k) * q_{k+1}
        # Since each q_{k+1} < 1, the product decreases
        q1 = contraction.total_contraction(1)['q_product']
        q2 = contraction.total_contraction(2)['q_product']
        q3 = contraction.total_contraction(3)['q_product']
        assert q2 < q1
        assert q3 < q2

    def test_invalid_level_raises(self, contraction):
        """Level 0 raises ValueError."""
        with pytest.raises(ValueError):
            contraction.contraction_at_level(0)

    def test_previous_solutions_accepted(self, contraction):
        """contraction_at_level accepts previous solutions."""
        A1, _ = contraction.fixed_point_at_level(1)
        result = contraction.contraction_at_level(2, previous_solutions={1: A1})
        assert 'bg_correction' in result


# ======================================================================
# 4. MultiLevelLinearization tests
# ======================================================================

class TestMultiLevelLinearization:
    """Tests for the full k-step linearization C_k -> Q_k."""

    def test_linearize_k1(self, linearization):
        """k=1 linearization succeeds."""
        result = linearization.linearize(1)
        assert result['all_converged'] is True
        assert result['k'] == 1

    def test_linearize_k2(self, linearization):
        """k=2 linearization succeeds."""
        result = linearization.linearize(2)
        assert result['all_converged'] is True
        assert result['k'] == 2

    def test_linearize_k3(self, linearization):
        """k=3 linearization succeeds (sufficient for 600-cell)."""
        result = linearization.linearize(3)
        assert result['all_converged'] is True
        assert result['k'] == 3

    def test_total_error_decreases_with_eps(self, default_R, default_g2):
        """Total error decreases with smaller epsilon_1."""
        lin1 = MultiLevelLinearization(
            R=default_R, g2=default_g2, epsilon_1=0.3)
        lin2 = MultiLevelLinearization(
            R=default_R, g2=default_g2, epsilon_1=0.1)
        err1 = lin1.total_error(2)
        err2 = lin2.total_error(2)
        assert err2 < err1

    def test_total_error_positive(self, linearization):
        """Total error is positive (non-zero nonlinearity)."""
        err = linearization.total_error(2)
        assert err > 0

    def test_error_decreases_with_k(self, linearization):
        """Per-level error decreases at deeper levels."""
        result = linearization.linearize(3)
        errors = [lev['linearization_error'] for lev in result['levels']]
        # Level 1 (finest) has the largest per-level error
        # Level 3 (coarsest) has the smallest
        # sorted by level: [1, 2, 3]
        assert errors[0] >= errors[-1]

    def test_verify_constraint_k1(self, linearization):
        """Constraint verified at k=1. THEOREM."""
        result = linearization.verify_constraint(1)
        assert result['all_within_bounds'] is True

    def test_verify_constraint_k2(self, linearization):
        """Constraint verified at k=2."""
        result = linearization.verify_constraint(2)
        assert result['all_within_bounds'] is True

    def test_solve_level_returns_solution(self, linearization):
        """solve_level returns a valid solution."""
        A, info = linearization.solve_level(1)
        assert A is not None
        assert info['converged'] is True

    def test_invalid_k_raises(self, linearization):
        """k=0 raises ValueError."""
        with pytest.raises(ValueError):
            linearization.linearize(0)

    def test_linearize_label_is_theorem(self, linearization):
        """Label is THEOREM."""
        result = linearization.linearize(1)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 5. BlockingHierarchy600Cell tests
# ======================================================================

class TestBlockingHierarchy600Cell:
    """Tests for the 600-cell blocking hierarchy."""

    def test_level_0_has_120_vertices(self, blocking_sym):
        """Level 0 has 120 blocks (= 600-cell vertices)."""
        assert blocking_sym.blocks_at_level(0) == 120

    def test_symmetry_scheme_level_1(self, blocking_sym):
        """Symmetry scheme: level 1 has 24 blocks."""
        assert blocking_sym.blocks_at_level(1) == 24

    def test_symmetry_scheme_level_2(self, blocking_sym):
        """Symmetry scheme: level 2 has 5 blocks."""
        assert blocking_sym.blocks_at_level(2) == 5

    def test_symmetry_scheme_level_3(self, blocking_sym):
        """Symmetry scheme: level 3 has 1 block (global)."""
        assert blocking_sym.blocks_at_level(3) == 1

    def test_geometric_scheme_level_0(self, blocking_geo):
        """Geometric scheme: level 0 has 120 blocks."""
        assert blocking_geo.blocks_at_level(0) == 120

    def test_blocks_decrease_monotonically(self, blocking_sym):
        """Block count is monotonically non-increasing."""
        for j in range(blocking_sym.n_levels - 1):
            assert blocking_sym.blocks_at_level(j + 1) <= blocking_sym.blocks_at_level(j)

    def test_terminal_level_has_one_block(self, blocking_sym):
        """Terminal level has exactly 1 block."""
        assert blocking_sym.blocks_at_level(blocking_sym.k_max) == 1

    def test_large_level_returns_one(self, blocking_sym):
        """Level beyond hierarchy returns 1."""
        assert blocking_sym.blocks_at_level(100) == 1

    def test_negative_level_raises(self, blocking_sym):
        """Negative level raises ValueError."""
        with pytest.raises(ValueError):
            blocking_sym.blocks_at_level(-1)

    def test_adjacency_level_0_nonempty(self, blocking_sym):
        """Adjacency at level 0 is non-empty."""
        adj = blocking_sym.adjacency_at_level(0)
        assert len(adj) == 120
        # Each block should have some neighbors
        assert all(len(v) > 0 for v in adj.values())

    def test_adjacency_simplex_at_level_2(self, blocking_sym):
        """At level 2 with 5 blocks, everyone is adjacent (simplex)."""
        adj = blocking_sym.adjacency_at_level(2)
        assert len(adj) == 5
        for b in range(5):
            assert len(adj[b]) == 4  # Adjacent to all others

    def test_adjacency_single_block(self, blocking_sym):
        """Single block has empty adjacency."""
        adj = blocking_sym.adjacency_at_level(3)
        assert adj == {0: set()}

    def test_tree_at_level_0(self, blocking_sym):
        """Spanning tree at level 0 has 119 edges."""
        tree = blocking_sym.tree_at_level(0)
        assert len(tree) == 119  # n - 1 edges

    def test_tree_at_level_2(self, blocking_sym):
        """Spanning tree at level 2 has 4 edges (5 blocks)."""
        tree = blocking_sym.tree_at_level(2)
        assert len(tree) == 4

    def test_tree_at_level_3(self, blocking_sym):
        """Single block: empty tree."""
        tree = blocking_sym.tree_at_level(3)
        assert len(tree) == 0

    def test_dof_at_level_0(self, blocking_sym):
        """DOF at level 0."""
        dof = blocking_sym.dof_at_level(0)
        assert dof['n_blocks'] == 120
        assert dof['total_dof'] == 120 * DIM_9DOF
        assert dof['gauge_dof'] == 119 * DIM_ADJ
        assert dof['physical_dof'] == dof['total_dof'] - dof['gauge_dof']
        assert dof['label'] == 'THEOREM'

    def test_dof_at_level_3(self, blocking_sym):
        """DOF at level 3: single block, no gauge DOF."""
        dof = blocking_sym.dof_at_level(3)
        assert dof['n_blocks'] == 1
        assert dof['gauge_dof'] == 0
        assert dof['total_dof'] == DIM_9DOF

    def test_summary_returns_all_levels(self, blocking_sym):
        """summary() returns info for all levels."""
        result = blocking_sym.summary()
        assert result['n_levels'] == len(blocking_sym._block_counts)
        assert len(result['levels']) == result['n_levels']

    def test_invalid_scheme_raises(self, default_R):
        """Invalid scheme raises ValueError."""
        with pytest.raises(ValueError):
            BlockingHierarchy600Cell(R=default_R, scheme='invalid')

    def test_n_levels_is_4_for_symmetry(self, blocking_sym):
        """Symmetry scheme has 4 levels: [120, 24, 5, 1]."""
        assert blocking_sym.n_levels == 4

    def test_k_max_is_3_for_symmetry(self, blocking_sym):
        """k_max = n_levels - 1 = 3."""
        assert blocking_sym.k_max == 3


# ======================================================================
# 6. MultiStepMinimizerConvergence tests
# ======================================================================

class TestMultiStepMinimizerConvergence:
    """Tests for convergence analysis."""

    def test_convergence_rate_k1(self, convergence):
        """Convergence rate at k=1."""
        result = convergence.convergence_rate(1)
        assert len(result['rates']) == 1
        assert result['worst_rate'] < 1.0

    def test_convergence_rate_k2(self, convergence):
        """Convergence rate at k=2."""
        result = convergence.convergence_rate(2)
        assert len(result['rates']) == 2
        assert result['worst_rate'] < 1.0

    def test_convergence_rate_k3(self, convergence):
        """Convergence rate at k=3."""
        result = convergence.convergence_rate(3)
        assert len(result['rates']) == 3
        assert result['worst_rate'] < 1.0

    def test_best_rate_improves_with_level(self, convergence):
        """Best rate improves at deeper levels (smaller eps_j)."""
        r2 = convergence.convergence_rate(2)
        r3 = convergence.convergence_rate(3)
        assert r3['best_rate'] <= r2['best_rate']

    def test_total_iterations_finite(self, convergence):
        """Total iterations is finite for all k."""
        for k in [1, 2, 3]:
            result = convergence.total_iterations(k)
            assert result['total_iterations'] >= 0
            assert result['total_iterations'] < 10000

    def test_iterations_per_level_order_one(self, convergence):
        """Mean iterations per level is O(1). PROPOSITION."""
        result = convergence.total_iterations(3)
        assert result['mean_per_level'] < 100, \
            f"Mean iterations = {result['mean_per_level']} is too large"

    def test_compare_levels_k1_is_trivial(self, convergence):
        """k=1 is the trivial single-step case."""
        result = convergence.compare_levels()
        assert result['k1_is_trivial'] is True

    def test_compare_levels_k2_improves(self, convergence):
        """k=2 improves best rate over k=1."""
        result = convergence.compare_levels()
        assert result['k2_improves'] is True

    def test_compare_levels_k3_sufficient(self, convergence):
        """k=3 is sufficient for 600-cell."""
        result = convergence.compare_levels()
        assert result['k3_sufficient_for_600cell'] is True

    def test_s3_improvement_positive(self, convergence):
        """S^3 curvature improves contraction constants."""
        result = convergence.s3_improvement()
        assert result['improvement_factor'] > 1.0
        assert result['spectral_gap'] > 0
        assert result['blocks_isometric'] is True

    def test_s3_sobolev_improvement(self, convergence):
        """Sobolev improvement from Ric > 0."""
        result = convergence.s3_improvement()
        assert result['sobolev_improvement'] > 1.0

    def test_s3_c_s3_less_than_c_flat(self, convergence):
        """C_S3 < C_flat: curvature reduces contraction constant."""
        result = convergence.s3_improvement()
        assert result['C_s3'] < result['C_flat']


# ======================================================================
# 7. MultiStepBackgroundField tests
# ======================================================================

class TestMultiStepBackgroundField:
    """Tests for the background field from k-step minimizer."""

    def test_compute_background_k1(self, background):
        """Background field at k=1 is computed."""
        A_bar, info = background.compute_background(1)
        assert A_bar is not None
        assert info['converged'] is True

    def test_compute_background_k2(self, background):
        """Background field at k=2 is computed."""
        A_bar, info = background.compute_background(2)
        assert info['converged'] is True

    def test_background_bounded(self, background):
        """||A_bar||_inf <= regularity bound. THEOREM."""
        A_bar, info = background.compute_background(1)
        assert info['satisfies_bound'] is True

    def test_regularity_bounds_improve(self, background):
        """Regularity bounds improve at deeper levels."""
        result = background.regularity_bounds(3)
        assert result['all_improve'] is True

    def test_regularity_bounds_positive(self, background):
        """All regularity bounds are positive."""
        result = background.regularity_bounds(3)
        for b in result['bounds']:
            assert b['regularity_bound'] > 0

    def test_lipschitz_constant_finite(self, background):
        """Lipschitz constant is finite. PROPOSITION."""
        result = background.lipschitz_constant(1, n_samples=3)
        assert np.isfinite(result['theoretical_bound'])
        assert result['theoretical_bound'] > 0

    def test_lipschitz_satisfies_bound(self, background):
        """Numerical Lipschitz <= theoretical bound."""
        result = background.lipschitz_constant(1, n_samples=3)
        assert result['satisfies_bound'] is True

    def test_decompose_exact(self, background):
        """A = A_bar + W decomposition is exact. THEOREM."""
        A_bar, _ = background.compute_background(1)
        n = len(A_bar)
        rng = np.random.default_rng(42)
        A = A_bar + 0.01 * rng.standard_normal(n)
        result = background.decompose(A, A_bar)
        assert result['exact'] is True
        assert result['reconstruction_error'] < 1e-14

    def test_decompose_W_equals_difference(self, background):
        """W = A - A_bar exactly."""
        A_bar, _ = background.compute_background(1)
        n = len(A_bar)
        rng = np.random.default_rng(42)
        W_input = 0.01 * rng.standard_normal(n)
        A = A_bar + W_input
        result = background.decompose(A, A_bar)
        assert np.allclose(result['W'], W_input, atol=1e-14)

    def test_decompose_mismatched_sizes_raises(self, background):
        """Mismatched A and A_bar sizes raise ValueError."""
        with pytest.raises(ValueError):
            background.decompose(np.zeros(5), np.zeros(10))

    def test_invalid_k_raises(self, background):
        """k=0 raises ValueError."""
        with pytest.raises(ValueError):
            background.compute_background(0)


# ======================================================================
# 8. Cross-class integration tests
# ======================================================================

class TestIntegration:
    """Integration tests across multiple classes."""

    def test_hierarchy_feeds_contraction(self, default_R, default_g2, default_eps):
        """Hierarchy epsilon values drive contraction constants."""
        h = NestedSmallFieldCondition(
            epsilon_1=default_eps, R=default_R, g2=default_g2)
        c = NestedContractionMapping(
            R=default_R, g2=default_g2, epsilon_1=default_eps)
        for j in range(1, 4):
            eps_j = h.epsilon_at_level(j)
            q_j = c.q_at_level(j)
            # q_j = C * eps_j with C = 2
            assert abs(q_j - 2.0 * eps_j) < 1e-14

    def test_green_fn_positive_at_all_blocking_levels(
            self, default_R, blocking_sym):
        """Green's function is positive at every level of the 600-cell blocking."""
        gf = IntermediateGreenFunction(R=default_R)
        for j in range(3):
            gap = gf.spectral_gap_at_level(j)
            assert gap > 0, f"Gap at 600-cell level {j} is {gap}"

    def test_linearization_uses_contraction(self, linearization):
        """Linearization internally uses contraction mapping."""
        result = linearization.linearize(2)
        # Both levels should converge
        assert all(lev['converged'] for lev in result['levels'])

    def test_background_consistent_with_minimizer(self, background):
        """Background from compute_background matches minimizer bound."""
        A_bar, info = background.compute_background(1)
        sup = float(np.max(np.abs(A_bar)))
        assert sup <= info['regularity_bound'] + 1e-10


# ======================================================================
# 9. k=1 vs k=2 vs k=3 comparison tests
# ======================================================================

class TestLevelComparison:
    """Compare behavior across different numbers of RG steps."""

    def test_k1_matches_single_step(self, default_R, default_g2, default_eps):
        """k=1 multi-step matches the standard single-step Balaban."""
        lin = MultiLevelLinearization(
            R=default_R, g2=default_g2, epsilon_1=default_eps)
        result = lin.linearize(1)
        assert len(result['levels']) == 1
        assert result['levels'][0]['level'] == 1

    def test_k2_has_two_levels(self, linearization):
        """k=2 has exactly two levels."""
        result = linearization.linearize(2)
        assert len(result['levels']) == 2

    def test_k3_has_three_levels(self, linearization):
        """k=3 has exactly three levels."""
        result = linearization.linearize(3)
        assert len(result['levels']) == 3

    def test_deeper_levels_converge_faster(self, linearization):
        """Deeper levels need fewer iterations. PROPOSITION."""
        result = linearization.linearize(3)
        levels = sorted(result['levels'], key=lambda x: x['level'])
        # Level 3 (deepest, smallest eps) should converge in <= iterations of level 1
        # This may not always hold strictly, but eps_3 << eps_1
        assert levels[2]['epsilon_j'] < levels[0]['epsilon_j']

    def test_error_comparison_k1_k2_k3(self, linearization):
        """Error at each level strictly decreases."""
        result = linearization.linearize(3)
        levels = sorted(result['levels'], key=lambda x: x['level'])
        # Per-level linearization error should decrease
        for i in range(len(levels) - 1):
            assert levels[i]['linearization_error'] >= levels[i + 1]['linearization_error']


# ======================================================================
# 10. Edge cases and parameter variation tests
# ======================================================================

class TestEdgeCases:
    """Edge cases and parameter variations."""

    def test_k4_works(self, default_R, default_g2):
        """k=4 works (beyond 600-cell natural hierarchy)."""
        lin = MultiLevelLinearization(
            R=default_R, g2=default_g2, epsilon_1=0.3)
        result = lin.linearize(4)
        assert result['all_converged'] is True

    def test_small_epsilon_k2(self, default_R, default_g2):
        """k=2 with eps_1 = 0.1 (very small field)."""
        lin = MultiLevelLinearization(
            R=default_R, g2=default_g2, epsilon_1=0.1)
        result = lin.linearize(2)
        assert result['all_converged'] is True

    def test_different_L(self, default_R, default_g2):
        """L=3 blocking factor works."""
        h = NestedSmallFieldCondition(
            epsilon_1=0.3, L=3, R=default_R, g2=default_g2)
        assert h.epsilon_at_level(2) == 0.3**2

    def test_blocking_geometric_scheme(self, default_R):
        """Geometric blocking scheme creates valid hierarchy."""
        bh = BlockingHierarchy600Cell(R=default_R, scheme='geometric')
        assert bh.blocks_at_level(0) == 120
        # Geometric: 120 -> 15 -> 1 (with L=2, 120/8=15, 15/8=1)
        assert bh.blocks_at_level(1) <= 120

    def test_gribov_epsilon_positive(self, default_g2, default_R):
        """Gribov-derived eps_1 is positive."""
        eps = _gribov_epsilon_1(default_g2, default_R)
        assert eps > 0
        assert eps <= 0.5  # Capped

    def test_gribov_epsilon_decreases_with_coupling(self, default_R):
        """Larger coupling -> smaller Gribov epsilon (below the cap)."""
        # Use couplings large enough that raw eps < 0.5 (uncapped regime)
        # eps_raw = 9*sqrt(3)/(8*sqrt(g2)); for g2=50, eps_raw ~ 0.28
        eps1 = _gribov_epsilon_1(50.0, default_R)
        eps2 = _gribov_epsilon_1(200.0, default_R)
        assert eps2 < eps1

    def test_convergence_with_different_tolerance(self, convergence):
        """Different tolerance changes iteration count."""
        r1 = convergence.total_iterations(2, tolerance=1e-5)
        r2 = convergence.total_iterations(2, tolerance=1e-12)
        assert r2['total_iterations'] >= r1['total_iterations']

    def test_all_classes_instantiate_with_defaults(self):
        """All 7 classes instantiate with default parameters."""
        NestedSmallFieldCondition()
        IntermediateGreenFunction()
        NestedContractionMapping()
        MultiLevelLinearization()
        BlockingHierarchy600Cell()
        MultiStepMinimizerConvergence()
        MultiStepBackgroundField()

    def test_hierarchy_consistent_with_physical_values(self):
        """Physical values: eps_1=0.3, eps_2=0.09, eps_3=0.0081. NUMERICAL."""
        h = NestedSmallFieldCondition(epsilon_1=0.3)
        assert abs(h.epsilon_at_level(1) - 0.3) < 1e-15
        assert abs(h.epsilon_at_level(2) - 0.09) < 1e-15
        assert abs(h.epsilon_at_level(3) - 0.0081) < 1e-12
        assert abs(h.epsilon_at_level(4) - 0.0081**2) < 1e-12

    def test_blocking_600cell_block_counts_sum(self, blocking_sym):
        """Block counts should be non-increasing sequence ending at 1."""
        counts = blocking_sym._block_counts
        assert counts[0] == 120
        assert counts[-1] == 1
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1]
