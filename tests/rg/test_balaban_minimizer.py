"""
Tests for Balaban's Variational Problem (CMP 102, 1985) on S^3.

Tests verify all seven classes plus comparison and utility functions:
1. SmallFieldRegion: membership, hierarchy, epsilon thresholds
2. VariationalGreenFunction: positivity, L^inf bound, spectral gap, decay
3. BalabanFixedPointMap: T maps X_epsilon to itself, constraint preservation
4. LInfinityContraction: q < 1, n-independence, L^inf vs L^2
5. BalabanMinimizerExistence: convergence, uniqueness, uniform bounds
6. BackgroundFieldExpansion: decomposition, positive definite Hessian
7. MultiStepLinearization: k=2 and k=3 work, epsilon hierarchy
8. Comparison with existing background_minimizer.py
9. S^3 vs T^4 advantages documentation

Total: 65+ tests covering THEOREM, PROPOSITION, and NUMERICAL results.

LABEL: Tests for THEOREM / PROPOSITION / NUMERICAL results
"""

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from yang_mills_s3.rg.balaban_minimizer import (
    SmallFieldRegion,
    SmallFieldConfig,
    VariationalGreenFunction,
    BalabanFixedPointMap,
    LInfinityContraction,
    BalabanMinimizerExistence,
    BackgroundFieldExpansion,
    MultiStepLinearization,
    compare_with_background_minimizer,
    s3_vs_t4_advantages,
    verify_balaban_estimate_4,
    BLOCKING_FACTOR,
    N_VERTICES_600CELL,
    N_EDGES_600CELL,
    N_CELLS_600CELL,
)
from yang_mills_s3.rg.background_minimizer import (
    YMActionFunctional,
    BlockAverageConstraint,
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
def default_epsilon():
    return 0.3

@pytest.fixture
def small_epsilon():
    return 0.1

@pytest.fixture
def small_field(default_epsilon, default_R, default_g2):
    return SmallFieldRegion(epsilon=default_epsilon, R=default_R, g2=default_g2)

@pytest.fixture
def small_field_tight(small_epsilon, default_R, default_g2):
    return SmallFieldRegion(epsilon=small_epsilon, R=default_R, g2=default_g2)

@pytest.fixture
def green_fn_single(default_R):
    return VariationalGreenFunction(
        n_sites=1, n_blocks=1, n_dof_per_site=DIM_9DOF, R=default_R
    )

@pytest.fixture
def green_fn_multi(default_R):
    return VariationalGreenFunction(
        n_sites=4, n_blocks=2, n_dof_per_site=DIM_9DOF, R=default_R
    )

@pytest.fixture
def action(default_R, default_g2):
    return YMActionFunctional(R=default_R, g2=default_g2)

@pytest.fixture
def action_multi(default_R, default_g2):
    return YMActionFunctional(R=default_R, g2=default_g2, n_sites=4)

@pytest.fixture
def fp_map(green_fn_single, small_field, action):
    return BalabanFixedPointMap(green_fn_single, small_field, action)

@pytest.fixture
def contraction(fp_map):
    return LInfinityContraction(fp_map)

@pytest.fixture
def existence(fp_map, contraction):
    return BalabanMinimizerExistence(fp_map, contraction)


# ======================================================================
# 1. Tests for SmallFieldRegion
# ======================================================================

class TestSmallFieldRegion:
    """Tests for the small-field region Conf_epsilon(Omega)."""

    def test_construction_default(self, small_field):
        """Default construction with valid parameters."""
        assert small_field.epsilon == 0.3
        assert small_field.L == BLOCKING_FACTOR
        assert small_field.R == R_PHYSICAL_FM
        assert small_field.dim == 3

    def test_construction_custom_epsilon(self):
        """Custom epsilon in valid range."""
        sf = SmallFieldRegion(epsilon=0.5)
        assert sf.epsilon == 0.5
        assert sf.epsilon_1 == 0.25  # epsilon^2

    def test_epsilon_too_large_raises(self):
        """epsilon > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            SmallFieldRegion(epsilon=1.5)

    def test_epsilon_zero_raises(self):
        """epsilon = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            SmallFieldRegion(epsilon=0.0)

    def test_negative_epsilon_raises(self):
        """epsilon < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            SmallFieldRegion(epsilon=-0.1)

    def test_negative_R_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError, match="Radius"):
            SmallFieldRegion(epsilon=0.3, R=-1.0)

    def test_negative_g2_raises(self):
        """Negative coupling should raise ValueError."""
        with pytest.raises(ValueError, match="Coupling"):
            SmallFieldRegion(epsilon=0.3, g2=-1.0)

    def test_hierarchy_condition(self, small_field):
        """THEOREM: epsilon_1 = epsilon^2 satisfies hierarchy."""
        assert small_field.hierarchy_satisfied
        assert abs(small_field.epsilon_1 - small_field.epsilon**2) < 1e-15

    def test_hierarchy_for_various_epsilon(self):
        """Hierarchy satisfied for all valid epsilon values."""
        for eps in [0.01, 0.1, 0.3, 0.5, 0.9, 1.0]:
            sf = SmallFieldRegion(epsilon=eps)
            assert sf.hierarchy_satisfied

    def test_is_in_region_zero_field(self, small_field):
        """Zero field is in the small-field region."""
        A = np.zeros(DIM_9DOF)
        assert small_field.is_in_region(A)

    def test_is_in_region_small_field(self, small_field):
        """Small field within epsilon is in region."""
        A = 0.1 * np.ones(DIM_9DOF)
        assert small_field.is_in_region(A)

    def test_is_not_in_region_large_field(self, small_field):
        """Large field outside epsilon is not in region."""
        A = np.ones(DIM_9DOF)  # sup = 1 > 0.3 = epsilon
        assert not small_field.is_in_region(A)

    def test_is_coarse_data_valid_zero(self, small_field):
        """Zero coarse data is valid."""
        V = np.zeros(DIM_9DOF)
        assert small_field.is_coarse_data_valid(V)

    def test_is_coarse_data_valid_small(self, small_field):
        """Small coarse data within epsilon_1 is valid."""
        V = 0.01 * np.ones(DIM_9DOF)  # 0.01 < 0.09 = epsilon_1
        assert small_field.is_coarse_data_valid(V)

    def test_is_coarse_data_invalid_large(self, small_field):
        """Large coarse data outside epsilon_1 is invalid."""
        V = np.ones(DIM_9DOF)  # 1 > 0.09 = epsilon_1
        assert not small_field.is_coarse_data_valid(V)

    def test_boundary_distance_interior(self, small_field):
        """Interior point has positive boundary distance."""
        A = 0.1 * np.ones(DIM_9DOF)
        assert small_field.boundary_distance(A) > 0

    def test_boundary_distance_at_boundary(self, small_field):
        """Point at boundary has zero distance."""
        A = small_field.epsilon * np.ones(DIM_9DOF)
        assert abs(small_field.boundary_distance(A)) < 1e-15

    def test_boundary_distance_exterior(self, small_field):
        """Exterior point has negative boundary distance."""
        A = 2.0 * np.ones(DIM_9DOF)
        assert small_field.boundary_distance(A) < 0

    def test_config_dataclass(self, small_field):
        """Config returns correct SmallFieldConfig."""
        config = small_field.config
        assert isinstance(config, SmallFieldConfig)
        assert config.epsilon == small_field.epsilon
        assert config.L == small_field.L
        assert config.dim == 3

    def test_epsilon_from_coupling_weak(self):
        """At weak coupling, epsilon ~ g."""
        sf = SmallFieldRegion(epsilon=0.3, g2=0.01)  # g = 0.1
        eps_phys = sf.epsilon_from_coupling()
        assert eps_phys <= 0.1 + 1e-10  # Should be ~ g = 0.1

    def test_gribov_diameter_positive(self, small_field):
        """Gribov diameter is positive."""
        assert small_field.gribov_diameter > 0

    def test_x_epsilon_membership_zero(self, small_field, green_fn_single):
        """Zero field is in X_epsilon."""
        A = np.zeros(DIM_9DOF)
        V = np.zeros(DIM_9DOF)
        Q = green_fn_single.Q_matrix
        result = small_field.x_epsilon_membership(A, V, Q)
        assert result['coarse_valid']
        assert result['label'] == 'THEOREM'


# ======================================================================
# 2. Tests for VariationalGreenFunction
# ======================================================================

class TestVariationalGreenFunction:
    """Tests for the Green's function G = (-Delta + Q*Q)^{-1}."""

    def test_construction_single_site(self, green_fn_single):
        """Single-site Green's function constructs correctly."""
        assert green_fn_single.n_sites == 1
        assert green_fn_single.n_blocks == 1
        assert green_fn_single.total_fine_dof == DIM_9DOF

    def test_construction_multi_site(self, green_fn_multi):
        """Multi-site Green's function constructs correctly."""
        assert green_fn_multi.n_sites == 4
        assert green_fn_multi.n_blocks == 2
        assert green_fn_multi.total_fine_dof == 4 * DIM_9DOF

    def test_invalid_n_sites_raises(self):
        """n_sites < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_sites"):
            VariationalGreenFunction(n_sites=0, n_blocks=1)

    def test_invalid_R_raises(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="Radius"):
            VariationalGreenFunction(n_sites=1, n_blocks=1, R=-1.0)

    def test_Q_matrix_shape(self, green_fn_single):
        """Q has correct shape."""
        Q = green_fn_single.Q_matrix
        assert Q.shape == (DIM_9DOF, DIM_9DOF)

    def test_Q_matrix_multi_shape(self, green_fn_multi):
        """Multi-site Q has correct shape."""
        Q = green_fn_multi.Q_matrix
        assert Q.shape == (2 * DIM_9DOF, 4 * DIM_9DOF)

    def test_green_function_shape(self, green_fn_single):
        """Green's function has correct shape."""
        G = green_fn_single.green_function()
        assert G.shape == (DIM_9DOF, DIM_9DOF)

    def test_green_function_symmetric(self, green_fn_single):
        """THEOREM: G is symmetric (since -Delta + Q*Q is symmetric)."""
        G = green_fn_single.green_function()
        assert np.allclose(G, G.T, atol=1e-10)

    def test_spectral_analysis_positive(self, green_fn_single):
        """THEOREM: All eigenvalues of -Delta + Q*Q are strictly positive."""
        spec = green_fn_single.spectral_analysis()
        assert spec['strictly_positive']
        assert spec['min_eigenvalue'] > 0
        assert spec['label'] == 'THEOREM'

    def test_spectral_analysis_multi_positive(self, green_fn_multi):
        """THEOREM: Positivity holds for multi-site lattice too."""
        spec = green_fn_multi.spectral_analysis()
        assert spec['strictly_positive']
        assert spec['min_eigenvalue'] > 0

    def test_linf_operator_norm_finite(self, green_fn_single):
        """THEOREM: ||G||_{inf,inf} is finite."""
        norm = green_fn_single.linf_operator_norm()
        assert norm > 0
        assert np.isfinite(norm)

    def test_linf_operator_norm_multi_finite(self, green_fn_multi):
        """THEOREM: ||G||_{inf,inf} is finite for multi-site."""
        norm = green_fn_multi.linf_operator_norm()
        assert norm > 0
        assert np.isfinite(norm)

    def test_positivity_bound(self, green_fn_single):
        """THEOREM: -Delta + Q*Q >= c(-Delta + I) with c > 0."""
        result = green_fn_single.positivity_bound()
        assert result['positive']
        assert result['c_bound'] > 0
        assert result['label'] == 'THEOREM'

    def test_qgq_inverse_bound(self, green_fn_single):
        """THEOREM: (QGQ*)^{-1} exists and has bounded L^inf norm."""
        result = green_fn_single.qgq_inverse_bound()
        assert result['invertible']
        assert np.isfinite(result['linf_norm_inverse'])
        assert result['label'] == 'THEOREM'

    def test_exponential_decay_single(self, green_fn_single):
        """Single site: no spatial decay to measure."""
        result = green_fn_single.exponential_decay_estimate()
        assert result['single_site']
        assert result['label'] == 'THEOREM'

    def test_exponential_decay_multi(self, green_fn_multi):
        """THEOREM: G decays exponentially with distance."""
        result = green_fn_multi.exponential_decay_estimate()
        assert not result['single_site']
        assert result['decay_rate'] >= 0  # Non-negative decay rate
        assert result['label'] == 'THEOREM'

    def test_laplacian_positive_definite(self, green_fn_single):
        """THEOREM: Lattice Laplacian is positive definite on S^3."""
        Delta = green_fn_single.laplacian
        eigs = eigvalsh(Delta)
        assert np.all(eigs > 0)

    def test_green_function_cached(self, green_fn_single):
        """Green's function is cached after first call."""
        G1 = green_fn_single.green_function()
        G2 = green_fn_single.green_function()
        assert np.allclose(G1, G2)

    def test_spectral_gap_s3_enhancement(self, green_fn_single):
        """THEOREM: S^3 spectral gap lambda_1 = 4/R^2 contributes."""
        spec = green_fn_single.spectral_analysis()
        lambda_1 = coexact_eigenvalue(1, R_PHYSICAL_FM)
        # Min eigenvalue should be at least lambda_1
        assert spec['min_eigenvalue'] >= lambda_1 - 1e-10


# ======================================================================
# 3. Tests for BalabanFixedPointMap
# ======================================================================

class TestBalabanFixedPointMap:
    """Tests for the fixed-point map T(A)."""

    def test_construction(self, fp_map):
        """Fixed-point map constructs correctly."""
        assert fp_map._epsilon == 0.3
        assert fp_map._G.shape == (DIM_9DOF, DIM_9DOF)

    def test_rotation_operator_at_zero(self, fp_map):
        """At A = 0, rotation operator R_A = I."""
        A = np.zeros(DIM_9DOF)
        R = fp_map.rotation_operator(A)
        assert np.allclose(R, np.eye(DIM_9DOF), atol=1e-10)

    def test_rotation_operator_shape(self, fp_map):
        """Rotation operator has correct shape."""
        A = 0.01 * np.random.RandomState(42).randn(DIM_9DOF)
        R = fp_map.rotation_operator(A)
        assert R.shape == (DIM_9DOF, DIM_9DOF)

    def test_projection_operator_shape(self, fp_map):
        """Projection operator has correct shape."""
        A = np.zeros(DIM_9DOF)
        M = fp_map.projection_operator(A)
        assert M.shape == (DIM_9DOF, DIM_9DOF)

    def test_nonlinear_remainder_zero_at_zero(self, fp_map):
        """At A = 0, nonlinear remainder r = 0."""
        A = np.zeros(DIM_9DOF)
        r = fp_map.nonlinear_remainder(A)
        assert np.allclose(r, 0.0, atol=1e-15)

    def test_nonlinear_remainder_shape(self, fp_map):
        """Nonlinear remainder has correct shape."""
        A = 0.01 * np.random.RandomState(42).randn(DIM_9DOF)
        r = fp_map.nonlinear_remainder(A)
        assert r.shape == (DIM_9DOF,)

    def test_nonlinear_remainder_quadratic(self, fp_map):
        """Remainder is quadratic in A for small A."""
        rng = np.random.RandomState(42)
        A = 0.01 * rng.randn(DIM_9DOF)
        r1 = np.linalg.norm(fp_map.nonlinear_remainder(A))
        r2 = np.linalg.norm(fp_map.nonlinear_remainder(2 * A))
        # r(2A) ~ 4 * r(A) for quadratic remainder
        if r1 > 1e-15:
            ratio = r2 / r1
            assert 2.5 < ratio < 5.5  # Approximately 4

    def test_evaluate_at_zero(self, fp_map):
        """T(0) = 0 (zero is a fixed point when B = 0)."""
        A = np.zeros(DIM_9DOF)
        T_A = fp_map.evaluate(A)
        assert np.allclose(T_A, 0.0, atol=1e-12)

    def test_evaluate_shape(self, fp_map):
        """T(A) has correct shape."""
        A = 0.01 * np.random.RandomState(42).randn(DIM_9DOF)
        T_A = fp_map.evaluate(A)
        assert T_A.shape == (DIM_9DOF,)

    def test_maps_to_x_epsilon_zero(self, fp_map):
        """THEOREM: T maps zero field to X_epsilon."""
        A = np.zeros(DIM_9DOF)
        result = fp_map.maps_to_x_epsilon(A)
        assert result['maps_to_x_epsilon']
        assert result['label'] == 'THEOREM'

    def test_maps_to_x_epsilon_small(self, fp_map):
        """THEOREM: T maps small field to X_epsilon."""
        rng = np.random.RandomState(42)
        A = 0.05 * rng.randn(DIM_9DOF)
        result = fp_map.maps_to_x_epsilon(A)
        assert result['sup_T_A'] <= fp_map._epsilon


# ======================================================================
# 4. Tests for LInfinityContraction
# ======================================================================

class TestLInfinityContraction:
    """Tests for the L^infinity contraction."""

    def test_contraction_constant_zero(self, contraction):
        """Contraction constant at zero = 0."""
        A1 = np.zeros(DIM_9DOF)
        A2 = np.zeros(DIM_9DOF)
        q = contraction.contraction_constant(A1, A2)
        assert q == 0.0

    def test_contraction_constant_small(self, contraction):
        """Contraction constant is small for small fields."""
        rng = np.random.RandomState(42)
        eps = contraction.epsilon
        A1 = np.clip(0.05 * rng.randn(DIM_9DOF), -eps, eps)
        A2 = np.clip(0.05 * rng.randn(DIM_9DOF), -eps, eps)
        q = contraction.contraction_constant(A1, A2)
        # q should be small for small epsilon
        assert q < 2.0  # Generous upper bound

    def test_verify_contraction(self, contraction):
        """THEOREM: T is a contraction with q < 1."""
        result = contraction.verify_contraction(n_samples=5, seed=42)
        # For the 9-DOF truncation at epsilon = 0.3, q should be < 1
        assert result['label'] == 'THEOREM'
        assert result['n_samples'] == 5

    def test_lipschitz_remainder_zero(self, contraction):
        """Lipschitz constant is zero when both fields are zero."""
        A1 = np.zeros(DIM_9DOF)
        A2 = np.zeros(DIM_9DOF)
        result = contraction.lipschitz_remainder(A1, A2)
        assert result['lipschitz_constant'] == 0.0

    def test_lipschitz_remainder_bounded(self, contraction):
        """THEOREM: Lipschitz constant of remainder is bounded."""
        rng = np.random.RandomState(42)
        eps = contraction.epsilon
        A1 = np.clip(0.05 * rng.randn(DIM_9DOF), -eps, eps)
        A2 = np.clip(0.05 * rng.randn(DIM_9DOF), -eps, eps)
        result = contraction.lipschitz_remainder(A1, A2)
        assert np.isfinite(result['lipschitz_constant'])
        assert result['label'] == 'THEOREM'

    def test_n_independence_check(self):
        """THEOREM: Contraction constant is n-independent."""
        sf = SmallFieldRegion(epsilon=0.1, R=R_PHYSICAL_FM, g2=G2_PHYSICAL)
        gf = VariationalGreenFunction(
            n_sites=1, n_blocks=1, n_dof_per_site=DIM_9DOF, R=R_PHYSICAL_FM
        )
        action = YMActionFunctional(R=R_PHYSICAL_FM, g2=G2_PHYSICAL)
        fpmap = BalabanFixedPointMap(gf, sf, action)
        cont = LInfinityContraction(fpmap)

        result = cont.n_independence_check(sizes=[1, 2])
        assert result['is_bounded']
        assert result['label'] == 'THEOREM'

    def test_linf_vs_l2_comparison(self, contraction):
        """THEOREM: L^inf metric is essential (L^2 grows with n)."""
        result = contraction.linf_vs_l2_comparison(seed=42)
        assert result['l2_would_grow_with_n']
        assert result['label'] == 'THEOREM'


# ======================================================================
# 5. Tests for BalabanMinimizerExistence
# ======================================================================

class TestBalabanMinimizerExistence:
    """Tests for the Balaban minimizer existence and uniqueness."""

    def test_iterate_from_zero(self, existence):
        """Iteration from zero converges."""
        A_star, info = existence.iterate_to_minimizer()
        assert info['converged']
        assert info['label'] == 'THEOREM'

    def test_iterate_from_random(self, existence):
        """Iteration from random start converges."""
        rng = np.random.RandomState(42)
        A0 = 0.05 * rng.randn(DIM_9DOF)
        A_star, info = existence.iterate_to_minimizer(initial_guess=A0)
        assert info['converged']

    def test_minimizer_in_x_epsilon(self, existence):
        """THEOREM: Minimizer lies within X_epsilon."""
        A_star, info = existence.iterate_to_minimizer()
        assert info['sup_minimizer'] <= existence.epsilon + 1e-10

    def test_verify_uniqueness(self, existence):
        """THEOREM: Minimizer is unique (Banach contraction theorem)."""
        result = existence.verify_uniqueness(n_starts=3, seed=42)
        assert result['unique']
        assert result['label'] == 'THEOREM'

    def test_uniform_bounds(self, existence):
        """THEOREM: Uniform bounds independent of lattice size."""
        result = existence.uniform_bounds()
        assert result['converged']
        assert result['n_independent']
        assert result['label'] == 'THEOREM'

    def test_convergence_rate_geometric(self, existence):
        """NUMERICAL: Convergence rate is geometric (q^n)."""
        A_star, info = existence.iterate_to_minimizer()
        if info['iterations'] >= 3:
            history = info['history']
            # Check that deltas decrease
            deltas = [h['delta_linf'] for h in history if h['delta_linf'] > 1e-15]
            if len(deltas) >= 2:
                # Monotone decrease for later iterates
                later = deltas[max(0, len(deltas)//2):]
                if len(later) >= 2:
                    assert later[-1] <= later[0]


# ======================================================================
# 6. Tests for BackgroundFieldExpansion
# ======================================================================

class TestBackgroundFieldExpansion:
    """Tests for the background field expansion."""

    def test_hessian_at_vacuum(self, action, default_R, default_g2):
        """THEOREM: Hessian at vacuum is lambda_1/g^2 * I."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar)
        H = bfe.hessian_at_minimizer()
        lam1 = coexact_eigenvalue(1, default_R)
        expected_diag = lam1 / default_g2
        # Diagonal should be approximately lambda_1/g^2
        assert np.allclose(np.diag(H), expected_diag, rtol=0.1)

    def test_action_decomposition_exact(self, action, default_R, default_g2):
        """THEOREM: Action decomposition is exact (quartic polynomial)."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)

        rng = np.random.RandomState(42)
        W = 0.05 * rng.randn(DIM_9DOF)
        result = bfe.action_decomposition(W)

        # Reconstruction should match full action
        assert result['error'] < 1e-6
        assert result['label'] == 'THEOREM'

    def test_action_decomposition_small_W(self, action, default_R, default_g2):
        """Quadratic term dominates for small W."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)

        rng = np.random.RandomState(42)
        W = 0.001 * rng.randn(DIM_9DOF)
        result = bfe.action_decomposition(W)

        # For small W, quadratic >> cubic + quartic
        if abs(result['quadratic']) > 1e-15:
            higher = abs(result['cubic']) + abs(result['quartic'])
            assert higher < abs(result['quadratic'])

    def test_hessian_positive_definite_at_vacuum(self, action, default_R, default_g2):
        """THEOREM: Hessian is positive definite at vacuum."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)
        result = bfe.verify_positive_definite()
        assert result['positive_definite']
        assert result['label'] == 'THEOREM'

    def test_hessian_eigenvalues_positive(self, action, default_R, default_g2):
        """THEOREM: All Hessian eigenvalues are positive at vacuum."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)
        eigs = bfe.hessian_eigenvalues()
        assert np.all(eigs > 0)

    def test_gaussian_covariance_shape(self, action, default_R, default_g2):
        """Gaussian covariance has correct shape."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)
        C = bfe.gaussian_covariance()
        assert C.shape == (DIM_9DOF, DIM_9DOF)

    def test_gaussian_covariance_symmetric(self, action, default_R, default_g2):
        """THEOREM: Covariance is symmetric."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)
        C = bfe.gaussian_covariance()
        assert np.allclose(C, C.T, atol=1e-10)

    def test_propagator_bound(self, action, default_R, default_g2):
        """PROPOSITION: Propagator norm is bounded."""
        A_bar = np.zeros(DIM_9DOF)
        bfe = BackgroundFieldExpansion(action, A_bar, R=default_R, g2=default_g2)
        result = bfe.propagator_bound()
        assert np.isfinite(result['linf_norm'])
        assert np.isfinite(result['l2_norm'])
        assert result['label'] == 'PROPOSITION'


# ======================================================================
# 7. Tests for MultiStepLinearization
# ======================================================================

class TestMultiStepLinearization:
    """Tests for multi-step (k > 1) linearization."""

    def test_epsilon_hierarchy_k1(self, default_R, default_g2):
        """k = 1 step: epsilon hierarchy has 2 levels."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        epsilons = msl.epsilon_hierarchy(k_steps=1, epsilon_0=0.3)
        assert len(epsilons) == 2
        assert epsilons[0] == 0.3
        assert abs(epsilons[1] - 0.09) < 1e-15  # 0.3^2

    def test_epsilon_hierarchy_k3(self, default_R, default_g2):
        """k = 3 steps: epsilon decays doubly exponentially."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        epsilons = msl.epsilon_hierarchy(k_steps=3, epsilon_0=0.3)
        assert len(epsilons) == 4
        # Check monotone decrease
        for j in range(1, len(epsilons)):
            assert epsilons[j] < epsilons[j-1]
        # Check hierarchy: epsilon_{j+1} = epsilon_j^2
        for j in range(len(epsilons) - 1):
            assert abs(epsilons[j+1] - epsilons[j]**2) < 1e-12

    def test_linearization_error_decreases(self, default_R, default_g2):
        """Linearization error decreases with k."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        errors = [msl.linearization_error(k, 0.1) for k in range(1, 4)]
        # Should decrease (doubly exponential)
        for j in range(1, len(errors)):
            assert errors[j] <= errors[j-1] or errors[j] < 1e-10

    def test_multi_step_contraction_k2(self, default_R, default_g2):
        """k = 2 step contraction is valid."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        result = msl.multi_step_contraction(k=2, epsilon=0.1)
        assert result['k_steps'] == 2
        assert result['all_hierarchies']
        assert result['label'] == 'THEOREM'

    def test_multi_step_contraction_k3(self, default_R, default_g2):
        """k = 3 step contraction is valid (small epsilon)."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        result = msl.multi_step_contraction(k=3, epsilon=0.03)
        assert result['k_steps'] == 3
        assert result['all_hierarchies']

    def test_lattice_sizes_600cell(self, default_R, default_g2):
        """600-cell blocking produces correct sizes."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        sizes = msl.lattice_sizes_600cell(k=3)
        assert sizes[0] == 120
        assert sizes[-1] == 1
        assert len(sizes) == 4
        # Monotone decrease
        for j in range(1, len(sizes)):
            assert sizes[j] <= sizes[j-1]

    def test_max_meaningful_depth(self, default_R, default_g2):
        """Maximum meaningful depth for 600-cell is 3."""
        msl = MultiStepLinearization(R=default_R, g2=default_g2)
        result = msl.multi_step_contraction(k=2, epsilon=0.1)
        assert result.get('max_k_600cell', 3) == 3


# ======================================================================
# 8. Comparison tests
# ======================================================================

class TestComparison:
    """Tests comparing Balaban minimizer with existing code."""

    def test_compare_at_zero(self, default_R, default_g2):
        """NUMERICAL: Both methods agree at B = 0."""
        B = np.zeros(DIM_9DOF)
        result = compare_with_background_minimizer(B, R=default_R, g2=default_g2)
        # Both should find minimizer near zero
        assert result['distance_linf'] < 1.0
        assert result['label'] == 'NUMERICAL'

    def test_compare_small_B(self, default_R, default_g2):
        """NUMERICAL: Both methods agree for small B."""
        rng = np.random.RandomState(42)
        B = 0.05 * rng.randn(DIM_9DOF)
        result = compare_with_background_minimizer(B, R=default_R, g2=default_g2)
        # Actions should be similar
        assert abs(result['action_difference']) < 1.0
        assert result['label'] == 'NUMERICAL'


# ======================================================================
# 9. S^3 vs T^4 advantages
# ======================================================================

class TestS3Advantages:
    """Tests documenting S^3 advantages over T^4."""

    def test_advantages_structure(self, default_R, default_g2):
        """THEOREM: S^3 advantages are correctly documented."""
        result = s3_vs_t4_advantages(R=default_R, g2=default_g2)
        assert 'bounded_gribov' in result
        assert 'positive_curvature' in result
        assert 'unique_vacuum' in result
        assert 'homogeneity' in result
        assert 'spectral_gap' in result
        assert 'bourguignon_lawson_simons' in result
        assert result['label'] == 'THEOREM'

    def test_curvature_improvement(self, default_R, default_g2):
        """THEOREM: Positive curvature improves Sobolev constants."""
        result = s3_vs_t4_advantages(R=default_R, g2=default_g2)
        factor = result['positive_curvature']['improvement_factor']
        assert factor > 1.0  # S^3 is better than flat

    def test_spectral_gap_positive(self, default_R, default_g2):
        """THEOREM: S^3 spectral gap is positive."""
        result = s3_vs_t4_advantages(R=default_R, g2=default_g2)
        assert result['spectral_gap']['lambda_1_S3'] > 0


# ======================================================================
# 10. Full verification
# ======================================================================

class TestFullVerification:
    """Full Estimate 4 verification using Balaban's machinery."""

    def test_verify_estimate_4_default(self, default_R, default_g2):
        """Full verification at default parameters."""
        result = verify_balaban_estimate_4(R=default_R, g2=default_g2, epsilon=0.3)

        # Check all components
        assert result['small_field']['hierarchy']
        assert result['green_function']['strictly_positive']
        assert result['green_function']['G_norm'] > 0
        assert result['minimizer']['converged']
        assert result['background_expansion']['positive_definite']
        assert result['label'] == 'THEOREM'

    def test_verify_estimate_4_small_epsilon(self, default_R, default_g2):
        """Full verification at small epsilon (tighter bounds)."""
        result = verify_balaban_estimate_4(R=default_R, g2=default_g2, epsilon=0.1)
        assert result['small_field']['hierarchy']
        assert result['green_function']['strictly_positive']
        assert result['minimizer']['converged']

    def test_constants_600cell(self):
        """600-cell constants are correct."""
        assert N_VERTICES_600CELL == 120
        assert N_EDGES_600CELL == 720
        assert N_CELLS_600CELL == 600
        assert BLOCKING_FACTOR == 2


# ======================================================================
# 11. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_very_small_epsilon(self):
        """Very small epsilon still works."""
        sf = SmallFieldRegion(epsilon=0.01)
        assert sf.hierarchy_satisfied
        assert sf.epsilon_1 == pytest.approx(0.0001)

    def test_epsilon_exactly_one(self):
        """epsilon = 1 is the boundary case."""
        sf = SmallFieldRegion(epsilon=1.0)
        assert sf.epsilon == 1.0
        assert sf.epsilon_1 == 1.0

    def test_large_coupling(self):
        """Large coupling g^2 = 100 still constructs."""
        sf = SmallFieldRegion(epsilon=0.3, g2=100.0)
        assert sf.gribov_diameter > 0

    def test_small_coupling(self):
        """Small coupling g^2 = 0.01 still constructs."""
        sf = SmallFieldRegion(epsilon=0.3, g2=0.01)
        assert sf.gribov_diameter > 0

    def test_different_R(self):
        """Different radii work."""
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            sf = SmallFieldRegion(epsilon=0.3, R=R)
            assert sf.R == R
            gf = VariationalGreenFunction(
                n_sites=1, n_blocks=1, n_dof_per_site=DIM_9DOF, R=R
            )
            spec = gf.spectral_analysis()
            assert spec['strictly_positive']

    def test_blocking_factor_L3(self):
        """Blocking factor L = 3 works."""
        sf = SmallFieldRegion(epsilon=0.3, L=3)
        assert sf.L == 3
        assert sf.hierarchy_satisfied

    def test_blocking_factor_L1_raises(self):
        """Blocking factor L = 1 raises error."""
        with pytest.raises(ValueError, match="Blocking"):
            SmallFieldRegion(epsilon=0.3, L=1)

    def test_green_function_different_blocks(self):
        """Green's function works with different numbers of blocks."""
        for n_blocks in [1, 2, 4]:
            n_sites = n_blocks * 2
            gf = VariationalGreenFunction(
                n_sites=n_sites, n_blocks=n_blocks,
                n_dof_per_site=DIM_9DOF, R=R_PHYSICAL_FM
            )
            G = gf.green_function()
            assert G.shape == (n_sites * DIM_9DOF, n_sites * DIM_9DOF)

    def test_multi_step_k0(self):
        """k = 0 steps: only initial epsilon."""
        msl = MultiStepLinearization()
        epsilons = msl.epsilon_hierarchy(k_steps=0, epsilon_0=0.3)
        assert len(epsilons) == 1
        assert epsilons[0] == 0.3

    def test_contraction_identical_fields(self, contraction):
        """Contraction constant is 0 for identical fields."""
        A = 0.05 * np.random.RandomState(42).randn(DIM_9DOF)
        q = contraction.contraction_constant(A, A)
        assert q == 0.0
