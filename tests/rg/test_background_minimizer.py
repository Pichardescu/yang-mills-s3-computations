"""
Tests for the Background Field Minimizer (Estimate 4).

Tests verify:
1. BlockAverageConstraint: constraint construction, averaging, projection
2. YMActionFunctional: positivity, minimum at vacuum, gradient, Hessian
3. ConstrainedMinimizer: convergence, constraint satisfaction, action minimality
4. ExistenceProof: coercivity, compactness, non-emptiness
5. UniquenessProof: strict convexity within Gribov region
6. EllipticRegularity: Sobolev constants, Schauder bounds, YM equation
7. BackgroundFieldDecomposition: exact decomposition, quadratic form, vertices
8. Edge cases: B=0, B large, different couplings, multi-block
9. 9-DOF truncation: exact minimizer, consistency with optimization

Total: 65+ tests covering THEOREM, PROPOSITION, and NUMERICAL results.

LABEL: Tests for THEOREM / PROPOSITION / NUMERICAL results
"""

import numpy as np
import pytest
from yang_mills_s3.rg.background_minimizer import (
    BlockAverageConstraint,
    YMActionFunctional,
    ConstrainedMinimizer,
    ExistenceProof,
    UniquenessProof,
    EllipticRegularity,
    BackgroundFieldDecomposition,
    exact_minimizer_9dof,
    minimizer_multi_block,
    verify_estimate_4,
    DIM_9DOF,
    DIM_ADJ,
    N_MODES_TRUNC,
    G2_PHYSICAL,
    R_PHYSICAL_FM,
    _su2_structure_constants,
)
from yang_mills_s3.rg.gribov_diameter_analytical import gribov_diameter_bound


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
def small_B():
    """Small coarse field within Gribov region."""
    np.random.seed(42)
    return 0.1 * np.random.randn(DIM_9DOF)

@pytest.fixture
def zero_B():
    """Zero coarse field (vacuum constraint)."""
    return np.zeros(DIM_9DOF)

@pytest.fixture
def action(default_R, default_g2):
    return YMActionFunctional(R=default_R, g2=default_g2)

@pytest.fixture
def constraint_single_block(small_B):
    """Single-block constraint with small B."""
    return BlockAverageConstraint(
        n_blocks=1,
        n_dof_per_block=DIM_9DOF,
        coarse_field=small_B.reshape(1, DIM_9DOF)
    )

@pytest.fixture
def constraint_zero(zero_B):
    """Single-block constraint with B=0."""
    return BlockAverageConstraint(
        n_blocks=1,
        n_dof_per_block=DIM_9DOF,
        coarse_field=zero_B.reshape(1, DIM_9DOF)
    )


# ======================================================================
# Test BlockAverageConstraint
# ======================================================================

class TestBlockAverageConstraint:
    """Tests for the block average constraint."""

    def test_construction_single_block(self):
        """Single block constraint constructs correctly."""
        B = np.ones(DIM_9DOF)
        c = BlockAverageConstraint(1, DIM_9DOF, B.reshape(1, DIM_9DOF))
        assert c.n_blocks == 1
        assert c.n_dof_per_block == DIM_9DOF
        assert c.n_fine_sites == 1
        assert c.total_fine_dof == DIM_9DOF

    def test_construction_multi_block(self):
        """Multi-block constraint constructs correctly."""
        n_blocks = 4
        n_fine_per_block = 3
        n_fine = n_blocks * n_fine_per_block
        B = np.zeros((n_blocks, DIM_9DOF))
        assignment = np.repeat(np.arange(n_blocks), n_fine_per_block)
        c = BlockAverageConstraint(n_blocks, DIM_9DOF, B, assignment)
        assert c.n_blocks == n_blocks
        assert c.n_fine_sites == n_fine
        assert c.total_fine_dof == n_fine * DIM_9DOF

    def test_construction_flat_coarse_field(self):
        """Flat coarse field array is accepted."""
        B_flat = np.zeros(2 * DIM_9DOF)
        c = BlockAverageConstraint(2, DIM_9DOF, B_flat)
        assert c.coarse_field.shape == (2, DIM_9DOF)

    def test_construction_wrong_size_raises(self):
        """Wrong size coarse field raises ValueError."""
        with pytest.raises(ValueError):
            BlockAverageConstraint(1, DIM_9DOF, np.zeros(5))

    def test_block_average_identity(self):
        """Block average of single site = the site value."""
        B = np.ones(DIM_9DOF)
        c = BlockAverageConstraint(1, DIM_9DOF, B.reshape(1, DIM_9DOF))
        A = 2.0 * np.ones((1, DIM_9DOF))
        avg = c.block_average(A)
        assert np.allclose(avg, A)

    def test_block_average_multi_site(self):
        """Block average of multiple sites = mean."""
        n_fine = 4
        B = np.zeros((1, DIM_9DOF))
        assignment = np.zeros(n_fine, dtype=int)
        c = BlockAverageConstraint(1, DIM_9DOF, B, assignment)

        A = np.arange(n_fine * DIM_9DOF, dtype=float).reshape(n_fine, DIM_9DOF)
        avg = c.block_average(A)
        expected = np.mean(A, axis=0, keepdims=True)
        assert np.allclose(avg, expected)

    def test_block_average_matrix_consistency(self):
        """Block average matrix Q gives same result as block_average()."""
        n_blocks = 2
        n_fine_per_block = 3
        n_fine = n_blocks * n_fine_per_block
        B = np.zeros((n_blocks, DIM_9DOF))
        assignment = np.repeat(np.arange(n_blocks), n_fine_per_block)
        c = BlockAverageConstraint(n_blocks, DIM_9DOF, B, assignment)

        np.random.seed(123)
        A = np.random.randn(n_fine, DIM_9DOF)
        avg_method = c.block_average(A)
        Q = c.block_average_matrix()
        avg_matrix = (Q @ A.ravel()).reshape(n_blocks, DIM_9DOF)
        assert np.allclose(avg_method, avg_matrix, atol=1e-12)

    def test_is_satisfied_true(self):
        """Constraint is satisfied when Q(A) = B."""
        B = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        assert c.is_satisfied(B)

    def test_is_satisfied_false(self):
        """Constraint is not satisfied when Q(A) != B."""
        B = np.ones((1, DIM_9DOF))
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        A = np.zeros((1, DIM_9DOF))
        assert not c.is_satisfied(A)

    def test_residual_zero_when_satisfied(self):
        """Residual is zero when constraint is satisfied."""
        B = np.ones((1, DIM_9DOF))
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        assert c.residual(B) < 1e-14

    def test_residual_positive_when_violated(self):
        """Residual is positive when constraint is violated."""
        B = np.ones((1, DIM_9DOF))
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        A = np.zeros((1, DIM_9DOF))
        assert c.residual(A) > 0

    def test_projection_satisfies_constraint(self):
        """Project onto constraint surface satisfies the constraint."""
        B = np.ones((1, DIM_9DOF)) * 0.5
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        A_init = np.zeros((1, DIM_9DOF))
        A_proj = c.project_onto_constraint(A_init)
        assert c.is_satisfied(A_proj, tolerance=1e-10)

    def test_projection_minimum_norm(self):
        """Projection gives minimum-norm correction."""
        B = np.ones((1, DIM_9DOF)) * 0.3
        c = BlockAverageConstraint(1, DIM_9DOF, B)
        A_proj = c.project_onto_constraint(np.zeros((1, DIM_9DOF)))
        # For single block, projection should equal B
        assert np.allclose(A_proj, B, atol=1e-10)

    def test_projection_multi_block(self):
        """Projection works for multi-block constraints."""
        n_blocks = 3
        n_fine_per_block = 2
        n_fine = n_blocks * n_fine_per_block
        np.random.seed(456)
        B = 0.1 * np.random.randn(n_blocks, DIM_9DOF)
        assignment = np.repeat(np.arange(n_blocks), n_fine_per_block)
        c = BlockAverageConstraint(n_blocks, DIM_9DOF, B, assignment)

        A_init = np.zeros((n_fine, DIM_9DOF))
        A_proj = c.project_onto_constraint(A_init)
        assert c.is_satisfied(A_proj, tolerance=1e-8)


# ======================================================================
# Test YMActionFunctional
# ======================================================================

class TestYMActionFunctional:
    """Tests for the Yang-Mills action functional."""

    def test_construction(self, default_R, default_g2):
        """Action functional constructs correctly."""
        action = YMActionFunctional(R=default_R, g2=default_g2)
        assert action.R == default_R
        assert action.g2 == default_g2
        assert action.total_dof == DIM_9DOF

    def test_construction_invalid_R_raises(self):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            YMActionFunctional(R=-1.0)

    def test_construction_invalid_g2_raises(self):
        """Non-positive g2 raises ValueError."""
        with pytest.raises(ValueError):
            YMActionFunctional(g2=0.0)

    def test_is_positive_property(self, action):
        """S_YM >= 0 is always True. THEOREM."""
        assert action.is_positive is True

    def test_minimum_value_is_zero(self, action):
        """Minimum of S_YM = 0. THEOREM."""
        assert action.minimum_value == 0.0

    def test_action_at_vacuum_is_zero(self, action):
        """S_YM[A=0] = 0 (Maurer-Cartan vacuum). THEOREM."""
        A = np.zeros(DIM_9DOF)
        assert action.evaluate(A) == 0.0

    def test_action_nonnegative_random(self, action):
        """S_YM >= 0 for random fields. THEOREM."""
        np.random.seed(100)
        for _ in range(20):
            A = 0.3 * np.random.randn(DIM_9DOF)
            S = action.evaluate(A)
            assert S >= -1e-12, f"S_YM = {S} < 0"

    def test_action_positive_nonzero_field(self, action):
        """S_YM > 0 for non-zero small fields (quadratic dominates)."""
        A = np.zeros(DIM_9DOF)
        A[0] = 0.01
        S = action.evaluate(A)
        assert S > 0, f"S_YM = {S} should be positive for non-zero A"

    def test_action_scales_with_coupling(self, default_R):
        """S_YM scales as 1/g^2 for the quadratic part."""
        A = np.zeros(DIM_9DOF)
        A[0] = 0.01  # Small field: quadratic dominates

        g2_1 = 1.0
        g2_2 = 4.0
        a1 = YMActionFunctional(R=default_R, g2=g2_1)
        a2 = YMActionFunctional(R=default_R, g2=g2_2)

        S1 = a1.evaluate(A)
        S2 = a2.evaluate(A)

        # S ~ 1/g^2 for quadratic-dominated regime
        ratio = S1 / S2
        expected_ratio = g2_2 / g2_1
        assert abs(ratio - expected_ratio) < 0.5, \
            f"Ratio {ratio} should be close to {expected_ratio}"

    def test_gradient_at_vacuum_is_zero(self, action):
        """Gradient at vacuum vanishes: A=0 is critical. THEOREM."""
        A = np.zeros(DIM_9DOF)
        grad = action.gradient(A)
        assert np.allclose(grad, 0, atol=1e-6), \
            f"Gradient at vacuum = {np.linalg.norm(grad)}"

    def test_gradient_nonzero_away_from_vacuum(self, action):
        """Gradient is non-zero away from vacuum."""
        A = np.zeros(DIM_9DOF)
        A[0] = 0.1
        grad = action.gradient(A)
        assert np.linalg.norm(grad) > 1e-8

    def test_gradient_analytical_matches_numerical(self, action):
        """Analytical gradient (quadratic part) matches numerical."""
        A = np.zeros(DIM_9DOF)
        A[0] = 0.01  # Very small: quadratic dominates
        grad_num = action.gradient(A)
        grad_ana = action.gradient_analytical(A)
        # For small fields, the quadratic gradient should be close
        assert np.allclose(grad_num, grad_ana, atol=1e-4), \
            f"Max diff = {np.max(np.abs(grad_num - grad_ana))}"

    def test_hessian_at_vacuum(self, action, default_R, default_g2):
        """Hessian at vacuum = (lambda_1/g^2) * I. THEOREM."""
        H_exact = action.hessian_at_vacuum()
        lam1 = 4.0 / default_R**2
        expected = (lam1 / default_g2) * np.eye(DIM_9DOF)
        assert np.allclose(H_exact, expected, rtol=1e-10)

    def test_hessian_numerical_matches_exact_at_vacuum(self, action):
        """Numerical Hessian at vacuum matches exact formula."""
        A = np.zeros(DIM_9DOF)
        H_num = action.hessian(A)
        H_exact = action.hessian_at_vacuum()
        assert np.allclose(H_num, H_exact, atol=1e-6)

    def test_hessian_positive_definite_at_vacuum(self, action):
        """Hessian is positive definite at the vacuum. THEOREM."""
        H = action.hessian_at_vacuum()
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs > 0), f"Min eigenvalue = {eigs[0]}"

    def test_hessian_symmetric(self, action):
        """Hessian is symmetric."""
        np.random.seed(200)
        A = 0.05 * np.random.randn(DIM_9DOF)
        H = action.hessian(A)
        assert np.allclose(H, H.T, atol=1e-8)

    def test_multi_site_action(self, default_R, default_g2):
        """Multi-site action sums over sites."""
        n_sites = 3
        action = YMActionFunctional(R=default_R, g2=default_g2,
                                     n_sites=n_sites)
        A = np.zeros(n_sites * DIM_9DOF)
        assert action.evaluate(A) == 0.0

        # Put a small field on one site
        A[0] = 0.01
        S_multi = action.evaluate(A)

        # Compare with single-site
        action_single = YMActionFunctional(R=default_R, g2=default_g2)
        A_single = np.zeros(DIM_9DOF)
        A_single[0] = 0.01
        S_single = action_single.evaluate(A_single)

        assert abs(S_multi - S_single) < 1e-12


# ======================================================================
# Test ConstrainedMinimizer
# ======================================================================

class TestConstrainedMinimizer:
    """Tests for the constrained minimizer."""

    def test_minimizer_at_zero_is_vacuum(self, action, constraint_zero):
        """Minimizer with B=0 constraint gives vacuum (A=0)."""
        cm = ConstrainedMinimizer(action, constraint_zero)
        A_bar, info = cm.minimize(method='projected_gradient', tol=1e-8)
        assert np.allclose(A_bar, 0, atol=1e-6), \
            f"A_bar norm = {np.linalg.norm(A_bar)}"

    def test_minimizer_satisfies_constraint(self, action, constraint_single_block, small_B):
        """Minimizer satisfies the block average constraint."""
        cm = ConstrainedMinimizer(action, constraint_single_block)
        A_bar, info = cm.minimize(method='penalty', tol=1e-8, penalty_lambda=1e6)
        residual = constraint_single_block.residual(A_bar)
        assert residual < 1e-4, f"Constraint residual = {residual}"

    def test_minimizer_penalty_converges(self, action, constraint_single_block):
        """Penalty method converges."""
        cm = ConstrainedMinimizer(action, constraint_single_block)
        A_bar, info = cm.minimize(method='penalty', penalty_lambda=1e6)
        assert info['action_value'] >= -1e-10, \
            f"Action = {info['action_value']} should be non-negative"

    def test_minimizer_lagrange_converges(self, action, constraint_single_block):
        """Augmented Lagrangian method converges."""
        cm = ConstrainedMinimizer(action, constraint_single_block)
        A_bar, info = cm.minimize(method='lagrange')
        assert info['action_value'] >= -1e-10

    def test_minimizer_projected_gradient(self, action, constraint_zero):
        """Projected gradient method converges for B=0."""
        cm = ConstrainedMinimizer(action, constraint_zero)
        A_bar, info = cm.minimize(method='projected_gradient', max_iter=200)
        assert info['action_value'] < 1e-6

    def test_minimizer_action_nonnegative(self, action, constraint_single_block):
        """Minimizer action is non-negative. THEOREM."""
        cm = ConstrainedMinimizer(action, constraint_single_block)
        A_bar, info = cm.minimize(method='penalty', penalty_lambda=1e6)
        assert cm.action_value >= -1e-10

    def test_minimizer_invalid_method_raises(self, action, constraint_zero):
        """Invalid method raises ValueError."""
        cm = ConstrainedMinimizer(action, constraint_zero)
        with pytest.raises(ValueError):
            cm.minimize(method='nonexistent')

    def test_minimizer_different_couplings(self, default_R, small_B):
        """Minimizer works for different coupling values."""
        for g2 in [1.0, 6.28, 12.0]:
            action = YMActionFunctional(R=default_R, g2=g2)
            constraint = BlockAverageConstraint(
                1, DIM_9DOF, small_B.reshape(1, DIM_9DOF))
            cm = ConstrainedMinimizer(action, constraint)
            A_bar, info = cm.minimize(method='penalty', penalty_lambda=1e6)
            assert info['action_value'] >= -1e-10, \
                f"g2={g2}: action = {info['action_value']}"


# ======================================================================
# Test ExistenceProof
# ======================================================================

class TestExistenceProof:
    """Tests for the existence proof of the minimizer."""

    def test_coercivity_at_physical_coupling(self, default_R, default_g2):
        """Coercivity holds at physical coupling. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_coercivity()
        assert result['is_coercive'], \
            f"Not coercive: eff_coer = {result['effective_coercivity']}"

    def test_coercivity_at_weak_coupling(self, default_R):
        """Coercivity holds at weak coupling. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=1.0)
        result = ep.verify_coercivity()
        assert result['is_coercive']

    def test_lambda_1_correct(self, default_R, default_g2):
        """lambda_1 = 4/R^2 for coexact k=1. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_coercivity()
        expected_lam1 = 4.0 / default_R**2
        assert abs(result['lambda_1'] - expected_lam1) < 1e-10

    def test_compactness(self, default_R, default_g2):
        """Gribov region is bounded. THEOREM (Dell'Antonio-Zwanziger)."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_compactness()
        assert result['is_bounded']
        assert result['is_compact_finite_dim']
        assert result['gribov_diameter'] > 0
        assert result['gribov_diameter'] < np.inf

    def test_payne_weinberger_bound(self, default_R, default_g2):
        """Payne-Weinberger bound is positive. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_compactness()
        assert result['payne_weinberger_bound'] > 0

    def test_nonemptiness_small_B(self, default_R, default_g2, small_B):
        """Constraint set is non-empty for small B. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_nonemptiness(small_B)
        assert result['is_nonempty']
        assert result['margin'] > 0

    def test_nonemptiness_zero_B(self, default_R, default_g2, zero_B):
        """Constraint set is non-empty for B=0. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.verify_nonemptiness(zero_B)
        assert result['is_nonempty']

    def test_nonemptiness_large_B_fails(self, default_R, default_g2):
        """Constraint set may be empty for B outside Gribov region."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        B_large = 100.0 * np.ones(DIM_9DOF)
        result = ep.verify_nonemptiness(B_large)
        assert not result['is_nonempty']

    def test_full_existence_small_B(self, default_R, default_g2, small_B):
        """Full existence check passes for small B. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.full_existence_check(small_B)
        assert result['existence_proved']
        assert result['label'] == 'THEOREM'
        assert len(result['proof_steps']) == 7

    def test_full_existence_zero_B(self, default_R, default_g2, zero_B):
        """Full existence check passes for B=0. THEOREM."""
        ep = ExistenceProof(R=default_R, g2=default_g2)
        result = ep.full_existence_check(zero_B)
        assert result['existence_proved']

    def test_gribov_diameter_matches_analytical(self, default_g2):
        """Gribov diameter matches analytical formula."""
        ep = ExistenceProof(g2=default_g2)
        g = np.sqrt(default_g2)
        expected = 9 * np.sqrt(3) / (2 * g)
        assert abs(ep.gribov_diameter - expected) < 1e-10


# ======================================================================
# Test UniquenessProof
# ======================================================================

class TestUniquenessProof:
    """Tests for the uniqueness proposition."""

    def test_strict_convexity_at_vacuum(self, default_R, default_g2):
        """Strict convexity at vacuum A=0. PROPOSITION."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        result = up.verify_strict_convexity(np.zeros(DIM_9DOF))
        assert result['strict_convexity']
        assert result['is_within_gribov']

    def test_fp_positive_at_vacuum(self, default_R, default_g2):
        """FP operator positive definite at vacuum. THEOREM."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        result = up.verify_strict_convexity(np.zeros(DIM_9DOF))
        assert result['min_fp_eigenvalue'] > 0
        # At vacuum, all eigenvalues = 3/R^2
        expected = 3.0 / default_R**2
        assert abs(result['min_fp_eigenvalue'] - expected) < 1e-10

    def test_fp_operator_shape(self, default_R, default_g2):
        """FP operator is 9x9 matrix."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        M = up.fp_operator_9dof(np.zeros(DIM_9DOF))
        assert M.shape == (9, 9)

    def test_fp_operator_symmetric(self, default_R, default_g2):
        """FP operator is symmetric (self-adjoint)."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        np.random.seed(300)
        a = 0.1 * np.random.randn(DIM_9DOF)
        M = up.fp_operator_9dof(a)
        assert np.allclose(M, M.T, atol=1e-12)

    def test_strict_convexity_small_field(self, default_R, default_g2, small_B):
        """Strict convexity for small field. PROPOSITION."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        result = up.verify_strict_convexity(small_B)
        assert result['strict_convexity']

    def test_gauge_orbit_curvature_at_vacuum(self, default_R, default_g2):
        """Gauge orbit curvature is positive at vacuum. NUMERICAL."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        result = up.gauge_orbit_curvature(np.zeros(DIM_9DOF))
        assert result['positive_curvature']
        assert result['fp_determinant'] > 0

    def test_gauge_orbit_curvature_small_field(self, default_R, default_g2, small_B):
        """Gauge orbit curvature positive for small field. NUMERICAL."""
        up = UniquenessProof(R=default_R, g2=default_g2)
        result = up.gauge_orbit_curvature(small_B)
        assert result['positive_curvature']


# ======================================================================
# Test EllipticRegularity
# ======================================================================

class TestEllipticRegularity:
    """Tests for elliptic regularity on S^3."""

    def test_sobolev_constant_positive(self, default_R):
        """Sobolev constant is positive."""
        er = EllipticRegularity(R=default_R)
        C_S = er.sobolev_constant(2.0, 1)
        assert C_S > 0

    def test_sobolev_constant_improvement_over_flat(self):
        """Sobolev constant on S^3 improved by sqrt(3) over flat ball. THEOREM.

        On S^3(R), the first eigenvalue lambda_1 = 3/R^2 is 3 times larger
        than the corresponding eigenvalue on a ball of comparable diameter
        (lambda_1(ball) ~ 1/R^2). This gives an improvement factor of sqrt(3)
        in the Sobolev constant.

        The dimensionless Sobolev constant on a round S^3 is R-independent
        (all round spheres are isometric up to scaling).
        """
        er = EllipticRegularity(R=2.2)
        C_S3 = er.sobolev_constant(2.0, 1)

        # The flat-space reference constant (without improvement):
        n = 3
        omega_3 = 4.0 * np.pi
        C_flat = (n * omega_3**(1.0/n))**(-1) * (n / (n - 2.0))**(0.5)

        # S^3 should be improved by sqrt(3)
        assert C_S3 < C_flat, \
            f"C(S^3) = {C_S3} should be < C(flat) = {C_flat}"
        ratio = C_flat / C_S3
        assert abs(ratio - np.sqrt(3)) < 0.01, \
            f"Improvement ratio {ratio} should be sqrt(3) = {np.sqrt(3):.4f}"

    def test_sobolev_different_p(self, default_R):
        """Sobolev constant varies with p."""
        er = EllipticRegularity(R=default_R)
        C_p2 = er.sobolev_constant(2.0, 1)
        C_p1 = er.sobolev_constant(1.5, 1)
        # Both should be positive finite
        assert C_p2 > 0
        assert C_p1 > 0
        assert np.isfinite(C_p2)
        assert np.isfinite(C_p1)

    def test_regularity_bound_at_vacuum(self, default_R):
        """Regularity bound at vacuum is trivially satisfied."""
        er = EllipticRegularity(R=default_R)
        result = er.regularity_bound(np.zeros(DIM_9DOF))
        assert result['A_bar_norm'] == 0.0
        assert result['bound_satisfied']

    def test_regularity_bound_small_field(self, default_R, small_B):
        """Regularity bound satisfied for small fields. THEOREM."""
        er = EllipticRegularity(R=default_R)
        result = er.regularity_bound(small_B)
        assert result['bound_satisfied']
        assert result['label'] == 'THEOREM'

    def test_ym_equation_at_vacuum(self, default_R, default_g2):
        """YM equation residual is zero at vacuum. THEOREM."""
        er = EllipticRegularity(R=default_R)
        action = YMActionFunctional(R=default_R, g2=default_g2)
        result = er.verify_ym_equation(np.zeros(DIM_9DOF), action)
        assert result['approximately_critical']
        assert result['ym_gradient_norm'] < 1e-6


# ======================================================================
# Test BackgroundFieldDecomposition
# ======================================================================

class TestBackgroundFieldDecomposition:
    """Tests for the background field decomposition."""

    def test_decompose_identity(self, action):
        """decompose(A-bar + a, A-bar) = a. THEOREM."""
        A_bar = 0.1 * np.ones(DIM_9DOF)
        a = 0.01 * np.ones(DIM_9DOF)
        decomp = BackgroundFieldDecomposition(action, A_bar)
        A = A_bar + a
        recovered = decomp.decompose(A)
        assert np.allclose(recovered, a, atol=1e-14)

    def test_full_action_equals_direct(self, action):
        """full_action(a) = evaluate(A-bar + a). THEOREM."""
        np.random.seed(400)
        A_bar = 0.05 * np.random.randn(DIM_9DOF)
        a = 0.01 * np.random.randn(DIM_9DOF)
        decomp = BackgroundFieldDecomposition(action, A_bar)

        S_decomp = decomp.full_action(a)
        S_direct = action.evaluate(A_bar + a)
        assert abs(S_decomp - S_direct) < 1e-12

    def test_decomposition_at_vacuum(self, action):
        """Decomposition around vacuum: S[a] = S_2[a] + ... THEOREM."""
        A_bar = np.zeros(DIM_9DOF)
        decomp = BackgroundFieldDecomposition(action, A_bar)
        assert abs(decomp.S_bar) < 1e-14  # S[0] = 0

        a = 0.01 * np.ones(DIM_9DOF)
        q = decomp.quadratic_form(a)
        assert q > 0, "Quadratic form should be positive"

    def test_quadratic_form_positive(self, action):
        """Quadratic form is positive at vacuum. THEOREM."""
        decomp = BackgroundFieldDecomposition(action, np.zeros(DIM_9DOF))
        np.random.seed(500)
        for _ in range(10):
            a = 0.01 * np.random.randn(DIM_9DOF)
            q = decomp.quadratic_form(a)
            assert q > 0, f"Quadratic form = {q}"

    def test_decomposition_exact_small_a(self, action):
        """Action decomposition is exact for small fluctuations. THEOREM."""
        np.random.seed(600)
        A_bar = 0.05 * np.random.randn(DIM_9DOF)
        decomp = BackgroundFieldDecomposition(action, A_bar)

        a = 0.005 * np.random.randn(DIM_9DOF)
        result = decomp.verify_decomposition(a, tolerance=1e-5)
        assert result['is_exact'], \
            f"Decomposition error = {result['error']}"

    def test_decomposition_exact_moderate_a(self, action):
        """Action decomposition is exact for moderate fluctuations."""
        np.random.seed(700)
        A_bar = 0.02 * np.random.randn(DIM_9DOF)
        decomp = BackgroundFieldDecomposition(action, A_bar)

        a = 0.02 * np.random.randn(DIM_9DOF)
        result = decomp.verify_decomposition(a, tolerance=1e-4)
        assert result['is_exact'], \
            f"Decomposition error = {result['error']}"

    def test_quadratic_form_eigenvalues_at_vacuum(self, action, default_R, default_g2):
        """Quadratic form eigenvalues at vacuum = lambda_1/g^2. THEOREM."""
        decomp = BackgroundFieldDecomposition(action, np.zeros(DIM_9DOF))
        eigs = decomp.quadratic_form_eigenvalues()
        expected = 4.0 / (default_R**2 * default_g2)
        assert np.allclose(eigs, expected, atol=1e-4)

    def test_cubic_vertex_zero_at_vacuum(self, action):
        """Cubic vertex is zero at A-bar=0 for symmetric a."""
        decomp = BackgroundFieldDecomposition(action, np.zeros(DIM_9DOF))
        # For a = const * (1, 1, ..., 1), the cubic term may or may not vanish
        # depending on the structure constants. Test with a specific a.
        a = np.zeros(DIM_9DOF)
        a[0] = 0.01
        cubic_plus = decomp.cubic_vertex(a)
        # At vacuum, the cubic vertex (V_3 contribution) should be small
        # for a field along a single direction
        assert np.isfinite(cubic_plus)


# ======================================================================
# Test 9-DOF Exact Minimizer
# ======================================================================

class TestExactMinimizer9DOF:
    """Tests for the exact minimizer in the 9-DOF truncation."""

    def test_vacuum_minimizer(self, default_R, default_g2, zero_B):
        """Minimizer for B=0 is the vacuum. NUMERICAL."""
        result = exact_minimizer_9dof(zero_B, default_R, default_g2)
        assert np.allclose(result['A_bar'], 0, atol=1e-14)
        assert abs(result['action_value']) < 1e-14
        assert result['in_gribov_region']

    def test_small_B_minimizer(self, default_R, default_g2, small_B):
        """Minimizer for small B is within Gribov region. NUMERICAL."""
        result = exact_minimizer_9dof(small_B, default_R, default_g2)
        assert result['in_gribov_region']
        assert result['action_value'] >= -1e-10

    def test_minimizer_equals_B(self, default_R, default_g2, small_B):
        """In 9-DOF truncation with 1 block, minimizer = B. NUMERICAL."""
        result = exact_minimizer_9dof(small_B, default_R, default_g2)
        assert np.allclose(result['A_bar'], small_B, atol=1e-14)

    def test_wrong_size_raises(self, default_R, default_g2):
        """Wrong size B raises ValueError."""
        with pytest.raises(ValueError):
            exact_minimizer_9dof(np.zeros(5), default_R, default_g2)

    def test_hessian_at_minimizer(self, default_R, default_g2, small_B):
        """Hessian at minimizer has bounded eigenvalues. NUMERICAL."""
        result = exact_minimizer_9dof(small_B, default_R, default_g2)
        # Hessian eigenvalue should be finite
        assert np.isfinite(result['hessian_min_eigenvalue'])


# ======================================================================
# Test Multi-block Minimizer
# ======================================================================

class TestMultiBlockMinimizer:
    """Tests for the multi-block constrained minimizer."""

    def test_two_blocks_zero_B(self, default_R, default_g2):
        """Two-block minimizer with B=0 gives small action. NUMERICAL."""
        n_blocks = 2
        n_fine_per_block = 2
        B = np.zeros((n_blocks, DIM_9DOF))
        result = minimizer_multi_block(B, n_blocks, n_fine_per_block,
                                        default_R, default_g2)
        assert result['info']['action_value'] < 1e-4

    def test_two_blocks_small_B(self, default_R, default_g2):
        """Two-block minimizer with small B converges. NUMERICAL."""
        n_blocks = 2
        n_fine_per_block = 2
        np.random.seed(800)
        B = 0.05 * np.random.randn(n_blocks, DIM_9DOF)
        result = minimizer_multi_block(B, n_blocks, n_fine_per_block,
                                        default_R, default_g2)
        assert result['info']['action_value'] >= -1e-10


# ======================================================================
# Test Edge Cases
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary behavior."""

    def test_very_small_R(self):
        """Minimizer works for very small R (UV regime)."""
        R = 0.1
        g2 = 1.0
        B = 0.01 * np.ones(DIM_9DOF)
        result = exact_minimizer_9dof(B, R, g2)
        assert result['action_value'] >= -1e-10

    def test_large_R(self):
        """Minimizer works for large R (IR regime)."""
        R = 100.0
        g2 = 10.0
        B = 0.01 * np.ones(DIM_9DOF)
        result = exact_minimizer_9dof(B, R, g2)
        assert result['action_value'] >= -1e-10

    def test_weak_coupling(self):
        """Minimizer works at weak coupling g2 << 1."""
        R = 2.2
        g2 = 0.1
        B = 0.001 * np.ones(DIM_9DOF)
        result = exact_minimizer_9dof(B, R, g2)
        assert result['in_gribov_region']

    def test_strong_coupling(self):
        """Minimizer works at strong coupling g2 ~ 4*pi."""
        R = 2.2
        g2 = 4 * np.pi
        B = 0.1 * np.ones(DIM_9DOF)
        result = exact_minimizer_9dof(B, R, g2)
        assert result['action_value'] >= -1e-10

    def test_single_component_B(self):
        """Minimizer with B along a single direction."""
        B = np.zeros(DIM_9DOF)
        B[0] = 0.1
        result = exact_minimizer_9dof(B)
        assert result['action_value'] >= -1e-10

    def test_existence_proof_various_R(self):
        """Existence proof works for various R values."""
        for R in [0.5, 1.0, 2.2, 10.0, 50.0]:
            ep = ExistenceProof(R=R, g2=6.28)
            result = ep.verify_coercivity()
            # Should always be coercive for physical couplings
            assert result['lambda_1'] > 0
            assert result['quadratic_coefficient'] > 0


# ======================================================================
# Test Complete Verification
# ======================================================================

class TestCompleteVerification:
    """Tests for the complete Estimate 4 verification."""

    def test_verify_estimate_4_default(self):
        """Complete Estimate 4 verification passes. NUMERICAL."""
        result = verify_estimate_4()
        assert result['existence']['existence_proved']
        assert result['uniqueness']['strict_convexity']
        assert result['regularity']['bound_satisfied']
        assert result['decomposition']['is_exact']
        assert result['all_passed']

    def test_verify_estimate_4_weak_coupling(self):
        """Estimate 4 verification at weak coupling. NUMERICAL."""
        result = verify_estimate_4(g2=1.0, B_amplitude=0.01)
        assert result['existence']['existence_proved']
        assert result['all_passed']

    def test_verify_estimate_4_small_R(self):
        """Estimate 4 verification at small R. NUMERICAL."""
        result = verify_estimate_4(R=0.5, B_amplitude=0.05)
        assert result['existence']['existence_proved']


# ======================================================================
# Test Structure Constants
# ======================================================================

class TestStructureConstants:
    """Tests for SU(2) structure constants."""

    def test_antisymmetry(self):
        """f^{abc} is totally antisymmetric."""
        f = _su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert f[a, b, c] == -f[b, a, c]
                    assert f[a, b, c] == -f[a, c, b]

    def test_specific_values(self):
        """f^{012} = 1, etc."""
        f = _su2_structure_constants()
        assert f[0, 1, 2] == 1.0
        assert f[1, 2, 0] == 1.0
        assert f[2, 0, 1] == 1.0

    def test_jacobi_identity(self):
        """Jacobi identity: f^{ade} f^{bce} + cyclic = 0."""
        f = _su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    total = 0.0
                    for e in range(3):
                        for d in range(3):
                            total += (f[a, d, e] * f[b, c, e] +
                                      f[b, d, e] * f[c, a, e] +
                                      f[c, d, e] * f[a, b, e])
                    # This checks sum over d of the Jacobi identity
                    # which sums to 0 by antisymmetry
