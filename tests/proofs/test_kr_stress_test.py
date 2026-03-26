"""
Tests for the Kato-Rellich Safety Factor Stress Test.

Tests the numerical computation of alpha = sup ||V(a) psi|| / ||H_0 psi||
on truncated coexact modes of S^3, validating the analytic claim
that alpha < 1 at physical coupling (gap survives).

Two operators are tested:
    1. LINEAR vertex V = g^2 [theta ^ a, .] (old, WRONG for Theorem 4.1)
    2. QUADRATIC vertex V(a)psi = g^2 f^{abc}(a^b ^ a^c) * psi (CORRECT)

Test categories:
    1. H_0 eigenvalue correctness
    2. V matrix properties (symmetry, selection rules, scaling)
    3. Block coupling strengths
    4. alpha convergence with truncation
    5. alpha at physical parameters vs analytic prediction
    6. Gap stability (alpha < 1)
    7. Monotonicity of alpha with g^2
    8. Safety factor validation
    9. Mode index bookkeeping
    10. CORRECT quadratic operator tests (NEW)
    11. L^6 and H^1 norm properties (NEW)
    12. Sobolev constant comparison (NEW)
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.kr_stress_test import (
    build_H0_matrix,
    build_V_matrix,
    build_V_matrix_linear,
    compute_alpha,
    compute_alpha_linear,
    compute_alpha_quadratic,
    compute_alpha_block,
    stress_test_scan,
    validate_safety_factor,
    validate_safety_factor_correct,
    alpha_vs_coupling,
    build_mode_index,
    coexact_degeneracy,
    truncated_dimension,
    block_coupling_matrix,
    _reduced_matrix_element_sq,
    _su2_structure_constants,
    L6_norm_eigenmode,
    H1_norm_eigenmode,
    L6_over_H1_ratio,
    PHYSICAL_G_SQUARED,
    PHYSICAL_R_FM,
    ANALYTIC_C_ALPHA,
    ANALYTIC_G_C_SQUARED,
    C_SOBOLEV_SCALAR,
    C_SOBOLEV_1FORM_ACTUAL,
    C_ALPHA_CORRECTED,
    G_C_SQUARED_CORRECTED,
    VOL_S3_UNIT,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def small_N():
    """Small truncation for fast tests."""
    return 3


@pytest.fixture
def medium_N():
    """Medium truncation for convergence tests."""
    return 7


@pytest.fixture
def physical_params():
    """Physical parameter values."""
    return {'R': PHYSICAL_R_FM, 'g_squared': PHYSICAL_G_SQUARED}


# ======================================================================
# 1. H_0 eigenvalue correctness
# ======================================================================

class TestH0Matrix:
    """The H_0 matrix must have the correct coexact eigenvalues."""

    def test_diagonal(self, small_N):
        """H_0 is diagonal."""
        H0 = build_H0_matrix(small_N, R=1.0)
        off_diag = H0 - np.diag(np.diag(H0))
        assert np.max(np.abs(off_diag)) < 1e-15

    def test_eigenvalues_unit_sphere(self, small_N):
        """Eigenvalues are (k+1)^2 on the unit S^3."""
        H0 = build_H0_matrix(small_N, R=1.0)
        diag = np.diag(H0)
        index = build_mode_index(small_N)

        for i in range(len(diag)):
            k = index['k_values'][i]
            expected = (k + 1)**2
            assert abs(diag[i] - expected) < 1e-12, \
                f"Mode {i}: k={k}, expected {expected}, got {diag[i]}"

    def test_gap_is_4(self):
        """Lowest eigenvalue is 4/R^2 (k=1 coexact gap)."""
        for R in [0.5, 1.0, 2.0, 2.2]:
            H0 = build_H0_matrix(5, R=R)
            gap = np.min(np.diag(H0))
            expected = 4.0 / R**2
            assert abs(gap - expected) < 1e-12, \
                f"R={R}: gap={gap}, expected={expected}"

    def test_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R1, R2 = 1.0, 3.0
        H0_1 = build_H0_matrix(3, R=R1)
        H0_2 = build_H0_matrix(3, R=R2)
        ratio = np.diag(H0_1) / np.diag(H0_2)
        expected_ratio = (R2 / R1)**2
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)

    def test_positive_definite(self, small_N):
        """H_0 is strictly positive definite (no zero modes)."""
        H0 = build_H0_matrix(small_N, R=1.0)
        assert np.min(np.diag(H0)) > 0

    def test_dimension(self, small_N):
        """Matrix dimension matches expected degeneracy sum."""
        H0 = build_H0_matrix(small_N, R=1.0)
        expected_dim = truncated_dimension(small_N)
        assert H0.shape == (expected_dim, expected_dim)

    def test_degeneracies(self):
        """Each level k has correct degeneracy 6k(k+2)."""
        for k in [1, 2, 3, 5]:
            assert coexact_degeneracy(k) == 6 * k * (k + 2)
        # k=1: 6*1*3 = 18
        assert coexact_degeneracy(1) == 18
        # k=2: 6*2*4 = 48
        assert coexact_degeneracy(2) == 48


# ======================================================================
# 2. V matrix properties (LINEAR vertex)
# ======================================================================

class TestVMatrix:
    """The perturbation matrix V must be symmetric and respect selection rules."""

    def test_symmetric(self, small_N):
        """V is a symmetric matrix."""
        V = build_V_matrix(small_N, R=1.0, g_squared=6.28)
        np.testing.assert_allclose(V, V.T, atol=1e-14)

    def test_zero_at_zero_coupling(self, small_N):
        """V = 0 when g^2 = 0."""
        V = build_V_matrix(small_N, R=1.0, g_squared=0.0)
        assert np.max(np.abs(V)) < 1e-15

    def test_scales_with_g_squared(self, small_N):
        """V scales linearly with g^2."""
        V1 = build_V_matrix(small_N, R=1.0, g_squared=1.0)
        V2 = build_V_matrix(small_N, R=1.0, g_squared=3.0)
        # V2 should be 3 * V1
        np.testing.assert_allclose(V2, 3.0 * V1, atol=1e-14)

    def test_scales_with_R_squared(self, small_N):
        """V scales as 1/R^2 (same as H_0)."""
        V1 = build_V_matrix(small_N, R=1.0, g_squared=1.0)
        V2 = build_V_matrix(small_N, R=2.0, g_squared=1.0)
        # V2 should be V1 / 4
        np.testing.assert_allclose(V2, V1 / 4.0, atol=1e-14)

    def test_block_tridiagonal(self, small_N):
        """V is block-tridiagonal: V_{k,k'} = 0 for |k-k'| > 1."""
        V = build_V_matrix(small_N, R=1.0, g_squared=6.28)
        index = build_mode_index(small_N)

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                k_i = index['k_values'][i]
                k_j = index['k_values'][j]
                if abs(k_i - k_j) > 1:
                    assert abs(V[i, j]) < 1e-15, \
                        f"V[{i},{j}] = {V[i,j]} but |k-k'| = {abs(k_i-k_j)} > 1"

    def test_nonzero_off_diagonal_blocks(self, small_N):
        """V has nonzero elements in the adjacent k-blocks."""
        V = build_V_matrix(small_N, R=1.0, g_squared=6.28)
        # Check that V is not entirely zero
        assert np.max(np.abs(V)) > 1e-15, "V matrix is all zeros"

    def test_correct_dimension(self, small_N):
        """V has the same dimension as H_0."""
        H0 = build_H0_matrix(small_N, R=1.0)
        V = build_V_matrix(small_N, R=1.0, g_squared=6.28)
        assert V.shape == H0.shape


# ======================================================================
# 3. Block coupling strengths
# ======================================================================

class TestBlockCoupling:
    """Block coupling matrix from reduced matrix elements."""

    def test_selection_rule(self):
        """Coupling vanishes for |k - k'| > 1."""
        assert _reduced_matrix_element_sq(1, 5) == 0.0
        assert _reduced_matrix_element_sq(2, 10) == 0.0

    def test_positive_adjacent(self):
        """Coupling is positive for adjacent levels."""
        for k in range(1, 10):
            assert _reduced_matrix_element_sq(k, k + 1) > 0
            assert _reduced_matrix_element_sq(k, k) > 0

    def test_symmetry(self):
        """C(k, k') = C(k', k)."""
        for k in range(1, 8):
            for kp in range(1, 8):
                c1 = _reduced_matrix_element_sq(k, kp)
                c2 = _reduced_matrix_element_sq(kp, k)
                assert abs(c1 - c2) < 1e-14, \
                    f"C({k},{kp}) = {c1} != C({kp},{k}) = {c2}"

    def test_coupling_matrix_symmetric(self):
        """Block coupling matrix is symmetric."""
        C = block_coupling_matrix(10)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_coupling_matrix_tridiagonal(self):
        """Block coupling matrix is tridiagonal."""
        C = block_coupling_matrix(10)
        for i in range(10):
            for j in range(10):
                if abs(i - j) > 1:
                    assert abs(C[i, j]) < 1e-15

    def test_coupling_decays_with_k(self):
        """Off-diagonal coupling normalized by eigenvalue decays."""
        ratios = []
        for k in range(1, 10):
            c = _reduced_matrix_element_sq(k, k + 1)
            ratio = c / ((k + 1)**2 * (k + 2)**2)
            ratios.append(ratio)
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10, \
                f"Coupling ratio not decaying: {ratios[i]} < {ratios[i+1]}"

    def test_structure_constants_antisymmetric(self):
        """f^{abc} = -f^{bac} (antisymmetry)."""
        f = _su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert abs(f[a, b, c] + f[b, a, c]) < 1e-15

    def test_structure_constants_epsilon(self):
        """f^{123} = 1 for su(2)."""
        f = _su2_structure_constants()
        assert abs(f[0, 1, 2] - 1.0) < 1e-15
        assert abs(f[1, 2, 0] - 1.0) < 1e-15
        assert abs(f[2, 0, 1] - 1.0) < 1e-15


# ======================================================================
# 4. alpha convergence with truncation (CORRECT quadratic operator)
# ======================================================================

class TestAlphaConvergence:
    """alpha(N) should converge as truncation N increases."""

    def test_alpha_bounded(self, small_N):
        """alpha is finite for any truncation level."""
        res = compute_alpha(small_N, R=1.0, g_squared=6.28)
        assert np.isfinite(res['alpha'])
        assert res['alpha'] >= 0

    def test_alpha_below_one_physical(self, medium_N):
        """alpha < 1 at physical coupling for reasonable truncation."""
        res = compute_alpha(medium_N, R=PHYSICAL_R_FM, g_squared=PHYSICAL_G_SQUARED)
        assert res['alpha'] < 1.0, \
            f"alpha = {res['alpha']} >= 1 at physical coupling"

    def test_alpha_not_zero(self, small_N):
        """alpha > 0 at nonzero coupling."""
        res = compute_alpha(small_N, R=1.0, g_squared=6.28)
        assert res['alpha'] > 0, \
            f"alpha = {res['alpha']} should be positive"

    def test_scan_convergence(self):
        """Stress test scan produces valid results."""
        scan = stress_test_scan(R=1.0, g_squared=6.28,
                                N_range=[2, 3, 5])
        for alpha in scan['alpha_values']:
            assert np.isfinite(alpha)
            assert alpha >= 0

    def test_dimension_grows(self):
        """Truncated space dimension grows with N."""
        dims = [truncated_dimension(N) for N in [2, 3, 5, 7, 10]]
        for i in range(len(dims) - 1):
            assert dims[i] < dims[i + 1]

    def test_block_alpha_bounded(self, small_N):
        """Block-level alpha is finite and bounded."""
        res = compute_alpha_block(small_N, R=1.0, g_squared=6.28)
        assert np.isfinite(res['alpha'])
        assert res['alpha'] >= 0


# ======================================================================
# 5. Gap stability (alpha < 1)
# ======================================================================

class TestGapStability:
    """The mass gap survives: alpha < 1 at physical coupling."""

    def test_alpha_below_one(self):
        """alpha < 1 at physical coupling, multiple truncations."""
        for N in [2, 3, 5]:
            res = compute_alpha(N, R=PHYSICAL_R_FM, g_squared=PHYSICAL_G_SQUARED)
            assert res['alpha'] < 1.0, \
                f"N={N}: alpha = {res['alpha']} >= 1"

    def test_safety_factor_positive(self, small_N):
        """Safety factor 1/alpha > 1."""
        res = compute_alpha(small_N, R=PHYSICAL_R_FM, g_squared=PHYSICAL_G_SQUARED)
        assert res['safety_factor'] > 1.0

    def test_gap_fraction_retained(self, small_N):
        """Significant fraction of the linearized gap is retained."""
        res = compute_alpha(small_N, R=PHYSICAL_R_FM, g_squared=PHYSICAL_G_SQUARED)
        retained = 1.0 - res['alpha']
        assert retained > 0.1, \
            f"Only {retained*100:.1f}% of gap retained (alpha={res['alpha']})"

    def test_block_alpha_below_one(self, small_N):
        """Block-level alpha < 1 at physical coupling."""
        res = compute_alpha_block(small_N, R=PHYSICAL_R_FM, g_squared=PHYSICAL_G_SQUARED)
        assert res['alpha'] < 1.0, \
            f"Block alpha = {res['alpha']} >= 1"


# ======================================================================
# 6. Monotonicity of alpha with g^2
# ======================================================================

class TestMonotonicity:
    """alpha should increase monotonically with g^2."""

    def test_alpha_increases_with_g2(self, small_N):
        """alpha(g^2) is monotonically increasing."""
        g2_values = [1.0, 5.0, 10.0, 20.0, 50.0]
        alphas = []
        for g2 in g2_values:
            res = compute_alpha(small_N, R=1.0, g_squared=g2)
            alphas.append(res['alpha'])

        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1] + 1e-14, \
                f"alpha not monotone: g2={g2_values[i]} -> alpha={alphas[i]}, " \
                f"g2={g2_values[i+1]} -> alpha={alphas[i+1]}"

    def test_alpha_linear_in_g2(self, small_N):
        """alpha is approximately linear in g^2 (from Sobolev bound)."""
        result = alpha_vs_coupling(N_max=small_N, R=1.0)
        # R^2 of linear fit should be > 0.95
        assert result['linearity_r2'] > 0.95, \
            f"Linearity R^2 = {result['linearity_r2']} < 0.95"

    def test_block_alpha_increases_with_g2(self, small_N):
        """Block-level alpha increases with g^2."""
        g2_values = [1.0, 10.0, 50.0]
        alphas = []
        for g2 in g2_values:
            res = compute_alpha_block(small_N, R=1.0, g_squared=g2)
            alphas.append(res['alpha'])

        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1] + 1e-14


# ======================================================================
# 7. Safety factor validation
# ======================================================================

class TestSafetyFactor:
    """The safety factor 1/alpha should confirm gap survival."""

    def test_validate_at_physical(self):
        """Safety factor validation at physical parameters."""
        result = validate_safety_factor(N_max=5)
        assert result['gap_survives'], \
            "Gap does not survive at physical coupling!"

    def test_numerical_alpha_bounded(self):
        """Numerical alpha should be bounded and positive."""
        result = validate_safety_factor(N_max=5)
        assert 0 < result['numerical_alpha'] < 1.0, \
            f"Numerical alpha = {result['numerical_alpha']} out of range"

    def test_analytic_alpha_small(self):
        """Analytic prediction: alpha ~ 0.0375 (paper's constant)."""
        analytic = ANALYTIC_C_ALPHA * PHYSICAL_G_SQUARED
        assert 0.03 < analytic < 0.05, \
            f"Analytic alpha = {analytic} out of expected range"
        # More precise check
        assert abs(analytic - 0.0375) < 0.003, \
            f"Analytic alpha = {analytic} != 0.0375"

    def test_analytic_safety_factor(self):
        """Analytic safety factor = g^2_c / g^2_phys ~ 26.7 (paper's constant)."""
        safety = ANALYTIC_G_C_SQUARED / PHYSICAL_G_SQUARED
        assert 25 < safety < 28, \
            f"Analytic safety factor = {safety} out of range"

    def test_R_independence_of_alpha(self, small_N):
        """alpha should be R-independent (both H_0 and V scale as 1/R^2)."""
        res1 = compute_alpha(small_N, R=1.0, g_squared=6.28)
        res2 = compute_alpha(small_N, R=2.2, g_squared=6.28)
        res3 = compute_alpha(small_N, R=5.0, g_squared=6.28)
        # All alphas should be approximately equal
        np.testing.assert_allclose(res1['alpha'], res2['alpha'], rtol=1e-10)
        np.testing.assert_allclose(res1['alpha'], res3['alpha'], rtol=1e-10)


# ======================================================================
# 8. Mode index bookkeeping
# ======================================================================

class TestModeIndex:
    """Mode index construction must be consistent."""

    def test_total_dimension(self):
        """Total dimension matches sum of degeneracies."""
        for N in [1, 2, 3, 5, 10]:
            expected = sum(6 * k * (k + 2) for k in range(1, N + 1))
            assert truncated_dimension(N) == expected

    def test_index_coverage(self, small_N):
        """All modes are indexed exactly once."""
        index = build_mode_index(small_N)
        assert index['dim'] == truncated_dimension(small_N)
        assert len(index['k_values']) == index['dim']
        assert len(index['m_values']) == index['dim']
        assert len(index['a_values']) == index['dim']

    def test_k_values_in_range(self, small_N):
        """All k values are in [1, N_max]."""
        index = build_mode_index(small_N)
        assert np.all(index['k_values'] >= 1)
        assert np.all(index['k_values'] <= small_N)

    def test_adjoint_values(self, small_N):
        """Adjoint indices are 0, 1, 2."""
        index = build_mode_index(small_N)
        assert set(index['a_values'].tolist()).issubset({0, 1, 2})

    def test_k_ranges_consistent(self, small_N):
        """k_ranges boundaries are consistent with the index."""
        index = build_mode_index(small_N)
        for k in range(1, small_N + 1):
            start, end = index['k_ranges'][k]
            expected_count = 6 * k * (k + 2)
            assert end - start == expected_count, \
                f"k={k}: range [{start},{end}) has {end-start} modes, expected {expected_count}"
            assert np.all(index['k_values'][start:end] == k)

    def test_specific_dimensions(self):
        """Check specific dimension values."""
        assert truncated_dimension(1) == 18
        assert truncated_dimension(2) == 66
        assert truncated_dimension(3) == 156


# ======================================================================
# 9. CORRECT quadratic operator tests (NEW)
# ======================================================================

class TestQuadraticOperator:
    """Tests for the CORRECT V(a)psi = g^2 f^{abc}(a^b ^ a^c) * psi."""

    def test_alpha_quadratic_is_positive(self):
        """Alpha for the quadratic operator is positive at nonzero coupling."""
        res = compute_alpha_quadratic(5, g_squared=6.28)
        assert res['alpha'] > 0
        assert np.isfinite(res['alpha'])

    def test_alpha_quadratic_below_one(self):
        """Alpha for the quadratic operator is < 1 at physical coupling."""
        res = compute_alpha_quadratic(10, g_squared=PHYSICAL_G_SQUARED)
        assert res['alpha'] < 1.0, \
            f"alpha = {res['alpha']} >= 1"

    def test_alpha_quadratic_much_less_than_linear(self):
        """
        The quadratic operator gives a MUCH smaller alpha than the linear.

        The linear vertex V = g^2[theta ^ a, .] gives alpha ~ 0.356.
        The quadratic vertex V(a) = g^2[a ^ a, .] gives alpha ~ 0.043.
        This is because the quadratic vertex involves |a|^2, not |theta|*|a|.
        """
        res_quad = compute_alpha(5, g_squared=PHYSICAL_G_SQUARED)
        res_lin = compute_alpha_linear(5, g_squared=PHYSICAL_G_SQUARED)

        assert res_quad['alpha'] < res_lin['alpha'], \
            f"Quadratic alpha {res_quad['alpha']} >= linear alpha {res_lin['alpha']}"
        # At least 5x smaller
        assert res_quad['alpha'] < res_lin['alpha'] / 5.0

    def test_alpha_scales_linearly_with_g2(self):
        """Alpha = C * g^2 (linear in coupling)."""
        alpha1 = compute_alpha_quadratic(5, g_squared=1.0)['alpha']
        alpha2 = compute_alpha_quadratic(5, g_squared=2.0)['alpha']
        alpha3 = compute_alpha_quadratic(5, g_squared=10.0)['alpha']
        np.testing.assert_allclose(alpha2 / alpha1, 2.0, rtol=1e-10)
        np.testing.assert_allclose(alpha3 / alpha1, 10.0, rtol=1e-10)

    def test_alpha_R_independent(self):
        """Alpha is R-independent for the quadratic operator."""
        for g2 in [1.0, 6.28, 50.0]:
            a1 = compute_alpha_quadratic(5, R=0.5, g_squared=g2)['alpha']
            a2 = compute_alpha_quadratic(5, R=1.0, g_squared=g2)['alpha']
            a3 = compute_alpha_quadratic(5, R=2.2, g_squared=g2)['alpha']
            a4 = compute_alpha_quadratic(5, R=10.0, g_squared=g2)['alpha']
            np.testing.assert_allclose(a1, a2, rtol=1e-10)
            np.testing.assert_allclose(a2, a3, rtol=1e-10)
            np.testing.assert_allclose(a3, a4, rtol=1e-10)

    def test_alpha_N_independent(self):
        """Alpha is N-independent (Sobolev bound is analytic, not truncation-dependent)."""
        alphas = [compute_alpha_quadratic(N, g_squared=6.28)['alpha']
                  for N in [2, 5, 10, 20]]
        for a in alphas:
            np.testing.assert_allclose(a, alphas[0], rtol=1e-10)

    def test_alpha_zero_at_zero_coupling(self):
        """Alpha = 0 when g^2 = 0."""
        res = compute_alpha_quadratic(5, g_squared=0.0)
        assert abs(res['alpha']) < 1e-15

    def test_worst_case_is_k1(self):
        """The worst-case a is at level k=1."""
        res = compute_alpha_quadratic(10, g_squared=PHYSICAL_G_SQUARED)
        # All alpha_per_k should be equal (since we use the same Sobolev constant)
        # but the worst case is k=1
        assert res['worst_k'] >= 1
        # alpha_per_k at k=1 should be >= all others
        alpha_k1 = res['alpha_per_k'][0]
        for alpha_k in res['alpha_per_k']:
            assert alpha_k <= alpha_k1 + 1e-14

    def test_paper_vs_honest_comparison(self):
        """Compare paper's C_S vs honest C for 1-forms."""
        res_paper = compute_alpha_quadratic(10, g_squared=PHYSICAL_G_SQUARED,
                                             use_paper_constant=True)
        res_honest = compute_alpha_quadratic(10, g_squared=PHYSICAL_G_SQUARED,
                                              use_paper_constant=False)
        # The honest alpha should be larger than the paper's
        # (because the 1-form Sobolev constant is larger)
        assert res_honest['alpha'] > res_paper['alpha'], \
            f"Honest alpha {res_honest['alpha']} <= paper alpha {res_paper['alpha']}"
        # But both should be < 1 (gap survives either way)
        assert res_paper['alpha'] < 1.0
        assert res_honest['alpha'] < 1.0

    def test_safety_factor_correct(self):
        """Validate the corrected safety factor."""
        result = validate_safety_factor_correct(N_max=10)
        assert result['validated'], "Gap should survive"
        assert result['gap_survives'], "Gap must survive"
        # Safety factor should be > 10 (honest) and > 20 (paper)
        assert result['safety_factor_honest'] > 10
        assert result['safety_factor_paper'] > 20

    def test_corrected_constants(self):
        """Corrected Sobolev constant and critical coupling."""
        # C_S_1form > C_S_scalar (1-form constant is larger)
        assert C_SOBOLEV_1FORM_ACTUAL > C_SOBOLEV_SCALAR
        # Corrected C_alpha > paper's C_alpha
        assert C_ALPHA_CORRECTED > ANALYTIC_C_ALPHA
        # Corrected g^2_c < paper's g^2_c (tighter bound)
        assert G_C_SQUARED_CORRECTED < ANALYTIC_G_C_SQUARED
        # But still g^2_c > physical g^2 (gap survives)
        assert G_C_SQUARED_CORRECTED > PHYSICAL_G_SQUARED


# ======================================================================
# 10. L^6 and H^1 norm properties (NEW)
# ======================================================================

class TestEigenmodeNorms:
    """Tests for L^6 and H^1 norms of coexact eigenmodes on S^3."""

    def test_H1_norm_k1(self):
        """H^1 norm of k=1 mode on unit S^3 is sqrt(3)."""
        H1 = H1_norm_eigenmode(1, R=1.0)
        np.testing.assert_allclose(H1, np.sqrt(3.0), rtol=1e-12)

    def test_H1_norm_formula(self):
        """H^1 norm follows the formula sqrt(1 + ((k+1)^2 - 2)/R^2)."""
        for k in range(1, 8):
            for R in [0.5, 1.0, 2.0]:
                H1 = H1_norm_eigenmode(k, R)
                expected = np.sqrt(1.0 + ((k + 1)**2 - 2.0) / R**2)
                np.testing.assert_allclose(H1, expected, rtol=1e-12)

    def test_H1_norm_increases_with_k(self):
        """H^1 norm increases with k (higher modes are rougher)."""
        for R in [0.5, 1.0, 2.0]:
            norms = [H1_norm_eigenmode(k, R) for k in range(1, 10)]
            for i in range(len(norms) - 1):
                assert norms[i] < norms[i + 1]

    def test_L6_norm_k1_constant(self):
        """L^6 norm of k=1 mode = vol(S^3)^{-1/3} on unit S^3."""
        L6 = L6_norm_eigenmode(1, R=1.0)
        expected = VOL_S3_UNIT**(-1.0 / 3.0)
        np.testing.assert_allclose(L6, expected, rtol=1e-12)

    def test_L6_norm_positive(self):
        """L^6 norms are positive for all k."""
        for k in range(1, 10):
            assert L6_norm_eigenmode(k, R=1.0) > 0

    def test_L6_over_H1_k1_value(self):
        """L^6/H^1 ratio at k=1 on unit S^3 = vol^{-1/3}/sqrt(3)."""
        ratio = L6_over_H1_ratio(1, R=1.0)
        expected = VOL_S3_UNIT**(-1.0 / 3.0) / np.sqrt(3.0)
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)

    def test_L6_over_H1_bounded_by_k1(self):
        """L^6/H^1 ratio is maximized at k=1 (worst case for Sobolev)."""
        ratio_k1 = L6_over_H1_ratio(1, R=1.0)
        for k in range(2, 10):
            ratio_k = L6_over_H1_ratio(k, R=1.0)
            assert ratio_k <= ratio_k1 + 1e-14, \
                f"k={k}: ratio {ratio_k} > k=1 ratio {ratio_k1}"

    def test_L6_over_H1_exceeds_paper_CS(self):
        """
        The actual L^6/H^1 ratio at k=1 EXCEEDS the paper's C_S.

        This is the key finding: the Aubin-Talenti scalar constant
        on R^3 is NOT sufficient for 1-forms on compact S^3.
        The ratio 0.2136 > C_S = 0.1826.
        """
        ratio = L6_over_H1_ratio(1, R=1.0)
        assert ratio > C_SOBOLEV_SCALAR, \
            f"L^6/H^1 ratio {ratio} should exceed C_S = {C_SOBOLEV_SCALAR}"

    def test_sobolev_constant_value(self):
        """C_SOBOLEV_1FORM_ACTUAL = vol^{-1/3}/sqrt(3)."""
        expected = VOL_S3_UNIT**(-1.0 / 3.0) / np.sqrt(3.0)
        np.testing.assert_allclose(C_SOBOLEV_1FORM_ACTUAL, expected, rtol=1e-12)


# ======================================================================
# 11. Sobolev constant comparison (NEW)
# ======================================================================

class TestSobolevComparison:
    """
    Compare the paper's Sobolev constant with the corrected value.

    The paper uses C_S = (4/3)(2*pi^2)^{-2/3} ~ 0.1826 (Aubin-Talenti
    for scalars on R^3). For 1-forms on compact S^3, the actual H^1 -> L^6
    Sobolev constant is C_actual = vol^{-1/3}/sqrt(3) ~ 0.2136 > C_S.

    The discrepancy reduces the safety factor from ~37 to ~23 but does
    NOT invalidate the mass gap theorem (alpha < 1 in both cases).
    """

    def test_paper_constant_value(self):
        """C_S = (4/3)(2*pi^2)^{-2/3}."""
        expected = (4.0 / 3.0) * (2.0 * np.pi**2)**(-2.0 / 3.0)
        np.testing.assert_allclose(C_SOBOLEV_SCALAR, expected, rtol=1e-12)

    def test_actual_constant_larger(self):
        """The actual constant for 1-forms on S^3 is larger than the paper's."""
        assert C_SOBOLEV_1FORM_ACTUAL > C_SOBOLEV_SCALAR
        # Ratio should be about 1.17
        ratio = C_SOBOLEV_1FORM_ACTUAL / C_SOBOLEV_SCALAR
        assert 1.1 < ratio < 1.3, f"Ratio = {ratio}"

    def test_alpha_paper_value(self):
        """Paper's alpha at physical coupling ~ 0.0375."""
        alpha = ANALYTIC_C_ALPHA * PHYSICAL_G_SQUARED
        np.testing.assert_allclose(alpha, 0.0375, atol=0.003)

    def test_alpha_corrected_value(self):
        """Corrected alpha at physical coupling ~ 0.043."""
        alpha = C_ALPHA_CORRECTED * PHYSICAL_G_SQUARED
        assert 0.03 < alpha < 0.06, f"alpha = {alpha}"

    def test_both_give_gap_survival(self):
        """Both the paper's and corrected alpha give alpha < 1."""
        alpha_paper = ANALYTIC_C_ALPHA * PHYSICAL_G_SQUARED
        alpha_corrected = C_ALPHA_CORRECTED * PHYSICAL_G_SQUARED
        assert alpha_paper < 1.0
        assert alpha_corrected < 1.0

    def test_corrected_safety_factor(self):
        """Corrected safety factor is > 20 (still very safe)."""
        safety = G_C_SQUARED_CORRECTED / PHYSICAL_G_SQUARED
        assert safety > 20, f"Safety factor = {safety}"

    def test_volume_unit_S3(self):
        """Volume of unit S^3 = 2*pi^2."""
        np.testing.assert_allclose(VOL_S3_UNIT, 2.0 * np.pi**2, rtol=1e-12)

    def test_alpha_cubic_in_C(self):
        """
        C_ALPHA_CORRECTED = sqrt(2) * C_SOBOLEV_1FORM_ACTUAL^3 / 2
        and ANALYTIC_C_ALPHA = sqrt(2)/(24*pi^2).
        Both are small enough that physical coupling is safe.
        """
        # Verify C_ALPHA_CORRECTED formula
        expected = np.sqrt(2) * C_SOBOLEV_1FORM_ACTUAL**3 / 2.0
        np.testing.assert_allclose(C_ALPHA_CORRECTED, expected, rtol=1e-10)
        # Both give alpha < 1 at physical coupling
        assert C_ALPHA_CORRECTED * PHYSICAL_G_SQUARED < 1.0
        assert ANALYTIC_C_ALPHA * PHYSICAL_G_SQUARED < 1.0


# ======================================================================
# 12. Linear vertex comparison (NEW)
# ======================================================================

class TestLinearComparison:
    """Tests comparing the linear (wrong) and quadratic (correct) operators."""

    def test_linear_alpha_value(self):
        """Linear vertex gives alpha ~ 0.36 at physical coupling."""
        res = compute_alpha_linear(5, R=1.0, g_squared=PHYSICAL_G_SQUARED)
        assert 0.2 < res['alpha'] < 0.5, \
            f"Linear alpha = {res['alpha']}"

    def test_linear_still_below_one(self):
        """Even the linear vertex gives alpha < 1 (gap survives)."""
        res = compute_alpha_linear(5, R=1.0, g_squared=PHYSICAL_G_SQUARED)
        assert res['alpha'] < 1.0

    def test_linear_R_independent(self):
        """Linear alpha is also R-independent."""
        a1 = compute_alpha_linear(3, R=1.0, g_squared=6.28)['alpha']
        a2 = compute_alpha_linear(3, R=2.2, g_squared=6.28)['alpha']
        np.testing.assert_allclose(a1, a2, rtol=1e-10)

    def test_build_V_matrix_alias(self, small_N):
        """build_V_matrix is an alias for build_V_matrix_linear."""
        V1 = build_V_matrix(small_N, R=1.0, g_squared=6.28)
        V2 = build_V_matrix_linear(small_N, R=1.0, g_squared=6.28)
        np.testing.assert_allclose(V1, V2, atol=1e-15)
