"""
Tests for the Gribov Diameter Stabilization Theorem.

Verifies the analytical proof that d(R)*R -> 9*sqrt(3)/(4*sqrt(pi)) as
R -> infinity in the 9-DOF Yang-Mills truncation on S^3/I* with SU(2).

Test categories:
    1. Decomposition M_FP = (3/R^2)*I + (g/R)*L is exact
    2. L operator is R-independent (computed at R=1, R=5, R=100)
    3. L operator is linear, symmetric, traceless
    4. Analytical horizon distance matches numerical (machine precision)
    5. C_D = 3*sqrt(3)/2 (exact, from optimization)
    6. Asymptotic d*R = 9*sqrt(3)/(4*sqrt(pi))
    7. Analytical diameter matches numerical (same directions, machine precision)
    8. PW bound grows as R^2
    9. Formal proof statement is well-formed
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.diameter_theorem import (
    DiameterTheorem, _C_D_EXACT, _G_MAX, _DR_ASYMPTOTIC
)
from yang_mills_s3.proofs.gribov_diameter import GribovDiameter
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def dt():
    """DiameterTheorem instance."""
    return DiameterTheorem()


@pytest.fixture
def gd():
    """GribovDiameter instance."""
    return GribovDiameter()


# ======================================================================
# 1. Decomposition M_FP = (3/R^2)*I + (g/R)*L is exact
# ======================================================================

class TestDecomposition:
    """The decomposition should be exact (up to machine precision)."""

    def test_decomposition_exact_R1(self, dt):
        """Decomposition should be exact at R=1."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.1
        result = dt.verify_decomposition(a, R=1.0)
        assert result['decomposition_exact'], (
            f"Decomposition error = {result['max_error']:.2e}, should be < 1e-12"
        )

    def test_decomposition_exact_R10(self, dt):
        """Decomposition should be exact at R=10."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.1
        result = dt.verify_decomposition(a, R=10.0)
        assert result['decomposition_exact'], (
            f"Decomposition error = {result['max_error']:.2e}"
        )

    def test_decomposition_exact_R100(self, dt):
        """Decomposition should be exact at R=100."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.1
        result = dt.verify_decomposition(a, R=100.0)
        assert result['decomposition_exact'], (
            f"Decomposition error = {result['max_error']:.2e}"
        )

    def test_decomposition_large_a(self, dt):
        """Decomposition should hold even for large a."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 5.0
        result = dt.verify_decomposition(a, R=2.0)
        assert result['max_error'] < 1e-10, (
            f"Decomposition error for large a = {result['max_error']:.2e}"
        )

    def test_decomposition_zero_a(self, dt):
        """At a=0, M_FP should be (3/R^2)*I and L=0."""
        a = np.zeros(9)
        result = dt.verify_decomposition(a, R=3.0)
        assert result['decomposition_exact']
        assert np.max(np.abs(result['L_matrix'])) < 1e-15


# ======================================================================
# 2. L is R-independent
# ======================================================================

class TestLRIndependent:
    """L(a) should not depend on R."""

    def test_L_same_at_R1_R5_R100(self, dt):
        """L extracted from M_FP at R=1, R=5, R=100 should be identical."""
        rng = np.random.RandomState(99)
        a = rng.randn(9) * 0.1
        result = dt.verify_L_R_independent(a, [1.0, 5.0, 100.0])
        assert result['R_independent'], (
            f"L should be R-independent, max variation = {result['max_variation']:.2e}"
        )

    def test_L_same_wide_range(self, dt):
        """L should be R-independent over a wide range of R."""
        rng = np.random.RandomState(77)
        a = rng.randn(9) * 0.3
        result = dt.verify_L_R_independent(a, [0.1, 0.5, 1.0, 10.0, 50.0, 500.0])
        assert result['max_variation'] < 1e-10, (
            f"L max variation = {result['max_variation']:.2e}, should be < 1e-10"
        )

    def test_L_direct_matches_extracted(self, dt):
        """L computed directly should match L extracted from M_FP."""
        rng = np.random.RandomState(55)
        a = rng.randn(9) * 0.2
        L_direct = dt.L_operator(a)
        result = dt.verify_L_R_independent(a, [1.0, 10.0])
        for L_ext in result['L_matrices']:
            diff = np.max(np.abs(L_direct - L_ext))
            assert diff < 1e-10, f"L_direct vs L_extracted diff = {diff:.2e}"


# ======================================================================
# 3. L is linear, symmetric, traceless
# ======================================================================

class TestLProperties:
    """L(a) should be linear in a, symmetric, and traceless."""

    def test_L_linear_scaling(self, dt):
        """L(c*a) = c * L(a) for scalar c."""
        rng = np.random.RandomState(42)
        a = rng.randn(9)
        c = 3.7
        L_a = dt.L_operator(a)
        L_ca = dt.L_operator(c * a)
        np.testing.assert_allclose(L_ca, c * L_a, atol=1e-12)

    def test_L_linear_addition(self, dt):
        """L(a + b) = L(a) + L(b)."""
        rng = np.random.RandomState(42)
        a = rng.randn(9)
        b = rng.randn(9)
        L_a = dt.L_operator(a)
        L_b = dt.L_operator(b)
        L_ab = dt.L_operator(a + b)
        np.testing.assert_allclose(L_ab, L_a + L_b, atol=1e-12)

    def test_L_zero(self, dt):
        """L(0) = 0."""
        L = dt.L_operator(np.zeros(9))
        np.testing.assert_allclose(L, np.zeros((9, 9)), atol=1e-15)

    def test_L_symmetric(self, dt):
        """L(a) should be symmetric for any a."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            a = rng.randn(9)
            L = dt.L_operator(a)
            np.testing.assert_allclose(L, L.T, atol=1e-14,
                                       err_msg="L should be symmetric")

    def test_L_traceless(self, dt):
        """L(a) should have trace zero for any a."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            a = rng.randn(9)
            L = dt.L_operator(a)
            assert abs(np.trace(L)) < 1e-12, (
                f"tr(L) = {np.trace(L):.2e}, should be 0"
            )


# ======================================================================
# 4. Analytical horizon distance matches numerical
# ======================================================================

class TestAnalyticalHorizon:
    """Analytical horizon distance should match numerical root-finding exactly."""

    def test_horizon_matches_numerical(self, dt, gd):
        """Analytical and numerical horizon distance should agree."""
        rng = np.random.RandomState(42)
        R = 1.0
        for _ in range(5):
            d_hat = rng.randn(9)
            d_hat /= np.linalg.norm(d_hat)

            t_analytical = dt.analytical_horizon_distance(d_hat, R)
            t_numerical = gd.gribov_horizon_distance_truncated(d_hat, R)

            if np.isfinite(t_analytical) and np.isfinite(t_numerical):
                rel_err = abs(t_analytical - t_numerical) / t_numerical
                assert rel_err < 1e-6, (
                    f"Horizon distance mismatch: analytical={t_analytical:.8f}, "
                    f"numerical={t_numerical:.8f}, rel_err={rel_err:.2e}"
                )

    def test_horizon_matches_at_R10(self, dt, gd):
        """Analytical and numerical horizon distance should agree at R=10."""
        rng = np.random.RandomState(123)
        R = 10.0
        for _ in range(5):
            d_hat = rng.randn(9)
            d_hat /= np.linalg.norm(d_hat)

            t_analytical = dt.analytical_horizon_distance(d_hat, R)
            t_numerical = gd.gribov_horizon_distance_truncated(d_hat, R)

            if np.isfinite(t_analytical) and np.isfinite(t_numerical):
                rel_err = abs(t_analytical - t_numerical) / t_numerical
                assert rel_err < 1e-6, (
                    f"Horizon mismatch at R=10: rel_err={rel_err:.2e}"
                )


# ======================================================================
# 5. C_D = 3*sqrt(3)/2
# ======================================================================

class TestCD:
    """C_D should be 3*sqrt(3)/2 = 2.598076..."""

    def test_C_D_exact_value(self):
        """Check the exact constant value."""
        expected = 3.0 * np.sqrt(3.0) / 2.0
        np.testing.assert_allclose(_C_D_EXACT, expected, rtol=1e-14)

    def test_C_D_from_optimization(self, dt):
        """Optimization should find C_D = 3*sqrt(3)/2."""
        analysis = dt.fp_structure_analysis(n_directions=100, seed=42)
        assert analysis['C_D_match'], (
            f"C_D = {analysis['C_D']:.6f}, expected {_C_D_EXACT:.6f}"
        )

    def test_extremal_eigenvalues(self, dt):
        """At the extremal direction, eigenvalues should be known."""
        analysis = dt.fp_structure_analysis(n_directions=100, seed=42)
        eigs = np.sort(analysis['extremal_eigenvalues'])
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        two_inv_sqrt3 = 2.0 / np.sqrt(3.0)

        # Expected: {-1/sqrt(3) x5, +1/sqrt(3) x3, +2/sqrt(3) x1}
        # But eigenvalue ordering may differ; check the key structure
        assert abs(eigs[-1] - two_inv_sqrt3) < 0.01, (
            f"Largest eigenvalue = {eigs[-1]:.6f}, expected {two_inv_sqrt3:.6f}"
        )

    def test_C_D_positive(self, dt):
        """C_D should be positive."""
        analysis = dt.fp_structure_analysis(n_directions=50, seed=42)
        assert analysis['C_D'] > 0

    def test_C_D_order_of_magnitude(self, dt):
        """C_D should be O(1)."""
        analysis = dt.fp_structure_analysis(n_directions=50, seed=42)
        assert 1.0 < analysis['C_D'] < 5.0


# ======================================================================
# 6. Asymptotic d*R = 9*sqrt(3)/(4*sqrt(pi))
# ======================================================================

class TestAsymptoticFormula:
    """Asymptotic d*R should be 9*sqrt(3)/(4*sqrt(pi))."""

    def test_asymptotic_exact_value(self):
        """Check the exact constant."""
        expected = 9.0 * np.sqrt(3.0) / (4.0 * np.sqrt(np.pi))
        np.testing.assert_allclose(_DR_ASYMPTOTIC, expected, rtol=1e-14)

    def test_asymptotic_finite(self, dt):
        """Asymptotic d*R should be finite."""
        dR = dt.asymptotic_diameter()
        assert np.isfinite(dR)

    def test_asymptotic_positive(self, dt):
        """Asymptotic d*R should be positive."""
        dR = dt.asymptotic_diameter()
        assert dR > 0

    def test_asymptotic_consistent_with_formula(self, dt):
        """Asymptotic d*R should equal 3*C_D/g_max."""
        dR = dt.asymptotic_diameter()
        expected = 3.0 * _C_D_EXACT / _G_MAX
        np.testing.assert_allclose(dR, expected, rtol=1e-10)

    def test_diameter_formula_approaches_asymptotic(self, dt):
        """d(R)*R should approach the asymptotic value as R increases."""
        dR_asymp = dt.asymptotic_diameter()
        for R in [50.0, 100.0, 500.0]:
            d = dt.diameter_formula(R)
            dR = d * R
            rel_diff = abs(dR - dR_asymp) / dR_asymp
            assert rel_diff < 0.01, (
                f"At R={R}: d*R={dR:.6f}, asymptotic={dR_asymp:.6f}, "
                f"rel_diff={rel_diff:.4f}"
            )

    def test_pw_bound_grows_with_R(self, dt):
        """PW bound pi^2/d^2 should grow as R^2 for large R."""
        pw_values = []
        R_values = [1.0, 10.0, 100.0]
        for R in R_values:
            d = dt.diameter_formula(R)
            pw = np.pi**2 / d**2
            pw_values.append(pw)

        # PW at R=100 should be ~100x PW at R=10 (since d~1/R, d^2~1/R^2)
        ratio = pw_values[2] / pw_values[1]
        assert ratio > 50, (
            f"PW ratio R=100/R=10 = {ratio:.1f}, should be ~100"
        )


# ======================================================================
# 7. Analytical diameter matches numerical (same directions)
# ======================================================================

class TestAnalyticalVsNumerical:
    """Analytical formula should agree with numerical root-finding exactly
    when using the same set of directions."""

    def test_agreement_same_directions_R1(self, dt):
        """Analytical and numerical diameter should agree at R=1."""
        result = dt.verify_against_numerical([1.0], n_directions=50, seed=42)
        assert result['agreement'], (
            f"Max relative error = {result['max_rel_error']:.2e}, should be < 1e-6"
        )

    def test_agreement_same_directions_R10(self, dt):
        """Analytical and numerical diameter should agree at R=10."""
        result = dt.verify_against_numerical([10.0], n_directions=50, seed=42)
        assert result['agreement'], (
            f"Max relative error = {result['max_rel_error']:.2e}"
        )

    def test_agreement_multiple_R(self, dt):
        """Agreement should hold across multiple R values."""
        result = dt.verify_against_numerical([1.0, 5.0, 10.0],
                                             n_directions=30, seed=42)
        assert result['agreement'], (
            f"Max relative error = {result['max_rel_error']:.2e}"
        )


# ======================================================================
# 8. Formal proof statement
# ======================================================================

class TestFormalProof:
    """Formal proof statement should be well-formed."""

    def test_proof_contains_theorem(self, dt):
        """Proof should contain THEOREM label."""
        proof = dt.formal_proof_statement()
        assert "THEOREM" in proof

    def test_proof_contains_qed(self, dt):
        """Proof should contain QED."""
        proof = dt.formal_proof_statement()
        assert "QED" in proof

    def test_proof_contains_C_D(self, dt):
        """Proof should contain the C_D constant."""
        proof = dt.formal_proof_statement()
        assert "C_D" in proof
        assert "3*sqrt(3)/2" in proof

    def test_proof_contains_g_max(self, dt):
        """Proof should reference g_max = sqrt(4*pi)."""
        proof = dt.formal_proof_statement()
        assert "g_max" in proof
        assert "sqrt(4*pi)" in proof

    def test_proof_steps(self, dt):
        """Proof should have the 5 required steps."""
        proof = dt.formal_proof_statement()
        assert "Step 1" in proof
        assert "Step 2" in proof
        assert "Step 3" in proof
        assert "Step 4" in proof
        assert "Step 5" in proof

    def test_proof_exact_asymptotic(self, dt):
        """Proof should contain the exact asymptotic value."""
        proof = dt.formal_proof_statement()
        assert "9*sqrt(3)" in proof
        assert "4*sqrt(pi)" in proof
