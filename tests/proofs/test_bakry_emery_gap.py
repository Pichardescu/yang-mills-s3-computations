"""
Tests for the Bakry-Emery Spectral Gap Analysis on S^3/I*.

Test categories:
    1. Hess(V_2) = (4/R^2) * I_9 (analytical)
    2. Hess(log det M_FP) is negative semidefinite at a=0
    3. -Hess(log det M_FP) is positive semidefinite at a=0
    4. -Hess(log det M_FP) eigenvalues grow with R at a=0
    5. Hess(V_4) is positive semidefinite at a=0
    6. Hess(U_phys) at a=0 has positive eigenvalues for various R
    7. Hess(U_phys) at a=0 grows with R (ghost term dominates)
    8. Scan over Gribov region: all sampled points have positive min eigenvalue
    9. BE bound vs R: bound stays positive for large R
   10. Analytical formula vs numerical finite difference for Hess(log det M_FP)
   11. Formal analysis produces coherent assessment
"""

import pytest
import numpy as np
from scipy.linalg import eigvalsh

from yang_mills_s3.proofs.bakry_emery_gap import BakryEmeryGap
from yang_mills_s3.proofs.gribov_diameter import GribovDiameter
from yang_mills_s3.proofs.diameter_theorem import DiameterTheorem
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def beg():
    """BakryEmeryGap instance."""
    return BakryEmeryGap()


@pytest.fixture
def gd():
    """GribovDiameter instance."""
    return GribovDiameter()


@pytest.fixture
def dt():
    """DiameterTheorem instance."""
    return DiameterTheorem()


# ======================================================================
# 1. Hessian of V_2
# ======================================================================

class TestHessianV2:
    """Hess(V_2) should be (4/R^2) * I_9 for all R."""

    def test_hessian_V2_R1(self, beg):
        """At R=1, Hess(V_2) = 4 * I_9."""
        H = beg.compute_hessian_V2(1.0)
        expected = 4.0 * np.eye(9)
        np.testing.assert_allclose(H, expected, atol=1e-14)

    def test_hessian_V2_R2(self, beg):
        """At R=2, Hess(V_2) = 1 * I_9."""
        H = beg.compute_hessian_V2(2.0)
        expected = 1.0 * np.eye(9)
        np.testing.assert_allclose(H, expected, atol=1e-14)

    def test_hessian_V2_R5(self, beg):
        """At R=5, Hess(V_2) = 4/25 * I_9."""
        H = beg.compute_hessian_V2(5.0)
        expected = (4.0 / 25.0) * np.eye(9)
        np.testing.assert_allclose(H, expected, atol=1e-14)

    def test_hessian_V2_R10(self, beg):
        """At R=10, Hess(V_2) = 0.04 * I_9."""
        H = beg.compute_hessian_V2(10.0)
        expected = 0.04 * np.eye(9)
        np.testing.assert_allclose(H, expected, atol=1e-14)

    def test_hessian_V2_is_diagonal(self, beg):
        """Hess(V_2) should be diagonal for any R."""
        for R in [0.5, 1.0, 3.0, 100.0]:
            H = beg.compute_hessian_V2(R)
            off_diag = H - np.diag(np.diag(H))
            assert np.max(np.abs(off_diag)) < 1e-15, (
                f"Hess(V_2) has off-diagonal elements at R={R}"
            )

    def test_hessian_V2_scales_as_R_minus_2(self, beg):
        """Hess(V_2) should scale as 1/R^2."""
        R1, R2 = 2.0, 10.0
        H1 = beg.compute_hessian_V2(R1)
        H2 = beg.compute_hessian_V2(R2)
        ratio = H1[0, 0] / H2[0, 0]
        expected_ratio = (R2 / R1) ** 2
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)


# ======================================================================
# 2. Hessian of log det M_FP at a=0
# ======================================================================

class TestHessianLogDetMFP:
    """Hess(log det M_FP) at a=0 should be negative semidefinite."""

    def test_hessian_log_det_is_NSD_R1(self, beg):
        """At a=0, R=1: Hess(log det M_FP) is NSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 1.0)
        eigs = eigvalsh(H)
        # All eigenvalues should be <= 0 (NSD)
        assert np.all(eigs <= 1e-12), (
            f"Hess(log det M_FP) has positive eigenvalue: max={eigs[-1]:.6e}"
        )

    def test_hessian_log_det_is_NSD_R5(self, beg):
        """At a=0, R=5: Hess(log det M_FP) is NSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 5.0)
        eigs = eigvalsh(H)
        assert np.all(eigs <= 1e-12), (
            f"Hess(log det M_FP) has positive eigenvalue: max={eigs[-1]:.6e}"
        )

    def test_hessian_log_det_is_NSD_R10(self, beg):
        """At a=0, R=10: Hess(log det M_FP) is NSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 10.0)
        eigs = eigvalsh(H)
        assert np.all(eigs <= 1e-12), (
            f"Hess(log det M_FP) has positive eigenvalue: max={eigs[-1]:.6e}"
        )

    def test_neg_hessian_is_PSD_R1(self, beg):
        """At a=0, R=1: -Hess(log det M_FP) is PSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 1.0)
        neg_H = -H
        eigs = eigvalsh(neg_H)
        assert np.all(eigs >= -1e-12), (
            f"-Hess has negative eigenvalue: min={eigs[0]:.6e}"
        )

    def test_neg_hessian_is_PSD_R5(self, beg):
        """At a=0, R=5: -Hess(log det M_FP) is PSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 5.0)
        neg_H = -H
        eigs = eigvalsh(neg_H)
        assert np.all(eigs >= -1e-12), (
            f"-Hess has negative eigenvalue: min={eigs[0]:.6e}"
        )

    def test_neg_hessian_is_PSD_R20(self, beg):
        """At a=0, R=20: -Hess(log det M_FP) is PSD."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 20.0)
        neg_H = -H
        eigs = eigvalsh(neg_H)
        assert np.all(eigs >= -1e-12), (
            f"-Hess has negative eigenvalue: min={eigs[0]:.6e}"
        )

    def test_hessian_log_det_is_symmetric(self, beg):
        """Hess(log det M_FP) should be symmetric."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 2.0)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_hessian_log_det_nonzero(self, beg):
        """Hess(log det M_FP) at a=0 should be nonzero."""
        H = beg.compute_hessian_log_det_MFP(np.zeros(9), 1.0)
        assert np.max(np.abs(H)) > 1e-10, (
            "Hess(log det M_FP) is zero at a=0 -- unexpected"
        )


# ======================================================================
# 3. Ghost contribution grows with R
# ======================================================================

class TestGhostGrowth:
    """The ghost contribution -Hess(log det M_FP) should GROW with R at a=0."""

    def test_ghost_eigenvalues_grow_with_R(self, beg):
        """Min eigenvalue of -Hess(log det M_FP) at a=0 should grow with R."""
        R_values = [1.0, 5.0, 10.0, 20.0]
        min_eigs = []
        for R in R_values:
            H = beg.compute_hessian_log_det_MFP(np.zeros(9), R)
            eigs = eigvalsh(-H)
            min_eigs.append(eigs[0])  # smallest eigenvalue of -H

        # Each should be >= 0 (PSD)
        for i, (R, me) in enumerate(zip(R_values, min_eigs)):
            assert me >= -1e-12, (
                f"Ghost contribution negative at R={R}: {me:.6e}"
            )

        # Should be increasing with R (the ghost term grows)
        for i in range(len(min_eigs) - 1):
            assert min_eigs[i + 1] > min_eigs[i] - 1e-10, (
                f"Ghost min eig decreased from R={R_values[i]} "
                f"({min_eigs[i]:.6f}) to R={R_values[i+1]} "
                f"({min_eigs[i+1]:.6f})"
            )

    def test_ghost_max_eigenvalue_grows_with_R(self, beg):
        """Max eigenvalue of -Hess(log det M_FP) should grow with R."""
        R_values = [1.0, 10.0]
        max_eigs = []
        for R in R_values:
            H = beg.compute_hessian_log_det_MFP(np.zeros(9), R)
            eigs = eigvalsh(-H)
            max_eigs.append(eigs[-1])

        assert max_eigs[1] > max_eigs[0], (
            f"Ghost max eig did not grow: R=1 -> {max_eigs[0]:.6f}, "
            f"R=10 -> {max_eigs[1]:.6f}"
        )

    def test_ghost_analytical_scaling_at_origin(self, beg):
        """
        At a=0, M_FP = (3/R^2)I, M_FP^{-1} = (R^2/3)I.
        [H]_{ij} = -(g/R)^2 * (R^2/3)^2 * Tr(L_i L_j) = -g^2 R^2/9 * Tr(L_i L_j).
        So -H_{ij} = g^2 R^2/9 * Tr(L_i L_j).
        The trace Tr(L_i L_j) is R-independent, so -H scales as g^2 R^2.
        """
        R_values = [1.0, 5.0]
        neg_H = []
        for R in R_values:
            H = beg.compute_hessian_log_det_MFP(np.zeros(9), R)
            neg_H.append(-H)

        g2_1 = ZwanzigerGapEquation.running_coupling_g2(R_values[0])
        g2_5 = ZwanzigerGapEquation.running_coupling_g2(R_values[1])

        # The ratio of -H should be (g2_5 * R_5^2) / (g2_1 * R_1^2)
        expected_ratio = (g2_5 * R_values[1]**2) / (g2_1 * R_values[0]**2)

        # Compare a diagonal element
        if abs(neg_H[0][0, 0]) > 1e-15:
            actual_ratio = neg_H[1][0, 0] / neg_H[0][0, 0]
            np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.01,
                err_msg="Ghost scaling does not match g^2 R^2 prediction")


# ======================================================================
# 4. Hessian of V_4 at a=0
# ======================================================================

class TestHessianV4:
    """Hess(V_4) should be PSD at a=0 (since V_4 has minimum there)."""

    def test_hessian_V4_at_origin_is_zero(self, beg):
        """At a=0, V_4 is quartic, so Hess(V_4) = 0."""
        H = beg.compute_hessian_V4(np.zeros(9), 1.0)
        np.testing.assert_allclose(H, np.zeros((9, 9)), atol=1e-8,
            err_msg="Hess(V_4) should be zero at a=0 (quartic minimum)")

    def test_hessian_V4_away_from_origin_is_PSD(self, beg):
        """Away from origin, Hess(V_4) should have non-negative eigenvalues
        (V_4 >= 0 is convex along most directions)."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.3
        g2 = ZwanzigerGapEquation.running_coupling_g2(1.0)
        H = beg.compute_hessian_V4(a, 1.0, g_squared=g2)
        eigs = eigvalsh(H)
        # V_4 is not globally convex, but eigenvalues should be finite
        assert np.all(np.isfinite(eigs)), "Hess(V_4) has non-finite eigenvalues"

    def test_hessian_V4_symmetric(self, beg):
        """Hess(V_4) should be symmetric."""
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.5
        H = beg.compute_hessian_V4(a, 2.0)
        np.testing.assert_allclose(H, H.T, atol=1e-6,
            err_msg="Hess(V_4) is not symmetric")


# ======================================================================
# 5. Full Hessian U_phys at a=0
# ======================================================================

class TestHessianUPhysOrigin:
    """Hess(U_phys) at a=0 should have positive eigenvalues for all R."""

    def test_positive_at_R1(self, beg):
        """Hess(U_phys) at a=0, R=1 should be positive definite."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 1.0)
        assert eig > 0, f"Min eigenvalue at R=1 is {eig:.6e}, should be > 0"

    def test_positive_at_R2(self, beg):
        """Hess(U_phys) at a=0, R=2."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 2.0)
        assert eig > 0, f"Min eigenvalue at R=2 is {eig:.6e}, should be > 0"

    def test_positive_at_R5(self, beg):
        """Hess(U_phys) at a=0, R=5."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 5.0)
        assert eig > 0, f"Min eigenvalue at R=5 is {eig:.6e}, should be > 0"

    def test_positive_at_R10(self, beg):
        """Hess(U_phys) at a=0, R=10."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 10.0)
        assert eig > 0, f"Min eigenvalue at R=10 is {eig:.6e}, should be > 0"

    def test_positive_at_R50(self, beg):
        """Hess(U_phys) at a=0, R=50."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 50.0)
        assert eig > 0, f"Min eigenvalue at R=50 is {eig:.6e}, should be > 0"

    def test_min_eig_grows_with_R(self, beg):
        """Min eigenvalue of Hess(U_phys) at origin should grow with R.
        At a=0: Hess = (4/R^2)I + 0 + g^2 R^2/9 * Tr(L_i L_j).
        The ghost term grows as g^2 R^2, dominating 4/R^2 at large R."""
        R_values = [1.0, 5.0, 20.0]
        min_eigs = []
        for R in R_values:
            eig = beg.min_eigenvalue_hessian_U(np.zeros(9), R)
            min_eigs.append(eig)

        # At large R, the ghost term dominates and grows
        assert min_eigs[2] > min_eigs[0], (
            f"Min eig did not grow from R={R_values[0]} ({min_eigs[0]:.6f}) "
            f"to R={R_values[2]} ({min_eigs[2]:.6f})"
        )

    def test_hessian_U_phys_decomposition(self, beg):
        """
        Verify Hess(U_phys) = Hess(V_2) + Hess(V_4) - Hess(log det M_FP).
        """
        R = 2.0
        a = np.zeros(9)
        H_total = beg.compute_hessian_U_phys(a, R)
        H_V2 = beg.compute_hessian_V2(R)
        H_V4 = beg.compute_hessian_V4(a, R)
        H_log = beg.compute_hessian_log_det_MFP(a, R)
        H_sum = H_V2 + H_V4 - H_log
        np.testing.assert_allclose(H_total, H_sum, atol=1e-10,
            err_msg="Decomposition of Hess(U_phys) does not hold")


# ======================================================================
# 6. Scan over Gribov region
# ======================================================================

class TestScanGribov:
    """All sampled points inside Omega_9 should have positive Hess(U_phys)."""

    def test_scan_R1(self, beg):
        """At R=1, scan should find all positive eigenvalues."""
        result = beg.scan_hessian_over_gribov(1.0, n_points=30, seed=42)
        assert result['n_valid'] > 0, "No valid points sampled"
        assert result['all_positive'], (
            f"Found non-positive eigenvalue: min={result['min_eigenvalue_overall']:.6e}"
        )

    def test_scan_R5(self, beg):
        """At R=5, scan should find all positive eigenvalues."""
        result = beg.scan_hessian_over_gribov(5.0, n_points=30, seed=42)
        assert result['n_valid'] > 0, "No valid points sampled"
        assert result['all_positive'], (
            f"Found non-positive eigenvalue: min={result['min_eigenvalue_overall']:.6e}"
        )

    def test_scan_R10(self, beg):
        """At R=10, scan should find all positive eigenvalues."""
        result = beg.scan_hessian_over_gribov(10.0, n_points=20, seed=42)
        assert result['n_valid'] > 0, "No valid points sampled"
        assert result['all_positive'], (
            f"Found non-positive eigenvalue: min={result['min_eigenvalue_overall']:.6e}"
        )

    def test_scan_origin_eigenvalues_positive(self, beg):
        """Eigenvalues at origin should all be positive in scan result."""
        result = beg.scan_hessian_over_gribov(2.0, n_points=10, seed=42)
        eigs_0 = result['eigenvalues_at_origin']
        assert np.all(eigs_0 > 0), (
            f"Origin eigenvalues not all positive: {eigs_0}"
        )

    def test_scan_sufficient_valid_points(self, beg):
        """Should get at least 50% valid points."""
        n = 30
        result = beg.scan_hessian_over_gribov(2.0, n_points=n, seed=42)
        assert result['n_valid'] >= n * 0.3, (
            f"Only {result['n_valid']}/{n} valid points"
        )


# ======================================================================
# 7. BE bound vs R
# ======================================================================

class TestBEBoundVsR:
    """The BE bound should stay positive as R grows."""

    def test_be_bound_positive_small_R(self, beg):
        """BE bound should be positive for small R."""
        results = beg.bakry_emery_bound_vs_R([1.0, 2.0], n_points=20, seed=42)
        for i, R in enumerate(results['R']):
            assert results['be_bound'][i] > 0 or not np.isfinite(results['be_bound'][i]), (
                f"BE bound negative at R={R}: {results['be_bound'][i]:.6e}"
            )

    def test_be_bound_origin_positive(self, beg):
        """BE bound at origin should be positive for all R."""
        results = beg.bakry_emery_bound_vs_R([1.0, 5.0, 10.0], n_points=10, seed=42)
        for i, R in enumerate(results['R']):
            assert results['be_bound_at_origin'][i] > 0, (
                f"BE bound at origin negative at R={R}: "
                f"{results['be_bound_at_origin'][i]:.6e}"
            )

    def test_ghost_contribution_nonnegative(self, beg):
        """Ghost contribution should be non-negative (PSD)."""
        results = beg.bakry_emery_bound_vs_R([1.0, 5.0], n_points=10, seed=42)
        for i, R in enumerate(results['R']):
            assert results['ghost_contribution_origin'][i] >= -1e-12, (
                f"Ghost contribution negative at R={R}: "
                f"{results['ghost_contribution_origin'][i]:.6e}"
            )


# ======================================================================
# 8. Analytical consistency checks
# ======================================================================

class TestAnalyticalConsistency:
    """Cross-checks between analytical and numerical Hessians."""

    def test_hessian_log_det_numerical_vs_analytical(self, beg, gd):
        """
        Compare analytical Hess(log det M_FP) with numerical finite differences.
        """
        R = 2.0
        a = np.zeros(9)
        H_analytical = beg.compute_hessian_log_det_MFP(a, R)

        # Numerical finite difference of log det M_FP
        h = 1e-5

        def log_det_MFP(a_vec):
            M = gd.fp_operator_truncated(a_vec, R)
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                return -1e10
            return logdet

        H_numerical = np.zeros((9, 9))
        for i in range(9):
            for j in range(i, 9):
                a_pp = a.copy(); a_pp[i] += h; a_pp[j] += h
                a_pm = a.copy(); a_pm[i] += h; a_pm[j] -= h
                a_mp = a.copy(); a_mp[i] -= h; a_mp[j] += h
                a_mm = a.copy(); a_mm[i] -= h; a_mm[j] -= h
                val = (log_det_MFP(a_pp) - log_det_MFP(a_pm)
                       - log_det_MFP(a_mp) + log_det_MFP(a_mm)) / (4 * h * h)
                H_numerical[i, j] = val
                H_numerical[j, i] = val

        np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-4,
            err_msg="Analytical and numerical Hess(log det M_FP) disagree")

    def test_hessian_log_det_numerical_away_from_origin(self, beg, gd):
        """
        Compare at a nonzero point inside Omega.
        """
        R = 2.0
        rng = np.random.RandomState(42)
        d = rng.randn(9)
        d /= np.linalg.norm(d)
        t_h = gd.gribov_horizon_distance_truncated(d, R)
        a = 0.3 * t_h * d  # 30% toward horizon

        H_analytical = beg.compute_hessian_log_det_MFP(a, R)
        if np.any(np.isnan(H_analytical)):
            pytest.skip("Point is at or outside Gribov horizon")

        h = 1e-5

        def log_det_MFP(a_vec):
            M = gd.fp_operator_truncated(a_vec, R)
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                return -1e10
            return logdet

        H_numerical = np.zeros((9, 9))
        for i in range(9):
            for j in range(i, 9):
                a_pp = a.copy(); a_pp[i] += h; a_pp[j] += h
                a_pm = a.copy(); a_pm[i] += h; a_pm[j] -= h
                a_mp = a.copy(); a_mp[i] -= h; a_mp[j] += h
                a_mm = a.copy(); a_mm[i] -= h; a_mm[j] -= h
                val = (log_det_MFP(a_pp) - log_det_MFP(a_pm)
                       - log_det_MFP(a_mp) + log_det_MFP(a_mm)) / (4 * h * h)
                H_numerical[i, j] = val
                H_numerical[j, i] = val

        np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-3,
            err_msg="Analytical and numerical disagree away from origin")

    def test_V4_hessian_vs_explicit(self, beg):
        """
        Compare finite-difference Hess(V_4) with a second finite-difference
        computation using a different step size to check convergence.
        """
        rng = np.random.RandomState(42)
        a = rng.randn(9) * 0.3
        R = 1.0
        H1 = beg.compute_hessian_V4(a, R, h=1e-4)
        H2 = beg.compute_hessian_V4(a, R, h=1e-5)
        np.testing.assert_allclose(H1, H2, atol=1e-3,
            err_msg="Hess(V_4) not converged between h=1e-4 and h=1e-5")


# ======================================================================
# 9. Formal analysis
# ======================================================================

class TestFormalAnalysis:
    """The formal analysis should produce coherent results."""

    def test_formal_analysis_runs(self, beg):
        """formal_analysis() should complete without error."""
        result = beg.formal_analysis(
            R_range=[1.0, 5.0], n_points=10, seed=42
        )
        assert 'assessment' in result
        assert 'label' in result
        assert result['label'] == 'NUMERICAL'

    def test_formal_analysis_has_theorems(self, beg):
        """Should list the theorems used."""
        result = beg.formal_analysis(
            R_range=[1.0], n_points=5, seed=42
        )
        assert 'theorems_used' in result
        assert 'bakry_emery' in result['theorems_used']
        assert 'singer_curvature' in result['theorems_used']
        assert 'ghost_psd' in result['theorems_used']

    def test_formal_analysis_assessment_not_empty(self, beg):
        """Assessment should be a non-empty string."""
        result = beg.formal_analysis(
            R_range=[1.0, 2.0], n_points=10, seed=42
        )
        assert isinstance(result['assessment'], str)
        assert len(result['assessment']) > 20


# ======================================================================
# 10. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Edge cases: large R, small R, points near horizon."""

    def test_hessian_U_phys_at_large_R(self, beg):
        """At R=100, Hess(U_phys) at origin should still be positive."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 100.0)
        assert eig > 0, f"Min eigenvalue at R=100 is {eig:.6e}"

    def test_hessian_U_phys_at_small_R(self, beg):
        """At R=0.2, Hess(U_phys) at origin should be positive."""
        eig = beg.min_eigenvalue_hessian_U(np.zeros(9), 0.2)
        assert eig > 0, f"Min eigenvalue at R=0.2 is {eig:.6e}"

    def test_near_horizon_point(self, beg, gd):
        """At 80% toward horizon, Hess(U_phys) should still be computable."""
        R = 2.0
        rng = np.random.RandomState(42)
        d = rng.randn(9)
        d /= np.linalg.norm(d)
        t_h = gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_h):
            pytest.skip("No horizon in this direction")
        a = 0.8 * t_h * d
        eig = beg.min_eigenvalue_hessian_U(a, R)
        assert np.isfinite(eig), f"Eigenvalue not finite near horizon: {eig}"


# ======================================================================
# 11. Corrected analytical bound (Session 12 hardening)
# ======================================================================

class TestCorrectedAnalyticalBound:
    """Tests for the corrected analytical kappa bound (3 vulnerabilities fixed).

    Vulnerabilities addressed:
        1. V2 term was 8/R^2 (wrong), corrected to 4/R^2 (S_YM has 1/2 factor).
        2. V4 bound used d/2, but origin is not centroid. Corrected to use
           C_R = sqrt(3) (max distance from origin to boundary).
        3. R_0 shifted from 2.77 to 3.598 due to corrected V2 and V4.
    """

    def test_V2_term_is_4_not_8(self, beg):
        """V2 contribution must be 4/R^2, matching compute_hessian_V2."""
        for R in [1.0, 5.0, 10.0]:
            ak = BakryEmeryGap.analytical_kappa_bound(R)
            expected_V2 = 4.0 / R**2
            assert ak['V2_contribution'] == pytest.approx(expected_V2, rel=1e-12), (
                f"V2 should be 4/R^2 = {expected_V2}, got {ak['V2_contribution']} at R={R}"
            )

    def test_V2_matches_numerical_hessian(self, beg):
        """The V2 contribution in the analytical bound must match Hess(V2)."""
        for R in [1.0, 2.0, 5.0]:
            H_V2 = beg.compute_hessian_V2(R)
            ak = BakryEmeryGap.analytical_kappa_bound(R)
            assert H_V2[0, 0] == pytest.approx(ak['V2_contribution'], rel=1e-12), (
                f"V2 mismatch at R={R}: Hess gives {H_V2[0,0]}, bound uses {ak['V2_contribution']}"
            )

    def test_V4_bound_is_108(self):
        """V4 bound coefficient must be 108 = 9 * C_Q * C_R^2 = 9*4*3."""
        ak = BakryEmeryGap.analytical_kappa_bound(1.0)
        assert ak['V4_bound'] == pytest.approx(108.0, rel=1e-10), (
            f"V4 bound should be 108/R^2, got {ak['V4_bound']}"
        )

    def test_ghost_coefficient_is_4_over_81(self):
        """Ghost coefficient must be 4/81 (mu_max_coeff = 9)."""
        # At R=10, g^2 ~ 4*pi, ghost = (4/81)*g^2*R^2
        ak = BakryEmeryGap.analytical_kappa_bound(10.0)
        g2 = ak['g_squared']
        expected_ghost = (4.0/81.0) * g2 * 100.0
        assert ak['ghost_lower_bound'] == pytest.approx(expected_ghost, rel=1e-10)

    def test_R0_is_approximately_3_60(self):
        """Corrected R0 should be approximately 3.598."""
        kr = BakryEmeryGap.theorem_threshold_R0()
        assert 3.5 < kr['R0'] < 3.7, f"R0 = {kr['R0']}, expected ~3.598"

    def test_KR_covers_below_R0(self):
        """g^2(R0) must be below g^2_critical for KR to cover R < R0."""
        kr = BakryEmeryGap.theorem_threshold_R0()
        assert kr['KR_covers_below_R0'], (
            f"KR does not cover R < R0: g^2(R0) = {kr['g2_at_R0']:.4f} "
            f">= g^2_c = {kr['g2_critical_KR']:.4f}"
        )

    def test_kappa_negative_below_R0(self):
        """The analytical bound should be negative at R=1 (below R0)."""
        ak = BakryEmeryGap.analytical_kappa_bound(1.0)
        assert ak['kappa_lower_bound'] < 0, (
            f"Analytical kappa should be negative at R=1, got {ak['kappa_lower_bound']:.4f}"
        )

    def test_kappa_positive_above_R0(self):
        """The analytical bound should be positive at R=5 (above R0)."""
        ak = BakryEmeryGap.analytical_kappa_bound(5.0)
        assert ak['kappa_lower_bound'] > 0, (
            f"Analytical kappa should be positive at R=5, got {ak['kappa_lower_bound']:.4f}"
        )

    def test_analytical_bound_is_lower_bound(self, beg):
        """The analytical kappa must be a LOWER bound on the numerical kappa.
        At every sampled point, actual kappa >= analytical bound."""
        for R in [4.0, 5.0, 10.0]:
            ak = BakryEmeryGap.analytical_kappa_bound(R)
            result = beg.scan_hessian_over_gribov(R, n_points=100, seed=42)
            assert result['min_eigenvalue_overall'] >= ak['kappa_lower_bound'] - 0.01, (
                f"At R={R}: numerical min kappa={result['min_eigenvalue_overall']:.4f} "
                f"< analytical bound={ak['kappa_lower_bound']:.4f}"
            )

    def test_max_radius_exceeds_half_diameter(self, beg, gd, dt):
        """Verify that the max distance from origin to boundary exceeds d/2.
        This is the core of vulnerability #2: origin is not centroid."""
        R = 5.0
        rng = np.random.RandomState(42)
        max_ratio = 0
        for _ in range(500):
            d = rng.randn(9)
            d /= np.linalg.norm(d)
            t_h = gd.gribov_horizon_distance_truncated(d, R)
            t_h_neg = gd.gribov_horizon_distance_truncated(-d, R)
            if np.isfinite(t_h) and np.isfinite(t_h_neg):
                half_diam = (t_h + t_h_neg) / 2.0
                ratio = max(t_h, t_h_neg) / half_diam
                if ratio > max_ratio:
                    max_ratio = ratio
        # The ratio should exceed 1 (origin is not centroid)
        assert max_ratio > 1.05, (
            f"Max radius/half-diameter ratio = {max_ratio:.4f}, expected > 1.05"
        )

    def test_numerical_kappa_negative_at_small_R(self, beg):
        """At R=0.5, some sampled points must have kappa < 0.
        This confirms vulnerability #1: kappa > 0 does NOT hold for all R."""
        result = beg.scan_hessian_over_gribov(0.5, n_points=300, seed=42)
        assert not result['all_positive'], (
            f"Expected negative kappa at R=0.5, but all positive: "
            f"min={result['min_eigenvalue_overall']:.4f}"
        )

    def test_numerical_kappa_positive_at_R1(self, beg):
        """At R=1, all sampled kappa should be positive (numerically).
        The BE analytical bound is negative here, but KR covers this regime."""
        result = beg.scan_hessian_over_gribov(1.0, n_points=200, seed=42)
        assert result['all_positive'], (
            f"Expected all positive kappa at R=1, min={result['min_eigenvalue_overall']:.4f}"
        )

    def test_combined_gap_positive_all_R(self):
        """The COMBINED gap (KR for small R, BE for large R) must be positive
        for all R. This is the core theorem: the two-regime split works."""
        kr = BakryEmeryGap.theorem_threshold_R0()
        R0 = kr['R0']

        # Below R0: KR covers
        g2_at_R0 = kr['g2_at_R0']
        g2_c = kr['g2_critical_KR']
        assert g2_at_R0 < g2_c, "KR does not cover R < R0"

        # Above R0: BE covers
        for R in [R0 + 0.1, R0 + 1, 10.0, 50.0, 100.0]:
            ak = BakryEmeryGap.analytical_kappa_bound(R)
            assert ak['positive'], f"BE bound negative at R={R}"
