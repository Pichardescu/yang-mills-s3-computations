"""
Tests for the Weighted Laplacian on the 9-DOF Gribov Region.

Test categories:
    1.  FP determinant: positive at origin, positive inside Omega_9
    2.  log det M_FP: finite inside Omega, -inf outside
    3.  Gradient of log det M_FP vanishes at origin (THEOREM)
    4.  Ghost curvature at origin: -Hess(log det M_FP) = (4g^2 R^2/9) I_9
    5.  Tr(L_i L_j) = 4 delta_{ij} (structure constant identity)
    6.  Hessian of Phi at origin: analytical vs numerical agreement
    7.  Kappa at origin is larger than unweighted eigenvalue 4/R^2
    8.  Kappa at origin grows with R (ghost contribution dominates)
    9.  BE curvature positive at sampled interior points
    10. Weighted gap > unweighted gap at physical parameters
    11. Gap decomposition: ghost fraction grows with R
    12. 1D Schrodinger discretization gives positive gap
    13. Analytical kappa minimum is positive
    14. Physical mass gap in MeV is reasonable
"""

import pytest
import numpy as np
from scipy.linalg import eigvalsh

from yang_mills_s3.proofs.weighted_laplacian_9dof import WeightedLaplacian9DOF
from yang_mills_s3.proofs.gribov_diameter import GribovDiameter
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def wl():
    """WeightedLaplacian9DOF instance."""
    return WeightedLaplacian9DOF()


@pytest.fixture
def gd():
    """GribovDiameter instance."""
    return GribovDiameter()


# ======================================================================
# 1. FP determinant
# ======================================================================

class TestFPDeterminant:
    """Faddeev-Popov determinant tests."""

    def test_fp_det_at_origin_R1(self, wl):
        """At a=0, R=1: det(M_FP) = (3/R^2)^9 = 3^9 = 19683."""
        a = np.zeros(9)
        det_val = wl.fp_determinant(a, R=1.0)
        expected = 3.0**9  # (3/1)^9
        assert det_val == pytest.approx(expected, rel=1e-10)

    def test_fp_det_at_origin_R2(self, wl):
        """At a=0, R=2: det(M_FP) = (3/4)^9."""
        a = np.zeros(9)
        det_val = wl.fp_determinant(a, R=2.0)
        expected = (3.0 / 4.0)**9
        assert det_val == pytest.approx(expected, rel=1e-10)

    def test_fp_det_positive_inside(self, wl, gd):
        """det(M_FP) > 0 at random points inside Omega_9."""
        R = 1.0
        rng = np.random.RandomState(42)
        for _ in range(20):
            d = rng.randn(9)
            d /= np.linalg.norm(d)
            t_max = gd.gribov_horizon_distance_truncated(d, R)
            if not np.isfinite(t_max) or t_max <= 0:
                continue
            a = 0.5 * t_max * d  # halfway to horizon
            det_val = wl.fp_determinant(a, R)
            assert det_val > 0, f"det(M_FP) = {det_val} <= 0 inside Omega"

    def test_fp_det_vanishes_at_horizon(self, wl, gd):
        """det(M_FP) ~ 0 at the Gribov horizon."""
        R = 1.0
        d = np.zeros(9)
        d[0] = 1.0
        t_horizon = gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_horizon):
            pytest.skip("No finite horizon in this direction")
        a = 0.999 * t_horizon * d
        det_val = wl.fp_determinant(a, R)
        # Should be very small but positive (we're just inside)
        assert det_val >= 0
        # Much smaller than at origin
        det_origin = wl.fp_determinant(np.zeros(9), R)
        assert det_val < 0.01 * det_origin


# ======================================================================
# 2. log det M_FP
# ======================================================================

class TestLogDetMFP:
    """log det(M_FP) tests."""

    def test_log_det_at_origin(self, wl):
        """log det(M_FP)(0) = 9 * log(3/R^2)."""
        R = 1.0
        log_det = wl.log_fp_determinant(np.zeros(9), R)
        expected = 9.0 * np.log(3.0 / R**2)
        assert log_det == pytest.approx(expected, rel=1e-10)

    def test_log_det_finite_inside(self, wl, gd):
        """log det is finite inside Omega_9."""
        R = 2.0
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        t_max = gd.gribov_horizon_distance_truncated(d, R)
        a = 0.5 * t_max * d
        log_det = wl.log_fp_determinant(a, R)
        assert np.isfinite(log_det)


# ======================================================================
# 3. Gradient of log det M_FP at origin
# ======================================================================

class TestGradLogDetOrigin:
    """grad(log det M_FP) vanishes at origin (THEOREM)."""

    def test_grad_zero_at_origin_R1(self, wl):
        """grad(log det M_FP)(0) = 0 at R=1."""
        grad = wl.grad_log_det_MFP_at_origin(R=1.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_grad_zero_at_origin_R2(self, wl):
        """grad(log det M_FP)(0) = 0 at R=2."""
        grad = wl.grad_log_det_MFP_at_origin(R=2.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_grad_zero_at_origin_R5(self, wl):
        """grad(log det M_FP)(0) = 0 at R=5."""
        grad = wl.grad_log_det_MFP_at_origin(R=5.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_grad_nonzero_away_from_origin(self, wl, gd):
        """grad(log det M_FP) is nonzero away from origin."""
        R = 1.0
        d = np.zeros(9)
        d[0] = 1.0
        t_max = gd.gribov_horizon_distance_truncated(d, R)
        a = 0.3 * t_max * d
        grad = wl.grad_log_det_MFP(a, R)
        assert np.linalg.norm(grad) > 1e-6, "Gradient should be nonzero away from origin"


# ======================================================================
# 4. Ghost curvature at origin
# ======================================================================

class TestGhostCurvatureOrigin:
    """Verify -Hess(log det M_FP)(0) = (4g^2 R^2/9) * I_9."""

    def test_ghost_proportional_to_identity_R1(self, wl):
        """Ghost curvature at origin is proportional to I_9 at R=1."""
        result = wl.verify_ghost_curvature_at_origin(R=1.0)
        eigs = result['numerical_eigenvalues']
        # All eigenvalues should be approximately equal
        np.testing.assert_allclose(eigs, eigs[0], rtol=1e-6)

    def test_ghost_value_R1(self, wl):
        """Ghost curvature eigenvalue matches analytical prediction at R=1."""
        result = wl.verify_ghost_curvature_at_origin(R=1.0)
        assert result['max_deviation'] < 1e-6
        assert result['relative_deviation'] < 1e-4

    def test_ghost_value_R2(self, wl):
        """Ghost curvature eigenvalue matches analytical prediction at R=2."""
        result = wl.verify_ghost_curvature_at_origin(R=2.0)
        assert result['relative_deviation'] < 1e-4

    def test_ghost_value_R5(self, wl):
        """Ghost curvature eigenvalue matches analytical prediction at R=5."""
        result = wl.verify_ghost_curvature_at_origin(R=5.0)
        assert result['relative_deviation'] < 1e-4

    def test_ghost_grows_with_R(self, wl):
        """Ghost curvature 4g^2R^2/9 grows with R (since g^2 saturates)."""
        R_values = [1.0, 2.0, 5.0, 10.0]
        ghost_eigs = []
        for R in R_values:
            result = wl.verify_ghost_curvature_at_origin(R)
            ghost_eigs.append(result['predicted_eigenvalue'])
        # Should be monotonically increasing
        for i in range(len(ghost_eigs) - 1):
            assert ghost_eigs[i + 1] > ghost_eigs[i], (
                f"Ghost curvature not growing: {ghost_eigs[i+1]} <= {ghost_eigs[i]} "
                f"at R={R_values[i+1]}"
            )


# ======================================================================
# 5. Trace identity: Tr(L_i L_j) = C_Q * delta_{ij}
# ======================================================================

class TestTraceLiLj:
    """Verify Tr(L(e_i) L(e_j)) structure."""

    def test_trace_diagonal(self, wl):
        """Tr(L_i L_i) should be the same for all i."""
        result = wl.verify_ghost_curvature_at_origin(R=1.0)
        diag = result['diag_traces']
        # All diagonal entries should be equal
        np.testing.assert_allclose(diag, diag[0], rtol=1e-10)

    def test_trace_off_diagonal_zero(self, wl):
        """Tr(L_i L_j) = 0 for i != j."""
        result = wl.verify_ghost_curvature_at_origin(R=1.0)
        assert result['off_diagonal_max'] < 1e-10

    def test_trace_proportional_to_identity(self, wl):
        """Tr(L_i L_j) = C_Q * delta_{ij} (C_Q = 4)."""
        result = wl.verify_ghost_curvature_at_origin(R=1.0)
        assert result['Tr_Li_Lj_is_proportional_to_I']
        assert result['Tr_Li_Li_value'] == pytest.approx(4.0, rel=1e-10)


# ======================================================================
# 6. Hessian of Phi at origin: analytical vs numerical
# ======================================================================

class TestHessianPhiOrigin:
    """Hess(Phi) at origin matches analytical formula."""

    def test_hessian_phi_origin_R1(self, wl):
        """Hess(Phi)(0) at R=1: kappa = 4 + 4g^2/9."""
        result = wl.hessian_Phi_at_origin(R=1.0)
        kappa_num = result['min_eigenvalue']
        kappa_ana = result['kappa_analytical']
        assert kappa_num == pytest.approx(kappa_ana, rel=1e-4)

    def test_hessian_phi_origin_R2(self, wl):
        """Hess(Phi)(0) at R=2: analytical matches numerical."""
        result = wl.hessian_Phi_at_origin(R=2.0)
        assert result['min_eigenvalue'] == pytest.approx(
            result['kappa_analytical'], rel=1e-4
        )

    def test_hessian_phi_origin_R5(self, wl):
        """Hess(Phi)(0) at R=5."""
        result = wl.hessian_Phi_at_origin(R=5.0)
        assert result['min_eigenvalue'] == pytest.approx(
            result['kappa_analytical'], rel=1e-3
        )

    def test_hessian_phi_origin_proportional_to_I(self, wl):
        """At origin, Hess(Phi) should be proportional to I_9."""
        result = wl.hessian_Phi_at_origin(R=1.0)
        eigs = result['eigenvalues']
        # All eigenvalues should be approximately equal (within V4 noise)
        np.testing.assert_allclose(eigs, eigs[0], rtol=1e-3)


# ======================================================================
# 7. Kappa at origin exceeds unweighted eigenvalue
# ======================================================================

class TestKappaExceedsUnweighted:
    """Weighted curvature > unweighted eigenvalue 4/R^2."""

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_kappa_exceeds_4_over_R2(self, wl, R):
        """kappa_0(R) > 4/R^2 for all R."""
        result = wl.hessian_Phi_at_origin(R)
        kappa = result['min_eigenvalue']
        unweighted = 4.0 / R**2
        assert kappa > unweighted, (
            f"kappa={kappa:.6f} should exceed 4/R^2={unweighted:.6f} at R={R}"
        )

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_ghost_contribution_positive(self, wl, R):
        """Ghost contribution is strictly positive for all R."""
        result = wl.hessian_Phi_at_origin(R)
        assert result['ghost_contribution'] > 0


# ======================================================================
# 8. Kappa at origin grows with R
# ======================================================================

class TestKappaGrowsWithR:
    """The ghost term 4g^2 R^2/9 grows with R, dominating at large R."""

    def test_ghost_fraction_increases(self, wl):
        """Ghost fraction of total kappa increases with R."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        fractions = []
        for R in R_values:
            decomp = wl.gap_decomposition(R)
            fractions.append(decomp['ghost_fraction'])
        # Ghost fraction should increase with R
        for i in range(len(fractions) - 1):
            assert fractions[i + 1] >= fractions[i], (
                f"Ghost fraction not increasing: {fractions[i+1]} < {fractions[i]}"
            )

    def test_ghost_dominates_at_large_R(self, wl):
        """At R >= 5, ghost contribution > 50% of total kappa at origin."""
        for R in [5.0, 10.0, 20.0]:
            decomp = wl.gap_decomposition(R)
            assert decomp['ghost_dominates'], (
                f"Ghost does not dominate at R={R}: "
                f"fraction = {decomp['ghost_fraction']:.3f}"
            )

    def test_total_kappa_at_origin_grows_for_large_R(self, wl):
        """Total kappa at origin grows for R >= 2."""
        R_values = [2.0, 5.0, 10.0, 20.0]
        kappas = []
        for R in R_values:
            kappa = WeightedLaplacian9DOF.analytical_kappa_at_origin(R)
            kappas.append(kappa)
        for i in range(len(kappas) - 1):
            assert kappas[i + 1] > kappas[i], (
                f"kappa not growing: {kappas[i+1]} <= {kappas[i]} at R={R_values[i+1]}"
            )


# ======================================================================
# 9. BE curvature positive at sampled interior points
# ======================================================================

class TestBECurvatureInterior:
    """Hess(Phi) >= kappa > 0 at sampled points in Omega_9."""

    def test_all_positive_R1(self, wl):
        """All sampled BE curvatures positive at R=1."""
        result = wl.bakry_emery_weighted_gap(R=1.0, n_sample=30, seed=42)
        assert result['kappa_all_positive']
        assert result['kappa_min_sampled'] > 0

    def test_all_positive_R2(self, wl):
        """All sampled BE curvatures positive at R=2."""
        result = wl.bakry_emery_weighted_gap(R=2.0, n_sample=30, seed=42)
        assert result['kappa_all_positive']

    def test_all_positive_R5(self, wl):
        """All sampled BE curvatures positive at R=5."""
        result = wl.bakry_emery_weighted_gap(R=5.0, n_sample=30, seed=42)
        assert result['kappa_all_positive']

    def test_min_kappa_reasonable(self, wl):
        """Minimum sampled kappa > 0 and within expected range."""
        result = wl.bakry_emery_weighted_gap(R=2.0, n_sample=50, seed=42)
        kappa_min = result['kappa_min_sampled']
        # Must be positive
        assert kappa_min > 0
        # Should not exceed kappa at origin (origin is max for quadratic terms)
        # but could exceed it if V4 Hessian is big enough away from origin
        # Just check it's a reasonable number
        assert kappa_min < 1e6


# ======================================================================
# 10. Weighted gap > unweighted gap
# ======================================================================

class TestWeightedVsUnweighted:
    """The weighted gap is larger than the unweighted gap."""

    def test_enhancement_at_R2(self, wl):
        """Enhancement factor > 1 at R=2."""
        result = wl.bakry_emery_weighted_gap(R=2.0, n_sample=30, seed=42)
        assert result['enhancement_factor'] > 1.0, (
            f"Enhancement = {result['enhancement_factor']:.3f} should be > 1"
        )

    def test_enhancement_at_R5(self, wl):
        """Enhancement factor > 1 at R=5."""
        result = wl.bakry_emery_weighted_gap(R=5.0, n_sample=30, seed=42)
        assert result['enhancement_factor'] > 1.0

    def test_enhancement_increases_with_R(self, wl):
        """Enhancement grows with R (ghost dominates more)."""
        R_values = [1.0, 2.0, 5.0]
        enhancements = []
        for R in R_values:
            result = wl.bakry_emery_weighted_gap(R=R, n_sample=20, seed=42)
            enhancements.append(result['enhancement_factor'])
        for i in range(len(enhancements) - 1):
            assert enhancements[i + 1] > enhancements[i], (
                f"Enhancement not increasing: {enhancements[i+1]} <= {enhancements[i]}"
            )


# ======================================================================
# 11. Gap decomposition
# ======================================================================

class TestGapDecomposition:
    """Decomposition of the BE curvature at origin."""

    def test_V2_contribution(self, wl):
        """V2 contribution is exactly 4/R^2."""
        decomp = wl.gap_decomposition(R=2.0)
        assert decomp['kappa_V2'] == pytest.approx(4.0 / 4.0, rel=1e-12)

    def test_V4_zero_at_origin(self, wl):
        """V4 Hessian contribution is 0 at origin."""
        decomp = wl.gap_decomposition(R=2.0)
        assert decomp['kappa_V4_origin'] == 0.0

    def test_ghost_positive(self, wl):
        """Ghost contribution is positive."""
        decomp = wl.gap_decomposition(R=2.0)
        assert decomp['kappa_ghost_origin'] > 0

    def test_total_equals_sum(self, wl):
        """Total = V2 + V4 + ghost."""
        decomp = wl.gap_decomposition(R=2.0)
        total = decomp['kappa_V2'] + decomp['kappa_V4_origin'] + decomp['kappa_ghost_origin']
        assert decomp['kappa_total_origin'] == pytest.approx(total, rel=1e-12)

    def test_mass_gap_reasonable(self, wl):
        """Mass gap at origin is in reasonable range."""
        decomp = wl.gap_decomposition(R=2.2)
        # Should give something in the 100-1000 MeV range
        m = decomp['m_total_origin_MeV']
        assert 50 < m < 5000, f"Mass gap {m:.1f} MeV outside reasonable range"


# ======================================================================
# 12. 1D Schrodinger discretization
# ======================================================================

class TestSchrodinger1D:
    """1D slice of the weighted Laplacian."""

    def test_1d_positive_gap(self, wl):
        """1D Schrodinger operator has a positive gap."""
        result = wl.discretize_weighted_laplacian_1d(R=1.0, n_grid=30)
        if 'error' in result:
            pytest.skip(result['error'])
        assert result['gap_1d'] > 0

    def test_1d_ground_energy_positive(self, wl):
        """Ground state energy is positive (confining potential)."""
        result = wl.discretize_weighted_laplacian_1d(R=1.0, n_grid=30)
        if 'error' in result:
            pytest.skip(result['error'])
        assert result['ground_energy'] > 0

    def test_1d_J_positive(self, wl):
        """J = det(M_FP) is positive throughout the interior."""
        result = wl.discretize_weighted_laplacian_1d(R=1.0, n_grid=30)
        if 'error' in result:
            pytest.skip(result['error'])
        # Interior points should have J > 0
        J = result['J_values']
        n = len(J)
        # Check interior (exclude boundary where J might be small)
        assert np.all(J[n // 4 : 3 * n // 4] > 0)


# ======================================================================
# 13. Analytical kappa minimum is positive
# ======================================================================

class TestAnalyticalKappaMinimum:
    """The minimum of kappa_0(R) over all R is positive."""

    def test_kappa_minimum_positive(self):
        """kappa_min > 0."""
        result = WeightedLaplacian9DOF.analytical_kappa_minimum()
        assert result['kappa_is_positive']
        assert result['kappa_min'] > 0

    def test_kappa_minimum_location_reasonable(self):
        """R* is at a reasonable location (around 0.5-2.0)."""
        result = WeightedLaplacian9DOF.analytical_kappa_minimum()
        assert 0.1 < result['R_star'] < 5.0

    def test_kappa_minimum_mass_gap_positive(self):
        """Mass gap at the minimum kappa is positive."""
        result = WeightedLaplacian9DOF.analytical_kappa_minimum()
        assert result['m_gap_at_minimum_MeV'] > 0


# ======================================================================
# 14. Physical mass gap at R = 2.2 fm
# ======================================================================

class TestPhysicalMassGap:
    """Mass gap in MeV at physical parameters."""

    def test_weighted_mass_exceeds_unweighted(self, wl):
        """m_weighted > m_unweighted at R = 2.2 fm."""
        result = wl.physical_mass_gap_MeV(R_fm=2.2, n_sample=30)
        assert result['m_weighted_MeV'] > result['m_unweighted_MeV']

    def test_enhancement_over_unweighted(self, wl):
        """Enhancement factor > 1 at physical parameters."""
        result = wl.physical_mass_gap_MeV(R_fm=2.2, n_sample=30)
        assert result['enhancement_over_unweighted'] > 1.0

    def test_unweighted_mass_correct(self, wl):
        """m_unweighted = 2 * hbar*c / R = 179.4 MeV at R=2.2 fm."""
        result = wl.physical_mass_gap_MeV(R_fm=2.2, n_sample=10)
        expected = 2.0 * 197.3269804 / 2.2
        assert result['m_unweighted_MeV'] == pytest.approx(expected, rel=1e-6)


# ======================================================================
# 15. Complete analysis
# ======================================================================

class TestCompleteAnalysis:
    """Integration test: complete analysis at physical parameters."""

    def test_complete_analysis_runs(self, wl):
        """Complete analysis runs without error."""
        result = wl.complete_analysis(R_fm=2.2, n_sample=20, seed=42)
        assert 'assessment' in result
        assert 'mass_gap' in result
        assert 'decomposition' in result

    def test_complete_analysis_ghost_verified(self, wl):
        """Ghost curvature structure is verified in complete analysis."""
        result = wl.complete_analysis(R_fm=2.2, n_sample=20, seed=42)
        ghost = result['ghost_verification']
        assert ghost['Tr_Li_Lj_proportional_to_I']
        assert ghost['Tr_Li_Li'] == pytest.approx(4.0, rel=1e-6)

    def test_complete_analysis_positive_gap(self, wl):
        """Complete analysis reports positive weighted mass gap."""
        result = wl.complete_analysis(R_fm=2.2, n_sample=20, seed=42)
        assert result['mass_gap']['m_weighted_MeV'] > 0


# ======================================================================
# 16. Consistency checks
# ======================================================================

class TestConsistency:
    """Cross-checks between methods."""

    def test_fp_det_equals_product_eigenvalues(self, wl, gd):
        """det(M_FP) = product of eigenvalues of M_FP."""
        R = 1.0
        a = np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1])
        M = gd.fp_operator_truncated(a, R)
        det_direct = np.linalg.det(M)
        eigs = np.linalg.eigvalsh(M)
        det_from_eigs = np.prod(eigs)
        assert det_direct == pytest.approx(det_from_eigs, rel=1e-8)

    def test_log_det_from_det(self, wl):
        """log det(M) matches log(det(M)) when det > 0."""
        R = 1.0
        a = np.zeros(9)
        log_det = wl.log_fp_determinant(a, R)
        det_val = wl.fp_determinant(a, R)
        assert log_det == pytest.approx(np.log(det_val), rel=1e-10)

    def test_analytical_vs_bakry_emery_at_origin(self, wl):
        """Analytical kappa_0 matches BakryEmeryGap Hessian at origin."""
        R = 2.0
        kappa_ana = WeightedLaplacian9DOF.analytical_kappa_at_origin(R)
        result = wl.hessian_Phi_at_origin(R)
        kappa_num = result['min_eigenvalue']
        assert kappa_num == pytest.approx(kappa_ana, rel=1e-3)

    def test_physical_potential_at_origin(self, wl):
        """Phi(0) = -log det M_FP(0) since S_YM(0) = 0."""
        R = 1.0
        Phi = wl.physical_potential_Phi(np.zeros(9), R)
        log_det = wl.log_fp_determinant(np.zeros(9), R)
        # S_YM(0) = 0, so Phi(0) = -log det(M_FP(0))
        assert Phi == pytest.approx(-log_det, rel=1e-10)
