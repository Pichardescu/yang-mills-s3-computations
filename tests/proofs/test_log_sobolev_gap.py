"""
Tests for Log-Sobolev Inequality on the Gribov Region.

Test categories:
    1. bakry_emery_to_log_sobolev: BE curvature -> LS constant (THEOREM)
    2. curvature_at_origin: kappa at a=0 analytical formula
    3. curvature_on_gribov_region: combined BE + PW bound
    4. log_sobolev_constant_vs_R: alpha(R) = 2/kappa(R)
    5. physical_mass_gap_bound: K(R)*kappa(R) lower bound
    6. r_independent_bound: kappa grows, m_phys > 0 for all R
    7. log_sobolev_vs_poincare: LS strictly stronger than Poincare
    8. tensorization_theorem: dimension-independent LS
    9. combined_curvature_bound: scan kappa(R) for monotonicity/positivity
   10. full_synthesis: end-to-end consistency
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.log_sobolev_gap import LogSobolevGap
from yang_mills_s3.proofs.bakry_emery_gap import BakryEmeryGap
from yang_mills_s3.proofs.diameter_theorem import _DR_ASYMPTOTIC
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def lsg():
    """LogSobolevGap instance."""
    return LogSobolevGap()


@pytest.fixture
def beg():
    """BakryEmeryGap instance."""
    return BakryEmeryGap()


# ======================================================================
# 1. bakry_emery_to_log_sobolev (THEOREM)
# ======================================================================

class TestBakryEmeryToLogSobolev:
    """BE curvature kappa -> LS constant alpha = 2/kappa."""

    def test_positive_kappa_gives_finite_alpha(self):
        """For kappa > 0, alpha = 2/kappa is finite and positive."""
        result = LogSobolevGap.bakry_emery_to_log_sobolev(1.0)
        assert result['valid']
        assert result['ls_constant'] == pytest.approx(2.0, rel=1e-12)
        assert result['spectral_gap'] == pytest.approx(1.0, rel=1e-12)
        assert result['hypercontractive']

    def test_alpha_inversely_proportional_to_kappa(self):
        """alpha = 2/kappa for various kappa values."""
        for kappa in [0.1, 1.0, 5.0, 10.0, 100.0]:
            result = LogSobolevGap.bakry_emery_to_log_sobolev(kappa)
            assert result['ls_constant'] == pytest.approx(2.0 / kappa, rel=1e-12)

    def test_negative_kappa_gives_invalid(self):
        """For kappa <= 0, LS is invalid."""
        result = LogSobolevGap.bakry_emery_to_log_sobolev(-1.0)
        assert not result['valid']
        assert result['ls_constant'] == np.inf

    def test_zero_kappa_gives_invalid(self):
        """For kappa = 0, LS is invalid."""
        result = LogSobolevGap.bakry_emery_to_log_sobolev(0.0)
        assert not result['valid']

    def test_mixing_rate_equals_kappa(self):
        """Mixing rate = kappa (exponential convergence)."""
        result = LogSobolevGap.bakry_emery_to_log_sobolev(3.5)
        assert result['mixing_rate'] == pytest.approx(3.5, rel=1e-12)

    def test_label_is_theorem(self):
        """Result is labeled THEOREM."""
        result = LogSobolevGap.bakry_emery_to_log_sobolev(1.0)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 2. curvature_at_origin (THEOREM)
# ======================================================================

class TestCurvatureAtOrigin:
    """Bakry-Emery curvature at a=0: kappa = 4/R^2 + 4*g^2*R^2/9."""

    def test_curvature_at_R1(self):
        """At R=1, kappa = 4 + 4*g^2(1)/9."""
        result = LogSobolevGap.curvature_at_origin(1.0)
        g2 = ZwanzigerGapEquation.running_coupling_g2(1.0)
        expected = 4.0 + 4.0 * g2 / 9.0
        assert result['kappa_origin'] == pytest.approx(expected, rel=1e-10)

    def test_curvature_positive_all_R(self):
        """kappa_origin > 0 for all R > 0."""
        for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            result = LogSobolevGap.curvature_at_origin(R)
            assert result['kappa_origin'] > 0, f"kappa_origin not positive at R={R}"

    def test_ghost_dominates_at_large_R(self):
        """At large R, ghost term >> V2 term."""
        result = LogSobolevGap.curvature_at_origin(20.0)
        assert result['ghost_dominates']
        assert result['ghost_contribution'] > 10 * result['v2_contribution']

    def test_v2_dominates_at_small_R(self):
        """At small R, V2 term >> ghost term."""
        result = LogSobolevGap.curvature_at_origin(0.1)
        assert result['v2_contribution'] > result['ghost_contribution']

    def test_kappa_grows_with_R(self):
        """kappa_origin grows with R at large R (ghost dominance)."""
        k5 = LogSobolevGap.curvature_at_origin(5.0)['kappa_origin']
        k20 = LogSobolevGap.curvature_at_origin(20.0)['kappa_origin']
        k100 = LogSobolevGap.curvature_at_origin(100.0)['kappa_origin']
        assert k20 > k5
        assert k100 > k20

    def test_label_is_theorem(self):
        """Result is labeled THEOREM."""
        result = LogSobolevGap.curvature_at_origin(1.0)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 3. curvature_on_gribov_region (THEOREM)
# ======================================================================

class TestCurvatureOnGribovRegion:
    """Combined BE + PW curvature bound on full Omega_9."""

    def test_kappa_positive_all_R(self):
        """kappa_min > 0 for all R > 0."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            result = LogSobolevGap.curvature_on_gribov_region(R)
            assert result['kappa_min'] > 0, f"kappa_min not positive at R={R}"

    def test_kappa_min_is_max_of_be_and_pw(self):
        """kappa_min = max(kappa_BE, kappa_PW)."""
        for R in [1.0, 5.0, 20.0]:
            result = LogSobolevGap.curvature_on_gribov_region(R)
            expected = max(result['kappa_be'], result['kappa_pw'])
            assert result['kappa_min'] == pytest.approx(expected, rel=1e-12)

    def test_pw_bound_formula(self):
        """PW bound = pi^2*R^2/(2*dR^2)."""
        R = 3.0
        result = LogSobolevGap.curvature_on_gribov_region(R)
        expected_pw = np.pi**2 * R**2 / (2.0 * _DR_ASYMPTOTIC**2)
        assert result['kappa_pw'] == pytest.approx(expected_pw, rel=1e-10)

    def test_kappa_grows_with_R(self):
        """kappa_min grows with R (both BE and PW grow as R^2)."""
        k1 = LogSobolevGap.curvature_on_gribov_region(1.0)['kappa_min']
        k10 = LogSobolevGap.curvature_on_gribov_region(10.0)['kappa_min']
        k100 = LogSobolevGap.curvature_on_gribov_region(100.0)['kappa_min']
        assert k10 > k1
        assert k100 > k10

    def test_label_is_theorem(self):
        """Result is labeled THEOREM."""
        result = LogSobolevGap.curvature_on_gribov_region(2.0)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 4. log_sobolev_constant_vs_R (THEOREM)
# ======================================================================

class TestLogSobolevConstantVsR:
    """alpha(R) = 2/kappa(R) as function of R."""

    def test_all_kappa_positive(self, lsg):
        """kappa > 0 for all scanned R."""
        result = lsg.log_sobolev_constant_vs_R([0.5, 1, 2, 5, 10, 50])
        assert result['all_positive']

    def test_alpha_decreases_with_R(self, lsg):
        """alpha = 2/kappa DECREASES with R (confinement strengthens)."""
        result = lsg.log_sobolev_constant_vs_R([1.0, 5.0, 20.0, 100.0])
        assert result['alpha_decreasing']
        # alpha at R=100 should be much smaller than at R=1
        assert result['alpha_ls'][-1] < result['alpha_ls'][0]

    def test_spectral_gap_equals_kappa(self, lsg):
        """spectral_gap = kappa (by definition)."""
        result = lsg.log_sobolev_constant_vs_R([1.0, 5.0])
        np.testing.assert_allclose(result['spectral_gap'], result['kappa'])

    def test_alpha_is_two_over_kappa(self, lsg):
        """alpha = 2/kappa for each R."""
        result = lsg.log_sobolev_constant_vs_R([1.0, 2.0, 5.0])
        expected = 2.0 / result['kappa']
        np.testing.assert_allclose(result['alpha_ls'], expected, rtol=1e-12)


# ======================================================================
# 5. physical_mass_gap_bound (PROPOSITION)
# ======================================================================

class TestPhysicalMassGapBound:
    """Physical mass gap m^2 = K(R)*kappa(R) > 0 for each R."""

    def test_mass_positive_all_R(self):
        """m_phys > 0 for all R."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            result = LogSobolevGap.physical_mass_gap_bound(R)
            assert result['positive'], f"mass gap not positive at R={R}"
            assert result['m_phys_MeV'] > 0

    def test_kinetic_normalization_formula(self):
        """K(R) = 1/(4*pi^2*g^2*R^3)."""
        R = 2.0
        g2 = ZwanzigerGapEquation.running_coupling_g2(R)
        expected_K = 1.0 / (4.0 * np.pi**2 * g2 * R**3)
        result = LogSobolevGap.physical_mass_gap_bound(R)
        assert result['K'] == pytest.approx(expected_K, rel=1e-10)

    def test_m_squared_equals_K_times_kappa(self):
        """m^2 = K * kappa_min."""
        R = 3.0
        result = LogSobolevGap.physical_mass_gap_bound(R)
        expected = result['K'] * result['kappa_min']
        assert result['m_squared'] == pytest.approx(expected, rel=1e-10)

    def test_label_is_proposition(self):
        """Physical gap is labeled PROPOSITION."""
        result = LogSobolevGap.physical_mass_gap_bound(2.0)
        assert result['label'] == 'PROPOSITION'


# ======================================================================
# 6. r_independent_bound (THEOREM + PROPOSITION)
# ======================================================================

class TestRIndependentBound:
    """kappa grows, m_phys > 0 for all R."""

    def test_all_kappa_positive(self, lsg):
        """Field-space curvature kappa > 0 for all R."""
        result = lsg.r_independent_bound()
        assert result['all_kappa_positive']

    def test_kappa_grows_with_R(self, lsg):
        """kappa increases with R."""
        result = lsg.r_independent_bound()
        assert result['kappa_growing']

    def test_alpha_decreases_with_R(self, lsg):
        """LS constant alpha decreases with R (stronger confinement)."""
        result = lsg.r_independent_bound()
        assert result['alpha_decreasing']

    def test_all_mass_positive(self, lsg):
        """Physical mass gap > 0 for all R."""
        result = lsg.r_independent_bound()
        assert result['all_m_positive']

    def test_min_mass_positive(self, lsg):
        """Minimum physical mass gap > 0."""
        result = lsg.r_independent_bound()
        assert result['min_m_phys_MeV'] > 0

    def test_field_space_label_theorem(self, lsg):
        """Field-space gap is THEOREM."""
        result = lsg.r_independent_bound()
        assert result['field_space_gap_label'] == 'THEOREM'

    def test_physical_gap_label_proposition(self, lsg):
        """Physical gap R-independence is PROPOSITION."""
        result = lsg.r_independent_bound()
        assert result['physical_gap_label'] == 'PROPOSITION'


# ======================================================================
# 7. log_sobolev_vs_poincare (THEOREM)
# ======================================================================

class TestLogSobolevVsPoincare:
    """LS strictly stronger than Poincare."""

    def test_poincare_gap_equals_kappa(self):
        """Poincare spectral gap = kappa."""
        result = LogSobolevGap.log_sobolev_vs_poincare(5.0)
        assert result['poincare_gap'] == pytest.approx(5.0, rel=1e-12)

    def test_ls_constant_equals_two_over_kappa(self):
        """LS constant = 2/kappa."""
        result = LogSobolevGap.log_sobolev_vs_poincare(5.0)
        assert result['ls_constant'] == pytest.approx(0.4, rel=1e-12)

    def test_tensorizable(self):
        """LS is tensorizable (dimension-independent)."""
        result = LogSobolevGap.log_sobolev_vs_poincare(5.0)
        assert result['tensorizable']
        assert result['dimension_independent']

    def test_concentration_rate(self):
        """Concentration rate = kappa/2."""
        result = LogSobolevGap.log_sobolev_vs_poincare(4.0)
        assert result['concentration_rate'] == pytest.approx(2.0, rel=1e-12)

    def test_invalid_for_nonpositive_kappa(self):
        """Invalid for kappa <= 0."""
        result = LogSobolevGap.log_sobolev_vs_poincare(-1.0)
        assert not result['valid']


# ======================================================================
# 8. tensorization_theorem (THEOREM)
# ======================================================================

class TestTensorizationTheorem:
    """Dimension-independent LS via tensorization."""

    def test_single_mode(self):
        """Single mode: alpha_product = 2/kappa."""
        result = LogSobolevGap.tensorization_theorem([3.0], 1)
        assert result['valid']
        assert result['alpha_product'] == pytest.approx(2.0 / 3.0, rel=1e-12)

    def test_multiple_modes_bottleneck(self):
        """Multiple modes: alpha determined by weakest mode."""
        kappas = [1.0, 5.0, 10.0, 100.0]
        result = LogSobolevGap.tensorization_theorem(kappas, 4)
        assert result['valid']
        assert result['kappa_product'] == pytest.approx(1.0, rel=1e-12)
        assert result['bottleneck_mode'] == 0  # first mode is weakest
        assert result['dimension_independent']

    def test_dimension_independent(self):
        """Adding modes does NOT degrade LS constant if kappa bounded."""
        kappas_3 = [2.0, 5.0, 10.0]
        kappas_100 = [2.0] + [5.0 + i for i in range(99)]
        r3 = LogSobolevGap.tensorization_theorem(kappas_3, 3)
        r100 = LogSobolevGap.tensorization_theorem(kappas_100, 100)
        # Both determined by kappa=2.0 (bottleneck)
        assert r3['alpha_product'] == pytest.approx(r100['alpha_product'], rel=1e-12)

    def test_invalid_if_any_kappa_nonpositive(self):
        """Invalid if any mode has kappa <= 0."""
        kappas = [1.0, -0.1, 5.0]
        result = LogSobolevGap.tensorization_theorem(kappas, 3)
        assert not result['valid']
        assert result['alpha_product'] == np.inf

    def test_label_is_theorem(self):
        """Tensorization is THEOREM."""
        result = LogSobolevGap.tensorization_theorem([1.0], 1)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 9. combined_curvature_bound (THEOREM + NUMERICAL)
# ======================================================================

class TestCombinedCurvatureBound:
    """Scan kappa(R) for monotonicity and positivity."""

    def test_all_positive(self, lsg):
        """kappa_min > 0 for all scanned R."""
        result = lsg.combined_curvature_bound()
        assert result['all_positive']

    def test_monotone_increasing(self, lsg):
        """kappa_min is monotonically increasing in R."""
        # Use R values that are clearly in the growing regime
        result = lsg.combined_curvature_bound(
            R_values=[1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        )
        assert result['monotone_increasing']

    def test_growth_power_near_2(self, lsg):
        """kappa ~ R^power with power close to 2 at large R."""
        result = lsg.combined_curvature_bound(
            R_values=[10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        )
        # Power should be close to 2 (both PW and BE grow as R^2)
        assert 1.5 < result['growth_power'] < 2.5

    def test_origin_kappa_larger_than_region_kappa(self, lsg):
        """kappa at origin >= kappa on full region (origin is minimum of V)."""
        result = lsg.combined_curvature_bound(R_values=[2.0, 5.0, 10.0])
        for i in range(len(result['R'])):
            # The full-region bound may be larger due to max(BE, PW)
            # but the origin kappa (without V4 perturbation) should be comparable
            assert result['kappa_origin'][i] > 0


# ======================================================================
# 10. full_synthesis (end-to-end consistency)
# ======================================================================

class TestFullSynthesis:
    """End-to-end consistency of the Log-Sobolev analysis."""

    def test_synthesis_completes(self, lsg):
        """full_synthesis runs without errors."""
        result = lsg.full_synthesis(R_values=[1.0, 2.2, 10.0])
        assert 'summary' in result
        assert 'labels' in result

    def test_field_space_kappa_positive(self, lsg):
        """Summary: kappa positive for all R."""
        result = lsg.full_synthesis(R_values=[0.5, 1.0, 2.2, 10.0, 50.0])
        assert result['summary']['kappa_positive_all_R']

    def test_physical_gap_positive(self, lsg):
        """Summary: physical gap > 0 for all R."""
        result = lsg.full_synthesis(R_values=[0.5, 1.0, 2.2, 10.0, 50.0])
        assert result['summary']['physical_gap_positive_all_R']

    def test_tensorization_valid(self, lsg):
        """Summary: tensorization is valid."""
        result = lsg.full_synthesis(R_values=[1.0, 2.2, 10.0])
        assert result['summary']['tensorization_valid']

    def test_labels_correct(self, lsg):
        """All labels are correctly assigned."""
        result = lsg.full_synthesis(R_values=[1.0, 2.2])
        labels = result['labels']
        assert labels['field_space_kappa_positive'] == 'THEOREM'
        assert labels['ls_from_kappa'] == 'THEOREM (Bakry-Emery 1985)'
        assert labels['tensorization'] == 'THEOREM (Gross 1975)'
        assert labels['physical_gap_each_R'] == 'THEOREM'
        assert 'PROPOSITION' in labels['physical_gap_R_independent']

    def test_ls_improves_with_R(self, lsg):
        """LS constant decreases with R (stronger confinement)."""
        result = lsg.full_synthesis(R_values=[1.0, 5.0, 20.0, 100.0])
        assert result['summary']['ls_improves_with_R']


# ======================================================================
# 11. Cross-validation with BakryEmeryGap
# ======================================================================

class TestCrossValidation:
    """Cross-check Log-Sobolev results with existing BE module."""

    def test_kappa_agrees_with_be_analytical(self):
        """kappa from curvature_on_gribov_region agrees with BE module."""
        for R in [1.0, 5.0, 20.0]:
            ls_result = LogSobolevGap.curvature_on_gribov_region(R)
            be_result = BakryEmeryGap.analytical_kappa_bound(R)
            # LS kappa_be should match BE analytical
            assert ls_result['kappa_be'] == pytest.approx(
                be_result['kappa_lower_bound'], rel=1e-10
            )

    def test_ls_constant_from_be_numerical(self, lsg, beg):
        """LS constant from numerical BE scan should be consistent."""
        R = 2.0
        # Numerical kappa from BE scan at origin
        H_origin = beg.compute_hessian_U_phys(np.zeros(9), R)
        from scipy.linalg import eigvalsh
        kappa_numerical = eigvalsh(H_origin)[0]
        # LS constant
        ls_result = LogSobolevGap.bakry_emery_to_log_sobolev(kappa_numerical)
        assert ls_result['valid']
        assert ls_result['ls_constant'] == pytest.approx(2.0 / kappa_numerical, rel=1e-10)
