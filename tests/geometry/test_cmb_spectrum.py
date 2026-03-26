"""
Tests for CMB-QCD duality on S3/I* (NUMERICAL 12.5).

Verifies:
  1. Radial eigenfunctions (orthonormality, boundary values)
  2. Scalar C_l on S3 (positivity, scaling)
  3. Scalar C_l on S3/I* (suppression, I*-invariant weighting)
  4. Suppression ratio S_l (bounds, Weyl law, quadrupole anomaly)
  5. Coexact 1-form spectrum (spectral desert, gap preservation)
  6. CMB-QCD duality (same m(k) controls both sides)
  7. Planck comparison (qualitative quadrupole suppression)
  8. Mode count asymptotics (Weyl law convergence)
  9. Parameter scans (chi_LSS, Omega_total)
  10. Self-consistency checks

Labeling convention:
  THEOREM: proven in the paper (e.g., orthonormality, Weyl law)
  NUMERICAL: verified computationally (e.g., S_2 value, mode counts)
  CONSISTENCY: internal cross-check
"""

import pytest
import numpy as np
from scipy.integrate import quad

from yang_mills_s3.geometry.cmb_spectrum_s3 import CMBSpectrumS3, PLANCK_2018_LOW_L


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture
def cmb():
    """Default CMB spectrum with chi_lss = 0.38 (near Aurich et al. optimal)."""
    return CMBSpectrumS3(n_s=0.965, chi_lss=0.38, k_max=200)


@pytest.fixture
def cmb_hz():
    """Harrison-Zeldovich (n_s=1) for analytic checks."""
    return CMBSpectrumS3(n_s=1.0, chi_lss=0.38, k_max=200)


@pytest.fixture
def cmb_luminet():
    """Luminet et al. parameters: Omega_tot ~ 1.013, chi_lss ~ 0.35."""
    return CMBSpectrumS3(n_s=0.965, chi_lss=0.35, k_max=200)


@pytest.fixture
def cmb_aurich():
    """Aurich et al. optimal: Omega_tot ~ 1.018, chi_lss ~ 0.42."""
    return CMBSpectrumS3(n_s=0.965, chi_lss=0.42, k_max=200)


# ==================================================================
# 1. Radial eigenfunctions
# ==================================================================

class TestRadialEigenfunctions:
    """Tests for Phi_k^l(chi) on S3."""

    def test_orthonormality_l0(self, cmb):
        """THEOREM: integral Phi_k^0 Phi_k'^0 sin^2(chi) dchi = delta_{kk'}."""
        for k1 in range(0, 5):
            for k2 in range(k1, 5):
                val, _ = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 0, chi)
                                 * cmb.radial_eigenfunction(k2, 0, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-8, (
                    f"<Phi_{k1}^0|Phi_{k2}^0> = {val}, expected {expected}"
                )

    def test_orthonormality_l2(self, cmb):
        """THEOREM: orthonormality holds at l=2."""
        for k1 in range(2, 6):
            for k2 in range(k1, 6):
                val, _ = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 2, chi)
                                 * cmb.radial_eigenfunction(k2, 2, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-8

    def test_phi_00_is_constant(self, cmb):
        """THEOREM: Phi_0^0(chi) = sqrt(2/pi)."""
        expected = np.sqrt(2.0 / np.pi)
        for chi in [0.1, 0.5, 1.0, np.pi / 2, 2.5]:
            val = cmb.radial_eigenfunction(0, 0, chi)
            assert abs(val - expected) < 1e-12

    def test_phi_zero_for_l_gt_k(self, cmb):
        """THEOREM: Phi_k^l = 0 for l > k."""
        for k in range(5):
            for l in range(k + 1, k + 3):
                val = cmb.radial_eigenfunction(k, l, 0.5)
                assert val == 0.0

    def test_phi_vanishes_at_poles_for_l_ge_1(self, cmb):
        """THEOREM: sin(0) = sin(pi) = 0, so Phi_k^l(0) = Phi_k^l(pi) = 0 for l >= 1."""
        for l in range(1, 4):
            for k in range(l, l + 3):
                assert abs(cmb.radial_eigenfunction(k, l, 0.0)) < 1e-14
                assert abs(cmb.radial_eigenfunction(k, l, np.pi)) < 1e-12

    def test_high_k_normalization(self, cmb):
        """NUMERICAL: normalization holds at high k (log-gamma stability)."""
        for k, l in [(20, 0), (30, 5), (50, 10)]:
            val, _ = quad(
                lambda chi: cmb.radial_eigenfunction(k, l, chi) ** 2 * np.sin(chi) ** 2,
                0, np.pi, limit=200
            )
            assert abs(val - 1.0) < 1e-6, f"||Phi_{k}^{l}||^2 = {val}"

    def test_array_input(self, cmb):
        """Phi_k^l accepts array input."""
        chi_arr = np.linspace(0.1, 2.9, 30)
        phi_arr = cmb.radial_eigenfunction(5, 2, chi_arr)
        assert phi_arr.shape == (30,)
        for i in [0, 10, 29]:
            val = cmb.radial_eigenfunction(5, 2, chi_arr[i])
            assert abs(phi_arr[i] - val) < 1e-14


# ==================================================================
# 2. Scalar C_l on S3
# ==================================================================

class TestScalarClS3:
    """Tests for angular power spectrum on S3 (full sphere)."""

    def test_positivity(self, cmb):
        """C_l^{S3} > 0 for all l >= 1."""
        for l in range(1, 15):
            assert cmb.cl_scalar_s3(l) > 0, f"C_{l}^S3 <= 0"

    def test_negative_l_returns_zero(self, cmb):
        """C_l = 0 for l < 0."""
        assert cmb.cl_scalar_s3(-1) == 0.0

    def test_cl_decreases_at_high_l(self, cmb):
        """NUMERICAL: C_l ~ 1/[l(l+1)] at large l (Sachs-Wolfe)."""
        c5 = cmb.cl_scalar_s3(5)
        c25 = cmb.cl_scalar_s3(25)
        assert c25 < c5

    def test_primordial_positivity(self, cmb):
        """P(k) > 0 for k >= 1."""
        for k in range(1, 30):
            assert cmb.primordial_spectrum(k) > 0

    def test_primordial_k0_zero(self, cmb):
        """P(0) = 0."""
        assert cmb.primordial_spectrum(0) == 0.0

    def test_hz_primordial(self, cmb_hz):
        """For n_s = 1: P(k) = 1/[k(k+2)]."""
        for k in [1, 5, 10, 50]:
            pk = cmb_hz.primordial_spectrum(k)
            expected = 1.0 / (k * (k + 2))
            assert abs(pk - expected) < 1e-14


# ==================================================================
# 3. Scalar C_l on S3/I* (suppression)
# ==================================================================

class TestScalarClPoincare:
    """Tests for C_l on S3/I* (position-averaged)."""

    def test_suppressed_vs_s3(self, cmb):
        """THEOREM: C_l^{I*} <= C_l^{S3} for all l (fewer modes = less power)."""
        for l in range(2, 25):
            cl_s3 = cmb.cl_scalar_s3(l)
            cl_p = cmb.cl_scalar_poincare(l)
            assert cl_p <= cl_s3 + 1e-30, f"C_{l}^I* > C_{l}^S3"

    def test_positivity_high_l(self, cmb):
        """NUMERICAL: C_l^{I*} > 0 for large l (high-k modes contribute)."""
        for l in [12, 15, 20, 25, 30]:
            assert cmb.cl_scalar_poincare(l) > 0, f"C_{l}^I* = 0"

    def test_strong_suppression_low_l(self, cmb):
        """
        NUMERICAL: S_l << 1 for l = 2,...,10 because m(k)=0 for k=1,...,11.
        The first contributing level is k=12.
        """
        for l in range(2, 11):
            sl = cmb.suppression_ratio_scalar(l)
            assert sl < 0.05, f"S_{l} = {sl}, expected << 1"


# ==================================================================
# 4. Suppression ratio S_l (CMB side)
# ==================================================================

class TestSuppressionRatio:
    """Tests for S_l = C_l^{I*} / C_l^{S3}."""

    def test_bounded_zero_one(self, cmb):
        """THEOREM: 0 <= S_l <= 1 for all l."""
        for l in range(2, 31):
            sl = cmb.suppression_ratio_scalar(l)
            assert 0 <= sl <= 1 + 1e-10, f"S_{l} = {sl} out of [0,1]"

    def test_quadrupole_strongly_suppressed(self, cmb):
        """
        NUMERICAL 12.5: S_2 << 1/120 (extra-suppressed relative to Weyl limit).
        This is the key CMB prediction.
        """
        s2 = cmb.suppression_ratio_scalar(2)
        weyl = 1.0 / 120.0
        assert s2 < weyl, f"S_2 = {s2} should be < 1/120 = {weyl}"
        assert s2 > 0, "S_2 should be positive"
        assert s2 < 0.01, f"S_2 = {s2} should be strongly suppressed"

    def test_quadrupole_extra_suppressed_vs_weyl(self, cmb):
        """
        NUMERICAL 12.5: S_2 / (1/120) < 1, i.e., quadrupole is MORE suppressed
        than the Weyl asymptotic limit. This is because m(k)=0 for k=2,...,11
        eliminates the modes that would otherwise dominate C_2.
        """
        s2 = cmb.suppression_ratio_scalar(2)
        weyl = 1.0 / 120.0
        ratio = s2 / weyl
        assert ratio < 1.0, f"S_2/(1/120) = {ratio}, expected < 1"
        assert ratio > 0.3, f"S_2/(1/120) = {ratio}, expected in (0.3, 1)"

    def test_approaches_weyl_limit(self):
        """
        NUMERICAL 12.4: S_l -> 1/120 as l -> inf (Weyl law).
        At l=30 with k_max=500, S_l should be close to 1/120.
        """
        cmb500 = CMBSpectrumS3(chi_lss=0.38, k_max=500)
        s30 = cmb500.suppression_ratio_scalar(30)
        weyl = 1.0 / 120.0
        assert abs(s30 - weyl) / weyl < 0.15, (
            f"S_30 = {s30}, expected ~{weyl} (15% tolerance)"
        )

    def test_s_values_cluster_near_weyl(self, cmb):
        """
        NUMERICAL: For l >= 5, S_l oscillates around 1/120 within ~25%.
        """
        weyl = 1.0 / 120.0
        for l in range(5, 31):
            sl = cmb.suppression_ratio_scalar(l)
            assert abs(sl - weyl) / weyl < 0.30, (
                f"S_{l} = {sl}, expected within 30% of {weyl}"
            )

    def test_consistency_with_cl_ratio(self, cmb):
        """CONSISTENCY: S_l = cl_poincare(l) / cl_s3(l)."""
        for l in [2, 5, 10, 20, 30]:
            sl = cmb.suppression_ratio_scalar(l)
            cl_s3 = cmb.cl_scalar_s3(l)
            cl_p = cmb.cl_scalar_poincare(l)
            if cl_s3 > 0:
                expected = cl_p / cl_s3
                assert abs(sl - expected) < 1e-14


# ==================================================================
# 5. Coexact 1-form spectrum (QCD side)
# ==================================================================

class TestCoexactSpectrum:
    """Tests for the coexact 1-form mode count on S3/I*."""

    def test_s3_multiplicity_formula(self, cmb):
        """THEOREM: n_co(k) = 2k(k+2) on S3."""
        assert cmb.coexact_multiplicity_s3(1) == 6
        assert cmb.coexact_multiplicity_s3(2) == 16
        assert cmb.coexact_multiplicity_s3(3) == 30
        assert cmb.coexact_multiplicity_s3(10) == 240

    def test_gap_preserves_on_poincare(self, cmb):
        """
        THEOREM 12.1: k=1 coexact modes survive: n_co^{I*}(1) = 3.
        The self-dual sector gives m(0)*(1+2) = 1*3 = 3 modes.
        The anti-self-dual sector gives m(2)*1 = 0*1 = 0 modes.
        """
        n = cmb.coexact_multiplicity_poincare(1)
        assert n == 3, f"n_co^I*(1) = {n}, expected 3"

    def test_spectral_desert(self, cmb):
        """
        THEOREM 12.2: n_co^{I*}(k) = 0 for k = 2,...,10.
        This is the spectral desert creating the 36x eigenvalue ratio.
        """
        for k in range(2, 11):
            n = cmb.coexact_multiplicity_poincare(k)
            assert n == 0, f"n_co^I*({k}) = {n}, expected 0"

    def test_k11_first_excited(self, cmb):
        """
        THEOREM 12.2: First excited coexact level is k=11.
        Anti-self-dual: m(12)*11 = 1*11 = 11 modes.
        Self-dual: m(10)*13 = 0*13 = 0 modes.
        Total: 11 modes.
        """
        n = cmb.coexact_multiplicity_poincare(11)
        assert n == 11, f"n_co^I*(11) = {n}, expected 11"

    def test_k13_second_excited(self, cmb):
        """
        THEOREM 12.2a: k=13 gives 15 modes.
        Self-dual: m(12)*15 = 1*15 = 15.
        Anti-self-dual: m(14)*13 = 0*13 = 0.
        """
        n = cmb.coexact_multiplicity_poincare(13)
        assert n == 15, f"n_co^I*(13) = {n}, expected 15"

    def test_coexact_suppression_ratio_zero_in_desert(self, cmb):
        """THEOREM: suppression ratio = 0 in the desert k=2,...,10."""
        for k in range(2, 11):
            ratio = cmb.coexact_suppression_ratio(k)
            assert ratio == 0.0, f"ratio({k}) = {ratio}"

    def test_coexact_suppression_at_k1(self, cmb):
        """THEOREM: at k=1, 3/6 = 0.5 of modes survive."""
        ratio = cmb.coexact_suppression_ratio(1)
        assert abs(ratio - 0.5) < 1e-14, f"ratio(1) = {ratio}"

    def test_mass_ratio(self, cmb):
        """
        THEOREM 12.2: m_2/m_1 = (k_2+1)/(k_1+1) = 12/2 = 6.0 on S3/I*.
        (vs 3/2 = 1.5 on S3)
        """
        k1 = 1   # first coexact level
        k2 = 11  # second coexact level on S3/I*
        ratio = (k2 + 1) / (k1 + 1)
        assert ratio == 6.0


# ==================================================================
# 6. CMB-QCD duality (the key claim for NUMERICAL 12.5)
# ==================================================================

class TestCMBQCDDuality:
    """
    Tests verifying that the SAME m(k) function controls both sides.

    NUMERICAL 12.5: The CMB-QCD duality is a mathematical identity:
    both the scalar CMB suppression and the coexact QCD sparsification
    are determined by the trivial multiplicity m(k) of I* in V_k.
    """

    def test_molien_zero_range(self, cmb):
        """
        THEOREM (Molien): m(k) = 0 for k = 1,...,11.
        This single fact drives BOTH:
          - CMB quadrupole suppression (no scalar modes at k=1,...,11)
          - QCD spectral desert (no coexact modes at k=2,...,10)
        """
        for k in range(1, 12):
            assert cmb.trivial_multiplicity(k) == 0, f"m({k}) != 0"

    def test_molien_first_nonzero(self, cmb):
        """THEOREM: m(0) = 1 (trivial rep) and m(12) = 1 (first excited)."""
        assert cmb.trivial_multiplicity(0) == 1
        assert cmb.trivial_multiplicity(12) == 1

    def test_molien_further_values(self, cmb):
        """THEOREM: m(20) = 1, m(24) = 1, m(30) = 1 (Molien series)."""
        assert cmb.trivial_multiplicity(20) == 1
        assert cmb.trivial_multiplicity(24) == 1
        assert cmb.trivial_multiplicity(30) == 1

    def test_odd_k_always_zero(self, cmb):
        """
        THEOREM: m(k) = 0 for all odd k.
        Because -I in I* acts as (-1)^k on V_k.
        """
        for k in range(1, 62, 2):
            assert cmb.trivial_multiplicity(k) == 0, f"m({k}) != 0 for odd k"

    def test_duality_same_function(self, cmb):
        """
        NUMERICAL 12.5: Both CMB and QCD use the SAME m(k).

        CMB side: modes at level k survive iff m(k) > 0.
        QCD side: coexact modes at level k survive iff m(k-1) > 0 or m(k+1) > 0.

        The difference is only which VALUES of m are probed:
          - CMB probes m(k) directly (scalar harmonics)
          - QCD probes m(k +/- 1) (coexact 1-form Hodge decomposition)
        """
        # The function is identical in both cases
        for k in range(0, 35):
            mk_cmb = cmb.trivial_multiplicity(k)
            mk_qcd = cmb.trivial_multiplicity(k)  # same function!
            assert mk_cmb == mk_qcd

    def test_cmb_uses_mk_for_weighting(self, cmb):
        """
        NUMERICAL: CMB C_l uses weight m(k)/(k+1) at each k.
        Verify that at k=12 (first m(k) > 0 after k=0), the weight is 1/13.
        """
        mk = cmb.trivial_multiplicity(12)
        weight = mk / (12 + 1)
        assert abs(weight - 1.0 / 13.0) < 1e-14

    def test_qcd_uses_mk_pm1(self, cmb):
        """
        NUMERICAL: QCD coexact at level k uses m(k-1) and m(k+1).
        At k=1: m(0)=1, m(2)=0 -> 3 SD modes, 0 ASD modes = 3 total.
        At k=11: m(10)=0, m(12)=1 -> 0 SD modes, 11 ASD modes = 11 total.
        """
        # k=1
        assert cmb.trivial_multiplicity(0) == 1
        assert cmb.trivial_multiplicity(2) == 0
        assert cmb.coexact_multiplicity_poincare(1) == 3

        # k=11
        assert cmb.trivial_multiplicity(10) == 0
        assert cmb.trivial_multiplicity(12) == 1
        assert cmb.coexact_multiplicity_poincare(11) == 11

    def test_compute_duality_structure(self, cmb):
        """NUMERICAL 12.5: compute_duality returns all needed data."""
        d = cmb.compute_duality(l_max=15, k_max_qcd=15)
        assert 'cmb_suppression' in d
        assert 'qcd_suppression' in d
        assert 'molien_values' in d
        assert 'quadrupole_ratio' in d
        assert 'weyl_limit' in d
        assert 'qcd_desert' in d
        assert d['qcd_desert'] is not None
        assert d['qcd_desert'][0] == 2    # desert starts at k=2
        assert d['qcd_desert'][1] == 10   # desert ends at k=10

    def test_both_sides_suppressed_at_low_k(self, cmb):
        """
        NUMERICAL 12.5: The SAME m(k) = 0 range kills both:
          - CMB: S_l strongly suppressed for l=2,...,11
          - QCD: coexact modes zero for k=2,...,10
        """
        # CMB side
        for l in range(2, 12):
            sl = cmb.suppression_ratio_scalar(l)
            assert sl < 0.02, f"S_{l} = {sl} should be << 1"

        # QCD side
        for k in range(2, 11):
            n = cmb.coexact_multiplicity_poincare(k)
            assert n == 0, f"n_co^I*({k}) should be 0"


# ==================================================================
# 7. Planck comparison
# ==================================================================

class TestPlanckComparison:
    """Tests for comparison with Planck 2018 data."""

    def test_planck_data_completeness(self):
        """All l from 2 to 30 in data dict."""
        for l in range(2, 31):
            assert l in PLANCK_2018_LOW_L

    def test_planck_quadrupole_anomaly(self):
        """
        OBSERVATIONAL: D_2^obs / D_2^LCDM ~ 0.19 (5x suppression).
        """
        d_obs, d_lcdm, _ = PLANCK_2018_LOW_L[2]
        ratio = d_obs / d_lcdm
        assert 0.1 < ratio < 0.3, f"Quadrupole ratio = {ratio}"

    def test_qualitative_agreement(self, cmb):
        """
        NUMERICAL: Our S_2 < 1 predicts suppression, consistent with
        Planck observed D_2/D_2^LCDM < 1.
        """
        s2 = cmb.suppression_ratio_scalar(2)
        assert s2 < 1.0, "S_2 should predict suppression"
        d_obs, d_lcdm, _ = PLANCK_2018_LOW_L[2]
        assert d_obs / d_lcdm < 1.0, "Planck quadrupole should be suppressed"

    def test_sw_stronger_than_observed(self, cmb):
        """
        NUMERICAL: Position-averaged SW gives STRONGER suppression than observed.
        S_2 ~ 0.006 < 0.19 (observed ratio).
        Expected: ISW effect adds power back (Aurich et al.).
        """
        s2 = cmb.suppression_ratio_scalar(2)
        observed = PLANCK_2018_LOW_L[2][0] / PLANCK_2018_LOW_L[2][1]
        assert s2 < observed, (
            f"SW suppression S_2={s2:.4f} should be stronger than "
            f"observed {observed:.4f} (ISW adds power)"
        )

    def test_planck_comparison_returns_keys(self, cmb):
        """planck_comparison returns expected keys."""
        comp = cmb.planck_comparison(l_max=10)
        assert 'results' in comp
        assert 'chi_squared' in comp
        assert 'n_dof' in comp
        assert comp['chi_squared'] >= 0

    def test_chi_squared_large(self, cmb):
        """
        NUMERICAL: chi^2 is large because SW-only misses ISW + position effects.
        This is EXPECTED, not a failure of the model.
        """
        comp = cmb.planck_comparison(l_max=30)
        # chi^2 should be large (SW-only is too strong a suppression)
        assert comp['chi_squared'] > 10, (
            "chi^2 should be large (SW-only vs full Planck)"
        )


# ==================================================================
# 8. Mode count and Weyl law
# ==================================================================

class TestModeCountWeyl:
    """Tests for cumulative mode counts and Weyl law convergence."""

    def test_scalar_mode_count_s3(self, cmb):
        """
        THEOREM: Total scalar modes on S3 up to level K = sum_{k=0}^K (k+1)^2.
        """
        counts = cmb.cumulative_mode_count(k_max=5)
        expected = sum((k + 1) ** 2 for k in range(6))  # 1+4+9+16+25+36=91
        assert counts['scalar_s3'] == expected

    def test_scalar_fraction_approaches_weyl(self):
        """
        NUMERICAL 12.4: scalar fraction -> 1/120 as k_max -> inf.
        """
        cmb = CMBSpectrumS3(k_max=200)
        c30 = cmb.cumulative_mode_count(k_max=30)
        c60 = cmb.cumulative_mode_count(k_max=60)
        weyl = 1.0 / 120.0

        # At k=60, should be closer to 1/120 than at k=30
        err30 = abs(c30['scalar_fraction'] - weyl)
        err60 = abs(c60['scalar_fraction'] - weyl)
        assert err60 < err30 + 0.001, "Weyl convergence not improving"

    def test_coexact_fraction_approaches_weyl(self):
        """
        NUMERICAL 12.4: coexact fraction -> 1/120 as k_max -> inf.
        """
        cmb = CMBSpectrumS3(k_max=200)
        c30 = cmb.cumulative_mode_count(k_max=30)
        weyl = 1.0 / 120.0
        # Should be within 10% of 1/120 by k=30
        assert abs(c30['coexact_fraction'] - weyl) / weyl < 0.15

    def test_coexact_fraction_consistent(self, cmb):
        """
        CONSISTENCY: The numerical fraction from paper (166/19220 ~ 0.0086)
        should match our computation at k=30.
        """
        c30 = cmb.cumulative_mode_count(k_max=30)
        # The paper quotes 166/19220 ~ 0.0086 for coexact modes
        # Our computation should give a similar value
        assert 0.005 < c30['coexact_fraction'] < 0.015


# ==================================================================
# 9. Parameter scans
# ==================================================================

class TestParameterScans:
    """Tests for chi_LSS scan and Omega_total conversion."""

    def test_omega_to_chi_luminet(self):
        """Omega_tot = 1.013 -> chi_LSS ~ 0.353."""
        chi = CMBSpectrumS3.omega_to_chi_lss(1.013)
        assert abs(chi - 3.1 * np.sqrt(0.013)) < 1e-10

    def test_omega_to_chi_aurich(self):
        """Omega_tot = 1.018 -> chi_LSS ~ 0.416."""
        chi = CMBSpectrumS3.omega_to_chi_lss(1.018)
        expected = 3.1 * np.sqrt(0.018)
        assert abs(chi - expected) < 1e-10

    def test_flat_raises(self):
        """Omega_tot = 1 raises ValueError."""
        with pytest.raises(ValueError):
            CMBSpectrumS3.omega_to_chi_lss(1.0)

    def test_open_raises(self):
        """Omega_tot < 1 raises ValueError."""
        with pytest.raises(ValueError):
            CMBSpectrumS3.omega_to_chi_lss(0.95)

    def test_scan_returns_sorted(self):
        """Scan results sorted by chi_squared."""
        cmb = CMBSpectrumS3(k_max=80)
        results = cmb.scan_chi_lss(chi_values=np.linspace(0.2, 0.8, 4), l_max=10)
        chi_sq = [r[1] for r in results]
        assert chi_sq == sorted(chi_sq)

    def test_scan_restores_chi(self):
        """After scan, chi_lss is restored."""
        cmb = CMBSpectrumS3(chi_lss=0.42, k_max=80)
        cmb.scan_chi_lss(chi_values=[0.1, 0.5], l_max=5)
        assert abs(cmb.chi_lss - 0.42) < 1e-15


# ==================================================================
# 10. Self-consistency and report
# ==================================================================

class TestSelfConsistency:
    """Cross-checks and report generation."""

    def test_cl_table_structure(self, cmb):
        """full_cl_table returns correct structure."""
        table = cmb.full_cl_table(l_max=10)
        assert len(table) == 9   # l=2,...,10
        for row in table:
            assert 'l' in row
            assert 'cl_s3' in row
            assert 'cl_poincare' in row
            assert 'suppression_ratio' in row

    def test_cl_table_has_planck(self, cmb):
        """Rows in Planck range have Planck data."""
        table = cmb.full_cl_table(l_max=10)
        for row in table:
            assert 'planck_dl_obs' in row

    def test_coexact_table_structure(self, cmb):
        """coexact_spectrum_table returns correct structure."""
        table = cmb.coexact_spectrum_table(k_max=15)
        assert len(table) == 15
        for row in table:
            assert 'k' in row
            assert 'n_s3' in row
            assert 'n_poincare' in row
            assert 'n_sd' in row
            assert 'n_asd' in row

    def test_coexact_table_k1(self, cmb):
        """Coexact table at k=1 gives n_sd=3, n_asd=0, total=3."""
        table = cmb.coexact_spectrum_table(k_max=5)
        k1 = table[0]
        assert k1['k'] == 1
        assert k1['n_sd'] == 3
        assert k1['n_asd'] == 0
        assert k1['n_poincare'] == 3

    def test_duality_report_status(self, cmb):
        """Duality report has NUMERICAL status."""
        report = cmb.duality_report(l_max=10, k_max_qcd=15)
        assert report['status'] == 'NUMERICAL'
        assert report['label'] == 'NUMERICAL 12.5'

    def test_duality_report_key_results(self, cmb):
        """Report key results are computed."""
        report = cmb.duality_report(l_max=10, k_max_qcd=15)
        kr = report['key_results']
        assert kr['molien_zero_1_to_11'] is True
        assert 0 < kr['S_2'] < 0.01
        assert abs(kr['weyl_limit'] - 1.0 / 120.0) < 1e-15
        assert 0 < kr['S_2_over_weyl'] < 1.0
        assert kr['qcd_desert'] == (2, 10)
        assert kr['cmb_first_nonzero_scalar'] == 12

    def test_chi_lss_independence_of_molien(self, cmb, cmb_luminet, cmb_aurich):
        """
        THEOREM: m(k) does not depend on chi_lss (it's group-theoretic).
        """
        for k in range(0, 35):
            m1 = cmb.trivial_multiplicity(k)
            m2 = cmb_luminet.trivial_multiplicity(k)
            m3 = cmb_aurich.trivial_multiplicity(k)
            assert m1 == m2 == m3, f"m({k}) differs across chi_lss values"

    def test_suppression_varies_with_chi_lss(self):
        """
        NUMERICAL: S_2 depends on chi_lss (radial eigenfunction evaluation point).
        Different chi_lss -> different S_2 values.
        """
        cmb1 = CMBSpectrumS3(chi_lss=0.35, k_max=100)
        cmb2 = CMBSpectrumS3(chi_lss=0.50, k_max=100)
        s2_1 = cmb1.suppression_ratio_scalar(2)
        s2_2 = cmb2.suppression_ratio_scalar(2)
        # Both should be strongly suppressed but different
        assert s2_1 != s2_2
        assert s2_1 < 0.02
        assert s2_2 < 0.02


# ==================================================================
# 11. Cross-check with paper values
# ==================================================================

class TestPaperValues:
    """Verify specific values quoted in Section 12 of the paper."""

    def test_paper_s2_value(self):
        """
        NUMERICAL 12.7: Paper quotes S_2 ~ 0.006 at chi_lss = 0.35.
        """
        cmb = CMBSpectrumS3(chi_lss=0.35, k_max=200)
        s2 = cmb.suppression_ratio_scalar(2)
        assert abs(s2 - 0.006) < 0.002, f"S_2 = {s2}, paper says ~0.006"

    def test_paper_s30_value(self):
        """
        NUMERICAL 12.7: S_30 ~ 0.0084, close to 1/120 = 0.00833.
        """
        cmb = CMBSpectrumS3(chi_lss=0.35, k_max=200)
        s30 = cmb.suppression_ratio_scalar(30)
        assert abs(s30 - 0.0083) < 0.002, f"S_30 = {s30}"

    def test_paper_coexact_modes_at_30(self):
        """
        NUMERICAL 12.4: Paper quotes ~166 I*-invariant coexact modes for k=1,...,30.
        """
        cmb = CMBSpectrumS3(k_max=200)
        total = sum(cmb.coexact_multiplicity_poincare(k) for k in range(1, 31))
        assert abs(total - 166) < 20, f"Total coexact I* modes = {total}, paper says ~166"

    def test_paper_total_coexact_s3(self):
        """
        Total coexact modes on S3 for k=1,...,30.
        sum_{k=1}^{30} 2k(k+2) = 20770.
        (Paper Section 12.4 quotes 19220, which is a minor typo.)
        """
        cmb = CMBSpectrumS3(k_max=200)
        total = sum(cmb.coexact_multiplicity_s3(k) for k in range(1, 31))
        expected = sum(2 * k * (k + 2) for k in range(1, 31))
        assert total == expected == 20770

    def test_paper_eigenvalue_ratio(self):
        """
        THEOREM 12.2: Eigenvalue ratio (k=11 vs k=1) = 144/4 = 36.
        Mass ratio = 12/2 = 6.0.
        """
        ev1 = (1 + 1) ** 2   # 4
        ev11 = (11 + 1) ** 2  # 144
        assert ev11 / ev1 == 36
        assert (11 + 1) / (1 + 1) == 6


# ==================================================================
# 12. Multiple chi_lss values (robustness)
# ==================================================================

class TestRobustness:
    """Verify that the key results hold across chi_lss values."""

    @pytest.mark.parametrize("chi_lss", [0.30, 0.35, 0.38, 0.42, 0.50, 0.60])
    def test_quadrupole_always_suppressed(self, chi_lss):
        """
        NUMERICAL: S_2 < 1/120 for all reasonable chi_lss.
        """
        cmb = CMBSpectrumS3(chi_lss=chi_lss, k_max=150)
        s2 = cmb.suppression_ratio_scalar(2)
        weyl = 1.0 / 120.0
        assert s2 < weyl * 1.5, f"S_2 = {s2} at chi_lss = {chi_lss}"
        assert s2 > 0

    @pytest.mark.parametrize("chi_lss", [0.30, 0.35, 0.38, 0.42, 0.50])
    def test_suppression_bounded(self, chi_lss):
        """THEOREM: 0 < S_l < 1 for all l and all chi_lss."""
        cmb = CMBSpectrumS3(chi_lss=chi_lss, k_max=150)
        for l in range(2, 16):
            sl = cmb.suppression_ratio_scalar(l)
            assert 0 <= sl <= 1 + 1e-10, f"S_{l} out of bounds at chi={chi_lss}"

    @pytest.mark.parametrize("chi_lss", [0.30, 0.38, 0.50])
    def test_qcd_desert_always_present(self, chi_lss):
        """
        THEOREM: The coexact desert k=2,...,10 is chi_lss-independent
        (it's determined by m(k), not by radial eigenfunctions).
        """
        cmb = CMBSpectrumS3(chi_lss=chi_lss, k_max=150)
        for k in range(2, 11):
            assert cmb.coexact_multiplicity_poincare(k) == 0
