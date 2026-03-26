"""
Tests for the S(1/2) angular correlation statistic.

Verifies:
  1. d_l_to_c_l conversion (D_l <-> C_l)
  2. angular_correlation_function: C(theta) from power spectrum
  3. s_half_statistic: S(60) integral
  4. s_half_comparison: model vs LCDM comparison
  5. Legendre polynomial summation identities
  6. Physical sanity: S(60) for LCDM vs S3/I*
  7. Edge cases and robustness

All CMB results labeled NUMERICAL.

Physical ground truth:
  - C(0) = sum_l (2l+1) C_l / (4*pi) = total variance
  - C(theta) for flat spectrum (C_l = const) is sharply peaked at theta=0
  - S(60) for Planck observed data is anomalously low vs LCDM expectation
  - S(60) for S3/I* should be lower than LCDM (suppressed large-angle correlations)

References:
  - Spergel et al., ApJ 583, 553 (2003)
  - Copi, Huterer, Schwarz, Starkman, PRD 75, 023507 (2007)
  - Aurich, Lustig, Steiner, CQG 22, 2061 (2005)
"""

import pytest
import numpy as np

from yang_mills_s3.spectral.angular_correlation import (
    d_l_to_c_l,
    angular_correlation_function,
    s_half_statistic,
    s_half_from_d_l,
    s_half_comparison,
    planck_observed_c_l,
    planck_lcdm_c_l,
)
from yang_mills_s3.spectral.cmb_boltzmann import PLANCK_LOW_L


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def planck_obs_cl():
    """Planck observed C_l (raw) for l=2..30."""
    return planck_observed_c_l()


@pytest.fixture
def lcdm_cl():
    """LCDM best-fit C_l (raw) for l=2..30."""
    return planck_lcdm_c_l()


@pytest.fixture
def flat_spectrum_cl():
    """Flat spectrum: C_l = 1 for l=2..30."""
    return {l: 1.0 for l in range(2, 31)}


@pytest.fixture
def single_mode_cl():
    """Single mode: only l=2 (quadrupole)."""
    return {2: 1000.0}


@pytest.fixture
def planck_obs_dl():
    """Planck observed D_l for l=2..30."""
    return {l: vals[0] for l, vals in PLANCK_LOW_L.items()}


@pytest.fixture
def lcdm_dl():
    """LCDM best-fit D_l for l=2..30."""
    return {l: vals[1] for l, vals in PLANCK_LOW_L.items()}


# ======================================================================
# 1. D_l to C_l conversion
# ======================================================================

class TestDlToClConversion:
    """Tests for d_l_to_c_l conversion."""

    def test_returns_dict(self, planck_obs_dl):
        """d_l_to_c_l returns a dictionary."""
        result = d_l_to_c_l(planck_obs_dl)
        assert isinstance(result, dict)

    def test_same_keys(self, planck_obs_dl):
        """Output has same l keys as input (for l >= 2)."""
        result = d_l_to_c_l(planck_obs_dl)
        for l in planck_obs_dl:
            if l >= 2:
                assert l in result

    def test_conversion_formula(self):
        """C_l = D_l * 2*pi / (l*(l+1))."""
        d_l = {2: 1000.0, 10: 500.0}
        c_l = d_l_to_c_l(d_l)
        for l, d_val in d_l.items():
            expected = d_val * 2.0 * np.pi / (l * (l + 1))
            np.testing.assert_allclose(c_l[l], expected, rtol=1e-12)

    def test_roundtrip(self):
        """Converting D_l -> C_l -> D_l recovers original."""
        d_l_orig = {l: float(100 * l) for l in range(2, 20)}
        c_l = d_l_to_c_l(d_l_orig)
        # Convert back
        d_l_back = {}
        for l, c_val in c_l.items():
            d_l_back[l] = c_val * l * (l + 1) / (2.0 * np.pi)
        for l in d_l_orig:
            np.testing.assert_allclose(d_l_back[l], d_l_orig[l], rtol=1e-12)

    def test_c_l_positive(self, planck_obs_dl):
        """C_l > 0 for positive D_l input."""
        c_l = d_l_to_c_l(planck_obs_dl)
        for l, c_val in c_l.items():
            assert c_val > 0, f"C_{l} = {c_val} <= 0"

    def test_c_l_smaller_than_d_l(self, planck_obs_dl):
        """C_l < D_l for all l >= 2 (since 2*pi/(l*(l+1)) < 1 for l >= 3)."""
        c_l = d_l_to_c_l(planck_obs_dl)
        for l in planck_obs_dl:
            if l >= 3:
                assert c_l[l] < planck_obs_dl[l], \
                    f"C_{l} = {c_l[l]} >= D_{l} = {planck_obs_dl[l]}"

    def test_empty_dict(self):
        """Empty input gives empty output."""
        assert d_l_to_c_l({}) == {}

    def test_skips_l_below_2(self):
        """l=0 and l=1 are skipped."""
        d_l = {0: 100.0, 1: 200.0, 2: 300.0}
        c_l = d_l_to_c_l(d_l)
        assert 0 not in c_l
        assert 1 not in c_l
        assert 2 in c_l


# ======================================================================
# 2. Angular correlation function C(theta)
# ======================================================================

class TestAngularCorrelationFunction:
    """Tests for angular_correlation_function."""

    def test_returns_ndarray(self, lcdm_cl):
        """Returns a numpy array."""
        result = angular_correlation_function(lcdm_cl, [0, 60, 90, 120, 180])
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self, lcdm_cl):
        """Output length equals number of input angles."""
        theta = [10, 20, 30, 40, 50]
        result = angular_correlation_function(lcdm_cl, theta)
        assert len(result) == len(theta)

    def test_c_zero_equals_total_variance(self, lcdm_cl):
        """
        C(0) = sum_l (2l+1)/(4*pi) * C_l.

        At theta=0, P_l(1)=1 for all l, so C(0) is the total variance.
        """
        c_at_zero = angular_correlation_function(lcdm_cl, [0.0])[0]
        expected = sum(
            (2 * l + 1) / (4.0 * np.pi) * c_l
            for l, c_l in lcdm_cl.items()
        )
        np.testing.assert_allclose(c_at_zero, expected, rtol=1e-10)

    def test_c_zero_positive(self, lcdm_cl):
        """C(0) > 0 for any spectrum with positive C_l."""
        c_at_zero = angular_correlation_function(lcdm_cl, [0.0])[0]
        assert c_at_zero > 0

    def test_c_180_value_for_quadrupole(self, single_mode_cl):
        """
        C(180) for pure l=2: P_2(-1) = 1, so C(180) = 5/(4*pi) * C_2.
        """
        c_at_180 = angular_correlation_function(single_mode_cl, [180.0])[0]
        # P_2(cos(180)) = P_2(-1) = 1
        expected = 5 / (4.0 * np.pi) * single_mode_cl[2]
        np.testing.assert_allclose(c_at_180, expected, rtol=1e-10)

    def test_c_90_for_quadrupole(self, single_mode_cl):
        """
        C(90) for pure l=2: P_2(0) = -1/2, so C(90) = 5/(4*pi)*C_2*(-1/2).
        """
        c_at_90 = angular_correlation_function(single_mode_cl, [90.0])[0]
        expected = 5 / (4.0 * np.pi) * single_mode_cl[2] * (-0.5)
        np.testing.assert_allclose(c_at_90, expected, rtol=1e-10)

    def test_symmetry_under_reflection(self, lcdm_cl):
        """
        C(theta) depends only on cos(theta), so C(theta) at any theta
        should equal the Legendre sum evaluated at cos(theta).

        Test: C(theta) is a real-valued function with no imaginary parts.
        """
        theta = np.linspace(0, 180, 50)
        result = angular_correlation_function(lcdm_cl, theta)
        assert np.all(np.isreal(result))
        assert np.all(np.isfinite(result))

    def test_single_angle(self, lcdm_cl):
        """Works with a single angle."""
        result = angular_correlation_function(lcdm_cl, [90.0])
        assert len(result) == 1

    def test_accepts_array(self, lcdm_cl):
        """Accepts numpy array as theta input."""
        theta = np.array([0.0, 90.0, 180.0])
        result = angular_correlation_function(lcdm_cl, theta)
        assert len(result) == 3

    def test_l_max_truncation(self, lcdm_cl):
        """l_max parameter truncates the sum."""
        theta = [0.0, 60.0, 120.0]
        result_full = angular_correlation_function(lcdm_cl, theta)
        result_trunc = angular_correlation_function(lcdm_cl, theta, l_max=5)
        # Truncated result uses fewer modes, so should differ
        assert not np.allclose(result_full, result_trunc, rtol=0.01)

    def test_array_input_for_cl(self):
        """Accepts array-like C_l input (indexed by l)."""
        cl_arr = np.zeros(31)
        cl_arr[2] = 1000.0
        cl_arr[3] = 800.0
        result = angular_correlation_function(cl_arr, [0.0])
        expected = (5 * 1000.0 + 7 * 800.0) / (4.0 * np.pi)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)


# ======================================================================
# 3. Legendre polynomial summation identities
# ======================================================================

class TestLegendreIdentities:
    """
    Tests that exploit known Legendre polynomial identities.
    These verify our summation is mathematically correct.
    """

    def test_p_l_at_one(self):
        """P_l(1) = 1 for all l. Verified via C(0) = sum (2l+1)C_l/(4pi)."""
        from scipy.special import legendre
        for l in range(2, 31):
            P_l = legendre(l)
            np.testing.assert_allclose(P_l(1.0), 1.0, atol=1e-12)

    def test_p_l_at_minus_one(self):
        """P_l(-1) = (-1)^l for all l."""
        from scipy.special import legendre
        for l in range(2, 31):
            P_l = legendre(l)
            expected = (-1) ** l
            np.testing.assert_allclose(P_l(-1.0), expected, atol=1e-12)

    def test_p_2_at_zero(self):
        """P_2(0) = -1/2."""
        from scipy.special import legendre
        P2 = legendre(2)
        np.testing.assert_allclose(P2(0.0), -0.5, atol=1e-14)

    def test_p_3_at_zero(self):
        """P_3(0) = 0 (odd function)."""
        from scipy.special import legendre
        P3 = legendre(3)
        np.testing.assert_allclose(P3(0.0), 0.0, atol=1e-14)

    def test_addition_theorem_consistency(self, flat_spectrum_cl):
        """
        For C_l = const, C(theta) = const * sum_l (2l+1)/(4*pi) * P_l(cos theta).

        The Legendre addition theorem gives:
        sum_{l=0}^{L} (2l+1) P_l(x) = ... (Christoffel-Darboux formula).
        Not a simple closed form, but we verify C(0) dominates.
        """
        c0 = angular_correlation_function(flat_spectrum_cl, [0.0])[0]
        c90 = angular_correlation_function(flat_spectrum_cl, [90.0])[0]
        # C(0) should be much larger than |C(90)| for flat spectrum
        assert c0 > abs(c90)


# ======================================================================
# 4. S(1/2) statistic
# ======================================================================

class TestSHalfStatistic:
    """Tests for s_half_statistic (S(60) integral)."""

    def test_returns_float(self, lcdm_cl):
        """S(60) returns a scalar float."""
        result = s_half_statistic(lcdm_cl)
        assert isinstance(result, (float, np.floating))

    def test_positive(self, lcdm_cl):
        """S(60) >= 0 (integral of a squared quantity)."""
        result = s_half_statistic(lcdm_cl)
        assert result >= 0

    def test_finite(self, lcdm_cl):
        """S(60) is finite."""
        result = s_half_statistic(lcdm_cl)
        assert np.isfinite(result)

    def test_s_half_for_zero_spectrum(self):
        """S(60) = 0 for zero spectrum."""
        cl_zero = {l: 0.0 for l in range(2, 31)}
        result = s_half_statistic(cl_zero)
        np.testing.assert_allclose(result, 0.0, atol=1e-20)

    def test_s_half_increases_with_amplitude(self, lcdm_cl):
        """
        Doubling C_l should quadruple S(60) (since [C(theta)]^2 scales as C_l^2).
        """
        s1 = s_half_statistic(lcdm_cl)
        cl_doubled = {l: 2.0 * c for l, c in lcdm_cl.items()}
        s2 = s_half_statistic(cl_doubled)
        np.testing.assert_allclose(s2, 4.0 * s1, rtol=1e-6)

    def test_s_60_vs_s_90(self, lcdm_cl):
        """
        S(90) <= S(60) because the integration range is smaller.
        """
        s_60 = s_half_statistic(lcdm_cl, theta_min=60)
        s_90 = s_half_statistic(lcdm_cl, theta_min=90)
        assert s_90 <= s_60 + 1e-10

    def test_s_0_vs_s_60(self, lcdm_cl):
        """S(0) >= S(60) because the integration range is larger."""
        s_0 = s_half_statistic(lcdm_cl, theta_min=0.01)  # avoid sin(0)=0 edge
        s_60 = s_half_statistic(lcdm_cl, theta_min=60)
        assert s_0 >= s_60 - 1e-10

    def test_from_d_l_convenience(self, lcdm_dl, lcdm_cl):
        """s_half_from_d_l gives same result as converting + s_half_statistic."""
        s_from_dl = s_half_from_d_l(lcdm_dl)
        s_from_cl = s_half_statistic(lcdm_cl)
        np.testing.assert_allclose(s_from_dl, s_from_cl, rtol=1e-6)


# ======================================================================
# 5. S(1/2) comparison: model vs LCDM
# ======================================================================

class TestSHalfComparison:
    """Tests for s_half_comparison."""

    def test_returns_dict(self, lcdm_cl):
        """Returns a dictionary."""
        result = s_half_comparison(lcdm_cl, lcdm_cl)
        assert isinstance(result, dict)

    def test_required_keys(self, lcdm_cl):
        """Result has all required keys."""
        result = s_half_comparison(lcdm_cl, lcdm_cl)
        required = {'s_model', 's_lcdm', 'ratio', 'theta_min'}
        assert required.issubset(set(result.keys()))

    def test_same_spectrum_ratio_one(self, lcdm_cl):
        """Comparing a spectrum with itself gives ratio = 1."""
        result = s_half_comparison(lcdm_cl, lcdm_cl)
        np.testing.assert_allclose(result['ratio'], 1.0, rtol=1e-6)

    def test_default_theta_min(self, lcdm_cl):
        """Default theta_min is 60 degrees."""
        result = s_half_comparison(lcdm_cl, lcdm_cl)
        assert result['theta_min'] == 60

    def test_custom_theta_min(self, lcdm_cl):
        """Custom theta_min is stored correctly."""
        result = s_half_comparison(lcdm_cl, lcdm_cl, theta_min=90)
        assert result['theta_min'] == 90

    def test_s_values_positive(self, lcdm_cl):
        """Both S values are positive."""
        result = s_half_comparison(lcdm_cl, lcdm_cl)
        assert result['s_model'] > 0
        assert result['s_lcdm'] > 0


# ======================================================================
# 6. Physical tests: Planck data
# ======================================================================

class TestPhysicalPlanck:
    """
    NUMERICAL: Physical tests using Planck 2018 data.

    These test that S(60) for Planck observed data is low,
    and that LCDM prediction is higher.
    """

    def test_planck_observed_s60_positive(self, planck_obs_cl):
        """S(60) from Planck observed data is positive and finite."""
        s60 = s_half_statistic(planck_obs_cl)
        assert s60 > 0
        assert np.isfinite(s60)

    def test_lcdm_s60_positive(self, lcdm_cl):
        """S(60) from LCDM best-fit is positive and finite."""
        s60 = s_half_statistic(lcdm_cl)
        assert s60 > 0
        assert np.isfinite(s60)

    def test_planck_s60_lower_than_lcdm(self, planck_obs_cl, lcdm_cl):
        """
        NUMERICAL: Planck observed S(60) < LCDM predicted S(60).

        This is the celebrated "lack of large-angle correlations" anomaly.
        Planck data shows less large-angle correlation than LCDM expects.

        Note: with only l=2..30, this is dominated by the low quadrupole.
        """
        s60_planck = s_half_statistic(planck_obs_cl)
        s60_lcdm = s_half_statistic(lcdm_cl)
        assert s60_planck < s60_lcdm, \
            f"Expected S60(Planck) < S60(LCDM): {s60_planck} vs {s60_lcdm}"

    def test_comparison_ratio_below_one(self, planck_obs_cl, lcdm_cl):
        """
        NUMERICAL: S(60)_Planck / S(60)_LCDM < 1.

        The ratio quantifies the anomaly.
        """
        result = s_half_comparison(planck_obs_cl, lcdm_cl)
        assert result['ratio'] < 1.0, \
            f"Expected ratio < 1, got {result['ratio']}"

    def test_c_theta_planck_near_zero_at_large_angles(self, planck_obs_cl):
        """
        NUMERICAL: C(theta) from Planck data is near zero for theta > 60.

        The observed lack of large-angle correlations means C(theta)
        hovers around zero for theta > 60 degrees.
        """
        theta_large = np.linspace(60, 180, 50)
        c_theta = angular_correlation_function(planck_obs_cl, theta_large)
        # The RMS of C(theta) at large angles should be small compared to C(0)
        c_zero = angular_correlation_function(planck_obs_cl, [0.0])[0]
        rms_large = np.sqrt(np.mean(c_theta ** 2))
        # The ratio should be much less than 1
        assert rms_large / abs(c_zero) < 0.5, \
            f"Large-angle RMS/C(0) = {rms_large/abs(c_zero)}: not suppressed enough"


# ======================================================================
# 7. Physical tests: S3/I* prediction (CAMB-dependent)
# ======================================================================

class TestS3IstarPrediction:
    """
    NUMERICAL: Test S(60) prediction for S3/I*.

    These tests require CAMB to compute the S3/I* spectrum.
    They verify the key prediction: S3/I* produces lower S(60) than LCDM.
    """

    @pytest.fixture(scope="class")
    def cmb_boltzmann(self):
        """CMBBoltzmann instance (requires CAMB)."""
        camb = pytest.importorskip("camb", reason="CAMB not installed")
        from yang_mills_s3.spectral.cmb_boltzmann import CMBBoltzmann
        return CMBBoltzmann(omega_tot=1.018, l_max=30)

    @pytest.fixture(scope="class")
    def istar_dl(self, cmb_boltzmann):
        """D_l for S3/I* from full Boltzmann."""
        return cmb_boltzmann.cls_s3_istar()

    @pytest.fixture(scope="class")
    def istar_cl(self, istar_dl):
        """Raw C_l for S3/I*."""
        return d_l_to_c_l(istar_dl)

    def test_istar_s60_positive(self, istar_cl):
        """S(60) for S3/I* is positive."""
        s60 = s_half_statistic(istar_cl)
        assert s60 > 0
        assert np.isfinite(s60)

    def test_istar_s60_lower_than_lcdm(self, istar_cl, lcdm_cl):
        """
        NUMERICAL: S(60) for S3/I* < S(60) for LCDM.

        This is the central prediction: S3/I* naturally explains the
        low S(60) anomaly because the spectral desert (m(k)=0, k=1..11)
        suppresses exactly the large-angle modes.
        """
        s60_istar = s_half_statistic(istar_cl)
        s60_lcdm = s_half_statistic(lcdm_cl)
        assert s60_istar < s60_lcdm, \
            f"S60(I*) = {s60_istar} not less than S60(LCDM) = {s60_lcdm}"

    def test_istar_comparison_ratio(self, istar_cl, lcdm_cl):
        """
        NUMERICAL: Report the S(60) ratio for S3/I* vs LCDM.

        Aurich et al. (2005) found acceptable fits, meaning the ratio
        should be significantly below 1.
        """
        result = s_half_comparison(istar_cl, lcdm_cl)
        ratio = result['ratio']
        assert 0 < ratio < 1, f"Ratio = {ratio}, expected in (0, 1)"
        # Print for NUMERICAL record
        print(f"\nNUMERICAL: S(60) comparison at Omega_tot=1.018")
        print(f"  S(60) S3/I* = {result['s_model']:.6e} muK^4*sr")
        print(f"  S(60) LCDM  = {result['s_lcdm']:.6e} muK^4*sr")
        print(f"  Ratio       = {ratio:.4f}")


# ======================================================================
# 8. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_mode_s60(self, single_mode_cl):
        """S(60) works for a single-mode spectrum."""
        s60 = s_half_statistic(single_mode_cl)
        assert s60 > 0
        assert np.isfinite(s60)

    def test_c_theta_at_boundary_0(self, lcdm_cl):
        """C(theta) at exactly 0 degrees works."""
        result = angular_correlation_function(lcdm_cl, [0.0])
        assert np.isfinite(result[0])

    def test_c_theta_at_boundary_180(self, lcdm_cl):
        """C(theta) at exactly 180 degrees works."""
        result = angular_correlation_function(lcdm_cl, [180.0])
        assert np.isfinite(result[0])

    def test_many_angles(self, lcdm_cl):
        """C(theta) computed at many angles is well-behaved."""
        theta = np.linspace(0, 180, 1000)
        result = angular_correlation_function(lcdm_cl, theta)
        assert np.all(np.isfinite(result))
        assert len(result) == 1000

    def test_s_half_with_l_max(self, lcdm_cl):
        """S(60) respects l_max parameter."""
        s_full = s_half_statistic(lcdm_cl, l_max=30)
        s_trunc = s_half_statistic(lcdm_cl, l_max=5)
        # Both should be positive
        assert s_full > 0
        assert s_trunc > 0
        # They should differ (unless spectrum is degenerate)
        assert s_full != s_trunc

    def test_theta_min_180_gives_zero(self, lcdm_cl):
        """S(180) should be essentially zero (empty integration range)."""
        s180 = s_half_statistic(lcdm_cl, theta_min=179.9)
        assert s180 < 1e-5 * s_half_statistic(lcdm_cl, theta_min=60)

    def test_c_theta_smooth(self, lcdm_cl):
        """C(theta) should vary smoothly (no discontinuities)."""
        theta = np.linspace(1, 179, 500)
        result = angular_correlation_function(lcdm_cl, theta)
        # Check that consecutive differences are small relative to range
        diffs = np.abs(np.diff(result))
        max_diff = np.max(diffs)
        total_range = np.max(result) - np.min(result)
        if total_range > 0:
            assert max_diff / total_range < 0.1, \
                f"C(theta) has sharp jumps: max_diff/range = {max_diff/total_range}"


# ======================================================================
# 9. Analytic S(60) for pure quadrupole
# ======================================================================

class TestAnalyticQuadrupole:
    """
    Analytic test: S(60) for a pure l=2 spectrum.

    For C_l = C_2 * delta_{l,2}:
      C(theta) = (5/(4*pi)) * C_2 * P_2(cos theta)
      S(60) = (5/(4*pi))^2 * C_2^2 * integral_{60}^{180} [P_2(cos t)]^2 sin(t) dt

    The integral can be computed analytically using the substitution x = cos(theta).
    integral_{60}^{180} P_2(cos t)^2 sin(t) dt = integral_{-1}^{1/2} P_2(x)^2 dx
    P_2(x) = (3x^2 - 1)/2
    P_2(x)^2 = (9x^4 - 6x^2 + 1)/4
    integral = [9x^5/5 - 6x^3/3 + x]_{-1}^{1/2} / 4
             = [9x^5/5 - 2x^3 + x]_{-1}^{1/2} / 4
    """

    def test_analytic_s60_pure_l2(self):
        """NUMERICAL: S(60) for pure l=2 matches analytic integral."""
        C_2 = 1000.0
        cl = {2: C_2}

        # Numerical
        s60_numerical = s_half_statistic(cl)

        # Analytic: integral_{-1}^{1/2} P_2(x)^2 dx
        # P_2(x) = (3x^2 - 1)/2
        def p2_sq_integral(x):
            return (9 * x ** 5 / 5 - 2 * x ** 3 + x) / 4

        integral_value = p2_sq_integral(0.5) - p2_sq_integral(-1.0)
        s60_analytic = (5 / (4 * np.pi)) ** 2 * C_2 ** 2 * integral_value

        np.testing.assert_allclose(s60_numerical, s60_analytic, rtol=1e-6,
                                   err_msg="S(60) for pure l=2 doesn't match analytic")

    def test_analytic_c_theta_pure_l2(self):
        """C(theta) for pure l=2: C(theta) = (5/(4pi)) * C_2 * P_2(cos theta)."""
        C_2 = 500.0
        cl = {2: C_2}
        theta_test = [0, 30, 45, 60, 90, 120, 150, 180]

        result = angular_correlation_function(cl, theta_test)
        from scipy.special import legendre
        P2 = legendre(2)

        for i, theta in enumerate(theta_test):
            x = np.cos(np.deg2rad(theta))
            expected = (5 / (4 * np.pi)) * C_2 * P2(x)
            np.testing.assert_allclose(result[i], expected, rtol=1e-10,
                                       err_msg=f"C({theta}) mismatch")
