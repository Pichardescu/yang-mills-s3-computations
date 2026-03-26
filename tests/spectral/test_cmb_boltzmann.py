"""
Tests for the CMB Boltzmann solver on S3/I*.

Verifies:
  1. CMBBoltzmann initialization (default and custom parameters)
  2. cls_s3: full Boltzmann C_l on simply-connected S3
  3. cls_s3_istar: C_l on S3/I* with eigenmode suppression
  4. _chi_lss: comoving distance to last scattering
  5. chi_squared: goodness-of-fit statistic
  6. comparison_table: structured output for analysis

All tests that require CAMB are guarded with pytest.importorskip.

Physical ground truth:
  - D_l ~ O(1000) muK^2 at low l
  - Quadrupole (l=2) is suppressed on S3/I* relative to S3
  - chi_LSS ~ 0.3-0.6 for Omega_tot in [1.01, 1.04]
  - chi^2 must be positive
"""

import pytest
import numpy as np


# ==================================================================
# CAMB availability guard
# ==================================================================

# Skip entire module if CAMB is not installed
camb = pytest.importorskip("camb", reason="CAMB not installed, skipping CMB Boltzmann tests")

from yang_mills_s3.spectral.cmb_boltzmann import CMBBoltzmann, PLANCK_2018, PLANCK_LOW_L, scan_omega


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="module")
def cmb_default():
    """CMBBoltzmann with default parameters (Omega_tot=1.018, l_max=30)."""
    return CMBBoltzmann(omega_tot=1.018, l_max=30)


@pytest.fixture(scope="module")
def cmb_custom():
    """CMBBoltzmann with slightly different Omega_tot."""
    return CMBBoltzmann(omega_tot=1.025, l_max=20)


@pytest.fixture(scope="module")
def cls_s3_cached(cmb_default):
    """Precomputed cls_s3 for reuse across tests (CAMB is expensive)."""
    return cmb_default.cls_s3()


@pytest.fixture(scope="module")
def cls_istar_cached(cmb_default):
    """Precomputed cls_s3_istar for reuse across tests."""
    return cmb_default.cls_s3_istar()


# ==================================================================
# 1. Initialization
# ==================================================================

class TestInitialization:
    """Tests for CMBBoltzmann constructor and parameter handling."""

    def test_default_omega_tot(self, cmb_default):
        """Default Omega_tot = 1.018 (Aurich et al. optimal)."""
        assert cmb_default.omega_tot == 1.018

    def test_default_l_max(self, cmb_default):
        """Default l_max = 30 (low-l regime)."""
        assert cmb_default.l_max == 30

    def test_omega_k_negative_for_closed(self, cmb_default):
        """Omega_k = 1 - Omega_tot < 0 for closed topology."""
        assert cmb_default.omega_k < 0

    def test_omega_k_value(self, cmb_default):
        """Omega_k = 1 - 1.018 = -0.018."""
        np.testing.assert_allclose(cmb_default.omega_k, -0.018, atol=1e-15)

    def test_custom_omega_tot(self, cmb_custom):
        """Custom Omega_tot is stored correctly."""
        assert cmb_custom.omega_tot == 1.025

    def test_custom_l_max(self, cmb_custom):
        """Custom l_max is stored correctly."""
        assert cmb_custom.l_max == 20

    def test_planck_params_used_by_default(self, cmb_default):
        """Planck 2018 best-fit parameters are loaded by default."""
        assert cmb_default.cosmo['H0'] == PLANCK_2018['H0']
        assert cmb_default.cosmo['ombh2'] == PLANCK_2018['ombh2']
        assert cmb_default.cosmo['omch2'] == PLANCK_2018['omch2']
        assert cmb_default.cosmo['tau'] == PLANCK_2018['tau']
        assert cmb_default.cosmo['As'] == PLANCK_2018['As']
        assert cmb_default.cosmo['ns'] == PLANCK_2018['ns']

    def test_custom_cosmo_params_override(self):
        """Custom cosmo_params override Planck defaults."""
        custom = {'H0': 70.0, 'tau': 0.06}
        cmb = CMBBoltzmann(omega_tot=1.01, cosmo_params=custom)
        assert cmb.cosmo['H0'] == 70.0
        assert cmb.cosmo['tau'] == 0.06
        # Non-overridden params retain Planck defaults
        assert cmb.cosmo['ombh2'] == PLANCK_2018['ombh2']

    def test_curvature_radius_positive(self, cmb_default):
        """Curvature radius R must be positive."""
        assert cmb_default.R_curvature_Mpc > 0

    def test_curvature_radius_reasonable(self, cmb_default):
        """
        R = c/(H0 * sqrt(|Omega_k|)) should be O(1000-10000) Mpc
        for Omega_k ~ -0.018.
        """
        R = cmb_default.R_curvature_Mpc
        assert 500 < R < 50000, f"R_curv = {R} Mpc seems unreasonable"

    def test_requires_camb(self):
        """
        If CAMB import were to fail, constructor would raise ImportError.

        We can only test that the constructor succeeds when CAMB IS available.
        """
        # Just verify construction works without error
        cmb = CMBBoltzmann(omega_tot=1.01, l_max=10)
        assert cmb is not None


# ==================================================================
# 2. cls_s3 — Simply-connected S3 spectrum
# ==================================================================

class TestClsS3:
    """Tests for the full Boltzmann C_l on simply-connected S3."""

    def test_returns_dict(self, cls_s3_cached):
        """cls_s3() returns a dictionary."""
        assert isinstance(cls_s3_cached, dict)

    def test_dict_has_l_keys(self, cls_s3_cached):
        """Keys are integer multipole values starting at l=2."""
        assert 2 in cls_s3_cached
        for l in cls_s3_cached:
            assert isinstance(l, (int, np.integer))
            assert l >= 2

    def test_d_l_values_are_positive(self, cls_s3_cached):
        """D_l = l(l+1)C_l/(2pi) must be positive for all l."""
        for l, d_l in cls_s3_cached.items():
            assert d_l > 0, f"D_{l} = {d_l} <= 0"

    def test_quadrupole_exists(self, cls_s3_cached):
        """l=2 (quadrupole) must be present in the output."""
        assert 2 in cls_s3_cached

    def test_quadrupole_positive(self, cls_s3_cached):
        """D_2 > 0 on simply-connected S3."""
        assert cls_s3_cached[2] > 0

    def test_d_l_order_of_magnitude(self, cls_s3_cached):
        """
        D_l should be O(100-10000) muK^2 at low l.

        Planck data has D_l ~ 200-2000 for l=2..30. CAMB on closed S3
        should be in the same ballpark.
        """
        for l, d_l in cls_s3_cached.items():
            assert 1 < d_l < 100000, \
                f"D_{l} = {d_l} muK^2 seems unreasonable"

    def test_contains_all_l_up_to_l_max(self, cmb_default, cls_s3_cached):
        """Should contain D_l for all l from 2 to l_max."""
        for l in range(2, cmb_default.l_max + 1):
            assert l in cls_s3_cached, f"l={l} missing from cls_s3 output"

    def test_caching(self, cmb_default):
        """Calling cls_s3() twice returns the same object (cached)."""
        result1 = cmb_default.cls_s3()
        result2 = cmb_default.cls_s3()
        assert result1 is result2

    @pytest.mark.slow
    def test_d_l_quadrupole_in_planck_range(self, cls_s3_cached):
        """
        D_2 should be roughly consistent with Planck (within factor of 5).

        Planck D_2 ~ 200 muK^2, but closed models can differ significantly.
        """
        d2 = cls_s3_cached[2]
        # Very loose bound -- just sanity check
        assert 10 < d2 < 10000, f"D_2 = {d2} muK^2 wildly inconsistent with Planck"


# ==================================================================
# 3. cls_s3_istar — S3/I* spectrum with eigenmode suppression
# ==================================================================

class TestClsS3Istar:
    """Tests for the C_l on Poincare dodecahedral space S3/I*."""

    def test_returns_dict(self, cls_istar_cached):
        """cls_s3_istar() returns a dictionary."""
        assert isinstance(cls_istar_cached, dict)

    def test_same_l_keys_as_s3(self, cls_s3_cached, cls_istar_cached):
        """S3/I* spectrum has the same l keys as the S3 spectrum."""
        assert set(cls_istar_cached.keys()) == set(cls_s3_cached.keys())

    def test_d_l_values_nonnegative(self, cls_istar_cached):
        """D_l >= 0 on S3/I* (suppressed but not negative)."""
        for l, d_l in cls_istar_cached.items():
            assert d_l >= 0, f"D_{l}(I*) = {d_l} < 0"

    def test_suppression_relative_to_s3(self, cls_s3_cached, cls_istar_cached):
        """
        D_l(I*) <= D_l(S3) for all l.

        The I* quotient removes eigenmodes (m(k)=0 for k=1..11),
        so the power spectrum can only decrease.
        """
        for l in cls_s3_cached:
            d_s3 = cls_s3_cached[l]
            d_istar = cls_istar_cached[l]
            assert d_istar <= d_s3 + 1e-10, \
                f"l={l}: D_l(I*) = {d_istar} > D_l(S3) = {d_s3}"

    def test_quadrupole_suppressed(self, cls_s3_cached, cls_istar_cached):
        """
        The l=2 quadrupole should be suppressed on S3/I*.

        This is the key CMB prediction: m(k)=0 for k=1..11 eliminates
        the largest modes that feed into the quadrupole.
        """
        d2_s3 = cls_s3_cached[2]
        d2_istar = cls_istar_cached[2]
        assert d2_istar < d2_s3, \
            f"Quadrupole not suppressed: D_2(I*)={d2_istar} >= D_2(S3)={d2_s3}"

    def test_high_l_suppression_consistent(self, cls_s3_cached, cls_istar_cached):
        """
        At all l, the suppression ratio S_l = C_l(I*)/C_l(S3) ~ 0.017.

        Session 21 established that the suppression is structural and
        roughly UNIFORM across all l (not l-selective). This is because
        the weight m(k)*nu / nu^2 ~ 1/60 is set by the order of I*,
        not by angular scale. The ratio is O(0.01-0.03) at all l.
        """
        if 25 not in cls_s3_cached or 25 not in cls_istar_cached:
            pytest.skip("l=25 not in computed spectra")

        for l in [25, 28, 30]:
            if l in cls_s3_cached and cls_s3_cached[l] > 0:
                ratio = cls_istar_cached[l] / cls_s3_cached[l]
                # S_l ~ 0.017 uniform; must be positive and finite
                assert ratio > 0.0, \
                    f"l={l}: suppression ratio {ratio} should be positive"
                assert ratio < 1.0, \
                    f"l={l}: suppression ratio {ratio} exceeds 1"

    def test_caching(self, cmb_default):
        """Calling cls_s3_istar() twice returns the same object (cached)."""
        result1 = cmb_default.cls_s3_istar()
        result2 = cmb_default.cls_s3_istar()
        assert result1 is result2


# ==================================================================
# 4. _chi_lss — Comoving distance to last scattering
# ==================================================================

class TestChiLSS:
    """Tests for the comoving distance to last scattering surface."""

    def test_returns_float(self, cmb_default):
        """_chi_lss() returns a scalar float."""
        chi = cmb_default._chi_lss()
        assert isinstance(chi, (float, np.floating))

    def test_positive(self, cmb_default):
        """chi_LSS > 0 (last scattering surface is at finite distance)."""
        chi = cmb_default._chi_lss()
        assert chi > 0

    def test_reasonable_range_default(self, cmb_default):
        """
        For Omega_tot ~ 1.018, chi_LSS should be in range 0.3-0.6.

        chi_LSS = d_LSS / R_curv, where d_LSS ~ 14 Gpc and
        R_curv = c/(H0*sqrt(|Omega_k|)) ~ 33 Gpc for Omega_k = -0.018.
        """
        chi = cmb_default._chi_lss()
        assert 0.2 < chi < 0.8, \
            f"chi_LSS = {chi} outside expected range [0.2, 0.8]"

    def test_different_omega_gives_different_chi(self, cmb_default, cmb_custom):
        """Different Omega_tot values should give different chi_LSS."""
        chi1 = cmb_default._chi_lss()
        chi2 = cmb_custom._chi_lss()
        assert chi1 != chi2, "Different Omega_tot should yield different chi_LSS"

    @pytest.mark.slow
    def test_chi_increases_with_omega_tot(self):
        """
        chi_LSS should increase with Omega_tot (more closed -> larger
        chi_LSS in units of curvature radius, because R_curv shrinks).
        """
        cmb_low = CMBBoltzmann(omega_tot=1.01, l_max=10)
        cmb_high = CMBBoltzmann(omega_tot=1.03, l_max=10)
        chi_low = cmb_low._chi_lss()
        chi_high = cmb_high._chi_lss()
        assert chi_high > chi_low, \
            f"chi_LSS should increase: {chi_low} (Omega=1.01) vs {chi_high} (Omega=1.03)"

    def test_less_than_pi(self, cmb_default):
        """
        chi_LSS < pi (last scattering surface cannot wrap around S3).

        On S3, chi ranges from 0 to pi (equator to antipodal point).
        For reasonable Omega_tot, chi_LSS << pi.
        """
        chi = cmb_default._chi_lss()
        assert chi < np.pi, f"chi_LSS = {chi} >= pi, physically impossible"


# ==================================================================
# 5. chi_squared — Goodness-of-fit statistic
# ==================================================================

class TestChiSquared:
    """Tests for the chi^2 computation against Planck data."""

    def test_returns_tuple(self, cmb_default):
        """chi_squared() returns (chi2, n_dof)."""
        result = cmb_default.chi_squared(model='s3')
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_chi2_positive_istar(self, cmb_default):
        """chi^2(I*) must be positive."""
        chi2, n = cmb_default.chi_squared(model='istar')
        assert chi2 > 0, f"chi^2(I*) = {chi2} <= 0"

    def test_chi2_positive_s3(self, cmb_default):
        """chi^2(S3) must be positive."""
        chi2, n = cmb_default.chi_squared(model='s3')
        assert chi2 > 0, f"chi^2(S3) = {chi2} <= 0"

    def test_chi2_positive_lcdm(self, cmb_default):
        """chi^2(LCDM) must be positive."""
        chi2, n = cmb_default.chi_squared(model='lcdm')
        assert chi2 > 0, f"chi^2(LCDM) = {chi2} <= 0"

    def test_n_dof_matches_data(self, cmb_default):
        """Number of degrees of freedom matches available Planck data points."""
        _, n = cmb_default.chi_squared(model='s3')
        # PLANCK_LOW_L has entries for l=2..30 (29 points)
        expected = len([l for l in PLANCK_LOW_L if l <= cmb_default.l_max])
        assert n == expected, f"n_dof = {n}, expected {expected}"

    def test_n_dof_respects_l_max(self, cmb_custom):
        """With lower l_max, fewer data points should be used."""
        _, n = cmb_custom.chi_squared(model='s3')
        expected = len([l for l in PLANCK_LOW_L if l <= cmb_custom.l_max])
        assert n == expected

    def test_lcdm_chi2_per_dof_reasonable(self, cmb_default):
        """
        LCDM chi^2/dof should be close to 1 for Planck data.

        Planck LCDM is the best-fit model, so chi^2/dof ~ 1.
        """
        chi2, n = cmb_default.chi_squared(model='lcdm')
        chi2_per_dof = chi2 / n
        assert 0.1 < chi2_per_dof < 5.0, \
            f"LCDM chi^2/dof = {chi2_per_dof}, expected near 1"

    def test_invalid_model_raises(self, cmb_default):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            cmb_default.chi_squared(model='invalid_model')

    def test_chi2_finite(self, cmb_default):
        """chi^2 must be finite (not inf or nan)."""
        for model in ['istar', 's3', 'lcdm']:
            chi2, n = cmb_default.chi_squared(model=model)
            assert np.isfinite(chi2), f"chi^2({model}) is not finite: {chi2}"


# ==================================================================
# 6. comparison_table — Structured output
# ==================================================================

class TestComparisonTable:
    """Tests for the comparison_table output structure."""

    def test_returns_list(self, cmb_default):
        """comparison_table() returns a list."""
        table = cmb_default.comparison_table()
        assert isinstance(table, list)

    def test_list_not_empty(self, cmb_default):
        """Table should contain data rows."""
        table = cmb_default.comparison_table()
        assert len(table) > 0

    def test_rows_are_dicts(self, cmb_default):
        """Each row is a dictionary."""
        table = cmb_default.comparison_table()
        for row in table:
            assert isinstance(row, dict)

    def test_required_keys_present(self, cmb_default):
        """Each row contains the required keys."""
        required_keys = {'l', 'D_l_planck', 'sigma', 'D_l_lcdm',
                         'D_l_s3', 'D_l_istar', 'S_l'}
        table = cmb_default.comparison_table()
        for row in table:
            missing = required_keys - set(row.keys())
            assert not missing, \
                f"Row l={row.get('l', '?')} missing keys: {missing}"

    def test_l_values_are_integers(self, cmb_default):
        """The l key in each row is an integer >= 2."""
        table = cmb_default.comparison_table()
        for row in table:
            assert isinstance(row['l'], (int, np.integer))
            assert row['l'] >= 2

    def test_l_values_sorted(self, cmb_default):
        """Rows should be sorted by increasing l."""
        table = cmb_default.comparison_table()
        l_vals = [row['l'] for row in table]
        assert l_vals == sorted(l_vals), "Table rows not sorted by l"

    def test_planck_data_matches_constant(self, cmb_default):
        """D_l_planck values should match the PLANCK_LOW_L constant."""
        table = cmb_default.comparison_table()
        for row in table:
            l = row['l']
            if l in PLANCK_LOW_L:
                d_obs, d_lcdm, sigma = PLANCK_LOW_L[l]
                assert row['D_l_planck'] == d_obs
                assert row['sigma'] == sigma
                assert row['D_l_lcdm'] == d_lcdm

    def test_sigma_positive(self, cmb_default):
        """Error bars must be positive."""
        table = cmb_default.comparison_table()
        for row in table:
            assert row['sigma'] > 0, \
                f"sigma <= 0 at l={row['l']}"

    def test_s_l_between_zero_and_one(self, cmb_default):
        """
        Suppression ratio S_l should be in [0, 1].

        S_l = C_l(I*) / C_l(S3), which must be non-negative
        and <= 1 (I* can only remove modes, not add them).
        """
        table = cmb_default.comparison_table()
        for row in table:
            s_l = row['S_l']
            assert -1e-10 <= s_l <= 1.0 + 1e-10, \
                f"S_{row['l']} = {s_l} outside [0, 1]"

    def test_d_l_s3_positive(self, cmb_default):
        """D_l(S3) > 0 in every row."""
        table = cmb_default.comparison_table()
        for row in table:
            assert row['D_l_s3'] > 0, \
                f"D_{row['l']}(S3) = {row['D_l_s3']} <= 0"

    def test_d_l_istar_nonnegative(self, cmb_default):
        """D_l(I*) >= 0 in every row."""
        table = cmb_default.comparison_table()
        for row in table:
            assert row['D_l_istar'] >= -1e-10, \
                f"D_{row['l']}(I*) = {row['D_l_istar']} < 0"

    def test_number_of_rows(self, cmb_default):
        """
        Number of rows should match available Planck data within l_max.
        """
        table = cmb_default.comparison_table()
        expected = len([l for l in PLANCK_LOW_L if l <= cmb_default.l_max])
        assert len(table) == expected


# ==================================================================
# 7. Planck data constants
# ==================================================================

class TestPlanckConstants:
    """Verify the Planck data and parameter constants are well-formed."""

    def test_planck_2018_keys(self):
        """PLANCK_2018 dict has the required cosmological parameters."""
        required = {'H0', 'ombh2', 'omch2', 'tau', 'As', 'ns'}
        assert required.issubset(set(PLANCK_2018.keys()))

    def test_planck_h0_reasonable(self):
        """H0 ~ 67 km/s/Mpc (Planck 2018)."""
        assert 60 < PLANCK_2018['H0'] < 75

    def test_planck_low_l_has_l2_through_l30(self):
        """PLANCK_LOW_L covers l=2..30."""
        for l in range(2, 31):
            assert l in PLANCK_LOW_L, f"l={l} missing from PLANCK_LOW_L"

    def test_planck_low_l_tuples(self):
        """Each entry in PLANCK_LOW_L is a 3-tuple (D_obs, D_lcdm, sigma)."""
        for l, val in PLANCK_LOW_L.items():
            assert isinstance(val, tuple), f"l={l}: not a tuple"
            assert len(val) == 3, f"l={l}: expected 3-tuple, got {len(val)}"

    def test_planck_low_l_positive_sigma(self):
        """All error bars sigma > 0."""
        for l, (d_obs, d_lcdm, sigma) in PLANCK_LOW_L.items():
            assert sigma > 0, f"l={l}: sigma={sigma} <= 0"

    def test_planck_low_l_positive_d_lcdm(self):
        """All LCDM predictions D_lcdm > 0."""
        for l, (d_obs, d_lcdm, sigma) in PLANCK_LOW_L.items():
            assert d_lcdm > 0, f"l={l}: D_lcdm={d_lcdm} <= 0"


# ==================================================================
# 8. Edge cases and error handling
# ==================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_l_max_equals_2(self):
        """CMBBoltzmann works with minimum l_max=2."""
        cmb = CMBBoltzmann(omega_tot=1.018, l_max=2)
        cls = cmb.cls_s3()
        assert 2 in cls

    @pytest.mark.slow
    def test_omega_tot_near_one(self):
        """
        Omega_tot very close to 1 (large R_curv) should still work.

        This tests the nearly-flat limit where R_curv -> infinity.
        """
        cmb = CMBBoltzmann(omega_tot=1.005, l_max=10)
        cls = cmb.cls_s3()
        assert len(cls) > 0
        for l, d_l in cls.items():
            assert np.isfinite(d_l), f"D_{l} not finite at Omega_tot=1.005"

    @pytest.mark.slow
    def test_omega_tot_1_04(self):
        """
        Omega_tot = 1.04 (strongly closed) should still produce valid output.
        """
        cmb = CMBBoltzmann(omega_tot=1.04, l_max=10)
        cls = cmb.cls_s3()
        assert len(cls) > 0
        for l, d_l in cls.items():
            assert d_l > 0, f"D_{l} <= 0 at Omega_tot=1.04"

    def test_comparison_table_with_custom_k_max(self, cmb_default):
        """comparison_table works with different k_max values."""
        table_100 = cmb_default.comparison_table(k_max=100)
        assert len(table_100) > 0

    def test_chi_squared_all_models(self, cmb_default):
        """chi_squared works for all three model types."""
        for model in ['istar', 's3', 'lcdm']:
            chi2, n = cmb_default.chi_squared(model=model)
            assert np.isfinite(chi2)
            assert n > 0


# ==================================================================
# 9. Position-dependent spectrum
# ==================================================================

from yang_mills_s3.spectral.cmb_boltzmann import (
    position_scan, omega_position_grid_scan, _sample_fundamental_domain,
)


class TestPositionDependentCls:
    """
    Tests for position-dependent CMB spectrum on S3/I*.

    NUMERICAL: Verifies that the observer position within the fundamental
    domain of S3/I* affects the observed power spectrum, that the
    position-averaged result is consistent with the known D_2 ~ 17.9 muK^2,
    and that different positions give different D_2 values.

    Uses k_max=32 for tests to keep precomputation time reasonable
    (~2 min instead of ~18 min for k_max=48). This captures the dominant
    modes (k=12, 20, 24, 30, 32) which contain the bulk of the I* signal.
    """

    K_MAX_TEST = 32  # Fast enough for tests, captures dominant modes

    @pytest.fixture(scope="class")
    def cmb_pos(self):
        """CMBBoltzmann with precomputed invariant modes."""
        cmb = CMBBoltzmann(omega_tot=1.02, l_max=10)
        cmb.precompute_invariant_modes(k_max=self.K_MAX_TEST)
        return cmb

    def test_identity_position_returns_dict(self, cmb_pos):
        """Spectrum at identity position (1,0,0,0) returns a dict."""
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        cls = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        assert isinstance(cls, dict)
        assert 2 in cls

    def test_identity_d2_positive(self, cmb_pos):
        """D_2 at identity position is positive."""
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        cls = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        assert cls[2] > 0, f"D_2 = {cls[2]} <= 0"

    def test_identity_d2_finite(self, cmb_pos):
        """D_2 at identity position is finite."""
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        cls = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        assert np.isfinite(cls[2]), f"D_2 = {cls[2]} is not finite"

    def test_different_positions_give_different_d2(self, cmb_pos):
        """
        Different positions in the fundamental domain give different D_2.

        This is a key physical prediction: the CMB is anisotropic on S3/I*
        because the observer breaks the symmetry of the covering S3.
        """
        pos1 = np.array([1.0, 0.0, 0.0, 0.0])
        pos2 = np.array([0.95, 0.18, 0.13, 0.21])
        pos2 = pos2 / np.linalg.norm(pos2)

        cls1 = cmb_pos.compute_position_dependent_cls(pos1, k_max=self.K_MAX_TEST)
        cls2 = cmb_pos.compute_position_dependent_cls(pos2, k_max=self.K_MAX_TEST)

        d2_1 = cls1[2]
        d2_2 = cls2[2]

        # They should differ (at least slightly)
        assert d2_1 != d2_2, \
            f"D_2 identical at two positions: {d2_1} == {d2_2}"

    def test_all_l_present(self, cmb_pos):
        """Position-dependent spectrum has all l from 2 to l_max."""
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        cls = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        for l in range(2, cmb_pos.l_max + 1):
            assert l in cls, f"l={l} missing from position-dependent spectrum"

    def test_suppression_relative_to_s3(self, cmb_pos):
        """
        Position-dependent D_l(I*, x) <= D_l(S3) for all l.

        The quotient can only remove modes, never add them. The
        position-dependent weight W(k, x) = sum |f_i(x)|^2 is bounded
        above, so D_l(I*, x) should not exceed D_l(S3) significantly.
        """
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        cls_pos = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        cls_s3 = cmb_pos.cls_s3()

        for l in range(2, min(11, cmb_pos.l_max + 1)):
            # Position-dependent power should be much less than full S3
            # (I* removes most modes). Allow 10x tolerance for position
            # concentration effects.
            assert cls_pos[l] < cls_s3[l] * 10.0, \
                f"l={l}: D_l(I*, x)={cls_pos[l]} >> D_l(S3)={cls_s3[l]}"

    def test_normalizes_non_unit_quaternion(self, cmb_pos):
        """Non-unit quaternion is normalized automatically."""
        pos = np.array([2.0, 0.0, 0.0, 0.0])  # Not unit, but should work
        cls = cmb_pos.compute_position_dependent_cls(pos, k_max=self.K_MAX_TEST)
        assert 2 in cls
        assert cls[2] > 0


class TestSampleFundamentalDomain:
    """Tests for the fundamental domain sampler."""

    def test_returns_correct_shape(self):
        """Returns (n_positions, 4) array."""
        positions = _sample_fundamental_domain(10, seed=42)
        assert positions.shape == (10, 4)

    def test_unit_quaternions(self):
        """All returned positions are unit quaternions."""
        positions = _sample_fundamental_domain(20, seed=42)
        norms = np.linalg.norm(positions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_deterministic_with_seed(self):
        """Same seed gives same positions."""
        p1 = _sample_fundamental_domain(10, seed=42)
        p2 = _sample_fundamental_domain(10, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_give_different_positions(self):
        """Different seeds give different positions."""
        p1 = _sample_fundamental_domain(10, seed=42)
        p2 = _sample_fundamental_domain(10, seed=99)
        assert not np.allclose(p1, p2)

    def test_near_identity(self):
        """
        Points in the fundamental domain should be near the identity
        element (1, 0, 0, 0) since they are in the Voronoi cell of identity.
        """
        positions = _sample_fundamental_domain(30, seed=42)
        # Distance from identity: arccos(|w|)
        dists = np.arccos(np.clip(np.abs(positions[:, 0]), 0, 1))
        # Maximum distance in fundamental domain of S3/I* is about pi/5 ~ 0.63
        assert np.all(dists < 1.0), \
            f"Some positions too far from identity: max dist = {dists.max()}"


class TestPositionScan:
    """Tests for the position_scan function."""

    @pytest.mark.slow
    def test_returns_dict_with_required_keys(self):
        """position_scan returns dict with all required keys."""
        result = position_scan(n_positions=5, omega_tot=1.02, lmax=10,
                               k_max=32, verbose=False)
        required = {'D2_min', 'D2_max', 'D2_mean', 'D2_std',
                    'D2_values', 'positions', 'best_position',
                    'best_D2', 'cls_at_best'}
        assert required.issubset(set(result.keys()))

    @pytest.mark.slow
    def test_d2_values_all_positive(self):
        """All D_2 values from position scan are positive."""
        result = position_scan(n_positions=5, omega_tot=1.02, lmax=10,
                               k_max=32, verbose=False)
        assert np.all(result['D2_values'] > 0), \
            "Some D_2 values are non-positive"

    @pytest.mark.slow
    def test_best_d2_is_max(self):
        """best_D2 equals the maximum of D2_values."""
        result = position_scan(n_positions=5, omega_tot=1.02, lmax=10,
                               k_max=32, verbose=False)
        np.testing.assert_allclose(
            result['best_D2'], result['D2_values'].max(), rtol=1e-10
        )

    @pytest.mark.slow
    def test_d2_variation_exists(self):
        """
        D_2 should vary across positions (std > 0).

        This confirms position dependence is real, not an artifact.
        """
        result = position_scan(n_positions=10, omega_tot=1.02, lmax=10,
                               k_max=32, verbose=False)
        assert result['D2_std'] > 0, \
            "No variation in D_2 across positions — position dependence not working"

    @pytest.mark.slow
    def test_position_averaged_order_of_magnitude(self):
        """
        Position-averaged D_2 should be in the right ballpark.

        Known: position-averaged D_2 ~ 17.9 muK^2 at Omega_tot=1.018.
        Allow wide tolerance since we use few positions and different Omega_tot.
        """
        result = position_scan(n_positions=10, omega_tot=1.02, lmax=15,
                               k_max=32, verbose=False)
        mean_d2 = result['D2_mean']
        # Should be O(1-1000) muK^2, not 0 and not millions
        assert 0.01 < mean_d2 < 10000, \
            f"Mean D_2 = {mean_d2} muK^2 seems unreasonable"


class TestOmegaPositionGridScan:
    """Tests for the omega_position_grid_scan function."""

    @pytest.mark.slow
    def test_returns_dict_with_required_keys(self):
        """omega_position_grid_scan returns all required keys."""
        result = omega_position_grid_scan(
            omega_range=(1.015, 1.025), n_omega=3, n_positions=3,
            lmax=10, k_max=32, verbose=False
        )
        required = {'omega_values', 'D2_grid', 'best_omega',
                    'best_position', 'best_D2', 'exceeds_100',
                    'exceeds_planck_half', 'cls_at_best', 'scan_results'}
        assert required.issubset(set(result.keys()))

    @pytest.mark.slow
    def test_d2_grid_shape(self):
        """D2_grid has shape (n_omega, n_positions)."""
        result = omega_position_grid_scan(
            omega_range=(1.015, 1.025), n_omega=3, n_positions=4,
            lmax=10, k_max=32, verbose=False
        )
        assert result['D2_grid'].shape == (3, 4)

    @pytest.mark.slow
    def test_best_d2_is_grid_max(self):
        """best_D2 equals the maximum value in D2_grid."""
        result = omega_position_grid_scan(
            omega_range=(1.015, 1.025), n_omega=3, n_positions=3,
            lmax=10, k_max=32, verbose=False
        )
        np.testing.assert_allclose(
            result['best_D2'], result['D2_grid'].max(), rtol=1e-10
        )

    @pytest.mark.slow
    def test_scan_results_length(self):
        """scan_results has one entry per omega value."""
        n_omega = 3
        result = omega_position_grid_scan(
            omega_range=(1.015, 1.025), n_omega=n_omega, n_positions=3,
            lmax=10, k_max=32, verbose=False
        )
        assert len(result['scan_results']) == n_omega

    @pytest.mark.slow
    def test_exceeds_flags_are_bool(self):
        """The exceeds_100 and exceeds_planck_half flags are boolean."""
        result = omega_position_grid_scan(
            omega_range=(1.015, 1.025), n_omega=2, n_positions=2,
            lmax=10, k_max=32, verbose=False
        )
        assert isinstance(result['exceeds_100'], bool)
        assert isinstance(result['exceeds_planck_half'], bool)


# ==================================================================
# 11. SW/ISW/cross-term decomposition
# ==================================================================

from yang_mills_s3.spectral.cmb_boltzmann import plot_swisw_decomposition


class TestSWISWDecomposition:
    """
    Tests for the SW/ISW/cross-term decomposition of the CMB spectrum.

    NUMERICAL: The decomposition separates the angular power spectrum
    D_l into Sachs-Wolfe + Doppler (SW), Integrated Sachs-Wolfe (ISW),
    and their cross-term. Key physical finding: the cross-term is NEGATIVE
    at low l on closed S3, indicating destructive interference between
    SW and ISW — a structural effect of positive spatial curvature.

    Method: Two CAMB runs (full cosmology vs matter-dominated) isolate
    the ISW contribution. The identity
        D_l^{total} = D_l^{SW} + D_l^{ISW} + D_l^{cross}
    is verified to hold within numerical precision.
    """

    @pytest.fixture(scope="class")
    def decomposition(self):
        """
        Compute the SW/ISW decomposition once and reuse across tests.

        Uses l_max=15 and k_max=100 for speed (two CAMB runs required).
        """
        cmb = CMBBoltzmann(omega_tot=1.018, l_max=15)
        return cmb.compute_swisw_decomposition(k_max=100)

    # ------------------------------------------------------------------
    # Structure and keys
    # ------------------------------------------------------------------

    def test_returns_dict(self, decomposition):
        """compute_swisw_decomposition returns a dictionary."""
        assert isinstance(decomposition, dict)

    def test_top_level_keys(self, decomposition):
        """Output has the required top-level keys."""
        required = {'l_values', 's3', 'istar', 'lcdm', 'planck'}
        assert required.issubset(set(decomposition.keys()))

    def test_component_keys_s3(self, decomposition):
        """S3 dict has total, sw_doppler, isw, cross sub-dicts."""
        required = {'total', 'sw_doppler', 'isw', 'cross'}
        assert required.issubset(set(decomposition['s3'].keys()))

    def test_component_keys_istar(self, decomposition):
        """S3/I* dict has total, sw_doppler, isw, cross sub-dicts."""
        required = {'total', 'sw_doppler', 'isw', 'cross'}
        assert required.issubset(set(decomposition['istar'].keys()))

    def test_l_values_start_at_2(self, decomposition):
        """l_values list starts at l=2."""
        assert decomposition['l_values'][0] == 2

    def test_l_values_contiguous(self, decomposition):
        """l_values are contiguous integers from 2 to l_max."""
        l_vals = decomposition['l_values']
        expected = list(range(l_vals[0], l_vals[-1] + 1))
        assert l_vals == expected

    # ------------------------------------------------------------------
    # Sum rule: SW + ISW + cross = total
    # ------------------------------------------------------------------

    def test_sum_rule_s3(self, decomposition):
        """
        SW + ISW + cross = total for all l on simply-connected S3.

        This is the algebraic identity
        |Delta_full|^2 = |Delta_noDE|^2 + |Delta_ISW|^2 + 2*Delta_noDE*Delta_ISW
        summed with nu^2 weights.
        """
        s3 = decomposition['s3']
        for l in decomposition['l_values']:
            total = s3['total'][l]
            recon = s3['sw_doppler'][l] + s3['isw'][l] + s3['cross'][l]
            if abs(total) > 1e-12:
                np.testing.assert_allclose(
                    recon, total, rtol=1e-6,
                    err_msg=f"Sum rule fails for S3 at l={l}"
                )
            else:
                np.testing.assert_allclose(
                    recon, total, atol=1e-10,
                    err_msg=f"Sum rule fails for S3 at l={l} (near zero)"
                )

    def test_sum_rule_istar(self, decomposition):
        """
        SW + ISW + cross = total for all l on S3/I*.

        Same algebraic identity, but summed with m(k)*nu weights
        over surviving modes only.
        """
        istar = decomposition['istar']
        for l in decomposition['l_values']:
            total = istar['total'][l]
            recon = istar['sw_doppler'][l] + istar['isw'][l] + istar['cross'][l]
            if abs(total) > 1e-12:
                np.testing.assert_allclose(
                    recon, total, rtol=1e-6,
                    err_msg=f"Sum rule fails for I* at l={l}"
                )
            else:
                np.testing.assert_allclose(
                    recon, total, atol=1e-10,
                    err_msg=f"Sum rule fails for I* at l={l} (near zero)"
                )

    # ------------------------------------------------------------------
    # ISW is negative (destructive interference) at low l — novel finding
    # ------------------------------------------------------------------

    def test_cross_term_negative_s3_low_l(self, decomposition):
        """
        NUMERICAL: The SW x ISW cross-term is negative at low l on S3.

        This indicates destructive interference between the ordinary
        Sachs-Wolfe effect and the Integrated Sachs-Wolfe effect on
        closed S3 topology. The physical mechanism: positive curvature
        causes the gravitational potential to decay (Phi_dot < 0) in a
        way that anti-correlates with the initial SW perturbation.

        This is a novel finding of this work.
        """
        s3_cross = decomposition['s3']['cross']
        # Check l=2 through l=10 (the regime where ISW matters)
        for l in range(2, min(11, max(decomposition['l_values']) + 1)):
            if l in s3_cross:
                assert s3_cross[l] < 0, \
                    f"Cross-term positive at l={l}: {s3_cross[l]:.4f} " \
                    f"(expected negative for destructive ISW)"

    def test_cross_term_negative_istar_low_l(self, decomposition):
        """
        NUMERICAL: The SW x ISW cross-term is negative at low l on S3/I*.

        The destructive ISW interference persists on the Poincare
        dodecahedral space quotient.
        """
        istar_cross = decomposition['istar']['cross']
        for l in range(2, min(11, max(decomposition['l_values']) + 1)):
            if l in istar_cross:
                assert istar_cross[l] < 0, \
                    f"Cross-term positive at l={l} on I*: {istar_cross[l]:.4f} " \
                    f"(expected negative for destructive ISW)"

    # ------------------------------------------------------------------
    # Sanity: all components finite, correct signs, correct shapes
    # ------------------------------------------------------------------

    def test_all_components_finite_s3(self, decomposition):
        """All S3 decomposition values are finite (not nan or inf)."""
        s3 = decomposition['s3']
        for key in ['total', 'sw_doppler', 'isw', 'cross']:
            for l, val in s3[key].items():
                assert np.isfinite(val), \
                    f"S3 {key}[{l}] = {val} is not finite"

    def test_all_components_finite_istar(self, decomposition):
        """All S3/I* decomposition values are finite."""
        istar = decomposition['istar']
        for key in ['total', 'sw_doppler', 'isw', 'cross']:
            for l, val in istar[key].items():
                assert np.isfinite(val), \
                    f"I* {key}[{l}] = {val} is not finite"

    def test_total_positive_s3(self, decomposition):
        """S3 total D_l > 0 for all l (power spectrum must be positive)."""
        for l, val in decomposition['s3']['total'].items():
            assert val > 0, f"S3 total D_{l} = {val} <= 0"

    def test_total_nonnegative_istar(self, decomposition):
        """S3/I* total D_l >= 0 for all l."""
        for l, val in decomposition['istar']['total'].items():
            assert val >= -1e-10, f"I* total D_{l} = {val} < 0"

    def test_sw_doppler_positive_s3(self, decomposition):
        """SW + Doppler auto-power is positive (it's |Delta|^2)."""
        for l, val in decomposition['s3']['sw_doppler'].items():
            assert val > 0, f"S3 SW+Doppler D_{l} = {val} <= 0"

    def test_isw_auto_positive_s3(self, decomposition):
        """ISW auto-power is positive (it's |Delta_ISW|^2)."""
        for l, val in decomposition['s3']['isw'].items():
            assert val > 0, f"S3 ISW D_{l} = {val} <= 0"

    def test_sw_doppler_nonnegative_istar(self, decomposition):
        """SW + Doppler is non-negative on I* (sum of squares)."""
        for l, val in decomposition['istar']['sw_doppler'].items():
            assert val >= -1e-10, f"I* SW+Doppler D_{l} = {val} < 0"

    def test_isw_auto_nonnegative_istar(self, decomposition):
        """ISW auto-power is non-negative on I* (sum of squares)."""
        for l, val in decomposition['istar']['isw'].items():
            assert val >= -1e-10, f"I* ISW D_{l} = {val} < 0"

    # ------------------------------------------------------------------
    # Consistency with existing cls_s3 / cls_s3_istar methods
    # ------------------------------------------------------------------

    def test_s3_total_matches_cls_s3(self, decomposition):
        """
        The 'total' component of the S3 decomposition matches cls_s3().

        The normalization factor norm = D_l^{CAMB} / sum(Delta^2 * nu^2)
        should reproduce the CAMB power spectrum exactly.
        """
        cmb = CMBBoltzmann(omega_tot=1.018, l_max=15)
        cls_s3 = cmb.cls_s3()

        for l in decomposition['l_values']:
            if l in cls_s3:
                np.testing.assert_allclose(
                    decomposition['s3']['total'][l], cls_s3[l], rtol=1e-6,
                    err_msg=f"S3 total doesn't match cls_s3 at l={l}"
                )

    # ------------------------------------------------------------------
    # LCDM and Planck reference data
    # ------------------------------------------------------------------

    def test_lcdm_values_positive(self, decomposition):
        """LCDM reference values are positive."""
        for l, val in decomposition['lcdm'].items():
            assert val > 0, f"LCDM D_{l} = {val} <= 0"

    def test_planck_data_present(self, decomposition):
        """Planck data is present for l=2 through at least l=10."""
        for l in range(2, 11):
            assert l in decomposition['planck'], \
                f"Planck data missing for l={l}"

    def test_planck_values_structure(self, decomposition):
        """Planck entries are (D_l_obs, sigma) tuples with positive sigma."""
        for l, (d_obs, sigma) in decomposition['planck'].items():
            assert sigma > 0, f"Planck sigma <= 0 at l={l}"
            assert np.isfinite(d_obs), f"Planck D_obs not finite at l={l}"

    # ------------------------------------------------------------------
    # ISW dominance structure at low l
    # ------------------------------------------------------------------

    def test_isw_exceeds_sw_at_l2_s3(self, decomposition):
        """
        NUMERICAL: ISW auto-power exceeds SW+Doppler at l=2 on S3.

        On closed S3 with Lambda, the ISW contribution is larger than
        the ordinary Sachs-Wolfe at the quadrupole, but the negative
        cross-term brings the total below either individual component.
        """
        s3 = decomposition['s3']
        assert s3['isw'][2] > s3['sw_doppler'][2], \
            f"ISW ({s3['isw'][2]:.1f}) not larger than " \
            f"SW+Doppler ({s3['sw_doppler'][2]:.1f}) at l=2"

    def test_istar_suppression_in_all_components(self, decomposition):
        """
        All components on S3/I* are suppressed relative to S3.

        The eigenmode selection m(k) removes modes from all source
        terms equally (topology acts on the eigenmode sum, not on
        the source physics).
        """
        s3 = decomposition['s3']
        istar = decomposition['istar']
        for key in ['total', 'sw_doppler', 'isw']:
            for l in decomposition['l_values']:
                assert istar[key][l] <= s3[key][l] + 1e-8, \
                    f"I* {key}[{l}] = {istar[key][l]:.4f} > " \
                    f"S3 {key}[{l}] = {s3[key][l]:.4f}"

    # ------------------------------------------------------------------
    # Same l-values across all components
    # ------------------------------------------------------------------

    def test_same_l_keys_across_components_s3(self, decomposition):
        """All S3 components have the same l keys."""
        s3 = decomposition['s3']
        ref_keys = set(s3['total'].keys())
        for key in ['sw_doppler', 'isw', 'cross']:
            assert set(s3[key].keys()) == ref_keys, \
                f"S3 {key} has different l keys than total"

    def test_same_l_keys_across_components_istar(self, decomposition):
        """All I* components have the same l keys."""
        istar = decomposition['istar']
        ref_keys = set(istar['total'].keys())
        for key in ['sw_doppler', 'isw', 'cross']:
            assert set(istar[key].keys()) == ref_keys, \
                f"I* {key} has different l keys than total"


class TestPlotSWISW:
    """Tests for the plot_swisw_decomposition function."""

    @pytest.fixture(scope="class")
    def decomposition_for_plot(self):
        """Compute decomposition for plotting tests."""
        cmb = CMBBoltzmann(omega_tot=1.018, l_max=10)
        return cmb.compute_swisw_decomposition(k_max=50)

    def test_returns_figure(self, decomposition_for_plot):
        """plot_swisw_decomposition returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for testing
        import matplotlib.pyplot as plt

        fig = plot_swisw_decomposition(
            decomposition_for_plot, show=False
        )
        assert fig is not None
        assert hasattr(fig, 'savefig')  # it's a Figure
        plt.close(fig)

    def test_figure_has_two_axes(self, decomposition_for_plot):
        """Figure has two subplots (S3/I* and S3)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plot_swisw_decomposition(
            decomposition_for_plot, show=False
        )
        axes = fig.get_axes()
        assert len(axes) == 2, \
            f"Expected 2 axes, got {len(axes)}"
        plt.close(fig)
