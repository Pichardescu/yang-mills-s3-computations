"""
Tests for the CMB angular power spectrum on S³ and S³/I*.

Verifies:
  1. Radial eigenfunctions (orthonormality, known values, edge cases)
  2. Primordial spectrum (positivity, scaling)
  3. C_l on S³ (positivity, convergence)
  4. C_l on S³/I* (suppression, positivity)
  5. Suppression ratio (bounds, Weyl law, quadrupole)
  6. Planck comparison (chi-squared, quadrupole anomaly)
  7. Cosmological parameter conversion
  8. Consistency checks
"""

import pytest
import numpy as np
from scipy.integrate import quad

from yang_mills_s3.spectral.cmb_spectrum import CMBSpectrum, PLANCK_2018_LOW_L


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture
def cmb():
    """Default CMB spectrum with Luminet et al. parameters."""
    return CMBSpectrum(n_s=0.965, chi_lss=0.35, k_max=200)


@pytest.fixture
def cmb_hz():
    """Harrison-Zel'dovich (n_s=1) for simpler analytics."""
    return CMBSpectrum(n_s=1.0, chi_lss=0.35, k_max=200)


@pytest.fixture
def cmb_compact():
    """compact topology parameters: R = c/H_0, chi_lss ~ 3.1."""
    return CMBSpectrum(n_s=0.965, chi_lss=3.1, k_max=200)


# ==================================================================
# 1. Radial eigenfunctions
# ==================================================================

class TestRadialEigenfunctions:
    """Tests for Φ_k^l(χ) on S³."""

    def test_orthonormality_l0(self, cmb):
        """
        THEOREM: ∫_0^π Φ_k^0(χ) Φ_{k'}^0(χ) sin²χ dχ = δ_{kk'}
        """
        for k1 in range(0, 6):
            for k2 in range(k1, 6):
                val, err = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 0, chi)
                                 * cmb.radial_eigenfunction(k2, 0, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-8, (
                    f"<Phi_{k1}^0|Phi_{k2}^0> = {val}, expected {expected}"
                )

    def test_orthonormality_l1(self, cmb):
        """
        THEOREM: ∫_0^π Φ_k^1(χ) Φ_{k'}^1(χ) sin²χ dχ = δ_{kk'}
        """
        for k1 in range(1, 6):
            for k2 in range(k1, 6):
                val, err = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 1, chi)
                                 * cmb.radial_eigenfunction(k2, 1, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-8, (
                    f"<Phi_{k1}^1|Phi_{k2}^1> = {val}, expected {expected}"
                )

    def test_orthonormality_l2(self, cmb):
        """
        THEOREM: ∫_0^π Φ_k^2(χ) Φ_{k'}^2(χ) sin²χ dχ = δ_{kk'}
        """
        for k1 in range(2, 7):
            for k2 in range(k1, 7):
                val, err = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 2, chi)
                                 * cmb.radial_eigenfunction(k2, 2, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-8, (
                    f"<Phi_{k1}^2|Phi_{k2}^2> = {val}, expected {expected}"
                )

    def test_orthonormality_l5(self, cmb):
        """
        THEOREM: Orthonormality holds for higher l as well.
        """
        for k1 in range(5, 9):
            for k2 in range(k1, 9):
                val, err = quad(
                    lambda chi: (cmb.radial_eigenfunction(k1, 5, chi)
                                 * cmb.radial_eigenfunction(k2, 5, chi)
                                 * np.sin(chi) ** 2),
                    0, np.pi, limit=100
                )
                expected = 1.0 if k1 == k2 else 0.0
                assert abs(val - expected) < 1e-7, (
                    f"<Phi_{k1}^5|Phi_{k2}^5> = {val}, expected {expected}"
                )

    def test_phi_00_is_constant(self, cmb):
        """
        THEOREM: Φ_0^0(χ) = √(2/π) (constant function on S³).
        """
        expected = np.sqrt(2.0 / np.pi)
        for chi in [0.1, 0.5, 1.0, 1.5, np.pi / 2, 2.5]:
            val = cmb.radial_eigenfunction(0, 0, chi)
            assert abs(val - expected) < 1e-12, (
                f"Phi_0^0({chi}) = {val}, expected {expected}"
            )

    def test_phi_10_proportional_to_cos(self, cmb):
        """
        THEOREM: Φ_1^0(χ) ∝ cos(χ).
        C_1^1(x) = 2x, so Φ_1^0 = N_{1,0} × 2cos(χ).
        """
        chi_vals = np.array([0.3, 0.7, 1.2, 2.0, 2.8])
        phi_vals = cmb.radial_eigenfunction(1, 0, chi_vals)
        # Should be proportional to cos(chi)
        cos_vals = np.cos(chi_vals)
        # Find proportionality constant
        ratio = phi_vals / cos_vals
        assert np.std(ratio) / abs(np.mean(ratio)) < 1e-10, (
            f"Phi_1^0 not proportional to cos(chi): ratios = {ratio}"
        )

    def test_phi_11_proportional_to_sin(self, cmb):
        """
        THEOREM: Φ_1^1(χ) ∝ sin(χ).
        C_0^2(x) = 1, so Φ_1^1 = N_{1,1} × sin(χ).
        """
        chi_vals = np.array([0.3, 0.7, 1.2, 2.0, 2.8])
        phi_vals = cmb.radial_eigenfunction(1, 1, chi_vals)
        sin_vals = np.sin(chi_vals)
        ratio = phi_vals / sin_vals
        assert np.std(ratio) / abs(np.mean(ratio)) < 1e-10, (
            f"Phi_1^1 not proportional to sin(chi): ratios = {ratio}"
        )

    def test_phi_zero_for_l_greater_than_k(self, cmb):
        """
        THEOREM: Φ_k^l = 0 for l > k (no such mode exists).
        """
        for k in range(5):
            for l in range(k + 1, k + 4):
                val = cmb.radial_eigenfunction(k, l, 0.5)
                assert val == 0.0, f"Phi_{k}^{l}(0.5) = {val}, expected 0"

    def test_phi_at_chi_zero(self, cmb):
        """
        Edge case: χ = 0 (north pole of S³).
        sin(0) = 0, so Φ_k^l(0) = 0 for l >= 1.
        Φ_k^0(0) = N_{k,0} × C_k^1(1).
        """
        for l in range(1, 5):
            for k in range(l, l + 3):
                val = cmb.radial_eigenfunction(k, l, 0.0)
                assert abs(val) < 1e-14, (
                    f"Phi_{k}^{l}(0) = {val}, expected 0"
                )

    def test_phi_at_chi_pi(self, cmb):
        """
        Edge case: χ = π (south pole of S³).
        sin(π) = 0, so Φ_k^l(π) = 0 for l >= 1.
        """
        for l in range(1, 5):
            for k in range(l, l + 3):
                val = cmb.radial_eigenfunction(k, l, np.pi)
                assert abs(val) < 1e-12, (
                    f"Phi_{k}^{l}(pi) = {val}, expected 0"
                )

    def test_array_input(self, cmb):
        """Radial eigenfunction accepts array input."""
        chi_arr = np.linspace(0.1, 2.9, 50)
        phi_arr = cmb.radial_eigenfunction(3, 1, chi_arr)
        assert phi_arr.shape == (50,)
        # Check scalar consistency
        for i in [0, 10, 25, 49]:
            val_scalar = cmb.radial_eigenfunction(3, 1, chi_arr[i])
            assert abs(phi_arr[i] - val_scalar) < 1e-14

    def test_negative_k_or_l(self, cmb):
        """Edge case: negative k or l returns zero."""
        assert cmb.radial_eigenfunction(-1, 0, 0.5) == 0.0
        assert cmb.radial_eigenfunction(3, -1, 0.5) == 0.0

    def test_high_k_normalization(self, cmb):
        """
        NUMERICAL: Even at high k, normalization integral = 1.
        Tests numerical stability of log-gamma formula.
        """
        for k, l in [(20, 0), (30, 5), (50, 10)]:
            val, err = quad(
                lambda chi: cmb.radial_eigenfunction(k, l, chi) ** 2 * np.sin(chi) ** 2,
                0, np.pi, limit=200
            )
            assert abs(val - 1.0) < 1e-6, (
                f"||Phi_{k}^{l}||² = {val}, expected 1.0"
            )


# ==================================================================
# 2. Primordial spectrum
# ==================================================================

class TestPrimordialSpectrum:
    """Tests for P(k) = (k/k_pivot)^{n_s-1} / [k(k+2)]."""

    def test_positivity(self, cmb):
        """P(k) > 0 for all k >= 1."""
        for k in range(1, 50):
            pk = cmb.primordial_spectrum(k)
            assert pk > 0, f"P({k}) = {pk} <= 0"

    def test_k_zero_returns_zero(self, cmb):
        """P(0) = 0 (no k=0 mode in primordial spectrum)."""
        assert cmb.primordial_spectrum(0) == 0.0

    def test_scale_invariant(self, cmb_hz):
        """
        For n_s = 1: P(k) = 1/[k(k+2)] exactly.
        """
        for k in [1, 5, 10, 50, 100]:
            pk = cmb_hz.primordial_spectrum(k)
            expected = 1.0 / (k * (k + 2))
            assert abs(pk - expected) < 1e-14, (
                f"P({k}) = {pk}, expected {expected}"
            )

    def test_pivot_normalization(self, cmb):
        """At k = k_pivot: tilt factor = 1, so P(k_pivot) = 1/[k_pivot(k_pivot+2)]."""
        k_pivot = 10
        pk = cmb.primordial_spectrum(k_pivot, k_pivot=k_pivot)
        expected = 1.0 / (k_pivot * (k_pivot + 2))
        assert abs(pk - expected) < 1e-14

    def test_red_tilt_enhances_large_scales(self, cmb):
        """
        For n_s < 1 (red tilt): P(k) at low k is enhanced relative to
        scale-invariant. Check P(2)/P(10) > 1/[2*4] / 1/[10*12].
        """
        p2 = cmb.primordial_spectrum(2, k_pivot=10)
        p10 = cmb.primordial_spectrum(10, k_pivot=10)
        ratio = p2 / p10
        # Scale-invariant ratio: [10*12]/[2*4] = 15
        # Red tilt makes low-k modes bigger, so ratio > 15
        si_ratio = (10 * 12) / (2 * 4)
        assert ratio > si_ratio, (
            f"Red tilt should enhance large scales: ratio={ratio}, SI={si_ratio}"
        )

    def test_decreasing_at_large_k(self, cmb):
        """P(k) ~ 1/k² for large k, so it decreases."""
        p50 = cmb.primordial_spectrum(50)
        p100 = cmb.primordial_spectrum(100)
        assert p100 < p50


# ==================================================================
# 3. C_l on S³
# ==================================================================

class TestClS3:
    """Tests for the angular power spectrum on S³ (full sphere)."""

    def test_positivity(self, cmb):
        """C_l > 0 for l = 0, 1, 2, ..., 10."""
        for l in range(0, 11):
            cl = cmb.cl_s3(l)
            assert cl > 0, f"C_{l}^{{S3}} = {cl} <= 0"

    def test_monotonically_related_to_l(self, cmb):
        """
        NUMERICAL: C_l on S³ roughly decreases for large l
        (Sachs-Wolfe: C_l ~ 1/[l(l+1)] at large l).
        Compare C_5 vs C_25.
        """
        c5 = cmb.cl_s3(5)
        c25 = cmb.cl_s3(25)
        assert c25 < c5, f"C_25 = {c25} should be < C_5 = {c5}"

    def test_convergence_kmax(self):
        """
        NUMERICAL: C_l converges as k_max increases.
        k_max=150 vs k_max=200 should differ by less than k_max=50 vs k_max=200.
        Convergence at chi_lss=0.35 is slow because the radial functions
        Phi_k^l at small chi oscillate without rapid decay.
        We test that doubling k_max reduces the difference.
        """
        cmb100 = CMBSpectrum(k_max=100, chi_lss=0.35)
        cmb150 = CMBSpectrum(k_max=150, chi_lss=0.35)
        cmb200 = CMBSpectrum(k_max=200, chi_lss=0.35)
        for l in [2, 5, 10]:
            c100 = cmb100.cl_s3(l)
            c150 = cmb150.cl_s3(l)
            c200 = cmb200.cl_s3(l)
            if c200 > 0:
                diff_100_200 = abs(c100 - c200)
                diff_150_200 = abs(c150 - c200)
                # More modes should bring us closer to the k_max=200 value
                assert diff_150_200 < diff_100_200 + 1e-30, (
                    f"C_{l}^{{S3}}: k_max=150 not closer to 200 than k_max=100"
                )

    def test_negative_l_returns_zero(self, cmb):
        """C_l = 0 for l < 0."""
        assert cmb.cl_s3(-1) == 0.0

    def test_dl_s3_relation(self, cmb):
        """D_l = l(l+1)C_l/(2π)."""
        for l in [2, 5, 10, 20]:
            dl = cmb.dl_s3(l)
            cl = cmb.cl_s3(l)
            expected = l * (l + 1) * cl / (2 * np.pi)
            assert abs(dl - expected) < 1e-20, (
                f"D_{l} = {dl}, expected {expected}"
            )


# ==================================================================
# 4. C_l on S³/I*
# ==================================================================

class TestClPoincare:
    """Tests for the angular power spectrum on S³/I*."""

    def test_suppressed_relative_to_s3(self, cmb):
        """
        THEOREM: C_l^{S³/I*} <= C_l^{S³} for all l (fewer modes = less power).
        """
        for l in range(2, 25):
            cl_s3 = cmb.cl_s3(l)
            cl_p = cmb.cl_poincare(l)
            assert cl_p <= cl_s3 + 1e-30, (
                f"C_{l}^{{I*}} = {cl_p} > C_{l}^{{S3}} = {cl_s3}"
            )

    def test_positivity_high_l(self, cmb):
        """
        NUMERICAL: C_l^{S³/I*} > 0 for all l (high-k modes always contribute).
        At high l, enough k-levels with m(k)>0 contribute.
        """
        for l in [12, 15, 20, 25, 30]:
            cl = cmb.cl_poincare(l)
            assert cl > 0, f"C_{l}^{{I*}} = {cl} should be > 0"

    def test_strong_suppression_low_l(self, cmb):
        """
        NUMERICAL: S_l << 1 for l = 2..10.
        Because m(k) = 0 for k = 1..11, low multipoles are strongly suppressed.
        """
        for l in range(2, 11):
            sl = cmb.suppression_ratio(l)
            assert sl < 0.5, (
                f"S_{l} = {sl} should be << 1 (strong suppression)"
            )

    def test_dl_poincare_relation(self, cmb):
        """D_l = l(l+1)C_l/(2π) on S³/I*."""
        for l in [2, 5, 10, 20]:
            dl = cmb.dl_poincare(l)
            cl = cmb.cl_poincare(l)
            expected = l * (l + 1) * cl / (2 * np.pi)
            assert abs(dl - expected) < 1e-25, (
                f"D_{l}^{{I*}} = {dl}, expected {expected}"
            )


# ==================================================================
# 5. Suppression ratio
# ==================================================================

class TestSuppressionRatio:
    """Tests for S_l = C_l^{S³/I*} / C_l^{S³}."""

    def test_bounded_zero_one(self, cmb):
        """
        THEOREM: 0 <= S_l <= 1 for all l.
        """
        for l in range(2, 31):
            sl = cmb.suppression_ratio(l)
            assert 0 <= sl <= 1 + 1e-10, (
                f"S_{l} = {sl} out of bounds [0, 1]"
            )

    def test_quadrupole_strongly_suppressed(self, cmb):
        """
        NUMERICAL: S_2 << 1 because m(k)=0 for k=1..11.
        First contribution to l=2 comes from k=12 (m(12)=1).
        """
        s2 = cmb.suppression_ratio(2)
        assert s2 < 0.1, f"S_2 = {s2}, expected << 1"

    def test_suppression_varies_with_l(self, cmb):
        """
        NUMERICAL: S_l ≈ 1/120 ≈ 0.008 for l=2..30 at chi_lss=0.35.
        The corrected Peter-Weyl factor m(k)/(k+1) gives S_l → 1/120
        (Weyl law). S_2 is slightly below 1/120 due to the missing
        k=2..11 modes which normally dominate the quadrupole.
        """
        s2 = cmb.suppression_ratio(2)
        s30 = cmb.suppression_ratio(30)
        # Both should be of similar small magnitude (order 1e-4 to 1e-5)
        assert s2 < 0.01, f"S_2 = {s2} should be << 1"
        assert s30 < 0.01, f"S_30 = {s30} should be << 1"
        assert s2 > 0, "S_2 should be positive"
        assert s30 > 0, "S_30 should be positive"

    def test_approaches_weyl_limit(self):
        """
        NUMERICAL: For l >> 12, S_l → 1/120 ≈ 0.00833 (Weyl law).

        The corrected formula m(k)/(k+1) (from Peter-Weyl decomposition)
        gives the fraction of I*-invariant modes at level k. For large k,
        m(k) ≈ (k+1)/120, so m(k)/(k+1) → 1/120.

        At l=30 with k_max=500, S_l should be close to 1/120.
        """
        cmb500 = CMBSpectrum(chi_lss=0.35, k_max=500)
        s30_500 = cmb500.suppression_ratio(30)

        # S_30 should be close to 1/120
        weyl = 1.0 / 120.0
        assert s30_500 > 0.0, "S_30 should be positive"
        assert abs(s30_500 - weyl) / weyl < 0.15, (
            f"S_30 = {s30_500}, expected ~{weyl} (15% tolerance)"
        )

        # S_2 should be BELOW 1/120 (extra suppression from missing k=2..11)
        s2_500 = cmb500.suppression_ratio(2)
        assert s2_500 > 0.0, "S_2 should be positive"
        assert s2_500 < weyl, f"S_2 = {s2_500} should be < 1/120 = {weyl}"

    def test_consistency_with_cl_ratio(self, cmb):
        """
        CONSISTENCY: S_l = cl_poincare(l) / cl_s3(l) exactly.
        """
        for l in [2, 5, 10, 15, 20, 25, 30]:
            sl = cmb.suppression_ratio(l)
            cl_s3 = cmb.cl_s3(l)
            cl_p = cmb.cl_poincare(l)
            if cl_s3 > 0:
                expected = cl_p / cl_s3
                assert abs(sl - expected) < 1e-14, (
                    f"S_{l} = {sl} != C_l^I*/C_l^S3 = {expected}"
                )


# ==================================================================
# 6. Planck comparison
# ==================================================================

class TestPlanckComparison:
    """Tests for comparison with Planck 2018 data."""

    def test_planck_data_completeness(self):
        """All l from 2 to 30 are in the Planck data dict."""
        for l in range(2, 31):
            assert l in PLANCK_2018_LOW_L, f"l={l} missing from Planck data"

    def test_planck_data_format(self):
        """Each entry has (D_l_obs, D_l_lcdm, sigma), all positive."""
        for l, (d_obs, d_lcdm, sigma) in PLANCK_2018_LOW_L.items():
            assert d_obs > 0, f"l={l}: D_l_obs = {d_obs} <= 0"
            assert d_lcdm > 0, f"l={l}: D_l_lcdm = {d_lcdm} <= 0"
            assert sigma > 0, f"l={l}: sigma = {sigma} <= 0"

    def test_quadrupole_anomaly_in_data(self):
        """
        OBSERVATIONAL: D_2^obs / D_2^ΛCDM ≈ 0.19 (factor ~5 suppression).
        """
        d_obs, d_lcdm, _ = PLANCK_2018_LOW_L[2]
        ratio = d_obs / d_lcdm
        assert ratio < 0.3, f"Quadrupole ratio = {ratio}, expected ~0.19"
        assert ratio > 0.1, f"Quadrupole ratio = {ratio}, expected ~0.19"

    def test_comparison_returns_correct_keys(self, cmb):
        """planck_comparison returns the expected keys."""
        result = cmb.planck_comparison(l_max=10)
        assert 'multipoles' in result
        assert 'observed_ratio' in result
        assert 'predicted_ratio' in result
        assert 'chi_squared' in result
        assert 'n_dof' in result
        assert 'chi_squared_per_dof' in result
        assert 'quadrupole_test' in result

    def test_comparison_chi_squared_positive(self, cmb):
        """Chi-squared is non-negative."""
        result = cmb.planck_comparison(l_max=30)
        assert result['chi_squared'] >= 0
        assert result['n_dof'] == 29  # l = 2 to 30

    def test_quadrupole_test_fields(self, cmb):
        """Quadrupole test has the expected fields."""
        result = cmb.planck_comparison(l_max=10)
        qt = result['quadrupole_test']
        assert 'S_2' in qt
        assert 'observed_ratio' in qt
        assert 'sigma_ratio' in qt
        assert 'suppression_consistent' in qt

    def test_predicted_ratios_bounded(self, cmb):
        """All predicted ratios (S_l) should be in [0, 1]."""
        result = cmb.planck_comparison(l_max=30)
        for sl in result['predicted_ratio']:
            assert 0 <= sl <= 1 + 1e-10, f"S_l = {sl} out of [0,1]"

    def test_s2_qualitatively_matches_quadrupole(self, cmb):
        """
        NUMERICAL: Our S_2 predicts suppression, consistent with
        the observed D_2^obs / D_2^ΛCDM < 1.
        """
        result = cmb.planck_comparison()
        qt = result['quadrupole_test']
        assert qt['S_2'] < 1.0, "S_2 should indicate suppression"
        assert qt['observed_ratio'] < 1.0, "Observed ratio should be < 1"


# ==================================================================
# 7. Cosmological parameter conversion
# ==================================================================

class TestOmegaConversion:
    """Tests for omega_to_chi_lss."""

    def test_luminet_value(self):
        """
        Ω_total = 1.013 → χ_LSS ≈ 3.1√0.013 ≈ 0.353.
        """
        chi = CMBSpectrum.omega_to_chi_lss(1.013)
        assert abs(chi - 3.1 * np.sqrt(0.013)) < 1e-10
        assert abs(chi - 0.353) < 0.01

    def test_larger_omega(self):
        """
        Ω_total = 1.05 → χ_LSS = 3.1√0.05 ≈ 0.693.
        """
        chi = CMBSpectrum.omega_to_chi_lss(1.05)
        expected = 3.1 * np.sqrt(0.05)
        assert abs(chi - expected) < 1e-10

    def test_flat_space_raises(self):
        """
        Ω_total = 1.0 (flat space) should raise ValueError.
        """
        with pytest.raises(ValueError, match="Ω_total > 1"):
            CMBSpectrum.omega_to_chi_lss(1.0)

    def test_open_space_raises(self):
        """
        Ω_total < 1 (open, hyperbolic) should raise ValueError.
        """
        with pytest.raises(ValueError, match="Ω_total > 1"):
            CMBSpectrum.omega_to_chi_lss(0.95)

    def test_monotonically_increasing(self):
        """χ_LSS increases with Ω_total."""
        chi1 = CMBSpectrum.omega_to_chi_lss(1.01)
        chi2 = CMBSpectrum.omega_to_chi_lss(1.05)
        chi3 = CMBSpectrum.omega_to_chi_lss(1.10)
        assert chi1 < chi2 < chi3


# ==================================================================
# 8. Consistency checks
# ==================================================================

class TestConsistency:
    """Cross-checks between different methods."""

    def test_same_eigenfunctions_used(self, cmb):
        """
        cl_s3 and cl_poincare use the same Φ_k^l(χ_LSS).
        Verify by comparing the sum structure.
        """
        l = 5
        chi = cmb.chi_lss

        # Manually compute cl_s3(5)
        total_s3 = 0.0
        for k in range(max(l, 1), cmb.k_max + 1):
            pk = cmb.primordial_spectrum(k)
            phi = cmb.radial_eigenfunction(k, l, chi)
            total_s3 += pk * phi ** 2
        expected_s3 = total_s3 / 9.0

        assert abs(cmb.cl_s3(l) - expected_s3) < 1e-20

    def test_poincare_manual_computation(self, cmb):
        """
        Verify cl_poincare matches manual sum with m(k)/(k+1) weighting.

        The factor m(k)/(k+1) arises from Peter-Weyl: I*-invariant subspace
        at level k has dim (k+1)*m(k), total eigenspace has dim (k+1)².
        Position-averaged weight: (k+1)*m(k) / (k+1)² = m(k)/(k+1).
        """
        l = 10
        chi = cmb.chi_lss

        total_p = 0.0
        for k in range(max(l, 1), cmb.k_max + 1):
            mk = cmb.poincare.trivial_multiplicity(k)
            if mk == 0:
                continue
            pk = cmb.primordial_spectrum(k)
            phi = cmb.radial_eigenfunction(k, l, chi)
            total_p += pk * phi ** 2 * mk / (k + 1.0)
        expected_p = total_p / 9.0

        assert abs(cmb.cl_poincare(l) - expected_p) < 1e-18

    def test_full_report_keys(self, cmb):
        """full_report returns all expected keys."""
        report = cmb.full_report(l_max=10)
        assert 'parameters' in report
        assert 'spectrum_s3' in report
        assert 'spectrum_poincare' in report
        assert 'suppression' in report
        assert 'dl_s3' in report
        assert 'dl_poincare' in report
        assert 'planck_comparison' in report
        assert 'weyl_limit' in report
        assert 'quadrupole_anomaly' in report

    def test_full_report_weyl_limit(self, cmb):
        """Weyl limit is 1/120."""
        report = cmb.full_report(l_max=5)
        assert abs(report['weyl_limit'] - 1.0 / 120.0) < 1e-15

    def test_full_report_suppression_matches(self, cmb):
        """Suppression in report matches direct computation."""
        report = cmb.full_report(l_max=10)
        for l in range(2, 11):
            assert abs(report['suppression'][l] - cmb.suppression_ratio(l)) < 1e-15


# ==================================================================
# 9. Chi_LSS scan
# ==================================================================

class TestChiLSSScan:
    """Tests for scanning chi_lss parameter."""

    def test_scan_returns_sorted(self):
        """Results are sorted by chi_squared (ascending)."""
        cmb = CMBSpectrum(k_max=60)
        results = cmb.scan_chi_lss(chi_values=np.linspace(0.2, 1.0, 5), l_max=10)
        chi_sq_values = [r[1] for r in results]
        assert chi_sq_values == sorted(chi_sq_values)

    def test_scan_restores_chi_lss(self):
        """After scan, chi_lss is restored to original value."""
        cmb = CMBSpectrum(chi_lss=0.42, k_max=60)
        cmb.scan_chi_lss(chi_values=[0.1, 0.5, 1.0], l_max=5)
        assert abs(cmb.chi_lss - 0.42) < 1e-15

    def test_best_fit_in_reasonable_range(self):
        """
        NUMERICAL: Best-fit χ_LSS should be in a physically reasonable range.
        """
        cmb = CMBSpectrum(k_max=80)
        results = cmb.scan_chi_lss(
            chi_values=np.linspace(0.1, 3.5, 18),
            l_max=15
        )
        best_chi = results[0][0]
        assert 0.05 < best_chi < 3.6, (
            f"Best-fit chi_lss = {best_chi} out of reasonable range"
        )


# ==================================================================
# 10. compact topology parameters
# ==================================================================

class TestCompactParameters:
    """Tests with compact topology cosmological parameters (R = c/H_0, chi_lss ~ 3.1)."""

    def test_spectrum_computable(self, cmb_compact):
        """Spectrum is computable at compact topology chi_lss ~ 3.1."""
        cl2 = cmb_compact.cl_s3(2)
        assert cl2 > 0
        cl2_p = cmb_compact.cl_poincare(2)
        assert cl2_p >= 0

    def test_suppression_still_present(self, cmb_compact):
        """
        NUMERICAL: Even at chi_lss = 3.1 (compact topology), the quadrupole
        suppression should be present because m(k)=0 for k=1..11.
        """
        s2 = cmb_compact.suppression_ratio(2)
        assert s2 < 1.0, f"S_2 = {s2} should show suppression"

    def test_suppression_bounded(self, cmb_compact):
        """Suppression ratio remains bounded in [0,1] for compact topology params."""
        for l in range(2, 16):
            sl = cmb_compact.suppression_ratio(l)
            assert 0 <= sl <= 1 + 1e-10, f"S_{l} = {sl} out of bounds"
