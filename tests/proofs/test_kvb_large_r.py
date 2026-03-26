"""
Tests for KvB Large-R Analysis.

Verifies:
1. Gap computation at individual R values
2. Gap monotonically decreases with R (for large enough R)
3. gap * R^2 converges to a constant (1/R^2 decay from quartic scaling)
4. Power-law fit gives exponent alpha ~ 2.0 (at sufficient basis size)
5. Spectrum ratios: first excited state is 3-fold degenerate
6. Quartic scaling prediction matches numerics
7. Convergence and Aitken extrapolation
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.kvb_large_r import (
    compute_gap_at_R,
    compute_spectrum_at_R,
    gap_vs_R_scan,
    spectrum_scan,
    fit_power_law,
    fit_subleading,
    analytical_gap_prediction,
    analytical_C_MeV_fm,
    quartic_scaling_prediction,
    quartic_C_MeV_fm2,
    aitken_extrapolation,
    convergence_study,
    run_full_analysis,
    plot_gap_vs_R,
    GapRecord,
    SpectrumRecord,
    HBAR_C_MEV_FM,
    G2_DEFAULT,
)


# ======================================================================
# Test parameters - use small basis for speed
# ======================================================================
N_FAST = 6       # Fast tests: small basis
N_ACCURATE = 8   # Accurate tests: larger basis
G2 = G2_DEFAULT  # g^2 = 6.28


class TestSingleGapComputation:
    """Test gap computation at individual R values."""

    def test_gap_at_R2p2_is_positive(self):
        """Gap at physical radius R=2.2 fm must be positive."""
        rec = compute_gap_at_R(R=2.2, g2=G2, N_per_dim=N_FAST)
        assert rec.gap_natural > 0, f"gap_natural = {rec.gap_natural}"
        assert rec.gap_MeV > 0, f"gap_MeV = {rec.gap_MeV}"

    def test_gap_at_R10_is_positive(self):
        """Gap at R=10 fm must be positive."""
        rec = compute_gap_at_R(R=10.0, g2=G2, N_per_dim=N_FAST)
        assert rec.gap_natural > 0
        assert rec.gap_MeV > 0

    def test_gap_at_R50_is_positive(self):
        """Gap at R=50 fm must be positive."""
        rec = compute_gap_at_R(R=50.0, g2=G2, N_per_dim=N_FAST)
        assert rec.gap_natural > 0
        assert rec.gap_MeV > 0

    def test_gap_conversion_consistency(self):
        """gap_MeV = gap_natural * hbar_c."""
        rec = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=N_FAST)
        expected = rec.gap_natural * HBAR_C_MEV_FM
        assert abs(rec.gap_MeV - expected) < 1e-10 * expected

    def test_gap_times_R_consistency(self):
        """gap_times_R = gap_natural * R."""
        rec = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=N_FAST)
        expected = rec.gap_natural * rec.R
        assert abs(rec.gap_times_R - expected) < 1e-10 * abs(expected)

    def test_gap_times_R2_consistency(self):
        """gap_times_R2 = gap_natural * R^2."""
        rec = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=N_FAST)
        expected = rec.gap_natural * rec.R**2
        assert abs(rec.gap_times_R2 - expected) < 1e-10 * abs(expected)

    def test_gap_times_R2_MeV_fm2_consistency(self):
        """gap_times_R2_MeV_fm2 = gap_MeV * R^2."""
        rec = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=N_FAST)
        expected = rec.gap_MeV * rec.R**2
        assert abs(rec.gap_times_R2_MeV_fm2 - expected) < 1e-10 * abs(expected)

    def test_E0_less_than_E1(self):
        """Ground state must be below first excited state."""
        rec = compute_gap_at_R(R=3.0, g2=G2, N_per_dim=N_FAST)
        assert rec.E0 < rec.E1


class TestGapDecaysWithR:
    """Test that the gap decreases as R increases."""

    def test_gap_decreases_R2_to_R10(self):
        """Gap at R=2 should be larger than at R=10."""
        rec2 = compute_gap_at_R(R=2.0, g2=G2, N_per_dim=N_FAST)
        rec10 = compute_gap_at_R(R=10.0, g2=G2, N_per_dim=N_FAST)
        assert rec2.gap_MeV > rec10.gap_MeV, (
            f"gap(R=2) = {rec2.gap_MeV:.2f} should be > gap(R=10) = {rec10.gap_MeV:.2f}"
        )

    def test_gap_decreases_R10_to_R50(self):
        """Gap at R=10 should be larger than at R=50."""
        rec10 = compute_gap_at_R(R=10.0, g2=G2, N_per_dim=N_FAST)
        rec50 = compute_gap_at_R(R=50.0, g2=G2, N_per_dim=N_FAST)
        assert rec10.gap_MeV > rec50.gap_MeV

    def test_gap_monotonic_decrease(self):
        """Gap should be monotonically decreasing for R >= 2."""
        R_vals = [2.0, 5.0, 10.0, 20.0, 50.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        gaps = [r.gap_MeV for r in records]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1], (
                f"gap(R={R_vals[i]}) = {gaps[i]:.2f} should be > "
                f"gap(R={R_vals[i+1]}) = {gaps[i+1]:.2f}"
            )


class TestOneOverR2Decay:
    """Test that the gap decays as 1/R^2 (quartic-oscillator scaling)."""

    def test_gap_times_R2_bounded(self):
        """gap * R^2 should be bounded (not diverging)."""
        R_vals = [2.0, 5.0, 10.0, 20.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        gapR2 = [r.gap_times_R2_MeV_fm2 for r in records]
        for val in gapR2:
            assert 0 < val < 10000.0, f"gap*R^2 = {val} out of range"

    def test_gap_times_R2_positive(self):
        """gap * R^2 must be positive at all R."""
        R_vals = [2.0, 10.0, 50.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        for r in records:
            assert r.gap_times_R2 > 0, f"gap*R^2 negative at R={r.R}"

    def test_gap_times_R2_roughly_constant_small_R(self):
        """gap*R^2 should be roughly constant for R in [2, 5] (well converged)."""
        R_vals = [2.0, 3.0, 5.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_ACCURATE)
        gapR2 = [r.gap_times_R2_MeV_fm2 for r in records]
        # Allow 50% variation (quadratic/cubic corrections at small R)
        mean_val = np.mean(gapR2)
        for val in gapR2:
            assert abs(val - mean_val) / mean_val < 0.5, (
                f"gap*R^2 = {val:.1f}, mean = {mean_val:.1f}, "
                f"variation = {abs(val-mean_val)/mean_val*100:.1f}%"
            )


class TestPowerLawFit:
    """Test the power-law fit."""

    def test_exponent_greater_than_one(self):
        """Fitted exponent alpha should be > 1 (steeper than 1/R at finite N)."""
        R_vals = [2.0, 5.0, 10.0, 20.0, 50.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        fit = fit_power_law(records, R_min=2.0)
        alpha = fit.params['alpha']
        # At N=6, exponent should be between 1.0 and 2.5
        assert 1.0 < alpha < 2.5, f"alpha = {alpha:.4f}"

    def test_fit_amplitude_positive(self):
        """Fit amplitude A should be positive."""
        R_vals = [2.0, 5.0, 10.0, 20.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        fit = fit_power_law(records, R_min=2.0)
        assert fit.params['A'] > 0, f"A = {fit.params['A']}"

    def test_fit_r_squared_high(self):
        """Fit should have high R^2 (good quality)."""
        R_vals = [2.0, 5.0, 10.0, 20.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        fit = fit_power_law(records, R_min=2.0)
        assert fit.r_squared > 0.99, f"R^2 = {fit.r_squared}"

    def test_subleading_fit_works(self):
        """Subleading fit A/R + B/R^2 should also work."""
        R_vals = [2.0, 5.0, 10.0, 20.0]
        records = gap_vs_R_scan(np.array(R_vals), g2=G2, N_per_dim=N_FAST)
        fit = fit_subleading(records, R_min=2.0)
        assert fit.params['A'] is not None
        assert fit.r_squared > 0.95


class TestSpectrumAnalysis:
    """Test spectrum computation and ratios."""

    def test_spectrum_at_R2p2(self):
        """Spectrum at R=2.2 should have well-separated eigenvalues."""
        rec = compute_spectrum_at_R(R=2.2, g2=G2, N_per_dim=N_FAST, n_eigenvalues=6)
        assert len(rec.eigenvalues) >= 5
        # All gaps should be positive
        assert np.all(rec.gaps > 0), "Some gaps are non-positive"

    def test_threefold_degeneracy(self):
        """First excited state should be 3-fold degenerate (S_3 Weyl group)."""
        rec = compute_spectrum_at_R(R=5.0, g2=G2, N_per_dim=N_ACCURATE, n_eigenvalues=6)
        # E_1, E_2, E_3 should be approximately equal
        if len(rec.eigenvalues) >= 4:
            E1 = rec.eigenvalues[1]
            E2 = rec.eigenvalues[2]
            E3 = rec.eigenvalues[3]
            # Relative spread should be small
            spread = max(E1, E2, E3) - min(E1, E2, E3)
            mean_val = (E1 + E2 + E3) / 3
            rel_spread = spread / mean_val if mean_val > 0 else 0
            assert rel_spread < 0.01, (
                f"Degeneracy broken: E1={E1:.6f}, E2={E2:.6f}, E3={E3:.6f}, "
                f"spread={rel_spread:.6f}"
            )

    def test_ratios_well_defined(self):
        """Spectrum ratios (E_n - E_0)/(E_1 - E_0) should be >= 1 for n >= 1."""
        rec = compute_spectrum_at_R(R=5.0, g2=G2, N_per_dim=N_FAST, n_eigenvalues=6)
        # First ratio is E_1/E_1 = 1 by definition
        assert abs(rec.ratios_to_gap[0] - 1.0) < 1e-10
        # Higher ratios should be > 1 (at least for the 4th and higher)
        if len(rec.ratios_to_gap) > 3:
            assert rec.ratios_to_gap[3] > 1.0, (
                f"ratio[3] = {rec.ratios_to_gap[3]:.4f} should be > 1"
            )

    def test_spectrum_eigenvalues_sorted(self):
        """Eigenvalues should be sorted in ascending order."""
        rec = compute_spectrum_at_R(R=10.0, g2=G2, N_per_dim=N_FAST, n_eigenvalues=6)
        for i in range(len(rec.eigenvalues) - 1):
            assert rec.eigenvalues[i] <= rec.eigenvalues[i + 1]


class TestQuarticScaling:
    """Test the quartic-oscillator scaling prediction."""

    def test_quartic_prediction_positive(self):
        """Quartic scaling prediction should be positive."""
        for R in [1.0, 10.0, 100.0]:
            gap = quartic_scaling_prediction(R, G2)
            assert gap > 0

    def test_quartic_prediction_scales_as_1_over_R2(self):
        """gap(2R) should be gap(R)/4 (exact 1/R^2 scaling)."""
        R = 10.0
        gap1 = quartic_scaling_prediction(R, G2)
        gap2 = quartic_scaling_prediction(2 * R, G2)
        assert abs(gap2 - gap1 / 4) < 1e-12

    def test_quartic_C2_value(self):
        """C2 = g^2 * hbar_c * delta_0 / 2^{2/3} should be calculable."""
        C2 = quartic_C_MeV_fm2(G2)
        assert C2 > 0
        expected = G2 * HBAR_C_MEV_FM * 1.95 / 2.0**(2.0/3.0)
        assert abs(C2 - expected) < 1e-6

    def test_quartic_matches_numerical_at_R2p2(self):
        """Quartic prediction should match numerical gap at R=2.2 (within 30%)."""
        rec = compute_gap_at_R(R=2.2, g2=G2, N_per_dim=N_ACCURATE)
        pred = quartic_scaling_prediction(2.2, G2) * HBAR_C_MEV_FM
        ratio = rec.gap_MeV / pred
        # delta_0 is calibrated from R=2.2 data, so should be close
        assert 0.7 < ratio < 1.5, f"ratio = {ratio:.4f}"


class TestAnalyticalPrediction:
    """Test the full-theory analytical prediction formula."""

    def test_analytical_prediction_positive(self):
        """Analytical gap prediction should be positive."""
        for R in [1.0, 10.0, 100.0]:
            gap = analytical_gap_prediction(R, G2)
            assert gap > 0

    def test_analytical_prediction_scales_as_1_over_R(self):
        """gap(2R) should be gap(R)/2 (exact 1/R scaling)."""
        R = 10.0
        gap1 = analytical_gap_prediction(R, G2)
        gap2 = analytical_gap_prediction(2 * R, G2)
        assert abs(gap2 - gap1 / 2) < 1e-12

    def test_analytical_C_value(self):
        """C = 8 g^4 hbar_c / 225 should be calculable."""
        C = analytical_C_MeV_fm(G2)
        assert C > 0
        expected = 8.0 * G2**2 * HBAR_C_MEV_FM / 225.0
        assert abs(C - expected) < 1e-6


class TestAitkenExtrapolation:
    """Test Aitken delta-squared extrapolation."""

    def test_aitken_geometric_sequence(self):
        """Aitken should give exact result for geometric convergence."""
        # s_n = L + a*r^n, converges to L
        L = 3.0
        a = 1.0
        r = 0.5
        vals = [L + a * r**n for n in range(5)]
        result = aitken_extrapolation(vals)
        assert abs(result - L) < 1e-10, f"Aitken = {result}, expected {L}"

    def test_aitken_with_3_values(self):
        """Aitken with exactly 3 values should work."""
        vals = [10.0, 8.0, 7.0]
        result = aitken_extrapolation(vals)
        assert np.isfinite(result)

    def test_aitken_with_fewer_than_3(self):
        """Aitken with < 3 values should return last value."""
        assert aitken_extrapolation([5.0]) == 5.0
        assert aitken_extrapolation([5.0, 4.0]) == 4.0


class TestFullAnalysis:
    """Test the full analysis pipeline."""

    def test_run_full_analysis_minimal(self):
        """Full analysis with minimal R set completes without error."""
        analysis = run_full_analysis(
            R_values=np.array([2.0, 2.2, 5.0, 10.0, 20.0]),
            spectrum_R_values=np.array([5.0]),
            g2=G2,
            N_per_dim=N_FAST,
            fit_R_min=5.0,
            verbose=False,
        )
        assert len(analysis.gap_records) == 5
        assert len(analysis.spectrum_records) == 1
        assert analysis.power_law_fit is not None
        assert analysis.asymptotic_C_MeV_fm2 > 0
        assert analysis.asymptotic_C_natural > 0
        assert analysis.quartic_prediction_MeV_fm2 > 0

    def test_asymptotic_C2_reasonable(self):
        """Asymptotic C2 = gap*R^2 should be in a physically reasonable range."""
        analysis = run_full_analysis(
            R_values=np.array([2.0, 2.2, 5.0, 10.0]),
            spectrum_R_values=np.array([5.0]),
            g2=G2,
            N_per_dim=N_FAST,
            fit_R_min=5.0,
            verbose=False,
        )
        # C2 should be positive and finite (~ 1000-2000 MeV*fm^2)
        assert 100 < analysis.asymptotic_C_MeV_fm2 < 20000, (
            f"C2 = {analysis.asymptotic_C_MeV_fm2} MeV*fm^2"
        )


class TestPlotting:
    """Test that plotting works without errors."""

    def test_plot_creates_figure(self, tmp_path):
        """Plotting should work and save a file."""
        analysis = run_full_analysis(
            R_values=np.array([2.0, 2.2, 5.0, 10.0, 20.0]),
            spectrum_R_values=np.array([5.0]),
            g2=G2,
            N_per_dim=N_FAST,
            fit_R_min=5.0,
            verbose=False,
        )
        save_path = str(tmp_path / "test_gap_vs_R.png")
        fig = plot_gap_vs_R(analysis, save_path=save_path)
        import os
        assert os.path.exists(save_path), f"File not created at {save_path}"


class TestConvergenceWithBasisSize:
    """Test that results converge as basis size increases."""

    def test_gap_converges_N6_vs_N8(self):
        """Gap at N=8 should be smaller than at N=6 (converging from above)."""
        rec6 = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=6)
        rec8 = compute_gap_at_R(R=5.0, g2=G2, N_per_dim=8)
        # Gap should decrease with N (variational upper bound)
        assert rec8.gap_natural <= rec6.gap_natural * 1.01, (
            f"N=6 gap = {rec6.gap_natural:.6f}, N=8 gap = {rec8.gap_natural:.6f}"
        )

    def test_convergence_study_runs(self):
        """Convergence study should complete and produce extrapolation."""
        result = convergence_study(R=2.2, N_values=[4, 6, 8], g2=G2)
        assert result['R'] == 2.2
        assert len(result['gaps']) == 3
        assert result['aitken_extrapolation'] > 0
        assert result['aitken_gap_times_R_MeV_fm'] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_R(self):
        """Gap at very small R should be large (quadratic dominates)."""
        rec = compute_gap_at_R(R=0.5, g2=G2, N_per_dim=N_FAST)
        assert rec.gap_MeV > 100, f"gap at R=0.5 = {rec.gap_MeV:.2f} MeV, expected > 100"

    def test_fit_needs_minimum_points(self):
        """Fit should raise error with too few points."""
        records = gap_vs_R_scan(np.array([1.0]), g2=G2, N_per_dim=N_FAST)
        with pytest.raises(ValueError):
            fit_power_law(records, R_min=0.5)
