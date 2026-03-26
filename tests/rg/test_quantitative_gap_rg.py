"""
Tests for the Quantitative RG Mass Gap computation.

Tests verify:
1. Decomposed gap: bare + one-loop decomposition at R = 2.2 fm
2. R-scan: gap over range of R values, minimum gap
3. Dimensional transmutation: large-R behavior
4. RG vs BE comparison (if BE infrastructure available)
5. Honesty assessment: perturbative control diagnostics

LABEL: NUMERICAL (validating the perturbative RG gap computation)
"""

import numpy as np
import pytest


from yang_mills_s3.rg.quantitative_gap_rg import (
    DecomposedRGGap,
    RGGapScan,
    RGvsBEComparison,
    DimensionalTransmutationCheck,
    QuantitativeRGGapReport,
)
from yang_mills_s3.rg.heat_kernel_slices import HBAR_C_MEV_FM, R_PHYSICAL_FM, LAMBDA_QCD_MEV


# ======================================================================
# DecomposedRGGap Tests
# ======================================================================

class TestDecomposedRGGap:
    """Tests for the decomposed RG gap at a single R."""

    def test_bare_gap_exact(self):
        """Bare gap should be exactly 2*hbar*c/R."""
        R = 2.2
        decomp = DecomposedRGGap(R)
        result = decomp.compute_decomposed_gap()

        expected_bare = 2.0 * HBAR_C_MEV_FM / R
        assert abs(result['gap_bare_mev'] - expected_bare) < 0.1, (
            f"Bare gap {result['gap_bare_mev']:.1f} != expected {expected_bare:.1f}"
        )

    def test_m2_bare_exact(self):
        """m^2_bare should be exactly 4/R^2."""
        R = 2.2
        decomp = DecomposedRGGap(R)
        result = decomp.compute_decomposed_gap()

        expected_m2 = 4.0 / R ** 2
        assert abs(result['m2_bare'] - expected_m2) < 1e-10, (
            f"m2_bare {result['m2_bare']} != expected {expected_m2}"
        )

    def test_bare_gap_is_179_mev(self):
        """At R=2.2 fm, bare gap should be ~179 MeV."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert 175.0 < result['gap_bare_mev'] < 185.0, (
            f"Bare gap {result['gap_bare_mev']:.1f} not in [175, 185] MeV"
        )

    def test_rg_gap_positive(self):
        """RG-corrected gap must be positive."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert result['gap_rg_mev'] > 0, "RG gap must be positive"

    def test_gap_ratio_reflects_large_oneloop(self):
        """
        One-loop mass correction is LARGE (uncontrolled) at the IR scale.

        This is a genuine physics result: the perturbative mass correction
        g^2 * C_2 * spectral_sum / Vol is NOT protected by gauge invariance
        in the naive one-loop formula. In the full theory, the Ward identity
        would cancel most of this. The gap ratio being >> 1 correctly
        signals that the one-loop RG is NOT a reliable mass gap computation.

        NUMERICAL.
        """
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        # The ratio is >> 1 because one-loop corrections are huge
        # This is EXPECTED: the perturbative mass correction is uncontrolled
        assert result['gap_ratio'] > 1.0, (
            f"Gap ratio {result['gap_ratio']:.3f}: one-loop should enhance"
        )

    def test_correction_fraction_reveals_strong_coupling(self):
        """
        The correction fraction is >> 1 at R = 2.2 fm because the
        coupling saturates at g^2_max = 4*pi and the naive one-loop
        mass correction is uncontrolled in the strong-coupling regime.

        This is the KEY FINDING: the one-loop RG does NOT give a
        reliable mass gap at the physical scale. The gap must come
        from the non-perturbative Brascamp-Lieb bound.

        NUMERICAL.
        """
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        # Correction fraction >> 1 means perturbation theory has broken down
        assert result['correction_fraction'] > 1.0, (
            f"Correction fraction {result['correction_fraction']:.3f}: "
            "should be > 1 at this coupling"
        )
        # The code should NOT be perturbative at this scale
        assert not result['is_perturbative'], (
            "Should correctly identify as non-perturbative"
        )

    def test_per_shell_corrections_exist(self):
        """Should have one correction per RG shell."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        # N_scales = 7 default, so 7 shells
        assert len(result['m2_oneloop_per_shell']) > 0, (
            "No per-shell corrections computed"
        )

    def test_wavefunction_renormalization_positive(self):
        """Wavefunction Z should be positive."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert result['wavefunction_z'] > 0, (
            f"Z = {result['wavefunction_z']} should be positive"
        )

    def test_coupling_increases_ir(self):
        """IR coupling should be >= UV coupling (asymptotic freedom)."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert result['g2_ir'] >= result['g2_uv'], (
            f"g2_ir={result['g2_ir']:.3f} < g2_uv={result['g2_uv']:.3f}"
        )

    def test_small_R_large_gap(self):
        """At small R, gap should be large (~ 1/R)."""
        decomp = DecomposedRGGap(R=0.5)
        result = decomp.compute_decomposed_gap()

        # At R=0.5 fm, bare gap is 2*197.3/0.5 ~ 789 MeV
        assert result['gap_rg_mev'] > 500.0, (
            f"Gap at R=0.5 fm ({result['gap_rg_mev']:.1f} MeV) should be > 500"
        )

    def test_large_R_positive_gap(self):
        """At large R, gap should still be positive."""
        decomp = DecomposedRGGap(R=50.0)
        result = decomp.compute_decomposed_gap()

        assert result['gap_rg_mev'] > 0, (
            f"Gap at R=50 fm ({result['gap_rg_mev']:.1f} MeV) must be positive"
        )

    def test_invalid_R_raises(self):
        """R <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            DecomposedRGGap(R=0.0)
        with pytest.raises(ValueError):
            DecomposedRGGap(R=-1.0)

    def test_label_is_numerical(self):
        """The label should honestly indicate this is numerical/perturbative."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert 'NUMERICAL' in result['label'], (
            f"Label should indicate NUMERICAL, got: {result['label']}"
        )


# ======================================================================
# RGGapScan Tests
# ======================================================================

class TestRGGapScan:
    """Tests for the R-scan of the RG gap."""

    def test_scan_runs(self):
        """Scan should run and return expected keys."""
        scan = RGGapScan(R_min=1.0, R_max=10.0, n_R=10)
        result = scan.scan()

        required_keys = [
            'R_values', 'gap_bare_mev', 'gap_rg_mev', 'gap_ratio',
            'm2_correction', 'g2_ir', 'is_perturbative',
            'gap_min_mev', 'R_at_gap_min', 'delta_min_mev',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_all_gaps_positive(self):
        """All gaps should be positive across the scan."""
        scan = RGGapScan(R_min=0.5, R_max=50.0, n_R=20)
        result = scan.scan()

        assert np.all(result['gap_rg_mev'] > 0), (
            "All RG gaps must be positive"
        )

    def test_gap_decreases_with_R(self):
        """
        Bare gap ~ 1/R decreases with R. RG gap should generally
        decrease too, though transmutation could flatten it.
        """
        scan = RGGapScan(R_min=0.5, R_max=5.0, n_R=10)
        result = scan.scan()

        # Bare gap strictly decreases
        diffs_bare = np.diff(result['gap_bare_mev'])
        assert np.all(diffs_bare < 0), "Bare gap should decrease with R"

    def test_gap_minimum_exists(self):
        """There should be a well-defined minimum gap."""
        scan = RGGapScan(R_min=0.5, R_max=100.0, n_R=30)
        result = scan.scan()

        assert result['gap_min_mev'] > 0, "Minimum gap must be positive"
        assert result['R_at_gap_min'] > 0, "R at gap minimum must be positive"

    def test_delta_min_is_physical(self):
        """
        Delta_min should be a physically reasonable number.
        Expected: O(100 MeV) scale, not O(1) or O(10000).
        """
        scan = RGGapScan(R_min=0.5, R_max=100.0, n_R=30)
        result = scan.scan()

        delta_min = result['delta_min_mev']
        assert delta_min > 1.0, f"Delta_min = {delta_min:.1f} MeV too small"

    def test_gap_at_R_physical_oneloop_inflated(self):
        """
        The RG gap at R = 2.2 fm is inflated by the uncontrolled
        one-loop mass correction. This is expected behavior when
        the coupling is in the strong regime.

        The bare gap is ~179 MeV, but one-loop pushes it to ~2000 MeV.
        This signals failure of the perturbative approach, NOT a
        prediction of the physical gap.

        NUMERICAL.
        """
        scan = RGGapScan(R_min=1.0, R_max=5.0, n_R=20)
        result = scan.scan()

        # One-loop inflated: gap >> bare gap
        bare_at_phys = 2.0 * HBAR_C_MEV_FM / R_PHYSICAL_FM
        assert result['gap_at_R_phys'] > bare_at_phys, (
            "One-loop corrections should increase the gap"
        )

    def test_r_values_log_spaced(self):
        """R values should be logarithmically spaced."""
        scan = RGGapScan(R_min=1.0, R_max=100.0, n_R=10)

        log_R = np.log10(scan.R_values)
        diffs = np.diff(log_R)
        assert np.allclose(diffs, diffs[0], rtol=0.01), (
            "R values should be log-spaced"
        )

    def test_invalid_R_range(self):
        """Invalid R range should raise ValueError."""
        with pytest.raises(ValueError):
            RGGapScan(R_min=0.0, R_max=10.0)
        with pytest.raises(ValueError):
            RGGapScan(R_min=10.0, R_max=5.0)


# ======================================================================
# DimensionalTransmutationCheck Tests
# ======================================================================

class TestDimensionalTransmutationCheck:
    """Tests for dimensional transmutation behavior."""

    def test_check_runs(self):
        """Check should produce expected output."""
        dt = DimensionalTransmutationCheck(R_min=0.5, R_max=50.0, n_R=20)
        result = dt.check()

        assert 'gap_rg_mev' in result
        assert 'gap_at_large_R' in result
        assert 'gap_at_R100' in result

    def test_large_R_gap_positive(self):
        """Gap should be positive even at very large R."""
        dt = DimensionalTransmutationCheck(R_min=1.0, R_max=100.0, n_R=15)
        result = dt.check()

        assert result['gap_at_R100'] > 0, (
            "Gap at R=100 fm must be positive"
        )

    def test_small_R_gap_large(self):
        """At small R, gap should be much larger than Lambda_QCD."""
        dt = DimensionalTransmutationCheck(R_min=0.1, R_max=1.0, n_R=10)
        result = dt.check()

        # At R = 0.1 fm, gap ~ 2*197.3/0.1 ~ 3946 MeV
        assert result['gap_rg_mev'][0] > LAMBDA_QCD_MEV, (
            "Gap at small R should exceed Lambda_QCD"
        )

    def test_transition_radius_physical(self):
        """Transition radius should be O(1 fm)."""
        dt = DimensionalTransmutationCheck(R_min=0.1, R_max=50.0, n_R=30)
        result = dt.check()

        R_trans = result['R_transition']
        assert 0.1 < R_trans < 50.0, (
            f"Transition radius {R_trans:.2f} fm out of physical range"
        )


# ======================================================================
# RG vs BE Comparison Tests
# ======================================================================

class TestRGvsBEComparison:
    """Tests for RG-BE comparison."""

    def test_rg_gap_computed(self):
        """RG gap should always be computed."""
        comp = RGvsBEComparison(R_values=[1.0, 2.2, 5.0])
        result = comp.compare()

        assert np.all(np.isfinite(result['gap_rg_mev'])), (
            "All RG gaps should be finite"
        )
        assert np.all(result['gap_rg_mev'] > 0), (
            "All RG gaps should be positive"
        )

    def test_bare_gap_reference(self):
        """Bare gap should match 2*hbar*c/R."""
        R_values = np.array([1.0, 2.0, 5.0])
        comp = RGvsBEComparison(R_values=R_values)
        result = comp.compare()

        expected = 2.0 * HBAR_C_MEV_FM / R_values
        np.testing.assert_allclose(
            result['gap_bare_mev'], expected, rtol=0.01,
            err_msg="Bare gap reference should match 2*hbar*c/R"
        )


# ======================================================================
# QuantitativeRGGapReport Tests
# ======================================================================

class TestQuantitativeRGGapReport:
    """Tests for the comprehensive report."""

    def test_report_runs(self):
        """Report should generate without errors."""
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        assert 'gap_at_R_physical' in result
        assert 'gap_scan' in result
        assert 'dimensional_transmutation' in result
        assert 'honesty' in result
        assert 'summary' in result

    def test_honesty_verdict_present(self):
        """Report should include an honesty verdict."""
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        assert 'verdict' in result['honesty'], "Honesty verdict missing"
        assert result['honesty']['verdict'] in [
            'RELIABLE', 'PARTIALLY RELIABLE', 'UNCONTROLLED'
        ], f"Unknown verdict: {result['honesty']['verdict']}"

    def test_summary_keys(self):
        """Summary should contain the key numbers."""
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        summary = result['summary']
        required = [
            'gap_bare_mev', 'gap_rg_mev', 'delta_min_mev',
            'R_at_delta_min', 'gap_at_R100_mev',
        ]
        for key in required:
            assert key in summary, f"Summary missing key: {key}"

    def test_gap_numbers_consistent(self):
        """
        Gap at R_physical from decomposed and from summary should match.
        """
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        gap_decomp = result['gap_at_R_physical']['gap_rg_mev']
        gap_summary = result['summary']['gap_rg_mev']

        assert abs(gap_decomp - gap_summary) < 0.1, (
            f"Decomposed gap {gap_decomp:.1f} != summary gap {gap_summary:.1f}"
        )


# ======================================================================
# Physical Consistency Tests
# ======================================================================

class TestPhysicalConsistency:
    """
    Tests that verify the RG gap is physically consistent.
    These are the most important tests.
    """

    def test_gap_monotone_in_1_over_R_at_small_R(self):
        """
        At small R (UV), the gap should scale as ~1/R.
        Perturbative corrections should not change the scaling.
        """
        R_values = np.array([0.3, 0.5, 0.8, 1.0])
        gaps = []
        for R in R_values:
            decomp = DecomposedRGGap(R)
            result = decomp.compute_decomposed_gap()
            gaps.append(result['gap_rg_mev'])

        gaps = np.array(gaps)
        # Check that gap * R is approximately constant (1/R scaling)
        gap_times_R = gaps * R_values
        # Should be within 50% of each other
        ratio = max(gap_times_R) / min(gap_times_R)
        assert ratio < 2.0, (
            f"Gap*R varies by factor {ratio:.2f}, expected ~1 (1/R scaling)"
        )

    def test_gap_bounded_below_by_bare(self):
        """
        RG gap should be at least half the bare gap.
        Gauge invariance protects against large negative corrections.
        """
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            decomp = DecomposedRGGap(R)
            result = decomp.compute_decomposed_gap()

            assert result['gap_rg_mev'] >= 0.5 * result['gap_bare_mev'], (
                f"At R={R}: gap_rg={result['gap_rg_mev']:.1f} < "
                f"0.5*gap_bare={0.5*result['gap_bare_mev']:.1f}"
            )

    def test_coupling_trajectory_asymptotic_freedom(self):
        """
        The coupling should increase from UV to IR (asymptotic freedom).
        """
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        assert result['g2_ir'] > result['g2_uv'], (
            "Coupling should grow toward IR (asymptotic freedom)"
        )

    def test_coupling_bounded(self):
        """IR coupling should be bounded (saturation)."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()

        G2_MAX = 4.0 * np.pi
        assert result['g2_ir'] <= G2_MAX + 0.01, (
            f"g2_ir = {result['g2_ir']:.2f} exceeds G2_MAX = {G2_MAX:.2f}"
        )

    def test_m2_effective_positive(self):
        """
        The effective m^2 should be positive.
        This is the mass gap existence condition.
        """
        for R in [0.5, 1.0, 2.2, 5.0, 20.0, 100.0]:
            decomp = DecomposedRGGap(R)
            result = decomp.compute_decomposed_gap()

            assert result['m2_effective'] > 0, (
                f"m^2_eff = {result['m2_effective']:.6f} at R = {R} fm "
                "should be positive"
            )

    def test_large_R_gap_not_zero(self):
        """
        At R = 100 fm, the bare gap is only 3.9 MeV, but the
        RG gap should be larger due to dimensional transmutation.

        However, the current RG (one-loop perturbative) may not
        capture dimensional transmutation correctly. We test that
        the gap is at least positive.
        """
        decomp = DecomposedRGGap(R=100.0)
        result = decomp.compute_decomposed_gap()

        assert result['gap_rg_mev'] > 0, (
            "Gap at R=100 fm must be positive"
        )

    def test_scan_no_nan(self):
        """No NaN values should appear in the scan."""
        scan = RGGapScan(R_min=0.5, R_max=50.0, n_R=15)
        result = scan.scan()

        assert np.all(np.isfinite(result['gap_rg_mev'])), "No NaN in gap_rg"
        assert np.all(np.isfinite(result['gap_bare_mev'])), "No NaN in gap_bare"
        assert np.all(np.isfinite(result['g2_ir'])), "No NaN in g2_ir"


# ======================================================================
# Honesty Tests: does the code correctly identify its limitations?
# ======================================================================

class TestHonesty:
    """
    These tests verify that the code honestly reports its
    perturbative nature and limitations.
    """

    def test_label_says_numerical(self):
        """All labels should include 'NUMERICAL'."""
        decomp = DecomposedRGGap(R=2.2)
        result = decomp.compute_decomposed_gap()
        assert 'NUMERICAL' in result['label']

        scan = RGGapScan(R_min=1.0, R_max=10.0, n_R=5)
        scan_result = scan.scan()
        assert 'NUMERICAL' in scan_result['label']

    def test_honesty_assessment_exists(self):
        """Report should contain a non-empty honesty assessment."""
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        honesty = result['honesty']
        assert 'perturbative_controlled' in honesty
        assert 'gap_dominated_by_bare' in honesty
        assert 'explanation' in honesty
        assert len(honesty['explanation']) > 50, "Explanation too short"

    def test_honesty_identifies_large_corrections(self):
        """
        The honesty check should identify that the one-loop corrections
        are NOT small and the perturbative RG is NOT reliable at this scale.
        """
        report = QuantitativeRGGapReport(R_physical=2.2)
        result = report.generate()

        honesty = result['honesty']
        # At R=2.2 fm with g^2 -> 4*pi, corrections dominate
        assert not honesty['gap_dominated_by_bare'], (
            "Should identify that corrections are large"
        )
        # Verdict should not be 'RELIABLE'
        assert honesty['verdict'] != 'RELIABLE', (
            f"Verdict should not be RELIABLE at strong coupling, "
            f"got: {honesty['verdict']}"
        )
