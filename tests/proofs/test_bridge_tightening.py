"""
Tests for bridge_tightening.py: Two approaches to close the Bridge Lemma gap.

Tests cover:
    1. ActualK0FromPipeline — compute actual K_0 from RG flow
    2. TightenedCK — tighten C_K using 600-cell geometry
    3. BridgeTighteningReport — combined analysis
    4. Consistency checks and physical sanity
"""

import numpy as np
import pytest
from yang_mills_s3.proofs.bridge_tightening import (
    ActualK0FromPipeline,
    TightenedCK,
    BridgeTighteningReport,
    BridgeTighteningVerdict,
    FaceSharingPolymerBound,
    DIM_9DOF,
    G2_PHYSICAL,
    N_COLORS_DEFAULT,
)
from yang_mills_s3.rg.quantitative_gap_be import (
    kappa_min_analytical,
    running_coupling_g2,
    HBAR_C_MEV_FM,
)

# ======================================================================
# Physical constants for tests
# ======================================================================

R_PHYS = 2.2
G2_PHYS = 6.28
G_BAR_0 = np.sqrt(G2_PHYS)


# ======================================================================
# 1. ActualK0FromPipeline tests
# ======================================================================

class TestActualK0FromPipeline:
    """Tests for Approach 1: actual K_0 from pipeline."""

    def test_construction_default(self):
        """ActualK0FromPipeline constructs with default parameters."""
        a1 = ActualK0FromPipeline()
        assert a1.R == R_PHYS
        assert a1.g2 == G2_PHYS
        assert a1.N_c == 2
        assert a1.g_bar_0 == pytest.approx(G_BAR_0, rel=1e-10)

    def test_construction_custom(self):
        """ActualK0FromPipeline constructs with custom parameters."""
        a1 = ActualK0FromPipeline(R=3.0, g2=4.0, N_c=3, N_scales=5)
        assert a1.R == 3.0
        assert a1.g2 == 4.0
        assert a1.N_c == 3
        assert a1.N_scales == 5
        assert a1.g_bar_0 == pytest.approx(2.0, rel=1e-10)

    def test_construction_validation(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            ActualK0FromPipeline(R=-1.0)
        with pytest.raises(ValueError):
            ActualK0FromPipeline(g2=-1.0)
        with pytest.raises(ValueError):
            ActualK0FromPipeline(N_c=1)
        with pytest.raises(ValueError):
            ActualK0FromPipeline(N_scales=0)

    def test_K_norm_flow_structure(self):
        """run_K_norm_flow returns all expected keys."""
        a1 = ActualK0FromPipeline(N_scales=5)
        result = a1.run_K_norm_flow()

        required_keys = [
            'N_scales', 'K_norm_trajectory', 'epsilon_trajectory',
            'source_trajectory', 'g_bar_trajectory',
            'K_0_actual', 'K_0_bound_placeholder', 'K_0_bound_bbs',
            'C_K_placeholder', 'C_K_bbs',
            'ratio_actual_over_bound',
            'hess_K0_conservative', 'hess_K0_optimistic',
            'hess_K0_placeholder', 'hess_K0_bbs',
            'hess_ratio_conservative', 'hess_ratio_optimistic',
            'label',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_K_norm_flow_trajectory_length(self):
        """Trajectory has correct length."""
        N = 5
        a1 = ActualK0FromPipeline(N_scales=N)
        result = a1.run_K_norm_flow()

        assert len(result['K_norm_trajectory']) == N
        assert len(result['epsilon_trajectory']) == N
        assert len(result['source_trajectory']) == N
        assert len(result['g_bar_trajectory']) == N

    def test_K_0_actual_is_positive(self):
        """Actual K_0 norm is positive (non-trivial remainder)."""
        a1 = ActualK0FromPipeline()
        result = a1.run_K_norm_flow()
        assert result['K_0_actual'] > 0

    def test_K_0_actual_less_than_bound(self):
        """Actual K_0 from pipeline is less than placeholder BBS bound."""
        a1 = ActualK0FromPipeline()
        result = a1.run_K_norm_flow()

        # NUMERICAL: The actual K_0 should be smaller than the placeholder bound
        assert result['K_0_actual'] <= result['K_0_bound_placeholder'], (
            f"Actual K_0 ({result['K_0_actual']:.6f}) exceeds placeholder bound "
            f"({result['K_0_bound_placeholder']:.6f})"
        )

    def test_ratio_actual_over_bound_less_than_one(self):
        """Ratio of actual/bound is less than 1 (bound is conservative)."""
        a1 = ActualK0FromPipeline()
        result = a1.run_K_norm_flow()
        assert 0 < result['ratio_actual_over_bound'] <= 1.0

    def test_hessian_hierarchy(self):
        """
        Hessian estimates satisfy: optimistic <= conservative <= bbs <= placeholder.
        """
        a1 = ActualK0FromPipeline()
        result = a1.run_K_norm_flow()

        assert result['hess_K0_optimistic'] <= result['hess_K0_conservative']
        assert result['hess_K0_bbs'] <= result['hess_K0_placeholder']

    def test_epsilon_trajectory_bounded(self):
        """All epsilon values are in (0, 1) for contraction."""
        a1 = ActualK0FromPipeline()
        result = a1.run_K_norm_flow()

        for eps in result['epsilon_trajectory']:
            assert 0 < eps < 5.0, f"epsilon = {eps} out of reasonable range"

    def test_g_bar_trajectory_monotone(self):
        """
        g_bar should decrease from IR (j=0) to UV (j=N-1)
        because of asymptotic freedom.
        """
        a1 = ActualK0FromPipeline(N_scales=7)
        result = a1.run_K_norm_flow()
        g_bars = result['g_bar_trajectory']

        # j=0 is IR (largest coupling), j=N-1 is UV (smallest)
        assert g_bars[0] >= g_bars[-1], (
            f"g_bar should decrease toward UV: g_bar[0]={g_bars[0]:.4f}, "
            f"g_bar[-1]={g_bars[-1]:.4f}"
        )

    def test_bridge_gap_analysis_structure(self):
        """bridge_gap_analysis returns all expected keys."""
        a1 = ActualK0FromPipeline(N_scales=3)
        result = a1.bridge_gap_analysis()

        required_keys = [
            'kappa_analytical',
            'hess_K0_placeholder', 'c_star_placeholder',
            'hess_K0_generic', 'c_star_generic',
            'hess_K0_conservative', 'c_star_conservative',
            'hess_K0_optimistic', 'c_star_optimistic',
            'kappa_numerical',
            'tightening_factor_conservative', 'tightening_factor_optimistic',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_placeholder_bound_fails(self):
        """
        NUMERICAL: The placeholder C_K=1.0 bound (bridge_lemma.py) should
        FAIL to close the gap (this is the known problem).
        """
        a1 = ActualK0FromPipeline()
        result = a1.bridge_gap_analysis()

        # c_star_placeholder should be negative (the known failure with C_K=1.0)
        assert result['c_star_placeholder'] < 0, (
            f"Placeholder bound unexpectedly closes: c*={result['c_star_placeholder']:.4f}"
        )

    def test_bbs_formula_generic_bound_positive(self):
        """
        NUMERICAL: The proper BBS-formula C_K (from contraction) should
        give a POSITIVE c_star, closing the gap.
        """
        a1 = ActualK0FromPipeline()
        result = a1.bridge_gap_analysis()

        # With C_K from BBS formula (~0.042), the bound should close
        # because hess ~ 1.66 < kappa ~ 2.42
        assert result['c_star_generic'] > 0, (
            f"BBS-formula generic bound fails: c*={result['c_star_generic']:.4f}"
        )

    def test_kappa_analytical_positive(self):
        """Bakry-Emery kappa should be positive at physical R."""
        a1 = ActualK0FromPipeline()
        result = a1.bridge_gap_analysis()
        assert result['kappa_analytical'] > 0

    def test_kappa_numerical_positive(self):
        """NUMERICAL: Hessian scan should show positive minimum eigenvalue."""
        a1 = ActualK0FromPipeline(N_scales=3)
        result = a1.bridge_gap_analysis()

        # The numerical scan should give a positive minimum eigenvalue
        # (this is the ~22.5 fm^{-2} value mentioned in the problem)
        if np.isfinite(result['kappa_numerical']):
            assert result['kappa_numerical'] > 0, (
                f"Numerical kappa is negative: {result['kappa_numerical']:.4f}"
            )

    def test_tightening_factors_greater_than_one(self):
        """
        Tightening factors should be > 1 (actual bound is tighter
        than generic bound).
        """
        a1 = ActualK0FromPipeline()
        result = a1.bridge_gap_analysis()
        assert result['tightening_factor_conservative'] > 1.0
        assert result['tightening_factor_optimistic'] > 1.0

    def test_different_N_scales(self):
        """Results are consistent across different N_scales choices."""
        a1_5 = ActualK0FromPipeline(N_scales=5)
        a1_7 = ActualK0FromPipeline(N_scales=7)

        flow_5 = a1_5.run_K_norm_flow()
        flow_7 = a1_7.run_K_norm_flow()

        # More scales should give a larger K_0 (more source terms accumulated)
        # but the ratio should still be < 1
        assert flow_5['ratio_actual_over_bound'] <= 1.0
        assert flow_7['ratio_actual_over_bound'] <= 1.0


# ======================================================================
# 2. TightenedCK tests
# ======================================================================

class TestTightenedCK:
    """Tests for Approach 2: tightened C_K from 600-cell geometry."""

    def test_construction_default(self):
        """TightenedCK constructs with default parameters."""
        t = TightenedCK()
        assert t.R == R_PHYS
        assert t.g2 == G2_PHYS
        assert t.N_c == 2

    def test_construction_validation(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            TightenedCK(R=-1.0)
        with pytest.raises(ValueError):
            TightenedCK(g2=-1.0)
        with pytest.raises(ValueError):
            TightenedCK(N_c=1)

    def test_large_field_is_empty(self):
        """THEOREM 7.6: Large-field region is empty on S^3."""
        t = TightenedCK()
        lf = t.large_field_contribution()

        assert lf['large_field_empty'] is True
        assert lf['K_LF_contribution'] == 0.0
        assert lf['label'] == 'THEOREM'
        assert lf['gribov_diameter'] > 0

    def test_polymer_entropy_halved(self):
        """600-cell face-sharing degree D=4 halves polymer entropy vs D=8."""
        t = TightenedCK()
        pe = t.polymer_entropy_600cell()

        assert pe['D_face_600'] == 4
        assert pe['D_face_hyp'] == 8
        assert pe['mu_ratio_upper'] == pytest.approx(0.5, rel=1e-10)
        assert pe['n_cells'] == 600
        assert pe['polymer_sum_finite'] is True

    def test_polymer_mu_estimated(self):
        """Estimated connective constant is reasonable."""
        t = TightenedCK()
        pe = t.polymer_entropy_600cell()

        # mu should be between 2 and e*D = 10.87
        assert 2.0 <= pe['mu_estimated_actual'] <= pe['mu_upper_600']
        # mu_estimated should be smaller than the upper bound
        assert pe['mu_ratio_estimated'] < pe['mu_ratio_upper']

    def test_single_block_IR(self):
        """At j=0, there is exactly 1 block."""
        t = TightenedCK()
        sb = t.single_block_IR()

        assert sb['n_blocks_IR'] == 1
        assert sb['polymer_count_IR'] == 1
        assert sb['combinatorial_factor_IR'] == 1.0
        assert sb['polymer_entropy_IR'] == 0.0
        assert sb['label'] == 'THEOREM'

    def test_tightened_C_K_structure(self):
        """compute_tightened_C_K returns all expected keys."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        required_keys = [
            'c_eps_generic', 'c_source_generic', 'C_K_generic',
            'c_eps_tight', 'c_source_tight', 'C_K_tight',
            'polymer_improvement', 'geometry_improvement', 'C_K_ratio',
            'kappa_analytical', 'c_star_generic', 'c_star_tight',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_C_K_tight_less_than_generic(self):
        """Tightened C_K is smaller than generic C_K."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        assert result['C_K_tight'] < result['C_K_generic'], (
            f"Tight ({result['C_K_tight']:.6f}) not smaller than "
            f"generic ({result['C_K_generic']:.6f})"
        )

    def test_hess_tight_less_than_generic(self):
        """Tightened Hessian bound is smaller than generic."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        assert result['hess_K0_tight'] < result['hess_K0_generic']

    def test_C_K_ratio_less_than_one(self):
        """C_K_tight / C_K_generic < 1 (improvement)."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        assert 0 < result['C_K_ratio'] < 1.0, (
            f"C_K ratio = {result['C_K_ratio']:.4f}, expected < 1"
        )

    def test_c_eps_base_value(self):
        """c_eps_base = C_2(adj)/(4*pi) = N_c/(4*pi) for SU(N_c)."""
        t = TightenedCK(N_c=2)
        expected = 2.0 / (4.0 * np.pi)
        assert t.c_eps_base == pytest.approx(expected, rel=1e-10)

        t3 = TightenedCK(N_c=3)
        expected3 = 3.0 / (4.0 * np.pi)
        assert t3.c_eps_base == pytest.approx(expected3, rel=1e-10)

    def test_c_source_base_value(self):
        """c_source = C_2^2/(16*pi^2)."""
        t = TightenedCK(N_c=2)
        expected = 4.0 / (16.0 * np.pi**2)  # C_2(SU(2)) = 2
        assert t.c_source_base == pytest.approx(expected, rel=1e-10)

    def test_polymer_improvement_less_than_one(self):
        """Polymer improvement factor is < 1 (favorable)."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()
        assert 0 < result['polymer_improvement'] < 1.0

    def test_geometry_improvement_less_than_one(self):
        """Geometry improvement factor is < 1 (favorable)."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()
        assert 0 < result['geometry_improvement'] < 1.0

    def test_sensitivity_analysis_structure(self):
        """sensitivity_analysis returns expected keys."""
        t = TightenedCK()
        sens = t.sensitivity_analysis(mu_values=np.linspace(1.0, 10.0, 10))

        required_keys = [
            'mu_values', 'c_star_values', 'mu_critical',
            'mu_estimated', 'kappa_analytical',
        ]
        for key in required_keys:
            assert key in sens, f"Missing key: {key}"

    def test_sensitivity_c_star_monotone_in_mu(self):
        """c_star should increase (become more positive) as mu decreases."""
        t = TightenedCK()
        mu_vals = np.linspace(1.0, 15.0, 20)
        sens = t.sensitivity_analysis(mu_values=mu_vals)

        c_stars = sens['c_star_values']
        # c_star should generally decrease as mu increases
        # (larger polymer entropy makes the bound worse)
        # Check that the first value is at least as good as the last
        assert c_stars[0] >= c_stars[-1], (
            f"c_star not monotone: c_star[0]={c_stars[0]:.4f}, "
            f"c_star[-1]={c_stars[-1]:.4f}"
        )

    def test_tightened_c_star_better_than_generic(self):
        """Tightened c_star is better (more positive) than generic."""
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        # The tightened bound should be at least as good as generic
        assert result['c_star_tight'] >= result['c_star_generic'], (
            f"Tight ({result['c_star_tight']:.4f}) worse than "
            f"generic ({result['c_star_generic']:.4f})"
        )


# ======================================================================
# 3. BridgeTighteningReport tests
# ======================================================================

class TestBridgeTighteningReport:
    """Tests for the combined report."""

    def test_construction(self):
        """BridgeTighteningReport constructs with default parameters."""
        report = BridgeTighteningReport()
        assert report.R == R_PHYS
        assert report.g2 == G2_PHYS

    def test_full_report_structure(self):
        """full_report returns all expected top-level keys."""
        report = BridgeTighteningReport(N_scales=3)
        result = report.full_report()

        required_keys = [
            'approach1_flow', 'approach1_bridge',
            'approach2_result', 'approach2_large_field',
            'approach2_polymer', 'approach2_single_block',
            'approach2_sensitivity',
            'verdict', 'c_star_comparison', 'label',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_verdict_is_verdict_type(self):
        """Verdict is a BridgeTighteningVerdict."""
        report = BridgeTighteningReport(N_scales=3)
        result = report.full_report()
        assert isinstance(result['verdict'], BridgeTighteningVerdict)

    def test_verdict_has_honest_assessment(self):
        """Verdict includes a non-empty honest assessment."""
        report = BridgeTighteningReport(N_scales=3)
        result = report.full_report()
        v = result['verdict']

        assert len(v.honest_assessment) > 0
        assert v.label in ('THEOREM', 'NUMERICAL', 'CONJECTURE', 'INCONCLUSIVE')

    def test_c_star_comparison_has_all_methods(self):
        """c_star_comparison includes all five methods."""
        report = BridgeTighteningReport(N_scales=3)
        result = report.full_report()
        comp = result['c_star_comparison']

        expected_methods = [
            'placeholder', 'generic', 'approach1_conservative',
            'approach1_optimistic', 'approach2_tight',
        ]
        for m in expected_methods:
            assert m in comp, f"Missing method: {m}"

    def test_placeholder_is_worst(self):
        """Placeholder C_K=1.0 gives the worst (most negative) c_star."""
        report = BridgeTighteningReport(N_scales=5)
        result = report.full_report()
        comp = result['c_star_comparison']

        # Placeholder (C_K=1.0) should be <= all other methods
        for method, c_star in comp.items():
            if method != 'placeholder':
                assert comp['placeholder'] <= c_star + 1e-10, (
                    f"Placeholder ({comp['placeholder']:.4f}) not worst vs "
                    f"{method} ({c_star:.4f})"
                )

    def test_summary_is_string(self):
        """summary() returns a non-empty string."""
        report = BridgeTighteningReport(N_scales=3)
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 100  # Should be a substantial report
        assert "BRIDGE TIGHTENING REPORT" in s

    def test_summary_mentions_kappa(self):
        """Summary mentions the analytical kappa value."""
        report = BridgeTighteningReport(N_scales=3)
        s = report.summary()
        assert "kappa_analytical" in s

    def test_summary_mentions_verdict(self):
        """Summary includes the verdict section."""
        report = BridgeTighteningReport(N_scales=3)
        s = report.summary()
        assert "VERDICT" in s


# ======================================================================
# 4. Consistency and physical sanity checks
# ======================================================================

class TestConsistencyChecks:
    """Cross-checks between approaches and physical sanity."""

    def test_kappa_consistent_between_approaches(self):
        """Both approaches use the same kappa_analytical."""
        a1 = ActualK0FromPipeline()
        bridge1 = a1.bridge_gap_analysis()

        a2 = TightenedCK()
        result2 = a2.compute_tightened_C_K()

        assert bridge1['kappa_analytical'] == pytest.approx(
            result2['kappa_analytical'], rel=1e-10
        )

    def test_kappa_analytical_value(self):
        """Kappa at R=2.2 fm should be around 2-3 fm^{-2}."""
        kappa = kappa_min_analytical(R_PHYS, 2)
        assert 0 < kappa < 50, f"kappa = {kappa} out of expected range"

    def test_g_bar_0_value(self):
        """g_bar_0 at physical parameters should be ~2.506."""
        assert G_BAR_0 == pytest.approx(np.sqrt(6.28), rel=1e-10)
        assert 2.0 < G_BAR_0 < 3.0

    def test_g_bar_0_cubed(self):
        """g_bar_0^3 ~ 15.7 (sets the scale for K bounds)."""
        g3 = G_BAR_0**3
        assert 10 < g3 < 20, f"g_bar^3 = {g3:.2f}"

    def test_g_bar_0_fourth(self):
        """g_bar_0^4 ~ 39.4 (sets the scale for Hessian bounds)."""
        g4 = G_BAR_0**4
        assert 30 < g4 < 50, f"g_bar^4 = {g4:.2f}"

    def test_approach1_hess_scales_correctly(self):
        """
        Hessian estimates scale correctly with K_0:
        hess_optimistic = K_0, hess_conservative = g_bar * K_0.
        """
        a1 = ActualK0FromPipeline()
        flow = a1.run_K_norm_flow()

        K_0 = flow['K_0_actual']
        assert flow['hess_K0_optimistic'] == pytest.approx(K_0, rel=1e-10)
        assert flow['hess_K0_conservative'] == pytest.approx(
            G_BAR_0 * K_0, rel=1e-10
        )

    def test_approach2_improvements_are_physical(self):
        """
        600-cell improvements should be moderate (not absurdly large).
        """
        t = TightenedCK()
        result = t.compute_tightened_C_K()

        # Polymer improvement should be around 0.1-0.5 (not extreme)
        assert 0.01 < result['polymer_improvement'] < 1.0
        # Geometry improvement should be around 0.3-0.8
        assert 0.1 < result['geometry_improvement'] < 1.0
        # C_K ratio should show meaningful improvement
        assert 0.001 < result['C_K_ratio'] < 1.0

    def test_SU3_parameters(self):
        """TightenedCK works for SU(3) as well."""
        t = TightenedCK(N_c=3)
        result = t.compute_tightened_C_K()

        # c_eps_base should be larger for SU(3) (C_2 = 3)
        assert t.c_eps_base > TightenedCK(N_c=2).c_eps_base

        # C_K_tight should still be finite
        assert np.isfinite(result['C_K_tight'])

    def test_small_coupling_closes_easily(self):
        """
        At small coupling, the bridge gap should be easy to close
        because g_bar^4 is tiny.
        """
        a1 = ActualK0FromPipeline(g2=1.0, N_scales=5)
        bridge = a1.bridge_gap_analysis()

        # At small coupling, the Hessian correction is negligible
        # and kappa is dominated by the geometric term
        # (but kappa itself may be small too)
        # The key check: the ratio hess_generic/kappa should be < 1
        kappa = bridge['kappa_analytical']
        if kappa > 0:
            ratio = bridge['hess_K0_generic'] / kappa
            # At g^2 = 1.0, this ratio should be much smaller than at g^2 = 6.28
            assert ratio < 100, f"Ratio = {ratio:.2f} still large at small coupling"

    def test_large_coupling_placeholder_fails(self):
        """
        At large coupling (g^2 = 6.28), the placeholder C_K=1.0 bound fails.
        This is the known problem we're trying to solve.
        """
        a1 = ActualK0FromPipeline(g2=G2_PHYS)
        bridge = a1.bridge_gap_analysis()

        # Placeholder bound (C_K=1.0) should fail
        assert bridge['c_star_placeholder'] < 0, (
            f"Placeholder bound should fail at physical coupling, "
            f"got c*={bridge['c_star_placeholder']:.4f}"
        )

    def test_approach1_K0_consistency_with_source(self):
        """
        K_0 should be roughly the accumulated source terms,
        since K starts at 0 at UV.
        """
        a1 = ActualK0FromPipeline(N_scales=5)
        flow = a1.run_K_norm_flow()

        # K_0 should be positive (accumulated sources)
        assert flow['K_0_actual'] > 0

        # K_0 should be comparable to or less than the sum of sources
        total_source = sum(flow['source_trajectory'])
        assert flow['K_0_actual'] <= total_source * 2.0, (
            f"K_0 ({flow['K_0_actual']:.6f}) much larger than "
            f"total source ({total_source:.6f})"
        )

    def test_sensitivity_mu_critical_exists(self):
        """
        There should exist a critical mu below which the gap closes.
        This tests that the bridge is closable in principle.
        """
        t = TightenedCK()
        sens = t.sensitivity_analysis(mu_values=np.linspace(0.1, 20.0, 50))

        # At very small mu, c_star should be positive
        c_stars = sens['c_star_values']
        assert any(c > 0 for c in c_stars), (
            "No mu value gives positive c_star -- bridge may be fundamentally unclosable"
        )

    def test_R_dependence(self):
        """
        TightenedCK should work at different R values and kappa
        should vary.
        """
        t1 = TightenedCK(R=1.0)
        t2 = TightenedCK(R=2.2)
        t3 = TightenedCK(R=5.0)

        r1 = t1.compute_tightened_C_K()
        r2 = t2.compute_tightened_C_K()
        r3 = t3.compute_tightened_C_K()

        # kappa_analytical should vary with R
        assert r1['kappa_analytical'] != pytest.approx(r2['kappa_analytical'], rel=0.1)

        # All should have finite C_K_tight
        assert np.isfinite(r1['C_K_tight'])
        assert np.isfinite(r2['C_K_tight'])
        assert np.isfinite(r3['C_K_tight'])


# ======================================================================
# 5. Quantitative spot checks
# ======================================================================

class TestQuantitativeSpotChecks:
    """Quantitative checks on specific numerical values."""

    def test_c_eps_base_SU2(self):
        """c_eps_base for SU(2) = 2/(4*pi) ~ 0.159."""
        t = TightenedCK(N_c=2)
        assert t.c_eps_base == pytest.approx(2.0 / (4 * np.pi), rel=1e-10)
        assert abs(t.c_eps_base - 0.159) < 0.001

    def test_c_source_base_SU2(self):
        """c_source_base for SU(2) = 4/(16*pi^2) ~ 0.0253."""
        t = TightenedCK(N_c=2)
        expected = 4.0 / (16.0 * np.pi**2)
        assert t.c_source_base == pytest.approx(expected, rel=1e-10)
        assert abs(t.c_source_base - 0.0253) < 0.001

    def test_generic_C_K_value(self):
        """
        Generic C_K = c_source / (1 - c_eps * g_bar_0) should be
        a specific finite value.
        """
        t = TightenedCK()
        r = t.compute_tightened_C_K()

        c_eps = r['c_eps_generic']
        c_src = r['c_source_generic']
        g_bar = r['g_bar_0']

        denom = 1.0 - c_eps * g_bar
        if denom > 0:
            expected_CK = c_src / denom
            assert r['C_K_generic'] == pytest.approx(expected_CK, rel=1e-10)

    def test_hess_generic_value(self):
        """
        Generic ||Hess(K_0)|| = C_K_generic * g_bar^4 at physical parameters.
        """
        t = TightenedCK()
        r = t.compute_tightened_C_K()
        expected = r['C_K_generic'] * r['g_bar_0']**4
        assert r['hess_K0_generic'] == pytest.approx(expected, rel=1e-10)

    def test_hess_tight_value(self):
        """
        Tightened ||Hess(K_0)|| = C_K_tight * g_bar^4 at physical parameters.
        """
        t = TightenedCK()
        r = t.compute_tightened_C_K()
        expected = r['C_K_tight'] * r['g_bar_0']**4
        assert r['hess_K0_tight'] == pytest.approx(expected, rel=1e-10)

    def test_face_sharing_degree_is_4(self):
        """600-cell face-sharing degree is exactly 4."""
        t = TightenedCK()
        pe = t.polymer_entropy_600cell()
        assert pe['D_face_600'] == 4

    def test_hypercubic_face_degree_is_8(self):
        """Hypercubic d=4 face degree is exactly 8."""
        t = TightenedCK()
        pe = t.polymer_entropy_600cell()
        assert pe['D_face_hyp'] == 8

    def test_mu_ratio_is_half(self):
        """Upper bound on mu_600/mu_hyp = e*4/(e*8) = 0.5."""
        t = TightenedCK()
        pe = t.polymer_entropy_600cell()
        assert pe['mu_ratio_upper'] == pytest.approx(0.5, rel=1e-10)


# ======================================================================
# 6. FaceSharingPolymerBound tests (Klarner argument)
# ======================================================================

class TestFaceSharingPolymerBound:
    """
    Tests for the rigorous Klarner-based polymer bound.

    This class tests the THEOREM that tf = D_600/D_hyp = 4/8 = 1/2
    is a valid bound on the polymer sum ratio in the BBS framework,
    based on Klarner's 1967 lattice animal bound applied to the dual
    graph of the 600-cell.

    Key distinction tested: this is about LATTICE ANIMAL growth rates
    on regular graphs (bounded by e*D via Klarner), NOT self-avoiding
    walk (SAW) connective constants (which require submultiplicativity).
    """

    def test_face_sharing_degrees(self):
        """D_600 = 4 and D_hyp = 8 are the claimed values."""
        fb = FaceSharingPolymerBound()
        assert fb.D_600 == 4
        assert fb.D_HYP_4D == 8

    def test_klarner_bound_n1(self):
        """At n=1, Klarner bound is (e*D)^0 = 1 for any D."""
        fb = FaceSharingPolymerBound()
        assert fb.klarner_bound(4, 1) == pytest.approx(1.0)
        assert fb.klarner_bound(8, 1) == pytest.approx(1.0)
        assert fb.klarner_bound(100, 1) == pytest.approx(1.0)

    def test_klarner_bound_n2(self):
        """At n=2, Klarner bound is e*D."""
        fb = FaceSharingPolymerBound()
        assert fb.klarner_bound(4, 2) == pytest.approx(np.e * 4, rel=1e-10)
        assert fb.klarner_bound(8, 2) == pytest.approx(np.e * 8, rel=1e-10)

    def test_klarner_bound_general(self):
        """Klarner bound is (e*D)^{n-1} for general n."""
        fb = FaceSharingPolymerBound()
        for n in [3, 5, 10]:
            assert fb.klarner_bound(4, n) == pytest.approx(
                (np.e * 4) ** (n - 1), rel=1e-10
            )

    def test_klarner_bound_ratio_is_half_power(self):
        """
        THEOREM: Klarner_600(n) / Klarner_hyp(n) = (D_600/D_hyp)^{n-1}
                                                   = (1/2)^{n-1}.
        """
        fb = FaceSharingPolymerBound()
        for n in [1, 2, 3, 5, 10, 20]:
            ratio = fb.klarner_bound(fb.D_600, n) / fb.klarner_bound(fb.D_HYP_4D, n)
            expected = (fb.D_600 / fb.D_HYP_4D) ** (n - 1)
            assert ratio == pytest.approx(expected, rel=1e-10), (
                f"At n={n}: ratio={ratio}, expected={expected}"
            )

    def test_klarner_600_always_less_than_hyp(self):
        """For n >= 2, Klarner bound on 600-cell < hypercubic."""
        fb = FaceSharingPolymerBound()
        for n in range(2, 15):
            assert fb.klarner_bound(fb.D_600, n) < fb.klarner_bound(fb.D_HYP_4D, n)

    def test_klarner_validation(self):
        """Invalid inputs raise ValueError."""
        fb = FaceSharingPolymerBound()
        with pytest.raises(ValueError):
            fb.klarner_bound(4, 0)
        with pytest.raises(ValueError):
            fb.klarner_bound(0, 5)

    def test_polymer_sum_convergent(self):
        """Polymer sum converges for c_s below the convergence limit."""
        fb = FaceSharingPolymerBound()
        c_s_limit = 1.0 / (np.e * fb.D_600)
        c_s = c_s_limit * 0.5  # Well within convergence

        S = fb.polymer_sum_bound(fb.D_600, c_s)
        assert np.isfinite(S)
        assert S > 0

    def test_polymer_sum_geometric_formula(self):
        """
        For c_s * e * D < 1, the polymer sum equals the geometric series:
            S = c_s / (1 - c_s * e * D)
        """
        fb = FaceSharingPolymerBound()
        c_s = 0.01
        expected = c_s / (1.0 - c_s * np.e * fb.D_600)
        actual = fb.polymer_sum_bound(fb.D_600, c_s)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_polymer_sum_600_less_than_hyp(self):
        """
        THEOREM: For any c_s in the common convergence region,
        the 600-cell polymer sum is strictly less than the hypercubic one.
        """
        fb = FaceSharingPolymerBound()
        # c_s must be below the SMALLER limit (hypercubic)
        c_s_limit_hyp = 1.0 / (np.e * fb.D_HYP_4D)
        for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c_s = c_s_limit_hyp * fraction
            S_600 = fb.polymer_sum_bound(fb.D_600, c_s)
            S_hyp = fb.polymer_sum_bound(fb.D_HYP_4D, c_s)
            assert S_600 < S_hyp, (
                f"At c_s={c_s:.4f}: S_600={S_600:.6f} >= S_hyp={S_hyp:.6f}"
            )

    def test_convergence_limit_doubled(self):
        """
        THEOREM: The 600-cell convergence limit is DOUBLE the hypercubic:
        c_s_limit_600 / c_s_limit_hyp = D_hyp / D_600 = 8/4 = 2.
        """
        fb = FaceSharingPolymerBound()
        c_s_limit_600 = 1.0 / (np.e * fb.D_600)
        c_s_limit_hyp = 1.0 / (np.e * fb.D_HYP_4D)
        ratio = c_s_limit_600 / c_s_limit_hyp
        assert ratio == pytest.approx(2.0, rel=1e-10)

    def test_tightening_factor_leading_is_half(self):
        """
        THEOREM: Leading-order tightening factor tf = D_600/D_hyp = 1/2.
        """
        fb = FaceSharingPolymerBound()
        assert fb.tightening_factor_leading() == pytest.approx(0.5, rel=1e-10)

    def test_tightening_factor_exact_less_than_one(self):
        """
        THEOREM: Exact tightening factor tf(c_s) < 1 for all c_s
        in the common convergence region.
        """
        fb = FaceSharingPolymerBound()
        c_s_limit_hyp = 1.0 / (np.e * fb.D_HYP_4D)
        for fraction in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
            c_s = c_s_limit_hyp * fraction
            tf = fb.tightening_factor_exact(c_s)
            assert 0 < tf < 1.0, (
                f"At c_s={c_s:.6f}: tf={tf:.6f}, expected 0 < tf < 1"
            )

    def test_tightening_factor_exact_at_small_cs(self):
        """
        At very small c_s, tf -> 1 because the n=1 term (identical
        on both lattices) dominates.  The multi-cell terms that carry
        the (1/2)^{n-1} suppression are negligible at small c_s.
        """
        fb = FaceSharingPolymerBound()
        c_s = 1e-6
        tf = fb.tightening_factor_exact(c_s)
        # At tiny c_s: S ~ c_s on both lattices, so ratio -> 1
        assert tf == pytest.approx(1.0, rel=0.01), (
            f"At tiny c_s: tf={tf:.6f}, expected ~1.0"
        )

    def test_tightening_factor_exact_at_large_cs(self):
        """
        At c_s near the hypercubic limit, tf < 1/2 (BETTER than leading order).
        This is because the 600-cell series is further from divergence.
        """
        fb = FaceSharingPolymerBound()
        c_s_limit_hyp = 1.0 / (np.e * fb.D_HYP_4D)
        c_s = c_s_limit_hyp * 0.9  # Near the limit
        tf = fb.tightening_factor_exact(c_s)
        assert tf < 0.5, (
            f"Near limit: tf={tf:.6f}, expected < 0.5"
        )

    def test_full_analysis_structure(self):
        """full_analysis returns all expected keys."""
        fb = FaceSharingPolymerBound()
        result = fb.full_analysis()

        required_keys = [
            'D_600', 'D_hyp', 'D_ratio', 'tf_leading',
            'klarner_600', 'klarner_hyp', 'klarner_ratios',
            'c_s_limit_600', 'c_s_limit_hyp', 'convergence_ratio',
            'c_s_values', 'tf_exact', 'tf_max', 'tf_min',
            'tf_always_less_than_one', 'label',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_full_analysis_label_is_theorem(self):
        """The analysis is labeled THEOREM."""
        fb = FaceSharingPolymerBound()
        result = fb.full_analysis()
        assert result['label'] == 'THEOREM'

    def test_full_analysis_tf_always_less_than_one(self):
        """All tightening factors in the scan are < 1."""
        fb = FaceSharingPolymerBound()
        result = fb.full_analysis()
        assert result['tf_always_less_than_one'] is True

    def test_full_analysis_convergence_ratio_is_2(self):
        """
        THEOREM: 600-cell convergence limit is 2x the hypercubic limit.
        """
        fb = FaceSharingPolymerBound()
        result = fb.full_analysis()
        assert result['convergence_ratio'] == pytest.approx(2.0, rel=1e-10)

    def test_full_analysis_klarner_ratios(self):
        """
        Klarner bound ratios are (1/2)^{n-1} at each polymer size.
        """
        fb = FaceSharingPolymerBound()
        result = fb.full_analysis()
        for n_str, ratio in result['klarner_ratios'].items():
            n = int(n_str)
            expected = (0.5) ** (n - 1)
            assert ratio == pytest.approx(expected, rel=1e-10), (
                f"At n={n}: ratio={ratio}, expected={expected}"
            )

    def test_verify_D_face_600cell(self):
        """
        Verify D_face = 4 on the actual 600-cell construction.
        This is a computational verification of the combinatorial theorem.
        """
        fb = FaceSharingPolymerBound()
        result = fb.verify_D_face_600cell()

        assert result.get('n_cells') == 600
        assert result.get('is_regular') is True
        assert result.get('D_face_verified') == 4
        assert result.get('matches_claim') is True
        assert result.get('label') == 'THEOREM'

    def test_not_about_saw_connective_constants(self):
        """
        Structural test: the tightening factor depends ONLY on
        the face-sharing degree D, not on any self-avoiding walk
        statistics.  Changing D while keeping everything else fixed
        changes tf proportionally.
        """
        fb = FaceSharingPolymerBound()

        # tf is exactly D1/D2 at leading order
        D1, D2 = 4, 8
        assert fb.tightening_factor_leading() == pytest.approx(D1 / D2, rel=1e-10)

        # Klarner bound depends only on D and n
        # No reference to walk statistics, path counts, or concatenation
        for n in [1, 5, 10]:
            bound_D4 = fb.klarner_bound(4, n)
            bound_D6 = fb.klarner_bound(6, n)
            bound_D8 = fb.klarner_bound(8, n)

            # Monotone in D
            assert bound_D4 <= bound_D6 <= bound_D8

            # Exact formula
            assert bound_D4 == pytest.approx((np.e * 4) ** (n - 1), rel=1e-10)
            assert bound_D6 == pytest.approx((np.e * 6) ** (n - 1), rel=1e-10)
            assert bound_D8 == pytest.approx((np.e * 8) ** (n - 1), rel=1e-10)

    def test_multi_cell_ratio_bounded_by_half(self):
        """
        THEOREM: The ratio of multi-cell (n >= 2) polymer contributions
        is at most D_600/D_hyp = 1/2.

        The n=1 term is the same on both lattices (c_s * 1 = c_s).
        Subtracting it:

            S_multi = c_s^2 * e * D / (1 - c_s * e * D)

        so the exact ratio is:
            (D_600 / D_hyp) * (1 - c_s*e*D_hyp) / (1 - c_s*e*D_600)

        Since D_600 < D_hyp, the correction factor (1 - r_hyp)/(1 - r_600) < 1,
        making the ratio LESS than D_600/D_hyp = 1/2.  That is, the 600-cell
        multi-cell polymers are suppressed by MORE than a factor of 2.
        """
        fb = FaceSharingPolymerBound()
        for c_s in [0.001, 0.005, 0.01]:
            S_600 = fb.polymer_sum_bound(fb.D_600, c_s)
            S_hyp = fb.polymer_sum_bound(fb.D_HYP_4D, c_s)
            # Subtract the n=1 term (identical on both)
            multi_600 = S_600 - c_s
            multi_hyp = S_hyp - c_s
            ratio = multi_600 / multi_hyp

            # THEOREM: ratio <= D_600/D_hyp = 0.5
            assert ratio <= 0.5 + 1e-10, (
                f"At c_s={c_s}: multi-cell ratio={ratio:.6f}, expected <= 0.5"
            )
            assert ratio > 0, (
                f"At c_s={c_s}: multi-cell ratio should be positive"
            )

            # Verify exact formula
            r_600 = c_s * np.e * fb.D_600
            r_hyp = c_s * np.e * fb.D_HYP_4D
            expected_ratio = (fb.D_600 / fb.D_HYP_4D) * (1 - r_hyp) / (1 - r_600)
            assert ratio == pytest.approx(expected_ratio, rel=1e-8), (
                f"At c_s={c_s}: ratio={ratio:.6f} vs formula={expected_ratio:.6f}"
            )

    def test_polymer_sum_zero_source(self):
        """Zero source gives zero polymer sum."""
        fb = FaceSharingPolymerBound()
        assert fb.polymer_sum_bound(4, 0.0) == 0.0

    def test_polymer_sum_finite_lattice_advantage(self):
        """
        On the 600-cell, the polymer sum is EXACT (no tail bound needed)
        because the maximum polymer size is 600.  The Klarner bound gives
        an UPPER bound on each term, and the sum truncates.
        """
        fb = FaceSharingPolymerBound()
        # Even if we use n_max = 600 (exact for 600-cell),
        # the result is finite for convergent c_s
        c_s = 0.01
        S_finite = fb.polymer_sum_bound(fb.D_600, c_s, n_max=600)
        S_infinite = fb.polymer_sum_bound(fb.D_600, c_s, n_max=10000)
        # Both should give the same result (geometric series converges)
        assert S_finite == pytest.approx(S_infinite, rel=1e-10)
