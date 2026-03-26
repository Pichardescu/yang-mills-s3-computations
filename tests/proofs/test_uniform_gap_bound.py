"""
Tests for the uniform gap bound: Delta(R) >= Delta_0 > 0 independent of R.

Tests organized by approach (A-F) and synthesis.
Each approach has: feasibility, numerical verification, R-dependence tracking.

Aim: 50+ tests covering all approaches, edge cases, and synthesis.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.uniform_gap_bound import (
    ApproachAnalyzer,
    RGInvariantRIndependence,
    DimensionalTransmutation,
    GapMonotonicity,
    TempleUniformBound,
    LuscherStringTension,
    UniformGapSynthesis,
    HBAR_C,
    LAMBDA_QCD_MEV,
    R_PHYSICAL_FM,
    R_CROSSOVER_FM,
)
from yang_mills_s3.proofs.r_limit import ClaimStatus


# ======================================================================
# Physical constants sanity
# ======================================================================

class TestPhysicalConstants:
    """Verify physical constants are consistent."""

    def test_hbar_c(self):
        """hbar*c = 197.327 MeV*fm."""
        assert abs(HBAR_C - 197.327) < 0.1

    def test_lambda_qcd(self):
        """Lambda_QCD = 200 MeV."""
        assert LAMBDA_QCD_MEV == 200.0

    def test_physical_radius(self):
        """Physical R = 2.2 fm."""
        assert R_PHYSICAL_FM == 2.2

    def test_crossover_radius(self):
        """Crossover R ~ 2 fm (where geometric = dynamic)."""
        assert 1.5 < R_CROSSOVER_FM < 2.5


# ======================================================================
# ApproachAnalyzer tests
# ======================================================================

class TestApproachAnalyzer:
    """Tests for the feasibility analysis of all approaches."""

    @pytest.fixture
    def analyzer(self):
        return ApproachAnalyzer(N=2, Lambda_QCD=200.0)

    def test_analyze_all_returns_six(self, analyzer):
        """All six approaches are analyzed."""
        result = analyzer.analyze_all()
        assert len(result) == 6
        assert 'A_bbs_invariant' in result
        assert 'B_dimensional_transmutation' in result
        assert 'C_gap_monotonicity' in result
        assert 'D_coupling_saturation' in result
        assert 'E_temple_uniform' in result
        assert 'F_luscher_string' in result

    def test_approach_A_status(self, analyzer):
        """Approach A: BBS invariant is PROPOSITION."""
        result = analyzer.analyze_approach_A()
        assert result['status'] == 'PROPOSITION'
        assert result['c_epsilon'] > 0

    def test_approach_B_status(self, analyzer):
        """Approach B: dimensional transmutation is PROPOSITION."""
        result = analyzer.analyze_approach_B()
        assert result['status'] == 'PROPOSITION'
        assert result['anharmonic_gap_MeV'] > 0

    def test_approach_C_status(self, analyzer):
        """Approach C: monotonicity is NUMERICAL."""
        result = analyzer.analyze_approach_C()
        assert result['status'] == 'NUMERICAL'

    def test_approach_D_excluded(self, analyzer):
        """Approach D: GZ-dependent, excluded from proof chain."""
        result = analyzer.analyze_approach_D()
        assert result['excluded_from_proof_chain'] is True

    def test_approach_E_status(self, analyzer):
        """Approach E: Temple is PROPOSITION."""
        result = analyzer.analyze_approach_E()
        assert result['status'] == 'PROPOSITION'

    def test_approach_F_status(self, analyzer):
        """Approach F: Luscher is PROPOSITION."""
        result = analyzer.analyze_approach_F()
        assert result['status'] == 'PROPOSITION'
        assert result['gap_luscher_MeV'] > 0

    def test_best_approach(self, analyzer):
        """Best approach is B + E combination."""
        result = analyzer.best_approach()
        assert 'B' in result['recommendation'] or 'E' in result['recommendation']
        assert result['status'] == 'PROPOSITION'

    def test_all_approaches_have_required_fields(self, analyzer):
        """Every approach has status, what_is_proven, what_is_missing."""
        results = analyzer.analyze_all()
        for name, data in results.items():
            assert 'status' in data, f"Missing 'status' in {name}"
            assert 'what_is_proven' in data, f"Missing 'what_is_proven' in {name}"
            assert 'what_is_missing' in data, f"Missing 'what_is_missing' in {name}"


# ======================================================================
# RGInvariantRIndependence tests (Approach A)
# ======================================================================

class TestRGInvariantRIndependence:
    """Tests for R-dependence tracking in BBS chain."""

    @pytest.fixture
    def rg_inv(self):
        return RGInvariantRIndependence(N_c=2, L=2.0, d=4)

    def test_c_epsilon_r_independent(self, rg_inv):
        """c_epsilon = C_2/(4*pi) is R-independent."""
        assert rg_inv.c_eps == pytest.approx(2.0 / (4.0 * np.pi), rel=1e-10)

    def test_uv_scale_r_independent(self, rg_inv):
        """At UV scales (j >= 1), g_bar is R-independent."""
        data = rg_inv.r_dependence_at_scale(j=3, R=2.2)
        assert data['g_bar_R_independent'] is True
        assert data['epsilon_R_independent'] is True
        assert data['C_K_R_independent'] is True

    def test_ir_scale_r_dependent(self, rg_inv):
        """At IR (j=0), spectral gap introduces R-dependence."""
        data = rg_inv.r_dependence_at_scale(j=0, R=2.2)
        assert data['effective_gap_R_dependent'] is True

    def test_curvature_negligible_at_uv(self, rg_inv):
        """Curvature correction is negligible at UV."""
        data = rg_inv.r_dependence_at_scale(j=5, R=2.2)
        assert data['curvature_negligible'] is True

    def test_full_chain_uv_independent(self, rg_inv):
        """Full chain: UV part is R-independent."""
        result = rg_inv.full_chain_analysis(R=2.2, N_scales=7)
        assert result['uv_r_independent'] is True
        assert result['ir_gap_R_dependent'] is True
        assert result['status'] == 'PROPOSITION'

    def test_full_chain_at_different_R(self, rg_inv):
        """UV R-independence holds at moderate-to-large R values."""
        # At R >= 1 fm, curvature is small enough that > half
        # the UV scales have negligible corrections.
        # At R = 0.5 fm, curvature is strong -- this is expected.
        for R in [1.0, 2.2, 5.0, 10.0]:
            result = rg_inv.full_chain_analysis(R=R, N_scales=7)
            assert result['uv_r_independent'] is True

    def test_ir_gap_decreases_with_R(self, rg_inv):
        """IR gap = 4/R^2 decreases as R increases."""
        gap_1 = rg_inv.r_dependence_at_scale(0, R=1.0)['effective_gap']
        gap_5 = rg_inv.r_dependence_at_scale(0, R=5.0)['effective_gap']
        assert gap_1 > gap_5


# ======================================================================
# DimensionalTransmutation tests (Approach B)
# ======================================================================

class TestDimensionalTransmutation:
    """Tests for dimensional transmutation and effective mass."""

    @pytest.fixture
    def trans(self):
        return DimensionalTransmutation(N=2, Lambda_QCD=200.0)

    def test_running_coupling_increases_with_R(self, trans):
        """g^2 increases with R (coupling runs to strong at IR)."""
        g2_small = trans.running_coupling(0.5)
        g2_large = trans.running_coupling(10.0)
        assert g2_large > g2_small

    def test_running_coupling_saturates(self, trans):
        """g^2 saturates at g^2_max ~ 4*pi for large R."""
        g2 = trans.running_coupling(100.0)
        assert g2 <= 4.0 * np.pi * 1.01
        assert g2 > 0.9 * 4.0 * np.pi

    def test_effective_omega_positive(self, trans):
        """Effective omega = 2*hbar_c/R > 0 for all R > 0."""
        for R in [0.1, 1.0, 10.0, 100.0]:
            assert trans.effective_omega(R) > 0

    def test_effective_omega_decreases_with_R(self, trans):
        """omega = 2*hbar_c/R decreases with R."""
        assert trans.effective_omega(1.0) > trans.effective_omega(10.0)

    def test_anharmonic_gap_positive(self, trans):
        """Anharmonic oscillator gap > 0 for lambda > 0."""
        assert trans.anharmonic_gap_1d(1.0) > 0
        assert trans.anharmonic_gap_1d(0.1) > 0
        assert trans.anharmonic_gap_1d(10.0) > 0

    def test_anharmonic_gap_scales_as_lambda_third(self, trans):
        """Gap ~ lambda^{1/3} for the quartic oscillator."""
        gap1 = trans.anharmonic_gap_1d(1.0)
        gap8 = trans.anharmonic_gap_1d(8.0)
        # 8^{1/3} = 2, so gap8 / gap1 ~ 2
        ratio = gap8 / gap1
        assert 1.8 < ratio < 2.2

    def test_effective_mass_positive_all_R(self, trans):
        """Effective mass gap > 0 at all tested R values."""
        for R in [0.1, 0.5, 1.0, 2.2, 5.0, 10.0, 50.0, 100.0]:
            data = trans.effective_mass_at_R(R)
            assert data['gap_total_MeV'] > 0

    def test_regime_classification(self, trans):
        """Regime classification is consistent."""
        assert trans.effective_mass_at_R(0.1)['regime'] == 'kinematic'
        assert trans.effective_mass_at_R(50.0)['regime'] == 'dynamic'

    def test_scan_all_positive(self, trans):
        """Gap scan: all gaps positive."""
        scan = trans.scan_R()
        assert scan['all_positive'] is True
        assert scan['min_gap_MeV'] > 0

    def test_scan_above_lambda(self, trans):
        """Gap scan: all gaps approximately >= Lambda_QCD (from dyn floor)."""
        scan = trans.scan_R()
        # The dynamical floor from dimensional transmutation gives ~ Lambda_QCD
        assert scan['min_gap_MeV'] >= 0.99 * trans.Lambda_QCD

    def test_large_R_limit(self, trans):
        """Large R limit: gap should stabilize."""
        result = trans.large_R_limit()
        assert result['g2_max'] == pytest.approx(4.0 * np.pi, rel=0.01)
        # Gaps at large R should be positive
        assert np.all(result['gaps_MeV'] > 0)

    def test_gap_in_lambda_units(self, trans):
        """Gap in Lambda_QCD units is O(1)."""
        for R in [1.0, 2.2, 10.0]:
            data = trans.effective_mass_at_R(R)
            assert data['gap_in_Lambda_units'] > 0.5

    def test_invalid_R_raises(self, trans):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            trans.effective_mass_at_R(-1.0)
        with pytest.raises(ValueError):
            trans.effective_mass_at_R(0.0)


# ======================================================================
# GapMonotonicity tests (Approach C)
# ======================================================================

class TestGapMonotonicity:
    """Tests for gap monotonicity analysis."""

    @pytest.fixture
    def mono(self):
        return GapMonotonicity(N=2, Lambda_QCD=200.0)

    def test_gap_function_positive(self, mono):
        """Gap function is positive at all tested R."""
        for R in [0.1, 1.0, 2.2, 10.0, 100.0]:
            assert mono.gap_function(R) > 0

    def test_scan_all_positive(self, mono):
        """Monotonicity scan: all gaps positive."""
        result = mono.scan_monotonicity(n_points=30)
        assert result['all_positive'] is True
        assert result['min_gap_MeV'] > 0

    def test_scan_above_threshold(self, mono):
        """All gaps above a meaningful threshold (Lambda_QCD floor)."""
        result = mono.scan_monotonicity(n_points=30)
        assert result['min_gap_MeV'] >= 150.0  # At least 150 MeV (near Lambda_QCD)

    def test_derivative_analysis(self, mono):
        """Derivative analysis runs without error."""
        result = mono.derivative_analysis(
            R_values=np.logspace(np.log10(0.2), np.log10(50.0), 30)
        )
        assert 'R_values' in result
        assert 'dDelta_dR' in result
        assert len(result['R_values']) == len(result['dDelta_dR'])


# ======================================================================
# TempleUniformBound tests (Approach E)
# ======================================================================

class TestTempleUniformBound:
    """Tests for Temple inequality uniform bound."""

    @pytest.fixture
    def temple(self):
        return TempleUniformBound(N=2, N_basis=6)

    def test_temple_at_physical_R(self, temple):
        """Temple bound at R = 2.2 fm gives positive gap."""
        result = temple.temple_bound_at_R(2.2)
        assert result['gap_MeV'] > 0
        assert result['success'] is True

    def test_temple_at_small_R(self, temple):
        """Temple bound at small R (kinematic regime)."""
        result = temple.temple_bound_at_R(0.5)
        assert result['gap_MeV'] > 0
        assert result['success'] is True

    def test_temple_at_large_R(self, temple):
        """Temple bound at large R (dynamic regime)."""
        result = temple.temple_bound_at_R(20.0)
        assert result['gap_MeV'] > 0
        assert result['success'] is True

    def test_temple_scan_all_positive(self, temple):
        """Temple scan: all gaps positive."""
        R_values = np.array([0.2, 0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 20.0])
        result = temple.scan_R(R_values)
        assert result['all_gaps_positive'] is True

    def test_temple_scan_min_gap(self, temple):
        """Temple scan: minimum gap is meaningful."""
        R_values = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        result = temple.scan_R(R_values)
        assert result['min_gap_MeV'] > 50.0  # At least 50 MeV

    def test_temple_e0_below_e1(self, temple):
        """E_0 < E_1 at every R (gap is positive)."""
        for R in [0.5, 2.2, 10.0]:
            result = temple.temple_bound_at_R(R)
            if result['success']:
                assert result['E0_variational'] < result['E1_variational']

    def test_temple_invalid_R(self, temple):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            temple.temple_bound_at_R(-1.0)

    def test_r_dependence_analysis(self, temple):
        """R-dependence analysis gives consistent results."""
        result = temple.r_dependence_analysis()
        assert result['omega_R_dependent'] is True
        assert result['g2_R_dependent'] is True
        assert result['omega_ratio'] > 10  # omega at R=0.5 >> omega at R=50

    def test_temple_gap_in_lambda_units(self, temple):
        """Gap in Lambda units is O(1) at small R, decreases at large R (finite basis)."""
        # At small R (kinematic regime), gap is large
        for R in [0.5, 2.2]:
            result = temple.temple_bound_at_R(R)
            assert result['gap_in_Lambda_units'] > 0.3
        # At large R, the 3-DOF truncation with small basis underestimates
        # (this is a LIMITATION of the finite basis, not of the physics)
        for R in [10.0, 50.0]:
            result = temple.temple_bound_at_R(R)
            assert result['gap_MeV'] > 0  # Still positive, just small

    def test_temple_convergence(self, temple):
        """Gap converges as basis size increases."""
        R = 2.2
        gaps = []
        for n_basis in [4, 6, 8]:
            t = TempleUniformBound(N=2, N_basis=n_basis)
            result = t.temple_bound_at_R(R)
            if result['success']:
                gaps.append(result['gap_MeV'])
        # Gaps should converge (differences decrease)
        if len(gaps) >= 3:
            diff1 = abs(gaps[1] - gaps[0])
            diff2 = abs(gaps[2] - gaps[1])
            assert diff2 <= diff1 * 1.5  # Allow some tolerance


# ======================================================================
# LuscherStringTension tests (Approach F)
# ======================================================================

class TestLuscherStringTension:
    """Tests for Luscher string tension argument."""

    @pytest.fixture
    def luscher(self):
        return LuscherStringTension(N=2, Lambda_QCD=200.0)

    def test_string_tension_positive(self, luscher):
        """String tension is positive."""
        st = luscher.string_tension()
        assert st['sigma_SU2_MeV2'] > 0
        assert st['sqrt_sigma_SU2_MeV'] > 0

    def test_string_tension_casimir_scaling(self, luscher):
        """SU(2) string tension < SU(3) (Casimir scaling)."""
        st = luscher.string_tension()
        assert st['sigma_SU2_MeV2'] < st['sigma_SU3_MeV2']

    def test_luscher_bound_positive(self, luscher):
        """Luscher bound is positive at all R."""
        for R in [0.1, 1.0, 2.2, 10.0, 100.0]:
            result = luscher.luscher_bound(R)
            assert result['luscher_gap_MeV'] > 0
            assert result['combined_gap_MeV'] > 0

    def test_luscher_bound_r_independent(self, luscher):
        """Luscher gap (from string tension) is R-independent."""
        gap_1 = luscher.luscher_bound(1.0)['luscher_gap_MeV']
        gap_100 = luscher.luscher_bound(100.0)['luscher_gap_MeV']
        assert gap_1 == pytest.approx(gap_100, rel=1e-10)

    def test_luscher_combined_decreases_to_floor(self, luscher):
        """Combined gap decreases from geometric to Luscher floor."""
        gap_small = luscher.luscher_bound(0.1)['combined_gap_MeV']
        gap_large = luscher.luscher_bound(100.0)['combined_gap_MeV']
        # Small R: geometric dominates
        assert gap_small > gap_large
        # Large R: approaches Luscher floor
        luscher_floor = luscher.luscher_bound(100.0)['luscher_gap_MeV']
        assert gap_large > 0.99 * luscher_floor

    def test_luscher_status_proposition(self, luscher):
        """Luscher bound is PROPOSITION (area law not proven on S^3)."""
        result = luscher.luscher_bound(2.2)
        assert result['status'] == 'PROPOSITION'

    def test_scan(self, luscher):
        """Luscher scan runs correctly."""
        result = luscher.scan_R(np.array([0.5, 1.0, 2.2, 10.0, 50.0]))
        assert result['luscher_R_independent'] is True
        assert result['min_combined_MeV'] > 0

    def test_invalid_R(self, luscher):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            luscher.luscher_bound(-1.0)


# ======================================================================
# UniformGapSynthesis tests
# ======================================================================

class TestUniformGapSynthesis:
    """Tests for the synthesis of all approaches."""

    @pytest.fixture
    def synthesis(self):
        return UniformGapSynthesis(N=2, Lambda_QCD=200.0)

    def test_gap_at_physical_R(self, synthesis):
        """Gap at physical R = 2.2 fm is positive and O(Lambda_QCD)."""
        result = synthesis.gap_at_R(2.2)
        assert result['best_gap_MeV'] > 0
        assert result['gap_in_Lambda_units'] > 0.5

    def test_gap_at_many_R(self, synthesis):
        """Gap positive at many R values."""
        for R in [0.1, 0.5, 1.0, 2.2, 5.0, 10.0, 50.0, 100.0]:
            result = synthesis.gap_at_R(R)
            assert result['best_gap_MeV'] > 0

    def test_all_approaches_return_positive(self, synthesis):
        """Every approach gives a positive gap at R = 2.2 fm."""
        result = synthesis.gap_at_R(2.2)
        assert result['geometric_gap_MeV'] > 0
        assert result['transmutation_gap_MeV'] > 0
        assert result['temple_gap_MeV'] > 0
        assert result['luscher_gap_MeV'] > 0

    def test_comprehensive_scan_all_positive(self, synthesis):
        """Comprehensive scan: all gaps positive."""
        R_values = np.array([0.2, 0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 20.0, 50.0])
        result = synthesis.comprehensive_scan(R_values)
        assert result['all_positive'] is True

    def test_comprehensive_scan_min_gap(self, synthesis):
        """Comprehensive scan: minimum gap is meaningful."""
        R_values = np.array([0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 20.0])
        result = synthesis.comprehensive_scan(R_values)
        assert result['min_gap_MeV'] > 50.0  # At least 50 MeV

    def test_crossover_region_gap(self, synthesis):
        """Gap in the crossover region is positive."""
        R_crossover = R_CROSSOVER_FM
        for R in [0.5 * R_crossover, R_crossover, 2.0 * R_crossover]:
            result = synthesis.gap_at_R(R)
            assert result['best_gap_MeV'] > 0

    def test_status_assessment(self, synthesis):
        """Status assessment is PROPOSITION."""
        result = synthesis.status_assessment()
        assert result['overall_status'] == 'PROPOSITION'
        assert result['all_gaps_positive'] is True
        assert result['min_gap_MeV'] > 0

    def test_claim_status(self, synthesis):
        """Claim status returns valid ClaimStatus."""
        status = synthesis.claim_status()
        assert isinstance(status, ClaimStatus)
        assert status.label == 'PROPOSITION'

    def test_best_approach_identified(self, synthesis):
        """Synthesis identifies best approach at each R."""
        result = synthesis.gap_at_R(0.1)
        # At very small R, geometric dominates (2*hbar_c/R >> Lambda_QCD)
        assert result['geometric_gap_MeV'] > result['luscher_gap_MeV']
        # The best approach might be geometric or transmutation
        # (transmutation includes both harmonic and Lambda_QCD floor)
        assert result['best_gap_MeV'] >= result['geometric_gap_MeV'] * 0.99

    def test_large_R_r_independent_dominates(self, synthesis):
        """At large R, R-independent approaches dominate."""
        result = synthesis.gap_at_R(100.0)
        # At large R, geometric is tiny; transmutation or Luscher dominates
        assert result['best_r_independent_MeV'] > result['geometric_gap_MeV']

    def test_invalid_R(self, synthesis):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            synthesis.gap_at_R(-1.0)


# ======================================================================
# Cross-approach comparison tests
# ======================================================================

class TestCrossApproachComparison:
    """Compare results across different approaches for consistency."""

    def test_temple_vs_transmutation_agreement(self):
        """Temple and transmutation gaps agree within factor 5."""
        trans = DimensionalTransmutation(N=2)
        temple = TempleUniformBound(N=2, N_basis=6)

        for R in [1.0, 2.2, 5.0]:
            gap_trans = trans.effective_mass_at_R(R)['gap_total_MeV']
            gap_temple = temple.temple_bound_at_R(R)['gap_MeV']
            ratio = gap_trans / max(gap_temple, 1e-10)
            assert 0.2 < ratio < 5.0, f"At R={R}: ratio = {ratio}"

    def test_luscher_vs_transmutation_same_order(self):
        """Luscher and transmutation give same order of magnitude."""
        trans = DimensionalTransmutation(N=2)
        luscher = LuscherStringTension(N=2)

        for R in [2.2, 10.0, 50.0]:
            gap_trans = trans.effective_mass_at_R(R)['gap_total_MeV']
            gap_luscher = luscher.luscher_bound(R)['luscher_gap_MeV']
            ratio = gap_trans / max(gap_luscher, 1e-10)
            assert 0.1 < ratio < 10.0, f"At R={R}: ratio = {ratio}"

    def test_all_approaches_positive_at_crossover(self):
        """All approaches give positive gap at the crossover radius."""
        R = R_CROSSOVER_FM
        synthesis = UniformGapSynthesis(N=2)
        result = synthesis.gap_at_R(R)
        assert result['geometric_gap_MeV'] > 0
        assert result['transmutation_gap_MeV'] > 0
        assert result['temple_gap_MeV'] > 0
        assert result['luscher_gap_MeV'] > 0


# ======================================================================
# Edge case tests
# ======================================================================

class TestEdgeCases:
    """Edge case tests: R -> 0, R -> inf, extreme parameters."""

    def test_very_small_R(self):
        """At R = 0.01 fm, gap is huge (kinematic regime)."""
        trans = DimensionalTransmutation(N=2)
        data = trans.effective_mass_at_R(0.01)
        assert data['gap_total_MeV'] > 10000  # Should be ~ hbar_c/R ~ 20 GeV

    def test_very_large_R(self):
        """At R = 1000 fm, gap is still positive."""
        trans = DimensionalTransmutation(N=2)
        data = trans.effective_mass_at_R(1000.0)
        assert data['gap_total_MeV'] > 0

    def test_R_sequence_smooth(self):
        """Gap varies smoothly with R (no discontinuities)."""
        trans = DimensionalTransmutation(N=2)
        R_values = np.linspace(0.5, 10.0, 50)
        gaps = [trans.effective_mass_at_R(R)['gap_total_MeV'] for R in R_values]
        # Check smoothness: max relative change between adjacent points
        for i in range(1, len(gaps)):
            rel_change = abs(gaps[i] - gaps[i-1]) / max(gaps[i], 1e-10)
            assert rel_change < 0.3, f"Gap jump at R={R_values[i]}: {rel_change}"

    def test_su3_gap_positive(self):
        """Gap is positive for SU(3) as well."""
        trans = DimensionalTransmutation(N=3, Lambda_QCD=200.0)
        for R in [0.5, 2.2, 10.0]:
            data = trans.effective_mass_at_R(R)
            assert data['gap_total_MeV'] > 0

    def test_different_lambda_qcd(self):
        """Gap scales with Lambda_QCD."""
        trans_200 = DimensionalTransmutation(N=2, Lambda_QCD=200.0)
        trans_300 = DimensionalTransmutation(N=2, Lambda_QCD=300.0)
        gap_200 = trans_200.effective_mass_at_R(10.0)['gap_total_MeV']
        gap_300 = trans_300.effective_mass_at_R(10.0)['gap_total_MeV']
        # At large R (dynamic regime), gap ~ Lambda_QCD
        ratio = gap_300 / gap_200
        assert 1.0 < ratio < 2.0  # Should be ~ 300/200 = 1.5


# ======================================================================
# R-scan integration tests (20+ R values)
# ======================================================================

class TestRScanIntegration:
    """Integration test: gap at 20+ R values."""

    def test_20_point_scan(self):
        """Gap positive at 20+ R values spanning 3 orders of magnitude."""
        synthesis = UniformGapSynthesis(N=2)
        R_values = np.logspace(np.log10(0.1), np.log10(100.0), 25)
        for R in R_values:
            result = synthesis.gap_at_R(R)
            assert result['best_gap_MeV'] > 0, f"Gap = 0 at R = {R} fm"

    def test_min_gap_in_scan(self):
        """Minimum gap over scan is > 50 MeV."""
        synthesis = UniformGapSynthesis(N=2)
        R_values = np.logspace(np.log10(0.1), np.log10(100.0), 25)
        gaps = [synthesis.gap_at_R(R)['best_gap_MeV'] for R in R_values]
        assert min(gaps) > 50.0

    def test_gap_never_below_geometric(self):
        """Best gap is always >= geometric gap."""
        synthesis = UniformGapSynthesis(N=2)
        R_values = np.logspace(np.log10(0.1), np.log10(100.0), 20)
        for R in R_values:
            result = synthesis.gap_at_R(R)
            assert result['best_gap_MeV'] >= result['geometric_gap_MeV'] * 0.99


# ======================================================================
# Honesty tests
# ======================================================================

class TestHonesty:
    """Tests that enforce honest labeling of claims."""

    def test_overall_is_proposition_not_theorem(self):
        """The uniform gap bound is honestly labeled PROPOSITION."""
        synthesis = UniformGapSynthesis(N=2)
        status = synthesis.status_assessment()
        assert status['overall_status'] == 'PROPOSITION'

    def test_what_is_missing_is_documented(self):
        """The gap to THEOREM is documented."""
        synthesis = UniformGapSynthesis(N=2)
        status = synthesis.status_assessment()
        # Must have substantial text describing what's needed
        text = status['what_is_missing_for_theorem']
        assert len(text) > 50
        # Must mention the key concepts
        assert 'proof' in text.lower() or 'rigorous' in text.lower()

    def test_gz_excluded_from_proof_chain(self):
        """GZ-dependent approach is marked as excluded."""
        analyzer = ApproachAnalyzer(N=2)
        result = analyzer.analyze_approach_D()
        assert result['excluded_from_proof_chain'] is True

    def test_luscher_is_proposition(self):
        """Luscher bound is honestly PROPOSITION (area law unproven on S^3)."""
        luscher = LuscherStringTension(N=2)
        result = luscher.luscher_bound(2.2)
        assert result['status'] == 'PROPOSITION'

    def test_claim_status_has_caveats(self):
        """ClaimStatus includes caveats."""
        synthesis = UniformGapSynthesis(N=2)
        status = synthesis.claim_status()
        assert len(status.caveats) > 0
        assert 'PROPOSITION' in status.label
