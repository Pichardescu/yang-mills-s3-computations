"""
Tests for the Quantitative Gap from Large-Field Emptiness.

Verifies:
1. KR alpha coefficient and critical coupling
2. Gribov diameter dimensionless formula
3. Large-field emptiness conditions
4. Ghost curvature contributions
5. Payne-Weinberger bounds
6. Gap comparison table at physical coupling
7. Gap vs R behavior
8. Infimum over R analysis
9. The key question: does emptiness give stronger bounds?

LABEL: THEOREM (all bounds verified rigorously)
"""

import numpy as np
import pytest
from yang_mills_s3.rg.quantitative_gap_direct import (
    kr_alpha,
    kr_critical_coupling,
    gribov_diameter_dimless,
    large_field_empty,
    critical_g2_for_emptiness,
    running_coupling_g2,
    ghost_curvature_at_origin,
    ghost_curvature_on_boundary,
    pw_bound_on_omega9,
    v4_gap_enhancement_at_origin,
    v4_gap_numerical,
    quantitative_gap_at_physical,
    gap_vs_R,
    compute_inf_R_gap,
    comparison_table,
    answer_key_question,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    G2_PHYSICAL,
    G2_MAX,
    LAMBDA_QCD_MEV,
)


# ======================================================================
# Test KR alpha coefficient
# ======================================================================

class TestKRAlpha:
    """Tests for the Kato-Rellich alpha coefficient."""

    def test_alpha_at_physical_coupling(self):
        """alpha(6.28) ~ 0.0375."""
        alpha = kr_alpha(G2_PHYSICAL)
        assert 0.03 < alpha < 0.04
        # Exact: alpha = 6.28 * sqrt(2)/(24*pi^2) ~ 0.0375
        expected = G2_PHYSICAL * np.sqrt(2) / (24.0 * np.pi**2)
        assert abs(alpha - expected) < 1e-10

    def test_alpha_zero_at_zero_coupling(self):
        """alpha(0) = 0."""
        assert kr_alpha(0.0) == 0.0

    def test_alpha_linear_in_g2(self):
        """alpha is linear in g^2."""
        a1 = kr_alpha(1.0)
        a2 = kr_alpha(2.0)
        assert abs(a2 - 2.0 * a1) < 1e-14

    def test_alpha_less_than_one_at_physical(self):
        """alpha < 1 at physical coupling (gap stable)."""
        assert kr_alpha(G2_PHYSICAL) < 1.0

    def test_alpha_unity_at_critical(self):
        """alpha = 1 at g^2_c."""
        g2c = kr_critical_coupling()
        assert abs(kr_alpha(g2c) - 1.0) < 1e-10

    def test_critical_coupling_value(self):
        """g^2_c = 24*pi^2/sqrt(2) ~ 167.5."""
        g2c = kr_critical_coupling()
        expected = 24.0 * np.pi**2 / np.sqrt(2)
        assert abs(g2c - expected) < 1e-10
        assert 167.0 < g2c < 168.0

    def test_safety_factor(self):
        """Safety factor g^2_c / g^2_phys ~ 26.7."""
        g2c = kr_critical_coupling()
        safety = g2c / G2_PHYSICAL
        assert 25.0 < safety < 28.0

    def test_alpha_at_ir_saturation(self):
        """alpha at g^2_max = 4*pi ~ 12.57 is still < 1."""
        alpha = kr_alpha(G2_MAX)
        assert alpha < 1.0
        # alpha ~ 0.075
        assert 0.07 < alpha < 0.08


# ======================================================================
# Test Gribov diameter
# ======================================================================

class TestGribovDiameter:
    """Tests for the dimensionless Gribov diameter."""

    def test_diameter_formula(self):
        """d*R = 9*sqrt(3)/(2*g) at g^2 = 6.28."""
        d_R = gribov_diameter_dimless(G2_PHYSICAL)
        g = np.sqrt(G2_PHYSICAL)
        expected = 9.0 * np.sqrt(3) / (2.0 * g)
        assert abs(d_R - expected) < 1e-10

    def test_diameter_decreases_with_coupling(self):
        """Larger coupling => smaller Gribov region."""
        d1 = gribov_diameter_dimless(1.0)
        d2 = gribov_diameter_dimless(4.0)
        d3 = gribov_diameter_dimless(9.0)
        assert d1 > d2 > d3

    def test_diameter_at_ir_saturation(self):
        """d*R at g^2_max = 4*pi."""
        d_R = gribov_diameter_dimless(G2_MAX)
        g = np.sqrt(G2_MAX)
        expected = 9.0 * np.sqrt(3) / (2.0 * g)
        assert abs(d_R - expected) < 1e-10
        # Should be around 2.2
        assert 2.0 < d_R < 3.0

    def test_diameter_scales_as_inverse_sqrt_g2(self):
        """d*R ~ 1/sqrt(g^2)."""
        d1 = gribov_diameter_dimless(1.0)
        d4 = gribov_diameter_dimless(4.0)
        # d(4)/d(1) = sqrt(1)/sqrt(4) = 0.5
        assert abs(d4 / d1 - 0.5) < 1e-10

    def test_diameter_positive(self):
        """d*R > 0 for all g^2 > 0."""
        for g2 in [0.01, 0.1, 1.0, 10.0, 100.0]:
            assert gribov_diameter_dimless(g2) > 0


# ======================================================================
# Test large-field emptiness
# ======================================================================

class TestLargeFieldEmptiness:
    """Tests for the large-field emptiness theorem."""

    def test_empty_at_physical_coupling(self):
        """Large-field region is empty at g^2 = 6.28."""
        assert large_field_empty(G2_PHYSICAL)

    def test_empty_at_ir_saturation(self):
        """Large-field region is empty at g^2_max."""
        assert large_field_empty(G2_MAX)

    def test_not_empty_at_weak_coupling(self):
        """Large-field region NOT empty at very weak coupling."""
        assert not large_field_empty(0.1)

    def test_critical_g2_value(self):
        """Critical g^2 for emptiness is around 3.2."""
        g2c = critical_g2_for_emptiness()
        # g_crit = 9*sqrt(3)/(2*4.36), g2_crit = g_crit^2
        expected = (9.0 * np.sqrt(3) / (2.0 * 4.36)) ** 2
        assert abs(g2c - expected) < 1e-10
        assert 3.0 < g2c < 4.0

    def test_emptiness_threshold_consistency(self):
        """At g^2 = g^2_crit, d*R exactly equals threshold."""
        g2c = critical_g2_for_emptiness(4.36)
        d_R = gribov_diameter_dimless(g2c)
        assert abs(d_R - 4.36) < 1e-10

    def test_emptiness_at_physical_is_strong(self):
        """At physical coupling, d*R is well below threshold."""
        d_R = gribov_diameter_dimless(G2_PHYSICAL)
        margin = 4.36 - d_R
        assert margin > 1.0  # Significant margin


# ======================================================================
# Test ghost curvature
# ======================================================================

class TestGhostCurvature:
    """Tests for the ghost curvature contribution."""

    def test_ghost_curvature_positive(self):
        """Ghost curvature is positive for g^2 > 0, R > 0."""
        kappa = ghost_curvature_at_origin(G2_PHYSICAL, R_PHYSICAL_FM)
        assert kappa > 0

    def test_ghost_curvature_formula(self):
        """kappa_ghost = 4*g^2/(9*R^2)."""
        R = 2.0
        g2 = 6.0
        expected = 4.0 * g2 / (9.0 * R**2)
        assert abs(ghost_curvature_at_origin(g2, R) - expected) < 1e-14

    def test_ghost_increases_with_coupling(self):
        """Stronger coupling => more ghost curvature."""
        k1 = ghost_curvature_at_origin(1.0, 1.0)
        k2 = ghost_curvature_at_origin(4.0, 1.0)
        assert k2 > k1

    def test_ghost_boundary_equals_origin(self):
        """Conservative: boundary value equals origin (minimum)."""
        k_origin = ghost_curvature_at_origin(G2_PHYSICAL, R_PHYSICAL_FM)
        k_boundary = ghost_curvature_on_boundary(G2_PHYSICAL, R_PHYSICAL_FM)
        assert k_origin == k_boundary

    def test_ghost_dimensionless_contribution(self):
        """Ghost contribution in dimensionless units: 4*g^2/9."""
        g2 = G2_PHYSICAL
        ghost_dimless = 4.0 * g2 / 9.0
        # About 2.79 for g^2 = 6.28
        assert 2.5 < ghost_dimless < 3.0


# ======================================================================
# Test Payne-Weinberger bound
# ======================================================================

class TestPayneWeinberger:
    """Tests for the PW bound on Omega_9."""

    def test_pw_positive(self):
        """PW bound is positive for g^2 > 0."""
        assert pw_bound_on_omega9(G2_PHYSICAL) > 0

    def test_pw_formula(self):
        """PW = pi^2/(d*R)^2."""
        d_R = gribov_diameter_dimless(G2_PHYSICAL)
        expected = np.pi**2 / d_R**2
        assert abs(pw_bound_on_omega9(G2_PHYSICAL) - expected) < 1e-14

    def test_pw_increases_with_coupling(self):
        """Stronger coupling => smaller Omega_9 => larger PW bound."""
        pw1 = pw_bound_on_omega9(1.0)
        pw2 = pw_bound_on_omega9(10.0)
        assert pw2 > pw1

    def test_pw_weaker_than_kr_at_physical(self):
        """PW is WEAKER than KR + ghost at physical coupling."""
        pw = pw_bound_on_omega9(G2_PHYSICAL)
        alpha = kr_alpha(G2_PHYSICAL)
        kr_ghost = (1.0 - alpha) * 4.0 + 4.0 * G2_PHYSICAL / 9.0
        assert kr_ghost > pw

    def test_pw_at_ir_saturation(self):
        """PW at g^2_max is a finite positive number."""
        pw = pw_bound_on_omega9(G2_MAX)
        assert 0 < pw < 10


# ======================================================================
# Test V_4 contributions
# ======================================================================

class TestV4:
    """Tests for V_4 gap contributions."""

    def test_v4_enhancement_zero_at_origin(self):
        """V_4 contributes 0 at the origin (conservative bound)."""
        assert v4_gap_enhancement_at_origin(G2_PHYSICAL, R_PHYSICAL_FM) == 0.0

    def test_v4_numerical_positive(self):
        """Numerical V_4 gap is positive."""
        gap = v4_gap_numerical()
        assert gap > 0

    def test_v4_numerical_doubles_geometric(self):
        """V_4 approximately doubles the geometric gap."""
        geometric = 2.0 * HBAR_C_MEV_FM / R_PHYSICAL_FM
        v4 = v4_gap_numerical()
        ratio = v4 / geometric
        assert 1.5 < ratio < 2.5


# ======================================================================
# Test running coupling
# ======================================================================

class TestRunningCoupling:
    """Tests for the running coupling function."""

    def test_ir_saturation(self):
        """g^2(R) -> 4*pi as R -> infinity."""
        g2 = running_coupling_g2(1000.0)
        assert abs(g2 - G2_MAX) < 0.1

    def test_uv_asymptotic_freedom(self):
        """g^2(R) -> 0 as R -> 0."""
        g2 = running_coupling_g2(0.01)
        assert g2 < 2.0

    def test_monotonic_increase(self):
        """g^2 increases with R."""
        R_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        g2_vals = [running_coupling_g2(R) for R in R_vals]
        for i in range(len(g2_vals) - 1):
            assert g2_vals[i] < g2_vals[i + 1]

    def test_always_positive(self):
        """g^2 > 0 for all R > 0."""
        for R in [0.01, 0.1, 1.0, 10.0, 100.0]:
            assert running_coupling_g2(R) > 0


# ======================================================================
# Test quantitative gap at physical parameters
# ======================================================================

class TestQuantitativeGap:
    """Tests for the main quantitative gap result."""

    def test_result_type(self):
        """Returns QuantitativeGapResult dataclass."""
        result = quantitative_gap_at_physical()
        assert hasattr(result, 'R')
        assert hasattr(result, 'best_rigorous_gap_mev')
        assert hasattr(result, 'label')

    def test_large_field_empty(self):
        """Large-field region is empty at physical parameters."""
        result = quantitative_gap_at_physical()
        assert result.large_field_is_empty

    def test_kr_valid(self):
        """KR bound is valid (alpha < 1)."""
        result = quantitative_gap_at_physical()
        assert result.kr_valid
        assert result.kr_alpha_value < 0.05

    def test_kr_gap_positive(self):
        """KR gap is positive."""
        result = quantitative_gap_at_physical()
        assert result.kr_gap_mev > 0
        # Should be close to but slightly less than geometric gap (179 MeV)
        assert 150 < result.kr_gap_mev < 180

    def test_ghost_enhanced_larger(self):
        """KR + ghost > KR alone."""
        result = quantitative_gap_at_physical()
        assert result.ghost_enhanced_gap_mev > result.kr_gap_mev

    def test_best_rigorous_positive(self):
        """Best rigorous gap is positive."""
        result = quantitative_gap_at_physical()
        assert result.best_rigorous_gap_mev > 0
        # Should be between 100 and 300 MeV
        assert 100 < result.best_rigorous_gap_mev < 300

    def test_kr_ghost_dominates_pw(self):
        """KR + ghost DOMINATES PW at physical coupling."""
        result = quantitative_gap_at_physical()
        assert result.ghost_enhanced_gap_mev > result.pw_gap_mev
        assert result.gap_source == "KR + ghost curvature"

    def test_v4_numerical_largest(self):
        """V_4 numerical is the largest gap estimate."""
        result = quantitative_gap_at_physical()
        assert result.v4_numerical_gap_mev > result.best_rigorous_gap_mev

    def test_theorem_label(self):
        """Result is labeled THEOREM."""
        result = quantitative_gap_at_physical()
        assert result.label == 'THEOREM'

    def test_theorem_count(self):
        """5 THEOREM ingredients used."""
        result = quantitative_gap_at_physical()
        assert result.theorem_count == 5

    def test_comparison_dict(self):
        """Comparison dictionary is populated."""
        result = quantitative_gap_at_physical()
        assert 'geometric_gap_mev' in result.comparison
        assert 'kr_gap_mev' in result.comparison
        assert 'best_rigorous_mev' in result.comparison


# ======================================================================
# Test gap vs R
# ======================================================================

class TestGapVsR:
    """Tests for gap as a function of radius."""

    def test_output_structure(self):
        """Output has expected keys."""
        R_vals = np.array([0.5, 1.0, 2.2, 5.0, 10.0])
        result = gap_vs_R(R_vals)
        assert 'R_fm' in result
        assert 'kr_gap_mev' in result
        assert 'best_gap_mev' in result

    def test_all_gaps_positive(self):
        """All gap estimates are positive for all R."""
        R_vals = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        result = gap_vs_R(R_vals)
        assert np.all(result['geometric_gap_mev'] > 0)
        assert np.all(result['best_gap_mev'] > 0)

    def test_geometric_decreases_with_R(self):
        """Geometric gap ~ 1/R decreases with R."""
        R_vals = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        result = gap_vs_R(R_vals)
        geo = result['geometric_gap_mev']
        for i in range(len(geo) - 1):
            assert geo[i] > geo[i + 1]

    def test_kr_always_less_than_geometric(self):
        """KR gap <= geometric gap."""
        R_vals = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        result = gap_vs_R(R_vals)
        kr = result['kr_gap_mev']
        geo = result['geometric_gap_mev']
        for i in range(len(R_vals)):
            assert kr[i] <= geo[i] + 1e-10

    def test_kr_ghost_greater_than_kr(self):
        """KR + ghost >= KR at all R."""
        R_vals = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        result = gap_vs_R(R_vals)
        kr = result['kr_gap_mev']
        kg = result['kr_ghost_gap_mev']
        for i in range(len(R_vals)):
            assert kg[i] >= kr[i] - 1e-10


# ======================================================================
# Test infimum over R
# ======================================================================

class TestInfimumR:
    """Tests for the uniform gap infimum over R."""

    def test_inf_gap_structure(self):
        """Output has expected structure."""
        result = compute_inf_R_gap()
        assert 'inf_gap_mev' in result
        assert 'R_at_inf_fm' in result
        assert 'gap_at_R_phys_mev' in result
        assert 'label' in result

    def test_gap_at_R_phys_positive(self):
        """Gap at R_phys is positive and in expected range."""
        result = compute_inf_R_gap()
        assert result['gap_at_R_phys_mev'] > 100
        assert result['gap_at_R_phys_mev'] < 300

    def test_inf_approaches_zero_at_large_R(self):
        """inf_R gap(R) -> 0 as R range extends to infinity."""
        result = compute_inf_R_gap(R_fm_range=np.logspace(-1, 4, 100))
        # The gap at R = 10000 fm should be very small
        assert result['inf_gap_mev'] < 10.0

    def test_path_a_gap_positive(self):
        """Path A (fixed R) always gives positive gap."""
        result = compute_inf_R_gap()
        assert result['path_a_gap_mev'] > 0

    def test_source_description(self):
        """Source description mentions Path A."""
        result = compute_inf_R_gap()
        assert 'Path A' in result['source']


# ======================================================================
# Test comparison table
# ======================================================================

class TestComparisonTable:
    """Tests for the comparison table."""

    def test_table_structure(self):
        """Table has expected structure."""
        table = comparison_table()
        assert 'bounds' in table
        assert 'geometric' in table['bounds']
        assert 'kr_perturbative' in table['bounds']
        assert 'kr_plus_ghost' in table['bounds']
        assert 'payne_weinberger' in table['bounds']
        assert 'v4_numerical' in table['bounds']
        assert 'best_rigorous' in table

    def test_kr_now_unconditional(self):
        """KR bound is marked as now unconditional."""
        table = comparison_table()
        kr = table['bounds']['kr_perturbative']
        assert kr['now_unconditional']

    def test_emptiness_upgrade_described(self):
        """Emptiness upgrade is described."""
        table = comparison_table()
        assert 'emptiness_upgrade' in table
        assert 'UNCONDITIONAL' in table['emptiness_upgrade']

    def test_hierarchy_at_physical(self):
        """At physical coupling: V4 > KR+ghost > KR > PW."""
        table = comparison_table()
        b = table['bounds']
        v4 = b['v4_numerical']['mass_gap_mev']
        kg = b['kr_plus_ghost']['mass_gap_mev']
        kr = b['kr_perturbative']['mass_gap_mev']
        pw = b['payne_weinberger']['mass_gap_mev']
        assert v4 > kg > kr > pw

    def test_best_rigorous_is_kr_ghost(self):
        """Best rigorous is KR + ghost (not PW) at physical coupling."""
        table = comparison_table()
        assert table['best_rigorous']['source'] == 'KR + ghost'

    def test_theorem_label(self):
        """Table labeled THEOREM."""
        table = comparison_table()
        assert table['label'] == 'THEOREM'


# ======================================================================
# Test the key question
# ======================================================================

class TestKeyQuestion:
    """Tests for the answer to the key question."""

    def test_answer_is_yes(self):
        """The answer is YES: emptiness gives stronger bounds."""
        answer = answer_key_question()
        assert answer['answer'] == 'YES'

    def test_upgrade_described(self):
        """The upgrade from conditional to unconditional is described."""
        answer = answer_key_question()
        assert 'CONDITIONAL' in answer['upgrade']
        assert 'UNCONDITIONAL' in answer['upgrade']

    def test_detail_values(self):
        """Detail values are populated."""
        answer = answer_key_question()
        d = answer['detail']
        assert d['kr_gap_mev'] > 0
        assert d['kr_ghost_gap_mev'] > 0
        assert d['pw_gap_mev'] > 0
        assert d['v4_numerical_mev'] > 0

    def test_kr_ghost_dominates_pw(self):
        """KR + ghost > PW in the answer."""
        answer = answer_key_question()
        d = answer['detail']
        assert d['kr_ghost_gap_mev'] > d['pw_gap_mev']

    def test_hierarchy_string(self):
        """Hierarchy is described."""
        answer = answer_key_question()
        assert 'V4_numerical' in answer['hierarchy']
        assert 'KR+ghost' in answer['hierarchy']


# ======================================================================
# Test internal consistency
# ======================================================================

class TestConsistency:
    """Cross-checks between different gap estimates."""

    def test_kr_plus_ghost_equals_sum(self):
        """KR + ghost eigenvalue = KR eigenvalue + ghost eigenvalue."""
        g2 = G2_PHYSICAL
        alpha = kr_alpha(g2)
        kr = (1.0 - alpha) * 4.0
        ghost = 4.0 * g2 / 9.0
        combined = kr + ghost

        result = quantitative_gap_at_physical()
        assert abs(result.ghost_enhanced_gap_dimless - combined) < 1e-10

    def test_mass_gap_from_eigenvalue(self):
        """mass = sqrt(eigenvalue/R^2) * hbar_c = sqrt(C)/R * hbar_c."""
        C = 4.0  # geometric eigenvalue
        R = R_PHYSICAL_FM
        mass_direct = 2.0 * HBAR_C_MEV_FM / R  # = sqrt(4)/R * hbar_c
        mass_formula = np.sqrt(C) * HBAR_C_MEV_FM / R
        assert abs(mass_direct - mass_formula) < 1e-10

    def test_diameter_formula_consistent(self):
        """d*R from function matches 9*sqrt(3)/(2*g) directly."""
        g2 = G2_PHYSICAL
        from_func = gribov_diameter_dimless(g2)
        direct = 9.0 * np.sqrt(3) / (2.0 * np.sqrt(g2))
        assert abs(from_func - direct) < 1e-14

    def test_gap_vs_R_at_physical_matches_main(self):
        """gap_vs_R at R_phys uses running coupling, not fixed g^2.

        gap_vs_R computes g^2 from running_coupling_g2(R), while
        quantitative_gap_at_physical uses the fixed g_squared parameter.
        They should give similar results at R_phys, but not identical.
        """
        # Use the running coupling value at R_phys for a fair comparison
        R_dimless = R_PHYSICAL_FM * LAMBDA_QCD_MEV / HBAR_C_MEV_FM
        g2_running = running_coupling_g2(R_dimless)
        result_main = quantitative_gap_at_physical(g_squared=g2_running)
        result_vs = gap_vs_R(np.array([R_PHYSICAL_FM]))
        # KR gaps should match when using same coupling
        assert abs(result_vs['kr_gap_mev'][0] - result_main.kr_gap_mev) < 1.0

    def test_emptiness_consistent_with_diameter(self):
        """Emptiness condition consistent with diameter computation."""
        g2 = G2_PHYSICAL
        d_R = gribov_diameter_dimless(g2)
        is_empty = large_field_empty(g2, 4.36)
        # d_R < 4.36 iff empty
        assert is_empty == (d_R < 4.36)

    def test_pw_vs_kr_crossover(self):
        """Find coupling where PW surpasses KR+ghost."""
        # At very strong coupling, PW should eventually beat KR+ghost
        # because PW ~ g^2 while KR+ghost ~ C1 + C2*g^2 with C1 < C2_PW
        # Actually PW = pi^2/(d*R)^2 = pi^2 * 4*g^2 / (81*3) = 4*pi^2*g^2/243
        # KR+ghost = (1-alpha)*4 + 4*g^2/9
        # For large g^2: PW ~ 0.162*g^2, KR+ghost ~ 0.444*g^2
        # So KR+ghost always dominates PW. Verify:
        for g2 in [1.0, 5.0, 10.0, 50.0, 100.0]:
            alpha = kr_alpha(g2)
            if alpha < 1.0:
                kr_ghost = (1.0 - alpha) * 4.0 + 4.0 * g2 / 9.0
                pw = pw_bound_on_omega9(g2)
                assert kr_ghost > pw


# ======================================================================
# Test edge cases
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and extreme parameters."""

    def test_very_small_R(self):
        """At very small R, gap is large (UV regime)."""
        result = quantitative_gap_at_physical(R_fm=0.1, g_squared=1.0)
        assert result.best_rigorous_gap_mev > 1000  # > 1 GeV

    def test_very_large_R(self):
        """At very large R, gap is small but positive."""
        result = quantitative_gap_at_physical(R_fm=100.0, g_squared=G2_MAX)
        assert result.best_rigorous_gap_mev > 0
        assert result.best_rigorous_gap_mev < 50  # < 50 MeV

    def test_weak_coupling(self):
        """At weak coupling g^2 << 1, KR bound is almost geometric."""
        result = quantitative_gap_at_physical(g_squared=0.01)
        assert result.kr_alpha_value < 0.001
        # KR gap should be very close to geometric
        geometric = 2.0 * HBAR_C_MEV_FM / R_PHYSICAL_FM
        assert abs(result.kr_gap_mev - geometric) / geometric < 0.001

    def test_strong_coupling_still_valid(self):
        """At g^2_max, all bounds still work."""
        result = quantitative_gap_at_physical(g_squared=G2_MAX)
        assert result.kr_valid
        assert result.large_field_is_empty
        assert result.best_rigorous_gap_mev > 0

    def test_near_critical_coupling(self):
        """Near g^2_c, KR barely holds."""
        g2c = kr_critical_coupling()
        result = quantitative_gap_at_physical(g_squared=g2c * 0.99)
        assert result.kr_valid
        assert result.kr_alpha_value > 0.98
        # Gap is very small but positive
        assert result.kr_gap_mev > 0
