"""
Tests for Phase 4: The R -> infinity limit.

Tests the RLimitAnalysis class which analyzes whether the Yang-Mills mass gap
persists as the radius R of S^3 goes to infinity (recovering flat space).

Test categories:
    1. Geometric gap properties
    2. Running coupling behavior (asymptotic freedom)
    3. Dynamical gap R-independence
    4. Total gap positivity (KEY: gap > 0 for all R)
    5. Crossover radius
    6. Path A consistency checks
    7. Path B: gap > 0 for all finite R
    8. Confinement argument structure
    9. Honest assessment label correctness
   10. Gap vs radius table monotonic approach to Lambda_QCD
   11. Dimensional transmutation consistency
   12. Edge cases and limits
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.r_limit import (
    RLimitAnalysis,
    ClaimStatus,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
    SQRT5,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def analysis():
    """RLimitAnalysis instance for SU(2), Lambda_QCD = 200 MeV."""
    return RLimitAnalysis(N=2, Lambda_QCD=200.0)


@pytest.fixture
def analysis_su3():
    """RLimitAnalysis instance for SU(3), Lambda_QCD = 200 MeV."""
    return RLimitAnalysis(N=3, Lambda_QCD=200.0)


@pytest.fixture
def crossover(analysis):
    """Crossover radius data."""
    return analysis.crossover_radius()


# ======================================================================
# 1. Geometric gap
# ======================================================================

class TestGeometricGap:
    """The geometric gap is 2*hbar_c/R. THEOREM status."""

    def test_formula(self, analysis):
        """Gap = 2 * hbar_c / R."""
        R = 2.2
        expected = 2.0 * HBAR_C_MEV_FM / R
        actual = analysis.geometric_gap(R)
        assert abs(actual - expected) < 1e-10

    def test_decreases_as_one_over_R(self, analysis):
        """Gap decreases as 1/R."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        gaps = [analysis.geometric_gap(R) for R in R_values]

        # Each subsequent gap should be smaller
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1], \
                f"Gap should decrease: gap({R_values[i]}) = {gaps[i]} " \
                f"should be > gap({R_values[i+1]}) = {gaps[i+1]}"

    def test_scaling(self, analysis):
        """Doubling R halves the gap."""
        R = 3.0
        gap1 = analysis.geometric_gap(R)
        gap2 = analysis.geometric_gap(2 * R)
        assert abs(gap1 / gap2 - 2.0) < 1e-10

    def test_positive_for_all_R(self, analysis):
        """Gap is positive for any finite R > 0."""
        for R in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1e6]:
            gap = analysis.geometric_gap(R)
            assert gap > 0, f"Geometric gap should be positive at R={R}"

    def test_approaches_zero(self, analysis):
        """Gap -> 0 as R -> infinity."""
        gap = analysis.geometric_gap(1e10)
        assert gap < 1e-4, f"Gap should approach 0 for large R, got {gap}"

    def test_diverges_at_small_R(self, analysis):
        """Gap -> infinity as R -> 0."""
        gap = analysis.geometric_gap(0.001)
        assert gap > 1e5, f"Gap should be large for small R, got {gap}"

    def test_invalid_R(self, analysis):
        """Negative or zero R should raise ValueError."""
        with pytest.raises(ValueError):
            analysis.geometric_gap(0.0)
        with pytest.raises(ValueError):
            analysis.geometric_gap(-1.0)

    def test_at_physical_radius(self, analysis):
        """At R = 2.2 fm, gap ~ 179 MeV."""
        gap = analysis.geometric_gap(2.2)
        # 2 * 197.3 / 2.2 ~ 179.4
        assert 175 < gap < 185, \
            f"At R=2.2 fm, gap should be ~179 MeV, got {gap:.1f}"


# ======================================================================
# 2. Running coupling (asymptotic freedom)
# ======================================================================

class TestRunningCoupling:
    """Running coupling g^2(R) from 1-loop beta function."""

    def test_perturbative_regime(self, analysis):
        """At small R (high energy), coupling is small and perturbative."""
        result = analysis.running_coupling(0.1)  # R = 0.1 fm, mu ~ 2 GeV
        assert result['perturbative'] is True
        assert result['g_squared'] > 0
        assert result['g_squared'] < 100  # should be finite

    def test_coupling_increases_with_R(self, analysis):
        """g^2 increases as R increases (asymptotic freedom in reverse)."""
        # Only in perturbative regime (R < hbar_c/Lambda)
        R_pert_max = HBAR_C_MEV_FM / analysis.Lambda_QCD  # ~ 0.99 fm
        R_values = [0.05, 0.1, 0.2, 0.5]
        couplings = [analysis.running_coupling(R)['g_squared'] for R in R_values]

        for i in range(len(couplings) - 1):
            assert couplings[i] < couplings[i + 1], \
                f"Coupling should increase: g^2({R_values[i]}) = {couplings[i]} " \
                f"should be < g^2({R_values[i+1]}) = {couplings[i+1]}"

    def test_nonperturbative_regime(self, analysis):
        """At large R (low energy), coupling diverges."""
        # R > hbar_c / Lambda_QCD ~ 1 fm => non-perturbative
        result = analysis.running_coupling(5.0)
        assert result['perturbative'] is False
        assert result['g_squared'] == float('inf')

    def test_landau_pole(self, analysis):
        """At R = hbar_c/Lambda_QCD, log argument = 1, coupling diverges."""
        R_pole = HBAR_C_MEV_FM / analysis.Lambda_QCD
        result = analysis.running_coupling(R_pole)
        # At the pole, log_arg = 1, ln(1) = 0, g^2 = infinity
        assert result['perturbative'] is False
        assert result['g_squared'] == float('inf')

    def test_alpha_s_reasonable(self, analysis):
        """alpha_s at typical perturbative scales should be O(0.1-0.5)."""
        result = analysis.running_coupling(0.1)  # mu ~ 2 GeV
        alpha_s = result['alpha_s']
        # At 2 GeV, alpha_s ~ 0.3 experimentally
        assert 0.05 < alpha_s < 2.0, \
            f"alpha_s should be reasonable, got {alpha_s}"

    def test_b0_coefficient(self, analysis):
        """Beta function coefficient b_0 = 11*N/3."""
        assert abs(analysis.b0 - 11.0 * 2 / 3.0) < 1e-14

    def test_b0_su3(self, analysis_su3):
        """b_0 = 11 for SU(3)."""
        assert abs(analysis_su3.b0 - 11.0) < 1e-14

    def test_energy_scale(self, analysis):
        """mu = hbar_c / R."""
        R = 0.5
        result = analysis.running_coupling(R)
        expected_mu = HBAR_C_MEV_FM / R
        assert abs(result['mu'] - expected_mu) < 1e-10


# ======================================================================
# 3. Dynamical gap R-independence
# ======================================================================

class TestDynamicalGap:
    """The dynamical gap ~ Lambda_QCD, independent of R."""

    def test_equals_lambda_qcd(self, analysis):
        """Dynamical gap estimate = Lambda_QCD (conservative)."""
        for R in [0.5, 1.0, 2.2, 5.0, 100.0]:
            result = analysis.dynamical_gap_estimate(R)
            assert abs(result['gap_MeV'] - analysis.Lambda_QCD) < 1e-10, \
                f"Dynamical gap at R={R} should equal Lambda_QCD"

    def test_independent_of_R(self, analysis):
        """Gap should be the SAME at all R."""
        gaps = [analysis.dynamical_gap_estimate(R)['gap_MeV']
                for R in [0.1, 1.0, 10.0, 100.0, 1000.0]]
        for i in range(1, len(gaps)):
            assert abs(gaps[i] - gaps[0]) < 1e-10, \
                f"Dynamical gap should be R-independent"

    def test_regime_classification(self, analysis):
        """Correct regime classification at various R."""
        R_star = analysis.crossover_radius()['R_star_fm']

        # Small R: geometric dominates
        result_small = analysis.dynamical_gap_estimate(0.1 * R_star)
        assert result_small['regime'] == 'geometric_dominates'

        # Large R: dynamical dominates
        result_large = analysis.dynamical_gap_estimate(10.0 * R_star)
        assert result_large['regime'] == 'dynamical_dominates'

    def test_positive(self, analysis):
        """Dynamical gap is always positive."""
        for R in [0.01, 1.0, 100.0, 1e6]:
            result = analysis.dynamical_gap_estimate(R)
            assert result['gap_MeV'] > 0


# ======================================================================
# 4. Total gap positivity (KEY TEST)
# ======================================================================

class TestTotalGap:
    """
    Total gap = max(geometric, dynamical) is positive for ALL R.

    This is the KEY result of Phase 4: the gap never reaches zero.
    """

    def test_positive_for_all_R_tested(self, analysis):
        """Gap > 0 for R from 0.01 fm to 10^6 fm."""
        R_values = np.logspace(-2, 6, 100)  # 0.01 to 10^6 fm
        for R in R_values:
            result = analysis.total_gap(R)
            assert result['gap_positive'], \
                f"Total gap should be positive at R={R} fm, " \
                f"got {result['total_gap_MeV']} MeV"

    def test_gap_floor_at_lambda_qcd(self, analysis):
        """At large R, gap approaches Lambda_QCD from above."""
        for R in [10.0, 50.0, 100.0, 1000.0]:
            result = analysis.total_gap(R)
            assert result['total_gap_MeV'] >= analysis.Lambda_QCD - 1e-10, \
                f"Gap at R={R} should be >= Lambda_QCD"

    def test_gap_large_at_small_R(self, analysis):
        """At small R, gap is dominated by geometry and much larger than Lambda."""
        result = analysis.total_gap(0.1)
        assert result['total_gap_MeV'] > analysis.Lambda_QCD * 5, \
            "At small R, geometric gap should dominate"

    def test_geometric_dominates_small_R(self, analysis):
        """Geometric gap dominates at small R."""
        result = analysis.total_gap(0.1)
        assert result['geometric_dominates'] == True

    def test_dynamical_dominates_large_R(self, analysis):
        """Dynamical gap dominates at large R."""
        result = analysis.total_gap(100.0)
        assert result['geometric_dominates'] == False

    def test_smooth_gap_always_larger(self, analysis):
        """Smooth gap (quadrature sum) >= max gap."""
        for R in [0.1, 1.0, 2.2, 10.0, 100.0]:
            result = analysis.total_gap(R)
            assert result['smooth_gap_MeV'] >= result['total_gap_MeV'] - 1e-10

    def test_regime_transitions(self, analysis):
        """Regime transitions at expected radii."""
        R_star = analysis.crossover_radius()['R_star_fm']

        # Well below crossover
        assert analysis.total_gap(0.1 * R_star)['regime'] == 'geometric_dominates'

        # Well above crossover
        assert analysis.total_gap(10.0 * R_star)['regime'] == 'dynamical_dominates'

    def test_gap_never_zero(self, analysis):
        """Explicit check: gap > 0 at crossover (minimum region)."""
        R_star = analysis.crossover_radius()['R_star_fm']
        result = analysis.total_gap(R_star)
        assert result['total_gap_MeV'] > 0
        # At crossover, both gaps ~ Lambda_QCD, so total ~ Lambda_QCD
        assert result['total_gap_MeV'] >= analysis.Lambda_QCD - 1e-10

    def test_su3_gap_positive(self, analysis_su3):
        """Gap is also positive for SU(3)."""
        for R in [0.1, 1.0, 2.2, 10.0, 100.0]:
            result = analysis_su3.total_gap(R)
            assert result['gap_positive'], \
                f"SU(3) gap should be positive at R={R}"


# ======================================================================
# 5. Crossover radius
# ======================================================================

class TestCrossoverRadius:
    """R* = 2 * hbar_c / Lambda_QCD ~ 1.97 fm."""

    def test_formula(self, analysis, crossover):
        """R* = 2 * hbar_c / Lambda_QCD."""
        expected = 2.0 * HBAR_C_MEV_FM / analysis.Lambda_QCD
        assert abs(crossover['R_star_fm'] - expected) < 1e-10

    def test_approximately_2_fm(self, crossover):
        """R* ~ 1.97 fm for Lambda_QCD = 200 MeV."""
        R_star = crossover['R_star_fm']
        # 2 * 197.3 / 200 ~ 1.973
        assert 1.5 < R_star < 2.5, \
            f"R* should be ~1.97 fm, got {R_star:.3f}"

    def test_agrees_with_R_phys(self, crossover):
        """R* agrees with physical radius within 15%."""
        agreement = crossover['agreement_percent']
        assert agreement < 15.0, \
            f"R* should agree with R_phys = 2.2 fm within 15%, " \
            f"disagreement: {agreement:.1f}%"

    def test_gap_at_crossover_equals_lambda(self, analysis, crossover):
        """At the crossover, gap = Lambda_QCD."""
        gap_at_xover = crossover['gap_at_crossover_MeV']
        assert abs(gap_at_xover - analysis.Lambda_QCD) < 1e-10

    def test_both_gaps_equal_at_crossover(self, analysis, crossover):
        """At R*, geometric gap = dynamical gap."""
        R_star = crossover['R_star_fm']
        geom = analysis.geometric_gap(R_star)
        dyn = analysis.Lambda_QCD
        assert abs(geom - dyn) < 1e-8, \
            f"At crossover: geom={geom}, dyn={dyn}"

    def test_crossover_scales_with_lambda(self):
        """R* scales as 1/Lambda_QCD."""
        a1 = RLimitAnalysis(N=2, Lambda_QCD=200.0)
        a2 = RLimitAnalysis(N=2, Lambda_QCD=400.0)
        R1 = a1.crossover_radius()['R_star_fm']
        R2 = a2.crossover_radius()['R_star_fm']
        # R* propto 1/Lambda, so R1/R2 = Lambda2/Lambda1 = 2
        assert abs(R1 / R2 - 2.0) < 1e-10


# ======================================================================
# 6. Path A consistency checks
# ======================================================================

class TestPathA:
    """Path A: Ontological — R is physical, R -> inf is unphysical."""

    def test_lambda_match(self, analysis):
        """R = 2.2 fm implies Lambda_QCD ~ 179 MeV."""
        result = analysis.path_a_ontological(R=2.2)
        # The implied Lambda from 2*hbar_c/R
        implied = result['implied_Lambda_MeV']
        # Should be ~179 MeV
        assert abs(implied - 179.4) < 15.0, \
            f"Implied Lambda should be ~179 MeV, got {implied:.1f}"

    def test_lambda_match_fraction(self, analysis):
        """Lambda match should be better than 85% (179 vs 200 MeV)."""
        result = analysis.path_a_ontological(R=2.2)
        assert result['Lambda_match_fraction'] > 0.85

    def test_glueball_ratio(self, analysis):
        """Predicted glueball ratio sqrt(2) matches lattice within 5%."""
        result = analysis.path_a_ontological()
        agreement = result['glueball_ratio_agreement_percent']
        assert agreement > 95.0, \
            f"Glueball ratio agreement should be > 95%, got {agreement:.1f}%"

    def test_glueball_ratio_value(self, analysis):
        """m(2++)/m(0++) = sqrt(2) ~ 1.414."""
        result = analysis.path_a_ontological()
        assert abs(result['glueball_ratio_predicted'] - np.sqrt(2.0)) < 1e-10

    def test_gap_positive(self, analysis):
        """Gap at R = 2.2 fm is positive."""
        result = analysis.path_a_ontological(R=2.2)
        assert result['gap_MeV'] > 0

    def test_gap_approximately_179_MeV(self, analysis):
        """Gap at R = 2.2 fm is approximately 179 MeV."""
        result = analysis.path_a_ontological(R=2.2)
        assert 150.0 < result['gap_MeV'] < 210.0

    def test_kr_gap_positive(self, analysis):
        """KR-corrected gap is also positive."""
        result = analysis.path_a_ontological(R=2.2)
        assert result['kr_gap_MeV'] > 0

    def test_claims_have_correct_labels(self, analysis):
        """Each claim has a valid status label."""
        result = analysis.path_a_ontological()
        valid_labels = {'THEOREM', 'PROPOSITION', 'NUMERICAL', 'CONJECTURE', 'POSTULATE'}
        for claim in result['claims']:
            assert claim.label in valid_labels, \
                f"Invalid label: {claim.label}"

    def test_has_postulate_claim(self, analysis):
        """Path A must include a POSTULATE claim (R is physical)."""
        result = analysis.path_a_ontological()
        labels = [c.label for c in result['claims']]
        assert 'POSTULATE' in labels, \
            "Path A must have a POSTULATE (that R is physical)"

    def test_crossover_reference(self, analysis):
        """Path A includes crossover radius data."""
        result = analysis.path_a_ontological()
        assert 'crossover_radius' in result
        assert result['crossover_radius']['R_star_fm'] > 0


# ======================================================================
# 7. Path B: gap > 0 for all finite R
# ======================================================================

class TestPathB:
    """Path B: Conservative — gap survives R -> infinity."""

    def test_all_gaps_positive(self, analysis):
        """All tested gaps are positive."""
        result = analysis.path_b_conservative()
        assert result['all_gaps_positive'] is True

    def test_min_gap_positive(self, analysis):
        """Minimum gap found is positive."""
        result = analysis.path_b_conservative()
        assert result['min_gap_MeV'] > 0

    def test_min_gap_is_lambda_qcd(self, analysis):
        """Minimum gap should be at or near Lambda_QCD."""
        result = analysis.path_b_conservative()
        min_gap = result['min_gap_MeV']
        # The minimum gap should be Lambda_QCD (at large R)
        assert abs(min_gap - analysis.Lambda_QCD) < 1.0, \
            f"Min gap should be ~Lambda_QCD={analysis.Lambda_QCD}, got {min_gap}"

    def test_gap_at_large_R(self, analysis):
        """Gap at R = 1000 fm is Lambda_QCD."""
        result = analysis.path_b_conservative()
        assert abs(result['gap_at_R1000_MeV'] - analysis.Lambda_QCD) < 1e-10

    def test_limiting_gap(self, analysis):
        """Limiting gap is Lambda_QCD."""
        result = analysis.path_b_conservative()
        assert abs(result['limiting_gap_MeV'] - analysis.Lambda_QCD) < 1e-10

    def test_gap_data_coverage(self, analysis):
        """Gap data covers from small to large R."""
        result = analysis.path_b_conservative()
        R_values = [d['R_fm'] for d in result['gap_data']]
        assert min(R_values) <= 0.1
        assert max(R_values) >= 1000.0

    def test_claims_include_theorem(self, analysis):
        """Path B must include a THEOREM claim (gap > 0 for finite R)."""
        result = analysis.path_b_conservative()
        labels = [c.label for c in result['claims']]
        assert 'THEOREM' in labels

    def test_claims_include_conjecture(self, analysis):
        """Path B must include a CONJECTURE (R -> inf limit)."""
        result = analysis.path_b_conservative()
        labels = [c.label for c in result['claims']]
        assert 'CONJECTURE' in labels, \
            "Path B must honestly label the R->inf limit as CONJECTURE"

    def test_gap_data_all_positive(self, analysis):
        """Every entry in gap_data has gap_positive = True."""
        result = analysis.path_b_conservative()
        for d in result['gap_data']:
            assert d['gap_positive'], \
                f"Gap should be positive at R={d['R_fm']}"

    def test_su3_all_positive(self, analysis_su3):
        """SU(3) Path B also has all gaps positive."""
        result = analysis_su3.path_b_conservative()
        assert result['all_gaps_positive'] is True


# ======================================================================
# 8. Confinement argument
# ======================================================================

class TestConfinement:
    """Confinement on S^3 at T=0 implies mass gap."""

    def test_confined_at_T0(self, analysis):
        """At T=0, theory is always confined."""
        result = analysis.confinement_argument(R=2.2)
        assert result['confined'] is True

    def test_polyakov_zero(self, analysis):
        """Polyakov loop expectation value is zero (confined)."""
        result = analysis.confinement_argument(R=2.2)
        assert result['polyakov_loop'] == 0.0

    def test_center_symmetry(self, analysis):
        """Center symmetry is Z_N."""
        result = analysis.confinement_argument(R=2.2)
        assert result['center_symmetry'] == 'Z_2'

    def test_center_symmetry_su3(self, analysis_su3):
        """SU(3) has Z_3 center symmetry."""
        result = analysis_su3.confinement_argument(R=2.2)
        assert result['center_symmetry'] == 'Z_3'

    def test_gap_from_confinement_positive(self, analysis):
        """Confinement implies positive mass gap."""
        result = analysis.confinement_argument(R=2.2)
        assert result['gap_from_confinement_MeV'] > 0

    def test_deconfinement_temp_positive(self, analysis):
        """Deconfinement temperature is positive."""
        result = analysis.confinement_argument(R=2.2)
        assert result['deconfinement_temp_MeV'] > 0

    def test_string_tension_positive(self, analysis):
        """String tension is positive (area law)."""
        result = analysis.confinement_argument(R=2.2)
        assert result['string_tension_MeV2'] > 0

    def test_status_is_proposition(self, analysis):
        """Confinement claim should be labeled PROPOSITION."""
        result = analysis.confinement_argument(R=2.2)
        assert result['status'].label == 'PROPOSITION'

    def test_invalid_R(self, analysis):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            analysis.confinement_argument(R=-1.0)


# ======================================================================
# 9. Honest assessment label correctness
# ======================================================================

class TestHonestAssessment:
    """Every claim must be labeled honestly and correctly."""

    def test_has_proven_claims(self, analysis):
        """Assessment includes proven claims."""
        result = analysis.honest_assessment()
        assert len(result['proven']) > 0

    def test_has_supported_claims(self, analysis):
        """Assessment includes strongly supported claims."""
        result = analysis.honest_assessment()
        assert len(result['strongly_supported']) > 0

    def test_has_not_proven_claims(self, analysis):
        """Assessment includes not-proven claims (honesty!)."""
        result = analysis.honest_assessment()
        assert len(result['not_proven']) > 0

    def test_proven_are_theorems(self, analysis):
        """Proven claims are labeled THEOREM."""
        result = analysis.honest_assessment()
        for claim in result['proven']:
            assert claim.label == 'THEOREM', \
                f"Proven claim should be THEOREM, got {claim.label}"

    def test_supported_are_propositions(self, analysis):
        """Strongly supported claims are labeled PROPOSITION."""
        result = analysis.honest_assessment()
        for claim in result['strongly_supported']:
            assert claim.label == 'PROPOSITION', \
                f"Supported claim should be PROPOSITION, got {claim.label}"

    def test_not_proven_are_conjectures(self, analysis):
        """Not proven claims are labeled CONJECTURE."""
        result = analysis.honest_assessment()
        for claim in result['not_proven']:
            assert claim.label == 'CONJECTURE', \
                f"Not proven claim should be CONJECTURE, got {claim.label}"

    def test_r_infinity_is_conjecture(self, analysis):
        """The R -> infinity limit is honestly labeled CONJECTURE."""
        result = analysis.honest_assessment()
        conjecture_statements = [c.statement for c in result['not_proven']]
        # Should mention R -> infinity or equivalent
        has_r_limit = any('R' in s and ('infinity' in s or 'limit' in s or 'inf' in s)
                         for s in conjecture_statements)
        assert has_r_limit, \
            "R -> infinity limit should be in the not-proven (CONJECTURE) list"

    def test_conclusion_mentions_open(self, analysis):
        """Conclusion honestly states the R->inf limit is OPEN."""
        result = analysis.honest_assessment()
        conclusion = result['conclusion']
        assert 'OPEN' in conclusion or 'open' in conclusion, \
            "Conclusion should mention that R->inf is OPEN"

    def test_both_paths_have_verdicts(self, analysis):
        """Both Path A and Path B have verdicts."""
        result = analysis.honest_assessment()
        assert 'path_a_verdict' in result
        assert 'path_b_verdict' in result
        assert len(result['path_a_verdict']) > 0
        assert len(result['path_b_verdict']) > 0

    def test_all_claims_have_evidence(self, analysis):
        """Every claim has non-empty evidence."""
        result = analysis.honest_assessment()
        all_claims = (result['proven'] + result['strongly_supported'] +
                      result['not_proven'])
        for claim in all_claims:
            assert len(claim.evidence) > 0, \
                f"Claim '{claim.statement}' has no evidence"

    def test_all_claims_have_caveats(self, analysis):
        """Every claim has non-empty caveats."""
        result = analysis.honest_assessment()
        all_claims = (result['proven'] + result['strongly_supported'] +
                      result['not_proven'])
        for claim in all_claims:
            assert len(claim.caveats) > 0, \
                f"Claim '{claim.statement}' has no caveats"


# ======================================================================
# 10. Gap vs radius table
# ======================================================================

class TestGapTable:
    """Gap vs R table: gap monotonically approaches Lambda_QCD from above."""

    def test_table_not_empty(self, analysis):
        """Table has entries."""
        table = analysis.gap_vs_radius_table()
        assert len(table) > 0

    def test_all_gaps_positive(self, analysis):
        """Every entry in the table has positive gap."""
        table = analysis.gap_vs_radius_table()
        for row in table:
            assert row['total_gap_MeV'] > 0, \
                f"Gap should be positive at R={row['R_fm']}"

    def test_total_gap_bounded_below(self, analysis):
        """Total gap >= Lambda_QCD for all R."""
        table = analysis.gap_vs_radius_table()
        for row in table:
            assert row['total_gap_MeV'] >= analysis.Lambda_QCD - 1e-10, \
                f"Gap at R={row['R_fm']} = {row['total_gap_MeV']} " \
                f"should be >= {analysis.Lambda_QCD}"

    def test_geometric_gap_decreases(self, analysis):
        """Geometric gap strictly decreases with R."""
        table = analysis.gap_vs_radius_table()
        for i in range(len(table) - 1):
            assert table[i]['geometric_gap_MeV'] > table[i + 1]['geometric_gap_MeV'], \
                f"Geometric gap should decrease: R={table[i]['R_fm']} -> R={table[i+1]['R_fm']}"

    def test_total_gap_monotone_decrease_at_large_R(self, analysis):
        """
        At large R (R >> R*), total gap is constant at Lambda_QCD.
        The total gap should be non-increasing for R > R*.
        """
        table = analysis.gap_vs_radius_table()
        R_star = analysis.crossover_radius()['R_star_fm']

        large_R_entries = [row for row in table if row['R_fm'] > 3 * R_star]
        for i in range(len(large_R_entries) - 1):
            # Total gap should be approximately constant (Lambda_QCD)
            gap_i = large_R_entries[i]['total_gap_MeV']
            gap_next = large_R_entries[i + 1]['total_gap_MeV']
            # Should be non-increasing (both ~ Lambda_QCD)
            assert gap_i >= gap_next - 1e-10, \
                f"Total gap should be non-increasing at large R"

    def test_approaches_lambda_qcd(self, analysis):
        """At large R, total gap converges to Lambda_QCD."""
        table = analysis.gap_vs_radius_table()
        last_row = table[-1]  # largest R
        assert abs(last_row['total_gap_MeV'] - analysis.Lambda_QCD) < 1.0, \
            f"At R={last_row['R_fm']}, gap should approach Lambda_QCD"

    def test_smooth_gap_always_positive(self, analysis):
        """Smooth gap is always positive."""
        table = analysis.gap_vs_radius_table()
        for row in table:
            assert row['smooth_gap_MeV'] > 0

    def test_custom_R_values(self, analysis):
        """Custom R values work correctly."""
        custom = [1.0, 2.0, 3.0]
        table = analysis.gap_vs_radius_table(R_values=custom)
        assert len(table) == 3
        assert table[0]['R_fm'] == 1.0
        assert table[1]['R_fm'] == 2.0
        assert table[2]['R_fm'] == 3.0


# ======================================================================
# 11. Dimensional transmutation
# ======================================================================

class TestDimensionalTransmutation:
    """Dimensional transmutation on S^3: Lambda_QCD is R-independent."""

    def test_ir_scale(self, analysis):
        """IR scale = hbar_c / R."""
        R = 1.0
        result = analysis.dimensional_transmutation(R)
        assert abs(result['IR_scale_MeV'] - HBAR_C_MEV_FM / R) < 1e-10

    def test_geometry_dominates_small_R(self, analysis):
        """At small R, IR scale >> Lambda_QCD."""
        result = analysis.dimensional_transmutation(0.1)
        assert result['geometry_dominates'] is True

    def test_dynamics_dominates_large_R(self, analysis):
        """At large R, IR scale << Lambda_QCD."""
        result = analysis.dimensional_transmutation(100.0)
        assert result['geometry_dominates'] is False

    def test_lambda_consistency_perturbative(self, analysis):
        """In perturbative regime, Lambda_QCD is recovered from running coupling."""
        result = analysis.dimensional_transmutation(0.1)
        if result['lambda_check_MeV'] is not None:
            error = result['lambda_consistency_error']
            assert error < 0.01, \
                f"Lambda consistency error should be < 1%, got {error*100:.2f}%"

    def test_beta_function_sign(self, analysis):
        """Beta function is negative (asymptotic freedom)."""
        result = analysis.dimensional_transmutation(1.0)
        assert result['beta_function_sign'] == 'negative'

    def test_b0_value(self, analysis):
        """b_0 is correct for the gauge group."""
        result = analysis.dimensional_transmutation(1.0)
        expected_b0 = 11.0 * analysis.N / 3.0
        assert abs(result['b0'] - expected_b0) < 1e-14

    def test_status_label(self, analysis):
        """Status is labeled THEOREM (RG invariance is standard)."""
        result = analysis.dimensional_transmutation(0.1)
        assert result['status'].label == 'THEOREM'


# ======================================================================
# 12. Edge cases and limits
# ======================================================================

class TestEdgeCases:
    """Edge cases, limits, and consistency checks."""

    def test_very_small_R(self, analysis):
        """At R = 0.001 fm (far UV), everything should work."""
        gap = analysis.total_gap(0.001)
        assert gap['gap_positive']
        assert gap['geometric_gap_MeV'] > 1e5  # very large

    def test_very_large_R(self, analysis):
        """At R = 10^8 fm, gap should be Lambda_QCD."""
        gap = analysis.total_gap(1e8)
        assert gap['gap_positive']
        assert abs(gap['total_gap_MeV'] - analysis.Lambda_QCD) < 1e-6

    def test_different_lambda_qcd(self):
        """Analysis works with different Lambda_QCD values."""
        for lam in [100.0, 200.0, 300.0, 500.0]:
            a = RLimitAnalysis(N=2, Lambda_QCD=lam)
            gap = a.total_gap(10.0)
            assert gap['dynamical_gap_MeV'] == lam

    def test_different_N(self):
        """Analysis works for various SU(N)."""
        for N in [2, 3, 4, 5]:
            a = RLimitAnalysis(N=N, Lambda_QCD=200.0)
            gap = a.total_gap(2.2)
            assert gap['gap_positive']
            # b0 should be 11*N/3
            assert abs(a.b0 - 11.0 * N / 3.0) < 1e-14

    def test_full_analysis_runs(self, analysis):
        """Full analysis completes without error."""
        result = analysis.full_analysis()
        assert 'crossover' in result
        assert 'path_a' in result
        assert 'path_b' in result
        assert 'confinement' in result
        assert 'transmutation' in result
        assert 'gap_table' in result
        assert 'assessment' in result

    def test_full_analysis_all_gaps_positive(self, analysis):
        """Every gap in the full analysis is positive."""
        result = analysis.full_analysis()
        for row in result['gap_table']:
            assert row['total_gap_MeV'] > 0

    def test_claim_status_repr(self):
        """ClaimStatus has a sensible string representation."""
        claim = ClaimStatus(
            label='THEOREM',
            statement='Test statement',
            evidence='Test evidence',
            caveats='Test caveats',
        )
        s = repr(claim)
        assert 'THEOREM' in s
        assert 'Test statement' in s

    def test_constants(self):
        """Physical constants are correct."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-4
        assert abs(LAMBDA_QCD_DEFAULT - 200.0) < 1e-10
        assert abs(SQRT5 - np.sqrt(5.0)) < 1e-14

    def test_crossover_at_physical_radius(self, analysis):
        """
        Key physical result: the crossover radius is near the compact topology
        physical radius. With the corrected gap, R* ~ 1.97 fm vs R_phys = 2.2 fm.
        """
        R_star = analysis.crossover_radius()['R_star_fm']
        R_phys = 2.2  # fm
        # Agreement within 15% (R* = 1.97 vs 2.2)
        assert abs(R_star - R_phys) / R_phys < 0.15, \
            f"Crossover R*={R_star:.3f} should be near R_phys={R_phys} within 15%"

    def test_gap_vs_radius_monotonicity(self, analysis):
        """
        The total gap starts large (small R), decreases, and plateaus at Lambda_QCD.
        It should be monotonically non-increasing.
        """
        R_values = np.logspace(-1, 3, 50)
        gaps = [analysis.total_gap(R)['total_gap_MeV'] for R in R_values]

        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 1e-10, \
                f"Total gap should be non-increasing: " \
                f"gap({R_values[i]:.3f}) = {gaps[i]:.2f} " \
                f"vs gap({R_values[i+1]:.3f}) = {gaps[i+1]:.2f}"
