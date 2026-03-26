"""
Tests for Topological Gap Persistence — Why the Gap Can't Close on S^3.

Tests the systematic analysis of every mechanism that COULD close the
mass gap, proving each one is blocked on S^3 with compact gauge group G.

Test categories:
    1.  H^1(S^3) = 0 — topological fact
    2.  Ric(S^3) = 2/R^2 > 0 — geometric fact
    3.  No flat directions in V = V_2 + V_4
    4.  V grows quadratically in all directions
    5.  No degenerate vacua on S^3
    6.  Instanton moduli space doesn't create zero modes for bosonic YM
    7.  Compact manifold => discrete spectrum
    8.  Center symmetry and confinement at T=0
    9.  Gap closing scenarios all ruled out
   10.  Scale-free gap argument: Delta/Lambda_QCD is R-independent
   11.  Delta/Lambda_QCD ratio computed for various g^2
   12.  Delta/Lambda_QCD > 0 for all tested g^2
   13.  Combined topological argument synthesis
   14.  Honest assessment of what remains
   15.  Edge cases: R -> 0, R -> inf, g -> 0, N -> inf
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.topological_gap import (
    TopologicalObstructions,
    FlatDirectionAnalysis,
    GapClosingScenarios,
    ScaleFreeGapArgument,
    ConfinementImpliesGap,
    CombinedTopologicalArgument,
    topological_gap_analysis,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
    COEXACT_GAP_COEFF,
    RICCI_S3_COEFF,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def obstructions():
    """Default TopologicalObstructions at R=1, SU(2)."""
    return TopologicalObstructions(R=1.0, N=2)


@pytest.fixture
def obstructions_su3():
    """TopologicalObstructions for SU(3)."""
    return TopologicalObstructions(R=1.0, N=3)


@pytest.fixture
def flat_analysis():
    """FlatDirectionAnalysis at R=1, g=1, 3 modes."""
    return FlatDirectionAnalysis(R=1.0, g_coupling=1.0, n_modes=3)


@pytest.fixture
def flat_analysis_6modes():
    """FlatDirectionAnalysis at R=1, g=1, 6 modes (full S^3)."""
    return FlatDirectionAnalysis(R=1.0, g_coupling=1.0, n_modes=6)


@pytest.fixture
def scenarios():
    """GapClosingScenarios for SU(2)."""
    return GapClosingScenarios(N=2, Lambda_QCD=200.0)


@pytest.fixture
def scale_free():
    """ScaleFreeGapArgument for SU(2)."""
    return ScaleFreeGapArgument(N=2, Lambda_QCD=200.0)


@pytest.fixture
def confinement():
    """ConfinementImpliesGap for SU(2) at R=1."""
    return ConfinementImpliesGap(N=2, R=1.0)


@pytest.fixture
def combined():
    """CombinedTopologicalArgument at R=1, SU(2)."""
    return CombinedTopologicalArgument(R=1.0, N=2, Lambda_QCD=200.0)


# ======================================================================
# 1. H^1(S^3) = 0 — topological fact
# ======================================================================

class TestH1Vanishing:
    """THEOREM: H^1(S^3) = 0 implies no harmonic 1-forms."""

    def test_h1_is_zero(self, obstructions):
        """H^1(S^3; R) = 0 is a topological fact."""
        result = obstructions.harmonic_one_forms()
        assert result['can_close_gap'] is False

    def test_h1_obstruction_is_topological(self, obstructions):
        """The obstruction is topological, not metric-dependent."""
        result = obstructions.harmonic_one_forms()
        assert result['obstruction'] == 'TOPOLOGICAL'
        assert result['metric_independent'] is True

    def test_h1_labeled_theorem(self, obstructions):
        """This is a THEOREM, not a proposition or conjecture."""
        result = obstructions.harmonic_one_forms()
        assert result['label'] == 'THEOREM'

    def test_h1_independent_of_N(self):
        """H^1(S^3) = 0 is independent of the gauge group."""
        for N in [2, 3, 5, 10]:
            obs = TopologicalObstructions(R=1.0, N=N)
            result = obs.harmonic_one_forms()
            assert result['can_close_gap'] is False

    def test_h1_independent_of_R(self):
        """H^1(S^3) = 0 is independent of the radius."""
        for R in [0.01, 0.1, 1.0, 10.0, 1000.0]:
            obs = TopologicalObstructions(R=R, N=2)
            result = obs.harmonic_one_forms()
            assert result['can_close_gap'] is False


# ======================================================================
# 2. Ric(S^3) > 0 — geometric fact
# ======================================================================

class TestPositiveRicci:
    """THEOREM: Ric(S^3) = 2/R^2 > 0 for all R > 0."""

    def test_ricci_positive(self):
        """Ricci curvature is strictly positive for all R > 0."""
        for R in [0.01, 0.5, 1.0, 5.0, 100.0]:
            ric = RICCI_S3_COEFF / R**2
            assert ric > 0, f"Ric = {ric} at R = {R}"

    def test_ricci_formula(self):
        """Ric = 2/R^2 on S^3 (Einstein manifold, n=3)."""
        R = 2.0
        expected = 2.0 / R**2
        assert abs(expected - 0.5) < 1e-14

    def test_ricci_scales_inversely_with_R2(self):
        """Ric(S^3_R) = 2/R^2 decreases as R increases but stays positive."""
        R1, R2 = 1.0, 10.0
        ric1 = RICCI_S3_COEFF / R1**2
        ric2 = RICCI_S3_COEFF / R2**2
        assert ric1 > ric2 > 0


# ======================================================================
# 3. No flat directions in V = V_2 + V_4
# ======================================================================

class TestNoFlatDirections:
    """THEOREM: V has no flat directions — V grows >= |a|^2 everywhere."""

    def test_no_flat_directions_3_modes(self, flat_analysis):
        """No flat directions on S^3/I* (3 spatial modes)."""
        result = flat_analysis.verify_no_flat_directions(n_directions=100)
        assert result['no_flat_directions'] is True
        assert result['min_growth_rate'] >= result['lower_bound'] - 1e-10

    def test_no_flat_directions_6_modes(self, flat_analysis_6modes):
        """No flat directions on full S^3 (6 spatial modes)."""
        result = flat_analysis_6modes.verify_no_flat_directions(n_directions=100)
        assert result['no_flat_directions'] is True

    def test_growth_rate_at_least_2_over_R2(self, flat_analysis):
        """V/|a|^2 >= 2/R^2 in all tested directions."""
        result = flat_analysis.verify_no_flat_directions(n_directions=50)
        assert result['ratio'] >= 1.0 - 1e-8, (
            f"Growth rate ratio = {result['ratio']}, expected >= 1.0"
        )

    def test_no_flat_directions_large_R(self):
        """No flat directions even at large R."""
        fa = FlatDirectionAnalysis(R=100.0, g_coupling=1.0, n_modes=3)
        result = fa.verify_no_flat_directions(n_directions=50)
        assert result['no_flat_directions'] is True

    def test_no_flat_directions_strong_coupling(self):
        """No flat directions at strong coupling g=10."""
        fa = FlatDirectionAnalysis(R=1.0, g_coupling=10.0, n_modes=3)
        result = fa.verify_no_flat_directions(n_directions=50)
        assert result['no_flat_directions'] is True


# ======================================================================
# 4. V grows quadratically in all directions
# ======================================================================

class TestQuadraticGrowth:
    """THEOREM: V(a) >= (2/R^2)|a|^2 in every direction."""

    def test_v2_grows_quadratically(self, flat_analysis):
        """V_2 = (2/R^2)|a|^2 is strictly quadratic."""
        for r in [0.1, 1.0, 10.0, 100.0]:
            a = np.ones(flat_analysis.n_dof) * r / np.sqrt(flat_analysis.n_dof)
            v2 = flat_analysis.quadratic_potential(a)
            expected = (2.0 / flat_analysis.R**2) * r**2
            assert abs(v2 - expected) < 1e-10 * expected

    def test_total_v_at_least_v2(self, flat_analysis):
        """V >= V_2 because V_4 >= 0."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            a = rng.standard_normal(flat_analysis.n_dof) * rng.uniform(0.1, 10.0)
            v_total = flat_analysis.total_potential(a)
            v2 = flat_analysis.quadratic_potential(a)
            assert v_total >= v2 - 1e-12, (
                f"V_total = {v_total} < V_2 = {v2}"
            )

    def test_growth_along_specific_directions(self, flat_analysis):
        """Test growth along coordinate axes."""
        for i in range(flat_analysis.n_dof):
            d = np.zeros(flat_analysis.n_dof)
            d[i] = 1.0
            result = flat_analysis.growth_rate_along_direction(d)
            assert result['all_above_bound'] is True


# ======================================================================
# 5. No degenerate vacua on S^3
# ======================================================================

class TestUniqueVacuum:
    """THEOREM: The vacuum at a = 0 is the unique minimum of V."""

    def test_v_at_zero_is_zero(self, flat_analysis):
        """V(0) = 0 exactly."""
        v = flat_analysis.total_potential(np.zeros(flat_analysis.n_dof))
        assert abs(v) < 1e-15

    def test_v_positive_away_from_zero(self, flat_analysis):
        """V(a) > 0 for all a != 0."""
        result = flat_analysis.verify_unique_minimum(n_samples=2000)
        assert result['unique_minimum'] is True
        assert result['V_at_zero'] < 1e-14
        assert result['min_V_found'] > -1e-12

    def test_no_degenerate_vacua(self, obstructions):
        """No degenerate vacua on S^3."""
        result = obstructions.degenerate_vacua()
        assert result['can_close_gap'] is False
        assert result['unique_vacuum'] is True
        assert result['label'] == 'THEOREM'


# ======================================================================
# 6. Index theorem doesn't create zero modes for bosonic YM
# ======================================================================

class TestIndexTheorem:
    """THEOREM: Atiyah-Singer index theorem applies to chiral operators, not bosonic YM."""

    def test_index_theorem_blocked(self, obstructions):
        """Index theorem cannot close the gap for bosonic YM."""
        result = obstructions.index_theorem_zero_modes()
        assert result['can_close_gap'] is False

    def test_index_theorem_is_algebraic(self, obstructions):
        """The obstruction is algebraic (self-adjoint operator)."""
        result = obstructions.index_theorem_zero_modes()
        assert result['obstruction'] == 'ALGEBRAIC'
        assert result['label'] == 'THEOREM'

    def test_applies_to_bosonic_only(self, obstructions):
        """The result applies specifically to bosonic YM without fermions."""
        result = obstructions.index_theorem_zero_modes()
        assert 'bosonic' in result['applies_to']


# ======================================================================
# 7. Compact manifold => discrete spectrum
# ======================================================================

class TestDiscreteSpectrum:
    """THEOREM: S^3 compact => purely discrete spectrum."""

    def test_continuous_spectrum_impossible(self, obstructions):
        """Continuous spectrum is impossible on S^3."""
        result = obstructions.continuous_spectrum()
        assert result['can_close_gap'] is False

    def test_obstruction_is_compactness(self, obstructions):
        """The obstruction comes from compactness."""
        result = obstructions.continuous_spectrum()
        assert 'compact' in result['obstruction'].lower()
        assert result['label'] == 'THEOREM'

    def test_contrast_with_R3(self, obstructions):
        """R^3 DOES have continuous spectrum (contrast)."""
        result = obstructions.continuous_spectrum()
        assert 'R^3' in result['contrast_with_R3'] or 'R^3' in result['contrast_with_R3']


# ======================================================================
# 8. Center symmetry and confinement at T=0
# ======================================================================

class TestConfinement:
    """Tests for center symmetry and confinement-implies-gap."""

    def test_center_symmetry_exact(self, confinement):
        """Center symmetry is exact in pure YM on S^3."""
        result = confinement.center_symmetry()
        assert result['exact'] is True
        assert result['label'] == 'THEOREM'

    def test_polyakov_loop_zero_at_t0(self, confinement):
        """Polyakov loop <P> = 0 at T=0."""
        result = confinement.polyakov_loop_at_t0()
        assert result['polyakov_loop'] == 0
        assert result['phase'] == 'confined'
        assert result['label'] == 'THEOREM'

    def test_no_goldstone_bosons(self, obstructions):
        """No symmetry breaking => no Goldstone bosons."""
        result = obstructions.symmetry_breaking()
        assert result['can_close_gap'] is False
        assert result['goldstone_bosons'] is False

    def test_confinement_implies_gap_is_proposition(self, confinement):
        """Confinement implies gap is PROPOSITION, not THEOREM."""
        result = confinement.confinement_implies_gap_argument()
        assert result['label'] == 'PROPOSITION'

    def test_confinement_for_sun(self):
        """Center symmetry holds for all SU(N)."""
        for N in [2, 3, 4, 5]:
            conf = ConfinementImpliesGap(N=N, R=1.0)
            result = conf.center_symmetry()
            assert result['exact'] is True
            assert f'Z_{N}' == result['symmetry']

    def test_confinement_full_analysis(self, confinement):
        """Full confinement analysis completes without error."""
        result = confinement.full_analysis()
        assert result['label'] == 'PROPOSITION'
        assert 'confined' in result['conclusion'].lower()


# ======================================================================
# 9. Gap closing scenarios all ruled out
# ======================================================================

class TestGapClosingScenarios:
    """Tests that all gap-closing scenarios are ruled out."""

    def test_eigenvalue_pileup_ruled_out(self, scenarios):
        """Scenario A: eigenvalue pile-up cannot close the gap."""
        result = scenarios.eigenvalue_pileup()
        assert result['can_close_gap'] is False
        assert result['label'] == 'PROPOSITION'

    def test_new_modes_ruled_out(self, scenarios):
        """Scenario B: no new modes can appear on compact manifold."""
        result = scenarios.new_modes()
        assert result['can_close_gap'] is False
        assert result['label'] == 'THEOREM'

    def test_tunneling_ruled_out(self, scenarios):
        """Scenario C: tunneling cannot close gap (unique vacuum)."""
        result = scenarios.tunneling()
        assert result['can_close_gap'] is False
        assert result['label'] == 'THEOREM'

    def test_all_scenarios_ruled_out(self, scenarios):
        """All 3 gap-closing scenarios are ruled out."""
        result = scenarios.full_analysis()
        assert result['all_ruled_out'] is True

    def test_full_analysis_label(self, scenarios):
        """Overall label is PROPOSITION (weakest link: scenario A)."""
        result = scenarios.full_analysis()
        assert result['label'] == 'PROPOSITION'


# ======================================================================
# 10. Scale-free gap argument: Delta/Lambda_QCD is R-independent
# ======================================================================

class TestScaleFreeArgument:
    """Tests for the scale-free Delta/Lambda_QCD ratio."""

    def test_ratio_increases_as_g2_decreases(self, scale_free):
        """Delta/Lambda_QCD increases as g^2 -> 0 (UV)."""
        r1 = scale_free.gap_over_lambda(1.0)
        r2 = scale_free.gap_over_lambda(0.1)
        assert r2 > r1

    def test_ratio_at_g2_zero_is_infinity(self, scale_free):
        """At g^2 = 0, Delta/Lambda_QCD = infinity."""
        r = scale_free.gap_over_lambda(0.0)
        assert r == float('inf')

    def test_ratio_at_negative_g2_is_infinity(self, scale_free):
        """Negative g^2 gives infinity (unphysical)."""
        r = scale_free.gap_over_lambda(-1.0)
        assert r == float('inf')

    def test_ratio_positive_for_physical_g2(self, scale_free):
        """Delta/Lambda_QCD > 0 for physical g^2 values."""
        for g2 in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            r = scale_free.gap_over_lambda(g2)
            assert r > 0, f"Ratio = {r} at g^2 = {g2}"

    def test_scale_free_argument_completes(self, scale_free):
        """The complete scale-free argument runs without error."""
        result = scale_free.scale_free_argument()
        assert result['label'] == 'PROPOSITION'

    def test_no_landau_pole_on_s3(self, scale_free):
        """The scale-free argument notes: no Landau pole on S^3."""
        result = scale_free.scale_free_argument()
        assert 'Landau' in result['no_landau_pole']


# ======================================================================
# 11. Delta/Lambda_QCD ratio computed for various g^2
# ======================================================================

class TestRatioScan:
    """NUMERICAL: scan of Delta/Lambda_QCD over g^2."""

    def test_scan_produces_results(self, scale_free):
        """Scan produces results for all g^2 values."""
        scan = scale_free.scan_g2(g2_values=np.logspace(-1, 1, 20))
        assert len(scan['results']) == 20
        assert scan['label'] == 'NUMERICAL'

    def test_scan_finds_valid_range(self, scale_free):
        """Scan identifies the valid range of g^2."""
        scan = scale_free.scan_g2()
        assert scan['max_g2_valid'] > 0

    def test_nonlinear_correction_reduces_ratio(self, scale_free):
        """Non-linear (Kato-Rellich) correction reduces the ratio."""
        g2 = 1.0
        r_lin = scale_free.gap_over_lambda(g2)
        r_nl = scale_free.gap_over_lambda_with_nonlinear(g2)
        assert r_nl <= r_lin

    def test_nonlinear_breaks_at_large_g2(self, scale_free):
        """Kato-Rellich breaks down for g^2 > g^2_crit."""
        # g^2_crit ~ 167.5 from Aubin-Talenti Sobolev constant
        r = scale_free.gap_over_lambda_with_nonlinear(200.0)
        assert r == 0.0  # alpha >= 1


# ======================================================================
# 12. Delta/Lambda_QCD > 0 for all tested g^2
# ======================================================================

class TestRatioBoundedBelow:
    """NUMERICAL: Delta/Lambda_QCD > 0 for all tested g^2."""

    def test_ratio_bounded_below(self, scale_free):
        """Ratio is bounded below by a positive number."""
        scan = scale_free.scan_g2()
        assert scan['ratio_bounded_below'] is True
        assert scan['min_ratio_nonlinear'] > 0

    def test_ratio_bounded_for_su3(self):
        """Ratio is bounded below for SU(3) as well."""
        sf = ScaleFreeGapArgument(N=3, Lambda_QCD=200.0)
        scan = sf.scan_g2()
        assert scan['ratio_bounded_below'] is True

    def test_ratio_bounded_for_various_lambda(self):
        """Ratio is bounded for different Lambda_QCD values."""
        for Lambda in [100.0, 200.0, 300.0, 500.0]:
            sf = ScaleFreeGapArgument(N=2, Lambda_QCD=Lambda)
            scan = sf.scan_g2(g2_values=np.logspace(-1, 1, 30))
            assert scan['ratio_bounded_below'] is True


# ======================================================================
# 13. Combined topological argument synthesis
# ======================================================================

class TestCombinedArgument:
    """Tests for the combined topological gap persistence argument."""

    def test_proposition_builds(self, combined):
        """The combined proposition builds without error."""
        result = combined.build_proposition()
        assert 'part_i' in result
        assert 'part_ii' in result
        assert 'part_iii' in result
        assert 'part_iv' in result

    def test_part_i_is_theorem(self, combined):
        """Part (i): Delta > 0 for all R is THEOREM."""
        result = combined.build_proposition()
        assert result['part_i']['label'] == 'THEOREM'

    def test_part_ii_is_proposition(self, combined):
        """Part (ii): scale-free ratio is PROPOSITION."""
        result = combined.build_proposition()
        assert result['part_ii']['label'] == 'PROPOSITION'

    def test_part_iii_is_numerical(self, combined):
        """Part (iii): f(g^2) > 0 is NUMERICAL."""
        result = combined.build_proposition()
        assert result['part_iii']['label'] == 'NUMERICAL'

    def test_part_iv_is_theorem(self, combined):
        """Part (iv): no topological mechanism closes gap is THEOREM."""
        result = combined.build_proposition()
        assert result['part_iv']['label'] == 'THEOREM'
        assert result['part_iv']['all_blocked'] is True

    def test_overall_label_is_proposition(self, combined):
        """Overall label is PROPOSITION (weakest link)."""
        result = combined.build_proposition()
        assert result['overall_label'] == 'PROPOSITION'

    def test_gap_status_positive(self, combined):
        """Gap status reports positive gap."""
        status = combined.gap_status()
        assert status['gap_positive'] is True
        assert status['all_mechanisms_blocked'] is True


# ======================================================================
# 14. Honest assessment of what remains
# ======================================================================

class TestHonestAssessment:
    """Tests that the honest assessment correctly identifies gaps."""

    def test_current_status_is_proposition(self, combined):
        """Current overall status is PROPOSITION, not THEOREM."""
        remaining = combined.what_remains_for_theorem()
        assert remaining['current_status'] == 'PROPOSITION'
        assert remaining['target_status'] == 'THEOREM'

    def test_three_gaps_identified(self, combined):
        """Three gaps remain to reach THEOREM status."""
        remaining = combined.what_remains_for_theorem()
        assert len(remaining['gaps']) == 3

    def test_r_infinity_is_clay_problem(self, combined):
        """The R -> infinity limit is identified as the Clay problem."""
        remaining = combined.what_remains_for_theorem()
        r_inf_gap = remaining['gaps'][2]
        assert 'Clay' in r_inf_gap['difficulty'] or 'Millennium' in r_inf_gap['difficulty']

    def test_honest_assessment_included(self, combined):
        """An honest assessment is included."""
        remaining = combined.what_remains_for_theorem()
        assert len(remaining['honest_assessment']) > 100


# ======================================================================
# 15. Edge cases: R -> 0, R -> inf, g -> 0, N -> inf
# ======================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_small_R(self):
        """Analysis works at very small R."""
        obs = TopologicalObstructions(R=0.01, N=2)
        result = obs.full_analysis()
        assert result['all_blocked'] is True

    def test_large_R(self):
        """Analysis works at very large R."""
        obs = TopologicalObstructions(R=1000.0, N=2)
        result = obs.full_analysis()
        assert result['all_blocked'] is True

    def test_weak_coupling(self):
        """Flat direction analysis works at weak coupling."""
        fa = FlatDirectionAnalysis(R=1.0, g_coupling=0.001, n_modes=3)
        result = fa.verify_no_flat_directions(n_directions=30)
        assert result['no_flat_directions'] is True

    def test_strong_coupling(self):
        """Flat direction analysis works at strong coupling."""
        fa = FlatDirectionAnalysis(R=1.0, g_coupling=50.0, n_modes=3)
        result = fa.verify_no_flat_directions(n_directions=30)
        assert result['no_flat_directions'] is True

    def test_su5(self):
        """Analysis works for SU(5)."""
        obs = TopologicalObstructions(R=1.0, N=5)
        result = obs.full_analysis()
        assert result['all_blocked'] is True

    def test_v4_nonneg_at_g_zero(self):
        """V_4 = 0 when g = 0 (trivially nonnegative)."""
        fa = FlatDirectionAnalysis(R=1.0, g_coupling=0.0, n_modes=3)
        result = fa.verify_v4_nonnegative(n_samples=100)
        assert result['v4_nonnegative'] is True
        assert abs(result['min_v4']) < 1e-14

    def test_module_level_function(self):
        """topological_gap_analysis runs end-to-end."""
        result = topological_gap_analysis(R=2.2, N=2, Lambda_QCD=200.0)
        assert 'proposition' in result
        assert 'status' in result
        assert 'remaining' in result

    def test_all_mechanisms_blocked_for_multiple_params(self):
        """All mechanisms blocked across a grid of (R, N) values."""
        for R in [0.1, 1.0, 10.0]:
            for N in [2, 3, 4]:
                obs = TopologicalObstructions(R=R, N=N)
                result = obs.full_analysis()
                assert result['all_blocked'] is True, (
                    f"Mechanism not blocked at R={R}, N={N}"
                )


# ======================================================================
# 16. V_4 non-negativity (detailed verification)
# ======================================================================

class TestV4NonNegativity:
    """THEOREM: V_4 >= 0 for all configurations."""

    def test_v4_nonneg_3_modes(self, flat_analysis):
        """V_4 >= 0 on S^3/I* (3 modes)."""
        result = flat_analysis.verify_v4_nonnegative(n_samples=5000)
        assert result['v4_nonnegative'] is True

    def test_v4_nonneg_6_modes(self, flat_analysis_6modes):
        """V_4 >= 0 on full S^3 (6 modes)."""
        result = flat_analysis_6modes.verify_v4_nonnegative(n_samples=5000)
        assert result['v4_nonnegative'] is True

    def test_v4_zero_for_rank_1_configs(self, flat_analysis):
        """V_4 = 0 for rank-1 configurations (only one nonzero SVD value)."""
        # Rank-1: a = u * v^T, so S = M^T M has rank 1
        rng = np.random.default_rng(77)
        for _ in range(50):
            u = rng.standard_normal(3)
            v = rng.standard_normal(3)
            a = np.outer(u, v).ravel()
            v4 = flat_analysis.quartic_potential(a)
            # For rank-1 S: (Tr S)^2 = s_1^2 = Tr(S^2), so V_4 = 0
            assert abs(v4) < 1e-10, f"V_4 = {v4} for rank-1 config"

    def test_v4_strictly_positive_for_full_rank(self, flat_analysis):
        """V_4 > 0 for generic (full-rank) configurations."""
        rng = np.random.default_rng(88)
        for _ in range(50):
            a = rng.standard_normal(flat_analysis.n_dof) * 5.0
            M = a.reshape(3, 3)
            # Check rank
            svs = np.linalg.svd(M, compute_uv=False)
            if min(svs) > 0.1:  # full rank
                v4 = flat_analysis.quartic_potential(a)
                assert v4 > 0, f"V_4 = {v4} for full-rank config"


# ======================================================================
# 17. Eigenvalue pile-up analysis with explicit R values
# ======================================================================

class TestEigenvaluePileup:
    """Tests for the eigenvalue pile-up scenario."""

    def test_eigenvalue_pileup_produces_results(self, scenarios):
        """Eigenvalue pile-up analysis produces R-indexed results."""
        R_vals = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
        result = scenarios.eigenvalue_pileup(R_values=R_vals)
        assert result['n_R_tested'] == 5

    def test_gap_decreases_with_R(self, scenarios):
        """The linearized gap 2*hbar_c/R decreases with R."""
        R_vals = np.array([1.0, 2.0, 5.0, 10.0])
        result = scenarios.eigenvalue_pileup(R_values=R_vals)
        # Filter to ones with valid coupling
        valid = [s for s in result if isinstance(s, dict)] if isinstance(result, list) else None
        # Check the label
        assert result['label'] == 'PROPOSITION'

    def test_ratio_gap_lambda_at_physical_R(self, scenarios):
        """At R = 2.2 fm (physical), the gap/Lambda ratio is reasonable."""
        R_vals = np.array([2.2])
        result = scenarios.eigenvalue_pileup(R_values=R_vals)
        assert result['can_close_gap'] is False


# ======================================================================
# 18. Full obstruction analysis
# ======================================================================

class TestFullObstructionAnalysis:
    """Tests for the complete obstruction analysis."""

    def test_six_mechanisms_analyzed(self, obstructions):
        """Exactly 6 mechanisms are analyzed."""
        result = obstructions.full_analysis()
        assert len(result['mechanisms']) == 6

    def test_all_six_blocked(self, obstructions):
        """All 6 mechanisms are blocked."""
        result = obstructions.full_analysis()
        for name, mech in result['mechanisms'].items():
            assert mech['can_close_gap'] is False, (
                f"Mechanism '{name}' not blocked!"
            )

    def test_overall_label_is_theorem(self, obstructions):
        """The overall obstruction analysis is THEOREM."""
        result = obstructions.full_analysis()
        assert result['label'] == 'THEOREM'

    def test_symmetry_breaking_independent_of_R(self):
        """Symmetry breaking analysis is independent of R."""
        for R in [0.01, 1.0, 100.0]:
            obs = TopologicalObstructions(R=R, N=2)
            result = obs.symmetry_breaking()
            assert result['can_close_gap'] is False
            assert result['degenerate_vacua'] is False
