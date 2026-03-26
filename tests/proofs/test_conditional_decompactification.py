"""
Tests for the Conditional Decompactification Theorem.

Tests cover all 4 classes:
    1. ConditionalDecompactificationTheorem: hypotheses, conclusions, proof steps,
       verification at fixed R, curvature decoupling, status
    2. UniformClusteringHypothesis: clustering at fixed R, correlation length,
       uniform bound check
    3. CurvatureDecouplingLemma: metric deviation, Christoffel bounds,
       Schwinger function error
    4. BridgeStatus: what is proven, what is needed, approaches, Clay connection

Test count target: 60+

Standards:
    - The conditional theorem itself is THEOREM
    - The bridge (H1-H3 uniformly) is PROPOSITION (computer-assisted)
    - At each fixed R, H1-H3 are THEOREM
    - Curvature decoupling is THEOREM
    - All labels must be precise and honest
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.conditional_decompactification import (
    ConditionalDecompactificationTheorem,
    UniformClusteringHypothesis,
    CurvatureDecouplingLemma,
    BridgeStatus,
    R_0_FM,
    G_SQUARED_PHYS,
    ALPHA_S_PHYS,
)
from yang_mills_s3.proofs.decompactification import (
    HBAR_C,
    LAMBDA_QCD_MEV,
    R_PHYSICAL_FM,
)
from yang_mills_s3.proofs.r_limit import ClaimStatus


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def cond_thm():
    """ConditionalDecompactificationTheorem for SU(2)."""
    return ConditionalDecompactificationTheorem(N=2, Lambda_QCD=200.0)


@pytest.fixture
def cond_thm_su3():
    """ConditionalDecompactificationTheorem for SU(3)."""
    return ConditionalDecompactificationTheorem(N=3, Lambda_QCD=200.0)


@pytest.fixture
def clustering():
    """UniformClusteringHypothesis for SU(2)."""
    return UniformClusteringHypothesis(N=2, Lambda_QCD=200.0)


@pytest.fixture
def curvature():
    """CurvatureDecouplingLemma instance."""
    return CurvatureDecouplingLemma()


@pytest.fixture
def bridge():
    """BridgeStatus for SU(2)."""
    return BridgeStatus(N=2, Lambda_QCD=200.0)


@pytest.fixture
def bridge_su3():
    """BridgeStatus for SU(3)."""
    return BridgeStatus(N=3, Lambda_QCD=200.0)


# ======================================================================
# 1. ConditionalDecompactificationTheorem
# ======================================================================

class TestConditionalDecompactificationTheorem:
    """Tests for the main conditional theorem."""

    def test_hypotheses_count(self, cond_thm):
        """The theorem has exactly 3 hypotheses."""
        hyps = cond_thm.hypotheses()
        assert len(hyps) == 3

    def test_hypotheses_names(self, cond_thm):
        """The three hypotheses are H1, H2, H3."""
        hyps = cond_thm.hypotheses()
        assert 'H1_local_moment_bounds' in hyps
        assert 'H2_uniform_local_coercivity' in hyps
        assert 'H3_uniform_clustering' in hyps

    def test_hypotheses_have_statements(self, cond_thm):
        """Each hypothesis has a non-empty precise statement."""
        hyps = cond_thm.hypotheses()
        for key, hyp in hyps.items():
            assert len(hyp['statement']) > 50, f"{key} statement too short"
            assert 'name' in hyp
            assert 'physical_meaning' in hyp

    def test_hypotheses_status_at_fixed_R_is_theorem(self, cond_thm):
        """At each fixed R, every hypothesis is THEOREM."""
        hyps = cond_thm.hypotheses()
        for key, hyp in hyps.items():
            assert 'THEOREM' in hyp['status_at_fixed_R'], (
                f"{key}: status at fixed R should be THEOREM, "
                f"got {hyp['status_at_fixed_R']}"
            )

    def test_hypotheses_uniform_status_is_proposition(self, cond_thm):
        """The uniform versions of H1-H3 are PROPOSITION."""
        hyps = cond_thm.hypotheses()
        for key, hyp in hyps.items():
            assert 'PROPOSITION' in hyp['status_uniform'], (
                f"{key}: uniform status should contain PROPOSITION, "
                f"got {hyp['status_uniform']}"
            )

    def test_conclusions_count(self, cond_thm):
        """The theorem has exactly 4 conclusions."""
        conc = cond_thm.conclusions()
        assert len(conc) == 4

    def test_conclusions_names(self, cond_thm):
        """The four conclusions are C1-C4."""
        conc = cond_thm.conclusions()
        assert 'C1_local_tightness' in conc
        assert 'C2_os_positive_limit' in conc
        assert 'C3_mass_gap' in conc
        assert 'C4_clay_solution' in conc

    def test_conclusions_are_theorem_given_hypotheses(self, cond_thm):
        """Each conclusion is THEOREM given H1-H3."""
        conc = cond_thm.conclusions()
        for key, c in conc.items():
            assert 'THEOREM' in c['status'], (
                f"{key}: status should contain THEOREM, got {c['status']}"
            )

    def test_conclusions_have_proof_sketches(self, cond_thm):
        """Each conclusion has a proof sketch."""
        conc = cond_thm.conclusions()
        for key, c in conc.items():
            assert 'proof_sketch' in c
            assert len(c['proof_sketch']) > 30, (
                f"{key}: proof sketch too short"
            )

    def test_proof_steps_count(self, cond_thm):
        """The proof outline has exactly 6 steps."""
        steps = cond_thm.proof_steps()
        assert len(steps) == 6

    def test_proof_steps_sequential(self, cond_thm):
        """Proof steps are numbered 1-6 in order."""
        steps = cond_thm.proof_steps()
        for i, step in enumerate(steps, start=1):
            assert step['step'] == i, f"Step {i} has wrong number {step['step']}"

    def test_proof_steps_are_all_theorem(self, cond_thm):
        """Each proof step is THEOREM given the hypotheses."""
        steps = cond_thm.proof_steps()
        for step in steps:
            assert 'THEOREM' in step['status'], (
                f"Step {step['step']}: status should contain THEOREM, "
                f"got {step['status']}"
            )

    def test_proof_steps_have_references(self, cond_thm):
        """Each proof step has references."""
        steps = cond_thm.proof_steps()
        for step in steps:
            assert 'references' in step
            assert len(step['references']) > 0

    def test_verify_at_fixed_R_physical(self, cond_thm):
        """All hypotheses satisfied at physical R = 2.2 fm."""
        result = cond_thm.verify_at_fixed_R(R=2.2)
        assert result['all_satisfied']
        assert result['H1_local_moment_bounds']['satisfied']
        assert result['H2_uniform_local_coercivity']['satisfied']
        assert result['H3_uniform_clustering']['satisfied']

    def test_verify_at_fixed_R_small(self, cond_thm):
        """All hypotheses satisfied at small R = 0.5 fm."""
        result = cond_thm.verify_at_fixed_R(R=0.5)
        assert result['all_satisfied']

    def test_verify_at_fixed_R_large(self, cond_thm):
        """All hypotheses satisfied at large R = 50 fm."""
        result = cond_thm.verify_at_fixed_R(R=50.0)
        assert result['all_satisfied']

    def test_verify_gap_positive_at_fixed_R(self, cond_thm):
        """Mass gap is positive at the physical radius."""
        result = cond_thm.verify_at_fixed_R(R=2.2)
        assert result['H3_uniform_clustering']['clustering_rate_MeV'] > 0

    def test_verify_fp_coercivity_positive(self, cond_thm):
        """FP coercivity constant is positive at physical R."""
        result = cond_thm.verify_at_fixed_R(R=2.2)
        assert result['H2_uniform_local_coercivity']['coercivity_constant'] > 0

    def test_verify_at_fixed_R_invalid_raises(self, cond_thm):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            cond_thm.verify_at_fixed_R(R=-1.0)

    def test_curvature_decoupling_bound_small_L(self, cond_thm):
        """Curvature decoupling error is small for L << R."""
        result = cond_thm.curvature_decoupling_bound(R=50.0, L=0.5)
        assert result['is_small']
        assert result['schwinger_error_bound'] < 0.01

    def test_curvature_decoupling_bound_scaling(self, cond_thm):
        """Error scales as O(L^2/R^2)."""
        err1 = cond_thm.curvature_decoupling_bound(R=10.0, L=1.0)
        err2 = cond_thm.curvature_decoupling_bound(R=20.0, L=1.0)
        # Doubling R should reduce error by ~4x
        ratio = err1['schwinger_error_bound'] / err2['schwinger_error_bound']
        assert 3.0 < ratio < 5.0, f"Expected ~4x reduction, got {ratio:.2f}x"

    def test_status_is_theorem(self, cond_thm):
        """The conditional theorem status is THEOREM."""
        status = cond_thm.status()
        assert status.label == 'THEOREM'

    def test_status_mentions_conditional(self, cond_thm):
        """The status clearly says this is a CONDITIONAL theorem."""
        status = cond_thm.status()
        assert 'CONDITIONAL' in status.statement.upper() or 'conditional' in status.statement.lower() or 'If' in status.statement

    def test_status_mentions_clay(self, cond_thm):
        """The status mentions the Clay Millennium Problem."""
        status = cond_thm.status()
        assert 'Clay' in status.statement

    def test_su3_hypotheses(self, cond_thm_su3):
        """SU(3) version has same number of hypotheses."""
        hyps = cond_thm_su3.hypotheses()
        assert len(hyps) == 3

    def test_su3_verify_at_fixed_R(self, cond_thm_su3):
        """SU(3) version also passes at fixed R."""
        result = cond_thm_su3.verify_at_fixed_R(R=2.2)
        assert result['all_satisfied']

    def test_multiple_R_values_all_pass(self, cond_thm):
        """Hypotheses hold at every tested R value."""
        R_values = [0.3, 0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 50.0]
        for R in R_values:
            result = cond_thm.verify_at_fixed_R(R)
            assert result['all_satisfied'], f"Failed at R = {R} fm"


# ======================================================================
# 2. UniformClusteringHypothesis
# ======================================================================

class TestUniformClusteringHypothesis:
    """Tests for the clustering hypothesis analysis."""

    def test_clustering_at_physical_R(self, clustering):
        """Clustering holds at the physical radius R = 2.2 fm."""
        result = clustering.check_clustering_numerically(R=2.2)
        assert result['clustering_holds']
        assert result['gap_MeV'] > 0
        assert result['status'] == 'THEOREM (at fixed R)'

    def test_clustering_at_various_R(self, clustering):
        """Clustering holds at all tested R values."""
        for R in [0.2, 0.5, 1.0, 2.2, 5.0, 20.0, 100.0]:
            result = clustering.check_clustering_numerically(R=R)
            assert result['clustering_holds'], f"Clustering failed at R={R}"

    def test_clustering_invalid_R_raises(self, clustering):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            clustering.check_clustering_numerically(R=-1.0)

    def test_correlation_length_physical(self, clustering):
        """Correlation length at physical R is ~ 1 fm."""
        result = clustering.estimate_correlation_length(R=2.2)
        xi = result['xi_fm']
        # xi ~ hbar_c / Lambda_QCD ~ 197.3/200 ~ 0.99 fm
        assert 0.1 < xi < 5.0, f"Unexpected xi = {xi} fm"

    def test_correlation_length_small_R(self, clustering):
        """At small R, correlation length is ~ R/2 (kinematic regime)."""
        R_small = 0.2
        result = clustering.estimate_correlation_length(R=R_small)
        xi = result['xi_fm']
        # In kinematic regime: gap ~ 2*hbar_c/R, so xi ~ R/2
        assert xi < R_small, f"xi = {xi} should be < R = {R_small}"

    def test_correlation_length_large_R(self, clustering):
        """At large R, correlation length approaches hbar_c/Lambda_QCD."""
        R_large = 100.0
        result = clustering.estimate_correlation_length(R=R_large)
        xi = result['xi_fm']
        xi_dynamic = HBAR_C / 200.0  # ~ 0.99 fm
        # Should be close to the dynamic limit
        assert abs(xi - xi_dynamic) / xi_dynamic < 0.1, (
            f"xi = {xi} should approach {xi_dynamic}"
        )

    def test_clustering_is_local(self, clustering):
        """Clustering is local: xi << R for large R."""
        result = clustering.estimate_correlation_length(R=10.0)
        assert result['clustering_is_local'], "xi/R should be < 0.5"

    def test_uniform_bound_check(self, clustering):
        """Uniform bound check over a range of R values."""
        R_values = np.logspace(np.log10(0.5), np.log10(50.0), 30)
        result = clustering.uniform_bound_check(R_values)
        assert result['all_positive']
        assert result['min_gap_MeV'] > 0

    def test_uniform_bound_dynamic_regime(self, clustering):
        """In the dynamic regime, the gap is roughly R-independent."""
        R_values = np.logspace(np.log10(3.0), np.log10(100.0), 20)
        result = clustering.uniform_bound_check(R_values)
        # In the dynamic regime, gap ~ Lambda_QCD (constant)
        assert result['variation_in_dynamic_regime'] < 0.2, (
            f"Gap varies by {result['variation_in_dynamic_regime']:.2%} "
            "in the dynamic regime, expected < 20%"
        )

    def test_uniform_bound_empty_raises(self, clustering):
        """Empty R_values raises ValueError."""
        with pytest.raises(ValueError):
            clustering.uniform_bound_check(np.array([]))

    def test_status_is_proposition(self, clustering):
        """Uniform clustering status is PROPOSITION."""
        status = clustering.status()
        assert status.label == 'PROPOSITION'


# ======================================================================
# 3. CurvatureDecouplingLemma
# ======================================================================

class TestCurvatureDecouplingLemma:
    """Tests for the curvature decoupling lemma."""

    def test_metric_deviation_small_L_over_R(self, curvature):
        """Metric deviation is small when L << R."""
        result = curvature.metric_deviation(R=10.0, L=0.5)
        assert result['is_small']
        assert result['max_relative_error'] < 0.01

    def test_metric_deviation_scales_as_L2_over_R2(self, curvature):
        """Metric deviation scales as (L/R)^2."""
        result1 = curvature.metric_deviation(R=10.0, L=1.0)
        result2 = curvature.metric_deviation(R=20.0, L=1.0)
        # Doubling R should reduce error by ~4x
        ratio = result1['error_bound_L2_over_R2'] / result2['error_bound_L2_over_R2']
        assert abs(ratio - 4.0) < 0.1, f"Expected ratio ~4, got {ratio:.2f}"

    def test_metric_deviation_zero_L(self, curvature):
        """At L=0 (center), metric is exact."""
        result = curvature.metric_deviation(R=5.0, L=0.0)
        assert result['max_relative_error'] == 0.0
        assert result['error_bound_L2_over_R2'] == 0.0

    def test_metric_deviation_conformal_factor(self, curvature):
        """Conformal factor at center is exactly 2."""
        result = curvature.metric_deviation(R=5.0, L=1.0)
        assert result['Omega_center'] == 2.0

    def test_metric_deviation_invalid_R(self, curvature):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            curvature.metric_deviation(R=-1.0, L=0.5)

    def test_metric_deviation_invalid_L(self, curvature):
        """Negative L raises ValueError."""
        with pytest.raises(ValueError):
            curvature.metric_deviation(R=5.0, L=-1.0)

    def test_christoffel_bound_small_L_over_R(self, curvature):
        """Christoffel symbols are small when L << R."""
        result = curvature.christoffel_bound(R=50.0, L=0.5)
        assert result['is_small']

    def test_christoffel_bound_scales_as_L_over_R2(self, curvature):
        """Christoffel symbols scale as O(L/R^2)."""
        r1 = curvature.christoffel_bound(R=10.0, L=1.0)
        r2 = curvature.christoffel_bound(R=20.0, L=1.0)
        # Doubling R with fixed L: leading order scales as 1/R^2
        ratio = r1['leading_order'] / r2['leading_order']
        assert abs(ratio - 4.0) < 0.1, f"Expected ratio ~4, got {ratio:.2f}"

    def test_christoffel_bound_zero_L(self, curvature):
        """Christoffel symbols vanish at L=0 (center of normal coords)."""
        result = curvature.christoffel_bound(R=5.0, L=0.0)
        assert result['christoffel_bound'] == 0.0

    def test_christoffel_invalid_R(self, curvature):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            curvature.christoffel_bound(R=-1.0, L=0.5)

    def test_schwinger_error_2pt(self, curvature):
        """2-point Schwinger function error is O(L^2/R^2)."""
        result = curvature.schwinger_function_error(R=10.0, L=1.0, n=2)
        assert result['n_point'] == 2
        assert result['error_vanishes_as_R_to_inf']
        assert result['rate'] == 'O(L^2/R^2)'

    def test_schwinger_error_decreases_with_R(self, curvature):
        """Error decreases as R increases."""
        err5 = curvature.schwinger_function_error(R=5.0, L=1.0)
        err10 = curvature.schwinger_function_error(R=10.0, L=1.0)
        err50 = curvature.schwinger_function_error(R=50.0, L=1.0)
        assert err5['schwinger_error_bound'] > err10['schwinger_error_bound']
        assert err10['schwinger_error_bound'] > err50['schwinger_error_bound']

    def test_schwinger_error_increases_with_n(self, curvature):
        """Higher n-point functions have larger error bounds."""
        err2 = curvature.schwinger_function_error(R=10.0, L=1.0, n=2)
        err4 = curvature.schwinger_function_error(R=10.0, L=1.0, n=4)
        assert err4['schwinger_error_bound'] > err2['schwinger_error_bound']

    def test_schwinger_error_large_R_is_small(self, curvature):
        """At large R with fixed L, error is negligible."""
        result = curvature.schwinger_function_error(R=100.0, L=1.0)
        assert result['is_small']
        assert result['schwinger_error_bound'] < 0.01

    def test_schwinger_error_invalid_n(self, curvature):
        """n < 1 raises ValueError."""
        with pytest.raises(ValueError):
            curvature.schwinger_function_error(R=5.0, L=1.0, n=0)

    def test_status_is_theorem(self, curvature):
        """Curvature decoupling is a THEOREM."""
        status = curvature.status()
        assert status.label == 'THEOREM'

    def test_multiple_R_values_error_monotone(self, curvature):
        """Error monotonically decreases as R increases (fixed L)."""
        L = 1.0
        R_values = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        errors = []
        for R in R_values:
            result = curvature.schwinger_function_error(R=R, L=L)
            errors.append(result['schwinger_error_bound'])
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Error at R={R_values[i]} ({errors[i]:.6f}) should be > "
                f"error at R={R_values[i+1]} ({errors[i+1]:.6f})"
            )


# ======================================================================
# 4. BridgeStatus
# ======================================================================

class TestBridgeStatus:
    """Tests for the bridge status analysis."""

    def test_what_is_proven_not_empty(self, bridge):
        """Proven results are non-empty."""
        proven = bridge.what_is_proven()
        assert len(proven) > 0

    def test_what_is_proven_has_theorems(self, bridge):
        """Most proven results are labeled THEOREM."""
        proven = bridge.what_is_proven()
        theorem_count = sum(
            1 for v in proven.values()
            if isinstance(v, dict) and v.get('label') == 'THEOREM'
        )
        assert theorem_count >= 7, f"Expected >= 7 THEOREM, got {theorem_count}"

    def test_proof_chain_18_theorems(self, bridge):
        """The proof chain has 18 THEOREM."""
        proven = bridge.what_is_proven()
        assert proven['proof_chain']['count'] == 18
        assert proven['proof_chain']['label'] == 'THEOREM'

    def test_conditional_theorem_is_proven(self, bridge):
        """The conditional theorem itself is listed as THEOREM."""
        proven = bridge.what_is_proven()
        assert proven['conditional_theorem']['label'] == 'THEOREM'

    def test_what_is_needed_has_bridge(self, bridge):
        """What is needed includes the bridge statement."""
        needed = bridge.what_is_needed()
        assert 'bridge_statement' in needed
        assert len(needed['bridge_statement']) > 50

    def test_what_is_needed_has_three_components(self, bridge):
        """What is needed specifies all three hypotheses."""
        needed = bridge.what_is_needed()
        assert 'H1_what_is_needed' in needed
        assert 'H2_what_is_needed' in needed
        assert 'H3_what_is_needed' in needed

    def test_what_is_needed_status_proposition(self, bridge):
        """Overall status of what is needed is PROPOSITION."""
        needed = bridge.what_is_needed()
        assert needed['status'] == 'PROPOSITION'

    def test_approaches_non_empty(self, bridge):
        """There are multiple promising approaches."""
        approaches = bridge.approaches()
        assert len(approaches) >= 3

    def test_approaches_have_status(self, bridge):
        """Each approach has a status and difficulty rating."""
        for approach in bridge.approaches():
            assert 'status' in approach
            assert 'difficulty' in approach
            assert 'references' in approach

    def test_clay_connection_logical_chain(self, bridge):
        """Clay connection has a logical chain."""
        clay = bridge.clay_connection()
        assert 'logical_chain' in clay
        chain = clay['logical_chain']
        assert len(chain) >= 4

    def test_clay_connection_steps_ordered(self, bridge):
        """Clay connection steps are numbered in order."""
        clay = bridge.clay_connection()
        for i, step in enumerate(clay['logical_chain'], start=1):
            assert step['step'] == i

    def test_clay_connection_identifies_proposition(self, bridge):
        """The logical chain identifies exactly which step is PROPOSITION."""
        clay = bridge.clay_connection()
        proposition_steps = [
            s for s in clay['logical_chain']
            if 'PROPOSITION' in s['status']
        ]
        assert len(proposition_steps) >= 1, "Must identify at least one PROPOSITION step"

    def test_clay_connection_has_key_insight(self, bridge):
        """The Clay connection explains the key insight."""
        clay = bridge.clay_connection()
        assert 'key_insight' in clay
        assert len(clay['key_insight']) > 50

    def test_clay_distance_assessment(self, bridge):
        """The distance to Clay is clearly stated."""
        clay = bridge.clay_connection()
        assert 'distance_to_clay' in clay
        assert 'PROPOSITION' in clay['distance_to_clay']

    def test_status_is_proposition(self, bridge):
        """Overall bridge status is PROPOSITION."""
        status = bridge.status()
        assert status.label == 'PROPOSITION'

    def test_su3_bridge_same_structure(self, bridge_su3):
        """SU(3) bridge has the same structure."""
        proven = bridge_su3.what_is_proven()
        assert proven['proof_chain']['count'] == 18

    def test_comparison_with_previous_decompactification(self, bridge):
        """The Clay connection compares with the previous PROPOSITION approach."""
        clay = bridge.clay_connection()
        assert 'comparison_with_previous' in clay
        assert 'PROPOSITION' in clay['comparison_with_previous']


# ======================================================================
# 5. Cross-cutting / integration tests
# ======================================================================

class TestIntegration:
    """Cross-cutting tests verifying consistency across classes."""

    def test_theorem_vs_proposition_distinction(self, cond_thm, bridge):
        """The conditional theorem is THEOREM; the bridge is PROPOSITION."""
        assert cond_thm.status().label == 'THEOREM'
        assert bridge.status().label == 'PROPOSITION'

    def test_curvature_decoupling_is_theorem(self, curvature):
        """Curvature decoupling is labeled THEOREM, not PROPOSITION."""
        assert curvature.status().label == 'THEOREM'

    def test_hypotheses_at_fixed_R_are_theorem(self, cond_thm):
        """At physical R, all hypotheses are THEOREM (not just numerical)."""
        result = cond_thm.verify_at_fixed_R(R=2.2)
        assert result['H1_local_moment_bounds']['status_at_this_R'] == 'THEOREM'
        assert result['H2_uniform_local_coercivity']['status_at_this_R'] == 'THEOREM'
        assert result['H3_uniform_clustering']['status_at_this_R'] == 'THEOREM'

    def test_gap_at_crossover_is_positive(self, cond_thm, clustering):
        """The gap at the crossover radius is positive."""
        R_cross = clustering._gap_bound.crossover_R()
        result = cond_thm.verify_at_fixed_R(R=R_cross)
        assert result['all_satisfied']
        assert result['H3_uniform_clustering']['clustering_rate_MeV'] > 100

    def test_decoupling_and_clustering_consistent(self, cond_thm, clustering, curvature):
        """Curvature decoupling and clustering estimates are consistent."""
        R = 10.0
        L = 1.0
        # Clustering gives correlation length
        cl_data = clustering.estimate_correlation_length(R=R)
        xi = cl_data['xi_fm']
        # Curvature decoupling gives metric error
        curv_data = curvature.schwinger_function_error(R=R, L=L)
        err = curv_data['schwinger_error_bound']
        # The error from curvature should be small compared to 1
        # (correlators have O(1) magnitude)
        assert err < 1.0, f"Curvature error {err} should be << 1"
        # The correlation length should be finite
        assert xi < R, f"xi = {xi} should be < R = {R}"

    def test_full_logical_chain_status(self, cond_thm, bridge):
        """The full logical chain has clear status labels."""
        # The conditional theorem: THEOREM
        assert cond_thm.status().label == 'THEOREM'
        # The bridge: PROPOSITION
        assert bridge.status().label == 'PROPOSITION'
        # The clay connection identifies exactly one gap
        clay = bridge.clay_connection()
        # Count PROPOSITION steps
        prop_count = sum(
            1 for s in clay['logical_chain']
            if 'PROPOSITION' in s['status']
        )
        assert prop_count == 1, f"Expected exactly 1 PROPOSITION step, got {prop_count}"

    def test_scan_R_range_no_failures(self, cond_thm):
        """Scan over a range of R values: all pass at fixed R."""
        R_values = np.logspace(np.log10(0.2), np.log10(100.0), 25)
        for R in R_values:
            result = cond_thm.verify_at_fixed_R(R=R)
            assert result['all_satisfied'], f"Failed at R = {R:.3f} fm"

    def test_physical_constants_consistent(self):
        """Physical constants are consistent across modules."""
        assert abs(R_0_FM - R_PHYSICAL_FM) < 0.01
        assert abs(G_SQUARED_PHYS - 6.28) < 0.01
        assert abs(ALPHA_S_PHYS - G_SQUARED_PHYS / (4 * np.pi)) < 0.001

    def test_gap_never_zero_in_dynamic_regime(self, clustering):
        """In the dynamic regime R > R_cross, gap is always > Lambda_QCD."""
        R_cross = clustering._gap_bound.crossover_R()
        R_values = np.linspace(R_cross * 1.5, 100.0, 30)
        result = clustering.uniform_bound_check(R_values)
        assert result['all_positive']
        # In dynamic regime, gap should be ~ Lambda_QCD = 200 MeV
        assert result['min_gap_dynamic_MeV'] >= 190.0, (
            f"Min dynamic gap = {result['min_gap_dynamic_MeV']:.1f} MeV, "
            "expected >= 190 MeV"
        )
