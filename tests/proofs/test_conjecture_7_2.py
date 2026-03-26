"""
Tests for Conjecture 7.2 Attack: Synthesis of Phase 1 Results.

Tests the proof chain that combines:
    1. Effective Hamiltonian on S^3/I* (finite-dim gap)
    2. S^4 Compactification (conformal bridge to R^4)
    3. Gap Monotonicity (Delta(R) > 0 for all R)

Test categories:
    1. Proof chain structure and integrity
    2. Rigor labels (no THEOREM claims for CONJECTURE-level results)
    3. Finite-dim gap theorem verification (various R, g^2)
    4. Lifting from S^3/I* to S^3 (spectrum containment)
    5. V_4 >= 0 on full S^3 k=1 space (18 DOF)
    6. Spectral desert verification (k=1 to k=11 on S^3/I*)
    7. R -> infinity consistency
    8. Conformal bridge ingredients
    9. proof_status() deliverable
   10. Edge cases: R -> 0, R -> inf, g -> 0, g -> inf
   11. Integration with Phase 1 modules
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.conjecture_7_2 import (
    ProofChain,
    ProofStep,
    LiftingArgument,
    RInfinityArgument,
    SpectralDesert,
    GapToClay,
    NumericalVerification,
    RigorLevel,
    proof_status,
    full_analysis,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
    COEXACT_GAP_COEFF,
    COEXACT_MASS_COEFF,
    K1_LEVEL,
    K2_LEVEL_POINCARE,
    K2_LEVEL_S3,
    EIGENVALUE_K1,
    EIGENVALUE_K2_POINCARE,
    EIGENVALUE_K2_S3,
    SPECTRAL_DESERT_RATIO,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def proof_chain():
    """Default SU(2) proof chain."""
    return ProofChain(N=2, Lambda_QCD=200.0)


@pytest.fixture
def lifting():
    """LiftingArgument with R=1, g=1."""
    return LiftingArgument(R=1.0, g_coupling=1.0)


@pytest.fixture
def r_infinity():
    """RInfinityArgument for SU(2)."""
    return RInfinityArgument(N=2, Lambda_QCD=200.0)


@pytest.fixture
def desert():
    """SpectralDesert at R=2.2 fm."""
    return SpectralDesert(R=2.2)


@pytest.fixture
def verification():
    """NumericalVerification for SU(2)."""
    return NumericalVerification(N=2, Lambda_QCD=200.0)


@pytest.fixture
def gap_clay():
    """GapToClay assessor."""
    return GapToClay(N=2, Lambda_QCD=200.0)


# ======================================================================
# 1. Proof chain structure and integrity
# ======================================================================

class TestProofChainStructure:
    """Verify the 8-step proof chain is well-formed."""

    def test_chain_has_8_steps(self, proof_chain):
        """The proof chain must have exactly 8 steps."""
        chain = proof_chain.build_chain()
        assert len(chain) == 8

    def test_steps_numbered_consecutively(self, proof_chain):
        """Steps are numbered 1 through 8."""
        chain = proof_chain.build_chain()
        numbers = [s.number for s in chain]
        assert numbers == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_all_steps_are_ProofStep(self, proof_chain):
        """Each step is a ProofStep dataclass."""
        chain = proof_chain.build_chain()
        for step in chain:
            assert isinstance(step, ProofStep)

    def test_final_step_is_conjecture(self, proof_chain):
        """The final step (Conjecture 7.2) must be labeled CONJECTURE."""
        chain = proof_chain.build_chain()
        assert chain[-1].label == 'CONJECTURE'

    def test_final_step_depends_on_all(self, proof_chain):
        """Step 8 depends on steps 1-7."""
        chain = proof_chain.build_chain()
        final = chain[-1]
        assert set(final.dependencies) == {1, 2, 3, 4, 5, 6, 7}

    def test_step_1_is_theorem(self, proof_chain):
        """Step 1 (Kato-Rellich) is THEOREM."""
        chain = proof_chain.build_chain()
        assert chain[0].label == 'THEOREM'

    def test_step_2_is_theorem(self, proof_chain):
        """Step 2 (finite-dim gap) is THEOREM."""
        chain = proof_chain.build_chain()
        assert chain[1].label == 'THEOREM'

    def test_step_3_is_theorem(self, proof_chain):
        """Step 3 (covering space lift) is THEOREM."""
        chain = proof_chain.build_chain()
        assert chain[2].label == 'THEOREM'

    def test_step_4_is_theorem(self, proof_chain):
        """Step 4 (operator comparison: gap(H_full) >= gap(H_3)) is THEOREM."""
        chain = proof_chain.build_chain()
        assert chain[3].label == 'THEOREM'

    def test_step_5_is_proposition(self, proof_chain):
        """Step 5 (Lambda_QCD floor) is PROPOSITION."""
        chain = proof_chain.build_chain()
        assert chain[4].label == 'PROPOSITION'

    def test_step_6_is_numerical(self, proof_chain):
        """Step 6 (gap scan + anharmonic scaling) is NUMERICAL."""
        chain = proof_chain.build_chain()
        assert chain[5].label == 'NUMERICAL'

    def test_step_7_is_theorem(self, proof_chain):
        """Step 7 (conformal bridge ingredients) is THEOREM."""
        chain = proof_chain.build_chain()
        assert chain[6].label == 'THEOREM'

    def test_dependencies_only_reference_earlier_steps(self, proof_chain):
        """No step depends on a later step (acyclic graph)."""
        chain = proof_chain.build_chain()
        for step in chain:
            for dep in step.dependencies:
                assert dep < step.number, (
                    f"Step {step.number} depends on step {dep} which is not earlier"
                )

    def test_chain_rigor_summary(self, proof_chain):
        """chain_rigor_summary returns a dict of step -> label."""
        summary = proof_chain.chain_rigor_summary()
        assert len(summary) == 8
        assert summary[1] == 'THEOREM'
        assert summary[3] == 'THEOREM'  # Covering space lift
        assert summary[8] == 'CONJECTURE'

    def test_weakest_link_is_conjecture(self, proof_chain):
        """The weakest link in the chain should be the CONJECTURE step."""
        weakest = proof_chain.weakest_link()
        assert weakest.label == 'CONJECTURE'


# ======================================================================
# 2. Rigor labels (honest labeling)
# ======================================================================

class TestRigorLabeling:
    """Ensure no THEOREM claims for CONJECTURE-level results."""

    def test_r_infinity_is_not_theorem(self, proof_chain):
        """The R -> inf persistence must NOT be labeled THEOREM."""
        chain = proof_chain.build_chain()
        r_inf_step = [s for s in chain if 'R ->' in s.statement or 'R -> inf' in s.statement or 'infinity' in s.statement]
        for step in r_inf_step:
            assert step.label != 'THEOREM', (
                f"Step {step.number} claims R->inf as THEOREM: {step.statement}"
            )

    def test_operator_comparison_is_theorem(self, proof_chain):
        """Step 4 (operator comparison: gap(H_full) >= gap(H_3)) is THEOREM.

        Upgraded from PROPOSITION via adiabatic_comparison.py:
        V_coupling >= 0 (eigenspace orthogonality) + Reed-Simon operator comparison.
        """
        chain = proof_chain.build_chain()
        step4 = chain[3]
        assert step4.label == 'THEOREM', (
            f"Step 4 (operator comparison) should be THEOREM, got {step4.label}"
        )

    def test_confinement_argument_is_not_theorem(self, proof_chain):
        """The confinement -> gap argument is PROPOSITION (not proven)."""
        chain = proof_chain.build_chain()
        step5 = chain[4]  # Step 5 in new chain (was Step 4)
        assert step5.label in ('PROPOSITION', 'NUMERICAL'), (
            f"Step 5 (confinement) labeled as {step5.label}"
        )

    def test_all_steps_have_caveats(self, proof_chain):
        """Every step must have non-empty caveats."""
        chain = proof_chain.build_chain()
        for step in chain:
            assert step.caveats, f"Step {step.number} has empty caveats"

    def test_all_steps_have_evidence(self, proof_chain):
        """Every step must have non-empty evidence."""
        chain = proof_chain.build_chain()
        for step in chain:
            assert step.evidence, f"Step {step.number} has empty evidence"


# ======================================================================
# 3. Finite-dim gap theorem verification (various R, g^2)
# ======================================================================

class TestFiniteDimGap:
    """Verify the finite-dim gap theorem from effective_hamiltonian."""

    def test_v4_nonneg_unit_sphere(self, lifting):
        """V_4 >= 0 on the I*-invariant (9 DOF) space at R=1."""
        result = lifting.verify_v4_nonnegative(n_samples=2000)
        assert result['nonnegative']
        assert result['min_value'] >= -1e-12

    def test_v4_nonneg_large_R(self):
        """V_4 >= 0 at R=100 fm."""
        lift = LiftingArgument(R=100.0, g_coupling=1.0)
        result = lift.verify_v4_nonnegative(n_samples=1000)
        assert result['nonnegative']

    def test_v4_nonneg_strong_coupling(self):
        """V_4 >= 0 at strong coupling g=10."""
        lift = LiftingArgument(R=1.0, g_coupling=10.0)
        result = lift.verify_v4_nonnegative(n_samples=1000)
        assert result['nonnegative']

    def test_total_potential_nonneg(self, lifting):
        """V = V_2 + V_4 >= 0 at random configs."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            a = rng.standard_normal((6, 3)) * rng.uniform(0.01, 10.0)
            v = lifting.total_potential_full_s3(a)
            assert v >= -1e-12, f"V = {v} < 0 at config"

    def test_potential_zero_at_origin(self, lifting):
        """V(0) = 0."""
        v = lifting.total_potential_full_s3(np.zeros((6, 3)))
        assert abs(v) < 1e-15

    def test_potential_positive_away_from_origin(self, lifting):
        """V(a) > 0 for a != 0."""
        rng = np.random.default_rng(88)
        for _ in range(100):
            a = rng.standard_normal((6, 3))
            a_norm = np.linalg.norm(a)
            if a_norm > 0.01:
                v = lifting.total_potential_full_s3(a)
                assert v > 0, f"V = {v} <= 0 at |a| = {a_norm}"

    def test_harmonic_gap_value(self, lifting):
        """Harmonic gap = 2/R."""
        omega = np.sqrt(COEXACT_GAP_COEFF / lifting.R**2)
        assert abs(omega - 2.0 / lifting.R) < 1e-14


# ======================================================================
# 4. Lifting from S^3/I* to S^3
# ======================================================================

class TestLiftingArgument:
    """Test the lifting from S^3/I* gap to full S^3 gap."""

    def test_spectral_containment(self, lifting):
        """Spectrum of S^3/I* is contained in spectrum of S^3."""
        result = lifting.spectral_containment()
        assert result['containment'] is True

    def test_k1_eigenvalue_same(self, lifting):
        """k=1 eigenvalue is 4/R^2 on both S^3 and S^3/I*."""
        result = lifting.spectral_containment()
        expected = EIGENVALUE_K1 / lifting.R**2
        assert abs(result['k1_eigenvalue'] - expected) < 1e-14

    def test_more_modes_on_s3(self, lifting):
        """S^3 has 6 coexact modes at k=1, S^3/I* has 3."""
        result = lifting.spectral_containment()
        assert result['k1_modes_s3'] == 6
        assert result['k1_modes_poincare'] == 3

    def test_desert_ratio_36(self, lifting):
        """Spectral desert ratio is 36 on S^3/I*."""
        result = lifting.spectral_containment()
        assert abs(result['spectral_desert_ratio'] - 36.0) < 1e-10

    def test_gap_relation_label_is_theorem(self, lifting):
        """The gap relation (S^3/I* -> S^3) is labeled THEOREM (covering space lift)."""
        result = lifting.gap_relation()
        assert result['label'] == 'THEOREM'

    def test_v4_nonneg_full_s3_18dof(self, lifting):
        """V_4 >= 0 on the full 18-DOF space on S^3."""
        result = lifting.verify_v4_nonnegative(n_samples=3000)
        assert result['nonnegative']
        assert result['n_dof'] == 18

    def test_v4_algebraic_identity(self):
        """
        V_4 = (g^2/2)[(Tr S)^2 - Tr(S^2)] where S = M^T M >= 0.
        Since eigenvalues s_i >= 0: (sum s_i)^2 >= sum s_i^2.
        """
        rng = np.random.default_rng(77)
        for n_rows in [3, 6, 10, 50]:
            for _ in range(100):
                M = rng.standard_normal((n_rows, 3))
                S = M.T @ M
                tr_S = np.trace(S)
                tr_S2 = np.trace(S @ S)
                assert tr_S**2 >= tr_S2 - 1e-10, (
                    f"(Tr S)^2 = {tr_S**2} < Tr(S^2) = {tr_S2} for {n_rows}x3 M"
                )


# ======================================================================
# 5. Spectral desert verification
# ======================================================================

class TestSpectralDesert:
    """Verify the spectral desert on S^3/I* vs S^3."""

    def test_k1_eigenvalue(self, desert):
        """k=1 eigenvalue is 4/R^2."""
        ev = desert.eigenvalue_k(1)
        assert abs(ev - 4.0 / desert.R**2) < 1e-14

    def test_k2_eigenvalue_s3(self, desert):
        """k=2 eigenvalue on S^3 is 9/R^2."""
        ev = desert.eigenvalue_k(2)
        assert abs(ev - 9.0 / desert.R**2) < 1e-14

    def test_k11_eigenvalue(self, desert):
        """k=11 eigenvalue is 144/R^2."""
        ev = desert.eigenvalue_k(11)
        assert abs(ev - 144.0 / desert.R**2) < 1e-14

    def test_spectral_gap_s3_ratio(self, desert):
        """On S^3: ratio of k=2 to k=1 eigenvalues is 9/4 = 2.25."""
        result = desert.spectral_gap_s3()
        assert abs(result['ratio'] - 9.0 / 4.0) < 1e-14

    def test_spectral_gap_poincare_ratio(self, desert):
        """On S^3/I*: ratio of k=11 to k=1 eigenvalues is 144/4 = 36."""
        result = desert.spectral_gap_poincare()
        assert abs(result['ratio'] - 36.0) < 1e-14

    def test_enhancement_factor(self, desert):
        """Enhancement from S^3 to S^3/I* is 36 / 2.25 = 16."""
        comp = desert.desert_comparison()
        expected = 36.0 / 2.25
        assert abs(comp['enhancement'] - expected) < 1e-10

    def test_truncation_error_poincare_small(self, desert):
        """Truncation error on S^3/I* is ~ 4/144 ~ 0.028 (small)."""
        result = desert.truncation_error_estimate()
        assert result['truncation_error_poincare'] < 0.03
        assert result['poincare_reliable'] is True

    def test_truncation_error_s3_large(self, desert):
        """Truncation error on S^3 is ~ 4/9 ~ 0.44 (too large)."""
        result = desert.truncation_error_estimate()
        assert result['truncation_error_s3'] > 0.4
        assert result['s3_reliable'] is False

    def test_mass_at_k1(self, desert):
        """Mass at k=1 is 2*hbar_c/R."""
        m = desert.mass_k(1)
        expected = 2.0 * HBAR_C_MEV_FM / desert.R
        assert abs(m - expected) < 1e-10

    def test_mass_at_k11(self, desert):
        """Mass at k=11 is 12*hbar_c/R."""
        m = desert.mass_k(11)
        expected = 12.0 * HBAR_C_MEV_FM / desert.R
        assert abs(m - expected) < 1e-10


# ======================================================================
# 6. R -> infinity consistency
# ======================================================================

class TestRInfinity:
    """Test the R -> infinity argument."""

    def test_geometric_gap_decreases_with_R(self, r_infinity):
        """Geometric gap 2*hbar_c/R decreases as R increases."""
        g1 = r_infinity.geometric_gap_MeV(1.0)
        g2 = r_infinity.geometric_gap_MeV(10.0)
        assert g1 > g2

    def test_dynamical_gap_constant(self, r_infinity):
        """Dynamical gap = Lambda_QCD (constant)."""
        assert abs(r_infinity.dynamical_gap_MeV() - 200.0) < 1e-10

    def test_best_gap_always_positive(self, r_infinity):
        """Best gap > 0 for all tested R."""
        for R in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            result = r_infinity.best_gap_MeV(R)
            assert result['gap_MeV'] > 0, f"Gap <= 0 at R = {R}"

    def test_best_gap_at_small_R_is_geometric(self, r_infinity):
        """At R = 0.01 fm, geometric gap dominates."""
        result = r_infinity.best_gap_MeV(0.01)
        assert result['geometric_MeV'] > result['dynamical_MeV']

    def test_best_gap_at_large_R_is_dynamical(self, r_infinity):
        """At R = 1000 fm, dynamical gap dominates."""
        result = r_infinity.best_gap_MeV(1000.0)
        assert result['dynamical_MeV'] >= result['geometric_MeV']

    def test_gap_infimum_positive(self, r_infinity):
        """inf Delta(R) > 0 over tested R values."""
        result = r_infinity.gap_infimum()
        assert result['infimum_MeV'] > 0
        assert result['all_positive'] is True
        assert result['conjecture_7_2_supported'] is True

    def test_gap_infimum_at_least_lambda_qcd(self, r_infinity):
        """Infimum of gap >= Lambda_QCD."""
        result = r_infinity.gap_infimum()
        assert result['infimum_MeV'] >= r_infinity.Lambda_QCD - 1.0  # 1 MeV tolerance

    def test_three_regimes(self, r_infinity):
        """Three regime summary is well-formed."""
        regimes = r_infinity.three_regime_summary()
        assert 'regime_1' in regimes
        assert 'regime_2' in regimes
        assert 'regime_3' in regimes
        assert regimes['regime_1']['rigor'] == 'THEOREM (Kato-Rellich)'

    def test_running_coupling_small_R(self, r_infinity):
        """Running coupling is finite for small R (perturbative)."""
        g2 = r_infinity.running_coupling(0.01)
        assert np.isfinite(g2)
        assert g2 > 0

    def test_running_coupling_large_R(self, r_infinity):
        """Running coupling is inf for large R (non-perturbative)."""
        g2 = r_infinity.running_coupling(100.0)
        assert np.isinf(g2)


# ======================================================================
# 7. Conformal bridge consistency
# ======================================================================

class TestConformalBridge:
    """Test the conformal bridge ingredients."""

    def test_ym_conformally_invariant_4d(self, verification):
        """YM action is conformally invariant in dim 4."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['ym_conformally_invariant'] is True

    def test_ym_conformal_weight_zero_4d(self, verification):
        """YM conformal weight is 0 in dim 4."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['ym_conformal_weight_4d'] == 0

    def test_point_capacity_small_in_4d(self, verification):
        """Point capacity in dim 4 is effectively zero."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['capacity_effectively_zero']

    def test_sobolev_unchanged_4d(self, verification):
        """W^{1,2} is unchanged by point removal in dim 4."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['sobolev_unchanged'] is True

    def test_all_bridge_ingredients_pass(self, verification):
        """All conformal bridge ingredients pass."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['all_passed'] is True


# ======================================================================
# 8. proof_status() deliverable
# ======================================================================

class TestProofStatus:
    """Test the main deliverable function."""

    def test_returns_dict(self):
        """proof_status() returns a dict."""
        result = proof_status()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Result has all required keys."""
        result = proof_status()
        for key in ['proven', 'proposed', 'numerical', 'conjectured', 'gap_to_clay']:
            assert key in result, f"Missing key: {key}"

    def test_proven_is_list(self):
        """'proven' is a non-empty list."""
        result = proof_status()
        assert isinstance(result['proven'], list)
        assert len(result['proven']) > 0

    def test_proposed_is_list(self):
        """'proposed' is a non-empty list."""
        result = proof_status()
        assert isinstance(result['proposed'], list)
        assert len(result['proposed']) > 0

    def test_conjectured_is_list(self):
        """'conjectured' is a non-empty list."""
        result = proof_status()
        assert isinstance(result['conjectured'], list)
        assert len(result['conjectured']) > 0

    def test_gap_to_clay_is_string(self):
        """'gap_to_clay' is a non-empty string."""
        result = proof_status()
        assert isinstance(result['gap_to_clay'], str)
        assert len(result['gap_to_clay']) > 100  # substantial paragraph

    def test_proven_items_are_theorem(self):
        """All proven items are labeled THEOREM."""
        result = proof_status()
        for item in result['proven']:
            assert item['label'] == 'THEOREM', (
                f"Proven item labeled {item['label']}: {item['statement']}"
            )

    def test_proposed_items_are_proposition(self):
        """All proposed items are labeled PROPOSITION."""
        result = proof_status()
        for item in result['proposed']:
            assert item['label'] == 'PROPOSITION', (
                f"Proposed item labeled {item['label']}: {item['statement']}"
            )

    def test_conjectured_items_are_conjecture(self):
        """All conjectured items are labeled CONJECTURE."""
        result = proof_status()
        for item in result['conjectured']:
            assert item['label'] == 'CONJECTURE', (
                f"Conjectured item labeled {item['label']}: {item['statement']}"
            )

    def test_gap_to_clay_mentions_clay(self):
        """The assessment mentions the Clay problem."""
        result = proof_status()
        assert 'Clay' in result['gap_to_clay'] or 'clay' in result['gap_to_clay']

    def test_gap_to_clay_is_honest(self):
        """The assessment mentions what is NOT proven."""
        result = proof_status()
        text = result['gap_to_clay'].lower()
        assert 'not' in text or 'remain' in text or 'conjecture' in text

    def test_su3_proof_status(self):
        """proof_status works for SU(3)."""
        result = proof_status(N=3)
        assert len(result['proven']) > 0

    def test_different_lambda_qcd(self):
        """proof_status works with Lambda_QCD = 300 MeV."""
        result = proof_status(Lambda_QCD=300.0)
        assert isinstance(result['gap_to_clay'], str)


# ======================================================================
# 9. Edge cases
# ======================================================================

class TestEdgeCases:
    """Test edge cases: R -> 0, R -> inf, g -> 0, g -> inf."""

    def test_small_R_gap_large(self, r_infinity):
        """At R = 0.001 fm, gap is huge."""
        result = r_infinity.best_gap_MeV(0.001)
        assert result['gap_MeV'] > 100000  # > 100 GeV

    def test_large_R_gap_lambda(self, r_infinity):
        """At R = 10000 fm, gap is Lambda_QCD."""
        result = r_infinity.best_gap_MeV(10000.0)
        assert abs(result['gap_MeV'] - 200.0) < 1.0

    def test_zero_coupling_v4_zero(self):
        """At g=0, V_4 = 0."""
        lift = LiftingArgument(R=1.0, g_coupling=0.0)
        a = np.random.default_rng(42).standard_normal((6, 3))
        v4 = lift.quartic_potential_full_s3(a)
        assert abs(v4) < 1e-15

    def test_strong_coupling_v4_large(self):
        """At g=100, V_4 is large for non-zero a with structure."""
        lift = LiftingArgument(R=1.0, g_coupling=100.0)
        # Use a config with non-degenerate singular values (not proportional columns)
        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ])
        v4 = lift.quartic_potential_full_s3(a)
        assert v4 > 0

    def test_identity_config_v4(self):
        """V_4 for specific a = identity-like config."""
        lift = LiftingArgument(R=1.0, g_coupling=1.0)
        a = np.zeros((6, 3))
        a[0, 0] = 1.0
        a[1, 1] = 1.0
        a[2, 2] = 1.0
        v4 = lift.quartic_potential_full_s3(a)
        assert v4 >= 0

    def test_very_small_R_geometric_gap(self, r_infinity):
        """Geometric gap at R = 1e-6 fm is 2*hbar_c/R."""
        R = 1e-6
        g = r_infinity.geometric_gap_MeV(R)
        expected = 2.0 * HBAR_C_MEV_FM / R
        assert abs(g - expected) / expected < 1e-10

    def test_negative_R_raises(self, r_infinity):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            r_infinity.running_coupling(-1.0)

    def test_zero_R_raises(self, r_infinity):
        """R = 0 raises ValueError."""
        with pytest.raises(ValueError):
            r_infinity.running_coupling(0.0)


# ======================================================================
# 10. Numerical verification suite
# ======================================================================

class TestNumericalVerification:
    """Test the NumericalVerification class."""

    def test_spectral_desert_verified(self, verification):
        """Spectral desert ratio is 36."""
        result = verification.verify_spectral_desert()
        assert result['passed'] is True
        assert abs(result['ratio'] - 36.0) < 1e-10

    def test_confining_potential_verified(self, verification):
        """V_4 >= 0 and V is confining."""
        result = verification.verify_confining_potential(n_samples=1000)
        assert result['v4_nonnegative']
        assert result['v_total_nonnegative']

    def test_conformal_bridge_verified(self, verification):
        """All conformal bridge ingredients pass."""
        result = verification.verify_conformal_bridge_consistency()
        assert result['all_passed'] is True

    def test_effective_vs_full_gap(self, verification):
        """Effective and full gap estimates are consistent."""
        result = verification.verify_effective_vs_full_gap(R=2.2)
        assert result['effective_gap_MeV'] > 0
        assert result['full_gap_MeV'] > 0


# ======================================================================
# 11. Constants consistency
# ======================================================================

class TestConstants:
    """Verify internal constants are consistent."""

    def test_eigenvalue_k1(self):
        """EIGENVALUE_K1 = (1+1)^2 = 4."""
        assert EIGENVALUE_K1 == 4

    def test_eigenvalue_k2_poincare(self):
        """EIGENVALUE_K2_POINCARE = (11+1)^2 = 144."""
        assert EIGENVALUE_K2_POINCARE == 144

    def test_eigenvalue_k2_s3(self):
        """EIGENVALUE_K2_S3 = (2+1)^2 = 9."""
        assert EIGENVALUE_K2_S3 == 9

    def test_spectral_desert_ratio(self):
        """SPECTRAL_DESERT_RATIO = 144/4 = 36."""
        assert abs(SPECTRAL_DESERT_RATIO - 36.0) < 1e-10

    def test_hbar_c(self):
        """hbar*c ~ 197.3 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-4

    def test_coexact_gap_coeff(self):
        """Coexact gap coefficient is 4."""
        assert COEXACT_GAP_COEFF == 4.0

    def test_coexact_mass_coeff(self):
        """Mass coefficient is 2 (sqrt of 4)."""
        assert abs(COEXACT_MASS_COEFF - 2.0) < 1e-14


# ======================================================================
# 12. Integration with GapToClay
# ======================================================================

class TestGapToClay:
    """Test the honest gap-to-Clay assessment."""

    def test_proven_has_multiple_theorems(self, gap_clay):
        """There are multiple THEOREM-level results."""
        proven = gap_clay.proven_results()
        assert len(proven) >= 5

    def test_proposed_has_upgrade_paths(self, gap_clay):
        """Every PROPOSITION has an upgrade path."""
        proposed = gap_clay.proposed_results()
        for item in proposed:
            assert 'upgrade_path' in item
            assert len(item['upgrade_path']) > 10

    def test_conjectured_has_upgrade_paths(self, gap_clay):
        """Every CONJECTURE has an upgrade path."""
        conjectured = gap_clay.conjectured_results()
        for item in conjectured:
            assert 'upgrade_path' in item

    def test_gap_to_clay_not_empty(self, gap_clay):
        """Gap to Clay paragraph is substantial."""
        text = gap_clay.gap_to_clay()
        assert len(text) > 200  # at least a paragraph


# ======================================================================
# 13. full_analysis() integration
# ======================================================================

class TestFullAnalysis:
    """Test the full_analysis convenience function."""

    def test_returns_dict(self):
        """full_analysis() returns a dict."""
        result = full_analysis()
        assert isinstance(result, dict)

    def test_has_proof_chain(self):
        """Result has proof_chain."""
        result = full_analysis()
        assert 'proof_chain' in result
        assert len(result['proof_chain']) == 8

    def test_has_lifting(self):
        """Result has lifting analysis."""
        result = full_analysis()
        assert 'lifting' in result

    def test_has_r_infinity(self):
        """Result has R->inf analysis."""
        result = full_analysis()
        assert 'r_infinity' in result

    def test_has_spectral_desert(self):
        """Result has spectral desert."""
        result = full_analysis()
        assert 'spectral_desert' in result

    def test_has_verification(self):
        """Result has numerical verification."""
        result = full_analysis()
        assert 'verification' in result

    def test_has_status(self):
        """Result has proof_status."""
        result = full_analysis()
        assert 'status' in result
        assert 'gap_to_clay' in result['status']
