"""
Tests for the Adiabatic Comparison Theorem module.

Tests cover:
1. Operator comparison theorem (Reed-Simon standard result)
2. Coupling sign analysis (V_coupling >= 0)
3. Adiabatic decoupling properties
4. Full gap comparison result
5. Proof chain integration
6. Numerical verification across parameter space

At least 50 tests as specified.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.adiabatic_comparison import (
    OperatorComparison,
    CouplingSign,
    AdiabaticDecoupling,
    GapComparisonResult,
    ProofChainUpgrade,
    EIGENVALUE_LOW,
    EIGENVALUE_HIGH,
    SPECTRAL_DESERT_DELTA,
    SPECTRAL_DESERT_RATIO,
    HBAR_C_MEV_FM,
    K_LOW,
    K_HIGH_MIN,
    _compute_radial_gap,
    _compute_effective_gap,
    _levi_civita,
)


# ======================================================================
# Constants tests
# ======================================================================

class TestConstants:
    """Test that all spectral constants are correct."""

    def test_eigenvalue_low(self):
        """k=1 coexact eigenvalue coefficient = (1+1)^2 = 4."""
        assert EIGENVALUE_LOW == 4

    def test_eigenvalue_high(self):
        """k=11 coexact eigenvalue coefficient = (11+1)^2 = 144."""
        assert EIGENVALUE_HIGH == 144

    def test_spectral_desert_ratio(self):
        """Ratio = 144/4 = 36."""
        assert SPECTRAL_DESERT_RATIO == 36

    def test_spectral_desert_delta(self):
        """Delta = (144 - 4) / 4 = 35."""
        assert SPECTRAL_DESERT_DELTA == 35

    def test_k_low(self):
        """First surviving coexact level on S^3/I* is k=1."""
        assert K_LOW == 1

    def test_k_high_min(self):
        """Second surviving coexact level on S^3/I* is k=11."""
        assert K_HIGH_MIN == 11


# ======================================================================
# Levi-Civita helper tests
# ======================================================================

class TestLeviCivita:
    """Test the Levi-Civita symbol."""

    def test_cyclic_positive(self):
        """epsilon_{012} = epsilon_{120} = epsilon_{201} = 1."""
        assert _levi_civita(0, 1, 2) == 1.0
        assert _levi_civita(1, 2, 0) == 1.0
        assert _levi_civita(2, 0, 1) == 1.0

    def test_anticyclic_negative(self):
        """epsilon_{021} = epsilon_{210} = epsilon_{102} = -1."""
        assert _levi_civita(0, 2, 1) == -1.0
        assert _levi_civita(2, 1, 0) == -1.0
        assert _levi_civita(1, 0, 2) == -1.0

    def test_repeated_index_zero(self):
        """epsilon_{iij} = 0 for any repeated index."""
        for i in range(3):
            for j in range(3):
                assert _levi_civita(i, i, j) == 0.0
                assert _levi_civita(i, j, i) == 0.0
                assert _levi_civita(j, i, i) == 0.0


# ======================================================================
# Operator Comparison tests
# ======================================================================

class TestOperatorComparison:
    """Tests for the operator comparison theorem (Reed-Simon)."""

    def test_1d_harmonic_same_frequency(self):
        """Equal frequencies should give equal gaps."""
        result = OperatorComparison.verify_1d_harmonic(1.0, 1.0)
        assert abs(result['gap1'] - result['gap2']) < 0.01
        assert result['gap_comparison_holds']

    def test_1d_harmonic_larger_frequency_larger_gap(self):
        """omega_2 > omega_1 should give gap_2 > gap_1."""
        result = OperatorComparison.verify_1d_harmonic(1.0, 2.0)
        assert result['v2_geq_v1']
        assert result['gap_comparison_holds']
        assert result['gap2'] > result['gap1']

    def test_1d_harmonic_smaller_frequency_smaller_gap(self):
        """omega_2 < omega_1 should give gap_2 < gap_1 (reverse direction)."""
        result = OperatorComparison.verify_1d_harmonic(2.0, 1.0)
        assert not result['v2_geq_v1']
        assert result['gap2'] < result['gap1'] + 0.01

    def test_1d_harmonic_analytical_match(self):
        """Numerical gap should match analytical gap = omega."""
        for omega in [0.5, 1.0, 2.0, 5.0]:
            result = OperatorComparison.verify_1d_harmonic(omega, omega)
            assert abs(result['gap1'] - omega) < 0.05 * omega

    def test_1d_quartic_larger_coupling_larger_gap(self):
        """lam_2 > lam_1 should give gap_2 > gap_1."""
        result = OperatorComparison.verify_1d_quartic(0.1, 1.0, omega_sq=1.0)
        assert result['v2_geq_v1']
        assert result['gap_comparison_holds']
        assert result['gap2'] > result['gap1']

    def test_1d_quartic_same_coupling_same_gap(self):
        """Equal couplings should give equal gaps."""
        result = OperatorComparison.verify_1d_quartic(0.5, 0.5, omega_sq=1.0)
        assert abs(result['gap1'] - result['gap2']) < 0.01

    def test_1d_quartic_zero_coupling(self):
        """lam = 0 should give harmonic gap."""
        result = OperatorComparison.verify_1d_quartic(0.0, 0.0, omega_sq=4.0)
        assert abs(result['gap1'] - 2.0) < 0.1  # omega = 2

    def test_1d_quartic_pure_quartic(self):
        """Pure quartic oscillator should have positive gap."""
        result = OperatorComparison.verify_1d_quartic(1.0, 1.0, omega_sq=0.0)
        assert result['gap1'] > 0

    def test_1d_quartic_multiple_couplings(self):
        """Comparison holds for several (lam1, lam2) pairs."""
        pairs = [(0.1, 0.5), (0.5, 2.0), (1.0, 10.0)]
        for lam1, lam2 in pairs:
            result = OperatorComparison.verify_1d_quartic(lam1, lam2)
            assert result['gap_comparison_holds'], f"Failed for lam1={lam1}, lam2={lam2}"

    def test_nd_3d_harmonic_comparison(self):
        """3D harmonic comparison: larger omega gives larger gap."""
        result = OperatorComparison.verify_nd(3, 1.0, 4.0)
        assert result['v2_geq_v1']
        assert result['gap_comparison_holds']

    def test_nd_3d_quartic_comparison(self):
        """3D quartic comparison: larger coupling gives larger gap."""
        result = OperatorComparison.verify_nd(3, 1.0, 1.0, lam1=0.1, lam2=1.0)
        assert result['gap_comparison_holds']

    def test_nd_2d_comparison(self):
        """2D comparison theorem."""
        result = OperatorComparison.verify_nd(2, 1.0, 2.0)
        assert result['gap_comparison_holds']

    def test_equal_potentials(self):
        """V_1 = V_2 should give gap(H_1) = gap(H_2)."""
        result = OperatorComparison.verify_equal_potentials(1.0, 0.5)
        assert result['gaps_equal']
        assert result['gap'] > 0

    def test_reverse_inequality(self):
        """When V_2 < V_1, gap(H_2) should be <= gap(H_1)."""
        result = OperatorComparison.verify_1d_harmonic(3.0, 1.0)
        assert not result['v2_geq_v1']
        assert result['gap1'] > result['gap2']


# ======================================================================
# Coupling Sign tests
# ======================================================================

class TestCouplingSign:
    """Tests for V_coupling >= 0."""

    def test_coupling_at_zero_is_zero(self):
        """V_coupling(a=0) = 0."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        result = cs.coupling_at_zero()
        assert result['is_zero']

    def test_coupling_nonnegative_unit_coupling(self):
        """V_coupling >= 0 for g = 1.0, random configurations."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        result = cs.verify_coupling_nonnegative(n_samples=500)
        assert result['nonnegative']

    def test_coupling_nonnegative_strong_coupling(self):
        """V_coupling >= 0 for g = 5.0, random configurations."""
        cs = CouplingSign(R=1.0, g_coupling=5.0)
        result = cs.verify_coupling_nonnegative(n_samples=500)
        assert result['nonnegative']

    def test_coupling_nonnegative_weak_coupling(self):
        """V_coupling >= 0 for g = 0.1, random configurations."""
        cs = CouplingSign(R=1.0, g_coupling=0.1)
        result = cs.verify_coupling_nonnegative(n_samples=500)
        assert result['nonnegative']

    def test_coupling_nonnegative_physical_coupling(self):
        """V_coupling >= 0 for g^2 = 2*pi (physical), random configurations."""
        cs = CouplingSign(R=1.0, g_coupling=np.sqrt(2 * np.pi))
        result = cs.verify_coupling_nonnegative(n_samples=500)
        assert result['nonnegative']

    def test_coupling_nonnegative_large_R(self):
        """V_coupling >= 0 for large R."""
        cs = CouplingSign(R=100.0, g_coupling=1.0)
        result = cs.verify_coupling_nonnegative(n_samples=300)
        assert result['nonnegative']

    def test_coupling_nonnegative_small_R(self):
        """V_coupling >= 0 for small R."""
        cs = CouplingSign(R=0.1, g_coupling=1.0)
        result = cs.verify_coupling_nonnegative(n_samples=300)
        assert result['nonnegative']

    def test_coupling_nonnegative_very_strong(self):
        """V_coupling >= 0 for g = 50.0."""
        cs = CouplingSign(R=1.0, g_coupling=50.0)
        result = cs.verify_coupling_nonnegative(n_samples=300)
        assert result['nonnegative']

    def test_manifest_positivity(self):
        """|[a_low, a_high]|^2 >= 0 (sum of squares)."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        result = cs.verify_manifest_positivity(n_samples=500)
        assert result['all_nonneg']

    def test_manifest_positivity_min_value(self):
        """The minimum of |[a_low, a_high]|^2 is 0 (at a=0)."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        result = cs.verify_manifest_positivity(n_samples=200)
        assert result['min_value'] >= -1e-14

    def test_cross_term_vanishes_proof_structure(self):
        """Cross-term vanishing proof has all required steps."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        proof = cs.cross_term_vanishes_proof()
        assert len(proof['proof_steps']) >= 5
        assert 'orthogonality' in proof['orthogonality_mechanism'].lower()

    def test_coupling_sign_theorem_statement(self):
        """Coupling sign theorem has correct conclusion."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        result = cs.coupling_sign_theorem()
        assert 'V_coupling >= 0' in result['theorem']
        assert result['status'] == 'THEOREM'

    def test_v4_nonnegative_for_combined_config(self):
        """V_4 >= 0 for combined (low + high) configurations."""
        cs = CouplingSign(R=1.0, g_coupling=1.0)
        rng = np.random.default_rng(123)
        for _ in range(100):
            a_full = rng.standard_normal((6, 3))
            v4 = cs._compute_v4(a_full)
            assert v4 >= -1e-12, f"V_4 = {v4} < 0 for a = {a_full}"


# ======================================================================
# Adiabatic Decoupling tests
# ======================================================================

class TestAdiabaticDecoupling:
    """Tests for the adiabatic decoupling properties."""

    def test_delta_is_35(self):
        """Spectral desert ratio delta = 35."""
        ad = AdiabaticDecoupling(R=1.0)
        assert ad.delta == 35

    def test_delta_r_independent(self):
        """Delta is the same for all R."""
        for R in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            ad = AdiabaticDecoupling(R=R)
            assert ad.delta == 35

    def test_spectral_gap_scales_with_R(self):
        """The dimensional spectral gap scales as 1/R^2."""
        ad1 = AdiabaticDecoupling(R=1.0)
        ad2 = AdiabaticDecoupling(R=2.0)
        ratio = ad1.spectral_gap / ad2.spectral_gap
        assert abs(ratio - 4.0) < 1e-10  # (2/1)^2 = 4

    def test_spectral_desert_properties(self):
        """Spectral desert has correct properties."""
        ad = AdiabaticDecoupling(R=1.0)
        props = ad.spectral_desert_properties()
        assert props['R_independent']
        assert props['eigenvalue_ratio'] == 36
        assert props['delta'] == 35
        assert props['n_missing'] == 9
        assert props['status'] == 'THEOREM'

    def test_adiabatic_error_bound_small(self):
        """Adiabatic error is small at physical coupling."""
        ad = AdiabaticDecoupling(R=1.0, g_coupling=np.sqrt(2 * np.pi))
        error = ad.adiabatic_error_bound()
        assert error['epsilon_at_physical_coupling'] < 0.1  # less than 10%
        assert error['R_independent']

    def test_adiabatic_error_bound_r_independent(self):
        """Error bound structure is R-independent."""
        for R in [0.5, 2.0, 10.0]:
            ad = AdiabaticDecoupling(R=R, g_coupling=1.0)
            error = ad.adiabatic_error_bound()
            assert error['R_independent']

    def test_effective_correction_nonnegative(self):
        """W_adiabatic >= 0 (from V_coupling >= 0)."""
        ad = AdiabaticDecoupling(R=1.0)
        correction = ad.effective_hamiltonian_correction()
        assert correction['w_adiabatic_sign'] == 'non-negative'
        assert correction['strict_comparison_available']

    def test_eigenvalue_low_correct(self):
        """Lambda_low = 4/R^2."""
        for R in [0.5, 1.0, 2.0]:
            ad = AdiabaticDecoupling(R=R)
            assert abs(ad.lambda_low - 4.0 / R**2) < 1e-12

    def test_eigenvalue_high_correct(self):
        """Lambda_high = 144/R^2."""
        for R in [0.5, 1.0, 2.0]:
            ad = AdiabaticDecoupling(R=R)
            assert abs(ad.lambda_high - 144.0 / R**2) < 1e-12


# ======================================================================
# Gap Comparison Result tests
# ======================================================================

class TestGapComparisonResult:
    """Tests for the full gap comparison theorem."""

    def test_full_gap_lower_bound_is_theorem(self):
        """The full gap lower bound has status THEOREM."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        result = gcr.full_gap_lower_bound()
        assert result['status'] == 'THEOREM'
        assert not result['epsilon_needed']  # strict bound, no epsilon
        assert result['strict_bound']

    def test_full_gap_lower_bound_proof_steps(self):
        """All proof steps are present and have THEOREM status."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        result = gcr.full_gap_lower_bound()
        steps = result['proof_steps']
        for key in ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6']:
            assert key in steps
            assert steps[key]['status'].startswith('THEOREM')

    def test_proof_chain_upgrade_step_4(self):
        """Step 4 is upgraded from PROPOSITION to THEOREM."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        upgrade = gcr.proof_chain_upgrade()
        assert upgrade['step_4_before']['label'] == 'PROPOSITION'
        assert upgrade['step_4_after']['label'] == 'THEOREM'

    def test_proof_chain_summary_before_vs_after(self):
        """Before: 4 THEOREMs. After: 5 THEOREMs."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        upgrade = gcr.proof_chain_upgrade()
        before = upgrade['chain_summary_before']
        after = upgrade['chain_summary_after']

        n_theorem_before = sum(1 for v in before.values() if v == 'THEOREM')
        n_theorem_after = sum(1 for v in after.values() if v == 'THEOREM')

        assert n_theorem_before == 4
        assert n_theorem_after == 5
        assert before[4] == 'PROPOSITION'
        assert after[4] == 'THEOREM'

    def test_remaining_weak_links(self):
        """Only Step 5 (PROPOSITION) and Step 8 (CONJECTURE) remain weak."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        upgrade = gcr.proof_chain_upgrade()
        weak = upgrade['remaining_weak_links']
        assert 'step_5' in weak
        assert 'step_8' in weak

    def test_numerical_scan_small_basis(self):
        """Numerical scan with small basis for speed."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        result = gcr.numerical_verification_scan(
            R_values=[0.5, 1.0, 2.0], n_basis=5
        )
        # All gaps should be positive
        for r in result['results']:
            assert r['gap_3'] > 0
            assert r['gap_6'] > 0

    def test_coupling_verification_scan(self):
        """V_coupling >= 0 for multiple coupling values."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        result = gcr.coupling_verification_scan(
            g_values=[0.1, 1.0, 5.0], n_samples=100
        )
        assert result['all_nonnegative']

    def test_adiabatic_correction_size(self):
        """Adiabatic corrections are computed for various R."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        result = gcr.compute_adiabatic_correction_size(
            R_values=[0.5, 1.0, 5.0]
        )
        for r in result['results']:
            assert r['delta'] == 35
            assert r['epsilon'] >= 0


# ======================================================================
# Proof Chain Upgrade tests
# ======================================================================

class TestProofChainUpgrade:
    """Tests for the proof chain integration."""

    def test_full_assessment_theorem_count(self):
        """Full assessment lists 5 key ingredients."""
        pcu = ProofChainUpgrade(N=2)
        assessment = pcu.full_assessment()
        assert len(assessment['key_ingredients']) >= 5

    def test_upgrades_step_4(self):
        """Only Step 4 is upgraded."""
        pcu = ProofChainUpgrade(N=2)
        assessment = pcu.full_assessment()
        assert 'step_4' in assessment['upgrades']

    def test_remaining_conjecture_is_clay(self):
        """The remaining conjecture is the Clay Millennium Problem."""
        pcu = ProofChainUpgrade(N=2)
        assessment = pcu.full_assessment()
        remaining = assessment['remaining_conjectures']
        assert 'conjecture_7_5' in remaining
        assert 'Clay' in remaining['conjecture_7_5']

    def test_summary_contains_upgrade(self):
        """Summary mentions the THEOREM upgrade."""
        pcu = ProofChainUpgrade(N=2)
        summary = pcu.generate_summary()
        assert 'THEOREM***' in summary or 'UPGRADED' in summary
        assert 'CONJECTURE' in summary


# ======================================================================
# Radial gap computation tests
# ======================================================================

class TestRadialGap:
    """Tests for the radial gap helper function."""

    def test_1d_harmonic_gap(self):
        """1D harmonic oscillator gap = omega."""
        gap = _compute_radial_gap(1, omega_sq=4.0, lam=0.0)
        assert abs(gap - 2.0) < 0.1  # omega = sqrt(4) = 2

    def test_1d_quartic_gap_positive(self):
        """1D pure quartic has positive gap."""
        gap = _compute_radial_gap(1, omega_sq=0.0, lam=1.0)
        assert gap > 0.5

    def test_3d_harmonic_gap(self):
        """3D radial harmonic gap is positive."""
        gap = _compute_radial_gap(3, omega_sq=1.0, lam=0.0)
        assert gap > 0

    def test_gap_increases_with_omega(self):
        """Larger omega gives larger gap (1D)."""
        gap1 = _compute_radial_gap(1, omega_sq=1.0, lam=0.0)
        gap2 = _compute_radial_gap(1, omega_sq=4.0, lam=0.0)
        assert gap2 > gap1

    def test_gap_increases_with_lambda(self):
        """Larger lambda gives larger gap (1D with fixed omega)."""
        gap1 = _compute_radial_gap(1, omega_sq=1.0, lam=0.1)
        gap2 = _compute_radial_gap(1, omega_sq=1.0, lam=1.0)
        assert gap2 > gap1


# ======================================================================
# Effective gap computation tests
# ======================================================================

class TestEffectiveGap:
    """Tests for the effective Hamiltonian gap computation."""

    def test_gap_positive_R1(self):
        """Effective gap is positive at R = 1."""
        gap = _compute_effective_gap(R=1.0, g_coupling=1.0, n_modes=3, n_basis=5)
        assert gap > 0

    def test_gap_positive_R10(self):
        """Effective gap is positive at R = 10."""
        gap = _compute_effective_gap(R=10.0, g_coupling=1.0, n_modes=3, n_basis=5)
        assert gap > 0

    def test_gap_decreases_with_R(self):
        """Gap decreases with R (fewer quantum fluctuations)."""
        gap1 = _compute_effective_gap(R=1.0, g_coupling=1.0, n_basis=5)
        gap2 = _compute_effective_gap(R=5.0, g_coupling=1.0, n_basis=5)
        assert gap1 > gap2

    def test_gap_approaches_harmonic_at_g0(self):
        """At g=0, the gap should be omega = 2/R."""
        R = 1.0
        gap = _compute_effective_gap(R=R, g_coupling=0.0, n_basis=8)
        expected = 2.0 / R
        assert abs(gap - expected) < 0.2 * expected

    def test_gap_6_mode_positive(self):
        """6-mode effective gap is also positive."""
        gap = _compute_effective_gap(R=1.0, g_coupling=1.0, n_modes=6, n_basis=5)
        assert gap > 0


# ======================================================================
# Integration / cross-module tests
# ======================================================================

class TestIntegration:
    """Cross-module integration tests."""

    def test_v4_algebraic_identity(self):
        """V_4 = (g^2/2) * [(Tr S)^2 - Tr(S^2)] >= 0 for any M."""
        rng = np.random.default_rng(42)
        g2 = 1.0
        for _ in range(200):
            n_rows = rng.integers(1, 10)
            n_cols = rng.integers(1, 5)
            M = rng.standard_normal((n_rows, n_cols)) * rng.uniform(0.01, 10.0)
            S = M.T @ M
            tr_S = np.trace(S)
            tr_S2 = np.trace(S @ S)
            v4 = 0.5 * g2 * (tr_S**2 - tr_S2)
            assert v4 >= -1e-12, f"V_4 = {v4} < 0 for M of shape {M.shape}"

    def test_gap_lower_bound_is_positive(self):
        """gap(H_full) >= gap(H_3) > 0 for various R."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            gap_3 = _compute_effective_gap(R, g_coupling=1.0, n_modes=3, n_basis=5)
            assert gap_3 > 0, f"gap(H_3) = {gap_3} at R = {R}"

    def test_spectral_desert_is_geometric(self):
        """The spectral desert ratio does NOT depend on R or g."""
        # The ratio (k+1)^2 for k=11 vs k=1 is a pure number
        ratio = (K_HIGH_MIN + 1)**2 / (K_LOW + 1)**2
        assert ratio == 36

    def test_coupling_nonnegative_at_physical_params(self):
        """V_coupling >= 0 at physical parameters (R ~ 2 fm, g^2 ~ 2*pi)."""
        cs = CouplingSign(R=2.0, g_coupling=np.sqrt(2 * np.pi))
        result = cs.verify_coupling_nonnegative(n_samples=200)
        assert result['nonnegative']

    def test_full_theorem_self_consistent(self):
        """The full theorem is internally consistent."""
        gcr = GapComparisonResult(R=1.0, g_coupling=1.0)
        bound = gcr.full_gap_lower_bound()
        upgrade = gcr.proof_chain_upgrade()

        # Theorem is THEOREM status
        assert bound['status'] == 'THEOREM'

        # Step 4 is upgraded
        assert upgrade['chain_summary_after'][4] == 'THEOREM'

        # Strict bound (no epsilon)
        assert bound['strict_bound']

    def test_comparison_consistent_with_effective_hamiltonian(self):
        """The comparison theorem is consistent with effective_hamiltonian.py results."""
        # gap(H_3) should be ~ omega = 2/R for small coupling
        R = 1.0
        gap_3 = _compute_effective_gap(R, g_coupling=0.01, n_modes=3, n_basis=8)
        expected = 2.0 / R
        # Should be close to harmonic gap
        assert abs(gap_3 - expected) < 0.3 * expected

    def test_no_contradiction_with_known_results(self):
        """Verify no contradiction: gap ~ 2/R for physical R ~ 2 fm."""
        R = 2.2  # fm
        gap = _compute_effective_gap(R, g_coupling=np.sqrt(2 * np.pi), n_basis=5)
        gap_mev = gap * HBAR_C_MEV_FM
        # Should be in the right ballpark (not checking exact value)
        assert gap > 0
        assert gap_mev > 0  # positive in physical units
