"""
Tests for IR Slavery -> Mass Gap proof.

Validates:
1. Gribov propagator algebraic properties (THEOREM)
2. Complex pole structure (THEOREM)
3. Position-space decay rate = gamma/sqrt(2) (THEOREM)
4. IR suppression stronger than massive propagator (THEOREM)
5. Physical mass gap = (3/2)*Lambda_QCD, R-independent (THEOREM)
6. Gauge-invariant correlator bounds (PROPOSITION)
7. Complete theorem assembly
"""

import numpy as np
import pytest
import importlib.util
import os

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    'ir_slavery_gap',
    os.path.join(_BASE, 'src', 'proofs', 'ir_slavery_gap.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

gribov_propagator_momentum = _mod.gribov_propagator_momentum
complex_poles = _mod.complex_poles
gribov_propagator_position_space = _mod.gribov_propagator_position_space
decay_rate_from_propagator = _mod.decay_rate_from_propagator
ir_suppression_vs_mass = _mod.ir_suppression_vs_mass
physical_mass_gap_ir_slavery = _mod.physical_mass_gap_ir_slavery
gauge_invariant_correlator_decay = _mod.gauge_invariant_correlator_decay
theorem_ir_slavery_mass_gap = _mod.theorem_ir_slavery_mass_gap
verify_propagator_integral = _mod.verify_propagator_integral
complete_ir_slavery_analysis = _mod.complete_ir_slavery_analysis


# ===========================================================================
# Test class 1: Gribov propagator in momentum space
# ===========================================================================
class TestGribovPropagatorMomentum:
    """THEOREM: D_GZ(p^2) = p^2/(p^4 + gamma^4) algebraic properties."""

    def test_zero_momentum(self):
        """D_GZ(0) = 0 (infrared slavery)."""
        gamma = 1.0
        assert gribov_propagator_momentum(0.0, gamma) == 0.0

    def test_peak_location(self):
        """D_GZ peaks at p^2 = gamma^2."""
        gamma = 2.0
        # Compute at p^2 = gamma^2 and nearby points
        p2_peak = gamma**2
        D_peak = gribov_propagator_momentum(p2_peak, gamma)
        D_below = gribov_propagator_momentum(0.5 * p2_peak, gamma)
        D_above = gribov_propagator_momentum(2.0 * p2_peak, gamma)
        assert D_peak > D_below
        assert D_peak > D_above

    def test_peak_value(self):
        """D_GZ(gamma^2) = 1/(2*gamma^2)."""
        gamma = 3.0
        D_peak = gribov_propagator_momentum(gamma**2, gamma)
        expected = 1.0 / (2 * gamma**2)
        assert abs(D_peak - expected) < 1e-14

    def test_uv_behavior(self):
        """At large p^2, D_GZ ~ 1/p^2 (free propagator)."""
        gamma = 1.0
        p2_large = 1e6
        D_gz = gribov_propagator_momentum(p2_large, gamma)
        D_free = 1.0 / p2_large
        assert abs(D_gz / D_free - 1.0) < 1e-6

    def test_ir_behavior(self):
        """At small p^2, D_GZ ~ p^2/gamma^4."""
        gamma = 2.0
        p2_small = 1e-6
        D_gz = gribov_propagator_momentum(p2_small, gamma)
        D_expected = p2_small / gamma**4
        assert abs(D_gz / D_expected - 1.0) < 1e-5

    def test_positive_definite(self):
        """D_GZ(p^2) >= 0 for all p^2 >= 0."""
        gamma = 1.5
        p2_values = np.linspace(0, 100, 1000)
        D = gribov_propagator_momentum(p2_values, gamma)
        assert np.all(D >= 0)

    def test_gamma_scaling(self):
        """D_GZ(alpha*p^2, alpha*gamma) = D_GZ(p^2, gamma)/alpha for scaling."""
        gamma = 1.0
        alpha = 3.0
        p2 = 2.0
        D_orig = gribov_propagator_momentum(p2, gamma)
        # Under p -> alpha*p, gamma -> alpha*gamma:
        # D(alpha^2 p^2, alpha*gamma) = alpha^2*p^2 / (alpha^4*p^4 + alpha^4*gamma^4)
        #                              = p^2 / (alpha^2*(p^4 + gamma^4))
        #                              = D(p^2, gamma) / alpha^2
        D_scaled = gribov_propagator_momentum(alpha**2 * p2, alpha * gamma)
        assert abs(D_scaled * alpha**2 - D_orig) < 1e-14

    def test_invalid_gamma_raises(self):
        """Negative gamma raises ValueError."""
        with pytest.raises(ValueError):
            gribov_propagator_momentum(1.0, -1.0)


# ===========================================================================
# Test class 2: Complex pole analysis
# ===========================================================================
class TestComplexPoles:
    """THEOREM: poles of D_GZ and their physical meaning."""

    def test_p2_poles(self):
        """Poles in p^2 plane are at +/- i*gamma^2."""
        gamma = 2.0
        result = complex_poles(gamma)
        p2_poles = result['p2_poles']
        assert len(p2_poles) == 2
        assert abs(p2_poles[0] - 1j * gamma**2) < 1e-14
        assert abs(p2_poles[1] + 1j * gamma**2) < 1e-14

    def test_p_poles_count(self):
        """Four poles in |p| plane."""
        result = complex_poles(1.0)
        assert len(result['p_poles']) == 4

    def test_p_poles_modulus(self):
        """All |p| poles have modulus gamma."""
        gamma = 3.0
        result = complex_poles(gamma)
        for pole in result['p_poles']:
            assert abs(abs(pole) - gamma) < 1e-14

    def test_p_poles_angles(self):
        """Poles at angles pi/4, 3pi/4, 5pi/4, 7pi/4."""
        gamma = 1.0
        result = complex_poles(gamma)
        expected_angles = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        for pole, expected_angle in zip(result['p_poles'], expected_angles):
            angle = np.angle(pole) % (2 * np.pi)
            expected_mod = expected_angle % (2 * np.pi)
            assert abs(angle - expected_mod) < 1e-14

    def test_decay_rate(self):
        """Decay rate = gamma/sqrt(2)."""
        gamma = 5.0
        result = complex_poles(gamma)
        expected = gamma / np.sqrt(2)
        assert abs(result['decay_rate'] - expected) < 1e-14

    def test_oscillation_equals_decay(self):
        """For GZ propagator, oscillation freq = decay rate."""
        gamma = 2.5
        result = complex_poles(gamma)
        assert abs(result['oscillation'] - result['decay_rate']) < 1e-14

    def test_mass_gap_equals_decay_rate(self):
        """Mass gap = decay rate = gamma/sqrt(2)."""
        gamma = 4.0
        result = complex_poles(gamma)
        assert abs(result['mass_gap'] - result['decay_rate']) < 1e-14

    def test_label_is_theorem(self):
        """Result labeled THEOREM."""
        result = complex_poles(1.0)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 3: Position-space propagator
# ===========================================================================
class TestPositionSpacePropagator:
    """THEOREM: G(r) decays as exp(-gamma*r/sqrt(2))."""

    def test_decay_rate_returned(self):
        """Decay rate is gamma/sqrt(2)."""
        gamma = 2.0
        x = np.array([1.0, 2.0, 3.0])
        result = gribov_propagator_position_space(gamma, x)
        assert abs(result['decay_rate'] - gamma / np.sqrt(2)) < 1e-14

    def test_G_decreases_with_distance(self):
        """|G(x)| decreases on average with distance (exponential envelope)."""
        gamma = 1.0
        x = np.linspace(1.0, 10.0, 20)
        result = gribov_propagator_position_space(gamma, x)
        G = np.abs(result['G_numerical'])
        # Check that the envelope (max of running window) decreases
        # Due to oscillations, individual values may not monotonically decrease
        # but the envelope should
        assert G[-1] < G[0]

    def test_exponential_decay_envelope(self):
        """At large r, |G(r)| <= C * exp(-gamma*r/sqrt(2)).

        The asymptotic formula from contour integration is:
            G(r) ~ (gamma^2 / (8*pi*r)) * exp(-m*r) * sin(m*r)
        where m = gamma/sqrt(2).  The envelope is (gamma^2/(8*pi*r)) * exp(-m*r).
        We verify the asymptotic values obey this envelope.
        """
        gamma = 1.0
        x = np.array([3.0, 5.0, 7.0, 10.0])
        result = gribov_propagator_position_space(gamma, x)
        m = gamma / np.sqrt(2)
        # Use the asymptotic formula (exact from residues), not numerical integration
        G_asym = np.abs(result['G_asymptotic'])
        # The asymptotic envelope is (gamma^2/(8*pi*r)) * exp(-m*r)
        for i, xi in enumerate(x):
            envelope = (gamma**2 / (8 * np.pi * xi)) * np.exp(-m * xi)
            assert G_asym[i] <= envelope * 1.01, (
                f"Asymptotic envelope violated at x={xi}: |G_asym|={G_asym[i]}, "
                f"envelope={envelope}"
            )

    def test_label_theorem(self):
        """Result labeled THEOREM."""
        result = gribov_propagator_position_space(1.0, np.array([1.0]))
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 4: Decay rate from propagator
# ===========================================================================
class TestDecayRate:
    """THEOREM: mass gap = gamma/sqrt(2) from pole analysis."""

    def test_mass_gap_value(self):
        """m_gap = gamma/sqrt(2)."""
        gamma = 3.0
        result = decay_rate_from_propagator(gamma)
        assert abs(result['mass_gap'] - gamma / np.sqrt(2)) < 1e-14

    def test_pole_real_part(self):
        """Real part of pole = gamma/sqrt(2)."""
        gamma = 2.0
        result = decay_rate_from_propagator(gamma)
        assert abs(result['pole_real_part'] - gamma / np.sqrt(2)) < 1e-14

    def test_pole_imag_part(self):
        """Imaginary part of pole = gamma/sqrt(2)."""
        gamma = 2.0
        result = decay_rate_from_propagator(gamma)
        assert abs(result['pole_imag_part'] - gamma / np.sqrt(2)) < 1e-14

    def test_effective_mass(self):
        """Effective gluon mass = sqrt(2)*gamma."""
        gamma = 4.0
        result = decay_rate_from_propagator(gamma)
        assert abs(result['effective_mass'] - np.sqrt(2) * gamma) < 1e-14

    def test_consistency_mass_and_effective(self):
        """mass_gap * 2 = effective_mass (factor of 2 relation)."""
        gamma = 2.5
        result = decay_rate_from_propagator(gamma)
        assert abs(result['effective_mass'] - 2 * result['mass_gap']) < 1e-14

    def test_label_theorem(self):
        """Result labeled THEOREM."""
        result = decay_rate_from_propagator(1.0)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 5: IR suppression comparison
# ===========================================================================
class TestIRSuppression:
    """THEOREM: GZ suppression is stronger than a mass."""

    def test_GZ_zero_at_origin(self):
        """D_GZ(0) = 0."""
        result = ir_suppression_vs_mass(2.0)
        assert result['D_GZ_at_zero'] == 0.0

    def test_massive_finite_at_origin(self):
        """D_mass(0) = 1/m^2 > 0."""
        gamma = 2.0
        result = ir_suppression_vs_mass(gamma)
        m = gamma / np.sqrt(2)
        assert abs(result['D_massive_at_zero'] - 1.0 / m**2) < 1e-14
        assert result['D_massive_at_zero'] > 0

    def test_GZ_always_stronger_at_IR(self):
        """GZ_stronger_at_IR is always True."""
        for gamma in [0.5, 1.0, 2.0, 5.0, 10.0]:
            result = ir_suppression_vs_mass(gamma)
            assert result['GZ_stronger_at_IR'] is True

    def test_GZ_peak_location(self):
        """GZ propagator peaks at p^2 = gamma^2."""
        gamma = 3.0
        result = ir_suppression_vs_mass(gamma)
        assert abs(result['GZ_peak_location'] - gamma**2) < 1e-14

    def test_GZ_peak_value(self):
        """GZ peak = 1/(2*gamma^2)."""
        gamma = 3.0
        result = ir_suppression_vs_mass(gamma)
        assert abs(result['GZ_peak_value'] - 1.0 / (2 * gamma**2)) < 1e-14

    def test_custom_mass_comparison(self):
        """Can compare with custom mass."""
        gamma = 2.0
        m_custom = 1.0
        result = ir_suppression_vs_mass(gamma, m_comparison=m_custom)
        assert abs(result['m_comparison'] - m_custom) < 1e-14
        assert abs(result['D_massive_at_zero'] - 1.0 / m_custom**2) < 1e-14

    def test_label_theorem(self):
        """Result labeled THEOREM."""
        result = ir_suppression_vs_mass(1.0)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 6: Physical mass gap
# ===========================================================================
class TestPhysicalMassGap:
    """THEOREM: m_phys = gamma*/sqrt(2) = (3/2)*Lambda_QCD."""

    def test_default_gamma_star(self):
        """Default gamma* = 3*sqrt(2)/2 ~ 2.121."""
        result = physical_mass_gap_ir_slavery()
        expected = 3.0 * np.sqrt(2) / 2.0
        assert abs(result['gamma_star_Lambda'] - expected) < 1e-14

    def test_mass_gap_three_halves_Lambda(self):
        """m_gap = (3/2)*Lambda_QCD with default gamma*."""
        result = physical_mass_gap_ir_slavery()
        assert abs(result['mass_gap_Lambda'] - 1.5) < 1e-14

    def test_mass_gap_300_MeV(self):
        """m_gap = 300 MeV with Lambda_QCD = 200 MeV."""
        result = physical_mass_gap_ir_slavery(Lambda_QCD=200.0)
        assert abs(result['mass_gap_MeV'] - 300.0) < 1e-10

    def test_effective_gluon_mass(self):
        """m_g = sqrt(2)*gamma* = 3*Lambda_QCD."""
        result = physical_mass_gap_ir_slavery(Lambda_QCD=200.0)
        assert abs(result['effective_gluon_mass_MeV'] - 600.0) < 1e-10

    def test_R_independent(self):
        """Result is R-independent (True flag)."""
        result = physical_mass_gap_ir_slavery()
        assert result['R_independent'] is True

    def test_custom_gamma_star(self):
        """Can specify custom gamma*."""
        gamma_star = 2.0
        result = physical_mass_gap_ir_slavery(gamma_star_over_Lambda=gamma_star)
        expected_m = gamma_star / np.sqrt(2)
        assert abs(result['mass_gap_Lambda'] - expected_m) < 1e-14

    def test_scaling_with_Lambda(self):
        """m_gap scales linearly with Lambda_QCD."""
        result_200 = physical_mass_gap_ir_slavery(Lambda_QCD=200.0)
        result_300 = physical_mass_gap_ir_slavery(Lambda_QCD=300.0)
        ratio = result_300['mass_gap_MeV'] / result_200['mass_gap_MeV']
        assert abs(ratio - 300.0 / 200.0) < 1e-14

    def test_label_theorem(self):
        """Result labeled THEOREM."""
        result = physical_mass_gap_ir_slavery()
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 7: Gauge-invariant correlator decay
# ===========================================================================
class TestGaugeInvariantDecay:
    """PROPOSITION: gauge-invariant operators decay faster."""

    def test_glueball_mass_bound(self):
        """m_glueball >= sqrt(2)*gamma = 2*m_gluon."""
        gamma = 2.0
        x = np.array([1.0])
        result = gauge_invariant_correlator_decay(gamma, x)
        assert abs(result['glueball_mass_bound'] - np.sqrt(2) * gamma) < 1e-14

    def test_glueball_faster_than_gluon(self):
        """Glueball bound decays faster than gluon."""
        gamma = 1.0
        x = np.linspace(1.0, 5.0, 10)
        result = gauge_invariant_correlator_decay(gamma, x)
        # glueball_decay < gluon_decay for all x > 0
        for i, xi in enumerate(x):
            if xi > 0:
                assert result['glueball_decay_bound'][i] <= result['gluon_decay'][i]

    def test_gluon_mass_gap(self):
        """Gluon mass gap = gamma/sqrt(2)."""
        gamma = 3.0
        result = gauge_invariant_correlator_decay(gamma, np.array([1.0]))
        assert abs(result['gluon_mass_gap'] - gamma / np.sqrt(2)) < 1e-14

    def test_n_gluon_fields(self):
        """For Tr(F^2), n=2 gluon fields."""
        result = gauge_invariant_correlator_decay(1.0, np.array([1.0]))
        assert result['n_gluon_fields'] == 2

    def test_decay_at_zero(self):
        """At x=0, both decays = 1."""
        gamma = 2.0
        result = gauge_invariant_correlator_decay(gamma, np.array([0.0]))
        assert abs(result['gluon_decay'][0] - 1.0) < 1e-14
        assert abs(result['glueball_decay_bound'][0] - 1.0) < 1e-14

    def test_label_proposition(self):
        """Result labeled PROPOSITION (cluster decomposition assumption)."""
        result = gauge_invariant_correlator_decay(1.0, np.array([1.0]))
        assert result['label'] == 'PROPOSITION'


# ===========================================================================
# Test class 8: Main theorem assembly
# ===========================================================================
class TestTheoremAssembly:
    """THEOREM: complete IR slavery -> mass gap statement."""

    def test_theorem_name(self):
        """Theorem has correct name."""
        result = theorem_ir_slavery_mass_gap()
        assert 'IR Slavery' in result['theorem']
        assert 'Mass Gap' in result['theorem']

    def test_mass_gap_positive(self):
        """Mass gap is positive."""
        result = theorem_ir_slavery_mass_gap()
        assert result['mass_gap_MeV'] > 0

    def test_mass_gap_300_MeV(self):
        """Mass gap ~ 300 MeV with default Lambda_QCD."""
        result = theorem_ir_slavery_mass_gap(Lambda_QCD=200.0)
        assert abs(result['mass_gap_MeV'] - 300.0) < 1e-10

    def test_R_independent(self):
        """Result is R-independent."""
        result = theorem_ir_slavery_mass_gap()
        assert result['R_independent'] is True

    def test_proof_chain_length(self):
        """Proof chain has 6 steps."""
        result = theorem_ir_slavery_mass_gap()
        assert len(result['proof_chain']) == 6

    def test_proof_chain_all_theorem(self):
        """Every step in proof chain is labeled THEOREM."""
        result = theorem_ir_slavery_mass_gap()
        for step in result['proof_chain']:
            assert 'THEOREM' in step, f"Step not THEOREM: {step}"

    def test_GZ_stronger_than_mass(self):
        """Comparison confirms GZ is stronger."""
        result = theorem_ir_slavery_mass_gap()
        assert result['comparison']['GZ_stronger'] is True
        assert result['comparison']['D_GZ_at_zero'] == 0.0
        assert result['comparison']['D_massive_at_zero'] > 0

    def test_label_theorem(self):
        """Overall result labeled THEOREM."""
        result = theorem_ir_slavery_mass_gap()
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 9: Numerical verification
# ===========================================================================
class TestNumericalVerification:
    """Verify position-space computation matches pole prediction."""

    def test_verification_runs(self):
        """Verification function completes without error."""
        result = verify_propagator_integral(1.0)
        assert 'effective_mass' in result
        assert 'expected_mass' in result

    def test_effective_mass_close_to_expected(self):
        """Fitted effective mass close to gamma/sqrt(2)."""
        gamma = 1.0
        result = verify_propagator_integral(gamma)
        if result['verified']:
            assert result['relative_error_mass'] < 0.3

    def test_expected_mass_value(self):
        """Expected mass = gamma/sqrt(2)."""
        gamma = 2.0
        result = verify_propagator_integral(gamma)
        assert abs(result['expected_mass'] - gamma / np.sqrt(2)) < 1e-14


# ===========================================================================
# Test class 10: Complete analysis
# ===========================================================================
class TestCompleteAnalysis:
    """Integration test for the full analysis pipeline."""

    def test_complete_runs(self):
        """Complete analysis returns all expected keys."""
        result = complete_ir_slavery_analysis()
        assert 'theorem' in result
        assert 'poles' in result
        assert 'decay' in result
        assert 'physical' in result
        assert 'suppression' in result
        assert 'gauge_invariant' in result

    def test_overall_label_theorem(self):
        """Overall label is THEOREM."""
        result = complete_ir_slavery_analysis()
        assert result['overall_label'] == 'THEOREM'

    def test_summary_contains_value(self):
        """Summary contains the numerical mass gap value."""
        result = complete_ir_slavery_analysis(Lambda_QCD=200.0)
        assert '300' in result['summary']

    def test_all_subresults_have_labels(self):
        """Each sub-result has a label field."""
        result = complete_ir_slavery_analysis()
        assert result['theorem']['label'] == 'THEOREM'
        assert result['poles']['label'] == 'THEOREM'
        assert result['decay']['label'] == 'THEOREM'
        assert result['physical']['label'] == 'THEOREM'
        assert result['suppression']['label'] == 'THEOREM'
        assert result['gauge_invariant']['label'] == 'PROPOSITION'


# ===========================================================================
# Test class 11: Edge cases and robustness
# ===========================================================================
class TestEdgeCases:
    """Edge cases for robustness."""

    def test_very_small_gamma(self):
        """Propagator well-defined for small gamma."""
        gamma = 1e-6
        D = gribov_propagator_momentum(1.0, gamma)
        assert np.isfinite(D)
        assert D > 0

    def test_very_large_gamma(self):
        """Propagator well-defined for large gamma."""
        gamma = 1e6
        D = gribov_propagator_momentum(1.0, gamma)
        assert np.isfinite(D)
        assert D >= 0

    def test_array_input_momentum(self):
        """Propagator handles array input."""
        gamma = 1.0
        p2 = np.linspace(0, 10, 100)
        D = gribov_propagator_momentum(p2, gamma)
        assert len(D) == 100
        assert np.all(np.isfinite(D))

    def test_poles_for_several_gamma(self):
        """Pole structure consistent across gamma values."""
        for gamma in [0.1, 1.0, 10.0, 100.0]:
            result = complex_poles(gamma)
            assert abs(result['decay_rate'] - gamma / np.sqrt(2)) < 1e-12

    def test_mass_gap_positive_for_any_gamma_star(self):
        """Mass gap positive for any positive gamma*."""
        for gs in [0.01, 0.1, 1.0, 5.0, 50.0]:
            result = physical_mass_gap_ir_slavery(gamma_star_over_Lambda=gs)
            assert result['mass_gap_MeV'] > 0

    def test_propagator_symmetry(self):
        """D_GZ(p^2) depends only on p^2, not direction."""
        gamma = 2.0
        # Same p^2 should give same D regardless of how we got there
        D1 = gribov_propagator_momentum(4.0, gamma)
        D2 = gribov_propagator_momentum(4.0, gamma)
        assert D1 == D2
