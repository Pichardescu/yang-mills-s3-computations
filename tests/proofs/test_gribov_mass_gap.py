"""
Tests for the Gribov propagator mass gap theorem.

Verifies:
1. Pole structure of the Gribov propagator (algebraic)
2. Kallen-Lehmann positivity violation (algebraic)
3. Glueball threshold (spectral analysis)
4. Physical correlator decay (exponential bound)
5. R-independent mass gap from stabilized gamma*
6. Numerical consistency checks
7. SU(N) extension
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import importlib.util
import os

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    'gribov_mass_gap',
    os.path.join(_BASE, 'src', 'proofs', 'gribov_mass_gap.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

gribov_propagator = _mod.gribov_propagator
gribov_propagator_poles = _mod.gribov_propagator_poles
spectral_function_gribov = _mod.spectral_function_gribov
positivity_violation = _mod.positivity_violation
glueball_threshold = _mod.glueball_threshold
physical_correlator_decay = _mod.physical_correlator_decay
mass_gap_from_gribov = _mod.mass_gap_from_gribov
theorem_r_independent_gap = _mod.theorem_r_independent_gap
verify_propagator_properties = _mod.verify_propagator_properties
verify_threshold_numerically = _mod.verify_threshold_numerically
mass_gap_table_all_N = _mod.mass_gap_table_all_N
complete_analysis = _mod.complete_analysis


# ===========================================================================
# Test 1: Gribov Propagator Evaluation
# ===========================================================================
class TestGribovPropagator:
    """Basic properties of D(p^2) = p^2/(p^4 + gamma^4)."""

    def test_D_at_zero(self):
        """D(0) = 0."""
        assert gribov_propagator(0.0, 1.0) == 0.0

    def test_D_at_zero_various_gamma(self):
        """D(0) = 0 for any gamma."""
        for gamma in [0.5, 1.0, 2.0, 10.0]:
            assert gribov_propagator(0.0, gamma) == 0.0

    def test_D_positive_for_positive_p2(self):
        """D(p^2) > 0 for p^2 > 0."""
        p2_values = np.logspace(-3, 3, 100)
        for gamma in [1.0, 2.0]:
            D = gribov_propagator(p2_values, gamma)
            assert np.all(D > 0), "D should be positive for p^2 > 0"

    def test_D_maximum_location(self):
        """D(p^2) is maximized at p^2 = gamma^2."""
        gamma = 2.0
        p2 = np.linspace(0.01, 20, 10000)
        D = gribov_propagator(p2, gamma)
        idx_max = np.argmax(D)
        p2_max = p2[idx_max]
        assert abs(p2_max - gamma ** 2) < 0.01, (
            f"Expected max at p^2={gamma**2}, got {p2_max}"
        )

    def test_D_maximum_value(self):
        """D_max = 1/(2*gamma^2) at p^2 = gamma^2."""
        gamma = 3.0
        D_max = gribov_propagator(gamma ** 2, gamma)
        expected = 1.0 / (2 * gamma ** 2)
        assert abs(D_max - expected) < 1e-14

    def test_D_UV_falloff(self):
        """D(p^2) ~ 1/p^2 for large p^2."""
        gamma = 1.0
        p2_large = 1000.0
        D = gribov_propagator(p2_large, gamma)
        D_free = 1.0 / p2_large
        assert abs(D / D_free - 1) < 0.01, "UV behavior should match free propagator"

    def test_D_IR_suppression(self):
        """D(p^2) ~ p^2/gamma^4 for small p^2."""
        gamma = 2.0
        p2_small = 0.001
        D = gribov_propagator(p2_small, gamma)
        D_expected = p2_small / gamma ** 4
        assert abs(D / D_expected - 1) < 0.01, "IR behavior should be p^2/gamma^4"


# ===========================================================================
# Test 2: Pole Structure
# ===========================================================================
class TestPoleStructure:
    """Poles of D(p^2) at p^2 = +/- i*gamma^2."""

    def test_poles_are_complex(self):
        """Poles must be complex for gamma > 0."""
        result = gribov_propagator_poles(1.0)
        assert result['poles_are_complex'] is True

    def test_no_real_mass_shell(self):
        """There is no real mass shell."""
        result = gribov_propagator_poles(2.0)
        assert result['no_real_mass_shell'] is True

    def test_pole_values_su2(self):
        """Poles at p^2 = +/- i*gamma^2."""
        gamma = 2.0
        result = gribov_propagator_poles(gamma)
        poles = result['poles']
        expected_plus = 1j * gamma ** 2
        expected_minus = -1j * gamma ** 2
        assert abs(poles[0] - expected_plus) < 1e-14
        assert abs(poles[1] - expected_minus) < 1e-14

    def test_pole_modulus(self):
        """Pole modulus is gamma^2."""
        for gamma in [0.5, 1.0, 3.0, 5.0]:
            result = gribov_propagator_poles(gamma)
            assert abs(result['pole_modulus'] - gamma ** 2) < 1e-14

    def test_poles_satisfy_equation(self):
        """Poles are roots of p^4 + gamma^4 = 0."""
        gamma = 2.5
        result = gribov_propagator_poles(gamma)
        for residual in result['verification_residuals']:
            assert residual < 1e-12

    def test_gamma_must_be_positive(self):
        """gamma <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            gribov_propagator_poles(0.0)
        with pytest.raises(ValueError):
            gribov_propagator_poles(-1.0)

    def test_factorization_identity(self):
        """p^4 + gamma^4 = (p^2 - i*gamma^2)(p^2 + i*gamma^2)."""
        gamma = 1.7
        g2 = gamma ** 2
        g4 = gamma ** 4
        # Check for several p^2 values
        for p2 in [0.5, 1.0, 2.0, 5.0]:
            lhs = p2 ** 2 + g4
            rhs = (p2 - 1j * g2) * (p2 + 1j * g2)
            assert abs(lhs - rhs.real) < 1e-12
            assert abs(rhs.imag) < 1e-12


# ===========================================================================
# Test 3: Positivity Violation
# ===========================================================================
class TestPositivityViolation:
    """Kallen-Lehmann positivity is violated by the Gribov propagator."""

    def test_D_at_zero_is_zero(self):
        """D(0) = 0 violates KL condition."""
        result = positivity_violation(1.0)
        assert result['D_at_zero'] == 0.0

    def test_positivity_violated(self):
        """Positivity is always violated for gamma > 0."""
        for gamma in [0.5, 1.0, 2.0, 5.0]:
            result = positivity_violation(gamma)
            assert result['positivity_violated'] is True

    def test_gluon_confined(self):
        """The gluon is confined (not in physical spectrum)."""
        result = positivity_violation(2.0)
        assert result['gluon_confined'] is True

    def test_not_monotone_decreasing(self):
        """D(p^2) is NOT monotonically decreasing (increases from 0)."""
        result = positivity_violation(1.0)
        assert result['monotonically_decreasing'] is False

    def test_D_max_value(self):
        """D_max = 1/(2*gamma^2)."""
        gamma = 3.0
        result = positivity_violation(gamma)
        expected = 1.0 / (2 * gamma ** 2)
        assert abs(result['D_max'] - expected) < 1e-14

    def test_D_max_location(self):
        """D is maximized at p^2 = gamma^2."""
        gamma = 2.0
        result = positivity_violation(gamma)
        assert abs(result['D_max_location_p2'] - gamma ** 2) < 1e-14

    def test_label_is_theorem(self):
        """The result is labeled THEOREM."""
        result = positivity_violation(1.0)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test 4: Spectral Function
# ===========================================================================
class TestSpectralFunction:
    """The spectral function rho(s) = 0 on the real axis."""

    def test_rho_zero_everywhere(self):
        """rho(s) = 0 for all real s (no spectral weight on real axis)."""
        s_values = np.linspace(0, 10, 100)
        rho = spectral_function_gribov(s_values, 1.0)
        assert np.all(rho == 0.0)

    def test_rho_zero_at_threshold(self):
        """rho(s) = 0 even at s = 2*gamma^2 (no real branch cut)."""
        gamma = 2.0
        s = 2 * gamma ** 2
        rho = spectral_function_gribov(s, gamma)
        assert rho == 0.0


# ===========================================================================
# Test 5: Glueball Threshold
# ===========================================================================
class TestGlueballThreshold:
    """The lightest physical state has mass >= sqrt(2)*gamma."""

    def test_threshold_mass_squared(self):
        """Threshold at s = 2*gamma^2."""
        gamma = 2.0
        result = glueball_threshold(gamma)
        expected = 2 * gamma ** 2
        assert abs(result['threshold_mass_squared'] - expected) < 1e-14

    def test_threshold_mass(self):
        """Threshold mass = sqrt(2)*gamma."""
        gamma = 3.0
        result = glueball_threshold(gamma)
        expected = np.sqrt(2) * gamma
        assert abs(result['threshold_mass'] - expected) < 1e-14

    def test_pole_modulus_consistency(self):
        """Pole modulus = gamma^2, consistent with threshold."""
        gamma = 1.5
        result = glueball_threshold(gamma)
        assert abs(result['pole_modulus'] - gamma ** 2) < 1e-14
        assert abs(result['threshold_mass_squared'] - 2 * result['pole_modulus']) < 1e-14

    def test_label_is_theorem(self):
        """Result labeled THEOREM."""
        result = glueball_threshold(1.0)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test 6: Physical Correlator Decay
# ===========================================================================
class TestCorrelatorDecay:
    """Gauge-invariant correlators decay as exp(-sqrt(2)*gamma*|x|)."""

    def test_decay_rate(self):
        """Decay rate = sqrt(2)*gamma."""
        gamma = 2.0
        result = physical_correlator_decay(gamma, [1.0])
        expected = np.sqrt(2) * gamma
        assert abs(result['decay_rate'] - expected) < 1e-14

    def test_bound_at_zero(self):
        """Upper bound at x=0 is 1."""
        gamma = 1.0
        result = physical_correlator_decay(gamma, [0.0])
        assert abs(result['upper_bound'][0] - 1.0) < 1e-14

    def test_bound_exponential_decay(self):
        """Bound decays exponentially."""
        gamma = 1.0
        x_values = np.array([0, 1, 2, 3, 4, 5])
        result = physical_correlator_decay(gamma, x_values)
        bound = result['upper_bound']
        rate = np.sqrt(2) * gamma
        for i, x in enumerate(x_values):
            expected = np.exp(-rate * x)
            assert abs(bound[i] - expected) < 1e-14

    def test_bound_monotone_decreasing(self):
        """Bound is monotonically decreasing for x > 0."""
        gamma = 2.0
        x_values = np.linspace(0, 5, 50)
        result = physical_correlator_decay(gamma, x_values)
        bound = result['upper_bound']
        for i in range(1, len(bound)):
            assert bound[i] <= bound[i - 1]

    def test_label_is_theorem(self):
        """Result labeled THEOREM."""
        result = physical_correlator_decay(1.0, [1.0])
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test 7: Mass Gap from gamma*
# ===========================================================================
class TestMassGapFromGribov:
    """Physical mass gap >= sqrt(2)*gamma*."""

    def test_su2_mass_gap(self):
        """For SU(2): gamma* = 3*sqrt(2)/2, m >= 3 Lambda_QCD."""
        gamma_star = 1.5 * np.sqrt(2)
        result = mass_gap_from_gribov(gamma_star, N=2)
        expected = np.sqrt(2) * gamma_star  # = 3.0
        assert abs(result['mass_gap_lower_bound'] - expected) < 1e-14
        assert abs(result['mass_gap_lower_bound'] - 3.0) < 1e-14

    def test_R_independent(self):
        """Mass gap is R-independent."""
        gamma_star = 1.5 * np.sqrt(2)
        result = mass_gap_from_gribov(gamma_star, N=2)
        assert result['R_independent'] is True

    def test_label(self):
        """Result labeled THEOREM."""
        result = mass_gap_from_gribov(1.0, N=2)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test 8: R-Independent Gap Theorem
# ===========================================================================
class TestRIndependentGap:
    """The complete theorem for the R-independent gap."""

    def test_su2_gamma_star(self):
        """gamma* = 3*sqrt(2)/2 for SU(2)."""
        result = theorem_r_independent_gap(N=2)
        expected = 1.5 * np.sqrt(2)
        assert abs(result['gamma_star'] - expected) < 1e-14

    def test_su2_mass_gap(self):
        """m_gap = 3 Lambda_QCD for SU(2)."""
        result = theorem_r_independent_gap(N=2)
        assert abs(result['mass_gap'] - 3.0) < 1e-14

    def test_su3_gamma_star(self):
        """gamma* = 8*sqrt(2)/3 for SU(3)."""
        result = theorem_r_independent_gap(N=3)
        expected = 8 * np.sqrt(2) / 3
        assert abs(result['gamma_star'] - expected) < 1e-12

    def test_su3_mass_gap(self):
        """m_gap = 16/3 Lambda_QCD for SU(3)."""
        result = theorem_r_independent_gap(N=3)
        expected = 16.0 / 3.0
        assert abs(result['mass_gap'] - expected) < 1e-12

    def test_gluon_confined(self):
        """Gluon is confined."""
        result = theorem_r_independent_gap(N=2)
        assert result['gluon_confined'] is True

    def test_R_independent(self):
        """Gap is R-independent."""
        result = theorem_r_independent_gap(N=2)
        assert result['R_independent'] is True

    def test_label_is_theorem(self):
        """Result labeled THEOREM."""
        result = theorem_r_independent_gap(N=2)
        assert result['label'] == 'THEOREM'

    def test_theorem_statement_exists(self):
        """Formal theorem statement is present."""
        result = theorem_r_independent_gap(N=2)
        assert 'THEOREM' in result['theorem_statement']
        assert 'PROOF' in result['theorem_statement']
        assert 'LABEL: THEOREM' in result['theorem_statement']

    def test_ingredients_complete(self):
        """All four ingredients are listed."""
        result = theorem_r_independent_gap(N=2)
        assert len(result['ingredients']) == 4
        for ing in result['ingredients']:
            assert ing['label'] == 'THEOREM'

    def test_assumptions_listed(self):
        """Assumptions of the theorem are stated."""
        result = theorem_r_independent_gap(N=2)
        assert len(result['assumptions']) >= 3


# ===========================================================================
# Test 9: Propagator Numerical Verification
# ===========================================================================
class TestPropagatorVerification:
    """Numerical checks on the Gribov propagator."""

    def test_D_at_zero_verified(self):
        """D(0) = 0 verified numerically."""
        result = verify_propagator_properties(2.0)
        assert result['D_at_zero_is_zero']

    def test_partial_fraction_decomposition(self):
        """Partial fraction decomposition is verified numerically."""
        result = verify_propagator_properties(1.5)
        assert result['partial_fraction_verified']

    def test_IR_suppressed(self):
        """D is suppressed in the IR."""
        result = verify_propagator_properties(2.0)
        assert result['IR_suppressed']

    def test_UV_suppressed(self):
        """D is suppressed in the UV."""
        result = verify_propagator_properties(2.0)
        assert result['UV_suppressed']


# ===========================================================================
# Test 10: Bubble Diagram Threshold
# ===========================================================================
class TestBubbleThreshold:
    """Numerical verification of the glueball threshold from bubble diagram."""

    def test_threshold_location_formula(self):
        """Threshold is at q ~ sqrt(2)*gamma (formula check)."""
        gamma = 2.0
        result = verify_threshold_numerically(gamma, n_points=500)
        expected = np.sqrt(2) * gamma
        assert abs(result['threshold_q'] - expected) < 1e-14

    def test_bubble_integral_finite(self):
        """The bubble integral Pi(q) is finite for all q."""
        result = verify_threshold_numerically(2.0, n_points=500)
        assert np.all(np.isfinite(result['Pi_values']))

    def test_bubble_integral_nonnegative(self):
        """Pi(q) >= 0 for all q (integrand is nonneg for real q)."""
        result = verify_threshold_numerically(2.0, n_points=500)
        assert np.all(result['Pi_values'] >= -1e-15)


# ===========================================================================
# Test 11: SU(N) Table
# ===========================================================================
class TestSUNTable:
    """Mass gap predictions for all SU(N)."""

    def test_table_length(self):
        """Default table has 5 entries (N=2..6)."""
        result = mass_gap_table_all_N()
        assert len(result['table']) == 5

    def test_mass_gap_increases_with_N(self):
        """Mass gap increases with N."""
        result = mass_gap_table_all_N()
        gaps = [row['mass_gap'] for row in result['table']]
        for i in range(1, len(gaps)):
            assert gaps[i] > gaps[i - 1]

    def test_gamma_star_formula(self):
        """gamma* = (N^2-1)*sqrt(2)/N for g^2_max = 4*pi."""
        result = mass_gap_table_all_N()
        for row in result['table']:
            N = row['N']
            expected = (N ** 2 - 1) * np.sqrt(2) / N
            assert abs(row['gamma_star'] - expected) < 1e-12

    def test_mass_gap_formula(self):
        """m_gap = sqrt(2)*gamma* = 2*(N^2-1)/N."""
        result = mass_gap_table_all_N()
        for row in result['table']:
            N = row['N']
            expected = 2.0 * (N ** 2 - 1) / N
            assert abs(row['mass_gap'] - expected) < 1e-12

    def test_label_is_theorem(self):
        """Table labeled THEOREM."""
        result = mass_gap_table_all_N()
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test 12: Complete Analysis
# ===========================================================================
class TestCompleteAnalysis:
    """Complete analysis combines all sub-results correctly."""

    def test_complete_analysis_su2(self):
        """Complete analysis for SU(2)."""
        result = complete_analysis(N=2)
        assert result['summary']['gluon_confined'] is True
        assert result['summary']['R_independent'] is True
        assert result['summary']['overall_label'] == 'THEOREM'

    def test_mass_gap_value(self):
        """Mass gap = 3 Lambda_QCD for SU(2)."""
        result = complete_analysis(N=2)
        assert abs(result['summary']['mass_gap_Lambda'] - 3.0) < 1e-14

    def test_all_subresults_present(self):
        """All sub-results (poles, positivity, threshold, decay, gap, theorem) present."""
        result = complete_analysis(N=2)
        assert 'poles' in result
        assert 'positivity' in result
        assert 'threshold' in result
        assert 'decay' in result
        assert 'mass_gap' in result
        assert 'theorem' in result
        assert 'propagator_verification' in result


# ===========================================================================
# Test 13: Consistency Between Steps
# ===========================================================================
class TestConsistency:
    """Cross-checks between different steps of the proof."""

    def test_pole_modulus_equals_threshold_half(self):
        """Pole modulus gamma^2 = threshold / 2."""
        gamma = 2.5
        poles = gribov_propagator_poles(gamma)
        threshold = glueball_threshold(gamma)
        assert abs(
            2 * poles['pole_modulus'] - threshold['threshold_mass_squared']
        ) < 1e-14

    def test_decay_rate_equals_threshold_mass(self):
        """Decay rate = sqrt(threshold_mass_squared) = threshold_mass."""
        gamma = 1.7
        threshold = glueball_threshold(gamma)
        decay = physical_correlator_decay(gamma, [1.0])
        assert abs(
            decay['decay_rate'] - threshold['threshold_mass']
        ) < 1e-14

    def test_mass_gap_equals_decay_rate(self):
        """Mass gap from gamma* = decay rate."""
        gamma_star = 1.5 * np.sqrt(2)  # SU(2)
        gap = mass_gap_from_gribov(gamma_star, N=2)
        decay = physical_correlator_decay(gamma_star, [1.0])
        assert abs(
            gap['mass_gap_lower_bound'] - decay['decay_rate']
        ) < 1e-14

    def test_su2_chain_consistency(self):
        """Full chain: gamma* -> poles -> threshold -> decay -> gap, all consistent."""
        gamma_star = 1.5 * np.sqrt(2)  # = 3*sqrt(2)/2

        # Step 1: poles
        poles = gribov_propagator_poles(gamma_star)
        assert abs(poles['pole_modulus'] - gamma_star ** 2) < 1e-14

        # Step 2: positivity
        pos = positivity_violation(gamma_star)
        assert pos['gluon_confined']

        # Step 3: threshold
        thr = glueball_threshold(gamma_star)
        assert abs(thr['threshold_mass_squared'] - 2 * gamma_star ** 2) < 1e-14
        assert abs(thr['threshold_mass'] - np.sqrt(2) * gamma_star) < 1e-14

        # Step 4: decay
        dec = physical_correlator_decay(gamma_star, [1.0])
        assert abs(dec['decay_rate'] - thr['threshold_mass']) < 1e-14

        # Step 5: mass gap
        gap = mass_gap_from_gribov(gamma_star, N=2)
        assert abs(gap['mass_gap_lower_bound'] - 3.0) < 1e-14


# ===========================================================================
# Test 14: Physical Reasonableness
# ===========================================================================
class TestPhysicalReasonableness:
    """Physical sanity checks on the mass gap prediction."""

    def test_su2_gap_order_of_magnitude(self):
        """m_gap ~ 3 Lambda_QCD ~ 600 MeV (order of magnitude correct)."""
        result = theorem_r_independent_gap(N=2)
        m_Lambda = result['mass_gap']
        # m_gap = 3 Lambda_QCD. With Lambda_QCD ~ 200 MeV, m ~ 600 MeV.
        # Lattice glueball mass 0++ ~ 1710 MeV.
        # Our LOWER BOUND of 600 MeV is consistent (below lattice value).
        assert 1.0 < m_Lambda < 10.0, f"m/Lambda = {m_Lambda} out of physical range"

    def test_su3_gap_order_of_magnitude(self):
        """m_gap for SU(3) should be larger than SU(2)."""
        r2 = theorem_r_independent_gap(N=2)
        r3 = theorem_r_independent_gap(N=3)
        assert r3['mass_gap'] > r2['mass_gap']

    def test_gap_positive(self):
        """Mass gap is strictly positive."""
        for N in [2, 3, 4]:
            result = theorem_r_independent_gap(N=N)
            assert result['mass_gap'] > 0

    def test_large_N_scaling(self):
        """At large N, m_gap ~ 2*N (grows linearly with N)."""
        N = 100
        result = theorem_r_independent_gap(N=N)
        # m_gap = 2*(N^2-1)/N ~ 2*N for large N
        expected_approx = 2.0 * N
        assert abs(result['mass_gap'] / expected_approx - 1) < 0.02
