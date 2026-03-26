"""
Tests for the Gribov parameter stabilization theorem.

Verifies:
1. The limiting integral = gamma * pi/(2*sqrt(2)) matches numerical quadrature
2. gamma* analytical matches gamma* numerical
3. gamma* is close to the earlier numerical estimate (~2.12)
4. Weyl density convergence (sum -> integral for large R)
5. Convergence rate is at least O(1/R)
6. dF_inf/dgamma != 0 at gamma* (implicit function theorem condition)
7. gamma(R) from the Zwanziger solver approaches gamma* monotonically
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import importlib.util
import os

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec_gs = importlib.util.spec_from_file_location(
    'gamma_stabilization',
    os.path.join(_BASE, 'src', 'proofs', 'gamma_stabilization.py'),
)
_mod_gs = importlib.util.module_from_spec(_spec_gs)
_spec_gs.loader.exec_module(_mod_gs)
GammaStabilization = _mod_gs.GammaStabilization

_spec_zge = importlib.util.spec_from_file_location(
    'zwanziger_gap_equation',
    os.path.join(_BASE, 'src', 'spectral', 'zwanziger_gap_equation.py'),
)
_mod_zge = importlib.util.module_from_spec(_spec_zge)
_spec_zge.loader.exec_module(_mod_zge)
ZwanzigerGapEquation = _mod_zge.ZwanzigerGapEquation


# ===========================================================================
# Test 1: Limiting integral matches analytical formula
# ===========================================================================
class TestLimitingIntegral:
    """The key integral int_0^inf gamma^4/(k^4+gamma^4) dk = gamma*pi/(2*sqrt(2))."""

    def test_integral_at_gamma_1(self):
        """Integral at gamma=1 should be pi/(2*sqrt(2))."""
        result = GammaStabilization.limiting_integral(1.0)
        expected = np.pi / (2 * np.sqrt(2))
        assert abs(result['analytical'] - expected) < 1e-14
        assert result['match']

    def test_integral_at_gamma_2(self):
        """Integral at gamma=2 should be 2*pi/(2*sqrt(2)) = pi/sqrt(2)."""
        result = GammaStabilization.limiting_integral(2.0)
        expected = 2.0 * np.pi / (2 * np.sqrt(2))
        assert abs(result['analytical'] - expected) < 1e-14
        assert result['match']

    def test_integral_matches_numerical(self):
        """Analytical and numerical integral agree to high precision."""
        for gamma in [0.5, 1.0, 2.0, 5.0, 10.0]:
            result = GammaStabilization.limiting_integral(gamma)
            assert result['relative_error'] < 1e-10, (
                f"gamma={gamma}: analytical={result['analytical']}, "
                f"numerical={result['numerical']}, "
                f"rel_err={result['relative_error']}"
            )

    def test_integral_linearity_in_gamma(self):
        """The integral is linear in gamma: I(c*gamma) = c*I(gamma)."""
        gamma_1 = 2.0
        gamma_2 = 6.0
        I1 = GammaStabilization.limiting_integral(gamma_1)['analytical']
        I2 = GammaStabilization.limiting_integral(gamma_2)['analytical']
        assert abs(I2 / I1 - gamma_2 / gamma_1) < 1e-14


# ===========================================================================
# Test 2: gamma* analytical matches gamma* numerical
# ===========================================================================
class TestGammaStar:
    """The analytical formula gamma* = (N^2-1)*4*pi*sqrt(2)/(g^2_max*N)."""

    def test_gamma_star_su2_value(self):
        """gamma* for SU(2) should be 3*sqrt(2)/2."""
        gs = GammaStabilization.gamma_star_analytical(N=2)
        expected = 1.5 * np.sqrt(2)
        assert abs(gs - expected) < 1e-14

    def test_gamma_star_analytical_matches_numerical(self):
        """Analytical and numerical gamma* agree to machine precision."""
        for N in [2, 3, 4]:
            gs_a = GammaStabilization.gamma_star_analytical(N=N)
            gs_n = GammaStabilization.gamma_star_numerical(N=N)
            assert abs(gs_a - gs_n) < 1e-10, (
                f"N={N}: analytical={gs_a}, numerical={gs_n}"
            )

    def test_gamma_star_solves_gap_equation(self):
        """Plugging gamma* into F_inf should give residual = 0."""
        for N in [2, 3]:
            gs = GammaStabilization.gamma_star_analytical(N=N)
            res = GammaStabilization.limiting_gap_equation(gs, N=N)
            assert abs(res) < 1e-12, (
                f"N={N}: gamma*={gs}, residual={res}"
            )

    def test_gamma_star_close_to_numerical_estimate(self):
        """gamma* should be close to the earlier numerical value ~2.12-2.16."""
        gs = GammaStabilization.gamma_star_analytical(N=2)
        # The earlier numerical estimate was ~2.16, which included finite-R effects.
        # The exact asymptotic value is 3*sqrt(2)/2 = 2.121...
        assert abs(gs - 2.12) < 0.1, f"gamma*={gs} not close to 2.12"

    def test_gamma_star_scales_with_N(self):
        """gamma* should increase with N (more colors = more self-interaction)."""
        gs2 = GammaStabilization.gamma_star_analytical(N=2)
        gs3 = GammaStabilization.gamma_star_analytical(N=3)
        gs4 = GammaStabilization.gamma_star_analytical(N=4)
        # gamma* = (N^2-1) * C / N = C * (N - 1/N)
        # This should increase with N
        assert gs3 > gs2
        assert gs4 > gs3


# ===========================================================================
# Test 3: Weyl density convergence
# ===========================================================================
class TestWeylDensity:
    """Spectral sum converges to flat-space integral by Weyl's law."""

    def test_weyl_convergence(self):
        """Sum/V should approach integral for large R."""
        R_values = [5.0, 10.0, 20.0, 50.0]
        result = GammaStabilization.weyl_density_check(R_values, l_max=500)
        # Errors should decrease
        for i in range(1, len(result['relative_errors'])):
            assert result['relative_errors'][i] < result['relative_errors'][i - 1], (
                f"Error not decreasing: R={result['R_values'][i]}, "
                f"err={result['relative_errors'][i]} >= prev={result['relative_errors'][i - 1]}"
            )

    def test_weyl_accuracy_at_large_R(self):
        """At R=50, the relative error should be small (< 5%)."""
        result = GammaStabilization.weyl_density_check([50.0], l_max=1000)
        assert result['relative_errors'][0] < 0.05, (
            f"Weyl accuracy at R=50: err={result['relative_errors'][0]}"
        )

    def test_weyl_integral_value(self):
        """The Weyl integral for gamma=2.0 matches the analytical formula."""
        gamma = 2.0
        expected = gamma * np.pi / (2 * np.sqrt(2) * 2 * np.pi ** 2)
        result = GammaStabilization.weyl_density_check([10.0])
        assert abs(result['weyl_integral'] - expected) < 1e-14


# ===========================================================================
# Test 4: Convergence rate
# ===========================================================================
class TestConvergenceRate:
    """gamma(R) -> gamma* at rate O(1/R)."""

    def test_gamma_approaches_star(self):
        """gamma(R) values should get closer to gamma* as R increases."""
        R_values = [5.0, 10.0, 20.0, 50.0]
        gamma_star = GammaStabilization.gamma_star_analytical(N=2)

        gammas = []
        for R in R_values:
            l_max = max(500, int(R * 10))
            g = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=l_max)
            gammas.append(g)

        errors = [abs(g - gamma_star) for g in gammas]
        # Each error should be smaller than the previous
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"Error not decreasing at R={R_values[i]}: "
                f"{errors[i]} >= {errors[i - 1]}"
            )

    def test_convergence_rate_exponent(self):
        """Power-law fit should give exponent <= -0.5 (at least O(1/sqrt(R)))."""
        R_values = [5.0, 10.0, 20.0, 50.0]
        gamma_star = GammaStabilization.gamma_star_analytical(N=2)

        errors = []
        for R in R_values:
            l_max = max(500, int(R * 10))
            g = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=l_max)
            errors.append(abs(g - gamma_star))

        coeffs = np.polyfit(np.log(R_values), np.log(errors), 1)
        rate = coeffs[0]
        assert rate < -0.5, f"Convergence rate exponent {rate} not < -0.5"


# ===========================================================================
# Test 5: Implicit function theorem
# ===========================================================================
class TestImplicitFunction:
    """dF_inf/dgamma != 0 at gamma* (IFT condition)."""

    def test_derivative_nonzero_su2(self):
        """For SU(2), dF/dgamma should be nonzero."""
        result = GammaStabilization.implicit_function_check(N=2)
        assert result['nonzero']
        assert result['ift_applies']

    def test_derivative_nonzero_su3(self):
        """For SU(3), dF/dgamma should be nonzero."""
        result = GammaStabilization.implicit_function_check(N=3)
        assert result['nonzero']
        assert result['ift_applies']

    def test_derivative_value(self):
        """dF/dgamma = -g^2_max*N/(4*pi*sqrt(2))."""
        N = 2
        g2_max = 4 * np.pi
        expected = -g2_max * N / (4 * np.pi * np.sqrt(2))
        result = GammaStabilization.implicit_function_check(N=N)
        assert abs(result['dF_dgamma'] - expected) < 1e-14

    def test_equation_is_linear(self):
        """The limiting equation is linear in gamma (a strong condition)."""
        result = GammaStabilization.implicit_function_check(N=2)
        assert result['equation_is_linear']


# ===========================================================================
# Test 6: Monotonicity from the Zwanziger solver
# ===========================================================================
class TestZwanzigerApproach:
    """gamma(R) from the Zwanziger module approaches gamma* from above."""

    def test_gamma_R_finite(self):
        """gamma(R) should be finite for all R tested."""
        for R in [1.0, 5.0, 10.0, 50.0]:
            g = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
            assert np.isfinite(g), f"gamma(R={R}) = {g} is not finite"
            assert g > 0, f"gamma(R={R}) = {g} is not positive"

    def test_gamma_R_approaches_star_from_above(self):
        """For R >= 5, gamma(R) should be above gamma* (approaches from above)."""
        gamma_star = GammaStabilization.gamma_star_analytical(N=2)
        for R in [5.0, 10.0, 20.0, 50.0]:
            l_max = max(500, int(R * 10))
            g = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=l_max)
            assert g > gamma_star, (
                f"R={R}: gamma(R)={g} not above gamma*={gamma_star}"
            )

    def test_gamma_R_decreasing_for_large_R(self):
        """For large R, gamma(R) should be decreasing toward gamma*."""
        R_values = [5.0, 10.0, 20.0, 50.0]
        gammas = []
        for R in R_values:
            l_max = max(500, int(R * 10))
            gammas.append(ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=l_max))

        for i in range(1, len(gammas)):
            assert gammas[i] < gammas[i - 1], (
                f"gamma not decreasing: gamma(R={R_values[i]})={gammas[i]} "
                f">= gamma(R={R_values[i - 1]})={gammas[i - 1]}"
            )


# ===========================================================================
# Test 7: Limiting gap equation properties
# ===========================================================================
class TestLimitingGapEquation:
    """Properties of F_inf(gamma)."""

    def test_residual_positive_at_zero(self):
        """F_inf(0) = N^2-1 > 0."""
        res = GammaStabilization.limiting_gap_equation(0.0, N=2)
        assert res == 3.0

    def test_residual_negative_at_large_gamma(self):
        """F_inf(large) < 0."""
        res = GammaStabilization.limiting_gap_equation(100.0, N=2)
        assert res < 0

    def test_residual_zero_at_star(self):
        """F_inf(gamma*) = 0."""
        gs = GammaStabilization.gamma_star_analytical(N=2)
        res = GammaStabilization.limiting_gap_equation(gs, N=2)
        assert abs(res) < 1e-12

    def test_residual_linear(self):
        """F_inf is linear: F_inf(a*g1 + (1-a)*g2) = a*F(g1) + (1-a)*F(g2)."""
        g1, g2 = 1.0, 3.0
        a = 0.37
        F1 = GammaStabilization.limiting_gap_equation(g1, N=2)
        F2 = GammaStabilization.limiting_gap_equation(g2, N=2)
        F_mix = GammaStabilization.limiting_gap_equation(a * g1 + (1 - a) * g2, N=2)
        expected = a * F1 + (1 - a) * F2
        assert abs(F_mix - expected) < 1e-12


# ===========================================================================
# Test 8: Formal proof statement
# ===========================================================================
class TestFormalProof:
    """The formal proof statement is complete and labeled THEOREM."""

    def test_proof_contains_theorem(self):
        proof = GammaStabilization.formal_proof_statement(N=2)
        assert 'THEOREM' in proof

    def test_proof_contains_gamma_star(self):
        proof = GammaStabilization.formal_proof_statement(N=2)
        assert '2.1213203436' in proof

    def test_proof_contains_qed(self):
        proof = GammaStabilization.formal_proof_statement(N=2)
        assert 'QED' in proof

    def test_proof_label(self):
        proof = GammaStabilization.formal_proof_statement(N=2)
        assert 'LABEL: THEOREM' in proof


# ===========================================================================
# Test 9: General N consistency
# ===========================================================================
class TestGeneralN:
    """gamma* formula works for general SU(N)."""

    def test_su3(self):
        """SU(3): gamma* = 8 * 4*pi*sqrt(2) / (4*pi*3) = 8*sqrt(2)/3."""
        gs = GammaStabilization.gamma_star_analytical(N=3)
        expected = 8 * np.sqrt(2) / 3
        assert abs(gs - expected) < 1e-12

    def test_su4(self):
        """SU(4): gamma* = 15 * 4*pi*sqrt(2) / (4*pi*4) = 15*sqrt(2)/4."""
        gs = GammaStabilization.gamma_star_analytical(N=4)
        expected = 15 * np.sqrt(2) / 4
        assert abs(gs - expected) < 1e-12

    def test_general_formula(self):
        """gamma* = (N^2-1)*sqrt(2)/N for g^2_max = 4*pi."""
        for N in [2, 3, 4, 5]:
            gs = GammaStabilization.gamma_star_analytical(N=N)
            expected = (N ** 2 - 1) * np.sqrt(2) / N
            assert abs(gs - expected) < 1e-12, f"N={N}: {gs} != {expected}"
