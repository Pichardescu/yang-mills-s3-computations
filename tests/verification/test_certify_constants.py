"""
Tests for computer-assisted proof Stage 3: constant certification.

Verifies all Bronze and Silver level certifications at high precision.
"""

import pytest
import mpmath

# Set high precision for tests
mpmath.mp.dps = 50

import sys
sys.path.insert(0, '.')
from verification.interval_arithmetic.certify_constants import (
    certify_alpha_kr,
    certify_beta_0,
    certify_g_bar_0,
    certify_g2_critical,
    certify_kappa,
    certify_r0_from_hessian,
    certify_contraction_chain,
    certify_c_epsilon_factors,
    G2_BARE,
    HBAR_C,
    LAMBDA_QCD,
)


# ======================================================================
# BRONZE: Algebraic Constants
# ======================================================================

class TestBronzeCertification:
    """All algebraic constant certifications."""

    def test_alpha_kr_less_than_one(self):
        """CRITICAL: alpha < 1 ensures Kato-Rellich stability."""
        result = certify_alpha_kr()
        assert result.inequality_holds
        assert result.value_float < 1.0
        assert result.value_float < 0.04  # Much less than 1

    def test_alpha_kr_value(self):
        """alpha = g^2 * sqrt(2) / (24*pi^2) ~ 0.0375."""
        result = certify_alpha_kr()
        assert abs(result.value_float - 0.0375) < 0.001

    def test_alpha_kr_exact(self):
        """Verify alpha at 50 digits matches independent computation."""
        g2 = mpmath.mpf('6.28')
        alpha = g2 * mpmath.sqrt(2) / (24 * mpmath.pi**2)
        result = certify_alpha_kr()
        assert abs(result.value_mpmath - alpha) < mpmath.mpf('1e-45')

    def test_beta_0_value(self):
        """beta_0 = 22/(48*pi^2) ~ 0.04644."""
        result = certify_beta_0()
        assert abs(result.value_float - 0.04644) < 0.0001

    def test_beta_0_exact(self):
        """beta_0 at full precision."""
        beta_0 = mpmath.mpf(22) / (48 * mpmath.pi**2)
        result = certify_beta_0()
        assert abs(result.value_mpmath - beta_0) < mpmath.mpf('1e-45')

    def test_g_bar_0_value(self):
        """g_bar_0 = sqrt(6.28) ~ 2.506."""
        result = certify_g_bar_0()
        assert abs(result.value_float - 2.506) < 0.001

    def test_g2_critical_value(self):
        """g^2_c = 24*pi^2/sqrt(2) ~ 167.5."""
        result = certify_g2_critical()
        assert abs(result.value_float - 167.5) < 0.5

    def test_g2_less_than_critical(self):
        """CRITICAL: g^2 = 6.28 < g^2_c ~ 167.5."""
        result = certify_g2_critical()
        assert result.inequality_holds
        assert result.value_float > 100  # Safety margin > 15x

    def test_safety_margin(self):
        """Safety margin g^2_c / g^2 ~ 26.7."""
        g2_c = 24 * mpmath.pi**2 / mpmath.sqrt(2)
        margin = float(g2_c / G2_BARE)
        assert margin > 25  # At least 25x safety
        assert margin < 28  # Sanity check

    def test_kappa_value(self):
        """kappa = g^2/R^3 at R=2.2 fm."""
        result = certify_kappa()
        assert abs(result.value_float - 0.5898) < 0.01

    def test_r0_self_consistency(self):
        """R_0 = 2*hbar_c/Lambda_QCD ~ 1.97 fm."""
        result = certify_r0_from_hessian()
        assert abs(result.value_float - 1.973) < 0.01


# ======================================================================
# SILVER: Contraction Chain
# ======================================================================

class TestSilverCertification:
    """Contraction chain verification."""

    def test_epsilon_0_less_than_one(self):
        """CRITICAL: epsilon_0 < 1 for BBS contraction."""
        result = certify_contraction_chain(c_epsilon=0.275, verbose=False)
        assert result.contraction_holds
        assert float(result.epsilon_0) < 1.0

    def test_epsilon_0_value(self):
        """epsilon_0 = c_epsilon * g_bar_0 ~ 0.689."""
        result = certify_contraction_chain(c_epsilon=0.275, verbose=False)
        eps_0 = float(result.epsilon_0)
        assert abs(eps_0 - 0.689) < 0.01

    def test_margin_to_failure(self):
        """Margin: 1 - epsilon_0 ~ 0.31 (31% margin)."""
        result = certify_contraction_chain(c_epsilon=0.275, verbose=False)
        margin = float(1 - result.epsilon_0)
        assert margin > 0.3  # At least 30% margin
        assert margin < 0.35

    def test_doubly_exponential_decay(self):
        """epsilon_k = epsilon_0^{2^{k-1}} decreases doubly exponentially."""
        result = certify_contraction_chain(c_epsilon=0.275, verbose=False)
        eps_0 = float(result.epsilon_0)
        eps_steps = [float(e) for e in result.epsilon_steps]

        # epsilon_1 = epsilon_0^2
        assert abs(eps_steps[0] - eps_0**2) < 1e-10
        # epsilon_2 = epsilon_1^2
        assert abs(eps_steps[1] - eps_steps[0]**2) < 1e-10
        # epsilon_3 = epsilon_2^2
        assert abs(eps_steps[2] - eps_steps[1]**2) < 1e-10

        # Each step squares -> rapid decay
        assert eps_steps[0] < eps_0
        assert eps_steps[1] < eps_steps[0]
        assert eps_steps[2] < eps_steps[1]

    def test_epsilon_3_small(self):
        """epsilon_3 < 0.1 (very small after 3 multi-step iterations)."""
        result = certify_contraction_chain(c_epsilon=0.275, verbose=False)
        eps_3 = float(result.epsilon_steps[2])
        assert eps_3 < 0.1

    def test_contraction_robust_to_c_epsilon_increase(self):
        """Contraction holds even if c_epsilon increases by 20%."""
        result = certify_contraction_chain(c_epsilon=0.275 * 1.20, verbose=False)
        assert result.contraction_holds, (
            f"Contraction fails at c_epsilon = {0.275 * 1.20}: "
            f"epsilon_0 = {float(result.epsilon_0)}"
        )

    def test_contraction_fails_at_large_c_epsilon(self):
        """Sanity: contraction fails if c_epsilon is too large."""
        # At c_epsilon = 0.5, epsilon_0 = 0.5 * 2.506 = 1.253 > 1
        result = certify_contraction_chain(c_epsilon=0.5, verbose=False)
        assert not result.contraction_holds

    def test_c_epsilon_computed_below_pessimistic(self):
        """Computed c_epsilon (0.135) is well below pessimistic (0.275)."""
        factors = certify_c_epsilon_factors()
        c_eps = float(factors['c_epsilon'])
        assert c_eps < 0.275, f"Computed c_epsilon = {c_eps} >= 0.275"
        assert c_eps > 0.1, f"Computed c_epsilon = {c_eps} suspiciously small"


# ======================================================================
# Cross-checks
# ======================================================================

class TestCrossChecks:
    """Cross-check consistency between constants."""

    def test_alpha_equals_g2_over_g2c(self):
        """alpha = g^2 / g^2_c (by definition)."""
        alpha_result = certify_alpha_kr()
        g2c_result = certify_g2_critical()

        alpha = alpha_result.value_mpmath
        ratio = G2_BARE / g2c_result.value_mpmath

        assert abs(float(alpha - ratio)) < 1e-10

    def test_gap_formula_consistency(self):
        """Linear gap = 2*hbar_c/R gives ~200 MeV at R=1.97 fm."""
        R = 2 * HBAR_C / LAMBDA_QCD  # = R_0
        gap = 2 * HBAR_C / R  # = Lambda_QCD by construction
        assert abs(float(gap) - 200.0) < 0.01

    def test_s3_gap_formula(self):
        """Spectral gap on S^3: lambda_1 = 4/R^2 (coexact 1-forms)."""
        R = mpmath.mpf('2.2')  # fm
        lambda_1 = 4 / R**2
        gap_mev = mpmath.sqrt(lambda_1) * HBAR_C
        # At R=2.2 fm: sqrt(4/2.2^2) * 197.3 = (2/2.2)*197.3 = 179.4 MeV
        assert abs(float(gap_mev) - 179.4) < 0.5
