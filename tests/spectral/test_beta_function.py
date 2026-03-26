"""
Tests for the beta function module.

Verifies the 1-loop coefficient, running coupling, and
consistency with asymptotic freedom.
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.beta_function import BetaFunction


class TestOneLoopFlat:
    """1-loop beta function coefficient b₀ = 11N/(48π²)."""

    def test_su3_coefficient(self):
        """
        SU(3): b₀ = 33/(48π²) ≈ 0.06948.
        This MUST match the known value exactly (up to numerics).
        """
        b0 = BetaFunction.one_loop_flat(3)
        expected = 11 * 3 / (48 * np.pi**2)
        assert abs(b0 - expected) < 1e-14

        # Known numerical value: 33/(48π²) ≈ 0.06966
        assert abs(b0 - 0.06966) < 0.0001, (
            f"b₀(SU(3)) = {b0:.5f}, expected ≈ 0.06966"
        )

    def test_su2_coefficient(self):
        """SU(2): b₀ = 22/(48π²) ≈ 0.04632."""
        b0 = BetaFunction.one_loop_flat(2)
        expected = 22 / (48 * np.pi**2)
        assert abs(b0 - expected) < 1e-14

    def test_scales_linearly_with_N(self):
        """b₀ is proportional to N."""
        for N in range(2, 8):
            b0 = BetaFunction.one_loop_flat(N)
            expected = 11 * N / (48 * np.pi**2)
            assert abs(b0 - expected) < 1e-14

    def test_positive_for_asymptotic_freedom(self):
        """b₀ > 0 for all SU(N) — this means asymptotic freedom."""
        for N in range(2, 10):
            assert BetaFunction.one_loop_flat(N) > 0

    def test_invalid_N(self):
        """N < 2 should raise."""
        with pytest.raises(ValueError):
            BetaFunction.one_loop_flat(1)


class TestOneLoopS3:
    """1-loop coefficient on S³ (should match flat space at leading order)."""

    def test_matches_flat_space(self):
        """
        The leading-order S³ coefficient MUST equal the flat-space value.
        This is a consistency check of the framework.
        """
        for N in [2, 3, 5]:
            result = BetaFunction.one_loop_s3(N, R=2.0)
            flat = BetaFunction.one_loop_flat(N)

            assert abs(result['b0_leading'] - flat) < 1e-14, (
                f"SU({N}): S³ leading coefficient {result['b0_leading']} "
                f"!= flat space {flat}"
            )

    def test_returns_expected_keys(self):
        """Result dict has expected structure."""
        result = BetaFunction.one_loop_s3(3, R=1.0)
        assert 'b0_flat' in result
        assert 'b0_leading' in result
        assert 'finite_volume_note' in result


class TestRunningCoupling:
    """1-loop running coupling g²(μ)."""

    def test_decreases_with_energy(self):
        """g²(μ) decreases as μ increases (asymptotic freedom)."""
        Lambda = 200.0
        g2_low = BetaFunction.running_coupling(3, 500, Lambda)
        g2_high = BetaFunction.running_coupling(3, 5000, Lambda)

        assert g2_low > g2_high, (
            f"g²(500 MeV) = {g2_low:.4f} should be > "
            f"g²(5000 MeV) = {g2_high:.4f}"
        )

    def test_mu_equals_lambda_raises(self):
        """μ = Λ is the Landau pole — should raise."""
        with pytest.raises(ValueError):
            BetaFunction.running_coupling(3, 200, 200)

    def test_mu_below_lambda_raises(self):
        """μ < Λ is non-perturbative — should raise."""
        with pytest.raises(ValueError):
            BetaFunction.running_coupling(3, 100, 200)

    def test_alpha_s_at_mz(self):
        """
        α_s(M_Z) at 1-loop for pure YM (no quarks).

        At M_Z = 91.2 GeV with Λ = 200 MeV:
        α_s = g²/(4π) = 1/(4π × b₀ × ln(M_Z²/Λ²))

        This won't match the experimental α_s(M_Z) ≈ 0.118 exactly
        because we're doing pure YM (no quarks), but it should be
        in the right ballpark.
        """
        alpha = BetaFunction.alpha_s(3, 91200, 200)
        # Pure YM gives a different value than QCD with quarks
        # but it should be positive and small
        assert 0 < alpha < 0.5, f"α_s(M_Z) = {alpha:.4f}, should be O(0.1)"


class TestCouplingAtGap:
    """Coupling at the gap scale μ = ℏc/R."""

    def test_gap_scale_value(self):
        """At R = 2.2 fm, μ = 197.3/2.2 ≈ 89.7 MeV."""
        result = BetaFunction.coupling_at_gap(3, R_fm=2.2)
        expected_mu = 197.3269804 / 2.2
        assert abs(result['mu_MeV'] - expected_mu) < 0.01

    def test_non_perturbative_at_large_R(self):
        """
        At R = 2.2 fm, μ ≈ 90 MeV < Λ_QCD = 200 MeV.
        This is NON-PERTURBATIVE. The coupling is not computable
        from the 1-loop formula here.
        """
        result = BetaFunction.coupling_at_gap(3, R_fm=2.2, Lambda_MeV=200.0)
        assert result['is_perturbative'] is False
        assert result['g_squared'] is None

    def test_perturbative_at_small_R(self):
        """
        At R = 0.1 fm, μ ≈ 1973 MeV >> Λ_QCD.
        This IS perturbative.
        """
        result = BetaFunction.coupling_at_gap(3, R_fm=0.1, Lambda_MeV=200.0)
        assert result['is_perturbative'] is True
        assert result['g_squared'] is not None
        assert result['alpha_s'] is not None
        assert result['alpha_s'] > 0

    def test_returns_expected_keys(self):
        """Result dict has expected structure."""
        result = BetaFunction.coupling_at_gap(3, R_fm=1.0)
        for key in ['mu_MeV', 'g_squared', 'alpha_s',
                     'is_perturbative', 'note']:
            assert key in result


class TestAsymptoticFreedom:
    """Verify asymptotic freedom explicitly."""

    def test_su3_asymptotic_freedom(self):
        """α_s monotonically decreases with energy for SU(3)."""
        result = BetaFunction.verify_asymptotic_freedom(3, Lambda_MeV=200.0)
        assert result['asymptotic_freedom'] is True

    def test_su2_asymptotic_freedom(self):
        """α_s monotonically decreases for SU(2) too."""
        result = BetaFunction.verify_asymptotic_freedom(2, Lambda_MeV=200.0)
        assert result['asymptotic_freedom'] is True

    def test_b0_in_result(self):
        """b₀ is included in the result."""
        result = BetaFunction.verify_asymptotic_freedom(3)
        assert abs(result['b0'] - 11 * 3 / (48 * np.pi**2)) < 1e-14
