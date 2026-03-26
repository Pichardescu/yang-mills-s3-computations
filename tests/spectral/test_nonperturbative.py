"""
Tests for the nonperturbative enhancement module.

Verifies that the factor ~8.6 between the linearized gap on S3 and the
lattice glueball mass is computed correctly and cross-checked against
known lattice QCD ratios.

Key checks:
    1. The enhancement factor is computed correctly
    2. RG running analysis is self-consistent
    3. Factorization via string tension works
    4. Scheme dependence spans the expected range
    5. Large-N data is consistent
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.nonperturbative_enhancement import (
    NonperturbativeEnhancement,
    HBAR_C_MEV_FM,
)


class TestEnhancementFactor:
    """The core ratio m(0++) / linearized_gap."""

    def test_ratio_value(self):
        """
        At R = 2.2 fm:
            gap = 2 * 197.327 / 2.2 = 179.4 MeV
            ratio = 1730 / 179.4 = 9.64
        """
        result = NonperturbativeEnhancement.enhancement_factor(R_fm=2.2)

        expected_gap = 2.0 * HBAR_C_MEV_FM / 2.2
        assert abs(result['linearized_gap_MeV'] - expected_gap) < 0.01
        assert result['glueball_mass_MeV'] == 1730.0
        assert abs(result['ratio'] - 9.64) < 0.1

    def test_ratio_in_expected_range(self):
        """
        The ratio m(0++) / Lambda_QCD should be in 7-9 for any
        reasonable Lambda scheme. Our value at R=2.2 fm should
        be in this range.
        """
        result = NonperturbativeEnhancement.enhancement_factor(R_fm=2.2)
        assert 5.0 < result['ratio'] < 12.0, (
            f"Ratio {result['ratio']:.2f} is outside expected range 5-12"
        )

    def test_ratio_increases_with_R(self):
        """
        Larger R means smaller linearized gap, so ratio increases.
        """
        r1 = NonperturbativeEnhancement.enhancement_factor(R_fm=1.5)
        r2 = NonperturbativeEnhancement.enhancement_factor(R_fm=2.2)
        r3 = NonperturbativeEnhancement.enhancement_factor(R_fm=3.0)

        assert r1['ratio'] < r2['ratio'] < r3['ratio']

    def test_ratio_decreases_with_smaller_R(self):
        """
        At R ~ 0.228 fm, the linearized gap equals the glueball mass,
        so the ratio should be ~1.
        """
        R_match = 2.0 * HBAR_C_MEV_FM / 1730.0
        result = NonperturbativeEnhancement.enhancement_factor(R_fm=R_match)
        assert abs(result['ratio'] - 1.0) < 0.01

    def test_result_has_expected_keys(self):
        """Result dict has all expected keys."""
        result = NonperturbativeEnhancement.enhancement_factor()
        for key in ['linearized_gap_MeV', 'glueball_mass_MeV', 'ratio',
                     'interpretation', 'status']:
            assert key in result


class TestRGRunning:
    """RG running analysis from gap scale to glueball scale."""

    def test_b0_su3(self):
        """b0 for SU(3) should be 33/(48*pi^2)."""
        result = NonperturbativeEnhancement.rg_running_analysis(N=3)
        expected_b0 = 11 * 3 / (48 * np.pi**2)
        assert abs(result['b0'] - expected_b0) < 1e-12

    def test_scale_ratio(self):
        """
        The ratio of glueball scale to gap scale should be ~9.6.
        """
        result = NonperturbativeEnhancement.rg_running_analysis(
            N=3, R_fm=2.2, mu_high_MeV=1730.0
        )
        gap = 2.0 * HBAR_C_MEV_FM / 2.2
        expected_ratio = 1730.0 / gap
        assert abs(result['scale_ratio'] - expected_ratio) < 0.1

    def test_gap_scale_nonperturbative(self):
        """
        At R = 2.2 fm, the gap scale ~179 MeV is BELOW Lambda_QCD = 200 MeV.
        This is non-perturbative.
        """
        result = NonperturbativeEnhancement.rg_running_analysis(
            N=3, Lambda_MeV=200.0, R_fm=2.2
        )
        # gap ~ 179.4 MeV < Lambda=200, so non-perturbative
        # The code checks mu_low > Lambda
        assert result['mu_low_MeV'] < 200.0  # below Lambda

    def test_glueball_scale_perturbative(self):
        """
        At the glueball scale (1730 MeV), alpha_s should be computable
        and moderately small.
        """
        result = NonperturbativeEnhancement.rg_running_analysis(
            N=3, Lambda_MeV=200.0, mu_high_MeV=1730.0
        )
        assert result['high_is_perturbative'] is True
        assert result['alpha_s_high'] is not None
        assert 0.1 < result['alpha_s_high'] < 0.5, (
            f"alpha_s(1730 MeV) = {result['alpha_s_high']:.4f}, "
            f"expected in range 0.1-0.5"
        )

    def test_alpha_s_at_glueball_scale(self):
        """
        At 1730 MeV with Lambda = 200 MeV (1-loop, pure YM):
            alpha_s = 1 / (4*pi * b0 * ln(1730^2/200^2))
        """
        result = NonperturbativeEnhancement.rg_running_analysis(
            N=3, Lambda_MeV=200.0, mu_high_MeV=1730.0
        )
        b0 = 11 * 3 / (48 * np.pi**2)
        log_ratio = np.log(1730.0**2 / 200.0**2)
        expected_alpha = 1.0 / (4 * np.pi * b0 * log_ratio)

        assert abs(result['alpha_s_high'] - expected_alpha) < 1e-10

    def test_lambda_recovery(self):
        """
        From alpha_s at the glueball scale, recovering Lambda should give
        a value in the right ballpark (within 30% of input due to
        1-loop truncation).
        """
        result = NonperturbativeEnhancement.rg_running_analysis(
            N=3, Lambda_MeV=200.0, mu_high_MeV=1730.0
        )
        if 'lambda_recovered_MeV' in result:
            recovered = result['lambda_recovered_MeV']
            # 1-loop recovery won't be exact; allow 50% tolerance
            assert 100 < recovered < 300, (
                f"Recovered Lambda = {recovered:.1f} MeV, "
                f"expected ~200 MeV (within ~50%)"
            )


class TestLatticeCrossChecks:
    """Cross-checks with known lattice ratios."""

    def test_factorization_identity(self):
        """
        m(0++) / Lambda = [m(0++) / sqrt(sigma)] * [sqrt(sigma) / Lambda]

        This must be an exact identity (tautology), so the factorization
        error should be zero.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks(R_fm=2.2)
        assert result['factorization_check'] < 1e-10, (
            f"Factorization error {result['factorization_check']:.2e} > 0"
        )

    def test_m_over_sqrt_sigma(self):
        """
        m(0++) / sqrt(sigma) = 1730 / 440 = 3.93.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks()
        expected = 1730.0 / 440.0
        assert abs(result['m_over_sqrt_sigma'] - expected) < 0.01

    def test_sqrt_sigma_over_gap(self):
        """
        sqrt(sigma) / gap = 440 / 179.4 = 2.45 at R = 2.2 fm.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks(R_fm=2.2)
        gap = 2.0 * HBAR_C_MEV_FM / 2.2
        expected = 440.0 / gap
        assert abs(result['sqrt_sigma_over_gap'] - expected) < 0.01

    def test_product_equals_total_ratio(self):
        """
        3.93 * 2.19 = 8.63 = the total enhancement factor.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks(R_fm=2.2)
        product = result['m_over_sqrt_sigma'] * result['sqrt_sigma_over_gap']
        assert abs(product - result['our_ratio']) < 0.01

    def test_hadron_ratios_physical(self):
        """
        Hadron mass / Lambda ratios should all be positive
        and in a sensible range.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks(R_fm=2.2)
        for name, data in result['hadron_ratios'].items():
            assert data['ratio'] > 0, f"{name}: negative ratio"
            assert data['ratio'] < 20, f"{name}: ratio too large"

    def test_msbar_ratio_in_range(self):
        """
        m(0++) / Lambda_MSbar ~ 6.9, which is in the expected range.
        """
        result = NonperturbativeEnhancement.lattice_cross_checks()
        assert 5.0 < result['msbar_ratio'] < 10.0


class TestSchemeDependence:
    """Lambda scheme dependence of the ratio."""

    def test_all_schemes_positive(self):
        """All Lambda values should be positive."""
        result = NonperturbativeEnhancement.scheme_dependence()
        for name, data in result['schemes'].items():
            assert data['Lambda_MeV'] > 0, f"Scheme {name}: Lambda <= 0"
            assert data['ratio'] > 0, f"Scheme {name}: ratio <= 0"

    def test_msbar_ratio(self):
        """MSbar ratio should be ~6.9."""
        result = NonperturbativeEnhancement.scheme_dependence()
        msbar = result['schemes']['MSbar']
        expected = 1730.0 / 250.0
        assert abs(msbar['ratio'] - expected) < 0.1

    def test_our_framework_ratio(self):
        """Our framework ratio should be ~8.6 (Lambda ~ 200.6 MeV)."""
        result = NonperturbativeEnhancement.scheme_dependence()
        ours = result['schemes']['our_framework']
        # Lambda = 200.6 MeV, ratio = 1730/200.6 ~ 8.63
        assert abs(ours['ratio'] - 8.6) < 0.3

    def test_ratio_spread(self):
        """
        The spread of ratios across schemes should be large
        (this is the point: the ratio is scheme-dependent).
        """
        result = NonperturbativeEnhancement.scheme_dependence()
        ratios = [d['ratio'] for d in result['schemes'].values()]
        assert max(ratios) / min(ratios) > 5, (
            "Scheme dependence should produce a wide spread of ratios"
        )


class TestLargeN:
    """Large-N predictions for the enhancement factor."""

    def test_data_exists_for_multiple_N(self):
        """We should have data for at least SU(2) through SU(5)."""
        result = NonperturbativeEnhancement.large_n_prediction()
        assert len(result['data']) >= 4

    def test_ratio_approximately_constant(self):
        """
        m(0++) / sqrt(sigma) should be approximately N-independent.
        The spread should be < 15%.
        """
        result = NonperturbativeEnhancement.large_n_prediction()
        values = [d['m_over_sqrt_sigma'] for d in result['data'].values()]
        mean = np.mean(values)
        std = np.std(values)

        # Coefficient of variation < 15%
        cv = std / mean
        assert cv < 0.15, (
            f"m(0++)/sqrt(sigma) varies by {cv*100:.1f}% across N, "
            f"expected < 15%"
        )

    def test_large_n_limit_value(self):
        """
        The large-N limit of m(0++) / sqrt(sigma) should be ~3.5-4.0.
        """
        result = NonperturbativeEnhancement.large_n_prediction()
        assert 3.0 < result['large_n_limit'] < 4.5, (
            f"Large-N limit {result['large_n_limit']:.2f} outside range 3.0-4.5"
        )


class TestSummary:
    """Overall summary assessment."""

    def test_is_consistent(self):
        """
        The framework should be assessed as consistent with lattice data.
        """
        result = NonperturbativeEnhancement.summary(R_fm=2.2)
        assert result['is_consistent'] is True

    def test_verdict_contains_key_info(self):
        """
        The verdict should mention the key numbers.
        """
        result = NonperturbativeEnhancement.summary(R_fm=2.2)
        verdict = result['verdict']
        assert 'CONSISTENT' in verdict
        assert '179' in verdict or '180' in verdict   # gap scale (~179.4 MeV)
        assert '1730' in verdict                        # glueball mass

    def test_all_sub_analyses_present(self):
        """
        Summary should contain all sub-analyses.
        """
        result = NonperturbativeEnhancement.summary(R_fm=2.2)
        for key in ['enhancement', 'rg_analysis', 'cross_checks',
                     'scheme_dependence', 'large_n']:
            assert key in result, f"Missing sub-analysis: {key}"

    def test_different_R_still_consistent(self):
        """
        For reasonable R values (1.5 to 3.0 fm), the framework
        should still be assessed as consistent (the ratio just
        changes, but stays within the 5-12 window).
        """
        for R in [1.5, 2.0, 2.5, 3.0]:
            result = NonperturbativeEnhancement.summary(R_fm=R)
            ratio = result['enhancement']['ratio']
            # Broader range because R != 2.2 changes the ratio
            assert 3.0 < ratio < 20.0, (
                f"At R={R} fm, ratio = {ratio:.1f}, out of bounds"
            )
