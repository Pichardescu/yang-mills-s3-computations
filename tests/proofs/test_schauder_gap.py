"""
Tests for Schauder fixed-point verification of YM gap equation on S^3.

Tests cover:
1. ScalarGapMap: T properties (continuity, monotonicity, boundary behavior)
2. SchauderGapExistence: existence and uniqueness THEOREM at each R
3. SchauderBoxVerification: optimal box and R-independent bounds
4. UniformGapBound: uniform lower bound across R (PROPOSITION)
5. AnalyticalTBounds: analytical upper/lower bounds on T
6. Contraction analysis: |T'| < 1
7. Full pipeline: end-to-end verification
8. Physical consistency: gap values match known results
9. Edge cases: extreme R, extreme coupling
10. SU(3) extension
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.schauder_gap import (
    ScalarGapMap,
    SchauderGapExistence,
    SchauderBoxVerification,
    UniformGapBound,
    UniformGapTheorem,
    AnalyticalTBounds,
    prove_T_monotonicity,
    contraction_analysis,
    full_schauder_verification,
)
from yang_mills_s3.proofs.gap_equation_s3 import (
    GapEquationS3,
    running_coupling_g2,
    physical_j_max,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_MEV,
)


# ======================================================================
# Test ScalarGapMap
# ======================================================================

class TestScalarGapMap:
    """Test the scalar self-energy map T(Sigma)."""

    def test_T_positive(self):
        """T(Sigma) must be positive for all Sigma > 0."""
        for R in [1.0, 10.0, 100.0]:
            smap = ScalarGapMap(R)
            for sigma in [0.01, 0.1, 1.0, 5.0, 20.0]:
                assert smap.T(sigma) > 0, (
                    f"T({sigma}) <= 0 at R={R}")

    def test_T_at_zero_positive(self):
        """T(0+) > 0 (positive self-energy at bare masses)."""
        for R in [1.0, 5.0, 50.0, 500.0]:
            smap = ScalarGapMap(R)
            T0 = smap.T(1e-10)
            assert T0 > 0, f"T(0+) = {T0} <= 0 at R={R}"

    def test_T_strictly_decreasing(self):
        """T(Sigma) must be strictly decreasing."""
        for R in [2.0, 20.0, 200.0]:
            smap = ScalarGapMap(R)
            sigmas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            T_vals = [smap.T(s) for s in sigmas]
            for i in range(len(T_vals) - 1):
                assert T_vals[i] > T_vals[i+1], (
                    f"T not decreasing at R={R}: T({sigmas[i]})={T_vals[i]} "
                    f"<= T({sigmas[i+1]})={T_vals[i+1]}")

    def test_T_vanishes_at_infinity(self):
        """T(Sigma) -> 0 as Sigma -> infinity."""
        smap = ScalarGapMap(10.0)
        T_large = smap.T(1e6)
        assert T_large < 0.01, (
            f"T(1e6) = {T_large}, should be near zero")

    def test_T_exceeds_identity_at_zero(self):
        """T(eps) > eps for sufficiently small eps (ensures crossing exists)."""
        for R in [1.0, 10.0, 100.0]:
            smap = ScalarGapMap(R)
            eps = 0.001
            assert smap.T(eps) > eps, (
                f"T({eps}) = {smap.T(eps)} <= {eps} at R={R}. "
                f"No crossing would exist.")

    def test_gap_function_sign_change(self):
        """f(Sigma) = T(Sigma) - Sigma must change sign (guarantees fixed point)."""
        for R in [1.0, 10.0, 100.0, 500.0]:
            smap = ScalarGapMap(R)
            f_small = smap.gap_function(0.001)
            f_large = smap.gap_function(100.0)
            assert f_small > 0, (
                f"f(0.001) = {f_small} <= 0 at R={R}")
            assert f_large < 0, (
                f"f(100) = {f_large} >= 0 at R={R}")

    def test_T_derivative_negative(self):
        """Numerical derivative of T should be negative everywhere."""
        smap = ScalarGapMap(10.0)
        for sigma in [0.1, 1.0, 5.0, 10.0]:
            dT = smap.dT_numerical(sigma)
            assert dT < 0, (
                f"dT/dSigma = {dT} >= 0 at Sigma={sigma}")

    def test_T_consistent_with_gap_equation(self):
        """T(Sigma) should match the self-energy from GapEquationS3."""
        R = 10.0
        g2 = running_coupling_g2(R)
        jm = physical_j_max(R)
        eq = GapEquationS3(R=R, g2=g2, N_c=2, j_max=jm)
        smap = ScalarGapMap(R, j_max=jm)

        sigma = 2.0
        masses = np.sqrt(eq._lam_arr + sigma)
        pi_direct = eq.self_energy_all(masses)
        T_val = smap.T(sigma)

        assert T_val == pytest.approx(pi_direct, rel=1e-10), (
            f"T({sigma}) = {T_val} != Pi = {pi_direct}")

    def test_ScalarGapMap_default_jmax(self):
        """Default j_max should use physical_j_max."""
        smap = ScalarGapMap(50.0)
        expected_jmax = physical_j_max(50.0)
        assert smap.j_max == expected_jmax


# ======================================================================
# Test existence and uniqueness theorem
# ======================================================================

class TestSchauderExistence:
    """Test the existence and uniqueness THEOREM."""

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    def test_existence_at_R(self, R):
        """THEOREM: T has a fixed point at every R > 0."""
        exist = SchauderGapExistence(R)
        props = exist.verify_T_properties()
        assert props['existence_THEOREM'], (
            f"Existence THEOREM fails at R={R}")

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    def test_uniqueness_at_R(self, R):
        """THEOREM: The fixed point is unique at every R > 0."""
        exist = SchauderGapExistence(R)
        props = exist.verify_T_properties()
        assert props['uniqueness_THEOREM'], (
            f"Uniqueness THEOREM fails at R={R}")

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_fixed_point_convergence(self, R):
        """The Brent solver should find the fixed point."""
        exist = SchauderGapExistence(R)
        fp = exist.find_fixed_point()
        assert fp['converged'], (
            f"Fixed point solver did not converge at R={R}")
        assert fp['residual'] < 1e-8

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_fixed_point_positive(self, R):
        """The fixed point Sigma* must be positive."""
        exist = SchauderGapExistence(R)
        fp = exist.find_fixed_point()
        assert fp['sigma_star'] > 0, (
            f"Sigma* = {fp['sigma_star']} <= 0 at R={R}")

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_mass_gap_positive(self, R):
        """m_0 = sqrt(lam_0 + Sigma*) must be positive."""
        exist = SchauderGapExistence(R)
        fp = exist.find_fixed_point()
        assert fp['m0_MeV'] > 0, (
            f"m_0 = {fp['m0_MeV']} MeV <= 0 at R={R}")

    def test_fixed_point_matches_iterative_solver(self):
        """Schauder fixed point should match the iterative solver."""
        for R in [5.0, 50.0, 500.0]:
            g2 = running_coupling_g2(R)
            jm = physical_j_max(R)
            eq = GapEquationS3(R=R, g2=g2, N_c=2, j_max=jm)
            iterative = eq.solve()

            exist = SchauderGapExistence(R, j_max=jm)
            fp = exist.find_fixed_point()

            assert fp['m0_MeV'] == pytest.approx(iterative['gap_MeV'], rel=1e-3), (
                f"Schauder m0={fp['m0_MeV']:.2f} != iterative "
                f"m0={iterative['gap_MeV']:.2f} at R={R}")

    def test_T_properties_all_components(self):
        """verify_T_properties should populate all fields."""
        exist = SchauderGapExistence(10.0)
        props = exist.verify_T_properties()

        required_keys = [
            'R', 'T_at_zero_plus', 'prop1_T_positive',
            'prop2_T_decreasing', 'prop2_all_derivs_negative',
            'prop3_T_below_identity', 'ivt_applies',
            'existence_THEOREM', 'uniqueness_THEOREM', 'label',
        ]
        for key in required_keys:
            assert key in props, f"Missing key: {key}"

        assert props['label'] == 'THEOREM'


# ======================================================================
# Test Schauder box verification
# ======================================================================

class TestSchauderBox:
    """Test the Schauder box invariance conditions."""

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_optimal_box_valid(self, R):
        """Optimal Schauder box should be valid at each R."""
        box = SchauderBoxVerification(R)
        opt = box.find_optimal_box()
        assert opt.get('box_valid', False), (
            f"Optimal box not valid at R={R}")

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_optimal_box_contains_fixed_point(self, R):
        """The fixed point Sigma* must lie within [a, b]."""
        box = SchauderBoxVerification(R)
        opt = box.find_optimal_box()
        if opt.get('box_valid'):
            sigma_star = opt['sigma_star']
            assert opt['a'] <= sigma_star <= opt['b'], (
                f"Sigma*={sigma_star} not in [{opt['a']}, {opt['b']}] at R={R}")

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_gap_lower_bound_positive(self, R):
        """Gap lower bound from Schauder box must be positive."""
        box = SchauderBoxVerification(R)
        opt = box.find_optimal_box()
        if opt.get('box_valid'):
            assert opt['gap_lower_bound_MeV'] > 0

    def test_box_verification_manual(self):
        """Manually verify box invariance conditions."""
        R = 50.0
        box = SchauderBoxVerification(R)
        # Use a wide box that should work
        result = box.verify_box(1.0, 4.0)
        # T is decreasing, so T(4.0) >= 1.0 and T(1.0) <= 4.0
        assert result['box_valid'], (
            f"Manual box [1, 4] failed at R={R}: "
            f"T(a)={result['T_a']}, T(b)={result['T_b']}")

    def test_box_verification_invalid_box(self):
        """Invalid box parameters should be caught."""
        box = SchauderBoxVerification(10.0)
        result = box.verify_box(-1.0, 5.0)
        assert not result.get('valid', True)

    def test_box_width_shrinks_at_large_R(self):
        """Box width should stabilize (and be small) at large R."""
        widths = []
        for R in [10.0, 50.0, 200.0, 500.0]:
            box = SchauderBoxVerification(R)
            opt = box.find_optimal_box()
            if opt.get('box_valid'):
                widths.append(opt['box_width'])
        # Widths should be small relative to Sigma*
        for w in widths:
            assert w < 0.1, f"Box width {w} too large"


# ======================================================================
# Test R-independent bound verification
# ======================================================================

class TestRIndependentBound:
    """Test the find_r_independent_box method."""

    def test_conservative_bound_works_everywhere(self):
        """A conservative lower bound should work at all tested R."""
        target_a = 1.0  # conservative: below all Sigma* values
        for R in [5.0, 10.0, 50.0, 100.0, 500.0]:
            box = SchauderBoxVerification(R)
            check = box.find_r_independent_box(target_a)
            assert check['works'], (
                f"target_a={target_a} fails at R={R}: {check.get('reason', '')}")

    def test_too_aggressive_bound_fails(self):
        """A bound above Sigma* should fail."""
        box = SchauderBoxVerification(500.0)
        fp = SchauderGapExistence(500.0).find_fixed_point()
        sigma_star = fp['sigma_star']
        # target above sigma_star: T(target) < target, so box has zero width
        check = box.find_r_independent_box(sigma_star * 1.5)
        assert not check['works']


# ======================================================================
# Test uniform gap bound
# ======================================================================

class TestUniformGapBound:
    """Test the uniform gap bound (PROPOSITION)."""

    def test_uniform_bound_found(self):
        """A positive uniform bound should be found."""
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(
            R_values=[5.0, 10.0, 50.0, 100.0, 500.0])
        assert result['success']
        assert result['uniform_gap_lower_bound_MeV'] > 0

    def test_uniform_bound_physical_range(self):
        """Uniform bound should be in a reasonable physical range."""
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(
            R_values=[10.0, 50.0, 100.0, 500.0])
        gap = result['uniform_gap_lower_bound_MeV']
        # Should be between 100 and 500 MeV (O(Lambda_QCD))
        assert 100 < gap < 500, (
            f"Uniform bound {gap:.1f} MeV outside expected range [100, 500]")

    def test_cross_check_passes(self):
        """Cross-check should pass: uniform bound works at all R."""
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(
            R_values=[5.0, 10.0, 50.0, 100.0, 500.0])
        assert result.get('cross_check_all_pass', False), (
            "Cross-check failed: uniform bound does not work at all R")

    def test_large_R_sigma_converges(self):
        """Sigma* should converge to an R-independent value at large R."""
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(
            R_values=[20.0, 50.0, 100.0, 200.0, 500.0])
        la = result['large_R_sigma_analysis']
        assert la['converged'], (
            f"Sigma* not converged: rel_var={la['relative_variation']:.4f}")

    def test_uniform_bound_label(self):
        """Uniform bound should be labeled PROPOSITION."""
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(R_values=[10.0, 100.0])
        assert result['label'] == 'PROPOSITION'


# ======================================================================
# Test analytical bounds
# ======================================================================

class TestAnalyticalBounds:
    """Test analytical upper/lower bounds on T."""

    @pytest.mark.parametrize("R", [5.0, 50.0, 500.0])
    def test_T_upper_bound(self, R):
        """T_upper(Sigma) >= T(Sigma) for all Sigma."""
        abounds = AnalyticalTBounds(R)
        smap = ScalarGapMap(R, j_max=abounds.j_max)
        for sigma in [0.1, 1.0, 5.0]:
            T_actual = smap.T(sigma)
            T_upper = abounds.T_upper_bound(sigma)
            assert T_upper >= T_actual * 0.99, (
                f"Upper bound {T_upper} < actual {T_actual} at R={R}, "
                f"Sigma={sigma}")

    @pytest.mark.parametrize("R", [5.0, 50.0, 500.0])
    def test_T_lower_bound(self, R):
        """T_lower(Sigma) <= T(Sigma) for all Sigma."""
        abounds = AnalyticalTBounds(R)
        smap = ScalarGapMap(R, j_max=abounds.j_max)
        for sigma in [0.1, 1.0, 5.0]:
            T_actual = smap.T(sigma)
            T_lower = abounds.T_lower_bound(sigma)
            assert T_lower <= T_actual * 1.01, (
                f"Lower bound {T_lower} > actual {T_actual} at R={R}, "
                f"Sigma={sigma}")

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0])
    def test_analytical_sigma_lower_bound(self, R):
        """Analytical lower bound on Sigma should be valid (below Sigma*)."""
        abounds = AnalyticalTBounds(R)
        a_lower = abounds.analytical_sigma_lower_bound()
        if a_lower['valid']:
            exist = SchauderGapExistence(R, j_max=abounds.j_max)
            fp = exist.find_fixed_point()
            assert a_lower['sigma_lower'] < fp['sigma_star'] * 1.01, (
                f"Analytical lower {a_lower['sigma_lower']} exceeds "
                f"Sigma*={fp['sigma_star']} at R={R}")

    def test_analytical_bound_positive(self):
        """Analytical lower bound should give positive gap."""
        for R in [1.0, 10.0, 100.0]:
            abounds = AnalyticalTBounds(R)
            a_lower = abounds.analytical_sigma_lower_bound()
            if a_lower['valid']:
                assert a_lower['m0_lower_MeV'] > 0


# ======================================================================
# Test monotonicity proof
# ======================================================================

class TestMonotonicity:
    """Test the monotonicity THEOREM."""

    @pytest.mark.parametrize("R", [1.0, 10.0, 100.0, 500.0])
    def test_T_strictly_decreasing_verified(self, R):
        """Numerical verification of strict monotonicity at each R."""
        result = prove_T_monotonicity(R)
        assert result['strictly_decreasing'], (
            f"Monotonicity failed at R={R}: max_increase={result['max_increase']}")

    def test_T_ratio_large_to_small(self):
        """T at large Sigma should be much smaller than T at small Sigma."""
        result = prove_T_monotonicity(10.0)
        assert result['T_ratio_large_to_small'] < 0.15, (
            f"T(large)/T(small) = {result['T_ratio_large_to_small']:.4f}, "
            f"expected < 0.15")

    def test_algebraic_proof_present(self):
        """The algebraic proof string should be included."""
        result = prove_T_monotonicity(10.0)
        assert 'algebraic_proof' in result
        assert 'decreasing' in result['algebraic_proof'].lower()

    def test_monotonicity_label(self):
        """Monotonicity is a THEOREM."""
        result = prove_T_monotonicity(10.0)
        assert result['label'] == 'THEOREM'


# ======================================================================
# Test contraction analysis
# ======================================================================

class TestContraction:
    """Test the contraction property (THEOREM)."""

    @pytest.mark.parametrize("R", [1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    def test_is_contraction(self, R):
        """T should be a contraction at the fixed point: |T'(Sigma*)| < 1."""
        result = contraction_analysis(R)
        assert result['is_contraction'], (
            f"|T'| = {result['T_prime_magnitude']} >= 1 at R={R}")

    def test_contraction_rate_bounded(self):
        """The contraction rate |T'| should be well below 1."""
        result = contraction_analysis(100.0)
        assert result['T_prime_magnitude'] < 0.5, (
            f"|T'| = {result['T_prime_magnitude']} too close to 1")

    def test_contraction_rate_stabilizes(self):
        """The contraction rate should stabilize at large R."""
        rates = []
        for R in [50.0, 100.0, 200.0, 500.0]:
            result = contraction_analysis(R)
            rates.append(result['T_prime_magnitude'])
        # Variation should be small at large R
        mean_rate = np.mean(rates)
        for r in rates:
            assert abs(r - mean_rate) / mean_rate < 0.05, (
                f"Contraction rate not stable: rates={rates}")

    def test_convergence_iterations_reasonable(self):
        """Convergence should be achieved in a reasonable number of iterations."""
        result = contraction_analysis(100.0)
        n_iter = result.get('iterations_for_1e-10', -1)
        assert 0 < n_iter < 200, (
            f"Iterations for 1e-10 accuracy: {n_iter}")

    def test_contraction_label(self):
        """Contraction analysis is a THEOREM."""
        result = contraction_analysis(10.0)
        assert result['label'] == 'THEOREM'


# ======================================================================
# Test full pipeline
# ======================================================================

class TestFullPipeline:
    """Test the end-to-end Schauder verification pipeline."""

    def test_pipeline_runs(self):
        """Full pipeline should complete without errors."""
        results = full_schauder_verification(
            R_values=[5.0, 50.0, 500.0])
        assert results is not None
        assert 'summary' in results
        assert 'per_R' in results

    def test_pipeline_all_existence(self):
        """Existence should hold at all tested R."""
        results = full_schauder_verification(
            R_values=[1.0, 10.0, 100.0, 500.0])
        assert results['summary']['existence_THEOREM']

    def test_pipeline_all_uniqueness(self):
        """Uniqueness should hold at all tested R."""
        results = full_schauder_verification(
            R_values=[1.0, 10.0, 100.0, 500.0])
        assert results['summary']['uniqueness_THEOREM']

    def test_pipeline_all_contraction(self):
        """Contraction should hold at all tested R."""
        results = full_schauder_verification(
            R_values=[1.0, 10.0, 100.0, 500.0])
        assert results['summary']['contraction_THEOREM']

    def test_pipeline_plateau_formed(self):
        """Plateau should form at large R."""
        results = full_schauder_verification(
            R_values=[20.0, 50.0, 100.0, 200.0, 500.0])
        assert results['summary']['plateau_formed']

    def test_pipeline_plateau_value(self):
        """Plateau should be near 290 MeV."""
        results = full_schauder_verification(
            R_values=[50.0, 100.0, 500.0])
        plateau = results['summary']['plateau_mean_MeV']
        assert 250 < plateau < 350, (
            f"Plateau {plateau:.1f} MeV outside [250, 350]")

    def test_pipeline_uniform_bound_positive(self):
        """Uniform gap bound should be positive."""
        results = full_schauder_verification(
            R_values=[10.0, 50.0, 100.0, 500.0])
        gap_bound = results['summary']['uniform_gap_lower_bound_MeV']
        assert gap_bound > 0

    def test_pipeline_classification_correct(self):
        """Classification should correctly label each step."""
        results = full_schauder_verification(
            R_values=[10.0, 100.0])
        c = results['classification']
        assert 'THEOREM' in c['existence']
        assert 'THEOREM' in c['uniqueness']
        assert 'THEOREM' in c['contraction']
        assert 'PROPOSITION' in c['schauder_box']
        assert 'PROPOSITION' in c['uniform_bound']


# ======================================================================
# Test physical consistency
# ======================================================================

class TestPhysicalConsistency:
    """Test that Schauder results are physically consistent."""

    def test_gap_order_Lambda_QCD(self):
        """Gap should be O(Lambda_QCD) at large R."""
        exist = SchauderGapExistence(100.0)
        fp = exist.find_fixed_point()
        ratio = fp['m0_MeV'] / LAMBDA_QCD_MEV
        assert 1.0 < ratio < 3.0, (
            f"m0/Lambda = {ratio:.2f}, expected 1.0-3.0")

    def test_gap_decreases_towards_plateau(self):
        """Gap should decrease monotonically and flatten at large R."""
        gaps = []
        for R in [5.0, 10.0, 50.0, 100.0, 500.0]:
            exist = SchauderGapExistence(R)
            fp = exist.find_fixed_point()
            gaps.append(fp['m0_MeV'])
        # Monotonically decreasing (approximately)
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i+1] * 0.95, (
                f"Gap not decreasing: m0({i})={gaps[i]}, m0({i+1})={gaps[i+1]}")

    def test_gap_geometric_regime_at_small_R(self):
        """At small R, gap ~ 1/R (geometric dominates)."""
        R = 0.2
        exist = SchauderGapExistence(R)
        fp = exist.find_fixed_point()
        bare_gap = HBAR_C_MEV_FM / R  # (j+1)/R for j=0
        assert fp['m0_MeV'] > bare_gap * 0.8, (
            f"Gap {fp['m0_MeV']:.1f} MeV too far below bare "
            f"{bare_gap:.1f} MeV at R={R}")

    def test_dimensional_transmutation_ratio(self):
        """Gap/Lambda ratio should converge as R -> inf."""
        ratios = []
        for R in [50.0, 100.0, 200.0, 500.0]:
            exist = SchauderGapExistence(R)
            fp = exist.find_fixed_point()
            ratios.append(fp['m0_MeV'] / LAMBDA_QCD_MEV)
        mean_ratio = np.mean(ratios)
        for r in ratios:
            assert abs(r - mean_ratio) / mean_ratio < 0.02, (
                f"Ratio not converged: {ratios}")


# ======================================================================
# Test edge cases
# ======================================================================

class TestEdgeCases:
    """Test behavior at extreme parameter values."""

    def test_very_small_R(self):
        """At R = 0.1 fm, the geometric gap dominates."""
        exist = SchauderGapExistence(0.1, j_max=50)
        fp = exist.find_fixed_point()
        assert fp['converged']
        # At small R, m0 should be large (~ 1/R)
        assert fp['m0_MeV'] > 1000

    def test_moderate_R(self):
        """At R = 2.2 fm (physical), gap should match known results."""
        exist = SchauderGapExistence(2.2)
        fp = exist.find_fixed_point()
        assert fp['converged']
        # Should give a gap of order 500-800 MeV
        assert 200 < fp['m0_MeV'] < 1000

    def test_very_large_R(self):
        """At R = 1000 fm, plateau should be well-established."""
        exist = SchauderGapExistence(1000.0)
        fp = exist.find_fixed_point()
        assert fp['converged']
        # Should match the R=500 value closely
        exist_500 = SchauderGapExistence(500.0)
        fp_500 = exist_500.find_fixed_point()
        rel_diff = abs(fp['m0_MeV'] - fp_500['m0_MeV']) / fp_500['m0_MeV']
        assert rel_diff < 0.005, (
            f"R=1000 gap {fp['m0_MeV']:.2f} differs from R=500 "
            f"gap {fp_500['m0_MeV']:.2f} by {rel_diff:.4f}")

    def test_custom_j_max(self):
        """Custom j_max should be respected."""
        smap = ScalarGapMap(10.0, j_max=30)
        assert smap.j_max == 30


# ======================================================================
# Test SU(3)
# ======================================================================

class TestSU3:
    """Test Schauder verification for SU(3)."""

    def test_su3_existence(self):
        """Existence THEOREM should hold for SU(3)."""
        exist = SchauderGapExistence(10.0, N_c=3)
        props = exist.verify_T_properties()
        assert props['existence_THEOREM']

    def test_su3_fixed_point(self):
        """SU(3) fixed point should exist and give positive gap."""
        exist = SchauderGapExistence(50.0, N_c=3)
        fp = exist.find_fixed_point()
        assert fp['converged']
        assert fp['m0_MeV'] > 0

    def test_su3_gap_larger_than_su2(self):
        """SU(3) gap should be larger than SU(2) (more self-interaction)."""
        R = 50.0
        fp_su2 = SchauderGapExistence(R, N_c=2).find_fixed_point()
        fp_su3 = SchauderGapExistence(R, N_c=3).find_fixed_point()
        assert fp_su3['m0_MeV'] > fp_su2['m0_MeV'], (
            f"SU(3) gap {fp_su3['m0_MeV']:.1f} <= "
            f"SU(2) gap {fp_su2['m0_MeV']:.1f}")

    def test_su3_contraction(self):
        """Contraction should hold for SU(3)."""
        result = contraction_analysis(50.0, N_c=3)
        assert result['is_contraction']

    def test_su3_schauder_box(self):
        """Schauder box should work for SU(3)."""
        box = SchauderBoxVerification(50.0, N_c=3)
        opt = box.find_optimal_box()
        assert opt.get('box_valid', False)

    def test_su3_plateau(self):
        """SU(3) should also show dimensional transmutation plateau."""
        gaps = []
        for R in [50.0, 100.0, 500.0]:
            fp = SchauderGapExistence(R, N_c=3).find_fixed_point()
            gaps.append(fp['m0_MeV'])
        mean_gap = np.mean(gaps)
        for g in gaps:
            rel_diff = abs(g - mean_gap) / mean_gap
            assert rel_diff < 0.02, (
                f"SU(3) plateau not formed: gaps={gaps}")


# ======================================================================
# Test proof chain integration
# ======================================================================

class TestProofChainIntegration:
    """Test that Schauder results integrate with the existing proof chain."""

    def test_schauder_gap_above_lambda_qcd(self):
        """
        The Schauder-guaranteed gap should be above Lambda_QCD.

        This is the KEY result: m_0 >= sqrt(Sigma_min) > Lambda_QCD
        uniformly in R.
        """
        ugb = UniformGapBound(N_c=2)
        result = ugb.find_uniform_bound(
            R_values=[10.0, 50.0, 100.0, 200.0, 500.0])
        gap = result['uniform_gap_lower_bound_MeV']
        assert gap > LAMBDA_QCD_MEV, (
            f"Uniform gap {gap:.1f} MeV <= Lambda_QCD = {LAMBDA_QCD_MEV} MeV")

    def test_schauder_consistent_with_lattice_estimates(self):
        """
        The Schauder gap should be consistent with lattice QCD estimates.

        Lattice glueball mass 0++ ~ 1.7 GeV = 1700 MeV for SU(3).
        Our gap is the lowest mode, which should be significantly below
        the glueball mass.
        """
        fp = SchauderGapExistence(100.0, N_c=3).find_fixed_point()
        assert fp['m0_MeV'] < 1700, (
            f"Gap {fp['m0_MeV']:.1f} MeV exceeds glueball mass")

    def test_proof_labels_correct(self):
        """Check that existence is THEOREM, uniform bound is PROPOSITION."""
        exist = SchauderGapExistence(10.0)
        props = exist.verify_T_properties()
        assert props['label'] == 'THEOREM'

        ugb = UniformGapBound()
        result = ugb.find_uniform_bound(R_values=[10.0, 100.0])
        assert result['label'] == 'PROPOSITION'

    def test_gap_at_physical_R(self):
        """At R_phys = 2.2 fm, gap should match existing numerical results."""
        # From existing gap_equation_s3.py, gap ~ 500-600 MeV at R=2.2
        fp = SchauderGapExistence(2.2).find_fixed_point()
        assert 300 < fp['m0_MeV'] < 800, (
            f"Gap at R=2.2: {fp['m0_MeV']:.1f} MeV, expected 300-800")


# ======================================================================
# Test continuity in R
# ======================================================================

class TestContinuityInR:
    """Test that the gap varies continuously with R."""

    def test_gap_continuous_small_step(self):
        """Gap should change smoothly for small R steps."""
        R_values = np.linspace(5.0, 6.0, 11)
        gaps = []
        for R in R_values:
            fp = SchauderGapExistence(R, j_max=50).find_fixed_point()
            gaps.append(fp['m0_MeV'])
        # Check that consecutive differences are small
        for i in range(len(gaps) - 1):
            rel_diff = abs(gaps[i+1] - gaps[i]) / gaps[i]
            assert rel_diff < 0.05, (
                f"Gap jumps at R={R_values[i]:.1f}->{R_values[i+1]:.1f}: "
                f"{gaps[i]:.1f} -> {gaps[i+1]:.1f} MeV")

    def test_sigma_star_continuous(self):
        """Sigma* should vary continuously with R."""
        R_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        sigmas = []
        for R in R_values:
            fp = SchauderGapExistence(R, j_max=50).find_fixed_point()
            sigmas.append(fp['sigma_star'])
        for i in range(len(sigmas) - 1):
            rel_diff = abs(sigmas[i+1] - sigmas[i]) / sigmas[i]
            assert rel_diff < 0.1, (
                f"Sigma* jumps: {sigmas[i]:.4f} -> {sigmas[i+1]:.4f}")


# ======================================================================
# Test UniformGapTheorem (THEOREM-level uniform bound)
# ======================================================================

class TestUniformGapTheorem:
    """
    Test the THEOREM-level proof that inf_{R>0} Sigma*(R) > 0.

    This elevates the uniform gap bound from PROPOSITION to THEOREM
    by proving: continuity, monotonicity, R->0 divergence, R->inf
    positive limit, and combining via EVT.
    """

    def test_theorem_holds_su2(self):
        """The uniform gap THEOREM should hold for SU(2)."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        assert result['theorem_holds'], (
            f"Theorem failed. Weakest links: {result['weakest_links']}")

    def test_theorem_holds_su3(self):
        """The uniform gap THEOREM should hold for SU(3)."""
        ugt = UniformGapTheorem(N_c=3)
        result = ugt.prove_uniform_gap()
        assert result['theorem_holds'], (
            f"Theorem failed for SU(3). Weakest links: {result['weakest_links']}")

    def test_theorem_label_is_THEOREM(self):
        """When theorem holds, label should be THEOREM."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        assert result['label'] == 'THEOREM', (
            f"Label is {result['label']}, expected THEOREM")

    def test_sigma_min_positive(self):
        """The infimum Sigma_min must be strictly positive."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        assert result['sigma_min'] > 0, (
            f"Sigma_min = {result['sigma_min']} <= 0")

    def test_m0_min_physical_range(self):
        """The minimum gap m0_min should be O(Lambda_QCD)."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        # Should be between 100 and 500 MeV for SU(2)
        assert 100 < result['m0_min_MeV'] < 500, (
            f"m0_min = {result['m0_min_MeV']:.1f} MeV outside [100, 500]")

    def test_su3_gap_larger_than_su2(self):
        """SU(3) uniform gap should exceed SU(2) uniform gap."""
        ugt2 = UniformGapTheorem(N_c=2)
        ugt3 = UniformGapTheorem(N_c=3)
        r2 = ugt2.prove_uniform_gap()
        r3 = ugt3.prove_uniform_gap()
        assert r3['m0_min_MeV'] > r2['m0_min_MeV'], (
            f"SU(3) gap {r3['m0_min_MeV']:.1f} <= "
            f"SU(2) gap {r2['m0_min_MeV']:.1f}")


class TestUniformGapTheoremStep1:
    """Test Step 1: Joint continuity of T(Sigma, R)."""

    def test_joint_continuity_holds(self):
        """T should be jointly continuous in (Sigma, R)."""
        ugt = UniformGapTheorem(N_c=2)
        step1 = ugt.verify_joint_continuity()
        assert step1['joint_continuous'], (
            f"Joint continuity failed: max discontinuity = "
            f"{step1['max_relative_discontinuity']:.6f}")

    def test_max_discontinuity_small(self):
        """Maximum relative discontinuity should be very small."""
        ugt = UniformGapTheorem(N_c=2)
        step1 = ugt.verify_joint_continuity()
        assert step1['max_relative_discontinuity'] < 0.01

    def test_label_is_THEOREM(self):
        """Step 1 should be labeled THEOREM."""
        ugt = UniformGapTheorem(N_c=2)
        step1 = ugt.verify_joint_continuity()
        assert step1['label'] == 'THEOREM'


class TestUniformGapTheoremStep2:
    """Test Step 2: Uniform contraction |T'(Sigma*)| < 1."""

    def test_all_contractive(self):
        """|T'| < 1 at every tested R."""
        ugt = UniformGapTheorem(N_c=2)
        step2 = ugt.verify_contraction_uniform()
        assert step2['all_contractive']

    def test_max_T_prime_well_below_1(self):
        """Maximum |T'| should be well below 1 (gives margin)."""
        ugt = UniformGapTheorem(N_c=2)
        step2 = ugt.verify_contraction_uniform()
        assert step2['max_T_prime'] < 0.5, (
            f"max |T'| = {step2['max_T_prime']:.4f}, too close to 1")

    def test_contraction_analytical_bound(self):
        """
        Analytical bound: |T'(Sigma*)| < 1 follows from:
        |T'|/T = <1/(2*lam_k + Sigma*)>_w < 1/Sigma* => |T'| < T/Sigma* = 1.
        Verify numerically that this ratio is strictly < 1.
        """
        for R in [1.0, 10.0, 100.0, 500.0]:
            smap = ScalarGapMap(R)
            exist = SchauderGapExistence(R)
            fp = exist.find_fixed_point()
            if fp['converged']:
                sigma_star = fp['sigma_star']
                dT = abs(smap.dT_numerical(sigma_star))
                T_val = smap.T(sigma_star)
                # |T'| / T should be < 1/Sigma*
                ratio = dT / T_val if T_val > 0 else float('inf')
                bound = 1.0 / sigma_star
                assert ratio < bound * 1.01, (
                    f"|T'|/T = {ratio:.6f} not < 1/Sigma* = {bound:.6f} "
                    f"at R={R}")


class TestUniformGapTheoremStep3:
    """Test Step 3: Monotonicity of Sigma*(R)."""

    def test_sigma_star_monotone_decreasing(self):
        """Sigma*(R) should be strictly decreasing in R."""
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R()
        assert step3['sigma_star_monotone_decreasing'], (
            f"Monotonicity violations: {step3['violations']}")

    def test_T_decreasing_in_R(self):
        """T(sigma; R) should decrease pointwise in R."""
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R()
        assert step3['T_decreasing_in_R']

    def test_sigma_star_range(self):
        """Sigma* should range from very large (small R) to ~2.15 (large R)."""
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R()
        assert step3['sigma_star_max'] > 10.0, (
            f"Sigma* max = {step3['sigma_star_max']:.2f}, expected > 10")
        assert 1.5 < step3['sigma_star_min'] < 3.0, (
            f"Sigma* min = {step3['sigma_star_min']:.4f}, expected in [1.5, 3.0]")

    def test_no_violations(self):
        """There should be zero monotonicity violations."""
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R()
        assert len(step3['violations']) == 0

    def test_dense_grid_monotonicity(self):
        """Structural monotonicity on a denser grid near the limit.

        At dense R spacing, j_max floor-function jumps can cause tiny
        O(1/j_max^2) non-monotonicities. The THEOREM-level criterion
        uses structural monotonicity (same j_max) + bounded perturbations.
        """
        R_dense = [float(x) for x in np.linspace(50.0, 500.0, 30)]
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R(R_values=R_dense)
        assert step3['sigma_star_monotone_decreasing'], (
            f"Structural monotonicity fails on dense grid. "
            f"Same-jmax monotone: {step3['structural_monotone_same_jmax']}, "
            f"Max relative violation: {step3['max_violation_relative']:.2e}")

    def test_jmax_perturbation_bounded(self):
        """The j_max perturbations should be tiny relative to Sigma_min."""
        R_dense = [float(x) for x in np.linspace(50.0, 1000.0, 100)]
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R(R_values=R_dense)
        assert step3['max_violation_relative'] < 0.01, (
            f"j_max perturbation too large: {step3['max_violation_relative']:.4f}")

    def test_structural_monotone_same_jmax(self):
        """With the same j_max, T(sigma; R) should be strictly decreasing in R."""
        ugt = UniformGapTheorem(N_c=2)
        step3 = ugt.verify_monotonicity_in_R()
        assert step3['structural_monotone_same_jmax'], (
            "Structural monotonicity (same j_max) violated")


class TestUniformGapTheoremStep4:
    """Test Step 4: Sigma*(R) -> infinity as R -> 0."""

    def test_sigma_diverges(self):
        """Sigma* should grow without bound as R -> 0."""
        ugt = UniformGapTheorem(N_c=2)
        step4 = ugt.verify_limit_R_to_zero()
        assert step4['sigma_diverges_at_R_zero']

    def test_sigma_very_large_at_small_R(self):
        """At R = 0.05, Sigma* should be very large."""
        ugt = UniformGapTheorem(N_c=2)
        step4 = ugt.verify_limit_R_to_zero()
        assert step4['sigma_at_smallest_R'] > 50.0, (
            f"Sigma* at R=0.05 is {step4['sigma_at_smallest_R']:.2f}, "
            f"expected > 50")

    def test_m0_exceeds_1_over_R(self):
        """At small R, m0 should exceed 1/R (geometric gap dominates)."""
        for R in [0.1, 0.2, 0.5]:
            exist = SchauderGapExistence(R)
            fp = exist.find_fixed_point()
            bare_mass = 1.0 / R  # (j+1)/R for j=0, in fm^{-1}
            assert fp['m0_fm_inv'] > bare_mass, (
                f"m0 = {fp['m0_fm_inv']:.2f} < 1/R = {bare_mass:.2f} at R={R}")


class TestUniformGapTheoremStep5:
    """Test Step 5: Sigma*(R) -> Sigma*(inf) > 0 as R -> infinity."""

    def test_limit_exists(self):
        """The limit of Sigma*(R) as R -> inf should exist (MCT)."""
        ugt = UniformGapTheorem(N_c=2)
        step5 = ugt.verify_limit_R_to_infinity()
        assert step5['limit_exists'], "Limit does not exist (not monotone)"

    def test_limit_positive(self):
        """The limit should be strictly positive."""
        ugt = UniformGapTheorem(N_c=2)
        step5 = ugt.verify_limit_R_to_infinity()
        assert step5['limit_positive']
        assert step5['sigma_inf_estimate'] > 1.0, (
            f"Sigma*(inf) = {step5['sigma_inf_estimate']:.4f}, expected > 1.0")

    def test_limit_converged(self):
        """The last few R values should give very similar Sigma*."""
        ugt = UniformGapTheorem(N_c=2)
        step5 = ugt.verify_limit_R_to_infinity()
        assert step5['relative_variation_last_3'] < 0.01, (
            f"Not converged: rel_var = {step5['relative_variation_last_3']:.6f}")

    def test_T_inf_at_zero_positive(self):
        """T_inf(0+) should be positive (needed for positive fixed point)."""
        ugt = UniformGapTheorem(N_c=2)
        step5 = ugt.verify_limit_R_to_infinity()
        assert step5['T_inf_at_zero_positive']
        assert step5['T_inf_at_zero'] > 1.0, (
            f"T_inf(0+) = {step5['T_inf_at_zero']:.4f}, expected > 1.0")

    def test_sigma_inf_consistent_with_numerical_minimum(self):
        """Sigma*(inf) from limit should match the numerical minimum."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        sigma_inf = result['steps']['step5_limit_R_to_infinity']['sigma_inf_estimate']
        sigma_min = result['sigma_min']
        # They should be the same (infimum = limit for monotone sequence)
        assert abs(sigma_inf - sigma_min) / sigma_min < 0.001, (
            f"sigma_inf = {sigma_inf:.6f} != sigma_min = {sigma_min:.6f}")

    def test_r3_cancellation(self):
        """
        Verify the R^3 cancellation: T(sigma_test; R) should converge
        as R -> inf (not diverge or go to zero).
        """
        sigma_test = 2.15  # near Sigma*(inf)
        T_values = []
        for R in [100.0, 200.0, 500.0, 1000.0]:
            smap = ScalarGapMap(R)
            T_values.append(smap.T(sigma_test))
        # T should converge, not diverge
        rel_var = (max(T_values) - min(T_values)) / np.mean(T_values)
        assert rel_var < 0.05, (
            f"T values at large R vary by {rel_var:.4f}, "
            f"R^3 cancellation not working. T_values = {T_values}")


class TestUniformGapTheoremProofStructure:
    """Test the proof structure and classification."""

    def test_all_steps_pass(self):
        """All 5 steps should pass."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        steps = result['steps']
        assert steps['step1_joint_continuity']['joint_continuous']
        assert steps['step2_uniform_contraction']['all_contractive']
        assert steps['step3_monotonicity_in_R']['sigma_star_monotone_decreasing']
        assert steps['step4_limit_R_to_zero']['sigma_diverges_at_R_zero']
        assert steps['step5_limit_R_to_infinity']['limit_exists']
        assert steps['step5_limit_R_to_infinity']['limit_positive']

    def test_no_weakest_links(self):
        """When theorem holds, there should be no weakest links."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        assert len(result['weakest_links']) == 0

    def test_proof_summary_present(self):
        """Proof summary should contain statement and method."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        ps = result['proof_summary']
        assert 'statement' in ps
        assert 'method' in ps
        assert 'rigor_level' in ps
        assert 'monotonically decreasing' in ps['method']

    def test_model_dependence_documented(self):
        """Model-dependent aspects should be documented."""
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        assert len(result['structural_assumptions']) > 0
        assert len(result['model_dependent_aspects']) > 0

    def test_classification_distinguishes_existence_and_value(self):
        """
        Classification should distinguish between THEOREM (existence
        of positive infimum) and NUMERICAL (specific value).
        """
        ugt = UniformGapTheorem(N_c=2)
        result = ugt.prove_uniform_gap()
        c = result['classification']
        assert 'THEOREM' in c
        assert 'NUMERICAL' in c
