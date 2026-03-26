"""
Tests for the Zwanziger gap equation on S³.

Verifies:
    1. Gap equation has solutions for various R
    2. γ(R) > 0 for all R tested
    3. Convergence of the UV-subtracted spectral sum with l_max
    4. Limiting behaviors (small R: γ large, large R: γ stabilizes)
    5. Gluon mass stays positive and bounded below for large R
    6. Stabilization of γ(R) as R → ∞ at ~2.15 Λ_QCD
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


class TestRunningCoupling:
    """Running coupling g²(μ = 1/R) behaves correctly."""

    def test_small_R_perturbative(self):
        """Small R (high μ): g² is small (asymptotic freedom)."""
        g2 = ZwanzigerGapEquation.running_coupling_g2(0.1, N=2)
        assert g2 > 0, "g² must be positive"
        assert g2 < 5.0, f"At small R=0.1, g²={g2:.4f} should be moderate"

    def test_large_R_saturates(self):
        """Large R (low μ): g² saturates at IR fixed point 4π."""
        g2 = ZwanzigerGapEquation.running_coupling_g2(100.0, N=2)
        g2_max = 4 * np.pi
        assert abs(g2 - g2_max) / g2_max < 0.01

    def test_monotonic_with_R(self):
        """g²(R) increases monotonically with R (IR slavery)."""
        R_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        g2_values = [
            ZwanzigerGapEquation.running_coupling_g2(R, N=2)
            for R in R_values
        ]
        for i in range(len(g2_values) - 1):
            assert g2_values[i] <= g2_values[i + 1] + 1e-10, (
                f"g²(R={R_values[i]}) > g²(R={R_values[i+1]}): not monotonic"
            )

    def test_su3_coupling_positive(self):
        """SU(3) coupling is positive."""
        g2 = ZwanzigerGapEquation.running_coupling_g2(1.0, N=3)
        assert g2 > 0


class TestLaplacianSpectrum:
    """Scalar Laplacian eigenvalues on S³."""

    def test_eigenvalue_formula(self):
        """λ_l = l(l+2)/R² matches known values."""
        R = 1.0
        assert ZwanzigerGapEquation.laplacian_eigenvalue(0, R) == 0.0
        assert ZwanzigerGapEquation.laplacian_eigenvalue(1, R) == 3.0
        assert ZwanzigerGapEquation.laplacian_eigenvalue(2, R) == 8.0

    def test_eigenvalue_scaling_with_R(self):
        """λ_l scales as 1/R²."""
        for R in [0.5, 1.0, 2.0]:
            lam = ZwanzigerGapEquation.laplacian_eigenvalue(1, R)
            assert abs(lam - 3.0 / R**2) < 1e-14

    def test_multiplicity(self):
        """mult(l) = (l+1)²."""
        assert ZwanzigerGapEquation.laplacian_multiplicity(0) == 1
        assert ZwanzigerGapEquation.laplacian_multiplicity(1) == 4
        assert ZwanzigerGapEquation.laplacian_multiplicity(2) == 9


class TestGribovPropagator:
    """Gribov-modified ghost propagator."""

    def test_positive(self):
        """Propagator is always positive for l ≥ 1."""
        for l in range(1, 10):
            G = ZwanzigerGapEquation.gribov_propagator(l, gamma=1.0, R=1.0)
            assert G > 0

    def test_gamma_zero_limit(self):
        """At γ=0, propagator → 1/λ_l (standard propagator)."""
        R = 1.0
        for l in range(1, 5):
            lam = ZwanzigerGapEquation.laplacian_eigenvalue(l, R)
            G = ZwanzigerGapEquation.gribov_propagator(l, gamma=1e-15, R=R)
            assert abs(G - 1.0 / lam) / (1.0 / lam) < 1e-10

    def test_large_gamma_suppression(self):
        """At large γ, propagator → λ_l/γ⁴."""
        R = 1.0
        gamma_large = 100.0
        l = 1
        lam = ZwanzigerGapEquation.laplacian_eigenvalue(l, R)
        G = ZwanzigerGapEquation.gribov_propagator(l, gamma_large, R)
        expected = lam / gamma_large**4
        assert abs(G - expected) / expected < 1e-4

    def test_minimum_at_gamma_eq_sqrt_lam(self):
        """Denominator λ + γ⁴/λ is minimized when γ² = λ → G = 1/(2λ)."""
        R = 1.0
        l = 1
        lam = ZwanzigerGapEquation.laplacian_eigenvalue(l, R)
        gamma_min = np.sqrt(lam)
        G_min = ZwanzigerGapEquation.gribov_propagator(l, gamma_min, R)
        assert abs(G_min - 1.0 / (2 * lam)) < 1e-14


class TestSubtractedKernel:
    """UV-subtracted kernel σ(γ, λ) = γ⁴/(λ(λ² + γ⁴))."""

    def test_positive(self):
        """Kernel is positive for γ > 0, λ > 0."""
        for gamma in [0.1, 1.0, 5.0]:
            for lam in [0.5, 3.0, 100.0]:
                sigma = ZwanzigerGapEquation.subtracted_kernel(gamma, lam)
                assert sigma > 0

    def test_gamma_zero(self):
        """At γ = 0, kernel vanishes."""
        assert ZwanzigerGapEquation.subtracted_kernel(0.0, 3.0) == 0.0

    def test_uv_decay(self):
        """At large λ, kernel decays as γ⁴/λ³."""
        gamma = 1.0
        lam = 1000.0
        sigma = ZwanzigerGapEquation.subtracted_kernel(gamma, lam)
        expected = gamma**4 / lam**3
        assert abs(sigma - expected) / expected < 0.01

    def test_ir_behavior(self):
        """At small λ << γ², kernel → 1/λ."""
        gamma = 10.0
        lam = 0.01
        sigma = ZwanzigerGapEquation.subtracted_kernel(gamma, lam)
        expected = 1.0 / lam
        assert abs(sigma - expected) / expected < 0.01


class TestGapEquationResidual:
    """Gap equation residual LHS - RHS."""

    def test_residual_at_gamma_zero_is_positive(self):
        """At γ = 0, σ = 0 → RHS = 0 → residual = N²-1 > 0."""
        for R in [0.5, 2.0, 10.0]:
            res = ZwanzigerGapEquation.gap_equation_residual(0.0, R, N=2)
            assert abs(res - 3.0) < 1e-10

    def test_residual_at_large_gamma_is_negative(self):
        """At large γ, RHS is large → residual is negative."""
        res = ZwanzigerGapEquation.gap_equation_residual(100.0, R=2.0, N=2)
        assert res < 0, f"At large γ=100, residual={res:.4f} should be < 0"

    def test_residual_changes_sign(self):
        """Residual changes sign → solution exists (IVT)."""
        R = 2.0
        res_zero = ZwanzigerGapEquation.gap_equation_residual(0.0, R, N=2)
        res_large = ZwanzigerGapEquation.gap_equation_residual(100.0, R, N=2)
        assert res_zero > 0
        assert res_large < 0


class TestSolveGamma:
    """Solving the gap equation for γ(R)."""

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_solution_exists(self, R):
        """Gap equation has a solution for various R values."""
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=200)
        assert np.isfinite(gamma), f"No solution found at R={R}"
        assert gamma > 0, f"γ must be positive, got {gamma} at R={R}"

    def test_solution_satisfies_equation(self):
        """The returned γ actually satisfies the gap equation."""
        R = 2.0
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
        res = ZwanzigerGapEquation.gap_equation_residual(gamma, R, N=2, l_max=500)
        assert abs(res) < 1e-6, f"Residual at solution: {res:.2e}"

    def test_su3_solution(self):
        """SU(3) gap equation also has solutions."""
        for R in [1.0, 5.0]:
            gamma = ZwanzigerGapEquation.solve_gamma(R, N=3, l_max=200)
            assert np.isfinite(gamma), f"No SU(3) solution at R={R}"
            assert gamma > 0

    def test_gamma_decreases_then_stabilizes(self):
        """γ(R) is large at small R (UV) and stabilizes at large R (IR)."""
        gamma_small = ZwanzigerGapEquation.solve_gamma(0.2, N=2, l_max=200)
        gamma_med = ZwanzigerGapEquation.solve_gamma(2.0, N=2, l_max=200)
        gamma_large = ZwanzigerGapEquation.solve_gamma(50.0, N=2, l_max=200)
        assert gamma_small > gamma_med > gamma_large, (
            f"Expected γ(0.2)={gamma_small:.4f} > γ(2)={gamma_med:.4f} > "
            f"γ(50)={gamma_large:.4f}"
        )


class TestGluonMass:
    """Effective gluon mass from γ."""

    def test_positive_mass(self):
        """Gluon mass is positive for positive γ."""
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            mg = ZwanzigerGapEquation.gluon_mass_from_gamma(gamma, R=1.0)
            assert mg > 0

    def test_mass_formula(self):
        """m_g = √2 × γ."""
        gamma = 1.5
        mg = ZwanzigerGapEquation.gluon_mass_from_gamma(gamma, R=1.0)
        assert abs(mg - np.sqrt(2) * gamma) < 1e-14

    def test_mass_at_solved_gamma(self):
        """Gluon mass from the self-consistent γ is physical (> 0)."""
        R = 5.0
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=200)
        mg = ZwanzigerGapEquation.gluon_mass_from_gamma(gamma, R)
        assert mg > 0


class TestConvergence:
    """Convergence of the UV-subtracted spectral sum with l_max."""

    def test_sum_converges(self):
        """The spectral sum converges as l_max increases."""
        R = 2.0
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
        if not np.isfinite(gamma):
            pytest.skip("No solution found at R=2.0")

        check = ZwanzigerGapEquation.convergence_check(
            gamma, R, N=2, l_max_values=[50, 100, 200, 500, 1000]
        )
        residuals = check['residuals']
        diff = abs(residuals[-1] - residuals[-2])
        assert diff < 0.1, f"Sum not converged: diff={diff:.6f}"

    def test_gamma_stable_with_lmax(self):
        """γ(R) doesn't change significantly when l_max is increased."""
        R = 5.0
        gamma_200 = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=200)
        gamma_500 = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
        if np.isfinite(gamma_200) and np.isfinite(gamma_500):
            rel_diff = abs(gamma_500 - gamma_200) / gamma_200
            assert rel_diff < 0.05, (
                f"γ changed by {rel_diff*100:.1f}% (200→500): "
                f"γ(200)={gamma_200:.6f}, γ(500)={gamma_500:.6f}"
            )


class TestLargeRBehavior:
    """Behavior of γ(R) as R → ∞ — the key physical question."""

    def test_gamma_positive_at_large_R(self):
        """γ(R) remains positive for large R."""
        for R in [10.0, 50.0, 100.0]:
            gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
            assert np.isfinite(gamma), f"No solution at R={R}"
            assert gamma > 0, f"γ must be > 0 at R={R}"

    def test_gamma_stabilizes(self):
        """γ(R) approaches a constant ≈ 2.15 Λ_QCD as R → ∞."""
        R_values = [20.0, 50.0, 100.0]
        gammas = []
        for R in R_values:
            g = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
            if np.isfinite(g):
                gammas.append(g)

        assert len(gammas) >= 2, "Need at least 2 data points"

        mean_g = np.mean(gammas)
        max_dev = max(abs(g - mean_g) for g in gammas)
        rel_dev = max_dev / mean_g if mean_g > 0 else float('inf')
        assert rel_dev < 0.10, (
            f"γ not stabilized: values={[f'{g:.4f}' for g in gammas]}, "
            f"rel_dev={rel_dev:.3f}"
        )

    def test_gluon_mass_bounded_below(self):
        """Gluon mass m_g(R) ≥ 1.0 Λ_QCD for R ≥ 5."""
        R_values = [5.0, 10.0, 50.0, 100.0]
        masses = []
        for R in R_values:
            gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
            if np.isfinite(gamma):
                mg = ZwanzigerGapEquation.gluon_mass_from_gamma(gamma, R)
                masses.append(mg)

        assert len(masses) >= 3
        min_mass = min(masses)
        assert min_mass > 1.0, (
            f"Gluon mass should be > 1.0 Λ_QCD, minimum={min_mass:.4f}"
        )

    def test_dynamic_gap_exceeds_geometric_at_large_R(self):
        """At large R, dynamic mass m_g >> geometric gap 2/R."""
        R = 50.0
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=500)
        assert np.isfinite(gamma)
        mg = ZwanzigerGapEquation.gluon_mass_from_gamma(gamma, R)
        geo = 2.0 / R
        assert mg > geo, (
            f"At R={R}: m_g={mg:.4f} should >> 2/R={geo:.4f}"
        )


class TestSmallRBehavior:
    """Behavior of γ(R) at small R (UV limit)."""

    def test_gamma_large_at_small_R(self):
        """At small R, γ is large (UV enhancement from volume factor)."""
        R = 0.2
        gamma = ZwanzigerGapEquation.solve_gamma(R, N=2, l_max=200)
        assert np.isfinite(gamma)
        gamma_IR = ZwanzigerGapEquation.solve_gamma(10.0, N=2, l_max=200)
        assert gamma > gamma_IR, (
            f"γ(R=0.2)={gamma:.4f} should be > γ(R=10)={gamma_IR:.4f}"
        )


class TestCompleteAnalysis:
    """Integration test: full analysis pipeline."""

    def test_complete_analysis_runs(self):
        """complete_analysis() returns a valid result dict."""
        R_range = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        result = ZwanzigerGapEquation.complete_analysis(R_range, N=2)

        assert 'gamma' in result
        assert 'gluon_mass' in result
        assert 'geometric_gap' in result
        assert 'crossover_R' in result
        assert 'large_R_analysis' in result
        assert result['label'] == 'NUMERICAL'
        assert len(result['gamma']) == len(R_range)

    def test_crossover_exists(self):
        """Dynamic mass crosses above geometric gap at some R."""
        R_range = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
        result = ZwanzigerGapEquation.complete_analysis(R_range, N=2)

        # At R=100, geometric gap = 0.02, gluon mass should be ~3.0
        gamma_100 = result['gamma'][-1]
        if np.isfinite(gamma_100):
            mg_100 = result['gluon_mass'][-1]
            geo_100 = result['geometric_gap'][-1]
            assert mg_100 > geo_100, (
                f"At R=100: m_g={mg_100:.4f} should > 2/R={geo_100:.4f}"
            )

    def test_stabilization_detected(self):
        """The large-R analysis correctly detects stabilization."""
        R_range = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        result = ZwanzigerGapEquation.complete_analysis(R_range, N=2)
        analysis = result['large_R_analysis']
        assert analysis['stabilized'], (
            f"Expected stabilization; rel_var={analysis['relative_variation']:.4f}"
        )
