"""
Tests for V₄ convexity investigation.

CRITICAL: These tests determine whether THEOREM 7.1d and THEOREM 10.7 Part II
are correct or need correction.
"""

import numpy as np
import pytest
from yang_mills_s3.proofs.v4_convexity import (
    v4_potential,
    v4_as_sum_of_squares,
    v2_potential,
    total_potential,
    hessian_numerical,
    hessian_v4_analytical,
    hessian_v2,
    hessian_total_analytical,
    hessian_eigenvalues,
    analyze_point,
    verify_bilinear_square_not_convex,
    task1_hessian_survey,
    task2_operator_norm_bound,
    task4_total_hessian_psd,
    find_worst_hessian_direction,
    kappa_corrected,
    full_investigation,
)


# ======================================================================
# Basic potential tests
# ======================================================================

class TestV4Potential:
    """Tests for V₄ potential computation."""

    def test_v4_at_origin_is_zero(self):
        """V₄(0) = 0."""
        assert abs(v4_potential(np.zeros(9))) < 1e-15

    def test_v4_nonnegative(self):
        """V₄(a) >= 0 for all a (algebraic identity)."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            a = rng.standard_normal(9) * rng.uniform(0.01, 10.0)
            assert v4_potential(a) >= -1e-14, f"V4 negative at {a}: {v4_potential(a)}"

    def test_v4_two_formulas_agree(self):
        """Matrix form and explicit sum-of-squares form agree."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            a = rng.standard_normal(9) * 2.0
            v4_matrix = v4_potential(a, g2=3.5)
            v4_explicit = v4_as_sum_of_squares(a, g2=3.5)
            np.testing.assert_allclose(v4_matrix, v4_explicit, rtol=1e-10,
                                       err_msg=f"Mismatch at a = {a}")

    def test_v4_gauge_invariant(self):
        """V₄ is invariant under SO(3) acting on color index."""
        rng = np.random.default_rng(77)
        for _ in range(100):
            a = rng.standard_normal((3, 3))
            # Random SO(3)
            angle = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            a_rot = a @ R.T
            np.testing.assert_allclose(
                v4_potential(a), v4_potential(a_rot), rtol=1e-10
            )

    def test_v4_homogeneous_degree_4(self):
        """V₄(λa) = λ⁴ V₄(a)."""
        rng = np.random.default_rng(55)
        for _ in range(100):
            a = rng.standard_normal(9)
            lam = rng.uniform(0.1, 5.0)
            v4_a = v4_potential(a)
            v4_la = v4_potential(lam * a)
            if abs(v4_a) > 1e-14:
                np.testing.assert_allclose(v4_la, lam**4 * v4_a, rtol=1e-10)

    def test_v4_zero_for_rank1_matrix(self):
        """V₄ = 0 when M has rank <= 1 (only one singular value nonzero)."""
        # Rank 1: M = u ⊗ v
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -1.0, 0.3])
        M = np.outer(u, v)
        assert abs(v4_potential(M.ravel())) < 1e-12

    def test_v4_positive_for_rank2_matrix(self):
        """V₄ > 0 when M has rank >= 2."""
        M = np.eye(3)
        M[2, :] = 0  # rank 2
        assert v4_potential(M.ravel()) > 0


# ======================================================================
# Hessian tests
# ======================================================================

class TestHessian:
    """Tests for Hessian computation."""

    def test_hessian_v4_at_origin_is_zero(self):
        """Hess(V₄)(0) = 0 since V₄ is quartic."""
        H = hessian_v4_analytical(np.zeros(9))
        np.testing.assert_allclose(H, np.zeros((9, 9)), atol=1e-14)

    def test_hessian_v2_is_constant(self):
        """Hess(V₂) = (4/R²)I₉."""
        R = 2.2
        H = hessian_v2(R)
        expected = (4.0 / R**2) * np.eye(9)
        np.testing.assert_allclose(H, expected, atol=1e-14)

    def test_hessian_v4_is_symmetric(self):
        """Hess(V₄) should be symmetric."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.standard_normal(9)
            H = hessian_v4_analytical(a)
            np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_hessian_v4_analytical_matches_numerical(self):
        """Analytical Hessian matches numerical finite-difference Hessian."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(9) * 0.5
            g2 = 2.0
            H_analytical = hessian_v4_analytical(a, g2)
            H_numerical = hessian_numerical(lambda x: v4_potential(x, g2), a, h=1e-5)
            np.testing.assert_allclose(
                H_analytical, H_numerical, atol=1e-4, rtol=1e-3,
                err_msg=f"Hessian mismatch at a = {a}"
            )

    def test_hessian_total_matches_numerical(self):
        """Analytical total Hessian matches numerical."""
        rng = np.random.default_rng(99)
        R, g2 = 1.5, 3.0
        for _ in range(10):
            a = rng.standard_normal(9) * 0.3
            H_analytical = hessian_total_analytical(a, R, g2)
            H_numerical = hessian_numerical(lambda x: total_potential(x, R, g2), a, h=1e-5)
            np.testing.assert_allclose(
                H_analytical, H_numerical, atol=1e-4, rtol=1e-3
            )

    def test_hessian_v4_homogeneous_degree_2(self):
        """Hess(V₄)(λa) = λ² Hess(V₄)(a) since V₄ is degree 4."""
        rng = np.random.default_rng(33)
        for _ in range(20):
            a = rng.standard_normal(9)
            lam = rng.uniform(0.5, 3.0)
            H_a = hessian_v4_analytical(a)
            H_la = hessian_v4_analytical(lam * a)
            np.testing.assert_allclose(H_la, lam**2 * H_a, atol=1e-10)


# ======================================================================
# THE CRITICAL TEST: Is Hess(V₄) PSD?
# ======================================================================

class TestV4Convexity:
    """THE CRITICAL TESTS: Determine if Hess(V₄) is PSD."""

    def test_bilinear_square_not_convex(self):
        """Verify the counterexample: (xy)² is not convex."""
        result = verify_bilinear_square_not_convex()
        assert result['confirms_not_convex'], "Expected (xy)^2 to be non-convex"
        np.testing.assert_allclose(result['min_eigenvalue'], -2.0, atol=0.01)

    def test_hess_v4_at_specific_points(self):
        """Check Hess(V₄) eigenvalues at specific points.

        KEY TEST: If V₄ = sum of squares of bilinear forms,
        Hess(V₄) should NOT be PSD at generic points.
        """
        # Point: a_{0,0} = a_{1,1} = 1, rest zero (diagonal-like)
        a = np.zeros(9)
        a[0] = 1.0  # a_{0,0}
        a[4] = 1.0  # a_{1,1}

        H = hessian_v4_analytical(a, g2=1.0)
        eigs = hessian_eigenvalues(H)

        # Record whether any eigenvalue is negative
        has_negative = eigs[0] < -1e-10
        # This is the key finding - record it
        print(f"  Hess(V4) at diag point: min eig = {eigs[0]:.6f}, max = {eigs[-1]:.6f}")
        print(f"  All eigenvalues: {eigs}")
        print(f"  Has negative eigenvalue: {has_negative}")

    def test_hess_v4_random_survey(self):
        """Survey Hess(V₄) over many random points.

        FINDING: What fraction of points have negative Hessian eigenvalues?
        """
        rng = np.random.default_rng(42)
        n_negative = 0
        n_total = 500
        min_eig_found = np.inf

        for _ in range(n_total):
            a = rng.standard_normal(9)
            a = a / np.linalg.norm(a)
            H = hessian_v4_analytical(a, g2=1.0)
            eigs = hessian_eigenvalues(H)
            if eigs[0] < -1e-10:
                n_negative += 1
            min_eig_found = min(min_eig_found, eigs[0])

        print(f"\n  Random survey ({n_total} points on unit sphere):")
        print(f"  Points with negative Hess(V4) eigenvalue: {n_negative}/{n_total}")
        print(f"  Most negative eigenvalue found: {min_eig_found:.6f}")

        # The test RECORDS the finding but does not assert PSD
        # because we EXPECT it to be non-PSD

    def test_hess_v4_worst_direction_search(self):
        """Find the worst direction and report."""
        result = find_worst_hessian_direction(g2=1.0, n_restarts=100)
        print(f"\n  Worst min eigenvalue of Hess(V4) on unit sphere (g2=1):")
        print(f"    {result['worst_min_eigenvalue']:.6f}")
        print(f"  Worst direction: {result['worst_direction']}")

    def test_hess_v4_single_minor_not_convex(self):
        """Verify that a single (a_{iα}a_{jβ} - a_{jα}a_{iβ})² is not convex.

        Take i=0, j=1, α=0, β=1.
        f = (a_{0,0}*a_{1,1} - a_{1,0}*a_{0,1})²

        This involves 4 variables: a_{0,0}, a_{0,1}, a_{1,0}, a_{1,1}.
        It's the square of a 2x2 determinant, which is NOT convex.
        """
        # Hessian of f = (x₁x₄ - x₂x₃)² where x₁=a00, x₂=a01, x₃=a10, x₄=a11
        # At point (1,0,0,1): det = 1, gradient = (1, 0, 0, 1) * 2
        # Hessian involves both the outer product of gradient and
        # the bilinear cross terms

        def single_minor_sq(a4):
            return (a4[0] * a4[3] - a4[1] * a4[2])**2

        point = np.array([1.0, 0.0, 0.0, 1.0])
        H = hessian_numerical(single_minor_sq, point, h=1e-5)
        eigs = np.linalg.eigvalsh(H)

        print(f"\n  Single 2x2 minor squared:")
        print(f"  Hessian at (1,0,0,1): eigenvalues = {eigs}")
        print(f"  Min eigenvalue = {eigs[0]:.4f}")

        # The square of det(2x2) is NOT convex
        assert eigs[0] < -0.1, "Expected negative eigenvalue for single minor squared"


# ======================================================================
# Total potential convexity
# ======================================================================

class TestTotalConvexity:
    """Tests for convexity of V₂ + V₄."""

    def test_total_hessian_psd_at_origin(self):
        """At origin, Hess(V) = (4/R²)I₉ > 0."""
        R = 2.2
        H = hessian_total_analytical(np.zeros(9), R, g2=11.33)
        eigs = hessian_eigenvalues(H)
        assert eigs[0] > 0, f"Hess(V) at origin should be positive: {eigs[0]}"
        np.testing.assert_allclose(eigs[0], 4.0 / R**2, atol=1e-10)

    def test_total_hessian_at_large_radius(self):
        """At large |a|, Hess(V) may become non-PSD.

        Hess(V₂) = (4/R²)I₉ (constant)
        Hess(V₄) grows as |a|² (can have negative eigenvalues)

        So for |a| large enough, Hess(V) could have negative eigenvalues.
        """
        R = 2.2
        g2 = 11.33

        # Find the worst direction
        wd = find_worst_hessian_direction(g2=1.0, n_restarts=50)
        worst_dir = wd['worst_direction']
        worst_eig_at_1 = wd['worst_min_eigenvalue']

        # At radius r in worst direction:
        # min eigenvalue of Hess(V) ~ 4/R² + g² * r² * worst_eig_at_1
        # This is zero when r² = -4/(R² * g² * worst_eig_at_1)

        if worst_eig_at_1 < -1e-10:
            r_critical = np.sqrt(-4.0 / (R**2 * g2 * worst_eig_at_1))
            print(f"\n  Critical radius where Hess(V) first becomes non-PSD:")
            print(f"    r_critical = {r_critical:.4f}")
            print(f"    (compare to Gribov radius)")

            # Verify at r_critical
            a_crit = r_critical * worst_dir
            H = hessian_total_analytical(a_crit, R, g2)
            eigs = hessian_eigenvalues(H)
            print(f"    Min eigenvalue at r_critical: {eigs[0]:.6f} (should be ~0)")

    def test_gribov_region_convexity(self):
        """On Ω₉ (Gribov region), is V convex?

        If r_critical > Gribov radius, then V IS convex on Ω₉.
        """
        result = task4_total_hessian_psd(R=2.2, g2=11.33, n_random=2000)

        print(f"\n  Gribov region convexity analysis:")
        print(f"    V convex on all R^9: {result['convex_everywhere_R9']}")
        print(f"    Critical radius: {result['r_critical']:.4f}")
        print(f"    Gribov radius: {result['gribov_radius']:.4f}")
        print(f"    Gribov saves convexity: {result['gribov_saves_convexity']}")
        print(f"    Ratio r_crit/r_Gribov: {result['ratio_r_critical_over_gribov']:.4f}")


# ======================================================================
# kappa_corrected tests
# ======================================================================

class TestKappaCorrected:
    """Tests for the corrected BE curvature."""

    def test_kappa_at_physical_R(self):
        """kappa_corrected at R = 2.2 fm."""
        t2 = task2_operator_norm_bound(g2=1.0, n_random=1000)
        C_op = t2['C_operator_norm']

        R = 2.2
        g2 = 11.33
        g = np.sqrt(g2)
        C_D = 3.0 * np.sqrt(3.0) / 2.0
        d_gribov = 3.0 * C_D / (R * g)

        res = kappa_corrected(R, g2, C_op, d_gribov)

        print(f"\n  kappa_corrected at R = {R} fm, g^2 = {g2}:")
        print(f"    Harmonic: {res['harmonic_4_over_R2']:.4f}")
        print(f"    V4 worst: {res['v4_worst_hessian']:.4f}")
        print(f"    Ghost:    {res['ghost_curvature']:.4f}")
        print(f"    Total:    {res['kappa_corrected']:.4f}")
        print(f"    Positive: {res['kappa_positive']}")

    def test_kappa_at_minimizer(self):
        """kappa_corrected at R* ~ 0.96 fm (the minimizer from paper)."""
        t2 = task2_operator_norm_bound(g2=1.0, n_random=1000)
        C_op = t2['C_operator_norm']

        R = 0.96
        g2 = 6.0  # approximate at this R
        g = np.sqrt(g2)
        C_D = 3.0 * np.sqrt(3.0) / 2.0
        d_gribov = 3.0 * C_D / (R * g)

        res = kappa_corrected(R, g2, C_op, d_gribov)

        print(f"\n  kappa_corrected at R = {R} fm, g^2 = {g2}:")
        print(f"    Harmonic: {res['harmonic_4_over_R2']:.4f}")
        print(f"    V4 worst: {res['v4_worst_hessian']:.4f}")
        print(f"    Ghost:    {res['ghost_curvature']:.4f}")
        print(f"    Total:    {res['kappa_corrected']:.4f}")
        print(f"    Positive: {res['kappa_positive']}")


# ======================================================================
# Integration test: full investigation
# ======================================================================

class TestFullInvestigation:
    """Integration test running the complete investigation."""

    def test_full_investigation_runs(self):
        """Run the full investigation and report all findings."""
        results = full_investigation(R_phys=2.2, g2_phys=11.33, verbose=True)

        # Report conclusions
        c = results['conclusions']
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"  Hess(V4) PSD? {c['hess_v4_is_psd']}")
        print(f"  kappa_corrected > 0 for all R? {c['kappa_corrected_positive_for_all_R']}")
        print(f"  Gribov saves convexity? {c['gribov_region_saves_convexity']}")
        print(f"  THEOREM 7.1d: {c['theorem_7_1d_status']}")
        print(f"  THEOREM 10.7: {c['theorem_10_7_status']}")
