"""
Tests for the Andrews-Clutterbuck Fundamental Gap Theorem applied to
the Gribov Region.

Test categories:
    1. AC bound is exactly 3x the PW bound
    2. gap_on_gribov_9dof returns positive values for various R
    3. AC gap increases with R (since d decreases as R grows)
    4. AC bound >= 3 * PW bound (exact relationship)
    5. Physical R = 2.2 gives meaningful gap
    6. improvement_factor is exactly 3
    7. AC bound on a ball (numerical eigenvalue check)
    8. Formal theorem statement
    9. Comparison analysis runs without error
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.fundamental_gap import FundamentalGap, HBAR_C_MEV_FM
from yang_mills_s3.proofs.diameter_theorem import _C_D_EXACT, _DR_ASYMPTOTIC


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def fg():
    """FundamentalGap instance."""
    return FundamentalGap()


# ======================================================================
# 1. AC bound is exactly 3x PW bound
# ======================================================================

class TestACvsPW:
    """The AC fundamental gap bound should be exactly 3x the PW bound."""

    def test_ac_equals_3_pw_d1(self):
        """For d=1, AC = 3*pi^2, PW = pi^2."""
        ac = FundamentalGap.ac_bound_pure_laplacian(1.0)
        pw = FundamentalGap.pw_bound(1.0)
        np.testing.assert_allclose(ac, 3.0 * pw, rtol=1e-14)

    def test_ac_equals_3_pw_d2(self):
        """For d=2, AC = 3*pi^2/4, PW = pi^2/4."""
        ac = FundamentalGap.ac_bound_pure_laplacian(2.0)
        pw = FundamentalGap.pw_bound(2.0)
        np.testing.assert_allclose(ac, 3.0 * pw, rtol=1e-14)

    def test_ac_equals_3_pw_d05(self):
        """For d=0.5, AC = 12*pi^2, PW = 4*pi^2."""
        ac = FundamentalGap.ac_bound_pure_laplacian(0.5)
        pw = FundamentalGap.pw_bound(0.5)
        np.testing.assert_allclose(ac, 3.0 * pw, rtol=1e-14)

    def test_ac_equals_3_pw_many_d(self):
        """For many d values, AC should always be 3*PW."""
        for d in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
            ac = FundamentalGap.ac_bound_pure_laplacian(d)
            pw = FundamentalGap.pw_bound(d)
            np.testing.assert_allclose(
                ac, 3.0 * pw, rtol=1e-14,
                err_msg=f"AC != 3*PW at d={d}"
            )

    def test_convex_potential_same_as_pure(self):
        """AC bound with convex potential should equal pure Laplacian bound."""
        for d in [0.5, 1.0, 5.0]:
            ac_pure = FundamentalGap.ac_bound_pure_laplacian(d)
            ac_convex = FundamentalGap.ac_bound_convex_potential(d)
            np.testing.assert_allclose(ac_pure, ac_convex, rtol=1e-14)


# ======================================================================
# 2. gap_on_gribov_9dof returns positive values
# ======================================================================

class TestGapPositive:
    """AC gap on the Gribov region should be positive for all R > 0."""

    @pytest.mark.parametrize("R", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_ac_gap_positive(self, fg, R):
        """AC gap should be positive at R={R}."""
        result = fg.gap_on_gribov_9dof(R)
        assert result['ac_gap'] > 0, \
            f"AC gap should be positive at R={R}, got {result['ac_gap']}"

    @pytest.mark.parametrize("R", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_pw_bound_positive(self, fg, R):
        """PW bound should be positive at R={R}."""
        result = fg.gap_on_gribov_9dof(R)
        assert result['pw_bound'] > 0, \
            f"PW bound should be positive at R={R}, got {result['pw_bound']}"

    @pytest.mark.parametrize("R", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_diameter_positive(self, fg, R):
        """Diameter should be positive and finite at R={R}."""
        result = fg.gap_on_gribov_9dof(R)
        assert result['diameter'] > 0
        assert np.isfinite(result['diameter'])

    @pytest.mark.parametrize("R", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_ac_equals_3_pw_on_gribov(self, fg, R):
        """AC gap should be exactly 3x PW bound on Gribov region."""
        result = fg.gap_on_gribov_9dof(R)
        np.testing.assert_allclose(
            result['ac_gap'], 3.0 * result['pw_bound'],
            rtol=1e-12,
            err_msg=f"AC != 3*PW on Gribov region at R={R}"
        )


# ======================================================================
# 3. AC gap increases with R
# ======================================================================

class TestGapIncreasesWithR:
    """
    The AC gap should increase with R because:
        d(R) = 3*C_D / (R*g(R))
    decreases with R (since R*g(R) increases), so
        3*pi^2/d^2 = 3*pi^2*(R*g)^2 / (3*C_D)^2
    increases with R.
    """

    def test_ac_gap_increases_small_to_large_R(self, fg):
        """AC gap at R=10 should be larger than at R=1."""
        gap_1 = fg.gap_on_gribov_9dof(1.0)['ac_gap']
        gap_10 = fg.gap_on_gribov_9dof(10.0)['ac_gap']
        assert gap_10 > gap_1, \
            f"AC gap should increase: gap(R=10)={gap_10} <= gap(R=1)={gap_1}"

    def test_ac_gap_increases_monotonically(self, fg):
        """AC gap should increase monotonically for R >= 0.5."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        gaps = [fg.gap_on_gribov_9dof(R)['ac_gap'] for R in R_values]
        for i in range(1, len(gaps)):
            assert gaps[i] > gaps[i - 1], \
                f"AC gap not increasing: gap(R={R_values[i]})={gaps[i]} " \
                f"<= gap(R={R_values[i-1]})={gaps[i-1]}"

    def test_diameter_decreases_with_R(self, fg):
        """Diameter d(R) should decrease as R increases."""
        R_values = [0.5, 1.0, 5.0, 10.0, 50.0]
        diameters = [fg.gap_on_gribov_9dof(R)['diameter'] for R in R_values]
        for i in range(1, len(diameters)):
            assert diameters[i] < diameters[i - 1], \
                f"Diameter not decreasing: d(R={R_values[i]})={diameters[i]} " \
                f">= d(R={R_values[i-1]})={diameters[i-1]}"


# ======================================================================
# 4. AC bound >= 3 * PW bound (relationship)
# ======================================================================

class TestACDominatesPW:
    """AC gap should always be >= 3 * PW bound."""

    def test_ac_dominates_in_gap_vs_R(self, fg):
        """In gap_vs_R, AC should dominate PW by factor 3."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        result = fg.gap_vs_R(R_values)
        for i, R in enumerate(R_values):
            np.testing.assert_allclose(
                result['ac_gap'][i],
                3.0 * result['pw_bound'][i],
                rtol=1e-12,
                err_msg=f"AC != 3*PW in gap_vs_R at R={R}"
            )

    def test_ac_dominates_flag(self, fg):
        """The ac_dominates_pw flag should be True everywhere."""
        R_values = [0.1, 1.0, 10.0, 100.0]
        result = fg.gap_vs_R(R_values)
        assert all(result['ac_dominates_pw']), \
            "AC should dominate PW at all R values"


# ======================================================================
# 5. Physical R = 2.2 gives meaningful gap
# ======================================================================

class TestPhysicalR:
    """At R = 2.2 (physical radius), the gap should be physically meaningful."""

    def test_ac_gap_at_R22_positive(self, fg):
        """AC gap at R=2.2 should be positive."""
        result = fg.gap_on_gribov_9dof(2.2)
        assert result['ac_gap'] > 0

    def test_ac_gap_MeV_at_R22(self, fg):
        """AC gap in MeV at R=2.2 should be a reasonable energy scale."""
        result = fg.gap_on_gribov_9dof(2.2)
        # The gap in MeV should be in a reasonable range (> 0)
        assert result['ac_gap_MeV'] > 0, \
            f"AC gap in MeV should be positive, got {result['ac_gap_MeV']}"

    def test_comparison_analysis_at_physical_R(self, fg):
        """Comparison analysis should include physical R data."""
        result = fg.comparison_analysis()
        assert result['ac_at_physical_R'] > 0
        assert result['pw_at_physical_R'] > 0
        assert result['physical_gap_MeV'] > 0

    def test_ac_gap_at_R22_larger_than_pw(self, fg):
        """At R=2.2, AC gap should be 3x the PW bound."""
        result = fg.gap_on_gribov_9dof(2.2)
        np.testing.assert_allclose(
            result['ac_gap'],
            3.0 * result['pw_bound'],
            rtol=1e-12
        )


# ======================================================================
# 6. Improvement factor is exactly 3
# ======================================================================

class TestImprovementFactor:
    """The improvement factor AC/PW should be exactly 3."""

    def test_improvement_factor_value(self):
        """improvement_factor() should return 3."""
        assert FundamentalGap.improvement_factor() == 3

    def test_improvement_factor_type(self):
        """improvement_factor() should return an integer."""
        assert isinstance(FundamentalGap.improvement_factor(), int)

    def test_numerical_ratio_matches(self):
        """Numerical ratio AC/PW should be 3 for any d."""
        for d in [0.1, 1.0, 10.0, 100.0]:
            ratio = (FundamentalGap.ac_bound_pure_laplacian(d)
                     / FundamentalGap.pw_bound(d))
            np.testing.assert_allclose(ratio, 3.0, rtol=1e-14)


# ======================================================================
# 7. AC bound on a ball (numerical eigenvalue verification)
# ======================================================================

class TestACBoundOnBall:
    """
    Verify the AC bound against known eigenvalues of -Delta on a ball.

    For the unit ball B_1 in R^n, the Dirichlet eigenvalues are:
        lambda_k = j_{nu,m}^2
    where j_{nu,m} are zeros of Bessel functions.

    In R^1 (interval [-1,1], d=2):
        lambda_k = (k*pi/2)^2, k=1,2,...
        lambda_1 = pi^2/4, lambda_2 = pi^2
        gap = lambda_2 - lambda_1 = 3*pi^2/4
        AC bound = 3*pi^2/d^2 = 3*pi^2/4
        => EQUALITY (the interval is a thin slab in R^1)

    This provides a sharp test of the AC bound.
    """

    def test_ac_sharp_on_interval(self):
        """AC bound should be exactly achieved on the interval [-1,1]."""
        d = 2.0  # diameter of [-1,1]
        ac_bound = FundamentalGap.ac_bound_pure_laplacian(d)

        # Exact eigenvalues: lambda_k = (k*pi/2)^2
        lambda_1 = (np.pi / 2.0)**2
        lambda_2 = (np.pi)**2
        exact_gap = lambda_2 - lambda_1  # = 3*pi^2/4

        # AC bound should equal the exact gap (sharp case)
        np.testing.assert_allclose(ac_bound, exact_gap, rtol=1e-12)

    def test_ac_valid_on_disk(self):
        """
        On the unit disk in R^2, the gap should satisfy AC.

        Dirichlet eigenvalues of -Delta on disk of radius 1:
            lambda_1 = j_{0,1}^2 ~ 2.4048^2 ~ 5.7832
            lambda_2 = j_{1,1}^2 ~ 3.8317^2 ~ 14.682
        (lambda_2 has multiplicity 2)

        Diameter d = 2.
        AC bound = 3*pi^2/4 ~ 7.4022

        Exact gap = 14.682 - 5.783 ~ 8.899

        AC bound should be <= exact gap.
        """
        # Bessel zeros (from standard tables)
        j_01 = 2.4048255577  # first zero of J_0
        j_11 = 3.8317059702  # first zero of J_1

        lambda_1 = j_01**2
        lambda_2 = j_11**2
        exact_gap = lambda_2 - lambda_1

        d = 2.0  # diameter of unit disk
        ac_bound = FundamentalGap.ac_bound_pure_laplacian(d)

        assert ac_bound <= exact_gap + 1e-10, \
            f"AC bound {ac_bound} should be <= exact gap {exact_gap} on disk"

    def test_ac_valid_on_ball_3d(self):
        """
        On the unit ball in R^3, the gap should satisfy AC.

        Dirichlet eigenvalues of -Delta on ball of radius 1:
            lambda_1 = pi^2 ~ 9.8696 (first zero of j_0(r) = sin(r)/r)
            lambda_2 ~ (4.493)^2 ~ 20.19 (first zero of j_1)

        Diameter d = 2.
        AC bound = 3*pi^2/4 ~ 7.4022

        Exact gap ~ 20.19 - 9.87 ~ 10.32

        AC bound should be <= exact gap.
        """
        # For the unit ball in R^3:
        # lambda_1 corresponds to l=0: j_{1/2,1} = pi
        # lambda_2 corresponds to l=1: j_{3/2,1} ~ 4.4934
        lambda_1 = np.pi**2
        j_32_1 = 4.4934094579  # first positive zero of j_{3/2}
        lambda_2 = j_32_1**2

        exact_gap = lambda_2 - lambda_1

        d = 2.0  # diameter of unit ball
        ac_bound = FundamentalGap.ac_bound_pure_laplacian(d)

        assert ac_bound <= exact_gap + 1e-10, \
            f"AC bound {ac_bound} should be <= exact gap {exact_gap} on 3D ball"


# ======================================================================
# 8. Formal theorem statement
# ======================================================================

class TestFormalStatement:
    """The formal theorem statement should be complete and correct."""

    def test_statement_is_string(self):
        """formal_theorem_statement should return a string."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert isinstance(stmt, str)

    def test_statement_mentions_AC(self):
        """Statement should mention Andrews-Clutterbuck."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "Andrews" in stmt
        assert "Clutterbuck" in stmt

    def test_statement_mentions_3pi2(self):
        """Statement should mention 3*pi^2/d^2."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "3*pi^2/d^2" in stmt

    def test_statement_mentions_PW(self):
        """Statement should mention Payne-Weinberger for comparison."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "Payne" in stmt or "PW" in stmt

    def test_statement_mentions_convex(self):
        """Statement should mention convexity requirement."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "convex" in stmt.lower()

    def test_statement_mentions_gribov(self):
        """Statement should mention the Gribov region."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "Gribov" in stmt

    def test_statement_has_label(self):
        """Statement should include THEOREM label."""
        stmt = FundamentalGap.formal_theorem_statement()
        assert "THEOREM" in stmt


# ======================================================================
# 9. Comparison analysis
# ======================================================================

class TestComparisonAnalysis:
    """comparison_analysis should run and produce sensible results."""

    def test_comparison_runs(self, fg):
        """comparison_analysis should run without error."""
        result = fg.comparison_analysis(R_range=[1.0, 5.0, 10.0])
        assert 'gap_comparison' in result
        assert 'improvement_factor' in result
        assert 'formal_statement' in result

    def test_improvement_factor_is_3(self, fg):
        """improvement_factor in analysis should be 3."""
        result = fg.comparison_analysis(R_range=[1.0, 5.0])
        assert result['improvement_factor'] == 3

    def test_all_values_finite(self, fg):
        """All gap values should be finite."""
        result = fg.comparison_analysis(R_range=[0.5, 1.0, 5.0, 10.0])
        comp = result['gap_comparison']
        assert all(np.isfinite(comp['ac_gap']))
        assert all(np.isfinite(comp['pw_bound']))
        assert all(np.isfinite(comp['geometric_gap']))
        assert all(np.isfinite(comp['diameter']))

    def test_assessment_present(self, fg):
        """Assessment string should be present."""
        result = fg.comparison_analysis(R_range=[1.0, 5.0, 10.0])
        assert 'assessment' in result
        assert len(result['assessment']) > 0


# ======================================================================
# 10. Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge cases for the AC bound."""

    def test_ac_zero_diameter(self):
        """AC bound for d=0 should return 0."""
        assert FundamentalGap.ac_bound_pure_laplacian(0.0) == 0.0

    def test_ac_negative_diameter(self):
        """AC bound for d<0 should return 0."""
        assert FundamentalGap.ac_bound_pure_laplacian(-1.0) == 0.0

    def test_ac_infinite_diameter(self):
        """AC bound for d=inf should return 0."""
        assert FundamentalGap.ac_bound_pure_laplacian(np.inf) == 0.0

    def test_pw_zero_diameter(self):
        """PW bound for d=0 should return 0."""
        assert FundamentalGap.pw_bound(0.0) == 0.0

    def test_pw_infinite_diameter(self):
        """PW bound for d=inf should return 0."""
        assert FundamentalGap.pw_bound(np.inf) == 0.0

    def test_ac_very_small_diameter(self):
        """AC bound for very small d should be very large."""
        ac = FundamentalGap.ac_bound_pure_laplacian(0.001)
        assert ac > 1e6

    def test_ac_bound_exact_values(self):
        """AC bound for d=1 should be exactly 3*pi^2."""
        np.testing.assert_allclose(
            FundamentalGap.ac_bound_pure_laplacian(1.0),
            3.0 * np.pi**2,
            rtol=1e-14
        )


# ======================================================================
# 11. Consistency with diameter_theorem constants
# ======================================================================

class TestConsistencyWithDiameterTheorem:
    """Results should be consistent with the diameter theorem constants."""

    def test_asymptotic_diameter_used(self, fg):
        """At large R, diameter*R should approach DR_ASYMPTOTIC."""
        result = fg.gap_on_gribov_9dof(100.0)
        dR = result['diameter_dimless']
        np.testing.assert_allclose(
            dR, _DR_ASYMPTOTIC,
            rtol=0.01,
            err_msg=f"d*R at R=100 should be close to {_DR_ASYMPTOTIC}"
        )

    def test_ac_gap_from_asymptotic_diameter(self, fg):
        """AC gap at large R should be ~ 3*pi^2/(DR_ASYMPTOTIC/R)^2."""
        R = 100.0
        result = fg.gap_on_gribov_9dof(R)
        # Expected: 3*pi^2 / (DR_ASYMPTOTIC/R)^2 = 3*pi^2*R^2/DR_ASYMPTOTIC^2
        expected_approx = 3.0 * np.pi**2 * R**2 / _DR_ASYMPTOTIC**2
        np.testing.assert_allclose(
            result['ac_gap'],
            expected_approx,
            rtol=0.01,
            err_msg="AC gap at large R should match asymptotic formula"
        )

    def test_gap_vs_R_has_correct_keys(self, fg):
        """gap_vs_R should return all expected keys."""
        result = fg.gap_vs_R([1.0, 5.0])
        expected_keys = ['R', 'ac_gap', 'pw_bound', 'geometric_gap',
                         'kr_gap', 'diameter', 'g_squared',
                         'ac_dominates_pw', 'label']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
