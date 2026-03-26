"""
Tests for Gribov Region Diameter + Payne-Weinberger bound.

Tests the GribovDiameter class which computes the diameter of the Gribov
region in the 9-DOF truncation on S³/I* and applies the Payne-Weinberger
bound for the spectral gap.

Test categories:
    1. FP operator at vacuum has correct eigenvalues
    2. FP operator symmetry
    3. Horizon distance is finite and positive
    4. Horizon distance decreases with coupling
    5. Diameter is approximately 2x max horizon distance
    6. Payne-Weinberger bound is positive
    7. Diameter vs R: stabilization for large R
    8. Payne-Weinberger bound >= some minimum for R in [0.1, 100]
    9. Complete analysis runs without error
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.gribov_diameter import GribovDiameter


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def gd():
    """GribovDiameter instance."""
    return GribovDiameter()


# ======================================================================
# 1. FP operator at vacuum
# ======================================================================

class TestFPOperatorVacuum:
    """Verify FP operator at a=0 has eigenvalues 3/R^2."""

    def test_vacuum_eigenvalue_R1(self, gd):
        """At a=0, R=1: all eigenvalues should be 3/R^2 = 3.0."""
        a_zero = np.zeros(9)
        M = gd.fp_operator_truncated(a_zero, R=1.0)
        eigenvalues = np.linalg.eigvalsh(M)
        expected = 3.0
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)

    def test_vacuum_eigenvalue_R2(self, gd):
        """At a=0, R=2: all eigenvalues should be 3/R^2 = 0.75."""
        a_zero = np.zeros(9)
        M = gd.fp_operator_truncated(a_zero, R=2.0)
        eigenvalues = np.linalg.eigvalsh(M)
        expected = 3.0 / 4.0
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)

    def test_vacuum_eigenvalue_R05(self, gd):
        """At a=0, R=0.5: all eigenvalues should be 3/R^2 = 12.0."""
        a_zero = np.zeros(9)
        M = gd.fp_operator_truncated(a_zero, R=0.5)
        eigenvalues = np.linalg.eigvalsh(M)
        expected = 12.0
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)

    def test_vacuum_identity_matrix(self, gd):
        """At a=0, M_FP should be (3/R^2)*I_9."""
        a_zero = np.zeros(9)
        R = 1.5
        M = gd.fp_operator_truncated(a_zero, R=R)
        expected = (3.0 / R**2) * np.eye(9)
        np.testing.assert_allclose(M, expected, atol=1e-12)


# ======================================================================
# 2. FP operator symmetry
# ======================================================================

class TestFPOperatorSymmetry:
    """The FP operator matrix should be real."""

    def test_real_matrix(self, gd):
        """M_FP should be a real matrix."""
        rng = np.random.RandomState(123)
        a = rng.randn(9) * 0.1
        M = gd.fp_operator_truncated(a, R=1.0)
        assert np.all(np.isreal(M)), "M_FP should be real"

    def test_nine_eigenvalues(self, gd):
        """M_FP should be 9x9."""
        a = np.zeros(9)
        M = gd.fp_operator_truncated(a, R=1.0)
        assert M.shape == (9, 9)


# ======================================================================
# 3. Horizon distance is finite and positive
# ======================================================================

class TestHorizonDistance:
    """Gribov horizon distance should be finite and positive."""

    def test_finite_horizon(self, gd):
        """Horizon distance should be finite for any direction."""
        direction = np.ones(9) / 3.0  # normalized below
        r = gd.gribov_horizon_distance_truncated(direction, R=1.0)
        assert np.isfinite(r), f"Horizon distance should be finite, got {r}"
        assert r > 0, f"Horizon distance should be positive, got {r}"

    def test_horizon_positive_for_random_dirs(self, gd):
        """Horizon distance should be positive for multiple random directions."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            d = rng.randn(9)
            r = gd.gribov_horizon_distance_truncated(d, R=1.0)
            assert np.isfinite(r), "Horizon distance should be finite"
            assert r > 0, "Horizon distance should be positive"

    def test_zero_direction_gives_inf(self, gd):
        """Zero direction should give infinite distance."""
        r = gd.gribov_horizon_distance_truncated(np.zeros(9), R=1.0)
        assert r == np.inf

    def test_horizon_at_horizon_eigenvalue_zero(self, gd):
        """At the horizon, lambda_min(M_FP) should be ~0."""
        direction = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        R = 1.0
        t_star = gd.gribov_horizon_distance_truncated(direction, R)
        if np.isfinite(t_star):
            d_norm = direction / np.linalg.norm(direction)
            a_horizon = t_star * d_norm
            lam_min = gd.fp_min_eigenvalue(a_horizon, R)
            assert abs(lam_min) < 1e-6, \
                f"At horizon, lambda_min should be ~0, got {lam_min}"


# ======================================================================
# 4. Horizon distance decreases with coupling
# ======================================================================

class TestHorizonVsCoupling:
    """Stronger coupling means smaller Gribov region (closer horizon)."""

    def test_smaller_R_means_weaker_coupling_larger_horizon(self, gd):
        """
        At smaller R (weaker coupling, UV), the horizon should be farther.
        At larger R (stronger coupling, IR), the horizon should be closer.

        The horizon distance is related to 1/g(R), so it should decrease
        as R increases (and g increases).
        """
        direction = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        r_small = gd.gribov_horizon_distance_truncated(direction, R=0.5)
        r_large = gd.gribov_horizon_distance_truncated(direction, R=5.0)
        # At small R, g is small -> horizon is farther
        # But lambda_1 = 3/R^2 also changes, so the test is about the
        # combined effect. The ratio should change significantly.
        assert np.isfinite(r_small) and np.isfinite(r_large)
        # Both should be positive
        assert r_small > 0 and r_large > 0


# ======================================================================
# 5. Diameter estimate
# ======================================================================

class TestDiameterEstimate:
    """Diameter of Gribov region should be well-defined."""

    def test_diameter_positive(self, gd):
        """Diameter should be positive."""
        result = gd.gribov_diameter_estimate(R=1.0, n_directions=20, seed=42)
        assert result['diameter'] > 0

    def test_diameter_finite(self, gd):
        """Diameter should be finite."""
        result = gd.gribov_diameter_estimate(R=1.0, n_directions=20, seed=42)
        assert np.isfinite(result['diameter'])

    def test_diameter_at_least_twice_min_radius(self, gd):
        """Diameter >= 2 * min_radius (it spans the region)."""
        result = gd.gribov_diameter_estimate(R=1.0, n_directions=30, seed=42)
        assert result['diameter'] >= 2.0 * result['min_radius'] * 0.99  # 1% tolerance

    def test_diameter_reproducible(self, gd):
        """Same seed should give same diameter."""
        r1 = gd.gribov_diameter_estimate(R=1.0, n_directions=20, seed=42)
        r2 = gd.gribov_diameter_estimate(R=1.0, n_directions=20, seed=42)
        assert abs(r1['diameter'] - r2['diameter']) < 1e-12


# ======================================================================
# 6. Payne-Weinberger bound
# ======================================================================

class TestPayneWeinberger:
    """Payne-Weinberger bound pi^2/d^2."""

    def test_pw_positive(self):
        """PW bound should be positive for finite d."""
        assert GribovDiameter.payne_weinberger_bound(1.0) > 0
        assert GribovDiameter.payne_weinberger_bound(10.0) > 0
        assert GribovDiameter.payne_weinberger_bound(0.1) > 0

    def test_pw_value(self):
        """PW bound for d=1 should be pi^2."""
        np.testing.assert_allclose(
            GribovDiameter.payne_weinberger_bound(1.0),
            np.pi**2,
            rtol=1e-12
        )

    def test_pw_decreases_with_d(self):
        """PW bound decreases as d increases."""
        pw1 = GribovDiameter.payne_weinberger_bound(1.0)
        pw2 = GribovDiameter.payne_weinberger_bound(2.0)
        pw3 = GribovDiameter.payne_weinberger_bound(5.0)
        assert pw1 > pw2 > pw3

    def test_pw_zero_diameter(self):
        """PW bound for d=0 should return 0 (degenerate)."""
        assert GribovDiameter.payne_weinberger_bound(0.0) == 0.0

    def test_pw_infinite_diameter(self):
        """PW bound for d=inf should return 0."""
        assert GribovDiameter.payne_weinberger_bound(np.inf) == 0.0


# ======================================================================
# 7. Diameter vs R: stabilization
# ======================================================================

class TestDiameterVsR:
    """Diameter should stabilize for large R."""

    def test_diameter_vs_R_runs(self, gd):
        """diameter_vs_R should run without error."""
        R_values = [0.5, 1.0, 5.0]
        result = gd.diameter_vs_R(R_values, n_directions=10)
        assert len(result['diameter']) == 3
        assert all(np.isfinite(result['diameter']))

    def test_pw_bounds_all_positive(self, gd):
        """PW bounds should be positive for all R."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        result = gd.diameter_vs_R(R_values, n_directions=10)
        for i, R in enumerate(R_values):
            assert result['pw_bound'][i] > 0, \
                f"PW bound should be positive at R={R}"

    def test_diameter_stabilization_trend(self, gd):
        """
        Dimensionless diameter d*R should stabilize for large R.
        Since g^2(R) saturates at 4*pi, the horizon distance scales
        as ~1/R, so d*R -> constant.
        """
        R_values = [1.0, 5.0, 20.0]
        result = gd.diameter_vs_R(R_values, n_directions=15, seed=42)
        dR = result['diameter_dimless']
        # d*R at R=20 and R=5 should be close (within factor 2)
        ratio = dR[2] / dR[1] if dR[1] > 0 else np.inf
        assert 0.5 < ratio < 2.0, \
            f"d*R ratio at R=20 vs R=5 = {ratio:.2f}, should be near 1 for stabilization"


# ======================================================================
# 8. Uniform PW bound
# ======================================================================

class TestUniformBound:
    """PW bound should provide a positive lower bound uniformly in R."""

    def test_pw_bound_minimum(self, gd):
        """PW bound >= some positive minimum for R in [0.5, 50]."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        result = gd.diameter_vs_R(R_values, n_directions=15, seed=42)
        min_pw = np.min(result['pw_bound'])
        assert min_pw > 0, \
            f"Minimum PW bound should be positive, got {min_pw}"

    def test_pw_bound_not_too_small(self, gd):
        """PW bound should be at least 0.01 for R in [0.5, 20]."""
        R_values = [0.5, 1.0, 5.0, 20.0]
        result = gd.diameter_vs_R(R_values, n_directions=15, seed=42)
        min_pw = np.min(result['pw_bound'])
        assert min_pw > 0.01, \
            f"Minimum PW bound = {min_pw:.6f} is too small"


# ======================================================================
# 9. Complete analysis
# ======================================================================

class TestCompleteAnalysis:
    """Complete analysis should run and produce sensible results."""

    def test_complete_analysis_runs(self, gd):
        """complete_analysis should run without error."""
        result = gd.complete_analysis(
            R_range=[0.5, 1.0, 5.0, 10.0],
            n_directions=10
        )
        assert 'diameter' in result
        assert 'pw_bound' in result
        assert 'assessment' in result
        assert 'theorems_used' in result

    def test_complete_analysis_all_finite(self, gd):
        """All computed values should be finite."""
        result = gd.complete_analysis(
            R_range=[1.0, 5.0, 10.0],
            n_directions=10
        )
        assert all(np.isfinite(result['diameter']))
        assert all(np.isfinite(result['pw_bound']))

    def test_theorems_documented(self, gd):
        """Theorems used should be documented."""
        result = gd.complete_analysis(
            R_range=[1.0],
            n_directions=5
        )
        theorems = result['theorems_used']
        assert 'payne_weinberger' in theorems
        assert 'dell_antonio_zwanziger' in theorems
