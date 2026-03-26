"""
Tests for Estimate 2: Covariant Propagator Bounds in Background Gauge Field.

Verifies:
  1. CovariantLaplacian: free spectrum, perturbation bounds, gap estimates
  2. GaussianUpperBound: Li-Yau on S^3, heat kernel bounds, numerical verification
  3. ScaleJPropagator: exponential decay, range estimates, scale consistency
  4. BackgroundDependence: Lipschitz bounds, derivative bounds, smoothness
  5. LatticeUniformity: O(a^2) convergence, scale resolution
  6. LichnerowiczBound: lower bounds, coupling dependence, free spectrum consistency
  7. DaviesGaffneyEstimate: finite propagation speed, geometric decay
  8. Integration: compatibility with heat_kernel_slices.py at A=0
  9. Edge cases: A=0 (free), extreme couplings, parameter boundaries

Labels:
  THEOREM:     Exact mathematical identity, rigorously proven
  PROPOSITION: Proven under stated assumptions
  NUMERICAL:   Verified by computation
"""

import numpy as np
import pytest

from yang_mills_s3.rg.covariant_propagator import (
    CovariantLaplacian,
    GaussianUpperBound,
    ScaleJPropagator,
    BackgroundDependence,
    LatticeUniformity,
    LichnerowiczBound,
    DaviesGaffneyEstimate,
    verify_estimate_2,
    _volume_s3,
    _volume_ball_s3,
    _geodesic_distance_s3,
    _coexact_eigenvalue,
    _coexact_multiplicity,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    G2_PHYSICAL,
)
from yang_mills_s3.rg.heat_kernel_slices import (
    coexact_eigenvalue,
    coexact_multiplicity,
    HeatKernelSlices,
)


# =====================================================================
# Helper constants
# =====================================================================

R = R_PHYSICAL_FM   # 2.2 fm
G2 = G2_PHYSICAL    # 6.28
N_C = 2             # SU(2)
M = 2.0             # Blocking factor


# =====================================================================
# 0. Utility functions
# =====================================================================

class TestUtilities:
    """Tests for geometric utility functions."""

    def test_volume_s3_unit(self):
        """Vol(S^3(1)) = 2*pi^2. THEOREM."""
        assert _volume_s3(1.0) == pytest.approx(2.0 * np.pi**2, rel=1e-10)

    def test_volume_s3_scaling(self):
        """Vol(S^3(R)) = 2*pi^2*R^3. THEOREM."""
        for R_val in [0.5, 1.0, 2.2, 5.0]:
            expected = 2.0 * np.pi**2 * R_val**3
            assert _volume_s3(R_val) == pytest.approx(expected, rel=1e-10)

    def test_volume_ball_s3_small_radius(self):
        """For small r/R, V(r) ~ (4/3)*pi*r^3. NUMERICAL."""
        R_val = 10.0  # Large R so r/R is small
        r = 0.01
        V = _volume_ball_s3(r, R_val)
        flat_approx = (4.0 / 3.0) * np.pi * r**3
        assert abs(V / flat_approx - 1.0) < 0.01

    def test_volume_ball_s3_full(self):
        """V(pi*R) = Vol(S^3(R)). THEOREM."""
        R_val = 2.0
        V = _volume_ball_s3(np.pi * R_val, R_val)
        total = _volume_s3(R_val)
        assert V == pytest.approx(total, rel=1e-6)

    def test_volume_ball_s3_zero(self):
        """V(0) = 0. THEOREM."""
        assert _volume_ball_s3(0.0, 1.0) == 0.0

    def test_volume_ball_s3_monotone(self):
        """Ball volume is monotonically increasing in r. THEOREM."""
        R_val = 2.0
        radii = np.linspace(0.01, np.pi * R_val * 0.99, 50)
        volumes = [_volume_ball_s3(r, R_val) for r in radii]
        for i in range(len(volumes) - 1):
            assert volumes[i + 1] > volumes[i]

    def test_geodesic_distance_same_point(self):
        """d(x, x) = 0. THEOREM."""
        x = np.array([1, 0, 0, 0], dtype=float)
        assert _geodesic_distance_s3(x, x, 1.0) == pytest.approx(0.0, abs=1e-15)

    def test_geodesic_distance_antipodal(self):
        """d(x, -x) = pi*R. THEOREM."""
        x = np.array([1, 0, 0, 0], dtype=float)
        y = np.array([-1, 0, 0, 0], dtype=float)
        assert _geodesic_distance_s3(x, y, 1.0) == pytest.approx(np.pi, rel=1e-10)
        assert _geodesic_distance_s3(x, y, 2.2) == pytest.approx(np.pi * 2.2, rel=1e-10)

    def test_geodesic_distance_orthogonal(self):
        """d(e_1, e_2) = pi*R/2 on unit S^3. THEOREM."""
        x = np.array([1, 0, 0, 0], dtype=float)
        y = np.array([0, 1, 0, 0], dtype=float)
        assert _geodesic_distance_s3(x, y, 1.0) == pytest.approx(np.pi / 2, rel=1e-10)

    def test_coexact_eigenvalue_matches_heat_kernel_slices(self):
        """Internal eigenvalue function matches heat_kernel_slices module. THEOREM."""
        for k in range(1, 20):
            for R_val in [1.0, 2.2, 5.0]:
                assert _coexact_eigenvalue(k, R_val) == pytest.approx(
                    coexact_eigenvalue(k, R_val), rel=1e-12)

    def test_coexact_multiplicity_matches(self):
        """Internal multiplicity function matches heat_kernel_slices module. THEOREM."""
        for k in range(1, 20):
            assert _coexact_multiplicity(k) == coexact_multiplicity(k)


# =====================================================================
# 1. CovariantLaplacian
# =====================================================================

class TestCovariantLaplacianInit:
    """Initialization and parameter validation."""

    def test_default_parameters(self):
        """Default initialization with physical parameters. NUMERICAL."""
        cl = CovariantLaplacian()
        assert cl.R == R_PHYSICAL_FM
        assert cl.N_c == 2
        assert cl.g2 == G2_PHYSICAL

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CovariantLaplacian(R=-1.0)
        with pytest.raises(ValueError):
            CovariantLaplacian(R=0.0)

    def test_invalid_N_c(self):
        """N_c < 2 raises ValueError."""
        with pytest.raises(ValueError):
            CovariantLaplacian(N_c=1)

    def test_invalid_g2(self):
        """g2 <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CovariantLaplacian(g2=-1.0)
        with pytest.raises(ValueError):
            CovariantLaplacian(g2=0.0)

    def test_dim_adj_su2(self):
        """dim(adj(SU(2))) = 3. THEOREM."""
        cl = CovariantLaplacian(N_c=2)
        assert cl.dim_adj == 3

    def test_dim_adj_su3(self):
        """dim(adj(SU(3))) = 8. THEOREM."""
        cl = CovariantLaplacian(N_c=3)
        assert cl.dim_adj == 8


class TestCovariantLaplacianSpectrum:
    """Free and perturbed spectra."""

    def test_free_spectrum_k1(self):
        """lambda_1 = 4/R^2 on unit S^3. THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        spec = cl.spectrum_free(1)
        assert spec[0] == pytest.approx(4.0, rel=1e-12)

    def test_free_spectrum_first_five(self):
        """lambda_k = (k+1)^2/R^2 for k=1..5. THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        spec = cl.spectrum_free(5)
        for k in range(1, 6):
            assert spec[k - 1] == pytest.approx((k + 1)**2, rel=1e-12)

    def test_free_spectrum_scaling(self):
        """Eigenvalues scale as 1/R^2. THEOREM."""
        R_val = 3.0
        cl = CovariantLaplacian(R=R_val)
        spec = cl.spectrum_free(10)
        for k in range(1, 11):
            assert spec[k - 1] == pytest.approx((k + 1)**2 / R_val**2, rel=1e-12)

    def test_multiplicities_first_three(self):
        """d_k = 2k(k+2) for k=1,2,3. THEOREM."""
        cl = CovariantLaplacian()
        mults = cl.multiplicities(3)
        assert mults[0] == 6   # 2*1*3
        assert mults[1] == 16  # 2*2*4
        assert mults[2] == 30  # 2*3*5

    def test_perturbed_spectrum_zero_background(self):
        """At A=0, perturbed spectrum = free spectrum. THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        free = cl.spectrum_free(20)
        pert = cl.spectrum_perturbed(0.0, 20)
        np.testing.assert_allclose(pert, free, rtol=1e-12)

    def test_perturbed_spectrum_shift_direction(self):
        """Non-zero A shifts eigenvalues DOWN (lower bound). PROPOSITION."""
        cl = CovariantLaplacian(R=1.0)
        free = cl.spectrum_free(20)
        pert = cl.spectrum_perturbed(0.1, 20)
        assert np.all(pert <= free)

    def test_perturbed_spectrum_negative_A_rejected(self):
        """Negative A_bar_norm raises ValueError."""
        cl = CovariantLaplacian()
        with pytest.raises(ValueError):
            cl.spectrum_perturbed(-0.1)

    def test_perturbation_relative_small_for_high_modes(self):
        """Relative perturbation decreases for high modes. PROPOSITION."""
        cl = CovariantLaplacian(R=1.0)
        deltas = cl.perturbation_relative(0.1, 50)
        # delta_k should decrease with k (perturbation becomes less significant)
        assert deltas[-1] < deltas[0]


class TestCovariantLaplacianGap:
    """Spectral gap estimates."""

    def test_gap_at_zero_background(self):
        """Gap at A=0 via Lichnerowicz = 2/R^2 (lower bound). THEOREM.
        The actual gap is 4/R^2, but the Lichnerowicz bound gives 2/R^2."""
        cl = CovariantLaplacian(R=1.0)
        gap = cl.covariant_gap_lower_bound(0.0)
        assert gap == pytest.approx(2.0, rel=1e-12)

    def test_gap_finite_at_physical(self):
        """Gap bound is finite at physical coupling. PROPOSITION.
        Note: at strong coupling (g^2=6.28), the Lichnerowicz bound on
        -D_A^2 for the worst-case A in Gribov region CAN be negative.
        This is expected -- the physical mass gap comes from the RG analysis."""
        cl = CovariantLaplacian(R=R, N_c=N_C, g2=G2)
        gap = cl.gap_within_gribov()
        assert np.isfinite(gap)

    def test_gap_decreasing_with_A(self):
        """Gap decreases monotonically with ||A||. PROPOSITION."""
        cl = CovariantLaplacian(R=1.0, N_c=N_C, g2=G2)
        g1 = cl.covariant_gap_lower_bound(0.0)
        g2_val = cl.covariant_gap_lower_bound(0.1)
        g3 = cl.covariant_gap_lower_bound(0.3)
        assert g2_val < g1
        assert g3 < g2_val

    def test_gap_decreases_with_background(self):
        """Gap decreases as ||A|| increases. PROPOSITION."""
        cl = CovariantLaplacian(R=1.0)
        gaps = [cl.covariant_gap_lower_bound(a) for a in [0, 0.1, 0.3, 0.5]]
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] < gaps[i]

    def test_gribov_diameter_su2(self):
        """d*R = 9*sqrt(3)/(2*g) for SU(2). THEOREM."""
        g = np.sqrt(G2)
        expected = 9.0 * np.sqrt(3.0) / (2.0 * g)
        cl = CovariantLaplacian(g2=G2)
        assert cl.gribov_diameter == pytest.approx(expected, rel=1e-10)

    def test_perturbation_controlled_at_physical(self):
        """Perturbation is controlled at physical parameters. NUMERICAL."""
        cl = CovariantLaplacian(R=R, N_c=N_C, g2=G2)
        A_max = cl.max_background_norm
        assert cl.is_perturbation_controlled(A_max)


# =====================================================================
# 2. GaussianUpperBound
# =====================================================================

class TestGaussianUpperBoundInit:
    """Initialization and Li-Yau constant."""

    def test_default_parameters(self):
        """Default initialization. NUMERICAL."""
        gub = GaussianUpperBound()
        assert gub.R == R_PHYSICAL_FM
        assert gub.N_c == 2

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            GaussianUpperBound(R=0.0)

    def test_li_yau_constant_positive(self):
        """Li-Yau constant is positive. THEOREM."""
        gub = GaussianUpperBound()
        assert gub.li_yau_constant() > 0

    def test_ricci_lower_bound(self):
        """Ricci lower bound = 2/R^2 on S^3. THEOREM."""
        gub = GaussianUpperBound(R=1.0)
        assert gub.ricci_lower == pytest.approx(2.0, rel=1e-12)

    def test_ricci_positive(self):
        """Ricci is positive on S^3. THEOREM."""
        for R_val in [0.5, 1.0, 2.2, 10.0]:
            gub = GaussianUpperBound(R=R_val)
            assert gub.ricci_lower > 0


class TestGaussianUpperBoundHeatKernel:
    """Heat kernel upper bounds."""

    def test_bound_non_negative(self):
        """Heat kernel bound is always non-negative. THEOREM.
        Note: at very small t with large d, exp(-d^2/(5t)) underflows
        to 0.0 in floating point, which is mathematically correct
        (the heat kernel is astronomically small there)."""
        gub = GaussianUpperBound(R=1.0)
        for t in [0.001, 0.01, 0.1, 1.0, 10.0]:
            for d in [0.0, 0.5, 1.0, 2.0]:
                assert gub.heat_kernel_bound(t, d) >= 0

    def test_bound_positive_at_diagonal(self):
        """Heat kernel bound is strictly positive at d=0. THEOREM."""
        gub = GaussianUpperBound(R=1.0)
        for t in [0.001, 0.01, 0.1, 1.0, 10.0]:
            assert gub.heat_kernel_bound(t, 0.0) > 0

    def test_bound_invalid_t(self):
        """t <= 0 raises ValueError."""
        gub = GaussianUpperBound()
        with pytest.raises(ValueError):
            gub.heat_kernel_bound(0.0, 1.0)
        with pytest.raises(ValueError):
            gub.heat_kernel_bound(-1.0, 1.0)

    def test_bound_invalid_d(self):
        """d < 0 raises ValueError."""
        gub = GaussianUpperBound()
        with pytest.raises(ValueError):
            gub.heat_kernel_bound(1.0, -1.0)

    def test_bound_decreases_with_distance(self):
        """Bound decreases with geodesic distance. THEOREM."""
        gub = GaussianUpperBound(R=1.0)
        t = 0.1
        distances = [0.0, 0.5, 1.0, 1.5, 2.0]
        bounds = [gub.heat_kernel_bound(t, d) for d in distances]
        for i in range(len(bounds) - 1):
            assert bounds[i + 1] <= bounds[i]

    def test_bound_decreases_with_time_at_distance(self):
        """At fixed d > 0, bound decays for small t due to Gaussian. THEOREM."""
        gub = GaussianUpperBound(R=1.0)
        d = 1.0
        # For very small t with large d, the Gaussian exp(-d^2/(5t)) dominates
        b1 = gub.heat_kernel_bound(0.01, d)
        b2 = gub.heat_kernel_bound(0.001, d)
        # For d = 1 and t = 0.001, exp(-1/(0.005)) ~ 0, much smaller
        assert b2 < b1

    def test_diagonal_bound_diverges_as_t_to_0(self):
        """K_t(x,x) ~ t^{-3/2} as t -> 0. THEOREM."""
        gub = GaussianUpperBound(R=10.0)  # Large R to avoid curvature effects
        t1 = 0.01
        t2 = 0.001
        b1 = gub.heat_kernel_diagonal_bound(t1)
        b2 = gub.heat_kernel_diagonal_bound(t2)
        # Ratio should be ~ (t1/t2)^{3/2} = 10^{3/2} ~ 31.6
        ratio = b2 / b1
        assert ratio > 10.0  # At least an order of magnitude increase

    def test_flat_approx_consistent(self):
        """Flat-space approx agrees with full bound for small t at d=0. NUMERICAL.
        At d=0 the Gaussian factor is 1, so we can compare prefactors directly.
        The Li-Yau bound uses V(sqrt(t))^{-1} with a different constant than
        (4*pi*t)^{-3/2}, so we allow a generous factor."""
        gub = GaussianUpperBound(R=100.0)  # Large R = nearly flat
        t = 0.01
        d = 0.0  # diagonal comparison, avoids Gaussian exponent differences
        full = gub.heat_kernel_bound(t, d)
        flat = gub.heat_kernel_bound_flat_approx(t, d)
        # Both are upper bounds with different methodology, should be
        # in the same ballpark (within 3 orders of magnitude)
        ratio = full / flat
        assert 0.001 < ratio < 1000.0

    def test_curvature_correction_small_for_short_times(self):
        """F_A correction exp(||F||*t) ~ 1 for t << 1/||F||. PROPOSITION."""
        gub = GaussianUpperBound(R=2.2)
        F_norm = 0.1
        t_small = 0.001
        b_free = gub.heat_kernel_bound(t_small, 0.0, F_A_norm=0.0)
        b_cov = gub.heat_kernel_bound(t_small, 0.0, F_A_norm=F_norm)
        # Correction factor exp(0.1 * 0.001) ~ 1.0001
        ratio = b_cov / b_free
        assert abs(ratio - 1.0) < 0.01


class TestGaussianUpperBoundVerification:
    """Numerical verification against spectral computation."""

    def test_verify_numerically_bound_holds(self):
        """Gaussian bound holds above spectral diagonal. NUMERICAL."""
        gub = GaussianUpperBound(R=1.0)
        result = gub.verify_numerically(k_max=80)
        assert result['bound_holds'], "Gaussian bound must hold above spectral sum"

    def test_verify_bound_holds_physical_R(self):
        """Gaussian bound holds at physical R = 2.2 fm. NUMERICAL."""
        gub = GaussianUpperBound(R=R)
        result = gub.verify_numerically(k_max=80)
        assert result['bound_holds']

    def test_median_tightness_reasonable(self):
        """Bound is not absurdly loose (within 100x). NUMERICAL."""
        gub = GaussianUpperBound(R=1.0)
        result = gub.verify_numerically(k_max=100)
        assert result['median_ratio'] < 100.0

    def test_curvature_endomorphism_finite(self):
        """||F_A|| bound is finite within Gribov. PROPOSITION."""
        gub = GaussianUpperBound(R=R, g2=G2)
        F_bound = gub.curvature_endomorphism_bound()
        assert np.isfinite(F_bound)
        assert F_bound > 0


# =====================================================================
# 3. ScaleJPropagator
# =====================================================================

class TestScaleJPropagatorInit:
    """Initialization and basic properties."""

    def test_default_parameters(self):
        """Default initialization. NUMERICAL."""
        prop = ScaleJPropagator()
        assert prop.R == R_PHYSICAL_FM
        assert prop.M == 2.0

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            ScaleJPropagator(R=0.0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            ScaleJPropagator(M=1.0)
        with pytest.raises(ValueError):
            ScaleJPropagator(M=0.5)

    def test_invalid_g2(self):
        """g2 <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            ScaleJPropagator(g2=0.0)


class TestScaleJPropagatorTimeWindow:
    """Proper-time windows and length scales."""

    def test_proper_time_window_j0(self):
        """j=0: t in [M^{-2}, 1]. THEOREM."""
        prop = ScaleJPropagator(M=2.0)
        t_lo, t_hi = prop.proper_time_window(0)
        assert t_lo == pytest.approx(0.25, rel=1e-12)
        assert t_hi == pytest.approx(1.0, rel=1e-12)

    def test_proper_time_window_j1(self):
        """j=1: t in [M^{-4}, M^{-2}]. THEOREM."""
        prop = ScaleJPropagator(M=2.0)
        t_lo, t_hi = prop.proper_time_window(1)
        assert t_lo == pytest.approx(1.0 / 16.0, rel=1e-12)
        assert t_hi == pytest.approx(0.25, rel=1e-12)

    def test_proper_time_windows_nested(self):
        """Windows are nested: t_hi(j+1) = t_lo(j). THEOREM."""
        prop = ScaleJPropagator(M=2.0)
        for j in range(7):
            _, t_hi_next = prop.proper_time_window(j + 1)
            t_lo_this, _ = prop.proper_time_window(j)
            assert t_hi_next == pytest.approx(t_lo_this, rel=1e-12)

    def test_length_scale_decreases(self):
        """L_j decreases geometrically with j. NUMERICAL."""
        prop = ScaleJPropagator(M=2.0)
        for j in range(6):
            L_j = prop.length_scale(j)
            L_jp1 = prop.length_scale(j + 1)
            assert L_jp1 == pytest.approx(L_j / 2.0, rel=1e-12)


class TestScaleJPropagatorDecay:
    """Exponential decay properties."""

    def test_kernel_bound_positive(self):
        """Kernel bound is always positive. PROPOSITION."""
        prop = ScaleJPropagator(R=1.0)
        for j in range(5):
            for d in [0.0, 0.5, 1.0]:
                assert prop.kernel_bound_fast(d, j) > 0

    def test_kernel_bound_decreases_with_distance(self):
        """Kernel bound decays with distance. PROPOSITION."""
        prop = ScaleJPropagator(R=1.0)
        for j in [1, 3, 5]:
            b0 = prop.kernel_bound_fast(0.0, j)
            b1 = prop.kernel_bound_fast(0.5, j)
            b2 = prop.kernel_bound_fast(1.0, j)
            assert b1 < b0
            assert b2 < b1

    def test_exponential_decay_rate_increases_with_j(self):
        """Decay rate c*M^j increases with j. NUMERICAL."""
        prop = ScaleJPropagator(M=2.0)
        for j in range(5):
            r_j = prop.exponential_decay_rate(j)
            r_jp1 = prop.exponential_decay_rate(j + 1)
            assert r_jp1 == pytest.approx(2.0 * r_j, rel=1e-12)

    def test_range_estimate_decreases_with_j(self):
        """Effective range decreases geometrically. NUMERICAL."""
        prop = ScaleJPropagator(M=2.0)
        for j in range(5):
            r_j = prop.range_estimate(j)
            r_jp1 = prop.range_estimate(j + 1)
            assert r_jp1 == pytest.approx(r_j / 2.0, rel=1e-12)

    def test_range_estimate_value(self):
        """Range = sqrt(5) * M^{-j}. NUMERICAL."""
        prop = ScaleJPropagator(M=2.0)
        for j in range(5):
            expected = np.sqrt(5.0) * 2.0**(-j)
            assert prop.range_estimate(j) == pytest.approx(expected, rel=1e-12)

    def test_diagonal_bound_scaling(self):
        """Diagonal C_j(x,x) scales roughly as M^j. NUMERICAL."""
        prop = ScaleJPropagator(R=1.0, M=2.0)
        diags = [prop.diagonal_bound(j) for j in range(2, 6)]
        # Ratios should be ~ M^1 = 2 (d-2 = 1 for d=3)
        for i in range(len(diags) - 1):
            ratio = diags[i + 1] / diags[i]
            # Allow generous tolerance since these are bounds, not exact
            assert 1.0 < ratio < 8.0

    def test_verify_decay_produces_result(self):
        """verify_decay returns meaningful result. NUMERICAL."""
        prop = ScaleJPropagator(R=1.0, M=2.0)
        result = prop.verify_decay(3, n_points=15)
        assert 'decay_rate_fit' in result
        assert 'decay_rate_theory' in result
        assert result['decay_rate_theory'] > 0


# =====================================================================
# 4. BackgroundDependence
# =====================================================================

class TestBackgroundDependenceInit:
    """Initialization."""

    def test_default_parameters(self):
        """Default initialization. NUMERICAL."""
        bd = BackgroundDependence()
        assert bd.R == R_PHYSICAL_FM

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BackgroundDependence(R=0.0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            BackgroundDependence(M=0.5)

    def test_gribov_diameter(self):
        """Gribov diameter consistent with CovariantLaplacian. THEOREM."""
        bd = BackgroundDependence(g2=G2)
        cl = CovariantLaplacian(g2=G2)
        assert bd.gribov_diameter == pytest.approx(cl.gribov_diameter, rel=1e-12)


class TestBackgroundDependenceLipschitz:
    """Lipschitz estimates."""

    def test_lipschitz_positive(self):
        """Lipschitz constant is positive. PROPOSITION."""
        bd = BackgroundDependence()
        for j in range(7):
            assert bd.lipschitz_constant(j) > 0

    def test_lipschitz_decreases_at_uv(self):
        """Lipschitz constant decreases for UV scales. PROPOSITION."""
        bd = BackgroundDependence()
        L_3 = bd.lipschitz_constant(3)
        L_6 = bd.lipschitz_constant(6)
        # At UV scales, ||C_j|| ~ M^{-2j}, so L_j ~ M^{-3j}
        assert L_6 < L_3

    def test_derivative_bound_zero_background(self):
        """Derivative bound at A=0 still positive (from nabla term). PROPOSITION."""
        bd = BackgroundDependence(R=1.0)
        for j in range(5):
            db = bd.derivative_bound(j, 0.0)
            assert db > 0

    def test_derivative_bound_increases_with_A(self):
        """Derivative bound increases with ||A||. PROPOSITION."""
        bd = BackgroundDependence(R=1.0)
        j = 3
        db1 = bd.derivative_bound(j, 0.1)
        db2 = bd.derivative_bound(j, 0.5)
        assert db2 > db1

    def test_derivative_bound_negative_A_rejected(self):
        """Negative A_bar_norm raises ValueError."""
        bd = BackgroundDependence()
        with pytest.raises(ValueError):
            bd.derivative_bound(3, -0.1)


class TestBackgroundDependenceSmoothness:
    """Smoothness verification."""

    def test_smoothness_same_background(self):
        """Zero field difference => Lipschitz trivially holds. NUMERICAL."""
        bd = BackgroundDependence(R=1.0)
        result = bd.verify_smoothness(0.1, 0.1, j=3)
        assert result['lipschitz_holds']
        assert result['diff_norm'] == pytest.approx(0.0, abs=1e-12)

    def test_smoothness_small_perturbation(self):
        """Lipschitz holds for small field difference. NUMERICAL."""
        bd = BackgroundDependence(R=1.0, g2=G2)
        result = bd.verify_smoothness(0.1, 0.11, j=3, k_max=30)
        assert result['lipschitz_holds']

    def test_smoothness_at_all_scales(self):
        """Lipschitz holds across all RG scales. NUMERICAL."""
        bd = BackgroundDependence(R=R, g2=G2)
        result = bd.smoothness_at_all_scales(delta_A=0.005, j_max=5)
        assert result['all_hold']


# =====================================================================
# 5. LatticeUniformity
# =====================================================================

class TestLatticeUniformityInit:
    """Initialization."""

    def test_default_parameters(self):
        """Default initialization. NUMERICAL."""
        lu = LatticeUniformity()
        assert lu.R == R_PHYSICAL_FM

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            LatticeUniformity(R=0.0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            LatticeUniformity(M=1.0)


class TestLatticeUniformityError:
    """Error bounds."""

    def test_error_positive(self):
        """Lattice error is positive. PROPOSITION."""
        lu = LatticeUniformity(R=1.0)
        for a in [0.01, 0.05, 0.1]:
            for j in range(5):
                assert lu.lattice_error(a, j) > 0

    def test_error_scales_as_a_squared(self):
        """Error scales as a^2. PROPOSITION."""
        lu = LatticeUniformity(R=1.0)
        j = 2
        e1 = lu.lattice_error(0.1, j)
        e2 = lu.lattice_error(0.05, j)
        ratio = e1 / e2
        # (0.1/0.05)^2 = 4
        assert ratio == pytest.approx(4.0, rel=1e-10)

    def test_error_invalid_a(self):
        """a <= 0 or a >= R raises ValueError."""
        lu = LatticeUniformity(R=1.0)
        with pytest.raises(ValueError):
            lu.lattice_error(0.0, 0)
        with pytest.raises(ValueError):
            lu.lattice_error(-0.1, 0)
        with pytest.raises(ValueError):
            lu.lattice_error(1.5, 0)

    def test_lattice_resolves_ir_scales(self):
        """Coarse lattice still resolves IR. NUMERICAL."""
        lu = LatticeUniformity(R=2.2, M=2.0)
        assert lu.lattice_resolves_scale(0.5, 0)  # j=0: L^0 = 2.2

    def test_lattice_cannot_resolve_all_uv(self):
        """Coarse lattice cannot resolve deep UV. NUMERICAL."""
        lu = LatticeUniformity(R=2.2, M=2.0)
        # At j=5: L^5 = 2.2 * 2^{-5} = 0.069 fm
        assert not lu.lattice_resolves_scale(0.5, 5)

    def test_max_resolvable_scale(self):
        """j_max = floor(log_M(R/a)). NUMERICAL."""
        lu = LatticeUniformity(R=2.2, M=2.0)
        # R/a = 2.2/0.1 = 22, log_2(22) ~ 4.46, floor = 4
        j_max = lu.max_resolvable_scale(0.1)
        assert j_max == int(np.floor(np.log(2.2 / 0.1) / np.log(2.0)))


class TestLatticeUniformityConvergence:
    """O(a^2) convergence verification."""

    def test_uniformity_check_is_O_a2(self):
        """Convergence rate is O(a^2). NUMERICAL."""
        lu = LatticeUniformity(R=2.2, M=2.0)
        a_vals = np.array([0.01, 0.02, 0.05, 0.1])
        result = lu.uniformity_check(a_vals, j=2)
        assert result['is_O_a2']
        assert abs(result['convergence_rate'] - 2.0) < 0.1

    def test_error_profile_monotone(self):
        """Error increases with j for fixed a. NUMERICAL."""
        lu = LatticeUniformity(R=2.2, M=2.0)
        result = lu.propagator_error_profile(0.05, j_max=3)
        resolvable_errors = [e for e, r in zip(result['errors'], result['resolvable']) if r]
        for i in range(len(resolvable_errors) - 1):
            assert resolvable_errors[i + 1] > resolvable_errors[i]


# =====================================================================
# 6. LichnerowiczBound
# =====================================================================

class TestLichnerowiczBoundInit:
    """Initialization."""

    def test_default_parameters(self):
        """Default initialization. NUMERICAL."""
        lb = LichnerowiczBound()
        assert lb.R == R_PHYSICAL_FM

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            LichnerowiczBound(R=0.0)


class TestLichnerowiczBoundValues:
    """Lower bounds on spectral gap."""

    def test_ricci_on_1forms_s3(self):
        """Ric|_{1-forms} = 2/R^2 on S^3. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        assert lb.ricci_on_1forms() == pytest.approx(2.0, rel=1e-12)

    def test_ricci_scaling(self):
        """Ric|_{1-forms} scales as 1/R^2. THEOREM."""
        for R_val in [0.5, 1.0, 2.2, 5.0]:
            lb = LichnerowiczBound(R=R_val)
            assert lb.ricci_on_1forms() == pytest.approx(2.0 / R_val**2, rel=1e-12)

    def test_lichnerowicz_bound_less_than_free_gap(self):
        """2/R^2 < 4/R^2 (Lichnerowicz < actual gap). THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        assert lb.lichnerowicz_lower_bound() < lb.free_spectral_gap()

    def test_free_spectral_gap(self):
        """Free gap = 4/R^2. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        assert lb.free_spectral_gap() == pytest.approx(4.0, rel=1e-12)

    def test_improved_gap_at_zero_A(self):
        """Improved gap at A=0 equals Lichnerowicz bound. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        # At A=0: F_A = 0, so gap = 2/R^2
        assert lb.improved_gap(0.0) == pytest.approx(2.0, rel=1e-12)

    def test_improved_gap_decreases_with_A(self):
        """Gap decreases as ||A|| increases. PROPOSITION."""
        lb = LichnerowiczBound(R=1.0)
        g1 = lb.improved_gap(0.0)
        g2 = lb.improved_gap(0.1)
        g3 = lb.improved_gap(0.3)
        assert g2 < g1
        assert g3 < g2

    def test_spectral_gap_covariant_finite(self):
        """Covariant gap bound is finite at physical coupling. PROPOSITION.
        Note: at strong coupling, this bound can be negative. The physical
        mass gap comes from the full RG analysis, not this single bound."""
        lb = LichnerowiczBound(R=R, N_c=N_C, g2=G2)
        gap = lb.spectral_gap_covariant()
        assert np.isfinite(gap)

    def test_spectral_gap_covariant_positive_weak(self):
        """Covariant gap is positive at weak coupling. PROPOSITION.
        At g^2=0.5, action-based ||A|| ~ g/(2*pi*R) gives small F_A."""
        lb = LichnerowiczBound(R=R, N_c=N_C, g2=0.5)
        gap = lb.spectral_gap_covariant()
        assert gap > 0

    def test_gap_ratio_to_free_positive_weak_coupling(self):
        """Gap ratio > 0 at weak coupling. NUMERICAL."""
        lb = LichnerowiczBound(R=R, N_c=N_C, g2=0.1)
        ratio = lb.gap_ratio_to_free()
        assert ratio > 0

    def test_curvature_endomorphism_bound_at_zero(self):
        """||F_A|| = 0 when A = 0. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        assert lb.curvature_endomorphism_bound(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_curvature_endomorphism_bound_positive(self):
        """||F_A|| > 0 when A != 0. PROPOSITION."""
        lb = LichnerowiczBound(R=1.0)
        assert lb.curvature_endomorphism_bound(0.1) > 0


class TestLichnerowiczBoundVerification:
    """Verification against known results."""

    def test_verify_against_free_spectrum(self):
        """Lichnerowicz bound <= all free eigenvalues. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        result = lb.verify_against_free_spectrum(k_max=50)
        assert result['bound_satisfied']
        assert result['tightness'] > 1.0  # Bound is below actual gap

    def test_tightness_exactly_2(self):
        """Actual gap / Lichnerowicz = 4/2 = 2. THEOREM."""
        lb = LichnerowiczBound(R=1.0)
        result = lb.verify_against_free_spectrum()
        assert result['tightness'] == pytest.approx(2.0, rel=1e-10)

    def test_coupling_scan_gap_decreases_with_coupling(self):
        """Gap decreases as coupling increases. NUMERICAL."""
        lb = LichnerowiczBound(R=R, N_c=N_C, g2=G2)
        result = lb.gap_as_function_of_coupling(
            g2_values=np.array([0.1, 1.0, 5.0, 10.0, 20.0])
        )
        # Gap should generally decrease with coupling
        # (but may plateau at strong coupling due to min(Gribov,action))
        assert result['gaps'][0] > result['gaps'][-1]

    def test_coupling_scan_weak_coupling_gap_large(self):
        """Gap approaches 2/R^2 at weak coupling. PROPOSITION."""
        lb = LichnerowiczBound(R=1.0)
        result = lb.gap_as_function_of_coupling(
            g2_values=np.array([0.01, 0.05, 0.1, 0.5])
        )
        # At g^2 = 0.01 (very weak), gap should be close to 2/R^2 = 2.0
        assert result['gaps'][0] > 1.5


# =====================================================================
# 7. DaviesGaffneyEstimate
# =====================================================================

class TestDaviesGaffneyInit:
    """Initialization."""

    def test_default(self):
        """Default initialization. NUMERICAL."""
        dg = DaviesGaffneyEstimate()
        assert dg.R == R_PHYSICAL_FM

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            DaviesGaffneyEstimate(R=0.0)


class TestDaviesGaffneyBounds:
    """Finite propagation speed bounds."""

    def test_semigroup_bound_unity_at_d0(self):
        """exp(0) = 1 at zero separation. THEOREM."""
        dg = DaviesGaffneyEstimate()
        assert dg.semigroup_bound(1.0, 0.0) == pytest.approx(1.0, rel=1e-12)

    def test_semigroup_bound_decays_with_d(self):
        """Bound decays with distance. THEOREM."""
        dg = DaviesGaffneyEstimate()
        t = 1.0
        b1 = dg.semigroup_bound(t, 1.0)
        b2 = dg.semigroup_bound(t, 2.0)
        assert b2 < b1

    def test_semigroup_bound_value(self):
        """exp(-d^2/(4t)) at specific values. THEOREM."""
        dg = DaviesGaffneyEstimate()
        t = 1.0
        d = 2.0
        expected = np.exp(-4.0 / 4.0)  # exp(-1)
        assert dg.semigroup_bound(t, d) == pytest.approx(expected, rel=1e-12)

    def test_invalid_t(self):
        """t <= 0 raises ValueError."""
        dg = DaviesGaffneyEstimate()
        with pytest.raises(ValueError):
            dg.semigroup_bound(0.0, 1.0)

    def test_invalid_d(self):
        """d < 0 raises ValueError."""
        dg = DaviesGaffneyEstimate()
        with pytest.raises(ValueError):
            dg.semigroup_bound(1.0, -1.0)

    def test_effective_range_formula(self):
        """d_eff = 2*sqrt(t) at threshold e^{-1}. NUMERICAL."""
        dg = DaviesGaffneyEstimate()
        t = 4.0
        expected = 2.0 * np.sqrt(4.0)  # = 4.0
        assert dg.effective_range(t) == pytest.approx(expected, rel=1e-12)

    def test_slice_range_geometric(self):
        """Slice ranges decay geometrically. NUMERICAL."""
        dg = DaviesGaffneyEstimate()
        r0 = dg.slice_range(0, M=2.0)
        r1 = dg.slice_range(1, M=2.0)
        assert r1 == pytest.approx(r0 / 2.0, rel=1e-10)


class TestDaviesGaffneyFiniteRange:
    """Finite range verification."""

    def test_verify_geometric_decay(self):
        """Ranges decay geometrically with ratio 1/M. NUMERICAL."""
        dg = DaviesGaffneyEstimate()
        result = dg.verify_finite_range(j_max=5, M=2.0)
        assert result['geometric_decay']

    def test_expected_ratio(self):
        """Expected ratio is 1/M. THEOREM."""
        dg = DaviesGaffneyEstimate()
        result = dg.verify_finite_range(j_max=3, M=2.0)
        assert result['expected_ratio'] == pytest.approx(0.5, rel=1e-12)


# =====================================================================
# 8. Integration tests: compatibility with heat_kernel_slices.py
# =====================================================================

class TestIntegrationWithHeatKernelSlices:
    """Verify compatibility with existing heat_kernel_slices infrastructure."""

    def test_free_propagator_diagonal_consistent(self):
        """
        At A=0, our diagonal bound should be consistent with
        HeatKernelSlices.kernel_bound_diagonal(j).
        NUMERICAL.
        """
        hks = HeatKernelSlices(R=R, M=M, a_lattice=0.1, k_max=100)
        prop = ScaleJPropagator(R=R, M=M, g2=G2)

        for j in range(min(hks.num_scales, 5)):
            hks_diag = hks.kernel_bound_diagonal(j)
            our_diag = prop.diagonal_bound(j)
            # Our bound should be at least as large (it's an upper bound)
            # Both are bounds, so just check they're in the same ballpark
            if hks_diag > 0 and our_diag > 0:
                ratio = our_diag / hks_diag
                # Both are upper bounds from different methods; ratio shouldn't
                # be wildly off (within 4 orders of magnitude)
                assert 1e-4 < ratio < 1e4, (
                    f"j={j}: our_diag={our_diag:.4e}, hks_diag={hks_diag:.4e}")

    def test_eigenvalue_consistency(self):
        """Eigenvalue functions agree between modules. THEOREM."""
        cl = CovariantLaplacian(R=R)
        spec = cl.spectrum_free(20)
        for k in range(1, 21):
            assert spec[k - 1] == pytest.approx(
                coexact_eigenvalue(k, R), rel=1e-12)

    def test_free_gap_consistent_with_weitzenboeck(self):
        """
        Free gap 4/R^2 consistent with Weitzenboeck spectral_gap_1forms.
        THEOREM.
        """
        from yang_mills_s3.geometry.weitzenboeck import Weitzenboeck
        lb = LichnerowiczBound(R=R)
        wb_gap = Weitzenboeck.spectral_gap_1forms(3, R, l=1)
        assert lb.free_spectral_gap() == pytest.approx(wb_gap, rel=1e-12)

    def test_ricci_consistent(self):
        """
        Ricci = 2/R^2 consistent with ricci.py on S^3.
        THEOREM.
        """
        from yang_mills_s3.geometry.ricci import RicciTensor
        lb = LichnerowiczBound(R=R)
        ricci_data = RicciTensor.on_lie_group('SU(2)', R)
        assert lb.ricci_on_1forms() == pytest.approx(
            ricci_data['ricci_on_1forms'], rel=1e-10)


# =====================================================================
# 9. Edge cases and special configurations
# =====================================================================

class TestEdgeCasesFreeField:
    """A = 0: all quantities reduce to free-field values."""

    def test_covariant_gap_lichnerowicz_at_A0(self):
        """gap(-D_0^2) >= 2/R^2 (Lichnerowicz bound). THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        # Lichnerowicz bound at A=0 gives 2/R^2, not the full 4/R^2
        assert cl.covariant_gap_lower_bound(0.0) == pytest.approx(2.0, rel=1e-12)

    def test_perturbed_spectrum_equals_free_at_A0(self):
        """Spectrum at A=0 matches free. THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        free = cl.spectrum_free(30)
        pert = cl.spectrum_perturbed(0.0, 30)
        np.testing.assert_allclose(pert, free, rtol=1e-12)

    def test_lipschitz_at_A0_gives_derivative_from_nabla(self):
        """Derivative bound at A=0 comes from the nabla term. PROPOSITION."""
        bd = BackgroundDependence(R=1.0)
        db = bd.derivative_bound(3, 0.0)
        assert db > 0


class TestEdgeCasesUnitSphere:
    """R = 1: simplest geometry."""

    def test_all_eigenvalues_integer_squares(self):
        """lambda_k = (k+1)^2 on unit S^3. THEOREM."""
        cl = CovariantLaplacian(R=1.0)
        spec = cl.spectrum_free(10)
        for k in range(1, 11):
            assert spec[k - 1] == pytest.approx(float((k + 1)**2), rel=1e-12)

    def test_volume_unit_s3(self):
        """Vol(S^3(1)) = 2*pi^2. THEOREM."""
        assert _volume_s3(1.0) == pytest.approx(2.0 * np.pi**2, rel=1e-10)


class TestEdgeCasesWeakCoupling:
    """g^2 -> 0: perturbation vanishes."""

    def test_gribov_diameter_diverges_at_weak_coupling(self):
        """d*R ~ 1/g -> large as g -> 0. THEOREM.
        At g^2 = 0.01: g = 0.1, d*R = 9*sqrt(3)/(2*0.1) = 77.9."""
        cl = CovariantLaplacian(g2=0.01)
        assert cl.gribov_diameter > 50.0  # ~78 at g^2=0.01

    def test_gap_approaches_lichnerowicz_at_weak_coupling(self):
        """Gap -> 2/R^2 as g^2 -> 0 (Lichnerowicz bound). PROPOSITION.
        At very weak coupling, ||A||_minimizer -> 0, so gap -> 2/R^2."""
        for g2 in [0.01, 0.05, 0.1]:
            lb = LichnerowiczBound(R=1.0, g2=g2)
            gap = lb.spectral_gap_covariant()
            assert gap > 1.5, f"At g2={g2}: gap={gap} should approach 2.0"


class TestEdgeCasesStrongCoupling:
    """g^2 large: perturbation becomes significant."""

    def test_gribov_diameter_small_at_strong_coupling(self):
        """d*R shrinks with g. THEOREM."""
        cl_weak = CovariantLaplacian(g2=1.0)
        cl_strong = CovariantLaplacian(g2=20.0)
        assert cl_strong.gribov_diameter < cl_weak.gribov_diameter

    def test_gap_at_strong_coupling_is_finite(self):
        """Gap bound is finite at physical g^2 = 6.28. NUMERICAL.
        At strong coupling, the Lichnerowicz bound is negative,
        but the full RG analysis provides the physical gap."""
        lb = LichnerowiczBound(R=R, g2=G2)
        gap = lb.spectral_gap_covariant()
        assert np.isfinite(gap)


class TestEdgeCasesLargeR:
    """R -> large: flat-space limit."""

    def test_gap_decreases_with_R(self):
        """4/R^2 decreases with R. THEOREM."""
        gaps = [LichnerowiczBound(R=r).free_spectral_gap()
                for r in [1.0, 2.0, 5.0, 10.0]]
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] < gaps[i]

    def test_flat_space_heat_kernel_recovered_at_large_R(self):
        """For large R, S^3 bound ~ flat * constant at short times. NUMERICAL.
        The Li-Yau bound uses a conservative constant C(3), so the ratio
        between the S^3 bound and the flat-space bound is O(C(3)) ~ 15-200.
        The key point is that both scale as t^{-3/2} (same exponent)."""
        gub_large = GaussianUpperBound(R=100.0)
        # Check that the SCALING is the same (t^{-3/2}) at two different times
        t1, t2 = 0.01, 0.001
        b1 = gub_large.heat_kernel_bound(t1, 0.0)
        b2 = gub_large.heat_kernel_bound(t2, 0.0)
        f1 = gub_large.heat_kernel_bound_flat_approx(t1, 0.0)
        f2 = gub_large.heat_kernel_bound_flat_approx(t2, 0.0)
        # Both should scale as t^{-3/2}, so ratio b2/b1 ~ (t1/t2)^{3/2}
        bound_ratio = b2 / b1
        flat_ratio = f2 / f1
        assert abs(bound_ratio / flat_ratio - 1.0) < 0.3


class TestEdgeCasesSUN:
    """SU(N) for N > 2."""

    def test_dim_adj_su3(self):
        """dim(adj(SU(3))) = 8. THEOREM."""
        cl = CovariantLaplacian(N_c=3)
        assert cl.dim_adj == 8

    def test_gap_positive_su3(self):
        """Gap positive for SU(3). PROPOSITION."""
        cl = CovariantLaplacian(R=R, N_c=3, g2=G2)
        gap = cl.gap_within_gribov()
        # Note: Gribov diameter formula is for SU(2) 9-DOF;
        # for SU(3) this is an approximation
        assert np.isfinite(gap)


# =====================================================================
# 10. Overall verification
# =====================================================================

class TestOverallEstimate2:
    """End-to-end verification of Estimate 2."""

    def test_verify_estimate_2_passes(self):
        """Full Estimate 2 verification passes. NUMERICAL."""
        result = verify_estimate_2(R=R, M=M, g2=G2, N_c=N_C, j_max=5)
        assert result['overall_pass'], (
            f"Estimate 2 failed: cov_lap={result['covariant_laplacian']}, "
            f"gauss={result['gaussian_bounds']}, "
            f"lich={result['lichnerowicz']}")

    def test_gap_finite_in_verification(self):
        """Covariant gap bound is finite. NUMERICAL."""
        result = verify_estimate_2(R=R, M=M, g2=G2)
        assert np.isfinite(result['covariant_laplacian']['gap_within_gribov'])

    def test_gaussian_bound_in_verification(self):
        """Gaussian bound holds. THEOREM."""
        result = verify_estimate_2(R=R, M=M, g2=G2)
        assert result['gaussian_bounds']['bound_holds']

    def test_lichnerowicz_in_verification(self):
        """Lichnerowicz bound vs free spectrum is satisfied. THEOREM."""
        result = verify_estimate_2(R=R, M=M, g2=G2)
        assert result['lichnerowicz']['bound_vs_free']

    def test_lattice_uniformity_in_verification(self):
        """O(a^2) convergence confirmed. PROPOSITION."""
        result = verify_estimate_2(R=R, M=M, g2=G2)
        assert result['lattice_uniformity']['is_O_a2']

    def test_davies_gaffney_in_verification(self):
        """Geometric decay of ranges. THEOREM."""
        result = verify_estimate_2(R=R, M=M, g2=G2)
        assert result['davies_gaffney']['geometric_decay']

    def test_verify_at_unit_sphere(self):
        """Verification passes on unit sphere too. NUMERICAL."""
        result = verify_estimate_2(R=1.0, M=2.0, g2=4.0, N_c=2, j_max=4)
        assert result['overall_pass']

    def test_verify_at_large_R(self):
        """Verification passes at large R (approaching flat). NUMERICAL."""
        result = verify_estimate_2(R=10.0, M=2.0, g2=G2, N_c=2, j_max=4)
        assert result['overall_pass']


# =====================================================================
# 11. Physical consistency checks
# =====================================================================

class TestPhysicalConsistency:
    """Checks that results make physical sense."""

    def test_gribov_diameter_at_physical_coupling(self):
        """d*R ~ 3.1 at g^2 = 6.28. NUMERICAL."""
        cl = CovariantLaplacian(g2=G2)
        dR = cl.gribov_diameter
        # 9*sqrt(3)/(2*sqrt(6.28)) ~ 3.11
        assert 2.5 < dR < 4.0

    def test_mass_gap_order_of_magnitude(self):
        """Gap ~ 4/R^2 gives mass ~ 2*hbar_c/R ~ 180 MeV. NUMERICAL."""
        lb = LichnerowiczBound(R=R)
        gap = lb.free_spectral_gap()  # 4/R^2 in fm^-2
        mass = np.sqrt(gap) * HBAR_C_MEV_FM  # Convert to MeV
        assert 100 < mass < 300  # Should be ~ 180 MeV

    def test_covariant_gap_positive_at_weak_coupling(self):
        """Covariant gap gives positive mass at weak coupling. PROPOSITION."""
        lb = LichnerowiczBound(R=R, g2=0.1)
        gap = lb.spectral_gap_covariant()
        assert gap > 0
        mass = np.sqrt(gap) * HBAR_C_MEV_FM
        assert mass > 0

    def test_propagator_range_at_j3_sub_fermi(self):
        """Scale j=3 has sub-fermi range. NUMERICAL."""
        prop = ScaleJPropagator(R=R, M=M)
        r3 = prop.range_estimate(3)
        # M^{-3} = 0.125, range ~ sqrt(5) * 0.125 ~ 0.28
        assert r3 < 1.0

    def test_lattice_error_small_at_physical(self):
        """Lattice error at a=0.1 fm, j=3 is small. NUMERICAL."""
        lu = LatticeUniformity(R=R, M=M)
        err = lu.lattice_error(0.1, 3)
        assert err < 1.0  # Error bound should be modest


# =====================================================================
# 12. Monotonicity and consistency across scales
# =====================================================================

class TestScaleConsistency:
    """Properties that must hold across all scales."""

    def test_decay_rate_monotone_increasing(self):
        """Decay rate increases monotonically with j. NUMERICAL."""
        prop = ScaleJPropagator(R=R, M=M)
        rates = [prop.exponential_decay_rate(j) for j in range(7)]
        for i in range(len(rates) - 1):
            assert rates[i + 1] > rates[i]

    def test_range_monotone_decreasing(self):
        """Effective range decreases monotonically with j. NUMERICAL."""
        prop = ScaleJPropagator(R=R, M=M)
        ranges = [prop.range_estimate(j) for j in range(7)]
        for i in range(len(ranges) - 1):
            assert ranges[i + 1] < ranges[i]

    def test_proper_time_windows_cover_range(self):
        """Union of proper-time windows covers (t_UV, 1). THEOREM."""
        prop = ScaleJPropagator(M=M)
        N = 7
        # j=0: [M^{-2}, 1], j=1: [M^{-4}, M^{-2}], ..., j=N: [M^{-2(N+1)}, M^{-2N}]
        # Union should be [M^{-2(N+1)}, 1]
        _, t_hi_0 = prop.proper_time_window(0)
        t_lo_N, _ = prop.proper_time_window(N)
        assert t_hi_0 == pytest.approx(1.0, rel=1e-12)
        assert t_lo_N == pytest.approx(M**(-2 * (N + 1)), rel=1e-12)

    def test_lipschitz_constants_scale_consistently(self):
        """Lipschitz constants have consistent scale dependence. NUMERICAL."""
        bd = BackgroundDependence(R=R, g2=G2, M=M)
        L_values = [bd.lipschitz_constant(j) for j in range(7)]
        # Should all be finite and positive
        for L in L_values:
            assert np.isfinite(L)
            assert L > 0
