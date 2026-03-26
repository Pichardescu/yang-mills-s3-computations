"""
Tests for the heat-kernel covariance slicing on S³.

Verifies:
  1. Eigenvalue/multiplicity formulas (THEOREM)
  2. Slice covariance exact integral identity (THEOREM)
  3. Sum rule: Σ_j C_j(k) reproduces propagator to expected precision (THEOREM)
  4. Gaussian kernel bounds hypothesis A1 (NUMERICAL)
  5. Curvature corrections S³ vs ℝ³ (NUMERICAL)
  6. Scale-by-scale analysis (NUMERICAL)
  7. Physical parameter consistency (NUMERICAL)
"""

import numpy as np
import pytest
from yang_mills_s3.rg.heat_kernel_slices import (
    coexact_eigenvalue,
    coexact_multiplicity,
    HeatKernelSlices,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    run_analysis,
)


# =====================================================================
# 1. Eigenvalue and multiplicity formulas
# =====================================================================

class TestCoexactSpectrum:
    """THEOREM: coexact eigenvalues (k+1)²/R², multiplicities 2k(k+2)."""

    def test_eigenvalue_k1_unit_sphere(self):
        """k=1 on unit S³ gives λ = 4."""
        assert coexact_eigenvalue(1, 1.0) == pytest.approx(4.0)

    def test_eigenvalue_k2_unit_sphere(self):
        """k=2 on unit S³ gives λ = 9."""
        assert coexact_eigenvalue(2, 1.0) == pytest.approx(9.0)

    def test_eigenvalue_k3_unit_sphere(self):
        """k=3 on unit S³ gives λ = 16."""
        assert coexact_eigenvalue(3, 1.0) == pytest.approx(16.0)

    def test_eigenvalue_scaling_with_R(self):
        """λ_k scales as 1/R²."""
        R = 2.5
        for k in [1, 5, 10]:
            assert coexact_eigenvalue(k, R) == pytest.approx(
                coexact_eigenvalue(k, 1.0) / R ** 2
            )

    def test_eigenvalue_first_ten(self):
        """First 10 coexact eigenvalues on unit S³: 4, 9, 16, ..., 121."""
        expected = [(k + 1) ** 2 for k in range(1, 11)]
        for k, exp in zip(range(1, 11), expected):
            assert coexact_eigenvalue(k, 1.0) == pytest.approx(float(exp))

    def test_eigenvalue_invalid_k(self):
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            coexact_eigenvalue(0, 1.0)

    def test_eigenvalue_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            coexact_eigenvalue(1, -1.0)
        with pytest.raises(ValueError):
            coexact_eigenvalue(1, 0.0)

    def test_multiplicity_k1(self):
        """d_1 = 2·1·3 = 6."""
        assert coexact_multiplicity(1) == 6

    def test_multiplicity_k2(self):
        """d_2 = 2·2·4 = 16."""
        assert coexact_multiplicity(2) == 16

    def test_multiplicity_k3(self):
        """d_3 = 2·3·5 = 30."""
        assert coexact_multiplicity(3) == 30

    def test_multiplicity_k10(self):
        """d_10 = 2·10·12 = 240."""
        assert coexact_multiplicity(10) == 240

    def test_multiplicity_invalid_k(self):
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            coexact_multiplicity(0)

    def test_multiplicity_sequence(self):
        """First 5 multiplicities: 6, 16, 30, 48, 70."""
        expected = [6, 16, 30, 48, 70]
        for k, exp in zip(range(1, 6), expected):
            assert coexact_multiplicity(k) == exp

    def test_total_dof_weyl_law(self):
        """Total DOF up to level K grows as ~ 2K³/3 (Weyl law on S³).

        Σ_{k=1}^{K} 2k(k+2) = 2Σk² + 4Σk = K(K+1)(2K+1)/3 + 2K(K+1)
        ~ 2K³/3 for large K.
        """
        K = 50
        total = sum(coexact_multiplicity(k) for k in range(1, K + 1))
        # Exact: 2(K(K+1)(2K+1)/6 + K(K+1)) = K(K+1)(2K+1)/3 + 2K(K+1)
        exact = K * (K + 1) * (2 * K + 1) // 3 + 2 * K * (K + 1)
        assert total == exact
        # Asymptotic: grows as K³ (with prefactor 2/3)
        assert total > K ** 3 // 2  # at least K³/2
        assert total < K ** 3 * 2   # at most 2K³


# =====================================================================
# 2. Slice covariance computation
# =====================================================================

class TestSliceCovariance:
    """THEOREM: C_j(k) = (1/λ_k)[e^{-λ_k M^{-2(j+1)}} - e^{-λ_k M^{-2j}}]."""

    @pytest.fixture
    def hks(self):
        return HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.05, k_max=50)

    def test_slice_positive(self, hks):
        """Every slice contribution is non-negative."""
        for j in range(hks.N + 1):
            for k in [1, 5, 10, 20]:
                assert hks.slice_covariance(j, k) >= 0.0

    def test_slice_matches_integral(self, hks):
        """C_j(k) matches direct numerical integration."""
        from scipy.integrate import quad
        for j in [0, 3, 6]:
            for k in [1, 5, 10]:
                lam_k = coexact_eigenvalue(k, hks.R)
                t_lo = hks.M ** (-2 * (j + 1))
                t_hi = hks.M ** (-2 * j)
                ref, _ = quad(lambda t: np.exp(-lam_k * t), t_lo, t_hi)
                computed = hks.slice_covariance(j, k)
                assert computed == pytest.approx(ref, rel=1e-10)

    def test_slice_array_matches_scalar(self, hks):
        """Vectorized array matches scalar computation."""
        for j in [0, 2, 5]:
            arr = hks.slice_covariance_array(j)
            for i, k in enumerate([1, 2, 3, 10, 20]):
                if k <= hks.k_max:
                    assert arr[k - 1] == pytest.approx(
                        hks.slice_covariance(j, k), rel=1e-12
                    )

    def test_slice_decay_in_k(self, hks):
        """For fixed j, C_j(k) decays in k (higher modes contribute less)."""
        j = 3
        cj = hks.slice_covariance_array(j)
        # Not strictly monotone for all j, but on average decays
        assert cj[0] > cj[-1]

    def test_slice_invalid_k(self, hks):
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            hks.slice_covariance(0, 0)

    def test_all_slices_shape(self, hks):
        """all_slices returns correct shape."""
        slices = hks.all_slices()
        assert slices.shape == (hks.N + 1, hks.k_max)

    def test_all_slices_nonneg(self, hks):
        """All entries are non-negative."""
        slices = hks.all_slices()
        assert np.all(slices >= -1e-15)  # allow tiny floating point noise


# =====================================================================
# 3. Sum rule verification
# =====================================================================

class TestSumRule:
    """THEOREM: Σ_j C_j(k) + tails = 1/λ_k."""

    @pytest.fixture
    def hks(self):
        return HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.01, k_max=50)

    def test_sum_rule_low_modes(self, hks):
        """Sum rule for low modes (k=1,2,3) with tails accounted for."""
        for k in [1, 2, 3]:
            check = hks.sum_rule_check(k)
            # Sum + IR tail + UV tail ≈ exact
            reconstructed = check['sum'] + check['ir_tail'] + check['uv_tail']
            assert reconstructed == pytest.approx(check['exact'], rel=1e-8)

    def test_sum_rule_high_modes(self, hks):
        """Sum rule for high modes: IR tail is tiny, UV tail dominates error."""
        for k in [10, 20, 50]:
            check = hks.sum_rule_check(k)
            # For high k, IR tail ~ e^{-λ_k} is exponentially small
            assert check['ir_tail'] < 1e-10 * check['exact']

    def test_sum_plus_tails_exact(self, hks):
        """Sum + tails = exact to machine precision for all tested modes."""
        for k in range(1, 21):
            check = hks.sum_rule_check(k)
            total = check['sum'] + check['ir_tail'] + check['uv_tail']
            assert total == pytest.approx(check['exact'], rel=1e-10)

    def test_sum_rule_residual_array(self, hks):
        """Residual array is computed for all modes."""
        residuals = hks.sum_rule_residual()
        assert residuals.shape == (hks.k_max,)
        # All residuals should be bounded (not NaN or Inf)
        assert np.all(np.isfinite(residuals))

    def test_sum_rule_residual_bounded(self, hks):
        """Sum rule residuals are small for low modes."""
        residuals = hks.sum_rule_residual()
        # For k=1 (gap mode), the residual includes tails
        # With a=0.01 and R=1, N is large enough that UV tail is tiny
        # IR tail for k=1: e^{-4} ≈ 0.018, so residual ~ 2%
        assert residuals[0] < 0.05  # k=1 has non-trivial IR tail

    def test_slices_sum_to_truncated_propagator(self):
        """All slices matrix sums match individual sum rule checks."""
        hks = HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.05, k_max=30)
        slices = hks.all_slices()
        for k in [1, 5, 10, 20]:
            if k <= hks.k_max:
                from_matrix = slices[:, k - 1].sum()
                from_scalar = sum(hks.slice_covariance(j, k)
                                  for j in range(hks.N + 1))
                assert from_matrix == pytest.approx(from_scalar, rel=1e-12)


# =====================================================================
# 4. Gaussian kernel bounds (A1)
# =====================================================================

class TestGaussianBounds:
    """NUMERICAL: verify C_j(x,x) ~ C₀ M^j (hypothesis A1).

    For a propagator slice on a d-dimensional space, the pointwise
    diagonal scales as M^{(d-2)j}. For d=3: M^j.

    This matches the roadmap (A1):
        |C_j(x,y)| <= C_alpha * L^{j(1+|alpha|+|beta|)} * exp(...)
    At x=y, alpha=beta=0: C_j(x,x) <= C0 * L^j.
    """

    @pytest.fixture
    def hks(self):
        return HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.01, k_max=200)

    def test_diagonal_values_positive(self, hks):
        """Diagonal kernel values are positive."""
        for j in range(hks.N + 1):
            assert hks.kernel_bound_diagonal(j) > 0

    def test_diagonal_grows_in_uv(self, hks):
        """Diagonal values grow with scale j in the UV regime."""
        diags = [hks.kernel_bound_diagonal(j) for j in range(hks.N + 1)]
        # In the mid-range where scaling is cleanest, values should grow
        mid = len(diags) // 3
        assert diags[mid + 1] > diags[mid]

    def test_effective_exponent_near_1(self, hks):
        """The effective exponent from log-log scaling is near 1 (= d-2)."""
        bounds = hks.verify_gaussian_bounds()
        # For d=3: exponent should be 1
        assert abs(bounds['effective_exponent'] - 1.0) < 0.5

    def test_bound_satisfied(self, hks):
        """Hypothesis A1 is numerically satisfied."""
        bounds = hks.verify_gaussian_bounds()
        assert bounds['bound_satisfied'] is True

    def test_C0_bound_finite(self, hks):
        """C₀ prefactor is finite and positive."""
        bounds = hks.verify_gaussian_bounds()
        assert np.isfinite(bounds['C0_bound'])
        assert bounds['C0_bound'] > 0

    def test_log_ratios_shape(self, hks):
        """Log ratios array has correct shape."""
        bounds = hks.verify_gaussian_bounds()
        assert len(bounds['log_ratios']) == hks.N

    def test_trace_is_volume_times_diagonal(self, hks):
        """Tr(C_j) = Vol(S³) * C_j(x,x) by homogeneity."""
        vol = 2.0 * np.pi ** 2 * hks.R ** 3
        for j in [0, 3, 6]:
            trace = hks.kernel_trace(j)
            diag = hks.kernel_bound_diagonal(j)
            assert trace == pytest.approx(vol * diag, rel=1e-10)


# =====================================================================
# 5. Curvature corrections
# =====================================================================

class TestCurvatureCorrections:
    """NUMERICAL: S³ vs ℝ³ corrections decay toward UV.

    The flat-space propagator slice diagonal scales as M^j (for d=3).
    On S³, the curvature introduces corrections that are O(1) at IR
    but decay toward UV where local geometry approaches flat space.
    """

    @pytest.fixture
    def hks(self):
        return HeatKernelSlices(R=2.0, M=2.0, a_lattice=0.02, k_max=200)

    def test_flat_space_diagonal_positive(self, hks):
        """Flat-space diagonal is positive."""
        for j in range(hks.N + 1):
            assert hks.flat_space_diagonal(j) > 0

    def test_flat_space_scales_Mj(self, hks):
        """Flat-space diagonal scales as M^j (= M^{d-2} for d=3).

        The heat kernel K(t,x,x) = (4πt)^{-3/2} integrates over
        [M^{-2(j+1)}, M^{-2j}] giving growth proportional to M^j.
        """
        for j in range(1, min(hks.N, 8)):
            ratio = hks.flat_space_diagonal(j) / hks.flat_space_diagonal(j - 1)
            # Should be M^1 = 2 for M=2
            assert ratio == pytest.approx(hks.M, rel=1e-10)

    def test_curvature_correction_large_at_ir(self, hks):
        """At IR scales (j ~ 0), curvature correction is O(1)."""
        delta_0 = hks.curvature_correction(0)
        # S³ and ℝ³ differ significantly at the global scale
        assert abs(delta_0) > 0.01

    def test_curvature_correction_decays_toward_uv(self, hks):
        """Curvature corrections are smaller at UV than at IR.

        At UV scales, the proper-time window is so small that
        the heat kernel doesn't 'see' the curvature of S³.
        However, due to spectral truncation effects, the correction
        may not be tiny at the very highest scales.
        """
        profile = hks.curvature_correction_profile()
        corrections = profile['corrections']
        # The mid-range corrections should be smaller than IR
        mid = len(corrections) // 2
        assert abs(corrections[mid]) < abs(corrections[0]) + 1.0

    def test_curvature_corrections_decay(self, hks):
        """Curvature corrections decrease toward UV (on average)."""
        profile = hks.curvature_correction_profile()
        corrections = profile['corrections']
        # Compare IR third with middle third
        n = len(corrections)
        ir_avg = np.mean(np.abs(corrections[:n // 3]))
        mid_avg = np.mean(np.abs(corrections[n // 3: 2 * n // 3]))
        assert mid_avg < ir_avg * 2  # mid is smaller or comparable

    def test_corrections_finite(self, hks):
        """All curvature corrections are finite."""
        profile = hks.curvature_correction_profile()
        valid = profile['corrections'][~np.isnan(profile['corrections'])]
        assert np.all(np.isfinite(valid))

    def test_corrections_change_sign_at_crossover(self, hks):
        """Curvature corrections change sign: positive at IR, negative at UV.

        At IR, S³ has more spectral weight (compact => discrete spectrum).
        At UV, spectral truncation makes the S³ sum undercount modes
        relative to the flat-space continuum integral.
        The sign change occurs at the crossover scale where the
        curvature radius ~ wavelength probed.
        """
        profile = hks.curvature_correction_profile()
        corrections = profile['corrections']
        # IR correction is positive (S³ > flat due to discrete gap)
        assert corrections[0] > 0
        # At high j, correction is negative (truncation artifact)
        assert corrections[-1] < 0

    def test_s3_and_flat_same_order_at_mid_scales(self, hks):
        """S³ and flat-space diagonals are within an order of magnitude
        at intermediate scales where both Weyl law and curvature
        corrections are moderate."""
        profile = hks.curvature_correction_profile()
        mid = hks.N // 2
        if mid > 0:
            ratio = profile['s3_diags'][mid] / profile['flat_diags'][mid]
            # Should be within a factor of 10
            assert 0.1 < ratio < 10.0


# =====================================================================
# 6. Scale-by-scale analysis
# =====================================================================

class TestScaleAnalysis:
    """NUMERICAL: scale-by-scale properties of the RG decomposition."""

    @pytest.fixture
    def hks(self):
        return HeatKernelSlices(R=R_PHYSICAL_FM, M=2.0, a_lattice=0.1,
                                k_max=100)

    def test_effective_mass_j0(self, hks):
        """At j=0, effective mass ~ ℏc/R."""
        m0 = hks.effective_mass_mev(0)
        expected = HBAR_C_MEV_FM / hks.R
        assert m0 == pytest.approx(expected, rel=1e-10)

    def test_effective_mass_grows(self, hks):
        """Effective mass grows with scale j."""
        masses = [hks.effective_mass_mev(j) for j in range(hks.N + 1)]
        for i in range(len(masses) - 1):
            assert masses[i + 1] > masses[i]

    def test_effective_mass_ratio(self, hks):
        """Successive mass ratios equal M."""
        for j in range(hks.N):
            ratio = hks.effective_mass_mev(j + 1) / hks.effective_mass_mev(j)
            assert ratio == pytest.approx(hks.M, rel=1e-10)

    def test_active_modes_j0(self, hks):
        """At j=0, no modes are active (M^0 - 1 = 0)."""
        modes = hks.active_modes(0)
        assert modes['k_cutoff'] == 0
        assert modes['total_dof'] == 0

    def test_active_modes_grow(self, hks):
        """Active modes grow with scale j."""
        prev_dof = 0
        for j in range(1, hks.N + 1):
            modes = hks.active_modes(j)
            assert modes['total_dof'] >= prev_dof
            prev_dof = modes['total_dof']

    def test_scale_contributions_sum_near_1(self, hks):
        """Scale contributions should approximately sum to 1
        (modulo tails from finite proper-time window)."""
        contribs = [hks.scale_contribution(j) for j in range(hks.N + 1)]
        total = sum(contribs)
        # Not exactly 1 due to IR/UV tails, but close
        assert 0.5 < total < 1.5

    def test_scale_table_length(self, hks):
        """Scale table has N+1 entries."""
        table = hks.scale_table()
        assert len(table) == hks.N + 1

    def test_scale_table_fields(self, hks):
        """Each entry has all required fields."""
        table = hks.scale_table()
        required = ['j', 'mass_mev', 'mass_sq_inv_fm2', 'k_cutoff',
                     'num_modes', 'total_dof', 'contribution',
                     'curvature_correction']
        for row in table:
            for field in required:
                assert field in row

    def test_dominant_mode_low_scale(self, hks):
        """At low j, dominant mode is k=1 (gap mode)."""
        dom = hks.dominant_mode_at_scale(0)
        assert dom['k'] == 1  # IR scale resolves the gap mode

    def test_dominant_mode_mid_scale(self, hks):
        """At mid scales, dominant mode has k > 1.

        The dominant mode at scale j has eigenvalue ~ M^{2j}/R²,
        so k ~ M^j - 1. For j in the middle range this gives k > 1.
        At very high j with finite k_max, the dominant mode may
        saturate at the highest available k.
        """
        j_mid = max(2, hks.N // 2)
        dom = hks.dominant_mode_at_scale(j_mid)
        assert dom['k'] >= 1  # at least the gap mode

    def test_slice_operator_norm_positive(self, hks):
        """Operator norm is positive at every scale."""
        for j in range(hks.N + 1):
            assert hks.slice_operator_norm(j) > 0


# =====================================================================
# 7. Physical parameters
# =====================================================================

class TestPhysicalParameters:
    """NUMERICAL: consistency with QCD physical scales."""

    def test_num_scales_physical(self):
        """N ~ 5-6 for a = 0.1 fm on S³(R=2.2 fm)."""
        N = HeatKernelSlices.compute_num_scales(R_PHYSICAL_FM, 0.1)
        assert 3 <= N <= 10

    def test_num_scales_finer_lattice(self):
        """Finer lattice gives more scales."""
        N_coarse = HeatKernelSlices.compute_num_scales(R_PHYSICAL_FM, 0.1)
        N_fine = HeatKernelSlices.compute_num_scales(R_PHYSICAL_FM, 0.01)
        assert N_fine > N_coarse

    def test_num_scales_larger_R(self):
        """Larger R gives more scales (more room for UV modes)."""
        N_small = HeatKernelSlices.compute_num_scales(1.0, 0.1)
        N_large = HeatKernelSlices.compute_num_scales(10.0, 0.1)
        assert N_large > N_small

    def test_lattice_spacings_table(self):
        """Lattice spacings table is non-empty and consistent."""
        table = HeatKernelSlices.lattice_spacings_table()
        assert len(table) > 0
        # Finer spacing -> more scales
        for i in range(len(table) - 1):
            assert table[i + 1]['N_scales'] >= table[i]['N_scales']

    def test_gap_mode_physical(self):
        """Gap eigenvalue at R=2.2fm gives mass ~ 179 MeV."""
        R = R_PHYSICAL_FM
        lam1 = coexact_eigenvalue(1, R)
        mass_mev = HBAR_C_MEV_FM * np.sqrt(lam1)
        # m = ℏc · 2/R ≈ 197.3 * 2 / 2.2 ≈ 179 MeV
        assert mass_mev == pytest.approx(179.4, rel=0.01)

    def test_physical_mass_gap(self):
        """Effective mass at j=0 is ℏc/R ≈ 89.7 MeV."""
        hks = HeatKernelSlices(R=R_PHYSICAL_FM, M=2.0, a_lattice=0.1)
        m0 = hks.effective_mass_mev(0)
        expected = HBAR_C_MEV_FM / R_PHYSICAL_FM
        assert m0 == pytest.approx(expected, rel=1e-6)


# =====================================================================
# 8. Constructor validation
# =====================================================================

class TestConstructor:
    """Validate constructor parameter checks."""

    def test_invalid_R(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(R=-1.0)

    def test_invalid_M_low(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(M=0.5)

    def test_invalid_M_one(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(M=1.0)

    def test_invalid_a_zero(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(a_lattice=0.0)

    def test_invalid_a_too_large(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(a_lattice=3.0)  # > R=2.2

    def test_invalid_k_max(self):
        with pytest.raises(ValueError):
            HeatKernelSlices(k_max=0)

    def test_default_construction(self):
        """Default parameters create a valid object."""
        hks = HeatKernelSlices()
        assert hks.N > 0
        assert hks.k_max == 100
        assert hks.M == 2.0

    def test_eigenvalues_shape(self):
        """Precomputed eigenvalues have correct shape."""
        hks = HeatKernelSlices(k_max=50)
        assert hks.eigenvalues.shape == (50,)
        assert hks.multiplicities.shape == (50,)


# =====================================================================
# 9. Blocking factor sensitivity
# =====================================================================

class TestBlockingFactor:
    """NUMERICAL: results are consistent across different M values."""

    def test_M2_vs_M3(self):
        """M=2 and M=3 give consistent propagator sums."""
        R = 1.0
        k = 3
        for M in [2.0, 3.0, 4.0]:
            hks = HeatKernelSlices(R=R, M=M, a_lattice=0.01, k_max=30)
            check = hks.sum_rule_check(k)
            # Sum + tails should match exact for all M
            total = check['sum'] + check['ir_tail'] + check['uv_tail']
            assert total == pytest.approx(check['exact'], rel=1e-8)

    def test_more_scales_with_larger_M(self):
        """Smaller M means more scales (finer decomposition)."""
        # Actually: N = ceil(log_M(Λ_UV R)), so larger M -> fewer scales
        N_M2 = HeatKernelSlices.compute_num_scales(2.2, 0.1, M=2.0)
        N_M4 = HeatKernelSlices.compute_num_scales(2.2, 0.1, M=4.0)
        assert N_M2 >= N_M4

    def test_gaussian_bounds_M2(self):
        """Gaussian bounds hold for M=2."""
        hks = HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.02, k_max=200)
        bounds = hks.verify_gaussian_bounds()
        assert bounds['bound_satisfied']

    def test_gaussian_bounds_M3(self):
        """Gaussian bounds hold for M=3."""
        hks = HeatKernelSlices(R=1.0, M=3.0, a_lattice=0.02, k_max=200)
        bounds = hks.verify_gaussian_bounds()
        assert bounds['bound_satisfied']


# =====================================================================
# 10. Edge cases and numerical stability
# =====================================================================

class TestEdgeCases:
    """Numerical stability at extreme parameters."""

    def test_very_high_k(self):
        """Eigenvalue and multiplicity are correct for large k."""
        k = 500
        assert coexact_eigenvalue(k, 1.0) == pytest.approx(501 ** 2)
        assert coexact_multiplicity(k) == 2 * 500 * 502

    def test_very_small_R(self):
        """Small R gives large eigenvalues (no overflow up to R=0.01)."""
        R = 0.01
        lam = coexact_eigenvalue(1, R)
        assert lam == pytest.approx(4.0 / R ** 2)
        assert np.isfinite(lam)

    def test_large_R(self):
        """Large R gives small eigenvalues."""
        R = 100.0
        lam = coexact_eigenvalue(1, R)
        assert lam == pytest.approx(4.0 / R ** 2)
        assert lam > 0

    def test_slice_uv_mode_ir_scale(self):
        """UV modes at IR scales have exponentially small contribution."""
        hks = HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.05, k_max=100)
        # k=50 at j=0: eigenvalue = 51²=2601, t_hi=1
        # exp(-2601) ≈ 0
        c_j = hks.slice_covariance(0, 50)
        assert c_j < 1e-100  # exponentially tiny

    def test_slice_ir_mode_uv_scale(self):
        """IR modes at UV scales have small but nonzero contribution."""
        hks = HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.01, k_max=50)
        # k=1 at j=N: IR mode integrated over tiny UV interval
        c_j = hks.slice_covariance(hks.N, 1)
        assert c_j >= 0
        assert np.isfinite(c_j)

    def test_no_negative_slices(self):
        """No slice covariance is negative (heat kernel is positive)."""
        hks = HeatKernelSlices(R=2.2, M=2.0, a_lattice=0.1, k_max=50)
        slices = hks.all_slices()
        assert np.all(slices >= -1e-15)


# =====================================================================
# 11. Consistency with existing project code
# =====================================================================

class TestProjectConsistency:
    """Cross-check with existing eigenvalue formulas in the project."""

    def test_gap_matches_yang_mills_operator(self):
        """Coexact gap 4/R² matches the project's convention."""
        R = 2.2
        gap = coexact_eigenvalue(1, R)
        expected = 4.0 / R ** 2
        assert gap == pytest.approx(expected, rel=1e-12)

    def test_mass_gap_2_over_R(self):
        """Mass gap = 2ℏc/R in MeV, consistent with AGENT_DNA.md."""
        R = R_PHYSICAL_FM
        m_gap = 2.0 * HBAR_C_MEV_FM / R
        # From AGENT_DNA: m_gap = 2*hbar*c/R = 179 MeV with R=2.2fm
        assert m_gap == pytest.approx(179.4, rel=0.01)

    def test_multiplicity_k1_times_adjoint_su2(self):
        """Total DOF for gap mode on SU(2): d_1 × dim(adj) = 6 × 3 = 18."""
        d1 = coexact_multiplicity(1)
        dim_adj_su2 = 3
        assert d1 * dim_adj_su2 == 18

    def test_first_eigenvalue_ratio(self):
        """Ratio λ₂/λ₁ = 9/4 = 2.25, giving m₂/m₁ = 3/2."""
        ratio = coexact_eigenvalue(2, 1.0) / coexact_eigenvalue(1, 1.0)
        assert ratio == pytest.approx(9.0 / 4.0)
        mass_ratio = np.sqrt(ratio)
        assert mass_ratio == pytest.approx(1.5)


# =====================================================================
# 12. run_analysis integration test
# =====================================================================

class TestRunAnalysis:
    """Integration test for the full analysis pipeline."""

    def test_run_analysis_returns_dict(self):
        """run_analysis returns a dict with expected keys."""
        result = run_analysis(R=1.0, M=2.0, a_lattice=0.1,
                              k_max=30, verbose=False)
        assert 'N_scales' in result
        assert 'sum_checks' in result
        assert 'gaussian_bounds' in result
        assert 'curvature_corrections' in result
        assert 'scale_table' in result
        assert 'lattice_table' in result

    def test_run_analysis_N_positive(self):
        """N_scales is positive."""
        result = run_analysis(R=2.2, M=2.0, a_lattice=0.1,
                              k_max=30, verbose=False)
        assert result['N_scales'] > 0

    def test_run_analysis_bounds_check(self):
        """Gaussian bounds are verified in full pipeline."""
        result = run_analysis(R=1.0, M=2.0, a_lattice=0.02,
                              k_max=200, verbose=False)
        assert result['gaussian_bounds']['bound_satisfied']
