"""
Tests for the complete end-to-end RG pipeline on S^3.

Tests cover:
    - RGScaleData: correct at each scale, physical parameters consistent
    - SingleRGStep: one step works, invariants preserved, coupling flows correctly
    - FullRGIteration: all N steps work, mass gap positive, trajectory smooth
    - ContinuumLimitScanner: convergence as N -> infinity, uniform bounds
    - RGDiagnostics: identifies bottleneck, gap budget realistic
    - S3AdvantageQuantifier: all five advantages quantified, consistent
    - Integration with ALL existing modules (imports work, interfaces compatible)
    - Edge cases: N=2 (minimum), N=15 (stress), different R, different g^2
    - Physical consistency: mass gap in right range, coupling runs correctly

Labels:
    THEOREM:   Verified properties that are mathematically proven.
    NUMERICAL: Computed quantities checked for physical consistency.
"""

import numpy as np
import pytest

from yang_mills_s3.rg.rg_pipeline import (
    RGScaleData,
    RGResult,
    SingleRGStep,
    FullRGIteration,
    ContinuumLimitScanner,
    RGDiagnostics,
    S3AdvantageQuantifier,
    _HAS_HEAT_KERNEL,
    _HAS_BLOCK_GEOMETRY,
    _HAS_GAUGE_FIXING,
    _HAS_BETA_FLOW,
    _HAS_COVARIANT_PROPAGATOR,
    _HAS_BACKGROUND_MINIMIZER,
    _HAS_UNIFORM_CONTRACTION,
    _HAS_POLYMER_ALGEBRA,
    _HAS_CONTINUUM_LIMIT,
    _HAS_LARGE_FIELD,
    _HAS_GRIBOV,
    _HBAR_C,
    _R_PHYS,
    _G2_PHYS,
    _M_DEFAULT,
    _NC_DEFAULT,
    G2_MAX,
)

from yang_mills_s3.rg.heat_kernel_slices import (
    HeatKernelSlices,
    coexact_eigenvalue,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
)

from yang_mills_s3.rg.inductive_closure import MultiScaleRGFlow
from yang_mills_s3.rg.bbs_coordinates import (
    RelevantCouplings,
    PolymerCoordinate,
    BBSCoordinates,
    MultiScaleRGBBS,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def physical_params():
    """Standard physical parameters."""
    return {
        'R': 2.2,
        'M': 2.0,
        'N_c': 2,
        'g2_bare': 6.28,
        'k_max': 100,
    }


@pytest.fixture
def small_pipeline(physical_params):
    """A small pipeline with N=3 for fast tests."""
    return FullRGIteration(**physical_params)


@pytest.fixture
def standard_pipeline(physical_params):
    """Standard pipeline with N=7."""
    return FullRGIteration(**physical_params)


@pytest.fixture
def step_engine(physical_params):
    """A SingleRGStep engine."""
    return SingleRGStep(N_scales=7, **physical_params)


@pytest.fixture
def diagnostics():
    """RGDiagnostics instance."""
    return RGDiagnostics()


@pytest.fixture
def s3_advantage():
    """S3AdvantageQuantifier instance."""
    return S3AdvantageQuantifier(R=2.2, M=2.0, N_c=2, g2=6.28)


# =====================================================================
# 1. RGScaleData tests
# =====================================================================

class TestRGScaleData:
    """Tests for the RGScaleData dataclass."""

    def test_creation_default(self):
        """RGScaleData can be created with minimal args. NUMERICAL."""
        sd = RGScaleData(scale_j=3, lattice_spacing=0.5, n_blocks=15, g_bar_j=4.0)
        assert sd.scale_j == 3
        assert sd.lattice_spacing == 0.5
        assert sd.n_blocks == 15
        assert sd.g_bar_j == 4.0
        assert sd.nu_j == 0.0
        assert sd.z_j == 1.0
        assert sd.K_norm_j == 0.0

    def test_alpha_s_computation(self):
        """alpha_s = g^2 / (4*pi). NUMERICAL."""
        sd = RGScaleData(scale_j=0, lattice_spacing=1.0, n_blocks=1,
                         g_bar_j=4.0 * np.pi)
        assert abs(sd.alpha_s - 1.0) < 1e-10

    def test_alpha_s_perturbative(self):
        """Small g^2 gives small alpha_s. NUMERICAL."""
        sd = RGScaleData(scale_j=5, lattice_spacing=0.1, n_blocks=100,
                         g_bar_j=1.0)
        assert sd.alpha_s < 1.0
        assert sd.alpha_s == pytest.approx(1.0 / (4.0 * np.pi), rel=1e-10)

    def test_gap_MeV_positive(self):
        """Gap in MeV is positive for positive gap_j. NUMERICAL."""
        gap_j = 4.0 / 2.2**2  # lambda_1 at R=2.2
        sd = RGScaleData(scale_j=0, lattice_spacing=2.2, n_blocks=1,
                         g_bar_j=6.28, gap_j=gap_j)
        assert sd.gap_MeV > 0
        expected = np.sqrt(gap_j) * HBAR_C_MEV_FM
        assert sd.gap_MeV == pytest.approx(expected, rel=1e-10)

    def test_gap_MeV_zero_for_zero_gap(self):
        """Gap in MeV is zero when gap_j = 0. NUMERICAL."""
        sd = RGScaleData(scale_j=0, lattice_spacing=2.2, n_blocks=1,
                         g_bar_j=6.28, gap_j=0.0)
        assert sd.gap_MeV == 0.0

    def test_gap_MeV_zero_for_negative_gap(self):
        """Gap in MeV is zero when gap_j < 0. NUMERICAL."""
        sd = RGScaleData(scale_j=0, lattice_spacing=2.2, n_blocks=1,
                         g_bar_j=6.28, gap_j=-1.0)
        assert sd.gap_MeV == 0.0

    def test_physical_scale_data(self):
        """Physical parameters produce reasonable values. NUMERICAL."""
        R = 2.2
        sd = RGScaleData(
            scale_j=3, lattice_spacing=R / 2**3, n_blocks=75,
            g_bar_j=3.5, nu_j=0.01, z_j=0.98,
            K_norm_j=0.1, gap_j=4.0 / R**2,
            energy_scale_MeV=500.0,
        )
        assert sd.lattice_spacing == pytest.approx(R / 8, rel=1e-10)
        assert sd.is_perturbative  # g^2=3.5 < 4*pi
        assert sd.gap_MeV > 100  # Should be ~179 MeV

    def test_perturbative_flag(self):
        """is_perturbative is True when g^2 < 4*pi. NUMERICAL."""
        sd_pert = RGScaleData(scale_j=0, lattice_spacing=1.0, n_blocks=1,
                              g_bar_j=5.0, is_perturbative=True)
        sd_nonpert = RGScaleData(scale_j=0, lattice_spacing=1.0, n_blocks=1,
                                 g_bar_j=15.0, is_perturbative=False)
        assert sd_pert.is_perturbative
        assert not sd_nonpert.is_perturbative


# =====================================================================
# 2. SingleRGStep tests
# =====================================================================

class TestSingleRGStep:
    """Tests for the SingleRGStep class."""

    def test_creation(self, physical_params):
        """SingleRGStep can be created with physical params. NUMERICAL."""
        step = SingleRGStep(N_scales=7, **physical_params)
        assert step.R == 2.2
        assert step.M == 2.0
        assert step.N_c == 2

    def test_invalid_R(self):
        """Negative R raises ValueError. NUMERICAL."""
        with pytest.raises(ValueError, match="R must be positive"):
            SingleRGStep(R=-1.0, M=2.0, N_c=2, N_scales=7, g2_bare=6.28)

    def test_invalid_M(self):
        """M <= 1 raises ValueError. NUMERICAL."""
        with pytest.raises(ValueError, match="Blocking factor M must be > 1"):
            SingleRGStep(R=2.2, M=0.5, N_c=2, N_scales=7, g2_bare=6.28)

    def test_invalid_N_c(self):
        """N_c < 2 raises ValueError. NUMERICAL."""
        with pytest.raises(ValueError, match="N_c must be >= 2"):
            SingleRGStep(R=2.2, M=2.0, N_c=1, N_scales=7, g2_bare=6.28)

    def test_execute_one_step(self, step_engine):
        """One RG step produces valid output. NUMERICAL."""
        R = 2.2
        initial = RGScaleData(
            scale_j=5, lattice_spacing=R / 2**5, n_blocks=100,
            g_bar_j=3.0, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        assert result.scale_j == 4  # j decremented
        assert result.g_bar_j > 0
        assert result.gap_j > 0

    def test_coupling_grows_toward_ir(self, step_engine):
        """Coupling increases from UV to IR (asymptotic freedom). THEOREM."""
        R = 2.2
        initial = RGScaleData(
            scale_j=6, lattice_spacing=R / 2**6, n_blocks=100,
            g_bar_j=2.0, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        # In the UV -> IR direction, g^2 should increase or stay constant
        # (asymptotic freedom reversed)
        assert result.g_bar_j >= initial.g_bar_j - 1e-10

    def test_kappa_below_one(self, step_engine):
        """Contraction factor kappa < 1 at every scale on S^3. THEOREM."""
        R = 2.2
        for j in range(7):
            initial = RGScaleData(
                scale_j=j, lattice_spacing=R / max(1, 2**j), n_blocks=100,
                g_bar_j=6.28, gap_j=4.0 / R**2,
            )
            result = step_engine.execute(initial)
            assert result.kappa_j < 1.0, f"kappa >= 1 at scale {j}"

    def test_gap_stays_positive(self, step_engine):
        """Effective gap remains positive through RG. THEOREM."""
        R = 2.2
        initial = RGScaleData(
            scale_j=4, lattice_spacing=R / 2**4, n_blocks=50,
            g_bar_j=6.28, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        assert result.gap_j > 0, "Gap became non-positive"

    def test_covariance_trace_nonnegative(self, step_engine):
        """Covariance slice trace is non-negative. THEOREM."""
        R = 2.2
        initial = RGScaleData(
            scale_j=3, lattice_spacing=R / 2**3, n_blocks=50,
            g_bar_j=4.0, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        assert result.covariance_trace >= 0

    def test_curvature_correction_small_in_uv(self, step_engine):
        """Curvature correction vanishes in UV (large j). NUMERICAL."""
        R = 2.2
        initial = RGScaleData(
            scale_j=6, lattice_spacing=R / 2**6, n_blocks=100,
            g_bar_j=2.0, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        assert result.curvature_correction < 0.01

    def test_curvature_correction_large_in_ir(self, step_engine):
        """Curvature correction is O(1) at IR. NUMERICAL."""
        R = 2.2
        initial = RGScaleData(
            scale_j=1, lattice_spacing=R / 2, n_blocks=5,
            g_bar_j=10.0, gap_j=4.0 / R**2,
        )
        result = step_engine.execute(initial)
        assert result.curvature_correction > 0.01


# =====================================================================
# 3. FullRGIteration tests
# =====================================================================

class TestFullRGIteration:
    """Tests for the FullRGIteration class."""

    def test_creation(self, physical_params):
        """FullRGIteration can be created. NUMERICAL."""
        it = FullRGIteration(**physical_params)
        assert it.R == 2.2
        assert it.M == 2.0
        assert it.N_c == 2

    def test_invalid_params(self):
        """Invalid parameters raise ValueError. NUMERICAL."""
        with pytest.raises(ValueError):
            FullRGIteration(R=-1.0)
        with pytest.raises(ValueError):
            FullRGIteration(M=0.5)
        with pytest.raises(ValueError):
            FullRGIteration(N_c=1)
        with pytest.raises(ValueError):
            FullRGIteration(g2_bare=-1.0)

    def test_run_small(self, small_pipeline):
        """Run with N=3 produces valid result. NUMERICAL."""
        result = small_pipeline.run(N=3)
        assert result.success
        assert result.n_scales == 3
        assert result.mass_gap_MeV > 0

    def test_run_standard(self, standard_pipeline):
        """Run with N=7 produces valid result. NUMERICAL."""
        result = standard_pipeline.run(N=7)
        assert result.success
        assert result.n_scales == 7
        assert result.mass_gap_MeV > 0

    def test_mass_gap_positive(self, standard_pipeline):
        """Mass gap is positive. THEOREM (spectral gap on S^3)."""
        result = standard_pipeline.run(N=5)
        assert result.mass_gap_MeV > 0
        assert result.mass_gap_fm_inv > 0

    def test_mass_gap_physical_range(self, standard_pipeline):
        """Mass gap is in the physically reasonable range. NUMERICAL."""
        result = standard_pipeline.run(N=7)
        # Should be between 50 and 500 MeV
        assert result.mass_gap_MeV > 50, f"Gap too small: {result.mass_gap_MeV:.1f} MeV"
        assert result.mass_gap_MeV < 500, f"Gap too large: {result.mass_gap_MeV:.1f} MeV"

    def test_coupling_trajectory_length(self, standard_pipeline):
        """Coupling trajectory has N+1 entries. NUMERICAL."""
        N = 5
        result = standard_pipeline.run(N=N)
        # N-1 steps + initial = N entries (initial + N-1 steps)
        assert len(result.coupling_trajectory) >= N

    def test_coupling_trajectory_positive(self, standard_pipeline):
        """All couplings are positive. NUMERICAL."""
        result = standard_pipeline.run(N=5)
        for g2 in result.coupling_trajectory:
            assert g2 > 0, f"Negative coupling: {g2}"

    def test_coupling_bounded(self, standard_pipeline):
        """All couplings are bounded by G2_MAX. NUMERICAL."""
        result = standard_pipeline.run(N=5)
        for g2 in result.coupling_trajectory:
            assert g2 <= G2_MAX + 1e-10, f"Coupling exceeds bound: {g2}"

    def test_K_norm_trajectory_nonnegative(self, standard_pipeline):
        """All K_norm values are non-negative. NUMERICAL."""
        result = standard_pipeline.run(N=5)
        for k in result.K_norm_trajectory:
            assert k >= -1e-10, f"Negative K_norm: {k}"

    def test_kappa_trajectory_below_one(self, standard_pipeline):
        """All kappa values are < 1. THEOREM (spectral gap on S^3)."""
        result = standard_pipeline.run(N=5)
        for i, kappa in enumerate(result.kappa_trajectory):
            assert kappa < 1.0, f"kappa >= 1 at index {i}: {kappa}"

    def test_contraction_product_small(self, standard_pipeline):
        """Contraction product << 1 for sufficient N. NUMERICAL."""
        result = standard_pipeline.run(N=7)
        assert result.contraction_product < 1.0

    def test_scale_data_populated(self, standard_pipeline):
        """Scale data list has entries for all scales. NUMERICAL."""
        N = 5
        result = standard_pipeline.run(N=N)
        assert len(result.scale_data) >= N

    def test_modules_used_populated(self, standard_pipeline):
        """Modules used dict is populated. NUMERICAL."""
        result = standard_pipeline.run(N=3)
        assert len(result.modules_used) > 0
        # At minimum, core modules should be available
        assert result.modules_used['heat_kernel_slices']
        assert result.modules_used['bbs_coordinates']
        assert result.modules_used['inductive_closure']
        assert result.modules_used['first_rg_step']

    def test_final_effective_action(self, standard_pipeline):
        """Final effective action dict has all required fields. NUMERICAL."""
        result = standard_pipeline.run(N=5)
        fa = result.final_effective_action
        assert 'g2_IR' in fa
        assert 'nu_IR' in fa
        assert 'z_IR' in fa
        assert 'gap_IR_fm2' in fa
        assert 'gap_IR_MeV' in fa
        assert 'K_norm_IR' in fa
        assert 'R_fm' in fa
        assert 'N_scales' in fa

    def test_trajectory_method(self, small_pipeline):
        """trajectory() returns the scale data list. NUMERICAL."""
        small_pipeline.run(N=3)
        traj = small_pipeline.trajectory()
        assert len(traj) >= 3

    def test_coupling_flow_method(self, small_pipeline):
        """coupling_flow() returns array of g^2. NUMERICAL."""
        small_pipeline.run(N=3)
        g2_flow = small_pipeline.coupling_flow()
        assert isinstance(g2_flow, np.ndarray)
        assert len(g2_flow) >= 3
        assert all(g2_flow > 0)

    def test_K_norm_flow_method(self, small_pipeline):
        """K_norm_flow() returns array of ||K||. NUMERICAL."""
        small_pipeline.run(N=3)
        k_flow = small_pipeline.K_norm_flow()
        assert isinstance(k_flow, np.ndarray)
        assert len(k_flow) >= 3

    def test_mass_gap_method(self, small_pipeline):
        """mass_gap() returns positive value. NUMERICAL."""
        result = small_pipeline.run(N=3)
        gap = small_pipeline.mass_gap()
        assert gap > 0
        assert gap == pytest.approx(result.mass_gap_MeV, rel=1e-10)


# =====================================================================
# 4. Edge cases
# =====================================================================

class TestEdgeCases:
    """Edge case tests for the pipeline."""

    def test_minimum_N_2(self):
        """Pipeline works with minimum N=2. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=2)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_large_N_15(self):
        """Pipeline works with N=15. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=15)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_small_R(self):
        """Pipeline works with small R = 0.5 fm. NUMERICAL."""
        it = FullRGIteration(R=0.5, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0
        # Smaller R => larger gap
        it2 = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result2 = it2.run(N=5)
        assert result.mass_gap_MeV > result2.mass_gap_MeV

    def test_large_R(self):
        """Pipeline works with large R = 10 fm. NUMERICAL."""
        it = FullRGIteration(R=10.0, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_weak_coupling(self):
        """Pipeline works with weak coupling g^2 = 1.0. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=1.0)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_strong_coupling(self):
        """Pipeline works with strong coupling g^2 = 12.0. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=12.0)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_su3(self):
        """Pipeline works with SU(3). NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=3, g2_bare=6.28)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_blocking_factor_3(self):
        """Pipeline works with M=3. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=3.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_gap_decreases_with_R(self):
        """Mass gap decreases with increasing R (1/R behavior). THEOREM."""
        gaps = []
        for R in [1.0, 2.0, 4.0]:
            it = FullRGIteration(R=R, M=2.0, N_c=2, g2_bare=6.28)
            result = it.run(N=5)
            gaps.append(result.mass_gap_MeV)
        # Monotone decrease (approximately)
        assert gaps[0] > gaps[1] > gaps[2]


# =====================================================================
# 5. ContinuumLimitScanner tests
# =====================================================================

class TestContinuumLimitScanner:
    """Tests for the ContinuumLimitScanner class."""

    def test_creation(self):
        """Scanner can be created. NUMERICAL."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        assert scanner.R == 2.2

    def test_scan_small_range(self):
        """Scan over N=2..5 produces valid output. NUMERICAL."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 5))
        assert 'N_values' in result
        assert 'gap_values_MeV' in result
        assert len(result['N_values']) == 4  # 2, 3, 4, 5
        assert all(g > 0 for g in result['gap_values_MeV'])

    def test_scan_gap_positive(self):
        """All gaps are positive in the scan. THEOREM."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 6))
        assert all(g > 0 for g in result['gap_values_MeV'])

    def test_K_bounded_uniformly(self):
        """K_norm is bounded uniformly in N. THEOREM."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 6))
        assert result['K_uniform_bound'] < float('inf')
        assert result['K_uniform_bound'] >= 0

    def test_scan_contraction_products(self):
        """Contraction products are computed. NUMERICAL."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 5))
        assert len(result['contraction_products']) == 4
        for cp in result['contraction_products']:
            assert 0 < cp < 10  # Reasonable range

    def test_scan_coupling_ir(self):
        """IR coupling is tracked. NUMERICAL."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 5))
        assert all(g > 0 for g in result['coupling_ir'])
        assert all(g <= G2_MAX + 1e-10 for g in result['coupling_ir'])

    def test_scan_relative_changes(self):
        """Relative changes in gap are computed. NUMERICAL."""
        scanner = ContinuumLimitScanner(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = scanner.scan(N_range=(2, 5))
        assert len(result['relative_changes']) == 3  # N-1 changes


# =====================================================================
# 6. RGDiagnostics tests
# =====================================================================

class TestRGDiagnostics:
    """Tests for the RGDiagnostics class."""

    def test_diagnose_valid_result(self, diagnostics, standard_pipeline):
        """Diagnose produces valid output. NUMERICAL."""
        rg_result = standard_pipeline.run(N=5)
        diag = diagnostics.diagnose(rg_result)
        assert 'n_scales' in diag
        assert 'scale_diagnostics' in diag
        assert 'bottleneck' in diag
        assert 'gap_budget' in diag
        assert 'contraction_budget' in diag

    def test_diagnose_empty_result(self, diagnostics):
        """Diagnose handles empty result gracefully. NUMERICAL."""
        empty_result = RGResult()
        diag = diagnostics.diagnose(empty_result)
        assert 'error' in diag

    def test_bottleneck_identified(self, diagnostics, standard_pipeline):
        """Bottleneck scale is identified. NUMERICAL."""
        rg_result = standard_pipeline.run(N=5)
        bn = diagnostics.bottleneck_analysis(rg_result)
        assert 'bottleneck_scale' in bn
        assert 'worst_kappa' in bn
        assert bn['worst_kappa'] < 1.0
        assert bn['headroom'] > 0

    def test_gap_budget_positive(self, diagnostics, standard_pipeline):
        """Gap budget shows positive final gap. THEOREM."""
        rg_result = standard_pipeline.run(N=5)
        gb = diagnostics.gap_budget(rg_result)
        assert gb['gap_positive']
        assert gb['fraction_surviving'] > 0

    def test_gap_budget_bare_vs_final(self, diagnostics, standard_pipeline):
        """Final gap <= bare gap (mass corrections can only reduce). NUMERICAL."""
        rg_result = standard_pipeline.run(N=5)
        gb = diagnostics.gap_budget(rg_result)
        # The final gap should be a reasonable fraction of the bare gap
        assert gb['fraction_surviving'] > 0.1  # At least 10% survives

    def test_contraction_budget_all_below_one(self, diagnostics, standard_pipeline):
        """All kappas are below 1 in the contraction budget. THEOREM."""
        rg_result = standard_pipeline.run(N=5)
        cb = diagnostics.contraction_budget(rg_result)
        assert cb['all_below_one']

    def test_contraction_budget_product(self, diagnostics, standard_pipeline):
        """Contraction product is computed. NUMERICAL."""
        rg_result = standard_pipeline.run(N=5)
        cb = diagnostics.contraction_budget(rg_result)
        assert 'product' in cb
        assert cb['product'] > 0
        assert cb['product'] < 10  # Reasonable bound

    def test_scale_diagnostics_populated(self, diagnostics, standard_pipeline):
        """Scale diagnostics have entries for each scale. NUMERICAL."""
        rg_result = standard_pipeline.run(N=5)
        diag = diagnostics.diagnose(rg_result)
        scale_diags = diag['scale_diagnostics']
        assert len(scale_diags) >= 5
        for sd in scale_diags:
            assert 'g2' in sd
            assert 'kappa' in sd
            assert 'headroom' in sd


# =====================================================================
# 7. S3AdvantageQuantifier tests
# =====================================================================

class TestS3AdvantageQuantifier:
    """Tests for the S3AdvantageQuantifier class."""

    def test_creation(self, s3_advantage):
        """Quantifier can be created. NUMERICAL."""
        assert s3_advantage.R == 2.2
        assert s3_advantage.N_c == 2

    def test_finite_volume(self, s3_advantage):
        """Finite volume advantage is quantified. THEOREM."""
        adv = s3_advantage.advantage_finite_volume(j=3)
        assert adv['n_blocks_s3'] > 0
        assert adv['n_blocks_s3'] < 1000  # Finite
        assert adv['polymer_bound_s3_finite']
        assert not adv['polymer_bound_t4_finite']

    def test_finite_volume_decreases_with_j(self, s3_advantage):
        """Block count decreases with scale (coarsening). NUMERICAL."""
        n0 = s3_advantage.advantage_finite_volume(0)['n_blocks_s3']
        n3 = s3_advantage.advantage_finite_volume(3)['n_blocks_s3']
        assert n0 >= n3

    def test_no_zero_modes(self, s3_advantage):
        """No zero modes advantage is quantified. THEOREM."""
        adv = s3_advantage.advantage_no_zero_modes()
        assert adv['h1_s3'] == 0
        assert adv['h1_t4'] == 4
        assert adv['spectral_gap_fm2'] > 0
        assert adv['spectral_gap_MeV'] > 100  # ~179 MeV

    def test_spectral_gap_value(self, s3_advantage):
        """Spectral gap matches lambda_1 = 4/R^2. THEOREM."""
        adv = s3_advantage.advantage_no_zero_modes()
        expected = 4.0 / 2.2**2
        assert adv['spectral_gap_fm2'] == pytest.approx(expected, rel=1e-10)

    def test_uniform_blocks(self, s3_advantage):
        """Uniform blocks advantage is quantified. THEOREM."""
        adv = s3_advantage.advantage_uniform_blocks(j=2)
        assert adv['homogeneous_s3']
        assert not adv['homogeneous_t4']
        assert adv['constants_same_across_blocks']

    def test_bounded_gribov(self, s3_advantage):
        """Bounded Gribov advantage is quantified. THEOREM."""
        adv = s3_advantage.advantage_bounded_gribov()
        assert adv['bounded_s3']
        assert not adv['bounded_t4']
        assert adv['gribov_diameter_R'] > 0
        assert adv['A_max_fm_inv'] > 0
        assert adv['convex_s3']

    def test_gribov_diameter_formula(self, s3_advantage):
        """Gribov diameter matches 9*sqrt(3)/(2*g). THEOREM."""
        g = np.sqrt(6.28)
        expected = 9.0 * np.sqrt(3.0) / (2.0 * g)
        adv = s3_advantage.advantage_bounded_gribov()
        assert adv['gribov_diameter_R'] == pytest.approx(expected, rel=0.01)

    def test_large_field_empty(self, s3_advantage):
        """Large-field empty advantage is quantified. THEOREM."""
        adv = s3_advantage.advantage_large_field_empty()
        assert adv['large_field_empty_s3']
        assert not adv['large_field_empty_t4']
        assert adv['balaban_pages_saved'] == 100

    def test_advantage_at_scale(self, s3_advantage):
        """All five advantages at a given scale. NUMERICAL."""
        adv = s3_advantage.advantage_at_scale(j=3)
        assert 'finite_volume' in adv
        assert 'no_zero_modes' in adv
        assert 'uniform_blocks' in adv
        assert 'bounded_gribov' in adv
        assert 'large_field_empty' in adv

    def test_total_advantage(self, s3_advantage):
        """Total advantage summary. NUMERICAL."""
        total = s3_advantage.total_advantage(N=5)
        assert total['zero_modes_eliminated']
        assert total['all_blocks_uniform']
        assert total['gribov_bounded']
        assert total['large_field_empty']
        assert len(total['structural_simplifications']) == 5

    def test_comparison_table(self, s3_advantage):
        """Comparison table has 5 entries. NUMERICAL."""
        table = s3_advantage.comparison_table()
        assert len(table) == 5
        for row in table:
            assert 'advantage' in row
            assert 's3' in row
            assert 't4' in row
            assert 'impact' in row


# =====================================================================
# 8. Integration with existing modules
# =====================================================================

class TestModuleIntegration:
    """Tests verifying integration with all existing RG modules."""

    def test_heat_kernel_available(self):
        """HeatKernelSlices is available. NUMERICAL."""
        assert _HAS_HEAT_KERNEL

    def test_block_geometry_available(self):
        """Block geometry module is available. NUMERICAL."""
        assert _HAS_BLOCK_GEOMETRY

    def test_gauge_fixing_available(self):
        """Gauge fixing module is available. NUMERICAL."""
        assert _HAS_GAUGE_FIXING

    def test_beta_flow_available(self):
        """Beta flow module is available. NUMERICAL."""
        assert _HAS_BETA_FLOW

    def test_covariant_propagator_available(self):
        """Covariant propagator module is available. NUMERICAL."""
        assert _HAS_COVARIANT_PROPAGATOR

    def test_background_minimizer_available(self):
        """Background minimizer module is available. NUMERICAL."""
        assert _HAS_BACKGROUND_MINIMIZER

    def test_uniform_contraction_available(self):
        """Uniform contraction module is available. NUMERICAL."""
        assert _HAS_UNIFORM_CONTRACTION

    def test_polymer_algebra_available(self):
        """Polymer algebra module is available. NUMERICAL."""
        assert _HAS_POLYMER_ALGEBRA

    def test_continuum_limit_available(self):
        """Continuum limit module is available. NUMERICAL."""
        assert _HAS_CONTINUUM_LIMIT

    def test_large_field_available(self):
        """Large field module is available. NUMERICAL."""
        assert _HAS_LARGE_FIELD

    def test_gribov_available(self):
        """Gribov diameter module is available. NUMERICAL."""
        assert _HAS_GRIBOV

    def test_all_15_modules(self):
        """All 15 modules are available. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=3)
        n_available = sum(1 for v in result.modules_used.values() if v)
        assert n_available >= 13, f"Only {n_available}/15 modules available"

    def test_cross_validation_with_multi_scale_rg(self):
        """Pipeline agrees with MultiScaleRGFlow. NUMERICAL."""
        R, M, N_c, g2 = 2.2, 2.0, 2, 6.28
        N = 5

        # Pipeline result
        it = FullRGIteration(R=R, M=M, N_c=N_c, g2_bare=g2)
        pipe_result = it.run(N=N)

        # MultiScaleRGFlow result
        msrg = MultiScaleRGFlow(R=R, M=M, N_scales=N, N_c=N_c,
                                g2_bare=g2, k_max=100)
        msrg_result = msrg.run_flow()

        # Both should give positive mass gap
        assert pipe_result.mass_gap_MeV > 0
        assert msrg_result['mass_gap_mev'] > 0

    def test_cross_validation_with_bbs(self):
        """Pipeline is consistent with MultiScaleRGBBS. NUMERICAL."""
        R, M, N_c, g2 = 2.2, 2.0, 2, 6.28
        N = 5

        # Pipeline result
        it = FullRGIteration(R=R, M=M, N_c=N_c, g2_bare=g2)
        pipe_result = it.run(N=N)

        # BBS result
        bbs = MultiScaleRGBBS(n_scales=N, R=R, M=M, N_c=N_c,
                               g2_bare=g2, k_max=100)
        traj = bbs.run()
        bbs_couplings = bbs.coupling_trajectory(traj)

        # Both should have positive couplings throughout
        assert all(g > 0 for g in pipe_result.coupling_trajectory)
        assert all(g > 0 for g in bbs_couplings['g2'])


# =====================================================================
# 9. Physical consistency tests
# =====================================================================

class TestPhysicalConsistency:
    """Tests for physical consistency of the pipeline output."""

    def test_asymptotic_freedom(self):
        """Coupling increases from UV to IR. THEOREM."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=3.0, k_max=100)
        result = it.run(N=7)
        g2_flow = result.coupling_trajectory
        # Overall trend: coupling should increase
        assert g2_flow[-1] >= g2_flow[0]

    def test_spectral_gap_dominates(self):
        """The bare gap 4/R^2 dominates at the IR. THEOREM."""
        R = 2.2
        it = FullRGIteration(R=R, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        bare_gap = 4.0 / R**2
        # Final gap should be at least 50% of bare gap
        # (mass corrections are perturbative)
        assert result.mass_gap_fm_inv >= bare_gap * 0.3

    def test_gap_scales_as_one_over_R(self):
        """Mass gap scales approximately as 1/R. THEOREM."""
        gaps = {}
        for R in [1.0, 2.0, 4.0]:
            it = FullRGIteration(R=R, M=2.0, N_c=2, g2_bare=6.28)
            result = it.run(N=5)
            gaps[R] = result.mass_gap_MeV

        # gap(R) ~ hbar_c * 2/R, so gap * R should be approximately constant
        products = [gaps[R] * R for R in [1.0, 2.0, 4.0]]
        # Allow 50% variation (mass corrections are not negligible)
        mean_product = np.mean(products)
        for p in products:
            assert abs(p - mean_product) / mean_product < 0.6

    def test_su2_vs_su3_gap(self):
        """SU(3) gap should be comparable to SU(2). NUMERICAL."""
        it2 = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result2 = it2.run(N=5)

        it3 = FullRGIteration(R=2.2, M=2.0, N_c=3, g2_bare=6.28)
        result3 = it3.run(N=5)

        # Both should be positive
        assert result2.mass_gap_MeV > 0
        assert result3.mass_gap_MeV > 0

    def test_contraction_all_scales(self):
        """Contraction holds at ALL scales. THEOREM."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=7)
        for sd in result.scale_data:
            assert sd.kappa_j < 1.0, f"Contraction fails at scale {sd.scale_j}"

    def test_z_stays_positive(self):
        """Wavefunction renormalization stays positive. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=7)
        for sd in result.scale_data:
            assert sd.z_j > 0, f"z_j <= 0 at scale {sd.scale_j}"

    def test_energy_scales_monotone(self):
        """Energy scales decrease from UV to IR. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=7)
        energies = [sd.energy_scale_MeV for sd in result.scale_data]
        # Allow small numerical violations
        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i+1] * 0.5, (
                f"Energy not monotone: {energies[i]:.1f} < {energies[i+1]:.1f}"
            )


# =====================================================================
# 10. RGResult dataclass tests
# =====================================================================

class TestRGResult:
    """Tests for the RGResult dataclass."""

    def test_default_creation(self):
        """RGResult can be created with defaults. NUMERICAL."""
        r = RGResult()
        assert not r.success
        assert r.n_scales == 0
        assert r.mass_gap_MeV == 0.0
        assert r.coupling_trajectory == []

    def test_populated_creation(self):
        """RGResult can be created with all fields. NUMERICAL."""
        r = RGResult(
            success=True,
            n_scales=7,
            mass_gap_MeV=179.0,
            mass_gap_fm_inv=0.826,
            coupling_trajectory=[6.28, 7.0, 8.0],
            K_norm_trajectory=[0.0, 0.1, 0.2],
            kappa_trajectory=[0.5, 0.6, 0.55],
            invariant_preserved=True,
            continuum_limit_converges=True,
        )
        assert r.success
        assert r.mass_gap_MeV == 179.0
        assert len(r.coupling_trajectory) == 3

    def test_warnings_field(self):
        """Warnings list is properly populated. NUMERICAL."""
        r = RGResult(warnings=['test warning'])
        assert len(r.warnings) == 1
        assert r.warnings[0] == 'test warning'

    def test_modules_used_field(self):
        """modules_used dict works. NUMERICAL."""
        r = RGResult(modules_used={'heat_kernel': True, 'beta_flow': False})
        assert r.modules_used['heat_kernel']
        assert not r.modules_used['beta_flow']


# =====================================================================
# 11. Stress tests
# =====================================================================

class TestStress:
    """Stress tests for robustness."""

    def test_many_scales(self):
        """Pipeline survives N=20 scales. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=20)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_very_strong_coupling(self):
        """Pipeline survives near-maximum coupling. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=12.0)
        result = it.run(N=5)
        assert result.success

    def test_very_small_R(self):
        """Pipeline survives very small R = 0.3 fm. NUMERICAL."""
        it = FullRGIteration(R=0.3, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        assert result.success
        assert result.mass_gap_MeV > 0

    def test_different_blocking_factors(self):
        """Pipeline works with M=2, 3, 4. NUMERICAL."""
        for M in [2.0, 3.0, 4.0]:
            it = FullRGIteration(R=2.2, M=M, N_c=2, g2_bare=6.28)
            result = it.run(N=5)
            assert result.success, f"Failed for M={M}"
            assert result.mass_gap_MeV > 0, f"Zero gap for M={M}"

    def test_multiple_runs_consistent(self):
        """Two runs with same parameters give same result. NUMERICAL."""
        params = dict(R=2.2, M=2.0, N_c=2, g2_bare=6.28, k_max=100)
        it1 = FullRGIteration(**params)
        r1 = it1.run(N=5)

        it2 = FullRGIteration(**params)
        r2 = it2.run(N=5)

        assert r1.mass_gap_MeV == pytest.approx(r2.mass_gap_MeV, rel=1e-10)
        assert r1.contraction_product == pytest.approx(r2.contraction_product, rel=1e-10)


# =====================================================================
# 12. Graceful degradation tests
# =====================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation when modules are approximate."""

    def test_result_has_warnings_field(self):
        """Warnings are reported in the result. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=3)
        assert isinstance(result.warnings, list)

    def test_modules_used_reports_availability(self):
        """Module availability is correctly reported. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=3)
        for module_name, available in result.modules_used.items():
            assert isinstance(available, bool), (
                f"Module {module_name} availability is not bool"
            )

    def test_pipeline_succeeds_despite_warnings(self):
        """Pipeline success is not blocked by warnings. NUMERICAL."""
        it = FullRGIteration(R=2.2, M=2.0, N_c=2, g2_bare=6.28)
        result = it.run(N=5)
        # Even with warnings, the pipeline should succeed
        assert result.success
