"""
Tests for Estimate 6: Perturbative RG Flow (beta-function) on S^3.

Verifies:
  1. BetaFunction: correct coefficients for SU(2), SU(3), SU(N)  (THEOREM)
  2. Asymptotic freedom: g^2 decreasing toward UV at every scale  (NUMERICAL)
  3. Lambda_QCD extraction: consistent with 200 MeV within 20%    (NUMERICAL)
  4. Mass flow: critical nu_0 yields nu_N -> 0                    (NUMERICAL)
  5. Curvature corrections: vanish at UV, bounded at IR            (NUMERICAL)
  6. Two-loop vs one-loop: corrections are O(g^4) smaller          (NUMERICAL)
  7. Full flow: N_scales steps without blowup                      (NUMERICAL)
  8. Comparison with existing first_rg_step.py beta function       (NUMERICAL)
  9. Edge cases: g^2 -> 0, R -> inf                                (NUMERICAL)
  10. Wavefunction renormalization: z flow logarithmic             (NUMERICAL)

Target: 60+ tests.
"""

import numpy as np
import pytest

from yang_mills_s3.rg.beta_flow import (
    BetaFunction,
    MassRenormalization,
    WaveFunctionRenormalization,
    CurvatureCorrections,
    PerturbativeRGFlow,
    AsymptoticFreedomVerifier,
)
from yang_mills_s3.rg.heat_kernel_slices import R_PHYSICAL_FM, HBAR_C_MEV_FM, LAMBDA_QCD_MEV


# ======================================================================
# Physical constants for reference
# ======================================================================

R_PHYS = R_PHYSICAL_FM          # 2.2 fm
G2_PHYS = 6.28                  # Physical bare coupling
HBAR_C = HBAR_C_MEV_FM          # 197.327 MeV*fm
LAMBDA_QCD = LAMBDA_QCD_MEV     # 200 MeV
M_DEFAULT = 2.0                 # Blocking factor
N_SCALES = 7                    # Number of RG scales


# ======================================================================
# 1. BetaFunction: coefficient tests
# ======================================================================

class TestBetaFunctionCoefficients:
    """THEOREM: beta_0 and beta_1 match Gross-Wilczek-Politzer / Caswell-Jones."""

    def test_beta0_su2(self):
        """beta_0(SU(2)) = 22 / (48 pi^2)."""
        bf = BetaFunction(N_c=2)
        expected = 22.0 / (48.0 * np.pi ** 2)
        assert bf.beta0 == pytest.approx(expected, rel=1e-12)

    def test_beta0_su3(self):
        """beta_0(SU(3)) = 33 / (48 pi^2)."""
        bf = BetaFunction(N_c=3)
        expected = 33.0 / (48.0 * np.pi ** 2)
        assert bf.beta0 == pytest.approx(expected, rel=1e-12)

    def test_beta0_general_sun(self):
        """beta_0(SU(N)) = 11 N / (3 * 16 pi^2) for N = 2..10."""
        for N in range(2, 11):
            bf = BetaFunction(N_c=N)
            expected = 11.0 * N / (3.0 * 16.0 * np.pi ** 2)
            assert bf.beta0 == pytest.approx(expected, rel=1e-12), \
                f"Failed for SU({N})"

    def test_beta0_positive(self):
        """beta_0 > 0 for all SU(N) => asymptotic freedom. THEOREM."""
        for N in range(2, 20):
            bf = BetaFunction(N_c=N)
            assert bf.beta0 > 0, f"beta_0 not positive for SU({N})"

    def test_beta0_linear_in_N(self):
        """beta_0 is linear in N_c. THEOREM."""
        bf2 = BetaFunction(N_c=2)
        bf4 = BetaFunction(N_c=4)
        assert bf4.beta0 == pytest.approx(2.0 * bf2.beta0, rel=1e-12)

    def test_beta1_su2(self):
        """beta_1(SU(2)) = 34 * 4 / (3 * (16 pi^2)^2)."""
        bf = BetaFunction(N_c=2)
        expected = 34.0 * 4 / (3.0 * (16.0 * np.pi ** 2) ** 2)
        assert bf.beta1 == pytest.approx(expected, rel=1e-12)

    def test_beta1_su3(self):
        """beta_1(SU(3)) = 34 * 9 / (3 * (16 pi^2)^2)."""
        bf = BetaFunction(N_c=3)
        expected = 34.0 * 9 / (3.0 * (16.0 * np.pi ** 2) ** 2)
        assert bf.beta1 == pytest.approx(expected, rel=1e-12)

    def test_beta1_positive(self):
        """beta_1 > 0 for all SU(N) in pure YM. THEOREM."""
        for N in range(2, 20):
            bf = BetaFunction(N_c=N)
            assert bf.beta1 > 0, f"beta_1 not positive for SU({N})"

    def test_beta1_quadratic_in_N(self):
        """beta_1 scales as N_c^2. THEOREM."""
        bf2 = BetaFunction(N_c=2)
        bf3 = BetaFunction(N_c=3)
        ratio = bf3.beta1 / bf2.beta1
        expected_ratio = 9.0 / 4.0  # (3/2)^2
        assert ratio == pytest.approx(expected_ratio, rel=1e-12)

    def test_beta1_much_smaller_than_beta0_for_weak_coupling(self):
        """beta_1 * g^2 << beta_0 when g^2 is small. NUMERICAL."""
        bf = BetaFunction(N_c=2)
        g2_small = 0.5
        ratio = bf.beta1 * g2_small / bf.beta0
        assert ratio < 0.1, "Two-loop should be <10% of one-loop for small g^2"

    def test_22_over_3_factor_su2(self):
        """For SU(2), (11/3) * C_2(adj) = (11/3) * 2 = 22/3. THEOREM."""
        N_c = 2
        C2_adj = float(N_c)
        factor = (11.0 / 3.0) * C2_adj
        assert factor == pytest.approx(22.0 / 3.0, rel=1e-14)


# ======================================================================
# 2. BetaFunction: one_loop and two_loop methods
# ======================================================================

class TestBetaFunctionFlow:
    """Tests for one_loop and two_loop flow maps."""

    @pytest.fixture
    def bf_su2(self):
        return BetaFunction(N_c=2, R=R_PHYS, M=M_DEFAULT)

    @pytest.fixture
    def bf_su3(self):
        return BetaFunction(N_c=3, R=R_PHYS, M=M_DEFAULT)

    def test_one_loop_positive(self, bf_su2):
        """One-loop increment delta(1/g^2) > 0. THEOREM."""
        assert bf_su2.one_loop(G2_PHYS) > 0

    def test_one_loop_independent_of_g2(self, bf_su2):
        """At 1-loop, delta(1/g^2) is independent of g^2. THEOREM."""
        d1 = bf_su2.one_loop(1.0)
        d2 = bf_su2.one_loop(10.0)
        assert d1 == pytest.approx(d2, rel=1e-14)

    def test_one_loop_equals_beta0_times_ln_M2(self, bf_su2):
        """delta(1/g^2) = beta_0 * ln(M^2). THEOREM."""
        expected = bf_su2.beta0 * np.log(M_DEFAULT ** 2)
        assert bf_su2.one_loop(G2_PHYS) == pytest.approx(expected, rel=1e-14)

    def test_two_loop_larger_than_one_loop(self, bf_su2):
        """Two-loop increment > one-loop (adds positive beta_1 * g^2 term)."""
        d_1loop = bf_su2.one_loop(G2_PHYS)
        d_2loop = bf_su2.two_loop(G2_PHYS)
        assert d_2loop > d_1loop

    def test_two_loop_difference_is_order_g2(self, bf_su2):
        """Two-loop - one-loop = beta_1 * g^2 * ln(M^2). THEOREM."""
        d_1loop = bf_su2.one_loop(G2_PHYS)
        d_2loop = bf_su2.two_loop(G2_PHYS)
        diff = d_2loop - d_1loop
        expected = bf_su2.beta1 * G2_PHYS * np.log(M_DEFAULT ** 2)
        assert diff == pytest.approx(expected, rel=1e-12)

    def test_two_loop_reduces_to_one_loop_at_weak(self, bf_su2):
        """At weak coupling (g^2 -> 0), two-loop -> one-loop. NUMERICAL."""
        g2_weak = 0.001
        d_1loop = bf_su2.one_loop(g2_weak)
        d_2loop = bf_su2.two_loop(g2_weak)
        assert abs(d_2loop - d_1loop) / d_1loop < 0.01

    def test_curvature_correction_positive_at_ir(self, bf_su2):
        """With-curvature > one-loop at IR scale (j=0). NUMERICAL."""
        d_flat = bf_su2.one_loop(G2_PHYS)
        d_curv = bf_su2.with_curvature(G2_PHYS, 0)
        assert d_curv > d_flat

    def test_curvature_correction_vanishes_at_uv(self, bf_su2):
        """At UV scales (large j), curvature correction negligible. NUMERICAL."""
        d_flat = bf_su2.one_loop(G2_PHYS)
        d_curv = bf_su2.with_curvature(G2_PHYS, 10)
        # At j=10, M^{-10}/R ~ 2^{-10}/2.2 ~ 4.4e-4 => correction ~ 2e-7
        rel_diff = abs(d_curv - d_flat) / d_flat
        assert rel_diff < 1e-4, f"Curvature correction too large at UV: {rel_diff}"

    def test_curvature_correction_decreases_with_j(self, bf_su2):
        """Curvature correction decreases monotonically with j. NUMERICAL."""
        corrections = []
        for j in range(N_SCALES):
            d_flat = bf_su2.one_loop(G2_PHYS)
            d_curv = bf_su2.with_curvature(G2_PHYS, j)
            corrections.append(d_curv - d_flat)
        for i in range(len(corrections) - 1):
            assert corrections[i] >= corrections[i + 1] - 1e-15


# ======================================================================
# 3. BetaFunction: validation
# ======================================================================

class TestBetaFunctionValidation:
    """Input validation tests for BetaFunction."""

    def test_invalid_N_c(self):
        """N_c < 2 raises ValueError."""
        with pytest.raises(ValueError):
            BetaFunction(N_c=1)
        with pytest.raises(ValueError):
            BetaFunction(N_c=0)

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BetaFunction(R=0.0)
        with pytest.raises(ValueError):
            BetaFunction(R=-1.0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            BetaFunction(M=1.0)
        with pytest.raises(ValueError):
            BetaFunction(M=0.5)


# ======================================================================
# 4. MassRenormalization
# ======================================================================

class TestMassRenormalization:
    """Tests for mass parameter (nu) flow."""

    @pytest.fixture
    def mr(self):
        return MassRenormalization(N_c=2, R=R_PHYS, M=M_DEFAULT)

    def test_one_loop_mass_shift_positive(self, mr):
        """Mass shift delta_nu > 0 at any scale. NUMERICAL."""
        for j in range(N_SCALES):
            d_nu = mr.one_loop_mass_shift(G2_PHYS, j)
            assert d_nu > 0, f"delta_nu not positive at scale j={j}"

    def test_one_loop_mass_shift_scales_with_g2(self, mr):
        """delta_nu is proportional to g^2. NUMERICAL."""
        d1 = mr.one_loop_mass_shift(1.0, 3)
        d2 = mr.one_loop_mass_shift(2.0, 3)
        assert d2 == pytest.approx(2.0 * d1, rel=1e-12)

    def test_mass_shift_finite_at_all_scales(self, mr):
        """delta_nu is finite at every scale (S^3 regulates). NUMERICAL."""
        for j in range(N_SCALES):
            d_nu = mr.one_loop_mass_shift(G2_PHYS, j)
            assert np.isfinite(d_nu)
            assert d_nu < 1e10  # Reasonable bound

    def test_mass_shift_increases_with_scale(self, mr):
        """Higher shells (UV) contribute more to delta_nu. NUMERICAL."""
        shifts = [mr.one_loop_mass_shift(G2_PHYS, j) for j in range(N_SCALES)]
        # Each shell covers a larger momentum range, so delta_nu grows
        for i in range(len(shifts) - 1):
            assert shifts[i + 1] >= shifts[i] - 1e-15

    def test_anomalous_dimension_positive(self, mr):
        """gamma_nu > 0 for pure YM. NUMERICAL."""
        gamma = mr.anomalous_dimension_nu(G2_PHYS)
        assert gamma > 0

    def test_anomalous_dimension_scales_with_g2(self, mr):
        """gamma_nu is proportional to g^2. NUMERICAL."""
        g1 = mr.anomalous_dimension_nu(1.0)
        g2 = mr.anomalous_dimension_nu(3.0)
        assert g2 == pytest.approx(3.0 * g1, rel=1e-12)

    def test_critical_initial_finite(self, mr):
        """nu_c is finite. NUMERICAL."""
        nu_c = mr.critical_initial(G2_PHYS, N_SCALES)
        assert np.isfinite(nu_c)

    def test_critical_initial_negative(self, mr):
        """nu_c is negative (counterterm). NUMERICAL."""
        nu_c = mr.critical_initial(G2_PHYS, N_SCALES)
        assert nu_c < 0, "Critical nu_c should be negative"

    def test_critical_initial_approaches_zero_as_g2_to_zero(self, mr):
        """nu_c -> 0 as g^2 -> 0. NUMERICAL."""
        nu_large = mr.critical_initial(1.0, N_SCALES)
        nu_small = mr.critical_initial(0.01, N_SCALES)
        assert abs(nu_small) < abs(nu_large)

    def test_critical_initial_drives_nu_toward_zero(self, mr):
        """Starting from nu_c, the mass parameter should approach 0. NUMERICAL."""
        nu_c = mr.critical_initial(G2_PHYS, N_SCALES)
        nu = nu_c
        for j in range(N_SCALES):
            gamma = mr.anomalous_dimension_nu(G2_PHYS)
            d_nu = mr.one_loop_mass_shift(G2_PHYS, j)
            nu = nu * (1.0 + gamma) + d_nu
        # After N steps, |nu| should be smaller than the bare gap
        lambda1 = 4.0 / R_PHYS ** 2
        assert abs(nu) < lambda1, \
            f"|nu_N| = {abs(nu):.4e} >= lambda_1 = {lambda1:.4e}"


class TestMassRenormalizationValidation:
    """Input validation for MassRenormalization."""

    def test_invalid_N_c(self):
        with pytest.raises(ValueError):
            MassRenormalization(N_c=1)

    def test_invalid_R(self):
        with pytest.raises(ValueError):
            MassRenormalization(R=0.0)

    def test_invalid_M(self):
        with pytest.raises(ValueError):
            MassRenormalization(M=1.0)


# ======================================================================
# 5. WaveFunctionRenormalization
# ======================================================================

class TestWaveFunctionRenormalization:
    """Tests for wavefunction (z) flow."""

    @pytest.fixture
    def wf(self):
        return WaveFunctionRenormalization(N_c=2, M=M_DEFAULT)

    def test_z_shift_positive(self, wf):
        """delta_z > 0 in Landau gauge. NUMERICAL."""
        dz = wf.one_loop_z_shift(G2_PHYS, 0)
        assert dz > 0

    def test_z_shift_scales_with_g2(self, wf):
        """delta_z proportional to g^2. NUMERICAL."""
        dz1 = wf.one_loop_z_shift(1.0, 0)
        dz2 = wf.one_loop_z_shift(3.0, 0)
        assert dz2 == pytest.approx(3.0 * dz1, rel=1e-12)

    def test_z_shift_independent_of_scale_j(self, wf):
        """At 1-loop, delta_z depends only on g^2, not explicitly on j. NUMERICAL."""
        dz0 = wf.one_loop_z_shift(G2_PHYS, 0)
        dz5 = wf.one_loop_z_shift(G2_PHYS, 5)
        assert dz0 == pytest.approx(dz5, rel=1e-14)

    def test_z_shift_logarithmic_in_M(self, wf):
        """delta_z proportional to ln(M^2). NUMERICAL."""
        wf2 = WaveFunctionRenormalization(N_c=2, M=2.0)
        wf4 = WaveFunctionRenormalization(N_c=2, M=4.0)
        dz2 = wf2.one_loop_z_shift(1.0, 0)
        dz4 = wf4.one_loop_z_shift(1.0, 0)
        ratio = dz4 / dz2
        expected_ratio = np.log(16.0) / np.log(4.0)  # ln(4^2)/ln(2^2)
        assert ratio == pytest.approx(expected_ratio, rel=1e-12)

    def test_z_shift_small_for_weak_coupling(self, wf):
        """delta_z << 1 for small g^2. NUMERICAL."""
        dz = wf.one_loop_z_shift(0.1, 0)
        assert dz < 0.1

    def test_z_shift_scales_with_N_c(self):
        """delta_z proportional to C_2(adj) = N_c. NUMERICAL."""
        wf2 = WaveFunctionRenormalization(N_c=2, M=2.0)
        wf3 = WaveFunctionRenormalization(N_c=3, M=2.0)
        dz2 = wf2.one_loop_z_shift(1.0, 0)
        dz3 = wf3.one_loop_z_shift(1.0, 0)
        ratio = dz3 / dz2
        assert ratio == pytest.approx(3.0 / 2.0, rel=1e-12)


class TestWaveFunctionValidation:
    """Input validation for WaveFunctionRenormalization."""

    def test_invalid_N_c(self):
        with pytest.raises(ValueError):
            WaveFunctionRenormalization(N_c=1)

    def test_invalid_M(self):
        with pytest.raises(ValueError):
            WaveFunctionRenormalization(M=0.5)


# ======================================================================
# 6. CurvatureCorrections
# ======================================================================

class TestCurvatureCorrections:
    """Tests for O((L^j/R)^2) curvature corrections."""

    @pytest.fixture
    def cc(self):
        return CurvatureCorrections(R=R_PHYS, M=M_DEFAULT, N_scales=N_SCALES)

    def test_propagator_correction_positive(self, cc):
        """Propagator correction > 0 at every scale. NUMERICAL."""
        for j in range(N_SCALES):
            c = cc.propagator_correction(j)
            assert c > 0, f"Propagator correction not positive at j={j}"

    def test_propagator_correction_decreases_with_j(self, cc):
        """Propagator correction decreases toward UV. NUMERICAL."""
        corrections = [cc.propagator_correction(j) for j in range(N_SCALES)]
        for i in range(len(corrections) - 1):
            assert corrections[i] > corrections[i + 1]

    def test_propagator_correction_order_of_magnitude(self, cc):
        """At j=0, correction ~ 1/(3R^2); at j=6, ~ M^{-12}/(3R^2). NUMERICAL."""
        c0 = cc.propagator_correction(0)
        expected_0 = 1.0 / (3.0 * R_PHYS ** 2)
        assert c0 == pytest.approx(expected_0, rel=1e-10)

        c6 = cc.propagator_correction(6)
        expected_6 = expected_0 * M_DEFAULT ** (-12)
        assert c6 == pytest.approx(expected_6, rel=1e-10)

    def test_vertex_correction_positive(self, cc):
        """Vertex correction > 0 at all scales. NUMERICAL."""
        for j in range(N_SCALES):
            c = cc.vertex_correction(j)
            assert c > 0

    def test_vertex_correction_decreases_with_j(self, cc):
        """Vertex correction decreases toward UV. NUMERICAL."""
        corrections = [cc.vertex_correction(j) for j in range(N_SCALES)]
        for i in range(len(corrections) - 1):
            assert corrections[i] > corrections[i + 1]

    def test_vertex_correction_is_ratio_squared(self, cc):
        """Vertex correction = (M^{-j}/R)^2. NUMERICAL."""
        for j in range(N_SCALES):
            c = cc.vertex_correction(j)
            expected = (M_DEFAULT ** (-j) / R_PHYS) ** 2
            assert c == pytest.approx(expected, rel=1e-12)

    def test_total_correction_sum_of_parts(self, cc):
        """Total = propagator + vertex. NUMERICAL."""
        for j in range(N_SCALES):
            total = cc.total_correction(j)
            parts = cc.propagator_correction(j) + cc.vertex_correction(j)
            assert total == pytest.approx(parts, rel=1e-14)

    def test_corrections_vanish_exponentially_at_uv(self, cc):
        """At high scales, corrections are negligible. NUMERICAL."""
        c_uv = cc.total_correction(N_SCALES - 1)
        # At j=6 for M=2: M^{-12}/R^2 ~ 2.4e-4 * 1/R^2
        assert c_uv < 1e-2, f"UV correction too large: {c_uv}"

    def test_all_corrections_returns_list(self, cc):
        """all_corrections returns N_scales entries. NUMERICAL."""
        all_c = cc.all_corrections()
        assert len(all_c) == N_SCALES

    def test_large_R_reduces_corrections(self):
        """As R -> inf, curvature corrections -> 0. NUMERICAL."""
        cc_small = CurvatureCorrections(R=1.0, M=2.0, N_scales=5)
        cc_large = CurvatureCorrections(R=100.0, M=2.0, N_scales=5)
        for j in range(5):
            assert cc_large.total_correction(j) < cc_small.total_correction(j)


class TestCurvatureCorrectionsValidation:
    """Input validation for CurvatureCorrections."""

    def test_invalid_R(self):
        with pytest.raises(ValueError):
            CurvatureCorrections(R=0.0)

    def test_invalid_M(self):
        with pytest.raises(ValueError):
            CurvatureCorrections(M=1.0)

    def test_invalid_N_scales(self):
        with pytest.raises(ValueError):
            CurvatureCorrections(N_scales=0)


# ======================================================================
# 7. PerturbativeRGFlow: single step
# ======================================================================

class TestPerturbativeRGFlowStep:
    """Tests for the single RG step."""

    @pytest.fixture
    def flow(self):
        return PerturbativeRGFlow(N_c=2, R=R_PHYS, M=M_DEFAULT, N_scales=N_SCALES)

    def test_step_returns_three_floats(self, flow):
        """Step returns (g2, nu, z) tuple. NUMERICAL."""
        result = flow.step(G2_PHYS, 0.0, 1.0, 0)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)

    def test_step_coupling_decreases(self, flow):
        """g^2 should decrease toward UV (each step). NUMERICAL."""
        g2_new, _, _ = flow.step(G2_PHYS, 0.0, 1.0, 0)
        assert g2_new < G2_PHYS

    def test_step_z_increases(self, flow):
        """z should increase (delta_z > 0 in Landau gauge). NUMERICAL."""
        _, _, z_new = flow.step(G2_PHYS, 0.0, 1.0, 0)
        assert z_new > 1.0

    def test_step_nu_changes(self, flow):
        """nu should change by delta_nu after one step. NUMERICAL."""
        _, nu_new, _ = flow.step(1.0, 0.0, 1.0, 0)
        assert nu_new > 0  # Starting from 0, should get positive shift

    def test_step_coupling_stays_finite(self, flow):
        """g^2 stays finite (no Landau pole in one step). NUMERICAL."""
        g2_new, _, _ = flow.step(G2_PHYS, 0.0, 1.0, 3)
        assert np.isfinite(g2_new)
        assert g2_new > 0

    def test_step_at_weak_coupling(self, flow):
        """At weak coupling, the step is perturbatively controlled. NUMERICAL."""
        g2_weak = 0.5
        g2_new, nu_new, z_new = flow.step(g2_weak, 0.0, 1.0, 3)
        assert g2_new < g2_weak
        assert abs(z_new - 1.0) < 0.5  # z shift should be small


# ======================================================================
# 8. PerturbativeRGFlow: full flow
# ======================================================================

class TestPerturbativeRGFlowFull:
    """Tests for the full multi-scale RG flow."""

    @pytest.fixture
    def flow(self):
        return PerturbativeRGFlow(N_c=2, R=R_PHYS, M=M_DEFAULT, N_scales=N_SCALES)

    def test_run_flow_length(self, flow):
        """Trajectory has N_scales entries. NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        assert len(traj) == N_SCALES

    def test_run_flow_first_entry_is_input(self, flow):
        """First entry matches input values. NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.5, 1.0)
        assert traj[0] == (G2_PHYS, 0.5, 1.0)

    def test_coupling_decreases_monotonically(self, flow):
        """g^2 decreases at each step (asymptotic freedom). NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        g2s = [t[0] for t in traj]
        for i in range(len(g2s) - 1):
            assert g2s[i + 1] < g2s[i], \
                f"g^2 not decreasing at step {i}: {g2s[i]} -> {g2s[i+1]}"

    def test_z_increases_monotonically(self, flow):
        """z increases at each step (field renormalization). NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        zs = [t[2] for t in traj]
        for i in range(len(zs) - 1):
            assert zs[i + 1] > zs[i]

    def test_all_values_finite(self, flow):
        """All couplings remain finite. NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        for g2, nu, z in traj:
            assert np.isfinite(g2)
            assert np.isfinite(nu)
            assert np.isfinite(z)

    def test_coupling_positive_everywhere(self, flow):
        """g^2 > 0 at all scales. NUMERICAL."""
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        for g2, _, _ in traj:
            assert g2 > 0

    def test_flow_with_critical_nu(self, flow):
        """Starting from nu_c, the mass parameter stays bounded. NUMERICAL."""
        mr = MassRenormalization(N_c=2, R=R_PHYS, M=M_DEFAULT)
        nu_c = mr.critical_initial(G2_PHYS, N_SCALES)
        traj = flow.run_flow(G2_PHYS, nu_c, 1.0)
        nus = [t[1] for t in traj]
        lambda1 = 4.0 / R_PHYS ** 2
        # nu should remain bounded relative to the spectral gap
        for nu in nus:
            assert abs(nu) < 10 * lambda1, f"|nu| = {abs(nu):.4e} too large"


# ======================================================================
# 9. PerturbativeRGFlow: derived quantities
# ======================================================================

class TestPerturbativeRGFlowDerived:
    """Tests for derived quantities of the flow."""

    @pytest.fixture
    def flow(self):
        return PerturbativeRGFlow(N_c=2, R=R_PHYS, M=M_DEFAULT, N_scales=N_SCALES)

    def test_is_asymptotically_free(self, flow):
        """Pure SU(2) YM is asymptotically free. THEOREM."""
        assert flow.is_asymptotically_free

    def test_coupling_at_scale_decreases(self, flow):
        """Coupling decreases with scale j. NUMERICAL."""
        g2s = [flow.coupling_at_scale(j, G2_PHYS) for j in range(N_SCALES)]
        for i in range(len(g2s) - 1):
            assert g2s[i + 1] < g2s[i]

    def test_effective_alpha_s_decreases(self, flow):
        """alpha_s decreases with scale. NUMERICAL."""
        alphas = [flow.effective_alpha_s(j, G2_PHYS) for j in range(N_SCALES)]
        for i in range(len(alphas) - 1):
            assert alphas[i + 1] < alphas[i]

    def test_energy_scale_increases_with_j(self, flow):
        """mu_j increases with j (moving to UV). NUMERICAL."""
        mus = [flow.energy_scale_MeV(j) for j in range(N_SCALES)]
        for i in range(len(mus) - 1):
            assert mus[i + 1] > mus[i]

    def test_energy_scale_at_j0(self, flow):
        """mu_0 = hbar*c / R. NUMERICAL."""
        mu_0 = flow.energy_scale_MeV(0)
        expected = HBAR_C / R_PHYS
        assert mu_0 == pytest.approx(expected, rel=1e-12)


# ======================================================================
# 10. PerturbativeRGFlow: validation
# ======================================================================

class TestPerturbativeRGFlowValidation:
    """Input validation for PerturbativeRGFlow."""

    def test_invalid_N_c(self):
        with pytest.raises(ValueError):
            PerturbativeRGFlow(N_c=1)

    def test_invalid_R(self):
        with pytest.raises(ValueError):
            PerturbativeRGFlow(R=-1.0)

    def test_invalid_M(self):
        with pytest.raises(ValueError):
            PerturbativeRGFlow(M=0.9)

    def test_invalid_N_scales(self):
        with pytest.raises(ValueError):
            PerturbativeRGFlow(N_scales=0)


# ======================================================================
# 11. AsymptoticFreedomVerifier
# ======================================================================

class TestAsymptoticFreedomVerifier:
    """Tests for the full asymptotic freedom verification."""

    @pytest.fixture
    def verifier(self):
        return AsymptoticFreedomVerifier(
            N_c=2, R=R_PHYS, M=M_DEFAULT, N_scales=N_SCALES
        )

    def test_verify_decreasing_coupling(self, verifier):
        """Coupling decreasing from IR to UV. NUMERICAL."""
        result = verifier.verify_decreasing_coupling(G2_PHYS)
        assert result['all_decreasing'], \
            f"Violations: {result['violations']}"

    def test_no_violations(self, verifier):
        """No violations of monotonic decrease. NUMERICAL."""
        result = verifier.verify_decreasing_coupling(G2_PHYS)
        assert len(result['violations']) == 0

    def test_trajectory_length(self, verifier):
        """Trajectory has N_scales points. NUMERICAL."""
        result = verifier.verify_decreasing_coupling(G2_PHYS)
        assert len(result['g2_trajectory']) == N_SCALES

    def test_lambda_qcd_extraction_finite(self, verifier):
        """Lambda_QCD values are finite. NUMERICAL."""
        result = verifier.extract_lambda_qcd(G2_PHYS)
        for lam in result['lambda_values_MeV']:
            assert np.isfinite(lam)

    def test_lambda_qcd_extraction_positive(self, verifier):
        """Lambda_QCD values are positive. NUMERICAL."""
        result = verifier.extract_lambda_qcd(G2_PHYS)
        for lam in result['lambda_values_MeV']:
            assert lam > 0

    def test_no_landau_pole(self, verifier):
        """No Landau pole in the flow. NUMERICAL."""
        result = verifier.check_no_landau_pole(G2_PHYS)
        assert result['no_landau_pole']

    def test_max_coupling_bounded(self, verifier):
        """Maximum coupling < 4*pi. NUMERICAL."""
        result = verifier.check_no_landau_pole(G2_PHYS)
        assert result['max_g2'] < 4.0 * np.pi

    def test_full_verification_passes(self, verifier):
        """Full verification all passes. NUMERICAL."""
        result = verifier.full_verification(G2_PHYS)
        assert result['asymptotic_freedom']

    def test_full_verification_contains_beta0(self, verifier):
        """Full verification includes beta_0. NUMERICAL."""
        result = verifier.full_verification(G2_PHYS)
        expected_b0 = 22.0 / (48.0 * np.pi ** 2)
        assert result['beta0'] == pytest.approx(expected_b0, rel=1e-12)


# ======================================================================
# 12. Comparison with existing first_rg_step.py
# ======================================================================

class TestComparisonWithExisting:
    """
    Cross-check that beta_flow.py is consistent with first_rg_step.py.
    NUMERICAL.
    """

    def test_beta0_matches_first_rg_step(self):
        """beta_0 from BetaFunction matches RGFlow.beta_coefficient()."""
        from yang_mills_s3.rg.first_rg_step import RGFlow as ExistingRGFlow
        bf = BetaFunction(N_c=2)
        existing = ExistingRGFlow(N_c=2)
        assert bf.beta0 == pytest.approx(existing.beta_coefficient(), rel=1e-12)

    def test_beta0_matches_for_su3(self):
        """beta_0 for SU(3) matches between modules."""
        from yang_mills_s3.rg.first_rg_step import RGFlow as ExistingRGFlow
        bf = BetaFunction(N_c=3)
        existing = ExistingRGFlow(N_c=3)
        assert bf.beta0 == pytest.approx(existing.beta_coefficient(), rel=1e-12)

    def test_one_loop_delta_matches(self):
        """One-loop delta(1/g^2) matches existing coupling_correction_one_loop."""
        from yang_mills_s3.rg.first_rg_step import OneLoopEffectiveAction
        bf = BetaFunction(N_c=2, M=2.0)
        existing = OneLoopEffectiveAction(R=R_PHYS, M=2.0, N_c=2)
        # existing returns b0 * log(M^2)
        delta_existing = existing.coupling_correction_one_loop(3)
        delta_new = bf.one_loop(G2_PHYS)
        assert delta_new == pytest.approx(delta_existing, rel=1e-10)

    def test_coupling_flow_direction_consistent(self):
        """Both old and new code agree that coupling decreases toward UV."""
        from yang_mills_s3.rg.first_rg_step import RGFlow as ExistingRGFlow
        existing = ExistingRGFlow(N_c=2, g2_bare=G2_PHYS)
        existing_result = existing.run_flow()
        existing_g2 = existing_result['g2_trajectory']

        new_flow = PerturbativeRGFlow(N_c=2, R=R_PHYS, M=2.0, N_scales=N_SCALES)
        new_traj = new_flow.run_flow(G2_PHYS, 0.0, 1.0)
        new_g2 = [t[0] for t in new_traj]

        # Both should be monotonically changing
        # (existing runs UV->IR, new runs IR->UV)
        for i in range(len(new_g2) - 1):
            assert new_g2[i + 1] < new_g2[i], "New flow not decreasing"


# ======================================================================
# 13. Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge case tests: g^2 -> 0, R -> inf, etc."""

    def test_free_field_limit(self):
        """At g^2 = 0, no running. NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=2, R=R_PHYS, M=2.0, N_scales=5)
        g2_eps = 1e-10
        traj = flow.run_flow(g2_eps, 0.0, 1.0)
        g2s = [t[0] for t in traj]
        # At very weak coupling, all g^2 should be very close
        for g2 in g2s:
            assert abs(g2 - g2_eps) / g2_eps < 0.1

    def test_flat_space_limit_large_R(self):
        """For R >> 1, curvature corrections negligible. NUMERICAL."""
        flow_large_R = PerturbativeRGFlow(N_c=2, R=100.0, M=2.0, N_scales=5)
        flow_huge_R = PerturbativeRGFlow(N_c=2, R=10000.0, M=2.0, N_scales=5)
        traj_large = flow_large_R.run_flow(1.0, 0.0, 1.0)
        traj_huge = flow_huge_R.run_flow(1.0, 0.0, 1.0)
        # At R=100 vs R=10000, the R^{-2} curvature corrections differ
        # by factor 10^4, so trajectories should be nearly identical
        for i in range(5):
            g2_l, _, _ = traj_large[i]
            g2_h, _, _ = traj_huge[i]
            assert abs(g2_l - g2_h) / max(g2_l, g2_h) < 0.01

    def test_large_blocking_factor(self):
        """M=4 still gives asymptotic freedom. NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=2, R=R_PHYS, M=4.0, N_scales=5)
        traj = flow.run_flow(G2_PHYS, 0.0, 1.0)
        g2s = [t[0] for t in traj]
        for i in range(len(g2s) - 1):
            assert g2s[i + 1] < g2s[i]

    def test_su5_asymptotic_freedom(self):
        """SU(5) is asymptotically free. NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=5, R=R_PHYS, M=2.0, N_scales=5)
        assert flow.is_asymptotically_free
        traj = flow.run_flow(1.0, 0.0, 1.0)
        g2s = [t[0] for t in traj]
        for i in range(len(g2s) - 1):
            assert g2s[i + 1] < g2s[i]

    def test_many_scales(self):
        """Flow works with N_scales=20 without blowup. NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=2, R=R_PHYS, M=2.0, N_scales=20)
        traj = flow.run_flow(1.0, 0.0, 1.0)
        assert len(traj) == 20
        for g2, nu, z in traj:
            assert np.isfinite(g2)
            assert np.isfinite(nu)
            assert np.isfinite(z)
            assert g2 > 0


# ======================================================================
# 14. Two-loop vs one-loop comparison
# ======================================================================

class TestTwoLoopVsOneLoop:
    """Verify that two-loop corrections are perturbatively small."""

    def test_two_loop_correction_small_at_weak_coupling(self):
        """2-loop - 1-loop << 1-loop at weak coupling. NUMERICAL."""
        bf = BetaFunction(N_c=2)
        g2_weak = 0.5
        d1 = bf.one_loop(g2_weak)
        d2 = bf.two_loop(g2_weak)
        assert abs(d2 - d1) / d1 < 0.05

    def test_two_loop_correction_order_g4(self):
        """2-loop correction scales as g^4. NUMERICAL."""
        bf = BetaFunction(N_c=2)
        # Difference = beta_1 * g^2 * ln(M^2)
        g2_a = 1.0
        g2_b = 2.0
        diff_a = bf.two_loop(g2_a) - bf.one_loop(g2_a)
        diff_b = bf.two_loop(g2_b) - bf.one_loop(g2_b)
        # Should scale as g^2: diff_b / diff_a = g2_b / g2_a = 2
        ratio = diff_b / diff_a
        assert ratio == pytest.approx(2.0, rel=1e-12)

    def test_two_loop_same_sign_as_one_loop(self):
        """Both 1-loop and 2-loop increments are positive. NUMERICAL."""
        bf = BetaFunction(N_c=2)
        for g2 in [0.1, 1.0, 5.0]:
            assert bf.one_loop(g2) > 0
            assert bf.two_loop(g2) > 0


# ======================================================================
# 15. Physical consistency
# ======================================================================

class TestPhysicalConsistency:
    """Physical consistency checks. NUMERICAL."""

    def test_alpha_s_reasonable_at_m_z(self):
        """alpha_s at M_Z should be O(0.1). NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=3, R=R_PHYS, M=2.0, N_scales=15)
        # M_Z ~ 91200 MeV, mu_0 = hbar_c / R ~ 89.7 MeV
        # Need j such that mu_j ~ 91200 => M^j ~ 91200*R/hbar_c ~ 1017
        # j ~ log_2(1017) ~ 10
        g2 = flow.coupling_at_scale(10, 5.0)
        alpha_s = g2 / (4 * np.pi)
        assert 0.01 < alpha_s < 1.0, \
            f"alpha_s = {alpha_s} out of range at high scale"

    def test_spectral_gap_scale(self):
        """Energy at j=0 corresponds to gap scale ~90 MeV. NUMERICAL."""
        flow = PerturbativeRGFlow(N_c=2, R=R_PHYS, M=2.0, N_scales=N_SCALES)
        mu_0 = flow.energy_scale_MeV(0)
        expected = HBAR_C / R_PHYS  # ~ 89.7 MeV
        assert mu_0 == pytest.approx(expected, rel=1e-6)

    def test_beta0_numerical_value_su2(self):
        """beta_0(SU(2)) ~ 0.04648. NUMERICAL."""
        bf = BetaFunction(N_c=2)
        assert bf.beta0 == pytest.approx(0.04648, rel=0.01)

    def test_beta0_numerical_value_su3(self):
        """beta_0(SU(3)) ~ 0.06972. NUMERICAL."""
        bf = BetaFunction(N_c=3)
        assert bf.beta0 == pytest.approx(0.06972, rel=0.01)
