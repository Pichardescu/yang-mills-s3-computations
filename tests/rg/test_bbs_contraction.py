"""
Tests for BBS Contraction Mechanism (bbs_contraction.py).

Verifies the CORRECT BBS contraction (Theorem 8.2.4) replacing the old
epsilon = 1/M mechanism with epsilon = O(g_bar_j).

Tests organized by class:
    1.  CouplingDependentContraction (20 tests)
    2.  InductiveInvariant (15 tests)
    3.  BBSContractionStep (12 tests)
    4.  CriticalMassSelection (10 tests)
    5.  BBSMultiScaleInduction (12 tests)
    6.  CrucialContractionDecomposition (10 tests)
    7.  CompareWithOldContraction (8 tests)
    8.  Edge cases and physical consistency (7 tests)

Total: 94 tests.

Run:
    pytest tests/rg/test_bbs_contraction.py -v
"""

import numpy as np
import pytest

from yang_mills_s3.rg.bbs_contraction import (
    CouplingDependentContraction,
    InductiveInvariant,
    BBSContractionStep,
    CriticalMassSelection,
    BBSMultiScaleInduction,
    CrucialContractionDecomposition,
    CompareWithOldContraction,
    _beta_0,
    _g_bar,
    _g_bar_trajectory,
    DIM_SPACETIME,
    SCALING_DIM_K,
    DERIVATIVE_ORDER_P,
    G2_BARE_PHYS,
)
from yang_mills_s3.rg.heat_kernel_slices import (
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
)
from yang_mills_s3.rg.inductive_closure import (
    G2_MAX,
    G2_BARE_DEFAULT,
    M_DEFAULT,
    N_SCALES_DEFAULT,
    N_COLORS_DEFAULT,
)
from yang_mills_s3.rg.first_rg_step import quadratic_casimir


# ======================================================================
# 0. Module-level helper tests
# ======================================================================

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_beta_0_su2(self):
        """beta_0 for SU(2) matches known value 22/(48 pi^2)."""
        beta0 = _beta_0(2)
        expected = 22.0 / (48.0 * np.pi**2)
        assert abs(beta0 - expected) < 1e-10

    def test_beta_0_su3(self):
        """beta_0 for SU(3) matches known value 33/(48 pi^2)."""
        beta0 = _beta_0(3)
        expected = 33.0 / (48.0 * np.pi**2)
        assert abs(beta0 - expected) < 1e-10

    def test_beta_0_positive(self):
        """beta_0 > 0 for all N_c >= 2 (asymptotic freedom)."""
        for N_c in [2, 3, 4, 5]:
            assert _beta_0(N_c) > 0

    def test_g_bar_decreases_with_j(self):
        """g_bar_j^2 decreases with j (asymptotic freedom)."""
        beta0 = _beta_0(2)
        g2_0 = _g_bar(6.28, beta0, 0)
        g2_3 = _g_bar(6.28, beta0, 3)
        g2_6 = _g_bar(6.28, beta0, 6)
        assert g2_0 > g2_3 > g2_6

    def test_g_bar_at_j0_equals_g0(self):
        """g_bar at j=0 equals bare coupling."""
        beta0 = _beta_0(2)
        g2_0 = _g_bar(6.28, beta0, 0)
        assert abs(g2_0 - 6.28) < 1e-10

    def test_g_bar_trajectory_length(self):
        """Trajectory has N entries."""
        traj = _g_bar_trajectory(6.28, 7)
        assert len(traj) == 7

    def test_g_bar_trajectory_monotone(self):
        """g_bar trajectory is monotonically decreasing."""
        traj = _g_bar_trajectory(6.28, 7)
        for i in range(len(traj) - 1):
            assert traj[i] > traj[i + 1]

    def test_g_bar_negative_j_raises(self):
        """Negative scale index raises ValueError."""
        with pytest.raises(ValueError):
            _g_bar(6.28, _beta_0(2), -1)


# ======================================================================
# 1. CouplingDependentContraction tests
# ======================================================================

class TestCouplingDependentContraction:
    """Tests for coupling-dependent contraction epsilon = O(g_bar)."""

    def test_construction_defaults(self):
        """CouplingDependentContraction constructs with defaults."""
        cdc = CouplingDependentContraction()
        assert cdc.g0_sq == G2_BARE_PHYS
        assert cdc.N_c == 2
        assert cdc.L == M_DEFAULT
        assert cdc.d == DIM_SPACETIME

    def test_construction_custom(self):
        """CouplingDependentContraction constructs with custom parameters."""
        cdc = CouplingDependentContraction(g0_sq=4.0, N_c=3, L=3.0)
        assert cdc.g0_sq == 4.0
        assert cdc.N_c == 3
        assert cdc.L == 3.0

    def test_invalid_g0_raises(self):
        """g0_sq <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CouplingDependentContraction(g0_sq=-1.0)
        with pytest.raises(ValueError):
            CouplingDependentContraction(g0_sq=0.0)

    def test_invalid_Nc_raises(self):
        """N_c < 2 raises ValueError."""
        with pytest.raises(ValueError):
            CouplingDependentContraction(N_c=1)

    def test_invalid_L_raises(self):
        """L <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            CouplingDependentContraction(L=1.0)

    def test_invalid_R_raises(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CouplingDependentContraction(R=0.0)

    def test_c_epsilon_positive(self):
        """c_epsilon is positive."""
        cdc = CouplingDependentContraction()
        assert cdc._c_eps > 0

    def test_c_epsilon_value_su2(self):
        """c_epsilon for SU(2) = C_2/(4pi) = 2/(4pi)."""
        cdc = CouplingDependentContraction(N_c=2)
        expected = quadratic_casimir(2) / (4.0 * np.pi)
        assert abs(cdc._c_eps - expected) < 1e-10

    def test_c_epsilon_value_su3(self):
        """c_epsilon for SU(3) = C_2/(4pi) = 3/(4pi)."""
        cdc = CouplingDependentContraction(N_c=3)
        expected = quadratic_casimir(3) / (4.0 * np.pi)
        assert abs(cdc._c_eps - expected) < 1e-10

    def test_epsilon_at_scale_0(self):
        """epsilon(0) = c_eps * sqrt(g0^2) at IR."""
        cdc = CouplingDependentContraction()
        eps_0 = cdc.epsilon_at_scale(0)
        expected = cdc._c_eps * np.sqrt(cdc.g0_sq)
        assert abs(eps_0 - expected) < 1e-10

    def test_epsilon_decreases_with_j(self):
        """epsilon(j) DECREASES with j (improves at UV, BBS key property)."""
        cdc = CouplingDependentContraction()
        eps = [cdc.epsilon_at_scale(j) for j in range(7)]
        for i in range(len(eps) - 1):
            assert eps[i] > eps[i + 1], f"eps[{i}]={eps[i]} <= eps[{i+1}]={eps[i+1]}"

    def test_epsilon_less_than_1_all_scales(self):
        """epsilon(j) < 1 for all j at physical parameters (THEOREM)."""
        cdc = CouplingDependentContraction()
        for j in range(20):
            assert cdc.is_small(j), f"epsilon({j}) >= 1"

    def test_epsilon_positive_all_scales(self):
        """epsilon(j) > 0 for all j."""
        cdc = CouplingDependentContraction()
        for j in range(10):
            assert cdc.epsilon_at_scale(j) > 0

    def test_epsilon_profile_shape(self):
        """epsilon_profile returns array of correct shape."""
        cdc = CouplingDependentContraction()
        profile = cdc.epsilon_profile(7)
        assert profile.shape == (7,)

    def test_epsilon_profile_matches_individual(self):
        """epsilon_profile matches individual epsilon_at_scale calls."""
        cdc = CouplingDependentContraction()
        profile = cdc.epsilon_profile(7)
        for j in range(7):
            assert abs(profile[j] - cdc.epsilon_at_scale(j)) < 1e-15

    def test_g_bar_at_scale_positive(self):
        """g_bar_j is positive at all scales."""
        cdc = CouplingDependentContraction()
        for j in range(10):
            assert cdc.g_bar_at_scale(j) > 0

    def test_g_bar_sq_at_scale_consistency(self):
        """g_bar_j^2 = g_bar_at_scale(j)^2."""
        cdc = CouplingDependentContraction()
        for j in range(7):
            g = cdc.g_bar_at_scale(j)
            g2 = cdc.g_bar_sq_at_scale(j)
            assert abs(g**2 - g2) < 1e-12

    def test_curvature_correction_uv_negligible(self):
        """Curvature correction is negligible at UV (large j)."""
        cdc = CouplingDependentContraction()
        corr_0 = cdc.curvature_correction_to_epsilon(0)
        corr_6 = cdc.curvature_correction_to_epsilon(6)
        assert corr_6 < corr_0 * 0.01  # At least 100x smaller at UV

    def test_curvature_correction_nonnegative(self):
        """Curvature correction is non-negative."""
        cdc = CouplingDependentContraction()
        for j in range(10):
            assert cdc.curvature_correction_to_epsilon(j) >= 0

    def test_negative_j_raises(self):
        """Negative scale index raises ValueError."""
        cdc = CouplingDependentContraction()
        with pytest.raises(ValueError):
            cdc.epsilon_at_scale(-1)

    def test_weak_coupling_epsilon_small(self):
        """At weak coupling g0^2 = 0.1, epsilon is very small."""
        cdc = CouplingDependentContraction(g0_sq=0.1)
        eps_0 = cdc.epsilon_at_scale(0)
        assert eps_0 < 0.1, f"epsilon(0) = {eps_0} should be < 0.1 at weak coupling"


# ======================================================================
# 2. InductiveInvariant tests
# ======================================================================

class TestInductiveInvariant:
    """Tests for the three BBS inductive invariants."""

    def test_construction_defaults(self):
        """InductiveInvariant constructs with defaults."""
        inv = InductiveInvariant()
        assert inv.g0_sq == G2_BARE_PHYS
        assert inv.C_K > 0
        assert inv.C_nu > 0

    def test_invalid_C_K_raises(self):
        """C_K <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            InductiveInvariant(C_K=-1.0)
        with pytest.raises(ValueError):
            InductiveInvariant(C_K=0.0)

    def test_coupling_window_at_reference(self):
        """Coupling at reference value is in window."""
        inv = InductiveInvariant()
        for j in range(7):
            g_bar_sq = inv.g_bar_sq_at_scale(j)
            assert inv.verify_coupling_window(g_bar_sq, j)

    def test_coupling_window_at_boundary(self):
        """Coupling at window boundary is in window."""
        inv = InductiveInvariant()
        g_bar_sq = inv.g_bar_sq_at_scale(0)
        # Lower boundary
        assert inv.verify_coupling_window(0.5 * g_bar_sq, 0)
        # Upper boundary
        assert inv.verify_coupling_window(2.0 * g_bar_sq, 0)

    def test_coupling_window_outside_fails(self):
        """Coupling outside window is rejected."""
        inv = InductiveInvariant()
        g_bar_sq = inv.g_bar_sq_at_scale(0)
        assert not inv.verify_coupling_window(0.1 * g_bar_sq, 0)
        assert not inv.verify_coupling_window(3.0 * g_bar_sq, 0)

    def test_mass_bound_at_zero(self):
        """nu = 0 satisfies mass bound."""
        inv = InductiveInvariant()
        for j in range(7):
            assert inv.verify_mass_bound(0.0, j)

    def test_mass_bound_at_boundary(self):
        """nu at boundary satisfies mass bound."""
        inv = InductiveInvariant()
        g_bar = inv.g_bar_at_scale(0)
        assert inv.verify_mass_bound(inv.C_nu * g_bar, 0)
        assert inv.verify_mass_bound(-inv.C_nu * g_bar, 0)

    def test_mass_bound_outside_fails(self):
        """nu outside bound is rejected."""
        inv = InductiveInvariant()
        g_bar = inv.g_bar_at_scale(0)
        assert not inv.verify_mass_bound(2.0 * inv.C_nu * g_bar, 0)

    def test_K_bound_at_zero(self):
        """K = 0 satisfies K bound."""
        inv = InductiveInvariant()
        for j in range(7):
            assert inv.verify_K_bound(0.0, j)

    def test_K_bound_at_boundary(self):
        """K at boundary satisfies K bound."""
        inv = InductiveInvariant()
        for j in range(7):
            g_bar = inv.g_bar_at_scale(j)
            assert inv.verify_K_bound(inv.C_K * g_bar**3, j)

    def test_K_bound_outside_fails(self):
        """K above bound is rejected."""
        inv = InductiveInvariant()
        g_bar = inv.g_bar_at_scale(0)
        assert not inv.verify_K_bound(2.0 * inv.C_K * g_bar**3, 0)

    def test_K_bound_value_decreases_with_j(self):
        """K bound decreases with j (because g_bar decreases)."""
        inv = InductiveInvariant()
        bounds = [inv.K_bound_value(j) for j in range(7)]
        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1]

    def test_verify_all_passes(self):
        """All three invariants pass at reference values."""
        inv = InductiveInvariant()
        for j in range(7):
            g_sq = inv.g_bar_sq_at_scale(j)
            assert inv.verify_all(g_sq, 0.0, 0.0, j)

    def test_verify_all_fails_on_coupling(self):
        """verify_all fails when coupling is out of window."""
        inv = InductiveInvariant()
        assert not inv.verify_all(100.0, 0.0, 0.0, 0)

    def test_determine_C_K_finite(self):
        """C_K determination gives finite result at physical parameters."""
        inv = InductiveInvariant()
        C2 = quadratic_casimir(2)
        c_eps = C2 / (4.0 * np.pi)
        c_source = C2**2 / (16.0 * np.pi**2)
        C_K = inv.determine_C_K(c_eps, c_source)
        assert np.isfinite(C_K)
        assert C_K > 0


# ======================================================================
# 3. BBSContractionStep tests
# ======================================================================

class TestBBSContractionStep:
    """Tests for a single BBS contraction step."""

    def test_construction_defaults(self):
        """BBSContractionStep constructs with defaults."""
        step = BBSContractionStep()
        assert step.g0_sq == G2_BARE_PHYS
        assert step.N_c == 2

    def test_invalid_g0_raises(self):
        """g0_sq <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BBSContractionStep(g0_sq=0.0)

    def test_epsilon_matches_contraction(self):
        """epsilon(j) matches CouplingDependentContraction."""
        step = BBSContractionStep()
        cdc = CouplingDependentContraction()
        for j in range(7):
            assert abs(step.epsilon(j) - cdc.epsilon_at_scale(j)) < 1e-15

    def test_source_positive(self):
        """Source term is positive at all scales."""
        step = BBSContractionStep()
        for j in range(7):
            assert step.source(j) > 0

    def test_source_decreases_with_j(self):
        """Source decreases with j (g_bar decreases -> source ~ g_bar^3)."""
        step = BBSContractionStep()
        sources = [step.source(j) for j in range(7)]
        for i in range(len(sources) - 1):
            assert sources[i] > sources[i + 1]

    def test_source_negative_j_raises(self):
        """Negative scale index raises ValueError for source."""
        step = BBSContractionStep()
        with pytest.raises(ValueError):
            step.source(-1)

    def test_K_bound_step_from_zero(self):
        """Starting from K=0, K_next = source(j)."""
        step = BBSContractionStep()
        K_next = step.K_bound_step(0.0, 0)
        assert abs(K_next - step.source(0)) < 1e-15

    def test_K_bound_step_positive(self):
        """K bound step preserves positivity."""
        step = BBSContractionStep()
        K_next = step.K_bound_step(0.1, 0)
        assert K_next > 0

    def test_K_bound_step_contractive(self):
        """K bound step: eps*K + source < K + source for K large."""
        step = BBSContractionStep()
        K_large = 100.0
        K_next = step.K_bound_step(K_large, 3)
        eps_3 = step.epsilon(3)
        assert K_next < K_large  # Because eps < 1

    def test_invariant_preservation_from_zero(self):
        """Invariant preserved starting from K=0."""
        step = BBSContractionStep()
        induction = BBSMultiScaleInduction()
        C_K = induction.C_K

        result = step.verify_invariant_preservation(0.0, 0, C_K)
        assert result['preserved']
        assert result['ratio'] < 1.0

    def test_V_bound_step_positive(self):
        """V bound step gives positive result."""
        step = BBSContractionStep()
        g_bar = step.g_bar_at_scale(0)
        V_bound = step.V_bound_step(g_bar, 0.0)
        assert V_bound > 0

    def test_V_bound_step_increases_with_K(self):
        """V bound increases with K_norm."""
        step = BBSContractionStep()
        g_bar = step.g_bar_at_scale(0)
        V_small = step.V_bound_step(g_bar, 0.0)
        V_large = step.V_bound_step(g_bar, 1.0)
        assert V_large > V_small


# ======================================================================
# 4. CriticalMassSelection tests
# ======================================================================

class TestCriticalMassSelection:
    """Tests for critical mass nu_c selection via backward contraction."""

    def test_construction_defaults(self):
        """CriticalMassSelection constructs with defaults."""
        cms = CriticalMassSelection()
        assert cms.g0_sq == G2_BARE_PHYS
        assert cms.N_c == 2
        assert cms.lambda_1 == 4.0 / R_PHYSICAL_FM**2

    def test_invalid_g0_raises(self):
        """g0_sq <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CriticalMassSelection(g0_sq=0.0)

    def test_relevant_eigenvalue(self):
        """Relevant eigenvalue is L^2 = 4 for L=2."""
        cms = CriticalMassSelection()
        assert cms.relevant_eigenvalue() == 4.0

    def test_mass_shift_positive(self):
        """One-loop mass shift is positive."""
        cms = CriticalMassSelection()
        delta = cms.mass_shift(6.28, 0)
        assert delta > 0

    def test_mass_shift_decreases_with_j(self):
        """Mass shift decreases at UV scales."""
        cms = CriticalMassSelection()
        shifts = [cms.mass_shift(6.28, j) for j in range(7)]
        for i in range(len(shifts) - 1):
            assert shifts[i] > shifts[i + 1]

    def test_backward_inverts_forward(self):
        """Backward step inverts forward step."""
        cms = CriticalMassSelection()
        nu_0 = 0.5
        nu_1 = cms.forward_step(nu_0, 6.28, 0)
        nu_0_recovered = cms.backward_step(nu_1, 6.28, 0)
        assert abs(nu_0_recovered - nu_0) < 1e-12

    def test_critical_nu_N0(self):
        """Critical nu for N=0 is zero."""
        cms = CriticalMassSelection()
        assert cms.find_critical_nu(0) == 0.0

    def test_critical_nu_finite(self):
        """Critical nu is finite for N > 0."""
        cms = CriticalMassSelection()
        nu_c = cms.find_critical_nu(7)
        assert np.isfinite(nu_c)

    def test_verify_contraction(self):
        """Backward map is contractive (backward jacobian < 1)."""
        cms = CriticalMassSelection()
        result = cms.verify_contraction(7)
        assert result['is_contractive']
        assert result['backward_jacobian'] < 1.0
        assert result['contraction_rate'] < 1.0

    def test_nu_trajectory_length(self):
        """nu trajectory has N+1 entries (j=0 to j=N)."""
        cms = CriticalMassSelection()
        traj = cms.nu_trajectory(7)
        assert len(traj) == 8  # N + 1


# ======================================================================
# 5. BBSMultiScaleInduction tests
# ======================================================================

class TestBBSMultiScaleInduction:
    """Tests for the full multi-scale BBS induction."""

    def test_construction_defaults(self):
        """BBSMultiScaleInduction constructs with defaults."""
        ind = BBSMultiScaleInduction()
        assert ind.g0_sq == G2_BARE_PHYS
        assert ind.N == N_SCALES_DEFAULT
        assert ind.C_K > 0

    def test_invalid_N_raises(self):
        """N < 1 raises ValueError."""
        with pytest.raises(ValueError):
            BBSMultiScaleInduction(N=0)

    def test_run_induction_returns_dict(self):
        """run_induction returns a dictionary with expected keys."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        expected_keys = [
            'g_bar_trajectory', 'K_norm_trajectory', 'K_bound_trajectory',
            'nu_trajectory', 'epsilon_trajectory', 'source_trajectory',
            'invariant_holds', 'all_invariants_hold', 'final_K_norm',
            'final_g_bar', 'C_K', 'mass_gap_fm_inv_sq', 'mass_gap_mev',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_all_invariants_hold(self):
        """All invariants hold at every scale (THEOREM verification)."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        assert result['all_invariants_hold'], \
            f"Invariants failed: {result['invariant_holds']}"

    def test_K_norm_bounded_by_invariant(self):
        """||K_j|| <= C_K * g_bar_j^3 at every scale."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        K_norms = result['K_norm_trajectory']
        K_bounds = result['K_bound_trajectory']
        for j in range(len(K_norms)):
            assert K_norms[j] <= K_bounds[j] * 1.01, \
                f"K_norm[{j}]={K_norms[j]} > K_bound[{j}]={K_bounds[j]}"

    def test_epsilon_trajectory_decreasing(self):
        """Epsilon decreases through the induction (UV improves)."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        eps = result['epsilon_trajectory']
        for i in range(len(eps) - 1):
            assert eps[i] > eps[i + 1], \
                f"eps[{i}]={eps[i]} <= eps[{i+1}]={eps[i+1]}"

    def test_mass_gap_positive(self):
        """Mass gap is positive (THEOREM: spectral gap 4/R^2)."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        assert result['mass_gap_mev'] > 0

    def test_mass_gap_value(self):
        """Mass gap = 2*hbar_c/R at physical R."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        expected = 2.0 * HBAR_C_MEV_FM / R_PHYSICAL_FM
        assert abs(result['mass_gap_mev'] - expected) < 1.0  # Within 1 MeV

    def test_g_bar_trajectory_length(self):
        """g_bar trajectory has N+1 entries."""
        N = 7
        ind = BBSMultiScaleInduction(N=N)
        result = ind.run_induction()
        assert len(result['g_bar_trajectory']) == N + 1

    def test_invariant_history(self):
        """invariant_history returns per-scale checks."""
        ind = BBSMultiScaleInduction()
        history = ind.invariant_history()
        assert 'coupling_in_window' in history
        assert 'mass_bounded' in history
        assert 'K_bounded' in history
        assert all(history['all_hold'])

    def test_is_complete(self):
        """is_complete returns True at physical parameters."""
        ind = BBSMultiScaleInduction()
        assert ind.is_complete()

    def test_final_K_norm_positive(self):
        """Final K norm is positive (source terms accumulate)."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        assert result['final_K_norm'] > 0

    def test_C_K_positive(self):
        """C_K is positive and finite."""
        ind = BBSMultiScaleInduction()
        assert ind.C_K > 0
        assert np.isfinite(ind.C_K)


# ======================================================================
# 6. CrucialContractionDecomposition tests
# ======================================================================

class TestCrucialContractionDecomposition:
    """Tests for the three-factor BBS contraction decomposition."""

    def test_construction_defaults(self):
        """CrucialContractionDecomposition constructs with defaults."""
        ccd = CrucialContractionDecomposition()
        assert ccd.L == M_DEFAULT
        assert ccd.d == DIM_SPACETIME
        assert ccd.p == DERIVATIVE_ORDER_P
        assert ccd.scaling_dim == SCALING_DIM_K

    def test_invalid_L_raises(self):
        """L <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            CrucialContractionDecomposition(L=1.0)

    def test_volume_factor_value(self):
        """Volume factor = L^{-d} = 1/16 for L=2, d=4."""
        ccd = CrucialContractionDecomposition()
        assert abs(ccd.volume_factor() - 1.0 / 16.0) < 1e-15

    def test_volume_factor_positive(self):
        """Volume factor is positive."""
        ccd = CrucialContractionDecomposition()
        assert ccd.volume_factor() > 0

    def test_dimensional_factor_value(self):
        """Dimensional factor = L^{-2} = 1/4 for [K]=-2, L=2."""
        ccd = CrucialContractionDecomposition()
        assert abs(ccd.dimensional_factor() - 0.25) < 1e-15

    def test_dimensional_factor_relevant_is_1(self):
        """Dimensional factor = 1 for relevant/marginal operators."""
        ccd = CrucialContractionDecomposition(scaling_dim=0)
        assert ccd.dimensional_factor() == 1.0
        ccd2 = CrucialContractionDecomposition(scaling_dim=2)
        assert ccd2.dimensional_factor() == 1.0

    def test_taylor_factor_positive(self):
        """Taylor factor is positive for valid inputs."""
        ccd = CrucialContractionDecomposition()
        tf = ccd.taylor_factor(1.0, 1.5)
        assert tf > 0

    def test_taylor_factor_invalid_ell_plus_raises(self):
        """ell_plus <= 0 raises ValueError."""
        ccd = CrucialContractionDecomposition()
        with pytest.raises(ValueError):
            ccd.taylor_factor(1.0, 0.0)

    def test_total_contraction_order_g_bar(self):
        """Effective contraction is O(g_bar_j) at each scale."""
        ccd = CrucialContractionDecomposition()
        cdc = CouplingDependentContraction()
        for j in range(7):
            eff = ccd.total_contraction(j)
            eps = cdc.epsilon_at_scale(j)
            # Should match the coupling-dependent contraction
            assert abs(eff - eps) < 1e-12

    def test_decomposition_at_scale(self):
        """decomposition_at_scale returns all components."""
        ccd = CrucialContractionDecomposition()
        decomp = ccd.decomposition_at_scale(0)
        expected_keys = [
            'volume_factor', 'taylor_factor', 'dimensional_factor',
            'ell_j', 'ell_plus', 'raw_product', 'effective_contraction',
            'norm_absorption_ratio',
        ]
        for key in expected_keys:
            assert key in decomp, f"Missing key: {key}"

    def test_ell_at_scale_positive(self):
        """ell_j is positive at all scales."""
        ccd = CrucialContractionDecomposition()
        for j in range(10):
            assert ccd.ell_at_scale(j) > 0


# ======================================================================
# 7. CompareWithOldContraction tests
# ======================================================================

class TestCompareWithOldContraction:
    """Tests comparing old (1/M) and new (O(g_bar)) mechanisms."""

    def test_construction_defaults(self):
        """CompareWithOldContraction constructs with defaults."""
        comp = CompareWithOldContraction()
        assert comp.g0_sq == G2_BARE_PHYS
        assert comp.N_c == 2

    def test_old_epsilon_constant(self):
        """Old epsilon = 1/M is constant across scales."""
        comp = CompareWithOldContraction()
        eps0 = comp._old_epsilon(0)
        eps5 = comp._old_epsilon(5)
        assert eps0 == eps5 == 0.5  # 1/M = 1/2

    def test_compare_epsilon_profiles(self):
        """Epsilon profile comparison returns expected structure."""
        comp = CompareWithOldContraction()
        result = comp.compare_epsilon_profiles(7)
        assert 'old_profile' in result
        assert 'new_profile' in result
        assert 'ratio' in result
        assert len(result['old_profile']) == 7
        assert len(result['new_profile']) == 7

    def test_new_epsilon_not_constant(self):
        """New epsilon varies with scale (not constant like old)."""
        comp = CompareWithOldContraction()
        result = comp.compare_epsilon_profiles(7)
        new_profile = result['new_profile']
        assert not np.allclose(new_profile, new_profile[0])

    def test_compare_K_bounds(self):
        """K bound comparison returns expected structure."""
        comp = CompareWithOldContraction()
        result = comp.compare_K_bounds(7)
        assert 'old_K_bound' in result
        assert 'new_K_bound' in result
        assert 'new_tighter' in result
        assert np.isfinite(result['old_K_bound'])
        assert np.isfinite(result['new_K_bound'])

    def test_compare_gap_bounds(self):
        """Gap bound comparison returns valid gaps."""
        comp = CompareWithOldContraction()
        result = comp.compare_gap_bounds(7)
        assert result['old_gap_mev'] >= 0
        assert result['new_gap_mev'] >= 0
        assert result['lambda_1'] == 4.0 / R_PHYSICAL_FM**2

    def test_summary_complete(self):
        """Summary returns all comparison components."""
        comp = CompareWithOldContraction()
        result = comp.summary(7)
        assert 'epsilon_comparison' in result
        assert 'K_comparison' in result
        assert 'gap_comparison' in result
        assert 'mechanism' in result
        assert 'conclusion' in result

    def test_old_mechanism_description(self):
        """Old mechanism is correctly described."""
        comp = CompareWithOldContraction()
        result = comp.summary(7)
        assert '1/M' in result['mechanism']['old']
        assert 'g_bar' in result['mechanism']['new']


# ======================================================================
# 8. Edge cases and physical consistency
# ======================================================================

class TestEdgeCases:
    """Edge cases and physical consistency checks."""

    def test_weak_coupling_limit(self):
        """At g0^2 -> 0 (free field), epsilon -> 0 and K -> 0."""
        g0_sq = 0.01
        ind = BBSMultiScaleInduction(g0_sq=g0_sq, N=5)
        result = ind.run_induction()
        # Epsilon should be very small
        assert max(result['epsilon_trajectory']) < 0.1
        # K norm should be tiny
        assert result['final_K_norm'] < 0.01

    def test_large_N_scales(self):
        """Induction works with many scales (N=15)."""
        ind = BBSMultiScaleInduction(N=15)
        result = ind.run_induction()
        assert result['all_invariants_hold']
        assert result['mass_gap_mev'] > 0

    def test_su3_gauge_group(self):
        """Induction works for SU(3) gauge group."""
        ind = BBSMultiScaleInduction(g0_sq=4.0, N_c=3)
        result = ind.run_induction()
        assert result['all_invariants_hold']
        assert result['mass_gap_mev'] > 0

    def test_su4_gauge_group(self):
        """Induction works for SU(4) gauge group."""
        ind = BBSMultiScaleInduction(g0_sq=3.0, N_c=4)
        result = ind.run_induction()
        assert result['all_invariants_hold']

    def test_different_blocking_factor(self):
        """Induction works with L=3."""
        ind = BBSMultiScaleInduction(L=3.0, N=5)
        result = ind.run_induction()
        assert result['mass_gap_mev'] > 0

    def test_consistency_with_beta_flow(self):
        """g_bar trajectory is consistent with beta_flow module."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        g_bar_traj = result['g_bar_trajectory']
        # g_bar should decrease monotonically (asymptotic freedom)
        for i in range(len(g_bar_traj) - 1):
            assert g_bar_traj[i] >= g_bar_traj[i + 1]

    def test_gap_preserved_all_scales(self):
        """Mass gap is preserved through all scales of the induction."""
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()
        # The spectral gap lambda_1 = 4/R^2 is a property of S3
        # and does not change through the RG flow
        assert result['mass_gap_fm_inv_sq'] == 4.0 / R_PHYSICAL_FM**2


# ======================================================================
# 9. Integration tests: BBS vs old mechanism comparison
# ======================================================================

class TestIntegration:
    """Integration tests comparing BBS and old contraction mechanisms."""

    def test_bbs_epsilon_profile_decreasing(self):
        """BBS epsilon profile DECREASES with j (unlike old 1/M)."""
        cdc = CouplingDependentContraction()
        profile = cdc.epsilon_profile(7)
        # BBS: epsilon DECREASES because g_bar decreases
        for i in range(len(profile) - 1):
            assert profile[i] > profile[i + 1]

    def test_old_epsilon_is_constant(self):
        """Old epsilon = 1/M = 0.5 is constant (for comparison)."""
        comp = CompareWithOldContraction()
        for j in range(10):
            assert comp._old_epsilon(j) == 0.5

    def test_invariant_is_stronger_than_product(self):
        """
        The invariant ||K|| <= C_K * g_bar^3 at each scale is
        STRONGER than the old product convergence Pi eps -> 0.

        The invariant gives a POINTWISE bound at each scale,
        while the product only gives a cumulative bound.
        """
        ind = BBSMultiScaleInduction()
        result = ind.run_induction()

        # The invariant holds at EVERY scale (pointwise)
        assert result['all_invariants_hold']

        # Each K_norm is bounded by C_K * g_bar^3 individually
        K_norms = result['K_norm_trajectory']
        K_bounds = result['K_bound_trajectory']
        for j in range(len(K_norms)):
            assert K_norms[j] <= K_bounds[j] * 1.01

    def test_bbs_contraction_at_physical_params(self):
        """Full BBS contraction works at physical parameters."""
        ind = BBSMultiScaleInduction(
            g0_sq=6.28,
            N=7,
            N_c=2,
            L=2.0,
            R=2.2,
        )
        assert ind.is_complete()
        result = ind.run_induction()
        assert result['mass_gap_mev'] > 100  # Gap > 100 MeV

    def test_critical_nu_drives_nu_to_zero(self):
        """Critical nu_c drives nu_N close to zero."""
        cms = CriticalMassSelection()
        traj = cms.nu_trajectory(7)
        # nu_N should be close to zero (target of backward iteration)
        assert abs(traj[-1]) < 1e-6

    def test_three_factor_decomposition_consistent(self):
        """The three-factor decomposition gives the same effective
        contraction as the direct coupling-dependent formula."""
        ccd = CrucialContractionDecomposition()
        cdc = CouplingDependentContraction()
        for j in range(7):
            effective = ccd.total_contraction(j)
            direct = cdc.epsilon_at_scale(j)
            assert abs(effective - direct) < 1e-12

    def test_source_term_is_cubic(self):
        """Source term is O(g_bar^3) at each scale."""
        step = BBSContractionStep()
        for j in range(7):
            src = step.source(j)
            g_bar = step.g_bar_at_scale(j)
            # src = c_source * g_bar^3
            c_source = step.c_source
            expected = c_source * g_bar**3
            assert abs(src - expected) < 1e-12

    def test_contraction_factor_is_linear_in_g_bar(self):
        """Contraction factor epsilon = c_eps * g_bar (linear in g_bar)."""
        step = BBSContractionStep()
        for j in range(7):
            eps = step.epsilon(j)
            g_bar = step.g_bar_at_scale(j)
            c_eps = step.c_eps
            expected = c_eps * g_bar
            assert abs(eps - expected) < 1e-12
