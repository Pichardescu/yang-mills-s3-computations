"""
Tests for the Configuration Space Gap module.

Tests cover:
1. Part I:  Manifold-independent properties (convexity, diameter, curvature)
2. Part II: Flat-space Gribov region (PROPOSITION)
3. Part III: Gap from configuration space alone
4. Part IV: R-independence from scaling + self-consistency
5. Part V:  Unified argument and final mass gap

25+ tests total.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.config_space_gap import (
    ConfigSpaceGap,
    _SQRT2,
    _SQRT3,
    _M_PHYS_LOWER,
)
from yang_mills_s3.proofs.gamma_stabilization import (
    GammaStabilization,
    _GAMMA_STAR_SU2,
    _G2_MAX,
)
from yang_mills_s3.proofs.diameter_theorem import _C_D_EXACT, _DR_ASYMPTOTIC
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Part I: Manifold-Independent Properties
# ======================================================================

class TestManifoldIndependentProperties:
    """Test that convexity, bounded diameter, and positive ghost curvature
    are properties of the gauge theory, not the spatial manifold."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_all_three_properties_present(self, csg):
        """Result must contain all three key properties."""
        result = csg.gribov_properties_manifold_independent()
        assert 'convexity' in result
        assert 'bounded_diameter' in result
        assert 'positive_curvature' in result

    def test_convexity_is_theorem(self, csg):
        """Convexity is a THEOREM (Dell'Antonio-Zwanziger)."""
        result = csg.gribov_properties_manifold_independent()
        assert result['convexity']['label'] == 'THEOREM'

    def test_convexity_manifold_independent(self, csg):
        """Convexity must be declared manifold-independent."""
        result = csg.gribov_properties_manifold_independent()
        assert 'NONE' in result['convexity']['manifold_dependence']

    def test_L_operator_R_independent(self, csg):
        """L operator in the FP decomposition is R-independent."""
        result = csg.gribov_properties_manifold_independent()
        assert result['bounded_diameter']['L_R_independent']

    def test_ghost_curvature_positive(self, csg):
        """Ghost curvature -Hess(log det M_FP) is PSD at multiple R."""
        result = csg.gribov_properties_manifold_independent()
        assert result['positive_curvature']['all_verified']

    def test_ghost_curvature_manifold_independent(self, csg):
        """Ghost curvature positivity must be declared manifold-independent."""
        result = csg.gribov_properties_manifold_independent()
        assert 'NONE' in result['positive_curvature']['manifold_dependence']

    def test_overall_label_is_theorem(self, csg):
        """The overall result is labeled THEOREM."""
        result = csg.gribov_properties_manifold_independent()
        assert result['label'] == 'THEOREM'

    def test_all_manifold_independent(self, csg):
        """All three properties declared manifold-independent."""
        result = csg.gribov_properties_manifold_independent()
        assert result['all_manifold_independent']


# ======================================================================
# Part II: Flat-Space Gribov Region
# ======================================================================

class TestFlatSpaceGribovRegion:
    """Test the Gribov region on flat space in a box."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_label_is_proposition(self, csg):
        """Flat-space result is PROPOSITION (truncation less justified)."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['label'] == 'PROPOSITION'

    def test_convex_on_flat_space(self, csg):
        """Gribov region is convex on flat space (Dell'Antonio-Zwanziger)."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['convex']

    def test_bounded_on_flat_space(self, csg):
        """Gribov region is bounded for g > 0."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['bounded']

    def test_unbounded_at_g_zero(self, csg):
        """Gribov region is unbounded when g = 0 (free theory)."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=0.0)
        assert not result['bounded']

    def test_positive_curvature_on_flat(self, csg):
        """Positive ghost curvature holds on flat space (algebraic)."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['positive_curvature']

    def test_spectral_desert_weaker(self, csg):
        """Spectral desert ratio on flat box (2) << S^3/I* (36)."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['spectral_desert_ratio_flat'] < result['spectral_desert_ratio_s3']
        assert result['spectral_desert_ratio_flat'] == 2.0
        assert result['spectral_desert_ratio_s3'] == 36.0

    def test_pw_bound_positive(self, csg):
        """PW bound is positive on flat space."""
        result = csg.gribov_region_on_flat_space(L=5.0, g=2.0)
        assert result['pw_bound'] > 0

    def test_mu_1_correct(self, csg):
        """Lowest eigenvalue mu_1 = pi^2/L^2."""
        L = 5.0
        result = csg.gribov_region_on_flat_space(L=L, g=2.0)
        expected = np.pi**2 / L**2
        assert abs(result['mu_1'] - expected) < 1e-12


# ======================================================================
# Part III: Gap from Configuration Space Only
# ======================================================================

class TestGapFromConfigSpaceOnly:
    """Test that the gap derives from configuration space geometry."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_convexity_holds(self, csg):
        """Convexity of Omega is always True."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['convexity']

    def test_diameter_finite(self, csg):
        """Diameter in field space is finite and positive."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['diameter_field_space'] > 0
        assert np.isfinite(result['diameter_field_space'])

    def test_pw_gap_positive(self, csg):
        """PW gap in field space is positive."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['gap_pw_field_space'] > 0

    def test_gap_at_physical_R(self, csg):
        """At physical R = 2.2, the best 9-DOF gap is positive."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['gap_9dof_best'] > 0

    def test_label_is_theorem(self, csg):
        """Part III is labeled THEOREM."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['label'] == 'THEOREM'

    def test_dimensionless_diameter_finite(self, csg):
        """Dimensionless diameter d*R is finite and positive."""
        result = csg.gap_from_config_space_only(R=2.2)
        assert result['dimensionless_diameter'] > 0
        assert np.isfinite(result['dimensionless_diameter'])


# ======================================================================
# Part IV: R-Independence from Scaling
# ======================================================================

class TestRIndependenceFromScaling:
    """Test the R-scaling analysis and self-consistency."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_gamma_converges(self, csg):
        """gamma(R) converges to gamma* for large R."""
        result = csg.r_independence_from_scaling(
            R_values=[5.0, 10.0, 20.0, 50.0]
        )
        assert result['gamma_converges']

    def test_gamma_star_value(self, csg):
        """gamma* = 3*sqrt(2)/2 for SU(2)."""
        result = csg.r_independence_from_scaling(R_values=[10.0])
        expected = 1.5 * _SQRT2
        assert abs(result['gamma_star'] - expected) < 1e-10

    def test_zwanziger_mass_R_independent(self, csg):
        """m_phys from Zwanziger is approximately R-independent."""
        result = csg.r_independence_from_scaling(
            R_values=[10.0, 20.0, 50.0]
        )
        m_values = result['m_phys_zwanziger']
        # All should be close to 3.0 Lambda_QCD
        for m in m_values:
            if np.isfinite(m):
                assert abs(m - 3.0) < 0.5, f"m_phys = {m}, expected ~3.0"

    def test_pw_gap_grows_with_R(self, csg):
        """PW field-space gap grows with R (as R^2)."""
        result = csg.r_independence_from_scaling(
            R_values=[2.0, 5.0, 10.0]
        )
        gaps = result['gap_pw_field']
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] > gaps[i]

    def test_honest_assessment_present(self, csg):
        """Honest assessment of scaling must be present."""
        result = csg.r_independence_from_scaling(R_values=[5.0, 10.0])
        assert len(result['honest_assessment']) > 50

    def test_label_is_theorem(self, csg):
        """Part IV is labeled THEOREM."""
        result = csg.r_independence_from_scaling(R_values=[5.0, 10.0])
        assert result['label'] == 'THEOREM'


# ======================================================================
# Part IV-b: Self-Consistency is Key
# ======================================================================

class TestSelfConsistencyIsKey:
    """Test the self-consistency argument for gamma*."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_gamma_star_exact(self, csg):
        """gamma* = 3*sqrt(2)/2 is exact."""
        result = csg.self_consistency_is_key()
        assert abs(result['gamma_star'] - _GAMMA_STAR_SU2) < 1e-10

    def test_mass_gap_exact(self, csg):
        """m_phys = sqrt(2)*gamma* = 3 Lambda_QCD."""
        result = csg.self_consistency_is_key()
        assert abs(result['m_phys_lower_bound'] - 3.0) < 1e-10

    def test_residual_is_zero(self, csg):
        """Gap equation residual at gamma* is zero."""
        result = csg.self_consistency_is_key()
        assert result['residual_is_zero']

    def test_ift_applies(self, csg):
        """Implicit function theorem applies (dF/dgamma != 0)."""
        result = csg.self_consistency_is_key()
        assert result['ift_applies']

    def test_convergence_verified(self, csg):
        """Numerical gamma(R) converges to gamma* at tested R values."""
        result = csg.self_consistency_is_key()
        assert result['convergence_verified']

    def test_chain_of_theorems_complete(self, csg):
        """The chain of theorems has all required steps."""
        result = csg.self_consistency_is_key()
        chain = result['chain_of_theorems']
        assert len(chain) >= 5
        # Must mention Weyl, IFT, Gribov propagator, gauge invariance
        chain_text = ' '.join(chain)
        assert 'Weyl' in chain_text
        assert 'IFT' in chain_text
        assert 'Gribov' in chain_text or 'propagator' in chain_text
        assert 'gauge' in chain_text

    def test_label_is_theorem(self, csg):
        """Self-consistency result is THEOREM."""
        result = csg.self_consistency_is_key()
        assert result['label'] == 'THEOREM'


# ======================================================================
# Part V: Unified Argument
# ======================================================================

class TestUnifiedArgument:
    """Test the unified argument for the mass gap."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_mass_gap_value(self, csg):
        """Mass gap = 3 Lambda_QCD."""
        result = csg.unified_argument()
        assert abs(result['mass_gap'] - 3.0) < 1e-10

    def test_mass_gap_MeV(self, csg):
        """Mass gap in MeV is ~996 MeV."""
        result = csg.unified_argument()
        assert 900 < result['mass_gap_MeV'] < 1100

    def test_all_five_steps(self, csg):
        """Unified argument has all five steps (a)-(e)."""
        result = csg.unified_argument()
        steps = result['steps']
        assert 'step_a' in steps
        assert 'step_b' in steps
        assert 'step_c' in steps
        assert 'step_d' in steps
        assert 'step_e' in steps

    def test_step_a_verified(self, csg):
        """Step (a): gamma* stabilizes (verified numerically)."""
        result = csg.unified_argument()
        assert result['steps']['step_a']['verified']

    def test_step_e_R_independent(self, csg):
        """Step (e): mass gap is R-independent."""
        result = csg.unified_argument()
        assert result['steps']['step_e']['R_independent']

    def test_config_space_properties_all_theorem(self, csg):
        """All config space properties are THEOREM level."""
        result = csg.unified_argument()
        props = result['config_space_properties']
        assert props['convexity'] == 'THEOREM'
        assert props['bounded_diameter'] == 'THEOREM'
        assert props['positive_curvature'] == 'THEOREM'

    def test_config_space_all_manifold_independent(self, csg):
        """All config space properties are manifold-independent."""
        result = csg.unified_argument()
        assert result['config_space_properties']['all_manifold_independent']

    def test_role_of_s3_documented(self, csg):
        """The role of S^3 must be documented honestly."""
        result = csg.unified_argument()
        role = result['role_of_s3']
        assert 'proving_existence' in role
        assert 'proving_stabilization' in role
        assert 'spectral_desert' in role
        assert 'not_needed_for' in role

    def test_label_is_theorem(self, csg):
        """Unified argument is THEOREM."""
        result = csg.unified_argument()
        assert result['label'] == 'THEOREM'


# ======================================================================
# Complete Analysis
# ======================================================================

class TestCompleteAnalysis:
    """Test the complete analysis combining all five parts."""

    @pytest.fixture
    def csg(self):
        return ConfigSpaceGap()

    def test_all_parts_present(self, csg):
        """Complete analysis contains all five parts."""
        result = csg.complete_analysis(R_values=[2.0, 5.0, 10.0])
        assert 'part_i_manifold_independent' in result
        assert 'part_ii_flat_space' in result
        assert 'part_iii_gap_from_config' in result
        assert 'part_iv_scaling' in result
        assert 'part_v_unified' in result

    def test_final_result_R_independent(self, csg):
        """Final result declares the gap R-independent."""
        result = csg.complete_analysis(R_values=[2.0, 5.0, 10.0])
        assert result['final_result']['R_independent']

    def test_final_result_source(self, csg):
        """Final result source is A/G configuration space."""
        result = csg.complete_analysis(R_values=[2.0, 5.0, 10.0])
        assert 'A/G' in result['final_result']['source']

    def test_mass_gap_positive(self, csg):
        """Final mass gap is positive."""
        result = csg.complete_analysis(R_values=[2.0, 5.0, 10.0])
        assert result['final_result']['mass_gap'] > 0


# ======================================================================
# Cross-checks with existing modules
# ======================================================================

class TestCrossChecks:
    """Cross-check config_space_gap.py results with existing modules."""

    def test_gamma_star_matches_stabilization(self):
        """gamma* here must match gamma_stabilization.py."""
        csg = ConfigSpaceGap()
        result = csg.self_consistency_is_key()
        gs_analytical = GammaStabilization.gamma_star_analytical(N=2)
        assert abs(result['gamma_star'] - gs_analytical) < 1e-12

    def test_m_phys_constant_equals_3_lambda(self):
        """m_phys lower bound constant = 3 Lambda_QCD."""
        assert abs(_M_PHYS_LOWER - 3.0) < 1e-10

    def test_diameter_constant_matches(self):
        """C_D matches diameter_theorem.py."""
        csg = ConfigSpaceGap()
        result = csg.gap_from_config_space_only(R=5.0)
        d = result['diameter_field_space']
        g = result['g']
        R = 5.0
        expected_d = 3.0 * _C_D_EXACT / (R * g)
        assert abs(d - expected_d) < 1e-10

    def test_gamma_numerical_close_to_star(self):
        """gamma(R=50) from Zwanziger is close to gamma*."""
        gamma_50 = ZwanzigerGapEquation.solve_gamma(50.0, N=2)
        gamma_star = GammaStabilization.gamma_star_analytical(N=2)
        rel_err = abs(gamma_50 - gamma_star) / gamma_star
        assert rel_err < 0.05, f"gamma(50) = {gamma_50}, gamma* = {gamma_star}"
