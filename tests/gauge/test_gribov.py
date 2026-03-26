"""
Tests for GribovAnalysis — Phase 1.3 of the Yang-Mills Lab Plan.

Verifies:
  - Gribov region is bounded on S³
  - FP lowest eigenvalue at vacuum = 3/R²
  - Gribov horizon is at finite distance from the vacuum
  - Fundamental modular region has finite volume
  - Singer's theorem is documented
  - Gap preservation argument is consistent
  - S³ vs R³ comparison highlights compactness advantages
  - Everything works for both SU(2) and SU(3)

KEY FINDING: On S³, the Gribov problem is TAME. The Gribov region is
bounded, the fundamental modular region is compact with finite volume,
and the mass gap is preserved under restriction to the Gribov region.
"""

import pytest
import numpy as np
from yang_mills_s3.gauge.gribov import GribovAnalysis


class TestFPLowestEigenvalue:
    """Tests for the FP lowest eigenvalue at the vacuum."""

    def test_eigenvalue_su2_R1(self):
        """FP lowest eigenvalue = 3/R² for SU(2) at R=1."""
        result = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R=1.0, N=2)
        assert abs(result - 3.0) < 1e-12

    def test_eigenvalue_su2_R2(self):
        """FP lowest eigenvalue = 3/4 for SU(2) at R=2."""
        result = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R=2.0, N=2)
        expected = 3.0 / 4.0
        assert abs(result - expected) < 1e-12

    def test_eigenvalue_su3_R1(self):
        """FP lowest eigenvalue = 3/R² = 3 for SU(3) at R=1."""
        result = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R=1.0, N=3)
        assert abs(result - 3.0) < 1e-12

    def test_eigenvalue_scales_as_R_minus_2(self):
        """FP eigenvalue scales as 1/R²."""
        R1, R2 = 1.0, 3.0
        e1 = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R1)
        e2 = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R2)
        ratio = e1 / e2
        expected = (R2 / R1) ** 2
        assert abs(ratio - expected) < 1e-10

    def test_eigenvalue_independent_of_N(self):
        """FP eigenvalue at vacuum is 3/R², independent of N."""
        R = 2.2
        for N in [2, 3, 4, 5]:
            result = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R, N)
            expected = 3.0 / R**2
            assert abs(result - expected) < 1e-12

    def test_eigenvalue_positive(self):
        """FP eigenvalue is always positive for R > 0."""
        for R in [0.1, 0.5, 1.0, 2.2, 5.0, 10.0]:
            result = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R)
            assert result > 0


class TestGribovRegionBound:
    """Tests for the Gribov region bound on S³."""

    def test_bounded_on_S3(self):
        """Gribov region is always bounded on S³."""
        result = GribovAnalysis.gribov_region_bound(R=1.0, N=2)
        assert result['bounded'] is True

    def test_not_bounded_on_R3(self):
        """Gribov region is NOT bounded on R³."""
        result = GribovAnalysis.gribov_region_bound(R=1.0, N=2)
        assert result['bounded_on_R3'] is False

    def test_lambda_1_correct(self):
        """λ₁ = 3/R² in the bound."""
        R = 2.0
        result = GribovAnalysis.gribov_region_bound(R)
        expected_lambda = 3.0 / R**2
        assert abs(result['lambda_1'] - expected_lambda) < 1e-12

    def test_Linfty_bound_positive(self):
        """L∞ bound on perturbation is positive."""
        result = GribovAnalysis.gribov_region_bound(R=1.0, N=2)
        assert result['a_Linfty_bound'] > 0

    def test_L2_bound_positive(self):
        """L² bound on perturbation is positive."""
        result = GribovAnalysis.gribov_region_bound(R=1.0, N=2)
        assert result['a_L2_bound'] > 0

    def test_bound_decreases_with_N(self):
        """Larger N => tighter bound (stronger structure constants)."""
        R = 1.0
        bound_su2 = GribovAnalysis.gribov_region_bound(R, N=2)['a_Linfty_bound']
        bound_su3 = GribovAnalysis.gribov_region_bound(R, N=3)['a_Linfty_bound']
        assert bound_su3 < bound_su2

    def test_volume_S3_correct(self):
        """Vol(S³) = 2π²R³."""
        R = 2.0
        result = GribovAnalysis.gribov_region_bound(R)
        expected = 2.0 * np.pi**2 * R**3
        assert abs(result['volume_S3'] - expected) < 1e-10

    def test_bound_scales_with_R(self):
        """L∞ bound scales as 1/R² (from λ₁ = 3/R²)."""
        R1, R2 = 1.0, 2.0
        b1 = GribovAnalysis.gribov_region_bound(R1, N=2)['a_Linfty_bound']
        b2 = GribovAnalysis.gribov_region_bound(R2, N=2)['a_Linfty_bound']
        ratio = b1 / b2
        expected = (R2 / R1)**2
        assert abs(ratio - expected) < 1e-10

    def test_label_is_proposition(self):
        """The bound is labeled PROPOSITION."""
        result = GribovAnalysis.gribov_region_bound(R=1.0)
        assert result['label'] == 'PROPOSITION'


class TestFundamentalModularRegion:
    """Tests for the fundamental modular region."""

    def test_finite_volume(self):
        """Fundamental modular region has finite volume on S³."""
        result = GribovAnalysis.fundamental_modular_region_volume(R=1.0, N=2)
        assert result['finite_volume'] is True

    def test_compact(self):
        """Λ is compact on S³."""
        result = GribovAnalysis.fundamental_modular_region_volume(R=1.0, N=2)
        assert result['compact'] is True

    def test_pi_3_is_Z(self):
        """π₃(SU(N)) = Z for all N ≥ 2."""
        for N in [2, 3, 4, 5]:
            result = GribovAnalysis.fundamental_modular_region_volume(R=1.0, N=N)
            assert result['pi_3_G'] == 'Z'

    def test_n_copies_positive(self):
        """Number of Gribov copies is positive."""
        result = GribovAnalysis.fundamental_modular_region_volume(R=1.0, N=2)
        assert result['n_copies_estimate'] > 0

    def test_volume_ratio_between_0_and_1(self):
        """Volume ratio Vol(Λ)/Vol(Ω) is between 0 and 1."""
        result = GribovAnalysis.fundamental_modular_region_volume(R=1.0, N=2)
        assert 0 < result['volume_ratio'] <= 1.0


class TestGribovHorizonDistance:
    """Tests for the Gribov horizon distance from the vacuum."""

    def test_finite_on_S3(self):
        """Horizon distance is finite on S³."""
        result = GribovAnalysis.gribov_horizon_distance(R=1.0, N=2)
        assert result['finite'] is True

    def test_not_finite_on_R3(self):
        """Horizon distance is generally not finite on R³."""
        result = GribovAnalysis.gribov_horizon_distance(R=1.0, N=2)
        assert result['finite_on_R3'] is False

    def test_Linfty_distance_positive(self):
        """L∞ horizon distance is positive."""
        result = GribovAnalysis.gribov_horizon_distance(R=1.0, N=2)
        assert result['horizon_distance_Linfty'] > 0

    def test_L2_distance_positive(self):
        """L² horizon distance is positive."""
        result = GribovAnalysis.gribov_horizon_distance(R=1.0, N=2)
        assert result['horizon_distance_L2'] > 0

    def test_lambda_1_matches_fp(self):
        """λ₁ in horizon analysis matches FP eigenvalue."""
        R = 2.2
        N = 2
        horizon = GribovAnalysis.gribov_horizon_distance(R, N)
        fp_ev = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R, N)
        assert abs(horizon['lambda_1'] - fp_ev) < 1e-12

    def test_horizon_distance_decreases_with_R(self):
        """Horizon distance (L∞) decreases as R increases (since λ₁ ~ 1/R²)."""
        d1 = GribovAnalysis.gribov_horizon_distance(1.0, 2)['horizon_distance_Linfty']
        d2 = GribovAnalysis.gribov_horizon_distance(2.0, 2)['horizon_distance_Linfty']
        assert d1 > d2

    def test_horizon_distance_decreases_with_N(self):
        """Horizon distance (L∞) decreases with N (stronger structure constants)."""
        d2 = GribovAnalysis.gribov_horizon_distance(1.0, 2)['horizon_distance_Linfty']
        d3 = GribovAnalysis.gribov_horizon_distance(1.0, 3)['horizon_distance_Linfty']
        assert d3 < d2


class TestSingerTheorem:
    """Tests for Singer's theorem documentation."""

    def test_has_statement(self):
        """Singer's theorem has a formal statement."""
        result = GribovAnalysis.singer_theorem()
        assert 'statement' in result
        assert len(result['statement']) > 0

    def test_has_consequence(self):
        """Singer's theorem has documented consequence."""
        result = GribovAnalysis.singer_theorem()
        assert 'consequence' in result
        assert len(result['consequence']) > 0

    def test_has_resolution_on_S3(self):
        """Resolution on S³ is documented."""
        result = GribovAnalysis.singer_theorem()
        assert 'resolution_on_S3' in result
        assert 'compact' in result['resolution_on_S3'].lower()

    def test_label_is_theorem(self):
        """Singer's theorem is labeled THEOREM."""
        result = GribovAnalysis.singer_theorem()
        assert result['label'] == 'THEOREM'


class TestGapPreservation:
    """Tests for the gap preservation argument."""

    def test_gap_preserved(self):
        """Gap is preserved under Gribov restriction."""
        result = GribovAnalysis.gap_preservation(R=1.0, N=2)
        assert result['gap_preserved'] is True

    def test_geometric_gap_correct(self):
        """Geometric gap = 4/R² (coexact spectrum) in the preservation argument."""
        R = 2.2
        result = GribovAnalysis.gap_preservation(R)
        expected = 4.0 / R**2
        assert abs(result['geometric_gap'] - expected) < 1e-12

    def test_fp_eigenvalue_correct(self):
        """FP eigenvalue = 3/R² in the preservation argument."""
        R = 2.2
        result = GribovAnalysis.gap_preservation(R)
        expected = 3.0 / R**2
        assert abs(result['fp_lowest_eigenvalue'] - expected) < 1e-12

    def test_horizon_distance_positive(self):
        """Horizon distance in the argument is positive."""
        result = GribovAnalysis.gap_preservation(R=1.0, N=2)
        assert result['horizon_distance'] > 0

    def test_four_step_argument(self):
        """Gap preservation has a 4-step argument."""
        result = GribovAnalysis.gap_preservation(R=1.0)
        arg = result['argument']
        assert 'step_1' in arg
        assert 'step_2' in arg
        assert 'step_3' in arg
        assert 'step_4' in arg

    def test_label_is_proposition(self):
        """Gap preservation is labeled PROPOSITION."""
        result = GribovAnalysis.gap_preservation(R=1.0)
        assert result['label'] == 'PROPOSITION'

    def test_gap_preserved_for_su3(self):
        """Gap is preserved for SU(3) too."""
        result = GribovAnalysis.gap_preservation(R=1.0, N=3)
        assert result['gap_preserved'] is True

    def test_gap_preserved_for_various_R(self):
        """Gap is preserved for various R values."""
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            result = GribovAnalysis.gap_preservation(R)
            assert result['gap_preserved'] is True


class TestS3VsR3Comparison:
    """Tests for the S³ vs R³ comparison."""

    def test_s3_gribov_bounded(self):
        """Gribov region is bounded on S³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['S3']['gribov_region_bounded'] is True

    def test_r3_gribov_not_bounded(self):
        """Gribov region is NOT bounded on R³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['R3']['gribov_region_bounded'] is False

    def test_s3_spectrum_discrete(self):
        """Spectrum is discrete on S³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['S3']['spectrum_discrete'] is True

    def test_r3_spectrum_not_discrete(self):
        """Spectrum is NOT discrete on R³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['R3']['spectrum_discrete'] is False

    def test_s3_mass_gap_protected(self):
        """Mass gap is protected on S³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['S3']['mass_gap_protected'] is True

    def test_s3_path_integral_well_defined(self):
        """Path integral is well-defined on S³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['S3']['path_integral_well_defined'] is True

    def test_r3_path_integral_not_well_defined(self):
        """Path integral is NOT well-defined on R³ (without additional care)."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['R3']['path_integral_well_defined'] is False

    def test_s3_fp_positive_in_lambda(self):
        """FP determinant is positive inside Λ on S³."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert result['S3']['fp_determinant_positive_in_lambda'] is True

    def test_advantage_documented(self):
        """Advantage of S³ is documented."""
        result = GribovAnalysis.s3_vs_r3_comparison()
        assert 'advantage_of_S3' in result
        assert len(result['advantage_of_S3']) > 0


class TestCompleteAnalysis:
    """Tests for the complete Gribov analysis."""

    def test_returns_all_components(self):
        """Complete analysis contains all sub-analyses."""
        result = GribovAnalysis.complete_analysis(R=1.0, N=2)
        assert 'fp_lowest_eigenvalue' in result
        assert 'region_bound' in result
        assert 'modular_region' in result
        assert 'horizon_distance' in result
        assert 'singer_theorem' in result
        assert 'gap_preservation' in result
        assert 'comparison' in result

    def test_complete_su2(self):
        """Complete analysis works for SU(2)."""
        result = GribovAnalysis.complete_analysis(R=2.2, N=2)
        assert result['gauge_group'] == 'SU(2)'
        assert result['adjoint_dim'] == 3
        assert result['gap_preservation']['gap_preserved'] is True

    def test_complete_su3(self):
        """Complete analysis works for SU(3)."""
        result = GribovAnalysis.complete_analysis(R=2.2, N=3)
        assert result['gauge_group'] == 'SU(3)'
        assert result['adjoint_dim'] == 8
        assert result['gap_preservation']['gap_preserved'] is True

    def test_internal_consistency(self):
        """FP eigenvalue is consistent across sub-analyses."""
        R = 2.2
        result = GribovAnalysis.complete_analysis(R=R, N=2)
        fp_direct = result['fp_lowest_eigenvalue']
        fp_from_bound = result['region_bound']['lambda_1']
        fp_from_horizon = result['horizon_distance']['lambda_1']
        fp_from_gap = result['gap_preservation']['fp_lowest_eigenvalue']
        assert abs(fp_direct - fp_from_bound) < 1e-12
        assert abs(fp_direct - fp_from_horizon) < 1e-12
        assert abs(fp_direct - fp_from_gap) < 1e-12
