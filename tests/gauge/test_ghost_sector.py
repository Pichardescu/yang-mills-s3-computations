"""
Tests for GhostSector — Phase 1.4 of the Yang-Mills Lab Plan.

Verifies:
  - FP operator spectrum matches Δ₀ on the adjoint bundle of S³
  - Ghost determinant is positive (no sign problem)
  - Zero mode count = dim(su(N))
  - After removing zero modes: lowest eigenvalue = 3/R²
  - Ghosts do NOT modify the physical mass gap
  - BRST cohomology is well-defined on S³
  - Ghost loop effective potential has correct sign
  - Everything works for both SU(2) and SU(3)

KEY FINDING: The ghost sector on S³ is clean — positive determinant,
finitely many zero modes from global gauge only, and no effect on the
physical mass gap (BRST cohomology argument).
"""

import pytest
import numpy as np
from yang_mills_s3.gauge.ghost_sector import GhostSector


class TestFPOperatorSpectrum:
    """Tests for the FP operator spectrum on S³."""

    def test_zero_mode_count_su2(self):
        """SU(2): zero mode count = dim(su(2)) = 3."""
        result = GhostSector.fp_operator_spectrum(R=1.0, l_max=10, N=2)
        assert result['zero_mode_count'] == 3

    def test_zero_mode_count_su3(self):
        """SU(3): zero mode count = dim(su(3)) = 8."""
        result = GhostSector.fp_operator_spectrum(R=1.0, l_max=10, N=3)
        assert result['zero_mode_count'] == 8

    def test_zero_mode_count_general(self):
        """SU(N): zero mode count = N² - 1."""
        for N in [2, 3, 4, 5]:
            result = GhostSector.fp_operator_spectrum(R=1.0, l_max=5, N=N)
            assert result['zero_mode_count'] == N**2 - 1

    def test_lowest_nonzero_su2(self):
        """Lowest non-zero eigenvalue = 3/R² for SU(2) at R=1."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=2)
        assert abs(result['lowest_nonzero'] - 3.0) < 1e-12

    def test_lowest_nonzero_su3(self):
        """Lowest non-zero eigenvalue = 3/R² for SU(3) at R=1."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=3)
        assert abs(result['lowest_nonzero'] - 3.0) < 1e-12

    def test_lowest_nonzero_scales_with_R(self):
        """Lowest non-zero eigenvalue = 3/R² scales correctly."""
        R = 2.2
        result = GhostSector.fp_operator_spectrum(R=R, N=2)
        expected = 3.0 / R**2
        assert abs(result['lowest_nonzero'] - expected) < 1e-12

    def test_lowest_multiplicity_su2(self):
        """SU(2): multiplicity of l=1 ghost mode = 4 × 3 = 12."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=2)
        assert result['lowest_multiplicity'] == 4 * 3

    def test_lowest_multiplicity_su3(self):
        """SU(3): multiplicity of l=1 ghost mode = 4 × 8 = 32."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=3)
        assert result['lowest_multiplicity'] == 4 * 8

    def test_spectrum_first_entry_is_zero(self):
        """First entry of spectrum is the zero mode (eigenvalue 0)."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=2)
        first_ev, first_mult = result['spectrum'][0]
        assert abs(first_ev) < 1e-12  # eigenvalue = 0
        assert first_mult == 3  # dim(su(2)) = 3

    def test_spectrum_second_entry_is_3_over_R2(self):
        """Second entry of spectrum is the l=1 mode (eigenvalue 3/R²)."""
        R = 1.0
        result = GhostSector.fp_operator_spectrum(R=R, N=2)
        second_ev, second_mult = result['spectrum'][1]
        expected_ev = 3.0 / R**2  # l=1: 1*3/R² = 3/R²
        assert abs(second_ev - expected_ev) < 1e-12
        # Scalar multiplicity for l=1 on S³ is (1+1)² = 4, times dim(adj)=3
        assert second_mult == 4 * 3

    def test_spectrum_eigenvalue_formula(self):
        """Eigenvalues follow l(l+2)/R² for each l."""
        R = 1.5
        result = GhostSector.fp_operator_spectrum(R=R, l_max=5, N=2)
        for l, (ev, mult) in enumerate(result['spectrum']):
            expected_ev = l * (l + 2) / R**2
            assert abs(ev - expected_ev) < 1e-10, f"l={l}: got {ev}, expected {expected_ev}"

    def test_spectrum_multiplicity_formula(self):
        """Multiplicities follow (l+1)² × dim(adj) for each l."""
        result = GhostSector.fp_operator_spectrum(R=1.0, l_max=5, N=2)
        dim_adj = 3  # SU(2)
        for l, (ev, mult) in enumerate(result['spectrum']):
            expected_mult = (l + 1)**2 * dim_adj
            assert mult == expected_mult, f"l={l}: got mult {mult}, expected {expected_mult}"

    def test_label_is_theorem(self):
        """FP spectrum is labeled THEOREM."""
        result = GhostSector.fp_operator_spectrum(R=1.0, N=2)
        assert result['label'] == 'THEOREM'


class TestGhostDeterminantSign:
    """Tests for the ghost determinant sign."""

    def test_sign_positive(self):
        """Ghost determinant sign is positive."""
        result = GhostSector.ghost_determinant_sign(R=1.0, N=2)
        assert result['sign'] == 'positive'

    def test_no_sign_problem(self):
        """No sign problem from ghosts on S³."""
        result = GhostSector.ghost_determinant_sign(R=1.0, N=2)
        assert result['sign_problem'] is False

    def test_sign_problem_on_R3(self):
        """Sign problem exists on R³."""
        result = GhostSector.ghost_determinant_sign(R=1.0, N=2)
        assert result['sign_problem_on_R3'] is True

    def test_eigenvalues_positive(self):
        """All eigenvalues of M_FP are positive (after zero mode removal)."""
        result = GhostSector.ghost_determinant_sign(R=1.0, N=2)
        assert result['eigenvalues_positive'] is True

    def test_sign_positive_su3(self):
        """Ghost determinant sign is positive for SU(3) too."""
        result = GhostSector.ghost_determinant_sign(R=1.0, N=3)
        assert result['sign'] == 'positive'

    def test_sign_positive_various_R(self):
        """Ghost determinant sign is positive for various R."""
        for R in [0.5, 1.0, 2.2, 5.0]:
            result = GhostSector.ghost_determinant_sign(R, N=2)
            assert result['sign'] == 'positive'

    def test_has_reason(self):
        """The positivity claim has a documented reason."""
        result = GhostSector.ghost_determinant_sign(R=1.0)
        assert 'reason' in result
        assert len(result['reason']) > 0


class TestGhostContributionToGap:
    """Tests for the ghost contribution to the mass gap."""

    def test_does_not_modify_gap(self):
        """Ghosts do NOT modify the physical mass gap."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=2)
        assert result['modifies_physical_gap'] is False

    def test_lowest_ghost_eigenvalue_correct(self):
        """Lowest ghost eigenvalue = 3/R²."""
        R = 2.0
        result = GhostSector.ghost_contribution_to_gap(R=R, N=2)
        expected = 3.0 / R**2
        assert abs(result['lowest_ghost_eigenvalue'] - expected) < 1e-12

    def test_ym_gap_eigenvalue_correct(self):
        """YM gap eigenvalue = 4/R² (coexact gap)."""
        R = 2.0
        result = GhostSector.ghost_contribution_to_gap(R=R, N=2)
        expected = 4.0 / R**2
        assert abs(result['ym_gap_eigenvalue'] - expected) < 1e-12

    def test_gap_ratio(self):
        """Ghost gap / YM gap = 3/4."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=2)
        expected_ratio = 3.0 / 4.0
        assert abs(result['gap_ratio_ghost_to_ym'] - expected_ratio) < 1e-12

    def test_does_not_modify_gap_su3(self):
        """Ghosts do NOT modify the physical mass gap for SU(3)."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=3)
        assert result['modifies_physical_gap'] is False

    def test_contributes_to_list(self):
        """Ghost contributes to effective action, running coupling, FP measure."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=2)
        assert len(result['contributes_to']) > 0

    def test_does_not_contribute_to_gap_in_list(self):
        """Mass gap is explicitly listed as not affected."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=2)
        does_not = result['does_not_contribute_to']
        assert 'physical mass gap' in does_not

    def test_ghost_beta_contribution_su2(self):
        """Ghost beta function contribution = N/3 for SU(N)."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=2)
        expected = 2.0 / 3.0
        assert abs(result['ghost_beta_contribution'] - expected) < 1e-12

    def test_ghost_beta_contribution_su3(self):
        """Ghost beta function contribution = 3/3 = 1 for SU(3)."""
        result = GhostSector.ghost_contribution_to_gap(R=1.0, N=3)
        expected = 3.0 / 3.0
        assert abs(result['ghost_beta_contribution'] - expected) < 1e-12


class TestGhostZeroModeAnalysis:
    """Tests for the ghost zero mode analysis."""

    def test_n_zero_modes_su2(self):
        """SU(2): n_zero_modes = dim(su(2)) = 3."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        assert result['n_zero_modes'] == 3

    def test_n_zero_modes_su3(self):
        """SU(3): n_zero_modes = dim(su(3)) = 8."""
        result = GhostSector.ghost_zero_mode_analysis(N=3)
        assert result['n_zero_modes'] == 8

    def test_n_zero_modes_general(self):
        """SU(N): n_zero_modes = N² - 1."""
        for N in [2, 3, 4, 5]:
            result = GhostSector.ghost_zero_mode_analysis(N=N)
            assert result['n_zero_modes'] == N**2 - 1

    def test_zero_mode_origin(self):
        """Zero modes are from constant (global) gauge transformations."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        assert 'constant' in result['zero_mode_origin'].lower()

    def test_after_removal_lowest_eigenvalue(self):
        """After removing zero modes: lowest eigenvalue = 3/R²."""
        R = 2.0
        result = GhostSector.ghost_zero_mode_analysis(R=R, N=2)
        expected = 3.0 / R**2
        assert abs(result['after_removal']['lowest_eigenvalue'] - expected) < 1e-12

    def test_after_removal_strictly_positive(self):
        """After removing zero modes: spectrum is strictly positive."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        assert result['after_removal']['strictly_positive'] is True

    def test_vol_gauge_group_su2(self):
        """Vol(SU(2)) = 2π²."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        expected = 2.0 * np.pi**2
        assert abs(result['vol_gauge_group'] - expected) < 1e-10

    def test_vol_gauge_group_su3(self):
        """Vol(SU(3)) = √3 π⁵ / 4."""
        result = GhostSector.ghost_zero_mode_analysis(N=3)
        expected = np.sqrt(3.0) * np.pi**5 / 4.0
        assert abs(result['vol_gauge_group'] - expected) < 1e-6

    def test_s3_vs_r3_comparison(self):
        """S³ has only constant zero modes; R³ has additional ones."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        comp = result['comparison_with_R3']
        assert comp['S3_zero_modes'] == 3
        assert 'constant only' in comp['S3_zero_mode_type'].lower()
        assert 'non-constant' in comp['R3_additional_problem'].lower()

    def test_label_is_theorem(self):
        """Zero mode analysis is labeled THEOREM."""
        result = GhostSector.ghost_zero_mode_analysis(N=2)
        assert result['label'] == 'THEOREM'


class TestBRSTCohomology:
    """Tests for the BRST cohomology analysis."""

    def test_well_defined(self):
        """BRST cohomology is well-defined on S³."""
        result = GhostSector.brst_cohomology_analysis(R=1.0, N=2)
        assert result['well_defined'] is True

    def test_physical_gap_equals_ym_gap(self):
        """Physical gap = YM gap (ghosts excluded by BRST)."""
        result = GhostSector.brst_cohomology_analysis(R=1.0, N=2)
        assert result['gap_equals_ym_gap'] is True

    def test_physical_gap_value(self):
        """Physical gap = 4/R² (coexact gap)."""
        R = 2.0
        result = GhostSector.brst_cohomology_analysis(R=R, N=2)
        expected = 4.0 / R**2
        assert abs(result['physical_gap'] - expected) < 1e-12

    def test_ghost_gap_value(self):
        """Ghost gap = 3/R²."""
        R = 2.0
        result = GhostSector.brst_cohomology_analysis(R=R, N=2)
        expected = 3.0 / R**2
        assert abs(result['ghost_gap'] - expected) < 1e-12

    def test_ghost_states_excluded(self):
        """Ghost states are excluded by BRST."""
        result = GhostSector.brst_cohomology_analysis(R=1.0, N=2)
        assert 'ghost states' in result['excluded_by_brst']

    def test_longitudinal_modes_excluded(self):
        """Longitudinal gauge modes are excluded by BRST."""
        result = GhostSector.brst_cohomology_analysis(R=1.0, N=2)
        assert 'longitudinal gauge modes' in result['excluded_by_brst']

    def test_physical_spectrum_first_eigenvalue(self):
        """First physical eigenvalue = 5/R² (l=1 of Δ₁)."""
        R = 1.0
        result = GhostSector.brst_cohomology_analysis(R=R, N=2)
        first_ev = result['physical_spectrum'][0][0]
        expected = (1 * 3 + 2) / R**2  # l=1: l(l+2)+2 = 5
        assert abs(first_ev - expected) < 1e-12

    def test_physical_spectrum_second_eigenvalue(self):
        """Second physical eigenvalue = 10/R² (l=2 of Δ₁)."""
        R = 1.0
        result = GhostSector.brst_cohomology_analysis(R=R, N=2)
        second_ev = result['physical_spectrum'][1][0]
        expected = (2 * 4 + 2) / R**2  # l=2: l(l+2)+2 = 10
        assert abs(second_ev - expected) < 1e-12

    def test_well_defined_su3(self):
        """BRST cohomology is well-defined for SU(3) too."""
        result = GhostSector.brst_cohomology_analysis(R=1.0, N=3)
        assert result['well_defined'] is True


class TestGhostLoopEffectivePotential:
    """Tests for the ghost loop effective potential."""

    def test_sign_negative(self):
        """Ghost loop potential sign is negative (anticommuting statistics)."""
        result = GhostSector.ghost_loop_effective_potential(R=1.0, N=2)
        assert result['sign'] == 'negative'

    def test_potential_negative(self):
        """Ghost loop potential partial sum is negative."""
        result = GhostSector.ghost_loop_effective_potential(R=1.0, N=2)
        assert result['V_ghost_partial'] < 0

    def test_n_modes_summed_positive(self):
        """Number of modes summed is positive."""
        result = GhostSector.ghost_loop_effective_potential(R=1.0, N=2, l_max=5)
        assert result['n_modes_summed'] > 0

    def test_n_modes_increases_with_l_max(self):
        """More modes summed with larger l_max."""
        r1 = GhostSector.ghost_loop_effective_potential(R=1.0, N=2, l_max=5)
        r2 = GhostSector.ghost_loop_effective_potential(R=1.0, N=2, l_max=10)
        assert r2['n_modes_summed'] > r1['n_modes_summed']

    def test_su3_more_modes_than_su2(self):
        """SU(3) has more modes than SU(2) (larger adjoint dim)."""
        r2 = GhostSector.ghost_loop_effective_potential(R=1.0, N=2, l_max=5)
        r3 = GhostSector.ghost_loop_effective_potential(R=1.0, N=3, l_max=5)
        assert r3['n_modes_summed'] > r2['n_modes_summed']


class TestCompleteGhostAnalysis:
    """Tests for the complete ghost analysis."""

    def test_returns_all_components(self):
        """Complete analysis contains all sub-analyses."""
        result = GhostSector.complete_ghost_analysis(R=1.0, N=2)
        assert 'spectrum' in result
        assert 'determinant_sign' in result
        assert 'gap_contribution' in result
        assert 'zero_modes' in result
        assert 'brst' in result
        assert 'effective_potential' in result

    def test_complete_su2(self):
        """Complete analysis works for SU(2)."""
        result = GhostSector.complete_ghost_analysis(R=2.2, N=2)
        assert result['gauge_group'] == 'SU(2)'
        assert result['adjoint_dim'] == 3
        assert result['determinant_sign']['sign'] == 'positive'
        assert result['gap_contribution']['modifies_physical_gap'] is False

    def test_complete_su3(self):
        """Complete analysis works for SU(3)."""
        result = GhostSector.complete_ghost_analysis(R=2.2, N=3)
        assert result['gauge_group'] == 'SU(3)'
        assert result['adjoint_dim'] == 8
        assert result['determinant_sign']['sign'] == 'positive'
        assert result['gap_contribution']['modifies_physical_gap'] is False

    def test_internal_consistency_lowest_eigenvalue(self):
        """Lowest non-zero eigenvalue is consistent across sub-analyses."""
        R = 2.2
        result = GhostSector.complete_ghost_analysis(R=R, N=2)
        # From spectrum
        ev_spectrum = result['spectrum']['lowest_nonzero']
        # From gap contribution
        ev_gap = result['gap_contribution']['lowest_ghost_eigenvalue']
        # From zero modes
        ev_zero = result['zero_modes']['after_removal']['lowest_eigenvalue']
        # From BRST
        ev_brst = result['brst']['ghost_gap']
        # All should be 3/R²
        expected = 3.0 / R**2
        assert abs(ev_spectrum - expected) < 1e-12
        assert abs(ev_gap - expected) < 1e-12
        assert abs(ev_zero - expected) < 1e-12
        assert abs(ev_brst - expected) < 1e-12

    def test_internal_consistency_zero_modes(self):
        """Zero mode count is consistent between spectrum and zero_modes."""
        result = GhostSector.complete_ghost_analysis(R=1.0, N=2)
        assert result['spectrum']['zero_mode_count'] == result['zero_modes']['n_zero_modes']


class TestConsistencyWithGribov:
    """Cross-checks between GhostSector and GribovAnalysis."""

    def test_fp_eigenvalue_consistent(self):
        """FP lowest eigenvalue matches between ghost and Gribov modules."""
        from yang_mills_s3.gauge.gribov import GribovAnalysis
        R = 2.2
        N = 2
        ghost_ev = GhostSector.fp_operator_spectrum(R, N=N)['lowest_nonzero']
        gribov_ev = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R, N)
        assert abs(ghost_ev - gribov_ev) < 1e-12

    def test_fp_eigenvalue_consistent_su3(self):
        """FP eigenvalue consistent for SU(3) too."""
        from yang_mills_s3.gauge.gribov import GribovAnalysis
        R = 1.5
        N = 3
        ghost_ev = GhostSector.fp_operator_spectrum(R, N=N)['lowest_nonzero']
        gribov_ev = GribovAnalysis.fp_lowest_eigenvalue_at_vacuum(R, N)
        assert abs(ghost_ev - gribov_ev) < 1e-12

    def test_zero_mode_count_matches_adjoint_dim(self):
        """Ghost zero modes = dim(adj) = Gribov's adjoint_dim."""
        from yang_mills_s3.gauge.gribov import GribovAnalysis
        for N in [2, 3, 4]:
            ghost_zeros = GhostSector.ghost_zero_mode_analysis(N=N)['n_zero_modes']
            gribov_adj = GribovAnalysis._adjoint_dim(N)
            assert ghost_zeros == gribov_adj

    def test_gap_not_modified_by_either(self):
        """Both modules agree: the physical mass gap is preserved."""
        from yang_mills_s3.gauge.gribov import GribovAnalysis
        R = 2.2
        N = 2
        gribov_preserved = GribovAnalysis.gap_preservation(R, N)['gap_preserved']
        ghost_no_modify = not GhostSector.ghost_contribution_to_gap(R, N)['modifies_physical_gap']
        assert gribov_preserved is True
        assert ghost_no_modify is True

    def test_both_agree_on_ym_gap(self):
        """Both modules reference the same YM gap = 4/R² (coexact spectrum)."""
        from yang_mills_s3.gauge.gribov import GribovAnalysis
        R = 2.2
        gribov_gap = GribovAnalysis.gap_preservation(R)['geometric_gap']
        ghost_gap = GhostSector.ghost_contribution_to_gap(R)['ym_gap_eigenvalue']
        expected = 4.0 / R**2
        assert abs(gribov_gap - expected) < 1e-12
        assert abs(ghost_gap - expected) < 1e-12
