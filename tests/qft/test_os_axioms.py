"""
Tests for the Osterwalder-Schrader axioms checker.

Verifies all 5 OS axioms for Yang-Mills on S^3 x R:
    OS0: Regularity
    OS1: Euclidean covariance
    OS2: Reflection positivity
    OS3: Symmetry (gauge invariance)
    OS4: Clustering (mass gap)

Status labels are verified to be honest:
    OS0, OS1, OS3: THEOREM
    OS2: THEOREM on lattice, OPEN in continuum
    OS4: PROPOSITION
"""

import pytest
import numpy as np
from yang_mills_s3.qft.os_axioms import OSAxioms


class TestOS0Regularity:
    """OS0: Schwinger functions are tempered distributions."""

    def test_os0_satisfied(self):
        """OS0 should be satisfied on S^3."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['satisfied'] is True

    def test_os0_status_is_theorem(self):
        """OS0 is a THEOREM on compact manifolds."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['status'] == 'THEOREM'

    def test_os0_compact_manifold(self):
        """The spatial manifold S^3 is compact."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['details']['compact_spatial_manifold'] is True

    def test_os0_discrete_spectrum(self):
        """Spectrum on S^3 is discrete."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['details']['discrete_spectrum'] is True

    def test_os0_first_eigenvalue_is_4(self):
        """First coexact eigenvalue of 1-form Laplacian on S^3(R=1) is 4."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert abs(result['details']['first_eigenvalue'] - 4.0) < 1e-10

    def test_os0_heat_trace_finite(self):
        """Heat kernel trace should be finite."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['details']['heat_trace_finite'] == True

    def test_os0_stronger_than_needed(self):
        """On S^3, regularity is stronger than OS0 requires."""
        result = OSAxioms.check_os0_regularity(R=1.0)
        assert result['details']['stronger_than_needed'] is True

    def test_os0_various_radii(self):
        """OS0 should hold for any radius."""
        for R in [0.5, 1.0, 2.2, 10.0]:
            result = OSAxioms.check_os0_regularity(R=R)
            assert result['satisfied'] is True
            expected_ev = 4.0 / R**2
            assert abs(result['details']['first_eigenvalue'] - expected_ev) < 1e-10


class TestOS1Covariance:
    """OS1: Euclidean covariance under isometries."""

    def test_os1_satisfied(self):
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['satisfied'] is True

    def test_os1_status_is_theorem(self):
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['status'] == 'THEOREM'

    def test_isometry_group_is_so4_cross_r(self):
        """Isometry group of S^3 x R is SO(4) x R."""
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['details']['isometry_group'] == 'SO(4) x R'

    def test_isometry_dimension_is_7(self):
        """dim(SO(4)) + dim(R) = 6 + 1 = 7."""
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['details']['isometry_dim'] == 7

    def test_not_so5(self):
        """
        Important: S^3 x R does NOT have SO(5) symmetry.
        This distinguishes it from R^4 or S^4.
        """
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['details']['not_so5'] is True

    def test_action_and_measure_invariant(self):
        result = OSAxioms.check_os1_covariance(R=1.0)
        assert result['details']['action_invariant'] is True
        assert result['details']['measure_invariant'] is True


class TestOS2ReflectionPositivity:
    """OS2: Reflection positivity -- the critical axiom."""

    def test_os2_satisfied_on_lattice(self):
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['satisfied'] is True

    def test_os2_status_honest(self):
        """
        Status must be honest: THEOREM on lattice, OPEN in continuum.
        This is the hardest axiom.
        """
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert 'THEOREM' in result['status']
        assert 'OPEN' in result['status']

    def test_lattice_proven(self):
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['details']['lattice_proven'] is True

    def test_continuum_not_proven(self):
        """Continuum reflection positivity is NOT proven."""
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['details']['continuum_proven'] is False

    def test_transfer_matrix_positive(self):
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['details']['transfer_matrix_positive'] is True

    def test_hamiltonian_bounded_below(self):
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['details']['hamiltonian_bounded_below'] is True

    def test_energy_gap_positive(self):
        """Energy gap lower bound should be positive for finite R."""
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert result['details']['energy_gap_lower_bound'] > 0

    def test_energy_gap_scales_with_R(self):
        """Gap should scale as 1/R."""
        r1 = OSAxioms.check_os2_reflection_positivity(R=1.0)
        r2 = OSAxioms.check_os2_reflection_positivity(R=2.0)
        gap1 = r1['details']['energy_gap_lower_bound']
        gap2 = r2['details']['energy_gap_lower_bound']
        ratio = gap1 / gap2
        assert abs(ratio - 2.0) < 1e-10, \
            f"Gap ratio should be 2.0 (1/R scaling), got {ratio}"

    def test_reference_osterwalder_seiler(self):
        result = OSAxioms.check_os2_reflection_positivity(R=1.0)
        assert '1978' in result['details']['reference']

    def test_with_lattice_data(self):
        """OS2 check should accept lattice data."""
        lattice_data = {'all_positive': True, 'min_value': 0.5}
        result = OSAxioms.check_os2_reflection_positivity(R=1.0, lattice_data=lattice_data)
        assert 'verified' in result['details']['lattice_check_status']


class TestOS3Symmetry:
    """OS3: Gauge invariance."""

    def test_os3_satisfied(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=2)
        assert result['satisfied'] is True

    def test_os3_status_is_theorem(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=2)
        assert result['status'] == 'THEOREM'

    def test_su2_adjoint_dim_3(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=2)
        assert result['details']['dim_adjoint'] == 3

    def test_su3_adjoint_dim_8(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=3)
        assert result['details']['dim_adjoint'] == 8

    def test_wilson_action_invariant(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=2)
        assert result['details']['wilson_action_invariant'] is True

    def test_haar_measure_invariant(self):
        result = OSAxioms.check_os3_symmetry(R=1.0, N=2)
        assert result['details']['haar_measure_invariant'] is True


class TestOS4Clustering:
    """OS4: Clustering = mass gap."""

    def test_os4_satisfied(self):
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert result['satisfied'] is True

    def test_os4_status_is_proposition(self):
        """
        OS4 is a PROPOSITION, not a THEOREM.
        It depends on the Kato-Rellich analysis from Phase 1,
        which is rigorous for the linearized operator with bounded perturbation,
        but the full non-perturbative statement is not yet proven.
        """
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert result['status'] == 'PROPOSITION'

    def test_linearized_gap_is_4(self):
        """Linearized gap = 4/R^2 (coexact spectrum)."""
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert abs(result['details']['gap_linearized'] - 4.0) < 1e-10

    def test_kr_corrected_gap_is_3_52(self):
        """KR-corrected gap = 4 - 0.48 = 3.52 on unit S^3."""
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert abs(result['details']['gap_kr_corrected'] - 3.52) < 1e-10

    def test_gap_positive(self):
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert result['details']['gap_positive'] is True

    def test_gap_positive_for_large_R(self):
        """Gap should remain positive even for large R (on S^3)."""
        result = OSAxioms.check_os4_clustering(R=100.0, N=2)
        assert result['details']['gap_positive'] is True
        assert result['details']['gap_kr_corrected'] > 0

    def test_clustering_rate_equals_mass(self):
        """Clustering rate should equal the mass gap."""
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        mass = result['details']['mass_kr_corrected']
        rate = result['details']['clustering_rate']
        assert abs(mass - rate) < 1e-10

    def test_extends_to_all_N(self):
        """Phase 2 result: gap extends to all N."""
        result = OSAxioms.check_os4_clustering(R=1.0, N=2)
        assert result['details']['extends_to_all_N'] is True

    def test_with_lattice_data(self):
        """OS4 check should accept lattice gap data."""
        gap_data = {'gap_estimate': 0.5, 'gap_positive': True}
        result = OSAxioms.check_os4_clustering(R=1.0, N=2, gap_data=gap_data)
        assert result['details']['lattice_gap'] == 0.5


class TestFullAxiomCheck:
    """Comprehensive axiom check."""

    def test_full_check_returns_all_axioms(self):
        result = OSAxioms.full_axiom_check(R=1.0, N=2)
        assert 'os0' in result
        assert 'os1' in result
        assert 'os2' in result
        assert 'os3' in result
        assert 'os4' in result

    def test_all_axioms_satisfied(self):
        """All axioms should be satisfied (on the lattice)."""
        result = OSAxioms.full_axiom_check(R=1.0, N=2)
        assert result['all_satisfied'] is True

    def test_summary_exists(self):
        result = OSAxioms.full_axiom_check(R=1.0, N=2)
        assert 'summary' in result
        assert len(result['summary']) > 50  # Non-trivial summary

    def test_statuses_honest(self):
        """
        Verify honesty of status labels:
        - OS0, OS1, OS3: THEOREM
        - OS2: contains both THEOREM and OPEN
        - OS4: PROPOSITION
        """
        result = OSAxioms.full_axiom_check(R=1.0, N=2)
        statuses = result['statuses']

        assert statuses['OS0 (Regularity)'] == 'THEOREM'
        assert statuses['OS1 (Covariance)'] == 'THEOREM'
        assert 'THEOREM' in statuses['OS2 (Reflection positivity)']
        assert 'OPEN' in statuses['OS2 (Reflection positivity)']
        assert statuses['OS3 (Gauge invariance)'] == 'THEOREM'
        assert statuses['OS4 (Clustering/mass gap)'] == 'PROPOSITION'

    def test_full_check_su3(self):
        """Full check should work for SU(3) as well."""
        result = OSAxioms.full_axiom_check(R=1.0, N=3)
        assert result['all_satisfied'] is True

    def test_full_check_various_radii(self):
        """Full check should pass for various radii."""
        for R in [0.5, 1.0, 2.2, 5.0]:
            result = OSAxioms.full_axiom_check(R=R, N=2)
            assert result['all_satisfied'] is True, f"Failed for R={R}"


class TestReconstructionTheorem:
    """OS reconstruction theorem applicability."""

    def test_reconstruction_applicable(self):
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert result['applicable'] is True

    def test_hilbert_space_exists(self):
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert result['hilbert_space_exists'] is True

    def test_vacuum_unique(self):
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert result['vacuum_unique'] is True

    def test_mass_gap_positive(self):
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert result['mass_gap_lower_bound'] > 0

    def test_open_problems_listed(self):
        """Open problems should be honestly documented."""
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert len(result['open_problems']) >= 2
        # Should mention continuum limit
        problems_text = ' '.join(result['open_problems'])
        assert 'continuum' in problems_text.lower()

    def test_status_is_contingent(self):
        """Status should be contingent on unproven parts."""
        result = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        assert 'PROPOSITION' in result['status'] or 'contingent' in result['status'].lower()
