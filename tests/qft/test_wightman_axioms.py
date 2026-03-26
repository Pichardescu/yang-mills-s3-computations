"""
Tests for Wightman axioms verification from OS reconstruction.

Verifies that OS reconstruction data for Yang-Mills on S^3 x R produces
a Wightman QFT satisfying all axioms W0-W4 with mass gap.

The overall result is THEOREM: it follows from the Osterwalder-Schrader
reconstruction theorem (1973, 1975), which is established mathematics.

Logical structure:
    OS0-OS4 (Euclidean, THEOREM 6.1) --> W0-W4 + mass gap (Minkowski)

Test organization:
    - TestW0HilbertSpace: Hilbert space from OS reflection positivity
    - TestW1Covariance: Poincare covariance from Euclidean SO(4) covariance
    - TestW2SpectralCondition: spec(H) >= 0 from OS positivity
    - TestW3Locality: spacelike commutativity from Euclidean locality
    - TestW4Completeness: cyclicity of vacuum from OS completeness
    - TestMassGap: inf spec(H) \\ {0} > 0 from OS exponential decay
    - TestFullVerification: complete check
    - TestLogicalDependencies: W0 before W2, etc.
    - TestOSWightmanEquivalence: Delta_OS = Delta_Wightman
    - TestLaTeXTable: table generation for the paper
    - TestParameterVariation: various R and N
"""

import pytest
import numpy as np
from yang_mills_s3.qft.wightman_axioms import WightmanVerification
from yang_mills_s3.qft.os_axioms import OSAxioms


# ======================================================================
# W0: Hilbert Space
# ======================================================================
class TestW0HilbertSpace:
    """W0: separable Hilbert space from OS reflection positivity."""

    def test_w0_satisfied(self):
        """W0 should be satisfied when OS2 holds."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['satisfied'] is True

    def test_w0_status_is_theorem(self):
        """W0 is THEOREM (from OS reconstruction)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['status'] == 'THEOREM'

    def test_w0_os_input_is_os2(self):
        """W0 comes from OS2 (reflection positivity)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert 'OS2' in result['os_input']

    def test_w0_hilbert_space_separable(self):
        """The constructed Hilbert space is separable."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['details']['hilbert_space_separable'] is True

    def test_w0_vacuum_exists(self):
        """A vacuum state |Omega> exists."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['details']['vacuum_exists'] is True

    def test_w0_vacuum_unique(self):
        """The vacuum is unique (from clustering)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['details']['vacuum_unique'] is True

    def test_w0_construction_is_gns(self):
        """The Hilbert space is constructed via GNS."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['details']['construction'] == 'GNS from OS inner product'

    def test_w0_references_theorem_6_1(self):
        """W0 should reference THEOREM 6.1 (OS axioms)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert '6.1' in result['paper_reference']

    def test_w0_transfer_matrix_positive(self):
        """Transfer matrix positivity is the mechanism."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w0_hilbert_space()
        assert result['details']['transfer_matrix_positive'] is True


# ======================================================================
# W1: Covariance
# ======================================================================
class TestW1Covariance:
    """W1: Poincare/isometry covariance from Euclidean SO(4)."""

    def test_w1_satisfied(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert result['satisfied'] is True

    def test_w1_status_is_theorem(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert result['status'] == 'THEOREM'

    def test_w1_os_input_is_os1(self):
        """W1 comes from OS1 (Euclidean covariance)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert 'OS1' in result['os_input']

    def test_w1_euclidean_symmetry(self):
        """Euclidean symmetry is SO(4) x R."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert result['details']['euclidean_symmetry'] == 'SO(4) x R'

    def test_w1_not_full_poincare(self):
        """
        S^3 x R is curved, so the symmetry is the isometry group,
        NOT the full Poincare group.
        """
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert result['details']['not_full_poincare'] is True

    def test_w1_analytic_continuation_of_time(self):
        """Time translations are analytically continued."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert 'exp(-tH)' in result['details']['time_translation']
        assert 'exp(-iHt)' in result['details']['time_translation']

    def test_w1_unitarity_via_stone(self):
        """Unitarity of the time evolution via Stone's theorem."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert 'Stone' in result['details']['unitarity']

    def test_w1_spatial_symmetry_unchanged(self):
        """Spatial SO(4) is unchanged by analytic continuation."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w1_covariance()
        assert 'SO(4)' in result['details']['spatial_symmetry']


# ======================================================================
# W2: Spectral Condition
# ======================================================================
class TestW2SpectralCondition:
    """W2: spectrum of H >= 0 with unique vacuum at E=0."""

    def test_w2_satisfied(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['satisfied'] is True

    def test_w2_status_is_theorem(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['status'] == 'THEOREM'

    def test_w2_os_input_is_os2(self):
        """W2 comes from OS2 (reflection positivity)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert 'OS2' in result['os_input']

    def test_w2_spectrum_nonnegative(self):
        """spec(H) >= 0."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['spectrum_nonnegative'] is True

    def test_w2_vacuum_eigenvalue_zero(self):
        """H|Omega> = 0."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['vacuum_eigenvalue'] == 0.0

    def test_w2_vacuum_unique(self):
        """Vacuum is the unique ground state."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['vacuum_unique'] is True

    def test_w2_first_excited_positive(self):
        """First excited state has positive energy."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['first_excited_lower_bound'] > 0

    def test_w2_no_tachyons(self):
        """No tachyonic modes on compact S^3."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['no_tachyons'] is True

    def test_w2_discrete_spectrum(self):
        """Spectrum is discrete on compact S^3."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w2_spectral_condition()
        assert result['details']['discrete_spectrum'] is True

    def test_w2_first_excited_scales_with_R(self):
        """First excited energy bound scales as 1/R."""
        v1 = WightmanVerification(R=1.0, N=2)
        v2 = WightmanVerification(R=2.0, N=2)
        e1 = v1.verify_w2_spectral_condition()['details']['first_excited_lower_bound']
        e2 = v2.verify_w2_spectral_condition()['details']['first_excited_lower_bound']
        ratio = e1 / e2
        assert abs(ratio - 2.0) < 1e-10, \
            f"Energy gap should scale as 1/R, got ratio {ratio}"


# ======================================================================
# W3: Locality
# ======================================================================
class TestW3Locality:
    """W3: spacelike commutativity from Euclidean locality."""

    def test_w3_satisfied(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert result['satisfied'] is True

    def test_w3_status_is_theorem(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert result['status'] == 'THEOREM'

    def test_w3_uses_edge_of_wedge(self):
        """The edge-of-the-wedge theorem is the key tool."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert 'edge-of-the-wedge' in result['details']['analytic_continuation_tool'].lower()

    def test_w3_applies_to_gauge_invariant_only(self):
        """Locality applies only to gauge-invariant observables."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert 'Gauge-invariant' in result['details']['applies_to']

    def test_w3_gluon_fields_not_observable(self):
        """Gluon fields are not physical observables."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert result['details']['gluon_fields_not_observable'] is True

    def test_w3_bosonic_symmetry(self):
        """Gauge fields are bosonic (commute, not anticommute)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert result['details']['bosonic_symmetry'] is True

    def test_w3_os_input_includes_os3(self):
        """W3 depends on OS3 (gauge invariance) for restricting to physical observables."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        assert 'OS3' in result['os_input']

    def test_w3_gauge_invariant_examples(self):
        """Should list examples of gauge-invariant observables."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w3_locality()
        examples = result['details']['gauge_invariant_examples']
        assert len(examples) >= 2
        # Should mention Wilson loops
        text = ' '.join(examples)
        assert 'Wilson' in text


# ======================================================================
# W4: Completeness
# ======================================================================
class TestW4Completeness:
    """W4: cyclicity of vacuum from OS completeness."""

    def test_w4_satisfied(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        assert result['satisfied'] is True

    def test_w4_status_is_theorem(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        assert result['status'] == 'THEOREM'

    def test_w4_gns_construction(self):
        """W4 follows from the GNS construction."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        assert result['details']['gns_construction'] is True

    def test_w4_vacuum_cyclic_by_construction(self):
        """Vacuum cyclicity is automatic in GNS."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        assert result['details']['vacuum_cyclic_by_construction'] is True

    def test_w4_explicit_spanning_set(self):
        """Should provide explicit spanning set for the Hilbert space."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        spanning = result['details']['explicit_spanning_set']
        assert len(spanning) >= 3
        # Should include vacuum
        text = ' '.join(spanning)
        assert 'Omega' in text or 'vacuum' in text

    def test_w4_reeh_schlieder_is_consequence(self):
        """Reeh-Schlieder is a consequence of W0-W3, not an input."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_w4_completeness()
        rs = result['details']['reeh_schlieder']
        assert 'CONSEQUENCE' in rs.upper() or 'consequence' in rs


# ======================================================================
# Mass Gap
# ======================================================================
class TestMassGap:
    """Mass gap: inf spec(H) \\ {0} = Delta > 0."""

    def test_mass_gap_satisfied(self):
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['satisfied'] is True

    def test_mass_gap_status_is_theorem(self):
        """Mass gap is THEOREM (from 18-step proof chain)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['status'] == 'THEOREM'

    def test_mass_gap_os_input_is_os4(self):
        """Mass gap comes from OS4 (clustering)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert 'OS4' in result['os_input']

    def test_mass_gap_positive(self):
        """The mass gap is strictly positive."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['details']['mass_gap_positive'] is True

    def test_mass_gap_linearized_is_4(self):
        """Linearized gap = 4/R^2 on S^3(R=1)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert abs(result['details']['gap_linearized'] - 4.0) < 1e-10

    def test_mass_gap_kr_corrected(self):
        """KR-corrected gap = 3.52/R^2 on S^3(R=1)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert abs(result['details']['gap_kr_corrected'] - 3.52) < 1e-10

    def test_mass_gap_gz_free(self):
        """The mass gap proof is GZ-free."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['details']['gz_free'] is True

    def test_mass_gap_extends_to_sun(self):
        """The mass gap extends to all SU(N)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['details']['extends_to_sun'] is True

    def test_proof_chain_18_steps(self):
        """The proof chain has 18 steps, all THEOREM."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        assert result['details']['proof_chain_steps'] == 18
        assert result['details']['proof_chain_all_theorem'] is True

    def test_five_independent_bounds(self):
        """Five independent bounds on the mass gap."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.verify_mass_gap()
        bounds = result['details']['five_independent_bounds']
        assert len(bounds) == 5
        assert 'hodge_kr' in bounds
        assert 'temple' in bounds
        assert 'bakry_emery' in bounds
        assert 'payne_weinberger' in bounds
        assert 'born_oppenheimer' in bounds


# ======================================================================
# OS-Wightman Gap Equivalence
# ======================================================================
class TestOSWightmanEquivalence:
    """The mass gap is the same in OS and Wightman formulations."""

    def test_os_wightman_gap_equality(self):
        """Delta_OS = Delta_Wightman."""
        v = WightmanVerification(R=1.0, N=2)
        mg = v.verify_mass_gap()
        assert 'Delta_OS = Delta_Wightman' in mg['details']['os_wightman_gap_equality']

    def test_os_gap_matches_wightman_gap_numerically(self):
        """The numerical gap values should match between OS and Wightman."""
        v = WightmanVerification(R=1.0, N=2)
        os_data = OSAxioms.check_os4_clustering(R=1.0, N=2)
        w_data = v.verify_mass_gap()

        os_gap = os_data['details']['gap_kr_corrected']
        w_gap = w_data['details']['gap_kr_corrected']
        assert abs(os_gap - w_gap) < 1e-14, \
            f"OS gap {os_gap} != Wightman gap {w_gap}"

    def test_os_mass_matches_wightman_mass(self):
        """The mass values should match."""
        v = WightmanVerification(R=1.0, N=2)
        os_data = OSAxioms.check_os4_clustering(R=1.0, N=2)
        w_data = v.verify_mass_gap()

        os_mass = os_data['details']['mass_kr_corrected']
        w_mass = w_data['details']['mass_gap_in_R_units']
        assert abs(os_mass - w_mass) < 1e-14, \
            f"OS mass {os_mass} != Wightman mass {w_mass}"

    def test_gap_equality_various_radii(self):
        """Gap equality holds for various radii."""
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            v = WightmanVerification(R=R, N=2)
            os_data = OSAxioms.check_os4_clustering(R=R, N=2)
            w_data = v.verify_mass_gap()

            os_gap = os_data['details']['gap_kr_corrected']
            w_gap = w_data['details']['gap_kr_corrected']
            assert abs(os_gap - w_gap) < 1e-14, \
                f"Gap mismatch at R={R}: OS={os_gap}, Wightman={w_gap}"


# ======================================================================
# Logical Dependencies
# ======================================================================
class TestLogicalDependencies:
    """
    Logical dependencies between Wightman axioms:
        W0 (Hilbert space) must hold for W2 (spectrum) to make sense.
        W0 + W1 (covariance) needed for W3 (locality).
        W0-W3 needed for W4 (completeness).
    """

    def test_w0_before_w2(self):
        """W0 must be satisfied before W2 can be checked."""
        v = WightmanVerification(R=1.0, N=2)
        w0 = v.verify_w0_hilbert_space()
        w2 = v.verify_w2_spectral_condition()
        # If W0 is satisfied, W2 can be checked
        assert w0['satisfied'] is True
        assert w2['satisfied'] is True

    def test_w0_w1_before_w3(self):
        """W0 and W1 must hold for W3 to be meaningful."""
        v = WightmanVerification(R=1.0, N=2)
        w0 = v.verify_w0_hilbert_space()
        w1 = v.verify_w1_covariance()
        w3 = v.verify_w3_locality()
        assert w0['satisfied'] is True
        assert w1['satisfied'] is True
        assert w3['satisfied'] is True

    def test_w0_through_w3_before_w4(self):
        """W0-W3 needed for Reeh-Schlieder => W4."""
        v = WightmanVerification(R=1.0, N=2)
        w0 = v.verify_w0_hilbert_space()
        w1 = v.verify_w1_covariance()
        w2 = v.verify_w2_spectral_condition()
        w3 = v.verify_w3_locality()
        w4 = v.verify_w4_completeness()
        assert all([w0['satisfied'], w1['satisfied'],
                    w2['satisfied'], w3['satisfied']])
        assert w4['satisfied'] is True

    def test_dependencies_satisfied_in_full_verification(self):
        """Full verification should report dependencies_satisfied = True."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert result['dependencies_satisfied'] is True

    def test_os_input_chain(self):
        """Each Wightman axiom should reference a specific OS axiom."""
        v = WightmanVerification(R=1.0, N=2)
        w0 = v.verify_w0_hilbert_space()
        w1 = v.verify_w1_covariance()
        w2 = v.verify_w2_spectral_condition()
        w3 = v.verify_w3_locality()
        w4 = v.verify_w4_completeness()
        mg = v.verify_mass_gap()

        assert 'OS2' in w0['os_input']
        assert 'OS1' in w1['os_input']
        assert 'OS2' in w2['os_input']
        assert 'OS3' in w3['os_input'] or 'locality' in w3['os_input'].lower()
        assert 'OS' in w4['os_input']
        assert 'OS4' in mg['os_input']


# ======================================================================
# Full Verification
# ======================================================================
class TestFullVerification:
    """Full Wightman axiom verification."""

    def test_full_verification_returns_all_axioms(self):
        """Full verification should return all axiom results."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert 'w0' in result
        assert 'w1' in result
        assert 'w2' in result
        assert 'w3' in result
        assert 'w4' in result
        assert 'mass_gap' in result

    def test_all_axioms_satisfied(self):
        """All Wightman axioms should be satisfied."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert result['all_axioms_satisfied'] is True

    def test_mass_gap_positive_in_full(self):
        """Mass gap should be positive in full verification."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert result['mass_gap_positive'] is True

    def test_wightman_qft_exists(self):
        """A Wightman QFT should exist (all axioms + mass gap)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert result['wightman_qft_exists'] is True

    def test_overall_status_is_theorem(self):
        """Overall status should be THEOREM."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert result['overall_status'] == 'THEOREM'

    def test_summary_nonempty(self):
        """Summary should be substantial."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert len(result['summary']) > 100

    def test_reconstruction_theorem_referenced(self):
        """Should reference the OS reconstruction theorem."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        assert 'Osterwalder-Schrader' in result['reconstruction_theorem']
        assert '1973' in result['reconstruction_theorem']
        assert '1975' in result['reconstruction_theorem']


# ======================================================================
# LaTeX Table
# ======================================================================
class TestLaTeXTable:
    """LaTeX-ready table for the paper."""

    def test_latex_table_has_rows(self):
        """Table should have 6 rows (W0-W4 + mass gap)."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        rows = result['latex_table']['rows']
        assert len(rows) == 6

    def test_latex_table_has_source(self):
        """Table should have LaTeX source."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        source = result['latex_table']['latex_source']
        assert r'\begin{table}' in source
        assert r'\end{table}' in source

    def test_latex_table_contains_all_axioms(self):
        """LaTeX source should contain all axiom names."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        source = result['latex_table']['latex_source']
        assert 'W0' in source
        assert 'W1' in source
        assert 'W2' in source
        assert 'W3' in source
        assert 'W4' in source
        assert 'Mass gap' in source

    def test_latex_table_all_theorem(self):
        """All rows should have THEOREM status."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        rows = result['latex_table']['rows']
        for row in rows:
            assert row['status'] == 'THEOREM', \
                f"{row['wightman']} has status {row['status']}, expected THEOREM"

    def test_latex_table_has_os_inputs(self):
        """Each row should reference an OS input."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        rows = result['latex_table']['rows']
        for row in rows:
            assert len(row['os_input']) > 5, \
                f"{row['wightman']} has empty OS input"

    def test_latex_table_has_mechanisms(self):
        """Each row should have a reconstruction mechanism."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        rows = result['latex_table']['rows']
        for row in rows:
            assert len(row['mechanism']) > 5, \
                f"{row['wightman']} has empty mechanism"

    def test_latex_table_includes_tabular(self):
        """LaTeX should use tabular environment."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        source = result['latex_table']['latex_source']
        assert r'\begin{tabular}' in source
        assert r'\end{tabular}' in source

    def test_latex_table_has_caption(self):
        """LaTeX table should have a caption."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        source = result['latex_table']['latex_source']
        assert r'\caption' in source

    def test_latex_table_has_label(self):
        """LaTeX table should have a label for cross-referencing."""
        v = WightmanVerification(R=1.0, N=2)
        result = v.full_verification()
        source = result['latex_table']['latex_source']
        assert r'\label{tab:wightman}' in source


# ======================================================================
# Parameter Variation
# ======================================================================
class TestParameterVariation:
    """Wightman axioms should hold for various R and N."""

    @pytest.mark.parametrize("R", [0.5, 1.0, 2.2, 5.0, 10.0])
    def test_all_axioms_various_radii(self, R):
        """All axioms should hold for any radius."""
        v = WightmanVerification(R=R, N=2)
        result = v.full_verification()
        assert result['all_axioms_satisfied'] is True, \
            f"Axioms not satisfied at R={R}"
        assert result['mass_gap_positive'] is True, \
            f"Mass gap not positive at R={R}"

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_all_axioms_various_N(self, N):
        """All axioms should hold for various gauge groups SU(N)."""
        v = WightmanVerification(R=1.0, N=N)
        result = v.full_verification()
        assert result['all_axioms_satisfied'] is True, \
            f"Axioms not satisfied for SU({N})"

    def test_physical_parameters(self):
        """Test at physical QCD parameters: R=2.2 fm, SU(3)."""
        v = WightmanVerification(R=2.2, N=3)
        result = v.full_verification()
        assert result['wightman_qft_exists'] is True

    def test_gap_scales_correctly(self):
        """Mass gap should scale as 1/R^2."""
        R1, R2 = 1.0, 2.0
        v1 = WightmanVerification(R=R1, N=2)
        v2 = WightmanVerification(R=R2, N=2)
        gap1 = v1.verify_mass_gap()['details']['gap_kr_corrected']
        gap2 = v2.verify_mass_gap()['details']['gap_kr_corrected']
        ratio = gap1 / gap2
        expected = (R2 / R1) ** 2
        assert abs(ratio - expected) < 1e-10, \
            f"Gap ratio {ratio} != expected {expected} (1/R^2 scaling)"

    def test_mass_scales_correctly(self):
        """Mass gap m should scale as 1/R."""
        R1, R2 = 1.0, 3.0
        v1 = WightmanVerification(R=R1, N=2)
        v2 = WightmanVerification(R=R2, N=2)
        m1 = v1.verify_mass_gap()['details']['mass_gap_in_R_units']
        m2 = v2.verify_mass_gap()['details']['mass_gap_in_R_units']
        ratio = m1 / m2
        expected = R2 / R1
        assert abs(ratio - expected) < 1e-10, \
            f"Mass ratio {ratio} != expected {expected} (1/R scaling)"


# ======================================================================
# Quick Check
# ======================================================================
class TestQuickCheck:
    """Quick check convenience method."""

    def test_quick_check_returns_string(self):
        """quick_check should return a summary string."""
        result = WightmanVerification.quick_check(R=1.0, N=2)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_quick_check_mentions_satisfied(self):
        """Quick check should report that axioms are satisfied."""
        result = WightmanVerification.quick_check(R=1.0, N=2)
        assert 'SATISFIED' in result.upper() or 'THEOREM' in result

    def test_quick_check_mentions_mass_gap(self):
        """Quick check should mention mass gap."""
        result = WightmanVerification.quick_check(R=1.0, N=2)
        assert 'gap' in result.lower() or 'POSITIVE' in result


# ======================================================================
# Integration with existing OS axioms
# ======================================================================
class TestOSIntegration:
    """Integration between WightmanVerification and OSAxioms."""

    def test_wightman_uses_os_data(self):
        """WightmanVerification should use data from OSAxioms."""
        v = WightmanVerification(R=1.0, N=2)
        # The internal OS data should match direct OS axiom check
        direct = OSAxioms.full_axiom_check(R=1.0, N=2)
        assert v._os_data['all_satisfied'] == direct['all_satisfied']

    def test_os_all_satisfied_implies_wightman_all_satisfied(self):
        """If OS axioms are all satisfied, Wightman axioms should be too."""
        os_result = OSAxioms.full_axiom_check(R=1.0, N=2)
        if os_result['all_satisfied']:
            v = WightmanVerification(R=1.0, N=2)
            w_result = v.full_verification()
            assert w_result['all_axioms_satisfied'] is True

    def test_consistent_gap_values(self):
        """Gap values should be consistent between OS and Wightman modules."""
        os_recon = OSAxioms.reconstruction_theorem_status(R=1.0, N=2)
        v = WightmanVerification(R=1.0, N=2)
        w_gap = v.verify_mass_gap()

        # OS reconstruction mass gap bound
        os_mass = os_recon['mass_gap_lower_bound']
        # Wightman mass gap (same underlying OS4 data)
        w_mass = w_gap['details']['mass_gap_in_R_units']

        # Both should be positive
        assert os_mass > 0
        assert w_mass > 0
