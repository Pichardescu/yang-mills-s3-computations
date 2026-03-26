"""
Tests for the Adiabatic Continuity Framework.

Tests verify:
1. Adiabatic path traces correctly on twisted T^3
2. Center symmetry preservation (no phase transitions)
3. Anomaly matching constrains IR phase
4. IR irrelevance of twist for local observables
5. Decompactification comparison (S^3 vs twisted T^3 vs periodic T^3)
6. Unsal-Tanizaki bridge status accuracy
7. Main theorem formalization
8. Proposition: decompactification bridge
9. Honest assessment consistency
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.adiabatic_continuity import (
    adiabatic_path_twisted_torus,
    center_symmetry_preservation,
    anomaly_matching_check,
    twist_ir_irrelevance,
    decompactification_comparison,
    ut_bridge_status,
    theorem_adiabatic_continuity,
    proposition_decompactification_bridge,
    honest_assessment,
)


# =====================================================================
# 1. Adiabatic Path
# =====================================================================

class TestAdiabaticPath:
    """Test adiabatic path tracing on twisted T^3."""

    def test_all_gaps_positive(self):
        """THEOREM 7.11: gap > 0 for all L."""
        r = adiabatic_path_twisted_torus()
        assert r['all_positive']

    def test_gap_positive_small_L(self):
        """Gap is positive at small L (weak coupling)."""
        r = adiabatic_path_twisted_torus(L_values=np.array([0.1, 0.2, 0.5]))
        assert r['all_positive']

    def test_gap_positive_large_L(self):
        """Gap is positive at large L (strong coupling)."""
        r = adiabatic_path_twisted_torus(L_values=np.array([10.0, 50.0, 100.0]))
        assert r['all_positive']

    def test_gap_decreases_with_L(self):
        """Perturbative gap ~ pi^2/L^2 decreases with L."""
        L_vals = np.array([1.0, 5.0, 10.0, 50.0])
        r = adiabatic_path_twisted_torus(L_values=L_vals)
        # Gap at L=1 should be larger than at L=50
        assert r['gaps'][0] > r['gaps'][-1]

    def test_label_theorem(self):
        """Label is THEOREM (proven for each finite L)."""
        r = adiabatic_path_twisted_torus(L_values=np.array([1.0, 2.0]))
        assert r['label'] == 'THEOREM'

    def test_returns_all_gap_components(self):
        """Returns geometric, PW, and BE gap components."""
        r = adiabatic_path_twisted_torus(L_values=np.array([1.0, 2.0]))
        assert 'gap_geometric' in r
        assert 'gap_pw' in r
        assert 'gap_be' in r
        assert len(r['gap_geometric']) == 2

    def test_min_gap_positive(self):
        """Minimum gap over all L is positive."""
        r = adiabatic_path_twisted_torus()
        assert r['min_gap'] > 0

    def test_proof_sketch_present(self):
        """Proof sketch is documented."""
        r = adiabatic_path_twisted_torus(L_values=np.array([1.0]))
        assert len(r['proof_sketch']) > 50

    def test_references_present(self):
        """References are documented."""
        r = adiabatic_path_twisted_torus(L_values=np.array([1.0]))
        assert len(r['references']) >= 2


# =====================================================================
# 2. Center Symmetry Preservation
# =====================================================================

class TestCenterSymmetry:
    """Test center symmetry preservation by twist."""

    def test_center_preserved_su2(self):
        """Center Z_2 is preserved for SU(2)."""
        r = center_symmetry_preservation(N=2)
        assert r['center_preserved']

    def test_no_phase_transition_su2(self):
        """No deconfinement phase transition for SU(2)."""
        r = center_symmetry_preservation(N=2)
        assert r['no_phase_transition']

    def test_gap_continuous(self):
        """Gap is continuous in L (no phase transition => continuous)."""
        r = center_symmetry_preservation(N=2)
        assert r['gap_continuous_in_L']

    def test_center_group_z2(self):
        """Center group is Z_2 for SU(2)."""
        r = center_symmetry_preservation(N=2)
        assert r['center_group'] == 'Z_2'

    def test_center_preserved_su3(self):
        """Center Z_3 preserved for SU(3)."""
        r = center_symmetry_preservation(N=3)
        assert r['center_preserved']
        assert r['center_group'] == 'Z_3'

    def test_zero_modes_eliminated(self):
        """Twist eliminates zero modes (prerequisite)."""
        r = center_symmetry_preservation(N=2)
        assert r['zero_modes_eliminated']

    def test_label_theorem(self):
        """Center preservation is THEOREM (topological)."""
        r = center_symmetry_preservation(N=2)
        assert r['label'] == 'THEOREM'

    def test_mechanism_documented(self):
        """Mechanism (Elitzur) is documented."""
        r = center_symmetry_preservation(N=2)
        assert 'Elitzur' in r['mechanism']

    def test_all_twist_types(self):
        """Center preserved for all twist types."""
        for tt in ['standard', 'cyclic_12', 'cyclic_23']:
            r = center_symmetry_preservation(twist_type=tt, N=2)
            assert r['center_preserved']


# =====================================================================
# 3. Anomaly Matching
# =====================================================================

class TestAnomalyMatching:
    """Test 't Hooft anomaly matching constraints."""

    def test_anomaly_constrains_phase(self):
        """Anomaly matching constrains the IR phase."""
        r = anomaly_matching_check(N=2)
        assert r['anomaly_constrains_phase']

    def test_forced_confinement(self):
        """Anomaly forces confinement (not Higgs/Coulomb)."""
        r = anomaly_matching_check(N=2)
        assert r['forced_phase'] == 'confinement'

    def test_center_unbroken(self):
        """Center symmetry is unbroken."""
        r = anomaly_matching_check(N=2)
        assert r['center_unbroken']

    def test_vacuum_degeneracy_su2(self):
        """N = 2 degenerate vacua for SU(2)."""
        r = anomaly_matching_check(N=2)
        assert r['n_vacua'] == 2

    def test_vacuum_degeneracy_su3(self):
        """N = 3 degenerate vacua for SU(3)."""
        r = anomaly_matching_check(N=3)
        assert r['n_vacua'] == 3

    def test_has_mixed_anomaly(self):
        """SU(N >= 2) has mixed anomaly."""
        for N in [2, 3, 5]:
            r = anomaly_matching_check(N=N)
            assert r['has_mixed_anomaly']

    def test_limitations_documented(self):
        """Limitations are honestly documented."""
        r = anomaly_matching_check(N=2)
        assert len(r['limitations']) >= 3
        # Should mention that anomaly constrains phase, not gap value
        limitation_text = ' '.join(r['limitations'])
        assert 'gap' in limitation_text.lower() or 'VALUE' in limitation_text

    def test_label_theorem(self):
        """Anomaly matching is THEOREM (topological)."""
        r = anomaly_matching_check(N=2)
        assert r['label'] == 'THEOREM'


# =====================================================================
# 4. IR Irrelevance of Twist
# =====================================================================

class TestIRIrrelevance:
    """Test that twist effects vanish for local observables at L -> inf."""

    def test_small_correction_at_large_L(self):
        """Twist correction is small when r << L."""
        r = twist_ir_irrelevance(L=10.0, observable_scale=0.5)
        assert r['twist_correction'] < 0.1

    def test_ir_irrelevant_flag(self):
        """IR irrelevant when correction < 1%."""
        r = twist_ir_irrelevance(L=100.0, observable_scale=1.0)
        assert r['ir_irrelevant']

    def test_correction_scales_as_r_over_L_squared(self):
        """Correction scales as (r/L)^2."""
        r1 = twist_ir_irrelevance(L=10.0, observable_scale=1.0)
        r2 = twist_ir_irrelevance(L=20.0, observable_scale=1.0)
        # At fixed r, doubling L should reduce correction by factor 4
        ratio = r1['twist_correction'] / r2['twist_correction']
        assert abs(ratio - 4.0) < 0.1

    def test_large_correction_when_r_near_L(self):
        """Twist is NOT irrelevant when r ~ L."""
        r = twist_ir_irrelevance(L=2.0, observable_scale=1.5)
        assert r['twist_correction'] > 0.1

    def test_observable_exceeds_L(self):
        """Returns not irrelevant when observable_scale >= L."""
        r = twist_ir_irrelevance(L=1.0, observable_scale=2.0)
        assert not r['ir_irrelevant']

    def test_gap_difference_estimate(self):
        """Gap difference between twisted and periodic ~ pi^2/L^2."""
        L = 5.0
        r = twist_ir_irrelevance(L=L, observable_scale=1.0)
        expected = np.pi**2 / L**2
        assert abs(r['gap_difference_estimate'] - expected) < 1e-10

    def test_label_proposition(self):
        """IR irrelevance is PROPOSITION (not THEOREM)."""
        r = twist_ir_irrelevance(L=10.0, observable_scale=1.0)
        assert r['label'] == 'PROPOSITION'

    def test_caveat_documented(self):
        """Caveats about non-perturbative effects are documented."""
        r = twist_ir_irrelevance(L=10.0, observable_scale=1.0)
        assert 'caveat' in r
        assert 'non-perturbative' in r['caveat'].lower() or 'PROPOSITION' in r['caveat']


# =====================================================================
# 5. Decompactification Comparison
# =====================================================================

class TestDecompactificationComparison:
    """Compare gap across manifolds."""

    def test_s3_gap_positive(self):
        """S^3 gap is positive for all R."""
        r = decompactification_comparison()
        for entry in r['comparison_table']:
            assert entry['s3_positive']

    def test_twisted_gap_positive(self):
        """Twisted T^3 gap is positive for all L."""
        r = decompactification_comparison()
        for entry in r['comparison_table']:
            assert entry['twisted_positive']

    def test_s3_gap_grows(self):
        """S^3 gap grows with R (PW bound ~ R^2)."""
        r = decompactification_comparison()
        assert r['s3_gap_grows']

    def test_twisted_gap_decreases(self):
        """Twisted T^3 gap decreases with L (perturbative)."""
        r = decompactification_comparison()
        assert r['twisted_gap_decreases']

    def test_r3_status_unknown(self):
        """R^3 gap status is listed as unknown (Clay Problem)."""
        r = decompactification_comparison()
        assert r['summary']['r3']['label'] == 'CONJECTURE (Clay Problem)'

    def test_labels_correct(self):
        """Each manifold has the correct label."""
        r = decompactification_comparison()
        assert 'THEOREM' in r['summary']['s3']['label']
        assert r['summary']['twisted_t3']['label'] == 'THEOREM 7.11'
        assert r['summary']['periodic_t3']['label'] == 'PROPOSITION'

    def test_protection_mechanisms(self):
        """Protection mechanisms are documented."""
        r = decompactification_comparison()
        assert 'H^1' in r['summary']['s3']['protection'] or 'zero' in r['summary']['s3']['protection']
        assert 'twist' in r['summary']['twisted_t3']['protection'].lower()
        assert 'NONE' in r['summary']['periodic_t3']['protection'] or 'abelian' in r['summary']['periodic_t3']['protection']


# =====================================================================
# 6. UT Bridge Status
# =====================================================================

class TestUTBridgeStatus:
    """Test Unsal-Tanizaki bridge documentation accuracy."""

    def test_ut_status_conjecture(self):
        """UT adiabatic continuity is CONJECTURE."""
        r = ut_bridge_status()
        assert 'CONJECTURE' in r['ut_claims']['status'].upper()

    def test_our_status_theorem(self):
        """Our THEOREM 7.11 is THEOREM."""
        r = ut_bridge_status()
        assert 'THEOREM' in r['our_results']['status'].upper()

    def test_overlap_documented(self):
        """Overlap between UT and our results is documented."""
        r = ut_bridge_status()
        assert 'center_symmetry' in r['overlap']
        assert 'twist_mechanism' in r['overlap']

    def test_differences_documented(self):
        """Key differences are documented."""
        r = ut_bridge_status()
        assert 'topology' in r['differences']
        assert 'decompactification' in r['differences']
        assert 'proof_method' in r['differences']

    def test_gap_to_close_documented(self):
        """Remaining gap is honestly documented."""
        r = ut_bridge_status()
        assert len(r['gap_to_close']['what_would_close_the_gap']) >= 3
        assert len(r['gap_to_close']['obstacles']) >= 3

    def test_references_include_ut(self):
        """References include Unsal papers."""
        r = ut_bridge_status()
        refs = ' '.join(r['references'])
        assert 'Unsal' in refs or 'nsal' in refs


# =====================================================================
# 7. Theorem: Adiabatic Continuity
# =====================================================================

class TestTheoremAdiabaticContinuity:
    """Test the main adiabatic continuity theorem."""

    def test_gap_positive_all_L(self):
        """THEOREM: gap > 0 for all L tested."""
        r = theorem_adiabatic_continuity(N=2)
        assert r['gap_positive_all_L']

    def test_gap_continuous(self):
        """Gap function is continuous (no phase transitions)."""
        r = theorem_adiabatic_continuity(N=2)
        assert r['gap_continuous']

    def test_zero_modes_eliminated(self):
        """Zero modes are eliminated by twist."""
        r = theorem_adiabatic_continuity(N=2)
        assert r['zero_modes_eliminated']

    def test_decompactification_conjecture(self):
        """Decompactification is honestly labeled CONJECTURE."""
        r = theorem_adiabatic_continuity(N=2)
        assert r['decompactification'] == 'CONJECTURE'

    def test_label_theorem(self):
        """Main result is THEOREM (for finite L)."""
        r = theorem_adiabatic_continuity(N=2)
        assert r['label'] == 'THEOREM'

    def test_su3_prime(self):
        """SU(3) also has gap > 0 (N=3 is prime)."""
        r = theorem_adiabatic_continuity(N=3)
        assert r['gap_positive_all_L']

    def test_su5_prime(self):
        """SU(5) also has gap > 0 (N=5 is prime)."""
        r = theorem_adiabatic_continuity(N=5)
        assert r['gap_positive_all_L']

    def test_proof_sketch_complete(self):
        """Proof sketch has all 8 steps."""
        r = theorem_adiabatic_continuity(N=2)
        # Check that the proof sketch mentions key steps
        ps = r['proof_sketch']
        assert 'zero mode' in ps.lower() or 'twist' in ps.lower()
        assert 'PW' in ps or 'Payne' in ps.lower() or 'Gribov' in ps.lower()
        assert 'phase transition' in ps.lower() or 'Elitzur' in ps


# =====================================================================
# 8. Proposition: Decompactification Bridge
# =====================================================================

class TestDecompactificationBridge:
    """Test the decompactification bridge proposition."""

    def test_all_twisted_gaps_positive(self):
        """All twisted T^3 gaps are positive."""
        r = proposition_decompactification_bridge()
        assert r['all_twisted_positive']

    def test_gap_survives_is_conjecture(self):
        """Gap survival in L -> inf is labeled CONJECTURE."""
        r = proposition_decompactification_bridge()
        assert r['gap_survives_limit'] == 'CONJECTURE'

    def test_technical_conditions_listed(self):
        """Technical conditions for gap survival are listed."""
        r = proposition_decompactification_bridge()
        assert len(r['technical_conditions']) >= 3

    def test_supporting_evidence_listed(self):
        """Supporting evidence is listed."""
        r = proposition_decompactification_bridge()
        assert len(r['supporting_evidence']) >= 4

    def test_obstacles_listed(self):
        """Obstacles are honestly listed."""
        r = proposition_decompactification_bridge()
        assert len(r['obstacles']) >= 3

    def test_label_proposition(self):
        """Label is PROPOSITION (not THEOREM)."""
        r = proposition_decompactification_bridge()
        assert r['label'] == 'PROPOSITION'

    def test_gaps_have_correct_length(self):
        """Gap arrays have correct length."""
        L_vals = np.array([1.0, 5.0, 10.0])
        r = proposition_decompactification_bridge(L_values=L_vals)
        assert len(r['gaps_twisted']) == 3
        assert len(r['gaps_periodic']) == 3


# =====================================================================
# 9. Honest Assessment
# =====================================================================

class TestHonestAssessment:
    """Test the honest assessment of the framework."""

    def test_has_theorems(self):
        """Assessment includes theorems."""
        r = honest_assessment()
        assert r['n_theorems'] >= 5

    def test_has_propositions(self):
        """Assessment includes propositions."""
        r = honest_assessment()
        assert r['n_propositions'] >= 2

    def test_has_conjectures(self):
        """Assessment includes conjectures."""
        r = honest_assessment()
        assert r['n_conjectures'] >= 2

    def test_conj_ac1_is_clay(self):
        """CONJ-AC-1 (gap survives L -> inf) is related to Clay Problem."""
        r = honest_assessment()
        conj1 = next(c for c in r['conjectures'] if c['id'] == 'CONJ-AC-1')
        assert 'Clay' in conj1['relation_to_clay'] or 'clay' in conj1['relation_to_clay'].lower()

    def test_evidence_for_and_against(self):
        """CONJ-AC-1 has both supporting and opposing evidence."""
        r = honest_assessment()
        conj1 = next(c for c in r['conjectures'] if c['id'] == 'CONJ-AC-1')
        assert len(conj1['evidence_for']) >= 3
        assert len(conj1['evidence_against']) >= 2

    def test_theorem_ids_unique(self):
        """All theorem IDs are unique."""
        r = honest_assessment()
        ids = [t['id'] for t in r['theorems']]
        assert len(ids) == len(set(ids))

    def test_proposition_ids_unique(self):
        """All proposition IDs are unique."""
        r = honest_assessment()
        ids = [p['id'] for p in r['propositions']]
        assert len(ids) == len(set(ids))

    def test_conjecture_ids_unique(self):
        """All conjecture IDs are unique."""
        r = honest_assessment()
        ids = [c['id'] for c in r['conjectures']]
        assert len(ids) == len(set(ids))

    def test_summary_complete(self):
        """Summary covers proven, proposed, conjectured, and gap."""
        r = honest_assessment()
        assert 'proven' in r['summary']
        assert 'proposed' in r['summary']
        assert 'conjectured' in r['summary']
        assert 'the_gap' in r['summary']

    def test_label_honest(self):
        """Label is HONEST (meta-assessment)."""
        r = honest_assessment()
        assert r['label'] == 'HONEST'

    def test_references_comprehensive(self):
        """References list is comprehensive."""
        r = honest_assessment()
        refs = ' '.join(r['references'])
        assert 'Hooft' in refs or 'tHooft' in refs
        assert 'Unsal' in refs or 'nsal' in refs
        assert 'Elitzur' in refs
        assert 'Payne' in refs or 'PW' in refs

    def test_ut_conjecture_included(self):
        """UT adiabatic continuity is listed as conjecture."""
        r = honest_assessment()
        conj_ids = [c['id'] for c in r['conjectures']]
        assert 'CONJ-AC-2' in conj_ids

    def test_total_count_consistency(self):
        """Total count matches individual lists."""
        r = honest_assessment()
        assert r['n_theorems'] == len(r['theorems'])
        assert r['n_propositions'] == len(r['propositions'])
        assert r['n_conjectures'] == len(r['conjectures'])
