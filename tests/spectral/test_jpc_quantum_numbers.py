"""
Tests for J^PC quantum number analysis of coexact 1-forms on S³.

Verifies:
    - SU(2)_L × SU(2)_R representations at each level
    - Angular momentum J content and multiplicities
    - Parity and charge conjugation assignments
    - Multiplicity consistency: Σ(2J+1) = k(k+2) per chirality
    - J = 0 absence (fundamental result)
    - Two-particle glueball composites
    - Consistency with hodge_spectrum.py
"""

import pytest
import numpy as np
from fractions import Fraction
from yang_mills_s3.spectral.jpc_quantum_numbers import JPCAnalysis, _adjoint_dimension
from yang_mills_s3.geometry.hodge_spectrum import HodgeSpectrum


# ======================================================================
# Representations: SU(2)_L × SU(2)_R
# ======================================================================
class TestCoexactRepresentations:
    """Verify the SO(4) = SU(2)_L × SU(2)_R representation content."""

    def test_k1_representations(self):
        """
        k=1: (1, 0) ⊕ (0, 1), dim = 6, eigenvalue = 4.

        j₊ = 1, j₋ = 0.
        dim per chirality = (2×1+1)(2×0+1) = 3×1 = 3.
        Total = 6.
        """
        reps = JPCAnalysis.coexact_representations(1)
        assert len(reps) == 1

        r = reps[0]
        assert r['k'] == 1
        assert r['eigenvalue'] == 4
        assert r['j_plus'] == Fraction(1, 1)
        assert r['j_minus'] == Fraction(0, 1)
        assert r['dim_per_chirality'] == 3
        assert r['dim_total'] == 6

    def test_k2_representations(self):
        """
        k=2: (3/2, 1/2) ⊕ (1/2, 3/2), dim = 16, eigenvalue = 9.

        j₊ = 3/2, j₋ = 1/2.
        dim per chirality = (2×3/2+1)(2×1/2+1) = 4×2 = 8.
        Total = 16.
        """
        reps = JPCAnalysis.coexact_representations(2)
        r = reps[1]  # k=2 is the second entry

        assert r['k'] == 2
        assert r['eigenvalue'] == 9
        assert r['j_plus'] == Fraction(3, 2)
        assert r['j_minus'] == Fraction(1, 2)
        assert r['dim_per_chirality'] == 8
        assert r['dim_total'] == 16

    def test_k3_representations(self):
        """
        k=3: (2, 1) ⊕ (1, 2), dim = 30, eigenvalue = 16.

        j₊ = 2, j₋ = 1.
        dim per chirality = (2×2+1)(2×1+1) = 5×3 = 15.
        Total = 30.
        """
        reps = JPCAnalysis.coexact_representations(3)
        r = reps[2]

        assert r['k'] == 3
        assert r['eigenvalue'] == 16
        assert r['j_plus'] == Fraction(2, 1)
        assert r['j_minus'] == Fraction(1, 1)
        assert r['dim_per_chirality'] == 15
        assert r['dim_total'] == 30

    def test_general_j_difference(self):
        """j₊ - j₋ = 1 at every level (fundamental property)."""
        reps = JPCAnalysis.coexact_representations(20)
        for r in reps:
            assert r['j_plus'] - r['j_minus'] == 1

    def test_general_dimension_formula(self):
        """dim_total = 2k(k+2) at every level."""
        reps = JPCAnalysis.coexact_representations(20)
        for r in reps:
            k = r['k']
            expected = 2 * k * (k + 2)
            assert r['dim_total'] == expected, (
                f"k={k}: dim_total={r['dim_total']}, expected {expected}"
            )

    def test_eigenvalue_formula(self):
        """eigenvalue = (k+1)² at every level (R=1 units)."""
        reps = JPCAnalysis.coexact_representations(15)
        for r in reps:
            k = r['k']
            assert r['eigenvalue'] == (k + 1) ** 2

    def test_k0_raises(self):
        """k_max < 1 should raise."""
        with pytest.raises(ValueError):
            JPCAnalysis.coexact_representations(0)

    def test_rep_chirality_swap(self):
        """rep_plus and rep_minus are swaps of each other."""
        reps = JPCAnalysis.coexact_representations(10)
        for r in reps:
            assert r['rep_plus'] == (r['j_plus'], r['j_minus'])
            assert r['rep_minus'] == (r['j_minus'], r['j_plus'])
            assert r['rep_plus'][0] == r['rep_minus'][1]
            assert r['rep_plus'][1] == r['rep_minus'][0]


# ======================================================================
# Angular momentum J content
# ======================================================================
class TestJContent:
    """Verify angular momentum decomposition under diagonal SU(2)."""

    def test_k1_j_content(self):
        """k=1: J = [1]."""
        assert JPCAnalysis.j_content(1) == [1]

    def test_k2_j_content(self):
        """k=2: J = [1, 2]."""
        assert JPCAnalysis.j_content(2) == [1, 2]

    def test_k3_j_content(self):
        """k=3: J = [1, 2, 3]."""
        assert JPCAnalysis.j_content(3) == [1, 2, 3]

    def test_k5_j_content(self):
        """k=5: J = [1, 2, 3, 4, 5]."""
        assert JPCAnalysis.j_content(5) == [1, 2, 3, 4, 5]

    def test_j0_never_appears(self):
        """CRITICAL: J = 0 never appears at any level."""
        for k in range(1, 101):
            content = JPCAnalysis.j_content(k)
            assert 0 not in content, f"J=0 found at k={k}!"
            assert content[0] == 1, f"Minimum J != 1 at k={k}"

    def test_j_min_always_1(self):
        """Minimum J is always 1."""
        for k in range(1, 50):
            assert min(JPCAnalysis.j_content(k)) == 1

    def test_j_max_equals_k(self):
        """Maximum J is always k."""
        for k in range(1, 50):
            assert max(JPCAnalysis.j_content(k)) == k

    def test_k0_raises(self):
        """k=0 should raise."""
        with pytest.raises(ValueError):
            JPCAnalysis.j_content(0)

    def test_negative_k_raises(self):
        """Negative k should raise."""
        with pytest.raises(ValueError):
            JPCAnalysis.j_content(-1)


# ======================================================================
# Multiplicity consistency
# ======================================================================
class TestMultiplicityConsistency:
    """
    Verify Σ(2J+1) = k(k+2) per chirality and 2k(k+2) total.
    This is the key identity linking representation theory to Hodge theory.
    """

    def test_per_chirality_sum_k1(self):
        """k=1: Σ(2J+1) = 3 = 1×3 = k(k+2)."""
        total = sum(2 * J + 1 for J in JPCAnalysis.j_content(1))
        assert total == 1 * 3

    def test_per_chirality_sum_k2(self):
        """k=2: Σ(2J+1) = 3+5 = 8 = 2×4 = k(k+2)."""
        total = sum(2 * J + 1 for J in JPCAnalysis.j_content(2))
        assert total == 2 * 4

    def test_per_chirality_sum_k3(self):
        """k=3: Σ(2J+1) = 3+5+7 = 15 = 3×5 = k(k+2)."""
        total = sum(2 * J + 1 for J in JPCAnalysis.j_content(3))
        assert total == 3 * 5

    def test_per_chirality_sum_general(self):
        """Σ_{J=1}^{k} (2J+1) = k(k+2) for k=1..50."""
        for k in range(1, 51):
            total = sum(2 * J + 1 for J in JPCAnalysis.j_content(k))
            expected = k * (k + 2)
            assert total == expected, (
                f"k={k}: sum(2J+1) = {total}, expected k(k+2) = {expected}"
            )

    def test_total_multiplicity_matches_hodge(self):
        """
        2 × Σ(2J+1) = 2k(k+2) matches the coexact Hodge multiplicity.
        Cross-check with HodgeSpectrum module.
        """
        hodge_spectrum = HodgeSpectrum.one_form_eigenvalues(3, 1.0, 20,
                                                             mode='coexact')
        for k_idx, (ev, hodge_mult) in enumerate(hodge_spectrum):
            k = k_idx + 1  # k starts at 1
            j_sum = sum(2 * J + 1 for J in JPCAnalysis.j_content(k))
            our_total = 2 * j_sum  # factor 2 for both parities

            assert our_total == hodge_mult, (
                f"k={k}: J^PC total = {our_total}, "
                f"Hodge multiplicity = {hodge_mult}"
            )

    def test_verify_multiplicity_method(self):
        """Test the dedicated verify_multiplicity method."""
        for k in range(1, 20):
            result = JPCAnalysis.verify_multiplicity(k)
            assert result['matches'], (
                f"k={k}: multiplicity mismatch! "
                f"expected={result['expected_total']}, "
                f"computed={result['computed_total']}"
            )
            assert result['expected_total'] == 2 * k * (k + 2)
            assert result['computed_per_chirality'] == k * (k + 2)

    def test_j_multiplicity_method(self):
        """j_multiplicity returns (2J+1) for valid J, 0 otherwise."""
        for k in range(1, 10):
            for J in range(1, k + 1):
                assert JPCAnalysis.j_multiplicity(k, J) == 2 * J + 1
            # J outside range returns 0
            assert JPCAnalysis.j_multiplicity(k, 0) == 0
            assert JPCAnalysis.j_multiplicity(k, k + 1) == 0

    def test_sum_j_multiplicity_matches_per_chirality(self):
        """Sum of j_multiplicity over J = 1..k equals k(k+2)."""
        for k in range(1, 20):
            total = sum(JPCAnalysis.j_multiplicity(k, J) for J in range(0, k + 2))
            assert total == k * (k + 2)


# ======================================================================
# Parity
# ======================================================================
class TestParity:
    """Verify parity assignments from the antipodal map."""

    def test_k1_parity(self):
        """k=1: J=1 with P=+1 and P=-1, two states."""
        parities = JPCAnalysis.parity(1)
        assert len(parities) == 2  # J=1 with P=+1, J=1 with P=-1

        # Check both parities present for J=1
        j1_entries = [p for p in parities if p['J'] == 1]
        assert len(j1_entries) == 2
        p_values = {p['P'] for p in j1_entries}
        assert p_values == {+1, -1}

    def test_k2_parity(self):
        """k=2: J=1,2 each with P=±1, four states."""
        parities = JPCAnalysis.parity(2)
        assert len(parities) == 4  # 2 J values × 2 parities

    def test_k3_parity(self):
        """k=3: J=1,2,3 each with P=±1, six states."""
        parities = JPCAnalysis.parity(3)
        assert len(parities) == 6

    def test_both_parities_at_every_j(self):
        """Both P=+1 and P=-1 appear for every J at every level."""
        for k in range(1, 10):
            parities = JPCAnalysis.parity(k)
            for J in range(1, k + 1):
                j_entries = [p for p in parities if p['J'] == J]
                p_values = {p['P'] for p in j_entries}
                assert p_values == {+1, -1}, (
                    f"k={k}, J={J}: missing parity in {p_values}"
                )

    def test_parity_labels(self):
        """P_label is '+' for P=+1 and '-' for P=-1."""
        parities = JPCAnalysis.parity(3)
        for p in parities:
            if p['P'] == +1:
                assert p['P_label'] == '+'
            else:
                assert p['P_label'] == '-'

    def test_parity_multiplicity(self):
        """Each (J, P) entry has multiplicity (2J+1)."""
        for k in range(1, 10):
            parities = JPCAnalysis.parity(k)
            for p in parities:
                assert p['multiplicity'] == 2 * p['J'] + 1

    def test_total_parity_multiplicity_matches(self):
        """
        Sum of multiplicities over all (J, P) pairs = 2k(k+2).
        This is the total coexact multiplicity at level k.
        """
        for k in range(1, 20):
            parities = JPCAnalysis.parity(k)
            total = sum(p['multiplicity'] for p in parities)
            expected = 2 * k * (k + 2)
            assert total == expected, (
                f"k={k}: parity total = {total}, expected {expected}"
            )


# ======================================================================
# Charge conjugation
# ======================================================================
class TestChargeConjugation:
    """Verify C-parity assignments."""

    def test_su2_c_positive(self):
        """SU(2): adjoint is real, C = +1 for single-particle modes."""
        c_info = JPCAnalysis.charge_conjugation('SU(2)')
        assert c_info['C_single_particle'] == +1
        assert c_info['adjoint_is_real'] is True

    def test_su3_c_positive(self):
        """SU(3): adjoint is also real, C = +1."""
        c_info = JPCAnalysis.charge_conjugation('SU(3)')
        assert c_info['C_single_particle'] == +1
        assert c_info['adjoint_is_real'] is True

    def test_formula_present(self):
        """C formula for multi-particle states is documented."""
        c_info = JPCAnalysis.charge_conjugation('SU(2)')
        assert 'C_formula' in c_info
        assert '(-1)^n' in c_info['C_formula']


# ======================================================================
# Single-particle J^PC table
# ======================================================================
class TestSingleParticleJPCTable:
    """Verify the complete J^PC table for single-particle modes."""

    def test_k1_jpc_entries(self):
        """
        k=1: J=1, P=±1, C=+1 → 1++ and 1-+.
        Two entries at k=1.
        """
        table = JPCAnalysis.single_particle_jpc_table(1)
        k1_entries = [e for e in table if e['k'] == 1]
        assert len(k1_entries) == 2

        labels = {e['JPC_label'] for e in k1_entries}
        assert labels == {'1++', '1-+'}

    def test_k2_jpc_entries(self):
        """
        k=2: J=1,2 × P=±1 → 1++, 1-+, 2++, 2-+.
        Four entries at k=2.
        """
        table = JPCAnalysis.single_particle_jpc_table(2)
        k2_entries = [e for e in table if e['k'] == 2]
        assert len(k2_entries) == 4

        labels = {e['JPC_label'] for e in k2_entries}
        assert labels == {'1++', '1-+', '2++', '2-+'}

    def test_no_j0_in_table(self):
        """CRITICAL: No J=0 entries in the entire table."""
        table = JPCAnalysis.single_particle_jpc_table(20)
        j0_entries = [e for e in table if e['J'] == 0]
        assert len(j0_entries) == 0, (
            f"Found {len(j0_entries)} entries with J=0! "
            "J=0 should NEVER appear in single-particle modes."
        )

    def test_no_0pp_in_table(self):
        """CRITICAL: No 0++ entries in the single-particle table."""
        table = JPCAnalysis.single_particle_jpc_table(20)
        opp_entries = [e for e in table if e['JPC_label'] == '0++']
        assert len(opp_entries) == 0, (
            "Found 0++ in single-particle table! "
            "0++ must be a composite state."
        )

    def test_mass_values(self):
        """mass = (k+1) in R=1 units."""
        table = JPCAnalysis.single_particle_jpc_table(10)
        for e in table:
            assert e['mass'] == e['k'] + 1

    def test_eigenvalue_values(self):
        """eigenvalue = (k+1)² in R=1 units."""
        table = JPCAnalysis.single_particle_jpc_table(10)
        for e in table:
            assert e['eigenvalue'] == (e['k'] + 1) ** 2

    def test_multiplicity_with_adjoint(self):
        """multiplicity = (2J+1) × dim(adj)."""
        for group, dim_adj in [('SU(2)', 3), ('SU(3)', 8)]:
            table = JPCAnalysis.single_particle_jpc_table(5, group)
            for e in table:
                expected = (2 * e['J'] + 1) * dim_adj
                assert e['multiplicity'] == expected, (
                    f"{group}, k={e['k']}, J={e['J']}: "
                    f"mult={e['multiplicity']}, expected {expected}"
                )

    def test_c_always_positive(self):
        """C = +1 for all single-particle modes."""
        table = JPCAnalysis.single_particle_jpc_table(10, 'SU(3)')
        for e in table:
            assert e['C'] == +1

    def test_total_entries_count(self):
        """
        Total number of (k, J, P) entries up to k_max:
        At each k: k values of J × 2 parities = 2k entries.
        Total = Σ_{k=1}^{k_max} 2k = k_max(k_max+1).
        """
        k_max = 10
        table = JPCAnalysis.single_particle_jpc_table(k_max)
        expected = k_max * (k_max + 1)
        assert len(table) == expected


# ======================================================================
# Two-particle glueball composites
# ======================================================================
class TestGlueballComposites:
    """Verify the two-particle composite analysis."""

    def test_0pp_exists_in_composites(self):
        """0++ must appear in the two-particle composites."""
        composites = JPCAnalysis.glueball_composites(1)
        opp = [c for c in composites if c['JPC_label'] == '0++']
        assert len(opp) > 0, "0++ not found in two-particle composites!"

    def test_lightest_0pp_from_k1_k1(self):
        """
        Lightest 0++ comes from two k=1 modes (J1=1, J2=1 → J=0).
        Threshold mass = 2 + 2 = 4 (units 1/R).
        """
        composites = JPCAnalysis.glueball_composites(1)
        opp = [c for c in composites
               if c['JPC_label'] == '0++' and c['k1'] == 1 and c['k2'] == 1]
        assert len(opp) > 0

        # Check threshold
        for c in opp:
            assert c['threshold_mass'] == 4  # 2 + 2

    def test_0pp_note_present(self):
        """0++ entries should have a descriptive note."""
        composites = JPCAnalysis.glueball_composites(1)
        opp = [c for c in composites if c['JPC_label'] == '0++']
        has_note = any(len(c['note']) > 0 for c in opp)
        assert has_note, "0++ entries should have a note about scalar glueball"

    def test_c_parity_two_gluon(self):
        """C = +1 for all two-gluon states (C = (-1)^2 = +1)."""
        composites = JPCAnalysis.glueball_composites(2)
        for c in composites:
            assert c['C_total'] == +1

    def test_threshold_formula(self):
        """Threshold = m_{k1} + m_{k2} = (k1+1) + (k2+1)."""
        composites = JPCAnalysis.glueball_composites(3)
        for c in composites:
            expected = (c['k1'] + 1) + (c['k2'] + 1)
            assert c['threshold_mass'] == expected

    def test_j_total_range(self):
        """J_total must be in |J1-J2|..J1+J2."""
        composites = JPCAnalysis.glueball_composites(3)
        for c in composites:
            J1, J2, J_total = c['J1'], c['J2'], c['J_total']
            assert abs(J1 - J2) <= J_total <= J1 + J2, (
                f"J_total={J_total} out of range for J1={J1}, J2={J2}"
            )

    def test_k1_k1_j_total_values(self):
        """
        Two k=1 modes: J1=1, J2=1 → J_total = 0, 1, 2.
        With P=±1 for each, gives 6 J^PC combinations.
        """
        composites = JPCAnalysis.glueball_composites(1)
        k1k1 = [c for c in composites if c['k1'] == 1 and c['k2'] == 1]

        j_values = sorted(set(c['J_total'] for c in k1k1))
        assert j_values == [0, 1, 2]

    def test_free_theory_degeneracy(self):
        """
        In free theory, all k1=k2=1 composites have the same threshold.
        This means 0++, 1++, 1-+, 2++, 2-+ are ALL degenerate.
        """
        composites = JPCAnalysis.glueball_composites(1)
        k1k1 = [c for c in composites if c['k1'] == 1 and c['k2'] == 1]

        thresholds = set(c['threshold_mass'] for c in k1k1)
        assert len(thresholds) == 1, (
            f"Expected all k1=k2=1 states to be degenerate, "
            f"got thresholds: {thresholds}"
        )
        assert thresholds.pop() == 4


# ======================================================================
# Mass ratio predictions
# ======================================================================
class TestMassRatioPredictions:
    """Verify the corrected mass ratio analysis."""

    def test_single_particle_ratios(self):
        """m_k/m_1 = (k+1)/2 for single-particle modes."""
        preds = JPCAnalysis.mass_ratio_predictions()
        sp = preds['single_particle_ratios']

        for k in range(1, 6):
            key = f"m_{k}/m_1"
            assert key in sp
            assert abs(sp[key]['value'] - (k + 1) / 2.0) < 1e-12

    def test_single_particle_j_min(self):
        """All single-particle ratios have J_min >= 1."""
        preds = JPCAnalysis.mass_ratio_predictions()
        sp = preds['single_particle_ratios']
        for key, val in sp.items():
            assert val['J_min'] >= 1, (
                f"{key}: J_min = {val['J_min']}, expected >= 1"
            )

    def test_two_particle_threshold(self):
        """0++ threshold is 2× the single-particle gap."""
        preds = JPCAnalysis.mass_ratio_predictions()
        tp = preds['two_particle_threshold']
        assert tp['ratio_0pp_to_gap'] == 2.0

    def test_lattice_comparison_honest(self):
        """The lattice comparison acknowledges the discrepancy."""
        preds = JPCAnalysis.mass_ratio_predictions()
        lc = preds['lattice_comparison']

        # Our free-theory prediction for 0++ ratios
        assert lc['our_free_0pp_ratio'] == 1.0  # degenerate

        # Lattice value
        assert abs(lc['lattice_2pp_over_0pp'] - 1.39) < 0.01

    def test_honest_assessment_present(self):
        """There is an honest assessment string."""
        preds = JPCAnalysis.mass_ratio_predictions()
        assert 'honest_assessment' in preds
        assert len(preds['honest_assessment']) > 100  # substantial text

    def test_honest_assessment_mentions_nonperturbative(self):
        """Assessment explicitly mentions non-perturbative effects."""
        preds = JPCAnalysis.mass_ratio_predictions()
        assessment = preds['honest_assessment'].upper()
        assert 'NON-PERTURBATIVE' in assessment or 'NONPERTURBATIVE' in assessment


# ======================================================================
# J = 0 absence proof
# ======================================================================
class TestJ0AbsenceProof:
    """Verify the formal proof that J=0 is absent."""

    def test_proof_is_theorem(self):
        """The proof has THEOREM status."""
        proof = JPCAnalysis.j0_absence_proof()
        assert proof['status'] == 'THEOREM'

    def test_proof_has_steps(self):
        """The proof has at least 3 steps."""
        proof = JPCAnalysis.j0_absence_proof()
        assert len(proof['proof_steps']) >= 3

    def test_proof_mentions_qed(self):
        """The proof concludes with QED."""
        proof = JPCAnalysis.j0_absence_proof()
        last_step = proof['proof_steps'][-1]
        assert 'QED' in last_step.upper() or 'qed' in last_step.lower()

    def test_proof_verified_numerically(self):
        """The proof is verified numerically up to high k."""
        proof = JPCAnalysis.j0_absence_proof()
        assert proof['verified_up_to_k'] >= 50

    def test_physical_consequence_mentioned(self):
        """The physical consequence (0++ is composite) is stated."""
        proof = JPCAnalysis.j0_absence_proof()
        assert '0++' in proof['physical_consequence'] or \
               'composite' in proof['physical_consequence'].lower()


# ======================================================================
# Consistency with hodge_spectrum.py
# ======================================================================
class TestConsistencyWithHodge:
    """Cross-check with the existing HodgeSpectrum module."""

    def test_eigenvalues_match(self):
        """Eigenvalues from JPCAnalysis match HodgeSpectrum."""
        reps = JPCAnalysis.coexact_representations(15)
        hodge = HodgeSpectrum.one_form_eigenvalues(3, 1.0, 15, mode='coexact')

        for r, (ev, mult) in zip(reps, hodge):
            assert abs(r['eigenvalue'] - ev) < 1e-12, (
                f"k={r['k']}: JPC eigenvalue={r['eigenvalue']}, "
                f"Hodge eigenvalue={ev}"
            )

    def test_multiplicities_match(self):
        """Total multiplicities from J^PC match Hodge multiplicities."""
        reps = JPCAnalysis.coexact_representations(15)
        hodge = HodgeSpectrum.one_form_eigenvalues(3, 1.0, 15, mode='coexact')

        for r, (ev, mult) in zip(reps, hodge):
            assert r['dim_total'] == mult, (
                f"k={r['k']}: JPC multiplicity={r['dim_total']}, "
                f"Hodge multiplicity={mult}"
            )

    def test_gap_eigenvalue_is_4(self):
        """The gap eigenvalue is 4/R² at R=1."""
        hodge_gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 1, 1.0,
                                                            mode='coexact')
        reps = JPCAnalysis.coexact_representations(1)
        jpc_gap = reps[0]['eigenvalue']

        assert abs(hodge_gap - 4.0) < 1e-12
        assert abs(jpc_gap - 4.0) < 1e-12

    def test_first_20_levels_consistent(self):
        """
        Full consistency check: for each level k = 1..20,
        verify dim_total from J^PC decomposition equals
        the coexact Hodge multiplicity 2k(k+2).
        """
        hodge = HodgeSpectrum.one_form_eigenvalues(3, 1.0, 20, mode='coexact')

        for k_idx, (ev, hodge_mult) in enumerate(hodge):
            k = k_idx + 1

            # From representations
            reps = JPCAnalysis.coexact_representations(k)
            rep = reps[-1]  # the k-th entry
            assert rep['dim_total'] == hodge_mult

            # From J content sum
            j_sum = sum(2 * J + 1 for J in JPCAnalysis.j_content(k))
            assert 2 * j_sum == hodge_mult

            # From verify_multiplicity
            ver = JPCAnalysis.verify_multiplicity(k)
            assert ver['matches']


# ======================================================================
# Helper: adjoint dimension
# ======================================================================
class TestAdjointDimension:
    """Verify the standalone adjoint dimension helper."""

    def test_su2(self):
        assert _adjoint_dimension('SU(2)') == 3

    def test_su3(self):
        assert _adjoint_dimension('SU(3)') == 8

    def test_su4(self):
        assert _adjoint_dimension('SU(4)') == 15

    def test_so3(self):
        assert _adjoint_dimension('SO(3)') == 3

    def test_g2(self):
        assert _adjoint_dimension('G2') == 14

    def test_e8(self):
        assert _adjoint_dimension('E8') == 248

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _adjoint_dimension('UNKNOWN')


# ======================================================================
# Edge cases and algebraic identities
# ======================================================================
class TestAlgebraicIdentities:
    """
    Verify the algebraic identity:
    Σ_{J=1}^{k} (2J+1) = Σ_{J=0}^{k} (2J+1) - 1 = (k+1)² - 1 = k(k+2)

    This is the fundamental link between the Clebsch-Gordan decomposition
    and the Hodge multiplicity formula.
    """

    def test_sum_formula(self):
        """
        Σ_{J=1}^{k} (2J+1) = k² + 2k = k(k+2).
        Proof: Σ_{J=1}^{k} (2J+1) = 2·k(k+1)/2 + k = k²+k+k = k²+2k.
        """
        for k in range(1, 100):
            s = sum(2 * J + 1 for J in range(1, k + 1))
            assert s == k * (k + 2), f"k={k}: sum={s}, expected {k*(k+2)}"

    def test_full_cg_decomposition(self):
        """
        Verify the Clebsch-Gordan decomposition dimensions match.
        For (j₊, j₋) with j₊ = j₋ + 1:
            dim = (2j₊+1)(2j₋+1) = Σ_{J=1}^{2j₋+1} (2J+1)

        At level k: j₋ = (k-1)/2, j₊ = (k+1)/2
            dim = (k+2)·k = k(k+2) ✓
        """
        for k in range(1, 50):
            j_minus = Fraction(k - 1, 2)
            j_plus = Fraction(k + 1, 2)

            dim_direct = int((2 * j_plus + 1) * (2 * j_minus + 1))
            dim_cg = sum(2 * J + 1 for J in range(1, k + 1))

            assert dim_direct == dim_cg == k * (k + 2), (
                f"k={k}: direct={dim_direct}, CG={dim_cg}, "
                f"expected={k*(k+2)}"
            )

    def test_two_chiralities_give_double(self):
        """
        With both chiralities (j₊,j₋) and (j₋,j₊), each J appears
        with BOTH parities P=±1, so total = 2 × k(k+2) = 2k(k+2).
        """
        for k in range(1, 30):
            parities = JPCAnalysis.parity(k)
            total = sum(p['multiplicity'] for p in parities)
            assert total == 2 * k * (k + 2)
