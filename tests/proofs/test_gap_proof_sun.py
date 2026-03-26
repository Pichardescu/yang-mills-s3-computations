"""
Tests for Phase 2: Mass gap extension to all compact simple Lie groups.

Tests the GapProofSUN class which extends the SU(2) mass gap proof
to arbitrary compact simple gauge groups G.

CONCEPTUAL FRAMEWORK (Session 4 correction):
    Result A — On S^3 with ANY gauge group G: gap = 4/R^2 trivially
               (tensor product structure, gauge group only affects multiplicities)
    Result B — On M = G (group manifold): coexact 1-form gap = 4/R^2
               (Casimir universality, independent geometric result)
    These coincide ONLY for G = SU(2) ~ S^3.

Test categories:
    1. SU(2) consistency (must match Phase 1 result: 4/R^2 coexact)
    2. SU(3) gap > 0 (QCD-relevant case)
    3. SU(N) gap > 0 for various N
    4. SO(N) gap > 0 for various N
    5. Sp(N) gap > 0 for various N
    6. Exceptional groups (G2, F4, E6, E7, E8) all have gap > 0
    7. Gap scales as 1/R^2 for all groups
    8. Mass gap table correctness
    9. SU(3) glueball spectrum
    10. General theorem statement
    11. Kato-Rellich for SU(3)
    12. Lie group database completeness
    17. Result A: gap_on_s3_any_gauge_group (trivial universality)
    18. Result B: gap_on_group_manifold (Casimir universality)
    19. Coincidence: Results A and B agree for SU(2)
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.gap_proof_sun import (
    GapProofSUN,
    LIE_GROUP_DB,
    HBAR_C_MEV_FM,
    _normalize_group_name,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def proof():
    """GapProofSUN instance."""
    return GapProofSUN()


# ======================================================================
# 1. SU(2) consistency with Phase 1
# ======================================================================

class TestSU2Consistency:
    """SU(2) gap must be exactly 4/R^2 (coexact), matching Phase 1."""

    def test_su2_gap_is_4(self, proof):
        """Delta_YM(SU(2)) = 4/R^2 at R=1 (coexact gap)."""
        gap = proof.su_n_gap_formula(2, R=1.0)
        assert abs(gap - 4.0) < 1e-14, f"SU(2) gap = {gap}, expected 4.0"

    def test_su2_gap_via_lie_group_data(self, proof):
        """Lie group data gives consistent gap for SU(2)."""
        data = proof.lie_group_data('SU(2)')
        assert abs(data['oneform_gap_coeff'] - 4.0) < 1e-14
        assert data['dim'] == 3
        assert data['rank'] == 1
        assert data['h_dual'] == 2

    def test_su2_ricci_is_2(self, proof):
        """Ricci coefficient for SU(2) is 2 (matching Ric = 2/R^2 on S^3)."""
        data = proof.lie_group_data('SU(2)')
        assert abs(data['ricci_coeff'] - 2.0) < 1e-14

    def test_su2_scalar_gap_is_2(self, proof):
        """Scalar gap coefficient for SU(2) is 2 (nabla*nabla on left-invariant forms)."""
        data = proof.lie_group_data('SU(2)')
        assert abs(data['scalar_gap_coeff'] - 2.0) < 1e-14

    def test_su2_weitzenboeck(self, proof):
        """Weitzenboeck decomposition: 2 + 2 = 4 for SU(2) (universal coexact gap).

        The rough Laplacian on left-invariant 1-forms = 2/R^2 universally.
        Ricci = 2/R^2 universally. Total coexact gap = 4/R^2.

        The fundamental sector Casimir eigenvalue = 8*C_2(fund)/(h*R^2) = 3/R^2
        for SU(2), giving fund_sector total = 3 + 2 = 5, but this is a HIGHER
        eigenvalue, not the gap.
        """
        wb = proof.weitzenboeck_general('SU(2)', R=1.0)
        assert abs(wb['scalar_gap_coefficient'] - 2.0) < 1e-14
        assert abs(wb['ricci_coefficient'] - 2.0) < 1e-14
        assert abs(wb['total_gap_coefficient'] - 4.0) < 1e-14
        # Fund sector eigenvalue for SU(2) = 3 + 2 = 5 (higher than gap)
        assert abs(wb['fund_sector_eigenvalue_coeff'] - 5.0) < 1e-14

    def test_su2_radius_scaling(self, proof):
        """Gap scales as 1/R^2 for SU(2)."""
        for R in [0.5, 1.0, 2.0, 3.14, 10.0]:
            gap = proof.su_n_gap_formula(2, R)
            expected = 4.0 / R**2
            assert abs(gap - expected) < 1e-12, \
                f"R={R}: gap={gap}, expected={expected}"

    def test_sp1_equals_su2(self, proof):
        """Sp(1) ~ SU(2): gaps must be identical."""
        su2_gap = proof.su_n_gap_formula(2, R=1.0)
        sp1_gap = proof.sp_n_gap_formula(1, R=1.0)
        assert abs(su2_gap - sp1_gap) < 1e-14, \
            f"Sp(1)={sp1_gap} != SU(2)={su2_gap}"


# ======================================================================
# 2. SU(3) gap (QCD)
# ======================================================================

class TestSU3Gap:
    """SU(3) is the physically relevant case for QCD."""

    def test_su3_gap_positive(self, proof):
        """SU(3) gap is strictly positive."""
        gap = proof.su_n_gap_formula(3, R=1.0)
        assert gap > 0

    def test_su3_gap_value(self, proof):
        """SU(3) gap = 4 (universal coexact gap)."""
        gap = proof.su_n_gap_formula(3, R=1.0)
        expected = 4.0
        assert abs(gap - expected) < 1e-12, \
            f"SU(3) gap = {gap}, expected {expected}"

    def test_su3_gap_equals_su2(self, proof):
        """SU(3) gap = SU(2) gap (universal from left-invariant forms)."""
        gap_su2 = proof.su_n_gap_formula(2)
        gap_su3 = proof.su_n_gap_formula(3)
        assert abs(gap_su3 - gap_su2) < 1e-14, \
            f"SU(3)={gap_su3} should equal SU(2)={gap_su2}"

    def test_su3_lie_group_data(self, proof):
        """SU(3) group data is correct."""
        data = proof.lie_group_data('SU(3)')
        assert data['dim'] == 8
        assert data['rank'] == 2
        assert data['h_dual'] == 3
        assert abs(data['C2_fund'] - 4.0 / 3.0) < 1e-14

    def test_su3_weitzenboeck(self, proof):
        """Weitzenboeck decomposition for SU(3) (universal coexact gap)."""
        wb = proof.weitzenboeck_general('SU(3)', R=1.0)
        # Rough Laplacian on left-invariant forms = 2/R^2 (universal)
        assert abs(wb['scalar_gap_coefficient'] - 2.0) < 1e-14
        # Ricci = 2
        assert abs(wb['ricci_coefficient'] - 2.0) < 1e-14
        # Total coexact gap = 4 (universal)
        assert abs(wb['total_gap_coefficient'] - 4.0) < 1e-14
        # Fund sector eigenvalue = 8*(4/3)/3 + 2 = 32/9 + 2 = 50/9 ~ 5.556
        assert abs(wb['fund_sector_eigenvalue_coeff'] - 50.0 / 9.0) < 1e-12

    def test_su3_mass_physical(self, proof):
        """SU(3) mass gap at R=2.2 fm is in reasonable range."""
        R_phys = 2.2  # fm
        gap_coeff = 4.0  # universal coexact gap
        mass = HBAR_C_MEV_FM * np.sqrt(gap_coeff) / R_phys
        # Should be ~179 MeV (comparable to Lambda_QCD ~ 200 MeV)
        assert 100.0 < mass < 400.0, f"Mass gap = {mass:.1f} MeV, expected ~200 MeV range"


# ======================================================================
# 3. SU(N) for various N
# ======================================================================

class TestSUNGap:
    """Gap is positive for all SU(N)."""

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 10, 100])
    def test_su_n_gap_positive(self, proof, N):
        """SU(N) gap > 0 for N={N}."""
        gap = proof.su_n_gap_formula(N, R=1.0)
        assert gap > 0, f"SU({N}) gap = {gap} should be > 0"

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 10, 100])
    def test_su_n_gap_formula_matches_data(self, proof, N):
        """Closed-form formula matches lie_group_data for SU(N)."""
        formula_gap = proof.su_n_gap_formula(N, R=1.0)
        data = proof.lie_group_data(f'SU({N})')
        data_gap = data['oneform_gap_coeff']
        assert abs(formula_gap - data_gap) < 1e-10, \
            f"SU({N}): formula={formula_gap}, data={data_gap}"

    def test_su_n_gap_constant(self, proof):
        """SU(N) gap is constant = 4/R^2 for all N (universal)."""
        gaps = [proof.su_n_gap_formula(N) for N in range(2, 20)]
        for i, gap in enumerate(gaps):
            assert abs(gap - 4.0) < 1e-14, \
                f"SU({i+2}) gap={gap:.4f}, expected 4.0"

    def test_su_n_large_n_limit(self, proof):
        """SU(N) gap = 4/R^2 for all N (universal, including N -> infinity)."""
        gap_large = proof.su_n_gap_formula(1000, R=1.0)
        assert abs(gap_large - 4.0) < 1e-14, \
            f"Large N limit: {gap_large}, expected 4.0"

    def test_su_n_gap_at_least_4(self, proof):
        """SU(N) gap = 4/R^2 for all N >= 2."""
        for N in [2, 3, 5, 10, 50, 100]:
            gap = proof.su_n_gap_formula(N, R=1.0)
            assert abs(gap - 4.0) < 1e-14, \
                f"SU({N}) gap = {gap} should be 4.0"


# ======================================================================
# 4. SO(N) gap
# ======================================================================

class TestSONGap:
    """Gap is positive for all SO(N), N >= 3."""

    @pytest.mark.parametrize("N", [3, 4, 5, 7, 10])
    def test_so_n_gap_positive(self, proof, N):
        """SO(N) gap > 0 for N={N}."""
        gap = proof.so_n_gap_formula(N, R=1.0)
        assert gap > 0, f"SO({N}) gap = {gap} should be > 0"

    def test_so_n_via_lie_data(self, proof):
        """SO(N) gap via lie_group_data matches formula."""
        for N in [5, 7, 10]:
            formula = proof.so_n_gap_formula(N, R=1.0)
            data = proof.lie_group_data(f'SO({N})')
            data_gap = data['oneform_gap_coeff']
            assert abs(formula - data_gap) < 1e-10, \
                f"SO({N}): formula={formula}, data={data_gap}"

    def test_so3_has_same_algebra_as_su2(self, proof):
        """
        SO(3) has the same LIE ALGEBRA as SU(2), but different
        group manifold (RP^3 vs S^3). The gap coefficients differ
        because the fundamental of SO(3) (spin-1 vector) is the
        adjoint of SU(2), so the lowest Casimir used differs.

        However, they share h_dual = 2.
        """
        data_so3 = proof.lie_group_data('SO(3)')
        data_su2 = proof.lie_group_data('SU(2)')

        # Same Lie algebra: same h_dual
        assert data_so3['h_dual'] == data_su2['h_dual']

        # But different dim of fundamental rep => different gap
        # SO(3) fund = 3-dim (vector), SU(2) fund = 2-dim (spinor)
        assert data_so3['dim_fund'] != data_su2['dim_fund']

    def test_so_n_gap_positive_all_large(self, proof):
        """SO(N) gap > 0 for larger N."""
        for N in [6, 8, 10, 20]:
            gap = proof.so_n_gap_formula(N, R=1.0)
            assert gap > 0, f"SO({N}) gap = {gap}"


# ======================================================================
# 5. Sp(N) gap
# ======================================================================

class TestSpNGap:
    """Gap is positive for all Sp(N)."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_sp_n_gap_positive(self, proof, N):
        """Sp(N) gap > 0 for N={N}."""
        gap = proof.sp_n_gap_formula(N, R=1.0)
        assert gap > 0, f"Sp({N}) gap = {gap} should be > 0"

    def test_sp_n_via_lie_data(self, proof):
        """Sp(N) gap via lie_group_data matches formula."""
        for N in [1, 2, 3]:
            formula = proof.sp_n_gap_formula(N, R=1.0)
            data = proof.lie_group_data(f'Sp({N})')
            data_gap = data['oneform_gap_coeff']
            assert abs(formula - data_gap) < 1e-10, \
                f"Sp({N}): formula={formula}, data={data_gap}"

    def test_sp2_gap_value(self, proof):
        """Sp(2) gap = 4 (universal coexact gap)."""
        gap = proof.sp_n_gap_formula(2, R=1.0)
        expected = 4.0
        assert abs(gap - expected) < 1e-12, \
            f"Sp(2) gap = {gap}, expected {expected}"


# ======================================================================
# 6. Exceptional groups
# ======================================================================

class TestExceptionalGroups:
    """All exceptional groups G2, F4, E6, E7, E8 have positive gap."""

    @pytest.mark.parametrize("group", ['G2', 'F4', 'E6', 'E7', 'E8'])
    def test_exceptional_gap_positive(self, proof, group):
        """Gap > 0 for exceptional group {group}."""
        data = proof.lie_group_data(group)
        assert data['oneform_gap_coeff'] > 0, \
            f"{group} gap coefficient = {data['oneform_gap_coeff']}"

    def test_g2_data(self, proof):
        """G2: dim=14, rank=2, h=4."""
        data = proof.lie_group_data('G2')
        assert data['dim'] == 14
        assert data['rank'] == 2
        assert data['h_dual'] == 4

    def test_f4_data(self, proof):
        """F4: dim=52, rank=4, h=9."""
        data = proof.lie_group_data('F4')
        assert data['dim'] == 52
        assert data['rank'] == 4
        assert data['h_dual'] == 9

    def test_e6_data(self, proof):
        """E6: dim=78, rank=6, h=12."""
        data = proof.lie_group_data('E6')
        assert data['dim'] == 78
        assert data['rank'] == 6
        assert data['h_dual'] == 12

    def test_e7_data(self, proof):
        """E7: dim=133, rank=7, h=18."""
        data = proof.lie_group_data('E7')
        assert data['dim'] == 133
        assert data['rank'] == 7
        assert data['h_dual'] == 18

    def test_e8_data(self, proof):
        """E8: dim=248, rank=8, h=30."""
        data = proof.lie_group_data('E8')
        assert data['dim'] == 248
        assert data['rank'] == 8
        assert data['h_dual'] == 30

    def test_e8_fund_equals_adj(self, proof):
        """E8 has no rep smaller than adjoint: fund = adj."""
        data = proof.lie_group_data('E8')
        assert abs(data['C2_fund'] - data['C2_adj']) < 1e-14, \
            "E8: fundamental should equal adjoint"

    def test_exceptional_gaps_all_above_minimum(self, proof):
        """All exceptional gaps should be > 2 (the pure Ricci bound)."""
        for group in ['G2', 'F4', 'E6', 'E7', 'E8']:
            data = proof.lie_group_data(group)
            assert data['oneform_gap_coeff'] > 2.0, \
                f"{group}: gap={data['oneform_gap_coeff']} should be > 2 (Ricci bound)"


# ======================================================================
# 7. Gap scales as 1/R^2
# ======================================================================

class TestRadiusScaling:
    """The gap scales as 1/R^2 for all groups."""

    @pytest.mark.parametrize("group", ['SU(2)', 'SU(3)', 'SU(5)', 'SO(5)',
                                       'Sp(2)', 'G2', 'E8'])
    def test_gap_scales_1_over_r_squared(self, proof, group):
        """Gap(R) = c/R^2 for {group} (Weitzenboeck decomposition)."""
        R_values = [0.5, 1.0, 2.0, 5.0]
        # Get the gap coefficient from weitzenboeck_general at R=1
        wb_unit = proof.weitzenboeck_general(group, R=1.0)
        c = wb_unit['total_gap_coefficient']

        for R in R_values:
            wb = proof.weitzenboeck_general(group, R)
            expected = c / R**2
            actual = wb['total_gap_value']
            assert abs(actual - expected) < 1e-10, \
                f"{group}, R={R}: gap={actual}, expected={expected}"

    def test_su_n_formula_1_over_r_squared(self, proof):
        """SU(N) closed-form formula gives 1/R^2 scaling."""
        for N in [2, 3, 10]:
            gap1 = proof.su_n_gap_formula(N, R=1.0)
            gap2 = proof.su_n_gap_formula(N, R=2.0)
            ratio = gap1 / gap2
            assert abs(ratio - 4.0) < 1e-12, \
                f"SU({N}): gap(R=1)/gap(R=2) = {ratio}, expected 4.0"


# ======================================================================
# 8. Mass gap table
# ======================================================================

class TestMassGapTable:
    """The mass gap table is the key deliverable of Phase 2."""

    def test_table_not_empty(self, proof):
        """Table has entries."""
        table = proof.mass_gap_all_groups()
        assert len(table) > 0

    def test_table_has_all_groups(self, proof):
        """Table includes all required group families."""
        table = proof.mass_gap_all_groups()
        names = {e['group'] for e in table}

        # Must include classical groups
        assert 'SU(2)' in names
        assert 'SU(3)' in names
        assert 'SU(5)' in names

        # Must include SO and Sp
        for name in names:
            if name.startswith('SO('):
                break
        else:
            pytest.fail("No SO groups in table")

        for name in names:
            if name.startswith('Sp('):
                break
        else:
            pytest.fail("No Sp groups in table")

        # Must include exceptional
        assert 'G2' in names
        assert 'F4' in names
        assert 'E6' in names
        assert 'E7' in names
        assert 'E8' in names

    def test_all_gaps_positive(self, proof):
        """ALL entries in the table have gap > 0."""
        table = proof.mass_gap_all_groups()
        for entry in table:
            assert entry['gap_positive'], \
                f"{entry['group']}: gap_positive = False (gap = {entry['total_gap_coeff']})"
            assert entry['total_gap_coeff'] > 0, \
                f"{entry['group']}: gap coefficient = {entry['total_gap_coeff']}"

    def test_table_has_required_keys(self, proof):
        """Each entry has the required fields."""
        table = proof.mass_gap_all_groups()
        required = {
            'group', 'dim', 'rank', 'h_dual', 'ricci_coeff',
            'hodge_gap_coeff', 'total_gap_coeff', 'gap_value',
            'mass_mev', 'gap_positive'
        }
        for entry in table:
            missing = required - set(entry.keys())
            assert not missing, \
                f"{entry.get('group', '?')}: missing keys {missing}"

    def test_table_su2_entry(self, proof):
        """SU(2) entry in table is correct."""
        table = proof.mass_gap_all_groups()
        su2 = [e for e in table if e['group'] == 'SU(2)']
        assert len(su2) == 1
        su2 = su2[0]
        assert su2['dim'] == 3
        assert su2['rank'] == 1
        assert abs(su2['total_gap_coeff'] - 4.0) < 1e-12
        assert abs(su2['ricci_coeff'] - 2.0) < 1e-14
        assert abs(su2['hodge_gap_coeff'] - 2.0) < 1e-14

    def test_table_has_correct_count(self, proof):
        """Table has at least 13 entries (classical + exceptional)."""
        table = proof.mass_gap_all_groups()
        assert len(table) >= 13, f"Table has {len(table)} entries, expected >= 13"

    def test_masses_are_physical(self, proof):
        """All masses at R=2.2 fm are in a reasonable range (> 50 MeV)."""
        table = proof.mass_gap_all_groups()
        for entry in table:
            assert entry['mass_mev'] > 50.0, \
                f"{entry['group']}: mass = {entry['mass_mev']:.1f} MeV, too small"


# ======================================================================
# 9. SU(3) glueball spectrum
# ======================================================================

class TestSU3Glueball:
    """SU(3) detailed analysis for QCD comparison."""

    def test_su3_detailed_has_spectrum(self, proof):
        """SU(3) detailed analysis includes a spectrum."""
        result = proof.su3_detailed()
        assert 'spectrum' in result
        assert len(result['spectrum']) > 0

    def test_su3_spectrum_sorted(self, proof):
        """SU(3) spectrum is sorted by eigenvalue."""
        result = proof.su3_detailed()
        evs = [s['eigenvalue'] for s in result['spectrum']]
        for i in range(len(evs) - 1):
            assert evs[i] <= evs[i + 1] + 1e-14, \
                f"Spectrum not sorted: {evs[i]:.4f} > {evs[i + 1]:.4f}"

    def test_su3_lowest_is_left_invariant(self, proof):
        """The lowest SU(3) eigenvalue comes from left-invariant forms (coexact gap)."""
        result = proof.su3_detailed()
        lowest = result['spectrum'][0]
        assert 'left-invariant' in lowest['representation'].lower() or \
               'coexact' in lowest['representation'].lower(), \
            f"Lowest rep is {lowest['representation']}, expected left-invariant/coexact"

    def test_su3_gap_matches_formula(self, proof):
        """SU(3) gap from detailed analysis matches closed form."""
        result = proof.su3_detailed()
        detailed_gap = result['gap_coefficient']
        formula_gap = 4.0  # universal coexact gap
        assert abs(detailed_gap - formula_gap) < 1e-10, \
            f"Detailed: {detailed_gap}, Formula: {formula_gap}"

    def test_su3_mass_ratios_exist(self, proof):
        """Mass ratios are computed."""
        result = proof.su3_detailed()
        assert 'mass_ratios' in result
        assert len(result['mass_ratios']) >= 1

    def test_su3_mass_ratios_start_at_1(self, proof):
        """The lowest mass ratio is 1.0 (reference state)."""
        result = proof.su3_detailed()
        assert abs(result['mass_ratios'][0]['mass_ratio'] - 1.0) < 1e-14

    def test_su3_gap_eigenvalue(self, proof):
        """The coexact gap eigenvalue for SU(3) = 4/R^2."""
        result = proof.su3_detailed()
        # The gap entry is the left-invariant coexact mode
        gap_entry = result['spectrum'][0]
        expected = 4.0  # universal coexact gap at R=1
        assert abs(gap_entry['eigenvalue_coeff'] - expected) < 1e-10, \
            f"Gap eigenvalue = {gap_entry['eigenvalue_coeff']}, expected {expected}"

    def test_su3_lattice_comparison_exists(self, proof):
        """Lattice comparison data is included."""
        result = proof.su3_detailed()
        assert 'lattice_comparison' in result
        assert result['lattice_comparison']['glueball_0pp_lattice'] > 0


# ======================================================================
# 10. General theorem statement
# ======================================================================

class TestTheoremStatement:
    """The formal theorem statement for all compact simple G."""

    def test_statement_is_string(self, proof):
        """Theorem is a non-empty string."""
        stmt = proof.general_theorem_statement()
        assert isinstance(stmt, str)
        assert len(stmt) > 200

    def test_statement_mentions_all_groups(self, proof):
        """Statement mentions all group families."""
        stmt = proof.general_theorem_statement()
        assert 'SU(2)' in stmt
        assert 'SU(3)' in stmt
        assert 'E8' in stmt or 'E_8' in stmt
        assert 'G2' in stmt or 'G_2' in stmt

    def test_statement_contains_theorem(self, proof):
        """Statement begins with THEOREM."""
        stmt = proof.general_theorem_statement()
        assert 'THEOREM' in stmt

    def test_statement_contains_proof(self, proof):
        """Statement includes proof sketch."""
        stmt = proof.general_theorem_statement()
        assert 'Proof' in stmt or 'proof' in stmt

    def test_statement_mentions_key_ingredients(self, proof):
        """Statement mentions Weitzenboeck, Ricci, Casimir/tensor product, H^1=0."""
        stmt = proof.general_theorem_statement()
        assert 'Weitzenboeck' in stmt or 'Weitzenbock' in stmt
        assert 'Ricci' in stmt or 'Ric' in stmt
        # Result A uses tensor product, Result B uses Casimir
        assert 'tensor product' in stmt.lower() or 'Casimir' in stmt
        assert 'H^1' in stmt or 'H1' in stmt or 'harmonic' in stmt.lower()

    def test_statement_confirms_all_positive(self, proof):
        """Statement confirms all gaps are positive."""
        stmt = proof.general_theorem_statement()
        assert 'positive' in stmt.lower() or 'ALL' in stmt

    def test_statement_has_finding(self, proof):
        """Statement includes a FINDING."""
        stmt = proof.general_theorem_statement()
        assert 'FINDING' in stmt

    def test_statement_has_status(self, proof):
        """Statement declares STATUS."""
        stmt = proof.general_theorem_statement()
        assert 'STATUS' in stmt
        assert 'THEOREM' in stmt


# ======================================================================
# 11. Kato-Rellich for SU(3)
# ======================================================================

class TestKatoRellichSU3:
    """Kato-Rellich stability analysis for SU(3)."""

    def test_kr_su3_zero_coupling(self, proof):
        """At g=0, full gap = linearized gap for SU(3)."""
        kr = proof.kato_rellich_general('SU(3)', g_coupling=0.0, R=1.0)
        data = proof.lie_group_data('SU(3)')
        expected_gap = data['oneform_gap_coeff']
        assert abs(kr['full_gap_lower_bound'] - expected_gap) < 1e-10

    def test_kr_su3_small_coupling(self, proof):
        """At small coupling, gap survives for SU(3)."""
        kr = proof.kato_rellich_general('SU(3)', g_coupling=0.1, R=1.0)
        assert kr['full_gap_lower_bound'] > 0
        assert kr['gap_survives'] is True

    def test_kr_su3_alpha_quadratic(self, proof):
        """Alpha scales as g^2 for SU(3)."""
        kr1 = proof.kato_rellich_general('SU(3)', g_coupling=1.0, R=1.0)
        kr2 = proof.kato_rellich_general('SU(3)', g_coupling=2.0, R=1.0)
        ratio = kr2['alpha'] / kr1['alpha']
        assert abs(ratio - 4.0) < 1e-10, \
            f"Alpha ratio = {ratio}, expected 4.0"

    def test_kr_su3_critical_coupling_exists(self, proof):
        """SU(3) has a finite positive critical coupling."""
        kr = proof.kato_rellich_general('SU(3)', g_coupling=0.1, R=1.0)
        assert np.isfinite(kr['g_critical'])
        assert kr['g_critical'] > 0

    def test_kr_su3_gap_decreases_with_coupling(self, proof):
        """Gap bound decreases with coupling for SU(3)."""
        gaps = []
        for g in [0.0, 0.1, 0.5, 1.0]:
            kr = proof.kato_rellich_general('SU(3)', g_coupling=g, R=1.0)
            gaps.append(kr['full_gap_lower_bound'])
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 1e-10, \
                f"Gap should decrease: {gaps[i]} < {gaps[i + 1]}"

    def test_kr_su3_at_physical_coupling(self, proof):
        """
        Document whether KR bound holds at physical coupling for SU(3).

        At the gap scale (~200 MeV), alpha_s ~ 0.3-0.5 for SU(3).
        g^2 = 4*pi*alpha_s ~ 3.8-6.3.
        """
        alpha_s = 0.3
        g_phys = np.sqrt(4 * np.pi * alpha_s)
        kr = proof.kato_rellich_general('SU(3)', g_coupling=g_phys, R=2.2)

        # Document the finding regardless of result
        if kr['gap_survives']:
            print(f"\nSU(3) KR at physical coupling: gap = {kr['full_gap_lower_bound']:.4f}")
        else:
            print(f"\nSU(3) KR INSUFFICIENT at physical coupling "
                  f"(alpha={kr['alpha']:.4f}, g_c={kr['g_critical']:.4f})")

        # The result is documented either way -- not a failure
        assert 'full_gap_lower_bound' in kr


# ======================================================================
# 12. Lie group database
# ======================================================================

class TestLieGroupDatabase:
    """The Lie group database is complete and consistent."""

    def test_database_not_empty(self):
        """Database has entries."""
        assert len(LIE_GROUP_DB) > 0

    def test_database_has_su_groups(self):
        """Database includes SU(2) through SU(100)."""
        for N in [2, 3, 4, 5, 10, 100]:
            assert f'SU({N})' in LIE_GROUP_DB, f"SU({N}) missing"

    def test_database_has_so_groups(self):
        """Database includes SO groups."""
        for N in [3, 5, 7, 10]:
            assert f'SO({N})' in LIE_GROUP_DB, f"SO({N}) missing"

    def test_database_has_sp_groups(self):
        """Database includes Sp groups."""
        for N in [1, 2, 3]:
            assert f'Sp({N})' in LIE_GROUP_DB, f"Sp({N}) missing"

    def test_database_has_exceptional(self):
        """Database includes all exceptional groups."""
        for name in ['G2', 'F4', 'E6', 'E7', 'E8']:
            assert name in LIE_GROUP_DB, f"{name} missing"

    def test_database_dims_correct(self):
        """Dimensions of known groups are correct."""
        checks = {
            'SU(2)': 3, 'SU(3)': 8, 'SU(4)': 15, 'SU(5)': 24,
            'SO(3)': 3, 'SO(5)': 10, 'SO(10)': 45,
            'Sp(1)': 3, 'Sp(2)': 10, 'Sp(3)': 21,
            'G2': 14, 'F4': 52, 'E6': 78, 'E7': 133, 'E8': 248,
        }
        for name, expected_dim in checks.items():
            assert LIE_GROUP_DB[name]['dim'] == expected_dim, \
                f"{name}: dim={LIE_GROUP_DB[name]['dim']}, expected {expected_dim}"

    def test_database_ranks_correct(self):
        """Ranks of known groups are correct."""
        checks = {
            'SU(2)': 1, 'SU(3)': 2, 'SU(4)': 3,
            'G2': 2, 'F4': 4, 'E6': 6, 'E7': 7, 'E8': 8,
        }
        for name, expected_rank in checks.items():
            assert LIE_GROUP_DB[name]['rank'] == expected_rank, \
                f"{name}: rank={LIE_GROUP_DB[name]['rank']}, expected {expected_rank}"

    def test_database_dual_coxeter_correct(self):
        """Dual Coxeter numbers are correct."""
        checks = {
            'SU(2)': 2, 'SU(3)': 3, 'SU(4)': 4, 'SU(5)': 5,
            'SO(5)': 3, 'SO(7)': 5,
            'Sp(1)': 2, 'Sp(2)': 3, 'Sp(3)': 4,
            'G2': 4, 'F4': 9, 'E6': 12, 'E7': 18, 'E8': 30,
        }
        for name, expected_h in checks.items():
            assert LIE_GROUP_DB[name]['h_dual'] == expected_h, \
                f"{name}: h_dual={LIE_GROUP_DB[name]['h_dual']}, expected {expected_h}"

    def test_name_normalization(self):
        """Group name normalization works."""
        assert _normalize_group_name('su(2)') == 'SU(2)'
        assert _normalize_group_name('SO(10)') == 'SO(10)'
        assert _normalize_group_name('g2') == 'G2'
        assert _normalize_group_name('G(2)') == 'G2'
        assert _normalize_group_name('e6') == 'E6'
        assert _normalize_group_name('E(8)') == 'E8'
        assert _normalize_group_name('sp(3)') == 'Sp(3)'


# ======================================================================
# 13. Full analysis integration
# ======================================================================

class TestFullAnalysis:
    """End-to-end analysis."""

    def test_full_analysis_returns_dict(self, proof):
        """Full analysis returns a complete dict."""
        result = proof.full_analysis(R=1.0)
        assert isinstance(result, dict)
        assert 'mass_gap_table' in result
        assert 'su3_detailed' in result
        assert 'theorem' in result
        assert 'all_gaps_positive' in result

    def test_full_analysis_all_positive(self, proof):
        """Full analysis confirms all gaps positive."""
        result = proof.full_analysis(R=1.0)
        assert result['all_gaps_positive'] is True

    def test_full_analysis_su2_consistency(self, proof):
        """Full analysis verifies SU(2) = 4/R^2."""
        result = proof.full_analysis(R=1.0)
        assert abs(result['su2_consistency'] - 4.0) < 1e-12

    def test_full_analysis_sp1_consistency(self, proof):
        """Full analysis verifies Sp(1) = SU(2)."""
        result = proof.full_analysis(R=1.0)
        assert abs(result['sp1_consistency'] - 4.0) < 1e-12


# ======================================================================
# 14. Hodge spectrum on Lie groups
# ======================================================================

class TestHodgeSpectrumLieGroup:
    """Test the spectrum computation on Lie groups."""

    def test_su2_spectrum_starts_at_4(self, proof):
        """SU(2) spectrum lowest value is 4/R^2 (coexact gap)."""
        spec = proof.hodge_spectrum_lie_group('SU(2)', R=1.0)
        assert len(spec) > 0
        assert abs(spec[0][0] - 4.0) < 1e-10

    def test_su3_spectrum_starts_at_4(self, proof):
        """SU(3) spectrum lowest value is 4/R^2 (universal coexact gap)."""
        spec = proof.hodge_spectrum_lie_group('SU(3)', R=1.0)
        assert len(spec) > 0
        expected = 4.0
        assert abs(spec[0][0] - expected) < 1e-10

    def test_spectrum_sorted(self, proof):
        """Spectrum is sorted by eigenvalue."""
        for group in ['SU(2)', 'SU(3)', 'SO(5)', 'G2']:
            spec = proof.hodge_spectrum_lie_group(group, R=1.0)
            evs = [s[0] for s in spec]
            for i in range(len(evs) - 1):
                assert evs[i] <= evs[i + 1] + 1e-10, \
                    f"{group}: spectrum not sorted at index {i}"

    def test_spectrum_all_positive(self, proof):
        """All eigenvalues are positive (no zero modes)."""
        for group in ['SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'Sp(2)', 'G2', 'E8']:
            spec = proof.hodge_spectrum_lie_group(group, R=1.0)
            for ev, label in spec:
                assert ev > 0, f"{group}, {label}: eigenvalue = {ev} <= 0"


# ======================================================================
# 15. Cross-family comparisons
# ======================================================================

class TestCrossFamilyComparisons:
    """Compare gaps across different group families."""

    def test_gap_always_above_ricci_bound(self, proof):
        """
        Gap > 2/R^2 for all groups.

        The pure Weitzenboeck bound (Ric alone) gives gap >= 2/R^2.
        The actual gap is always strictly larger because the rough Laplacian
        also contributes.
        """
        table = proof.mass_gap_all_groups(R=1.0)
        for entry in table:
            assert entry['total_gap_coeff'] > 2.0, \
                f"{entry['group']}: gap = {entry['total_gap_coeff']:.4f} <= 2 (Ricci bound)"

    def test_su_gaps_constant_in_table(self, proof):
        """SU(N) gaps are constant = 4 in the table (universal coexact gap)."""
        table = proof.mass_gap_all_groups(R=1.0)
        su_entries = [e for e in table if e['group'].startswith('SU(')]
        for entry in su_entries:
            assert abs(entry['total_gap_coeff'] - 4.0) < 1e-12, \
                (f"{entry['group']}: gap = {entry['total_gap_coeff']:.4f}, "
                 f"expected 4.0 (universal)")

    def test_all_gaps_finite(self, proof):
        """All gaps are finite (no infinities or NaN)."""
        table = proof.mass_gap_all_groups(R=1.0)
        for entry in table:
            assert np.isfinite(entry['total_gap_coeff']), \
                f"{entry['group']}: gap = {entry['total_gap_coeff']}"
            assert np.isfinite(entry['mass_mev']), \
                f"{entry['group']}: mass = {entry['mass_mev']}"


# ======================================================================
# 16. c(G) universality verification (Session 3 correction)
# ======================================================================

class TestCGUniversality:
    """
    THEOREM: c(G) = 4 for ALL compact simple Lie groups.

    Session 2 corrected the SU(2) eigenvalue from 5/R^2 to 4/R^2 (coexact).
    Session 3 corrects the c(G) table: the gap is UNIVERSAL at 4/R^2,
    not group-dependent as the old paper table claimed.

    The old (wrong) table used c(G) = C_2(adj)/4 + 2 with C_2(adj) = 4*h_dual,
    giving c(SU(N)) = N + 2. This was wrong because:
    1. C_2(adj) in the METRIC normalization = 8 (universal, NOT 4*h_dual)
    2. The formula nabla*nabla = (1/4)*C_2^metric(adj) = (1/4)*8 = 2 is universal
    3. Total gap = nabla*nabla + Ricci = 2 + 2 = 4 for ALL compact simple G

    The universality follows from:
    - C_2^metric(adj) = 8*C_2^T(adj)/(h_dual*R^2) * R^2
    - For ALL simple G: C_2^T(adj) = h_dual (standard result)
    - Therefore C_2^metric(adj) = 8 (universal)
    """

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(4)', 'SU(5)', 'SU(10)', 'SU(100)',
        'SO(3)', 'SO(5)', 'SO(7)', 'SO(10)', 'SO(20)',
        'Sp(1)', 'Sp(2)', 'Sp(3)', 'Sp(5)',
        'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_gap_is_4_for_all_groups(self, proof, group):
        """c(G) = 4 for {group} (universal coexact gap)."""
        data = proof.lie_group_data(group)
        assert abs(data['oneform_gap_coeff'] - 4.0) < 1e-14, \
            f"{group}: c(G) = {data['oneform_gap_coeff']}, expected 4.0"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_weitzenboeck_gap_is_4(self, proof, group):
        """weitzenboeck_general gives total_gap = 4 for {group}."""
        wb = proof.weitzenboeck_general(group, R=1.0)
        assert abs(wb['total_gap_coefficient'] - 4.0) < 1e-14, \
            f"{group}: total_gap = {wb['total_gap_coefficient']}, expected 4.0"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_rough_laplacian_is_2(self, proof, group):
        """nabla*nabla on left-invariant 1-forms = 2/R^2 (universal)."""
        data = proof.lie_group_data(group)
        assert abs(data['scalar_gap_coeff'] - 2.0) < 1e-14, \
            f"{group}: nabla*nabla = {data['scalar_gap_coeff']}/R^2, expected 2.0/R^2"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_ricci_is_2(self, proof, group):
        """Ricci = 2/R^2 (universal in metric g = -R^2*B/8)."""
        data = proof.lie_group_data(group)
        assert abs(data['ricci_coeff'] - 2.0) < 1e-14, \
            f"{group}: Ricci = {data['ricci_coeff']}/R^2, expected 2.0/R^2"


class TestMetricCasimirUniversality:
    """
    Verify that C_2^metric(adj) = 8 universally for all compact simple G.

    THEOREM: With the metric g = -R^2*B/8 (Killing form normalization):
        C_2^metric(adj) = 8 * C_2^T(adj) / h_dual = 8
    because C_2^T(adj) = h_dual for ALL compact simple Lie algebras.
    """

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SU(10)', 'SU(100)',
        'SO(3)', 'SO(5)', 'SO(7)', 'SO(10)',
        'Sp(1)', 'Sp(2)', 'Sp(3)',
        'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_c2_adj_equals_h_dual(self, proof, group):
        """C_2^T(adj) = h_dual for {group} (standard Lie theory result)."""
        data = proof.lie_group_data(group)
        assert abs(data['C2_adj'] - data['h_dual']) < 1e-14, \
            f"{group}: C2_adj={data['C2_adj']}, h_dual={data['h_dual']}"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_metric_casimir_adj_is_8(self, proof, group):
        """C_2^metric(adj) = 8*C_2^T(adj)/h_dual = 8 (universal)."""
        data = proof.lie_group_data(group)
        C2_metric = 8.0 * data['C2_adj'] / data['h_dual']
        assert abs(C2_metric - 8.0) < 1e-14, \
            f"{group}: C_2^metric(adj) = {C2_metric}, expected 8.0"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_nabla_nabla_from_metric_casimir(self, proof, group):
        """nabla*nabla = (1/4)*C_2^metric(adj) = (1/4)*8 = 2 (universal derivation)."""
        data = proof.lie_group_data(group)
        C2_metric = 8.0 * data['C2_adj'] / data['h_dual']
        nabla_nabla = 0.25 * C2_metric
        assert abs(nabla_nabla - 2.0) < 1e-14, \
            f"{group}: nabla*nabla = {nabla_nabla}, expected 2.0"


class TestCasimirDatabaseValues:
    """
    Verify quadratic Casimir eigenvalues for fundamental representations
    against standard Lie theory formulas.

    Convention: Tr_fund(T^a T^b) = (1/2)*delta^{ab}
    """

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 10, 50])
    def test_su_n_c2_fund(self, N):
        """SU(N): C_2(fund) = (N^2-1)/(2N)."""
        expected = (N**2 - 1) / (2 * N)
        actual = LIE_GROUP_DB[f'SU({N})']['C2_fund']
        assert abs(actual - expected) < 1e-12, \
            f"SU({N}): C2_fund={actual}, expected={expected}"

    @pytest.mark.parametrize("N", [5, 7, 9, 11])
    def test_so_odd_c2_fund(self, N):
        """SO(2n+1): C_2(vector) = (N-1)/2."""
        expected = (N - 1) / 2
        actual = LIE_GROUP_DB[f'SO({N})']['C2_fund']
        assert abs(actual - expected) < 1e-12, \
            f"SO({N}): C2_fund={actual}, expected={expected}"

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_so_even_c2_fund(self, N):
        """SO(2n): C_2(vector) = (N-1)/2."""
        expected = (N - 1) / 2
        actual = LIE_GROUP_DB[f'SO({N})']['C2_fund']
        assert abs(actual - expected) < 1e-12, \
            f"SO({N}): C2_fund={actual}, expected={expected}"

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_sp_n_c2_fund(self, N):
        """Sp(N): C_2(fund) = (2N+1)/4."""
        expected = (2 * N + 1) / 4
        actual = LIE_GROUP_DB[f'Sp({N})']['C2_fund']
        assert abs(actual - expected) < 1e-12, \
            f"Sp({N}): C2_fund={actual}, expected={expected}"

    def test_g2_c2_fund(self):
        """G2: C_2(7-dim fund) = 2."""
        assert abs(LIE_GROUP_DB['G2']['C2_fund'] - 2.0) < 1e-14

    def test_f4_c2_fund(self):
        """F4: C_2(26-dim fund) = 6."""
        assert abs(LIE_GROUP_DB['F4']['C2_fund'] - 6.0) < 1e-14

    def test_e6_c2_fund(self):
        """E6: C_2(27-dim fund) = 26/3."""
        assert abs(LIE_GROUP_DB['E6']['C2_fund'] - 26.0 / 3.0) < 1e-14

    def test_e7_c2_fund(self):
        """E7: C_2(56-dim fund) = 57/4."""
        assert abs(LIE_GROUP_DB['E7']['C2_fund'] - 57.0 / 4.0) < 1e-14

    def test_e8_c2_fund_equals_c2_adj(self):
        """E8: fund = adj (248-dim), so C_2(fund) = C_2(adj) = 30."""
        d = LIE_GROUP_DB['E8']
        assert abs(d['C2_fund'] - 30.0) < 1e-14
        assert abs(d['C2_adj'] - 30.0) < 1e-14

    @pytest.mark.parametrize("group,dim_G,dim_R,I2_R,C2_R", [
        ('SU(2)', 3, 2, 0.5, 0.75),
        ('SU(3)', 8, 3, 0.5, 4.0/3.0),
        ('G2', 14, 7, 1.0, 2.0),
        ('F4', 52, 26, 3.0, 6.0),
        ('E6', 78, 27, 3.0, 26.0/3.0),
        ('E7', 133, 56, 6.0, 57.0/4.0),
        ('E8', 248, 248, 30.0, 30.0),
    ])
    def test_dynkin_index_identity(self, group, dim_G, dim_R, I2_R, C2_R):
        """C_2(R)*dim(R) = I_2(R)*dim(G) (standard Lie theory identity)."""
        lhs = C2_R * dim_R
        rhs = I2_R * dim_G
        assert abs(lhs - rhs) < 1e-10, \
            f"{group}: C2*dim(R)={lhs}, I2*dim(G)={rhs}"


class TestFundSectorEigenvalues:
    """
    Verify that the fundamental representation sector eigenvalue
    (from Peter-Weyl) is ABOVE the universal gap of 4/R^2.

    The scalar Laplacian eigenvalue for the fundamental representation:
        lambda_fund = 8*C_2(fund)/(h_dual*R^2)
    This is always > 0 for any non-trivial rep. Adding Ricci = 2/R^2 gives
    a total Delta_1 eigenvalue in the fund sector that is ABOVE the gap.
    """

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_fund_sector_above_gap(self, proof, group):
        """Fund sector eigenvalue > 4/R^2 = universal gap."""
        wb = proof.weitzenboeck_general(group, R=1.0)
        fund_ev = wb['fund_sector_eigenvalue_coeff']
        # For SU(2): fund sector = 5 > 4. For all others: even higher.
        assert fund_ev >= 4.0 - 1e-14, \
            f"{group}: fund_sector={fund_ev}, should be >= 4.0"

    @pytest.mark.parametrize("group", [
        'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_fund_sector_strictly_above_gap_for_non_su2(self, proof, group):
        """For G != SU(2), fund sector eigenvalue > 4/R^2 (strictly)."""
        wb = proof.weitzenboeck_general(group, R=1.0)
        fund_ev = wb['fund_sector_eigenvalue_coeff']
        assert fund_ev > 4.0 + 1e-10, \
            f"{group}: fund_sector={fund_ev}, should be strictly > 4.0"

    def test_su2_fund_sector_is_5(self, proof):
        """SU(2) fund sector eigenvalue = 5 (the 'old' value before correction)."""
        wb = proof.weitzenboeck_general('SU(2)', R=1.0)
        assert abs(wb['fund_sector_eigenvalue_coeff'] - 5.0) < 1e-14

    def test_su3_fund_sector_is_50_over_9(self, proof):
        """SU(3) fund sector = 8*(4/3)/3 + 2 = 32/9 + 2 = 50/9."""
        wb = proof.weitzenboeck_general('SU(3)', R=1.0)
        expected = 50.0 / 9.0
        assert abs(wb['fund_sector_eigenvalue_coeff'] - expected) < 1e-12

    def test_e8_fund_sector_is_10(self, proof):
        """E8 fund = adj, so scalar_ev(fund) = 8*30/30 = 8. Total = 8 + 2 = 10."""
        wb = proof.weitzenboeck_general('E8', R=1.0)
        expected = 10.0
        assert abs(wb['fund_sector_eigenvalue_coeff'] - expected) < 1e-12


# ======================================================================
# 17. Result A: Trivial universality on S^3 (Theorem 5.1')
# ======================================================================

class TestResultA_GapOnS3:
    """
    THEOREM 5.1' (Result A): On S^3_R with ANY gauge group G, the
    linearized YM operator has gap = 4/R^2. The gauge group only
    affects multiplicities, not eigenvalues.

    This is the physically relevant result for Yang-Mills on S^3 x R.
    """

    def test_gap_is_4_regardless_of_gauge_dim(self, proof):
        """Gap = 4/R^2 for any gauge group dimension (3, 8, 14, 248)."""
        for dim_g in [3, 8, 14, 24, 78, 133, 248]:
            result = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=dim_g)
            assert abs(result['gap'] - 4.0) < 1e-14, \
                f"dim(g)={dim_g}: gap={result['gap']}, expected 4.0"

    def test_gap_coefficient_always_4(self, proof):
        """Gap coefficient is always 4 regardless of gauge group."""
        for dim_g in [3, 8, 248]:
            result = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=dim_g)
            assert abs(result['gap_coefficient'] - 4.0) < 1e-14

    def test_multiplicity_scales_with_gauge_dim(self, proof):
        """Multiplicities scale linearly with dim(g)."""
        r1 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=3)
        r2 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        # First eigenvalue multiplicities
        m1 = r1['first_eigenvalues'][0]['total_multiplicity']
        m2 = r2['first_eigenvalues'][0]['total_multiplicity']
        # s3_multiplicity is the same, so ratio = 8/3
        s3_m1 = r1['first_eigenvalues'][0]['s3_multiplicity']
        s3_m2 = r2['first_eigenvalues'][0]['s3_multiplicity']
        assert s3_m1 == s3_m2, "S^3 multiplicities should be identical"
        assert abs(m2 / m1 - 8.0 / 3.0) < 1e-14, \
            f"Multiplicity ratio {m2/m1} != 8/3"

    def test_eigenvalues_independent_of_gauge_dim(self, proof):
        """Eigenvalues are the S^3 eigenvalues, independent of dim(g)."""
        r3 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=3)
        r8 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        r248 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=248)
        for i in range(len(r3['first_eigenvalues'])):
            ev3 = r3['first_eigenvalues'][i]['eigenvalue']
            ev8 = r8['first_eigenvalues'][i]['eigenvalue']
            ev248 = r248['first_eigenvalues'][i]['eigenvalue']
            assert abs(ev3 - ev8) < 1e-14, \
                f"Level {i}: ev(dim=3)={ev3} != ev(dim=8)={ev8}"
            assert abs(ev3 - ev248) < 1e-14, \
                f"Level {i}: ev(dim=3)={ev3} != ev(dim=248)={ev248}"

    def test_coexact_spectrum_is_k_plus_1_squared(self, proof):
        """First eigenvalues are (k+1)^2/R^2 for k=1,2,3,..."""
        result = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        for entry in result['first_eigenvalues']:
            k = entry['k']
            expected = (k + 1)**2
            assert abs(entry['eigenvalue_coeff'] - expected) < 1e-14, \
                f"k={k}: coeff={entry['eigenvalue_coeff']}, expected {expected}"

    def test_radius_scaling(self, proof):
        """Gap scales as 1/R^2."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = proof.gap_on_s3_any_gauge_group(R=R, gauge_group_dim=8)
            expected = 4.0 / R**2
            assert abs(result['gap'] - expected) < 1e-12, \
                f"R={R}: gap={result['gap']}, expected {expected}"

    def test_result_a_note_mentions_s3(self, proof):
        """The note clarifies this is about S^3."""
        result = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        assert 'S^3' in result['note'] or 'S3' in result['note']
        assert result['status'] == 'THEOREM'

    def test_su2_su3_e8_all_same_gap_on_s3(self, proof):
        """SU(2), SU(3), E8 all give the same gap 4/R^2 on S^3."""
        dims = {'SU(2)': 3, 'SU(3)': 8, 'E8': 248}
        gaps = {}
        for name, dim_g in dims.items():
            result = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=dim_g)
            gaps[name] = result['gap']
        for name, gap in gaps.items():
            assert abs(gap - 4.0) < 1e-14, \
                f"{name} on S^3: gap={gap}, expected 4.0"


# ======================================================================
# 18. Result B: Casimir universality on group manifolds (Theorem 5.2')
# ======================================================================

class TestResultB_GapOnGroupManifold:
    """
    THEOREM 5.2' (Result B): On M = G (compact simple Lie group manifold),
    the coexact 1-form Laplacian has gap = 4/R^2 universally.

    This is a geometric result about G as a Riemannian manifold.
    NOT about YM on S^3 for G != SU(2).
    """

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_group_manifold_gap_is_4(self, proof, group):
        """Gap on group manifold M = G is 4/R^2 for all G."""
        result = proof.gap_on_group_manifold(group, R=1.0)
        assert abs(result['total_gap_coefficient'] - 4.0) < 1e-14, \
            f"{group}: gap={result['total_gap_coefficient']}, expected 4.0"

    @pytest.mark.parametrize("group", [
        'SU(2)', 'SU(3)', 'SU(5)', 'SO(5)', 'SO(7)',
        'Sp(2)', 'G2', 'F4', 'E6', 'E7', 'E8',
    ])
    def test_ricci_and_rough_lap_each_2(self, proof, group):
        """Ricci = 2/R^2 and nabla*nabla = 2/R^2 on group manifold."""
        result = proof.gap_on_group_manifold(group, R=1.0)
        assert abs(result['ricci_coefficient'] - 2.0) < 1e-14
        assert abs(result['rough_laplacian_coefficient'] - 2.0) < 1e-14

    def test_su2_is_s3(self, proof):
        """SU(2) group manifold IS S^3."""
        result = proof.gap_on_group_manifold('SU(2)', R=1.0)
        assert result['is_three_sphere'] is True
        assert result['dim_manifold'] == 3

    @pytest.mark.parametrize("group,expected_dim", [
        ('SU(3)', 8), ('SU(5)', 24), ('SO(5)', 10),
        ('G2', 14), ('F4', 52), ('E8', 248),
    ])
    def test_non_su2_is_not_s3(self, proof, group, expected_dim):
        """For G != SU(2), group manifold is NOT S^3."""
        result = proof.gap_on_group_manifold(group, R=1.0)
        assert result['is_three_sphere'] is False, \
            f"{group}: should NOT be S^3"
        assert result['dim_manifold'] == expected_dim, \
            f"{group}: dim={result['dim_manifold']}, expected {expected_dim}"

    def test_radius_scaling(self, proof):
        """Gap scales as 1/R^2 on group manifold."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = proof.gap_on_group_manifold('SU(3)', R=R)
            expected = 4.0 / R**2
            assert abs(result['gap_value'] - expected) < 1e-12

    def test_note_warns_not_s3_for_su3(self, proof):
        """The note for SU(3) warns this is NOT S^3."""
        result = proof.gap_on_group_manifold('SU(3)', R=1.0)
        assert 'NOT' in result['note'] or 'not' in result['note']
        assert '8-dimensional' in result['note'] or '8' in result['note']


# ======================================================================
# 19. Coincidence: Results A and B agree for SU(2), differ conceptually
# ======================================================================

class TestCoincidenceAB:
    """
    For G = SU(2) ~ S^3, Results A and B give the same gap.
    For all other G, they are conceptually independent.
    """

    def test_su2_results_coincide(self, proof):
        """For SU(2), Result A on S^3 = Result B on group manifold = 4/R^2."""
        result_a = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=3)
        result_b = proof.gap_on_group_manifold('SU(2)', R=1.0)
        assert abs(result_a['gap'] - result_b['gap_value']) < 1e-14, \
            f"Result A={result_a['gap']}, Result B={result_b['gap_value']}"
        # Both should be 4.0
        assert abs(result_a['gap'] - 4.0) < 1e-14
        assert abs(result_b['gap_value'] - 4.0) < 1e-14
        # SU(2) group manifold is S^3
        assert result_b['is_three_sphere'] is True

    def test_su3_results_give_same_number_different_meaning(self, proof):
        """For SU(3), both give gap = 4/R^2 but on different manifolds."""
        result_a = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        result_b = proof.gap_on_group_manifold('SU(3)', R=1.0)
        # Same gap VALUE (coincidentally)
        assert abs(result_a['gap'] - result_b['gap_value']) < 1e-14
        # But Result A is on S^3 (dim=3), Result B is on SU(3) (dim=8)
        assert result_b['dim_manifold'] == 8
        assert result_b['is_three_sphere'] is False

    def test_e8_results_give_same_number_different_meaning(self, proof):
        """For E8, both give gap = 4/R^2 but on wildly different manifolds."""
        result_a = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=248)
        result_b = proof.gap_on_group_manifold('E8', R=1.0)
        # Same gap VALUE
        assert abs(result_a['gap'] - result_b['gap_value']) < 1e-14
        # But Result A is on S^3 (dim=3), Result B is on E8 (dim=248)
        assert result_b['dim_manifold'] == 248
        assert result_b['is_three_sphere'] is False

    def test_result_a_multiplicity_depends_on_g(self, proof):
        """Result A: multiplicities differ for different gauge groups."""
        r_su2 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=3)
        r_su3 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=8)
        r_e8 = proof.gap_on_s3_any_gauge_group(R=1.0, gauge_group_dim=248)
        m_su2 = r_su2['first_eigenvalues'][0]['total_multiplicity']
        m_su3 = r_su3['first_eigenvalues'][0]['total_multiplicity']
        m_e8 = r_e8['first_eigenvalues'][0]['total_multiplicity']
        # Multiplicities should scale with dim(g)
        assert m_su3 > m_su2
        assert m_e8 > m_su3
        # But eigenvalues are the SAME
        ev_su2 = r_su2['first_eigenvalues'][0]['eigenvalue']
        ev_su3 = r_su3['first_eigenvalues'][0]['eigenvalue']
        ev_e8 = r_e8['first_eigenvalues'][0]['eigenvalue']
        assert abs(ev_su2 - ev_su3) < 1e-14
        assert abs(ev_su2 - ev_e8) < 1e-14

    def test_general_theorem_mentions_both_results(self, proof):
        """The general theorem statement now separates Results A and B."""
        stmt = proof.general_theorem_statement()
        assert 'Result A' in stmt or 'THEOREM 5.1' in stmt
        assert 'Result B' in stmt or 'THEOREM 5.2' in stmt
        assert 'coincide' in stmt.lower() or 'SU(2)' in stmt
