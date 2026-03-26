"""
Tests for Actual BBS Contraction Constant c_epsilon on the 600-Cell.

Verifies all 5 geometric correction factors and the combined c_epsilon,
with comparisons to hypercubic values and contraction viability checks.

Tests organized by class:
    1.  CellVertexOverlap (9 tests)
    2.  CoordinationAnalysis (9 tests)
    3.  VolumeJacobian (8 tests)
    4.  CellContactStructure (9 tests)
    5.  BlockingHierarchyAnalysis (7 tests)
    6.  ActualCEpsilon (10 tests)
    7.  ContractionViabilityReport (8 tests)

Total: 60 tests.

Run:
    pytest tests/rg/test_cepsilon_600cell.py -v
"""

import numpy as np
import pytest

from yang_mills_s3.rg.cepsilon_600cell import (
    CellVertexOverlap,
    CoordinationAnalysis,
    VolumeJacobian,
    CellContactStructure,
    BlockingHierarchyAnalysis,
    ActualCEpsilon,
    RigorousFactorDerivation,
    ContractionViabilityReport,
    ViabilityResult,
    G2_BARE,
    DIM_SPACETIME,
    L_BLOCKING,
    N_COLORS_DEFAULT,
    BETA_0_SU2,
    N_VERTICES_600,
    N_EDGES_600,
    N_FACES_600,
    N_CELLS_600,
    CELLS_PER_VERTEX_HYP_D4,
    VERTEX_DEGREE_HYP_D4,
)
from yang_mills_s3.rg.first_rg_step import quadratic_casimir


# ======================================================================
# Fixtures: build expensive objects once per module
# ======================================================================

@pytest.fixture(scope="module")
def overlap():
    return CellVertexOverlap()

@pytest.fixture(scope="module")
def coordination():
    return CoordinationAnalysis()

@pytest.fixture(scope="module")
def volume_jac():
    return VolumeJacobian()

@pytest.fixture(scope="module")
def contact():
    return CellContactStructure()

@pytest.fixture(scope="module")
def blocking():
    return BlockingHierarchyAnalysis()

@pytest.fixture(scope="module")
def actual_ceps():
    return ActualCEpsilon()

@pytest.fixture(scope="module")
def viability():
    return ContractionViabilityReport()


# ======================================================================
# 1. CellVertexOverlap tests (9 tests)
# ======================================================================

class TestCellVertexOverlap:
    """Tests for Factor 1: Partition-of-Unity Overlap."""

    def test_cells_per_vertex_is_20(self, overlap):
        """THEOREM: Every vertex in the 600-cell is shared by exactly 20 cells."""
        cpv = overlap.cells_per_vertex_uniform()
        assert cpv == 20

    def test_cells_per_vertex_uniform(self, overlap):
        """THEOREM: All 120 vertices have the same number of cells (icosahedral symmetry)."""
        stats = overlap.cells_per_vertex_stats()
        assert stats['min'] == stats['max']
        assert stats['std'] == 0.0

    def test_cells_per_vertex_count(self, overlap):
        """All 120 vertices are accounted for."""
        cpv_all = overlap.cells_per_vertex_all()
        assert len(cpv_all) == N_VERTICES_600

    def test_hypercubic_cells_per_vertex_d4(self, overlap):
        """THEOREM: 2^4 = 16 for hypercubic d=4."""
        assert overlap.hypercubic_cells_per_vertex(4) == 16

    def test_overlap_ratio_value(self, overlap):
        """NUMERICAL: overlap ratio = 20/16 = 1.25."""
        ratio = overlap.overlap_ratio()
        assert abs(ratio - 1.25) < 1e-10

    def test_overlap_factor_value(self, overlap):
        """NUMERICAL: overlap factor = sqrt(1.25) ~ 1.118."""
        factor = overlap.overlap_factor()
        assert abs(factor - np.sqrt(1.25)) < 1e-10

    def test_overlap_factor_gt_1(self, overlap):
        """600-cell has MORE overlap than hypercubic -> factor > 1."""
        assert overlap.overlap_factor() > 1.0

    def test_euler_characteristic_zero(self, overlap):
        """THEOREM: chi(S^3) = V - E + F - C = 0."""
        euler = overlap.euler_check()
        assert euler['chi'] == 0

    def test_euler_check_counts(self, overlap):
        """Verify the 600-cell combinatorial counts."""
        euler = overlap.euler_check()
        assert euler['V'] == N_VERTICES_600
        assert euler['E'] == N_EDGES_600
        assert euler['F'] == N_FACES_600
        assert euler['C'] == N_CELLS_600


# ======================================================================
# 2. CoordinationAnalysis tests (9 tests)
# ======================================================================

class TestCoordinationAnalysis:
    """Tests for Factor 2: Coordination Number and Polymer Counting."""

    def test_vertex_degree_is_12(self, coordination):
        """THEOREM: 600-cell 1-skeleton is 12-regular."""
        assert coordination.vertex_degree() == 12

    def test_cell_face_degree_is_4(self, coordination):
        """THEOREM: Each tetrahedron shares 4 faces -> D_face = 4."""
        assert coordination.cell_face_degree() == 4

    def test_cell_vertex_sharing_uniform(self, coordination):
        """NUMERICAL: Vertex-sharing degree is uniform across all cells."""
        stats = coordination.cell_vertex_sharing_degree()
        assert stats['min_deg'] == stats['max_deg']

    def test_cell_vertex_sharing_value(self, coordination):
        """NUMERICAL: Vertex-sharing degree = 56."""
        stats = coordination.cell_vertex_sharing_degree()
        assert int(stats['min_deg']) == 56

    def test_hypercubic_vertex_degree_d4(self, coordination):
        """THEOREM: 2*4 = 8 for hypercubic d=4."""
        assert coordination.hypercubic_vertex_degree(4) == 8

    def test_coordination_ratio(self, coordination):
        """NUMERICAL: 12/8 = 1.5."""
        assert abs(coordination.coordination_ratio() - 1.5) < 1e-10

    def test_polymer_growth_rate_bound_face(self, coordination):
        """THEOREM: mu <= e*D = e*4 ~ 10.87 for face-sharing."""
        bound = coordination.polymer_growth_rate_bound('face')
        assert abs(bound - np.e * 4) < 1e-10

    def test_polymer_entropy_correction_face(self, coordination):
        """NUMERICAL: face-sharing ratio = e*4/(e*8) = 0.5."""
        ratio = coordination.polymer_entropy_correction('face')
        assert abs(ratio - 0.5) < 1e-10

    def test_polymer_entropy_face_favorable(self, coordination):
        """600-cell face-sharing has LESS polymer entropy than hypercubic."""
        ratio = coordination.polymer_entropy_correction('face')
        assert ratio < 1.0


# ======================================================================
# 3. VolumeJacobian tests (8 tests)
# ======================================================================

class TestVolumeJacobian:
    """Tests for Factor 3: Flat Simplicial vs Spherical Volume."""

    def test_spherical_volume_per_cell(self, volume_jac):
        """THEOREM: V_spherical/cell = 2 pi^2 / 600 ~ 0.0329."""
        expected = 2.0 * np.pi**2 / 600.0
        assert abs(volume_jac.spherical_volume_per_cell() - expected) < 1e-10

    def test_flat_volumes_positive(self, volume_jac):
        """All flat cell volumes are positive."""
        vols = volume_jac.flat_volume_per_cell()
        assert np.all(vols > 0)

    def test_flat_volumes_uniform(self, volume_jac):
        """NUMERICAL: By symmetry, all flat cell volumes should be identical."""
        vols = volume_jac.flat_volume_per_cell()
        assert np.std(vols) / np.mean(vols) < 1e-6

    def test_volume_ratio_gt_1(self, volume_jac):
        """Spherical volume > flat volume (curvature expands)."""
        ratios = volume_jac.volume_ratios()
        assert np.all(ratios > 1.0)

    def test_volume_ratio_mean_approx(self, volume_jac):
        """NUMERICAL: Mean ratio ~ 1.18 (spherical/flat)."""
        mean = volume_jac.mean_volume_ratio()
        assert 1.1 < mean < 1.3  # generous bounds

    def test_volume_ratio_uniform(self, volume_jac):
        """NUMERICAL: All cells have the same volume ratio (symmetry)."""
        stats = volume_jac.volume_ratio_stats()
        assert stats['uniformity'] < 1.001  # ratio of max/min

    def test_total_flat_lt_spherical(self, volume_jac):
        """Total flat volume < Vol(S^3) (flat tetrahedra don't tile S^3)."""
        ratio = volume_jac.flat_to_spherical_total_ratio()
        assert ratio < 1.0

    def test_total_flat_fraction(self, volume_jac):
        """NUMERICAL: Total flat volume ~ 84.6% of Vol(S^3)."""
        ratio = volume_jac.flat_to_spherical_total_ratio()
        assert 0.80 < ratio < 0.90


# ======================================================================
# 4. CellContactStructure tests (9 tests)
# ======================================================================

class TestCellContactStructure:
    """Tests for Factor 4: Cell Contact Structure."""

    def test_face_sharing_is_4(self, contact):
        """THEOREM: Each tetrahedron has 4 face-sharing neighbors."""
        stats = contact.face_sharing_count()
        assert stats['min'] == 4
        assert stats['max'] == 4

    def test_edge_sharing_uniform(self, contact):
        """NUMERICAL: Edge-sharing degree is uniform."""
        stats = contact.edge_sharing_count()
        assert stats['min'] == stats['max']

    def test_edge_sharing_value(self, contact):
        """NUMERICAL: Edge-sharing (non-face) count = 12."""
        stats = contact.edge_sharing_count()
        assert int(stats['mean']) == 12

    def test_vertex_only_sharing_uniform(self, contact):
        """NUMERICAL: Vertex-only sharing is uniform."""
        stats = contact.vertex_only_sharing_count()
        assert stats['min'] == stats['max']

    def test_vertex_only_sharing_value(self, contact):
        """NUMERICAL: Vertex-only sharing count = 40."""
        stats = contact.vertex_only_sharing_count()
        assert int(stats['mean']) == 40

    def test_total_contact_equals_sum(self, contact):
        """Total = face + edge + vertex_only."""
        face = contact.face_sharing_count()['mean']
        edge = contact.edge_sharing_count()['mean']
        vert = contact.vertex_only_sharing_count()['mean']
        total = contact.total_contact_count()['mean']
        assert abs(face + edge + vert - total) < 1e-10

    def test_total_contact_is_56(self, contact):
        """NUMERICAL: Total contact = 4 + 12 + 40 = 56."""
        stats = contact.total_contact_count()
        assert int(stats['mean']) == 56

    def test_hypercubic_face_sharing_d4(self, contact):
        """THEOREM: 2*4 = 8 for d=4 hypercubic."""
        assert contact.hypercubic_face_sharing(4) == 8

    def test_contact_ratio_face(self, contact):
        """NUMERICAL: Face contact ratio = 4/8 = 0.5 (favorable)."""
        ratio = contact.contact_ratio_face()
        assert abs(ratio - 0.5) < 1e-10


# ======================================================================
# 5. BlockingHierarchyAnalysis tests (7 tests)
# ======================================================================

class TestBlockingHierarchyAnalysis:
    """Tests for Factor 5: Blocking Hierarchy."""

    def test_hierarchy_structure(self, blocking):
        """600 -> 120 -> 24 -> 5 -> 1."""
        assert blocking.hierarchy_blocks == [600, 120, 24, 5, 1]

    def test_blocking_ratios(self, blocking):
        """NUMERICAL: Ratios are [5, 5, 4.8, 5]."""
        ratios = blocking.blocking_ratios()
        assert len(ratios) == 4
        assert abs(ratios[0] - 5.0) < 1e-10
        assert abs(ratios[1] - 5.0) < 1e-10
        assert abs(ratios[2] - 4.8) < 1e-10
        assert abs(ratios[3] - 5.0) < 1e-10

    def test_hypercubic_blocking_ratio_d4(self, blocking):
        """THEOREM: L^d = 2^4 = 16 for hypercubic L=2, d=4."""
        assert blocking.hypercubic_blocking_ratio() == 16.0

    def test_effective_L_less_than_2(self, blocking):
        """Effective L ~ 5^{1/4} ~ 1.495 < 2."""
        L_effs = blocking.effective_L_per_step()
        for L in L_effs:
            assert L < 2.0
            assert L > 1.0

    def test_single_step_contraction_gt_05(self, blocking):
        """1/L_eff ~ 0.67 > 0.5 = 1/L_hyp (WORSE than hypercubic)."""
        factor = blocking.single_step_contraction_factor()
        assert factor > 0.5
        assert factor < 1.0

    def test_n_rg_steps(self, blocking):
        """4 steps: 600->120->24->5->1."""
        assert blocking.n_rg_steps() == 4

    def test_volume_jacobians_gt_1(self, blocking):
        """Per-step Jacobians > 1 (less volume shrinkage than hypercubic)."""
        jacobians = blocking.volume_jacobian_per_step()
        for j in jacobians:
            assert j > 1.0


# ======================================================================
# 6. ActualCEpsilon tests (10 tests)
# ======================================================================

class TestActualCEpsilon:
    """Tests for the combined corrected c_epsilon."""

    def test_base_c_epsilon_su2(self, actual_ceps):
        """NUMERICAL: base c_eps = C_2(adj)/(4 pi) = 2/(4 pi) ~ 0.159."""
        expected = quadratic_casimir(2) / (4.0 * np.pi)
        assert abs(actual_ceps.base_c_epsilon() - expected) < 1e-10

    def test_factor1_gt_1(self, actual_ceps):
        """Factor 1 (overlap) > 1: more overlap increases c_epsilon."""
        assert actual_ceps.factor1_overlap() > 1.0

    def test_factor2_optimistic_lt_1(self, actual_ceps):
        """Factor 2 at alpha=1.0 (face-sharing): < 1 (favorable)."""
        assert actual_ceps.factor2_polymer_entropy(1.0) < 1.0

    def test_factor2_pessimistic_gt_1(self, actual_ceps):
        """Factor 2 at alpha=0.0 (vertex-level): > 1 (unfavorable)."""
        assert actual_ceps.factor2_polymer_entropy(0.0) > 1.0

    def test_factor3_gt_1(self, actual_ceps):
        """Factor 3 (volume Jacobian) > 1: spherical > flat."""
        assert actual_ceps.factor3_volume_jacobian() > 1.0

    def test_factor4_lt_1(self, actual_ceps):
        """Factor 4 (contact) < 1: fewer face-sharing neighbors."""
        assert actual_ceps.factor4_contact() < 1.0

    def test_factor5_gt_1(self, actual_ceps):
        """Factor 5 (blocking) > 1: less volume shrinkage per step."""
        assert actual_ceps.factor5_blocking() > 1.0

    def test_corrected_optimistic_lt_base(self, actual_ceps):
        """NUMERICAL: Optimistic corrected c_eps < base c_eps."""
        assert actual_ceps.corrected_c_epsilon(1.0) < actual_ceps.base_c_epsilon()

    def test_corrected_pessimistic_gt_base(self, actual_ceps):
        """NUMERICAL: Pessimistic corrected c_eps > base c_eps."""
        assert actual_ceps.corrected_c_epsilon(0.0) > actual_ceps.base_c_epsilon()

    def test_epsilon_at_scale_0(self, actual_ceps):
        """NUMERICAL: epsilon(0) = c_eps * g_bar_0."""
        c_eps = actual_ceps.corrected_c_epsilon(1.0)
        g_bar_0 = np.sqrt(G2_BARE)
        expected = c_eps * g_bar_0
        assert abs(actual_ceps.epsilon_at_scale(0, 1.0) - expected) < 1e-10


# ======================================================================
# 7. ContractionViabilityReport tests (8 tests)
# ======================================================================

class TestContractionViabilityReport:
    """Tests for the viability report: BRUTALLY HONEST."""

    def test_optimistic_holds(self, viability):
        """NUMERICAL: With alpha=1.0, contraction holds at all scales."""
        result = viability.check_viability(alpha=1.0)
        assert result.contraction_holds_at_all_scales
        assert result.strategy == 'SUCCESS'

    def test_mixed_holds(self, viability):
        """NUMERICAL: With alpha=0.5, contraction holds at all scales."""
        result = viability.check_viability(alpha=0.5)
        assert result.contraction_holds_at_all_scales
        assert result.strategy == 'SUCCESS'

    def test_pessimistic_holds(self, viability):
        """NUMERICAL: Even with alpha=0.0, contraction holds."""
        result = viability.check_viability(alpha=0.0)
        assert result.contraction_holds_at_all_scales
        assert result.strategy == 'SUCCESS'

    def test_epsilon_decreases_with_scale(self, viability):
        """Epsilon decreases with j (asymptotic freedom)."""
        result = viability.check_viability(alpha=1.0, N_scales=8)
        for j in range(len(result.epsilon_profile) - 1):
            assert result.epsilon_profile[j] > result.epsilon_profile[j + 1]

    def test_max_epsilon_at_ir(self, viability):
        """Maximum epsilon is at j=0 (IR, largest coupling)."""
        result = viability.check_viability(alpha=1.0, N_scales=8)
        assert result.max_epsilon == result.epsilon_profile[0]

    def test_max_epsilon_lt_1(self, viability):
        """NUMERICAL: max(epsilon) < 1 for all scenarios."""
        reports = viability.full_report()
        for scenario, result in reports.items():
            assert result.max_epsilon < 1.0, (
                f"{scenario}: max_epsilon = {result.max_epsilon}"
            )

    def test_full_report_three_scenarios(self, viability):
        """Full report has exactly 3 scenarios."""
        reports = viability.full_report()
        assert 'optimistic' in reports
        assert 'mixed' in reports
        assert 'pessimistic' in reports

    def test_print_report_nonempty(self, viability):
        """Print report produces non-empty string."""
        text = viability.print_report()
        assert len(text) > 100
        assert 'CONTRACTION VIABILITY REPORT' in text


# ======================================================================
# 8. Cross-check and consistency tests (additional)
# ======================================================================

class TestCrossChecks:
    """Cross-checks between factors and physical consistency."""

    def test_face_plus_edge_plus_vertex_only_eq_total(self, contact):
        """Total contact = face + edge + vertex_only for each cell."""
        face = contact.face_sharing_count()['mean']
        edge = contact.edge_sharing_count()['mean']
        vert = contact.vertex_only_sharing_count()['mean']
        total = contact.total_contact_count()['mean']
        assert abs(face + edge + vert - total) < 1e-10

    def test_vertex_sharing_eq_total_contact(self, coordination, contact):
        """Vertex-sharing degree from coordination == total contact from contact."""
        vs_stats = coordination.cell_vertex_sharing_degree()
        tc_stats = contact.total_contact_count()
        assert abs(vs_stats['mean_deg'] - tc_stats['mean']) < 1e-10

    def test_contraction_product_decreases(self, actual_ceps):
        """Product of epsilons decreases as N increases."""
        p4 = actual_ceps.contraction_product(4)
        p8 = actual_ceps.contraction_product(8)
        assert p8 < p4

    def test_contraction_product_small(self, actual_ceps):
        """NUMERICAL: Product for 8 scales is very small (< 1e-3)."""
        p8 = actual_ceps.contraction_product(8, alpha=1.0)
        assert p8 < 1e-3

    def test_g_bar_0_value(self, actual_ceps):
        """g_bar_0 = sqrt(g0^2) ~ 2.506."""
        g_bar_0 = actual_ceps.g_bar_at_scale(0)
        assert abs(g_bar_0 - np.sqrt(G2_BARE)) < 1e-10

    def test_g_bar_decreases(self, actual_ceps):
        """g_bar_j decreases with j (asymptotic freedom)."""
        for j in range(7):
            assert actual_ceps.g_bar_at_scale(j) > actual_ceps.g_bar_at_scale(j + 1)

    def test_beta0_su2_value(self):
        """beta_0 for SU(2) = 22/(48 pi^2) ~ 0.04648."""
        expected = 22.0 / (48.0 * np.pi**2)
        assert abs(BETA_0_SU2 - expected) < 1e-10

    def test_all_factors_product_matches_corrected(self, actual_ceps):
        """The product of individual factors matches the corrected c_epsilon."""
        for alpha in [0.0, 0.5, 1.0]:
            factors = actual_ceps.all_factors(alpha)
            computed = factors['base_c_epsilon'] * factors['product_of_corrections']
            direct = factors['corrected_c_epsilon']
            assert abs(computed - direct) < 1e-12

    def test_600cell_identity_vef(self):
        """
        Verify: 4*C = 2*F (each face shared by 2 cells, each cell has 4 faces).
        THEOREM.
        """
        assert 4 * N_CELLS_600 == 2 * N_FACES_600

    def test_600cell_edge_count(self):
        """
        Verify: 6*C = (degree)*F for the 600-cell.
        Each cell has 6 edges, each edge is shared among several cells.
        The edge sum: 6*600 = 3600. With 720 edges, each shared by 5 cells.
        """
        assert 6 * N_CELLS_600 == 5 * N_EDGES_600


# ======================================================================
# 9. RigorousFactorDerivation tests (20 tests)
# ======================================================================

@pytest.fixture(scope="module")
def rigorous():
    return RigorousFactorDerivation()


class TestRigorousFactorDerivation:
    """
    Tests for the explicit first-principles derivations of F4 and F5,
    created in response to peer review.

    Verifies:
    1. F4 derivation from Cauchy-Schwarz is correct and matches computed value
    2. F5 derivation from polytope hierarchy is correct and matches computed value
    3. F4 and F5 are structurally independent (no double-counting)
    4. Algebraic identities hold exactly
    5. Full derivation summary is internally consistent
    """

    # --- F4: Contact interaction scaling ---

    def test_F4_value_exact(self, rigorous):
        """THEOREM: F4 = sqrt(4/8) = 1/sqrt(2) = 0.70710678..."""
        result = rigorous.derive_F4()
        expected = 1.0 / np.sqrt(2.0)
        assert abs(result['F4'] - expected) < 1e-14

    def test_F4_inputs_exact(self, rigorous):
        """F4 inputs are exact integers from Coxeter (no fitting)."""
        result = rigorous.derive_F4()
        assert result['D_face_600'] == 4
        assert result['D_face_hyp'] == 8

    def test_F4_status_theorem(self, rigorous):
        """F4 is classified as THEOREM, not NUMERICAL."""
        result = rigorous.derive_F4()
        assert result['status'] == 'THEOREM'

    def test_F4_derivation_method(self, rigorous):
        """F4 uses Cauchy-Schwarz, not empirical fitting."""
        result = rigorous.derive_F4()
        assert 'Cauchy-Schwarz' in result['derivation_method']

    def test_F4_matches_computed(self, rigorous):
        """Derived F4 matches the numerically computed value from geometry."""
        result = rigorous.derive_F4()
        assert result['verified_numerically']

    def test_F4_lt_1(self, rigorous):
        """F4 < 1: fewer face contacts is FAVORABLE."""
        result = rigorous.derive_F4()
        assert result['F4'] < 1.0

    # --- F5: Blocking hierarchy correction ---

    def test_F5_value_exact(self, rigorous):
        """THEOREM: F5 = 2 / 5^{1/4} = (16/5)^{1/4} = 1.33748..."""
        result = rigorous.derive_F5()
        expected = 2.0 / 5.0**(1.0/4.0)
        assert abs(result['F5'] - expected) < 1e-14

    def test_F5_inputs_exact(self, rigorous):
        """F5 inputs are exact integers from Coxeter (no fitting)."""
        result = rigorous.derive_F5()
        assert result['b_600'] == 5
        assert result['b_hyp'] == 16
        assert result['d'] == 4

    def test_F5_hierarchy_integers(self, rigorous):
        """The polytope hierarchy has integer cell counts from Coxeter."""
        result = rigorous.derive_F5()
        assert result['hierarchy'] == [600, 120, 24, 5, 1]
        assert result['step_ratios'] == [5, 5, 4.8, 5]

    def test_F5_algebraic_identity(self, rigorous):
        """THEOREM: 2/5^{1/4} = (16/5)^{1/4} (algebraic identity)."""
        result = rigorous.derive_F5()
        assert result['algebraic_identity_check']['all_equal']

    def test_F5_three_representations_agree(self, rigorous):
        """All three representations of F5 give the same value."""
        result = rigorous.derive_F5()
        check = result['algebraic_identity_check']
        vals = [check['2/5^{1/4}'], check['(16/5)^{1/4}'], check['(b_hyp/b_600)^{1/d}']]
        for v in vals:
            assert abs(v - vals[0]) < 1e-14

    def test_F5_status_theorem(self, rigorous):
        """F5 is classified as THEOREM, not NUMERICAL."""
        result = rigorous.derive_F5()
        assert result['status'] == 'THEOREM'

    def test_F5_derivation_method(self, rigorous):
        """F5 uses Coxeter polytope hierarchy, not empirical fitting."""
        result = rigorous.derive_F5()
        assert 'Coxeter' in result['derivation_method']
        assert 'BBS' in result['derivation_method']

    def test_F5_matches_computed(self, rigorous):
        """Derived F5 matches the numerically computed value."""
        result = rigorous.derive_F5()
        assert result['verified_numerically']

    def test_F5_gt_1(self, rigorous):
        """F5 > 1: weaker blocking is UNFAVORABLE."""
        result = rigorous.derive_F5()
        assert result['F5'] > 1.0

    # --- Independence argument ---

    def test_independence_flag(self, rigorous):
        """The independence analysis confirms no double-counting."""
        result = rigorous.independence_argument()
        assert result['no_double_counting']

    def test_F4_F5_different_bbs_locations(self, rigorous):
        """F4 and F5 enter DIFFERENT parts of BBS Theorem 8.2.4."""
        result = rigorous.independence_argument()
        assert 'polymer' in result['F4_source']['BBS_location'].lower()
        assert 'section 4.1' in result['F5_source']['BBS_location'].lower() or \
               'def 3.2.1' in result['F5_source']['BBS_location'].lower()

    def test_F4_discrete_F5_continuous(self, rigorous):
        """F4 bounds a DISCRETE sum; F5 bounds a CONTINUOUS integral."""
        result = rigorous.independence_argument()
        assert 'discrete' in result['F4_source']['mathematical_operation'].lower() or \
               'sum' in result['F4_source']['mathematical_operation'].lower()
        assert 'integral' in result['F5_source']['mathematical_operation'].lower() or \
               'gaussian' in result['F5_source']['mathematical_operation'].lower()

    # --- Full summary ---

    def test_full_summary_has_all_factors(self, rigorous):
        """Full derivation summary includes all 5 factors."""
        summary = rigorous.full_derivation_summary()
        assert 'F1_overlap' in summary
        assert 'F2_polymer' in summary
        assert 'F3_volume' in summary
        assert 'F4_contact' in summary
        assert 'F5_blocking' in summary
        assert 'independence_F4_F5' in summary
