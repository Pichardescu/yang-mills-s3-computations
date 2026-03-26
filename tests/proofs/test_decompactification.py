"""
Tests for the decompactification module: S^3(R) x R -> R^4 as R -> infinity.

Tests cover all 8 classes:
    1. UniformGapBound:       gap(R) > 0 for all R, crossover, minimum
    2. MoscoConvergence:      liminf, recovery, stereographic map
    3. ISO4Recovery:          generators, contraction, commutator -> 0
    4. OSAxiomsInLimit:       all axioms at each R, limit preserved
    5. WightmanReconstruction: mass gap extracted, spectrum condition
    6. DecompactificationTheorem: all ingredients, status tracked
    7. PhaseTransitionAbsence: no transition detected, gap smooth
    8. ClayMillenniumConnection: requirements mapped, gap identified

Test count target: 60+

Standards:
    - Claims labeled: THEOREM / PROPOSITION / NUMERICAL / CONJECTURE
    - Honest about what is proven vs conjectured
    - The decompactification itself is PROPOSITION
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.decompactification import (
    UniformGapBound,
    MoscoConvergence,
    ISO4Recovery,
    OSAxiomsInLimit,
    WightmanReconstruction,
    DecompactificationTheorem,
    PhaseTransitionAbsence,
    ClayMillenniumConnection,
    HBAR_C,
    LAMBDA_QCD_MEV,
    R_PHYSICAL_FM,
    DIM_SO5,
    DIM_ISO4,
    GAP_FACTOR,
)
from yang_mills_s3.proofs.r_limit import ClaimStatus


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def gap_bound():
    """UniformGapBound for SU(2)."""
    return UniformGapBound(N=2, Lambda_QCD=200.0)


@pytest.fixture
def gap_bound_su3():
    """UniformGapBound for SU(3)."""
    return UniformGapBound(N=3, Lambda_QCD=200.0)


@pytest.fixture
def mosco():
    """MoscoConvergence for SU(2)."""
    return MoscoConvergence(N=2, Lambda_QCD=200.0)


@pytest.fixture
def iso4():
    """ISO4Recovery instance."""
    return ISO4Recovery()


@pytest.fixture
def os_limit():
    """OSAxiomsInLimit for SU(2)."""
    return OSAxiomsInLimit(N=2, Lambda_QCD=200.0)


@pytest.fixture
def wightman():
    """WightmanReconstruction for SU(2)."""
    return WightmanReconstruction(N=2, Lambda_QCD=200.0)


@pytest.fixture
def decomp():
    """DecompactificationTheorem for SU(2)."""
    return DecompactificationTheorem(N=2, Lambda_QCD=200.0)


@pytest.fixture
def no_transition():
    """PhaseTransitionAbsence for SU(2)."""
    return PhaseTransitionAbsence(N=2, Lambda_QCD=200.0)


@pytest.fixture
def clay():
    """ClayMillenniumConnection for SU(2)."""
    return ClayMillenniumConnection(N=2, Lambda_QCD=200.0)


# ======================================================================
# 1. UniformGapBound
# ======================================================================

class TestUniformGapBound:
    """gap(R) > 0 for all R, crossover identified, minimum found."""

    def test_gap_positive_all_R(self, gap_bound):
        """Gap is positive at every radius tested."""
        for R in [0.1, 0.5, 1.0, 1.97, 2.2, 5.0, 10.0, 50.0, 100.0]:
            result = gap_bound.gap_at_R(R)
            assert result['gap_MeV'] > 0, f"Gap should be > 0 at R={R}"

    def test_gap_positive_logspace(self, gap_bound):
        """Gap is positive over a wide logspace scan."""
        R_vals = np.logspace(-1, 2, 50)
        for R in R_vals:
            result = gap_bound.gap_at_R(R)
            assert result['gap_MeV'] > 0

    def test_kinematic_regime(self, gap_bound):
        """At small R, geometric gap dominates."""
        result = gap_bound.gap_at_R(0.1)
        assert result['regime'] == 'kinematic'
        assert result['geometric_MeV'] > result['dynamical_MeV']

    def test_dynamic_regime(self, gap_bound):
        """At large R, dynamical gap dominates."""
        result = gap_bound.gap_at_R(50.0)
        assert result['regime'] == 'dynamic'
        assert result['dynamical_MeV'] >= result['geometric_MeV']

    def test_crossover_regime(self, gap_bound):
        """At intermediate R, crossover regime."""
        R_cross = gap_bound.crossover_R()
        result = gap_bound.gap_at_R(R_cross)
        assert result['regime'] == 'crossover'

    def test_crossover_radius_formula(self, gap_bound):
        """R* = GAP_FACTOR * hbar_c / Lambda_QCD."""
        R_star = gap_bound.crossover_R()
        expected = GAP_FACTOR * HBAR_C / 200.0
        assert abs(R_star - expected) < 1e-10

    def test_crossover_radius_approximately_2_fm(self, gap_bound):
        """R* ~ 1.97 fm for Lambda_QCD = 200 MeV."""
        R_star = gap_bound.crossover_R()
        assert 1.5 < R_star < 2.5

    def test_minimum_gap_positive(self, gap_bound):
        """The minimum gap over all R is positive."""
        result = gap_bound.minimum_gap()
        assert result['gap_positive']
        assert result['min_gap_MeV'] > 0

    def test_minimum_gap_at_crossover(self, gap_bound):
        """Minimum gap occurs near the crossover radius."""
        result = gap_bound.minimum_gap()
        R_cross = gap_bound.crossover_R()
        # The minimum should be near crossover
        assert abs(result['min_gap_R_fm'] - R_cross) < 0.5 * R_cross

    def test_minimum_gap_is_lambda_qcd(self, gap_bound):
        """At crossover, both gaps are Lambda_QCD."""
        result = gap_bound.minimum_gap()
        assert abs(result['min_gap_MeV'] - 200.0) < 10.0

    def test_is_uniform(self, gap_bound):
        """Gap is uniform over (0.1, 100) fm."""
        result = gap_bound.is_uniform()
        assert result['is_uniform']
        assert result['all_gaps_positive']
        assert result['lower_bound_MeV'] > 0

    def test_geometric_gap_1_over_R(self, gap_bound):
        """Geometric gap scales as 1/R."""
        g1 = gap_bound.gap_at_R(1.0)['geometric_MeV']
        g2 = gap_bound.gap_at_R(2.0)['geometric_MeV']
        assert abs(g1 / g2 - 2.0) < 1e-10

    def test_dynamical_gap_R_independent(self, gap_bound):
        """Dynamical gap is the same at all R."""
        d1 = gap_bound.gap_at_R(1.0)['dynamical_MeV']
        d2 = gap_bound.gap_at_R(100.0)['dynamical_MeV']
        assert abs(d1 - d2) < 1e-10

    def test_plot_data_shape(self, gap_bound):
        """plot_gap_vs_R returns arrays of correct shape."""
        data = gap_bound.plot_gap_vs_R(n_points=50)
        assert len(data['R_fm']) == 50
        assert len(data['gap_MeV']) == 50
        assert len(data['geometric_MeV']) == 50
        assert len(data['dynamical_MeV']) == 50
        assert data['crossover_R_fm'] > 0

    def test_invalid_R_raises(self, gap_bound):
        """Negative or zero R raises ValueError."""
        with pytest.raises(ValueError):
            gap_bound.gap_at_R(0.0)
        with pytest.raises(ValueError):
            gap_bound.gap_at_R(-1.0)

    def test_status_is_proposition(self, gap_bound):
        """Uniform gap status is PROPOSITION."""
        s = gap_bound.status()
        assert s.label == 'PROPOSITION'

    def test_su3_gap_also_positive(self, gap_bound_su3):
        """SU(3) gap is also positive at all R."""
        for R in [0.5, 2.2, 50.0]:
            result = gap_bound_su3.gap_at_R(R)
            assert result['gap_MeV'] > 0


# ======================================================================
# 2. MoscoConvergence
# ======================================================================

class TestMoscoConvergence:
    """liminf and recovery verified numerically, stereographic correct."""

    def test_stereographic_conformal_factor(self, mosco):
        """Conformal factor Omega(0) = 2 at the origin."""
        data = mosco.stereographic_map(10.0)
        # At r=0: Omega = 2R^2/R^2 = 2
        assert abs(data['conformal_factor'][0] - 2.0) < 1e-10

    def test_stereographic_converges_to_flat(self, mosco):
        """As R -> inf, conformal factor -> 2 everywhere."""
        data = mosco.stereographic_map(1000.0)
        assert data['converges_to_flat']
        # All errors near origin should be tiny
        assert data['metric_error'][0] < 1e-6

    def test_stereographic_large_R(self, mosco):
        """At very large R, metric error is negligible near the origin."""
        data = mosco.stereographic_map(1e6)
        # Only check points with r/R < 0.5 (near origin, where chart is good)
        near_origin = data['max_error_at_r_over_R'] < 0.5
        assert np.all(data['metric_error'][near_origin] < 0.01)

    def test_action_on_sphere_positive(self, mosco):
        """Action on S^3 is positive for non-zero curvature."""
        S = mosco.action_on_sphere(1.0, 2.2)
        assert S > 0

    def test_action_on_flat_positive(self, mosco):
        """Action on flat space is positive for non-zero curvature."""
        S = mosco.action_on_flat(1.0, 10.0)
        assert S > 0

    def test_liminf_satisfied(self, mosco):
        """liminf condition verified for increasing R."""
        R_seq = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0])
        result = mosco.verify_liminf(1.0, R_seq)
        assert result['liminf_satisfied']

    def test_liminf_convergence(self, mosco):
        """S_R -> S_flat as R -> inf."""
        R_seq = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        result = mosco.verify_liminf(1.0, R_seq)
        assert result['convergence']

    def test_recovery_converges(self, mosco):
        """Recovery sequence converges: S_R[A_R] -> S_flat[A]."""
        R_seq = np.array([2.0, 5.0, 10.0, 50.0, 100.0, 500.0])
        result = mosco.verify_recovery(1.0, R_seq)
        assert result['converges']

    def test_recovery_conformal_invariance(self, mosco):
        """Recovery uses conformal invariance in d=4."""
        R_seq = np.array([10.0, 100.0])
        result = mosco.verify_recovery(1.0, R_seq)
        assert result['conformal_invariance_used']

    def test_recovery_errors_decrease(self, mosco):
        """Recovery errors decrease with R."""
        R_seq = np.array([5.0, 10.0, 50.0, 100.0, 500.0])
        result = mosco.verify_recovery(1.0, R_seq)
        errors = result['recovery_errors']
        # Errors should be monotonically decreasing (O(1/R^2))
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-10

    def test_status_is_theorem(self, mosco):
        """Mosco convergence status is THEOREM."""
        s = mosco.status()
        assert s.label == 'THEOREM'


# ======================================================================
# 3. ISO4Recovery
# ======================================================================

class TestISO4Recovery:
    """Generators correct, contraction limit, commutator -> 0."""

    def test_dim_so5(self, iso4):
        """dim(SO(5)) = 10."""
        assert iso4.dim_so5 == 10

    def test_dim_iso4(self, iso4):
        """dim(ISO(4)) = 10."""
        assert iso4.dim_iso4 == 10

    def test_dim_match(self, iso4):
        """dim(SO(5)) = dim(ISO(4))."""
        assert iso4.dim_so5 == iso4.dim_iso4

    def test_so5_generators_count(self, iso4):
        """so(5) has 10 generators."""
        data = iso4.so5_generators()
        assert data['n_generators'] == 10

    def test_so5_generators_antisymmetric(self, iso4):
        """so(5) generators are antisymmetric."""
        data = iso4.so5_generators()
        for (a, b), M in data['generators'].items():
            assert np.allclose(M, -M.T), f"M_{a}{b} not antisymmetric"

    def test_iso4_dimensions(self, iso4):
        """iso(4) = 4 translations + 6 rotations."""
        data = iso4.iso4_generators()
        assert data['n_translations'] == 4
        assert data['n_rotations'] == 6
        assert data['dimension'] == 10

    def test_contraction_map_generators(self, iso4):
        """Contraction at R=10 gives 6 rotations + 4 translations."""
        data = iso4.contraction_map(10.0)
        assert data['n_rotations'] == 6
        assert data['n_translations'] == 4

    def test_commutator_error_decreases(self, iso4):
        """Commutator error decreases with R."""
        e1 = iso4.commutator_error(10.0)['max_error']
        e2 = iso4.commutator_error(100.0)['max_error']
        assert e2 < e1

    def test_commutator_error_1_over_R2(self, iso4):
        """Commutator error scales as 1/R^2."""
        e1 = iso4.commutator_error(10.0)['max_error']
        e2 = iso4.commutator_error(100.0)['max_error']
        # Ratio should be (100/10)^2 = 100
        ratio = e1 / e2
        assert abs(ratio - 100.0) < 1.0

    def test_commutator_small_at_large_R(self, iso4):
        """At large R, commutator error is small."""
        result = iso4.commutator_error(100.0)
        assert result['is_small']

    def test_verify_limit_converges(self, iso4):
        """Contraction converges along increasing R."""
        R_seq = np.array([5.0, 10.0, 50.0, 100.0, 500.0])
        result = iso4.verify_limit(R_seq)
        assert result['converges_to_zero']
        assert result['dim_match']

    def test_verify_limit_scaling(self, iso4):
        """Scaling exponent is approximately -2."""
        R_seq = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        result = iso4.verify_limit(R_seq)
        assert result['scaling_correct']

    def test_status_is_theorem(self, iso4):
        """ISO4 recovery status is THEOREM."""
        s = iso4.status()
        assert s.label == 'THEOREM'

    def test_invalid_R_raises(self, iso4):
        """Negative R raises ValueError."""
        with pytest.raises(ValueError):
            iso4.contraction_map(0.0)
        with pytest.raises(ValueError):
            iso4.commutator_error(-1.0)


# ======================================================================
# 4. OSAxiomsInLimit
# ======================================================================

class TestOSAxiomsInLimit:
    """All axioms at each R, limit preserved."""

    def test_os_at_R_satisfied(self, os_limit):
        """OS axioms satisfied at R = 2.2 fm."""
        result = os_limit.verify_at_R(2.2)
        assert result['all_satisfied']

    def test_reflection_positivity_theorem(self, os_limit):
        """Reflection positivity is THEOREM (R direction unchanged)."""
        result = os_limit.check_reflection_positivity(2.2)
        assert result['reflection_positivity_satisfied']
        assert result['status'] == 'THEOREM'
        assert result['R_direction_unchanged']

    def test_reflection_positivity_at_various_R(self, os_limit):
        """Reflection positivity holds at all R."""
        for R in [0.5, 1.0, 2.2, 10.0, 50.0]:
            result = os_limit.check_reflection_positivity(R)
            assert result['reflection_positivity_satisfied']

    def test_clustering_positive_gap(self, os_limit):
        """Clustering has positive gap at each R."""
        for R in [0.5, 2.2, 10.0]:
            result = os_limit.check_clustering(R)
            assert result['gap_positive']
            assert result['gap_MeV'] > 0
            assert result['clustering_length_fm'] > 0
            assert result['clustering_length_fm'] < float('inf')

    def test_clustering_status_proposition(self, os_limit):
        """Clustering status is PROPOSITION (depends on uniform gap)."""
        result = os_limit.check_clustering(2.2)
        assert result['status'] == 'PROPOSITION'

    def test_verify_limit_all_satisfied(self, os_limit):
        """OS axioms satisfied at all R in a sequence."""
        R_seq = np.array([1.0, 2.0, 5.0, 10.0, 50.0])
        result = os_limit.verify_limit(R_seq)
        assert result['all_os_satisfied_at_every_R']
        assert result['all_gaps_positive']

    def test_verify_limit_min_gap_positive(self, os_limit):
        """Min gap in the limit sequence is positive."""
        R_seq = np.array([1.0, 5.0, 10.0, 50.0])
        result = os_limit.verify_limit(R_seq)
        assert result['min_gap_MeV'] > 0

    def test_status_is_proposition(self, os_limit):
        """OS axioms in limit status is PROPOSITION."""
        s = os_limit.status()
        assert s.label == 'PROPOSITION'


# ======================================================================
# 5. WightmanReconstruction
# ======================================================================

class TestWightmanReconstruction:
    """Mass gap extracted, spectrum condition verified."""

    def test_reconstruct_at_R(self, wightman):
        """Wightman QFT exists at R = 2.2 fm."""
        result = wightman.reconstruct_at_R(2.2)
        assert result['wightman_qft_exists']
        assert result['mass_gap_positive']

    def test_reconstruct_at_various_R(self, wightman):
        """Wightman QFT exists at multiple radii."""
        for R in [0.5, 1.0, 2.2, 5.0]:
            result = wightman.reconstruct_at_R(R)
            assert result['wightman_qft_exists']

    def test_reconstruct_limit(self, wightman):
        """Wightman reconstruction along R sequence."""
        R_seq = np.array([1.0, 2.0, 5.0, 10.0])
        result = wightman.reconstruct_limit(R_seq)
        assert result['all_wightman_qft_exist']
        assert result['all_mass_gaps_positive']

    def test_verify_mass_gap(self, wightman):
        """Mass gap is positive at R = 2.2 fm."""
        result = wightman.verify_mass_gap(2.2)
        assert result['mass_gap_positive']
        assert result['gap_uniform_bound_MeV'] > 0

    def test_verify_spectrum_condition(self, wightman):
        """Spectral condition satisfied at R = 2.2 fm."""
        result = wightman.verify_spectrum_condition(2.2)
        assert result['spectral_condition_satisfied']
        assert result['spectrum_nonnegative']
        assert result['vacuum_eigenvalue'] == 0.0

    def test_spectrum_discrete(self, wightman):
        """Spectrum is discrete on compact S^3."""
        result = wightman.verify_spectrum_condition(2.2)
        assert result['discrete_spectrum']

    def test_status_is_proposition(self, wightman):
        """Wightman reconstruction status is PROPOSITION."""
        s = wightman.status()
        assert s.label == 'PROPOSITION'


# ======================================================================
# 6. DecompactificationTheorem
# ======================================================================

class TestDecompactificationTheorem:
    """All ingredients verified, status tracked."""

    def test_verify_all_ingredients(self, decomp):
        """All 6 ingredients are verified."""
        result = decomp.verify_all_ingredients(R_range=(0.5, 50.0), n_R=10)
        assert 'ingredients' in result
        assert len(result['ingredients']) == 6

    def test_each_ingredient_has_status(self, decomp):
        """Each ingredient has satisfied and status fields."""
        result = decomp.verify_all_ingredients(R_range=(0.5, 50.0), n_R=10)
        for name, ing in result['ingredients'].items():
            assert 'satisfied' in ing, f"Ingredient {name} missing 'satisfied'"
            assert 'status' in ing, f"Ingredient {name} missing 'status'"

    def test_iso4_contraction_is_theorem(self, decomp):
        """ISO4 contraction ingredient is THEOREM."""
        result = decomp.verify_all_ingredients(R_range=(0.5, 50.0), n_R=10)
        assert result['ingredients']['iso4_contraction']['status'] == 'THEOREM'

    def test_mosco_is_theorem(self, decomp):
        """Mosco convergence ingredients are THEOREM."""
        result = decomp.verify_all_ingredients(R_range=(0.5, 50.0), n_R=10)
        assert result['ingredients']['mosco_liminf']['status'] == 'THEOREM'
        assert result['ingredients']['mosco_recovery']['status'] == 'THEOREM'

    def test_uniform_gap_is_proposition(self, decomp):
        """Uniform gap ingredient is PROPOSITION."""
        result = decomp.verify_all_ingredients(R_range=(0.5, 50.0), n_R=10)
        assert result['ingredients']['uniform_gap']['status'] == 'PROPOSITION'

    def test_proof_status_steps(self, decomp):
        """Proof has 7 steps."""
        ps = decomp.proof_status()
        assert ps['n_steps'] == 7

    def test_proof_status_has_theorem_and_proposition(self, decomp):
        """Proof chain has both THEOREM and PROPOSITION steps."""
        ps = decomp.proof_status()
        assert ps['n_theorem'] >= 3
        assert ps['n_proposition'] >= 1
        assert ps['n_theorem'] + ps['n_proposition'] == ps['n_steps']

    def test_overall_status_proposition(self, decomp):
        """Overall decompactification status is PROPOSITION."""
        ps = decomp.proof_status()
        assert ps['overall_status'] == 'PROPOSITION'

    def test_identify_gaps_has_content(self, decomp):
        """identify_gaps returns meaningful content."""
        gaps = decomp.identify_gaps()
        assert len(gaps['gaps']) >= 1
        assert 'what_we_have' in gaps
        assert 'what_remains' in gaps
        assert 'distance_to_clay' in gaps

    def test_bottleneck_identified(self, decomp):
        """The bottleneck is the uniform gap bound."""
        ps = decomp.proof_status()
        assert 'uniform' in ps['bottleneck'].lower() or 'coupling' in ps['bottleneck'].lower()

    def test_status_is_proposition(self, decomp):
        """DecompactificationTheorem status is PROPOSITION."""
        s = decomp.status()
        assert s.label == 'PROPOSITION'


# ======================================================================
# 7. PhaseTransitionAbsence
# ======================================================================

class TestPhaseTransitionAbsence:
    """No transition detected, gap smooth."""

    def test_pi1_s3_is_zero(self, no_transition):
        """pi_1(S^3) = 0."""
        result = no_transition.polyakov_loop_argument()
        assert result['pi_1_S3'] == 0

    def test_no_noncontractible_loops(self, no_transition):
        """S^3 has no non-contractible loops."""
        result = no_transition.polyakov_loop_argument()
        assert result['has_noncontractible_loops'] is False

    def test_polyakov_trivial(self, no_transition):
        """Polyakov loop is trivial on S^3."""
        result = no_transition.polyakov_loop_argument()
        assert result['polyakov_loop_trivial']

    def test_no_deconfinement(self, no_transition):
        """No deconfinement transition on S^3."""
        result = no_transition.polyakov_loop_argument()
        assert result['deconfinement_transition'] is False

    def test_status_is_theorem(self, no_transition):
        """Polyakov loop argument status is THEOREM."""
        result = no_transition.polyakov_loop_argument()
        assert result['status'] == 'THEOREM'

    def test_gap_continuity_smooth(self, no_transition):
        """Gap is smooth over R range."""
        result = no_transition.gap_continuity()
        assert result['is_smooth']
        assert result['all_positive']

    def test_gap_continuity_no_jumps(self, no_transition):
        """No large jumps in gap function."""
        result = no_transition.gap_continuity(n_points=500)
        assert result['max_relative_jump'] < 0.1

    def test_verify_no_transition(self, no_transition):
        """Combined verification: no phase transition."""
        result = no_transition.verify_no_transition()
        assert result['no_transition']

    def test_verify_no_transition_summary(self, no_transition):
        """Verification summary mentions pi_1."""
        result = no_transition.verify_no_transition()
        assert 'pi_1' in result['summary']

    def test_center_symmetry_unbroken(self, no_transition):
        """Center symmetry Z_N is unbroken."""
        result = no_transition.polyakov_loop_argument()
        assert 'Z_2' in result['center_symmetry']


# ======================================================================
# 8. ClayMillenniumConnection
# ======================================================================

class TestClayMillenniumConnection:
    """Requirements mapped, gap identified clearly."""

    def test_clay_requirements(self, clay):
        """Clay requirements are correctly stated."""
        req = clay.clay_requirements()
        assert req['requirements']['spacetime'] == 'R^4 (4-dimensional Euclidean space)'
        assert 'mass_gap' in req['requirements']
        assert 'Wightman' in req['requirements']['axioms']

    def test_what_we_have(self, clay):
        """Our results are correctly reported."""
        have = clay.what_we_have()
        assert have['results']['gap_at_each_R']['status'] == 'THEOREM'
        assert have['results']['iso4_recovery']['status'] == 'THEOREM'
        assert have['results']['no_phase_transition']['status'] == 'THEOREM'

    def test_rg_pipeline_result(self, clay):
        """RG pipeline result is NUMERICAL."""
        have = clay.what_we_have()
        assert have['results']['rg_program']['status'] == 'NUMERICAL'

    def test_what_remains_identified(self, clay):
        """The remaining gap is identified."""
        rem = clay.what_remains()
        assert 'PROPOSITION' in rem['status_of_gap']

    def test_three_paths(self, clay):
        """Three potential paths are identified."""
        rem = clay.what_remains()
        assert len(rem['potential_paths']) == 3

    def test_path_a_is_postulate(self, clay):
        """Path A (ontological) is POSTULATE."""
        rem = clay.what_remains()
        path_a = [p for p in rem['potential_paths'] if p['path'] == 'Path A (Ontological)'][0]
        assert 'POSTULATE' in path_a['status']

    def test_gap_analysis_has_proof_chain(self, clay):
        """Gap analysis includes proof chain."""
        analysis = clay.gap_analysis()
        assert 'proof_chain' in analysis
        assert analysis['proof_chain']['n_steps'] == 7

    def test_gap_analysis_honest(self, clay):
        """Gap analysis is honest about status."""
        analysis = clay.gap_analysis()
        assert 'PROPOSITION' in analysis['honest_assessment']
        assert 'not THEOREM' in analysis['honest_assessment']

    def test_status_is_proposition(self, clay):
        """Clay connection status is PROPOSITION."""
        s = clay.status()
        assert s.label == 'PROPOSITION'

    def test_clay_distance(self, clay):
        """Distance assessment mentions uniform gap bound."""
        analysis = clay.gap_analysis()
        assert 'uniform' in analysis['distance_assessment']['bottleneck'].lower()


# ======================================================================
# 9. R-scan: gap(R) for R in [0.1, 100] fm
# ======================================================================

class TestRScan:
    """Scan gap(R) over a wide range of R values."""

    def test_gap_always_positive(self, gap_bound):
        """Gap > 0 for every R in [0.1, 100] fm."""
        R_vals = np.logspace(-1, 2, 100)
        for R in R_vals:
            assert gap_bound.gap_at_R(R)['gap_MeV'] > 0

    def test_gap_floor_at_lambda_qcd(self, gap_bound):
        """Gap never goes below Lambda_QCD."""
        R_vals = np.logspace(-1, 2, 100)
        for R in R_vals:
            gap = gap_bound.gap_at_R(R)['gap_MeV']
            assert gap >= 200.0 - 1e-10

    def test_gap_monotone_small_R(self, gap_bound):
        """For small R, gap decreases with R (1/R behavior)."""
        R_vals = [0.1, 0.2, 0.3, 0.5]
        gaps = [gap_bound.gap_at_R(R)['gap_MeV'] for R in R_vals]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_gap_constant_large_R(self, gap_bound):
        """For large R, gap is approximately Lambda_QCD."""
        for R in [20.0, 50.0, 100.0]:
            gap = gap_bound.gap_at_R(R)['gap_MeV']
            assert abs(gap - 200.0) < 1.0


# ======================================================================
# 10. Edge cases
# ======================================================================

class TestEdgeCases:
    """R -> 0 (kinematic limit) and R -> large (flat limit)."""

    def test_small_R_large_gap(self, gap_bound):
        """At R = 0.01 fm, gap is very large."""
        gap = gap_bound.gap_at_R(0.01)['gap_MeV']
        assert gap > 30000  # 2 * 197.3 / 0.01 ~ 39460

    def test_large_R_gap_equals_lambda(self, gap_bound):
        """At R = 10^4 fm, gap = Lambda_QCD."""
        gap = gap_bound.gap_at_R(1e4)['gap_MeV']
        assert abs(gap - 200.0) < 1e-5

    def test_iso4_error_tiny_at_large_R(self, iso4):
        """ISO(4) contraction error is negligible at large R."""
        err = iso4.commutator_error(1e6)['max_error']
        assert err < 1e-10

    def test_stereographic_flat_at_large_R(self, mosco):
        """Stereographic map gives flat metric at large R."""
        data = mosco.stereographic_map(1e6)
        assert data['converges_to_flat']

    def test_os_axioms_at_extreme_R(self, os_limit):
        """OS axioms satisfied at R = 0.1 and R = 100."""
        r1 = os_limit.verify_at_R(0.1)
        r2 = os_limit.verify_at_R(100.0)
        assert r1['all_satisfied']
        assert r2['all_satisfied']

    def test_phase_transition_absence_wide_range(self, no_transition):
        """No phase transition over R = 0.01 to 1000 fm."""
        result = no_transition.verify_no_transition(R_range=(0.01, 1000.0))
        assert result['no_transition']


# ======================================================================
# 11. Integration tests
# ======================================================================

class TestIntegration:
    """End-to-end integration of all components."""

    def test_full_decompactification_pipeline(self, decomp):
        """Full pipeline runs without errors."""
        result = decomp.verify_all_ingredients(R_range=(1.0, 20.0), n_R=5)
        assert 'ingredients' in result
        assert 'all_satisfied' in result
        assert result['overall_status'] == 'PROPOSITION'

    def test_all_statuses_consistent(self, decomp):
        """All status labels are valid."""
        valid = {'THEOREM', 'PROPOSITION', 'NUMERICAL', 'CONJECTURE', 'POSTULATE'}

        ps = decomp.proof_status()
        for step in ps['steps']:
            assert step['status'] in valid

        s = decomp.status()
        assert s.label in valid

    def test_claim_status_objects(self, gap_bound, mosco, iso4, os_limit, wightman,
                                  decomp, no_transition, clay):
        """All status() methods return ClaimStatus."""
        statuses = [
            gap_bound.status(),
            mosco.status(),
            iso4.status(),
            os_limit.status(),
            wightman.status(),
            decomp.status(),
            no_transition.status(),
            clay.status(),
        ]
        for s in statuses:
            assert isinstance(s, ClaimStatus)
            assert s.label in {'THEOREM', 'PROPOSITION', 'NUMERICAL', 'CONJECTURE', 'POSTULATE'}
            assert len(s.statement) > 10
            assert len(s.evidence) > 10
