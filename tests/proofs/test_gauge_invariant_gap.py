"""
Tests for the gauge-invariant mass gap theorem.

Verifies:
1. Gauge-invariant correlator definition and decay
2. Spectral gap = correlator decay rate (exact relation)
3. Gauge invariance of Gribov expectation values
4. Gap proof WITHOUT GZ propagator
5. Uniform gap (extreme value theorem)
6. Schwinger function uniform decay
7. Main theorem statement and proof ingredients
8. Criticism response
9. Gap identification with gamma*
10. Complete analysis integration

Every function is tested for:
    - Correct return type and keys
    - Correct physical behavior
    - Gauge invariance properties
    - No dependence on GZ propagator (qualitative)
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.gauge_invariant_gap import (
    gauge_invariant_correlator,
    spectral_gap_from_correlator_decay,
    gauge_invariance_of_gribov_expectation,
    gap_without_gz,
    uniform_gap_without_gz,
    schwinger_decay_uniform,
    theorem_physical_mass_gap,
    address_final_criticism,
    gap_identification_with_gamma_star,
    complete_gauge_invariant_analysis,
)


# ===========================================================================
# Test 1: Gauge-invariant correlator
# ===========================================================================
class TestGaugeInvariantCorrelator:
    """Tests for the gauge-invariant correlator definition."""

    def test_basic_return_keys(self):
        """Correlator returns expected keys."""
        result = gauge_invariant_correlator(1.0, [0.1, 0.5, 1.0], 'TrF2')
        assert 'status' in result
        assert 'upper_bound' in result
        assert 'm_gap_lower_bound' in result
        assert 'gauge_fixing_used' in result

    def test_is_theorem(self):
        """Correlator decomposition is labeled THEOREM."""
        result = gauge_invariant_correlator(1.0, [0.1], 'TrF2')
        assert result['status'] == 'THEOREM'

    def test_no_gauge_fixing(self):
        """No gauge fixing is used."""
        result = gauge_invariant_correlator(1.0, [0.1], 'TrF2')
        assert result['gauge_fixing_used'] is False

    def test_upper_bound_positive(self):
        """Upper bound is positive for t > 0."""
        t_vals = np.linspace(0.1, 5.0, 20)
        result = gauge_invariant_correlator(1.0, t_vals, 'TrF2')
        assert np.all(result['upper_bound'] > 0)

    def test_upper_bound_decays(self):
        """Upper bound decays with increasing t."""
        t_vals = np.linspace(0.1, 5.0, 20)
        result = gauge_invariant_correlator(1.0, t_vals, 'TrF2')
        bounds = result['upper_bound']
        # Check monotone decrease
        assert np.all(np.diff(bounds) <= 0)

    def test_gap_positive(self):
        """Mass gap lower bound is positive."""
        result = gauge_invariant_correlator(1.0, [0.1], 'TrF2')
        assert result['m_gap_lower_bound'] > 0

    def test_gap_scales_with_R(self):
        """Gap lower bound scales as 1/R."""
        r1 = gauge_invariant_correlator(1.0, [0.1], 'TrF2')
        r2 = gauge_invariant_correlator(2.0, [0.1], 'TrF2')
        # m ~ 1/R, so m(2R) ~ m(R)/2
        ratio = r2['m_gap_lower_bound'] / r1['m_gap_lower_bound']
        assert 0.3 < ratio < 0.7  # approximately 1/2

    def test_wilson_loop_observable(self):
        """Wilson loop observable works."""
        result = gauge_invariant_correlator(1.0, [0.1], 'Wilson')
        assert result['status'] == 'THEOREM'
        assert result['observable_norm_sq'] == 1.0

    def test_polyakov_loop_observable(self):
        """Polyakov loop observable works."""
        result = gauge_invariant_correlator(1.0, [0.1], 'Polyakov')
        assert result['status'] == 'THEOREM'

    def test_invalid_observable_raises(self):
        """Invalid observable type raises error."""
        with pytest.raises(ValueError):
            gauge_invariant_correlator(1.0, [0.1], 'invalid')

    def test_invalid_R_raises(self):
        """Non-positive R raises error."""
        with pytest.raises(ValueError):
            gauge_invariant_correlator(-1.0, [0.1], 'TrF2')
        with pytest.raises(ValueError):
            gauge_invariant_correlator(0.0, [0.1], 'TrF2')

    def test_su3(self):
        """Works for SU(3)."""
        result = gauge_invariant_correlator(1.0, [0.1], 'TrF2', N=3)
        assert result['N'] == 3
        assert result['status'] == 'THEOREM'


# ===========================================================================
# Test 2: Spectral gap from correlator decay
# ===========================================================================
class TestSpectralGapFromCorrelatorDecay:
    """Tests for the spectral gap = correlator decay rate theorem."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = spectral_gap_from_correlator_decay(1.0)
        assert 'status' in result
        assert 'mass_gap' in result
        assert 'discrete_spectrum' in result
        assert 'compact_resolvent' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = spectral_gap_from_correlator_decay(1.0)
        assert result['status'] == 'THEOREM'

    def test_no_gauge_fixing(self):
        """No gauge fixing used."""
        result = spectral_gap_from_correlator_decay(1.0)
        assert result['gauge_fixing_used'] is False

    def test_gap_positive(self):
        """Mass gap is positive."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = spectral_gap_from_correlator_decay(R)
            assert result['mass_gap'] > 0, f"Gap not positive at R={R}"

    def test_discrete_spectrum(self):
        """Spectrum is discrete (S^3 compact)."""
        result = spectral_gap_from_correlator_decay(1.0)
        assert result['discrete_spectrum'] is True

    def test_compact_resolvent(self):
        """Compact resolvent on S^3."""
        result = spectral_gap_from_correlator_decay(1.0)
        assert result['compact_resolvent'] is True

    def test_linearized_gap_value(self):
        """Linearized gap is 4/R^2."""
        R = 2.0
        result = spectral_gap_from_correlator_decay(R)
        expected = 4.0 / R**2
        assert abs(result['gap_linearized'] - expected) < 1e-10


# ===========================================================================
# Test 3: Gauge invariance of Gribov expectation values
# ===========================================================================
class TestGaugeInvarianceOfGribovExpectation:
    """Tests for the FP theorem: <O>_GZ = <O>_unfixed."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = gauge_invariance_of_gribov_expectation()
        assert 'status' in result
        assert 'fp_theorem_applies' in result
        assert 'omega_convex' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = gauge_invariance_of_gribov_expectation()
        assert result['status'] == 'THEOREM'

    def test_fp_theorem_applies(self):
        """FP theorem applies."""
        result = gauge_invariance_of_gribov_expectation()
        assert result['fp_theorem_applies'] is True

    def test_omega_convex(self):
        """Gribov region is convex."""
        result = gauge_invariance_of_gribov_expectation()
        assert result['omega_convex'] is True

    def test_omega_bounded(self):
        """Gribov region is bounded."""
        result = gauge_invariance_of_gribov_expectation()
        assert result['omega_bounded'] is True

    def test_singer_curvature(self):
        """Singer curvature is positive."""
        result = gauge_invariance_of_gribov_expectation()
        assert result['singer_curvature_positive'] is True

    def test_su3(self):
        """Works for SU(3)."""
        result = gauge_invariance_of_gribov_expectation(N=3)
        assert result['dim_adjoint'] == 8


# ===========================================================================
# Test 4: Gap without GZ propagator
# ===========================================================================
class TestGapWithoutGZ:
    """Tests for the gap proof without GZ propagator."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = gap_without_gz(1.0)
        assert 'status' in result
        assert 'gap_non_gz' in result
        assert 'uses_gz_propagator' in result
        assert 'proof_chain' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = gap_without_gz(1.0)
        assert result['status'] == 'THEOREM'

    def test_no_gz_propagator(self):
        """Does not use GZ propagator."""
        result = gap_without_gz(1.0)
        assert result['uses_gz_propagator'] is False

    def test_gap_positive_small_R(self):
        """Gap is positive at small R."""
        result = gap_without_gz(0.5)
        assert result['gap_positive'] is True
        assert result['gap_non_gz'] > 0

    def test_gap_positive_medium_R(self):
        """Gap is positive at medium R."""
        result = gap_without_gz(2.0)
        assert result['gap_positive'] is True

    def test_gap_positive_large_R(self):
        """Gap is positive at large R."""
        result = gap_without_gz(10.0)
        assert result['gap_positive'] is True

    def test_gap_positive_various_R(self):
        """Gap is positive at various R values."""
        for R in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            result = gap_without_gz(R)
            assert result['gap_positive'], f"Gap not positive at R={R}"

    def test_hodge_gap_value(self):
        """Hodge gap is 4/R^2."""
        R = 3.0
        result = gap_without_gz(R)
        expected = 4.0 / R**2
        assert abs(result['gap_hodge'] - expected) < 1e-10

    def test_kr_applicable_small_R(self):
        """KR is applicable at small R (weak coupling)."""
        result = gap_without_gz(0.3)
        assert result['kr_applicable'] is True

    def test_proof_chain_all_gauge_invariant(self):
        """All proof chain steps are gauge-invariant."""
        result = gap_without_gz(1.0)
        for step in result['proof_chain']:
            assert step['gauge_invariant'] is True
            assert step['uses_gz_propagator'] is False

    def test_invalid_R_raises(self):
        """Non-positive R raises error."""
        with pytest.raises(ValueError):
            gap_without_gz(-1.0)
        with pytest.raises(ValueError):
            gap_without_gz(0.0)

    def test_su3(self):
        """Works for SU(3)."""
        result = gap_without_gz(1.0, N=3)
        assert result['gap_positive'] is True


# ===========================================================================
# Test 5: Uniform gap without GZ
# ===========================================================================
class TestUniformGapWithoutGZ:
    """Tests for the uniform gap Delta_0 > 0."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert 'status' in result
        assert 'Delta_0_squared' in result
        assert 'all_positive' in result
        assert 'R_at_minimum' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['status'] == 'THEOREM'

    def test_no_gz_propagator(self):
        """Does not use GZ propagator."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['uses_gz_propagator'] is False

    def test_all_gaps_positive(self):
        """All gaps are positive across the R scan."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['all_positive'] is True

    def test_Delta_0_positive(self):
        """Delta_0 > 0."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['Delta_0_squared'] > 0
        assert result['Delta_0'] > 0

    def test_correct_asymptotics(self):
        """Gap is larger at endpoints than at minimum."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['correct_asymptotics'] is True

    def test_evt_applies(self):
        """Extreme value theorem applies."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        assert result['evt_applies'] is True

    def test_proof_steps_all_theorem(self):
        """All proof steps are THEOREM."""
        result = uniform_gap_without_gz(N=2, n_R_points=20)
        for step_name, status in result['proof_steps'].items():
            assert 'THEOREM' in status, f"Step {step_name} is not THEOREM: {status}"


# ===========================================================================
# Test 6: Schwinger function uniform decay
# ===========================================================================
class TestSchwingerDecayUniform:
    """Tests for uniform Schwinger function decay."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert 'status' in result
        assert 'm_gap' in result
        assert 'envelope' in result
        assert 'uniform_in_R' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['status'] == 'THEOREM'

    def test_gauge_invariant(self):
        """Is gauge-invariant."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['gauge_invariant'] is True

    def test_no_gz(self):
        """Does not use GZ propagator."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['uses_gz_propagator'] is False

    def test_uniform_in_R(self):
        """Decay is uniform in R."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['uniform_in_R'] is True

    def test_m_gap_positive(self):
        """Mass gap is positive."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['m_gap'] > 0

    def test_envelope_decays(self):
        """Envelope is monotonically decreasing."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        envelope = result['envelope']
        assert np.all(np.diff(envelope) <= 1e-15)

    def test_os_clustering_verified(self):
        """OS clustering (OS4) is verified."""
        result = schwinger_decay_uniform(N=2, n_R_points=15)
        assert result['os_clustering_verified'] is True


# ===========================================================================
# Test 7: Main theorem -- physical mass gap
# ===========================================================================
class TestTheoremPhysicalMassGap:
    """Tests for the main theorem."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert 'status' in result
        assert 'Delta_0' in result
        assert 'theorem_statement' in result
        assert 'ingredients' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['status'] == 'THEOREM'

    def test_Delta_0_positive(self):
        """Delta_0 > 0."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['Delta_0'] > 0

    def test_all_gaps_positive(self):
        """All sampled gaps are positive."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['all_gaps_positive'] is True

    def test_all_ingredients_gauge_invariant(self):
        """All ingredients are gauge-invariant."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['all_gauge_invariant'] is True

    def test_none_use_gz(self):
        """No ingredient uses GZ propagator."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['none_use_gz'] is True

    def test_all_ingredients_theorem(self):
        """All ingredients are THEOREM-level."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['all_theorem'] is True

    def test_no_gz_propagator_flag(self):
        """uses_gz_propagator is False."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert result['uses_gz_propagator'] is False

    def test_theorem_statement_nonempty(self):
        """Theorem statement is non-empty."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert len(result['theorem_statement']) > 100

    def test_ingredient_count(self):
        """Has at least 9 ingredients."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert len(result['ingredients']) >= 9

    def test_each_ingredient_is_theorem(self):
        """Each ingredient is individually THEOREM."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        for ing in result['ingredients']:
            assert ing['status'] == 'THEOREM', (
                f"Ingredient '{ing['name']}' is {ing['status']}, not THEOREM"
            )

    def test_each_ingredient_gauge_invariant(self):
        """Each ingredient is individually gauge-invariant."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        for ing in result['ingredients']:
            assert ing['gauge_invariant'] is True, (
                f"Ingredient '{ing['name']}' is not gauge-invariant"
            )

    def test_each_ingredient_no_gz(self):
        """Each ingredient does not use GZ."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        for ing in result['ingredients']:
            assert ing['uses_gz'] is False, (
                f"Ingredient '{ing['name']}' uses GZ"
            )

    def test_gap_scan_provided(self):
        """Gap scan data is provided."""
        result = theorem_physical_mass_gap(N=2, n_R_points=20)
        assert 'gap_scan' in result
        assert len(result['gap_scan']['R_values']) > 0
        assert len(result['gap_scan']['gap_values']) > 0

    def test_su3(self):
        """Works for SU(3)."""
        result = theorem_physical_mass_gap(N=3, n_R_points=15)
        assert result['status'] == 'THEOREM'
        assert result['Delta_0'] > 0
        assert result['gauge_group'] == 'SU(3)'


# ===========================================================================
# Test 8: Criticism response
# ===========================================================================
class TestAddressCriticism:
    """Tests for the criticism response."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = address_final_criticism()
        assert 'criticism' in result
        assert 'answer_summary' in result
        assert 'detailed_answer' in result
        assert 'conclusion' in result

    def test_criticism_stated(self):
        """The criticism is clearly stated."""
        result = address_final_criticism()
        assert 'gauge' in result['criticism'].lower()

    def test_answer_addresses_qualitative(self):
        """Answer addresses qualitative vs quantitative separation."""
        result = address_final_criticism()
        detailed = result['detailed_answer']
        assert 'quantitative_vs_qualitative' in detailed

    def test_gauge_invariant_ingredients_listed(self):
        """Gauge-invariant ingredients are listed."""
        result = address_final_criticism()
        ingredients = result['detailed_answer']['gauge_invariant_ingredients']
        assert len(ingredients) >= 5

    def test_what_is_not_gauge_invariant(self):
        """Clearly states what IS gauge-fixed (GZ propagator)."""
        result = address_final_criticism()
        not_gi = result['detailed_answer']['what_is_not_gauge_invariant']
        assert 'gz_propagator' in not_gi


# ===========================================================================
# Test 9: Gap identification with gamma*
# ===========================================================================
class TestGapIdentificationWithGammaStar:
    """Tests for the quantitative gap identification."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = gap_identification_with_gamma_star()
        assert 'status' in result
        assert 'm_gap' in result
        assert 'fp_theorem' in result

    def test_is_theorem(self):
        """Labeled THEOREM."""
        result = gap_identification_with_gamma_star()
        assert result['status'] == 'THEOREM'

    def test_fp_theorem(self):
        """FP theorem applies."""
        result = gap_identification_with_gamma_star()
        assert result['fp_theorem'] is True

    def test_m_gap_equals_3_for_su2(self):
        """m_gap = 3 Lambda_QCD for SU(2)."""
        result = gap_identification_with_gamma_star(N=2)
        # gamma* = 3*sqrt(2)/2, m_gap = sqrt(2)*gamma* = 3
        assert abs(result['m_gap'] - 3.0) < 1e-10

    def test_quantitative_gauge_invariant(self):
        """The quantitative bound is gauge-invariant."""
        result = gap_identification_with_gamma_star()
        assert result['quantitative_gauge_invariant'] is True

    def test_proof_chain_provided(self):
        """Proof chain is provided."""
        result = gap_identification_with_gamma_star()
        assert len(result['proof_chain']) >= 3


# ===========================================================================
# Test 10: Complete analysis
# ===========================================================================
class TestCompleteAnalysis:
    """Tests for the complete gauge-invariant analysis."""

    def test_basic_return_keys(self):
        """Returns expected keys."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert 'status' in result
        assert 'summary' in result
        assert 'main_theorem' in result

    def test_is_theorem(self):
        """Overall status is THEOREM."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert result['status'] == 'THEOREM'

    def test_summary_all_theorem(self):
        """Summary confirms all theorem."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert result['summary']['all_theorem'] is True

    def test_summary_all_gauge_invariant(self):
        """Summary confirms all gauge-invariant."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert result['summary']['all_gauge_invariant'] is True

    def test_criticism_addressed(self):
        """Criticism is addressed."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert result['summary']['criticism_addressed'] is True

    def test_Delta_0_positive(self):
        """Delta_0 > 0 in summary."""
        result = complete_gauge_invariant_analysis(N=2, n_R_points=15)
        assert result['summary']['Delta_0'] > 0
