"""
Tests for the functional measure on S^3.

Verifies:
    - Lattice partition function estimation
    - Two-point correlator computation
    - Mass gap extraction from correlators
    - Continuum limit analysis
    - Reflection positivity numerical check
    - Measure properties (well-definedness)

Note: Monte Carlo sample counts are LOW for fast tests.
"""

import pytest
import numpy as np
from yang_mills_s3.qft.functional_measure import FunctionalMeasure


class TestMeasureProperties:
    """Properties that make the YM measure well-defined on S^3."""

    def test_compact_spatial_manifold(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['compact_spatial_manifold'] is True

    def test_finite_volume(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        expected = 2 * np.pi**2  # Volume of unit S^3
        assert abs(result['finite_volume'] - expected) < 1e-10

    def test_haar_measure_exists(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['haar_measure_exists'] is True
        assert result['haar_measure_unique'] is True

    def test_action_bounded_below(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['action_bounded_below'] is True
        assert result['action_minimum'] == 0.0

    def test_no_ir_divergence(self):
        """On S^3, there are no infrared divergences (finite volume)."""
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['no_ir_divergence'] is True

    def test_no_boundary_terms(self):
        """S^3 is compact without boundary."""
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['no_boundary_terms'] is True

    def test_gauge_orbits_compact(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert result['gauge_orbits_compact'] is True

    def test_status_is_theorem(self):
        result = FunctionalMeasure.measure_properties(N=2, R=1.0)
        assert 'THEOREM' in result['status']

    def test_volume_scales_with_R(self):
        """Volume of S^3(R) = 2*pi^2*R^3."""
        for R in [0.5, 1.0, 2.0]:
            result = FunctionalMeasure.measure_properties(N=2, R=R)
            expected = 2 * np.pi**2 * R**3
            assert abs(result['finite_volume'] - expected) < 1e-10


class TestPartitionFunction:
    """Monte Carlo estimation of the lattice partition function."""

    @pytest.fixture
    def fm(self):
        """Functional measure with small sample count for speed."""
        return FunctionalMeasure(N=2, R=1.0, beta=2.0)

    def test_partition_function_runs(self, fm):
        """Monte Carlo estimation should complete without errors."""
        result = fm.partition_function_lattice(n_samples=20, n_therm=2, epsilon=0.3)
        assert 'plaquette_avg' in result
        assert 'action_avg' in result
        assert 'action_std' in result
        assert 'acceptance_rate' in result

    def test_plaquette_avg_in_range(self, fm):
        """Plaquette average should be between 0 and 1."""
        result = fm.partition_function_lattice(n_samples=20, n_therm=2, epsilon=0.3)
        assert 0 < result['plaquette_avg'] < 1.0

    def test_action_avg_positive(self, fm):
        """Average action should be positive."""
        result = fm.partition_function_lattice(n_samples=20, n_therm=2, epsilon=0.3)
        assert result['action_avg'] > 0

    def test_action_std_positive(self, fm):
        """Action fluctuations should be non-zero."""
        result = fm.partition_function_lattice(n_samples=20, n_therm=2, epsilon=0.3)
        assert result['action_std'] > 0

    def test_acceptance_rate_reasonable(self, fm):
        """Acceptance rate should be in a reasonable range."""
        result = fm.partition_function_lattice(n_samples=20, n_therm=2, epsilon=0.3)
        assert 0.05 < result['acceptance_rate'] < 0.99


class TestMassGapExtraction:
    """Mass gap extraction from correlator decay."""

    def test_mass_gap_from_exponential_decay(self):
        """
        Given exponentially decaying correlators,
        should extract the correct mass gap.
        """
        m_true = 0.5
        t_values = [0, 1, 2, 3, 4, 5]
        correlators = [np.exp(-m_true * t) for t in t_values]

        result = FunctionalMeasure.mass_gap_from_correlator(correlators, t_values)
        assert abs(result['mass_gap'] - m_true) < 0.01, \
            f"Extracted mass {result['mass_gap']:.4f} should be ~{m_true}"

    def test_mass_gap_positive_for_decaying_correlator(self):
        """Decaying correlator should give positive mass gap."""
        correlators = [1.0, 0.6, 0.36, 0.22, 0.13]
        t_values = [0, 1, 2, 3, 4]
        result = FunctionalMeasure.mass_gap_from_correlator(correlators, t_values)
        assert result['mass_gap'] > 0

    def test_effective_masses_computed(self):
        """Should compute effective masses at each time step."""
        correlators = [1.0, 0.5, 0.25, 0.125]
        t_values = [0, 1, 2, 3]
        result = FunctionalMeasure.mass_gap_from_correlator(correlators, t_values)
        # For pure exponential with m=ln(2), all effective masses should be ln(2)
        for m_eff in result['effective_masses']:
            if np.isfinite(m_eff):
                assert abs(m_eff - np.log(2)) < 1e-10

    def test_quality_good_for_clear_gap(self):
        """Quality should be 'good' when gap is clearly positive."""
        correlators = [1.0, 0.3, 0.09, 0.027]
        t_values = [0, 1, 2, 3]
        result = FunctionalMeasure.mass_gap_from_correlator(correlators, t_values)
        assert result['quality'] == 'good'

    def test_flat_correlator_no_gap(self):
        """Flat correlator (no decay) should give poor quality."""
        correlators = [1.0, 1.0, 1.0, 1.0]
        t_values = [0, 1, 2, 3]
        result = FunctionalMeasure.mass_gap_from_correlator(correlators, t_values)
        # No decay -> no valid effective mass -> gap = 0 or poor quality
        assert result['mass_gap'] == 0.0 or result['quality'] == 'poor'


class TestContinuumLimitAnalysis:
    """Continuum limit analysis as beta -> infinity."""

    def test_all_positive_gaps(self):
        """Should detect when all gaps are positive."""
        betas = [1.0, 2.0, 4.0, 8.0]
        gaps = [0.8, 0.6, 0.4, 0.3]
        result = FunctionalMeasure.continuum_limit_analysis(betas, gaps)
        assert result['gaps_positive'] is True

    def test_monotonic_decrease(self):
        """Should detect monotonic decrease of lattice-unit gap."""
        betas = [1.0, 2.0, 4.0]
        gaps = [0.8, 0.5, 0.3]
        result = FunctionalMeasure.continuum_limit_analysis(betas, gaps)
        assert result['monotonic_trend'] is True

    def test_extrapolated_gap_positive(self):
        """Extrapolated gap should be positive for good data."""
        betas = [1.0, 2.0, 4.0, 8.0]
        gaps = [0.8, 0.6, 0.5, 0.45]
        result = FunctionalMeasure.continuum_limit_analysis(betas, gaps)
        assert result['extrapolated_gap'] > 0

    def test_quality_good_for_positive_extrapolation(self):
        betas = [1.0, 2.0, 4.0, 8.0]
        gaps = [0.8, 0.6, 0.5, 0.45]
        result = FunctionalMeasure.continuum_limit_analysis(betas, gaps)
        assert result['scaling_quality'] == 'good'

    def test_single_beta(self):
        """Should handle single beta point gracefully."""
        result = FunctionalMeasure.continuum_limit_analysis([2.0], [0.5])
        assert result['gaps_positive'] is True
        assert result['extrapolated_gap'] == 0.5


class TestReflectionPositivityCheck:
    """Numerical verification of reflection positivity."""

    def test_rp_check_runs(self):
        """Reflection positivity check should complete without errors."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.reflection_positivity_check(n_samples=10, n_therm=2, epsilon=0.3)

        assert 'all_positive' in result
        assert 'min_value' in result
        assert 'mean_value' in result
        assert 'n_violations' in result
        assert 'status' in result

    def test_rp_status_mentions_theorem(self):
        """Status should reference the Osterwalder-Seiler theorem."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.reflection_positivity_check(n_samples=10, n_therm=2, epsilon=0.3)
        assert 'Osterwalder-Seiler' in result['status']

    def test_rp_mean_positive(self):
        """Mean inner product should be positive."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.reflection_positivity_check(n_samples=10, n_therm=2, epsilon=0.3)
        assert result['mean_value'] > 0, \
            f"Mean RP inner product should be > 0, got {result['mean_value']}"

    def test_rp_min_not_deeply_negative(self):
        """
        Minimum inner product should not be deeply negative.
        Small numerical fluctuations are OK; large violations are not.
        """
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.reflection_positivity_check(n_samples=10, n_therm=2, epsilon=0.3)
        assert result['min_value'] > -0.1, \
            f"Min RP value {result['min_value']} too negative"


class TestTwoPointCorrelator:
    """Two-point correlator estimation via MC."""

    def test_correlator_runs(self):
        """Two-point correlator should complete without errors."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.correlator_two_point(t_sep=0, n_samples=10, n_therm=2, epsilon=0.3)
        assert 'correlator' in result
        assert 'error' in result

    def test_correlator_at_zero_positive(self):
        """Correlator at t=0 (autocorrelation) should be positive."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.correlator_two_point(t_sep=0, n_samples=10, n_therm=2, epsilon=0.3)
        assert result['correlator'] > 0

    def test_correlator_is_finite(self):
        """Correlator values should be finite."""
        fm = FunctionalMeasure(N=2, R=1.0, beta=2.0)
        result = fm.correlator_two_point(t_sep=1, n_samples=10, n_therm=2, epsilon=0.3)
        assert np.isfinite(result['correlator'])
        assert np.isfinite(result['error'])
