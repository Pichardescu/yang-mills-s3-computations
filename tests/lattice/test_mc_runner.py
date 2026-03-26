"""
Tests for the Monte Carlo simulation runner.

These are integration tests that verify the full MC pipeline produces
valid numerical results. Uses minimal statistics for speed.

STATUS: NUMERICAL
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.mc_runner import (
    run_plaquette_scan,
    run_wilson_loops,
    run_mass_gap,
    _fit_mass_gap,
)


class TestFitMassGap:
    """Test the mass gap fitting utility."""

    def test_fit_synthetic_exponential(self):
        """Fit to clean exponential should recover mass."""
        d = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        m_true = 3.0
        A_true = 0.1
        c = A_true * np.exp(-m_true * d)

        result = _fit_mass_gap(d, c)
        assert abs(result['mass_gap'] - m_true) < 0.5, \
            f"Expected m~{m_true}, got {result['mass_gap']}"

    def test_fit_noisy_exponential(self):
        """Fit to noisy exponential gives positive mass."""
        rng = np.random.default_rng(42)
        d = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        c = 0.1 * np.exp(-2.0 * d) + rng.normal(0, 0.001, 5)
        c = np.maximum(c, 1e-10)

        result = _fit_mass_gap(d, c)
        assert result['mass_gap'] > 0

    def test_fit_all_zero(self):
        """All-zero correlator is handled gracefully."""
        d = np.array([0.2, 0.4, 0.6])
        c = np.zeros(3)
        result = _fit_mass_gap(d, c)
        assert 'mass_gap' in result

    def test_fit_negative_correlator(self):
        """Negative correlator values are handled."""
        d = np.array([0.2, 0.4, 0.6])
        c = np.array([-0.1, -0.2, -0.3])
        result = _fit_mass_gap(d, c)
        assert 'mass_gap' in result


class TestPlaquetteScan:
    """Plaquette scan produces valid results."""

    def test_scan_returns_list(self):
        """run_plaquette_scan returns a list of dicts."""
        results = run_plaquette_scan(
            [2.0, 4.0],
            n_therm=10, n_measure=5, n_skip=1,
            seed=42, verbose=False
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_scan_has_required_keys(self):
        """Each result has the expected keys."""
        results = run_plaquette_scan(
            [4.0],
            n_therm=10, n_measure=5, n_skip=1,
            seed=42, verbose=False
        )
        r = results[0]
        assert 'beta' in r
        assert 'plaq_mean' in r
        assert 'plaq_err' in r
        assert 'action' in r

    def test_plaquette_in_valid_range(self):
        """Plaquette average is between 0 and 1."""
        results = run_plaquette_scan(
            [2.0],
            n_therm=15, n_measure=5, n_skip=1,
            seed=42, verbose=False
        )
        P = results[0]['plaq_mean']
        assert 0.0 < P < 1.0, f"<P> = {P}"

    def test_plaquette_ordering(self):
        """Higher beta gives higher plaquette."""
        results = run_plaquette_scan(
            [1.0, 8.0],
            n_therm=15, n_measure=5, n_skip=1,
            seed=42, verbose=False
        )
        assert results[1]['plaq_mean'] > results[0]['plaq_mean'], \
            f"beta=8: {results[1]['plaq_mean']:.4f} <= beta=1: {results[0]['plaq_mean']:.4f}"


class TestWilsonLoops:
    """Wilson loop measurement produces valid results."""

    def test_returns_dict(self):
        """run_wilson_loops returns dict with required keys."""
        result = run_wilson_loops(
            beta=4.0, n_therm=10, n_measure=5, n_skip=1,
            max_loop_length=4, seed=42, verbose=False
        )
        assert 'wilson_loops' in result
        assert 'string_tension' in result
        assert 'beta' in result

    def test_wilson_loops_present(self):
        """Wilson loops are measured for at least length 3."""
        result = run_wilson_loops(
            beta=4.0, n_therm=10, n_measure=5, n_skip=1,
            max_loop_length=4, seed=42, verbose=False
        )
        assert 3 in result['wilson_loops']
        assert result['wilson_loops'][3]['W_mean'] > 0

    def test_wilson_loop_3_bounded(self):
        """W(3) is bounded by 1 for SU(2)."""
        result = run_wilson_loops(
            beta=4.0, n_therm=15, n_measure=5, n_skip=1,
            max_loop_length=4, seed=42, verbose=False
        )
        W3 = result['wilson_loops'][3]['W_mean']
        assert abs(W3) <= 1.0 + 0.01, f"|W(3)| = {abs(W3)}"


class TestMassGap:
    """Mass gap extraction produces valid results."""

    def test_returns_dict(self):
        """run_mass_gap returns dict with required keys."""
        result = run_mass_gap(
            beta=4.0, n_therm=10, n_measure=5, n_skip=1,
            n_bins=8, seed=42, verbose=False
        )
        assert 'gap_fit' in result
        assert 'analytical_gap' in result
        assert 'plaquette' in result
        assert 'correlator' in result

    def test_analytical_gap_correct(self):
        """Analytical gap = 2/R = 2 for R=1."""
        result = run_mass_gap(
            beta=4.0, n_therm=5, n_measure=3, n_skip=1,
            seed=42, verbose=False
        )
        assert abs(result['analytical_gap'] - 2.0) < 1e-10

    def test_plaquette_positive(self):
        """Plaquette is positive after thermalization."""
        result = run_mass_gap(
            beta=4.0, n_therm=15, n_measure=5, n_skip=1,
            seed=42, verbose=False
        )
        assert result['plaquette'] > 0

    def test_gap_nonnegative(self):
        """Fitted mass gap is non-negative."""
        result = run_mass_gap(
            beta=4.0, n_therm=15, n_measure=10, n_skip=1,
            seed=42, verbose=False
        )
        assert result['gap_fit']['mass_gap'] >= 0
