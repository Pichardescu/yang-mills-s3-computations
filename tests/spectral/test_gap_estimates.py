"""
Tests for the gap estimates module.

Verifies Weitzenböck bounds, Kato-Rellich stability, radius dependence,
and comparison with experimental QCD values.
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.gap_estimates import GapEstimates
from yang_mills_s3.spectral.yang_mills_operator import HBAR_C_MEV_FM


class TestWeitzenboeckBound:
    """Weitzenböck lower bound on the YM gap."""

    def test_su2_bound_equals_4(self):
        """
        On S³(R=1): Weitzenböck bound = nabla*nabla + Ric = 2 + 2 = 4.
        This is the coexact (physical) 1-form gap eigenvalue.
        """
        bound = GapEstimates.weitzenboeck_lower_bound('SU(2)', R=1.0)
        assert abs(bound - 4.0) < 1e-12

    def test_radius_scaling(self):
        """Bound scales as 1/R²."""
        R = 2.0
        bound = GapEstimates.weitzenboeck_lower_bound('SU(2)', R=R)
        expected = 4.0 / R**2
        assert abs(bound - expected) < 1e-12


class TestKatoRellich:
    """Kato-Rellich perturbation stability."""

    def test_gap_survives_small_perturbation(self):
        """If ||V|| < gap, the gap survives."""
        result = GapEstimates.kato_rellich_stability(
            gap_linear=5.0,
            perturbation_bound=2.0
        )
        assert result['gap_survives'] is True
        assert abs(result['shifted_gap'] - 3.0) < 1e-12
        assert abs(result['relative_bound'] - 0.4) < 1e-12

    def test_gap_destroyed_by_large_perturbation(self):
        """If ||V|| >= gap, the gap may be destroyed."""
        result = GapEstimates.kato_rellich_stability(
            gap_linear=5.0,
            perturbation_bound=6.0
        )
        assert result['gap_survives'] is False
        assert result['shifted_gap'] < 0
        assert result['relative_bound'] > 1.0

    def test_marginal_case(self):
        """Exactly ||V|| = gap: gap does NOT survive (strict inequality)."""
        result = GapEstimates.kato_rellich_stability(
            gap_linear=5.0,
            perturbation_bound=5.0
        )
        assert result['gap_survives'] is False
        assert abs(result['shifted_gap']) < 1e-12


class TestGapVsRadius:
    """Gap as a function of radius."""

    def test_gap_decreases_with_radius(self):
        """Gap → 0 as R → ∞."""
        R_values = [1.0, 2.0, 5.0, 10.0, 100.0]
        data = GapEstimates.gap_vs_radius('SU(2)', R_values)

        assert data.shape == (5, 2)
        gaps = data[:, 1]

        # Should be monotonically decreasing
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

        # Should be close to zero for large R
        assert gaps[-1] < 0.001

    def test_gap_increases_for_small_radius(self):
        """Gap → ∞ as R → 0."""
        R_values = [0.01, 0.1, 1.0, 10.0]
        data = GapEstimates.gap_vs_radius('SU(2)', R_values)
        gaps = data[:, 1]

        # Smallest R should have largest gap
        assert gaps[0] > gaps[1] > gaps[2] > gaps[3]

    def test_exact_values(self):
        """Gap = 4/R² for each radius."""
        R_values = [0.5, 1.0, 2.0, 3.0]
        data = GapEstimates.gap_vs_radius('SU(2)', R_values)

        for i, R in enumerate(R_values):
            expected = 4.0 / R**2
            assert abs(data[i, 1] - expected) < 1e-12


class TestComparisonWithQCD:
    """Comparison with experimental QCD observables."""

    def test_proton_radius(self):
        """
        Proton radius ≈ 0.84 fm.
        With R = 2.184 fm: predicted = R/2.6 ≈ 0.84 fm.
        """
        R_fm = 2.184  # Chosen so R/2.6 ≈ 0.84
        result = GapEstimates.comparison_with_qcd(R_fm)

        predicted = result['proton_radius']
        experimental = result['proton_radius_exp']

        assert abs(predicted - experimental) / experimental < 0.05, \
            f"Proton radius {predicted:.3f} fm should be ≈ {experimental} fm (within 5%)"

    def test_mass_gap_order_of_magnitude(self):
        """Mass gap should be in the right ballpark (100-400 MeV)."""
        R_fm = 2.2
        result = GapEstimates.comparison_with_qcd(R_fm)
        mass_gap = result['mass_gap']

        assert 100 < mass_gap < 400, \
            f"Mass gap {mass_gap:.1f} MeV should be O(200) MeV"

    def test_comparison_dict_keys(self):
        """All expected keys are present."""
        result = GapEstimates.comparison_with_qcd(2.0)
        expected_keys = {
            'mass_gap', 'lambda_qcd',
            'proton_radius', 'proton_radius_exp',
            'confinement_length', 'confinement_exp',
            'string_tension', 'string_tension_exp',
        }
        assert set(result.keys()) == expected_keys

    def test_lambda_qcd_value(self):
        """Reference Λ_QCD should be 200 MeV."""
        result = GapEstimates.comparison_with_qcd(2.0)
        assert result['lambda_qcd'] == 200.0
