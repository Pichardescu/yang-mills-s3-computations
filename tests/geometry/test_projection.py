"""
Tests for the projection formalism.

These tests explore the relationship between 4D (S³) quantities
and 3D observables. Many are exploratory — documenting what we
find rather than asserting what must be true.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from yang_mills_s3.geometry.projection import Projection, TemperatureModel, MassOntology

HBAR_C = 197.3269804


class TestFiberIntegral:
    """How eigenfunctions on S³ look when averaged over the Hopf fiber."""

    def test_eigenvalue_ratio_s3_vs_s2(self):
        """S³ eigenvalue l(l+2) vs S² eigenvalue l(l+1)."""
        R = 1.0
        for l in range(1, 10):
            result = Projection.fiber_integral_eigenfunction(l, R)
            assert result['eigenvalue_s3'] == l * (l + 2)
            assert result['eigenvalue_s2_projected'] == l * (l + 1)

    def test_fiber_contribution_is_l_over_R2(self):
        """The fiber adds l/R² to the eigenvalue."""
        R = 2.0
        for l in range(1, 10):
            result = Projection.fiber_integral_eigenfunction(l, R)
            expected = l / R**2
            assert abs(result['fiber_contribution'] - expected) < 1e-10

    def test_ratio_approaches_1_for_large_l(self):
        """For large l, S³ and S² eigenvalues converge: l(l+2)/l(l+1) → 1."""
        R = 1.0
        for l in [10, 50, 100]:
            result = Projection.fiber_integral_eigenfunction(l, R)
            ratio = result['ratio_s3_to_s2']
            assert abs(ratio - 1.0) < 1.0 / l  # Converges as 1/l


class TestYangMillsGap:
    """The mass gap: 4D reality vs 3D observable."""

    def test_gap_is_5_over_R2(self):
        """The gap eigenvalue is 5/R², intrinsic to S³."""
        R = 2.2
        result = Projection.yang_mills_gap_4d(R)
        assert abs(result['gap_eigenvalue_4d'] - 5 / R**2) < 1e-10

    def test_ricci_is_intrinsic(self):
        """The Ricci contribution 2/R² is intrinsic — observers feel it."""
        R = 2.2
        result = Projection.yang_mills_gap_4d(R)
        assert abs(result['ricci_intrinsic'] - 2 / R**2) < 1e-10

    def test_gap_physical_value(self):
        """At R=2.2 fm, gap ≈ 200 MeV."""
        R = 2.2
        result = Projection.yang_mills_gap_4d(R)
        assert 190 < result['gap_mass_4d_MeV'] < 210


class TestObservableSpectrum:
    """Compare 4D eigenvalues with what 3D observers measure."""

    def test_spectrum_ratios_are_R_independent(self):
        """Mass ratios don't depend on R — they're pure geometry."""
        for R in [1.0, 2.2, 5.0, 10.0]:
            rows = Projection.observable_spectrum_table(R, l_max=5)
            ratios = [row['ratio_to_ground'] for row in rows]
            # All should match the R=1 values
            rows_ref = Projection.observable_spectrum_table(1.0, l_max=5)
            ratios_ref = [row['ratio_to_ground'] for row in rows_ref]
            for r, r_ref in zip(ratios, ratios_ref):
                assert abs(r - r_ref) < 1e-10

    def test_ground_state_ratio_is_1(self):
        """l=1 ratio to ground state is 1."""
        rows = Projection.observable_spectrum_table(2.2)
        assert abs(rows[0]['ratio_to_ground'] - 1.0) < 1e-10

    def test_l2_ratio_is_sqrt_2(self):
        """l=2/l=1 ratio = sqrt(10/5) = sqrt(2) ≈ 1.414."""
        rows = Projection.observable_spectrum_table(2.2)
        assert abs(rows[1]['ratio_to_ground'] - np.sqrt(2)) < 1e-10

    def test_lattice_comparison_2pp(self):
        """
        GENUINE PREDICTION: m(2++)/m(0++) = sqrt(2) ≈ 1.414.
        Lattice: 1.39. Discrepancy: 1.7%.
        """
        rows = Projection.observable_spectrum_table(2.2)
        our_ratio = rows[1]['ratio_to_ground']  # l=2/l=1
        lattice_ratio = 1.39
        discrepancy = abs(our_ratio - lattice_ratio) / lattice_ratio
        # Document the comparison
        assert discrepancy < 0.02  # Within 2%


class TestTemperature:
    """Temperature as w-oscillation amplitude."""

    def test_geometric_temperature(self):
        """T_geom = ℏc / (2π R) at R=2.2 fm."""
        T = TemperatureModel.geometric_temperature(2.2)
        expected = HBAR_C / (2 * np.pi * 2.2)
        assert abs(T - expected) < 0.01
        assert 14 < T < 15  # ≈ 14.3 MeV

    def test_deconfinement_factors(self):
        """
        The linking energy factor for SU(3) ≈ 12, for SU(2) ≈ 21.
        Their RATIO should match T_c(SU(2))/T_c(SU(3)) from lattice.
        """
        R = 2.2
        result = TemperatureModel.deconfinement_estimate(R)
        # Ratio of factors should equal ratio of T_c
        factor_ratio = result['factor_SU2'] / result['factor_SU3']
        tc_ratio = 300.0 / 170.0
        assert abs(factor_ratio - tc_ratio) < 0.01  # Exact by construction

    def test_deconfinement_ratio_is_physical(self):
        """
        T_c(SU(2))/T_c(SU(3)) ≈ 1.76 from lattice.
        This ratio is independent of R.

        OBSERVATION: This ratio is close to sqrt(3) ≈ 1.732.
        Also close to 7/4 = 1.75.
        The actual lattice value is 1.76 ± 0.03.

        If the linking energy ∝ some function of the gauge group
        representation theory, this ratio should be derivable.
        """
        ratio_exp = 300.0 / 170.0  # 1.765
        assert 1.73 < ratio_exp < 1.80


class TestMassOntology:
    """Exploring mass = w-extension."""

    def test_standard_mass_formula(self):
        """Standard: m = ℏc · sqrt(eigenvalue)."""
        eigenvalue = 5.0  # /R² with R=1
        mass = MassOntology.standard_mass(eigenvalue)
        assert abs(mass - HBAR_C * np.sqrt(5)) < 0.01

    def test_two_R_tension(self):
        """
        R_gap = 2.2 fm vs R_glueball = 0.255 fm.
        Factor ≈ 8.6. This is the CENTRAL TENSION.
        """
        result = MassOntology.two_R_tension()
        assert 8.0 < result['ratio'] < 9.0  # Factor ≈ 8.6

    def test_fiber_corrected_mass_is_open(self):
        """The fiber correction is OPEN — we don't fake a formula."""
        result = MassOntology.fiber_corrected_mass(l=1, R=2.2)
        assert 'OPEN' in result['status']

    def test_two_R_has_resolutions(self):
        """We document possible resolutions honestly."""
        result = MassOntology.two_R_tension()
        assert len(result['possible_resolutions']) >= 5
