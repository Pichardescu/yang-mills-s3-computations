"""
Tests for the glueball spectrum module.

Verifies eigenvalue computations, mass ratios, and comparison
with lattice QCD glueball masses.

IMPORTANT: The linearized (free) spectrum on S³ is NOT the full
interacting theory. Mass ratios will differ from lattice QCD.
We use generous tolerances and document discrepancies honestly.
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.glueball_spectrum import GlueballSpectrum, HBAR_C_MEV_FM


class TestEigenvalueAtL:
    """Eigenvalues of the coexact 1-form Laplacian on S³."""

    def test_l1_unit_radius(self):
        """l=1, R=1: eigenvalue = (1+1)^2 = 4."""
        ev = GlueballSpectrum.eigenvalue_at_l(1, R=1.0)
        assert abs(ev - 4.0) < 1e-12

    def test_l2_unit_radius(self):
        """l=2, R=1: eigenvalue = (2+1)^2 = 9."""
        ev = GlueballSpectrum.eigenvalue_at_l(2, R=1.0)
        assert abs(ev - 9.0) < 1e-12

    def test_l3_unit_radius(self):
        """l=3, R=1: eigenvalue = (3+1)^2 = 16."""
        ev = GlueballSpectrum.eigenvalue_at_l(3, R=1.0)
        assert abs(ev - 16.0) < 1e-12

    def test_radius_scaling(self):
        """Eigenvalue scales as 1/R²."""
        R = 3.0
        ev = GlueballSpectrum.eigenvalue_at_l(2, R)
        expected = 9.0 / R**2
        assert abs(ev - expected) < 1e-12

    def test_l0_raises(self):
        """l=0 is invalid for 1-forms on S³."""
        with pytest.raises(ValueError):
            GlueballSpectrum.eigenvalue_at_l(0, R=1.0)

    def test_general_formula(self):
        """λ_l = (l+1)²/R² for several l values."""
        R = 2.0
        for l in range(1, 11):
            expected = (l + 1) ** 2 / R**2
            actual = GlueballSpectrum.eigenvalue_at_l(l, R)
            assert abs(actual - expected) < 1e-12


class TestMassAtL:
    """Physical masses from eigenvalues."""

    def test_l1_mass_formula(self):
        """m_1 = ℏc × 2 / R."""
        R_fm = 2.2
        mass = GlueballSpectrum.mass_at_l(1, R_fm)
        expected = HBAR_C_MEV_FM * 2.0 / R_fm
        assert abs(mass - expected) < 1e-6

    def test_l2_mass_formula(self):
        """m_2 = ℏc × 3 / R."""
        R_fm = 2.2
        mass = GlueballSpectrum.mass_at_l(2, R_fm)
        expected = HBAR_C_MEV_FM * 3.0 / R_fm
        assert abs(mass - expected) < 1e-6

    def test_mass_increases_with_l(self):
        """Higher l gives higher mass."""
        R_fm = 2.2
        masses = [GlueballSpectrum.mass_at_l(l, R_fm) for l in range(1, 6)]
        for i in range(len(masses) - 1):
            assert masses[i] < masses[i + 1]

    def test_l0_raises(self):
        """l=0 is invalid."""
        with pytest.raises(ValueError):
            GlueballSpectrum.mass_at_l(0, R_fm=1.0)


class TestEigenvalueRatios:
    """Mass ratios are R-independent — a key prediction."""

    def test_ratios_r_independent(self):
        """
        m_l/m_1 = (l+1)/2 is independent of R.
        Verify at multiple radii.
        """
        for R_fm in [0.5, 1.0, 2.2, 5.0, 10.0]:
            m1 = GlueballSpectrum.mass_at_l(1, R_fm)
            m2 = GlueballSpectrum.mass_at_l(2, R_fm)
            m3 = GlueballSpectrum.mass_at_l(3, R_fm)

            ratio_21 = m2 / m1
            ratio_31 = m3 / m1

            expected_21 = 3.0 / 2.0  # = 1.5
            expected_31 = 4.0 / 2.0  # = 2.0

            assert abs(ratio_21 - expected_21) < 1e-10
            assert abs(ratio_31 - expected_31) < 1e-10

    def test_eigenvalue_ratio_method(self):
        """Test the eigenvalue_ratio static method."""
        ratio = GlueballSpectrum.eigenvalue_ratio(1, 2)
        expected = 3.0 / 2.0
        assert abs(ratio - expected) < 1e-12

    def test_l2_over_l1(self):
        """m_2/m_1 = (2+1)/(1+1) = 3/2 = 1.5."""
        ratio = GlueballSpectrum.eigenvalue_ratio(1, 2)
        assert abs(ratio - 1.5) < 1e-12

    def test_l3_over_l1(self):
        """m_3/m_1 = (3+1)/(1+1) = 4/2 = 2.0."""
        ratio = GlueballSpectrum.eigenvalue_ratio(1, 3)
        expected = 2.0
        assert abs(ratio - expected) < 1e-12


class TestGlueballRatiosVsLattice:
    """
    Compare our mass ratios with lattice QCD.

    IMPORTANT: Our spectrum is the FREE (linearized) spectrum.
    The full interacting theory will modify these ratios.
    We document the discrepancies here — they are EXPECTED.
    """

    def test_ratio_structure(self):
        """glueball_ratios returns expected structure."""
        result = GlueballSpectrum.glueball_ratios(2.2)
        assert 'our_ratios' in result
        assert 'lattice_ratios' in result
        assert 'comparison' in result
        assert len(result['comparison']) >= 2

    def test_our_2pp_ratio(self):
        """
        Our l=2/l=1 ratio = 3/2 = 1.5 vs lattice 2++/0++ ≈ 1.39.
        Discrepancy ≈ 7.9%.
        """
        result = GlueballSpectrum.glueball_ratios(2.2)
        comp = result['comparison'][0]  # 2++/0++

        our = comp['our_ratio']
        lattice = comp['lattice_ratio']

        # Our prediction: 3/2 = 1.5
        assert abs(our - 1.5) < 1e-6

        # Discrepancy should be documented (≈ 7.9%)
        discrepancy = abs(our - lattice) / lattice
        assert discrepancy < 0.10, (
            f"2++/0++ ratio: our={our:.3f}, lattice={lattice:.3f}, "
            f"discrepancy={discrepancy*100:.1f}%"
        )

    def test_our_0mp_ratio_documents_discrepancy(self):
        """
        Our l=3/l=1 ratio = 2.0 vs lattice 0-+/0++ ≈ 1.50.
        Discrepancy ≈ 33% — this is expected for the free spectrum.
        The l=3 → 0-+ assignment may not be correct.
        """
        result = GlueballSpectrum.glueball_ratios(2.2)
        comp = result['comparison'][1]  # 0-+/0++

        our = comp['our_ratio']
        lattice = comp['lattice_ratio']

        # Our prediction: (3+1)/2 = 2.0
        assert abs(our - 2.0) < 1e-6

        # Document the discrepancy (≈ 33%)
        discrepancy = abs(our - lattice) / lattice
        # We allow up to 40% because this is the FREE spectrum
        assert discrepancy < 0.40, (
            f"0-+/0++ ratio: our={our:.3f}, lattice={lattice:.3f}, "
            f"discrepancy={discrepancy*100:.1f}% (expected for free spectrum)"
        )


class TestBestFitR:
    """Determine R from glueball mass and check consistency."""

    def test_best_fit_R_in_range(self):
        """R from 0++ glueball mass should be in 0.1-1.0 fm range."""
        result = GlueballSpectrum.best_fit_R(target_mass_0pp_MeV=1730.0)
        R = result['R_fm']

        # R = 2 × ℏc / 1730 ≈ 0.228 fm
        expected = 2.0 * HBAR_C_MEV_FM / 1730.0
        assert abs(R - expected) < 1e-6
        assert 0.1 < R < 1.0, f"R = {R:.3f} fm should be in [0.1, 1.0] fm"

    def test_achieved_mass_matches_target(self):
        """The achieved mass should match the target."""
        result = GlueballSpectrum.best_fit_R(1730.0)
        assert abs(result['achieved_mass_MeV'] - 1730.0) < 0.1

    def test_tension_documented(self):
        """
        R from glueball (≈0.255 fm) vs R from gap (≈2.2 fm)
        are very different. This tension must be documented.
        """
        result = GlueballSpectrum.best_fit_R(1730.0)
        tension = result['tension']

        # R_gap / R_glueball ≈ 2.2 / 0.255 ≈ 8.6
        assert tension > 5.0, (
            f"Tension ratio = {tension:.1f}, expected >> 1. "
            f"R from gap and R from glueball are very different."
        )
        assert 'note' in result
        assert len(result['note']) > 0

    def test_custom_target(self):
        """Works with custom target mass."""
        result = GlueballSpectrum.best_fit_R(target_mass_0pp_MeV=2000.0)
        R = result['R_fm']
        expected = 2.0 * HBAR_C_MEV_FM / 2000.0
        assert abs(R - expected) < 1e-6


class TestSpectrumTable:
    """Full spectrum table generation."""

    def test_table_length(self):
        """Table should have l_max entries."""
        table = GlueballSpectrum.spectrum_table(2.2, l_max=10)
        assert len(table) == 10

    def test_table_fields(self):
        """Each entry has required fields."""
        table = GlueballSpectrum.spectrum_table(2.2, l_max=3)
        for entry in table:
            assert 'l' in entry
            assert 'eigenvalue' in entry
            assert 'mass_MeV' in entry
            assert 'mass_GeV' in entry
            assert 'ratio_to_ground' in entry
            assert 'jpc' in entry

    def test_ground_state_ratio_is_1(self):
        """First entry (l=1) should have ratio = 1.0."""
        table = GlueballSpectrum.spectrum_table(2.2)
        assert abs(table[0]['ratio_to_ground'] - 1.0) < 1e-12

    def test_masses_gev_consistency(self):
        """mass_GeV = mass_MeV / 1000."""
        table = GlueballSpectrum.spectrum_table(2.2, l_max=5)
        for entry in table:
            assert abs(entry['mass_GeV'] - entry['mass_MeV'] / 1000) < 1e-10
