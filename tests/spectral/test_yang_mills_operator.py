"""
Tests for the Yang-Mills operator module.

Verifies the linearized YM spectrum, mass gap computations,
and physical predictions.
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.yang_mills_operator import YangMillsOperator, HBAR_C_MEV_FM


class TestAdjointDimension:
    """dim(adj(G)) for various gauge groups."""

    def test_su2(self):
        assert YangMillsOperator.adjoint_dimension('SU(2)') == 3

    def test_su3(self):
        assert YangMillsOperator.adjoint_dimension('SU(3)') == 8

    def test_su_n(self):
        for N in range(2, 8):
            assert YangMillsOperator.adjoint_dimension(f'SU({N})') == N**2 - 1

    def test_unknown_group(self):
        with pytest.raises(ValueError):
            YangMillsOperator.adjoint_dimension('FooBar')


class TestLinearizedSpectrum:
    """Spectrum of the linearized YM operator on S³."""

    def test_su2_first_eigenvalue(self):
        """First eigenvalue for SU(2) on S³(R=1) is 4 (coexact gap)."""
        spectrum = YangMillsOperator.linearized_spectrum('SU(2)', R=1.0, l_max=5)
        ev, mult = spectrum[0]
        assert abs(ev - 4.0) < 1e-12

    def test_su2_multiplicity_includes_adjoint(self):
        """Multiplicity includes factor of dim(adj) = 3."""
        spectrum = YangMillsOperator.linearized_spectrum('SU(2)', R=1.0, l_max=5)
        _, mult = spectrum[0]

        # Hodge mult for l=1 on S³ = 2*1*3 = 6
        # Total = 6 × 3(adj) = 18
        assert mult == 6 * 3  # = 18

    def test_su3_multiplicity(self):
        """SU(3): multiplicity scaled by dim(adj) = 8."""
        spectrum = YangMillsOperator.linearized_spectrum('SU(3)', R=1.0, l_max=3)
        _, mult = spectrum[0]
        assert mult == 6 * 8  # = 48


class TestMassGap:
    """The Yang-Mills mass gap."""

    def test_su2_gap_eigenvalue(self):
        """SU(2) on S³(R=1): gap eigenvalue = 4 (coexact)."""
        gap_ev = YangMillsOperator.mass_gap_eigenvalue('SU(2)', R=1.0)
        assert abs(gap_ev - 4.0) < 1e-12

    def test_su2_gap_mass(self):
        """SU(2) on S³(R=1): mass gap = 2."""
        gap = YangMillsOperator.mass_gap('SU(2)', R=1.0)
        assert abs(gap - 2.0) < 1e-12

    def test_gap_radius_scaling(self):
        """Mass gap scales as 1/R."""
        R = 3.0
        gap = YangMillsOperator.mass_gap('SU(2)', R=R)
        expected = 2.0 / R
        assert abs(gap - expected) < 1e-12

    def test_gap_eigenvalue_radius_scaling(self):
        """Gap eigenvalue scales as 1/R²."""
        R = 2.5
        gap_ev = YangMillsOperator.mass_gap_eigenvalue('SU(2)', R=R)
        expected = 4.0 / R**2
        assert abs(gap_ev - expected) < 1e-12


class TestMultiplicityLowestMode:
    """Multiplicity of the gap mode = coexact Hodge × adjoint."""

    def test_su2_multiplicity_18(self):
        """
        SU(2): 6 (coexact Hodge, k=1: 2*1*(1+2)=6) × 3 (adjoint) = 18.
        """
        mult = YangMillsOperator.multiplicity_lowest_mode('SU(2)')
        assert mult == 18

    def test_su3_multiplicity(self):
        """SU(3): 6 (coexact Hodge) × 8 (adjoint) = 48."""
        mult = YangMillsOperator.multiplicity_lowest_mode('SU(3)')
        assert mult == 48


class TestPhysicalMassGap:
    """Mass gap in MeV at physical radius."""

    def test_su2_at_2_2_fm(self):
        """
        SU(2) at R = 2.2 fm should give ≈ 179 MeV.

        m = hbar*c × 2 / R
          = 197.327 × 2 / 2.2
          ≈ 179.4 MeV
        """
        R_fm = 2.2
        mass = YangMillsOperator.physical_mass_gap('SU(2)', R_fm)
        expected = HBAR_C_MEV_FM * 2.0 / R_fm

        # Verify analytical formula
        assert abs(mass - expected) < 1e-6

        # Verify ≈ 179 MeV within 5%
        assert abs(mass - 179.4) / 179.4 < 0.05, \
            f"Mass gap {mass:.1f} MeV should be ≈ 179 MeV (within 5%)"

    def test_physical_units_consistency(self):
        """Check dimensional consistency: mass has units of MeV."""
        R_fm = 1.0
        mass = YangMillsOperator.physical_mass_gap('SU(2)', R_fm)
        expected = HBAR_C_MEV_FM * 2.0
        assert abs(mass - expected) < 1e-6
