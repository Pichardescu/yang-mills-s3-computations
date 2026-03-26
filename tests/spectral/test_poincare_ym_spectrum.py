"""
Tests for the Poincare YM spectrum module.

Verifies Yang-Mills spectrum on S^3/I*, glueball predictions,
topology comparison, and CMB-QCD connection.
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.poincare_ym_spectrum import PoincareYMSpectrum
from yang_mills_s3.geometry.poincare_homology import HBAR_C_MEV_FM


@pytest.fixture
def pym():
    """Create a PoincareYMSpectrum instance for SU(2) at R=2.2 fm."""
    return PoincareYMSpectrum(gauge_group='SU(2)', R_fm=2.2)


@pytest.fixture
def pym_su3():
    """Create a PoincareYMSpectrum instance for SU(3) at R=2.2 fm."""
    return PoincareYMSpectrum(gauge_group='SU(3)', R_fm=2.2)


# ======================================================================
# Single-particle spectrum
# ======================================================================

class TestSingleParticleSpectrum:
    """Test the single-particle YM spectrum on S^3/I*."""

    def test_gap_mass(self, pym):
        """Gap mass = 2*hbar_c/R = 2*197.33/2.2 = 179.4 MeV."""
        sp = pym.single_particle_spectrum(30)
        expected = 2 * HBAR_C_MEV_FM / 2.2
        assert abs(sp[0]['mass_mev'] - expected) < 0.1

    def test_gap_at_k1(self, pym):
        """First mode is at k=1."""
        sp = pym.single_particle_spectrum(30)
        assert sp[0]['k'] == 1

    def test_second_mode_at_k11(self, pym):
        """Second mode is at k=11."""
        sp = pym.single_particle_spectrum(30)
        assert sp[1]['k'] == 11

    def test_second_mode_mass(self, pym):
        """Second mode mass = 12*hbar_c/R."""
        sp = pym.single_particle_spectrum(30)
        expected = 12 * HBAR_C_MEV_FM / 2.2
        assert abs(sp[1]['mass_mev'] - expected) < 0.1

    def test_multiplicity_su2(self, pym):
        """SU(2): gap multiplicity = 3 (coexact) x 3 (adjoint) = 9."""
        sp = pym.single_particle_spectrum(5)
        assert sp[0]['multiplicity'] == 9
        assert sp[0]['geom_mult'] == 3
        assert sp[0]['adj_mult'] == 3

    def test_multiplicity_su3(self, pym_su3):
        """SU(3): gap multiplicity = 3 (coexact) x 8 (adjoint) = 24."""
        sp = pym_su3.single_particle_spectrum(5)
        assert sp[0]['multiplicity'] == 24
        assert sp[0]['geom_mult'] == 3
        assert sp[0]['adj_mult'] == 8

    def test_ratio_to_gap(self, pym):
        """Ratio of second mode mass to gap mass is 6.0."""
        sp = pym.single_particle_spectrum(30)
        assert abs(sp[1]['ratio_to_gap'] - 6.0) < 1e-10

    def test_eigenvalue_formula(self, pym):
        """Eigenvalue at level k is (k+1)^2/R^2."""
        R = pym.R_fm
        sp = pym.single_particle_spectrum(30)
        for entry in sp[:5]:
            k = entry['k']
            expected = (k + 1)**2 / R**2
            assert abs(entry['eigenvalue'] - expected) < 1e-10

    def test_at_least_5_modes(self, pym):
        """At least 5 single-particle modes up to k=60."""
        sp = pym.single_particle_spectrum(60)
        assert len(sp) >= 5


# ======================================================================
# Glueball spectrum
# ======================================================================

class TestGlueballSpectrum:
    """Test glueball composite predictions on S^3/I*."""

    def test_0pp_ground_state_exists(self, pym):
        """The 0++ glueball ground state exists (two k=1 composites)."""
        gb = pym.glueball_spectrum(30)
        ground = [g for g in gb if g['is_ground_state']]
        assert len(ground) == 1
        assert ground[0]['can_make_0pp'] is True

    def test_0pp_threshold_same_as_s3(self, pym):
        """0++ threshold = 2 * gap = 4/R (same as S^3)."""
        gb = pym.glueball_spectrum(30)
        ground = [g for g in gb if g['is_ground_state']][0]
        expected = 2 * 2 * HBAR_C_MEV_FM / 2.2  # 2 * m_1
        assert abs(ground['threshold_mev'] - expected) < 0.1

    def test_first_excited_composite(self, pym):
        """First excited composite uses k=1 + k=11."""
        gb = pym.glueball_spectrum(30)
        excited = [g for g in gb if not g['is_ground_state']]
        assert len(excited) > 0
        first = excited[0]
        assert first['k1'] == 1
        assert first['k2'] == 11

    def test_excited_composite_threshold(self, pym):
        """Excited composite threshold = m_1 + m_11 = 2/R + 12/R = 14/R."""
        gb = pym.glueball_spectrum(30)
        excited = [g for g in gb if not g['is_ground_state']]
        first = excited[0]
        expected = (2 + 12) * HBAR_C_MEV_FM / 2.2
        assert abs(first['threshold_mev'] - expected) < 0.1

    def test_huge_gap_to_excited(self, pym):
        """Ratio of excited to ground composite threshold > 3."""
        gb = pym.glueball_spectrum(30)
        ground = [g for g in gb if g['is_ground_state']][0]
        excited = [g for g in gb if not g['is_ground_state']][0]
        ratio = excited['threshold_mev'] / ground['threshold_mev']
        assert ratio > 3, f"Expected ratio > 3, got {ratio:.2f}"


# ======================================================================
# Topology comparison
# ======================================================================

class TestTopologyComparison:
    """Test the three-topology comparison."""

    def test_gap_same_on_both(self, pym):
        """Gap is same on S^3 and S^3/I*."""
        comp = pym.topology_comparison()
        s3_gap = comp['single_particle_gap']['s3']['mass_mev']
        pi_gap = comp['single_particle_gap']['poincare']['mass_mev']
        assert abs(s3_gap - pi_gap) < 0.01

    def test_second_excitation_different(self, pym):
        """Second excitation is dramatically different."""
        comp = pym.topology_comparison()
        s3_m2 = comp['second_excitation']['s3']['mass_mev']
        pi_m2 = comp['second_excitation']['poincare']['mass_mev']
        assert pi_m2 > 3 * s3_m2  # at least 3x heavier

    def test_spectrum_much_sparser(self, pym):
        """S^3/I* has much sparser spectrum."""
        comp = pym.topology_comparison()
        ratio = comp['spectrum_density']['ratio']
        assert ratio < 0.05  # less than 5% of modes survive


# ======================================================================
# Mass ratios
# ======================================================================

class TestMassRatios:
    """Test mass ratio predictions."""

    def test_s3_m2_over_m1(self, pym):
        """S^3: m2/m1 = 3/2."""
        mr = pym.mass_ratios()
        assert abs(mr['comparison']['s3_m2_over_m1'] - 1.5) < 1e-10

    def test_poincare_m2_over_m1(self, pym):
        """S^3/I*: m2/m1 = 12/2 = 6.0."""
        mr = pym.mass_ratios()
        assert abs(mr['comparison']['poincare_m2_over_m1'] - 6.0) < 1e-10

    def test_s3_single_particle_ratios(self, pym):
        """S^3 single-particle ratios are (k+1)/2."""
        mr = pym.mass_ratios()
        for k in range(1, 6):
            key = f'k={k}/k=1'
            expected = (k + 1) / 2.0
            assert abs(mr['s3_single_particle'][key] - expected) < 1e-10


# ======================================================================
# CMB-QCD connection
# ======================================================================

class TestCMBQCDConnection:
    """Test the CMB-QCD connection predictions."""

    def test_hypothesis(self, pym):
        """The hypothesis is stated."""
        conn = pym.cmb_qcd_connection()
        assert 'Poincare' in conn['hypothesis'] or 'S^3/I*' in conn['hypothesis']

    def test_cmb_suppression(self, pym):
        """CMB multipoles l=2..11 are suppressed."""
        conn = pym.cmb_qcd_connection()
        suppressed = conn['cmb_prediction']['suppressed_multipoles']
        for l in range(2, 12):
            assert l in suppressed

    def test_qcd_suppressed_levels(self, pym):
        """QCD has suppressed k levels between 2 and 10."""
        conn = pym.cmb_qcd_connection()
        suppressed_k = conn['qcd_prediction']['suppressed_k_levels']
        for k in range(2, 11):
            assert k in suppressed_k

    def test_status_is_conjecture(self, pym):
        """The overall status is CONJECTURE."""
        conn = pym.cmb_qcd_connection()
        assert conn['status'] == 'CONJECTURE'


# ======================================================================
# Thermodynamic predictions
# ======================================================================

class TestThermodynamics:
    """Test thermodynamic predictions."""

    def test_volume_ratio(self, pym):
        """Volume ratio is 1/120."""
        thermo = pym.thermodynamic_predictions()
        assert abs(thermo['volume_ratio'] - 1.0 / 120) < 1e-12

    def test_mode_density_small(self, pym):
        """Mode density ratio is small (< 5%)."""
        thermo = pym.thermodynamic_predictions()
        assert thermo['mode_density_ratio'] < 0.05

    def test_stefan_boltzmann_ratio(self, pym):
        """Stefan-Boltzmann ratio equals volume ratio."""
        thermo = pym.thermodynamic_predictions()
        assert abs(thermo['stefan_boltzmann_ratio'] - 1.0 / 120) < 1e-12


# ======================================================================
# Full report
# ======================================================================

class TestFullReport:
    """Test the full report generation."""

    def test_report_runs(self, pym):
        """Full report generates without error."""
        report = pym.full_report()
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_key_info(self, pym):
        """Full report contains key information."""
        report = pym.full_report()
        assert 'POINCARE' in report
        assert 'SU(2)' in report
        assert 'k=11' in report or 'k= 11' in report
