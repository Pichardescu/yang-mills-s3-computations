"""
Tests for 0++ Glueball Mass from the 9-DOF Effective Hamiltonian.

Tests the computation of the gauge-invariant 0++ glueball mass from the
effective Hamiltonian on S^3/I* reduced to the 3 singular values.

Test categories:
    1. Harmonic spectrum (free theory, exact)
    2. Gauge-invariant basis construction
    3. Hamiltonian matrix properties (symmetry, dimensions)
    4. Harmonic limit (g^2 -> 0 recovers free spectrum)
    5. Gap positivity (gap > 0 for all tested parameters)
    6. V_4 enhancement (gap > omega for g^2 > 0)
    7. Coupling dependence (gap increases with g^2)
    8. Radius scaling (gap * R finite for fixed g^2)
    9. Convergence with basis size
   10. Comparison with effective_hamiltonian.py
   11. Physical prediction at R = 2.2 fm
   12. Edge cases and robustness
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.glueball_spectrum import (
    harmonic_spectrum,
    gauge_invariant_basis,
    build_H_gauge_invariant,
    glueball_spectrum,
    glueball_mass_vs_R,
    convergence_study,
    physical_glueball_prediction,
    gap_vs_coupling,
    glueball_summary,
    HBAR_C_MEV_FM,
    _build_1d_operators,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def R_unit():
    """Unit radius."""
    return 1.0


@pytest.fixture
def R_physical():
    """Physical radius R = 2.2 fm."""
    return 2.2


@pytest.fixture
def g2_weak():
    """Weak coupling."""
    return 0.01


@pytest.fixture
def g2_moderate():
    """Moderate coupling."""
    return 1.0


@pytest.fixture
def g2_physical():
    """Physical coupling g^2 = 6.28 (alpha_s ~ 0.5)."""
    return 6.28


# ======================================================================
# 1. Harmonic spectrum (free theory)
# ======================================================================

class TestHarmonicSpectrum:
    """Test the free (harmonic) spectrum. THEOREM level."""

    def test_omega_value(self, R_unit):
        """omega = 2/R."""
        result = harmonic_spectrum(R_unit)
        assert abs(result['omega'] - 2.0) < 1e-14

    def test_omega_scaling(self):
        """omega = 2/R for various R."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = harmonic_spectrum(R)
            assert abs(result['omega'] - 2.0 / R) < 1e-14

    def test_ground_energy_reduced(self, R_unit):
        """E_0 = (3/2) * omega for 3 DOF."""
        result = harmonic_spectrum(R_unit)
        expected = 1.5 * result['omega']
        assert abs(result['E0_reduced'] - expected) < 1e-14

    def test_first_excited_reduced(self, R_unit):
        """E_1 = (5/2) * omega for 3 DOF."""
        result = harmonic_spectrum(R_unit)
        expected = 2.5 * result['omega']
        assert abs(result['E1_reduced'] - expected) < 1e-14

    def test_gap_equals_omega(self, R_unit):
        """Harmonic gap = omega = 2/R."""
        result = harmonic_spectrum(R_unit)
        assert abs(result['gap_reduced'] - result['omega']) < 1e-14

    def test_gap_MeV(self, R_physical):
        """Gap in MeV = omega * hbar_c."""
        result = harmonic_spectrum(R_physical)
        expected_MeV = (2.0 / R_physical) * HBAR_C_MEV_FM
        assert abs(result['gap_MeV'] - expected_MeV) < 1e-8

    def test_gap_MeV_at_physical_R(self, R_physical):
        """At R=2.2 fm, gap ~ 179 MeV."""
        result = harmonic_spectrum(R_physical)
        assert 170 < result['gap_MeV'] < 190  # ~179 MeV

    def test_energies_ordered(self, R_unit):
        """Energies are non-decreasing."""
        result = harmonic_spectrum(R_unit, n_levels=20)
        energies = result['energies_reduced']
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1] + 1e-14

    def test_label_is_theorem(self, R_unit):
        """Harmonic spectrum is exact: label = THEOREM."""
        result = harmonic_spectrum(R_unit)
        assert result['label'] == 'THEOREM'


# ======================================================================
# 2. Gauge-invariant basis
# ======================================================================

class TestGaugeInvariantBasis:
    """Test basis construction for the 3-SVD space."""

    def test_basis_size(self):
        """Basis size = n_max^3."""
        for n in [3, 5, 8, 10]:
            result = gauge_invariant_basis(n)
            assert result['basis_size'] == n ** 3

    def test_quantum_numbers_count(self):
        """Number of quantum number tuples = n_max^3."""
        result = gauge_invariant_basis(5)
        assert len(result['quantum_nums']) == 5 ** 3

    def test_quantum_numbers_range(self):
        """All quantum numbers in [0, n_max)."""
        n_max = 6
        result = gauge_invariant_basis(n_max)
        for (n1, n2, n3) in result['quantum_nums']:
            assert 0 <= n1 < n_max
            assert 0 <= n2 < n_max
            assert 0 <= n3 < n_max

    def test_ground_state_in_basis(self):
        """(0,0,0) is in the basis."""
        result = gauge_invariant_basis(3)
        assert (0, 0, 0) in result['quantum_nums']

    def test_first_excited_in_basis(self):
        """(1,0,0), (0,1,0), (0,0,1) are in the basis."""
        result = gauge_invariant_basis(3)
        assert (1, 0, 0) in result['quantum_nums']
        assert (0, 1, 0) in result['quantum_nums']
        assert (0, 0, 1) in result['quantum_nums']


# ======================================================================
# 3. Hamiltonian matrix properties
# ======================================================================

class TestHamiltonianMatrix:
    """Test properties of the Hamiltonian matrix."""

    def test_matrix_symmetric(self, R_unit, g2_moderate):
        """H is symmetric (real Hermitian)."""
        data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=5)
        H = data['matrix']
        assert np.allclose(H, H.T, atol=1e-12), \
            f"Max asymmetry: {np.max(np.abs(H - H.T))}"

    def test_matrix_dimension(self, R_unit, g2_moderate):
        """Matrix has dimension n_basis^3."""
        for n in [3, 5, 8]:
            data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=n)
            assert data['matrix'].shape == (n ** 3, n ** 3)

    def test_matrix_real(self, R_unit, g2_moderate):
        """Matrix is real."""
        data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=5)
        assert np.all(np.isreal(data['matrix']))

    def test_omega_in_output(self, R_unit, g2_moderate):
        """Output contains correct omega."""
        data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=5)
        assert abs(data['omega'] - 2.0 / R_unit) < 1e-14

    def test_diagonal_positive(self, R_unit, g2_moderate):
        """Diagonal elements are positive (kinetic + potential > 0)."""
        data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=5)
        H = data['matrix']
        assert np.all(np.diag(H) > 0), "Some diagonal elements are non-positive"

    def test_eigenvalues_positive(self, R_unit, g2_moderate):
        """All eigenvalues are positive (H > 0)."""
        data = build_H_gauge_invariant(R_unit, g2_moderate, n_basis=6)
        evals = np.linalg.eigvalsh(data['matrix'])
        assert np.all(evals > -1e-10), \
            f"Negative eigenvalue: {evals[0]}"


# ======================================================================
# 4. Harmonic limit (g^2 -> 0)
# ======================================================================

class TestHarmonicLimit:
    """When g^2 -> 0, the interacting spectrum reduces to harmonic."""

    def test_gap_approaches_omega(self, R_unit, g2_weak):
        """gap -> omega as g^2 -> 0."""
        result = glueball_spectrum(R_unit, g2_weak, n_basis=8)
        omega = result['omega']
        assert abs(result['gap'] - omega) / omega < 0.01, \
            f"gap = {result['gap']}, omega = {omega}"

    def test_ground_energy_approaches_harmonic(self, R_unit, g2_weak):
        """E_0 -> 3*omega/2 as g^2 -> 0."""
        result = glueball_spectrum(R_unit, g2_weak, n_basis=8)
        omega = result['omega']
        expected_E0 = 1.5 * omega
        assert abs(result['E0'] - expected_E0) / expected_E0 < 0.01

    def test_zero_coupling_exact(self, R_unit):
        """At g^2 = 0, spectrum is exactly harmonic."""
        result = glueball_spectrum(R_unit, 0.0, n_basis=8)
        omega = result['omega']
        # Ground state: E = 3*omega/2
        assert abs(result['E0'] - 1.5 * omega) < 1e-10 * omega
        # Gap: omega
        assert abs(result['gap'] - omega) < 1e-10 * omega


# ======================================================================
# 5. Gap positivity
# ======================================================================

class TestGapPositivity:
    """The spectral gap is positive for all parameters tested."""

    def test_gap_positive_weak(self, R_unit, g2_weak):
        """Gap > 0 for weak coupling."""
        result = glueball_spectrum(R_unit, g2_weak, n_basis=8)
        assert result['gap'] > 0

    def test_gap_positive_moderate(self, R_unit, g2_moderate):
        """Gap > 0 for moderate coupling."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=8)
        assert result['gap'] > 0

    def test_gap_positive_strong(self, R_unit):
        """Gap > 0 for strong coupling g^2 = 20."""
        result = glueball_spectrum(R_unit, 20.0, n_basis=8)
        assert result['gap'] > 0

    def test_gap_positive_physical(self, R_physical, g2_physical):
        """Gap > 0 at physical parameters."""
        result = glueball_spectrum(R_physical, g2_physical, n_basis=10)
        assert result['gap'] > 0

    def test_gap_positive_small_R(self, g2_moderate):
        """Gap > 0 for small R."""
        result = glueball_spectrum(0.5, g2_moderate, n_basis=8)
        assert result['gap'] > 0

    def test_gap_positive_large_R(self, g2_moderate):
        """Gap > 0 for large R."""
        result = glueball_spectrum(10.0, g2_moderate, n_basis=8)
        assert result['gap'] > 0


# ======================================================================
# 6. V_4 enhancement
# ======================================================================

class TestV4Enhancement:
    """The quartic interaction V_4 pushes the gap above the free value."""

    def test_gap_exceeds_omega_moderate(self, R_unit, g2_moderate):
        """gap > omega for moderate coupling."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=10)
        assert result['gap'] > result['omega'], \
            f"gap = {result['gap']}, omega = {result['omega']}"

    def test_gap_exceeds_omega_physical(self, R_physical, g2_physical):
        """gap > omega at physical parameters."""
        result = glueball_spectrum(R_physical, g2_physical, n_basis=10)
        assert result['gap'] > result['omega'], \
            f"gap = {result['gap']}, omega = {result['omega']}"

    def test_enhancement_increases_with_coupling(self, R_unit):
        """gap/omega increases with g^2."""
        ratios = []
        for g2 in [0.1, 1.0, 5.0, 10.0]:
            result = glueball_spectrum(R_unit, g2, n_basis=10)
            ratios.append(result['gap_over_omega'])
        # Should be monotonically increasing (or at least non-decreasing)
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] >= ratios[i] - 0.01, \
                f"Enhancement not increasing: {ratios}"

    def test_ground_energy_increases_with_coupling(self, R_unit):
        """E_0 increases with g^2 (V_4 raises all levels)."""
        E0s = []
        for g2 in [0.0, 1.0, 5.0, 10.0]:
            result = glueball_spectrum(R_unit, g2, n_basis=8)
            E0s.append(result['E0'])
        for i in range(len(E0s) - 1):
            assert E0s[i + 1] >= E0s[i] - 1e-10, \
                f"E0 not increasing: {E0s}"


# ======================================================================
# 7. Coupling dependence
# ======================================================================

class TestCouplingDependence:
    """Test how the gap depends on g^2."""

    def test_gap_vs_coupling_all_positive(self, R_unit):
        """All gaps positive in coupling scan."""
        result = gap_vs_coupling(R_unit, n_basis=8)
        assert result['all_positive'], \
            f"Some gaps non-positive: {result['gaps']}"

    def test_gap_vs_coupling_monotone(self, R_unit):
        """Gap increases with coupling (for moderate couplings)."""
        result = gap_vs_coupling(R_unit,
                                  g_squared_values=[0.0, 1.0, 5.0, 10.0],
                                  n_basis=8)
        gaps = result['gaps']
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] >= gaps[i] - 1e-10, \
                f"Gap not monotone at g^2={result['g_squared_values'][i+1]}: {gaps}"

    def test_weak_coupling_gap_near_omega(self, R_unit):
        """At very weak coupling, gap ~ omega."""
        result = gap_vs_coupling(R_unit,
                                  g_squared_values=[0.001],
                                  n_basis=8)
        omega = result['omega']
        assert abs(result['gaps'][0] - omega) / omega < 0.01


# ======================================================================
# 8. Radius scaling
# ======================================================================

class TestRadiusScaling:
    """Test gap behavior as a function of R."""

    def test_gap_x_R_finite(self, g2_weak):
        """gap * R is approximately constant for weak coupling."""
        R_values = [0.5, 1.0, 2.0, 5.0]
        result = glueball_mass_vs_R(R_values, g2_weak, n_basis=8)
        gap_x_R = result['gaps'] * result['R_values']
        # Should be close to 2 (harmonic value: gap = omega = 2/R)
        # At g^2=0.01, the V_4 correction is small but not zero;
        # gap*R ~ 2.0 to within ~10%
        for v in gap_x_R:
            assert abs(v - 2.0) / 2.0 < 0.10, \
                f"gap*R = {v}, expected ~2.0"

    def test_all_gaps_positive(self, g2_moderate):
        """Gaps positive for all R values."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        result = glueball_mass_vs_R(R_values, g2_moderate, n_basis=8)
        assert result['all_positive'], \
            f"Gaps: {result['gaps']}"

    def test_enhancement_array(self, g2_moderate):
        """Enhancement > 1 for all R when g^2 > 0."""
        R_values = [0.5, 1.0, 2.0]
        result = glueball_mass_vs_R(R_values, g2_moderate, n_basis=8)
        assert np.all(result['enhancement'] > 1.0 - 1e-10), \
            f"Enhancement: {result['enhancement']}"


# ======================================================================
# 9. Convergence with basis size
# ======================================================================

class TestConvergence:
    """Test that the gap converges as n_basis increases."""

    def test_convergence_study_runs(self, R_unit, g2_moderate):
        """Convergence study completes without error."""
        result = convergence_study(R_unit, g2_moderate,
                                    n_basis_values=[4, 6, 8])
        assert len(result['gaps']) == 3

    def test_convergence_gaps_positive(self, R_unit, g2_moderate):
        """All gaps in convergence study are positive."""
        result = convergence_study(R_unit, g2_moderate,
                                    n_basis_values=[4, 6, 8])
        assert np.all(result['gaps'] > 0)

    def test_convergence_decreasing_change(self, R_unit, g2_moderate):
        """Relative changes decrease with basis size."""
        result = convergence_study(R_unit, g2_moderate,
                                    n_basis_values=[4, 6, 8, 10, 12])
        gaps = result['gaps']
        if len(gaps) >= 3:
            change1 = abs(gaps[1] - gaps[0])
            change2 = abs(gaps[2] - gaps[1])
            # Second change should be smaller (convergence)
            assert change2 < change1 + 0.05 * abs(gaps[0]), \
                f"Not converging: changes = {change1}, {change2}"

    def test_convergence_physical(self, R_physical, g2_physical):
        """Convergence at physical parameters."""
        result = convergence_study(R_physical, g2_physical,
                                    n_basis_values=[6, 8, 10, 12])
        assert len(result['gaps']) >= 3
        assert np.all(result['gaps'] > 0)


# ======================================================================
# 10. Consistency with effective_hamiltonian.py
# ======================================================================

class TestConsistencyWithEffectiveHamiltonian:
    """Cross-check with the existing EffectiveHamiltonian class."""

    def test_free_spectrum_matches(self, R_unit):
        """Free spectrum matches EffectiveHamiltonian at g=0."""
        from yang_mills_s3.proofs.effective_hamiltonian import EffectiveHamiltonian

        h_eff = EffectiveHamiltonian(R=R_unit, g_coupling=0.001)
        spec_old = h_eff.compute_spectrum(n_basis=8, method='reduced')

        result_new = glueball_spectrum(R_unit, 0.001**2, n_basis=8)

        # Both should have gap ~ omega = 2/R
        omega = 2.0 / R_unit
        assert abs(spec_old['gap'] - omega) / omega < 0.02
        assert abs(result_new['gap'] - omega) / omega < 0.02

    def test_omega_matches(self, R_unit):
        """Harmonic frequency matches between implementations."""
        from yang_mills_s3.proofs.effective_hamiltonian import EffectiveHamiltonian

        h_eff = EffectiveHamiltonian(R=R_unit, g_coupling=1.0)
        data_old = h_eff.build_reduced_hamiltonian(n_basis=5)

        data_new = build_H_gauge_invariant(R_unit, 1.0, n_basis=5)

        assert abs(data_old['omega'] - data_new['omega']) < 1e-14

    def test_dimension_matches(self, R_unit):
        """Basis dimensions match."""
        from yang_mills_s3.proofs.effective_hamiltonian import EffectiveHamiltonian

        h_eff = EffectiveHamiltonian(R=R_unit, g_coupling=1.0)
        n = 5
        data_old = h_eff.build_reduced_hamiltonian(n_basis=n)
        data_new = build_H_gauge_invariant(R_unit, 1.0, n_basis=n)

        assert data_old['basis_size'] == data_new['basis_size']


# ======================================================================
# 11. Physical prediction
# ======================================================================

class TestPhysicalPrediction:
    """Test the physical 0++ glueball prediction."""

    def test_prediction_runs(self):
        """Physical prediction completes."""
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=8)
        assert 'gap_MeV' in result
        assert 'assessment' in result

    def test_prediction_gap_positive(self):
        """Gap is positive at physical parameters."""
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=10)
        assert result['gap_MeV'] > 0

    def test_prediction_above_free(self):
        """Gap exceeds the free (harmonic) value."""
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=10)
        assert result['gap_MeV'] > result['free_gap_MeV']

    def test_prediction_enhancement_factor(self):
        """Enhancement factor > 1."""
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=10)
        assert result['enhancement'] > 1.0

    def test_prediction_below_lattice(self):
        """Gap is below the full lattice 0++ (model is truncated).

        The 9-DOF model captures only k=1 modes, so it cannot reproduce
        the full 1730 MeV glueball mass. We expect the model gap to be
        somewhere between 179 MeV (free) and 1730 MeV (lattice).
        """
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=10)
        # Model gap should be less than lattice 0++ (model is truncated)
        assert result['gap_MeV'] < result['lattice_0pp_MeV'], \
            f"Model gap {result['gap_MeV']} >= lattice {result['lattice_0pp_MeV']}"

    def test_lattice_comparison_ratio(self):
        """Model/lattice ratio is between 0 and 1."""
        result = physical_glueball_prediction(R_fm=2.2, g_squared=6.28,
                                               n_basis=10)
        assert 0 < result['ratio_to_lattice'] < 1.0

    def test_summary_string(self):
        """Summary string is generated without error."""
        summary = glueball_summary(R_fm=2.2, g_squared=6.28, n_basis=8)
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "0++ GLUEBALL MASS" in summary


# ======================================================================
# 12. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_very_small_R(self):
        """Works for very small R (large gap)."""
        result = glueball_spectrum(0.1, 1.0, n_basis=6)
        assert result['gap'] > 0
        assert result['gap'] > 10.0  # omega = 2/0.1 = 20

    def test_very_large_R(self):
        """Works for very large R (small gap)."""
        result = glueball_spectrum(50.0, 1.0, n_basis=6)
        assert result['gap'] > 0
        assert result['gap'] < 1.0  # omega = 2/50 = 0.04

    def test_zero_coupling(self, R_unit):
        """g^2 = 0 gives exactly harmonic spectrum."""
        result = glueball_spectrum(R_unit, 0.0, n_basis=8)
        omega = 2.0 / R_unit
        assert abs(result['gap'] - omega) < 1e-10

    def test_small_basis(self, R_unit, g2_moderate):
        """Works even with very small basis (qualitative)."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=3)
        assert result['gap'] > 0

    def test_1d_operators(self, R_unit):
        """1D operators have correct properties."""
        omega = 2.0 / R_unit
        ops = _build_1d_operators(10, omega)

        # x is symmetric
        assert np.allclose(ops['x'], ops['x'].T, atol=1e-14)

        # x^2 is symmetric and PSD
        assert np.allclose(ops['x2'], ops['x2'].T, atol=1e-14)
        evals = np.linalg.eigvalsh(ops['x2'])
        assert np.all(evals >= -1e-14)

        # H0 is diagonal with correct values
        for n in range(10):
            assert abs(ops['H0'][n, n] - omega * (n + 0.5)) < 1e-14

    def test_eigenvalues_output(self, R_unit, g2_moderate):
        """Output eigenvalues are sorted."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=8,
                                    n_eigenvalues=8)
        evals = result['eigenvalues']
        for i in range(len(evals) - 1):
            assert evals[i] <= evals[i + 1] + 1e-10

    def test_gap_over_omega_output(self, R_unit, g2_moderate):
        """gap_over_omega is correctly computed."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=8)
        expected = result['gap'] / result['omega']
        assert abs(result['gap_over_omega'] - expected) < 1e-14

    def test_gap_MeV_output(self, R_unit, g2_moderate):
        """gap_MeV is correctly computed."""
        result = glueball_spectrum(R_unit, g2_moderate, n_basis=8)
        expected = result['gap'] * HBAR_C_MEV_FM
        assert abs(result['gap_MeV'] - expected) < 1e-8


# ======================================================================
# 13. Quantitative checks
# ======================================================================

class TestQuantitative:
    """Quantitative checks on the spectrum values."""

    def test_harmonic_gap_at_R_2_2(self):
        """Harmonic gap at R=2.2 fm is ~179 MeV."""
        result = harmonic_spectrum(2.2)
        expected = 2.0 / 2.2 * HBAR_C_MEV_FM  # ~179.4 MeV
        assert abs(result['gap_MeV'] - expected) < 0.1

    def test_interacting_gap_exceeds_harmonic(self):
        """At physical coupling, gap exceeds harmonic by measurable amount."""
        result = glueball_spectrum(2.2, 6.28, n_basis=12)
        harmonic_gap = result['omega']
        assert result['gap'] > 1.1 * harmonic_gap, \
            f"Enhancement only {result['gap']/harmonic_gap:.3f}x"

    def test_strong_coupling_large_enhancement(self, R_unit):
        """At strong coupling, V_4 significantly enhances the gap."""
        result = glueball_spectrum(R_unit, 50.0, n_basis=10)
        # At g^2=50, the quartic pushes the gap well above the harmonic value.
        # With n_basis=10 truncation, we see ~1.9x enhancement.
        assert result['gap_over_omega'] > 1.5, \
            f"Expected strong enhancement at g^2=50, got {result['gap_over_omega']}"

    def test_hbar_c_value(self):
        """hbar*c = 197.3269804 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-4
