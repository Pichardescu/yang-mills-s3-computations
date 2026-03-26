"""
Tests for Yang-Mills thermodynamics on S³.

Verifies that the Bose-Einstein partition function Z(T) constructed from
the known linearized YM spectrum produces consistent thermodynamic quantities:
    F (free energy), U (internal energy), S (entropy), C_V (specific heat)

The partition function is:
    ln Z = -sum_l d_l * ln(1 - exp(-E_l/T))

This is the Fock-space (free bosonic field) partition function.

STATUS: NUMERICAL (testing the free-field partition function)

HONEST LIMITATIONS:
    - All results are for the FREE (linearized) spectrum
    - The interacting theory has phase transitions we cannot capture
    - C_V is monotonically increasing (no peak = no phase transition)
    - Crossover scale != deconfinement temperature
"""

import pytest
import numpy as np
from yang_mills_s3.qft.thermodynamics import YMThermodynamics
from yang_mills_s3.spectral.yang_mills_operator import HBAR_C_MEV_FM


# ------------------------------------------------------------------
# Physical constants for tests
# ------------------------------------------------------------------
R_DEFAULT = 2.2   # fm
N_DEFAULT = 2     # SU(2)
E1_SU2 = HBAR_C_MEV_FM * np.sqrt(5.0) / R_DEFAULT  # ~ 200.5 MeV


class TestPartitionFunction:
    """Tests for ln Z = -sum d_l * ln(1 - exp(-E_l/T))."""

    def test_ln_Z_non_negative(self):
        """ln Z >= 0 for all T > 0 (since Z >= 1)."""
        for T in [10.0, 50.0, 100.0, 200.0, 500.0]:
            ln_Z = YMThermodynamics.log_partition_function(T, R_DEFAULT, N_DEFAULT)
            assert ln_Z >= 0, f"ln Z({T}) = {ln_Z} should be >= 0"

    def test_Z_greater_than_one(self):
        """Z(T) > 1 for all T > 0."""
        for T in [10.0, 50.0, 100.0, 200.0, 500.0]:
            Z = YMThermodynamics.partition_function(T, R_DEFAULT, N_DEFAULT)
            assert Z > 1.0, f"Z({T}) = {Z} should be > 1"

    def test_Z_approaches_one_at_low_T(self):
        """Z(T) -> 1 (ln Z -> 0) as T -> 0."""
        ln_Z_low = YMThermodynamics.log_partition_function(1.0, R_DEFAULT, N_DEFAULT)
        assert abs(ln_Z_low) < 1e-10, (
            f"ln Z(1 MeV) = {ln_Z_low}, should be ~0 (gap ~ {E1_SU2:.1f} MeV)"
        )

    def test_ln_Z_increases_with_T(self):
        """ln Z(T) is monotonically increasing in T."""
        T_values = [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]
        ln_Z_values = [
            YMThermodynamics.log_partition_function(T, R_DEFAULT, N_DEFAULT)
            for T in T_values
        ]
        for i in range(1, len(ln_Z_values)):
            assert ln_Z_values[i] > ln_Z_values[i - 1], (
                f"ln Z({T_values[i]}) = {ln_Z_values[i]} should be > "
                f"ln Z({T_values[i - 1]}) = {ln_Z_values[i - 1]}"
            )

    def test_ln_Z_exponentially_suppressed_below_gap(self):
        """At T << gap, ln Z should be exponentially small."""
        T_low = E1_SU2 / 20.0  # T = gap/20 ~ 10 MeV
        ln_Z = YMThermodynamics.log_partition_function(T_low, R_DEFAULT, N_DEFAULT)
        # ln Z ~ sum d_l * exp(-E_l/T) ~ d_1 * exp(-E_1/T) = 18 * exp(-20) ~ 4e-8
        assert ln_Z < 1e-5, (
            f"ln Z = {ln_Z} at T = {T_low:.1f} MeV should be exponentially small"
        )

    def test_Z_raises_for_negative_T(self):
        """T <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            YMThermodynamics.log_partition_function(0.0, R_DEFAULT, N_DEFAULT)
        with pytest.raises(ValueError):
            YMThermodynamics.log_partition_function(-10.0, R_DEFAULT, N_DEFAULT)

    def test_ln_Z_increases_with_N(self):
        """ln Z should increase with N (more adjoint degrees of freedom)."""
        T = 300.0
        ln_Z_su2 = YMThermodynamics.log_partition_function(T, R_DEFAULT, 2)
        ln_Z_su3 = YMThermodynamics.log_partition_function(T, R_DEFAULT, 3)
        assert ln_Z_su3 > ln_Z_su2, (
            f"ln Z_SU(3) = {ln_Z_su3} should be > ln Z_SU(2) = {ln_Z_su2}"
        )


class TestFreeEnergy:
    """Tests for F = -T * ln(Z)."""

    def test_F_is_non_positive(self):
        """F = -T*ln(Z) <= 0 since ln Z >= 0."""
        for T in [50.0, 100.0, 200.0, 500.0]:
            F = YMThermodynamics.free_energy(T, R_DEFAULT, N_DEFAULT)
            assert F <= 0, f"F({T}) = {F} should be <= 0"

    def test_F_approaches_zero_at_low_T(self):
        """F -> 0 as T -> 0 (ln Z -> 0)."""
        F_low = YMThermodynamics.free_energy(1.0, R_DEFAULT, N_DEFAULT)
        assert abs(F_low) < 1e-8, (
            f"F(1 MeV) = {F_low} should be ~ 0"
        )

    def test_F_decreases_with_T(self):
        """F is monotonically decreasing (more negative) with T."""
        T_values = [50.0, 100.0, 200.0, 300.0, 500.0]
        F_values = [
            YMThermodynamics.free_energy(T, R_DEFAULT, N_DEFAULT)
            for T in T_values
        ]
        for i in range(1, len(F_values)):
            assert F_values[i] < F_values[i - 1], (
                f"F({T_values[i]}) = {F_values[i]:.4f} should be < "
                f"F({T_values[i - 1]}) = {F_values[i - 1]:.4f}"
            )


class TestInternalEnergy:
    """Tests for U = sum d_l * E_l / (exp(E_l/T) - 1)."""

    def test_U_is_non_negative(self):
        """U >= 0 since all energies and occupations are non-negative."""
        for T in [10.0, 100.0, 200.0, 500.0]:
            U = YMThermodynamics.internal_energy(T, R_DEFAULT, N_DEFAULT)
            assert U >= 0, f"U({T}) = {U} should be >= 0"

    def test_U_approaches_zero_at_low_T(self):
        """U -> 0 as T -> 0 (vacuum dominates)."""
        U_low = YMThermodynamics.internal_energy(1.0, R_DEFAULT, N_DEFAULT)
        assert U_low < 1e-8, (
            f"U(1 MeV) = {U_low} should be ~ 0"
        )

    def test_U_increases_with_T(self):
        """U is monotonically increasing with T."""
        T_values = [50.0, 100.0, 200.0, 300.0, 500.0]
        U_values = [
            YMThermodynamics.internal_energy(T, R_DEFAULT, N_DEFAULT)
            for T in T_values
        ]
        for i in range(1, len(U_values)):
            assert U_values[i] > U_values[i - 1], (
                f"U({T_values[i]}) = {U_values[i]:.4f} should be > "
                f"U({T_values[i - 1]}) = {U_values[i - 1]:.4f}"
            )


class TestEntropy:
    """Tests for S = (U - F) / T."""

    def test_S_non_negative(self):
        """S >= 0 for all T > 0 (consistent with third law)."""
        for T in [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]:
            S = YMThermodynamics.entropy(T, R_DEFAULT, N_DEFAULT)
            assert S >= -1e-15, (
                f"S({T}) = {S} should be >= 0"
            )

    def test_S_approaches_zero_at_low_T(self):
        """S -> 0 as T -> 0 (third law)."""
        S_low = YMThermodynamics.entropy(1.0, R_DEFAULT, N_DEFAULT)
        assert abs(S_low) < 1e-10, (
            f"S(1 MeV) = {S_low} should be ~ 0"
        )

    def test_S_increases_with_T(self):
        """S is monotonically increasing (more modes populated)."""
        T_values = [50.0, 100.0, 200.0, 300.0, 500.0]
        S_values = [
            YMThermodynamics.entropy(T, R_DEFAULT, N_DEFAULT)
            for T in T_values
        ]
        for i in range(1, len(S_values)):
            assert S_values[i] > S_values[i - 1], (
                f"S({T_values[i]}) = {S_values[i]:.6f} should be > "
                f"S({T_values[i - 1]}) = {S_values[i - 1]:.6f}"
            )


class TestSpecificHeat:
    """Tests for C_V = sum d_l * (E_l/T)^2 * exp(E_l/T) / (exp(E_l/T)-1)^2."""

    def test_Cv_non_negative(self):
        """C_V >= 0 for all T > 0 (thermodynamic stability)."""
        for T in [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]:
            Cv = YMThermodynamics.specific_heat(T, R_DEFAULT, N_DEFAULT)
            assert Cv >= -1e-15, (
                f"C_V({T}) = {Cv} should be >= 0"
            )

    def test_Cv_monotonically_increasing(self):
        """
        For a free Bose-Einstein gas with an infinite tower of modes,
        C_V is monotonically increasing with T (no Schottky peak).
        """
        T_values = [50.0, 100.0, 200.0, 300.0, 500.0]
        Cv_values = [
            YMThermodynamics.specific_heat(T, R_DEFAULT, N_DEFAULT)
            for T in T_values
        ]
        for i in range(1, len(Cv_values)):
            assert Cv_values[i] > Cv_values[i - 1], (
                f"C_V({T_values[i]}) = {Cv_values[i]:.4f} should be > "
                f"C_V({T_values[i - 1]}) = {Cv_values[i - 1]:.4f} "
                f"(free BE gas: monotonically increasing)"
            )

    def test_Cv_exponentially_suppressed_at_low_T(self):
        """At T << gap, C_V should be exponentially small."""
        T_low = E1_SU2 / 20.0
        Cv = YMThermodynamics.specific_heat(T_low, R_DEFAULT, N_DEFAULT)
        assert Cv < 1e-3, (
            f"C_V at T = {T_low:.1f} MeV should be << 1, got {Cv}"
        )


class TestThermodynamicIdentities:
    """Verify standard thermodynamic identities hold."""

    @pytest.mark.parametrize("T", [50.0, 100.0, 200.0, 300.0, 500.0])
    def test_F_equals_U_minus_TS(self, T):
        """F = U - T*S should hold exactly (it's a definition)."""
        result = YMThermodynamics.verify_thermodynamic_identities(
            T, R_DEFAULT, N_DEFAULT
        )
        check = result['F_equals_U_minus_TS']
        assert check['passed'], (
            f"F = U - TS failed at T={T}: "
            f"F={check['F']:.6e}, U-TS={check['U_minus_TS']:.6e}, "
            f"rel error={check['relative_error']:.2e}"
        )

    @pytest.mark.parametrize("T", [100.0, 200.0, 300.0])
    def test_S_equals_minus_dFdT(self, T):
        """S = -dF/dT should hold (numerical derivative check)."""
        result = YMThermodynamics.verify_thermodynamic_identities(
            T, R_DEFAULT, N_DEFAULT
        )
        check = result['S_equals_minus_dFdT']
        assert check['passed'], (
            f"S = -dF/dT failed at T={T}: "
            f"S_direct={check['S_direct']:.6e}, "
            f"S_from_deriv={check['S_from_derivative']:.6e}, "
            f"rel error={check['relative_error']:.2e}"
        )

    @pytest.mark.parametrize("T", [100.0, 200.0, 300.0])
    def test_Cv_equals_T_dSdT(self, T):
        """C_V = T * dS/dT should hold (numerical derivative check)."""
        result = YMThermodynamics.verify_thermodynamic_identities(
            T, R_DEFAULT, N_DEFAULT
        )
        check = result['Cv_equals_T_dSdT']
        assert check['passed'], (
            f"C_V = T*dS/dT failed at T={T}: "
            f"Cv_direct={check['Cv_direct']:.6e}, "
            f"Cv_from_deriv={check['Cv_from_derivative']:.6e}, "
            f"rel error={check['relative_error']:.2e}"
        )

    def test_all_identities_at_T200(self):
        """All 5 identities should pass at T = 200 MeV."""
        result = YMThermodynamics.verify_thermodynamic_identities(
            200.0, R_DEFAULT, N_DEFAULT
        )
        assert result['all_passed'], (
            f"Not all identities passed at T=200 MeV: {result}"
        )


class TestGapFromLowT:
    """Low-T specific heat should recover the spectral gap."""

    def test_gap_extraction_consistent(self):
        """
        Extract E_1 from low-T C_V and compare with sqrt(5)*hbar*c/R.

        At T << E_1, the Bose-Einstein distribution reduces to the
        Boltzmann tail, and C_V ~ d_1 * (E_1/T)^2 * exp(-E_1/T).
        A linear fit of log(C_V * T^2) vs 1/T gives slope = -E_1.
        """
        result = YMThermodynamics.gap_from_low_T_behavior(R_DEFAULT, N_DEFAULT)
        assert result['consistent'], (
            f"Gap extraction failed: "
            f"E_1_exact = {result['E_1_exact_MeV']:.4f} MeV, "
            f"E_1_extracted = {result['E_1_extracted_MeV']:.4f} MeV, "
            f"rel error = {result['relative_error']:.4e}"
        )
        assert result['relative_error'] < 0.01, (
            f"Gap extraction error {result['relative_error']:.4e} > 1%"
        )

    def test_gap_extraction_SU3(self):
        """Gap extraction should also work for SU(3)."""
        result = YMThermodynamics.gap_from_low_T_behavior(R_DEFAULT, 3)
        assert result['consistent'], (
            f"SU(3) gap extraction failed: rel error = {result['relative_error']:.4e}"
        )


class TestStefanBoltzmann:
    """High-T limit should approach Stefan-Boltzmann."""

    def test_approaches_SB_at_high_T(self):
        """
        At T >> hbar*c/R, the Bose-Einstein energy density should approach
        the Stefan-Boltzmann law for (N²-1) massless gauge bosons
        (2 polarizations each):

            u = (N²-1) * pi²/15 * T⁴ / (hbar*c)³

        With adaptive l_max, the ratio should be between 0.5 and 2.0
        at T = 5000 MeV.
        """
        result = YMThermodynamics.stefan_boltzmann_check(R_DEFAULT, N_DEFAULT)
        assert result['approaches_SB'], (
            f"Does not approach SB at high T. "
            f"Highest T ratio = {result['highest_T_ratio']:.4f}"
        )

    def test_SB_ratio_in_right_ballpark(self):
        """
        At T = 5000 MeV >> hbar*c/R ~ 90 MeV, the ratio u/u_SB
        should be close to 1.0 (within 10%).
        """
        result = YMThermodynamics.stefan_boltzmann_check(R_DEFAULT, N_DEFAULT)
        highest = result['highest_T_ratio']
        assert 0.9 < highest < 1.1, (
            f"SB ratio at T=5000 MeV = {highest:.4f}, expected ~1.0 within 10%"
        )

    def test_SB_ratio_improves_with_T(self):
        """The ratio to SB should converge toward 1 at higher T."""
        result = YMThermodynamics.stefan_boltzmann_check(R_DEFAULT, N_DEFAULT)
        ratios = [r['ratio_to_SB'] for r in result['results']]
        # At the highest temperatures, the ratio should be closer to 1
        deviations = [abs(r - 1.0) for r in ratios]
        assert deviations[-1] < deviations[0], (
            f"SB deviation at highest T ({deviations[-1]:.4f}) should be "
            f"less than at lowest T ({deviations[0]:.4f})"
        )


class TestDeconfinement:
    """Tests for the deconfinement crossover analysis."""

    def test_crossover_exists(self):
        """A crossover scale T* should be identified."""
        result = YMThermodynamics.deconfinement_from_thermodynamics(
            R_DEFAULT, N_DEFAULT
        )
        assert result['T_crossover_MeV'] > 0, "Crossover temperature should be > 0"

    def test_crossover_at_gap_scale(self):
        """
        T* = E_1/ln(2) should be approximately 1.44 * E_1.
        """
        result = YMThermodynamics.deconfinement_from_thermodynamics(
            R_DEFAULT, N_DEFAULT
        )
        E_1 = result['E_1_MeV']
        T_cross = result['T_crossover_MeV']
        ratio = T_cross / E_1
        expected_ratio = 1.0 / np.log(2.0)  # ~ 1.4427
        assert abs(ratio - expected_ratio) < 0.01, (
            f"T*/E_1 = {ratio:.4f}, expected {expected_ratio:.4f}"
        )

    def test_honest_limitation_documented(self):
        """The result should document the free-gas limitation."""
        result = YMThermodynamics.deconfinement_from_thermodynamics(
            R_DEFAULT, N_DEFAULT
        )
        note = result['note']
        assert 'HONEST LIMITATION' in note
        assert 'FREE-GAS' in note or 'free-field' in note.lower()
        assert 'MONOTONICALLY INCREASING' in note


class TestThermodynamicTable:
    """Tests for the key deliverable: the thermodynamic table."""

    def test_table_has_entries(self):
        """Table should have entries."""
        table = YMThermodynamics.thermodynamic_table(R_DEFAULT, N_DEFAULT)
        assert len(table) > 10, f"Table has {len(table)} entries, expected > 10"

    def test_table_has_all_fields(self):
        """Each entry should have T, ln_Z, Z, F, U, S, Cv."""
        table = YMThermodynamics.thermodynamic_table(R_DEFAULT, N_DEFAULT)
        required_keys = {'T_MeV', 'ln_Z', 'Z', 'F_MeV', 'U_MeV', 'S', 'Cv'}
        for entry in table:
            assert required_keys.issubset(entry.keys()), (
                f"Entry missing keys: {required_keys - entry.keys()}"
            )

    def test_table_values_reasonable(self):
        """Spot-check values for reasonableness."""
        table = YMThermodynamics.thermodynamic_table(R_DEFAULT, N_DEFAULT)

        for entry in table:
            T = entry['T_MeV']
            assert entry['ln_Z'] >= 0, f"ln Z < 0 at T={T}"
            assert entry['Z'] >= 1.0, f"Z < 1 at T={T}"
            assert entry['F_MeV'] <= 0, f"F > 0 at T={T}"
            assert entry['U_MeV'] >= 0, f"U < 0 at T={T}"
            assert entry['S'] >= -1e-10, f"S < 0 at T={T}"
            assert entry['Cv'] >= -1e-10, f"Cv < 0 at T={T}"

    def test_table_T_ordering(self):
        """Table should be ordered by increasing T."""
        table = YMThermodynamics.thermodynamic_table(R_DEFAULT, N_DEFAULT)
        T_values = [entry['T_MeV'] for entry in table]
        for i in range(1, len(T_values)):
            assert T_values[i] > T_values[i - 1], (
                f"T not increasing at index {i}: {T_values[i - 1]} -> {T_values[i]}"
            )


class TestWOscillation:
    """Tests for the S³ w-oscillation connection."""

    def test_beta_correct(self):
        """beta = hbar*c / T should be computed correctly."""
        T = 200.0
        result = YMThermodynamics.w_oscillation_connection(T, R_DEFAULT)
        expected_beta = HBAR_C_MEV_FM / T
        assert abs(result['beta_fm'] - expected_beta) < 1e-10

    def test_omega_correct(self):
        """omega = T / hbar should be computed correctly."""
        T = 200.0
        HBAR_S = 6.582119569e-22
        result = YMThermodynamics.w_oscillation_connection(T, R_DEFAULT)
        expected_omega = T / HBAR_S
        assert abs(result['omega_Hz'] - expected_omega) / expected_omega < 1e-10

    def test_postulate_status_documented(self):
        """The result should document POSTULATE status."""
        result = YMThermodynamics.w_oscillation_connection(200.0, R_DEFAULT)
        assert 'POSTULATE' in result['note']

    def test_high_T_gives_dimensional_reduction(self):
        """At very high T (>> hbar*c/R), beta < R => dimensional reduction."""
        T_high = 1000.0  # Much larger than hbar*c/R ~ 90 MeV
        result = YMThermodynamics.w_oscillation_connection(T_high, R_DEFAULT)
        assert result['dimensional_reduction'][0] is True, (
            f"At T={T_high} MeV, beta={result['beta_fm']:.3f} fm should be < R={R_DEFAULT} fm"
        )

    def test_low_T_no_dimensional_reduction(self):
        """At low T (< hbar*c/R), beta > R => full 4d theory."""
        T_low = 50.0  # Much smaller than hbar*c/R ~ 90 MeV
        result = YMThermodynamics.w_oscillation_connection(T_low, R_DEFAULT)
        assert result['dimensional_reduction'][0] is False, (
            f"At T={T_low} MeV, beta={result['beta_fm']:.3f} fm should be > R={R_DEFAULT} fm"
        )


class TestEnergyAndDegeneracy:
    """Tests for the basic spectral data used in Z(T)."""

    def test_E1_matches_gap(self):
        """E_1 should equal hbar*c * sqrt(5) / R (the known gap)."""
        E_1 = YMThermodynamics.energy_at_l(1, R_DEFAULT)
        expected = HBAR_C_MEV_FM * np.sqrt(5.0) / R_DEFAULT
        assert abs(E_1 - expected) / expected < 1e-10, (
            f"E_1 = {E_1:.4f}, expected {expected:.4f}"
        )

    def test_degeneracy_SU2(self):
        """d(l=1, N=2) = 2*1*3 * 3 = 18."""
        d = YMThermodynamics.degeneracy(1, 2)
        assert d == 18, f"d(1, 2) = {d}, expected 18"

    def test_degeneracy_SU3(self):
        """d(l=1, N=3) = 2*1*3 * 8 = 48."""
        d = YMThermodynamics.degeneracy(1, 3)
        assert d == 48, f"d(1, 3) = {d}, expected 48"

    def test_energy_increases_with_l(self):
        """E_l should be strictly increasing in l."""
        for l in range(1, 20):
            E_l = YMThermodynamics.energy_at_l(l, R_DEFAULT)
            E_l1 = YMThermodynamics.energy_at_l(l + 1, R_DEFAULT)
            assert E_l1 > E_l, f"E_{l+1} = {E_l1} should be > E_{l} = {E_l}"

    def test_degeneracy_increases_with_l(self):
        """d(l) = 2*l*(l+2)*(N²-1) should increase with l."""
        for l in range(1, 20):
            d_l = YMThermodynamics.degeneracy(l, N_DEFAULT)
            d_l1 = YMThermodynamics.degeneracy(l + 1, N_DEFAULT)
            assert d_l1 > d_l, f"d({l + 1}) = {d_l1} should be > d({l}) = {d_l}"

    def test_energy_raises_for_l0(self):
        """l=0 should raise ValueError."""
        with pytest.raises(ValueError):
            YMThermodynamics.energy_at_l(0, R_DEFAULT)

    def test_degeneracy_raises_for_N1(self):
        """N=1 should raise ValueError (SU(1) is trivial)."""
        with pytest.raises(ValueError):
            YMThermodynamics.degeneracy(1, 1)
