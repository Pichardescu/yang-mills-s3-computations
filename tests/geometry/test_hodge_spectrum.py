"""
Tests for the Hodge spectrum module.

Verifies eigenvalues and multiplicities of the Hodge-de Rham Laplacian
on spheres, with emphasis on S^3 where the Yang-Mills mass gap lives.

CORRECTED (2026-03-10): Uses the correct two-branch spectrum for
1-forms on S^3 (exact + coexact), with physical gap = 4/R^2.
"""

import pytest
import numpy as np
from yang_mills_s3.geometry.hodge_spectrum import HodgeSpectrum


class TestScalarLaplacianS3:
    """Delta_0 on S^3: eigenvalue = l(l+2)/R^2, multiplicity = (l+1)^2."""

    def test_eigenvalues_unit_sphere(self):
        """Scalar eigenvalues on S^3 of radius 1."""
        spectrum = HodgeSpectrum.scalar_eigenvalues(3, R=1.0, l_max=10)

        for l, (ev, mult) in enumerate(spectrum):
            expected_ev = l * (l + 2)
            assert abs(ev - expected_ev) < 1e-12, \
                f"l={l}: got eigenvalue {ev}, expected {expected_ev}"

    def test_multiplicities(self):
        """Multiplicity of l-th eigenvalue on S^3 is (l+1)^2."""
        spectrum = HodgeSpectrum.scalar_eigenvalues(3, R=1.0, l_max=10)

        for l, (ev, mult) in enumerate(spectrum):
            expected_mult = (l + 1) ** 2
            assert mult == expected_mult, \
                f"l={l}: got multiplicity {mult}, expected {expected_mult}"

    def test_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R = 2.5
        spectrum = HodgeSpectrum.scalar_eigenvalues(3, R=R, l_max=5)

        for l, (ev, mult) in enumerate(spectrum):
            expected_ev = l * (l + 2) / R**2
            assert abs(ev - expected_ev) < 1e-12, \
                f"l={l}: eigenvalue not scaling as 1/R^2"

    def test_zero_mode(self):
        """l=0: eigenvalue = 0, multiplicity = 1 (the constants)."""
        spectrum = HodgeSpectrum.scalar_eigenvalues(3, R=1.0, l_max=0)
        ev, mult = spectrum[0]
        assert ev == 0.0
        assert mult == 1


class TestScalarLaplacianS2:
    """Delta_0 on S^2: eigenvalue = l(l+1)/R^2, multiplicity = 2l+1."""

    def test_eigenvalues(self):
        spectrum = HodgeSpectrum.scalar_eigenvalues(2, R=1.0, l_max=10)
        for l, (ev, mult) in enumerate(spectrum):
            assert abs(ev - l * (l + 1)) < 1e-12
            assert mult == 2 * l + 1


class TestOneFormLaplacianS3Coexact:
    """Delta_1 coexact on S^3: eigenvalue = (k+1)^2/R^2, starting at k=1."""

    def test_coexact_eigenvalues(self):
        """Coexact 1-form eigenvalues on S^3."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=10,
                                                      mode='coexact')
        for i, (ev, mult) in enumerate(spectrum):
            k = i + 1  # k starts at 1
            expected_ev = (k + 1) ** 2
            assert abs(ev - expected_ev) < 1e-12, \
                f"k={k}: got {ev}, expected {expected_ev}"

    def test_first_coexact_eigenvalue_is_4(self):
        """First coexact eigenvalue of Delta_1 on S^3(R=1) is 4."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=1,
                                                      mode='coexact')
        ev, _ = spectrum[0]
        assert abs(ev - 4.0) < 1e-12, \
            f"First coexact 1-form eigenvalue should be 4, got {ev}"

    def test_coexact_multiplicities(self):
        """Coexact multiplicity at k is 2*k*(k+2)."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=5,
                                                      mode='coexact')
        for i, (ev, mult) in enumerate(spectrum):
            k = i + 1
            expected_mult = 2 * k * (k + 2)
            assert mult == expected_mult, \
                f"k={k}: got multiplicity {mult}, expected {expected_mult}"

    def test_no_zero_eigenvalue(self):
        """No zero eigenvalue because H^1(S^3) = 0."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=20,
                                                      mode='coexact')
        for ev, mult in spectrum:
            assert ev > 0, "1-form Laplacian on S^3 should have no zero eigenvalue"

    def test_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R = 3.0
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=R, l_max=5,
                                                      mode='coexact')
        for i, (ev, mult) in enumerate(spectrum):
            k = i + 1
            expected = (k + 1) ** 2 / R**2
            assert abs(ev - expected) < 1e-12


class TestOneFormLaplacianS3Exact:
    """Delta_1 exact on S^3: eigenvalue = l(l+2)/R^2, starting at l=1."""

    def test_exact_eigenvalues(self):
        """Exact 1-form eigenvalues on S^3."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=5,
                                                      mode='exact')
        for i, (ev, mult) in enumerate(spectrum):
            l = i + 1
            expected_ev = l * (l + 2)
            assert abs(ev - expected_ev) < 1e-12, \
                f"l={l}: got {ev}, expected {expected_ev}"

    def test_first_exact_eigenvalue_is_3(self):
        """First exact eigenvalue is 3/R^2 (from l=1 scalar)."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=1,
                                                      mode='exact')
        ev, _ = spectrum[0]
        assert abs(ev - 3.0) < 1e-12

    def test_exact_multiplicities(self):
        """Exact multiplicity at l is (l+1)^2."""
        spectrum = HodgeSpectrum.one_form_eigenvalues(3, R=1.0, l_max=5,
                                                      mode='exact')
        for i, (ev, mult) in enumerate(spectrum):
            l = i + 1
            expected_mult = (l + 1) ** 2
            assert mult == expected_mult


class TestFirstNonzeroEigenvalue:
    """The spectral gap -- the key quantity for the mass gap."""

    def test_scalar_gap_s3(self):
        """Scalar gap on S^3(R=1) = 3 (from l=1: 1*3=3)."""
        gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 0, R=1.0)
        assert abs(gap - 3.0) < 1e-12

    def test_one_form_coexact_gap_s3(self):
        """Coexact 1-form gap on S^3(R=1) = 4. THIS IS THE MASS GAP."""
        gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 1, R=1.0,
                                                      mode='coexact')
        assert abs(gap - 4.0) < 1e-12

    def test_one_form_exact_gap_s3(self):
        """Exact 1-form gap on S^3(R=1) = 3."""
        gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 1, R=1.0,
                                                      mode='exact')
        assert abs(gap - 3.0) < 1e-12

    def test_one_form_all_gap_s3(self):
        """Overall 1-form gap on S^3(R=1) = 3 (exact is lower than coexact)."""
        gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 1, R=1.0,
                                                      mode='all')
        assert abs(gap - 3.0) < 1e-12

    def test_one_form_gap_s3_with_radius(self):
        """Coexact 1-form gap on S^3(R) = 4/R^2."""
        R = 2.2
        gap = HodgeSpectrum.first_nonzero_eigenvalue(3, 1, R=R,
                                                      mode='coexact')
        expected = 4.0 / R**2
        assert abs(gap - expected) < 1e-12


class TestBettiNumbers:
    """Betti numbers of S^n -- topology determines the gap."""

    def test_s3_betti(self):
        """S^3: b = [1, 0, 0, 1]."""
        betti = HodgeSpectrum.betti_numbers(3)
        assert betti == [1, 0, 0, 1]

    def test_s3_b1_is_zero(self):
        """b_1(S^3) = 0: no harmonic 1-forms => spectral gap => mass gap."""
        betti = HodgeSpectrum.betti_numbers(3)
        assert betti[1] == 0, "b_1(S^3) must be 0 -- this is why the mass gap exists"

    def test_s2_betti(self):
        betti = HodgeSpectrum.betti_numbers(2)
        assert betti == [1, 0, 1]

    def test_s1_betti(self):
        betti = HodgeSpectrum.betti_numbers(1)
        assert betti == [1, 1]
        assert betti[1] == 1, "S^1 has b_1=1, hence NO spectral gap"

    def test_s4_betti(self):
        betti = HodgeSpectrum.betti_numbers(4)
        assert betti == [1, 0, 0, 0, 1]


class TestHodgeDuality:
    """Hodge duality: Delta_p and Delta_{n-p} have the same nonzero spectrum."""

    def test_p_form_0_equals_n_form(self):
        """Delta_0 and Delta_3 on S^3 have the same eigenvalues."""
        spec_0 = HodgeSpectrum.p_form_eigenvalues(3, 0, R=1.0, l_max=5)
        spec_3 = HodgeSpectrum.p_form_eigenvalues(3, 3, R=1.0, l_max=5)

        for (ev0, _), (ev3, _) in zip(spec_0, spec_3):
            assert abs(ev0 - ev3) < 1e-12

    def test_p_form_1_equals_2_form(self):
        """Delta_1 and Delta_2 on S^3 have the same eigenvalues."""
        spec_1 = HodgeSpectrum.p_form_eigenvalues(3, 1, R=1.0, l_max=5)
        spec_2 = HodgeSpectrum.p_form_eigenvalues(3, 2, R=1.0, l_max=5)

        for (ev1, _), (ev2, _) in zip(spec_1, spec_2):
            assert abs(ev1 - ev2) < 1e-12

    def test_invalid_form_degree(self):
        """p < 0 or p > n should raise ValueError."""
        with pytest.raises(ValueError):
            HodgeSpectrum.p_form_eigenvalues(3, -1, R=1.0)
        with pytest.raises(ValueError):
            HodgeSpectrum.p_form_eigenvalues(3, 4, R=1.0)
