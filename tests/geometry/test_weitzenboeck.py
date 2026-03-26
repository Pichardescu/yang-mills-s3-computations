"""Tests for the Weitzenboeck decomposition and spectral gaps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yang_mills_s3.geometry.weitzenboeck import Weitzenboeck, HBAR_C_MEV_FM


class TestDecomposition:
    """Test the Weitzenboeck decomposition on S^3."""

    def test_s3_ricci_term(self):
        """On S^3 of radius R=1, Ricci term = 2."""
        result = Weitzenboeck.decomposition('S3', R=1.0)
        assert_allclose(result['ricci_term'], 2.0, atol=1e-14)

    def test_s3_flat_connection(self):
        """Curvature endomorphism vanishes for flat connection."""
        result = Weitzenboeck.decomposition('S3', R=1.0)
        assert result['curvature_endomorphism'] == 0

    def test_s3_radius_scaling(self):
        """Ricci term scales as 2/R^2."""
        R = 3.5
        result = Weitzenboeck.decomposition('S3', R)
        assert_allclose(result['ricci_term'], 2.0 / R**2, atol=1e-14)

    def test_unknown_manifold_raises(self):
        with pytest.raises(ValueError):
            Weitzenboeck.decomposition('CP2', 1.0)


class TestSpectralGap0Forms:
    """Test eigenvalues of Delta_0 on S^3: l(l+2)/R^2."""

    def test_eigenvalues_s3_R1(self):
        R = 1.0
        for l in range(11):
            expected = l * (l + 2) / R**2
            actual = Weitzenboeck.spectral_gap_0forms(3, R, l)
            assert_allclose(actual, expected, atol=1e-14,
                            err_msg=f"Failed for l={l}")

    def test_eigenvalues_s3_general_R(self):
        R = 2.5
        for l in range(11):
            expected = l * (l + 2) / R**2
            actual = Weitzenboeck.spectral_gap_0forms(3, R, l)
            assert_allclose(actual, expected, atol=1e-14)

    def test_spectrum_list(self):
        R = 1.0
        spectrum = Weitzenboeck.spectrum_0forms(3, R, l_max=10)
        assert len(spectrum) == 11
        for l, val in enumerate(spectrum):
            assert_allclose(val, l * (l + 2), atol=1e-14)


class TestSpectralGap1Forms:
    """Test coexact eigenvalues of Delta_1 on S^3: (k+1)^2/R^2."""

    def test_lowest_eigenvalue(self):
        """Lowest coexact eigenvalue of Delta_1 on S^3 (R=1) is 4."""
        val = Weitzenboeck.spectral_gap_1forms(3, 1.0, l=1)
        assert_allclose(val, 4.0, atol=1e-14)

    def test_eigenvalues_s3_R1(self):
        """Coexact eigenvalues of Delta_1 on S^3 (R=1) for k=1..10."""
        R = 1.0
        for k in range(1, 11):
            expected = (k + 1) ** 2 / R**2
            actual = Weitzenboeck.spectral_gap_1forms(3, R, l=k)
            assert_allclose(actual, expected, atol=1e-14,
                            err_msg=f"Failed for k={k}")

    def test_eigenvalues_s3_general_R(self):
        R = 1.7
        for k in range(1, 11):
            expected = (k + 1) ** 2 / R**2
            actual = Weitzenboeck.spectral_gap_1forms(3, R, l=k)
            assert_allclose(actual, expected, atol=1e-14)

    def test_spectrum_list(self):
        R = 1.0
        spectrum = Weitzenboeck.spectrum_1forms(3, R, l_max=10)
        assert len(spectrum) == 10  # k = 1..10
        for i, val in enumerate(spectrum):
            k = i + 1
            assert_allclose(val, (k + 1) ** 2, atol=1e-14)


class TestMassGap:
    """Test the Yang-Mills mass gap prediction."""

    def test_formula(self):
        """mass_gap = 2 * hbar*c / R."""
        R = 1.0
        gap = Weitzenboeck.mass_gap_yang_mills(R)
        expected = 2.0 * HBAR_C_MEV_FM
        assert_allclose(gap, expected, atol=1e-6)

    def test_at_R_2_2_fm(self):
        """At R=2.2 fm, gap should be ~179 MeV."""
        R = 2.2
        gap = Weitzenboeck.mass_gap_yang_mills(R)
        expected = 2.0 * 197.3269804 / 2.2
        assert_allclose(gap, expected, atol=1e-6)
        # Sanity: should be in the ballpark of 180 MeV
        assert 130 < gap < 230, f"Gap {gap} MeV not in QCD range"

    def test_scaling(self):
        """Gap scales as 1/R."""
        g1 = Weitzenboeck.mass_gap_yang_mills(1.0)
        g2 = Weitzenboeck.mass_gap_yang_mills(2.0)
        assert_allclose(g1 / g2, 2.0, atol=1e-12)
