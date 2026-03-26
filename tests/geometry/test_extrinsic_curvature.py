"""Tests for extrinsic curvature of S^n in R^{n+1}."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yang_mills_s3.geometry.extrinsic_curvature import ExtrinsicCurvature


class TestSecondFundamentalForm:
    """Test second fundamental form K_ij = (1/R) g_ij."""

    def test_s2_in_r3(self):
        """S^2(R) in R^3: all principal curvatures = 1/R."""
        R = 1.0
        result = ExtrinsicCurvature.second_fundamental_form(2, R)
        assert len(result['principal_curvatures']) == 2
        for kappa in result['principal_curvatures']:
            assert abs(kappa - 1.0 / R) < 1e-12

    def test_s3_in_r4(self):
        """S^3(R) in R^4: all 3 principal curvatures = 1/R."""
        R = 2.2
        result = ExtrinsicCurvature.second_fundamental_form(3, R)
        assert len(result['principal_curvatures']) == 3
        for kappa in result['principal_curvatures']:
            assert abs(kappa - 1.0 / R) < 1e-12

    def test_umbilical(self):
        """S^n is always umbilical."""
        for n in [2, 3, 4, 7]:
            result = ExtrinsicCurvature.second_fundamental_form(n, 1.0)
            assert result['is_umbilical'] is True

    def test_k_scalar(self):
        """K_scalar = 1/R for any n and R."""
        R = 3.5
        result = ExtrinsicCurvature.second_fundamental_form(5, R)
        assert abs(result['K_scalar'] - 1.0 / R) < 1e-12


class TestMeanCurvature:
    """Test mean curvature H = n/R."""

    def test_s2(self):
        """H(S^2) = 2/R."""
        R = 1.0
        assert abs(ExtrinsicCurvature.mean_curvature(2, R) - 2.0 / R) < 1e-12

    def test_s3(self):
        """H(S^3) = 3/R."""
        R = 2.2
        assert abs(ExtrinsicCurvature.mean_curvature(3, R) - 3.0 / R) < 1e-12

    def test_s4(self):
        """H(S^4) = 4/R."""
        R = 0.5
        assert abs(ExtrinsicCurvature.mean_curvature(4, R) - 4.0 / R) < 1e-12


class TestNormSquaredK:
    """Test |K|^2 = n/R^2."""

    def test_s3(self):
        """For S^3: |K|^2 = 3/R^2."""
        R = 1.0
        assert abs(ExtrinsicCurvature.norm_squared_K(3, R) - 3.0) < 1e-12

    def test_scaling(self):
        """Check 1/R^2 scaling."""
        R = 2.5
        assert abs(ExtrinsicCurvature.norm_squared_K(3, R) - 3.0 / R**2) < 1e-12


class TestJacobiOperator:
    """Test eigenvalues of the Jacobi operator J = Delta + |K|^2."""

    def test_all_positive_s3(self):
        """All Jacobi eigenvalues on S^3 are positive (stable embedding)."""
        R = 1.0
        spectrum = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R, l_max=20)
        for eigenvalue, mult in spectrum:
            assert eigenvalue > 0, f"Found non-positive eigenvalue: {eigenvalue}"

    def test_eigenvalues_s3(self):
        """Verify l(l+2)/R^2 + 3/R^2 for l=0..5 on S^3."""
        R = 1.0
        spectrum = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R, l_max=5)
        for l in range(6):
            expected = l * (l + 2) / R**2 + 3.0 / R**2
            actual = spectrum[l][0]
            assert abs(actual - expected) < 1e-12, (
                f"l={l}: expected {expected}, got {actual}"
            )

    def test_multiplicities_s3(self):
        """On S^3, scalar harmonic multiplicity is (l+1)^2."""
        R = 1.0
        spectrum = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R, l_max=5)
        for l in range(6):
            expected_mult = (l + 1) ** 2
            assert spectrum[l][1] == expected_mult, (
                f"l={l}: expected mult {expected_mult}, got {spectrum[l][1]}"
            )

    def test_multiplicities_s2(self):
        """On S^2, scalar harmonic multiplicity is 2l+1."""
        R = 1.0
        spectrum = ExtrinsicCurvature.jacobi_operator_eigenvalues(2, R, l_max=5)
        for l in range(6):
            expected_mult = 2 * l + 1
            assert spectrum[l][1] == expected_mult, (
                f"l={l}: expected mult {expected_mult}, got {spectrum[l][1]}"
            )

    def test_first_nontrivial_s3(self):
        """First nontrivial Jacobi eigenvalue on S^3 is 6/R^2."""
        R = 1.0
        spectrum = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R, l_max=5)
        # l=1 eigenvalue: 1*(1+2)/1 + 3/1 = 3 + 3 = 6
        assert abs(spectrum[1][0] - 6.0) < 1e-12

    def test_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R1 = 1.0
        R2 = 3.0
        spec1 = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R1, l_max=5)
        spec2 = ExtrinsicCurvature.jacobi_operator_eigenvalues(3, R2, l_max=5)
        for l in range(6):
            ratio = spec1[l][0] / spec2[l][0]
            expected_ratio = R2**2 / R1**2
            assert abs(ratio - expected_ratio) < 1e-10, (
                f"l={l}: ratio {ratio} != expected {expected_ratio}"
            )


class TestStability:
    """Test stability index of S^n in flat space."""

    def test_s3_stable(self):
        """S^3 is stable: index = 0."""
        result = ExtrinsicCurvature.stability_index(3, 1.0)
        assert result['index'] == 0
        assert result['stable'] is True

    def test_s2_stable(self):
        """S^2 is stable: index = 0."""
        result = ExtrinsicCurvature.stability_index(2, 1.0)
        assert result['index'] == 0
        assert result['stable'] is True

    def test_smallest_eigenvalue(self):
        """Smallest Jacobi eigenvalue is n/R^2."""
        R = 2.0
        for n in [2, 3, 4]:
            result = ExtrinsicCurvature.stability_index(n, R)
            assert abs(result['smallest_eigenvalue'] - n / R**2) < 1e-12


class TestJacobiMassGap:
    """Test mass gap from Jacobi operator."""

    def test_s3_trivial_mode(self):
        """Trivial mode on S^3: 3/R^2."""
        R = 1.0
        result = ExtrinsicCurvature.jacobi_mass_gap(3, R)
        assert abs(result['trivial_mode'] - 3.0) < 1e-12

    def test_s3_nontrivial_gap(self):
        """Nontrivial gap on S^3: 6/R^2."""
        R = 1.0
        result = ExtrinsicCurvature.jacobi_mass_gap(3, R)
        assert abs(result['nontrivial_gap'] - 6.0) < 1e-12

    def test_s4_nontrivial_gap(self):
        """Nontrivial gap on S^4: 8/R^2."""
        R = 1.0
        result = ExtrinsicCurvature.jacobi_mass_gap(4, R)
        assert abs(result['nontrivial_gap'] - 8.0) < 1e-12

    def test_l_gap(self):
        """Gap mode is always l=1."""
        result = ExtrinsicCurvature.jacobi_mass_gap(3, 1.0)
        assert result['l_gap'] == 1


class TestGaussCodazzi:
    """Test Gauss-Codazzi relations."""

    def test_constant_curvature(self):
        """Gauss equation derives sectional curvature 1/R^2."""
        R = 2.2
        result = ExtrinsicCurvature.gauss_codazzi(3, R)
        assert abs(result['intrinsic_curvature'] - 1.0 / R**2) < 1e-12

    def test_codazzi(self):
        """Codazzi equation is satisfied for umbilical spheres."""
        result = ExtrinsicCurvature.gauss_codazzi(3, 1.0)
        assert result['codazzi_satisfied'] is True

    def test_gauss_consistency_with_ricci(self):
        """Sectional curvature 1/R^2 implies Ricci = (n-1)/R^2."""
        R = 1.5
        n = 3
        result = ExtrinsicCurvature.gauss_codazzi(n, R)
        # From constant sectional curvature K = 1/R^2,
        # Ric = (n-1)*K*g = (n-1)/R^2 * g
        expected_einstein = (n - 1) * result['intrinsic_curvature']
        assert abs(expected_einstein - (n - 1) / R**2) < 1e-12


class TestConnectionToYangMills:
    """Test structural analogy with Yang-Mills operator."""

    def test_s3_gaps(self):
        """On S^3: Jacobi gap = 6/R^2, YM linearized gap = 4/R^2."""
        R = 1.0
        result = ExtrinsicCurvature.connection_to_yang_mills(3, R)
        assert abs(result['jacobi_gap'] - 6.0) < 1e-12
        assert abs(result['ym_gap_linearized'] - 4.0) < 1e-12

    def test_s4_gaps(self):
        """On S^4: Jacobi gap = 8/R^2, YM linearized gap = 6/R^2."""
        R = 1.0
        result = ExtrinsicCurvature.connection_to_yang_mills(4, R)
        assert abs(result['jacobi_gap'] - 8.0) < 1e-12
        assert abs(result['ym_gap_linearized'] - 6.0) < 1e-12

    def test_label(self):
        """This is a PROPOSITION, not a THEOREM."""
        result = ExtrinsicCurvature.connection_to_yang_mills(3, 1.0)
        assert result['label'] == 'PROPOSITION'
