"""Tests for Ricci tensor computations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yang_mills_s3.geometry.ricci import RicciTensor


class TestOnSphere:
    """Test Ricci tensor on round spheres."""

    def test_s3_einstein_constant(self):
        """On S^3 of radius R, Ric = 2/R^2 * g."""
        R = 1.0
        result = RicciTensor.on_sphere(3, R)
        assert_allclose(result['einstein_constant'], 2.0, atol=1e-14)

    def test_s3_ricci_scalar(self):
        """Scalar curvature of S^3: n*(n-1)/R^2 = 6/R^2."""
        R = 1.0
        result = RicciTensor.on_sphere(3, R)
        assert_allclose(result['ricci_scalar'], 6.0, atol=1e-14)

    def test_s3_radius_scaling(self):
        """Ric = 2/R^2 * g for general R."""
        R = 2.2
        result = RicciTensor.on_sphere(3, R)
        assert_allclose(result['einstein_constant'], 2.0 / R**2, atol=1e-14)

    def test_s2(self):
        """On S^2: Ric = 1/R^2 * g."""
        R = 1.0
        result = RicciTensor.on_sphere(2, R)
        assert_allclose(result['einstein_constant'], 1.0, atol=1e-14)

    def test_s4(self):
        """On S^4: Ric = 3/R^2 * g."""
        R = 1.5
        result = RicciTensor.on_sphere(4, R)
        assert_allclose(result['einstein_constant'], 3.0 / R**2, atol=1e-14)

    def test_dimension(self):
        result = RicciTensor.on_sphere(3, 1.0)
        assert result['dimension'] == 3

    def test_ricci_equals_2_over_R2_times_g(self):
        """Core validation: Ric = 2*g/R^2 on S^3."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = RicciTensor.on_sphere(3, R)
            assert_allclose(
                result['einstein_constant'],
                2.0 / R**2,
                atol=1e-14,
                err_msg=f"Failed for R={R}",
            )


class TestOnLieGroup:
    """Test Ricci tensor on compact Lie groups."""

    def test_su2_matches_s3(self):
        """SU(2) is isomorphic to S^3, so Ric = 2/R^2 * g."""
        R = 1.0
        result = RicciTensor.on_lie_group('SU(2)', R)
        assert_allclose(result['ricci_on_1forms'], 2.0, atol=1e-14)
        assert result['dimension'] == 3

    def test_su3(self):
        """SU(3): dim=8, Ric = 3/(4*R^2) * g."""
        R = 1.0
        result = RicciTensor.on_lie_group('SU(3)', R)
        assert_allclose(result['ricci_on_1forms'], 3.0 / 4.0, atol=1e-14)
        assert result['dimension'] == 8

    def test_su3_ricci_scalar(self):
        """Scalar curvature = dim * lambda."""
        R = 1.0
        result = RicciTensor.on_lie_group('SU(3)', R)
        assert_allclose(result['ricci_scalar'], 8 * 3.0 / 4.0, atol=1e-14)

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            RicciTensor.on_lie_group('SO(3)', 1.0)


class TestEinsteinConstant:
    """Test the Einstein constant lambda = (n-1)/R^2."""

    def test_s3(self):
        assert_allclose(RicciTensor.einstein_constant(3, 1.0), 2.0)

    def test_s3_r2(self):
        assert_allclose(RicciTensor.einstein_constant(3, 2.0), 0.5)

    def test_general(self):
        for n in range(2, 8):
            for R in [0.5, 1.0, 3.0]:
                assert_allclose(
                    RicciTensor.einstein_constant(n, R),
                    (n - 1) / R**2,
                )
