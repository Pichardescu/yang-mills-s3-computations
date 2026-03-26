"""Tests for S3 coordinate systems."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yang_mills_s3.geometry.s3_coordinates import S3Coordinates


class TestEulerAngles:
    """Test the Euler/Hopf angle parametrization."""

    def test_identity_quaternion(self):
        """chi=0, theta=0 gives (1,0,0,0)."""
        s3 = S3Coordinates()
        q = s3.euler_angles(0, 0, 0)
        assert_allclose(q, (1, 0, 0, 0), atol=1e-14)

    def test_unit_norm(self):
        """All quaternions should have unit norm."""
        s3 = S3Coordinates()
        rng = np.random.default_rng(42)
        for _ in range(100):
            chi = rng.uniform(0, np.pi)
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            q = s3.euler_angles(chi, theta, phi)
            norm = np.sqrt(sum(c**2 for c in q))
            assert_allclose(norm, 1.0, atol=1e-14)

    def test_specific_values(self):
        """chi=pi, theta=0, phi=0 => w=0, x=0, y=1, z=0."""
        s3 = S3Coordinates()
        q = s3.euler_angles(np.pi, 0, 0)
        assert_allclose(q, (0, 0, 1, 0), atol=1e-14)


class TestHopfCoordinates:
    """Test the Hopf coordinate parametrization in C^2."""

    def test_on_unit_sphere(self):
        """Points should satisfy |z1|^2 + |z2|^2 = R^2."""
        s3 = S3Coordinates(R=1.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            eta = rng.uniform(0, np.pi / 2)
            xi1 = rng.uniform(0, 2 * np.pi)
            xi2 = rng.uniform(0, 2 * np.pi)
            z1, z2 = s3.hopf_coordinates(eta, xi1, xi2)
            assert_allclose(abs(z1)**2 + abs(z2)**2, 1.0, atol=1e-14)

    def test_radius_scaling(self):
        """With R=3, |z1|^2 + |z2|^2 = 9."""
        s3 = S3Coordinates(R=3.0)
        z1, z2 = s3.hopf_coordinates(np.pi / 4, 0.5, 1.0)
        assert_allclose(abs(z1)**2 + abs(z2)**2, 9.0, atol=1e-14)

    def test_north_pole(self):
        """eta=0 => z2=0, z1 = R*exp(i*xi1)."""
        s3 = S3Coordinates(R=2.0)
        z1, z2 = s3.hopf_coordinates(0, 1.23, 0)
        assert_allclose(abs(z2), 0.0, atol=1e-14)
        assert_allclose(abs(z1), 2.0, atol=1e-14)


class TestQuaternionToRotation:
    """Test the 2:1 covering map SU(2) -> SO(3)."""

    def test_identity(self):
        """(1,0,0,0) -> identity matrix."""
        R = S3Coordinates.quaternion_to_rotation((1, 0, 0, 0))
        assert_allclose(R, np.eye(3), atol=1e-14)

    def test_minus_identity_also_maps_to_identity(self):
        """(-1,0,0,0) also maps to identity (2:1 covering)."""
        R = S3Coordinates.quaternion_to_rotation((-1, 0, 0, 0))
        assert_allclose(R, np.eye(3), atol=1e-14)

    def test_is_orthogonal(self):
        """Output should be an orthogonal matrix with det=+1."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = S3Coordinates.quaternion_to_rotation(tuple(q))
            # R^T R = I
            assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            # det R = 1
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_rotation_around_z(self):
        """Quaternion (cos(t/2), 0, 0, sin(t/2)) -> rotation by t around z."""
        t = np.pi / 3
        q = (np.cos(t / 2), 0, 0, np.sin(t / 2))
        R = S3Coordinates.quaternion_to_rotation(q)
        expected = np.array([
            [np.cos(t), -np.sin(t), 0],
            [np.sin(t),  np.cos(t), 0],
            [0,          0,         1],
        ])
        assert_allclose(R, expected, atol=1e-14)


class TestVolume:
    """Test the volume formula for S^3."""

    def test_unit_sphere(self):
        """Vol(S^3, R=1) = 2*pi^2."""
        vol = S3Coordinates.volume(1.0)
        assert_allclose(vol, 2 * np.pi**2, atol=1e-12)

    def test_radius_scaling(self):
        """Vol scales as R^3."""
        R = 2.5
        vol = S3Coordinates.volume(R)
        assert_allclose(vol, 2 * np.pi**2 * R**3, atol=1e-10)


class TestMetricRound:
    """Test the symbolic round metric on S^3."""

    def test_is_3x3(self):
        metric = S3Coordinates.metric_round()
        assert metric.shape == (3, 3)

    def test_symmetry(self):
        """Metric tensor must be symmetric."""
        g = S3Coordinates.metric_round()
        assert g.equals(g.T)

    def test_positive_definite_at_generic_point(self):
        """At theta=pi/3, the metric should be positive definite."""
        import sympy
        R_sym = sympy.Symbol('R', positive=True)
        g = S3Coordinates.metric_round(R_sym)
        # Substitute a generic interior point and R=1
        free = g.free_symbols
        subs = {s: 1 for s in free if str(s) == 'R'}
        subs.update({s: sympy.pi / 3 for s in free if str(s) == 'theta'})
        g_num = g.subs(subs)
        eigenvals = g_num.eigenvals()
        for ev in eigenvals:
            assert float(ev) > 0, f"Eigenvalue {ev} is not positive"

    def test_degenerate_at_theta0(self):
        """At theta=0, the metric has a coordinate singularity (det=0)."""
        import sympy
        R_sym = sympy.Symbol('R', positive=True)
        g = S3Coordinates.metric_round(R_sym)
        free = g.free_symbols
        subs = {s: 1 for s in free if str(s) == 'R'}
        subs.update({s: 0 for s in free if str(s) == 'theta'})
        g_num = g.subs(subs)
        det = float(g_num.det())
        assert det == 0, "Expected coordinate singularity at theta=0"
