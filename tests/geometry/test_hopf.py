"""Tests for the Hopf fibration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yang_mills_s3.geometry.hopf_fibration import HopfFibration


class TestProjection:
    """Test the Hopf map S^3 -> S^2."""

    def test_north_pole(self):
        """z1=1, z2=0 maps to north pole (0,0,1)."""
        x, y, z = HopfFibration.projection(1 + 0j, 0 + 0j)
        assert_allclose((x, y, z), (0, 0, 1), atol=1e-14)

    def test_south_pole(self):
        """z1=0, z2=1 maps to south pole (0,0,-1)."""
        x, y, z = HopfFibration.projection(0 + 0j, 1 + 0j)
        assert_allclose((x, y, z), (0, 0, -1), atol=1e-14)

    def test_equator(self):
        """z1=1/sqrt(2), z2=1/sqrt(2) maps to (1,0,0)."""
        s = 1.0 / np.sqrt(2)
        x, y, z = HopfFibration.projection(s + 0j, s + 0j)
        assert_allclose((x, y, z), (1, 0, 0), atol=1e-14)

    def test_output_on_unit_s2(self):
        """Projection should land on unit S^2."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            v = rng.normal(size=4)
            v /= np.linalg.norm(v)
            z1 = v[0] + 1j * v[1]
            z2 = v[2] + 1j * v[3]
            x, y, z = HopfFibration.projection(z1, z2)
            assert_allclose(x**2 + y**2 + z**2, 1.0, atol=1e-12)

    def test_fiber_invariance(self):
        """Points on the same fiber should project to the same point on S^2."""
        z1, z2 = 0.6 + 0.2j, 0.3 - 0.7j
        norm = np.sqrt(abs(z1)**2 + abs(z2)**2)
        z1, z2 = z1 / norm, z2 / norm
        p_base = HopfFibration.projection(z1, z2)

        for t in np.linspace(0, 2 * np.pi, 20):
            phase = np.exp(1j * t)
            p = HopfFibration.projection(phase * z1, phase * z2)
            assert_allclose(p, p_base, atol=1e-12)


class TestFiber:
    """Test the fiber reconstruction."""

    def test_fiber_on_s3(self):
        """All fiber points should lie on unit S^3."""
        pts = HopfFibration.fiber((0, 0, 1), num_points=200)
        norms = np.sqrt(np.sum(pts**2, axis=1))
        assert_allclose(norms, 1.0, atol=1e-12)

    def test_fiber_projects_to_base_point(self):
        """Every point on the fiber should project back to the base point."""
        base = (0.5, 0.5, 1 / np.sqrt(2))
        # normalise
        norm = np.sqrt(sum(c**2 for c in base))
        base = tuple(c / norm for c in base)

        pts = HopfFibration.fiber(base, num_points=100)
        for pt in pts:
            z1 = pt[0] + 1j * pt[1]
            z2 = pt[2] + 1j * pt[3]
            proj = HopfFibration.projection(z1, z2)
            assert_allclose(proj, base, atol=1e-10)

    def test_fiber_south_pole(self):
        """Fiber over south pole."""
        pts = HopfFibration.fiber((0, 0, -1), num_points=50)
        norms = np.sqrt(np.sum(pts**2, axis=1))
        assert_allclose(norms, 1.0, atol=1e-12)


class TestConnectionAndCurvature:
    """Test connection form and curvature descriptions."""

    def test_connection_returns_dict(self):
        result = HopfFibration.connection_1form()
        assert 'formula' in result
        assert 'Im' in result['formula']

    def test_curvature_returns_dict(self):
        result = HopfFibration.curvature()
        assert 'formula' in result
        assert 'total_flux' in result


class TestFirstChernNumber:
    """Test the first Chern number."""

    def test_chern_number_is_one(self):
        """c_1 of the Hopf bundle = 1."""
        assert HopfFibration.first_chern_number() == 1


class TestLinkingNumber:
    """Test that distinct Hopf fibers have linking number 1."""

    def test_linking_number_north_south(self):
        """Fibers over north and south poles should have |linking number| = 1."""
        f1 = HopfFibration.fiber((0, 0, 1), num_points=500)
        f2 = HopfFibration.fiber((0, 0, -1), num_points=500)
        lk = HopfFibration.linking_number(f1, f2)
        assert abs(lk) == 1, f"Expected |linking number| = 1, got {lk}"

    def test_linking_number_equatorial(self):
        """Fibers over two equatorial points should have |linking number| = 1."""
        f1 = HopfFibration.fiber((1, 0, 0), num_points=500)
        f2 = HopfFibration.fiber((0, 1, 0), num_points=500)
        lk = HopfFibration.linking_number(f1, f2)
        assert abs(lk) == 1, f"Expected |linking number| = 1, got {lk}"
