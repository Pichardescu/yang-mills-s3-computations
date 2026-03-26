"""
S3 Coordinates — Coordinate systems on the 3-sphere S^3.

Provides Euler/Hopf parametrizations, quaternion-to-rotation conversion,
and the round metric on S^3 of radius R.
"""

import numpy as np
import sympy as sp
from sympy import cos, sin, Matrix, symbols, pi, trigsimp


class S3Coordinates:
    """Coordinate systems and geometric quantities on S^3 of radius R."""

    def __init__(self, R=1.0):
        self.R = R

    # ------------------------------------------------------------------
    # Euler / Hopf angles  -->  quaternion on S^3
    # ------------------------------------------------------------------
    def euler_angles(self, chi, theta, phi):
        """
        Standard Hopf-Euler parametrization of S^3.

        Parameters
        ----------
        chi   : float, chi in [0, pi]
        theta : float, theta in [0, pi]
        phi   : float, phi in [0, 2*pi)

        Returns
        -------
        (w, x, y, z) : quaternion coordinates on the unit S^3

        Convention (standard Hopf coordinates):
            w = cos(chi/2) * cos(theta/2)
            x = cos(chi/2) * sin(theta/2)
            y = sin(chi/2) * cos(phi)
            z = sin(chi/2) * sin(phi)
        """
        c_chi = np.cos(chi / 2)
        s_chi = np.sin(chi / 2)
        c_th  = np.cos(theta / 2)
        s_th  = np.sin(theta / 2)

        w = c_chi * c_th
        x = c_chi * s_th
        y = s_chi * np.cos(phi)
        z = s_chi * np.sin(phi)
        return (w, x, y, z)

    # ------------------------------------------------------------------
    # Hopf coordinates  -->  C^2
    # ------------------------------------------------------------------
    def hopf_coordinates(self, eta, xi1, xi2):
        """
        Hopf coordinates on S^3 of radius R embedded in C^2.

        Parameters
        ----------
        eta  : float, eta in [0, pi/2]
        xi1  : float, xi1 in [0, 2*pi)
        xi2  : float, xi2 in [0, 2*pi)

        Returns
        -------
        (z1, z2) : complex numbers with |z1|^2 + |z2|^2 = R^2
        """
        R = self.R
        z1 = R * np.cos(eta) * np.exp(1j * xi1)
        z2 = R * np.sin(eta) * np.exp(1j * xi2)
        return (z1, z2)

    # ------------------------------------------------------------------
    # Quaternion  -->  SO(3) rotation matrix  (2:1 covering SU(2)->SO(3))
    # ------------------------------------------------------------------
    @staticmethod
    def quaternion_to_rotation(q):
        """
        Convert unit quaternion to 3x3 rotation matrix.

        Parameters
        ----------
        q : tuple (w, x, y, z) with w^2+x^2+y^2+z^2 = 1

        Returns
        -------
        R : 3x3 numpy array, element of SO(3)
        """
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
            [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
        ])

    # ------------------------------------------------------------------
    # Volume of S^3
    # ------------------------------------------------------------------
    @staticmethod
    def volume(R):
        """Volume of S^3 of radius R: 2*pi^2*R^3."""
        return 2 * np.pi**2 * R**3

    # ------------------------------------------------------------------
    # Round metric on S^3 (symbolic, sympy)
    # ------------------------------------------------------------------
    @staticmethod
    def metric_round(R_val=None):
        """
        The round metric on S^3 in Hopf-Euler coordinates (chi, theta, phi).

        ds^2 = (R^2/4)[ d(chi)^2 + d(theta)^2 + d(phi)^2
                         + 2*cos(theta)*d(chi)*d(phi) ]

        Parameters
        ----------
        R_val : numeric or sympy symbol for radius (default: sympy symbol R)

        Returns
        -------
        g : 3x3 sympy Matrix (metric tensor in coordinates chi, theta, phi)
        """
        R = symbols('R', positive=True) if R_val is None else R_val

        chi, theta, phi = symbols('chi theta phi', real=True)
        factor = R**2 / 4

        g = factor * Matrix([
            [1,            0,  cos(theta)],
            [0,            1,  0],
            [cos(theta),   0,  1],
        ])
        return g
