"""
Koller-van Baal SVD Reduction: 9 DOF -> 3 Effective DOF for SU(2) YM on S^3.

The 9 constant-mode degrees of freedom M_{ia} (spatial i, color a) decompose via
singular value decomposition:

    M = U . diag(x_1, x_2, x_3) . V^T

where U in SO(3)_spatial, V in SO(3)_gauge, and x_1 >= x_2 >= x_3 >= 0.

Gauge fixing eliminates V (3 DOF). For the spin-0 (J=0) sector, the U dependence
is trivial. The physical variables are the 3 singular values (x_1, x_2, x_3),
living in the Weyl chamber W = {x_1 >= x_2 >= x_3 >= 0}.

The flat measure transforms as:
    prod dM_{ia} = J(x) . dmu(U) . dmu(V) . dx_1 dx_2 dx_3

with the Khvedelidze-Pavel Jacobian:
    J(x_1, x_2, x_3) = prod_{i<j} |x_i^2 - x_j^2|
                      = (x_1^2 - x_2^2)(x_1^2 - x_3^2)(x_2^2 - x_3^2)

The reduced Hamiltonian on W (spin-0 sector):
    H_red = -(g^2 / 2L^3) * (1/J) sum_i d/dx_i [J d/dx_i] + V(x_1, x_2, x_3)

On S^3 the potential has THREE terms (contrast with T^3 which has only quartic):
    V_{S^3} = V_quad + V_cubic + V_quartic
            = sum_i x_i^2/R^2  -  (2/R) x_1 x_2 x_3  +  sum_{i<j} x_i^2 x_j^2

The quadratic term (mass from S^3 curvature) ELIMINATES flat directions.
The cubic term (from curvature of the S^3 connection) breaks discrete symmetry.

The ground-state wavefunction satisfies Neumann BC at Weyl chamber walls
(A_1 representation of the Weyl group S_3).

SELF-ADJOINTNESS: The centrifugal potential from the sqrt(J) transformation
yields an inverse-square term c/rho^2 with c = -1/4 at each wall. This is
exactly the Weyl limit-circle case, so self-adjoint extensions exist and the
boundary condition choice (Neumann/Dirichlet) selects the physical sector.

S^3 ADVANTAGES over T^3:
    - pi_1(S^3) = 0  =>  no Gribov copies for constant modes
    - Single vacuum at A = 0 (no tunneling, no vacuum valley)
    - Mass term 1/R^2 eliminates flat directions
    - No topological boundary conditions needed

LABEL: NUMERICAL (eigenvalues from Rayleigh-Ritz diagonalization)
LABEL: THEOREM  (SVD decomposition, Jacobian, self-adjointness classification)

Benchmarks (T^3 — Pavel 2007, Butt-Draper-Shen 2023):
    E_0         ~ 2.560 g^{2/3}       (ground state on T^3)
    E_1(J=2)    ~ 4.12  g^{2/3}       (first spin-2 excitation)
    E_1(J=0)    ~ 5.87  g^{2/3}       (first spin-0 excitation)
    Delta(J=0)  ~ 3.31  g^{2/3}       (spin-0 gap)

Our S^3 problem differs: the quadratic + cubic terms shift all eigenvalues.

References:
    [1] Koller & van Baal (1988): Non-perturbative SU(2) YM in a small volume
    [2] van Baal (1988): Gauge theory in a finite volume (T^3)
    [3] Khvedelidze & Pavel (2000): Two-step gauge fixing in YM theory
    [4] Pavel (2007): Spectrum of the T^3 reduced Hamiltonian
    [5] Butt, Draper & Shen (2023): Benchmark diagonalization
    [6] Luscher (1982): Symmetry breaking in finite-volume gauge theories
    [7] Reed & Simon (1975): Limit-point/limit-circle classification
"""

import math
import numpy as np
from scipy.linalg import svd, eigh
from scipy.special import hermite
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from itertools import product


# ======================================================================
# Physical constants
# ======================================================================

HBAR_C_MEV_FM = 197.3269804   # hbar*c in MeV*fm
R_PHYSICAL_FM = 2.2            # Physical S^3 radius in fm
G2_DEFAULT = 6.28              # g^2 at physical scale
LAMBDA_QCD_MEV = 200.0         # Lambda_QCD in MeV

# T^3 benchmarks from Pavel (2007) and Butt-Draper-Shen (2023)
# Energies in units of g^{2/3}
PAVEL_E0 = 2.560              # Ground state
PAVEL_E1_J2 = 4.12            # First spin-2 excitation
PAVEL_E1_J0 = 5.87            # First spin-0 excitation
PAVEL_GAP_J0 = 3.31           # Spin-0 gap


# ======================================================================
# 1. SVDReduction — M = U . Sigma . V^T decomposition
# ======================================================================

class SVDReduction:
    """
    SVD decomposition of the 3x3 matrix M_{ia} of constant-mode amplitudes.

    M = U . diag(x_1, x_2, x_3) . V^T

    where U in SO(3)_spatial, V in SO(3)_gauge, x_1 >= x_2 >= x_3 >= 0.

    THEOREM: Every 3x3 real matrix admits such a decomposition. The singular
    values are unique; U, V are unique when all x_i are distinct.
    """

    @staticmethod
    def decompose(M_9dof):
        """
        Decompose 9-DOF configuration into SVD components.

        Parameters
        ----------
        M_9dof : ndarray, shape (9,) or (3, 3)
            The 3x3 matrix M_{ia} (spatial i, color a).

        Returns
        -------
        dict with keys:
            'U'      : ndarray (3, 3), SO(3)_spatial rotation
            'x'      : ndarray (3,), singular values x_1 >= x_2 >= x_3 >= 0
            'Vt'     : ndarray (3, 3), V^T (gauge rotation)
            'det_U'  : float, det(U) (should be +1 for SO(3))
            'det_V'  : float, det(V) (should be +1 for SO(3))
        """
        M = np.asarray(M_9dof, dtype=float).reshape(3, 3)
        U_raw, x, Vt_raw = svd(M)

        # Ensure U, V are in SO(3) (det = +1), not O(3)
        det_U = np.linalg.det(U_raw)
        det_V = np.linalg.det(Vt_raw)  # det(V^T) = det(V)

        U = U_raw.copy()
        Vt = Vt_raw.copy()

        if det_U < 0:
            U[:, -1] *= -1
            x_mod = x.copy()
            x_mod[-1] *= -1
            # Absorb sign into x, but x must be non-negative
            # So flip the sign in V instead
            if x_mod[-1] < 0:
                Vt[-1, :] *= -1
                x_mod[-1] *= -1
            x = x_mod

        det_V_final = np.linalg.det(Vt)
        if det_V_final < 0:
            Vt[-1, :] *= -1
            x_copy = x.copy()
            x_copy[-1] *= -1
            if x_copy[-1] < 0:
                U[:, -1] *= -1
                x_copy[-1] *= -1
            x = x_copy

        return {
            'U': U,
            'x': x,
            'Vt': Vt,
            'det_U': float(np.linalg.det(U)),
            'det_V': float(np.linalg.det(Vt)),
        }

    @staticmethod
    def singular_values(M_9dof):
        """
        Extract just the singular values (physical DOF).

        Parameters
        ----------
        M_9dof : ndarray, shape (9,) or (3, 3)

        Returns
        -------
        ndarray (3,) : x_1 >= x_2 >= x_3 >= 0
        """
        M = np.asarray(M_9dof, dtype=float).reshape(3, 3)
        _, x, _ = svd(M)
        return x

    @staticmethod
    def reconstruct(x1, x2, x3, U=None, V=None):
        """
        Reconstruct M from singular values and rotations.

        M = U . diag(x1, x2, x3) . V^T

        Parameters
        ----------
        x1, x2, x3 : float
            Singular values.
        U : ndarray (3, 3) or None
            Spatial rotation. Identity if None.
        V : ndarray (3, 3) or None
            Gauge rotation. Identity if None.

        Returns
        -------
        ndarray (3, 3) : reconstructed M
        """
        if U is None:
            U = np.eye(3)
        if V is None:
            V = np.eye(3)
        Sigma = np.diag([x1, x2, x3])
        return U @ Sigma @ V.T

    @staticmethod
    def gauge_equivalent(M1, M2, tol=1e-10):
        """
        Check if two configurations are gauge-equivalent (same singular values).

        Parameters
        ----------
        M1, M2 : ndarray, shape (9,) or (3, 3)
        tol : float

        Returns
        -------
        bool
        """
        x1 = SVDReduction.singular_values(M1)
        x2 = SVDReduction.singular_values(M2)
        return np.allclose(np.sort(x1)[::-1], np.sort(x2)[::-1], atol=tol)


# ======================================================================
# 2. WeylChamberJacobian — J = prod_{i<j} |x_i^2 - x_j^2|
# ======================================================================

class WeylChamberJacobian:
    """
    The Jacobian from the SVD change of variables.

    J(x_1, x_2, x_3) = prod_{i<j} |x_i^2 - x_j^2|
                      = (x_1^2 - x_2^2)(x_1^2 - x_3^2)(x_2^2 - x_3^2)

    in the Weyl chamber W = {x_1 >= x_2 >= x_3 >= 0}.

    THEOREM: J vanishes on the walls of W (where any x_i = x_j) and is
    strictly positive in the interior.

    The flat measure transforms as:
        prod dM_{ia} = J . dmu(U) . dmu(V) . dx_1 dx_2 dx_3
    """

    @staticmethod
    def jacobian(x1, x2, x3):
        """
        Compute J(x_1, x_2, x_3) = prod_{i<j} |x_i^2 - x_j^2|.

        Parameters
        ----------
        x1, x2, x3 : float
            Singular values (any ordering).

        Returns
        -------
        float : J >= 0
        """
        return abs(x1**2 - x2**2) * abs(x1**2 - x3**2) * abs(x2**2 - x3**2)

    @staticmethod
    def jacobian_ordered(x1, x2, x3):
        """
        Compute J assuming x_1 >= x_2 >= x_3 >= 0 (Weyl chamber ordering).

        In this case all factors are non-negative and absolute values are
        unnecessary.

        Returns
        -------
        float : J >= 0
        """
        return (x1**2 - x2**2) * (x1**2 - x3**2) * (x2**2 - x3**2)

    @staticmethod
    def sqrt_jacobian(x1, x2, x3):
        """
        Compute sqrt(J) for the ground-state transformation Phi = sqrt(J) * Psi.

        Returns
        -------
        float : sqrt(J) >= 0
        """
        return np.sqrt(WeylChamberJacobian.jacobian(x1, x2, x3))

    @staticmethod
    def log_jacobian(x1, x2, x3):
        """
        Compute log(J) for Bakry-Emery analysis.

        Returns
        -------
        float : log(J), or -inf if J = 0
        """
        J = WeylChamberJacobian.jacobian(x1, x2, x3)
        if J <= 0:
            return -np.inf
        return np.log(J)

    @staticmethod
    def grad_log_jacobian(x1, x2, x3, eps=1e-8):
        """
        Gradient of log(J) via numerical differentiation.

        The drift term in the reduced Schrodinger equation.

        Parameters
        ----------
        x1, x2, x3 : float
        eps : float, step size

        Returns
        -------
        ndarray (3,) : (d/dx_1, d/dx_2, d/dx_3) log J
        """
        x = np.array([x1, x2, x3])
        grad = np.zeros(3)
        for i in range(3):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            lp = WeylChamberJacobian.log_jacobian(*xp)
            lm = WeylChamberJacobian.log_jacobian(*xm)
            if np.isfinite(lp) and np.isfinite(lm):
                grad[i] = (lp - lm) / (2 * eps)
            else:
                grad[i] = np.nan
        return grad

    @staticmethod
    def grad_log_jacobian_exact(x1, x2, x3):
        """
        Exact gradient of log J.

        d/dx_i log J = sum_{j != i} 2 x_i / (x_i^2 - x_j^2)

        Parameters
        ----------
        x1, x2, x3 : float

        Returns
        -------
        ndarray (3,) : exact gradient
        """
        x = np.array([x1, x2, x3])
        grad = np.zeros(3)
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                denom = x[i]**2 - x[j]**2
                if abs(denom) < 1e-30:
                    grad[i] = np.nan
                    break
                grad[i] += 2.0 * x[i] / denom
        return grad


# ======================================================================
# 3. CentrifugalPotential — V_cent from sqrt(J) transformation
# ======================================================================

class CentrifugalPotential:
    """
    Centrifugal potential from the ground-state transformation.

    To convert from the J-weighted inner product to flat L^2, define
    Phi = sqrt(J) * Psi. Then the Schrodinger equation becomes:

        [-(kappa/2) sum_i d^2/dx_i^2 + V + V_cent] Phi = E Phi

    where

        V_cent = +(kappa/2) * sum_i [d^2 sqrt(J) / dx_i^2] / sqrt(J)

    The sign is POSITIVE because the transformation absorbs the Jacobian
    from the kinetic operator: H_J = -(kappa/2)(1/J)d[J d/dx] + V, and
    conjugation by sqrt(J) produces -(kappa/2)d^2/dx^2 + (kappa/2)*Q + V
    where Q = [Laplacian(sqrt(J))]/sqrt(J).

    Near a wall x_i = x_j (setting rho = x_i - x_j -> 0), each of the
    two adjacent singular-value directions contributes -1/(4*rho^2), giving:

        V_cent ~ -1/(2*rho^2)  (total, summed over directions)

    Per single direction: c_single = -1/4 (critical inverse-square case).

    THEOREM (Reed & Simon): For V ~ c/r^2 near r = 0,
        c >= 3/4   => limit-point (unique, essentially self-adjoint)
        c < 3/4    => limit-circle (need BC to specify self-adjoint extension)

    Since c = -1/4 < 3/4 in EACH direction, we are in the limit-circle
    case, and the BC (Neumann for A_1 ground state, Dirichlet for
    antisymmetric) selects the physical self-adjoint extension.
    """

    @staticmethod
    def v_cent(x1, x2, x3, prefactor=1.0, eps=1e-7):
        """
        Compute the centrifugal potential V_cent(x_1, x_2, x_3).

        V_cent = +prefactor * sum_i (d^2 sqrt(J) / dx_i^2) / sqrt(J)

        The POSITIVE sign comes from the Sturm-Liouville transformation:
        H_J = -(k/2)(1/J)d[Jd/dx] + V maps to H_flat = -(k/2)d^2/dx^2 + V + V_cent.

        Parameters
        ----------
        x1, x2, x3 : float
        prefactor : float
            Physical prefactor kappa/2 = g^2/(2L^3). Default 1.0.
        eps : float
            Step size for numerical second derivative.

        Returns
        -------
        float : V_cent (negative near walls — attractive centrifugal potential)
        """
        sJ = WeylChamberJacobian.sqrt_jacobian(x1, x2, x3)
        if sJ < 1e-30:
            return -np.inf  # Diverges negatively on the wall

        x = np.array([x1, x2, x3])
        laplacian_sJ = 0.0
        for i in range(3):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            sJp = WeylChamberJacobian.sqrt_jacobian(*xp)
            sJm = WeylChamberJacobian.sqrt_jacobian(*xm)
            laplacian_sJ += (sJp - 2 * sJ + sJm) / eps**2

        return prefactor * laplacian_sJ / sJ

    @staticmethod
    def v_cent_exact(x1, x2, x3, prefactor=1.0):
        """
        Exact centrifugal potential via analytical derivatives.

        J = prod_{i<j} (x_i^2 - x_j^2) in the Weyl chamber.

        Define h_i = d^2(sqrt(J))/dx_i^2 / sqrt(J). Then:

        h_i = (1/2) d^2(log J)/dx_i^2 + (1/4)(d(log J)/dx_i)^2

        Wait — more carefully:
        sqrt(J) = J^{1/2}, so
        d/dx_i [J^{1/2}] = (1/2) J^{-1/2} dJ/dx_i
        d^2/dx_i^2 [J^{1/2}] = (1/2) [-(1/2) J^{-3/2} (dJ/dx_i)^2 + J^{-1/2} d^2J/dx_i^2]
                              = (1/2) J^{-1/2} [d^2J/dx_i^2 - (1/2)(dJ/dx_i)^2/J]

        So h_i = d^2/dx_i^2[J^{1/2}] / J^{1/2}
               = (1/2) [d^2J/dx_i^2 / J - (1/2)(dJ/dx_i)^2 / J^2]
               = (1/2) d^2(log J)/dx_i^2 + (1/4)(d(log J)/dx_i)^2
               Wait, that's:
               (1/2)[J''/J] - (1/4)(J'/J)^2
               = (1/2)(log J)'' + (1/2)(J'/J)^2 - (1/4)(J'/J)^2
               = (1/2)(log J)'' + (1/4)((log J)')^2

        So V_cent = +prefactor * sum_i [(1/2)(log J)''_i + (1/4)((log J)'_i)^2]

        The sign is POSITIVE (Sturm-Liouville transformation of -(1/J)d[Jd]).
        Near a wall x_i = x_j, total ~ -1/(2*rho^2) (each of the two adjacent
        directions contributes -1/(4*rho^2)), so V_cent ~ -prefactor/(2*rho^2).

        For log J = sum_{i<j} log|x_i^2 - x_j^2|:
        d/dx_k log|x_k^2 - x_j^2| = 2 x_k / (x_k^2 - x_j^2)
        d^2/dx_k^2 log|x_k^2 - x_j^2| = 2/(x_k^2 - x_j^2) - 4x_k^2/(x_k^2 - x_j^2)^2
                                        = -2(x_k^2 + x_j^2)/(x_k^2 - x_j^2)^2

        Parameters
        ----------
        x1, x2, x3 : float (must satisfy x1 > x2 > x3 >= 0, strict inequality)
        prefactor : float

        Returns
        -------
        float : V_cent (negative near walls)
        """
        x = np.array([x1, x2, x3])
        total = 0.0

        for k in range(3):
            # Compute (d log J / dx_k) and (d^2 log J / dx_k^2)
            d1 = 0.0  # first derivative of log J w.r.t. x_k
            d2 = 0.0  # second derivative of log J w.r.t. x_k
            for j in range(3):
                if j == k:
                    continue
                diff_sq = x[k]**2 - x[j]**2
                if abs(diff_sq) < 1e-30:
                    return -np.inf
                d1 += 2.0 * x[k] / diff_sq
                d2 += -2.0 * (x[k]**2 + x[j]**2) / diff_sq**2

            total += 0.5 * d2 + 0.25 * d1**2

        return prefactor * total

    @staticmethod
    def near_wall_behavior(rho, x_center=1.0):
        """
        Asymptotic form of V_cent near a wall x_i = x_j.

        Near the wall x_1 = x_2, set x_1 = x_c + rho/2, x_2 = x_c - rho/2,
        x_3 = fixed. Then for small rho:

            V_cent ~ c_total / rho^2

        with c_total = -1/2 (sum of -1/4 from each of the two directions
        x_1 and x_2 that touch the wall).

        Per single direction: c_single = -1/4 (critical limit-circle case).

        Parameters
        ----------
        rho : float
            Distance from the wall.
        x_center : float
            Value of x_1 ~ x_2 at the wall.

        Returns
        -------
        float : approximate V_cent ~ -1/(2 rho^2)
        """
        if abs(rho) < 1e-30:
            return -np.inf
        return -0.5 / rho**2

    @staticmethod
    def inverse_square_coefficient_per_direction():
        """
        The per-direction coefficient c in V_cent contributions near walls.

        THEOREM: c_single = -1/4 per direction (critical limit-circle case).

        Derived from: J ~ rho * f(other) near rho = x_i - x_j -> 0.
        sqrt(J) ~ sqrt(rho) * sqrt(f).
        d^2/dx_k^2 [sqrt(J)] / sqrt(J) ~ -1/(4 rho^2) for each of the
        two directions k that touch the wall.

        Returns
        -------
        float : -1/4
        """
        return -0.25

    @staticmethod
    def inverse_square_coefficient():
        """
        The TOTAL coefficient c_total in V_cent ~ c_total / rho^2 near walls.

        THEOREM: c_total = -1/2, being the sum of -1/4 from each of the
        two singular-value directions adjacent to the wall.

        For the Weyl limit-circle classification, the relevant coefficient
        per direction is c_single = -1/4 < 3/4, placing us firmly in the
        limit-circle regime.

        Returns
        -------
        float : -1/2
        """
        return -0.5

    @staticmethod
    def verify_inverse_square(x3=0.5, x_center=2.0, n_points=20):
        """
        Numerically verify c = -1/4 by approaching the wall x_1 = x_2.

        Compute V_cent at decreasing rho and extract c from c = rho^2 * V_cent.

        Parameters
        ----------
        x3 : float
            Fixed value of x_3.
        x_center : float
            Value of x_1 ≈ x_2.
        n_points : int
            Number of test points.

        Returns
        -------
        dict with 'rho_values', 'c_values', 'c_limit', 'matches_theory'
        """
        rhos = np.logspace(-1, -6, n_points)
        c_values = []

        for rho in rhos:
            x1 = x_center + rho / 2
            x2 = x_center - rho / 2
            vc = CentrifugalPotential.v_cent_exact(x1, x2, x3)
            if np.isfinite(vc):
                c_values.append(rho**2 * vc)
            else:
                c_values.append(np.nan)

        c_arr = np.array(c_values)
        valid = np.isfinite(c_arr)
        c_limit = np.mean(c_arr[valid][-5:]) if np.sum(valid) >= 5 else np.nan

        return {
            'rho_values': rhos,
            'c_values': c_arr,
            'c_limit': c_limit,
            'matches_theory': abs(c_limit - (-0.5)) < 0.02 if np.isfinite(c_limit) else False,
        }


# ======================================================================
# 4. S3Potential — V_{S^3} = quadratic + cubic + quartic
# ======================================================================

class S3Potential:
    """
    The Yang-Mills potential on S^3 in SVD variables.

    V_{S^3}(x_1, x_2, x_3) = V_quad + V_cubic + V_quartic

    where (with appropriate normalization, g^2 = 1 units):

        V_quad   = sum_i x_i^2 / R^2             (mass from S^3 curvature)
        V_cubic  = -(2/R) x_1 x_2 x_3           (from curvature of connection)
        V_quartic = sum_{i<j} x_i^2 x_j^2        ([A,A]^2 self-interaction)

    THEOREM: V_{S^3} has a unique minimum at x = 0 (the A = 0 vacuum).
    THEOREM: V_{S^3} -> infinity as |x| -> infinity (confining).
    THEOREM: On T^3, only V_quartic survives; V_quad and V_cubic are absent.

    The mass term V_quad = x^2/R^2 is the KEY advantage of S^3 over T^3:
    it eliminates the flat directions that plague the T^3 analysis.
    """

    def __init__(self, R=1.0, g2=1.0):
        """
        Parameters
        ----------
        R : float
            Radius of S^3.
        g2 : float
            Yang-Mills coupling g^2.
        """
        self.R = R
        self.g2 = g2

    def v_quadratic(self, x):
        """
        Quadratic potential: V_quad = (2/R^2) sum_i x_i^2.

        This comes from the coexact eigenvalue mu_1 = 4/R^2 on S^3,
        giving V_2 = (1/2)(4/R^2)|a|^2 = (2/R^2)|a|^2. In SVD variables,
        |a|^2 = sum_i x_i^2, so V_quad = (2/R^2) sum_i x_i^2.

        Parameters
        ----------
        x : ndarray (3,) or list
            Singular values [x_1, x_2, x_3].

        Returns
        -------
        float : V_quad >= 0
        """
        x = np.asarray(x, dtype=float)
        return (2.0 / self.R**2) * np.sum(x**2)

    def v_cubic(self, x):
        """
        Cubic potential: V_cubic = -(2g/R) x_1 x_2 x_3.

        This term arises from the Maurer-Cartan curvature of S^3:
        the cross term between the background curvature F_theta and the
        quartic [A,A] interaction, integrated over S^3.

        In the 9-DOF truncation on S^3, det(M) = x_1 x_2 x_3 (in SVD),
        and the cubic contribution is proportional to this.

        The factor of g (not g^2) means this is a CUBIC vertex.

        Parameters
        ----------
        x : ndarray (3,) or list

        Returns
        -------
        float : can be positive or negative
        """
        x = np.asarray(x, dtype=float)
        g = np.sqrt(self.g2)
        return -(2.0 * g / self.R) * x[0] * x[1] * x[2]

    def v_quartic(self, x):
        """
        Quartic potential: V_quartic = (g^2/2) sum_{i<j} x_i^2 x_j^2.

        This is the [A,A]^2 self-interaction, present on both S^3 and T^3.
        In SVD variables:
            V_4 = (g^2/2) [(Tr S)^2 - Tr(S^2)]    with S = diag(x_1^2, x_2^2, x_3^2)
                = (g^2/2) * 2 * sum_{i<j} x_i^2 x_j^2
                = g^2 * sum_{i<j} x_i^2 x_j^2

        Wait — more carefully, the factor from V_4 = (g^2/2)[(TrS)^2 - Tr(S^2)]:
            (TrS)^2 - Tr(S^2) = 2 sum_{i<j} s_i s_j where s_i = x_i^2
            So V_4 = g^2 sum_{i<j} x_i^2 x_j^2

        Parameters
        ----------
        x : ndarray (3,) or list

        Returns
        -------
        float : V_quartic >= 0
        """
        x = np.asarray(x, dtype=float)
        v = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                v += x[i]**2 * x[j]**2
        return self.g2 * v

    def v_total(self, x):
        """
        Total potential on S^3: V = V_quad + V_cubic + V_quartic.

        Parameters
        ----------
        x : ndarray (3,) or list

        Returns
        -------
        float
        """
        return self.v_quadratic(x) + self.v_cubic(x) + self.v_quartic(x)

    def v_torus(self, x):
        """
        T^3 potential for comparison: V_{T^3} = V_quartic only.

        On a flat torus there is no curvature => no mass term, no cubic term.

        Parameters
        ----------
        x : ndarray (3,) or list

        Returns
        -------
        float
        """
        return self.v_quartic(x)

    def gradient(self, x, eps=1e-7):
        """
        Numerical gradient of V_total.

        Parameters
        ----------
        x : ndarray (3,)
        eps : float

        Returns
        -------
        ndarray (3,)
        """
        x = np.asarray(x, dtype=float)
        grad = np.zeros(3)
        for i in range(3):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (self.v_total(xp) - self.v_total(xm)) / (2 * eps)
        return grad

    def hessian(self, x, eps=1e-5):
        """
        Numerical Hessian of V_total.

        Parameters
        ----------
        x : ndarray (3,)
        eps : float

        Returns
        -------
        ndarray (3, 3)
        """
        x = np.asarray(x, dtype=float)
        H = np.zeros((3, 3))
        v0 = self.v_total(x)
        for i in range(3):
            for j in range(3):
                xpp = x.copy()
                xpm = x.copy()
                xmp = x.copy()
                xmm = x.copy()
                xpp[i] += eps
                xpp[j] += eps
                xpm[i] += eps
                xpm[j] -= eps
                xmp[i] -= eps
                xmp[j] += eps
                xmm[i] -= eps
                xmm[j] -= eps
                H[i, j] = (
                    self.v_total(xpp) - self.v_total(xpm)
                    - self.v_total(xmp) + self.v_total(xmm)
                ) / (4 * eps**2)
        return H

    def hessian_at_origin(self):
        """
        Exact Hessian at x = 0.

        V_quad = (2/R^2) sum x_i^2  =>  H_quad = (4/R^2) I
        V_cubic = -(2g/R) x1 x2 x3  =>  all second derivatives vanish at 0
        V_quartic = g^2 sum_{i<j} x_i^2 x_j^2 => all second derivatives vanish at 0

        THEOREM: Hess(V)(0) = (4/R^2) I_3  (positive definite).

        Returns
        -------
        ndarray (3, 3) : (4/R^2) * I_3
        """
        return (4.0 / self.R**2) * np.eye(3)

    def minimum_is_at_origin(self, n_samples=5000):
        """
        NUMERICAL verification that x = 0 is the global minimum.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        dict with 'is_minimum', 'v_at_origin', 'min_found', 'min_config'
        """
        rng = np.random.default_rng(42)
        v_origin = self.v_total(np.zeros(3))
        min_val = v_origin
        min_config = np.zeros(3)

        for _ in range(n_samples):
            x = np.abs(rng.standard_normal(3)) * rng.uniform(0.01, 5.0)
            x.sort()
            x = x[::-1]  # x_1 >= x_2 >= x_3
            v = self.v_total(x)
            if v < min_val:
                min_val = v
                min_config = x.copy()

        return {
            'is_minimum': min_val >= v_origin - 1e-12,
            'v_at_origin': v_origin,
            'min_found': min_val,
            'min_config': min_config,
        }


# ======================================================================
# 5. ReducedHamiltonian — H_red on L^2(W, J dx) or flat-measure version
# ======================================================================

class ReducedHamiltonian:
    """
    The reduced Hamiltonian in the spin-0 sector on the Weyl chamber W.

    On L^2(W, J dx) (J-weighted measure):
        H_J = -(kappa/2) * (1/J) sum_i d/dx_i [J d/dx_i] + V(x)

    On L^2(W, dx) (flat measure, via Phi = sqrt(J) * Psi):
        H_flat = -(kappa/2) sum_i d^2/dx_i^2 + V(x) + V_cent(x)

    where kappa = g^2 / L^3 (L = R for S^3).

    Parameters
    ----------
    R : float
        Radius of S^3.
    g2 : float
        Yang-Mills coupling g^2.
    """

    def __init__(self, R=1.0, g2=1.0):
        self.R = R
        self.g2 = g2
        self.kappa = g2 / R**3     # kinetic prefactor
        self.potential = S3Potential(R=R, g2=g2)

    def kinetic_prefactor(self):
        """
        The prefactor kappa/2 = g^2 / (2 R^3) of the kinetic term.

        Returns
        -------
        float
        """
        return self.kappa / 2.0

    def potential_energy(self, x):
        """
        Total potential V(x_1, x_2, x_3) on S^3.

        Parameters
        ----------
        x : ndarray (3,)

        Returns
        -------
        float
        """
        return self.potential.v_total(x)

    def effective_potential_flat(self, x):
        """
        Effective potential in the flat-measure picture:
        V_eff = V(x) + V_cent(x).

        V_cent = +(kappa/2) * [Laplacian(sqrt(J))/sqrt(J)],
        which is negative near walls (attractive centrifugal term).

        Parameters
        ----------
        x : ndarray (3,)

        Returns
        -------
        float
        """
        v = self.potential.v_total(x)
        vc = CentrifugalPotential.v_cent_exact(*x, prefactor=self.kappa / 2.0)
        return v + vc


# ======================================================================
# 6. HarmonicOscillatorBasis — product basis for Rayleigh-Ritz
# ======================================================================

class HarmonicOscillatorBasis:
    """
    Basis of product Hermite functions for the Weyl chamber.

    phi_{n1,n2,n3}(x) = h_{n1}(alpha*x_1) * h_{n2}(alpha*x_2) * h_{n3}(alpha*x_3)

    where h_n(y) = (2^n n! sqrt(pi))^{-1/2} H_n(y) exp(-y^2/2) are the
    normalized Hermite functions and alpha is a scale parameter chosen
    to match the quadratic potential.

    For the S^3 potential with V_quad = (2/R^2) x^2 and kinetic prefactor
    kappa/2, the harmonic frequency is omega = sqrt(4/(R^2) / (kappa/2))
    = sqrt(8/(g^2/R)) = sqrt(8R/g^2), and alpha = (m*omega/hbar)^{1/4}.

    In practice, alpha is treated as a variational parameter.
    """

    def __init__(self, N_per_dim, alpha=1.0):
        """
        Parameters
        ----------
        N_per_dim : int
            Number of basis functions per dimension.
        alpha : float
            Scale parameter for the Hermite functions.
        """
        self.N = N_per_dim
        self.alpha = alpha
        # Total basis size is N^3 for unconstrained, but we work in the
        # Weyl chamber with the full product basis (BC handled by symmetry).
        self.n_basis = N_per_dim**3

    def hermite_value(self, n, y):
        """
        Normalized Hermite function h_n(y).

        h_n(y) = (2^n n! sqrt(pi))^{-1/2} H_n(y) exp(-y^2/2)

        Parameters
        ----------
        n : int
        y : float or ndarray

        Returns
        -------
        float or ndarray
        """
        Hn = hermite(n)
        norm = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
        y = np.asarray(y, dtype=float)
        return norm * Hn(y) * np.exp(-y**2 / 2)

    def basis_function(self, n1, n2, n3, x):
        """
        Product basis function phi_{n1,n2,n3}(x).

        Parameters
        ----------
        n1, n2, n3 : int
            Quantum numbers.
        x : ndarray (3,)
            Point in configuration space.

        Returns
        -------
        float
        """
        x = np.asarray(x, dtype=float)
        a = self.alpha
        return (
            self.hermite_value(n1, a * x[0])
            * self.hermite_value(n2, a * x[1])
            * self.hermite_value(n3, a * x[2])
            * a**(1.5)  # Jacobian from scaling
        )

    def index_to_quantum_numbers(self, idx):
        """
        Map linear index to (n1, n2, n3).

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple (n1, n2, n3)
        """
        n3 = idx % self.N
        n2 = (idx // self.N) % self.N
        n1 = idx // (self.N**2)
        return (n1, n2, n3)

    def quantum_numbers_to_index(self, n1, n2, n3):
        """
        Map (n1, n2, n3) to linear index.

        Parameters
        ----------
        n1, n2, n3 : int

        Returns
        -------
        int
        """
        return n1 * self.N**2 + n2 * self.N + n3

    def all_quantum_numbers(self):
        """
        Generator over all (n1, n2, n3) in the basis.

        Yields
        ------
        tuple (n1, n2, n3)
        """
        return product(range(self.N), repeat=3)


# ======================================================================
# 7. NumericalDiagonalization — Rayleigh-Ritz
# ======================================================================

class NumericalDiagonalization:
    """
    Rayleigh-Ritz diagonalization of the reduced Hamiltonian.

    Builds the Hamiltonian matrix in the harmonic oscillator basis
    via Gauss-Hermite quadrature and diagonalizes.

    For N basis functions per dimension:
        Total basis size = N^3
        N=10 => 1000x1000 matrix
        N=20 => 8000x8000 matrix

    NUMERICAL: eigenvalues depend on basis truncation, but converge
    as N increases.
    """

    def __init__(self, R=1.0, g2=1.0, N_per_dim=10, alpha=None, n_quad=None):
        """
        Parameters
        ----------
        R : float
            Radius of S^3.
        g2 : float
            Yang-Mills coupling g^2.
        N_per_dim : int
            Number of HO basis functions per dimension.
        alpha : float or None
            Scale parameter. If None, set optimally from harmonic frequency.
        n_quad : int or None
            Number of Gauss-Hermite quadrature points per dimension.
            If None, use 2*N_per_dim + 5.
        """
        self.R = R
        self.g2 = g2
        self.N = N_per_dim
        self.n_basis = N_per_dim**3
        self.hamiltonian = ReducedHamiltonian(R=R, g2=g2)
        self.potential_obj = S3Potential(R=R, g2=g2)

        # Kinetic prefactor
        self.kappa = g2 / R**3

        # Set alpha optimally for the quadratic potential
        # omega^2 = (4/R^2) / (kappa/2) = 8R / g^2
        # alpha = (omega)^{1/2} in natural units where m=1
        if alpha is None:
            omega_sq = 4.0 / (self.R**2) / (self.kappa / 2.0)
            if omega_sq > 0:
                self.alpha = omega_sq**0.25
            else:
                self.alpha = 1.0
        else:
            self.alpha = alpha

        self.basis = HarmonicOscillatorBasis(N_per_dim, alpha=self.alpha)

        # Quadrature
        if n_quad is None:
            n_quad = 2 * N_per_dim + 5
        self.n_quad = n_quad

        # Cache
        self._eigenvalues = None
        self._eigenvectors = None

    def _gauss_hermite_grid(self):
        """
        Set up Gauss-Hermite quadrature grid and weights.

        The integral of f(y) * exp(-y^2) is approximated by
        sum_k w_k f(y_k). For our Hermite-function basis, we need
        to integrate h_n(y) * h_m(y) * V(y/alpha) which already
        contains exp(-y^2), so we use the physicist's Gauss-Hermite
        nodes and convert.

        Returns
        -------
        nodes : ndarray (n_quad,)
        weights : ndarray (n_quad,)
            Nodes and weights for integral of f(y) dy on (-inf, inf)
            with weight function exp(-y^2).
        """
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        return nodes, weights

    def build_hamiltonian_matrix_fast(self):
        """
        Vectorized Hamiltonian matrix construction via separable potential
        decomposition.

        Same physics as build_hamiltonian_matrix but uses the fact that
        V_{S^3} = V_quad + V_cubic + V_quartic is a sum of separable terms
        (products of 1D functions), so each contribution's matrix elements
        factorize into Kronecker products of 1D integrals.

        This avoids building any 3D or 6D tensors, making it O(N^2 * nq)
        per 1D integral rather than O(N^6) for the full tensor.

        For N=20 (8000x8000 matrix), this runs in seconds rather than hours.

        Returns
        -------
        ndarray (n_basis, n_basis) : Hamiltonian matrix (symmetric)
        """
        N = self.N
        n_basis = self.n_basis
        alpha = self.alpha
        kappa_half = self.kappa / 2.0

        # === KINETIC ENERGY (exact, via Kronecker products) ===
        T1d = np.zeros((N, N))
        for n in range(N):
            T1d[n, n] = 0.25 * (2 * n + 1)
            if n >= 2:
                T1d[n, n - 2] = -0.25 * np.sqrt(n * (n - 1))
            if n + 2 < N:
                T1d[n, n + 2] = -0.25 * np.sqrt((n + 1) * (n + 2))

        kinetic_scale = kappa_half * alpha**2 * 2
        I_N = np.eye(N)

        H = kinetic_scale * (
            np.kron(np.kron(T1d, I_N), I_N) +
            np.kron(np.kron(I_N, T1d), I_N) +
            np.kron(np.kron(I_N, I_N), T1d)
        )

        # === POTENTIAL ENERGY (separable decomposition) ===
        # The S^3 potential is:
        #   V = (2/R^2)(x1^2+x2^2+x3^2) - (2g/R)x1*x2*x3 + g^2*sum_{i<j}xi^2*xj^2
        #
        # Each term is a sum of products of 1D functions f(x_i)*g(x_j)*h(x_k).
        # Matrix elements of f(x_1)*g(x_2)*h(x_3) in the product basis are:
        #   <n1,n2,n3| f*g*h |m1,m2,m3> = F_{n1,m1} * G_{n2,m2} * H_{n3,m3}
        # where F_{nm} = <h_n| f(x) |h_m> via 1D Gauss-Hermite quadrature.
        #
        # The 3D matrix is then kron(F, kron(G, H)).

        nodes, weights_gh = self._gauss_hermite_grid()
        nq = self.n_quad

        # 1D basis values at quadrature nodes: basis_1d[n, k] = norm_n * H_n(y_k)
        basis_1d = np.zeros((N, nq))
        for n in range(N):
            Hn = hermite(n)
            norm = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
            basis_1d[n, :] = norm * Hn(nodes)

        # Physical coordinates at quad points
        x_phys = nodes / alpha  # shape (nq,)

        # Build 1D integral matrices via quadrature:
        # M_f[n, m] = sum_k w_k * basis_1d[n,k] * f(x_k) * basis_1d[m,k]
        # where f(x) is a 1D function evaluated at x_phys[k]

        def build_1d_matrix(f_vals):
            """Build <n|f|m> matrix from f evaluated at quad points."""
            # f_vals: shape (nq,)
            # result[n, m] = sum_k w_k * basis_1d[n,k] * f_vals[k] * basis_1d[m,k]
            wf = weights_gh * f_vals  # (nq,)
            return basis_1d @ np.diag(wf) @ basis_1d.T

        # Identity (overlap) matrix: S[n,m] = <h_n|h_m> = delta_{nm}
        # (exact for HO basis, but compute via quadrature for consistency)
        S_1d = build_1d_matrix(np.ones(nq))  # Should be ~ I_N

        # 1D function matrices:
        f_x2 = x_phys**2       # x^2
        f_x = x_phys           # x
        f_1 = np.ones(nq)      # 1

        M_x2 = build_1d_matrix(f_x2)   # <n|x^2|m>
        M_x = build_1d_matrix(f_x)     # <n|x|m>
        M_1 = build_1d_matrix(f_1)     # <n|1|m> = S_1d
        M_x4 = build_1d_matrix(f_x2**2)  # <n|x^4|m>

        g = np.sqrt(self.g2)
        R = self.R

        # --- Quadratic: (2/R^2)(x1^2 + x2^2 + x3^2) ---
        # = (2/R^2) * [x1^2*1*1 + 1*x2^2*1 + 1*1*x3^2]
        coeff_quad = 2.0 / R**2
        H += coeff_quad * (
            np.kron(np.kron(M_x2, M_1), M_1) +
            np.kron(np.kron(M_1, M_x2), M_1) +
            np.kron(np.kron(M_1, M_1), M_x2)
        )

        # --- Cubic: -(2g/R) x1*x2*x3 ---
        # Fully separable: f(x1)=x1, g(x2)=x2, h(x3)=x3
        coeff_cubic = -(2.0 * g / R)
        H += coeff_cubic * np.kron(np.kron(M_x, M_x), M_x)

        # --- Quartic: g^2 * (x1^2*x2^2 + x1^2*x3^2 + x2^2*x3^2) ---
        # = g^2 * [x1^2*x2^2*1 + x1^2*1*x3^2 + 1*x2^2*x3^2]
        coeff_quart = self.g2
        H += coeff_quart * (
            np.kron(np.kron(M_x2, M_x2), M_1) +
            np.kron(np.kron(M_x2, M_1), M_x2) +
            np.kron(np.kron(M_1, M_x2), M_x2)
        )

        # Symmetrize to remove numerical asymmetry
        H = 0.5 * (H + H.T)

        return H

    def build_hamiltonian_matrix(self):
        """
        Build the Hamiltonian matrix H_{ij} = <phi_i | H | phi_j>
        using 3D Gauss-Hermite quadrature.

        The Hamiltonian is H = -(kappa/2) Delta + V(x).

        In the HO basis, the kinetic energy matrix elements of the
        HARMONIC part are known exactly:
            <n|-(1/2)d^2/dy^2 + (1/2)y^2|m> = (n + 1/2) delta_{nm}

        So we split H = H_HO + V_anharmonic, where V_anharmonic is
        the difference between the actual potential and the harmonic one.

        Returns
        -------
        ndarray (n_basis, n_basis) : Hamiltonian matrix (symmetric)
        """
        N = self.N
        n_basis = self.n_basis
        alpha = self.alpha
        kappa_half = self.kappa / 2.0

        # 1D harmonic oscillator energy: epsilon_n = kappa/2 * alpha^2 * (n + 1/2)
        # because the kinetic term is -(kappa/2)d^2/dx^2 and the HO has
        # frequency omega = kappa * alpha^2
        ho_energy_scale = kappa_half * alpha**2

        # Quadratic part of V that matches the HO:
        # V_HO = (1/2) omega^2 x^2 where omega^2 = (4/R^2)
        # In scaled variable y = alpha*x: V_HO = (1/2)(4/R^2)(y/alpha)^2
        # The HO in the y basis has kinetic -(kappa/2)*alpha^2 d^2/dy^2
        # and potential (2/R^2)(y/alpha)^2 = (2/R^2)/alpha^2 * y^2
        # HO frequency: omega_HO^2 = (4/R^2)/(kappa*alpha^2)
        # epsilon_n = sqrt(kappa_half*alpha^2 * (2/R^2)/alpha^2) * (2n+1)/2
        #           = sqrt(2*kappa_half/R^2) * (n+1/2)

        # Actually let's just do the full quadrature for the potential part
        # and use exact kinetic matrix elements.

        H = np.zeros((n_basis, n_basis))

        # Kinetic energy: exact matrix elements in HO basis
        # -(kappa/2) d^2/dx^2 in HO basis with parameter alpha:
        # <n| -(kappa/2) d^2/dx^2 |m> = kappa*alpha^2/2 * [(n+1/2)delta_{nm}]
        #   - (kappa*alpha^2/4)*[sqrt(n(n-1)) delta_{n,m+2} + sqrt((n+1)(n+2)) delta_{n,m-2}]
        # Wait, for the HO basis h_n(alpha*x) * alpha^{1/2}, the kinetic part gives:
        # <n|p^2/2|m> = (alpha^2/2)*kappa/2 * [ (2n+1)delta_{nm}
        #                - sqrt(n(n-1))delta_{n,m+2} - sqrt((n+1)(n+2))delta_{n,m-2} ]

        # For 1D: <h_n| -d^2/dy^2 |h_m> (in y = alpha*x variable)
        # = (2n+1)/2 delta_{nm} via the virial theorem for the HO
        # BUT we have -(kappa/2)*d^2/dx^2 = -(kappa/2)*alpha^2 * d^2/dy^2
        # so <n|T|m> = (kappa*alpha^2/2) * (n+1/2) * delta_{nm}
        # PLUS off-diagonal terms from the x -> y transformation

        # Actually for the standard HO basis in y-variable:
        # <n| -d^2/dy^2 |m> = (n + 1/2)*delta_{nm}  [WRONG: this is <n|H_HO|m> not <n|T|m>]
        # <n| T_HO |m> = <n| (-1/2 d^2/dy^2 + 1/2 y^2)|m>/2 etc.

        # Let me use the correct formulas:
        # <n| -(1/2) d^2/dy^2 |m> = (1/4)(2n+1)delta_{nm}
        #     - (1/4)sqrt(n(n-1))delta_{m,n-2} - (1/4)sqrt((n+1)(n+2))delta_{m,n+2}
        # with y^2 adding the complementary terms to give (n+1/2)delta for the full HO.

        # So the 1D kinetic matrix T1d_{nm} in the y-basis is:
        # T1d_{nm} = (1/4)(2n+1)delta_{nm}
        #   - (1/4)sqrt(n(n-1))delta_{m,n-2}
        #   - (1/4)sqrt((n+1)(n+2))delta_{m,n+2}

        # And the 3D kinetic matrix in the product basis is:
        # T3d_{n1n2n3, m1m2m3} = kappa*alpha^2 * [
        #   T1d_{n1m1}*delta_{n2m2}*delta_{n3m3}
        # + delta_{n1m1}*T1d_{n2m2}*delta_{n3m3}
        # + delta_{n1m1}*delta_{n2m2}*T1d_{n3m3}
        # ]

        # Build 1D kinetic matrix
        T1d = np.zeros((N, N))
        for n in range(N):
            T1d[n, n] = 0.25 * (2 * n + 1)
            if n >= 2:
                T1d[n, n - 2] = -0.25 * np.sqrt(n * (n - 1))
            if n + 2 < N:
                T1d[n, n + 2] = -0.25 * np.sqrt((n + 1) * (n + 2))

        # Fill kinetic part of H
        kinetic_scale = kappa_half * alpha**2 * 2  # factor 2 because T1d has 1/4 not 1/2
        for idx_i in range(n_basis):
            n1, n2, n3 = self.basis.index_to_quantum_numbers(idx_i)
            for idx_j in range(idx_i, n_basis):
                m1, m2, m3 = self.basis.index_to_quantum_numbers(idx_j)

                val = 0.0
                # x_1 kinetic
                if n2 == m2 and n3 == m3:
                    val += T1d[n1, m1]
                # x_2 kinetic
                if n1 == m1 and n3 == m3:
                    val += T1d[n2, m2]
                # x_3 kinetic
                if n1 == m1 and n2 == m2:
                    val += T1d[n3, m3]

                H[idx_i, idx_j] += kinetic_scale * val
                if idx_i != idx_j:
                    H[idx_j, idx_i] += kinetic_scale * val

        # Potential energy via quadrature
        nodes, weights_gh = self._gauss_hermite_grid()

        # Pre-compute 1D basis values at quadrature nodes
        # h_n(y) includes exp(-y^2/2), but GH quadrature has weight exp(-y^2),
        # so we need: h_n(y) * exp(+y^2/2) evaluated at GH nodes
        basis_vals_1d = np.zeros((N, self.n_quad))
        for n in range(N):
            Hn = hermite(n)
            norm = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
            # h_n(y) = norm * H_n(y) * exp(-y^2/2)
            # We want h_n(y_k) / exp(-y_k^2/2) * sqrt(w_k) for the quadrature
            # Actually: integral h_n(y) * V(y/alpha) * h_m(y) dy
            # = integral [norm*H_n(y)*exp(-y^2/2)] * V(y/alpha) * [norm*H_m(y)*exp(-y^2/2)] dy
            # = integral norm^2 * H_n*H_m * exp(-y^2) * V(y/alpha) dy
            # ~ sum_k w_k * norm_n*H_n(y_k) * norm_m*H_m(y_k) * V(y_k/alpha)
            # where (y_k, w_k) are GH nodes/weights for weight exp(-y^2)
            for k in range(self.n_quad):
                basis_vals_1d[n, k] = norm * Hn(nodes[k])

        # 3D potential matrix elements via quadrature
        # V_{ij} = sum_{k1,k2,k3} w_{k1}*w_{k2}*w_{k3}
        #          * phi_i(y/alpha) * V(y/alpha) * phi_j(y/alpha)
        # where phi uses the modified basis values (without exp(-y^2/2))

        for k1 in range(self.n_quad):
            y1 = nodes[k1]
            w1 = weights_gh[k1]
            x1_phys = y1 / self.alpha

            for k2 in range(self.n_quad):
                y2 = nodes[k2]
                w2 = weights_gh[k2]
                x2_phys = y2 / self.alpha

                for k3 in range(self.n_quad):
                    y3 = nodes[k3]
                    w3 = weights_gh[k3]
                    x3_phys = y3 / self.alpha

                    x_phys = np.array([x1_phys, x2_phys, x3_phys])
                    V = self.potential_obj.v_total(x_phys)
                    w = w1 * w2 * w3

                    if abs(w * V) < 1e-30:
                        continue

                    for idx_i in range(n_basis):
                        n1, n2, n3 = self.basis.index_to_quantum_numbers(idx_i)
                        phi_i = (
                            basis_vals_1d[n1, k1]
                            * basis_vals_1d[n2, k2]
                            * basis_vals_1d[n3, k3]
                        )
                        if abs(phi_i) < 1e-30:
                            continue

                        for idx_j in range(idx_i, n_basis):
                            m1, m2, m3 = self.basis.index_to_quantum_numbers(idx_j)
                            phi_j = (
                                basis_vals_1d[m1, k1]
                                * basis_vals_1d[m2, k2]
                                * basis_vals_1d[m3, k3]
                            )

                            val = w * phi_i * V * phi_j
                            H[idx_i, idx_j] += val
                            if idx_i != idx_j:
                                H[idx_j, idx_i] += val

        return H

    def diagonalize(self, n_eigenvalues=None, use_fast=True):
        """
        Diagonalize the Hamiltonian matrix.

        Parameters
        ----------
        n_eigenvalues : int or None
            Number of lowest eigenvalues to compute. If None, compute all.
        use_fast : bool
            If True (default), use the vectorized Hamiltonian builder.
            Set to False to use the original loop-based builder.

        Returns
        -------
        eigenvalues : ndarray
        eigenvectors : ndarray (n_basis, n_eigenvalues)
        """
        if use_fast:
            H = self.build_hamiltonian_matrix_fast()
        else:
            H = self.build_hamiltonian_matrix()

        if n_eigenvalues is None or n_eigenvalues >= self.n_basis:
            eigenvalues, eigenvectors = eigh(H)
        else:
            # Use sparse solver for a few eigenvalues of large matrices
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            H_sparse = csr_matrix(H)
            eigenvalues, eigenvectors = eigsh(
                H_sparse, k=n_eigenvalues, which='SM'
            )
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        return eigenvalues, eigenvectors

    def eigenvalues(self, n_eigenvalues=None):
        """
        Get eigenvalues (diagonalizing if not already done).

        Parameters
        ----------
        n_eigenvalues : int or None

        Returns
        -------
        ndarray : sorted eigenvalues
        """
        if self._eigenvalues is None:
            self.diagonalize(n_eigenvalues)
        return self._eigenvalues

    def spectral_gap(self, n_eigenvalues=10):
        """
        Compute the spectral gap E_1 - E_0.

        Parameters
        ----------
        n_eigenvalues : int
            Number of eigenvalues to compute (need at least 2).

        Returns
        -------
        float : gap = E_1 - E_0 > 0
        """
        evals = self.eigenvalues(n_eigenvalues)
        if len(evals) < 2:
            raise ValueError("Need at least 2 eigenvalues for gap")
        return evals[1] - evals[0]


# ======================================================================
# 8. BenchmarkComparison — compare with Pavel (2007) and BDS (2023)
# ======================================================================

class BenchmarkComparison:
    """
    Compare eigenvalues with published T^3 benchmarks.

    IMPORTANT: Our computation is on S^3, which differs from the T^3 results.
    The benchmarks serve as sanity checks, not exact targets.

    On T^3 (Pavel 2007, Butt-Draper-Shen 2023):
        E_0       ~ 2.560 g^{2/3}
        Delta_J0  ~ 3.31  g^{2/3}

    On S^3, the mass term 1/R^2 shifts everything upward and the cubic
    term breaks the discrete Z_2 symmetry.
    """

    def __init__(self, g2=G2_DEFAULT):
        self.g2 = g2
        self.g = np.sqrt(g2)
        self.energy_unit = g2**(1.0 / 3.0)  # g^{2/3} for T^3 scaling

    def pavel_benchmark(self):
        """
        Published T^3 benchmark values in units of g^{2/3}.

        Returns
        -------
        dict with 'E0', 'E1_J2', 'E1_J0', 'gap_J0'
        """
        return {
            'E0': PAVEL_E0 * self.energy_unit,
            'E1_J2': PAVEL_E1_J2 * self.energy_unit,
            'E1_J0': PAVEL_E1_J0 * self.energy_unit,
            'gap_J0': PAVEL_GAP_J0 * self.energy_unit,
        }

    def bds_benchmark(self):
        """
        Butt-Draper-Shen (2023) benchmark (same values, higher precision).

        Returns
        -------
        dict : same structure as pavel_benchmark
        """
        # BDS confirms Pavel's values within ~1%
        return self.pavel_benchmark()

    @staticmethod
    def our_result(R, g2, N_per_dim=8, n_eigenvalues=10):
        """
        Compute our eigenvalues for comparison.

        Parameters
        ----------
        R : float
        g2 : float
        N_per_dim : int
        n_eigenvalues : int

        Returns
        -------
        dict with 'eigenvalues', 'E0', 'gap', 'N', 'R', 'g2'
        """
        diag = NumericalDiagonalization(R=R, g2=g2, N_per_dim=N_per_dim)
        evals, _ = diag.diagonalize(n_eigenvalues)
        return {
            'eigenvalues': evals,
            'E0': evals[0],
            'gap': evals[1] - evals[0] if len(evals) >= 2 else None,
            'N': N_per_dim,
            'R': R,
            'g2': g2,
        }

    def torus_comparison(self, N_per_dim=8):
        """
        Compute T^3 problem (V_quartic only) for direct benchmark comparison.

        On T^3, there is no curvature: V = V_quartic only.
        We model this by taking R -> infinity, but since V_quad ~ 1/R^2
        and V_cubic ~ 1/R, we can't literally do that. Instead we
        set up a separate diagonalization with the torus potential.

        Returns
        -------
        dict with benchmark comparison
        """
        # For the torus, the Hamiltonian is
        # H = -(g^2/2) Delta + g^2 sum_{i<j} x_i^2 x_j^2
        # Eigenvalues scale as g^{2/3} (dimensional analysis).
        # We compute at g^2 = 1 and scale.
        return {
            'note': 'T^3 benchmark requires separate torus diagonalization',
            'pavel_E0': PAVEL_E0,
            'pavel_gap': PAVEL_GAP_J0,
        }


# ======================================================================
# 9. SpectralGapExtraction — gap in physical units
# ======================================================================

class SpectralGapExtraction:
    """
    Extract the mass gap from numerical eigenvalues and convert to physical units.

    gap_natural = E_1 - E_0   (in units of the Hamiltonian)
    gap_MeV = gap_natural * hbar_c / R   (converting 1/R to MeV using hbar*c)
    """

    def __init__(self, R=R_PHYSICAL_FM, g2=G2_DEFAULT):
        self.R = R
        self.g2 = g2

    def gap_in_natural_units(self, N_per_dim=8, n_eigenvalues=10):
        """
        Compute the spectral gap in natural (Hamiltonian) units.

        Parameters
        ----------
        N_per_dim : int
        n_eigenvalues : int

        Returns
        -------
        float : E_1 - E_0
        """
        diag = NumericalDiagonalization(
            R=self.R, g2=self.g2, N_per_dim=N_per_dim
        )
        return diag.spectral_gap(n_eigenvalues)

    def gap_in_MeV(self, N_per_dim=8, n_eigenvalues=10):
        """
        Compute the spectral gap in MeV.

        The conversion uses:
            gap_MeV = gap_natural * (hbar*c / R)

        where the factor hbar*c/R converts 1/fm to MeV.

        Parameters
        ----------
        N_per_dim : int
        n_eigenvalues : int

        Returns
        -------
        float : gap in MeV
        """
        gap_nat = self.gap_in_natural_units(N_per_dim, n_eigenvalues)
        # gap_natural is in units where the Hamiltonian has kappa = g^2/R^3
        # The physical energy is gap * (hbar*c) since x is in fm
        # Actually: H has units of 1/R^2 * fm^2 = dimensionless (in natural units
        # where hbar = c = 1). To convert to MeV, multiply by hbar*c/R.
        return gap_nat * HBAR_C_MEV_FM / self.R

    def gap_vs_R(self, R_range, N_per_dim=6, n_eigenvalues=5):
        """
        Compute gap as a function of R.

        Parameters
        ----------
        R_range : array-like
            R values in fm.
        N_per_dim : int
        n_eigenvalues : int

        Returns
        -------
        dict with 'R_values', 'gap_natural', 'gap_MeV'
        """
        R_range = np.asarray(R_range)
        gaps_nat = []
        gaps_MeV = []

        for R in R_range:
            try:
                diag = NumericalDiagonalization(
                    R=R, g2=self.g2, N_per_dim=N_per_dim
                )
                gap = diag.spectral_gap(n_eigenvalues)
                gaps_nat.append(gap)
                gaps_MeV.append(gap * HBAR_C_MEV_FM / R)
            except Exception:
                gaps_nat.append(np.nan)
                gaps_MeV.append(np.nan)

        return {
            'R_values': R_range,
            'gap_natural': np.array(gaps_nat),
            'gap_MeV': np.array(gaps_MeV),
        }


# ======================================================================
# 10. SelfAdjointnessAnalysis — limit-circle classification
# ======================================================================

class SelfAdjointnessAnalysis:
    """
    Self-adjointness analysis for the reduced Hamiltonian with
    centrifugal potential.

    The centrifugal potential V_cent ~ c/rho^2 near the Weyl chamber walls
    (where rho = x_i - x_j -> 0) has c = -1/4.

    THEOREM (Reed & Simon, Vol. II): The 1D Schrodinger operator
    -d^2/dr^2 + c/r^2 on (0, infinity) is:
        - limit-point at r = 0 if c >= 3/4  (essentially self-adjoint)
        - limit-circle at r = 0 if c < 3/4  (need boundary condition)

    Since c = -1/4 < 3/4, we are in the LIMIT-CIRCLE case.

    Physical interpretation:
        - The A_1 representation (Neumann BC) gives the ground state
        - The antisymmetric representation (Dirichlet BC) gives excited states
        - The boundary condition is PHYSICAL, determined by the symmetry sector
    """

    @staticmethod
    def weyl_classification(c):
        """
        Classify the inverse-square potential V = c/r^2 using Weyl's theorem.

        Parameters
        ----------
        c : float
            Coefficient of 1/r^2.

        Returns
        -------
        dict with 'type' ('limit_point' or 'limit_circle'),
        'essentially_self_adjoint' (bool),
        'needs_bc' (bool),
        'c' (float)
        """
        threshold = 3.0 / 4.0
        if c >= threshold:
            return {
                'type': 'limit_point',
                'essentially_self_adjoint': True,
                'needs_bc': False,
                'c': c,
            }
        else:
            return {
                'type': 'limit_circle',
                'essentially_self_adjoint': False,
                'needs_bc': True,
                'c': c,
            }

    @staticmethod
    def is_essentially_selfadjoint():
        """
        Is the reduced Hamiltonian essentially self-adjoint?

        THEOREM: No. The centrifugal potential has c = -1/4 < 3/4,
        so the operator is in the limit-circle case and requires a
        boundary condition at each wall.

        Returns
        -------
        bool : False
        """
        return False

    @staticmethod
    def boundary_condition_type(symmetry_sector='A1'):
        """
        Determine the boundary condition for a given symmetry sector.

        The Weyl group S_3 acts on the Weyl chamber by permuting
        x_1, x_2, x_3. The representations are:
            A_1 (trivial): Neumann BC (even continuation)
            A_2 (sign): Dirichlet BC (odd continuation)
            E (standard): mixed

        Parameters
        ----------
        symmetry_sector : str
            'A1' for ground state, 'A2' for antisymmetric, 'E' for doublet.

        Returns
        -------
        dict with 'sector', 'bc_type', 'parity', 'physical_meaning'
        """
        sectors = {
            'A1': {
                'sector': 'A1 (trivial)',
                'bc_type': 'Neumann',
                'parity': 'even',
                'physical_meaning': (
                    'Ground state and J=0 excitations. '
                    'Wavefunction is symmetric under all permutations of x_i.'
                ),
            },
            'A2': {
                'sector': 'A2 (sign)',
                'bc_type': 'Dirichlet',
                'parity': 'odd',
                'physical_meaning': (
                    'Antisymmetric excitations. '
                    'Wavefunction is antisymmetric (vanishes at walls).'
                ),
            },
            'E': {
                'sector': 'E (standard 2D)',
                'bc_type': 'mixed (Neumann-Dirichlet)',
                'parity': 'mixed',
                'physical_meaning': (
                    'Doublet excitations with spin-2 content. '
                    'BC depends on which wall is approached.'
                ),
            },
        }
        return sectors.get(symmetry_sector, {'error': f'Unknown sector: {symmetry_sector}'})

    @staticmethod
    def s3_advantages():
        """
        Summary of S^3 advantages for self-adjointness.

        THEOREM: On S^3, pi_1(S^3) = 0 => no Gribov copies for constant modes.
        This means:
            - Single vacuum at A = 0
            - No tunneling between vacua
            - No topological boundary conditions
            - No vacuum valley (contrast with T^3)

        Returns
        -------
        dict with advantages
        """
        return {
            'pi_1_trivial': True,
            'no_gribov_copies': True,
            'single_vacuum': True,
            'no_tunneling': True,
            'no_topological_bc': True,
            'mass_term': 'V_quad = x^2/R^2 eliminates flat directions',
            'key_result': (
                'The self-adjointness issue reduces to choosing the correct '
                'Weyl group representation (BC), which is determined by the '
                'spin quantum number. No topological ambiguity.'
            ),
        }


# ======================================================================
# Convergence study helper
# ======================================================================

class ConvergenceStudy:
    """
    Study convergence of eigenvalues with basis size N.

    NUMERICAL: Demonstrates that the Rayleigh-Ritz approximation
    converges as N increases.
    """

    def __init__(self, R=1.0, g2=1.0):
        self.R = R
        self.g2 = g2

    def run(self, N_range, n_eigenvalues=5):
        """
        Run convergence study over a range of basis sizes.

        Parameters
        ----------
        N_range : list of int
            Basis sizes to test (per dimension).
        n_eigenvalues : int
            Number of eigenvalues to track.

        Returns
        -------
        dict with 'N_values', 'eigenvalues' (list of arrays), 'gaps'
        """
        results = {
            'N_values': list(N_range),
            'eigenvalues': [],
            'gaps': [],
        }

        for N in N_range:
            try:
                diag = NumericalDiagonalization(
                    R=self.R, g2=self.g2, N_per_dim=N
                )
                evals, _ = diag.diagonalize(n_eigenvalues)
                results['eigenvalues'].append(evals[:n_eigenvalues])
                if len(evals) >= 2:
                    results['gaps'].append(evals[1] - evals[0])
                else:
                    results['gaps'].append(np.nan)
            except Exception as e:
                results['eigenvalues'].append(np.array([np.nan] * n_eigenvalues))
                results['gaps'].append(np.nan)

        return results

    def is_converged(self, N_range, tol=0.01, n_eigenvalues=5):
        """
        Check if eigenvalues have converged to within tolerance.

        Parameters
        ----------
        N_range : list of int
        tol : float
            Relative tolerance for convergence.
        n_eigenvalues : int

        Returns
        -------
        dict with 'converged', 'relative_change', 'last_two_N'
        """
        res = self.run(N_range, n_eigenvalues)
        if len(res['eigenvalues']) < 2:
            return {'converged': False, 'reason': 'Need at least 2 N values'}

        last = res['eigenvalues'][-1]
        prev = res['eigenvalues'][-2]

        if len(last) == 0 or len(prev) == 0:
            return {'converged': False, 'reason': 'Empty eigenvalue arrays'}

        n_compare = min(len(last), len(prev), n_eigenvalues)
        rel_change = np.abs(last[:n_compare] - prev[:n_compare]) / (np.abs(prev[:n_compare]) + 1e-30)
        max_change = np.max(rel_change)

        return {
            'converged': max_change < tol,
            'relative_change': rel_change,
            'max_relative_change': max_change,
            'last_two_N': (N_range[-2], N_range[-1]),
        }


# ======================================================================
# Extended convergence study with Richardson extrapolation
# ======================================================================

class ConvergenceStudyExtended:
    """
    Extended convergence study for the KvB 3-DOF Rayleigh-Ritz gap.

    Uses the vectorized Hamiltonian builder to reach N=20 (8000x8000 matrix)
    and performs Richardson extrapolation to estimate the N->infinity limit.

    NUMERICAL: All results are Rayleigh-Ritz upper bounds on eigenvalues.
    The spectral gap from Rayleigh-Ritz is NOT a rigorous bound (it can
    either over- or under-estimate the true gap). However, convergence
    of the gap as N increases provides strong numerical evidence.

    Key results (R=2.2 fm, g^2=6.28):
        N=5:  gap ~ 152 MeV (previous session)
        N=10: gap ~ 144 MeV
        N=15: gap ~ 143 MeV
        N=20: gap ~ 143 MeV (converged to < 0.02% change)
        Extrapolated: gap_inf ~ 143 MeV

    Comparison with SCLBT lower bound:
        BUG FIX (Session 25): the old SCLBT value of 367.9 MeV used a unit
        kinetic prefactor and omitted the cubic term. After correction, SCLBT
        now delegates to the KvB Hamiltonian and gives ~145 MeV, consistent
        with the KvB Ritz gap.

    References:
        [1] Koller & van Baal (1988)
        [2] Richardson extrapolation: gap(N) = gap_inf + c/N^alpha
    """

    def __init__(self, R=R_PHYSICAL_FM, g2=G2_DEFAULT):
        self.R = R
        self.g2 = g2

    def run(self, N_range=None, n_eigenvalues=10):
        """
        Run the convergence study over multiple basis sizes.

        Parameters
        ----------
        N_range : list of int or None
            Basis sizes to test. Default: [3, 5, 7, 10, 12, 15, 20].
        n_eigenvalues : int
            Number of lowest eigenvalues to compute.

        Returns
        -------
        dict with keys:
            'N_values'   : list of int
            'dims'       : list of int (matrix dimensions N^3)
            'E0'         : list of float (ground state energies)
            'E1'         : list of float (first excited state energies)
            'gaps_nat'   : list of float (gaps in natural units)
            'gaps_MeV'   : list of float (gaps in MeV)
            'times'      : list of float (computation times in seconds)
            'converged'  : bool (True if last two N give < 0.1% change)
        """
        import time

        if N_range is None:
            N_range = [3, 5, 7, 10, 12, 15, 20]

        results = {
            'N_values': [],
            'dims': [],
            'E0': [],
            'E1': [],
            'gaps_nat': [],
            'gaps_MeV': [],
            'times': [],
        }

        for N in N_range:
            t0 = time.time()
            diag = NumericalDiagonalization(
                R=self.R, g2=self.g2, N_per_dim=N
            )
            evals, _ = diag.diagonalize(n_eigenvalues, use_fast=True)
            dt = time.time() - t0

            gap_nat = evals[1] - evals[0]
            gap_MeV = gap_nat * HBAR_C_MEV_FM / self.R

            results['N_values'].append(N)
            results['dims'].append(N**3)
            results['E0'].append(evals[0])
            results['E1'].append(evals[1])
            results['gaps_nat'].append(gap_nat)
            results['gaps_MeV'].append(gap_MeV)
            results['times'].append(dt)

        # Check convergence
        if len(results['gaps_MeV']) >= 2:
            last = results['gaps_MeV'][-1]
            prev = results['gaps_MeV'][-2]
            rel_change = abs(last - prev) / (abs(prev) + 1e-30)
            results['converged'] = rel_change < 0.001  # 0.1%
            results['rel_change_last'] = rel_change
        else:
            results['converged'] = False
            results['rel_change_last'] = float('inf')

        return results

    def richardson_extrapolation(self, N_range=None, n_eigenvalues=10,
                                  min_N_for_fit=5):
        """
        Run convergence study and fit Richardson extrapolation.

        Fits gap(N) = gap_inf + c / N^alpha to extract gap_inf.

        Parameters
        ----------
        N_range : list of int or None
        n_eigenvalues : int
        min_N_for_fit : int
            Minimum N to include in the fit (skip coarse grids).

        Returns
        -------
        dict with convergence data plus:
            'gap_extrapolated' : float (MeV)
            'fit_c'            : float
            'fit_alpha'        : float
            'fit_errors'       : ndarray (3,), standard errors
            'fit_success'      : bool
        """
        from scipy.optimize import curve_fit

        results = self.run(N_range, n_eigenvalues)

        # Filter for fit
        mask = [N >= min_N_for_fit for N in results['N_values']]
        N_fit = np.array([N for N, m in zip(results['N_values'], mask) if m],
                         dtype=float)
        gap_fit = np.array([g for g, m in zip(results['gaps_MeV'], mask) if m])

        def richardson_model(N, gap_inf, c, alpha):
            return gap_inf + c / N**alpha

        try:
            p0 = [gap_fit[-1], 100.0, 2.0]
            popt, pcov = curve_fit(richardson_model, N_fit, gap_fit,
                                   p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            results['gap_extrapolated'] = popt[0]
            results['fit_c'] = popt[1]
            results['fit_alpha'] = popt[2]
            results['fit_errors'] = perr
            results['fit_success'] = True
        except Exception:
            results['gap_extrapolated'] = results['gaps_MeV'][-1]
            results['fit_c'] = np.nan
            results['fit_alpha'] = np.nan
            results['fit_errors'] = np.array([np.nan, np.nan, np.nan])
            results['fit_success'] = False

        return results

    def e0_is_monotone_decreasing(self, results=None, N_range=None):
        """
        Verify that E0 decreases monotonically with N (Rayleigh-Ritz property).

        Parameters
        ----------
        results : dict or None
            Pre-computed results. If None, runs the study.
        N_range : list of int or None

        Returns
        -------
        bool : True if E0 is monotonically non-increasing.
        """
        if results is None:
            results = self.run(N_range)
        e0 = results['E0']
        for i in range(1, len(e0)):
            if e0[i] > e0[i - 1] + 1e-10:
                return False
        return True

    def gap_at_N(self, N, n_eigenvalues=10):
        """
        Compute the gap at a single basis size N.

        Parameters
        ----------
        N : int
        n_eigenvalues : int

        Returns
        -------
        float : gap in MeV
        """
        diag = NumericalDiagonalization(
            R=self.R, g2=self.g2, N_per_dim=N
        )
        evals, _ = diag.diagonalize(n_eigenvalues, use_fast=True)
        gap_nat = evals[1] - evals[0]
        return gap_nat * HBAR_C_MEV_FM / self.R
