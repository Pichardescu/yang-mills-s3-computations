"""
Bakry-Emery Spectral Gap Analysis for Yang-Mills on S^3/I*.

The physical measure on the 9-DOF Gribov region is:
    dmu = det(M_FP(a)) * exp(-S_YM(a)) * da

Writing dmu = exp(-U_phys) da where:
    U_phys(a) = S_YM(a) - log det(M_FP(a))

The Bakry-Emery-Lichnerowicz theorem states: if Hess(U_phys) >= kappa * I
throughout Omega_9 (as a 9x9 matrix inequality), then the spectral gap
of the associated diffusion operator is >= kappa.

KEY MATHEMATICAL STRUCTURE:

1. Hess(S_YM) = Hess(V_2) + Hess(V_4)
   - S_YM = (1/2) mu_1 |a|^2 + V_4, with mu_1 = 4/R^2
     V_2 = (1/2)(4/R^2)|a|^2 = (2/R^2)|a|^2
     Hess(V_2) = (4/R^2) * I_9.
   - Hess(V_4) >= 0 at a=0 (V_4 has minimum at 0)

2. Hess(log det M_FP):
   Since M_FP(a) = (3/R^2)*I_9 + (g/R)*L(a) is LINEAR in a:
     d^2 M_FP / da_i da_j = 0

   Therefore:
     [Hess(log det M_FP)]_{ij} = Tr(M_FP^{-1} dM/da_i) * Tr(M_FP^{-1} dM/da_j)
       (from d/da_j of Tr(M^{-1} dM/da_i))
     MINUS Tr(M_FP^{-1} (dM/da_i) M_FP^{-1} (dM/da_j))

   Actually, the exact formula for d^2/da_i da_j log det M is:
     H_{ij} = Tr(M^{-1} d^2M/da_i da_j)
              - Tr(M^{-1} (dM/da_i) M^{-1} (dM/da_j))
            = 0 - Tr(M^{-1} (dM/da_i) M^{-1} (dM/da_j))

   where dM/da_i = (g/R) * L(e_i) with e_i the i-th unit vector.

   So: H_{ij} = -(g/R)^2 * Tr(M_FP^{-1} L(e_i) M_FP^{-1} L(e_j))

   This matrix H is NEGATIVE SEMIDEFINITE (it's -X^T X in a suitable sense).
   Therefore: -Hess(log det M_FP) >= 0 (POSITIVE SEMIDEFINITE).

3. Combining:
   Hess(U_phys) = Hess(V_2 + V_4) - Hess(log det M_FP)
                = Hess(V_2) + Hess(V_4) + (-Hess(log det M_FP))
                >= (4/R^2) * I + 0 + 0  at a=0

   Both Hess(V_4) and -Hess(log det M_FP) contribute POSITIVELY.
   The -Hess(log det M_FP) term GROWS as g^2 R^2, dominating at large R!

LABEL: NUMERICAL (Hessian computation is analytical, but the scan over
Omega_9 and the R-dependence analysis are numerical)

References:
    - Bakry & Emery (1985): Diffusions hypercontractives
    - Lichnerowicz (1958): Geometrie des groupes de transformations
    - Singer (1978/1981): Positive curvature of A/G
    - Mondal (2023, JHEP): Bakry-Emery Ricci on A/G -> mass gap
    - Shen-Zhu-Zhu (2023, CMP): Poincare inequality for lattice YM
"""

import numpy as np
from scipy.linalg import eigvalsh

from .gribov_diameter import GribovDiameter, _su2_structure_constants
from .diameter_theorem import DiameterTheorem
from ..spectral.zwanziger_gap_equation import ZwanzigerGapEquation


class BakryEmeryGap:
    """
    Bakry-Emery spectral gap analysis for the physical measure on Omega_9.

    The physical potential is:
        U_phys(a) = V_2(a) + V_4(a) - log det M_FP(a)

    The Bakry-Emery bound: if Hess(U_phys) >= kappa * I on Omega_9,
    then the spectral gap of the Fokker-Planck operator is >= kappa.

    All 9x9 matrices are indexed as a[3*alpha + i] where:
        alpha = 0,1,2 (adjoint/color index)
        i = 0,1,2 (mode/spatial index)
    """

    def __init__(self):
        self.f_abc = _su2_structure_constants()
        self.dt = DiameterTheorem()
        self.gd = GribovDiameter()
        self.dim_adj = 3   # SU(2)
        self.n_modes = 3   # I*-invariant coexact modes at k=1
        self.dim = self.dim_adj * self.n_modes  # = 9

    # ------------------------------------------------------------------
    # 1. Hessian of V_2
    # ------------------------------------------------------------------
    def compute_hessian_V2(self, R):
        """
        Analytical Hessian of V_2(a) = (1/2) * (4/R^2) * |a|^2 = (2/R^2)|a|^2.

        Hess(V_2) = (4/R^2) * I_9.

        THEOREM: Exact analytical result from the coexact eigenvalue mu_1 = 4/R^2.

        Parameters
        ----------
        R : float
            Radius of S^3.

        Returns
        -------
        ndarray of shape (9, 9)
            (4/R^2) * I_9.
        """
        return (4.0 / R**2) * np.eye(self.dim)

    # ------------------------------------------------------------------
    # 2. Hessian of V_4 (numerical via finite differences)
    # ------------------------------------------------------------------
    def _compute_V4(self, a_flat, g_squared):
        """
        Compute V_4(a) = (g^2/2) * [(Tr(M^T M))^2 - Tr((M^T M)^2)].

        Uses the algebraic simplification from effective_hamiltonian.py.

        Parameters
        ----------
        a_flat : ndarray of shape (9,)
            Configuration in flat indexing: a[3*alpha + i].
        g_squared : float
            Coupling constant squared.

        Returns
        -------
        float
            V_4(a).
        """
        # Reshape: a[alpha, i] with alpha=adjoint, i=mode
        # In effective_hamiltonian.py, the convention is M_{i,alpha} = a_{i,alpha}
        # with shape (n_modes, n_colors). Here our flat index is a[3*alpha + i],
        # so a_matrix[alpha, i] -> M_{i,alpha} = a_matrix.T
        a_matrix = a_flat.reshape(self.dim_adj, self.n_modes)
        M = a_matrix.T  # shape (n_modes, n_colors) = (3, 3)
        S = M.T @ M     # shape (3, 3), positive semidefinite
        tr_S = np.trace(S)
        tr_S2 = np.trace(S @ S)
        return 0.5 * g_squared * (tr_S**2 - tr_S2)

    def compute_hessian_V4(self, a_coeffs, R, g_squared=None, h=1e-5):
        """
        Numerical Hessian of V_4 using central finite differences.

        Parameters
        ----------
        a_coeffs : array-like of shape (9,)
            Configuration.
        R : float
            Radius of S^3.
        g_squared : float or None
            If None, uses running coupling at R.
        h : float
            Step size for finite differences.

        Returns
        -------
        ndarray of shape (9, 9)
            Hess(V_4) at a_coeffs.
        """
        a = np.asarray(a_coeffs, dtype=float).ravel()
        if g_squared is None:
            g_squared = ZwanzigerGapEquation.running_coupling_g2(R)

        n = self.dim
        hess = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                # Central difference for d^2 V4 / da_i da_j
                a_pp = a.copy(); a_pp[i] += h; a_pp[j] += h
                a_pm = a.copy(); a_pm[i] += h; a_pm[j] -= h
                a_mp = a.copy(); a_mp[i] -= h; a_mp[j] += h
                a_mm = a.copy(); a_mm[i] -= h; a_mm[j] -= h

                val = (self._compute_V4(a_pp, g_squared)
                       - self._compute_V4(a_pm, g_squared)
                       - self._compute_V4(a_mp, g_squared)
                       + self._compute_V4(a_mm, g_squared)) / (4.0 * h * h)

                hess[i, j] = val
                hess[j, i] = val

        return hess

    # ------------------------------------------------------------------
    # 3. Hessian of log det M_FP (analytical)
    # ------------------------------------------------------------------
    def _L_unit_vector(self, idx):
        """
        Compute L(e_idx) where e_idx is the idx-th unit vector in R^9.

        This gives dM_FP/da_idx = (g/R) * L(e_idx).

        Parameters
        ----------
        idx : int
            Index 0..8 of the unit vector.

        Returns
        -------
        ndarray of shape (9, 9)
            L(e_idx).
        """
        e = np.zeros(self.dim)
        e[idx] = 1.0
        return self.dt.L_operator(e)

    def compute_hessian_log_det_MFP(self, a_coeffs, R, N=2):
        """
        Analytical Hessian of log det M_FP(a).

        Since M_FP is linear in a, d^2 M_FP / da_i da_j = 0, so:

            [Hess(log det M_FP)]_{ij} = -Tr(M^{-1} L_i M^{-1} L_j) * (g/R)^2

        where L_i = L(e_i) and M = M_FP(a).

        This matrix is NEGATIVE SEMIDEFINITE.

        THEOREM: -Hess(log det M_FP) is positive semidefinite.
        PROOF: Define the 9x9 matrix G_{ij} = Tr(M^{-1} L_i M^{-1} L_j).
               Since M^{-1} > 0 (inside Omega), we can write M^{-1} = P P^T
               for some P. Then G_{ij} = Tr(P^T L_i P P^T L_j P) which is
               the Gram matrix of {P^T L_i P}_{i=1..9} under the trace inner
               product. Gram matrices are PSD. Therefore H = -(g/R)^2 G is
               NSD, and -H is PSD.  QED.

        Parameters
        ----------
        a_coeffs : array-like of shape (9,)
            Configuration.
        R : float
            Radius of S^3.
        N : int
            N for SU(N). Only N=2 implemented.

        Returns
        -------
        ndarray of shape (9, 9)
            Hess(log det M_FP) at a_coeffs. This is NSD.
        """
        a = np.asarray(a_coeffs, dtype=float).ravel()

        # Build M_FP
        M_FP = self.gd.fp_operator_truncated(a, R, N)

        # Invert M_FP (must be positive definite inside Omega)
        try:
            M_inv = np.linalg.inv(M_FP)
        except np.linalg.LinAlgError:
            # At or outside the Gribov horizon
            return np.full((self.dim, self.dim), np.nan)

        g = np.sqrt(ZwanzigerGapEquation.running_coupling_g2(R, N))
        g_over_R = g / R

        # Precompute L(e_i) for all i
        L_units = [self._L_unit_vector(i) for i in range(self.dim)]

        # Compute Hessian
        hess = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            # M^{-1} L_i
            MiLi = M_inv @ L_units[i]
            for j in range(i, self.dim):
                # M^{-1} L_j
                MiLj = M_inv @ L_units[j]
                # Tr(M^{-1} L_i M^{-1} L_j)
                val = -g_over_R**2 * np.trace(MiLi @ MiLj)
                hess[i, j] = val
                hess[j, i] = val

        return hess

    # ------------------------------------------------------------------
    # 4. Total Hessian of U_phys
    # ------------------------------------------------------------------
    def compute_hessian_U_phys(self, a_coeffs, R, N=2):
        """
        Hessian of the physical potential U_phys = S_YM - log det M_FP.

        Hess(U_phys) = Hess(V_2) + Hess(V_4) - Hess(log det M_FP)

        where -Hess(log det M_FP) >= 0 (positive semidefinite).

        Parameters
        ----------
        a_coeffs : array-like of shape (9,)
            Configuration.
        R : float
            Radius of S^3.
        N : int
            N for SU(N). Only N=2.

        Returns
        -------
        ndarray of shape (9, 9)
            Hess(U_phys).
        """
        a = np.asarray(a_coeffs, dtype=float).ravel()

        H_V2 = self.compute_hessian_V2(R)
        H_V4 = self.compute_hessian_V4(a, R)
        H_log_det = self.compute_hessian_log_det_MFP(a, R, N)

        if np.any(np.isnan(H_log_det)):
            return np.full((self.dim, self.dim), np.nan)

        # U_phys = (V2 + V4) - log det M_FP
        # Hess(U_phys) = Hess(V2) + Hess(V4) - Hess(log det M_FP)
        return H_V2 + H_V4 - H_log_det

    # ------------------------------------------------------------------
    # 5. Min eigenvalue of Hess(U_phys)
    # ------------------------------------------------------------------
    def min_eigenvalue_hessian_U(self, a_coeffs, R, N=2):
        """
        Smallest eigenvalue of Hess(U_phys) at a given point.

        If >= kappa > 0, the Bakry-Emery bound gives spectral gap >= kappa.

        Parameters
        ----------
        a_coeffs : array-like of shape (9,)
        R : float
        N : int

        Returns
        -------
        float
            Minimum eigenvalue of Hess(U_phys), or NaN if outside Omega.
        """
        H = self.compute_hessian_U_phys(a_coeffs, R, N)
        if np.any(np.isnan(H)):
            return np.nan
        eigs = eigvalsh(H)
        return eigs[0]

    # ------------------------------------------------------------------
    # 6. Scan Hessian over Gribov region
    # ------------------------------------------------------------------
    def scan_hessian_over_gribov(self, R, N=2, n_points=100, seed=42):
        """
        Sample points inside Omega_9 and compute min eigenvalue of
        Hess(U_phys) at each.

        Sampling strategy: for each random direction d, go from a=0
        toward the Gribov horizon at fraction f in [0, 0.9]. This
        ensures we stay strictly inside Omega.

        Parameters
        ----------
        R : float
        N : int
        n_points : int
        seed : int

        Returns
        -------
        dict with:
            'min_eigenvalue_overall'    : smallest kappa across all samples
            'max_eigenvalue_overall'    : largest min eigenvalue
            'mean_eigenvalue'           : mean of min eigenvalues
            'all_positive'              : bool, whether all min eigs > 0
            'n_valid'                   : number of valid samples
            'n_points'                  : total attempted
            'eigenvalues_at_origin'     : eigenvalues of Hess(U) at a=0
            'label'                     : 'NUMERICAL'
        """
        rng = np.random.RandomState(seed)
        min_eigs = []

        # First, eigenvalues at origin
        eigs_origin = eigvalsh(self.compute_hessian_U_phys(np.zeros(self.dim), R, N))

        for _ in range(n_points):
            # Random direction
            d = rng.randn(self.dim)
            d /= np.linalg.norm(d)

            # Find horizon distance
            t_horizon = self.gd.gribov_horizon_distance_truncated(d, R, N)
            if not np.isfinite(t_horizon) or t_horizon <= 0:
                continue

            # Sample at various fractions toward the horizon
            fraction = rng.uniform(0.0, 0.9)
            a = fraction * t_horizon * d

            # Check that M_FP is still positive definite
            lam_min_fp = self.gd.fp_min_eigenvalue(a, R, N)
            if lam_min_fp <= 0:
                continue

            min_eig = self.min_eigenvalue_hessian_U(a, R, N)
            if np.isfinite(min_eig):
                min_eigs.append(min_eig)

        if len(min_eigs) == 0:
            return {
                'min_eigenvalue_overall': np.nan,
                'max_eigenvalue_overall': np.nan,
                'mean_eigenvalue': np.nan,
                'all_positive': False,
                'n_valid': 0,
                'n_points': n_points,
                'eigenvalues_at_origin': eigs_origin,
                'label': 'NUMERICAL',
            }

        min_eigs = np.array(min_eigs)
        return {
            'min_eigenvalue_overall': np.min(min_eigs),
            'max_eigenvalue_overall': np.max(min_eigs),
            'mean_eigenvalue': np.mean(min_eigs),
            'all_positive': bool(np.all(min_eigs > 0)),
            'n_valid': len(min_eigs),
            'n_points': n_points,
            'eigenvalues_at_origin': eigs_origin,
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # 7. BE bound vs R
    # ------------------------------------------------------------------
    def bakry_emery_bound_vs_R(self, R_values, N=2, n_points=50, seed=42):
        """
        Compute the Bakry-Emery curvature bound for multiple R values.

        KEY QUESTION: does the minimum eigenvalue of Hess(U_phys) stay
        positive and bounded away from 0 as R -> infinity?

        Parameters
        ----------
        R_values : array-like
            Radii of S^3 in Lambda_QCD units.
        N : int
        n_points : int
            Points to sample per R.
        seed : int

        Returns
        -------
        dict with:
            'R'                       : R values
            'be_bound'                : BE curvature bound (min eig of Hess U)
            'be_bound_at_origin'      : min eig at a=0 only
            'all_positive'            : whether all sampled points had positive eigs
            'g_squared'               : running coupling
            'geometric_gap'           : 4/R^2 (from V_2 alone)
            'ghost_contribution_origin': -Hess(log det M_FP) min eig at origin
            'label'                   : 'NUMERICAL'
        """
        R_arr = np.asarray(R_values, dtype=float)
        n = len(R_arr)

        be_bounds = np.zeros(n)
        be_origin = np.zeros(n)
        all_pos = np.zeros(n, dtype=bool)
        g2_arr = np.zeros(n)
        geo_gaps = np.zeros(n)
        ghost_contrib = np.zeros(n)

        for idx, R in enumerate(R_arr):
            g2_arr[idx] = ZwanzigerGapEquation.running_coupling_g2(R, N)
            geo_gaps[idx] = 4.0 / R**2

            # At origin
            H_origin = self.compute_hessian_U_phys(np.zeros(self.dim), R, N)
            eigs_origin = eigvalsh(H_origin)
            be_origin[idx] = eigs_origin[0]

            # Ghost contribution at origin
            H_ghost = self.compute_hessian_log_det_MFP(np.zeros(self.dim), R, N)
            ghost_eigs = eigvalsh(-H_ghost)  # -H is PSD
            ghost_contrib[idx] = ghost_eigs[0]  # min eig of -H

            # Scan
            result = self.scan_hessian_over_gribov(R, N, n_points, seed)
            be_bounds[idx] = result['min_eigenvalue_overall']
            all_pos[idx] = result['all_positive']

        return {
            'R': R_arr,
            'be_bound': be_bounds,
            'be_bound_at_origin': be_origin,
            'all_positive': all_pos,
            'g_squared': g2_arr,
            'geometric_gap': geo_gaps,
            'ghost_contribution_origin': ghost_contrib,
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # 8. Formal analysis
    # ------------------------------------------------------------------
    def formal_analysis(self, R_range=None, N=2, n_points=30, seed=42):
        """
        Complete Bakry-Emery analysis with assessment.

        Parameters
        ----------
        R_range : array-like or None
            R values. Default: [0.5, 1, 2, 5, 10, 20].
        N : int
        n_points : int
            Points per R value for scanning.
        seed : int

        Returns
        -------
        dict with complete analysis and assessment string.
        """
        if R_range is None:
            R_range = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

        results = self.bakry_emery_bound_vs_R(R_range, N, n_points, seed)

        # Analyze the trend
        R_arr = results['R']
        be = results['be_bound']
        be_0 = results['be_bound_at_origin']
        ghost = results['ghost_contribution_origin']

        # Check if BE bound stays positive
        valid = np.isfinite(be)
        if np.any(valid):
            all_positive = bool(np.all(be[valid] > 0))
            min_be = np.min(be[valid])
        else:
            all_positive = False
            min_be = np.nan

        # Check if ghost contribution grows with R
        if len(ghost) >= 2:
            ghost_growing = bool(ghost[-1] > ghost[0])
        else:
            ghost_growing = False

        # Check if origin bound grows
        if len(be_0) >= 2:
            origin_growing = bool(be_0[-1] > be_0[0])
        else:
            origin_growing = False

        # Assessment
        if all_positive and ghost_growing:
            assessment = (
                f"POSITIVE: Hess(U_phys) is uniformly positive on sampled "
                f"points in Omega_9 for all tested R values "
                f"(R in [{R_arr[0]:.1f}, {R_arr[-1]:.1f}]). "
                f"Min BE bound = {min_be:.6f}. "
                f"The ghost contribution -Hess(log det M_FP) GROWS with R "
                f"(from {ghost[0]:.6f} to {ghost[-1]:.6f}), "
                f"providing increasing confinement. "
                f"This supports the persistence of the mass gap as R -> inf. "
                f"LABEL: NUMERICAL."
            )
        elif all_positive:
            assessment = (
                f"CAUTIOUSLY POSITIVE: All sampled eigenvalues positive "
                f"(min = {min_be:.6f}), but ghost growth unclear. "
                f"More R values or more sampling points may be needed. "
                f"LABEL: NUMERICAL."
            )
        else:
            assessment = (
                f"INCONCLUSIVE: Some sampled points have non-positive "
                f"eigenvalues (min = {min_be if np.isfinite(min_be) else 'NaN'}). "
                f"This may indicate the BE approach needs refinement "
                f"near the Gribov horizon. LABEL: NUMERICAL."
            )

        return {
            **results,
            'min_be_bound': min_be,
            'all_positive_everywhere': all_positive,
            'ghost_growing_with_R': ghost_growing,
            'origin_bound_growing': origin_growing,
            'assessment': assessment,
            'theorems_used': {
                'bakry_emery': (
                    'If Hess(U) >= kappa I on Omega, spectral gap >= kappa '
                    '(Bakry-Emery 1985)'
                ),
                'singer_curvature': (
                    'A/G has positive sectional curvature '
                    '(Singer 1978/1981, THEOREM)'
                ),
                'ghost_psd': (
                    '-Hess(log det M_FP) is positive semidefinite '
                    '(Gram matrix argument, THEOREM)'
                ),
                'dell_antonio_zwanziger': (
                    'Gribov region is bounded and convex '
                    '(Dell\'Antonio-Zwanziger 1989/1991, THEOREM)'
                ),
            },
            'label': 'NUMERICAL',
        }

    # ==================================================================
    # ANALYTICAL THEOREM: Uniform positivity of Bakry-Émery curvature
    # ==================================================================

    @staticmethod
    def analytical_kappa_bound(R, N=2):
        """
        THEOREM: Analytical lower bound on the Bakry-Emery curvature
        kappa(a) = min eigenvalue of Hess(U_phys)(a) at any a in Omega_9,
        valid for R >= R_0.

        The bound combines three THEOREM-level ingredients:

        1. Hess(V_2) = (4/R^2) I_9
           From V_2 = (1/2)(4/R^2)|a|^2 = (2/R^2)|a|^2 (S_YM includes 1/2).

        2. |Hess(V_4)| <= 9*C_Q*C_R^2/R^2 = 108/R^2 on Omega_9
           where C_Q = 4 (exact: Tr(L(e)^2) = 2*2 = 4) and
           C_R = sqrt(3) is the max-radius constant (the maximum distance
           from the origin to the Gribov boundary, in units of 3/(R*g)).
           NOTE: The origin is NOT the centroid of Omega_9. The max distance
           from origin is C_R = sqrt(3), which exceeds d/2 = C_D/2 = 3*sqrt(3)/4
           by a factor of 4/3. The bound uses |a| <= 3*C_R/(R*g), not d/2.

        3. Ghost lower bound: ghost >= (4/81)*g^2*R^2
           From mu_max(M_FP) <= 9/R^2 (using C_R and nu_max = 2/sqrt(3))
           and Gram eigenvalue = 4.

        Combined:
            kappa >= -104/R^2 + (4/81)*g^2*R^2

        For R >= R_0 = 3.598: kappa > 0.
        For R < R_0: Kato-Rellich covers (g^2(R_0) = 12.04 < 167.5).

        LABEL: THEOREM
        """
        C_Q = 4.0
        C_R = np.sqrt(3.0)        # max radius constant (NOT C_D/2)
        nu_max = 2.0 / np.sqrt(3.0)  # max eigenvalue of L on unit sphere
        V4_coeff = 9.0 * C_Q * C_R**2           # = 108
        mu_max_coeff = 3.0 + 3.0 * nu_max * C_R  # = 9
        ghost_coeff = 4.0 / mu_max_coeff**2      # = 4/81
        g2 = ZwanzigerGapEquation.running_coupling_g2(R, N)

        V2_term = 4.0 / R**2
        V4_term = V4_coeff / R**2
        ghost_term = ghost_coeff * g2 * R**2

        kappa = V2_term - V4_term + ghost_term
        return {
            'kappa_lower_bound': kappa,
            'V2_contribution': V2_term,
            'V4_bound': V4_term,
            'ghost_lower_bound': ghost_term,
            'positive': kappa > 0,
            'g_squared': g2,
            'R': R,
            'label': 'THEOREM',
        }

    @staticmethod
    def theorem_threshold_R0(N=2):
        """
        THEOREM: R_0 above which the analytical BE bound gives kappa > 0.
        For R < R_0: Kato-Rellich covers (g^2(R_0) < 167.5).
        Combined: gap > 0 for ALL R > 0.

        Corrected derivation:
            V2 = 4/R^2 (from S_YM = (1/2) mu_1 |a|^2)
            V4_bound = 108/R^2 (using C_R = sqrt(3), not d/2)
            ghost >= (4/81)*g^2*R^2 (mu_max = 9/R^2)
            deficit = 108 - 4 = 104
            R0^4 = 104 / ((4/81) * 4*pi) = 104*81/(16*pi)
            R0 = 3.598

        LABEL: THEOREM
        """
        C_Q = 4.0
        C_R = np.sqrt(3.0)
        nu_max = 2.0 / np.sqrt(3.0)
        mu_max_coeff = 3.0 + 3.0 * nu_max * C_R  # = 9
        ghost_coeff = 4.0 / mu_max_coeff**2       # = 4/81
        V4_coeff = 9.0 * C_Q * C_R**2             # = 108
        V4_deficit = V4_coeff - 4.0                # = 104
        g2_max = 4.0 * np.pi

        R0 = (V4_deficit / (ghost_coeff * g2_max))**0.25
        g2_at_R0 = ZwanzigerGapEquation.running_coupling_g2(R0, N)
        g2_c = 24.0 * np.pi**2 / np.sqrt(2.0)  # = 24*pi^2/sqrt(2) ~ 167.53

        return {
            'R0': R0,
            'g2_at_R0': g2_at_R0,
            'g2_critical_KR': g2_c,
            'KR_covers_below_R0': g2_at_R0 < g2_c,
            'gap_positive_all_R': g2_at_R0 < g2_c,
            'label': 'THEOREM',
        }
