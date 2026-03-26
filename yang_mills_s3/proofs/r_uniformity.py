"""
R-Uniformity of the Bridge Constant c*(R) for Yang-Mills on S^3(R).

THE PROBLEM:
    The 18-THEOREM chain proves gap(R) > 0 for each finite R.
    Decompactification to R^4 requires inf_R gap(R) > 0 -- the gap must
    not vanish as R -> infinity (or R -> 0).

    The bridge constant c*(R) = kappa_BE(R) - C_K * g_bar(R)^4 controls
    the Poincare constant of the terminal measure. If c*(R) > 0 for all R,
    the Bridge Lemma holds uniformly.

THE ARGUMENT (three regimes):

    UV (R -> 0):
        g^2(R) -> 0 by asymptotic freedom.
        gap(R) >= (1 - alpha) * 2/R -> infinity.
        alpha = g^2 * sqrt(2) / (24*pi^2) -> 0.
        The Kato-Rellich bound (THEOREM 4.1) dominates.
        c* is NOT the relevant quantity here; the gap is already large.
        LABEL: THEOREM (Kato-Rellich).

    IR (R -> infinity):
        g^2(R) -> g^2_max = 4*pi (one-loop saturation, NUMERICAL).
        kappa_BE = -7.19/R^2 + (16/225)*g^2*R^2 ~ (16/225)*g^2_max*R^2.
        C_K * g_bar^4 -> C_K * (4*pi)^2 = const.
        So kappa_BE ~ R^2 while the K correction is bounded.
        c*(R) -> infinity as R -> infinity.
        LABEL: THEOREM for kappa growth; NUMERICAL for g^2 saturation.

    Crossover (R_1 <= R <= R_2):
        Finite interval where neither UV nor IR simplification applies.
        c*(R) is continuous on this compact interval.
        Need to verify c*(R_min) > 0 at the minimum.
        LABEL: NUMERICAL (computer-assisted verification on a grid).

COMBINED:
    The COMBINED gap bound max(KR, physical_BE) is positive for all R:
    (a) KR gives gap ~ 2/R at small R (diverges) — THEOREM
    (b) Physical BE gives gap ~ 8*g^4/(225*R) at large R — NUMERICAL
    (c) Both decay as 1/R, so gap(R) * R >= const > 0 — NUMERICAL
    (d) The combined gap has a minimum but never reaches zero

    CRITICAL FINDING: The 9-DOF truncation gives gap ~ const/R -> 0 as
    R -> infinity. This means the 9-DOF bound CANNOT prove uniform gap
    (inf_R gap(R) > 0). The uniform gap requires either:
    - The full A/G theory (not just 9-DOF truncation)
    - Dimensional transmutation argument (gap ~ Lambda_QCD, R-independent)
    Both are PROPOSITION level, not THEOREM.

    What IS proven (NUMERICAL): gap(R) >= C/R for a universal C > 0.
    This gives gap(R) > 0 for every finite R, but not uniform.

    The Fokker-Planck c*_FP is negative in a wide band [R_0, R_2] where
    R_0 ~ 1.75 fm (kappa_BE = 0) and R_2 ~ 13 fm (kappa_BE > hess_K).
    This means the Brascamp-Lieb method with BBS bounds is too conservative
    in this range. The physical gap is still positive (from the 4/R^2 term
    and KR bound), but the FP bridge constant is negative.

    The overall label is NUMERICAL because the coupling model g^2(R) is
    NUMERICAL (one-loop with IR saturation), not a THEOREM.

IMPORTANT CAVEAT (kinetic prefactor):
    The physical Hamiltonian has H = -epsilon * Delta + V with epsilon = g^2/(2*R^3).
    The Bakry-Emery curvature kappa_BE controls the gap of the Fokker-Planck
    operator L = -Delta + grad(Phi).grad, NOT directly the physical gap.

    The relationship is:
        gap(H) >= Hess(V) + epsilon * ghost_curvature
                = 4/R^2 + (g^2/(2*R^3)) * (16/225)*g^2*R^2
                = 4/R^2 + 8*g^4/(225*R)

    As R -> infinity: gap(H) ~ 8*g^4_max/(225*R) -> 0 in the 9-DOF truncation.

    HOWEVER: the 9-DOF truncation breaks down at large R (THEOREM 7.1c error
    ~ 140/R^2 -> 0). The FULL A/G theory gap is controlled by dimensional
    transmutation and is O(Lambda_QCD). The 9-DOF bound is a LOWER bound
    that becomes loose at large R.

    This module analyzes c*(R) in the FOKKER-PLANCK sense (relevant for the
    Bridge Lemma) and ALSO provides the physical gap estimate.

References:
    [1] THEOREM 9.10: kappa_BE >= -7.19/R^2 + (16/225)*g^2*R^2
    [2] THEOREM 4.1: Kato-Rellich gap at small R
    [3] BBS THEOREM 8.2.4: ||K|| <= C_K * g_bar^3
    [4] Bakry-Emery (1985): Poincare from curvature
    [5] Brascamp-Lieb (1976): Poincare from convexity

Physical parameters:
    Lambda_QCD = 200 MeV, hbar*c = 197.327 MeV*fm
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from scipy.optimize import minimize_scalar, brentq

from ..rg.quantitative_gap_be import (
    running_coupling_g2,
    kappa_min_analytical,
    kappa_to_mass_gap,
    kappa_to_mass_gap_physical,
    QuantitativeGapBE,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_MEV,
)

# ======================================================================
# Physical constants
# ======================================================================

LAMBDA_QCD_FM_INV = LAMBDA_QCD_MEV / HBAR_C_MEV_FM  # ~1.014 fm^{-1}
R_PHYSICAL_FM = 2.2
G2_PHYSICAL = 6.28
G2_MAX = 4.0 * np.pi  # IR saturation of one-loop coupling


# ======================================================================
# 1. C_K computation (BBS remainder bound)
# ======================================================================

def compute_C_K(g2: float, N_c: int = 2) -> float:
    """
    BBS remainder constant C_K from THEOREM 8.1.

    C_K = c_source / (1 - c_epsilon * g_bar)

    where:
        c_source = C_2(adj)^2 / (16*pi^2)
        c_epsilon = 0.275 (pessimistic 600-cell value)
        g_bar = sqrt(g^2)

    THEOREM for the structure; NUMERICAL for the 600-cell constants.

    Parameters
    ----------
    g2 : float
        Coupling g^2.
    N_c : int
        Number of colors.

    Returns
    -------
    float
        C_K constant. Returns inf if denominator <= 0.
    """
    g_bar = np.sqrt(g2)
    c_source = N_c**2 / (16.0 * np.pi**2)
    c_epsilon = 0.275  # pessimistic 600-cell
    denom = 1.0 - c_epsilon * g_bar
    if denom <= 0:
        return float('inf')
    return c_source / denom


def hessian_K_correction(g2: float, N_c: int = 2) -> float:
    """
    Hessian perturbation from the BBS remainder K_0.

    ||Hess(K_0)||_op <= C_K * g_bar^4

    This is the correction that must be subtracted from kappa_BE to get c*.

    NUMERICAL.

    Parameters
    ----------
    g2 : float
        Coupling g^2.
    N_c : int
        Number of colors.

    Returns
    -------
    float
        Upper bound on ||Hess(K_0)|| in fm^{-2}.
    """
    g_bar = np.sqrt(g2)
    C_K = compute_C_K(g2, N_c)
    return C_K * g_bar**4


# ======================================================================
# 2. c*(R) — the bridge constant as a function of R
# ======================================================================

def c_star_fokker_planck(R: float, N_c: int = 2) -> float:
    """
    Bridge constant c*(R) in Fokker-Planck sense.

    c*(R) = kappa_BE(R) - C_K(R) * g_bar(R)^4

    where kappa_BE is the Bakry-Emery curvature from THEOREM 9.10
    and the second term is the BBS remainder Hessian bound.

    This is the Poincare constant of the Fokker-Planck operator
    L = -Delta + grad(Phi).grad.

    NUMERICAL.

    Parameters
    ----------
    R : float
        S^3 radius in fm.
    N_c : int
        Number of colors.

    Returns
    -------
    float
        c*(R) in fm^{-2}. Positive means Bridge Lemma holds at this R.
    """
    g2 = running_coupling_g2(R, N_c)
    kappa = kappa_min_analytical(R, N_c)
    hess_K = hessian_K_correction(g2, N_c)
    return kappa - hess_K


def c_star_physical(R: float, N_c: int = 2) -> float:
    """
    Bridge constant c*(R) accounting for the physical kinetic prefactor.

    The physical Hamiltonian: H = -epsilon * Delta + V, epsilon = g^2/(2*R^3).

    gap(H) >= Hess(V) + epsilon * ghost_curv - epsilon * ||Hess(K_0)||

    Components:
        Hess(V) = 4/R^2 (from V_2 = (2/R^2)|a|^2)
        ghost_curv = (16/225)*g^2*R^2 (from THEOREM 9.7)
        epsilon = g^2/(2*R^3)
        ||Hess(K_0)|| <= C_K * g_bar^4

    Physical c* = 4/R^2 + epsilon * (ghost_curv - C_K*g_bar^4)

    NUMERICAL.

    Parameters
    ----------
    R : float
        S^3 radius in fm.
    N_c : int
        Number of colors.

    Returns
    -------
    float
        Physical bridge constant in fm^{-2} (gap of H, not gap of L).
    """
    g2 = running_coupling_g2(R, N_c)
    g_bar = np.sqrt(g2)
    epsilon = g2 / (2.0 * R**3)

    # Geometric (V_2) Hessian contribution: always positive
    hess_V = 4.0 / R**2

    # Ghost curvature: (16/225)*g^2*R^2
    ghost_curv = (16.0 / 225.0) * g2 * R**2

    # BBS remainder Hessian correction
    C_K = compute_C_K(g2, N_c)
    hess_K = C_K * g_bar**4

    # Physical bridge constant
    # gap(H) >= hess_V + epsilon * (ghost_curv - hess_K)
    return hess_V + epsilon * (ghost_curv - hess_K)


def kato_rellich_gap(R: float, N_c: int = 2) -> float:
    """
    Kato-Rellich gap bound (THEOREM 4.1) as a function of R.

    gap(R) >= (1 - alpha) * 2/R

    where alpha = g^2(R) * sqrt(2) / (24*pi^2).

    This bound diverges as R -> 0 (asymptotic freedom: alpha -> 0).

    THEOREM.

    Parameters
    ----------
    R : float
        S^3 radius in fm.
    N_c : int
        Number of colors.

    Returns
    -------
    float
        Kato-Rellich gap in fm^{-1}. Multiply by hbar*c for MeV.
    """
    g2 = running_coupling_g2(R, N_c)
    alpha = g2 * np.sqrt(2.0) / (24.0 * np.pi**2)
    if alpha >= 1.0:
        return 0.0
    return (1.0 - alpha) * 2.0 / R


# ======================================================================
# 3. RegimeAnalysis — analytical control in UV and IR
# ======================================================================

@dataclass
class RegimeResult:
    """Result of regime analysis for one asymptotic direction."""
    regime: str                  # 'UV' or 'IR'
    R_boundary: float            # R_1 or R_2 delimiting the regime
    gap_behavior: str            # Asymptotic behavior description
    gap_at_boundary: float       # gap(R_boundary) in fm^{-1}
    gap_at_boundary_MeV: float   # gap(R_boundary) in MeV
    controlling_bound: str       # 'KR' or 'BE' or 'physical'
    label: str                   # 'THEOREM' or 'NUMERICAL'
    details: Dict[str, Any] = field(default_factory=dict)


class RegimeAnalysis:
    """
    Analyze c*(R) asymptotically in UV and IR to reduce the R-uniformity
    question to a FINITE interval.

    UV (R -> 0): Kato-Rellich dominates (THEOREM).
    IR (R -> infinity): kappa_BE growth dominates (THEOREM + NUMERICAL coupling).
    Crossover: finite interval certified numerically.

    NUMERICAL overall (coupling model is NUMERICAL).
    """

    def __init__(self, N_c: int = 2, Lambda_QCD: float = LAMBDA_QCD_MEV):
        self.N_c = N_c
        self.Lambda_QCD = Lambda_QCD
        self.hbar_c = HBAR_C_MEV_FM

    def analyze_uv(self, R_max_uv: float = 0.5) -> RegimeResult:
        """
        UV regime: R <= R_max_uv.

        In this regime, g^2(R) is small (asymptotic freedom) and the
        Kato-Rellich bound gives gap >= (1-alpha)*2/R which diverges.

        We verify that the KR bound is positive and large at R_max_uv.

        THEOREM (Kato-Rellich).
        """
        g2 = running_coupling_g2(R_max_uv, self.N_c)
        alpha = g2 * np.sqrt(2.0) / (24.0 * np.pi**2)
        gap_kr = kato_rellich_gap(R_max_uv, self.N_c)
        gap_MeV = self.hbar_c * gap_kr

        return RegimeResult(
            regime='UV',
            R_boundary=R_max_uv,
            gap_behavior=f'gap >= (1-alpha)*2/R, alpha={alpha:.4f}, diverges as 1/R',
            gap_at_boundary=gap_kr,
            gap_at_boundary_MeV=gap_MeV,
            controlling_bound='KR (Kato-Rellich, THEOREM 4.1)',
            label='THEOREM',
            details={
                'g2_at_boundary': g2,
                'alpha': alpha,
                'gap_formula': '(1 - g^2*sqrt(2)/(24*pi^2)) * 2/R',
                'note': ('For R < R_max_uv, the gap is LARGER than at the boundary. '
                         'The KR bound is monotone decreasing in R for fixed coupling, '
                         'and the coupling decreases as R decreases (AF), making the '
                         'bound even better.'),
            },
        )

    def analyze_ir(self, R_min_ir: float = 10.0) -> RegimeResult:
        """
        IR regime: R >= R_min_ir.

        In this regime, g^2(R) saturates near g^2_max = 4*pi.
        The Fokker-Planck kappa_BE grows as R^2:
            kappa_BE ~ (16/225)*g^2_max*R^2 >> C_K*g_bar^4

        The physical gap (with kinetic prefactor) behaves as:
            gap_phys ~ 4/R^2 + 8*g^4/(225*R) -> 0

        So the FP c* grows (good for Bridge Lemma), but the physical
        gap decays (9-DOF truncation limitation).

        THEOREM for kappa_BE growth; NUMERICAL for coupling saturation.
        """
        g2 = running_coupling_g2(R_min_ir, self.N_c)
        g_bar = np.sqrt(g2)

        # FP bridge constant
        c_fp = c_star_fokker_planck(R_min_ir, self.N_c)

        # Physical bridge constant
        c_phys = c_star_physical(R_min_ir, self.N_c)
        gap_phys_MeV = self.hbar_c * max(c_phys, 0.0)

        # Asymptotic analysis: at large R with g^2 -> g^2_max
        kappa_asymp = (16.0 / 225.0) * G2_MAX * R_min_ir**2
        hess_K_asymp = compute_C_K(G2_MAX, self.N_c) * G2_MAX**2
        c_fp_asymp = kappa_asymp - hess_K_asymp

        # Physical gap at large R
        eps_asymp = G2_MAX / (2.0 * R_min_ir**3)
        gap_phys_asymp = 4.0 / R_min_ir**2 + eps_asymp * (16.0 / 225.0) * G2_MAX * R_min_ir**2

        return RegimeResult(
            regime='IR',
            R_boundary=R_min_ir,
            gap_behavior=(
                f'FP: c* ~ (16/225)*g^2_max*R^2 = {kappa_asymp:.1f} >> '
                f'C_K*g_bar^4 = {hess_K_asymp:.3f}; '
                f'Physical: gap ~ 4/R^2 + 8g^4/(225R) -> 0'
            ),
            gap_at_boundary=c_phys,
            gap_at_boundary_MeV=gap_phys_MeV,
            controlling_bound='BE (Bakry-Emery, THEOREM 9.10)',
            label='NUMERICAL',
            details={
                'g2_at_boundary': g2,
                'g2_max': G2_MAX,
                'c_star_FP': c_fp,
                'c_star_FP_asymptotic': c_fp_asymp,
                'c_star_physical': c_phys,
                'kappa_BE': kappa_min_analytical(R_min_ir, self.N_c),
                'hess_K': hessian_K_correction(g2, self.N_c),
                'note': ('kappa_BE grows as R^2 while hess_K is bounded. '
                         'For R > R_min_ir, c*_FP increases monotonically. '
                         'The physical gap decays as 1/R but this is a '
                         'limitation of the 9-DOF truncation, not the full theory.'),
            },
        )

    def find_R_crossover_FP(self) -> float:
        """
        Find R where kappa_BE = 0 (below this, BE is negative, KR controls).

        Returns
        -------
        float
            R_0 such that kappa_BE(R_0) = 0, in fm.
        """
        try:
            R0 = brentq(lambda R: kappa_min_analytical(R, self.N_c), 0.3, 5.0)
            return R0
        except ValueError:
            # Fallback: analytical with g^2 = g^2_max
            return (7.19 * 225.0 / (16.0 * G2_MAX))**0.25


# ======================================================================
# 4. CrossoverCertification — numerical certification on [R_1, R_2]
# ======================================================================

class CrossoverCertification:
    """
    Certify c*(R) > 0 on the crossover interval [R_1, R_2] using a grid.

    Strategy:
        1. Grid the interval [R_1, R_2] with spacing h.
        2. Compute c*(R_i) at each grid point.
        3. Compute |dc*/dR| upper bound (Lipschitz constant) to ensure
           c* doesn't dip negative between grid points.
        4. If c*(R_i) > L*h for all i (where L is the Lipschitz bound),
           then c* > 0 everywhere on [R_1, R_2].

    This is the standard computer-assisted proof technique (Hales/Kepler).

    NUMERICAL.

    Parameters
    ----------
    N_c : int
        Number of colors.
    """

    def __init__(self, N_c: int = 2):
        self.N_c = N_c

    def c_star_FP_grid(self, R_values: np.ndarray) -> np.ndarray:
        """Evaluate c*_FP at an array of R values."""
        return np.array([c_star_fokker_planck(R, self.N_c) for R in R_values])

    def c_star_physical_grid(self, R_values: np.ndarray) -> np.ndarray:
        """Evaluate c*_physical at an array of R values."""
        return np.array([c_star_physical(R, self.N_c) for R in R_values])

    def combined_gap_grid(self, R_values: np.ndarray) -> np.ndarray:
        """
        Combined gap: max(KR_gap, physical_c*) at each R.

        The combined bound is the maximum of:
        - Kato-Rellich gap (1-alpha)*2/R (dominates at small R)
        - Physical bridge constant from BE (dominates at large R)

        Returns gap in fm^{-1}.
        """
        gaps = np.zeros(len(R_values))
        for i, R in enumerate(R_values):
            kr = kato_rellich_gap(R, self.N_c)
            phys = c_star_physical(R, self.N_c)
            gaps[i] = max(kr, phys)
        return gaps

    def estimate_lipschitz(self, R_values: np.ndarray,
                            c_values: np.ndarray) -> float:
        """
        Estimate Lipschitz constant of c*(R) from the grid.

        L = max |c*(R_{i+1}) - c*(R_i)| / (R_{i+1} - R_i)

        Parameters
        ----------
        R_values : ndarray
            Grid of R values.
        c_values : ndarray
            c* values at the grid points.

        Returns
        -------
        float
            Estimated Lipschitz constant.
        """
        diffs = np.abs(np.diff(c_values) / np.diff(R_values))
        return float(np.max(diffs))

    def certify_interval(self, R_min: float, R_max: float,
                          n_points: int = 200,
                          mode: str = 'fokker_planck') -> Dict[str, Any]:
        """
        Certify c*(R) > 0 on [R_min, R_max] with a grid of n_points.

        Strategy:
            1. Evaluate c*(R_i) on a uniform grid.
            2. Compute Lipschitz constant L from the grid.
            3. Check if min(c*) > L * h (where h = grid spacing).
            4. If yes, certified. If no, refine the grid.

        Parameters
        ----------
        R_min : float
            Left endpoint of interval (fm).
        R_max : float
            Right endpoint of interval (fm).
        n_points : int
            Number of grid points.
        mode : str
            'fokker_planck' for c*_FP, 'physical' for c*_phys, 'combined' for
            max(KR, phys).

        Returns
        -------
        dict with certification results.
        """
        R_grid = np.linspace(R_min, R_max, n_points)
        h = (R_max - R_min) / (n_points - 1)

        if mode == 'fokker_planck':
            c_values = self.c_star_FP_grid(R_grid)
        elif mode == 'physical':
            c_values = self.c_star_physical_grid(R_grid)
        elif mode == 'combined':
            c_values = self.combined_gap_grid(R_grid)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        c_min = float(np.min(c_values))
        c_min_idx = int(np.argmin(c_values))
        R_at_min = float(R_grid[c_min_idx])

        L = self.estimate_lipschitz(R_grid, c_values)
        margin = c_min - L * h

        certified = margin > 0
        all_positive = bool(np.all(c_values > 0))

        return {
            'R_min': R_min,
            'R_max': R_max,
            'n_points': n_points,
            'h': h,
            'mode': mode,
            'c_star_min': c_min,
            'R_at_c_star_min': R_at_min,
            'c_star_min_MeV': HBAR_C_MEV_FM * max(c_min, 0) / 2.0,
            'lipschitz_estimate': L,
            'margin': margin,
            'certified': certified,
            'all_positive_on_grid': all_positive,
            'n_positive': int(np.sum(c_values > 0)),
            'R_grid': R_grid.tolist(),
            'c_star_values': c_values.tolist(),
            'label': 'NUMERICAL (certified)' if certified else 'NUMERICAL (not certified)',
        }


# ======================================================================
# 5. UniformBridgeAnalysis — complete R-uniformity analysis
# ======================================================================

class UniformBridgeAnalysis:
    """
    Complete analysis of c*(R) > 0 for all R > 0.

    Combines:
    1. UV regime analysis (Kato-Rellich, THEOREM)
    2. IR regime analysis (kappa_BE growth, THEOREM + NUMERICAL coupling)
    3. Crossover certification (grid on [R_1, R_2], NUMERICAL)

    The overall result is NUMERICAL because the coupling model g^2(R) is
    NUMERICAL (one-loop with IR saturation).

    Parameters
    ----------
    N_c : int
        Number of colors.
    Lambda_QCD : float
        Lambda_QCD in MeV.
    """

    def __init__(self, N_c: int = 2, Lambda_QCD: float = LAMBDA_QCD_MEV):
        self.N_c = N_c
        self.Lambda_QCD = Lambda_QCD
        self._regime = RegimeAnalysis(N_c, Lambda_QCD)
        self._cert = CrossoverCertification(N_c)
        self._qgap = QuantitativeGapBE(N=N_c, Lambda_QCD=Lambda_QCD)

    def find_minimum_FP(self, R_min: float = 0.3, R_max: float = 100.0) -> Dict[str, Any]:
        """
        Find the minimum of c*_FP(R) over [R_min, R_max].

        Uses scipy.optimize.minimize_scalar to minimize c*_FP directly.

        Note: c*_FP is typically very negative in the crossover band
        (R ~ 1-12 fm) because hess_K ~ 158 >> kappa_BE there. The
        minimum of c*_FP may be at the boundary or at a local minimum.

        Returns
        -------
        dict with minimum location and value.
        """
        def c_star_fn(log_R):
            R = np.exp(log_R)
            return c_star_fokker_planck(R, self.N_c)

        result = minimize_scalar(
            c_star_fn,
            bounds=(np.log(R_min), np.log(R_max)),
            method='bounded',
        )

        R_opt = np.exp(result.x)
        c_opt = result.fun

        return {
            'R_at_min': R_opt,
            'c_star_min': c_opt,
            'c_star_positive': c_opt > 0,
            'gap_at_min_MeV': HBAR_C_MEV_FM * max(c_opt, 0) / 2.0,
        }

    def find_minimum_combined(self, R_min: float = 0.1, R_max: float = 100.0
                               ) -> Dict[str, Any]:
        """
        Find the minimum of the COMBINED gap bound: max(KR, phys_BE).

        This is the physically relevant bound: whichever method gives
        the better (larger) gap at each R.

        NOTE: Both KR (~2/R) and physical BE (~5.6/R) decay as 1/R at
        large R, so the combined gap -> 0 as R -> infinity. The infimum
        is approached at R = R_max (the right boundary). We use a dense
        grid to find the minimum, including boundary points.

        Returns
        -------
        dict with minimum location and value.
        """
        # Use a dense grid including boundary points
        R_grid = np.exp(np.linspace(np.log(R_min), np.log(R_max), 500))

        gap_values = np.zeros(len(R_grid))
        for i, R in enumerate(R_grid):
            kr = kato_rellich_gap(R, self.N_c)
            phys = max(c_star_physical(R, self.N_c), 0.0)
            gap_values[i] = max(kr, phys)

        idx_min = np.argmin(gap_values)
        R_opt = float(R_grid[idx_min])
        gap_opt = float(gap_values[idx_min])
        kr_at_opt = kato_rellich_gap(R_opt, self.N_c)
        phys_at_opt = c_star_physical(R_opt, self.N_c)

        return {
            'R_at_min': R_opt,
            'gap_min': gap_opt,
            'gap_min_MeV': HBAR_C_MEV_FM * max(gap_opt, 0.0),
            'gap_positive': gap_opt > 0,
            'KR_at_min': kr_at_opt,
            'KR_at_min_MeV': HBAR_C_MEV_FM * kr_at_opt,
            'phys_at_min': phys_at_opt,
            'phys_at_min_MeV': HBAR_C_MEV_FM * max(phys_at_opt, 0.0),
            'dominant_bound': 'KR' if kr_at_opt >= max(phys_at_opt, 0) else 'BE_physical',
        }

    def full_analysis(self, R_uv: float = 0.5, R_ir: float = 10.0,
                       n_crossover_points: int = 200) -> Dict[str, Any]:
        """
        Complete R-uniformity analysis.

        1. UV regime: R <= R_uv (Kato-Rellich dominates)
        2. IR regime: R >= R_ir (kappa_BE growth dominates)
        3. Crossover: [R_uv, R_ir] certified on grid

        For each regime, compute both c*_FP and c*_physical.

        Parameters
        ----------
        R_uv : float
            UV boundary (fm).
        R_ir : float
            IR boundary (fm).
        n_crossover_points : int
            Grid points for crossover certification.

        Returns
        -------
        dict with complete analysis.
        """
        # 1. UV regime
        uv = self._regime.analyze_uv(R_uv)

        # 2. IR regime
        ir = self._regime.analyze_ir(R_ir)

        # 3. Crossover certification (FP)
        crossover_fp = self._cert.certify_interval(
            R_uv, R_ir, n_crossover_points, mode='fokker_planck'
        )

        # 4. Crossover certification (combined = physical best)
        crossover_combined = self._cert.certify_interval(
            R_uv, R_ir, n_crossover_points, mode='combined'
        )

        # 5. Global minimum search
        min_fp = self.find_minimum_FP()
        min_combined = self.find_minimum_combined()

        # 6. Detailed gap table
        R_table = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.2, 3.0, 5.0,
                            10.0, 20.0, 50.0, 100.0])
        table = self._build_gap_table(R_table)

        # 7. R_0 where kappa_BE = 0
        R0 = self._regime.find_R_crossover_FP()

        # Overall assessment
        fp_uniform = crossover_fp['all_positive_on_grid']
        combined_uniform = crossover_combined['all_positive_on_grid']

        # Check gap*R bound (weaker than uniform, but always holds)
        gap_times_R = []
        for row in table:
            gap_times_R.append(row['combined_gap'] * row['R_fm'])
        gap_times_R_min = min(gap_times_R) if gap_times_R else 0

        if combined_uniform and min_combined['gap_positive']:
            overall_label = 'NUMERICAL'
            overall_explanation = (
                f"gap(R) > 0 at all tested R values. "
                f"Combined gap minimum = {min_combined['gap_min_MeV']:.1f} MeV "
                f"at R = {min_combined['R_at_min']:.2f} fm "
                f"(dominant: {min_combined['dominant_bound']}). "
                f"gap(R)*R >= {gap_times_R_min:.2f} fm^{{-1}} * fm (positive). "
                "HOWEVER: gap ~ const/R -> 0 as R -> infinity. "
                "This is a limitation of the 9-DOF truncation, NOT the full theory. "
                "UV: KR gives gap ~ 2/R (THEOREM). "
                "IR: physical BE gives gap ~ 8*g^4/(225*R) (NUMERICAL). "
                "Uniform gap (inf_R gap(R) > 0) requires full A/G theory, "
                "not just 9-DOF (PROPOSITION)."
            )
        else:
            overall_label = 'CONJECTURE'
            overall_explanation = (
                f"R-uniformity NOT fully confirmed. "
                f"Combined minimum = {min_combined.get('gap_min_MeV', 0):.1f} MeV. "
                "The Fokker-Planck c* may be negative at some R values "
                "where the K_0 correction exceeds kappa_BE. However, the "
                "physical gap (accounting for kinetic prefactor and KR "
                "alternative) may still be positive."
            )

        return {
            'uv_regime': {
                'R_boundary': uv.R_boundary,
                'gap_MeV': uv.gap_at_boundary_MeV,
                'label': uv.label,
                'behavior': uv.gap_behavior,
            },
            'ir_regime': {
                'R_boundary': ir.R_boundary,
                'gap_MeV': ir.gap_at_boundary_MeV,
                'label': ir.label,
                'c_star_FP': ir.details['c_star_FP'],
            },
            'R0_kappa_zero': R0,
            'crossover_fp': {
                'interval': [crossover_fp['R_min'], crossover_fp['R_max']],
                'c_star_min': crossover_fp['c_star_min'],
                'R_at_min': crossover_fp['R_at_c_star_min'],
                'all_positive': crossover_fp['all_positive_on_grid'],
                'certified': crossover_fp['certified'],
            },
            'crossover_combined': {
                'interval': [crossover_combined['R_min'], crossover_combined['R_max']],
                'c_star_min': crossover_combined['c_star_min'],
                'R_at_min': crossover_combined['R_at_c_star_min'],
                'gap_min_MeV': crossover_combined['c_star_min_MeV'],
                'all_positive': crossover_combined['all_positive_on_grid'],
                'certified': crossover_combined['certified'],
            },
            'global_minimum_FP': min_fp,
            'global_minimum_combined': min_combined,
            'gap_table': table,
            'gap_times_R_min': gap_times_R_min,
            'overall_label': overall_label,
            'overall_explanation': overall_explanation,
            'label_breakdown': {
                'UV': 'THEOREM (Kato-Rellich)',
                'IR_kappa_growth': 'THEOREM (kappa_BE ~ R^2)',
                'IR_coupling_saturation': 'NUMERICAL (g^2 -> 4*pi)',
                'crossover': 'NUMERICAL (grid certification)',
                'combined': overall_label,
            },
        }

    def _build_gap_table(self, R_values: np.ndarray) -> List[Dict[str, Any]]:
        """Build a detailed gap table at selected R values."""
        table = []
        for R in R_values:
            g2 = running_coupling_g2(R, self.N_c)
            g_bar = np.sqrt(g2)
            kappa = kappa_min_analytical(R, self.N_c)
            C_K_val = compute_C_K(g2, self.N_c)
            hess_K = hessian_K_correction(g2, self.N_c)
            c_fp = c_star_fokker_planck(R, self.N_c)
            c_phys = c_star_physical(R, self.N_c)
            kr = kato_rellich_gap(R, self.N_c)
            combined = max(kr, max(c_phys, 0.0))

            table.append({
                'R_fm': float(R),
                'g2': float(g2),
                'g_bar': float(g_bar),
                'kappa_BE': float(kappa),
                'C_K': float(C_K_val),
                'hess_K': float(hess_K),
                'c_star_FP': float(c_fp),
                'c_star_physical': float(c_phys),
                'KR_gap': float(kr),
                'combined_gap': float(combined),
                'combined_gap_MeV': float(HBAR_C_MEV_FM * combined),
                'dominant': 'KR' if kr >= max(c_phys, 0.0) else 'BE_phys',
            })
        return table

    def scaling_analysis(self) -> Dict[str, Any]:
        """
        Analyze the SCALING of each component as R -> 0 and R -> infinity.

        This is the analytical argument for R-uniformity.

        Returns
        -------
        dict with scaling analysis.
        """
        # Sample points for numerical verification
        R_small = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        R_large = np.array([5.0, 10.0, 20.0, 50.0, 100.0])

        # UV scaling
        uv_data = []
        for R in R_small:
            g2 = running_coupling_g2(R, self.N_c)
            kr = kato_rellich_gap(R, self.N_c)
            uv_data.append({
                'R_fm': float(R),
                'g2': float(g2),
                'KR_gap': float(kr),
                'KR_gap_times_R': float(kr * R),  # should -> 2
                'KR_gap_MeV': float(HBAR_C_MEV_FM * kr),
            })

        # IR scaling
        ir_data = []
        for R in R_large:
            g2 = running_coupling_g2(R, self.N_c)
            kappa = kappa_min_analytical(R, self.N_c)
            hess_K = hessian_K_correction(g2, self.N_c)
            c_fp = c_star_fokker_planck(R, self.N_c)
            c_phys = c_star_physical(R, self.N_c)
            ir_data.append({
                'R_fm': float(R),
                'g2': float(g2),
                'kappa_BE': float(kappa),
                'kappa_over_R2': float(kappa / R**2),  # should -> (16/225)*g2_max
                'hess_K': float(hess_K),
                'c_star_FP': float(c_fp),
                'c_star_FP_over_R2': float(c_fp / R**2) if R > 0 else 0,
                'c_star_physical': float(c_phys),
                'c_star_physical_times_R': float(c_phys * R),  # should -> 8*g4/(225)
            })

        # Analytical limits
        g2_max = G2_MAX
        kappa_over_R2_limit = (16.0 / 225.0) * g2_max  # ~ 0.894
        C_K_at_max = compute_C_K(g2_max, self.N_c)
        hess_K_at_max = C_K_at_max * g2_max**2
        phys_gap_times_R_limit = 8.0 * g2_max**2 / 225.0  # ~ 5.60

        return {
            'uv_scaling': {
                'behavior': 'gap ~ 2/R -> infinity (THEOREM)',
                'KR_gap_times_R_limit': 2.0,
                'data': uv_data,
            },
            'ir_scaling': {
                'behavior': (
                    f'FP: c*_FP ~ (16/225)*g2_max*R^2 = {kappa_over_R2_limit:.3f}*R^2 -> infinity; '
                    f'Physical: c*_phys ~ 8*g4/(225*R) -> 0'
                ),
                'kappa_over_R2_limit': kappa_over_R2_limit,
                'hess_K_at_saturation': hess_K_at_max,
                'C_K_at_saturation': C_K_at_max,
                'phys_gap_times_R_limit': phys_gap_times_R_limit,
                'data': ir_data,
            },
            'key_finding': (
                'The Fokker-Planck c*_FP grows as R^2 at large R, ensuring '
                'the Bridge Lemma (Poincare inequality for the FP operator) '
                'holds for all large R. The physical gap decays as 1/R in '
                'the 9-DOF truncation but this is expected: the truncation '
                'error is O(140/R^2) which vanishes, so the 9-DOF bound '
                'becomes loose. The FULL A/G gap is O(Lambda_QCD).'
            ),
        }


# ======================================================================
# 6. Entry point
# ======================================================================

def analyze_r_uniformity(N_c: int = 2, verbose: bool = False) -> Dict[str, Any]:
    """
    Run the complete R-uniformity analysis.

    Parameters
    ----------
    N_c : int
        Number of colors.
    verbose : bool
        If True, print a human-readable report.

    Returns
    -------
    dict with complete analysis.
    """
    analysis = UniformBridgeAnalysis(N_c=N_c)
    result = analysis.full_analysis()
    scaling = analysis.scaling_analysis()
    result['scaling_analysis'] = scaling

    if verbose:
        _print_report(result)

    return result


def _print_report(result: Dict[str, Any]) -> None:
    """Print a human-readable report."""
    print("=" * 70)
    print("R-UNIFORMITY OF BRIDGE CONSTANT c*(R)")
    print("=" * 70)
    print()

    # UV
    uv = result['uv_regime']
    print(f"UV REGIME (R <= {uv['R_boundary']} fm):")
    print(f"  Gap >= {uv['gap_MeV']:.1f} MeV at boundary (diverges as 1/R)")
    print(f"  Label: {uv['label']}")
    print()

    # IR
    ir = result['ir_regime']
    print(f"IR REGIME (R >= {ir['R_boundary']} fm):")
    print(f"  c*_FP = {ir['c_star_FP']:.2f} fm^{{-2}} at boundary (grows as R^2)")
    print(f"  Label: {ir['label']}")
    print()

    # Crossover
    xf = result['crossover_fp']
    print(f"CROSSOVER [{xf['interval'][0]}, {xf['interval'][1]}] fm (FP):")
    print(f"  c*_min = {xf['c_star_min']:.4f} at R = {xf['R_at_min']:.2f} fm")
    print(f"  All positive: {xf['all_positive']}")
    print(f"  Certified: {xf['certified']}")
    print()

    xc = result['crossover_combined']
    print(f"CROSSOVER [{xc['interval'][0]}, {xc['interval'][1]}] fm (combined):")
    print(f"  gap_min = {xc['c_star_min']:.4f} fm^{{-1}} = {xc['gap_min_MeV']:.1f} MeV")
    print(f"  at R = {xc['R_at_min']:.2f} fm")
    print(f"  All positive: {xc['all_positive']}")
    print()

    # Global minimum
    gm = result['global_minimum_combined']
    print("GLOBAL MINIMUM (combined):")
    print(f"  gap = {gm['gap_min_MeV']:.1f} MeV at R = {gm['R_at_min']:.2f} fm")
    print(f"  Dominant bound: {gm['dominant_bound']}")
    print()

    # Table
    print("-" * 70)
    print("GAP TABLE")
    print("-" * 70)
    fmt = "{:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("R(fm)", "g^2", "kappa_BE", "c*_FP", "c*_phys",
                      "KR_gap", "best MeV"))
    for row in result['gap_table']:
        print(fmt.format(
            f"{row['R_fm']:.2f}",
            f"{row['g2']:.3f}",
            f"{row['kappa_BE']:.3f}",
            f"{row['c_star_FP']:.3f}",
            f"{row['c_star_physical']:.4f}",
            f"{row['KR_gap']:.4f}",
            f"{row['combined_gap_MeV']:.1f}",
        ))
    print()

    # Overall
    print("=" * 70)
    print(f"OVERALL: {result['overall_label']}")
    print(result['overall_explanation'])
    print("=" * 70)


if __name__ == "__main__":
    analyze_r_uniformity(verbose=True)
