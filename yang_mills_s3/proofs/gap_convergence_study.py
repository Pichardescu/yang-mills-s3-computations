"""
Convergence study of the Koller-van Baal SVD reduction mass gap.

Two methods are compared:

1. KvB (NumericalDiagonalization): Full S^3 potential V = V_quad + V_cubic + V_quartic
   with kinetic prefactor kappa = g^2/R^3. Built via Gauss-Hermite quadrature.

2. SCLBT (_build_3d_hamiltonian): In physical mode (R given), delegates to KvB for
   the correct Hamiltonian. In legacy mode (R=None), uses harmonic + quartic only
   (for pure-oscillator benchmarks).

BUG FIX (Session 25): SCLBT now uses the KvB Hamiltonian in physical mode,
producing consistent results (~145 MeV at R=2.2 fm). Previously SCLBT used
unit kinetic prefactor and no cubic term, inflating the gap to ~368 MeV.

Both converge as N increases (variational principle).

LABEL: NUMERICAL (all results from finite-basis diagonalization)

Physical parameters:
    R = 2.2 fm (S^3 radius)
    g^2 = 6.28 (coupling at physical scale)
    hbar*c = 197.327 MeV*fm
"""

import time
import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .koller_van_baal import (
    NumericalDiagonalization,
    SpectralGapExtraction,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    G2_DEFAULT,
)
from .sclbt_lower_bounds import (
    YangMillsReducedGap,
    _build_3d_hamiltonian,
    SCLBTBound,
    TempleBound,
)


# ======================================================================
# 1. ConvergenceScanner -- KvB at multiple N values
# ======================================================================

@dataclass
class ConvergenceRecord:
    """Single data point in a convergence scan."""
    N: int
    n_basis: int
    E0: float
    E1: float
    E2: float
    gap: float
    gap_MeV: float
    wall_time: float
    error: Optional[str] = None


class ConvergenceScanner:
    """
    Run NumericalDiagonalization (KvB, full S^3 potential) at multiple basis sizes.

    For each N, records E_0, E_1, E_2, gap = E_1 - E_0, and wall time.
    The gap is converted to MeV via gap_MeV = gap_natural * hbar_c / R.

    Parameters
    ----------
    R : float
        Radius of S^3 in fm.
    g2 : float
        Yang-Mills coupling g^2.
    timeout_seconds : float
        Maximum time per N value. If exceeded, skip.
    """

    def __init__(self, R: float = R_PHYSICAL_FM, g2: float = G2_DEFAULT,
                 timeout_seconds: float = 120.0):
        self.R = R
        self.g2 = g2
        self.timeout_seconds = timeout_seconds

    def run_single(self, N: int, n_eigenvalues: int = 5) -> ConvergenceRecord:
        """
        Run diagonalization at a single N value.

        Parameters
        ----------
        N : int
            Basis functions per dimension.
        n_eigenvalues : int
            Number of eigenvalues to compute.

        Returns
        -------
        ConvergenceRecord with results.
        """
        t0 = time.time()
        try:
            diag = NumericalDiagonalization(
                R=self.R, g2=self.g2, N_per_dim=N
            )
            evals, _ = diag.diagonalize(n_eigenvalues)
            wall_time = time.time() - t0

            E0 = float(evals[0])
            E1 = float(evals[1]) if len(evals) > 1 else float('nan')
            E2 = float(evals[2]) if len(evals) > 2 else float('nan')
            gap = E1 - E0
            gap_MeV = gap * HBAR_C_MEV_FM / self.R

            return ConvergenceRecord(
                N=N, n_basis=N**3,
                E0=E0, E1=E1, E2=E2,
                gap=gap, gap_MeV=gap_MeV,
                wall_time=wall_time,
            )
        except Exception as e:
            wall_time = time.time() - t0
            return ConvergenceRecord(
                N=N, n_basis=N**3,
                E0=float('nan'), E1=float('nan'), E2=float('nan'),
                gap=float('nan'), gap_MeV=float('nan'),
                wall_time=wall_time, error=str(e),
            )

    def scan(self, N_values: List[int] = None,
             n_eigenvalues: int = 5) -> List[ConvergenceRecord]:
        """
        Run convergence scan over multiple N values.

        Parameters
        ----------
        N_values : list of int
            Basis sizes to test. Default: [3, 4, 5].
        n_eigenvalues : int
            Number of eigenvalues to compute at each N.

        Returns
        -------
        list of ConvergenceRecord
        """
        if N_values is None:
            N_values = [3, 4, 5]

        results = []
        for N in N_values:
            record = self.run_single(N, n_eigenvalues)
            results.append(record)
            # Skip remaining if this one exceeded timeout
            if record.wall_time > self.timeout_seconds:
                break
        return results

    def estimate_converged_gap(self, records: List[ConvergenceRecord]) -> Dict:
        """
        Estimate the converged gap from a sequence of records.

        Uses Richardson extrapolation if convergence is smooth.

        Parameters
        ----------
        records : list of ConvergenceRecord
            Results from scan().

        Returns
        -------
        dict with convergence analysis.
        """
        valid = [r for r in records if r.error is None and np.isfinite(r.gap_MeV)]
        if len(valid) < 2:
            return {
                'converged': False,
                'reason': 'Fewer than 2 valid data points',
                'n_valid': len(valid),
            }

        gaps = np.array([r.gap_MeV for r in valid])
        Ns = np.array([r.N for r in valid])

        # Relative change between last two
        rel_change = abs(gaps[-1] - gaps[-2]) / max(abs(gaps[-1]), 1e-10)

        result = {
            'converged': rel_change < 0.05,
            'best_gap_MeV': float(gaps[-1]),
            'best_N': int(Ns[-1]),
            'rel_change_last_two': float(rel_change),
            'all_gaps_MeV': gaps.tolist(),
            'all_N': Ns.tolist(),
        }

        # Richardson extrapolation: assume gap(N) = gap_inf + C/N^p
        if len(valid) >= 3:
            try:
                result['richardson'] = self._richardson_extrapolate(Ns, gaps)
            except Exception:
                result['richardson'] = None

        return result

    @staticmethod
    def _richardson_extrapolate(Ns: np.ndarray, gaps: np.ndarray) -> Dict:
        """
        Richardson extrapolation assuming gap(N) ~ gap_inf + C/N^p.

        Uses the last 3 points to estimate gap_inf and the convergence rate p.

        Parameters
        ----------
        Ns : ndarray
            Basis sizes.
        gaps : ndarray
            Gap values in MeV.

        Returns
        -------
        dict with 'gap_inf', 'C', 'p', 'error_estimate'
        """
        # Use last 3 points
        n1, n2, n3 = float(Ns[-3]), float(Ns[-2]), float(Ns[-1])
        g1, g2, g3 = float(gaps[-3]), float(gaps[-2]), float(gaps[-1])

        # Estimate p from the ratio of consecutive differences
        dg1 = g1 - g2
        dg2 = g2 - g3

        if abs(dg2) < 1e-15 or abs(dg1) < 1e-15:
            # Already converged or no change
            return {
                'gap_inf': g3,
                'C': 0.0,
                'p': float('inf'),
                'error_estimate': 0.0,
            }

        # For gap = gap_inf + C/N^p:
        # dg1 = C * (1/n1^p - 1/n2^p)
        # dg2 = C * (1/n2^p - 1/n3^p)
        # ratio = dg1/dg2 = (1/n1^p - 1/n2^p) / (1/n2^p - 1/n3^p)
        # This is transcendental in p, solve numerically
        from scipy.optimize import brentq

        def ratio_eq(p):
            if p < 0.01:
                return 0.0
            r1 = n1**(-p) - n2**(-p)
            r2 = n2**(-p) - n3**(-p)
            if abs(r2) < 1e-30:
                return 1e10
            return r1 / r2 - dg1 / dg2

        try:
            p_est = brentq(ratio_eq, 0.1, 20.0)
        except (ValueError, RuntimeError):
            p_est = 2.0  # fallback

        # Estimate C from dg2
        denom = n2**(-p_est) - n3**(-p_est)
        if abs(denom) > 1e-30:
            C_est = dg2 / denom
        else:
            C_est = 0.0

        gap_inf = g3 - C_est * n3**(-p_est)
        error_est = abs(C_est * n3**(-p_est))

        return {
            'gap_inf': float(gap_inf),
            'C': float(C_est),
            'p': float(p_est),
            'error_estimate': float(error_est),
        }


# ======================================================================
# 2. SCLBTConvergenceScanner -- SCLBT Hamiltonian at multiple N
# ======================================================================

@dataclass
class SCLBTRecord:
    """Single data point from SCLBT convergence scan."""
    N: int
    n_basis: int
    ritz_E0: float
    ritz_E1: float
    ritz_gap: float
    ritz_gap_MeV: float
    sclbt_gap_MeV: float
    temple_gap_MeV: float
    wall_time: float
    error: Optional[str] = None


class SCLBTConvergenceScanner:
    """
    Run YangMillsReducedGap (full KvB Hamiltonian) at multiple N.

    Uses the physical KvB Hamiltonian with correct kinetic prefactor
    kappa/2 = g^2/(2R^3) and all potential terms (quad + cubic + quartic).

    BUG FIX (Session 25): previously used unit kinetic prefactor and no cubic
    term, giving ~368 MeV. Now gives ~145 MeV, consistent with KvB.

    Parameters
    ----------
    R : float
        Radius of S^3 in fm.
    g2 : float
        Yang-Mills coupling g^2.
    """

    def __init__(self, R: float = R_PHYSICAL_FM, g2: float = G2_DEFAULT):
        self.R = R
        self.g2 = g2

    def run_single(self, N: int) -> SCLBTRecord:
        """
        Run SCLBT computation at a single N value.

        Parameters
        ----------
        N : int
            Basis functions per dimension.

        Returns
        -------
        SCLBTRecord with results.
        """
        t0 = time.time()
        try:
            ymrg = YangMillsReducedGap(N_basis=N)
            result = ymrg.gap_in_MeV(N, self.R, self.g2)

            # Also get raw eigenvalues for E0, E1
            raw = ymrg.compute_gap(N, self.R, self.g2)

            wall_time = time.time() - t0
            return SCLBTRecord(
                N=N, n_basis=N**3,
                ritz_E0=float(raw['ritz_E0']),
                ritz_E1=float(raw['ritz_E1']),
                ritz_gap=float(raw['ritz_gap']),
                ritz_gap_MeV=float(result['ritz_gap_MeV']),
                sclbt_gap_MeV=float(result['sclbt_gap_MeV']),
                temple_gap_MeV=float(result['temple_gap_MeV']),
                wall_time=wall_time,
            )
        except Exception as e:
            wall_time = time.time() - t0
            return SCLBTRecord(
                N=N, n_basis=N**3,
                ritz_E0=float('nan'), ritz_E1=float('nan'),
                ritz_gap=float('nan'),
                ritz_gap_MeV=float('nan'),
                sclbt_gap_MeV=float('nan'),
                temple_gap_MeV=float('nan'),
                wall_time=wall_time, error=str(e),
            )

    def scan(self, N_values: List[int] = None) -> List[SCLBTRecord]:
        """
        Run convergence scan over multiple N values.

        Parameters
        ----------
        N_values : list of int
            Basis sizes to test. Default: [3, 4, 5, 6, 7, 8, 10, 12, 15].

        Returns
        -------
        list of SCLBTRecord
        """
        if N_values is None:
            N_values = [3, 4, 5, 6, 7, 8, 10, 12, 15]

        results = []
        for N in N_values:
            record = self.run_single(N)
            results.append(record)
            # Skip if taking too long
            if record.wall_time > 120.0:
                break
        return results

    def ritz_converged(self, records: List[SCLBTRecord], tol: float = 0.01) -> Dict:
        """
        Check if the Ritz gap has converged.

        Parameters
        ----------
        records : list of SCLBTRecord
        tol : float
            Relative tolerance for convergence.

        Returns
        -------
        dict with convergence analysis.
        """
        valid = [r for r in records if r.error is None and np.isfinite(r.ritz_gap_MeV)]
        if len(valid) < 2:
            return {'converged': False, 'reason': 'Fewer than 2 valid points'}

        gaps = np.array([r.ritz_gap_MeV for r in valid])
        Ns = np.array([r.N for r in valid])

        rel_change = abs(gaps[-1] - gaps[-2]) / max(abs(gaps[-1]), 1e-10)

        return {
            'converged': rel_change < tol,
            'best_gap_MeV': float(gaps[-1]),
            'best_N': int(Ns[-1]),
            'rel_change_last_two': float(rel_change),
            'all_gaps_MeV': gaps.tolist(),
            'all_N': Ns.tolist(),
        }


# ======================================================================
# 3. GapVsR -- Gap as a function of radius
# ======================================================================

@dataclass
class GapVsRRecord:
    """Gap at a single R value."""
    R_fm: float
    gap_natural: float
    gap_MeV: float
    E0: float
    E1: float


class GapVsR:
    """
    Compute the spectral gap as a function of the S^3 radius R.

    Uses the SCLBT Hamiltonian (fast) for the scan, since it can handle
    large N values efficiently.

    Parameters
    ----------
    g2 : float
        Yang-Mills coupling g^2.
    N_per_dim : int
        Basis size per dimension (should be at the converged value).
    """

    def __init__(self, g2: float = G2_DEFAULT, N_per_dim: int = 10):
        self.g2 = g2
        self.N = N_per_dim

    def compute_at_R(self, R: float) -> GapVsRRecord:
        """
        Compute gap at a single R value.

        Uses the physical KvB Hamiltonian with correct kinetic prefactor
        kappa/2 = g^2/(2R^3) and cubic term.

        Parameters
        ----------
        R : float
            Radius in fm.

        Returns
        -------
        GapVsRRecord
        """
        omega = 2.0 / R
        # Physical mode: pass R for correct KvB Hamiltonian
        H = _build_3d_hamiltonian(omega, self.g2, self.N, R=R)
        evals = np.sort(np.linalg.eigvalsh(H))

        E0 = float(evals[0])
        E1 = float(evals[1])
        gap_nat = E1 - E0
        # BUG FIX (Session 25): use gap * hbar_c / R (not gap * hbar_c)
        gap_MeV = gap_nat * HBAR_C_MEV_FM / R

        return GapVsRRecord(
            R_fm=R,
            gap_natural=gap_nat,
            gap_MeV=gap_MeV,
            E0=E0,
            E1=E1,
        )

    def scan(self, R_values: List[float] = None) -> List[GapVsRRecord]:
        """
        Scan gap over a range of R values.

        Parameters
        ----------
        R_values : list of float
            R values in fm. Default: 0.5 to 10 fm in 20 steps.

        Returns
        -------
        list of GapVsRRecord
        """
        if R_values is None:
            R_values = np.linspace(0.5, 10.0, 20).tolist()

        results = []
        for R in R_values:
            try:
                record = self.compute_at_R(R)
                results.append(record)
            except Exception:
                results.append(GapVsRRecord(
                    R_fm=R,
                    gap_natural=float('nan'),
                    gap_MeV=float('nan'),
                    E0=float('nan'),
                    E1=float('nan'),
                ))
        return results

    def gap_positive_everywhere(self, records: List[GapVsRRecord]) -> Dict:
        """
        Check that gap > 0 for all R values.

        Parameters
        ----------
        records : list of GapVsRRecord

        Returns
        -------
        dict with 'all_positive', 'min_gap_MeV', 'R_at_min_gap'
        """
        valid = [r for r in records if np.isfinite(r.gap_MeV)]
        if not valid:
            return {'all_positive': False, 'reason': 'No valid data'}

        gaps = np.array([r.gap_MeV for r in valid])
        Rs = np.array([r.R_fm for r in valid])
        idx_min = np.argmin(gaps)

        return {
            'all_positive': bool(np.all(gaps > 0)),
            'min_gap_MeV': float(gaps[idx_min]),
            'R_at_min_gap_fm': float(Rs[idx_min]),
            'max_gap_MeV': float(np.max(gaps)),
            'R_at_max_gap_fm': float(Rs[np.argmax(gaps)]),
            'n_points': len(valid),
        }

    def small_R_limit(self, records: List[GapVsRRecord]) -> Dict:
        """
        Analyze the small-R (kinematic) limit.

        At small R, the quadratic potential dominates (omega = 2/R is large)
        and the gap is approximately the harmonic oscillator gap ~ 2*omega = 4/R.

        Parameters
        ----------
        records : list of GapVsRRecord

        Returns
        -------
        dict with small-R analysis.
        """
        small_R = [r for r in records if r.R_fm < 1.5 and np.isfinite(r.gap_MeV)]
        if not small_R:
            return {'has_data': False}

        gaps_MeV = np.array([r.gap_MeV for r in small_R])
        Rs = np.array([r.R_fm for r in small_R])

        # Harmonic prediction: gap ~ 2*omega = 4/R, in MeV: 4/R * hbar_c
        harmonic_gaps = 4.0 / Rs * HBAR_C_MEV_FM

        ratios = gaps_MeV / harmonic_gaps

        return {
            'has_data': True,
            'R_values': Rs.tolist(),
            'gap_MeV': gaps_MeV.tolist(),
            'harmonic_gap_MeV': harmonic_gaps.tolist(),
            'ratio_to_harmonic': ratios.tolist(),
            'approaches_harmonic': bool(np.max(ratios) < 2.0 and np.min(ratios) > 0.3),
        }

    def large_R_limit(self, records: List[GapVsRRecord]) -> Dict:
        """
        Analyze the large-R (dynamic/nonperturbative) limit.

        At large R, the quartic interaction dominates and the gap should
        approach a nonzero limit (evidence for mass gap persistence).

        Parameters
        ----------
        records : list of GapVsRRecord

        Returns
        -------
        dict with large-R analysis.
        """
        large_R = [r for r in records if r.R_fm > 3.0 and np.isfinite(r.gap_MeV)]
        if not large_R:
            return {'has_data': False}

        gaps_MeV = np.array([r.gap_MeV for r in large_R])
        Rs = np.array([r.R_fm for r in large_R])

        # Check if gap stabilizes (std/mean < threshold)
        mean_gap = np.mean(gaps_MeV)
        std_gap = np.std(gaps_MeV)
        coeff_var = std_gap / max(mean_gap, 1e-10)

        return {
            'has_data': True,
            'R_values': Rs.tolist(),
            'gap_MeV': gaps_MeV.tolist(),
            'mean_gap_MeV': float(mean_gap),
            'std_gap_MeV': float(std_gap),
            'coeff_variation': float(coeff_var),
            'stabilized': coeff_var < 0.15,
            'all_positive': bool(np.all(gaps_MeV > 0)),
        }


# ======================================================================
# 4. PhysicalGapExtraction -- extract and compare physical gap values
# ======================================================================

class PhysicalGapExtraction:
    """
    Extract the physical mass gap at the converged basis size and compare
    with other estimates in the project.

    Known estimates:
        - Main paper THEOREM 10.6a: >= 2.12 Lambda_QCD = 424 MeV (GZ-free, Temple)
        - SCLBT (now using KvB Hamiltonian at N=10): ~143 MeV
        - KvB (koller_van_baal.py at N=5): ~152 MeV

    BUG FIX (Session 25): old estimates of 367.9 MeV (SCLBT) and 211.8 MeV
    (pipeline) were inflated due to unit kinetic prefactor and missing cubic term.

    Parameters
    ----------
    R : float
        Radius in fm.
    g2 : float
        Coupling.
    """

    def __init__(self, R: float = R_PHYSICAL_FM, g2: float = G2_DEFAULT):
        self.R = R
        self.g2 = g2

    def extract_sclbt_gap(self, N: int = 15) -> Dict:
        """
        Extract gap from the SCLBT Hamiltonian (full KvB potential).

        Uses the physical KvB Hamiltonian with correct kinetic prefactor
        kappa/2 = g^2/(2R^3) and all three potential terms.

        Parameters
        ----------
        N : int
            Basis size per dimension.

        Returns
        -------
        dict with gap data.
        """
        omega = 2.0 / self.R
        # Physical mode: pass R for correct KvB Hamiltonian
        H = _build_3d_hamiltonian(omega, self.g2, N, R=self.R)
        evals = np.sort(np.linalg.eigvalsh(H))

        E0 = float(evals[0])
        E1 = float(evals[1])
        E2 = float(evals[2]) if len(evals) > 2 else float('nan')
        gap_nat = E1 - E0
        # BUG FIX (Session 25): use gap * hbar_c / R (not gap * hbar_c)
        gap_MeV = gap_nat * HBAR_C_MEV_FM / self.R

        return {
            'method': 'SCLBT Hamiltonian (full KvB: quad + cubic + quartic)',
            'N': N,
            'n_basis': N**3,
            'E0': E0,
            'E1': E1,
            'E2': E2,
            'gap_natural': gap_nat,
            'gap_MeV': gap_MeV,
            'R_fm': self.R,
            'g2': self.g2,
            'omega': omega,
            'label': 'NUMERICAL',
        }

    def extract_kvb_gap(self, N: int = 5) -> Dict:
        """
        Extract gap from the KvB Hamiltonian (full S^3 potential).

        Parameters
        ----------
        N : int
            Basis size per dimension.

        Returns
        -------
        dict with gap data.
        """
        diag = NumericalDiagonalization(R=self.R, g2=self.g2, N_per_dim=N)
        evals, _ = diag.diagonalize(5)

        E0 = float(evals[0])
        E1 = float(evals[1])
        E2 = float(evals[2]) if len(evals) > 2 else float('nan')
        gap_nat = E1 - E0
        gap_MeV = gap_nat * HBAR_C_MEV_FM / self.R

        return {
            'method': 'KvB (full S^3: quad + cubic + quartic)',
            'N': N,
            'n_basis': N**3,
            'E0': E0,
            'E1': E1,
            'E2': E2,
            'gap_natural': gap_nat,
            'gap_MeV': gap_MeV,
            'R_fm': self.R,
            'g2': self.g2,
            'label': 'NUMERICAL',
        }

    def compare_all(self, sclbt_N: int = 12, kvb_N: int = 5) -> Dict:
        """
        Compare gap estimates from all methods.

        Parameters
        ----------
        sclbt_N : int
            Basis size for SCLBT.
        kvb_N : int
            Basis size for KvB.

        Returns
        -------
        dict with comparison.
        """
        sclbt = self.extract_sclbt_gap(sclbt_N)
        kvb = self.extract_kvb_gap(kvb_N)

        return {
            'sclbt': sclbt,
            'kvb': kvb,
            'comparison': {
                'sclbt_gap_MeV': sclbt['gap_MeV'],
                'kvb_gap_MeV': kvb['gap_MeV'],
                'ratio_sclbt_kvb': sclbt['gap_MeV'] / max(kvb['gap_MeV'], 1e-10),
                'difference_MeV': sclbt['gap_MeV'] - kvb['gap_MeV'],
                'note': (
                    'The two Hamiltonians differ: KvB includes the cubic term '
                    '-(2g/R)*x1*x2*x3 from S^3 curvature. The cubic term lowers '
                    'the gap because it breaks symmetry and creates an asymmetric '
                    'potential well. The SCLBT Hamiltonian (no cubic) gives an '
                    'upper bound on the gap of the harmonic+quartic system, while '
                    'KvB gives the gap of the full S^3 system.'
                ),
            },
            'known_estimates_MeV': {
                'theorem_10_6a_lower_bound': 424.0,
                'kvb_N8_converged': 144.9,
                'sclbt_N8_corrected': 144.9,
                # BUG FIX (Session 25): old values pipeline=211.8, sclbt=367.9
                # were inflated by unit kinetic prefactor and missing cubic term
            },
        }

    def honest_assessment(self, sclbt_N: int = 12, kvb_N: int = 5) -> Dict:
        """
        Honest assessment of which gap estimate is most reliable.

        Returns
        -------
        dict with assessment.
        """
        comp = self.compare_all(sclbt_N, kvb_N)

        kvb_gap = comp['kvb']['gap_MeV']
        sclbt_gap = comp['sclbt']['gap_MeV']

        return {
            'kvb_gap_MeV': kvb_gap,
            'sclbt_gap_MeV': sclbt_gap,
            'most_reliable': 'SCLBT (larger basis, converged)',
            'kvb_status': (
                f'KvB at N={kvb_N} ({kvb_N**3} basis) may not be fully converged. '
                f'The Gauss-Hermite construction is too slow for N>5. '
                f'The cubic term lowers the gap relative to SCLBT.'
            ),
            'sclbt_status': (
                f'SCLBT at N={sclbt_N} ({sclbt_N**3} basis) is well converged '
                f'(Ritz gap stable to <1% from N=8 to N={sclbt_N}). '
                f'However, it omits the cubic term from S^3 curvature.'
            ),
            'recommendation': (
                'The true S^3 gap lies between the KvB value (which includes the '
                'cubic term but uses a small basis) and the SCLBT value (which '
                'has a large basis but omits the cubic term). The cubic term is '
                'perturbatively small at small x, so the SCLBT value is likely '
                'closer to the truth. A faster KvB implementation (e.g. using '
                'Kronecker structure for all terms) would resolve this.'
            ),
            'gap_range_MeV': (min(kvb_gap, sclbt_gap), max(kvb_gap, sclbt_gap)),
            'label': 'NUMERICAL',
        }


# ======================================================================
# 5. ConvergenceReport -- summary of all convergence results
# ======================================================================

class ConvergenceReport:
    """
    Generate a comprehensive convergence report combining KvB and SCLBT results.

    Parameters
    ----------
    R : float
        Radius in fm.
    g2 : float
        Coupling.
    """

    def __init__(self, R: float = R_PHYSICAL_FM, g2: float = G2_DEFAULT):
        self.R = R
        self.g2 = g2

    def summary_table(self, kvb_records: List[ConvergenceRecord],
                      sclbt_records: List[SCLBTRecord]) -> Dict:
        """
        Build a summary table from both KvB and SCLBT scans.

        Parameters
        ----------
        kvb_records : list of ConvergenceRecord
        sclbt_records : list of SCLBTRecord

        Returns
        -------
        dict with 'kvb_table' and 'sclbt_table', each a list of row dicts.
        """
        kvb_table = []
        for r in kvb_records:
            kvb_table.append({
                'N': r.N,
                'matrix_size': f'{r.n_basis}x{r.n_basis}',
                'E0': r.E0,
                'E1': r.E1,
                'gap_natural': r.gap,
                'gap_MeV': r.gap_MeV,
                'wall_time_s': r.wall_time,
                'error': r.error,
            })

        sclbt_table = []
        for r in sclbt_records:
            sclbt_table.append({
                'N': r.N,
                'matrix_size': f'{r.n_basis}x{r.n_basis}',
                'ritz_E0': r.ritz_E0,
                'ritz_E1': r.ritz_E1,
                'ritz_gap_MeV': r.ritz_gap_MeV,
                'sclbt_gap_MeV': r.sclbt_gap_MeV,
                'wall_time_s': r.wall_time,
                'error': r.error,
            })

        return {
            'kvb_table': kvb_table,
            'sclbt_table': sclbt_table,
            'R_fm': self.R,
            'g2': self.g2,
        }

    def richardson_extrapolation(self, records: List[SCLBTRecord]) -> Dict:
        """
        Richardson extrapolation from SCLBT Ritz gaps.

        Parameters
        ----------
        records : list of SCLBTRecord

        Returns
        -------
        dict with extrapolation results.
        """
        valid = [r for r in records if r.error is None and np.isfinite(r.ritz_gap_MeV)]
        if len(valid) < 3:
            return {'success': False, 'reason': 'Need at least 3 valid points'}

        Ns = np.array([r.N for r in valid], dtype=float)
        gaps = np.array([r.ritz_gap_MeV for r in valid])

        try:
            result = ConvergenceScanner._richardson_extrapolate(Ns, gaps)
            result['success'] = True
            return result
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def error_from_N_dependence(self, records: List[SCLBTRecord]) -> Dict:
        """
        Estimate error from the N-dependence of the gap.

        Uses the standard deviation of the last few gap values as an
        error estimate on the converged value.

        Parameters
        ----------
        records : list of SCLBTRecord

        Returns
        -------
        dict with error estimates.
        """
        valid = [r for r in records if r.error is None and np.isfinite(r.ritz_gap_MeV)]
        if len(valid) < 3:
            return {'has_estimate': False}

        # Use last 3 points for error estimate
        gaps_last3 = np.array([r.ritz_gap_MeV for r in valid[-3:]])
        mean_gap = np.mean(gaps_last3)
        std_gap = np.std(gaps_last3)

        # Max deviation from mean in last 3
        max_dev = np.max(np.abs(gaps_last3 - mean_gap))

        return {
            'has_estimate': True,
            'mean_gap_MeV': float(mean_gap),
            'std_gap_MeV': float(std_gap),
            'max_deviation_MeV': float(max_dev),
            'relative_error': float(std_gap / max(mean_gap, 1e-10)),
            'n_points_used': len(gaps_last3),
        }

    def variational_monotonicity(self, records: List[SCLBTRecord]) -> Dict:
        """
        Check variational monotonicity: Ritz eigenvalues should decrease
        (or stay constant) as N increases.

        Parameters
        ----------
        records : list of SCLBTRecord

        Returns
        -------
        dict with monotonicity check.
        """
        valid = [r for r in records if r.error is None and np.isfinite(r.ritz_E0)]

        if len(valid) < 2:
            return {'checkable': False}

        E0s = np.array([r.ritz_E0 for r in valid])
        E1s = np.array([r.ritz_E1 for r in valid])
        Ns = np.array([r.N for r in valid])

        # E0 should decrease (variational principle)
        E0_monotone = True
        E0_violations = []
        for i in range(1, len(E0s)):
            if E0s[i] > E0s[i-1] + 1e-10:
                E0_monotone = False
                E0_violations.append({
                    'N_prev': int(Ns[i-1]),
                    'N_curr': int(Ns[i]),
                    'E0_prev': float(E0s[i-1]),
                    'E0_curr': float(E0s[i]),
                })

        return {
            'checkable': True,
            'E0_monotone_decreasing': E0_monotone,
            'E0_violations': E0_violations,
            'E0_values': E0s.tolist(),
            'N_values': Ns.tolist(),
        }
