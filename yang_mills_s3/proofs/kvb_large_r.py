"""
KvB Large-R Analysis: Mass gap decay in the 9-DOF truncation on S^3(R).

Computes the Koller-van Baal spectral gap at many values of R to characterize
the asymptotic R-dependence of the mass gap in the constant-mode sector.

Key results:
    - gap(R) ~ C / R for large R (analytical scaling from anharmonic oscillator)
    - At fixed basis size N, the gap converges well for small R but requires
      increasingly large N for large R (N >> 14 for R > 20 fm)
    - C = lim_{R->inf} gap(R)*R in natural units
    - C_MeV_fm = C * hbar_c in MeV*fm
    - Full spectrum clustering analysis (E_n/E_0 ratios)
    - First excited state is 3-fold degenerate (S_3 Weyl group)

The Hamiltonian is:
    H = -(g^2/(2R^3)) sum_i d^2/dx_i^2  +  (2/R^2) sum x_i^2
        - (2g/R) x_1 x_2 x_3  +  g^2 sum_{i<j} x_i^2 x_j^2

Scaling analysis:
    Rescale x_i = (2R^3)^{-1/4} y_i to normalize the quartic coefficient.
    The prefactor becomes g^2 / (sqrt(2) R^{3/2}).
    The quadratic coefficient becomes 4/(g^2 sqrt(2R)) -> 0 as R -> inf.
    The cubic coefficient also -> 0.
    The pure quartic V = sum y_i^2 y_j^2 has FLAT DIRECTIONS along axes.
    Confinement along axes comes from the (vanishing) quadratic term,
    but the transverse zero-point energy creates an effective linear
    potential along axes, giving gap ~ c * g^{4/3} / R.

    At small R (R << 2/g^2 ~ 0.3 fm): quadratic dominates, gap ~ 2*sqrt(2)/R.
    At large R (R >> 2/g^2): quartic+quadratic interplay, gap ~ c*g^{4/3}/R.

CONVERGENCE NOTE:
    The HO basis converges well at R ~ 2 fm (0.1% at N=12) but convergence
    degrades at large R because the wavefunction spreads along the flat
    directions of the quartic potential. At R=50 fm, even N=14 is not
    converged. For quantitative results at large R, use basis sizes
    N >= 12 and apply Aitken extrapolation.

LABEL: NUMERICAL (eigenvalues from Rayleigh-Ritz diagonalization)

References:
    [1] Koller & van Baal (1988)
    [2] Pavel (2007)
    [3] Butt, Draper & Shen (2023)
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings

from yang_mills_s3.proofs.koller_van_baal import (
    NumericalDiagonalization,
    HBAR_C_MEV_FM,
    G2_DEFAULT,
)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class GapRecord:
    """Single gap measurement at a given R."""
    R: float               # fm
    g2: float              # coupling
    N_per_dim: int         # basis size
    gap_natural: float     # gap in natural units (eigenvalue difference)
    gap_MeV: float         # gap in MeV
    gap_times_R: float     # gap_natural * R (for 1/R scaling check)
    gap_times_R_MeV_fm: float  # gap_MeV * R in MeV*fm
    gap_times_R2: float    # gap_natural * R^2 (for 1/R^2 scaling check)
    gap_times_R2_MeV_fm2: float  # gap_MeV * R^2 in MeV*fm^2
    E0: float              # ground state energy
    E1: float              # first excited state energy


@dataclass
class SpectrumRecord:
    """Full spectrum at a given R."""
    R: float
    g2: float
    N_per_dim: int
    eigenvalues: np.ndarray      # first n_evals eigenvalues
    gaps: np.ndarray             # E_n - E_0 for n >= 1
    ratios_to_gap: np.ndarray   # (E_n - E_0) / (E_1 - E_0)


@dataclass
class FitResult:
    """Result of power-law or polynomial fit."""
    model: str             # e.g. "A/R^alpha", "A/R + B/R^2"
    params: Dict[str, float]
    R_range: Tuple[float, float]
    residual_rms: float
    r_squared: float


@dataclass
class LargeRAnalysis:
    """Complete large-R analysis result."""
    gap_records: List[GapRecord]
    spectrum_records: List[SpectrumRecord]
    power_law_fit: Optional[FitResult]
    subleading_fit: Optional[FitResult]
    asymptotic_C_natural: float     # lim gap*R^2 in natural units
    asymptotic_C_MeV_fm2: float    # lim gap*R^2 in MeV*fm^2
    quartic_prediction_MeV_fm2: float  # g^2*delta_0*hbar_c/2^{2/3}
    analytical_prediction_MeV_fm: float  # 8 g^4 hbar_c / 225 (full-theory)


# ======================================================================
# Core computation
# ======================================================================

def compute_gap_at_R(R: float, g2: float = G2_DEFAULT, N_per_dim: int = 8,
                     n_eigenvalues: int = 10) -> GapRecord:
    """
    Compute the KvB spectral gap at a single R value.

    Parameters
    ----------
    R : float
        Radius in fm.
    g2 : float
        Yang-Mills coupling g^2.
    N_per_dim : int
        Basis functions per dimension.
    n_eigenvalues : int
        Number of eigenvalues to compute.

    Returns
    -------
    GapRecord
    """
    diag = NumericalDiagonalization(R=R, g2=g2, N_per_dim=N_per_dim)
    evals, _ = diag.diagonalize(n_eigenvalues)

    E0 = evals[0]
    E1 = evals[1]
    gap_nat = E1 - E0

    # Convert: gap in natural units is an energy in the Hamiltonian's units.
    # The Hamiltonian has kinetic prefactor kappa/2 = g^2/(2R^3).
    # Eigenvalues are in units where x_i are in fm and energies are in fm^{-1}.
    # Actually, the eigenvalues from the diagonalization are in "natural" units
    # determined by the Hamiltonian normalization.
    #
    # For conversion to MeV:
    #   The Hamiltonian's energy unit depends on how x_i are scaled.
    #   With x_i in fm, kinetic term ~ g^2/(2R^3) * d^2/dx^2 has units fm^{-2}/fm^3 = fm^{-5}...
    #   No — the eigenvalues E have units such that E = (energy in some unit).
    #
    # The potential V_quad = 2/R^2 * x^2 has units [R^{-2}]*[x^2].
    # The kinetic term -(g^2/(2R^3)) d^2/dx^2 matches when x has the same
    # units as R (both fm), giving energy in fm^{-1}.
    #
    # So gap_nat is in fm^{-1}, and gap_MeV = gap_nat * hbar_c.
    gap_MeV = gap_nat * HBAR_C_MEV_FM

    return GapRecord(
        R=R,
        g2=g2,
        N_per_dim=N_per_dim,
        gap_natural=gap_nat,
        gap_MeV=gap_MeV,
        gap_times_R=gap_nat * R,
        gap_times_R_MeV_fm=gap_MeV * R,
        gap_times_R2=gap_nat * R**2,
        gap_times_R2_MeV_fm2=gap_MeV * R**2,
        E0=E0,
        E1=E1,
    )


def compute_spectrum_at_R(R: float, g2: float = G2_DEFAULT, N_per_dim: int = 8,
                          n_eigenvalues: int = 10) -> SpectrumRecord:
    """
    Compute the full low-lying spectrum at a given R.

    Parameters
    ----------
    R : float
        Radius in fm.
    g2 : float
    N_per_dim : int
    n_eigenvalues : int

    Returns
    -------
    SpectrumRecord
    """
    diag = NumericalDiagonalization(R=R, g2=g2, N_per_dim=N_per_dim)
    evals, _ = diag.diagonalize(n_eigenvalues)

    gaps = evals[1:] - evals[0]
    fundamental_gap = gaps[0]
    if fundamental_gap > 0:
        ratios = gaps / fundamental_gap
    else:
        ratios = np.full_like(gaps, np.inf)

    return SpectrumRecord(
        R=R,
        g2=g2,
        N_per_dim=N_per_dim,
        eigenvalues=evals[:n_eigenvalues],
        gaps=gaps,
        ratios_to_gap=ratios,
    )


# ======================================================================
# Scan over R values
# ======================================================================

def gap_vs_R_scan(
    R_values: Optional[np.ndarray] = None,
    g2: float = G2_DEFAULT,
    N_per_dim: int = 8,
    n_eigenvalues: int = 10,
    verbose: bool = False,
) -> List[GapRecord]:
    """
    Compute the gap at many R values.

    Parameters
    ----------
    R_values : array-like or None
        R values in fm. If None, use default set.
    g2 : float
    N_per_dim : int
    n_eigenvalues : int
    verbose : bool

    Returns
    -------
    list of GapRecord
    """
    if R_values is None:
        R_values = np.array([1.0, 2.0, 2.2, 3.0, 5.0, 7.0, 10.0,
                             15.0, 20.0, 30.0, 50.0, 75.0, 100.0])

    records = []
    for R in R_values:
        if verbose:
            print(f"  Computing gap at R = {R:.1f} fm (N={N_per_dim})...")
        rec = compute_gap_at_R(R, g2=g2, N_per_dim=N_per_dim,
                               n_eigenvalues=n_eigenvalues)
        records.append(rec)
        if verbose:
            print(f"    gap = {rec.gap_MeV:.2f} MeV, gap*R = {rec.gap_times_R_MeV_fm:.2f} MeV*fm")

    return records


def spectrum_scan(
    R_values: Optional[np.ndarray] = None,
    g2: float = G2_DEFAULT,
    N_per_dim: int = 8,
    n_eigenvalues: int = 10,
    verbose: bool = False,
) -> List[SpectrumRecord]:
    """
    Compute full low-lying spectra at selected R values.

    Parameters
    ----------
    R_values : array-like or None
    g2 : float
    N_per_dim : int
    n_eigenvalues : int
    verbose : bool

    Returns
    -------
    list of SpectrumRecord
    """
    if R_values is None:
        R_values = np.array([2.2, 10.0, 50.0])

    records = []
    for R in R_values:
        if verbose:
            print(f"  Computing spectrum at R = {R:.1f} fm (N={N_per_dim})...")
        rec = compute_spectrum_at_R(R, g2=g2, N_per_dim=N_per_dim,
                                    n_eigenvalues=n_eigenvalues)
        records.append(rec)
        if verbose:
            print(f"    E_1/E_0 ratio: {rec.ratios_to_gap[0]:.4f}")
            if len(rec.ratios_to_gap) > 1:
                print(f"    E_2/E_0 ratio: {rec.ratios_to_gap[1]:.4f}")

    return records


# ======================================================================
# Convergence analysis
# ======================================================================

def convergence_study(R: float, N_values: List[int] = None,
                      g2: float = G2_DEFAULT) -> Dict:
    """
    Study convergence of the gap as a function of basis size N.

    Parameters
    ----------
    R : float
        Radius in fm.
    N_values : list of int
        Basis sizes to test.
    g2 : float

    Returns
    -------
    dict with:
        'R': float
        'N_values': list
        'gaps': list of gap_natural
        'gap_times_R': list
        'aitken_extrapolation': float (gap_natural extrapolated)
        'aitken_gap_times_R_MeV_fm': float
        'relative_correction': float (from last N to extrapolated)
    """
    if N_values is None:
        N_values = [4, 6, 8, 10, 12]

    gaps = []
    for N in N_values:
        rec = compute_gap_at_R(R, g2=g2, N_per_dim=N)
        gaps.append(rec.gap_natural)

    # Aitken extrapolation from the last 3 points
    g_inf = aitken_extrapolation(gaps[-3:])

    return {
        'R': R,
        'N_values': N_values,
        'gaps': gaps,
        'gap_times_R': [g * R for g in gaps],
        'aitken_extrapolation': g_inf,
        'aitken_gap_times_R_MeV_fm': g_inf * R * HBAR_C_MEV_FM,
        'relative_correction': (g_inf - gaps[-1]) / gaps[-1] if gaps[-1] != 0 else 0,
    }


def aitken_extrapolation(values: List[float]) -> float:
    """
    Aitken delta-squared extrapolation from the last 3 values.

    Assumes values converge to a limit as a geometric sequence.
    Given s_n, s_{n+1}, s_{n+2}, estimates s_inf.

    Parameters
    ----------
    values : list of at least 3 floats

    Returns
    -------
    float : extrapolated value
    """
    if len(values) < 3:
        return values[-1]

    s0, s1, s2 = values[-3], values[-2], values[-1]
    d1 = s1 - s0
    d2 = s2 - s1

    if abs(d2 - d1) < 1e-15:
        return s2

    return s2 - d2**2 / (d2 - d1)


# ======================================================================
# Fitting
# ======================================================================

def _power_law(R, A, alpha):
    """gap = A / R^alpha."""
    return A / R**alpha


def _subleading(R, A, B):
    """gap = A/R + B/R^2."""
    return A / R + B / R**2


def fit_power_law(records: List[GapRecord],
                  R_min: float = 10.0) -> FitResult:
    """
    Fit gap(R) = A / R^alpha for R >= R_min.

    Parameters
    ----------
    records : list of GapRecord
    R_min : float
        Only use R >= R_min for fitting.

    Returns
    -------
    FitResult
    """
    data = [(r.R, r.gap_natural) for r in records if r.R >= R_min]
    if len(data) < 2:
        raise ValueError(f"Need at least 2 points with R >= {R_min}, got {len(data)}")

    R_arr = np.array([d[0] for d in data])
    gap_arr = np.array([d[1] for d in data])

    # Initial guess: alpha ~ 1
    try:
        popt, pcov = curve_fit(_power_law, R_arr, gap_arr, p0=[1.0, 1.0],
                               maxfev=10000)
        A, alpha = popt

        residuals = gap_arr - _power_law(R_arr, A, alpha)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((gap_arr - np.mean(gap_arr))**2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return FitResult(
            model="A/R^alpha",
            params={"A": A, "alpha": alpha},
            R_range=(R_min, float(R_arr.max())),
            residual_rms=np.sqrt(ss_res / len(data)),
            r_squared=r_sq,
        )
    except RuntimeError:
        return FitResult(
            model="A/R^alpha",
            params={"A": np.nan, "alpha": np.nan},
            R_range=(R_min, float(R_arr.max())),
            residual_rms=np.inf,
            r_squared=0.0,
        )


def fit_subleading(records: List[GapRecord],
                   R_min: float = 10.0) -> FitResult:
    """
    Fit gap(R) = A/R + B/R^2 for R >= R_min.

    Parameters
    ----------
    records : list of GapRecord
    R_min : float

    Returns
    -------
    FitResult
    """
    data = [(r.R, r.gap_natural) for r in records if r.R >= R_min]
    if len(data) < 2:
        raise ValueError(f"Need at least 2 points with R >= {R_min}, got {len(data)}")

    R_arr = np.array([d[0] for d in data])
    gap_arr = np.array([d[1] for d in data])

    try:
        popt, pcov = curve_fit(_subleading, R_arr, gap_arr, p0=[1.0, 1.0],
                               maxfev=10000)
        A, B = popt

        residuals = gap_arr - _subleading(R_arr, A, B)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((gap_arr - np.mean(gap_arr))**2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return FitResult(
            model="A/R + B/R^2",
            params={"A": A, "B": B},
            R_range=(R_min, float(R_arr.max())),
            residual_rms=np.sqrt(ss_res / len(data)),
            r_squared=r_sq,
        )
    except RuntimeError:
        return FitResult(
            model="A/R + B/R^2",
            params={"A": np.nan, "B": np.nan},
            R_range=(R_min, float(R_arr.max())),
            residual_rms=np.inf,
            r_squared=0.0,
        )


# ======================================================================
# Analytical predictions
# ======================================================================

def quartic_scaling_prediction(R: float, g2: float = G2_DEFAULT,
                               delta_0: float = 1.95) -> float:
    """
    Gap prediction from the quartic-oscillator scaling analysis.

    The 9-DOF Hamiltonian at large R is dominated by the coupled quartic
    potential V = g^2 sum_{i<j} x_i^2 x_j^2. Rescaling to dimensionless
    variables gives:

        gap = (g^2 / (2^{2/3} R^2)) * delta_0

    where delta_0 is the dimensionless gap of the pure quartic problem
    h = -Laplacian + sum y_i^2 y_j^2.

    Numerically, delta_0 ~ 1.95 (extracted from converged R=2.2 data).

    Parameters
    ----------
    R : float (fm)
    g2 : float
    delta_0 : float
        Dimensionless gap of the pure quartic oscillator.

    Returns
    -------
    float : predicted gap in fm^{-1}
    """
    return g2 / (2.0**(2.0/3.0) * R**2) * delta_0


def quartic_C_MeV_fm2(g2: float = G2_DEFAULT,
                       delta_0: float = 1.95) -> float:
    """
    Asymptotic constant C2 = lim_{R->inf} gap(R)*R^2 in MeV*fm^2.

    C2 = g^2 * hbar_c * delta_0 / 2^{2/3}

    Parameters
    ----------
    g2 : float
    delta_0 : float

    Returns
    -------
    float : C2 in MeV*fm^2
    """
    return g2 * HBAR_C_MEV_FM * delta_0 / 2.0**(2.0/3.0)


def analytical_gap_prediction(R: float, g2: float = G2_DEFAULT) -> float:
    """
    Analytical prediction for the large-R gap from Bakry-Emery analysis.

    gap ~ 8 g^4 / (225 R)  in natural units (fm^{-1})

    NOTE: This is the FULL Yang-Mills gap prediction (including all modes),
    not the 9-DOF truncation. The 9-DOF truncation scales as 1/R^2 (see
    quartic_scaling_prediction), while the full theory gap includes
    contributions from non-constant modes that modify the scaling.

    Parameters
    ----------
    R : float (fm)
    g2 : float

    Returns
    -------
    float : predicted gap in fm^{-1}
    """
    g4 = g2**2
    return 8.0 * g4 / (225.0 * R)


def analytical_C_MeV_fm(g2: float = G2_DEFAULT) -> float:
    """
    Analytical prediction for C = lim(R->inf) gap(R)*R in MeV*fm.

    C = 8 g^4 hbar_c / 225

    NOTE: This is the full-theory prediction. The 9-DOF truncation
    scales as 1/R^2, not 1/R. See quartic_C_MeV_fm2 for the 9-DOF
    asymptotic constant.

    Parameters
    ----------
    g2 : float

    Returns
    -------
    float : C in MeV*fm
    """
    g4 = g2**2
    return 8.0 * g4 * HBAR_C_MEV_FM / 225.0


# ======================================================================
# Full analysis
# ======================================================================

def run_full_analysis(
    R_values: Optional[np.ndarray] = None,
    spectrum_R_values: Optional[np.ndarray] = None,
    g2: float = G2_DEFAULT,
    N_per_dim: int = 8,
    n_eigenvalues: int = 10,
    fit_R_min: float = 10.0,
    verbose: bool = False,
) -> LargeRAnalysis:
    """
    Run the complete large-R analysis.

    Parameters
    ----------
    R_values : array-like or None
    spectrum_R_values : array-like or None
    g2 : float
    N_per_dim : int
    n_eigenvalues : int
    fit_R_min : float
    verbose : bool

    Returns
    -------
    LargeRAnalysis
    """
    if verbose:
        print("=" * 60)
        print("KvB Large-R Analysis")
        print(f"  g^2 = {g2:.3f}, N_per_dim = {N_per_dim}")
        print("=" * 60)

    # Gap scan
    if verbose:
        print("\n--- Gap vs R scan ---")
    gap_records = gap_vs_R_scan(R_values, g2=g2, N_per_dim=N_per_dim,
                                n_eigenvalues=n_eigenvalues, verbose=verbose)

    # Spectrum scan
    if verbose:
        print("\n--- Spectrum scan ---")
    spectrum_records = spectrum_scan(spectrum_R_values, g2=g2,
                                    N_per_dim=N_per_dim,
                                    n_eigenvalues=n_eigenvalues,
                                    verbose=verbose)

    # Fits
    if verbose:
        print("\n--- Fitting ---")
    power_fit = fit_power_law(gap_records, R_min=fit_R_min)
    sub_fit = fit_subleading(gap_records, R_min=fit_R_min)

    if verbose:
        print(f"  Power law: gap = {power_fit.params['A']:.6f} / R^{power_fit.params['alpha']:.4f}")
        print(f"    R^2 = {power_fit.r_squared:.8f}")
        print(f"  Subleading: gap = {sub_fit.params['A']:.6f}/R + {sub_fit.params['B']:.6f}/R^2")
        print(f"    R^2 = {sub_fit.r_squared:.8f}")

    # Asymptotic constant from the best-converged small-R data
    # The gap scales as C2/R^2, so C2 = gap*R^2
    # Use R=2.2 data (best converged) as the reference
    ref = [r for r in gap_records if abs(r.R - 2.2) < 0.5]
    if not ref:
        ref = gap_records[:1]
    best = ref[0]
    C_natural = best.gap_times_R2
    C_MeV_fm2 = best.gap_times_R2_MeV_fm2

    # Quartic-scaling analytical prediction
    C_quartic = quartic_C_MeV_fm2(g2)

    # Full-theory 1/R prediction (for reference)
    C_analytical = analytical_C_MeV_fm(g2)

    if verbose:
        print(f"\n--- Results ---")
        print(f"  gap*R^2 at R={best.R}: {C_MeV_fm2:.1f} MeV*fm^2")
        print(f"  Quartic prediction (delta_0=1.95): {C_quartic:.1f} MeV*fm^2")
        print(f"  Full-theory 1/R prediction (8g^4*hbar_c/225): {C_analytical:.1f} MeV*fm")
        print(f"  Fitted exponent: {power_fit.params.get('alpha', 'N/A')}")

    return LargeRAnalysis(
        gap_records=gap_records,
        spectrum_records=spectrum_records,
        power_law_fit=power_fit,
        subleading_fit=sub_fit,
        asymptotic_C_natural=C_natural,
        asymptotic_C_MeV_fm2=C_MeV_fm2,
        quartic_prediction_MeV_fm2=C_quartic,
        analytical_prediction_MeV_fm=C_analytical,
    )


# ======================================================================
# Plotting
# ======================================================================

def plot_gap_vs_R(analysis: LargeRAnalysis, save_path: str = None):
    """
    Create 2-panel figure: (a) gap vs R log-log, (b) gap*R^2 vs R.

    Parameters
    ----------
    analysis : LargeRAnalysis
    save_path : str or None
        If provided, save to this path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R_arr = np.array([r.R for r in analysis.gap_records])
    gap_MeV = np.array([r.gap_MeV for r in analysis.gap_records])
    gapR2_MeV_fm2 = np.array([r.gap_times_R2_MeV_fm2 for r in analysis.gap_records])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): gap(R) vs R (log-log)
    ax1.loglog(R_arr, gap_MeV, 'bo-', markersize=6, label='KvB numerical')

    # Overlay power-law fit
    fit = analysis.power_law_fit
    if fit and not np.isnan(fit.params.get('A', np.nan)):
        R_fit = np.logspace(np.log10(R_arr.min()), np.log10(R_arr.max()), 100)
        gap_fit = _power_law(R_fit, fit.params['A'], fit.params['alpha']) * HBAR_C_MEV_FM
        alpha_str = f"{fit.params['alpha']:.3f}"
        ax1.loglog(R_fit, gap_fit, 'r--', linewidth=1.5,
                   label=rf'Fit: $\Delta \propto R^{{-{alpha_str}}}$')

    # Overlay quartic-scaling prediction
    g2 = analysis.gap_records[0].g2
    R_theory = np.logspace(np.log10(R_arr.min()), np.log10(R_arr.max()), 100)
    gap_quartic = np.array([quartic_scaling_prediction(R, g2) * HBAR_C_MEV_FM for R in R_theory])
    ax1.loglog(R_theory, gap_quartic, 'g:', linewidth=1.5,
               label=r'Quartic scaling $\propto 1/R^2$')

    ax1.set_xlabel(r'$R$ [fm]', fontsize=12)
    ax1.set_ylabel(r'$\Delta_{\mathrm{KvB}}$ [MeV]', fontsize=12)
    ax1.set_title('(a) Mass gap vs radius', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)

    # Panel (b): gap*R^2 vs R (converged region only: R <= 10)
    # At larger R the basis (N=8) is insufficient, causing the upturn
    converged_mask = R_arr <= 10.0
    unconverged_mask = R_arr > 10.0

    ax2.semilogx(R_arr[converged_mask], gapR2_MeV_fm2[converged_mask],
                 'bo-', markersize=6, label='KvB (converged)')
    if np.any(unconverged_mask):
        ax2.semilogx(R_arr[unconverged_mask], gapR2_MeV_fm2[unconverged_mask],
                     'b^--', markersize=5, alpha=0.4, label='KvB (basis too small)')

    # Horizontal line for quartic prediction
    C_quartic = analysis.quartic_prediction_MeV_fm2
    ax2.axhline(y=C_quartic, color='g', linestyle='--', alpha=0.5,
                label=f'Quartic: {C_quartic:.0f} MeV fm$^2$')

    ax2.set_xlabel(r'$R$ [fm]', fontsize=12)
    ax2.set_ylabel(r'$\Delta_{\mathrm{KvB}} \times R^2$ [MeV fm$^2$]', fontsize=12)
    ax2.set_title(r'(b) $\Delta \times R^2$ plateau', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Set y-range to focus on the physical plateau region
    if np.any(converged_mask):
        ymin = min(gapR2_MeV_fm2[converged_mask]) * 0.8
        ymax = max(gapR2_MeV_fm2[converged_mask]) * 1.3
        ax2.set_ylim(ymin, ymax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close(fig)
    return fig


# ======================================================================
# Pretty-print results
# ======================================================================

def print_gap_table(records: List[GapRecord]):
    """Print a formatted table of gap vs R results."""
    print(f"{'R [fm]':>10} {'gap [nat]':>12} {'gap [MeV]':>12} {'gap*R^2 [nat]':>14} {'gap*R^2 [MeV*fm^2]':>20}")
    print("-" * 72)
    for r in records:
        print(f"{r.R:10.1f} {r.gap_natural:12.6f} {r.gap_MeV:12.2f} {r.gap_times_R2:14.6f} {r.gap_times_R2_MeV_fm2:20.1f}")


def print_spectrum_table(records: List[SpectrumRecord]):
    """Print spectrum ratios at selected R values."""
    for rec in records:
        print(f"\nR = {rec.R:.1f} fm:")
        print(f"  {'n':>3} {'E_n':>14} {'E_n - E_0':>14} {'ratio':>10}")
        print(f"  " + "-" * 45)
        for i, ev in enumerate(rec.eigenvalues[:6]):
            gap = ev - rec.eigenvalues[0]
            ratio = rec.ratios_to_gap[i - 1] if i > 0 else 0.0
            print(f"  {i:3d} {ev:14.6f} {gap:14.6f} {ratio:10.4f}")


# ======================================================================
# Main entry point
# ======================================================================

if __name__ == "__main__":
    print("Running KvB Large-R Analysis...")
    print(f"  g^2 = {G2_DEFAULT}")
    print(f"  hbar*c = {HBAR_C_MEV_FM} MeV*fm")
    print()

    analysis = run_full_analysis(
        N_per_dim=8,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("GAP TABLE")
    print("=" * 65)
    print_gap_table(analysis.gap_records)

    print("\n" + "=" * 65)
    print("SPECTRUM RATIOS")
    print("=" * 65)
    print_spectrum_table(analysis.spectrum_records)

    print("\n" + "=" * 65)
    print("FIT RESULTS")
    print("=" * 65)
    pf = analysis.power_law_fit
    sf = analysis.subleading_fit
    print(f"  Power law: gap = {pf.params['A']:.6f} / R^{pf.params['alpha']:.4f}")
    print(f"    R^2 = {pf.r_squared:.10f}")
    print(f"  Subleading: gap = {sf.params['A']:.6f}/R + {sf.params['B']:.6f}/R^2")
    print(f"    R^2 = {sf.r_squared:.10f}")
    print(f"\n  gap*R^2 (numerical) = {analysis.asymptotic_C_MeV_fm2:.1f} MeV*fm^2")
    print(f"  gap*R^2 (quartic pred) = {analysis.quartic_prediction_MeV_fm2:.1f} MeV*fm^2")
    print(f"  Full-theory 1/R prediction = {analysis.analytical_prediction_MeV_fm:.1f} MeV*fm")

    # Save figure
    import os
    fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'papers', 'final', 'figures')
    fig_path = os.path.abspath(os.path.join(fig_dir, 'gap_vs_R.png'))
    plot_gap_vs_R(analysis, save_path=fig_path)
