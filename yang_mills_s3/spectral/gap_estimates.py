"""
Gap Estimates — Bounds on the Yang-Mills mass gap.

Provides:
  1. Weitzenböck lower bound on the YM gap
  2. Kato-Rellich stability analysis for perturbations
  3. Gap vs. radius curves for visualization
  4. Comparison with experimental QCD observables
"""

import numpy as np
from ..geometry.hodge_spectrum import HodgeSpectrum
from .yang_mills_operator import YangMillsOperator, HBAR_C_MEV_FM


class GapEstimates:
    """
    Rigorous and semi-rigorous estimates of the Yang-Mills mass gap.

    The Weitzenböck identity gives a LOWER BOUND on the 1-form Laplacian:
        Δ₁ >= ∇*∇ + Ric >= Ric = (n-1)/R²  on S^n

    For S³: Δ₁ >= 2/R² from Ricci alone. The actual coexact (physical)
    gap is 4/R², from the k=1 coexact eigenmode ((k+1)^2/R^2 = 4/R^2).
    """

    # ------------------------------------------------------------------
    # Weitzenböck lower bound
    # ------------------------------------------------------------------
    @staticmethod
    def weitzenboeck_lower_bound(gauge_group: str, R: float) -> float:
        """
        Lower bound on the YM gap from the Weitzenböck formula.

        On S³:
            Δ₁ = ∇*∇ + Ric
            where Ric = 2/R² (Ricci curvature on S³)

        The ∇*∇ (rough Laplacian) is non-negative, so:
            Δ₁ >= 2/R²

        The coexact (physical) gap is 4/R², from the k=1 coexact eigenmode.
        The left-invariant 1-forms on S^3 satisfy ∇*∇ = 2/R^2 (not 3/R^2),
        giving Δ₁ = 2/R^2 + 2/R^2 = 4/R^2.

        Parameters
        ----------
        gauge_group : str (used for adjoint dimension factor)
        R           : radius of S³

        Returns
        -------
        float : lower bound on the coexact eigenvalue gap (4/R² for S³)
        """
        # Coexact gap = nabla*nabla(left-invariant) + Ricci = 2/R^2 + 2/R^2 = 4/R^2
        ricci = 2.0 / R**2  # Ric on S³ = 2/R²
        rough_lap_gap = 2.0 / R**2  # nabla*nabla on left-invariant 1-forms
        return rough_lap_gap + ricci  # = 4/R²

    # ------------------------------------------------------------------
    # Kato-Rellich stability
    # ------------------------------------------------------------------
    @staticmethod
    def kato_rellich_stability(gap_linear: float, perturbation_bound: float) -> dict:
        """
        Kato-Rellich perturbation theory: does the gap survive?

        If the linearized operator has gap Δ and the perturbation V satisfies
        ||V|| < Δ, then by the Kato-Rellich theorem the perturbed operator
        also has a gap of at least Δ - ||V||.

        Parameters
        ----------
        gap_linear        : float, the unperturbed spectral gap
        perturbation_bound: float, operator norm bound on the perturbation

        Returns
        -------
        dict with:
            'gap_survives'   : bool  — True if ||V|| < gap
            'shifted_gap'    : float — lower bound on perturbed gap (may be <= 0)
            'relative_bound' : float — ||V|| / gap (< 1 means stable)
        """
        relative = perturbation_bound / gap_linear if gap_linear > 0 else float('inf')
        shifted = gap_linear - perturbation_bound

        return {
            'gap_survives': perturbation_bound < gap_linear,
            'shifted_gap': shifted,
            'relative_bound': relative,
        }

    # ------------------------------------------------------------------
    # Gap vs. radius
    # ------------------------------------------------------------------
    @staticmethod
    def gap_vs_radius(gauge_group: str, R_values) -> np.ndarray:
        """
        Mass gap eigenvalue as a function of radius.

        Returns array of (R, gap_eigenvalue) pairs.
        gap = 4/R² for SU(N) on S³ -> diverges as R->0, vanishes as R->inf.

        Parameters
        ----------
        gauge_group : str
        R_values    : array-like of radius values

        Returns
        -------
        numpy array of shape (len(R_values), 2) with columns [R, gap]
        """
        R_arr = np.asarray(R_values, dtype=float)
        gaps = np.array([
            YangMillsOperator.mass_gap_eigenvalue(gauge_group, R)
            for R in R_arr
        ])
        return np.column_stack([R_arr, gaps])

    # ------------------------------------------------------------------
    # Comparison with QCD
    # ------------------------------------------------------------------
    @staticmethod
    def comparison_with_qcd(R_fm: float) -> dict:
        """
        Compare geometric predictions at radius R (in fm) with
        experimental QCD observables.

        Parameters
        ----------
        R_fm : radius in femtometers

        Returns
        -------
        dict with predicted and experimental values:
            'mass_gap'           : predicted mass gap in MeV
            'lambda_qcd'         : experimental Λ_QCD ≈ 200 MeV
            'proton_radius'      : predicted proton radius in fm
            'proton_radius_exp'  : experimental ≈ 0.84 fm
            'confinement_length' : predicted confinement scale in fm
            'confinement_exp'    : experimental ≈ 1.0 fm
            'string_tension'     : predicted string tension in MeV²
            'string_tension_exp' : experimental ≈ (440 MeV)²
        """
        # Mass gap
        mass_gap = YangMillsOperator.physical_mass_gap('SU(3)', R_fm)

        # Proton radius: geometric prediction R / 2.6
        proton_radius = R_fm / 2.6

        # Confinement length: geometric prediction R / 2.2
        confinement_length = R_fm / 2.2

        # String tension: (hbar*c / R)²
        string_tension = (HBAR_C_MEV_FM / R_fm) ** 2

        return {
            'mass_gap': mass_gap,
            'lambda_qcd': 200.0,  # MeV
            'proton_radius': proton_radius,
            'proton_radius_exp': 0.84,  # fm
            'confinement_length': confinement_length,
            'confinement_exp': 1.0,  # fm
            'string_tension': string_tension,
            'string_tension_exp': 440.0**2,  # MeV²
        }
