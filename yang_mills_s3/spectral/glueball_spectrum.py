"""
Glueball Spectrum — Eigenvalues of the linearized YM Laplacian on S^3
mapped to glueball masses and compared with lattice QCD predictions.

The linearized YM operator on S^3 of radius R has COEXACT eigenvalues:

    lambda_k = (k+1)^2 / R^2    for k = 1, 2, 3, ...
    Values on unit S^3: 4, 9, 16, 25, 36, ...

The physical mass associated with the k-th mode is:

    m_k = hbar*c x sqrt(lambda_k) = hbar*c x (k+1) / R

Lattice QCD glueball masses (Morningstar & Peardon 1999, Chen et al. 2006):
    0++ ~ 1730 MeV   (scalar)
    2++ ~ 2400 MeV   (tensor)
    0-+ ~ 2590 MeV   (pseudoscalar)

IMPORTANT NOTE ON GLUEBALL MASSES:
Glueballs are COMPOSITE (bound) states, not single eigenmodes of the
linearized operator. The single-particle spectrum gives the EXCITATION
threshold (~ Lambda_QCD ~ 200 MeV), while glueball masses (~ 1.7 GeV)
involve strong-coupling bound-state dynamics. Mass RATIOS from the
linearized spectrum are a rough guide, but the l -> J^PC assignment
is speculative and not rigorous.

CORRECTION (2026-03-10): Eigenvalues changed from (l(l+2)+2)/R^2 to
the correct coexact spectrum (k+1)^2/R^2. Mass ratios change accordingly.
"""

import numpy as np


# Physical constants
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm


class GlueballSpectrum:
    """
    Glueball mass predictions from the linearized YM Laplacian on S^3.

    The spectrum is that of the FREE (linearized) operator — interactions
    are not included. Mass ratios are a more robust prediction than
    absolute masses, since the overall scale depends on R.

    Uses the COEXACT (physical) eigenvalues: (k+1)^2/R^2 for k=1,2,3,...
    """

    # ------------------------------------------------------------------
    # Eigenvalue at quantum number k
    # ------------------------------------------------------------------
    @staticmethod
    def eigenvalue_at_l(l: int, R: float) -> float:
        """
        The l-th coexact eigenvalue of the 1-form Hodge Laplacian on S^3(R).

        lambda_k = (k+1)^2 / R^2

        Here l is used as the quantum number k (kept as 'l' for API
        compatibility).

        Parameters
        ----------
        l : int, quantum number (l >= 1)
        R : float, radius of S^3

        Returns
        -------
        float : eigenvalue in units of 1/R^2
        """
        if l < 1:
            raise ValueError(f"l must be >= 1 for 1-forms on S^3, got l={l}")
        return (l + 1) ** 2 / R**2

    # ------------------------------------------------------------------
    # Physical mass at level l
    # ------------------------------------------------------------------
    @staticmethod
    def mass_at_l(l: int, R_fm: float) -> float:
        """
        Physical mass in MeV for the l-th coexact eigenmode.

        m_l = hbar*c x sqrt(lambda_l) = hbar*c x (l+1) / R_fm

        Parameters
        ----------
        l    : int, quantum number (l >= 1)
        R_fm : float, radius in femtometers

        Returns
        -------
        float : mass in MeV
        """
        if l < 1:
            raise ValueError(f"l must be >= 1, got l={l}")
        return HBAR_C_MEV_FM * (l + 1) / R_fm

    # ------------------------------------------------------------------
    # Full spectrum table
    # ------------------------------------------------------------------
    @staticmethod
    def spectrum_table(R_fm: float, l_max: int = 10) -> list[dict]:
        """
        Table of glueball masses for l = 1 to l_max.

        Each entry includes:
            l, eigenvalue, mass_MeV, mass_GeV, ratio_to_ground,
            jpc_assignment (speculative)

        The J^PC assignment is speculative because our modes are
        labeled by k, not by spin-parity. A rough mapping:
            k=1 -> 0++ (scalar, ground state)
            k=2 -> 2++ (tensor)
            k=3 -> 0-+ (pseudoscalar) or 3++
            Higher k -> higher excitations

        NOTE: Glueballs are COMPOSITE states. The linearized spectrum
        gives the single-particle excitation threshold, not bound-state
        masses. The J^PC assignment here is a rough hypothesis.

        Parameters
        ----------
        R_fm  : float, radius in fm
        l_max : int, maximum l value

        Returns
        -------
        list of dicts with spectrum data
        """
        # Speculative J^PC assignments
        jpc_map = {
            1: '0++',
            2: '2++',
            3: '0-+',
            4: '3++',
            5: '2-+',
        }

        m_ground = GlueballSpectrum.mass_at_l(1, R_fm)

        table = []
        for l in range(1, l_max + 1):
            mass = GlueballSpectrum.mass_at_l(l, R_fm)
            ev = GlueballSpectrum.eigenvalue_at_l(l, R_fm)
            table.append({
                'l': l,
                'eigenvalue': ev,
                'mass_MeV': mass,
                'mass_GeV': mass / 1000.0,
                'ratio_to_ground': mass / m_ground,
                'jpc': jpc_map.get(l, f'{l}??'),
            })

        return table

    # ------------------------------------------------------------------
    # Glueball mass ratios vs lattice
    # ------------------------------------------------------------------
    @staticmethod
    def glueball_ratios(R_fm: float) -> dict:
        """
        Mass ratios m_l / m_1 compared with lattice QCD results.

        Our prediction (R-independent):
            m_l / m_1 = (l+1) / 2

        since m_l = hbar*c*(l+1)/R and m_1 = hbar*c*2/R.

        Lattice ratios (Morningstar & Peardon 1999):
            m(2++) / m(0++) ~ 1.39
            m(0-+) / m(0++) ~ 1.50

        Parameters
        ----------
        R_fm : float, radius (ratios are R-independent)

        Returns
        -------
        dict with:
            'our_ratios'     : list of (l, ratio) for l=1..5
            'lattice_ratios' : dict of known lattice ratios
            'comparison'     : list of dicts comparing our vs lattice
        """
        # Our ratios (R-independent): m_l/m_1 = (l+1)/2
        our_ratios = []
        for l in range(1, 6):
            ratio = (l + 1) / 2.0
            our_ratios.append((l, ratio))

        # Lattice QCD ratios
        lattice = {
            '2++/0++': 1.39,   # Morningstar & Peardon 1999
            '0-+/0++': 1.50,   # Morningstar & Peardon 1999
            '1+-/0++': 1.74,   # Chen et al. 2006
        }

        # Compare assuming l=1->0++, l=2->2++, l=3->0-+
        comparison = []

        # l=2 vs l=1 -> compare with 2++/0++
        our_21 = 3.0 / 2.0  # (2+1)/2 = 1.5
        comparison.append({
            'assignment': '2++/0++ (l=2/l=1)',
            'our_ratio': our_21,
            'lattice_ratio': 1.39,
            'discrepancy_pct': abs(our_21 - 1.39) / 1.39 * 100,
        })

        # l=3 vs l=1 -> compare with 0-+/0++
        our_31 = 4.0 / 2.0  # (3+1)/2 = 2.0
        comparison.append({
            'assignment': '0-+/0++ (l=3/l=1)',
            'our_ratio': our_31,
            'lattice_ratio': 1.50,
            'discrepancy_pct': abs(our_31 - 1.50) / 1.50 * 100,
        })

        return {
            'our_ratios': our_ratios,
            'lattice_ratios': lattice,
            'comparison': comparison,
        }

    # ------------------------------------------------------------------
    # Best-fit radius from glueball mass
    # ------------------------------------------------------------------
    @staticmethod
    def best_fit_R(target_mass_0pp_MeV: float = 1730.0) -> dict:
        """
        Find the S^3 radius R that makes the ground state mass (l=1)
        match the lattice 0++ glueball mass.

        m_1 = hbar*c x 2 / R  =>  R = hbar*c x 2 / m_1

        Parameters
        ----------
        target_mass_0pp_MeV : float, target 0++ mass in MeV (default 1730)

        Returns
        -------
        dict with:
            'R_fm'              : best-fit radius in fm
            'target_mass_MeV'   : the target mass used
            'achieved_mass_MeV' : mass at best-fit R (should equal target)
            'R_from_gap'        : R from mass gap = 200 MeV (= 2.0 fm)
            'tension'           : ratio of the two R values
            'note'              : explanation of any discrepancy
        """
        # R from glueball mass: m = 2*hbar_c/R => R = 2*hbar_c/m
        R_glueball = 2.0 * HBAR_C_MEV_FM / target_mass_0pp_MeV

        # R from mass gap identification (gap ~ Lambda_QCD ~ 200 MeV)
        R_gap = 2.0 * HBAR_C_MEV_FM / 200.0

        # Verify
        achieved = GlueballSpectrum.mass_at_l(1, R_glueball)

        ratio = R_gap / R_glueball

        note = (
            f"R from glueball 0++ = {R_glueball:.4f} fm vs "
            f"R from gap ~ Lambda_QCD = {R_gap:.4f} fm. "
            f"Ratio = {ratio:.2f}. "
        )
        if abs(ratio - 1.0) < 0.1:
            note += "These are consistent (within 10%)."
        else:
            note += (
                "TENSION: these differ significantly. "
                "The gap eigenvalue 2/R is NOT the same as the 0++ glueball mass. "
                "The gap (~200 MeV) represents the lowest excitation of the "
                "linearized operator, while the glueball mass (~1730 MeV) is a "
                "bound state in the full interacting theory. The ratio ~8.7 "
                "reflects the ratio between the free-field gap and the physical "
                "glueball mass, which includes strong-coupling effects."
            )

        return {
            'R_fm': R_glueball,
            'target_mass_MeV': target_mass_0pp_MeV,
            'achieved_mass_MeV': achieved,
            'R_from_gap': R_gap,
            'tension': ratio,
            'note': note,
        }

    # ------------------------------------------------------------------
    # Eigenvalue ratio (R-independent)
    # ------------------------------------------------------------------
    @staticmethod
    def eigenvalue_ratio(l1: int, l2: int) -> float:
        """
        Ratio of masses m_{l2}/m_{l1}.

        This is R-independent:
            m_{l2}/m_{l1} = (l2+1)/(l1+1)

        (since m_l = hbar*c*(l+1)/R)

        Parameters
        ----------
        l1, l2 : int, quantum numbers

        Returns
        -------
        float : mass ratio
        """
        if l1 < 1 or l2 < 1:
            raise ValueError("l values must be >= 1")
        return (l2 + 1) / (l1 + 1)
