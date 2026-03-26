"""
Yang-Mills Operator — Linearized YM spectrum and mass gap computation.

The linearized Yang-Mills operator around the Maurer-Cartan vacuum on S^3
is the 1-form Laplacian tensored with the adjoint representation:

    Delta_YM = Delta_1 (x) 1_{adj}

Its eigenvalues are those of Delta_1 on S^3, each with multiplicity multiplied
by dim(adj(G)).

Key results:
    SU(2) on S^3:
        - Delta_1 coexact gap = 4/R^2  (from k=1 coexact mode)
        - dim(adj(SU(2))) = 3
        - Mass gap = 2/R  in natural units
        - Total multiplicity of gap mode = (coexact Hodge mult) x 3

CORRECTION (2026-03-10): Gap eigenvalue changed from 5/R^2 to 4/R^2.
The physical modes are COEXACT (divergence-free) 1-forms. Exact 1-forms
(df) are pure gauge and unphysical. The coexact eigenvalues on S^3 are
(k+1)^2/R^2, giving a gap of 4/R^2 at k=1.
"""

import numpy as np
from ..geometry.hodge_spectrum import HodgeSpectrum


# Physical constants
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm


class YangMillsOperator:
    """
    Linearized Yang-Mills operator on S^3 and its spectrum.

    The operator acts on adjoint-valued 1-forms:
        Delta_YM : Omega^1(S^3, ad P) -> Omega^1(S^3, ad P)

    Around the Maurer-Cartan connection (flat vacuum), this reduces to
    the Hodge Laplacian on 1-forms tensored with the identity on the
    adjoint bundle.

    In Coulomb gauge, only the COEXACT (transverse, divergence-free)
    modes are physical. The coexact spectrum on S^3 is:
        (k+1)^2/R^2 for k = 1, 2, 3, ...
    with the mass gap at k=1 giving eigenvalue 4/R^2.
    """

    # ------------------------------------------------------------------
    # Gauge group data
    # ------------------------------------------------------------------
    @staticmethod
    def adjoint_dimension(gauge_group: str) -> int:
        """
        Dimension of the adjoint representation of the gauge group.

        Parameters
        ----------
        gauge_group : str, e.g. 'SU(2)', 'SU(3)', 'SU(N)'

        Returns
        -------
        int : dim(adj(G))
        """
        group = gauge_group.strip().upper().replace(' ', '')

        if group.startswith('SU(') and group.endswith(')'):
            N = int(group[3:-1])
            return N**2 - 1
        elif group.startswith('SO(') and group.endswith(')'):
            N = int(group[3:-1])
            return N * (N - 1) // 2
        elif group.startswith('SP(') and group.endswith(')'):
            N = int(group[3:-1])
            return N * (2 * N + 1)
        elif group in ('G2', 'G(2)'):
            return 14
        elif group in ('E6', 'E(6)'):
            return 78
        elif group in ('E7', 'E(7)'):
            return 133
        elif group in ('E8', 'E(8)'):
            return 248
        else:
            raise ValueError(f"Unknown gauge group: {gauge_group}")

    # ------------------------------------------------------------------
    # Linearized spectrum
    # ------------------------------------------------------------------
    @staticmethod
    def linearized_spectrum(gauge_group: str, R: float, l_max: int = 20):
        """
        Spectrum of the linearized YM operator around the Maurer-Cartan
        vacuum on S^3 of radius R.

        Returns the COEXACT (physical) spectrum. Each coexact eigenvalue
        lambda of Delta_1 on S^3 appears with multiplicity
        (coexact Hodge multiplicity) x dim(adj(G)).

        Parameters
        ----------
        gauge_group : str, e.g. 'SU(2)', 'SU(3)'
        R           : radius of S^3
        l_max       : max quantum number

        Returns
        -------
        list of (eigenvalue, total_multiplicity) tuples
        """
        dim_adj = YangMillsOperator.adjoint_dimension(gauge_group)
        hodge_spectrum = HodgeSpectrum.one_form_eigenvalues(
            3, R, l_max, mode='coexact')

        result = []
        for ev, hodge_mult in hodge_spectrum:
            total_mult = hodge_mult * dim_adj
            result.append((ev, total_mult))

        return result

    # ------------------------------------------------------------------
    # Mass gap (eigenvalue)
    # ------------------------------------------------------------------
    @staticmethod
    def mass_gap(gauge_group: str, R: float) -> float:
        """
        The Yang-Mills mass gap on S^3 of radius R in natural units (1/R).

        For SU(N) on S^3:
            coexact gap eigenvalue = 4/R^2
            mass gap = sqrt(4/R^2) = 2/R

        Parameters
        ----------
        gauge_group : str
        R           : radius of S^3

        Returns
        -------
        float : mass gap = sqrt(first coexact eigenvalue of Delta_YM)
        """
        gap_eigenvalue = HodgeSpectrum.first_nonzero_eigenvalue(
            3, 1, R, mode='coexact')
        return np.sqrt(gap_eigenvalue)

    # ------------------------------------------------------------------
    # Mass gap eigenvalue (Delta gap, not sqrt)
    # ------------------------------------------------------------------
    @staticmethod
    def mass_gap_eigenvalue(gauge_group: str, R: float) -> float:
        """
        The mass gap eigenvalue (not the mass itself).

        For SU(N) on S^3: 4/R^2  (coexact gap)

        Returns
        -------
        float : first coexact eigenvalue of linearized YM operator
        """
        return HodgeSpectrum.first_nonzero_eigenvalue(
            3, 1, R, mode='coexact')

    # ------------------------------------------------------------------
    # Multiplicity of the lowest mode
    # ------------------------------------------------------------------
    @staticmethod
    def multiplicity_lowest_mode(gauge_group: str) -> int:
        """
        Multiplicity of the mass gap mode.

        = (coexact Hodge multiplicity of first Delta_1 eigenvalue) x dim(adj(G))

        For SU(2): coexact k=1 mode has mult = 2*1*(1+2) = 6.
        Wait — that uses the old formula. Let me recalculate:
        k=1 coexact: mult = 2*k*(k+2) = 2*1*3 = 6.
        dim(adj(SU(2))) = 3.
        Total = 6 x 3 = 18.

        But physically, only the transverse (coexact) modes matter.
        The coexact multiplicity at k=1 is 2*1*3 = 6.

        Actually, for the LEFT-INVARIANT 1-forms (the k=1 coexact
        eigenmodes on S^3), there are exactly 3 independent forms
        (theta^1, theta^2, theta^3). These correspond to the 3
        generators of SU(2). The multiplicity 6 counts both
        +curl and -curl eigenvalues, but both give eigenvalue 4
        under Delta_1 = curl^2. So the physical multiplicity is 6.

        Total = 6 (coexact Hodge) x 3 (adjoint) = 18.

        BUT: if we think of the physical degrees of freedom in Coulomb
        gauge, the 3 left-invariant 1-forms give the k=1 mode.
        With 3 adjoint colors: 3 x 3 = 9 modes (matching original code
        for transverse modes only). This discrepancy comes from how we
        count +/- curl eigenmodes.

        The left-invariant forms span a 3-dim space. These are the
        SELF-DUAL coexact eigenmodes. The ANTI-SELF-DUAL ones give
        another 3. Together: 6 = 2*1*3. Both types are physical.

        Parameters
        ----------
        gauge_group : str

        Returns
        -------
        int : total multiplicity of mass gap mode
        """
        dim_adj = YangMillsOperator.adjoint_dimension(gauge_group)
        # Coexact multiplicity at k=1: 2*k*(k+2) = 6
        k = 1
        coexact_mult = 2 * k * (k + 2)  # = 6
        return coexact_mult * dim_adj

    # ------------------------------------------------------------------
    # Physical mass gap in MeV
    # ------------------------------------------------------------------
    @staticmethod
    def physical_mass_gap(gauge_group: str, R_fm: float) -> float:
        """
        Mass gap in MeV for S^3 of radius R_fm (in femtometers).

        m = hbar*c x sqrt(gap_eigenvalue)
          = hbar*c x 2/R     [for any gauge group on S^3]

        Using hbar*c = 197.3269804 MeV*fm:
            m = 197.327 x 2 / R_fm   MeV

        Parameters
        ----------
        gauge_group : str
        R_fm        : radius in femtometers

        Returns
        -------
        float : mass gap in MeV
        """
        gap = YangMillsOperator.mass_gap(gauge_group, R_fm)
        return HBAR_C_MEV_FM * gap
