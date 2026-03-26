"""
Beta Function — Perturbative running of the Yang-Mills coupling.

The 1-loop beta function for pure SU(N) Yang-Mills theory is:

    β(g) = -b₀ g³  where  b₀ = 11N / (48π²)

This gives asymptotic freedom: the coupling decreases at high energies.

On S³, the beta function receives finite-volume corrections that
go as O(1/(Rμ)²), but at leading order it matches flat space.
This is a crucial consistency check: our S³ framework MUST reproduce
the standard perturbative result in the large-R limit.

References:
    - Gross & Wilczek (1973), Politzer (1973): discovery of asymptotic freedom
    - Lüscher (1982): finite-volume effects in gauge theories
"""

import numpy as np


class BetaFunction:
    """
    Perturbative beta function for pure SU(N) Yang-Mills theory,
    including S³ finite-volume considerations.
    """

    # ------------------------------------------------------------------
    # 1-loop coefficient (flat space)
    # ------------------------------------------------------------------
    @staticmethod
    def one_loop_flat(N: int) -> float:
        """
        1-loop beta function coefficient b₀ for pure SU(N) YM in flat space.

        β(g) = -b₀ g³

        b₀ = 11N / (48π²)

        For SU(3): b₀ = 33/(48π²) ≈ 0.06948

        Parameters
        ----------
        N : int, number of colors (N >= 2)

        Returns
        -------
        float : b₀ (positive for asymptotic freedom)
        """
        if N < 2:
            raise ValueError(f"N must be >= 2, got {N}")
        return 11 * N / (48 * np.pi**2)

    # ------------------------------------------------------------------
    # 1-loop coefficient on S³
    # ------------------------------------------------------------------
    @staticmethod
    def one_loop_s3(N: int, R: float) -> dict:
        """
        1-loop beta function coefficient on S³ of radius R.

        At leading order, this matches the flat-space result.
        Finite-volume corrections go as O(1/(Rμ)²) and are
        suppressed when Rμ >> 1.

        The S³ computation uses the heat kernel on S³, which gives:

            b₀(S³) = b₀(flat) × [1 + c/(Rμ)² + ...]

        where c is a numerical constant from the S³ heat kernel expansion.
        At 1-loop order, the leading term is universal and MUST match
        flat space — this is a non-trivial check that our framework
        is consistent with asymptotic freedom.

        Parameters
        ----------
        N : int, number of colors
        R : float, radius of S³ (in natural units or fm)

        Returns
        -------
        dict with:
            'b0_flat'           : flat-space 1-loop coefficient
            'b0_leading'        : leading S³ coefficient (= flat)
            'finite_volume_note': description of corrections
        """
        b0 = BetaFunction.one_loop_flat(N)

        return {
            'b0_flat': b0,
            'b0_leading': b0,  # Matches flat space at leading order
            'finite_volume_note': (
                f"On S³(R={R}), the 1-loop coefficient matches flat space "
                f"at leading order: b₀ = {b0:.6f}. "
                f"Finite-volume corrections are O(1/(Rμ)²) and are "
                f"suppressed when the energy scale μ >> 1/R. "
                f"At the scale μ = 1/R (where the gap lives), these "
                f"corrections are O(1) and perturbation theory is "
                f"not reliable."
            ),
        }

    # ------------------------------------------------------------------
    # Running coupling (1-loop)
    # ------------------------------------------------------------------
    @staticmethod
    def running_coupling(N: int, mu: float, Lambda: float) -> float:
        """
        1-loop running coupling g²(μ) for pure SU(N) YM.

        g²(μ) = 1 / (b₀ × ln(μ²/Λ²))

        This is valid only for μ >> Λ (perturbative regime).

        Parameters
        ----------
        N      : int, number of colors
        mu     : float, energy scale in MeV
        Lambda : float, Λ_QCD in MeV

        Returns
        -------
        float : g²(μ) (dimensionless)

        Raises
        ------
        ValueError : if μ <= Λ (non-perturbative regime)
        """
        if mu <= Lambda:
            raise ValueError(
                f"μ = {mu} MeV <= Λ = {Lambda} MeV: "
                f"outside perturbative regime"
            )

        b0 = BetaFunction.one_loop_flat(N)
        log_ratio = np.log(mu**2 / Lambda**2)

        if log_ratio <= 0:
            raise ValueError("ln(μ²/Λ²) <= 0: Landau pole region")

        return 1.0 / (b0 * log_ratio)

    # ------------------------------------------------------------------
    # α_s = g²/(4π)
    # ------------------------------------------------------------------
    @staticmethod
    def alpha_s(N: int, mu: float, Lambda: float) -> float:
        """
        Strong coupling constant α_s(μ) = g²(μ) / (4π).

        Parameters
        ----------
        N      : int, number of colors
        mu     : float, energy scale in MeV
        Lambda : float, Λ_QCD in MeV

        Returns
        -------
        float : α_s(μ)
        """
        g2 = BetaFunction.running_coupling(N, mu, Lambda)
        return g2 / (4 * np.pi)

    # ------------------------------------------------------------------
    # Coupling at the gap scale
    # ------------------------------------------------------------------
    @staticmethod
    def coupling_at_gap(N: int, R_fm: float, Lambda_MeV: float = 200.0) -> dict:
        """
        Evaluate the coupling at the scale μ = ℏc/R, which is
        the natural scale where the S³ mass gap lives.

        This tells us whether perturbation theory is valid at
        the gap scale.

        Parameters
        ----------
        N          : int, number of colors
        R_fm       : float, S³ radius in fm
        Lambda_MeV : float, Λ_QCD in MeV (default 200)

        Returns
        -------
        dict with:
            'mu_MeV'             : scale μ = ℏc/R in MeV
            'g_squared'          : g²(μ) if perturbative, else None
            'alpha_s'            : α_s(μ) if perturbative, else None
            'is_perturbative'    : bool, True if μ > Λ
            'note'               : str, assessment of validity
        """
        HBAR_C = 197.3269804  # MeV·fm
        mu = HBAR_C / R_fm

        result = {
            'mu_MeV': mu,
            'g_squared': None,
            'alpha_s': None,
            'is_perturbative': mu > Lambda_MeV,
            'note': '',
        }

        if mu > Lambda_MeV:
            g2 = BetaFunction.running_coupling(N, mu, Lambda_MeV)
            a_s = g2 / (4 * np.pi)
            result['g_squared'] = g2
            result['alpha_s'] = a_s

            if a_s < 0.3:
                result['note'] = (
                    f"At μ = {mu:.1f} MeV (R = {R_fm} fm), "
                    f"α_s = {a_s:.4f}. Perturbation theory is "
                    f"marginally valid."
                )
            else:
                result['note'] = (
                    f"At μ = {mu:.1f} MeV (R = {R_fm} fm), "
                    f"α_s = {a_s:.4f}. This is large — perturbation "
                    f"theory is NOT reliable at this scale."
                )
        else:
            result['note'] = (
                f"At μ = {mu:.1f} MeV (R = {R_fm} fm), "
                f"μ < Λ_QCD = {Lambda_MeV} MeV. "
                f"This is deep in the non-perturbative regime. "
                f"The 1-loop formula is not valid here."
            )

        return result

    # ------------------------------------------------------------------
    # Verify asymptotic freedom
    # ------------------------------------------------------------------
    @staticmethod
    def verify_asymptotic_freedom(N: int, Lambda_MeV: float = 200.0) -> dict:
        """
        Verify that the coupling decreases at high energies
        (asymptotic freedom).

        Parameters
        ----------
        N          : int, number of colors
        Lambda_MeV : float, Λ_QCD in MeV

        Returns
        -------
        dict with coupling at several scales
        """
        scales = [500, 1000, 2000, 5000, 10000, 91200]  # MeV (last = M_Z)
        labels = ['500 MeV', '1 GeV', '2 GeV', '5 GeV', '10 GeV', 'M_Z']

        entries = []
        for mu, label in zip(scales, labels):
            g2 = BetaFunction.running_coupling(N, mu, Lambda_MeV)
            a_s = g2 / (4 * np.pi)
            entries.append({
                'scale': label,
                'mu_MeV': mu,
                'g_squared': g2,
                'alpha_s': a_s,
            })

        # Check monotonically decreasing
        alphas = [e['alpha_s'] for e in entries]
        is_af = all(alphas[i] > alphas[i + 1] for i in range(len(alphas) - 1))

        return {
            'entries': entries,
            'asymptotic_freedom': is_af,
            'b0': BetaFunction.one_loop_flat(N),
        }
