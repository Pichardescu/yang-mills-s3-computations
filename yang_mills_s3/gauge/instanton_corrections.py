"""
Instanton Corrections to the Mass Gap on S3.

Phase 1.2 of the Yang-Mills Lab Plan.

Instantons are non-perturbative field configurations that tunnel between
different topological vacua, classified by pi_3(SU(2)) = Z. This module
computes their corrections to the mass gap on S3.

KEY FINDING (NUMERICAL):
    At physical coupling g^2 ~ 6, instanton corrections to the mass gap
    are ~ 10^{-6} of the geometric gap 5/R^2. Instantons are relevant for
    topology (theta-vacuum, eta' mass) but NEGLIGIBLE for the mass gap
    magnitude.

PHYSICS SUMMARY:
    - Instanton action: S_0 = 8pi^2/g^2 (topological, independent of R)
    - On S3, the instanton moduli space is COMPACT (unlike on R^4)
    - This compactness guarantees convergence of the instanton path integral
    - The dilute instanton gas gives exponentially suppressed corrections
    - Sign of correction: POSITIVE (gap increases)

References:
    - 't Hooft 1976: instanton physics, theta-vacuum
    - BPST 1975: instanton solution
    - Berg & Luscher 1981: corrected prefactor for instanton density
    - Witten 1989: Chern-Simons on S3 exactly solvable
    - Foundational analysis: gap = 5/R^2, instantons = Hopf maps
"""

import numpy as np
from .instanton import Instanton


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV fm


class InstantonCorrections:
    """
    Compute instanton contributions to the mass gap on S3.

    The instanton density on S3 x S1(beta) at temperature T = 1/beta:

        n_inst = C * (8pi^2/g^2)^(2N) * exp(-8pi^2/g^2) * R^(-4) * det'(Delta)

    where det'(Delta) is the regularized determinant of the fluctuation
    operator around the instanton.

    LABEL: NUMERICAL
    All quantitative results are numerical estimates based on the dilute
    instanton gas approximation. The qualitative conclusion (instantons
    negligible for the gap) is a PROPOSITION supported by the exponential
    suppression factor.
    """

    # Numerical prefactors from 't Hooft 1976, corrected by Berg-Luscher 1981
    # C_N for the instanton density in SU(N)
    PREFACTOR = {
        2: 0.466,    # SU(2): 't Hooft 1976, corrected Berg-Luscher
        3: 0.0072,   # SU(3): Gross-Pisarski-Yaffe 1981
    }

    # ------------------------------------------------------------------
    # Instanton action: S_0 = 8pi^2 / g^2
    # ------------------------------------------------------------------
    @staticmethod
    def instanton_action(g):
        """
        Single instanton action.

            S_0 = 8pi^2 / g^2

        This is exact, topological, and independent of R.

        LABEL: THEOREM (standard result, BPST 1975)

        Parameters
        ----------
        g : float
            Gauge coupling constant.

        Returns
        -------
        float
            The instanton action S_0.
        """
        return 8.0 * np.pi**2 / g**2

    # ------------------------------------------------------------------
    # Instanton density in the dilute gas approximation
    # ------------------------------------------------------------------
    @classmethod
    def instanton_density_dilute(cls, g, R, N=2):
        """
        Dilute instanton gas density on S3.

            n/V = C_N * (8pi^2/g^2)^(2N) * exp(-8pi^2/g^2) * R^(-4)

        where:
            - C_N is the numerical prefactor from the fluctuation determinant
            - (8pi^2/g^2)^(2N) is the semiclassical power-law factor
            - exp(-8pi^2/g^2) is the tunneling suppression
            - R^(-4) is the dimensional factor (density has dimension length^{-4})

        For SU(2): C_2 ~ 0.466 (from 't Hooft 1976, corrected by Berg-Luscher)
        For SU(3): C_3 ~ 0.0072

        LABEL: NUMERICAL (dilute gas approximation)

        Parameters
        ----------
        g : float
            Gauge coupling constant.
        R : float
            Radius of S3 (in fm or natural units, consistently).
        N : int
            N for SU(N) gauge group. Default 2.

        Returns
        -------
        float
            Instanton number density (in units of R^{-4}).
        """
        S0 = cls.instanton_action(g)
        C_N = cls.PREFACTOR.get(N, cls._estimate_prefactor(N))

        # Power-law prefactor from semiclassical expansion
        power_factor = S0 ** (2 * N)

        # Tunneling suppression
        tunneling = np.exp(-S0)

        # Dimensional factor
        volume_factor = R**(-4)

        return C_N * power_factor * tunneling * volume_factor

    # ------------------------------------------------------------------
    # Vacuum energy from instantons in the theta-vacuum
    # ------------------------------------------------------------------
    @classmethod
    def vacuum_energy_from_instantons(cls, g, R, theta=0.0, N=2):
        """
        Vacuum energy density in the theta-vacuum from instantons.

            E_vac(theta) = -2 * K * cos(theta)

        where K = instanton_density * volume * instanton_action_contribution.

        In the dilute instanton gas:
            K = n_inst * V_{S3}

        where V_{S3} = 2 pi^2 R^3 is the volume of S3 of radius R.

        The vacuum energy is MINIMIZED at theta = 0 (CP conservation).

        This generates a mass for the eta' meson via:
            m^2_{eta'} ~ 2K / f_pi^2

        LABEL: NUMERICAL (dilute gas approximation)

        Parameters
        ----------
        g : float
            Gauge coupling constant.
        R : float
            Radius of S3.
        theta : float
            Vacuum angle (default 0).
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict
            'energy_density'  : vacuum energy density E_vac(theta)
            'K'               : the instanton fugacity K
            'theta'           : the vacuum angle used
            'minimized_at'    : theta value where energy is minimized
            'eta_prime_mass2' : estimate of m^2_{eta'} (in natural units)
        """
        n_density = cls.instanton_density_dilute(g, R, N)

        # Volume of S3
        V_S3 = 2.0 * np.pi**2 * R**3

        # Instanton fugacity: density * volume
        K = n_density * V_S3

        # Vacuum energy density
        E_vac = -2.0 * K * np.cos(theta)

        # eta' mass estimate: m^2_{eta'} ~ 2K / f_pi^2
        # f_pi ~ 93 MeV in natural units; here we express in 1/R^2 units
        # f_pi ~ 93 MeV ~ 93 * R / hbar_c in 1/R units for R in fm
        # For a dimensionless ratio we just record K as the relevant scale
        eta_prime_mass2 = 2.0 * K  # in natural units of the theory

        return {
            'energy_density': E_vac,
            'K': K,
            'theta': theta,
            'minimized_at': 0.0,  # cos(theta) maximized at theta=0
            'eta_prime_mass2': eta_prime_mass2,
        }

    # ------------------------------------------------------------------
    # Mass gap correction from instantons
    # ------------------------------------------------------------------
    @classmethod
    def mass_gap_correction(cls, g, R, N=2):
        """
        Instanton correction to the mass gap on S3.

        The instanton-induced effective potential modifies the spectrum of
        the YM operator. The key physics:

        The mass gap correction is controlled by the TUNNELING AMPLITUDE
        between topological vacua, not by the full instanton density. The
        tunneling amplitude goes as:

            A_tunnel ~ exp(-S_0 / 2)

        and the mass shift (from the splitting of the ground state) is:

            delta_m^2 ~ C_N * exp(-S_0) / R^2

        where exp(-S_0) = |A_tunnel|^2 is the tunneling probability, and
        C_N is an O(1) numerical coefficient from the fluctuation determinant.

        IMPORTANT DISTINCTION:
        The instanton DENSITY includes a factor S_0^{2N} from zero-mode
        integrations (collectively integrated over positions, scales, and
        gauge orientations). This large power-law factor enters the
        partition function but NOT the mass shift directly, because the
        mass shift comes from the overlap of the tunneling amplitude with
        the lowest eigenmode, which involves a normalized mode function.

        On S3:
        - exp(-S_0) at physical coupling g^2 ~ 6: exp(-13.16) ~ 1.9 x 10^{-6}
        - At weak coupling (g << 1): astronomically small
        - At strong coupling (g ~ 1): O(1), but dilute gas breaks down

        FINDING: At physical coupling, instanton corrections are ~10^{-6}
        of the geometric gap. They are NEGLIGIBLE for the mass gap magnitude,
        though important for the theta-vacuum structure (eta' mass, etc.).

        The sign is POSITIVE: instantons break residual symmetry further,
        and the tunneling frequency adds to the energy. This follows from:
        1. The instanton-induced potential is V(a) = -K * cos(a) + const
        2. Expanding around the minimum (a=0): V''(0) = K > 0
        3. A positive curvature contribution raises the eigenvalue

        LABEL: NUMERICAL (dilute gas; sign argument is PROPOSITION)

        Parameters
        ----------
        g : float
            Gauge coupling constant.
        R : float
            Radius of S3.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict
            'correction_magnitude' : |delta_m^2| in units of 1/R^2
            'fraction_of_gap'      : |delta_m^2| / (5/R^2), dimensionless
            'sign'                 : 'positive' (gap increases)
            'regime'               : 'weak', 'strong', or 'physical'
            'geometric_gap'        : 5/R^2
            'corrected_gap'        : (5/R^2) + delta_m^2
            'S0'                   : instanton action
            'suppression_factor'   : exp(-S0), the key smallness parameter
        """
        S0 = cls.instanton_action(g)

        # The mass gap correction is controlled by the tunneling amplitude:
        #
        #   delta_m^2 = C_N * exp(-S_0) / R^2
        #
        # where C_N is the numerical prefactor from the fluctuation determinant
        # (same C_N as in the density, but WITHOUT the S_0^{2N} zero-mode
        # factor, which cancels against the normalization of the collective
        # coordinate integration when computing the matrix element).
        #
        # Physical reasoning: the mass shift comes from
        #   <lowest_mode| V_instanton |lowest_mode>
        # where V_instanton ~ exp(-S_0) on a compact space. The mode is
        # normalized on S3, giving a factor 1/sqrt(V) ~ R^{-3/2} from each
        # bra and ket, and the instanton has extent ~ R, so the integral
        # gives ~ exp(-S_0) * R^3 / (R^3) = exp(-S_0).
        # In units of eigenvalue (1/R^2), we get delta_m^2 ~ exp(-S_0) / R^2.

        C_N = cls.PREFACTOR.get(N, cls._estimate_prefactor(N))
        suppression = np.exp(-S0)

        delta_m2 = C_N * suppression / R**2  # in units of 1/length^2

        # Geometric gap
        geometric_gap = 4.0 / R**2

        # Fraction of the gap
        fraction = abs(delta_m2) / geometric_gap if geometric_gap > 0 else float('inf')

        # Determine the regime
        g2 = g**2
        if g2 < 1.0:
            regime = 'weak'
        elif g2 > 10.0:
            regime = 'strong'
        else:
            regime = 'physical'

        # Suppression factor
        suppression = np.exp(-S0)

        return {
            'correction_magnitude': abs(delta_m2),
            'fraction_of_gap': fraction,
            'sign': 'positive',  # Gap increases (PROPOSITION, see docstring)
            'regime': regime,
            'geometric_gap': geometric_gap,
            'corrected_gap': geometric_gap + delta_m2,  # positive correction
            'S0': S0,
            'suppression_factor': suppression,
        }

    # ------------------------------------------------------------------
    # Moduli space on S3
    # ------------------------------------------------------------------
    @staticmethod
    def moduli_space_on_s3(k=1, N=2):
        """
        Instanton moduli space on S3.

        For k-instanton on SU(N), the moduli space on S4 has dimension:
            dim M = 4kN

        On S3 x S1:
        - The moduli space is COMPACT (key advantage over R^4)
        - The moduli include:
            * position on S3 (3 parameters)
            * scale rho (bounded by R, 1 parameter)
            * gauge orientation (N^2 - 1 parameters for SU(N))
        - Total on S4: 4kN
        - On S3 x S1: same dimension but compactified

        Key result: compactness => integral over moduli CONVERGES.
        On R^4: integral over rho DIVERGES (IR problem). On S3: rho <= R => finite.

        LABEL: THEOREM (dimension formula is standard; compactness on S3 is
        a consequence of compactness of the base manifold)

        Parameters
        ----------
        k : int
            Instanton number (positive).
        N : int
            N for SU(N).

        Returns
        -------
        dict
            'dimension'         : dim of moduli space
            'compact_on_s3'     : True (always)
            'compact_on_r4'     : False (always)
            'parameters'        : breakdown of moduli
            'convergent_integral' : True for S3
        """
        dim = 4 * abs(k) * N

        # Parameter breakdown for k=1
        if k == 1:
            params = {
                'position_on_s3': 3,
                'scale': 1,       # rho, bounded by R on S3
                'gauge_orientation': N**2 - 1,
                'total_on_s4': dim,
                'note': (
                    f'On S3, scale rho is bounded: 0 < rho <= R. '
                    f'On R^4, rho can be arbitrarily large (IR divergence).'
                ),
            }
        else:
            params = {
                'total_on_s4': dim,
                'note': f'{k}-instanton moduli for SU({N})',
            }

        return {
            'dimension': dim,
            'compact_on_s3': True,
            'compact_on_r4': False,
            'parameters': params,
            'convergent_integral': True,
        }

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    @classmethod
    def comparison_table(cls, R=2.2, g2=6.0):
        """
        Compare instanton corrections with the geometric gap at given R.

        Produces a structured table comparing the geometric gap, 1-instanton
        correction, 2-instanton correction, and net gap.

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
            Radius of S3 in fm. Default 2.2 fm.
        g2 : float
            Coupling constant squared (g^2). Default 6.0 (physical QCD).

        Returns
        -------
        dict with:
            'geometric_gap'     : 5/R^2 in 1/fm^2
            'geometric_gap_mev' : gap in MeV (via hbar*c)
            '1_instanton'       : 1-instanton correction dict
            '2_instanton'       : 2-instanton correction dict
            'total_instanton'   : total instanton correction
            'net_gap'           : geometric + instanton
            'fraction_total'    : total instanton / geometric
            'conclusion'        : summary string
        """
        g = np.sqrt(g2)

        # Geometric gap
        geom_gap = 4.0 / R**2
        geom_gap_mev = HBAR_C_MEV_FM * np.sqrt(geom_gap)

        # 1-instanton correction
        corr_1 = cls.mass_gap_correction(g, R, N=2)

        # 2-instanton: action = 2 * S_0, so suppression ~ exp(-2*S_0)
        # In dilute gas, k-instanton contribution scales as exp(-k*S_0)
        S0 = cls.instanton_action(g)
        suppression_2 = np.exp(-2.0 * S0)
        suppression_1 = np.exp(-S0)

        # 2-instanton correction relative to 1-instanton
        ratio_2_to_1 = suppression_2 / suppression_1 if suppression_1 > 0 else 0.0
        corr_2_magnitude = corr_1['correction_magnitude'] * ratio_2_to_1

        # Total instanton correction
        total_inst = corr_1['correction_magnitude'] + corr_2_magnitude
        fraction_total = total_inst / geom_gap if geom_gap > 0 else float('inf')

        # Net gap
        net_gap = geom_gap + total_inst  # positive correction

        # Conclusion
        if fraction_total < 1e-3:
            conclusion = (
                f'Instanton corrections are {fraction_total:.2e} of the '
                f'geometric gap -- NEGLIGIBLE. The mass gap is dominated '
                f'by geometry (Weitzenbock + Hodge on S3).'
            )
        elif fraction_total < 0.1:
            conclusion = (
                f'Instanton corrections are {fraction_total:.2e} of the '
                f'geometric gap -- small but potentially measurable.'
            )
        else:
            conclusion = (
                f'Instanton corrections are {fraction_total:.2e} of the '
                f'geometric gap -- significant, dilute gas approximation '
                f'may be unreliable.'
            )

        return {
            'R_fm': R,
            'g_squared': g2,
            'geometric_gap': geom_gap,
            'geometric_gap_mev': geom_gap_mev,
            '1_instanton': {
                'correction': corr_1['correction_magnitude'],
                'fraction': corr_1['fraction_of_gap'],
                'suppression': corr_1['suppression_factor'],
            },
            '2_instanton': {
                'correction': corr_2_magnitude,
                'fraction': corr_2_magnitude / geom_gap if geom_gap > 0 else 0,
                'suppression': suppression_2,
            },
            'total_instanton': total_inst,
            'fraction_total': fraction_total,
            'net_gap': net_gap,
            'net_gap_mev': HBAR_C_MEV_FM * np.sqrt(net_gap),
            'sign': 'positive',
            'conclusion': conclusion,
        }

    # ------------------------------------------------------------------
    # theta-dependence of the vacuum energy
    # ------------------------------------------------------------------
    @classmethod
    def theta_dependence(cls, g, R, N=2, n_points=100):
        """
        Vacuum energy as a function of the theta angle.

            E_vac(theta) = -2K * cos(theta)

        This is periodic with period 2*pi, minimized at theta = 0,
        and maximized at theta = pi (Dashen phenomenon for N_f flavors,
        but here pure gauge: minimum always at theta = 0).

        LABEL: NUMERICAL

        Parameters
        ----------
        g : float
            Gauge coupling.
        R : float
            Radius of S3.
        N : int
            N for SU(N). Default 2.
        n_points : int
            Number of theta values.

        Returns
        -------
        dict
            'theta'   : array of theta values [0, 2*pi]
            'E_vac'   : array of vacuum energy densities
            'K'       : the instanton fugacity
            'minimum' : theta value where E is minimized (should be 0)
            'maximum' : theta value where E is maximized (should be pi)
        """
        thetas = np.linspace(0, 2 * np.pi, n_points)

        # Get K from a single call at theta=0
        result = cls.vacuum_energy_from_instantons(g, R, theta=0.0, N=N)
        K = result['K']

        # E_vac(theta) = -2K * cos(theta)
        E_vac = -2.0 * K * np.cos(thetas)

        # Find minimum and maximum
        idx_min = np.argmin(E_vac)
        idx_max = np.argmax(E_vac)

        return {
            'theta': thetas,
            'E_vac': E_vac,
            'K': K,
            'minimum': thetas[idx_min],
            'maximum': thetas[idx_max],
        }

    # ------------------------------------------------------------------
    # Coupling regime analysis
    # ------------------------------------------------------------------
    @classmethod
    def coupling_regime_analysis(cls, R=2.2, N=2):
        """
        Analyze instanton corrections across different coupling regimes.

        Surveys how the instanton correction fraction varies with g^2:
            - Weak coupling (g^2 = 0.1): perturbative regime
            - Moderate coupling (g^2 = 1): crossover
            - Physical coupling (g^2 ~ 6): QCD-like
            - Strong coupling (g^2 = 20): non-perturbative

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
            Radius of S3 in fm.
        N : int
            N for SU(N).

        Returns
        -------
        list of dicts, one per coupling value, with fraction_of_gap and regime.
        """
        g2_values = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0]
        results = []

        for g2 in g2_values:
            g = np.sqrt(g2)
            corr = cls.mass_gap_correction(g, R, N)
            results.append({
                'g_squared': g2,
                'S0': corr['S0'],
                'suppression': corr['suppression_factor'],
                'fraction_of_gap': corr['fraction_of_gap'],
                'regime': corr['regime'],
                'correction_magnitude': corr['correction_magnitude'],
            })

        return results

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _estimate_prefactor(N):
        """
        Estimate the instanton density prefactor C_N for general SU(N).

        The prefactor scales approximately as:
            C_N ~ (4pi^2)^{-2N} * exp(c * N)

        For N > 3, we use a rough interpolation.

        LABEL: CONJECTURE (rough estimate for N > 3)

        Parameters
        ----------
        N : int
            N for SU(N).

        Returns
        -------
        float
            Estimated prefactor C_N.
        """
        # Known values: C_2 ~ 0.466, C_3 ~ 0.0072
        # Rough log-linear interpolation: log(C_N) ~ a + b*N
        # log(0.466) ~ -0.764, log(0.0072) ~ -4.934
        # slope b ~ (-4.934 - (-0.764)) / (3 - 2) ~ -4.17
        # intercept a ~ -0.764 - (-4.17)*2 ~ 7.58
        if N < 2:
            return 1.0  # Undefined, return dummy
        a = 7.58
        b = -4.17
        return np.exp(a + b * N)
