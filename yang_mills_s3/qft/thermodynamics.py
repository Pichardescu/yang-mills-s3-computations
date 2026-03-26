"""
Thermodynamics of Yang-Mills on S³ — Partition function from the spectral gap.

CONTEXT (Peer Review Q3):
    The peer review demanded: either rebrand temperature as "fiber excitation scale,"
    or construct explicit Z(beta) from H_eff and derive F, U, S, C_V.

    We choose option (1): construct Z(beta) properly from the KNOWN spectrum.

PHYSICS:
    Yang-Mills on S³ x S¹(beta) (Euclidean time circle of circumference beta = 1/T).

    The partition function for a FREE bosonic field on S³:

        Z(beta) = Tr_Fock exp(-beta * H_YM)

    For independent bosonic modes with energies E_l and degeneracy d_l:

        ln Z = -sum_l d_l * ln(1 - exp(-E_l/T))

    This is the FOCK SPACE partition function — each mode is an independent
    quantum harmonic oscillator that can be multiply occupied.

    The thermodynamic quantities follow:

        F = -T * ln Z = T * sum_l d_l * ln(1 - exp(-E_l/T))

        U = sum_l d_l * E_l / (exp(E_l/T) - 1)   [Bose-Einstein distribution]

        S = (U - F) / T

        C_V = sum_l d_l * (E_l/T)² * exp(E_l/T) / (exp(E_l/T) - 1)²

    With the LINEARIZED spectrum on S³:
        E_l = hbar*c * sqrt(mu_l)   where mu_l = (l(l+2)+2)/R²
        d_l = 2*l*(l+2) * (N²-1)   [Hodge multiplicity x adjoint dim]

    IMPORTANT LIMITATION:
        This is the FREE-FIELD (non-interacting, linearized) partition function.
        The full interacting theory has:
        - A genuine phase transition (2nd order for SU(2), 1st order for SU(3))
        - Polyakov loop deconfinement at T_c ~ 300 MeV (SU(2)), ~170 MeV (SU(3))
        - Non-perturbative effects (instantons, center vortices)

        Our free-gas Z(T) gives a SMOOTH CROSSOVER, not a sharp phase transition.
        This is expected and honest: the free-field approximation cannot capture
        the deconfinement transition, which is inherently non-perturbative.

STATUS: NUMERICAL (thermodynamic quantities from known spectrum;
        free-field approximation, not the full interacting theory)
"""

import numpy as np
from ..spectral.yang_mills_operator import YangMillsOperator, HBAR_C_MEV_FM


class YMThermodynamics:
    """
    Thermodynamics of Yang-Mills on S³ from the known spectrum.

    Constructs Z(beta) EXPLICITLY from the eigenvalue spectrum
    computed in Phases 1-2, using Bose-Einstein statistics for
    a free bosonic field. Then derives F, U, S, C_V and checks
    standard thermodynamic identities.

    The partition function is the FOCK SPACE trace:
        ln Z = -sum_l d_l * ln(1 - exp(-E_l/T))

    This correctly accounts for arbitrary occupation numbers of
    each bosonic mode, giving the Stefan-Boltzmann law at high T.

    All temperatures and energies are in MeV.
    """

    # ------------------------------------------------------------------
    # Spectrum: eigenvalues, energies, degeneracies
    # ------------------------------------------------------------------
    @staticmethod
    def energy_at_l(l: int, R: float) -> float:
        """
        Physical energy of the l-th mode in MeV.

        E_l = hbar*c * sqrt((l(l+2)+2) / R²)
            = hbar*c * sqrt(l(l+2)+2) / R

        Parameters
        ----------
        l : int, angular momentum quantum number (l >= 1)
        R : float, radius of S³ in fm

        Returns
        -------
        float : energy in MeV
        """
        if l < 1:
            raise ValueError(f"l must be >= 1 for 1-forms on S³, got l={l}")
        return HBAR_C_MEV_FM * np.sqrt(l * (l + 2) + 2) / R

    @staticmethod
    def degeneracy(l: int, N: int) -> int:
        """
        Total degeneracy of the l-th energy level.

        d(l, N) = (Hodge multiplicity) x (adjoint dimension)
                = 2*l*(l+2) * (N² - 1)

        The Hodge multiplicity 2*l*(l+2) counts both exact and coexact
        1-form modes on S³ at angular momentum l.

        Parameters
        ----------
        l : int, angular momentum (l >= 1)
        N : int, SU(N) gauge group parameter

        Returns
        -------
        int : total degeneracy
        """
        if l < 1:
            raise ValueError(f"l must be >= 1, got l={l}")
        if N < 2:
            raise ValueError(f"N must be >= 2 for SU(N), got N={N}")
        return 2 * l * (l + 2) * (N**2 - 1)

    # ------------------------------------------------------------------
    # Adaptive l_max
    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_l_max(T: float, R: float, safety: float = 10.0) -> int:
        """
        Compute l_max such that E(l_max) >> T, ensuring Bose-Einstein convergence.

        We need E(l_max) > safety * T, where E(l) ~ hbar*c * l / R for large l.
        So l_max ~ safety * T * R / hbar_c.

        A safety factor of 10 gives convergence to within ~1% of the
        Stefan-Boltzmann limit at high T.

        Parameters
        ----------
        T      : float, temperature in MeV
        R      : float, S³ radius in fm
        safety : float, E(l_max) / T ratio (default 10)

        Returns
        -------
        int : adaptive l_max, at least 20
        """
        l_est = int(safety * T * R / HBAR_C_MEV_FM) + 1
        return max(l_est, 20)

    # ------------------------------------------------------------------
    # Partition function (Fock space, Bose-Einstein)
    # ------------------------------------------------------------------
    @staticmethod
    def log_partition_function(T: float, R: float = 2.2, N: int = 2,
                               l_max: int = 20) -> float:
        """
        ln Z(T) = -sum_{l=1}^{l_max} d(l,N) * ln(1 - exp(-E(l,R)/T))

        This is the Fock space partition function for a free bosonic field.
        Each of the d_l degenerate modes at energy E_l contributes
        independently as a quantum harmonic oscillator.

        Parameters
        ----------
        T     : float, temperature in MeV (must be > 0)
        R     : float, S³ radius in fm (default 2.2)
        N     : int, SU(N) gauge group (default 2)
        l_max : int, maximum angular momentum (default 20)

        Returns
        -------
        float : ln Z(T) >= 0
        """
        if T <= 0:
            raise ValueError(f"Temperature must be > 0, got T={T}")

        ln_Z = 0.0
        for l in range(1, l_max + 1):
            E_l = YMThermodynamics.energy_at_l(l, R)
            d_l = YMThermodynamics.degeneracy(l, N)
            ratio = E_l / T
            if ratio < 700:  # safe from underflow
                x = np.exp(-ratio)
                ln_Z -= d_l * np.log(1.0 - x)
        return ln_Z

    @staticmethod
    def partition_function(T: float, R: float = 2.2, N: int = 2,
                           l_max: int = 20) -> float:
        """
        Z(T) = exp(ln Z) where ln Z = -sum d_l * ln(1 - exp(-E_l/T))

        For numerical stability, we return exp(ln Z).
        Note: Z can be astronomically large at high T.

        Parameters
        ----------
        T     : float, temperature in MeV (must be > 0)
        R     : float, S³ radius in fm (default 2.2)
        N     : int, SU(N) gauge group (default 2)
        l_max : int, maximum angular momentum (default 20)

        Returns
        -------
        float : partition function Z(T) >= 1
        """
        ln_Z = YMThermodynamics.log_partition_function(T, R, N, l_max)
        # Cap to avoid overflow; for Z > exp(700) just return a large number
        if ln_Z > 700:
            return np.exp(700.0)
        return np.exp(ln_Z)

    # ------------------------------------------------------------------
    # Free energy
    # ------------------------------------------------------------------
    @staticmethod
    def free_energy(T: float, R: float = 2.2, N: int = 2,
                    l_max: int = 20) -> float:
        """
        Helmholtz free energy: F = -T * ln(Z)

        F = T * sum_l d_l * ln(1 - exp(-E_l/T))

        Note: F <= 0 always (since ln(1-x) < 0 for 0 < x < 1).

        Parameters
        ----------
        T     : float, temperature in MeV
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, maximum angular momentum

        Returns
        -------
        float : free energy in MeV
        """
        ln_Z = YMThermodynamics.log_partition_function(T, R, N, l_max)
        return -T * ln_Z

    # ------------------------------------------------------------------
    # Internal energy
    # ------------------------------------------------------------------
    @staticmethod
    def internal_energy(T: float, R: float = 2.2, N: int = 2,
                        l_max: int = 20) -> float:
        """
        Internal energy (Bose-Einstein):

            U = sum_l d_l * E_l / (exp(E_l/T) - 1)

        Each mode is occupied with Bose-Einstein distribution <n_l> = 1/(exp(E_l/T)-1).

        Parameters
        ----------
        T     : float, temperature in MeV
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, maximum angular momentum

        Returns
        -------
        float : internal energy in MeV
        """
        if T <= 0:
            raise ValueError(f"Temperature must be > 0, got T={T}")

        U = 0.0
        for l in range(1, l_max + 1):
            E_l = YMThermodynamics.energy_at_l(l, R)
            d_l = YMThermodynamics.degeneracy(l, N)
            ratio = E_l / T
            if ratio < 700:
                U += d_l * E_l / (np.exp(ratio) - 1.0)
        return U

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(T: float, R: float = 2.2, N: int = 2,
                l_max: int = 20) -> float:
        """
        Entropy: S = (U - F) / T = ln(Z) + U/T

        For Bose-Einstein modes:
            S = sum_l d_l * [ (E_l/T)/(exp(E_l/T)-1) - ln(1-exp(-E_l/T)) ]

        Parameters
        ----------
        T     : float, temperature in MeV
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, maximum angular momentum

        Returns
        -------
        float : entropy (dimensionless, in natural units)
        """
        F = YMThermodynamics.free_energy(T, R, N, l_max)
        U = YMThermodynamics.internal_energy(T, R, N, l_max)
        return (U - F) / T

    # ------------------------------------------------------------------
    # Specific heat
    # ------------------------------------------------------------------
    @staticmethod
    def specific_heat(T: float, R: float = 2.2, N: int = 2,
                      l_max: int = 20) -> float:
        """
        Specific heat for Bose-Einstein gas:

            C_V = dU/dT = sum_l d_l * (E_l/T)² * exp(E_l/T) / (exp(E_l/T)-1)²

        This is always non-negative (thermodynamic stability).

        Parameters
        ----------
        T     : float, temperature in MeV
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, maximum angular momentum

        Returns
        -------
        float : specific heat (dimensionless, in natural units)
        """
        if T <= 0:
            raise ValueError(f"Temperature must be > 0, got T={T}")

        Cv = 0.0
        for l in range(1, l_max + 1):
            E_l = YMThermodynamics.energy_at_l(l, R)
            d_l = YMThermodynamics.degeneracy(l, N)
            ratio = E_l / T
            if ratio < 500:  # exp(500) ~ 1e217, safe from overflow in ratio^2*exp
                ex = np.exp(ratio)
                Cv += d_l * ratio**2 * ex / (ex - 1.0)**2
            # For ratio >= 500: contribution ~ ratio^2 * exp(-ratio) ~ 0
        return Cv

    # ------------------------------------------------------------------
    # Thermodynamic identity verification
    # ------------------------------------------------------------------
    @staticmethod
    def verify_thermodynamic_identities(T: float, R: float = 2.2,
                                        N: int = 2, l_max: int = 20) -> dict:
        """
        Check standard thermodynamic identities:

        1. F = U - T*S  (definition check)
        2. S = -dF/dT   (numerical derivative)
        3. C_V = T * dS/dT  (numerical derivative)
        4. C_V >= 0  (stability)
        5. S >= 0  (third law compatible)

        Parameters
        ----------
        T     : float, temperature in MeV
        R     : float, S³ radius in fm
        N     : int, SU(N)
        l_max : int, max angular momentum

        Returns
        -------
        dict with check results and relative errors
        """
        F = YMThermodynamics.free_energy(T, R, N, l_max)
        U = YMThermodynamics.internal_energy(T, R, N, l_max)
        S = YMThermodynamics.entropy(T, R, N, l_max)
        Cv = YMThermodynamics.specific_heat(T, R, N, l_max)

        # Check 1: F = U - T*S
        F_check = U - T * S
        if abs(F) > 1e-15:
            error_1 = abs(F - F_check) / abs(F)
        else:
            error_1 = abs(F - F_check)

        # Check 2: S = -dF/dT (numerical derivative)
        dT = T * 1e-5
        F_plus = YMThermodynamics.free_energy(T + dT, R, N, l_max)
        F_minus = YMThermodynamics.free_energy(T - dT, R, N, l_max)
        dFdT_numerical = (F_plus - F_minus) / (2 * dT)
        S_from_deriv = -dFdT_numerical
        if abs(S) > 1e-15:
            error_2 = abs(S - S_from_deriv) / abs(S)
        else:
            error_2 = abs(S - S_from_deriv)

        # Check 3: C_V = T * dS/dT (numerical derivative)
        S_plus = YMThermodynamics.entropy(T + dT, R, N, l_max)
        S_minus = YMThermodynamics.entropy(T - dT, R, N, l_max)
        dSdT_numerical = (S_plus - S_minus) / (2 * dT)
        Cv_from_deriv = T * dSdT_numerical
        if abs(Cv) > 1e-15:
            error_3 = abs(Cv - Cv_from_deriv) / abs(Cv)
        else:
            error_3 = abs(Cv - Cv_from_deriv)

        # Check 4: C_V >= 0
        cv_positive = Cv >= 0

        # Check 5: S >= 0
        s_positive = S >= -1e-15  # small numerical tolerance

        return {
            'F_equals_U_minus_TS': {
                'F': F,
                'U_minus_TS': F_check,
                'relative_error': error_1,
                'passed': error_1 < 1e-10,
            },
            'S_equals_minus_dFdT': {
                'S_direct': S,
                'S_from_derivative': S_from_deriv,
                'relative_error': error_2,
                'passed': error_2 < 1e-3,
            },
            'Cv_equals_T_dSdT': {
                'Cv_direct': Cv,
                'Cv_from_derivative': Cv_from_deriv,
                'relative_error': error_3,
                'passed': error_3 < 1e-3,
            },
            'Cv_non_negative': {
                'Cv': Cv,
                'passed': cv_positive,
            },
            'S_non_negative': {
                'S': S,
                'passed': s_positive,
            },
            'all_passed': (
                error_1 < 1e-10
                and error_2 < 1e-3
                and error_3 < 1e-3
                and cv_positive
                and s_positive
            ),
        }

    # ------------------------------------------------------------------
    # Gap extraction from low-T specific heat
    # ------------------------------------------------------------------
    @staticmethod
    def gap_from_low_T_behavior(R: float = 2.2, N: int = 2) -> dict:
        """
        At low T, the specific heat is dominated by the first excited state.

        For Bose-Einstein statistics with a gapped spectrum:
            C_V ~ d_1 * (E_1/T)² * exp(-E_1/T)  at T << E_1

        (Because when E_1/T >> 1, the BE distribution reduces to the
        Boltzmann tail: n ~ exp(-E/T), and C_V ~ (E/T)^2 * exp(-E/T).)

        By fitting log(C_V) vs 1/T at low T, we extract E_1
        and compare with the known spectral gap sqrt(5)*hbar*c/R.

        STATUS: NUMERICAL (consistency check, not a new result)

        Parameters
        ----------
        R : float, S³ radius in fm
        N : int, SU(N) gauge group

        Returns
        -------
        dict with extracted gap and comparison
        """
        E_1_exact = HBAR_C_MEV_FM * np.sqrt(5.0) / R

        # Use multiple low temperatures for a linear fit.
        # At low T: C_V ~ d_1 * (E_1/T)^2 * exp(-E_1/T)
        # log(C_V * T^2) ~ -E_1/T + log(d_1 * E_1^2)
        # Linear fit of log(C_V * T^2) vs 1/T gives slope = -E_1.
        #
        # We use T = E_1/20 to E_1/14, deep in the Boltzmann tail.

        factors = [20, 18, 16, 14]
        inv_T_vals = []
        log_CvT2_vals = []

        for f in factors:
            T_low = E_1_exact / f
            Cv = YMThermodynamics.specific_heat(T_low, R, N, l_max=5)
            if Cv > 0:
                inv_T_vals.append(1.0 / T_low)
                log_CvT2_vals.append(np.log(Cv * T_low**2))

        if len(inv_T_vals) >= 2:
            inv_T_arr = np.array(inv_T_vals)
            y_arr = np.array(log_CvT2_vals)
            coeffs = np.polyfit(inv_T_arr, y_arr, 1)
            slope = coeffs[0]
            E_1_extracted = -slope
            relative_error = abs(E_1_extracted - E_1_exact) / E_1_exact
        else:
            E_1_extracted = float('nan')
            relative_error = float('nan')

        T_range = [E_1_exact / f for f in factors]

        return {
            'E_1_exact_MeV': E_1_exact,
            'E_1_extracted_MeV': E_1_extracted,
            'relative_error': relative_error,
            'consistent': relative_error < 0.01 if np.isfinite(relative_error) else False,
            'T_range_MeV': T_range,
            'n_fit_points': len(inv_T_vals),
            'note': (
                'Low-T fit of C_V recovers the spectral gap E_1 = hbar*c*sqrt(5)/R. '
                'This is a CONSISTENCY CHECK: the partition function correctly '
                'encodes the spectral gap. At T << E_1, the Bose-Einstein '
                'distribution reduces to the Boltzmann tail, and the single-mode '
                'approximation is excellent.'
            ),
        }

    # ------------------------------------------------------------------
    # Stefan-Boltzmann check (high-T limit)
    # ------------------------------------------------------------------
    @staticmethod
    def stefan_boltzmann_check(R: float = 2.2, N: int = 2) -> dict:
        """
        At high T >> gap, the system should approach a gas of (N²-1) massless
        gauge bosons on S³ (2 polarizations each = d_l modes per level).

        In the flat-space limit (T >> hbar*c/R):

            U/V -> (N²-1) * pi²/15 * T⁴ / (hbar*c)³

        On S³: V = 2*pi²*R³ (volume of 3-sphere of radius R).

        The (N²-1) * pi²/15 factor arises because each of the (N²-1) adjoint
        colors has 2 transverse polarizations, giving 2*(N²-1) dof total,
        and the SB law per bosonic dof is pi²T⁴/30. So:
            u = 2*(N²-1) * pi²/30 * T⁴ = (N²-1) * pi²/15 * T⁴

        l_max is chosen adaptively so that E(l_max) >> T, ensuring
        that the Boltzmann sum has converged.

        STATUS: NUMERICAL (consistency with known flat-space limit)

        Parameters
        ----------
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group

        Returns
        -------
        dict with comparison to Stefan-Boltzmann
        """
        V_S3 = 2 * np.pi**2 * R**3  # Volume of S³ in fm³
        adj_dim = N**2 - 1

        results = []
        for T in [500.0, 1000.0, 2000.0, 5000.0]:
            l_max = YMThermodynamics._adaptive_l_max(T, R)
            U = YMThermodynamics.internal_energy(T, R, N, l_max)

            u_numerical = U / V_S3  # MeV / fm³

            # SB energy density in MeV/fm³
            u_SB = adj_dim * np.pi**2 / 15.0 * T**4 / HBAR_C_MEV_FM**3

            ratio = u_numerical / u_SB if u_SB > 0 else float('inf')

            results.append({
                'T_MeV': T,
                'l_max_used': l_max,
                'U_numerical_MeV': U,
                'u_numerical_MeV_per_fm3': u_numerical,
                'u_SB_MeV_per_fm3': u_SB,
                'ratio_to_SB': ratio,
            })

        highest_T_ratio = results[-1]['ratio_to_SB']

        return {
            'results': results,
            'V_S3_fm3': V_S3,
            'approaches_SB': 0.5 < highest_T_ratio < 2.0,
            'highest_T_ratio': highest_T_ratio,
            'note': (
                'At T >> hbar*c/R, the Bose-Einstein partition function on S³ '
                'reproduces the Stefan-Boltzmann law for (N²-1) massless gauge '
                'bosons (2 polarizations each). Deviations at moderate T reflect '
                'finite-size effects on the compact S³. l_max is chosen '
                'adaptively to ensure convergence. The free-field '
                'approximation is used throughout — interaction effects are '
                'not included.'
            ),
        }

    # ------------------------------------------------------------------
    # Deconfinement from thermodynamic signatures
    # ------------------------------------------------------------------
    @staticmethod
    def deconfinement_from_thermodynamics(R: float = 2.2, N: int = 2,
                                          l_max: int = 30) -> dict:
        """
        Analysis of deconfinement signatures in the free-gas approximation.

        In the full interacting theory:
        - SU(2): 2nd order phase transition at T_c ~ 300 MeV
        - SU(3): 1st order phase transition at T_c ~ 170 MeV

        Our free Bose-Einstein Z(T) CANNOT reproduce a genuine phase transition.
        For a free bosonic gas, C_V is monotonically increasing (no peak).

        Instead, we identify a "crossover scale" T* where the lowest mode
        becomes significantly populated: <n_1> = 1/(exp(E_1/T*) - 1) ~ 1,
        which gives T* ~ E_1/ln(2) ~ E_1 * 1.44.

        Below T*, the system is essentially in the vacuum (exponentially
        suppressed occupation). Above T*, modes are thermally populated
        and the system behaves as a hot gas.

        STATUS: NUMERICAL with HONEST LIMITATION.
        The free-gas approximation gives a smooth crossover, NOT a phase
        transition. This crossover scale is NOT the deconfinement
        temperature — it is where thermal excitations become O(1).

        Parameters
        ----------
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, max angular momentum

        Returns
        -------
        dict with crossover temperature and comparison to lattice
        """
        E_1 = YMThermodynamics.energy_at_l(1, R)
        d_1 = YMThermodynamics.degeneracy(1, N)

        # Crossover scale: T* where <n_1> = 1  =>  T* = E_1 / ln(2)
        T_crossover = E_1 / np.log(2.0)

        # Compute thermodynamic quantities at crossover
        Cv_at_cross = YMThermodynamics.specific_heat(T_crossover, R, N, l_max)
        S_at_cross = YMThermodynamics.entropy(T_crossover, R, N, l_max)

        # Compare with lattice values
        lattice_Tc = {2: 300.0, 3: 170.0}
        T_c_lattice = lattice_Tc.get(N, None)

        return {
            'T_crossover_MeV': T_crossover,
            'Cv_at_crossover': Cv_at_cross,
            'S_at_crossover': S_at_cross,
            'E_1_MeV': E_1,
            'T_crossover_over_E1': T_crossover / E_1,
            'T_c_lattice_MeV': T_c_lattice,
            'ratio_to_lattice': (
                T_crossover / T_c_lattice if T_c_lattice else None
            ),
            'note': (
                'HONEST LIMITATION: T* = E_1/ln(2) is the temperature where '
                'the lowest mode has occupation number <n_1> = 1 in the '
                'FREE-GAS (Bose-Einstein) approximation. For a free bosonic '
                'gas, C_V is MONOTONICALLY INCREASING — there is no peak and '
                'no phase transition. The true deconfinement transition is a '
                'genuinely non-perturbative phenomenon involving center '
                'symmetry breaking and Polyakov loop dynamics, which cannot '
                'be captured by a free-field partition function. '
                'For SU(2), the transition is 2nd order; for SU(3), 1st order.'
            ),
        }

    # ------------------------------------------------------------------
    # Thermodynamic table
    # ------------------------------------------------------------------
    @staticmethod
    def thermodynamic_table(R: float = 2.2, N: int = 2,
                            l_max: int = 20) -> list:
        """
        Table of F, U, S, C_V for T from 10 to 500 MeV.

        KEY DELIVERABLE: this is the explicit construction of Z(beta)
        demanded by the peer review.

        Parameters
        ----------
        R     : float, S³ radius in fm
        N     : int, SU(N) gauge group
        l_max : int, max angular momentum

        Returns
        -------
        list of dicts, each with keys:
            T_MeV, Z, F_MeV, U_MeV, S, Cv
        """
        T_values = np.concatenate([
            np.arange(10, 100, 10),    # 10, 20, ..., 90
            np.arange(100, 550, 50),   # 100, 150, ..., 500
        ])

        table = []
        for T in T_values:
            ln_Z = YMThermodynamics.log_partition_function(T, R, N, l_max)
            F = YMThermodynamics.free_energy(T, R, N, l_max)
            U = YMThermodynamics.internal_energy(T, R, N, l_max)
            S = YMThermodynamics.entropy(T, R, N, l_max)
            Cv = YMThermodynamics.specific_heat(T, R, N, l_max)

            table.append({
                'T_MeV': float(T),
                'ln_Z': ln_Z,
                'Z': np.exp(min(ln_Z, 700)),  # cap for display
                'F_MeV': F,
                'U_MeV': U,
                'S': S,
                'Cv': Cv,
            })

        return table

    # ------------------------------------------------------------------
    # S³ w-oscillation connection
    # ------------------------------------------------------------------
    @staticmethod
    def w_oscillation_connection(T: float, R: float = 2.2) -> dict:
        """
        THE S³ FRAMEWORK CONNECTION:

        In the Euclidean formulation, T = 1/beta where beta is the
        circumference of the time circle S¹.

        Interpretation 1 — Euclidean time circle:
            beta = 1/T is the circumference of the Euclidean time circle.
            High T = small beta = SMALL circle = CONFINED in imaginary time
            Low T  = large beta = LARGE circle = EXTENDED in imaginary time

        Interpretation 2 — w-oscillation frequency:
            The thermal frequency omega_T = T / hbar (in natural units)
            = T (in energy units, where hbar = 1)
            In conventional units: omega = T / hbar = T / (6.582e-22 MeV*s)

        In the compact topology framework, the 4th coordinate w parametrizes position on S³.
        The Euclidean time circle at temperature T has circumference:
            beta = hbar*c / T  (in fm, using hbar*c = 197.3 MeV*fm)

        This is the periodicity of the w-coordinate in the thermal theory.

        Physical implication:
            At T = 200 MeV (~ gap):
                beta = 197.3 / 200 ~ 0.987 fm
                omega = 200 / (6.582e-22) ~ 3.04e23 Hz

            At T = 0 (vacuum):
                beta -> infinity (no periodic identification)
                The full S³ is explored

            At T >> gap:
                beta -> 0 (tight circle)
                Dimensional reduction to 3d effective theory

        STATUS: POSTULATE (interpretive framework, not derivable from Z alone)

        Parameters
        ----------
        T : float, temperature in MeV
        R : float, S³ radius in fm

        Returns
        -------
        dict with both interpretations and physical quantities
        """
        HBAR_S = 6.582119569e-22  # hbar in MeV*s

        beta_fm = HBAR_C_MEV_FM / T  # Euclidean time circumference in fm
        omega_Hz = T / HBAR_S         # thermal frequency in Hz

        E_1 = HBAR_C_MEV_FM * np.sqrt(5.0) / R  # gap energy

        return {
            'T_MeV': T,
            'beta_fm': beta_fm,
            'beta_over_R': beta_fm / R,
            'omega_Hz': omega_Hz,
            'T_over_gap': T / E_1,
            'interpretation_euclidean': (
                f'Euclidean time circle: beta = {beta_fm:.4f} fm. '
                f'beta/R = {beta_fm / R:.4f}. '
                f'{"beta >> R: low-T regime, time circle much larger than S³" if beta_fm > 2 * R else ""}'
                f'{"beta ~ R: intermediate regime, time circle comparable to S³" if 0.5 * R < beta_fm <= 2 * R else ""}'
                f'{"beta << R: high-T regime, dimensional reduction to S³" if beta_fm <= 0.5 * R else ""}'
            ),
            'interpretation_frequency': (
                f'Thermal frequency: omega = {omega_Hz:.3e} Hz. '
                f'In the compact topology framework, this is the rate of w-oscillation at temperature T.'
            ),
            'dimensional_reduction': (
                beta_fm < R,
                f'At T = {T:.0f} MeV, beta = {beta_fm:.3f} fm '
                f'{"<" if beta_fm < R else ">="} R = {R} fm. '
                f'{"Dimensional reduction applies: effective 3d theory on S³." if beta_fm < R else "Full 4d theory on S³ x S¹."}'
            ),
            'note': (
                'Both interpretations are CONSISTENT with standard thermal QFT. '
                'The framework-specific claim is that the Euclidean time circle '
                'corresponds to oscillation in the w-coordinate of S³. '
                'This is interpretive (POSTULATE level), not derivable from '
                'the partition function alone. The thermodynamic quantities '
                '(F, U, S, Cv) are independent of this interpretation.'
            ),
        }
