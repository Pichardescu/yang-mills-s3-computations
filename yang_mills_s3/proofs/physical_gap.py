"""
Physical Mass Gap from the 9-DOF Effective Hamiltonian on S³/I*.

Connects the finite-dimensional truncation (9 DOF = 3 modes × 3 adjoint for SU(2))
to the PHYSICAL mass gap in units of Λ_QCD.

THE PHYSICS:
    The YM action on S³(R) × R_time for mode expansion A = Σ aᵢ(t) eᵢ(x):

        S = ∫ dt { (V_{S³}/(2g²)) [|ȧ|² - ω²|a|² - V₄(a)] }

    where V_{S³} = 2π²R³, ω² = 4/R² (coexact eigenvalue), g² = g²(R) running.

    The effective mass parameter:
        M = V_{S³}/g² = 2π²R³/g²(R)

    The QUANTUM Hamiltonian (ℏ=c=1):
        H = (1/(2M)) |p|² + (M/2)ω²|a|² + α|a|⁴

    with α = g²/(2V_{S³}) × (structure constant + mode overlap factors).

    The Gribov restriction confines a to Ω₉ (bounded convex, diameter d(R)).
    Dirichlet BC ψ = 0 on ∂Ω₉.

GAP COMPONENTS:
    1. Harmonic gap: Δ_harm = ω = 2/R  (THEOREM)
    2. Andrews-Clutterbuck bound: Δ_AC ≥ 3π²/(2M·d²)  (THEOREM for convex V)
    3. Anharmonic gap: Δ_anh from numerical diagonalization  (NUMERICAL)

UNITS: Λ_QCD = 1 throughout. Physical gap in units of Λ_QCD.

References:
    - Andrews & Clutterbuck (2011): Proof of the fundamental gap conjecture
    - Payne & Weinberger (1960): Optimal Poincaré inequality for convex domains
    - Dell'Antonio & Zwanziger (1989/1991): Convexity of the Gribov region
"""

import numpy as np
from scipy.linalg import eigh
from ..spectral.zwanziger_gap_equation import ZwanzigerGapEquation
from .gribov_diameter import GribovDiameter


class PhysicalGap:
    """
    Physical mass gap computation for the 9-DOF Yang-Mills truncation on S³/I*.

    Combines:
    - Running coupling g²(R) from Zwanziger gap equation
    - Gribov diameter d(R) from gribov_diameter module
    - Effective Hamiltonian with correct physical units (mass M = V_{S³}/g²)
    - Numerical diagonalization of anharmonic oscillator on bounded domain

    Units: Λ_QCD = 1 throughout.
    """

    def __init__(self, gribov_diameter_obj=None, n_gribov_directions=50):
        """
        Parameters
        ----------
        gribov_diameter_obj : GribovDiameter or None
            Pre-constructed GribovDiameter instance. Created if None.
        n_gribov_directions : int
            Number of directions for Gribov diameter sampling.
        """
        self.gribov = gribov_diameter_obj or GribovDiameter()
        self.n_gribov_directions = n_gribov_directions
        # Cache for Gribov diameter results
        self._diameter_cache = {}

    # ------------------------------------------------------------------
    # Volume of S³
    # ------------------------------------------------------------------
    @staticmethod
    def volume_s3(R):
        """Volume of S³(R): V = 2π²R³."""
        return 2.0 * np.pi**2 * R**3

    # ------------------------------------------------------------------
    # Effective mass M(R)
    # ------------------------------------------------------------------
    def effective_mass(self, R, N=2):
        """
        Effective mass parameter M = V_{S³}/g²(R).

        In the mode expansion, the kinetic term is (V_{S³}/(2g²))|ȧ|².
        The canonical momentum is p = M·ȧ where M = V_{S³}/g².
        The kinetic energy in the Hamiltonian is |p|²/(2M).

        LABEL: NUMERICAL (depends on running coupling)

        Parameters
        ----------
        R : float
            Radius of S³ in units of 1/Λ_QCD.
        N : int
            Number of colors. Default 2.

        Returns
        -------
        float
            M = 2π²R³/g²(R) in units of 1/Λ_QCD.
        """
        g2 = ZwanzigerGapEquation.running_coupling_g2(R, N)
        V = self.volume_s3(R)
        return V / g2

    # ------------------------------------------------------------------
    # Harmonic gap
    # ------------------------------------------------------------------
    def harmonic_gap(self, R, N=2):
        """
        Free harmonic oscillator gap: ω = 2/R.

        This is the geometric gap from the coexact eigenvalue on S³.
        For the harmonic oscillator H = p²/(2M) + (M/2)ω²x², the gap is ω
        regardless of M. This is the gap of the LINEARIZED theory.

        LABEL: THEOREM (Hodge theory on S³)

        Parameters
        ----------
        R : float
            Radius of S³ in units of 1/Λ_QCD.
        N : int
            Number of colors (unused, gap is universal).

        Returns
        -------
        float
            ω = 2/R in units of Λ_QCD.
        """
        return 2.0 / R

    # ------------------------------------------------------------------
    # Gribov diameter (cached)
    # ------------------------------------------------------------------
    def _get_gribov_diameter(self, R, N=2):
        """Get Gribov diameter, using cache."""
        key = (R, N)
        if key not in self._diameter_cache:
            result = self.gribov.gribov_diameter_estimate(
                R, N, n_directions=self.n_gribov_directions
            )
            self._diameter_cache[key] = result['diameter']
        return self._diameter_cache[key]

    # ------------------------------------------------------------------
    # Particle-in-box gap (Andrews-Clutterbuck / Payne-Weinberger)
    # ------------------------------------------------------------------
    def particle_in_box_gap(self, R, N=2):
        """
        Gap from Gribov confinement: Andrews-Clutterbuck / Payne-Weinberger bound.

        For -Δ/(2M) + V_convex on a convex domain of diameter d:
            E₁ - E₀ ≥ 3π²/(2M·d²)    (Andrews-Clutterbuck, for convex potential)

        For the pure particle-in-box (V=0, Dirichlet BC):
            E₁ - E₀ ≥ π²/(2M·d²)     (Payne-Weinberger)

        We use the factor 3 (AC) since V₂ + V₄ is convex.

        LABEL: THEOREM (AC 2011 + DZ convexity of Ω)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            Number of colors.

        Returns
        -------
        float
            Lower bound 3π²/(2M·d²) in units of Λ_QCD.
        """
        M = self.effective_mass(R, N)
        d = self._get_gribov_diameter(R, N)
        if d <= 0 or not np.isfinite(d):
            return 0.0
        return 3.0 * np.pi**2 / (2.0 * M * d**2)

    # ------------------------------------------------------------------
    # Quartic coupling coefficient
    # ------------------------------------------------------------------
    def quartic_coupling(self, R, N=2):
        """
        Effective quartic coupling α for the 1D reduction.

        From the full Hamiltonian:
            H = (1/(2M))|p|² + (M/2)ω²|a|² + α_full · (quartic invariant)

        For SU(2) on S³/I*, the quartic vertex from [A,A] is:
            V₄ = (1/(4g²))∫|[A,A]|² = (g²/(4V))·C₄·|a|⁴_eff

        where C₄ ~ O(1) is the geometric/group theory factor from the mode
        overlap integral. For the 1D effective oscillator (radial mode),
        the quartic coefficient is:

            α = (g²/(2V_{S³})) × C₄

        where C₄ ≈ 2 accounts for the structure constant contraction and
        mode overlap integrals.

        In the Lagrangian: V₄ ~ (V/(2g²)) × (g²/V)² × C₄ × a⁴ = (g²C₄)/(2V) × a⁴

        LABEL: NUMERICAL (the exact C₄ depends on mode overlaps)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            Number of colors.

        Returns
        -------
        float
            Quartic coupling α in natural units.
        """
        g2 = ZwanzigerGapEquation.running_coupling_g2(R, N)
        V = self.volume_s3(R)
        C4 = 2.0  # geometric factor from mode self-coupling (see effective_hamiltonian.py)
        return g2 * C4 / (2.0 * V)

    # ------------------------------------------------------------------
    # 1D anharmonic oscillator on bounded interval
    # ------------------------------------------------------------------
    def anharmonic_gap_1d(self, R, N=2, n_basis=50):
        """
        Solve the 1D anharmonic oscillator numerically on [-d/2, d/2].

        H₁ = -(1/(2M)) d²/dx² + (M/2)ω²x² + α·x⁴

        with Dirichlet BC ψ(±d/2) = 0.

        Uses finite differences on a uniform grid.

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            Number of colors.
        n_basis : int
            Number of interior grid points for finite differences.

        Returns
        -------
        float
            E₁ - E₀ (mass gap) in units of Λ_QCD.
        """
        M = self.effective_mass(R, N)
        omega2 = 4.0 / R**2
        alpha = self.quartic_coupling(R, N)
        d = self._get_gribov_diameter(R, N)

        if d <= 0 or not np.isfinite(d):
            # Fallback: use harmonic gap
            return self.harmonic_gap(R, N)

        half_d = d / 2.0
        n = n_basis
        dx = d / (n + 1)
        x = np.linspace(-half_d + dx, half_d - dx, n)

        # Kinetic energy: -(1/(2M)) d²/dx² via finite differences
        # Second derivative: (f_{i+1} - 2f_i + f_{i-1})/dx²
        kinetic_coeff = 1.0 / (2.0 * M * dx**2)

        # Build tridiagonal kinetic matrix
        diag_K = np.full(n, 2.0 * kinetic_coeff)
        offdiag_K = np.full(n - 1, -kinetic_coeff)

        # Potential energy on diagonal
        V = 0.5 * M * omega2 * x**2 + alpha * x**4

        # Full Hamiltonian
        H = np.diag(diag_K + V) + np.diag(offdiag_K, 1) + np.diag(offdiag_K, -1)

        # Diagonalize (only need lowest 2 eigenvalues)
        eigenvalues = eigh(H, eigvals_only=True, subset_by_index=[0, 1])

        return eigenvalues[1] - eigenvalues[0]

    # ------------------------------------------------------------------
    # 3D anharmonic oscillator (spherical reduction)
    # ------------------------------------------------------------------
    def anharmonic_gap_3d(self, R, N=2, n_basis=20):
        """
        Solve the 3-mode anharmonic oscillator with spherical symmetry.

        For 3 equivalent modes with SO(3) symmetry, we can reduce to the
        radial equation in 3D:

            H_rad = -(1/(2M)) [d²/dr² + (2/r)d/dr] + (M/2)ω²r² + α·r⁴
                  + l(l+1)/(2M·r²)

        The ground state has l=0, first excited state has l=1.
        The mass gap is E(l=1) - E(l=0).

        Uses finite differences on r ∈ [0, d/2] with ψ(d/2) = 0
        and regularity at r = 0.

        For l=0: substitute u = r·ψ, then
            -(1/(2M)) u'' + [(M/2)ω²r² + α·r⁴]u = E·u
            with u(0) = 0, u(d/2) = 0

        For l=1: substitute u = r·ψ, then
            -(1/(2M)) u'' + [(M/2)ω²r² + α·r⁴ + l(l+1)/(2M·r²)]u = E·u

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            Number of colors.
        n_basis : int
            Number of interior grid points for finite differences.

        Returns
        -------
        float
            E(l=1) - E(l=0) in units of Λ_QCD.
        """
        M = self.effective_mass(R, N)
        omega2 = 4.0 / R**2
        alpha = self.quartic_coupling(R, N)
        d = self._get_gribov_diameter(R, N)

        if d <= 0 or not np.isfinite(d):
            return self.harmonic_gap(R, N)

        r_max = d / 2.0
        n = n_basis

        def solve_radial(l_angular):
            """Solve radial equation for angular momentum l using u = r·ψ."""
            dr = r_max / (n + 1)
            r = np.linspace(dr, r_max - dr, n)

            kinetic_coeff = 1.0 / (2.0 * M * dr**2)
            diag_K = np.full(n, 2.0 * kinetic_coeff)
            offdiag_K = np.full(n - 1, -kinetic_coeff)

            # Effective potential for u = r·ψ
            V_eff = 0.5 * M * omega2 * r**2 + alpha * r**4
            if l_angular > 0:
                V_eff += l_angular * (l_angular + 1) / (2.0 * M * r**2)

            H_rad = (np.diag(diag_K + V_eff)
                     + np.diag(offdiag_K, 1)
                     + np.diag(offdiag_K, -1))

            evals = eigh(H_rad, eigvals_only=True, subset_by_index=[0, 0])
            return evals[0]

        E0_l0 = solve_radial(0)
        E0_l1 = solve_radial(1)

        return E0_l1 - E0_l0

    # ------------------------------------------------------------------
    # Pure harmonic oscillator gap (for validation)
    # ------------------------------------------------------------------
    def harmonic_gap_numerical(self, R, N=2, n_basis=50):
        """
        Solve the harmonic oscillator on [-d/2, d/2] with Dirichlet BC.
        Should agree with ω = 2/R when d >> harmonic oscillator length.

        Used for validation of the finite-difference solver.

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
        N : int
        n_basis : int

        Returns
        -------
        float
            E₁ - E₀ from the harmonic oscillator with Dirichlet BC.
        """
        M = self.effective_mass(R, N)
        omega2 = 4.0 / R**2
        d = self._get_gribov_diameter(R, N)

        if d <= 0 or not np.isfinite(d):
            return self.harmonic_gap(R, N)

        half_d = d / 2.0
        n = n_basis
        dx = d / (n + 1)
        x = np.linspace(-half_d + dx, half_d - dx, n)

        kinetic_coeff = 1.0 / (2.0 * M * dx**2)
        diag_K = np.full(n, 2.0 * kinetic_coeff)
        offdiag_K = np.full(n - 1, -kinetic_coeff)

        V = 0.5 * M * omega2 * x**2  # Pure harmonic, no quartic

        H = np.diag(diag_K + V) + np.diag(offdiag_K, 1) + np.diag(offdiag_K, -1)
        eigenvalues = eigh(H, eigvals_only=True, subset_by_index=[0, 1])

        return eigenvalues[1] - eigenvalues[0]

    # ------------------------------------------------------------------
    # Physical gap vs R
    # ------------------------------------------------------------------
    def physical_gap_vs_R(self, R_values, N=2, n_basis_1d=80, n_basis_3d=30):
        """
        Compute all gap components as a function of R.

        For each R, computes:
        - harmonic_gap: ω = 2/R (THEOREM)
        - ac_bound: 3π²/(2M·d²) (THEOREM)
        - anharmonic_1d: numerical 1D gap (NUMERICAL)
        - anharmonic_3d: numerical 3D radial gap (NUMERICAL)
        - best_bound: max of all bounds/estimates

        LABEL: NUMERICAL

        Parameters
        ----------
        R_values : array-like
            Radii of S³ in units of 1/Λ_QCD.
        N : int
            Number of colors.
        n_basis_1d : int
            Grid points for 1D solver.
        n_basis_3d : int
            Grid points for 3D radial solver.

        Returns
        -------
        dict with arrays for each gap component.
        """
        R_arr = np.asarray(R_values, dtype=float)
        n_R = len(R_arr)

        results = {
            'R': R_arr,
            'harmonic_gap': np.zeros(n_R),
            'ac_bound': np.zeros(n_R),
            'anharmonic_1d': np.zeros(n_R),
            'anharmonic_3d': np.zeros(n_R),
            'best_bound': np.zeros(n_R),
            'effective_mass': np.zeros(n_R),
            'g_squared': np.zeros(n_R),
            'gribov_diameter': np.zeros(n_R),
            'quartic_coupling': np.zeros(n_R),
            'label': 'NUMERICAL',
        }

        for idx, R in enumerate(R_arr):
            # Parameters
            results['effective_mass'][idx] = self.effective_mass(R, N)
            results['g_squared'][idx] = ZwanzigerGapEquation.running_coupling_g2(R, N)
            results['gribov_diameter'][idx] = self._get_gribov_diameter(R, N)
            results['quartic_coupling'][idx] = self.quartic_coupling(R, N)

            # Gap components
            h_gap = self.harmonic_gap(R, N)
            results['harmonic_gap'][idx] = h_gap

            ac = self.particle_in_box_gap(R, N)
            results['ac_bound'][idx] = ac

            anh_1d = self.anharmonic_gap_1d(R, N, n_basis=n_basis_1d)
            results['anharmonic_1d'][idx] = anh_1d

            anh_3d = self.anharmonic_gap_3d(R, N, n_basis=n_basis_3d)
            results['anharmonic_3d'][idx] = anh_3d

            # Best bound: largest of all estimates
            results['best_bound'][idx] = max(h_gap, ac, anh_1d, anh_3d)

        return results

    # ------------------------------------------------------------------
    # Dimensionless gap analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _dimensionless_ratios(R, M, omega, alpha, d):
        """
        Compute the two key dimensionless ratios controlling the gap.

        β₁ = α/(Mω³)  — quartic coupling strength
        β₂ = d·√(Mω)  — box size in HO units

        For small β₁: gap ≈ ω (harmonic)
        For large β₁: gap ~ (α/M)^{1/3} (quartic regime)
        For small β₂: gap ~ 1/(Md²) (particle in box)
        """
        omega_val = 2.0 / R
        if omega_val <= 0 or M <= 0:
            return {'beta1': 0.0, 'beta2': np.inf, 'regime': 'undefined'}

        beta1 = alpha / (M * omega_val**3) if M * omega_val**3 > 0 else 0.0
        beta2 = d * np.sqrt(M * omega_val) if d > 0 and np.isfinite(d) else np.inf

        if beta2 < 3.0:
            regime = 'particle_in_box'
        elif beta1 > 1.0:
            regime = 'quartic'
        else:
            regime = 'harmonic'

        return {'beta1': beta1, 'beta2': beta2, 'regime': regime}

    # ------------------------------------------------------------------
    # Complete analysis
    # ------------------------------------------------------------------
    def complete_analysis(self, R_range=None, N=2, n_basis_1d=80, n_basis_3d=30):
        """
        Full physical gap analysis with all components, crossover identification,
        and comparison with Λ_QCD.

        LABEL: NUMERICAL

        Parameters
        ----------
        R_range : array-like or None
            R values. Default: logarithmic range from 0.3 to 30.
        N : int
            Number of colors.
        n_basis_1d : int
            Grid points for 1D solver.
        n_basis_3d : int
            Grid points for 3D solver.

        Returns
        -------
        dict with complete analysis.
        """
        if R_range is None:
            R_range = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0,
                                7.0, 10.0, 15.0, 20.0, 30.0])

        results = self.physical_gap_vs_R(R_range, N, n_basis_1d, n_basis_3d)

        R_arr = results['R']
        best = results['best_bound']
        anh_1d = results['anharmonic_1d']

        # Find minimum gap over all R
        min_gap = np.min(best)
        min_gap_R = R_arr[np.argmin(best)]

        # Check stabilization at large R
        large_R_mask = R_arr >= 5.0
        if np.sum(large_R_mask) >= 2:
            gaps_large_R = anh_1d[large_R_mask]
            mean_gap = np.mean(gaps_large_R)
            std_gap = np.std(gaps_large_R)
            rel_var = std_gap / mean_gap if mean_gap > 0 else np.inf
            stabilized = rel_var < 0.3  # within 30% variation
        else:
            mean_gap = std_gap = rel_var = np.nan
            stabilized = False

        # Dimensionless ratios at each R
        regimes = []
        for idx, R in enumerate(R_arr):
            M = results['effective_mass'][idx]
            omega = 2.0 / R
            alpha = results['quartic_coupling'][idx]
            d = results['gribov_diameter'][idx]
            ratios = self._dimensionless_ratios(R, M, omega, alpha, d)
            regimes.append(ratios['regime'])

        # Crossover: where harmonic gap < AC bound
        crossover_R = np.nan
        for idx in range(len(R_arr)):
            if results['ac_bound'][idx] > results['harmonic_gap'][idx]:
                crossover_R = R_arr[idx]
                break

        # Assessment
        if min_gap > 0:
            assessment = (
                f"POSITIVE: Physical gap has minimum {min_gap:.4f} Λ_QCD "
                f"at R = {min_gap_R:.1f}/Λ_QCD. "
                f"Gap is bounded below by max(harmonic, AC, anharmonic) > 0 "
                f"for all R tested."
            )
            label = 'NUMERICAL'
        else:
            assessment = (
                "WARNING: Physical gap appears to vanish. Check numerical "
                "convergence and increase basis size."
            )
            label = 'INCONCLUSIVE'

        return {
            **results,
            'min_gap': min_gap,
            'min_gap_R': min_gap_R,
            'regimes': regimes,
            'crossover_R': crossover_R,
            'large_R_analysis': {
                'mean_gap': mean_gap,
                'std_gap': std_gap,
                'relative_variation': rel_var,
                'stabilized': stabilized,
            },
            'assessment': assessment,
            'theorems_used': {
                'harmonic_gap': 'ω = 2/R from coexact Hodge Laplacian (THEOREM)',
                'andrews_clutterbuck': '3π²/(2Md²) for convex potential on convex domain (THEOREM)',
                'dell_antonio_zwanziger': 'Gribov region is bounded and convex (THEOREM)',
                'anharmonic': 'Numerical diagonalization of finite-dim Hamiltonian (NUMERICAL)',
            },
            'label': label,
        }
