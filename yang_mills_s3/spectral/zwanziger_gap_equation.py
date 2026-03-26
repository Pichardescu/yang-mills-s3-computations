"""
Zwanziger Gap Equation on S³(R) — Gribov parameter determination.

The Gribov-Zwanziger framework restricts the Yang-Mills functional integral
to the Gribov region Ω = {A : ∂·A = 0, M_FP ≥ 0}, introducing a mass
parameter γ (the Gribov parameter) determined self-consistently by the
"horizon condition" (gap equation).

ON FLAT SPACE R^d:
    The gap equation is:
        d(N²-1) = g²N ∫ d^dk/(2π)^d × d / (k² + γ⁴/k²)

ON S³(R):
    The integral becomes a discrete sum over the spectrum of the scalar
    Laplacian on S³:
        λ_l = l(l+2)/R²  with multiplicity (l+1)²   (l = 0, 1, 2, ...)

    The horizon condition on S³, properly UV-renormalized and volume-normalized
    (trace per unit volume), is:

        (N²-1) = g²(R) × N × (1/V) × Σ_{l=1}^∞ (l+1)² × σ(γ, λ_l)

    where:
        V = Vol(S³) = 2π²R³
        σ(γ, λ) = γ⁴ / (λ(λ² + γ⁴))   (UV-subtracted kernel)

    The volume normalization (1/V) is essential: it converts the trace over
    eigenmodes into a trace per unit volume, matching the flat-space integral
    ∫ d³k/(2π)³ in the R → ∞ limit. Without it, the sum grows as R³ and
    γ(R) → 0 — an artifact of the growing number of modes, not physics.

    The UV subtraction removes the divergent Σ mult/λ_l piece, leaving a
    convergent sum: at large l, σ ~ γ⁴/λ³ and Σ (l+1)²/λ³ ~ R⁶ Σ 1/l⁴.

KEY RESULT: γ(R) → constant ≈ 2.15 Λ_QCD as R → ∞.
    The Gribov parameter stabilizes at a finite, R-independent value.
    The effective gluon mass m_g = √2 γ ≈ 3.0 Λ_QCD persists at all R.

LABEL: NUMERICAL (self-consistent gap equation solved numerically)

References:
    - Gribov 1978: Quantization of non-Abelian gauge theories
    - Zwanziger 1989: Local and renormalizable action from the Gribov horizon
    - Vandersickel & Zwanziger 2012: Review of the Gribov-Zwanziger framework
    - van Baal 1992: Gribov copies on compact spaces (S³, T³)
"""

import numpy as np
from scipy.optimize import brentq


class ZwanzigerGapEquation:
    """
    Solves the Zwanziger gap equation on S³(R) for the Gribov parameter γ.

    Units: Λ_QCD = 1. All dimensionful quantities in units of Λ_QCD.
    R is in units of 1/Λ_QCD, γ in units of Λ_QCD, etc.

    The gap equation is UV-renormalized (subtracted kernel) and volume-
    normalized (trace per unit volume) to match the flat-space limit.
    """

    # ------------------------------------------------------------------
    # Running coupling g²(μ) at 1-loop with smooth IR saturation
    # ------------------------------------------------------------------
    @staticmethod
    def running_coupling_g2(R, N=2):
        """
        1-loop running coupling g²(μ) at scale μ = 1/R with smooth IR behavior.

        Uses a single analytic formula:
            g²(R) = 1 / (1/g²_max + b₀ × ln(1 + 1/(R²Λ²)))

        Properties:
        - Matches 1/(b₀ ln(μ²/Λ²)) in the UV (μ >> Λ, i.e., small R)
        - Saturates at g²_max = 4π in the IR (μ << Λ, i.e., large R)
        - Monotonically increasing in R

        LABEL: NUMERICAL

        Parameters
        ----------
        R : float
            Radius of S³ in units of 1/Λ_QCD.
        N : int
            Number of colors. Default 2.

        Returns
        -------
        float
            g²(μ = 1/R), smoothly interpolated.
        """
        b0 = 11 * N / (48 * np.pi**2)
        # IR saturation value g^2_max = 4*pi (~12.57)
        # ASSUMPTION: NUMERICAL, not derived from first principles.
        # Lattice evidence: Cornwall 1982, Aguilar-Papavassiliou 2008,
        # Bogolubsky et al. 2009 show alpha_s(0) ~ 0.9-1.0, giving
        # g^2(0) ~ 11-13. Our choice g^2_max = 4*pi ~ 12.57 is within
        # this range. The precise value affects gamma* linearly:
        # gamma* = (N^2-1)*4*pi*sqrt(2)/(g^2_max*N), so a 20% change
        # in g^2_max gives a 20% change in gamma* and hence in the
        # mass gap estimate.
        g2_max = 4 * np.pi  # NUMERICAL assumption: IR saturation value

        log_term = np.log(1.0 + 1.0 / R**2)
        return 1.0 / (1.0 / g2_max + b0 * log_term)

    # ------------------------------------------------------------------
    # Scalar Laplacian eigenvalue on S³
    # ------------------------------------------------------------------
    @staticmethod
    def laplacian_eigenvalue(l, R):
        """
        Eigenvalue of the scalar Laplacian on S³(R).

        λ_l = l(l+2)/R²  for l = 0, 1, 2, ...

        Parameters
        ----------
        l : int
            Angular momentum quantum number (l ≥ 0).
        R : float
            Radius of S³.

        Returns
        -------
        float
            λ_l = l(l+2)/R².
        """
        return l * (l + 2) / R**2

    # ------------------------------------------------------------------
    # Scalar Laplacian multiplicity on S³
    # ------------------------------------------------------------------
    @staticmethod
    def laplacian_multiplicity(l):
        """
        Multiplicity of the l-th eigenvalue of the scalar Laplacian on S³.

        mult(l) = (l+1)²

        Parameters
        ----------
        l : int
            Angular momentum quantum number.

        Returns
        -------
        int
            (l+1)².
        """
        return (l + 1)**2

    # ------------------------------------------------------------------
    # Volume of S³
    # ------------------------------------------------------------------
    @staticmethod
    def volume_s3(R):
        """Volume of S³(R): V = 2π²R³."""
        return 2.0 * np.pi**2 * R**3

    # ------------------------------------------------------------------
    # Gribov-modified ghost propagator
    # ------------------------------------------------------------------
    @staticmethod
    def gribov_propagator(l, gamma, R):
        """
        The Gribov-modified ghost propagator at angular momentum l.

        G_ghost(l) = 1 / (λ_l + γ⁴/λ_l)

        Parameters
        ----------
        l : int
            Angular momentum quantum number (l ≥ 1).
        gamma : float
            Gribov parameter.
        R : float
            Radius of S³.

        Returns
        -------
        float
            G_ghost(l) = 1 / (λ_l + γ⁴/λ_l).
        """
        lam = ZwanzigerGapEquation.laplacian_eigenvalue(l, R)
        if lam <= 0:
            raise ValueError(f"λ_l must be > 0 for l ≥ 1, got l={l}")
        return 1.0 / (lam + gamma**4 / lam)

    # ------------------------------------------------------------------
    # UV-subtracted kernel
    # ------------------------------------------------------------------
    @staticmethod
    def subtracted_kernel(gamma, lam_l):
        """
        UV-subtracted ghost propagator kernel.

        σ(γ, λ) = 1/λ - 1/(λ + γ⁴/λ) = γ⁴ / (λ(λ² + γ⁴))

        At large λ: σ ~ γ⁴/λ³ → 0  (UV finite)
        At small λ: σ ~ 1/λ         (full IR enhancement)

        Parameters
        ----------
        gamma : float
            Gribov parameter.
        lam_l : float
            Eigenvalue λ_l.

        Returns
        -------
        float
            σ(γ, λ_l).
        """
        return gamma**4 / (lam_l * (lam_l**2 + gamma**4))

    # ------------------------------------------------------------------
    # Gap equation residual (UV-renormalized, volume-normalized)
    # ------------------------------------------------------------------
    @staticmethod
    def gap_equation_residual(gamma, R, N=2, l_max=500):
        """
        Residual of the Zwanziger gap equation on S³.

        The volume-normalized, UV-subtracted horizon condition:

            (N²-1) = g²(R) × N × (1/V) × Σ_{l=1}^{l_max} (l+1)² × σ(γ, λ_l)

        where V = 2π²R³ and σ(γ, λ) = γ⁴/(λ(λ² + γ⁴)).

        Returns LHS - RHS. The gap equation is satisfied when this is 0.

        Parameters
        ----------
        gamma : float
            Gribov parameter (candidate value, in Λ_QCD units).
        R : float
            Radius of S³ in Λ_QCD units.
        N : int
            Number of colors. Default 2.
        l_max : int
            UV cutoff for the spectral sum. Default 500.

        Returns
        -------
        float
            LHS - RHS of the gap equation.
        """
        dim_adj = N**2 - 1
        LHS = float(dim_adj)

        if gamma <= 0:
            return LHS

        g2 = ZwanzigerGapEquation.running_coupling_g2(R, N)
        V = ZwanzigerGapEquation.volume_s3(R)

        # Spectral sum with UV-subtracted kernel, volume-normalized
        spectral_sum = 0.0
        gamma4 = gamma**4
        for l in range(1, l_max + 1):
            lam_l = l * (l + 2) / R**2
            mult_l = (l + 1)**2
            spectral_sum += mult_l * gamma4 / (lam_l * (lam_l**2 + gamma4))

        RHS = g2 * N * spectral_sum / V
        return LHS - RHS

    # ------------------------------------------------------------------
    # Solve for γ(R)
    # ------------------------------------------------------------------
    @staticmethod
    def solve_gamma(R, N=2, l_max=500):
        """
        Solve the Zwanziger gap equation for the Gribov parameter γ at radius R.

        Uses Brent's method on the residual function.
        At γ = 0: residual = N²-1 > 0.
        As γ → ∞: RHS grows (sum → Σ mult/λ), so residual → -∞.
        By IVT, a root exists.

        Parameters
        ----------
        R : float
            Radius of S³ in Λ_QCD units.
        N : int
            Number of colors. Default 2.
        l_max : int
            UV cutoff for the spectral sum. Default 500.

        Returns
        -------
        float
            γ(R) satisfying the gap equation, or NaN if no solution found.
        """
        # Find upper bracket where residual < 0
        gamma_max = 1.0
        for _ in range(60):
            res_high = ZwanzigerGapEquation.gap_equation_residual(
                gamma_max, R, N, l_max
            )
            if res_high < 0:
                break
            gamma_max *= 2.0
        else:
            return float('nan')

        # Find lower bracket where residual > 0
        gamma_min = gamma_max
        while gamma_min > 1e-15:
            gamma_min /= 2.0
            res_low = ZwanzigerGapEquation.gap_equation_residual(
                gamma_min, R, N, l_max
            )
            if res_low > 0:
                break
        else:
            return float('nan')

        try:
            gamma_sol = brentq(
                ZwanzigerGapEquation.gap_equation_residual,
                gamma_min, gamma_max,
                args=(R, N, l_max),
                xtol=1e-12,
                rtol=1e-12,
                maxiter=200,
            )
            return gamma_sol
        except ValueError:
            return float('nan')

    # ------------------------------------------------------------------
    # Gluon mass from Gribov parameter
    # ------------------------------------------------------------------
    @staticmethod
    def gluon_mass_from_gamma(gamma, R):
        """
        Effective gluon mass from the Gribov parameter.

        In the GZ framework, the gluon propagator D(k²) = k²/(k⁴ + γ⁴)
        has complex poles at k² = ±iγ², giving m_g = √2 × γ.

        LABEL: NUMERICAL

        Parameters
        ----------
        gamma : float
            Gribov parameter (in Λ_QCD units).
        R : float
            Radius of S³.

        Returns
        -------
        float
            Effective gluon mass m_g = √2 × γ (in Λ_QCD units).
        """
        return np.sqrt(2) * gamma

    # ------------------------------------------------------------------
    # γ(R) for a range of R values
    # ------------------------------------------------------------------
    @staticmethod
    def gamma_vs_R(R_values, N=2, l_max=500):
        """
        Compute γ(R) for an array of R values.

        Parameters
        ----------
        R_values : array-like
            Radii of S³ in Λ_QCD units.
        N : int
            Number of colors. Default 2.
        l_max : int
            UV cutoff for the spectral sum. Default 500.

        Returns
        -------
        dict with R, gamma, gluon_mass, geometric_gap, g_squared arrays.
        """
        R_arr = np.asarray(R_values, dtype=float)
        gamma_arr = np.zeros_like(R_arr)
        g2_arr = np.zeros_like(R_arr)
        mg_arr = np.zeros_like(R_arr)
        geo_gap_arr = np.zeros_like(R_arr)

        for i, R in enumerate(R_arr):
            gamma_arr[i] = ZwanzigerGapEquation.solve_gamma(R, N, l_max)
            g2_arr[i] = ZwanzigerGapEquation.running_coupling_g2(R, N)
            if np.isfinite(gamma_arr[i]):
                mg_arr[i] = ZwanzigerGapEquation.gluon_mass_from_gamma(
                    gamma_arr[i], R
                )
            else:
                mg_arr[i] = float('nan')
            geo_gap_arr[i] = 2.0 / R

        return {
            'R': R_arr,
            'gamma': gamma_arr,
            'gamma_over_Lambda': gamma_arr,
            'gluon_mass': mg_arr,
            'geometric_gap': geo_gap_arr,
            'g_squared': g2_arr,
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # Complete analysis
    # ------------------------------------------------------------------
    @staticmethod
    def complete_analysis(R_range=None, N=2):
        """
        Full Zwanziger gap equation analysis.

        Computes γ(R), m_g(R), comparison with geometric gap 2/R,
        identifies crossover point, and assesses R → ∞ behavior.

        Parameters
        ----------
        R_range : array-like or None
            R values to scan. Default: logarithmic range from 0.1 to 100.
        N : int
            Number of colors. Default 2.

        Returns
        -------
        dict with complete analysis results, labeled NUMERICAL.
        """
        if R_range is None:
            R_range = np.concatenate([
                np.array([0.1, 0.2, 0.5]),
                np.arange(1.0, 11.0, 1.0),
                np.array([15, 20, 30, 50, 75, 100]),
            ])

        results = ZwanzigerGapEquation.gamma_vs_R(R_range, N, l_max=500)

        gamma = results['gamma']
        mg = results['gluon_mass']
        geo = results['geometric_gap']
        R_arr = results['R']

        # Find crossover: first R where gluon mass > geometric gap
        crossover_R = float('nan')
        for i in range(len(R_arr)):
            if np.isfinite(mg[i]) and np.isfinite(geo[i]):
                if mg[i] > geo[i]:
                    crossover_R = R_arr[i]
                    break

        # Assess stabilization at large R
        large_R_mask = R_arr >= 10.0
        if np.any(large_R_mask):
            gamma_large_R = gamma[large_R_mask & np.isfinite(gamma)]
            if len(gamma_large_R) >= 2:
                mean_gamma = np.mean(gamma_large_R)
                std_gamma = np.std(gamma_large_R)
                relative_variation = (
                    std_gamma / mean_gamma if mean_gamma > 0 else float('inf')
                )
                stabilized = relative_variation < 0.1
            else:
                mean_gamma = std_gamma = relative_variation = float('nan')
                stabilized = False
        else:
            mean_gamma = std_gamma = relative_variation = float('nan')
            stabilized = False

        return {
            'R': R_arr,
            'gamma': gamma,
            'gluon_mass': mg,
            'geometric_gap': geo,
            'g_squared': results['g_squared'],
            'crossover_R': crossover_R,
            'large_R_analysis': {
                'mean_gamma': mean_gamma,
                'std_gamma': std_gamma,
                'relative_variation': relative_variation,
                'stabilized': stabilized,
            },
            'N': N,
            'gauge_group': f'SU({N})',
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # Convergence check of the spectral sum
    # ------------------------------------------------------------------
    @staticmethod
    def convergence_check(gamma, R, N=2, l_max_values=None):
        """
        Check convergence of the spectral sum as a function of l_max.

        Parameters
        ----------
        gamma : float
            Gribov parameter.
        R : float
            Radius.
        N : int
            Number of colors.
        l_max_values : list or None
            Values of l_max to test. Default: [50, 100, 200, 500, 1000].

        Returns
        -------
        dict with l_max_values, residuals, and converged flag.
        """
        if l_max_values is None:
            l_max_values = [50, 100, 200, 500, 1000]

        residuals = []
        for lm in l_max_values:
            res = ZwanzigerGapEquation.gap_equation_residual(
                gamma, R, N, lm
            )
            residuals.append(res)

        return {
            'l_max_values': l_max_values,
            'residuals': residuals,
            'converged': (
                len(residuals) >= 2 and
                abs(residuals[-1] - residuals[-2]) <
                0.01 * (abs(residuals[-1]) + 1e-10)
            ),
        }
