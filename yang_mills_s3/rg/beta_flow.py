"""
Perturbative RG Flow (Estimate 6) -- beta-function on S^3.

Implements the full perturbative map (g_j, nu_j, z_j) -> (g_{j+1}, nu_{j+1}, z_{j+1})
for SU(N_c) Yang-Mills on S^3(R), adapted from Balaban Paper 8.

The key classes are:

    BetaFunction          -- 1-loop and 2-loop beta-function coefficients with
                             curvature corrections from the S^3 heat kernel.
    MassRenormalization   -- Mass parameter (nu) flow, including critical initial
                             condition nu_c that drives nu_N -> 0.
    WaveFunctionRenormalization -- Wavefunction (z) flow from vacuum polarization.
    CurvatureCorrections  -- O((L^j/R)^2) corrections from S^3 curvature at each
                             RG scale j.
    PerturbativeRGFlow    -- Full combined flow assembling beta + nu + z.
    AsymptoticFreedomVerifier -- Verifies g^2 decreasing toward UV, extracts
                                Lambda_QCD, checks Landau pole absence.

Mathematical framework:
    At RG scale j (with blocking factor M and lattice spacing a_j = a_0 * M^{-j}),
    the coupling evolves as:

        1/g^2_{j+1} = 1/g^2_j + beta_0 * ln(M^2) + beta_1 * g^2_j * ln(M^2)
                      + delta_curv(j)

    where beta_0 = 11 N_c / (3 * 16 pi^2)   [1-loop]
          beta_1 = 34 N_c^2 / (3 * (16 pi^2)^2) [2-loop]
          delta_curv(j) = c_curv * (M^{-j}/R)^2  [curvature correction]

    On S^3, the spectral gap lambda_1 = 4/R^2 provides a natural IR mass,
    eliminating the need for delicate zero-mode subtraction (unlike T^4).

Labels:
    THEOREM:     Proven rigorously under stated assumptions
    PROPOSITION: Proven with reasonable but unverified assumptions
    NUMERICAL:   Supported by computation, no formal proof
    CONJECTURE:  Motivated by evidence, not proved

Physical parameters (defaults):
    R = 2.2 fm, g^2 = 6.28, hbar*c = 197.327 MeV*fm
    Lambda_QCD = 200 MeV, N_c = 2 (SU(2)), M = 2, N_scales ~ 7

References:
    - Balaban (1984-89), Paper 8: small-field effective action, beta-function
    - Gross & Wilczek (1973), Politzer (1973): asymptotic freedom
    - Luscher (1982): finite-volume effects in gauge theories
    - Bauerschmidt-Brydges-Slade (2019): modern RG framework (BBS)
    - heat_kernel_slices.py: covariance decomposition infrastructure
    - first_rg_step.py: one-step RG contraction (existing)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from .heat_kernel_slices import (
    coexact_eigenvalue,
    coexact_multiplicity,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    LAMBDA_QCD_MEV,
)


# ======================================================================
# Physical constants
# ======================================================================

# Re-export for local use
_HBAR_C = HBAR_C_MEV_FM       # 197.327 MeV*fm
_R_PHYS = R_PHYSICAL_FM       # 2.2 fm
_LAMBDA_QCD = LAMBDA_QCD_MEV  # 200 MeV


# ======================================================================
# BetaFunction
# ======================================================================

class BetaFunction:
    """
    Beta-function coefficients for pure SU(N_c) Yang-Mills on S^3(R).

    1-loop:  beta_0 = 11 * N_c / (3 * 16 pi^2)
    2-loop:  beta_1 = 34 * N_c^2 / (3 * (16 pi^2)^2)

    Curvature correction from S^3 heat kernel:
        delta_beta(j) = c_curv * (L^j / R)^2

    where L^j = a_0 * M^j is the lattice spacing at scale j (IR counting)
    and c_curv is computable from the Seeley-DeWitt coefficient a_1 = Ric/6.

    THEOREM: beta_0, beta_1 are the standard Gross-Wilczek-Politzer values.
    NUMERICAL: curvature corrections vanish at UV scales (j << log_M(R/a)).

    Parameters
    ----------
    N_c : int
        Number of colours (N_c >= 2).
    R : float
        S^3 radius in fm (default 2.2).
    M : float
        Blocking factor (default 2).
    """

    def __init__(self, N_c: int = 2, R: float = _R_PHYS, M: float = 2.0):
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")
        if M <= 1:
            raise ValueError(f"Blocking factor M must be > 1, got {M}")

        self.N_c = N_c
        self.R = R
        self.M = M

        # Pre-compute standard coefficients (pure YM, no fermions)
        self._beta0 = 11.0 * N_c / (3.0 * 16.0 * np.pi ** 2)
        self._beta1 = 34.0 * N_c ** 2 / (3.0 * (16.0 * np.pi ** 2) ** 2)

        # Seeley-DeWitt a_1 coefficient on S^3:
        #   a_1 = Ric / 6  where Ric_{ij} = (2/R^2) g_{ij}  on S^3(R)
        #   For the trace: Tr(a_1) = (dim_adj) * (d-1) * 2 / (6 R^2)
        #     = (N_c^2 - 1) * 3 * 2 / (6 R^2) = (N_c^2 - 1) / R^2
        # The coefficient c_curv in the beta correction:
        #   delta_beta(j) = c_curv * (scale_j / R)^2
        # where scale_j = M^{-j} (in units of R) represents the ratio of
        # lattice spacing to radius.
        # From the heat-kernel expansion, c_curv = beta_0 / 6 (leading order).
        self._c_curv = self._beta0 / 6.0

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    @property
    def beta0(self) -> float:
        """
        1-loop coefficient beta_0 = 11 N_c / (3 * 16 pi^2).

        THEOREM (Gross-Wilczek-Politzer 1973).
        """
        return self._beta0

    @property
    def beta1(self) -> float:
        """
        2-loop coefficient beta_1 = 34 N_c^2 / (3 * (16 pi^2)^2).

        THEOREM (Caswell 1974, Jones 1974).
        """
        return self._beta1

    @property
    def c_curv(self) -> float:
        """Curvature correction coefficient from Seeley-DeWitt a_1. NUMERICAL."""
        return self._c_curv

    # ------------------------------------------------------------------
    # Flow maps
    # ------------------------------------------------------------------

    def one_loop(self, g2: float) -> float:
        """
        One-loop change in 1/g^2 per RG step.

            delta(1/g^2) = beta_0 * ln(M^2)

        THEOREM.

        Parameters
        ----------
        g2 : float
            Current coupling g^2 (unused at 1-loop, but kept for API
            consistency -- the 1-loop shift is independent of g^2).

        Returns
        -------
        float
            Increment delta(1/g^2).
        """
        return self._beta0 * np.log(self.M ** 2)

    def two_loop(self, g2: float) -> float:
        """
        Two-loop change in 1/g^2 per RG step.

            delta(1/g^2) = beta_0 * ln(M^2) + beta_1 * g^2 * ln(M^2)

        The second term is O(g^2) relative to the first, hence suppressed
        in the UV where g^2 is small.

        THEOREM (universal 2-loop coefficient).

        Parameters
        ----------
        g2 : float
            Current coupling g^2.

        Returns
        -------
        float
            Increment delta(1/g^2) at 2-loop.
        """
        ln_M2 = np.log(self.M ** 2)
        return self._beta0 * ln_M2 + self._beta1 * g2 * ln_M2

    def with_curvature(self, g2: float, scale_j: int, R: Optional[float] = None) -> float:
        """
        Full 1-loop beta with S^3 curvature correction at scale j.

            delta(1/g^2) = beta_0 * ln(M^2) * [1 + c_curv * (M^{-scale_j} / R)^2]

        The curvature correction comes from the difference between the S^3
        heat kernel and the flat-space heat kernel. It enters as:

            K_{S^3}(t,x,x) = K_{flat}(t) * [1 + (Ric/6) * t + O(t^2)]

        At RG scale j, the relevant proper time is t ~ M^{-2j}, so the
        correction is O((M^{-j}/R)^2) which vanishes exponentially at
        UV scales.

        NUMERICAL.

        Parameters
        ----------
        g2 : float
            Current coupling g^2 (unused at 1-loop order for the base term).
        scale_j : int
            RG scale index j.
        R : float or None
            S^3 radius. If None, uses self.R.

        Returns
        -------
        float
            delta(1/g^2) with curvature correction.
        """
        if R is None:
            R = self.R
        ln_M2 = np.log(self.M ** 2)
        # Ratio of lattice spacing at scale j to the radius
        ratio = self.M ** (-scale_j) / R
        correction = 1.0 + self._c_curv * ratio ** 2
        return self._beta0 * ln_M2 * correction


# ======================================================================
# MassRenormalization
# ======================================================================

class MassRenormalization:
    """
    Mass parameter (nu) flow for Yang-Mills on S^3(R).

    The mass parameter nu tracks the deviation from criticality. Under
    one RG step:

        nu_{j+1} = nu_j + delta_nu(g_j, j)

    where delta_nu is the one-loop self-energy mass shift.

    On S^3, the spectral gap lambda_1 = 4/R^2 provides a natural IR
    mass. No delicate zero-mode subtraction is needed (unlike T^4).

    The critical initial condition nu_c is chosen so that after N RG
    steps, nu_N = 0 (or O(g_N^4)):

        nu_c = - sum_{j=0}^{N-1} delta_nu(g_j) * prod_{k=0}^{j-1} (1 + d_nu/d_nu|_k)

    NUMERICAL: one-loop mass shift from self-energy on S^3.

    Parameters
    ----------
    N_c : int
        Number of colours.
    R : float
        S^3 radius in fm.
    M : float
        Blocking factor.
    """

    def __init__(self, N_c: int = 2, R: float = _R_PHYS, M: float = 2.0):
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")
        if M <= 1:
            raise ValueError(f"M must be > 1, got {M}")

        self.N_c = N_c
        self.R = R
        self.M = M
        self.dim_adj = N_c ** 2 - 1
        self._C2 = float(N_c)  # Quadratic Casimir C_2(adj) = N_c

        # Spectral gap of the coexact Laplacian on S^3(R)
        self.lambda1 = 4.0 / R ** 2

    def one_loop_mass_shift(self, g2: float, scale_j: int,
                            R: Optional[float] = None) -> float:
        """
        One-loop mass shift delta_nu at RG scale j.

        From the gluon self-energy on S^3, the mass counterterm is:

            delta_nu = g^2 * C_2(adj) / (16 pi^2)
                       * sum_{k in shell_j} d_k / lambda_k  / Vol(S^3)

        We approximate the shell sum by the continuum integral in the
        shell [M^j / R, M^{j+1} / R]:

            delta_nu ~ g^2 * C_2 / (16 pi^2) * (3 / (4 pi^2 R))

        times a scale-dependent factor.

        On S^3, this is FINITE at every scale (no quadratic divergence),
        unlike flat space where it diverges as Lambda^2.

        NUMERICAL.

        Parameters
        ----------
        g2 : float
            Coupling g^2 at scale j.
        scale_j : int
            RG scale index j.
        R : float or None
            S^3 radius. If None, uses self.R.

        Returns
        -------
        float
            Mass shift delta_nu (in units of 1/R^2).
        """
        if R is None:
            R = self.R
        vol = 2.0 * np.pi ** 2 * R ** 3

        # Shell momentum range
        p_lo = self.M ** scale_j / R
        p_hi = self.M ** (scale_j + 1) / R

        # Continuum approximation to the shell sum:
        #   sum_{k in shell} d_k / lambda_k ~ Vol * integral_{p_lo}^{p_hi}
        #                                       (4 pi p^2) / (2 pi)^3 * (1/p^2) dp
        #   = Vol / (2 pi^2) * (p_hi - p_lo)
        shell_integral = (p_hi - p_lo) / (2.0 * np.pi ** 2)

        # One-loop self-energy coefficient
        coeff = g2 * self._C2 / (16.0 * np.pi ** 2)

        return coeff * shell_integral

    def anomalous_dimension_nu(self, g2: float) -> float:
        """
        Anomalous dimension of the mass parameter.

        At one loop:
            gamma_nu = d(delta_nu)/d(nu) ~ g^2 * C_2 / (16 pi^2) * c_dim

        where c_dim is a numerical constant from the Feynman diagram.
        For pure YM in Landau gauge: c_dim = 13/6.

        NUMERICAL.

        Parameters
        ----------
        g2 : float
            Coupling g^2.

        Returns
        -------
        float
            gamma_nu (dimensionless).
        """
        c_dim = 13.0 / 6.0  # Landau gauge
        return g2 * self._C2 * c_dim / (16.0 * np.pi ** 2)

    def critical_initial(self, g2_bare: float, N_scales: int,
                         R: Optional[float] = None) -> float:
        """
        Critical initial mass parameter nu_c that drives nu_N -> 0.

        Uses the recursion:
            nu_c = - sum_{j=0}^{N-1} delta_nu(g_j)
                    * prod_{k=0}^{j-1} (1 + gamma_nu(g_k))^{-1}

        We approximate by setting g_j ~ g_bare at all scales (valid for
        small g_bare). For a more accurate result, use the full RG flow.

        NUMERICAL.

        Parameters
        ----------
        g2_bare : float
            Bare coupling g^2.
        N_scales : int
            Number of RG steps.
        R : float or None
            S^3 radius. If None, uses self.R.

        Returns
        -------
        float
            Critical initial condition nu_c (in 1/R^2 units).
        """
        if R is None:
            R = self.R

        gamma = self.anomalous_dimension_nu(g2_bare)
        nu_c = 0.0
        propagation_factor = 1.0

        for j in range(N_scales):
            d_nu = self.one_loop_mass_shift(g2_bare, j, R)
            nu_c -= d_nu / propagation_factor
            propagation_factor *= (1.0 + gamma)

        return nu_c


# ======================================================================
# WaveFunctionRenormalization
# ======================================================================

class WaveFunctionRenormalization:
    """
    Wavefunction renormalization (z) flow from vacuum polarization.

    The z parameter tracks the field-strength renormalization:

        z_{j+1} = z_j + delta_z(g_j)

    In d=4, z is a marginal coupling and flows logarithmically.

    From the vacuum polarization diagram in Landau gauge:
        delta_z = g^2 * C_2(adj) * (13 - 3*xi) / (6 * 16 pi^2) * ln(M^2)

    For Landau gauge (xi = 0):
        delta_z = g^2 * C_2(adj) * 13 / (6 * 16 pi^2) * ln(M^2)

    NUMERICAL.

    Parameters
    ----------
    N_c : int
        Number of colours.
    M : float
        Blocking factor.
    """

    def __init__(self, N_c: int = 2, M: float = 2.0):
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        if M <= 1:
            raise ValueError(f"M must be > 1, got {M}")

        self.N_c = N_c
        self.M = M
        self.dim_adj = N_c ** 2 - 1
        self._C2 = float(N_c)

    def one_loop_z_shift(self, g2: float, scale_j: int) -> float:
        """
        One-loop wavefunction renormalization shift at scale j.

            delta_z = g^2 * C_2(adj) * 13 / (6 * 16 pi^2) * ln(M^2)

        This is the standard Landau-gauge vacuum polarization result.
        The scale_j dependence enters only through g^2 (which runs),
        not through a separate geometric factor, because the vacuum
        polarization is a marginal (logarithmic) effect.

        NUMERICAL.

        Parameters
        ----------
        g2 : float
            Coupling g^2 at scale j.
        scale_j : int
            RG scale index (for documentation; the shift depends on g^2
            only, not explicitly on j at one-loop).

        Returns
        -------
        float
            delta_z (dimensionless).
        """
        xi_factor = 13.0 / 6.0  # Landau gauge
        ln_M2 = np.log(self.M ** 2)
        return g2 * self._C2 * xi_factor * ln_M2 / (16.0 * np.pi ** 2)


# ======================================================================
# CurvatureCorrections
# ======================================================================

class CurvatureCorrections:
    """
    O((L^j/R)^2) corrections from S^3 curvature at each RG scale.

    These arise from the difference between the heat kernel on S^3 and
    flat space:

        K_{S^3}(t, x, x) = K_{flat}(t) * [1 + a_1 * t + a_2 * t^2 + ...]

    where a_1 = Ric/6 = 1/(3 R^2) (scalar Ricci for S^3(R): Ric = 6/R^2,
    so for the 1-form Laplacian with Weitzenbock: a_1 = Ric/6 in trace).

    At RG scale j, the effective proper time is t ~ M^{-2j}, so
    corrections scale as (M^{-j}/R)^2.

    Key property: corrections vanish exponentially at UV scales
    (j << log_M(R/a)) where the lattice spacing is much smaller than R.

    NUMERICAL.

    Parameters
    ----------
    R : float
        S^3 radius in fm.
    M : float
        Blocking factor.
    N_scales : int
        Number of RG steps.
    """

    def __init__(self, R: float = _R_PHYS, M: float = 2.0, N_scales: int = 7):
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")
        if M <= 1:
            raise ValueError(f"M must be > 1, got {M}")
        if N_scales < 1:
            raise ValueError(f"N_scales must be >= 1, got {N_scales}")

        self.R = R
        self.M = M
        self.N_scales = N_scales

        # Seeley-DeWitt a_1 for the scalar Laplacian on S^3:
        #   Ric(S^3(R)) = 6/R^2  (scalar curvature)
        #   a_1 = Ric / 6 = 1/R^2
        # For the 1-form Laplacian (Hodge), the Weitzenbock identity
        # introduces an additional Ricci term:
        #   Delta_1 = nabla^* nabla + Ric  =>  a_1 = 1/(3R^2)
        self._a1 = 1.0 / (3.0 * R ** 2)

    def _scale_ratio(self, j: int) -> float:
        """Ratio (lattice spacing at scale j) / R = M^{-j} / R."""
        return self.M ** (-j) / self.R

    def propagator_correction(self, j: int, R: Optional[float] = None) -> float:
        """
        Curvature correction to the propagator at scale j.

        The free propagator 1/(p^2 + m^2) receives a correction
        from the a_1 heat-kernel coefficient:

            delta_C(j) = a_1 * M^{-2j} / (p^2 + m^2)^2
                       ~ (1/(3R^2)) * (M^{-j})^2

        NUMERICAL.

        Parameters
        ----------
        j : int
            RG scale index.
        R : float or None
            Override radius.

        Returns
        -------
        float
            Relative correction delta_C / C (dimensionless).
        """
        if R is None:
            R = self.R
        a1 = 1.0 / (3.0 * R ** 2)
        return a1 * self.M ** (-2 * j)

    def vertex_correction(self, j: int, R: Optional[float] = None) -> float:
        """
        Curvature correction to the 3-gluon vertex at scale j.

        The vertex receives corrections from covariant derivatives on S^3:
            nabla_S^3 = nabla_flat + Christoffel
        giving corrections proportional to the connection, which scales
        as 1/R.

        At scale j, the vertex correction is:
            delta_V(j) ~ (1/R) * M^{-j} * (vertex_flat)
                       = (M^{-j}/R) * V_flat

        Squaring and dividing by V_flat^2 gives relative correction
        of order (M^{-j}/R)^2.

        NUMERICAL.

        Parameters
        ----------
        j : int
            RG scale index.
        R : float or None
            Override radius.

        Returns
        -------
        float
            Relative correction (dimensionless).
        """
        if R is None:
            R = self.R
        ratio = self.M ** (-j) / R
        return ratio ** 2

    def total_correction(self, j: int, R: Optional[float] = None) -> float:
        """
        Total curvature correction at scale j (sum of propagator + vertex).

        NUMERICAL.

        Parameters
        ----------
        j : int
            RG scale index.
        R : float or None
            Override radius.

        Returns
        -------
        float
            Total relative correction (dimensionless).
        """
        return self.propagator_correction(j, R) + self.vertex_correction(j, R)

    def all_corrections(self, R: Optional[float] = None) -> List[float]:
        """
        Total curvature corrections at every scale j = 0, ..., N_scales-1.

        NUMERICAL.

        Returns
        -------
        list of float
            Corrections at each scale.
        """
        return [self.total_correction(j, R) for j in range(self.N_scales)]


# ======================================================================
# PerturbativeRGFlow
# ======================================================================

class PerturbativeRGFlow:
    """
    Full perturbative RG flow combining beta + nu + z on S^3(R).

    Assembles the three running couplings into a single flow:
        (g^2_j, nu_j, z_j) --step--> (g^2_{j+1}, nu_{j+1}, z_{j+1})

    Convention: j=0 is IR (scale ~ R), j=N_scales-1 is UV (scale ~ a).
    The flow is run from j=0 to j=N_scales-1 (IR to UV) to track how
    the coupling decreases toward high energy.

    NUMERICAL.

    Parameters
    ----------
    N_c : int
        Number of colours.
    R : float
        S^3 radius in fm.
    M : float
        Blocking factor.
    N_scales : int
        Number of RG steps.
    """

    def __init__(self, N_c: int = 2, R: float = _R_PHYS, M: float = 2.0,
                 N_scales: int = 7):
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")
        if M <= 1:
            raise ValueError(f"M must be > 1, got {M}")
        if N_scales < 1:
            raise ValueError(f"N_scales must be >= 1, got {N_scales}")

        self.N_c = N_c
        self.R = R
        self.M = M
        self.N_scales = N_scales

        self.beta = BetaFunction(N_c, R, M)
        self.mass = MassRenormalization(N_c, R, M)
        self.wf = WaveFunctionRenormalization(N_c, M)
        self.curvature = CurvatureCorrections(R, M, N_scales)

    # ------------------------------------------------------------------
    # Single RG step
    # ------------------------------------------------------------------

    def step(self, g2_j: float, nu_j: float, z_j: float,
             j: int) -> Tuple[float, float, float]:
        """
        Single RG step: (g^2_j, nu_j, z_j) -> (g^2_{j+1}, nu_{j+1}, z_{j+1}).

        Coupling evolution (2-loop with curvature):
            1/g^2_{j+1} = 1/g^2_j + delta(1/g^2)

        where delta includes 1-loop, 2-loop, and curvature correction.

        Mass evolution:
            nu_{j+1} = nu_j * (1 + gamma_nu(g_j)) + delta_nu(g_j, j)

        Wavefunction evolution:
            z_{j+1} = z_j + delta_z(g_j, j)

        NUMERICAL.

        Parameters
        ----------
        g2_j : float
            Coupling g^2 at scale j.
        nu_j : float
            Mass parameter at scale j (in 1/R^2 units).
        z_j : float
            Wavefunction renormalization at scale j.
        j : int
            Current RG scale index.

        Returns
        -------
        (g2_{j+1}, nu_{j+1}, z_{j+1}) : tuple of float
        """
        # --- Coupling flow ---
        # Two-loop beta with curvature correction
        delta_inv_g2_2loop = self.beta.two_loop(g2_j)
        curv_corr = self.curvature.total_correction(j, self.R)
        # Curvature correction to the beta function
        delta_inv_g2 = delta_inv_g2_2loop * (1.0 + curv_corr)

        inv_g2_new = 1.0 / g2_j + delta_inv_g2
        if inv_g2_new <= 0:
            # Landau pole: cap at strong coupling
            g2_new = 4.0 * np.pi
        else:
            g2_new = min(1.0 / inv_g2_new, 4.0 * np.pi)

        # --- Mass flow ---
        gamma_nu = self.mass.anomalous_dimension_nu(g2_j)
        delta_nu = self.mass.one_loop_mass_shift(g2_j, j, self.R)
        nu_new = nu_j * (1.0 + gamma_nu) + delta_nu

        # --- Wavefunction flow ---
        delta_z = self.wf.one_loop_z_shift(g2_j, j)
        z_new = z_j + delta_z

        return (g2_new, nu_new, z_new)

    # ------------------------------------------------------------------
    # Full flow
    # ------------------------------------------------------------------

    def run_flow(self, g2_bare: float, nu_bare: float,
                 z_bare: float = 1.0) -> List[Tuple[float, float, float]]:
        """
        Run the full RG flow from j=0 (IR) to j=N_scales-1 (UV).

        Starting from bare couplings at the IR scale, applies N_scales-1
        RG steps. Each step moves to a higher scale (UV direction).

        NUMERICAL.

        Parameters
        ----------
        g2_bare : float
            Bare coupling g^2 at the IR scale (j=0).
        nu_bare : float
            Bare mass parameter at j=0 (in 1/R^2 units).
        z_bare : float
            Bare wavefunction renormalization at j=0 (default 1.0).

        Returns
        -------
        list of (g2, nu, z) tuples
            Couplings at scales j = 0, 1, ..., N_scales-1.
            Entry [0] = (g2_bare, nu_bare, z_bare).
        """
        trajectory = [(g2_bare, nu_bare, z_bare)]
        g2, nu, z = g2_bare, nu_bare, z_bare

        for j in range(self.N_scales - 1):
            g2, nu, z = self.step(g2, nu, z, j)
            trajectory.append((g2, nu, z))

        return trajectory

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def is_asymptotically_free(self) -> bool:
        """
        Check whether the beta function gives asymptotic freedom.

        THEOREM: beta_0 > 0 for pure SU(N_c) YM => asymptotically free.
        """
        return self.beta.beta0 > 0

    def coupling_at_scale(self, j: int, g2_ir: float) -> float:
        """
        Coupling at scale j, starting from g^2 at IR (j=0).

        Uses 1-loop running:
            1/g^2(j) = 1/g^2(0) + j * beta_0 * ln(M^2)

        NUMERICAL.

        Parameters
        ----------
        j : int
            Target scale index.
        g2_ir : float
            Coupling at IR (j=0).

        Returns
        -------
        float
            g^2 at scale j.
        """
        inv_g2 = 1.0 / g2_ir + j * self.beta.one_loop(g2_ir)
        if inv_g2 <= 0:
            return 4.0 * np.pi
        return min(1.0 / inv_g2, 4.0 * np.pi)

    def effective_alpha_s(self, j: int, g2_ir: float) -> float:
        """
        Effective alpha_s = g^2/(4 pi) at scale j.

        NUMERICAL.

        Parameters
        ----------
        j : int
            Scale index.
        g2_ir : float
            Coupling at IR.

        Returns
        -------
        float
            alpha_s at scale j.
        """
        return self.coupling_at_scale(j, g2_ir) / (4.0 * np.pi)

    def energy_scale_MeV(self, j: int) -> float:
        """
        Energy scale mu_j in MeV corresponding to RG step j.

            mu_j = hbar*c / (R * M^{-j}) = hbar*c * M^j / R

        Parameters
        ----------
        j : int
            Scale index.

        Returns
        -------
        float
            Energy scale in MeV.
        """
        return _HBAR_C * self.M ** j / self.R


# ======================================================================
# AsymptoticFreedomVerifier
# ======================================================================

class AsymptoticFreedomVerifier:
    """
    Verify asymptotic freedom properties of the perturbative RG flow.

    Checks:
    1. g^2_{j+1} < g^2_j for all j (coupling decreasing toward UV).
    2. Lambda_QCD extraction from running: Lambda = mu * exp(-1/(2 beta_0 g^2)).
    3. Consistency: Lambda_QCD ~ 200 MeV at physical coupling.
    4. Landau pole check: coupling stays finite for all j <= N_scales.

    NUMERICAL.

    Parameters
    ----------
    N_c : int
        Number of colours.
    R : float
        S^3 radius in fm.
    M : float
        Blocking factor.
    N_scales : int
        Number of RG steps.
    """

    def __init__(self, N_c: int = 2, R: float = _R_PHYS, M: float = 2.0,
                 N_scales: int = 7):
        self.N_c = N_c
        self.R = R
        self.M = M
        self.N_scales = N_scales

        self.flow = PerturbativeRGFlow(N_c, R, M, N_scales)
        self.beta0 = self.flow.beta.beta0

    def verify_decreasing_coupling(self, g2_ir: float) -> Dict:
        """
        Verify g^2 strictly decreasing from IR to UV.

        NUMERICAL.

        Parameters
        ----------
        g2_ir : float
            Coupling at IR scale.

        Returns
        -------
        dict
            'g2_trajectory': list of g^2 at each scale.
            'all_decreasing': bool, True if monotonically decreasing toward UV.
            'violations': list of (j, g2_j, g2_{j+1}) where the condition fails.
        """
        trajectory = self.flow.run_flow(g2_ir, 0.0, 1.0)
        g2s = [t[0] for t in trajectory]

        violations = []
        for j in range(len(g2s) - 1):
            if g2s[j + 1] >= g2s[j]:
                violations.append((j, g2s[j], g2s[j + 1]))

        return {
            'g2_trajectory': g2s,
            'all_decreasing': len(violations) == 0,
            'violations': violations,
        }

    def extract_lambda_qcd(self, g2_ir: float) -> Dict:
        """
        Extract Lambda_QCD from the running coupling.

        At each scale j with coupling g^2_j and energy mu_j:
            Lambda = mu_j * exp(-1 / (2 * beta_0 * g^2_j))

        If the running is consistent, Lambda should be approximately
        constant across all scales.

        NUMERICAL.

        Parameters
        ----------
        g2_ir : float
            Coupling at IR scale.

        Returns
        -------
        dict
            'lambda_values_MeV': list of Lambda extracted at each scale.
            'lambda_mean_MeV': mean Lambda.
            'lambda_std_MeV': std of Lambda.
            'consistent_with_200_MeV': bool, True if within 20%.
        """
        trajectory = self.flow.run_flow(g2_ir, 0.0, 1.0)
        g2s = [t[0] for t in trajectory]

        lambdas = []
        for j, g2_j in enumerate(g2s):
            mu_j = self.flow.energy_scale_MeV(j)
            if g2_j > 0 and self.beta0 > 0:
                lam = mu_j * np.exp(-1.0 / (2.0 * self.beta0 * g2_j))
                lambdas.append(lam)

        if len(lambdas) == 0:
            return {
                'lambda_values_MeV': [],
                'lambda_mean_MeV': 0.0,
                'lambda_std_MeV': 0.0,
                'consistent_with_200_MeV': False,
            }

        mean_lam = np.mean(lambdas)
        std_lam = np.std(lambdas)

        return {
            'lambda_values_MeV': lambdas,
            'lambda_mean_MeV': float(mean_lam),
            'lambda_std_MeV': float(std_lam),
            'consistent_with_200_MeV': abs(mean_lam - _LAMBDA_QCD) / _LAMBDA_QCD < 0.20,
        }

    def check_no_landau_pole(self, g2_ir: float) -> Dict:
        """
        Check that the coupling stays finite for all j <= N_scales.

        A Landau pole would manifest as 1/g^2 -> 0 or g^2 -> infinity.
        On S^3 with finite N_scales, the pole is avoided as long as
        g^2 stays below 4*pi at all scales.

        NUMERICAL.

        Parameters
        ----------
        g2_ir : float
            Coupling at IR scale.

        Returns
        -------
        dict
            'g2_trajectory': list of g^2.
            'max_g2': float, maximum coupling in the trajectory.
            'no_landau_pole': bool, True if g^2 < 4*pi everywhere.
        """
        trajectory = self.flow.run_flow(g2_ir, 0.0, 1.0)
        g2s = [t[0] for t in trajectory]
        max_g2 = max(g2s)

        return {
            'g2_trajectory': g2s,
            'max_g2': float(max_g2),
            'no_landau_pole': max_g2 < 4.0 * np.pi,
        }

    def full_verification(self, g2_ir: float) -> Dict:
        """
        Run all verification checks.

        NUMERICAL.

        Parameters
        ----------
        g2_ir : float
            Coupling at IR scale.

        Returns
        -------
        dict with all verification results.
        """
        decreasing = self.verify_decreasing_coupling(g2_ir)
        lambda_qcd = self.extract_lambda_qcd(g2_ir)
        landau = self.check_no_landau_pole(g2_ir)

        return {
            'asymptotic_freedom': decreasing['all_decreasing'],
            'decreasing_coupling': decreasing,
            'lambda_qcd': lambda_qcd,
            'landau_pole': landau,
            'beta0': self.beta0,
            'N_c': self.N_c,
            'R_fm': self.R,
        }
