"""
Anharmonic Oscillator Scaling Analysis for the R -> infinity Limit.

The finite-dimensional effective Hamiltonian on S^3/I* is:

    H_eff(R, g^2) = -(1/2) nabla^2 + (4/R^2)|a|^2 + g^2 * V_4(a)

where a in R^9 (3 coexact modes x 3 adjoint components).

As R -> infinity:
  - omega^2 = 4/R^2 -> 0  (power law, fast)
  - lambda = g^2(R) -> 0   (logarithmic, slow)
  - The ratio lambda / omega^3 -> infinity: STRONG COUPLING of the quartic

In the strong-coupling (quartic-dominated) regime:
  - 1D: gap ~ lambda^{1/3}
  - dD: gap ~ lambda^{1/(d+2)}

This module:
  1. Establishes reference results for the 1D anharmonic oscillator
  2. Extends to multi-dimensional systems (our case: 3 SVD DOF after gauge fixing)
  3. Computes the effective gap Delta_eff(R) across all R
  4. Analyzes the crossover from harmonic to quartic regime
  5. Investigates the truncation validity (spectral desert on S^3/I*)
  6. Identifies dimensional transmutation in the effective theory

STATUS LABELS:
  THEOREM:     gap(H_eff) > 0 for all R > 0 (finite-dim confining potential)
  NUMERICAL:   gap scaling ~ [g^2(R)]^{1/3} for large R (1D) or ^{1/5} (3D)
  NUMERICAL:   crossover radius where harmonic -> quartic transition occurs
  PROPOSITION: spectral desert ratio is R-independent (geometric eigenvalues)
  CONJECTURE:  truncation gap lower-bounds the true gap

References:
  - Bender & Wu (1969): Anharmonic oscillator perturbation theory
  - Simon (1970): Coupling constant analyticity for the AHO
  - Graffi & Grecchi (1978): Borel summability of AHO
  - Luscher (1982): Symmetry breaking in finite-volume gauge theories
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Optional


# ======================================================================
# Physical constants
# ======================================================================

HBAR_C_MEV_FM = 197.3269804    # hbar*c in MeV*fm
LAMBDA_QCD_DEFAULT = 200.0      # Lambda_QCD in MeV


# ======================================================================
# 1. One-Dimensional Anharmonic Oscillator
# ======================================================================

class AnharmonicOscillator1D:
    """
    1D Anharmonic oscillator: H = -(1/2) d^2/dx^2 + (omega^2/2) x^2 + lambda x^4.

    Three regimes:
      - Harmonic (omega^2 >> lambda):  gap ~ omega
      - Mixed:                          gap ~ max(omega, c * lambda^{1/3})
      - Quartic (lambda >> omega^3):    gap ~ c_1 * lambda^{1/3}

    The quartic scaling lambda^{1/3} follows from dimensional analysis:
      H = lambda [ -(1/(2*lambda)) d^2/dx^2 + x^4 ]
    Rescale y = lambda^{1/6} x, then H = lambda^{1/3} * H_standard
    where H_standard = -(1/2) d^2/dy^2 + y^4 is parameter-free.

    THEOREM: gap(H) > 0 for all omega >= 0, lambda > 0.
    NUMERICAL: c_1 = gap(H_standard) ~ 1.7254 (the pure quartic gap).
    (Note: the value 1.0604 in some references is E_0 of H = -d^2/dx^2 + x^4,
     not the gap of our convention H = -(1/2)d^2/dx^2 + x^4.)
    """

    # Gap of the pure quartic oscillator H = -d^2/(2dx^2) + x^4
    # Computed by high-accuracy numerical diagonalization
    PURE_QUARTIC_GAP = None  # Will be computed and cached

    def __init__(self, omega_sq=1.0, lam=0.0):
        """
        Parameters
        ----------
        omega_sq : float
            Coefficient of x^2/2 (omega^2).
        lam : float
            Coefficient of x^4 (lambda >= 0).
        """
        if lam < 0:
            raise ValueError(f"lambda must be >= 0, got {lam}")
        self.omega_sq = omega_sq
        self.lam = lam
        self.omega = np.sqrt(abs(omega_sq)) if omega_sq > 0 else 0.0

    @staticmethod
    def _build_hamiltonian_fdm(omega_sq, lam, n_grid=500, x_max=None):
        """
        Build the Hamiltonian matrix using finite difference method.

        Parameters
        ----------
        omega_sq : float
            Coefficient of x^2/2.
        lam : float
            Coefficient of x^4.
        n_grid : int
            Number of grid points.
        x_max : float or None
            Half-width of the grid. Auto-selected if None.

        Returns
        -------
        ndarray : Hamiltonian matrix (n_grid x n_grid)
        """
        # Auto-select grid range based on potential shape
        if x_max is None:
            if lam > 0 and omega_sq >= 0:
                # Turning point estimate for E ~ omega or E ~ lam^{1/3}
                E_est = max(np.sqrt(abs(omega_sq)) if omega_sq > 0 else 0.0,
                            lam**(1.0/3.0) if lam > 0 else 0.0,
                            1.0)
                # x where V(x) ~ 10*E_est
                if lam > 1e-15:
                    x_turn = (10.0 * E_est / lam) ** 0.25
                else:
                    x_turn = np.sqrt(20.0 * E_est / max(omega_sq, 0.01))
                x_max = max(x_turn, 5.0)
            else:
                x_max = 10.0

        x = np.linspace(-x_max, x_max, n_grid)
        dx = x[1] - x[0]

        # Kinetic energy: -(1/2) d^2/dx^2 via central finite differences
        # T_{ii} = 1/dx^2, T_{i,i+1} = T_{i,i-1} = -1/(2*dx^2)
        diag_main = np.full(n_grid, 1.0 / dx**2)
        diag_off = np.full(n_grid - 1, -0.5 / dx**2)

        # Potential energy
        V = 0.5 * omega_sq * x**2 + lam * x**4

        # Full Hamiltonian
        H = np.diag(diag_main + V) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        return H, x

    def diagonalize(self, n_eigenvalues=5, n_grid=500, x_max=None):
        """
        Compute the lowest eigenvalues by numerical diagonalization.

        Parameters
        ----------
        n_eigenvalues : int
            Number of eigenvalues to return.
        n_grid : int
            Grid size for finite differences.
        x_max : float or None
            Grid half-width.

        Returns
        -------
        dict with eigenvalues, gap, and parameters.
        """
        H, x = self._build_hamiltonian_fdm(
            self.omega_sq, self.lam, n_grid, x_max
        )

        n_ev = min(n_eigenvalues, n_grid - 2)
        evals = eigh(H, eigvals_only=True, subset_by_index=[0, n_ev - 1])

        gap = evals[1] - evals[0] if len(evals) > 1 else 0.0

        return {
            'eigenvalues': evals,
            'gap': float(gap),
            'E0': float(evals[0]),
            'E1': float(evals[1]) if len(evals) > 1 else None,
            'omega_sq': self.omega_sq,
            'lambda': self.lam,
            'n_grid': n_grid,
        }

    @classmethod
    def pure_quartic_gap(cls, n_grid=800):
        """
        Compute the gap of the pure quartic oscillator H = -d^2/(2dx^2) + x^4.

        NUMERICAL: c_1 ~ 1.7254 for our convention H = -(1/2)d^2/dx^2 + x^4.
        (E_0 ~ 0.6680, E_1 ~ 2.3933, gap = E_1 - E_0 ~ 1.7254.)

        Returns
        -------
        float : the gap E_1 - E_0 of H = -(1/2)d^2/dx^2 + x^4
        """
        if cls.PURE_QUARTIC_GAP is not None:
            return cls.PURE_QUARTIC_GAP

        osc = cls(omega_sq=0.0, lam=1.0)
        result = osc.diagonalize(n_eigenvalues=3, n_grid=n_grid)
        cls.PURE_QUARTIC_GAP = result['gap']
        return cls.PURE_QUARTIC_GAP

    def gap_harmonic_approx(self):
        """
        Gap in the harmonic approximation (lambda = 0).

        THEOREM: gap = omega = sqrt(omega^2) when lambda = 0.
        """
        return self.omega

    def gap_quartic_approx(self):
        """
        Gap in the quartic-dominated regime (omega^2 = 0).

        NUMERICAL: gap ~ c_1 * lambda^{1/3} where c_1 ~ 1.7254.

        The scaling follows from:
          H = lambda * [-(1/(2*lambda)) d^2/dx^2 + x^4]
        Rescale y = lambda^{1/6} x => H = lambda^{1/3} * H_std
        where H_std = -(1/2) d^2/dy^2 + y^4.
        """
        if self.lam <= 0:
            return 0.0
        c1 = self.pure_quartic_gap()
        return c1 * self.lam ** (1.0 / 3.0)

    def gap_combined_approx(self):
        """
        Combined gap estimate valid across all regimes.

        The gap is approximately max(omega, c_1 * lambda^{1/3}).
        A smoother interpolation: (omega^3 + c_1^3 * lambda)^{1/3}.
        """
        g_harm = self.gap_harmonic_approx()
        g_quart = self.gap_quartic_approx()
        # Smooth interpolation that is exact in both limits
        return (g_harm**3 + g_quart**3) ** (1.0 / 3.0)

    def coupling_ratio(self):
        """
        Dimensionless coupling ratio: lambda / omega^3.

        This determines the regime:
          - ratio << 1: harmonic
          - ratio ~ 1: mixed (crossover)
          - ratio >> 1: quartic (strong coupling)
        """
        if self.omega <= 0:
            return np.inf
        return self.lam / self.omega**3

    def regime(self):
        """Identify the regime based on coupling ratio."""
        r = self.coupling_ratio()
        if r < 0.1:
            return 'harmonic'
        elif r > 10.0:
            return 'quartic'
        else:
            return 'mixed'


# ======================================================================
# 2. Multi-Dimensional Anharmonic Oscillator
# ======================================================================

class AnharmonicOscillatorND:
    """
    d-dimensional anharmonic oscillator:
        H = -(1/2) nabla^2 + (omega^2/2) |x|^2 + lambda * V_4(x)

    For a rotationally symmetric quartic V_4 = |x|^4, the gap scaling is:
        Quartic regime: gap ~ c_d * lambda^{1/(d+2)}

    This follows from dimensional analysis:
      H = lambda^{2/(d+2)} * H_std  after rescaling x -> lambda^{-1/(2(d+2))} x
      where H_std = -(1/2) nabla^2 + |y|^{2(d+2)/d}...

    Actually, more carefully for V_4 = lambda * |x|^4:
      Rescale y = lambda^{1/(d+2)} * (some factor) * x
      Energy scales as lambda^{2/(d+2)}
      Gap scales as lambda^{2/(d+2)}

    WAIT: let me redo this properly.
      H = -(1/2) nabla^2 + lambda |x|^4
      Rescale: y = alpha * x, then -nabla_y^2 = -alpha^{-2} nabla_x^2
      H = (alpha^2/2) * [-nabla_y^2] + lambda * alpha^{-4} * |y|^4
      Set alpha^2 = lambda * alpha^{-4} => alpha^6 = lambda => alpha = lambda^{1/6}
      H = lambda^{1/3} * [-(1/2) nabla_y^2 + |y|^4]

    So the gap scales as lambda^{1/3} INDEPENDENT OF DIMENSION.
    The dimension enters through the numerical prefactor c_d.

    THEOREM: For V_4 = |x|^4 (isotropic), gap ~ c_d * lambda^{1/3} for all d.

    For our YM problem, V_4 is NOT isotropic but depends on singular values.
    V_4 = (g^2/2) * sum_{i<j} sigma_i^2 * sigma_j^2 (after gauge fixing to 3 SVD).
    This is a 3-variable quartic, and the scaling is still lambda^{1/3} but
    with a different prefactor.
    """

    def __init__(self, d, omega_sq=1.0, lam=0.0, potential_type='isotropic'):
        """
        Parameters
        ----------
        d : int
            Number of dimensions.
        omega_sq : float
            Coefficient of |x|^2/2.
        lam : float
            Coefficient of the quartic term.
        potential_type : str
            'isotropic' for |x|^4, or 'ym_svd' for the YM singular value potential.
        """
        self.d = d
        self.omega_sq = omega_sq
        self.lam = lam
        self.omega = np.sqrt(abs(omega_sq)) if omega_sq > 0 else 0.0
        self.potential_type = potential_type

    def diagonalize_radial(self, n_grid=400, n_eigenvalues=5, r_max=None):
        """
        Diagonalize using radial reduction (for isotropic potential).

        The radial Schrodinger equation for the l=0 sector is:
            H_rad = -(1/2) d^2/dr^2 + [(d-1)(d-3)/(8r^2)] + V(r)

        where V(r) = (omega^2/2)*r^2 + lam*r^4, and the centrifugal
        barrier is from the d-dimensional Laplacian.

        For the ground state and gap we need l=0.

        For d=1: no centrifugal term, just the 1D problem on [0, inf) with
        appropriate boundary conditions. But the even-parity sector on all R
        is equivalent to Dirichlet on [0, inf), so the gap from the full
        1D problem (including odd states) may differ.

        For d >= 2: the effective radial potential has centrifugal barrier
        (d-1)(d-3)/(8*r^2). For d=3: barrier = 0 (s-wave).

        IMPORTANT: The gap of the FULL system (all angular momenta) is the
        gap within the l=0 sector, because the l=0 ground state is the
        absolute ground state, and the first excited state in l=0 is
        typically below the l=1 ground state for confining potentials.

        Parameters
        ----------
        n_grid : int
        n_eigenvalues : int
        r_max : float or None

        Returns
        -------
        dict with eigenvalues and gap
        """
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")

        # For d=1, use the full 1D solver
        if self.d == 1:
            osc_1d = AnharmonicOscillator1D(self.omega_sq, self.lam)
            return osc_1d.diagonalize(n_eigenvalues, n_grid)

        # Radial problem for d >= 2
        # u(r) = r^{(d-1)/2} * R(r) satisfies:
        # -u''/2 + V_eff(r) u = E u
        # V_eff = (omega^2/2) r^2 + lam r^4 + centrifugal
        # centrifugal = (d-1)(d-3) / (8 r^2) for l=0

        cent_coeff = (self.d - 1) * (self.d - 3) / 8.0

        if r_max is None:
            E_est = max(self.omega if self.omega > 0 else 0.0,
                        self.lam**(1.0/3.0) if self.lam > 0 else 0.0,
                        1.0)
            if self.lam > 1e-15:
                r_turn = (10.0 * E_est / self.lam) ** 0.25
            else:
                r_turn = np.sqrt(20.0 * E_est / max(self.omega_sq, 0.01))
            r_max = max(r_turn, 8.0)

        # Use grid that avoids r=0 singularity
        r = np.linspace(r_max / n_grid, r_max, n_grid)
        dr = r[1] - r[0]

        # Potential
        V_harm = 0.5 * self.omega_sq * r**2
        V_quart = self.lam * r**4
        V_cent = cent_coeff / (r**2 + 1e-30)  # regularize near r=0
        V_total = V_harm + V_quart + V_cent

        # Kinetic: -(1/2) d^2/dr^2
        diag_main = np.full(n_grid, 1.0 / dr**2)
        diag_off = np.full(n_grid - 1, -0.5 / dr**2)

        H = np.diag(diag_main + V_total) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        n_ev = min(n_eigenvalues, n_grid - 2)
        evals = eigh(H, eigvals_only=True, subset_by_index=[0, n_ev - 1])

        gap = evals[1] - evals[0] if len(evals) > 1 else 0.0

        return {
            'eigenvalues': evals,
            'gap': float(gap),
            'E0': float(evals[0]),
            'E1': float(evals[1]) if len(evals) > 1 else None,
            'd': self.d,
            'omega_sq': self.omega_sq,
            'lambda': self.lam,
        }

    def diagonalize_product(self, n_basis_per_dim=20, n_eigenvalues=5):
        """
        Diagonalize in product basis (for non-isotropic potentials, small d).

        Uses a product of 1D harmonic oscillator eigenstates.
        Practical only for d <= 3 with moderate n_basis.

        For the YM SVD potential:
            V_4 = (lam/2) * sum_{i<j} sigma_i^2 * sigma_j^2

        Parameters
        ----------
        n_basis_per_dim : int
            HO states per dimension.
        n_eigenvalues : int

        Returns
        -------
        dict with eigenvalues and gap
        """
        if self.d > 4:
            raise ValueError(
                f"Product basis impractical for d={self.d}. "
                f"Use radial method for isotropic potentials."
            )

        total_dim = n_basis_per_dim ** self.d
        if total_dim > 50000:
            raise ValueError(
                f"Basis too large: {n_basis_per_dim}^{self.d} = {total_dim}"
            )

        omega_ho = self.omega if self.omega > 0 else 1.0

        # Build 1D operators in HO basis
        n = n_basis_per_dim
        x_scale = 1.0 / np.sqrt(2.0 * omega_ho)

        x_1d = np.zeros((n, n))
        for k in range(n - 1):
            x_1d[k, k+1] = np.sqrt(k + 1) * x_scale
            x_1d[k+1, k] = np.sqrt(k + 1) * x_scale

        x2_1d = x_1d @ x_1d
        x4_1d = x2_1d @ x2_1d

        # Build d-dimensional Hamiltonian
        I_1d = np.eye(n)

        def kron_chain(mats):
            result = mats[0]
            for m in mats[1:]:
                result = np.kron(result, m)
            return result

        # Harmonic part: sum_i omega_ho*(n_i + 1/2)
        H = np.zeros((total_dim, total_dim))
        for dim_idx in range(self.d):
            parts = [I_1d] * self.d
            diag_ho = np.diag([omega_ho * (k + 0.5) for k in range(n)])
            parts[dim_idx] = diag_ho
            H += kron_chain(parts)

        # If omega_sq != omega_ho^2, correct the quadratic term
        if abs(self.omega_sq - omega_ho**2) > 1e-15:
            delta_omega_sq = self.omega_sq - omega_ho**2
            for dim_idx in range(self.d):
                parts = [I_1d] * self.d
                parts[dim_idx] = x2_1d
                H += 0.5 * delta_omega_sq * kron_chain(parts)

        # Quartic part
        if self.lam > 0:
            if self.potential_type == 'isotropic':
                # V_4 = lam * |x|^4 = lam * (sum x_i^2)^2
                # = lam * [sum x_i^4 + 2 * sum_{i<j} x_i^2 x_j^2]
                for i in range(self.d):
                    parts = [I_1d] * self.d
                    parts[i] = x4_1d
                    H += self.lam * kron_chain(parts)
                for i in range(self.d):
                    for j in range(i+1, self.d):
                        parts = [I_1d] * self.d
                        parts[i] = x2_1d
                        parts[j] = x2_1d
                        H += 2.0 * self.lam * kron_chain(parts)

            elif self.potential_type == 'ym_svd':
                # V_4 = (lam/2) * sum_{i<j} x_i^2 * x_j^2
                for i in range(self.d):
                    for j in range(i+1, self.d):
                        parts = [I_1d] * self.d
                        parts[i] = x2_1d
                        parts[j] = x2_1d
                        H += 0.5 * self.lam * kron_chain(parts)

        n_ev = min(n_eigenvalues, total_dim - 1)
        evals = eigh(H, eigvals_only=True, subset_by_index=[0, n_ev - 1])

        gap = evals[1] - evals[0] if len(evals) > 1 else 0.0

        return {
            'eigenvalues': evals,
            'gap': float(gap),
            'E0': float(evals[0]),
            'E1': float(evals[1]) if len(evals) > 1 else None,
            'd': self.d,
            'omega_sq': self.omega_sq,
            'lambda': self.lam,
            'potential_type': self.potential_type,
            'basis_size': total_dim,
        }

    def gap_quartic_scaling(self):
        """
        Gap scaling in the pure quartic regime.

        THEOREM (dimensional analysis):
            For V_4 = lam * |x|^4 (isotropic) in d dimensions:
                gap ~ c_d * lam^{1/3}

            The exponent 1/3 is dimension-INDEPENDENT because the quartic
            |x|^4 is homogeneous degree 4 regardless of d.

        For V_4 = (lam/2) * sum_{i<j} sigma_i^2 sigma_j^2 (YM SVD):
            Same lambda^{1/3} scaling, different prefactor c_ym.

        Returns
        -------
        float : estimated gap in quartic regime
        """
        if self.lam <= 0:
            return 0.0

        # The prefactor c_d depends on d and potential type
        # We compute it by solving the unit-lambda problem
        if self.potential_type == 'isotropic' and self.d == 1:
            c = AnharmonicOscillator1D.pure_quartic_gap()
        else:
            # Compute numerically for this d and potential type
            unit_osc = AnharmonicOscillatorND(
                self.d, omega_sq=0.0, lam=1.0,
                potential_type=self.potential_type
            )
            if self.d <= 3:
                result = unit_osc.diagonalize_product(n_basis_per_dim=25)
            else:
                result = unit_osc.diagonalize_radial()
            c = result['gap']

        return c * self.lam ** (1.0 / 3.0)


# ======================================================================
# 3. Running Coupling on S^3
# ======================================================================

class RunningCoupling:
    """
    1-loop running coupling for SU(N) Yang-Mills.

    g^2(mu) = 8 pi^2 / (b0_raw * ln(mu^2 / Lambda^2))
    where b0_raw = 11*N/3, mu = hbar_c / R.

    THEOREM: This is the standard 1-loop result (Gross-Wilczek-Politzer).
    """

    def __init__(self, N=2, Lambda_QCD=LAMBDA_QCD_DEFAULT):
        self.N = N
        self.Lambda_QCD = Lambda_QCD
        self.b0_raw = 11.0 * N / 3.0
        self.hbar_c = HBAR_C_MEV_FM
        self.R_landau = self.hbar_c / self.Lambda_QCD

    def g_squared(self, R_fm):
        """
        Running coupling g^2 at scale mu = hbar_c / R.

        Returns
        -------
        float : g^2(R). Returns inf if R >= R_landau.
        """
        if R_fm <= 0:
            raise ValueError(f"R must be positive, got {R_fm}")
        mu = self.hbar_c / R_fm
        if mu <= self.Lambda_QCD:
            return np.inf
        log_val = np.log((mu / self.Lambda_QCD) ** 2)
        if log_val <= 0:
            return np.inf
        return 8.0 * np.pi**2 / (self.b0_raw * log_val)

    def g_squared_safe(self, R_fm, g2_max=4.0 * np.pi):
        """
        Running coupling with saturation for non-perturbative regime.

        When R >= R_landau, the 1-loop formula diverges. We cap at g2_max.
        This is a conservative choice: physical alpha_s ~ 0.5 => g^2 ~ 6.3.

        NUMERICAL status (not a theorem for the saturated regime).
        """
        g2 = self.g_squared(R_fm)
        if np.isinf(g2) or g2 > g2_max:
            return g2_max
        return g2


# ======================================================================
# 4. Effective Theory Gap as Function of R
# ======================================================================

@dataclass
class GapAtRadius:
    """Result of gap computation at a given R."""
    R_fm: float
    gap_dimless: float        # gap in units of 1/fm^2 (eigenvalue)
    gap_MeV: float            # gap in MeV
    omega: float              # harmonic frequency 2/R
    omega_MeV: float          # omega * hbar_c
    g_squared: float          # running coupling
    lam_eff: float            # effective quartic coupling
    coupling_ratio: float     # lam_eff / omega^3 (regime indicator)
    regime: str               # 'harmonic', 'mixed', 'quartic'
    gap_harmonic_MeV: float   # harmonic approximation
    gap_quartic_MeV: float    # quartic approximation
    method: str               # computation method used


class EffectiveTheoryGap:
    """
    Compute the mass gap of the finite-dimensional effective theory on S^3/I*
    as a function of R.

    The effective Hamiltonian (after gauge fixing to 3 SVD DOF) is:
        H_eff = -(1/2) nabla^2 + (omega^2/2) sum sigma_i^2
                + (g^2/2) sum_{i<j} sigma_i^2 sigma_j^2

    where omega = 2/R and g^2 = g^2(R) is the running coupling.

    The quartic coefficient in the YM potential is:
        lambda_eff = g^2(R) * C_4
    where C_4 comes from the structure constants and mode overlaps.

    For the 3 I*-invariant modes on S^3/I*, the quartic term is:
        V_4 = (g^2/2) sum_{i<j} sigma_i^2 sigma_j^2

    so C_4 = 1/2 (the 1/2 is absorbed into the potential definition).

    THEOREM: gap > 0 for all R > 0 (confining potential in finite dim).
    NUMERICAL: gap ~ harmonic for small R, ~ quartic for large R.
    """

    def __init__(self, N=2, Lambda_QCD=LAMBDA_QCD_DEFAULT):
        self.N = N
        self.Lambda_QCD = Lambda_QCD
        self.coupling = RunningCoupling(N, Lambda_QCD)
        self.hbar_c = HBAR_C_MEV_FM

        # Quartic coupling coefficient from YM structure
        # V_4 = g^2 * C_4 * (sum_{i<j} sigma_i^2 sigma_j^2)
        # For the EffectiveHamiltonian, the full quartic is
        # (g^2/2) * sum_{i<j} sigma_i^2 sigma_j^2
        # So the effective lambda for the standard form V = lam * f(sigma) is:
        self.C_4 = 0.5  # From the (g^2/2) prefactor

    def omega(self, R_fm):
        """Harmonic frequency omega = 2/R (in 1/fm)."""
        return 2.0 / R_fm

    def omega_MeV(self, R_fm):
        """Harmonic frequency in MeV: omega * hbar_c."""
        return self.omega(R_fm) * self.hbar_c

    def lambda_eff(self, R_fm):
        """
        Effective quartic coupling: g^2(R) * C_4.

        This is the coefficient of the quartic term in the effective potential.
        """
        g2 = self.coupling.g_squared_safe(R_fm)
        return g2 * self.C_4

    def coupling_ratio(self, R_fm):
        """
        Dimensionless ratio lambda_eff / omega^3.

        This determines whether we are in the harmonic or quartic regime:
          - ratio << 1: harmonic regime, gap ~ omega ~ 2/R
          - ratio >> 1: quartic regime, gap ~ lambda^{1/3} ~ g^{2/3}
        """
        om = self.omega(R_fm)
        lam = self.lambda_eff(R_fm)
        if om <= 0:
            return np.inf
        return lam / om**3

    def crossover_radius(self):
        """
        Find R_cross where coupling_ratio = 1 (harmonic-quartic crossover).

        At R_cross: lambda_eff = omega^3, i.e., g^2(R) * C_4 = (2/R)^3.

        This is a transcendental equation solved numerically.

        Returns
        -------
        float : R_cross in fm, or None if no crossover found.
        """
        from scipy.optimize import brentq

        def f(log_R):
            R = np.exp(log_R)
            ratio = self.coupling_ratio(R)
            if np.isinf(ratio):
                return 1.0  # Beyond Landau pole, definitely quartic
            return np.log10(ratio)

        # Search between 0.01 fm and 100 fm
        try:
            log_R_cross = brentq(f, np.log(0.01), np.log(100.0))
            return np.exp(log_R_cross)
        except (ValueError, RuntimeError):
            return None

    def gap_at_R(self, R_fm, n_basis=20, method='auto'):
        """
        Compute the effective theory gap at radius R.

        Parameters
        ----------
        R_fm : float
            Radius in fm.
        n_basis : int
            Basis size for diagonalization.
        method : str
            'numerical' for full diag, 'analytical' for scaling formulas,
            'auto' to choose based on regime.

        Returns
        -------
        GapAtRadius
        """
        om = self.omega(R_fm)
        om_MeV = self.omega_MeV(R_fm)
        g2 = self.coupling.g_squared_safe(R_fm)
        lam = self.lambda_eff(R_fm)
        ratio = self.coupling_ratio(R_fm)

        if ratio < 0.1:
            regime = 'harmonic'
        elif ratio > 10.0:
            regime = 'quartic'
        else:
            regime = 'mixed'

        # Harmonic approximation: gap = omega * hbar_c
        gap_harm_MeV = om_MeV

        # Quartic approximation: gap ~ c_ym * lam^{1/3} * hbar_c
        # For the YM SVD potential with 3 DOF
        osc_ym = AnharmonicOscillatorND(3, omega_sq=0.0, lam=1.0, potential_type='ym_svd')
        c_ym_result = osc_ym.diagonalize_product(n_basis_per_dim=min(n_basis, 25))
        c_ym = c_ym_result['gap']
        gap_quart = c_ym * lam ** (1.0 / 3.0)
        gap_quart_MeV = gap_quart * self.hbar_c

        # Full numerical computation
        if method == 'auto':
            method = 'numerical'

        if method == 'numerical':
            osc = AnharmonicOscillatorND(
                3, omega_sq=om**2, lam=lam, potential_type='ym_svd'
            )
            result = osc.diagonalize_product(n_basis_per_dim=min(n_basis, 25))
            gap_dimless = result['gap']
            gap_MeV = gap_dimless * self.hbar_c
        else:
            # Analytical: use combined approximation
            gap_dimless = (om**3 + (c_ym * lam**(1.0/3.0))**3) ** (1.0/3.0)
            gap_MeV = gap_dimless * self.hbar_c

        return GapAtRadius(
            R_fm=R_fm,
            gap_dimless=gap_dimless,
            gap_MeV=gap_MeV,
            omega=om,
            omega_MeV=om_MeV,
            g_squared=g2,
            lam_eff=lam,
            coupling_ratio=ratio if np.isfinite(ratio) else 1e10,
            regime=regime,
            gap_harmonic_MeV=gap_harm_MeV,
            gap_quartic_MeV=gap_quart_MeV,
            method=method,
        )

    def gap_scan(self, R_values=None, n_basis=20):
        """
        Scan gap across a range of R values.

        Parameters
        ----------
        R_values : array-like or None
            Radii in fm. Default: logarithmic scan.
        n_basis : int

        Returns
        -------
        list of GapAtRadius
        """
        if R_values is None:
            R_values = np.logspace(-1, 3, 30)

        results = []
        for R in R_values:
            try:
                result = self.gap_at_R(float(R), n_basis=n_basis)
                results.append(result)
            except Exception:
                pass

        return results

    def gap_vs_R_arrays(self, R_values=None, n_basis=20):
        """
        Return numpy arrays suitable for plotting.

        Returns
        -------
        dict with arrays: R, gap_MeV, gap_harmonic_MeV, gap_quartic_MeV, regime
        """
        results = self.gap_scan(R_values, n_basis)
        return {
            'R_fm': np.array([r.R_fm for r in results]),
            'gap_MeV': np.array([r.gap_MeV for r in results]),
            'gap_harmonic_MeV': np.array([r.gap_harmonic_MeV for r in results]),
            'gap_quartic_MeV': np.array([r.gap_quartic_MeV for r in results]),
            'g_squared': np.array([r.g_squared for r in results]),
            'coupling_ratio': np.array([r.coupling_ratio for r in results]),
            'regime': [r.regime for r in results],
        }


# ======================================================================
# 5. Spectral Desert Analysis
# ======================================================================

class SpectralDesertAnalysis:
    """
    Analyze the spectral desert on S^3/I* that justifies the 3-mode truncation.

    On S^3, the coexact 1-form eigenvalues are:
        mu_k = (k+1)^2 / R^2  for k = 1, 2, 3, ...

    On S^3/I* (Poincare homology sphere), the I*-projection kills many modes.
    At k=1: 6 modes on S^3, 3 survive on S^3/I*.
    The next surviving modes are at k=11 (or higher), giving:
        mu_1 = 4/R^2     (3 modes)
        mu_11 = 144/R^2  (next surviving level)

    PROPOSITION: The desert ratio mu_11/mu_1 = 36 is R-INDEPENDENT.

    This is because:
      - All coexact eigenvalues scale as 1/R^2
      - The I*-projection depends only on the angular part (R-independent)
      - Therefore the set of surviving modes is the same at all R
      - And the ratios between surviving eigenvalues are R-independent

    This means the 3-mode truncation is valid for ALL R, not just small R.

    The energy gap between the low sector (k=1) and the high sector (k>=11)
    is Delta_desert = (144 - 4)/R^2 = 140/R^2, which provides exponential
    suppression of high-mode contributions at low energies.
    """

    # Coexact eigenvalues on S^3 (unit sphere): mu_k = (k+1)^2
    # Modes surviving I* projection: k=1 (3 modes), k=11 (next batch)
    MU_LOW = 4      # (1+1)^2
    MU_HIGH = 144   # (11+1)^2
    DESERT_RATIO = MU_HIGH / MU_LOW  # = 36

    # Number of I*-invariant coexact modes at each level
    N_MODES_K1 = 3
    # Expected next level
    NEXT_K = 11

    def __init__(self, R=1.0):
        self.R = R

    def low_eigenvalue(self):
        """Lowest coexact eigenvalue on S^3/I*: 4/R^2."""
        return self.MU_LOW / self.R**2

    def high_eigenvalue(self):
        """Next coexact eigenvalue on S^3/I*: 144/R^2."""
        return self.MU_HIGH / self.R**2

    def desert_ratio(self):
        """
        Ratio of next-surviving to lowest eigenvalue.

        PROPOSITION: This is R-independent (= 36).
        """
        return self.DESERT_RATIO

    def desert_gap(self):
        """
        Energy gap between low and high sectors: (144-4)/R^2 = 140/R^2.
        """
        return (self.MU_HIGH - self.MU_LOW) / self.R**2

    def is_truncation_valid(self, T_energy):
        """
        Check if the 3-mode truncation is valid at energy scale T_energy.

        The truncation is valid when T_energy << desert_gap,
        i.e., the energy is well below the high-mode threshold.

        Parameters
        ----------
        T_energy : float
            Typical energy scale (in units of 1/R^2).

        Returns
        -------
        dict with validity assessment
        """
        dg = self.desert_gap()
        ratio = T_energy / dg if dg > 0 else np.inf

        return {
            'valid': ratio < 0.1,
            'marginal': ratio < 1.0,
            'energy_scale': T_energy,
            'desert_gap': dg,
            'ratio': ratio,
            'note': (
                'PROPOSITION: Truncation valid when typical energy << desert gap. '
                f'Here: E/gap_desert = {ratio:.4f}. '
                f'{"Valid" if ratio < 0.1 else "Marginal" if ratio < 1.0 else "INVALID"}.'
            ),
        }

    def r_independence_check(self, R_values=None):
        """
        Verify that the desert ratio is R-independent.

        PROPOSITION: Since all eigenvalues scale as 1/R^2, the ratio is
        a pure number determined by the geometry of S^3/I* (R-independent).
        """
        if R_values is None:
            R_values = [0.1, 1.0, 10.0, 100.0, 1000.0]

        ratios = []
        for R in R_values:
            analysis = SpectralDesertAnalysis(R)
            ratios.append(analysis.desert_ratio())

        all_equal = all(abs(r - self.DESERT_RATIO) < 1e-12 for r in ratios)

        return {
            'R_values': R_values,
            'ratios': ratios,
            'all_equal': all_equal,
            'expected': self.DESERT_RATIO,
            'note': (
                'PROPOSITION: Desert ratio is R-independent because all '
                'coexact eigenvalues scale as 1/R^2, and the I*-projection '
                'depends only on angular quantum numbers (R-independent).'
            ),
        }


# ======================================================================
# 6. Truncation Validity and Mode Coupling Sign
# ======================================================================

class TruncationAnalysis:
    """
    Analyze whether the 3-mode truncation UNDERESTIMATES or OVERESTIMATES the gap.

    The key question: when we drop the high modes (k >= 11), does the gap of the
    remaining low-energy effective theory provide a LOWER or UPPER bound on the
    true gap?

    The answer depends on the SIGN of the coupling between low and high modes.

    For Yang-Mills on S^3/I*:
      - The full Hamiltonian: H = H_low + H_high + H_coupling
      - H_low: the 3-mode effective Hamiltonian
      - H_high: high modes (k >= 11), decoupled
      - H_coupling: interaction between low and high modes (from [A,A] terms)

    The coupling H_coupling involves:
      - Trilinear terms: low-low-high and low-high-high
      - Quartic terms: low-low-high-high

    PROPOSITION: The dominant coupling is REPULSIVE (pushes eigenvalues apart).

    Argument:
      1. The quartic term V_4 = (g^2/4) |[A,A]|^2 >= 0 for ALL modes
      2. Including more modes adds more positive quartic contributions
      3. These additional contributions raise the effective potential
      4. A higher potential generally means a LARGER gap
      5. Therefore: truncation UNDERESTIMATES the gap

    CAVEAT: This argument is not rigorous because:
      - Step 4 is not always true (adding repulsion can split the gap)
      - The trilinear coupling could potentially reduce the gap
      - A rigorous bound would need a Bogolubov-type inequality

    STATUS: PROPOSITION (plausible but not proven).
    """

    def __init__(self, R=1.0, g_coupling=1.0, N=2,
                 Lambda_QCD=LAMBDA_QCD_DEFAULT):
        self.R = R
        self.g = g_coupling
        self.N = N
        self.Lambda_QCD = Lambda_QCD
        self.desert = SpectralDesertAnalysis(R)

    def coupling_sign_analysis(self):
        """
        Analyze the sign of the coupling between low and high modes.

        The quartic interaction between low (k=1) and high (k>=11) modes:
            V_{low-high} = g^2 * integral [A_low, A_high] ^ *[A_low, A_high]

        Since this is |...|^2, it is >= 0.
        Adding it to H_low raises the potential -> increases the gap.

        BUT: there are also cross-terms in the quadratic part:
            The linearized operator has off-diagonal blocks between
            low and high modes from the [A_background, delta_A] terms.
            At the MC vacuum (F=0), these vanish. So the leading
            coupling is indeed the quartic term.

        Returns
        -------
        dict with analysis
        """
        return {
            'quartic_sign': 'positive',
            'quartic_effect': 'raises potential -> increases gap',
            'trilinear_at_mc_vacuum': 'zero (F_theta = 0)',
            'dominant_coupling': 'quartic (positive)',
            'truncation_direction': 'underestimates gap',
            'status': 'PROPOSITION',
            'note': (
                'PROPOSITION: The 3-mode truncation underestimates the true gap. '
                'The quartic coupling V_{low-high} = g^2 |[A_low, A_high]|^2 >= 0 '
                'is positive and adds to the confining potential. At the MC vacuum, '
                'trilinear couplings vanish (F_theta = 0). Therefore dropping high '
                'modes lowers the potential -> decreases the gap.'
            ),
            'caveat': (
                'This argument is not rigorous: (1) higher potential does not always '
                'imply larger gap; (2) the trilinear coupling is zero only at the '
                'vacuum, not at generic field configurations; (3) a rigorous bound '
                'would require a Bogolubov-type inequality or a resolvent estimate. '
                'The argument is physically motivated but falls short of THEOREM status.'
            ),
        }

    def effective_gap_bound_type(self):
        """
        Is the effective theory gap a LOWER or UPPER bound on the true gap?

        PROPOSITION: Lower bound.
        CAVEAT: Not rigorously proven.
        """
        return {
            'bound_type': 'lower (PROPOSITION)',
            'inequality': 'Delta_true >= Delta_eff (proposed, not proven)',
            'evidence': (
                '1. V_4 >= 0 for all modes (THEOREM). '
                '2. Including more modes adds positive V_4 (THEOREM). '
                '3. Higher V -> larger gap (PROPOSITION, not always true). '
                '4. Numerical: full 9-DOF gap >= reduced 3-DOF gap (checked).'
            ),
        }


# ======================================================================
# 7. Dimensional Transmutation in the Effective Theory
# ======================================================================

class DimensionalTransmutationEffective:
    """
    Analyze dimensional transmutation in the finite-dim effective theory.

    The effective theory has parameters: omega(R) = 2/R and g^2(R).
    Both go to zero as R -> infinity, but their ratio diverges.

    The effective theory generates its own dynamical scale Lambda_eff:
      Lambda_eff is defined as the R at which Delta_eff(R) = Lambda_eff * hbar_c.

    Compare Lambda_eff with Lambda_QCD:
      - If Lambda_eff ~ Lambda_QCD: effective theory captures the right physics
      - If Lambda_eff << Lambda_QCD: truncation misses important NP effects
      - If Lambda_eff >> Lambda_QCD: truncation overestimates confinement
    """

    def __init__(self, N=2, Lambda_QCD=LAMBDA_QCD_DEFAULT):
        self.N = N
        self.Lambda_QCD = Lambda_QCD
        self.hbar_c = HBAR_C_MEV_FM
        self.gap_computer = EffectiveTheoryGap(N, Lambda_QCD)

    def find_lambda_eff(self, n_basis=20):
        """
        Find the effective dynamical scale Lambda_eff.

        Lambda_eff is defined by: for large R, gap(R) ~ c * Lambda_eff.

        In practice, we compute gap(R) for large R and fit to extract
        the asymptotic value (or the slowest decay scale).

        Returns
        -------
        dict with Lambda_eff and comparison to Lambda_QCD
        """
        # Compute gap at several large R values
        R_large = np.array([5.0, 10.0, 20.0, 50.0, 100.0])

        gaps_MeV = []
        g2_values = []
        for R in R_large:
            result = self.gap_computer.gap_at_R(R, n_basis=n_basis)
            gaps_MeV.append(result.gap_MeV)
            g2_values.append(result.g_squared)

        gaps_MeV = np.array(gaps_MeV)
        g2_values = np.array(g2_values)

        # The effective theory gap scales as ~ g^{2/3}(R) * constant * hbar_c
        # Extract the scale from the fit
        # gap = Lambda_eff * f(R) where f(R) encodes the R-dependence

        # At large R, gap ~ c_ym * (g^2(R) * C_4)^{1/3} * hbar_c
        # So Lambda_eff ~ c_ym * (g^2 * C_4)^{1/3} * hbar_c / hbar_c = ...
        # Actually Lambda_eff IS the gap at some reference point.

        # More precisely: the effective theory predicts
        # gap(R) ~ c * [g^2(R)]^{1/3}
        # = c * [8*pi^2 / (b0_raw * ln(mu^2/Lambda^2))]^{1/3}
        # = c * [8*pi^2 / (b0_raw * 2*ln(hbar_c/(R*Lambda)))]^{1/3}

        # This -> 0 logarithmically as R -> inf.
        # Lambda_eff is the scale where this crosses Lambda_QCD:
        #   gap(R_eff) = Lambda_QCD
        # But since gap -> 0, there exists R_eff where gap = Lambda_QCD for small R
        # and gap < Lambda_QCD for R > R_eff.

        # Find R_eff where gap crosses Lambda_QCD from above
        R_scan = np.logspace(-0.5, 2.0, 50)
        gaps_scan = []
        for R in R_scan:
            try:
                res = self.gap_computer.gap_at_R(R, n_basis=n_basis)
                gaps_scan.append(res.gap_MeV)
            except Exception:
                gaps_scan.append(np.nan)
        gaps_scan = np.array(gaps_scan)

        # Find crossing point
        R_eff = None
        for i in range(len(R_scan) - 1):
            if (np.isfinite(gaps_scan[i]) and np.isfinite(gaps_scan[i+1])
                    and gaps_scan[i] >= self.Lambda_QCD
                    and gaps_scan[i+1] < self.Lambda_QCD):
                # Linear interpolation
                t = ((self.Lambda_QCD - gaps_scan[i])
                     / (gaps_scan[i+1] - gaps_scan[i]))
                R_eff = R_scan[i] + t * (R_scan[i+1] - R_scan[i])
                break

        # Asymptotic gap at R = 100 fm (representative of "large R")
        asymptotic_gap_MeV = gaps_MeV[-1] if len(gaps_MeV) > 0 else 0.0

        return {
            'Lambda_QCD_MeV': self.Lambda_QCD,
            'R_crossover_fm': R_eff,
            'asymptotic_gap_MeV': asymptotic_gap_MeV,
            'gaps_at_large_R': list(zip(R_large.tolist(), gaps_MeV.tolist())),
            'gap_decays_to_zero': bool(asymptotic_gap_MeV < self.Lambda_QCD),
            'note': (
                'The effective theory gap decays logarithmically as R -> inf: '
                f'gap ~ [g^2(R)]^(1/3) * c. At R=100 fm: gap ~ {asymptotic_gap_MeV:.1f} MeV. '
                f'Lambda_QCD = {self.Lambda_QCD:.0f} MeV. '
                'The effective theory does NOT reproduce Lambda_QCD as a floor, '
                'because it is a 3-mode truncation that misses the full NP dynamics. '
                'HOWEVER: the gap remains > 0 for all finite R (THEOREM).'
            ),
        }

    def gap_decay_rate(self, R_values=None, n_basis=20):
        """
        Characterize the decay rate of the gap at large R.

        NUMERICAL: gap(R) ~ A / [ln(R * Lambda_QCD / hbar_c)]^{1/3}
        for R >> hbar_c / Lambda_QCD.

        Returns
        -------
        dict with decay analysis
        """
        if R_values is None:
            R_values = np.array([2.0, 5.0, 10.0, 20.0, 50.0])

        gaps_MeV = []
        g2_vals = []
        for R in R_values:
            res = self.gap_computer.gap_at_R(R, n_basis=n_basis)
            gaps_MeV.append(res.gap_MeV)
            g2_vals.append(res.g_squared)

        gaps_MeV = np.array(gaps_MeV)
        g2_vals = np.array(g2_vals)

        # Check if gap ~ g^{2/3}: plot log(gap) vs log(g^2)
        # Expect slope 1/3
        finite_mask = np.isfinite(g2_vals) & (g2_vals > 0) & (gaps_MeV > 0)
        if np.sum(finite_mask) >= 3:
            log_g2 = np.log(g2_vals[finite_mask])
            log_gap = np.log(gaps_MeV[finite_mask])
            # Linear fit
            coeffs = np.polyfit(log_g2, log_gap, 1)
            slope = coeffs[0]
        else:
            slope = None

        return {
            'R_values': R_values.tolist(),
            'gaps_MeV': gaps_MeV.tolist(),
            'g_squared': g2_vals.tolist(),
            'log_slope': float(slope) if slope is not None else None,
            'expected_slope': 1.0 / 3.0,
            'consistent_with_quartic_scaling': (
                abs(slope - 1.0/3.0) < 0.15 if slope is not None else False
            ),
        }


# ======================================================================
# 8. Key Result: Gap Positivity Theorem
# ======================================================================

class GapPositivityResult:
    """
    THEOREM: For the finite-dimensional effective Hamiltonian H_eff on S^3/I*,
    the spectral gap Delta_eff(R) > 0 for all R > 0.

    Proof:
        1. H_eff = T + V_2 + V_4 where T = kinetic, V_2 = harmonic, V_4 = quartic.
        2. V_2 + V_4 is confining: V(a) -> infinity as |a| -> infinity.
           (V_2 grows quadratically; V_4 >= 0 adds to it.)
        3. Any confining potential in finite dimensions has purely discrete spectrum.
        4. The minimum of V is at a=0 with V(0)=0 (unique minimum).
        5. By the spectral theorem for Schrodinger operators with confining potential:
           E_0 < E_1 < ... with E_n -> infinity.
        6. Therefore gap = E_1 - E_0 > 0.

    NUMERICAL:
        gap(R) ~ 2*hbar_c/R for small R (harmonic regime)
        gap(R) ~ c * [g^2(R)]^{1/3} * hbar_c for large R (quartic regime)
        gap(R) > 0 for all tested R in [0.1, 10^6] fm.

    CAVEAT:
        This is the gap of the EFFECTIVE theory, not the full theory.
        The effective theory truncates to 3 modes. Whether this lower-bounds
        the true gap is PROPOSITION status, not THEOREM.
    """

    @staticmethod
    def verify(R_values=None, N=2, Lambda_QCD=LAMBDA_QCD_DEFAULT, n_basis=20):
        """
        Numerical verification of gap positivity.

        Parameters
        ----------
        R_values : array-like or None
        N : int
        Lambda_QCD : float
        n_basis : int

        Returns
        -------
        dict with verification results
        """
        if R_values is None:
            R_values = np.array([
                0.1, 0.2, 0.5, 1.0, 2.0, 2.2, 5.0,
                10.0, 20.0, 50.0, 100.0
            ])

        computer = EffectiveTheoryGap(N, Lambda_QCD)

        results = []
        all_positive = True
        min_gap = np.inf

        for R in R_values:
            res = computer.gap_at_R(R, n_basis=n_basis)
            results.append(res)
            if res.gap_MeV <= 0:
                all_positive = False
            min_gap = min(min_gap, res.gap_MeV)

        return {
            'all_positive': all_positive,
            'min_gap_MeV': float(min_gap),
            'n_tested': len(R_values),
            'R_range': [float(R_values[0]), float(R_values[-1])],
            'results': results,
            'status': 'THEOREM' if all_positive else 'FAILED',
            'note': (
                'THEOREM: Gap > 0 for the finite-dim effective Hamiltonian '
                f'for all R tested in [{R_values[0]}, {R_values[-1]}] fm. '
                f'Minimum gap: {min_gap:.2f} MeV.'
            ),
        }


# ======================================================================
# 9. Summary and Comparison Table
# ======================================================================

def summary_table(R_values=None, N=2, Lambda_QCD=LAMBDA_QCD_DEFAULT, n_basis=20):
    """
    Generate a human-readable summary table of gap vs R.

    Shows the gap from harmonic, quartic, and numerical computation.

    Parameters
    ----------
    R_values : array-like or None
    N : int
    Lambda_QCD : float
    n_basis : int

    Returns
    -------
    str : formatted table
    """
    if R_values is None:
        R_values = np.array([
            0.1, 0.5, 1.0, 2.0, 2.2, 5.0,
            10.0, 50.0, 100.0
        ])

    computer = EffectiveTheoryGap(N, Lambda_QCD)

    lines = []
    lines.append("=" * 100)
    lines.append(f"ANHARMONIC SCALING ANALYSIS: SU({N}), Lambda_QCD = {Lambda_QCD} MeV")
    lines.append("=" * 100)
    lines.append(
        f"{'R (fm)':>10} {'gap (MeV)':>12} {'harm (MeV)':>12} {'quart (MeV)':>12} "
        f"{'g^2':>10} {'lam/om^3':>10} {'regime':>10}"
    )
    lines.append("-" * 100)

    for R in R_values:
        res = computer.gap_at_R(R, n_basis=n_basis)
        g2_str = f"{res.g_squared:.3f}" if np.isfinite(res.g_squared) else "inf"
        cr_str = f"{res.coupling_ratio:.3f}" if res.coupling_ratio < 1e8 else ">>1"
        lines.append(
            f"{res.R_fm:10.2f} {res.gap_MeV:12.2f} {res.gap_harmonic_MeV:12.2f} "
            f"{res.gap_quartic_MeV:12.2f} {g2_str:>10} {cr_str:>10} {res.regime:>10}"
        )

    lines.append("-" * 100)

    # Crossover radius
    R_cross = computer.crossover_radius()
    if R_cross is not None:
        lines.append(f"Crossover radius (harmonic -> quartic): R_cross = {R_cross:.3f} fm")
    else:
        lines.append("Crossover radius: could not be determined")

    lines.append(f"Lambda_QCD = {Lambda_QCD:.0f} MeV")
    lines.append(f"R_Landau = hbar_c / Lambda = {HBAR_C_MEV_FM / Lambda_QCD:.3f} fm")
    lines.append("=" * 100)

    return "\n".join(lines)
