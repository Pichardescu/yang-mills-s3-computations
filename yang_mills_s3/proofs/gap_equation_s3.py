"""
Self-Consistent Gap Equation for Yang-Mills on S^3(R).

Demonstrates DIMENSIONAL TRANSMUTATION: the emergence of an R-independent
mass scale from the self-interaction of the gauge field.

THE PHYSICS:
    On S^3(R), the gluon field has discrete modes j = 0, 1, 2, ... with
    bare eigenvalues lambda_j = (j+1)^2/R^2 (coexact 1-forms on S^3).
    Quantum corrections (one-loop self-energy) shift these eigenvalues:

        m_j^2(R) = lambda_j(R) + Pi_j(R, {m_k})

    where Pi_j is the self-energy that depends on ALL other masses.
    Solving self-consistently gives a non-trivial mass even when the
    classical theory has no intrinsic scale.

THE GAP EQUATION (Cornwall-type on S^3):

    Pi_j = (C_2(adj) / Vol(S^3)) * sum_k d_k * g^2(mu_k) / (lam_k + m_k^2)

    where:
    - C_2(adj) = N for SU(N)
    - d_k = 2*(k+1)*(k+3) = coexact Hodge multiplicity on S^3
    - g^2(mu_k) = running coupling at scale mu_k = (k+1)/R
    - lam_k = (k+1)^2/R^2 = bare eigenvalue (plays role of q^2)
    - Vol(S^3) = 2*pi^2*R^3

    The running coupling provides automatic UV regularization:
    g^2(mu) ~ 1/ln(mu/Lambda) -> 0 as mu -> infinity.

DIMENSIONAL TRANSMUTATION:

    At large R, split the mode sum at k* = m_dyn * R:

    LOW MODES (k < k*): lam_k << m_dyn^2
        term ~ d_k * g^2_IR / m_dyn^2 ~ k^2 / m_dyn^2
        partial sum ~ (m_dyn*R)^3 / m_dyn^2 = m_dyn * R^3

    HIGH MODES (k > k*): lam_k >> m_dyn^2
        term ~ d_k * g^2(k/R) / lam_k ~ k^2 * g^2(k/R) * R^2 / k^2
        = R^2 * g^2(k/R)  [asymptotic freedom tames this]

    Pi ~ C_2/R^3 * (m_dyn*R^3 + R^2 * sum_high g^2(k/R))
       ~ C_2 * m_dyn + C_2/R * sum_high g^2(k/R)

    The second term becomes a convergent integral at large R:
    (1/R)*sum_k g^2(k/R) -> int dmu g^2(mu) [finite by AF]

    Self-consistency: m_dyn^2 ~ g^2_IR * m_dyn => m_dyn ~ g^2_IR
    => R-INDEPENDENT dynamical mass.

    j_max MUST scale with R to capture all relevant modes.
    We use j_max = max(j_max_min, alpha * Lambda_QCD * R)
    where alpha ~ 5 gives good convergence.

LABEL: NUMERICAL (self-consistent gap equation solved numerically)

References:
    - 't Hooft 1973: Dimensional transmutation
    - Cornwall 1982: Dynamical mass generation in continuum QCD
    - Aguilar & Papavassiliou 2008: Gluon mass generation
    - Roberts & Williams 1994: Dyson-Schwinger equations
"""

import numpy as np
from scipy.optimize import brentq


# ======================================================================
# Physical constants
# ======================================================================
HBAR_C_MEV_FM = 197.3269804   # hbar*c in MeV*fm
LAMBDA_QCD_MEV = 200.0        # MeV
LAMBDA_QCD_FM_INV = LAMBDA_QCD_MEV / HBAR_C_MEV_FM  # ~ 1.014 fm^{-1}


# ======================================================================
# Running coupling (vectorized)
# ======================================================================

def running_coupling_g2(R_fm, N_c=2, Lambda_QCD_MeV=LAMBDA_QCD_MEV):
    """
    1-loop running coupling g^2(mu) at scale mu = hbar*c/R.

    Uses smooth IR saturation:
        g^2(R) = 1 / (1/g2_max + b0 * ln(1 + mu^2/Lambda^2))

    where mu = hbar_c/R.

    At small R (high energy): g^2 -> 0 (asymptotic freedom).
    At large R (low energy): g^2 -> g2_max = 4*pi (IR saturation).

    Parameters
    ----------
    R_fm : float or ndarray
        Radius in fm (or effective R for mode-dependent coupling).
    N_c : int
        Number of colors.
    Lambda_QCD_MeV : float
        QCD scale in MeV.

    Returns
    -------
    float or ndarray
        g^2 at scale mu = hbar_c/R.
    """
    b0 = 11.0 * N_c / (48.0 * np.pi**2)
    g2_max = 4.0 * np.pi  # IR saturation ~ 12.57
    R_fm = np.asarray(R_fm, dtype=float)
    mu = HBAR_C_MEV_FM / np.maximum(R_fm, 1e-6)  # MeV, clamp to avoid div/0
    log_arg = 1.0 + (mu / Lambda_QCD_MeV)**2
    result = 1.0 / (1.0 / g2_max + b0 * np.log(log_arg))
    return float(result) if result.ndim == 0 else result


def _g2_array(R, j_max, N_c):
    """
    Precompute g^2(mu_k) for k = 0, ..., j_max where mu_k = (k+1)/R.

    Returns ndarray of shape (j_max+1,).
    """
    k_arr = np.arange(j_max + 1)
    R_eff = R / (k_arr + 1)  # effective "radius" for mode k
    R_eff = np.maximum(R_eff, 0.001)  # clamp
    return running_coupling_g2(R_eff, N_c)


class GapEquationS3:
    """
    Self-consistent gap equation for SU(N) Yang-Mills on S^3(R).

    Solves the Schwinger-Dyson-type gap equation:
        m_j^2 = bare_j + Pi_j({m_k})
    iteratively until convergence.

    Units: natural units where hbar = c = 1. Lengths in fm, masses
    in fm^{-1}. Convert to MeV by multiplying by HBAR_C_MEV_FM.
    """

    def __init__(self, R, g2, N_c=2, j_max=50):
        """
        Parameters
        ----------
        R : float
            Radius of S^3 in fm.
        g2 : float
            Yang-Mills coupling g^2 (dimensionless, used as IR value).
        N_c : int
            Number of colors (default 2 for SU(2)).
        j_max : int
            Maximum mode number (UV cutoff).
        """
        if R <= 0:
            raise ValueError(f"Radius must be positive, got {R}")
        if g2 <= 0:
            raise ValueError(f"Coupling must be positive, got {g2}")
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        if j_max < 1:
            raise ValueError(f"j_max must be >= 1, got {j_max}")

        self.R = R
        self.g2 = g2
        self.N_c = N_c
        self.j_max = j_max
        self.dim_adj = N_c**2 - 1  # dim of adjoint rep
        self.C2_adj = N_c          # quadratic Casimir of adjoint = N

        # Precompute arrays for vectorized operations
        self._k_arr = np.arange(j_max + 1)
        self._lam_arr = (self._k_arr + 1)**2 / R**2
        self._d_arr = 2.0 * (self._k_arr + 1) * (self._k_arr + 3)
        self._g2_arr = _g2_array(R, j_max, N_c)
        self._vol = 2.0 * np.pi**2 * R**3

    # ------------------------------------------------------------------
    # Bare spectrum
    # ------------------------------------------------------------------

    def bare_eigenvalue(self, j):
        """
        Bare Hodge Laplacian eigenvalue for coexact mode j on S^3(R).

        lambda_j = (j+1)^2 / R^2   for j = 0, 1, 2, ...

        Parameters
        ----------
        j : int
            Mode index (j >= 0).

        Returns
        -------
        float
            Eigenvalue in fm^{-2}.
        """
        return (j + 1)**2 / self.R**2

    def bare_mass(self, j):
        """Bare mass of mode j: sqrt(lambda_j) = (j+1)/R."""
        return (j + 1) / self.R

    def multiplicity(self, j):
        """
        Multiplicity of coexact mode j on S^3, including color.

        Hodge multiplicity at level j: 2*(j+1)*(j+3)
        Color factor: dim(adj) = N_c^2 - 1
        Total = 2*(j+1)*(j+3) * (N_c^2 - 1)

        Parameters
        ----------
        j : int
            Mode index.

        Returns
        -------
        int
            Total multiplicity (Hodge x color).
        """
        k = j + 1
        hodge_mult = 2 * k * (k + 2)
        return hodge_mult * self.dim_adj

    def hodge_multiplicity(self, j):
        """Hodge multiplicity only (no color factor)."""
        k = j + 1
        return 2 * k * (k + 2)

    # ------------------------------------------------------------------
    # Self-energy (vectorized)
    # ------------------------------------------------------------------

    def self_energy(self, j, masses):
        """
        One-loop self-energy Pi_j for mode j given all masses {m_k}.

        Pi_j = (C_2(adj) / Vol) * sum_k d_k * g^2(mu_k) / (lam_k + m_k^2)

        Uses precomputed arrays for speed.

        Parameters
        ----------
        j : int
            Mode index (currently unused; Pi_j is j-independent in the
            contact interaction approximation).
        masses : ndarray
            Current mass estimates m_k for k = 0, ..., j_max.

        Returns
        -------
        float
            Self-energy Pi_j in fm^{-2}.
        """
        m_sq = masses**2
        denom = self._lam_arr + m_sq
        # Running coupling provides UV convergence
        terms = self._d_arr * self._g2_arr / denom
        return self.C2_adj / self._vol * np.sum(terms)

    def self_energy_all(self, masses):
        """
        Compute self-energy for ALL modes simultaneously.

        In the contact interaction approximation, Pi_j is the same
        for all j (the external mode doesn't appear in the tadpole).

        Returns
        -------
        float
            The common self-energy value.
        """
        return self.self_energy(0, masses)

    # ------------------------------------------------------------------
    # Solver (vectorized)
    # ------------------------------------------------------------------

    def solve(self, tol=1e-8, max_iter=2000, damping=0.5):
        """
        Solve the gap equation self-consistently by damped iteration.

        Uses a mixing scheme:
            m^{n+1} = damping * m^{n+1}_{raw} + (1-damping) * m^{n}

        Parameters
        ----------
        tol : float
            Convergence tolerance (relative change in m_0).
        max_iter : int
            Maximum number of iterations.
        damping : float
            Mixing parameter in (0, 1]. 1.0 = no damping.

        Returns
        -------
        dict with solution data.
        """
        # Initialize from bare masses
        masses = np.sqrt(self._lam_arr.copy())
        history = [masses[0] * HBAR_C_MEV_FM]

        converged = False
        final_residual = 1.0

        for it in range(max_iter):
            # Self-energy (same for all modes in contact approximation)
            pi = self.self_energy_all(masses)

            # New mass^2 = bare + self-energy
            new_m_sq = self._lam_arr + pi
            new_masses_raw = np.sqrt(np.maximum(new_m_sq, 1e-30))

            # Damped update
            new_masses = damping * new_masses_raw + (1.0 - damping) * masses

            # Convergence check
            if masses[0] > 0:
                rel_change = abs(new_masses[0] - masses[0]) / masses[0]
            else:
                rel_change = abs(new_masses[0] - masses[0])

            masses = new_masses
            history.append(masses[0] * HBAR_C_MEV_FM)
            final_residual = rel_change

            if rel_change < tol:
                converged = True
                break

        masses_MeV = masses * HBAR_C_MEV_FM

        return {
            'masses': masses,
            'masses_MeV': masses_MeV,
            'gap_MeV': masses_MeV[0],
            'gap_fm_inv': masses[0],
            'converged': converged,
            'iterations': it + 1 if converged else max_iter,
            'history': history,
            'residual': final_residual,
            'R': self.R,
            'g2': self.g2,
            'N_c': self.N_c,
            'j_max': self.j_max,
            'self_energy_MeV2': pi * HBAR_C_MEV_FM**2,
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # Physical gap in MeV
    # ------------------------------------------------------------------

    def physical_gap_MeV(self, masses):
        """
        Extract the mass gap in MeV from the solved masses.

        Parameters
        ----------
        masses : ndarray
            Masses in fm^{-1} (as returned by solve()['masses']).

        Returns
        -------
        float
            Mass gap m_0 in MeV.
        """
        return masses[0] * HBAR_C_MEV_FM


# ======================================================================
# Compute physical j_max for a given R
# ======================================================================

def physical_j_max(R_fm, alpha=5.0, j_min=50):
    """
    Compute the physical UV cutoff for the gap equation.

    We need to include all modes up to ~alpha * Lambda_QCD. On S^3(R),
    mode k has momentum (k+1)/R, so the physical cutoff is:

        j_max = max(j_min, int(alpha * Lambda_QCD / hbar_c * R))

    Parameters
    ----------
    R_fm : float
        Radius in fm.
    alpha : float
        How many times Lambda_QCD to include (default 5).
    j_min : int
        Minimum j_max (for small R).

    Returns
    -------
    int
    """
    return max(j_min, int(alpha * LAMBDA_QCD_FM_INV * R_fm))


# ======================================================================
# Main analysis: gap vs R
# ======================================================================

def gap_vs_R(R_values_fm, N_c=2, j_max=None, j_max_fixed=None,
             Lambda_QCD_MeV=LAMBDA_QCD_MEV, tol=1e-8):
    """
    Solve the self-consistent gap equation at multiple radii.

    This is the KEY computation: it shows whether m_0(R) converges
    to an R-independent value at large R (dimensional transmutation).

    Parameters
    ----------
    R_values_fm : list of float
        Radii of S^3 in fm.
    N_c : int
        Number of colors.
    j_max : deprecated, use j_max_fixed.
    j_max_fixed : int or None
        If given, use this fixed j_max for all R.
        If None, j_max scales with R (physical cutoff).
    Lambda_QCD_MeV : float
        QCD scale in MeV.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict with arrays and analysis.
    """
    results_list = []
    R_arr = np.asarray(R_values_fm, dtype=float)
    gap_arr = np.zeros_like(R_arr)
    gap_bare_arr = np.zeros_like(R_arr)
    converged_arr = np.zeros_like(R_arr, dtype=bool)
    iters_arr = np.zeros_like(R_arr, dtype=int)
    g2_arr = np.zeros_like(R_arr)
    jmax_arr = np.zeros_like(R_arr, dtype=int)

    for i, R in enumerate(R_arr):
        g2 = running_coupling_g2(R, N_c, Lambda_QCD_MeV)
        g2_arr[i] = g2

        if j_max_fixed is not None:
            jm = j_max_fixed
        elif j_max is not None:
            jm = j_max
        else:
            jm = physical_j_max(R)
        jmax_arr[i] = jm

        eq = GapEquationS3(R=R, g2=g2, N_c=N_c, j_max=jm)
        result = eq.solve(tol=tol)

        gap_arr[i] = result['gap_MeV']
        gap_bare_arr[i] = eq.bare_mass(0) * HBAR_C_MEV_FM
        converged_arr[i] = result['converged']
        iters_arr[i] = result['iterations']
        results_list.append(result)

    # Analyze R-independence at large R
    large_R_mask = R_arr >= 10.0
    if np.any(large_R_mask):
        gaps_large_R = gap_arr[large_R_mask]
        mean_gap = np.mean(gaps_large_R)
        std_gap = np.std(gaps_large_R)
        rel_var = std_gap / mean_gap if mean_gap > 0 else float('inf')
        plateau = rel_var < 0.10  # Less than 10% variation
    else:
        mean_gap = std_gap = rel_var = float('nan')
        plateau = False

    return {
        'R': R_arr,
        'gap_MeV': gap_arr,
        'bare_gap_MeV': gap_bare_arr,
        'enhancement': gap_arr / gap_bare_arr,
        'g2': g2_arr,
        'converged': converged_arr,
        'iterations': iters_arr,
        'j_max_used': jmax_arr,
        'individual_results': results_list,
        'large_R_analysis': {
            'mean_gap_MeV': mean_gap,
            'std_gap_MeV': std_gap,
            'relative_variation': rel_var,
            'R_independent': plateau,
        },
        'N_c': N_c,
        'Lambda_QCD_MeV': Lambda_QCD_MeV,
        'label': 'NUMERICAL',
    }


# ======================================================================
# UV cutoff independence check
# ======================================================================

def uv_independence(R_fm, j_max_values=None, N_c=2,
                    Lambda_QCD_MeV=LAMBDA_QCD_MEV):
    """
    Check that the mass gap is insensitive to the UV cutoff j_max.

    The running coupling provides UV regularization, so results should
    converge as j_max increases.

    Parameters
    ----------
    R_fm : float
        Radius in fm.
    j_max_values : list of int or None
        Values of j_max to test. Default: [20, 50, 100, 200, 500].
    N_c : int
        Number of colors.
    Lambda_QCD_MeV : float
        QCD scale.

    Returns
    -------
    dict with j_max values and corresponding gaps.
    """
    if j_max_values is None:
        j_max_values = [20, 50, 100, 200, 500]

    g2 = running_coupling_g2(R_fm, N_c, Lambda_QCD_MeV)
    gaps = []

    for jm in j_max_values:
        eq = GapEquationS3(R=R_fm, g2=g2, N_c=N_c, j_max=jm)
        result = eq.solve()
        gaps.append(result['gap_MeV'])

    gaps = np.array(gaps)
    if len(gaps) >= 2 and gaps[-1] > 0:
        rel_change = abs(gaps[-1] - gaps[-2]) / gaps[-1]
    else:
        rel_change = float('inf')

    return {
        'R_fm': R_fm,
        'j_max_values': j_max_values,
        'gap_MeV': gaps.tolist(),
        'final_gap_MeV': gaps[-1],
        'last_relative_change': rel_change,
        'uv_insensitive': rel_change < 0.05,
        'label': 'NUMERICAL',
    }


# ======================================================================
# Dimensional transmutation analysis
# ======================================================================

def dimensional_transmutation_demo(N_c=2, Lambda_QCD_MeV=LAMBDA_QCD_MEV):
    """
    Complete demonstration of dimensional transmutation.

    Solves the gap equation at R = 1, 2, 3, 5, 10, 20, 50, 100, 200, 500 fm
    with PHYSICAL j_max scaling (j_max ~ 5 * Lambda * R).

    Returns
    -------
    dict with complete analysis.
    """
    R_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    # Main computation with physical j_max
    main = gap_vs_R(R_values, N_c=N_c,
                    Lambda_QCD_MeV=Lambda_QCD_MeV)

    # UV independence check at R = 10 fm
    uv_check = uv_independence(10.0, N_c=N_c,
                                Lambda_QCD_MeV=Lambda_QCD_MeV)

    # Identify crossover radius
    gap = main['gap_MeV']
    bare = main['bare_gap_MeV']
    R_arr = main['R']

    crossover_R = float('nan')
    for i in range(len(R_arr)):
        enhancement = gap[i] / bare[i] if bare[i] > 0 else 0
        if enhancement > 2.0:
            crossover_R = R_arr[i]
            break

    gap_over_Lambda = gap[-1] / Lambda_QCD_MeV if gap[-1] > 0 else 0

    return {
        'main_results': main,
        'uv_check': uv_check,
        'crossover_R_fm': crossover_R,
        'gap_over_Lambda': gap_over_Lambda,
        'dimensional_transmutation': main['large_R_analysis']['R_independent'],
        'summary': {
            'R_values_fm': R_values,
            'gap_MeV': gap.tolist(),
            'bare_gap_MeV': bare.tolist(),
            'plateau_value_MeV': main['large_R_analysis']['mean_gap_MeV'],
            'relative_variation': main['large_R_analysis']['relative_variation'],
        },
        'label': 'NUMERICAL',
    }


# ======================================================================
# Analytical explanation of dimensional transmutation
# ======================================================================

def analytical_DT_argument(R_fm, N_c=2, j_max=None):
    """
    Analytical verification of the R-cancellation mechanism.

    At large R, the self-energy Pi becomes:

        Pi ~ (C_2 / Vol) * sum_k d_k * g^2_IR / m_dyn^2  [for low modes k < m_dyn*R]
           ~ (N / R^3) * (m_dyn*R)^3 * g^2_IR / m_dyn^2
           = N * g^2_IR * m_dyn

    Self-consistency m_dyn^2 ~ Pi gives:
        m_dyn^2 ~ N * g^2_IR * m_dyn
        m_dyn ~ N * g^2_IR  [in natural units, fm^{-1}]

    This is R-INDEPENDENT. Converting to MeV:
        m_dyn ~ N * g^2_IR * hbar_c  [MeV]

    Parameters
    ----------
    R_fm : float
    N_c : int
    j_max : int or None
        If None, use physical cutoff.

    Returns
    -------
    dict with the analytical estimates.
    """
    g2 = running_coupling_g2(R_fm, N_c)

    # Analytical estimate: m_dyn ~ C_2(adj) * g^2_IR / (2*pi^2)
    # The 1/(2*pi^2) comes from the angular integration in the sum
    m_dyn_analytical = N_c * g2 / (2.0 * np.pi**2)  # fm^{-1}
    m_dyn_MeV = m_dyn_analytical * HBAR_C_MEV_FM

    # Numerical comparison
    if j_max is None:
        jm = physical_j_max(R_fm)
    else:
        jm = j_max
    eq = GapEquationS3(R=R_fm, g2=g2, N_c=N_c, j_max=jm)
    result = eq.solve()

    return {
        'R_fm': R_fm,
        'g2': g2,
        'j_max': jm,
        'analytical_m_dyn_MeV': m_dyn_MeV,
        'numerical_gap_MeV': result['gap_MeV'],
        'ratio': result['gap_MeV'] / m_dyn_MeV if m_dyn_MeV > 0 else float('nan'),
        'Lambda_QCD_MeV': LAMBDA_QCD_MEV,
        'label': 'NUMERICAL',
    }


# ======================================================================
# Pretty-print results table
# ======================================================================

def print_results_table(results):
    """
    Print a formatted table of gap equation results.

    Parameters
    ----------
    results : dict
        Output of gap_vs_R() or dimensional_transmutation_demo()['main_results'].
    """
    R = results['R']
    gap = results['gap_MeV']
    bare = results['bare_gap_MeV']
    enh = results['enhancement']
    g2 = results['g2']
    conv = results['converged']
    jmax = results.get('j_max_used', np.full_like(R, -1, dtype=int))

    print("=" * 100)
    print(f"{'R (fm)':>10} {'j_max':>7} {'g^2':>8} {'bare gap':>12} {'dressed gap':>12} "
          f"{'enhance':>9} {'conv':>6}")
    print(f"{'':>10} {'':>7} {'':>8} {'(MeV)':>12} {'(MeV)':>12} "
          f"{'(ratio)':>9} {'':>6}")
    print("-" * 100)

    for i in range(len(R)):
        print(f"{R[i]:10.1f} {jmax[i]:7d} {g2[i]:8.3f} {bare[i]:12.2f} {gap[i]:12.2f} "
              f"{enh[i]:9.3f} {'yes' if conv[i] else 'NO':>6}")

    print("-" * 100)
    la = results['large_R_analysis']
    print(f"Large-R plateau: {la['mean_gap_MeV']:.2f} +/- {la['std_gap_MeV']:.2f} MeV")
    print(f"Relative variation: {la['relative_variation']:.4f}")
    print(f"R-independent: {la['R_independent']}")
    print(f"Ratio gap/Lambda_QCD: {la['mean_gap_MeV']/LAMBDA_QCD_MEV:.3f}")
    print("=" * 100)
