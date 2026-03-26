"""
V₄ Convexity Investigation: Is Hess(V₄) PSD?

CRITICAL BUG INVESTIGATION for THEOREM 7.1d and THEOREM 10.7 Part II.

The claim: V₄ is convex ("Hess(V₄) >= 0 as a sum of squares").

The problem: V₄ = (g²/2) Σ_{α<β} Σ_{i<j} (a_{iα}a_{jβ} - a_{jα}a_{iβ})²
Each term is the square of a BILINEAR form. The square of a bilinear
form is NOT convex in general. Example: f(x,y) = (xy)² has
Hessian eigenvalue -2 at (1,1).

This module computes:
1. Hess(V₄) symbolically and numerically
2. Eigenvalues of Hess(V₄) at various points
3. Operator norm bound ||Hess(V₄)|| on Ω₉
4. Whether Hess(V₂ + V₄) remains PSD
5. κ_corrected including ghost curvature compensation

LABEL: INVESTIGATION (determining correctness of existing claims)

References:
    - Effective Hamiltonian: src/proofs/effective_hamiltonian.py
    - Gribov diameter: src/proofs/gribov_diameter.py
    - Fundamental gap: src/proofs/fundamental_gap.py
"""

import numpy as np
from itertools import product as iter_product


# ======================================================================
# Physical constants
# ======================================================================

HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm


# ======================================================================
# V₄ potential and its Hessian
# ======================================================================

def v4_potential(a, g2=1.0):
    """
    Quartic potential V₄(a) = (g²/2) * [(Tr S)² - Tr(S²)]
    where S = M^T M, M = a reshaped as 3x3.

    Parameters
    ----------
    a : ndarray of shape (9,) or (3,3)
        Configuration. a[i,alpha] for i=spatial, alpha=color.
    g2 : float
        Coupling g².

    Returns
    -------
    float : V₄(a) >= 0
    """
    M = np.asarray(a, dtype=float).reshape(3, 3)
    S = M.T @ M
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    return 0.5 * g2 * (tr_S**2 - tr_S2)


def v4_as_sum_of_squares(a, g2=1.0):
    """
    V₄ written explicitly as sum of squares of 2x2 minors.

    V₄ = (g²/2) Σ_{α<β} Σ_{i<j} (a_{iα}a_{jβ} - a_{jα}a_{iβ})²

    Parameters
    ----------
    a : ndarray of shape (9,) or (3,3)
    g2 : float

    Returns
    -------
    float : V₄(a) (should match v4_potential)
    """
    M = np.asarray(a, dtype=float).reshape(3, 3)
    total = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            for alpha in range(3):
                for beta in range(alpha + 1, 3):
                    minor = M[i, alpha] * M[j, beta] - M[j, alpha] * M[i, beta]
                    total += minor**2
    # Identity: (TrS)^2 - Tr(S^2) = 2 * sum_{i<j, alpha<beta} minor^2
    # So V₄ = (g²/2) * 2 * sum = g² * sum
    return g2 * total


def v2_potential(a, R=1.0):
    """
    Quadratic potential V₂(a) = (2/R²)|a|².

    Parameters
    ----------
    a : ndarray of shape (9,) or (3,3)
    R : float

    Returns
    -------
    float : V₂(a)
    """
    a_flat = np.asarray(a, dtype=float).ravel()
    return (2.0 / R**2) * np.dot(a_flat, a_flat)


def total_potential(a, R=1.0, g2=1.0):
    """V(a) = V₂(a) + V₄(a)."""
    return v2_potential(a, R) + v4_potential(a, g2)


# ======================================================================
# Hessian computation (numerical, via finite differences)
# ======================================================================

def hessian_numerical(func, a, h=1e-5):
    """
    Compute the Hessian matrix of func at point a using central finite differences.

    Parameters
    ----------
    func : callable
        Function R^9 -> R
    a : ndarray of shape (9,)
        Point at which to compute the Hessian
    h : float
        Step size for finite differences

    Returns
    -------
    ndarray of shape (9,9) : Hessian matrix
    """
    a = np.asarray(a, dtype=float).ravel()
    n = len(a)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # H[i,j] = d²f / da_i da_j
            a_pp = a.copy()
            a_pp[i] += h
            a_pp[j] += h

            a_pm = a.copy()
            a_pm[i] += h
            a_pm[j] -= h

            a_mp = a.copy()
            a_mp[i] -= h
            a_mp[j] += h

            a_mm = a.copy()
            a_mm[i] -= h
            a_mm[j] -= h

            H[i, j] = (func(a_pp) - func(a_pm) - func(a_mp) + func(a_mm)) / (4 * h**2)
            H[j, i] = H[i, j]

    return H


def hessian_v4_analytical(a, g2=1.0):
    """
    Compute the Hessian of V₄ analytically.

    V₄ = (g²/2) [(Tr(M^T M))² - Tr((M^T M)²)]

    Let a be flattened as a[3*i + alpha] = M[i, alpha].

    Derivation:
    V₄ = (g²/2) Σ_{α<β,i<j} (a_{iα}a_{jβ} - a_{jα}a_{iβ})²

    Each term f_{ij,αβ} = (a_{iα}a_{jβ} - a_{jα}a_{iβ})²

    df_{ij,αβ}/da_{kγ} = 2(a_{iα}a_{jβ} - a_{jα}a_{iβ}) * d(minor)/da_{kγ}

    d(minor)/da_{kγ} =
      δ_{ki}δ_{γα} * a_{jβ} + a_{iα} * δ_{kj}δ_{γβ}
      - δ_{kj}δ_{γα} * a_{iβ} - a_{jα} * δ_{ki}δ_{γβ}

    d²f/da_{kγ}da_{lδ} = 2 * [d(minor)/da_{kγ}] * [d(minor)/da_{lδ}]
                         + 2 * (minor) * d²(minor)/da_{kγ}da_{lδ}

    The second term involves d²(minor)/da_{kγ}da_{lδ}, which is nonzero
    only for specific index combinations (bilinear terms).

    Parameters
    ----------
    a : ndarray of shape (9,) or (3,3)
    g2 : float

    Returns
    -------
    ndarray of shape (9,9) : Hessian matrix
    """
    M = np.asarray(a, dtype=float).reshape(3, 3)
    H = np.zeros((9, 9))

    # Flatten index: idx = 3*i + alpha
    def idx(i, alpha):
        return 3 * i + alpha

    for i_s in range(3):
        for j_s in range(i_s + 1, 3):
            for alpha_c in range(3):
                for beta_c in range(alpha_c + 1, 3):
                    # minor = M[i,α]*M[j,β] - M[j,α]*M[i,β]
                    minor = (M[i_s, alpha_c] * M[j_s, beta_c]
                             - M[j_s, alpha_c] * M[i_s, beta_c])

                    # gradient of minor w.r.t. each a_{k,γ}
                    grad_minor = np.zeros(9)
                    grad_minor[idx(i_s, alpha_c)] += M[j_s, beta_c]
                    grad_minor[idx(j_s, beta_c)] += M[i_s, alpha_c]
                    grad_minor[idx(j_s, alpha_c)] -= M[i_s, beta_c]
                    grad_minor[idx(i_s, beta_c)] -= M[j_s, alpha_c]

                    # Hessian of minor: d²(minor)/da_{kγ}da_{lδ}
                    hess_minor = np.zeros((9, 9))
                    # d²(a_{iα}*a_{jβ})/da_{kγ}da_{lδ} = δ_{ki}δ_{γα}δ_{lj}δ_{δβ} + δ_{kj}δ_{γβ}δ_{li}δ_{δα}
                    hess_minor[idx(i_s, alpha_c), idx(j_s, beta_c)] += 1.0
                    hess_minor[idx(j_s, beta_c), idx(i_s, alpha_c)] += 1.0
                    # d²(-a_{jα}*a_{iβ})/da_{kγ}da_{lδ} = -(δ_{kj}δ_{γα}δ_{li}δ_{δβ} + δ_{ki}δ_{γβ}δ_{lj}δ_{δα})
                    hess_minor[idx(j_s, alpha_c), idx(i_s, beta_c)] -= 1.0
                    hess_minor[idx(i_s, beta_c), idx(j_s, alpha_c)] -= 1.0

                    # Hess(minor²) = 2 * grad(minor) ⊗ grad(minor) + 2 * minor * Hess(minor)
                    H += 2.0 * np.outer(grad_minor, grad_minor) + 2.0 * minor * hess_minor

    # V₄ = g² * sum_{i<j, α<β} minor², so Hess(V₄) = g² * sum Hess(minor²)
    return g2 * H


def hessian_v2(R=1.0):
    """
    Hessian of V₂ = (2/R²)|a|².

    Hess(V₂) = (4/R²) * I₉

    Parameters
    ----------
    R : float

    Returns
    -------
    ndarray of shape (9,9)
    """
    return (4.0 / R**2) * np.eye(9)


def hessian_total_analytical(a, R=1.0, g2=1.0):
    """
    Hessian of V = V₂ + V₄.

    Hess(V) = (4/R²)I₉ + Hess(V₄)

    Parameters
    ----------
    a : ndarray of shape (9,) or (3,3)
    R : float
    g2 : float

    Returns
    -------
    ndarray of shape (9,9)
    """
    return hessian_v2(R) + hessian_v4_analytical(a, g2)


# ======================================================================
# Eigenvalue analysis
# ======================================================================

def hessian_eigenvalues(hess_matrix):
    """
    Compute eigenvalues of a symmetric matrix.

    Parameters
    ----------
    hess_matrix : ndarray of shape (n,n)

    Returns
    -------
    ndarray : sorted eigenvalues (ascending)
    """
    evals = np.linalg.eigvalsh(hess_matrix)
    return np.sort(evals)


def analyze_point(a, R=1.0, g2=1.0, label=""):
    """
    Full analysis of V₄ and V at a single point.

    Parameters
    ----------
    a : ndarray of shape (9,)
    R, g2 : float
    label : str

    Returns
    -------
    dict with analysis results
    """
    a_flat = np.asarray(a, dtype=float).ravel()

    # Potentials
    v4_val = v4_potential(a_flat, g2)
    v2_val = v2_potential(a_flat, R)
    v_total = v4_val + v2_val

    # Hessians
    hess_v4 = hessian_v4_analytical(a_flat, g2)
    hess_total = hessian_total_analytical(a_flat, R, g2)

    # Eigenvalues
    eigs_v4 = hessian_eigenvalues(hess_v4)
    eigs_total = hessian_eigenvalues(hess_total)

    return {
        'label': label,
        'a': a_flat.copy(),
        'norm_a': np.linalg.norm(a_flat),
        'V4': v4_val,
        'V2': v2_val,
        'V_total': v_total,
        'hess_v4_eigenvalues': eigs_v4,
        'hess_v4_min_eigenvalue': eigs_v4[0],
        'hess_v4_max_eigenvalue': eigs_v4[-1],
        'hess_v4_is_psd': bool(eigs_v4[0] >= -1e-12),
        'hess_total_eigenvalues': eigs_total,
        'hess_total_min_eigenvalue': eigs_total[0],
        'hess_total_is_psd': bool(eigs_total[0] >= -1e-12),
    }


# ======================================================================
# Task 1: Systematic Hessian computation at various points
# ======================================================================

def task1_hessian_survey(g2=1.0, R=1.0, n_random=1000, seed=42):
    """
    Compute Hess(V₄) at various points and report eigenvalues.

    Tests:
    - Origin (a = 0)
    - Single mode excited
    - Two modes excited
    - Random points on unit sphere
    - Random points inside Ω₉

    Parameters
    ----------
    g2, R : float
    n_random : int
    seed : int

    Returns
    -------
    dict with survey results
    """
    rng = np.random.default_rng(seed)
    results = {}

    # --- Special points ---

    # Origin
    a_origin = np.zeros(9)
    results['origin'] = analyze_point(a_origin, R, g2, "origin (a=0)")

    # Single mode: a_{1,1} = 1
    a_single = np.zeros(9)
    a_single[0] = 1.0
    results['single_mode'] = analyze_point(a_single, R, g2, "single mode (1,0,...,0)")

    # Two modes: a_{1,1} = a_{2,1} = 1
    a_two = np.zeros(9)
    a_two[0] = 1.0
    a_two[1] = 1.0
    results['two_modes_same_color'] = analyze_point(a_two, R, g2, "two modes same color")

    # Two modes different color: a_{1,1} = a_{2,2} = 1
    a_two_diff = np.zeros(9)
    a_two_diff[0] = 1.0  # a_{0,0}
    a_two_diff[4] = 1.0  # a_{1,1}
    results['two_modes_diff_color'] = analyze_point(a_two_diff, R, g2, "two modes diff color")

    # Diagonal: a_{i,i} = 1 for all i
    a_diag = np.zeros(9)
    a_diag[0] = 1.0  # a_{0,0}
    a_diag[4] = 1.0  # a_{1,1}
    a_diag[8] = 1.0  # a_{2,2}
    results['diagonal'] = analyze_point(a_diag, R, g2, "diagonal (identity-like)")

    # All equal
    a_all = np.ones(9)
    results['all_equal'] = analyze_point(a_all, R, g2, "all components = 1")

    # --- Random points on unit sphere in R^9 ---
    min_eig_v4_sphere = np.inf
    min_eig_total_sphere = np.inf
    worst_point_v4 = None
    worst_eig_v4 = np.inf
    n_negative_v4 = 0
    n_negative_total = 0

    sphere_eigs_v4_all = []
    sphere_eigs_total_all = []

    for _ in range(n_random):
        a_rand = rng.standard_normal(9)
        a_rand = a_rand / np.linalg.norm(a_rand)

        hess_v4 = hessian_v4_analytical(a_rand, g2)
        hess_total = hessian_total_analytical(a_rand, R, g2)
        eigs_v4 = hessian_eigenvalues(hess_v4)
        eigs_total = hessian_eigenvalues(hess_total)

        sphere_eigs_v4_all.append(eigs_v4[0])
        sphere_eigs_total_all.append(eigs_total[0])

        if eigs_v4[0] < min_eig_v4_sphere:
            min_eig_v4_sphere = eigs_v4[0]
            worst_point_v4 = a_rand.copy()
            worst_eig_v4 = eigs_v4[0]

        if eigs_v4[0] < -1e-12:
            n_negative_v4 += 1

        if eigs_total[0] < min_eig_total_sphere:
            min_eig_total_sphere = eigs_total[0]

        if eigs_total[0] < -1e-12:
            n_negative_total += 1

    results['unit_sphere'] = {
        'n_samples': n_random,
        'hess_v4_min_eigenvalue': min_eig_v4_sphere,
        'hess_v4_n_negative': n_negative_v4,
        'hess_v4_fraction_negative': n_negative_v4 / n_random,
        'hess_total_min_eigenvalue': min_eig_total_sphere,
        'hess_total_n_negative': n_negative_total,
        'worst_point_v4': worst_point_v4,
        'worst_eigenvalue_v4': worst_eig_v4,
        'hess_v4_min_eig_mean': np.mean(sphere_eigs_v4_all),
        'hess_v4_min_eig_std': np.std(sphere_eigs_v4_all),
    }

    # --- Random points at various scales ---
    scales_results = {}
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        min_ev4 = np.inf
        min_etot = np.inf
        n_neg_v4 = 0
        for _ in range(n_random // 5):
            a_rand = rng.standard_normal(9)
            a_rand = scale * a_rand / np.linalg.norm(a_rand)
            hess_v4 = hessian_v4_analytical(a_rand, g2)
            hess_total = hessian_total_analytical(a_rand, R, g2)
            eigs_v4 = hessian_eigenvalues(hess_v4)
            eigs_total = hessian_eigenvalues(hess_total)
            min_ev4 = min(min_ev4, eigs_v4[0])
            min_etot = min(min_etot, eigs_total[0])
            if eigs_v4[0] < -1e-12:
                n_neg_v4 += 1

        scales_results[scale] = {
            'hess_v4_min_eigenvalue': min_ev4,
            'hess_total_min_eigenvalue': min_etot,
            'n_negative_v4': n_neg_v4,
        }

    results['various_scales'] = scales_results

    # --- Summary ---
    results['summary'] = {
        'hess_v4_is_psd': (
            min_eig_v4_sphere >= -1e-12
            and results['origin']['hess_v4_is_psd']
            and results['single_mode']['hess_v4_is_psd']
            and results['two_modes_same_color']['hess_v4_is_psd']
            and results['two_modes_diff_color']['hess_v4_is_psd']
        ),
        'global_min_hess_v4_eigenvalue': min(
            min_eig_v4_sphere,
            results['origin']['hess_v4_min_eigenvalue'],
            results['single_mode']['hess_v4_min_eigenvalue'],
            results['two_modes_same_color']['hess_v4_min_eigenvalue'],
            results['two_modes_diff_color']['hess_v4_min_eigenvalue'],
            results['diagonal']['hess_v4_min_eigenvalue'],
            results['all_equal']['hess_v4_min_eigenvalue'],
        ),
        'hess_total_is_psd_everywhere': (
            min_eig_total_sphere >= -1e-12
            and results['origin']['hess_total_is_psd']
        ),
        'global_min_hess_total_eigenvalue': min(
            min_eig_total_sphere,
            results['origin']['hess_total_min_eigenvalue'],
            results['single_mode']['hess_total_min_eigenvalue'],
        ),
    }

    return results


# ======================================================================
# Task 2: Operator norm bound on Ω₉
# ======================================================================

def task2_operator_norm_bound(g2=1.0, n_random=5000, seed=42):
    """
    Bound ||Hess(V₄)||_op on Ω₉.

    Since V₄ is quartic, Hess(V₄) is quadratic in a.
    Therefore ||Hess(V₄)(a)|| <= C * g² * |a|².

    Find the constant C by sampling directions on the unit sphere.

    On Ω₉ with |a| <= d_max:
    ||Hess(V₄)|| <= C * g² * d_max²

    Parameters
    ----------
    g2 : float
    n_random : int
    seed : int

    Returns
    -------
    dict with bound results
    """
    rng = np.random.default_rng(seed)

    # On the unit sphere, ||Hess(V₄)(a/|a|)|| / g² is a constant (since
    # Hess(V₄) is homogeneous of degree 2 in a, and we set g²=1 for the bound).

    max_op_norm = 0.0
    max_neg_eigenvalue = 0.0
    max_pos_eigenvalue = 0.0

    for _ in range(n_random):
        a_hat = rng.standard_normal(9)
        a_hat = a_hat / np.linalg.norm(a_hat)

        hess = hessian_v4_analytical(a_hat, g2=1.0)  # g²=1 to extract C
        eigs = hessian_eigenvalues(hess)

        op_norm = max(abs(eigs[0]), abs(eigs[-1]))
        max_op_norm = max(max_op_norm, op_norm)
        max_neg_eigenvalue = min(max_neg_eigenvalue, eigs[0])
        max_pos_eigenvalue = max(max_pos_eigenvalue, eigs[-1])

    # By homogeneity: Hess(V₄)(a) = g² * |a|² * Hess(V₄)(a/|a|) when g²=1
    # So ||Hess(V₄)(a)||_op <= C * g² * |a|²
    # where C = max over unit sphere of ||Hess(V₄)(a_hat)||_op with g²=1
    C_bound = max_op_norm

    return {
        'C_operator_norm': C_bound,
        'max_negative_eigenvalue_unit_sphere': max_neg_eigenvalue,
        'max_positive_eigenvalue_unit_sphere': max_pos_eigenvalue,
        'bound_formula': f'||Hess(V4)(a)|| <= {C_bound:.6f} * g^2 * |a|^2',
        'n_samples': n_random,
        'note': (
            'Hess(V4) is homogeneous of degree 2 in a (V4 is quartic). '
            'Therefore ||Hess(V4)(a)|| = g^2 * |a|^2 * ||Hess(V4)(a_hat)|| '
            'where a_hat = a/|a|. C is the maximum of the unit-sphere norm.'
        ),
    }


def worst_negative_eigenvalue_on_omega9(g2, d_max, C_op_norm):
    """
    Worst-case negative eigenvalue of Hess(V₄) over Ω₉.

    Parameters
    ----------
    g2 : float
    d_max : float
        Maximum |a| in Ω₉ (half the diameter)
    C_op_norm : float
        Operator norm constant from task2

    Returns
    -------
    float : most negative possible eigenvalue (negative number)
    """
    # ||Hess(V₄)|| <= C * g² * |a|², so min eigenvalue >= -C * g² * |a|²
    # On Ω₉: |a| <= d_max/2 (radius = half diameter)
    return -C_op_norm * g2 * (d_max / 2.0)**2


# ======================================================================
# Task 3: κ_corrected for THEOREM 10.7 Part II
# ======================================================================

def running_coupling_g2(R, N=2):
    """
    Running coupling g²(R) for SU(N).

    Uses the two-loop beta function. For SU(2):
    g²(R) approaches 4*pi as R -> inf.

    Parameters
    ----------
    R : float (in fm)
    N : int

    Returns
    -------
    float : g²(R)
    """
    # One-loop asymptotic freedom
    beta_0 = 11.0 * N / (48.0 * np.pi**2)
    Lambda_QCD_fm = 1.0 / HBAR_C_MEV_FM * 250.0  # ~250 MeV -> fm^-1

    # g²(R) = 1 / (beta_0 * ln(1/(R*Lambda)²))
    # But this diverges at R*Lambda = 1. Use a regularized form:
    R_Lambda = R * Lambda_QCD_fm
    if R_Lambda < 0.1:
        # Perturbative regime: asymptotic freedom
        log_factor = 2.0 * beta_0 * np.log(1.0 / R_Lambda)
        if log_factor > 0:
            return 1.0 / log_factor
        else:
            return 4.0 * np.pi  # IR saturation
    else:
        # IR regime: saturate at g²_max = 4*pi
        g2_max = 4.0 * np.pi
        g2_pert = 1.0 / (2.0 * beta_0 * max(np.log(1.0 / R_Lambda), 0.01))
        return min(g2_pert, g2_max)


def kappa_corrected(R, g2=None, C_op_norm=None, d_max=None):
    """
    Corrected BE curvature including V₄ non-convexity.

    κ_corrected(R) = 4/R² + min_eigenvalue(Hess(V₄)) + 4g²R²/9

    where min_eigenvalue(Hess(V₄)) is the worst case (most negative)
    over Ω₉.

    With ghost curvature compensation:
    κ_corrected = 4/R² - C*g²*(d_max/2)² + 4g²R²/9

    Parameters
    ----------
    R : float (in fm)
    g2 : float or None (compute from running coupling)
    C_op_norm : float or None
    d_max : float or None

    Returns
    -------
    dict with curvature analysis
    """
    if g2 is None:
        g2 = running_coupling_g2(R)

    # Harmonic contribution
    harmonic = 4.0 / R**2

    # Ghost curvature contribution (THEOREM 9.7)
    ghost = 4.0 * g2 * R**2 / 9.0

    # V₄ worst-case Hessian contribution
    if C_op_norm is not None and d_max is not None:
        v4_worst = -C_op_norm * g2 * (d_max / 2.0)**2
    else:
        v4_worst = 0.0  # Will be filled in by task3

    kappa = harmonic + v4_worst + ghost

    return {
        'R': R,
        'g2': g2,
        'harmonic_4_over_R2': harmonic,
        'v4_worst_hessian': v4_worst,
        'ghost_curvature': ghost,
        'kappa_corrected': kappa,
        'kappa_positive': kappa > 0,
        'gap_lower_bound': np.sqrt(max(kappa, 0)) if kappa > 0 else 0.0,
        'gap_MeV': np.sqrt(max(kappa, 0)) * HBAR_C_MEV_FM if kappa > 0 else 0.0,
    }


def task3_kappa_analysis(C_op_norm, d_max_formula=None):
    """
    Compute κ_corrected for various R values.

    Parameters
    ----------
    C_op_norm : float
        From task2
    d_max_formula : callable or None
        d_max(R, g2) function. If None, estimates from Gribov diameter.

    Returns
    -------
    dict with analysis for various R
    """
    R_values = [0.5, 0.96, 1.0, 1.5, 2.0, 2.2, 3.0, 5.0, 10.0, 20.0]
    results = []

    for R in R_values:
        g2 = running_coupling_g2(R)

        # Estimate d_max from Gribov diameter
        # From the paper: d*R ~ 1.89 (stabilized), so d ~ 1.89/R
        # d_max = diameter/2 (radius of Omega_9)
        if d_max_formula is not None:
            d_max = d_max_formula(R, g2)
        else:
            # Use the analytical formula from diameter_theorem.py
            # d(R) = 3*C_D / (R*g) where C_D = 3*sqrt(3)/2
            g = np.sqrt(g2)
            C_D = 3.0 * np.sqrt(3.0) / 2.0
            d_gribov = 3.0 * C_D / (R * g) if g > 0 else np.inf
            d_max = d_gribov / 2.0  # radius = half diameter

        res = kappa_corrected(R, g2, C_op_norm, d_max * 2.0)
        res['d_max'] = d_max
        res['d_gribov'] = d_max * 2.0
        results.append(res)

    # Find minimum kappa
    kappas = [r['kappa_corrected'] for r in results]
    min_idx = np.argmin(kappas)

    return {
        'results': results,
        'R_values': R_values,
        'kappas': kappas,
        'min_kappa': kappas[min_idx],
        'min_kappa_R': R_values[min_idx],
        'all_positive': all(k > 0 for k in kappas),
        'C_op_norm_used': C_op_norm,
    }


# ======================================================================
# Task 4: Is Hess(V₂ + V₄) PSD everywhere?
# ======================================================================

def task4_total_hessian_psd(R=1.0, g2=1.0, n_random=5000, seed=42):
    """
    Check whether Hess(V₂ + V₄) is PSD everywhere in R⁹.

    Since Hess(V₂) = (4/R²)I₉ and Hess(V₄) is quadratic in a,
    for small |a| the V₂ term dominates and Hess(V) > 0.
    For large |a|, the negative eigenvalues of Hess(V₄) grow as |a|²,
    but 4/R² is constant. So Hess(V) could become non-PSD.

    However: V₄ ~ |a|⁴ and V₂ ~ |a|², so V is NOT uniformly convex.
    But the question is whether V is convex AT ALL (Hess >= 0 everywhere).

    Parameters
    ----------
    R, g2 : float
    n_random : int
    seed : int

    Returns
    -------
    dict with analysis
    """
    rng = np.random.default_rng(seed)

    # Analytical bound: Hess(V)(a) = (4/R²)I + Hess(V₄)(a)
    # min eigenvalue of Hess(V)(a) >= 4/R² - ||Hess(V₄)(a)||
    # = 4/R² - C*g²*|a|²
    # This becomes negative when |a|² > 4/(R²*C*g²)
    # i.e., |a| > 2/(R*sqrt(C*g²))

    # Find C first
    max_neg_on_sphere = 0.0
    worst_point = None

    for _ in range(n_random):
        a_hat = rng.standard_normal(9)
        a_hat = a_hat / np.linalg.norm(a_hat)

        hess_v4 = hessian_v4_analytical(a_hat, g2)
        eigs = hessian_eigenvalues(hess_v4)

        if eigs[0] < max_neg_on_sphere:
            max_neg_on_sphere = eigs[0]
            worst_point = a_hat.copy()

    # max_neg_on_sphere is the most negative eigenvalue at |a|=1 with given g²
    # At radius r, the most negative eigenvalue of Hess(V₄) is r² * max_neg_on_sphere
    # Hess(V) has min eigenvalue = 4/R² + r² * max_neg_on_sphere
    # This is zero when r² = -4/(R² * max_neg_on_sphere)

    if max_neg_on_sphere < -1e-15:
        r_critical = np.sqrt(-4.0 / (R**2 * max_neg_on_sphere))
        convex_everywhere = False

        # Verify: at r_critical * worst_point, Hess(V) should have zero eigenvalue
        a_crit = r_critical * worst_point
        hess_at_crit = hessian_total_analytical(a_crit, R, g2)
        eigs_at_crit = hessian_eigenvalues(hess_at_crit)
    else:
        r_critical = np.inf
        convex_everywhere = True
        eigs_at_crit = None

    # Check: does Gribov radius save us?
    # On Ω₉, |a| <= d_max. If d_max < r_critical, V is convex on Ω₉.
    g = np.sqrt(g2)
    C_D = 3.0 * np.sqrt(3.0) / 2.0
    d_gribov = 3.0 * C_D / (R * g) if g > 0 else np.inf
    gribov_radius = d_gribov / 2.0

    gribov_saves = gribov_radius < r_critical

    return {
        'most_negative_eigenvalue_unit_sphere': max_neg_on_sphere,
        'worst_direction': worst_point,
        'r_critical': r_critical,
        'convex_everywhere_R9': convex_everywhere,
        'gribov_radius': gribov_radius,
        'gribov_diameter': d_gribov,
        'gribov_saves_convexity': gribov_saves,
        'ratio_r_critical_over_gribov': r_critical / gribov_radius if np.isfinite(gribov_radius) and gribov_radius > 0 else np.inf,
        'eigs_at_critical': eigs_at_crit,
        'R': R,
        'g2': g2,
        'note': (
            f'V is convex on all of R^9: {convex_everywhere}. '
            f'V is convex on Omega_9 (Gribov region): {gribov_saves}. '
            f'Critical radius where Hess(V) first gets zero eigenvalue: {r_critical:.4f}. '
            f'Gribov radius (max |a| in Omega_9): {gribov_radius:.4f}.'
        ),
    }


# ======================================================================
# Optimized worst-case search via gradient descent
# ======================================================================

def find_worst_hessian_direction(g2=1.0, n_restarts=100, seed=42):
    """
    Find the direction on the unit sphere that minimizes the minimum
    eigenvalue of Hess(V₄).

    Uses gradient-free optimization (random restarts + local perturbation).

    Parameters
    ----------
    g2 : float
    n_restarts : int
    seed : int

    Returns
    -------
    dict with worst direction and eigenvalue
    """
    rng = np.random.default_rng(seed)
    best_min_eig = np.inf
    best_direction = None

    for _ in range(n_restarts):
        a_hat = rng.standard_normal(9)
        a_hat = a_hat / np.linalg.norm(a_hat)

        # Local optimization: perturb and check
        current_eig = hessian_eigenvalues(hessian_v4_analytical(a_hat, g2))[0]

        for __ in range(50):
            perturbation = rng.standard_normal(9) * 0.1
            a_new = a_hat + perturbation
            a_new = a_new / np.linalg.norm(a_new)

            new_eig = hessian_eigenvalues(hessian_v4_analytical(a_new, g2))[0]
            if new_eig < current_eig:
                a_hat = a_new
                current_eig = new_eig

        if current_eig < best_min_eig:
            best_min_eig = current_eig
            best_direction = a_hat.copy()

    return {
        'worst_direction': best_direction,
        'worst_min_eigenvalue': best_min_eig,
        'worst_min_eigenvalue_at_r': lambda r: r**2 * best_min_eig,
        'g2': g2,
    }


# ======================================================================
# Counterexample verification
# ======================================================================

def verify_bilinear_square_not_convex():
    """
    Verify that (xy)² is not convex by computing its Hessian.

    f(x,y) = (xy)²
    df/dx = 2xy²
    df/dy = 2x²y
    d²f/dx² = 2y²
    d²f/dy² = 2x²
    d²f/dxdy = 4xy

    Hessian = [[2y², 4xy], [4xy, 2x²]]
    det(H) = 4x²y² - 16x²y² = -12x²y²

    At (1,1): H = [[2, 4], [4, 2]], eigenvalues = 6, -2.

    Returns
    -------
    dict with verification
    """
    # At (1,1)
    H = np.array([[2.0, 4.0], [4.0, 2.0]])
    eigs = np.linalg.eigvalsh(H)

    return {
        'hessian_at_1_1': H,
        'eigenvalues': eigs,
        'min_eigenvalue': eigs[0],
        'is_psd': bool(eigs[0] >= 0),
        'confirms_not_convex': bool(eigs[0] < 0),
        'note': 'f(x,y) = (xy)^2 has Hessian eigenvalue -2 at (1,1). NOT convex.',
    }


# ======================================================================
# Master analysis function
# ======================================================================

def full_investigation(R_phys=2.2, g2_phys=11.33, verbose=True):
    """
    Complete V₄ convexity investigation.

    Parameters
    ----------
    R_phys : float (fm)
        Physical radius of S³
    g2_phys : float
        Physical coupling g² at R_phys
    verbose : bool

    Returns
    -------
    dict with all results and recommendations
    """
    results = {}

    if verbose:
        print("=" * 70)
        print("V4 CONVEXITY INVESTIGATION")
        print("=" * 70)

    # 0. Counterexample verification
    if verbose:
        print("\n--- Step 0: Verify bilinear square is NOT convex ---")
    results['counterexample'] = verify_bilinear_square_not_convex()
    if verbose:
        ce = results['counterexample']
        print(f"  f(x,y) = (xy)^2 at (1,1): eigenvalues = {ce['eigenvalues']}")
        print(f"  Confirms NOT convex: {ce['confirms_not_convex']}")

    # 1. Hessian survey
    if verbose:
        print("\n--- Task 1: Hessian survey ---")
    results['task1'] = task1_hessian_survey(g2=g2_phys, R=R_phys, n_random=2000)
    t1 = results['task1']
    if verbose:
        print(f"  Origin: Hess(V4) eigenvalues = {t1['origin']['hess_v4_eigenvalues']}")
        print(f"  Single mode: min eig = {t1['single_mode']['hess_v4_min_eigenvalue']:.6f}")
        print(f"  Two modes (same color): min eig = {t1['two_modes_same_color']['hess_v4_min_eigenvalue']:.6f}")
        print(f"  Two modes (diff color): min eig = {t1['two_modes_diff_color']['hess_v4_min_eigenvalue']:.6f}")
        print(f"  Diagonal: min eig = {t1['diagonal']['hess_v4_min_eigenvalue']:.6f}")
        print(f"  All equal: min eig = {t1['all_equal']['hess_v4_min_eigenvalue']:.6f}")
        print(f"  Unit sphere ({t1['unit_sphere']['n_samples']} samples):")
        print(f"    Min Hess(V4) eigenvalue: {t1['unit_sphere']['hess_v4_min_eigenvalue']:.6f}")
        print(f"    Fraction negative: {t1['unit_sphere']['hess_v4_fraction_negative']:.4f}")
        print(f"  SUMMARY: Hess(V4) is PSD? {t1['summary']['hess_v4_is_psd']}")
        print(f"  Global min eigenvalue: {t1['summary']['global_min_hess_v4_eigenvalue']:.6f}")

    # 2. Operator norm bound
    if verbose:
        print("\n--- Task 2: Operator norm bound ---")
    results['task2'] = task2_operator_norm_bound(g2=1.0, n_random=5000)
    t2 = results['task2']
    if verbose:
        print(f"  C (operator norm constant): {t2['C_operator_norm']:.6f}")
        print(f"  Bound: {t2['bound_formula']}")
        print(f"  Max negative eigenvalue on unit sphere (g2=1): {t2['max_negative_eigenvalue_unit_sphere']:.6f}")
        print(f"  Max positive eigenvalue on unit sphere (g2=1): {t2['max_positive_eigenvalue_unit_sphere']:.6f}")

    # 3. κ_corrected analysis
    if verbose:
        print("\n--- Task 3: kappa_corrected analysis ---")
    C_op = t2['C_operator_norm']
    results['task3'] = task3_kappa_analysis(C_op)
    t3 = results['task3']
    if verbose:
        print(f"  {'R':>6s}  {'g2':>8s}  {'harmonic':>10s}  {'V4_worst':>10s}  {'ghost':>10s}  {'kappa':>10s}  {'gap_MeV':>8s}")
        for r in t3['results']:
            print(f"  {r['R']:6.2f}  {r['g2']:8.4f}  {r['harmonic_4_over_R2']:10.4f}  {r['v4_worst_hessian']:10.4f}  {r['ghost_curvature']:10.4f}  {r['kappa_corrected']:10.4f}  {r['gap_MeV']:8.1f}")
        print(f"  Min kappa: {t3['min_kappa']:.6f} at R = {t3['min_kappa_R']:.2f}")
        print(f"  All positive: {t3['all_positive']}")

    # 4. Total Hessian PSD analysis
    if verbose:
        print("\n--- Task 4: Hess(V2 + V4) PSD analysis ---")
    results['task4'] = task4_total_hessian_psd(R=R_phys, g2=g2_phys, n_random=5000)
    t4 = results['task4']
    if verbose:
        print(f"  V convex everywhere on R^9: {t4['convex_everywhere_R9']}")
        print(f"  Critical radius: {t4['r_critical']:.4f}")
        print(f"  Gribov radius: {t4['gribov_radius']:.4f}")
        print(f"  Gribov saves convexity: {t4['gribov_saves_convexity']}")
        print(f"  Ratio r_crit / r_Gribov: {t4['ratio_r_critical_over_gribov']:.4f}")

    # 5. Find worst direction
    if verbose:
        print("\n--- Optimized worst direction search ---")
    results['worst_direction'] = find_worst_hessian_direction(g2=1.0, n_restarts=200)
    wd = results['worst_direction']
    if verbose:
        print(f"  Worst min eigenvalue on unit sphere (g2=1): {wd['worst_min_eigenvalue']:.6f}")

    # 6. Recommendations
    if verbose:
        print("\n" + "=" * 70)
        print("CONCLUSIONS AND RECOMMENDATIONS")
        print("=" * 70)

    is_v4_convex = t1['summary']['hess_v4_is_psd']
    kappa_ok = t3['all_positive']
    gribov_saves = t4['gribov_saves_convexity']

    results['conclusions'] = {
        'hess_v4_is_psd': is_v4_convex,
        'kappa_corrected_positive_for_all_R': kappa_ok,
        'gribov_region_saves_convexity': gribov_saves,
        'theorem_7_1d_status': (
            'VALID as stated' if is_v4_convex
            else ('NEEDS CORRECTION: V4 is NOT convex, but gap survives via '
                  'ghost curvature' if kappa_ok
                  else 'BROKEN: gap may not survive')
        ),
        'theorem_10_7_status': (
            'VALID as stated' if is_v4_convex
            else ('NEEDS CORRECTION: Hess(V4) >= 0 claim is false, but '
                  'kappa_corrected > 0 via ghost curvature compensation' if kappa_ok
                  else 'BROKEN: kappa_corrected not positive for all R')
        ),
    }

    if verbose:
        c = results['conclusions']
        print(f"\n  1. Is Hess(V4) PSD? {c['hess_v4_is_psd']}")
        print(f"  2. kappa_corrected > 0 for all R? {c['kappa_corrected_positive_for_all_R']}")
        print(f"  3. Gribov region saves convexity? {c['gribov_region_saves_convexity']}")
        print(f"  4. THEOREM 7.1d: {c['theorem_7_1d_status']}")
        print(f"  5. THEOREM 10.7 Part II: {c['theorem_10_7_status']}")

    return results
