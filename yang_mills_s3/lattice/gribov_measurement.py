"""
Gribov Horizon Measurement on S³ Lattice — Lattice FP operator and gauge fixing.

Implements:
    1. Lattice Coulomb/Landau gauge fixing (maximize Re Tr sum of links)
    2. Lattice Faddeev-Popov operator construction
    3. Spectral measurement of the FP operator on thermalized configurations
    4. R-dependence scan to test gap persistence

The key physics question: does lambda_min(M_FP) * R² stay bounded away
from zero as R → ∞?  If yes, the Gribov region radius is R-independent
and the mass gap persists.

At vacuum (all links = identity): lambda_min(M_FP) = 3/R² (THEOREM).
For thermalized configs: lambda_min should be smaller as configs explore
toward the Gribov horizon.

LABEL: NUMERICAL (lattice measurements, no formal proofs)

References:
    - Gribov 1978: Quantization of non-Abelian gauge theories
    - Zwanziger 1994: Lattice Coulomb gauge and Gribov copies
    - Cucchieri 1997: Gribov copies in lattice Coulomb gauge
    - Our gribov.py: Theoretical Gribov analysis on S³
"""

import numpy as np
from scipy import linalg as la
from .s3_lattice import S3Lattice
from .lattice_ym import LatticeYM


# SU(2) Pauli basis for the Lie algebra (normalized: Tr(T^a T^b) = delta^{ab}/2)
_PAULI = np.array([
    [[0, 1], [1, 0]],       # sigma_1
    [[0, -1j], [1j, 0]],    # sigma_2
    [[1, 0], [0, -1]],      # sigma_3
], dtype=complex)

# Generators T^a = sigma^a / 2
_SU2_GENERATORS = _PAULI / 2.0


def _su_n_generators(N):
    """
    Return generators of su(N) as traceless anti-Hermitian matrices,
    normalized so Tr(T^a T^b) = delta^{ab}/2.

    For SU(2): T^a = sigma^a / 2 (Pauli matrices / 2).
    For SU(N>2): generalized Gell-Mann matrices / 2.

    Parameters
    ----------
    N : int
        Dimension of the fundamental representation.

    Returns
    -------
    list of (N, N) complex arrays : dim_adj = N²-1 generators
    """
    if N == 2:
        return [_SU2_GENERATORS[a] for a in range(3)]

    dim_adj = N * N - 1
    generators = []

    # Off-diagonal symmetric
    for i in range(N):
        for j in range(i + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[i, j] = 0.5
            T[j, i] = 0.5
            generators.append(T)

    # Off-diagonal antisymmetric
    for i in range(N):
        for j in range(i + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[i, j] = -0.5j
            T[j, i] = 0.5j
            generators.append(T)

    # Diagonal
    for k in range(1, N):
        T = np.zeros((N, N), dtype=complex)
        for i in range(k):
            T[i, i] = 1.0
        T[k, k] = -k
        T *= 1.0 / np.sqrt(2.0 * k * (k + 1))
        generators.append(T)

    assert len(generators) == dim_adj
    return generators


def _adjoint_rep(U, generators):
    """
    Compute the adjoint representation matrix R^{ab} of an SU(N) element U.

    R^{ab} = 2 Re Tr(T^a U T^b U^dag)

    Parameters
    ----------
    U : (N, N) complex array
        SU(N) matrix.
    generators : list of (N, N) arrays
        Lie algebra generators T^a.

    Returns
    -------
    (dim_adj, dim_adj) real array
    """
    dim_adj = len(generators)
    R = np.zeros((dim_adj, dim_adj))
    Udag = U.conj().T
    for a in range(dim_adj):
        for b in range(dim_adj):
            R[a, b] = 2.0 * np.real(np.trace(generators[a] @ U @ generators[b] @ Udag))
    return R


# ======================================================================
# Gauge fixing
# ======================================================================

def lattice_gauge_fix(lym, max_iter=500, tol=1e-8, omega=1.7):
    """
    Coulomb/Landau gauge fixing on the lattice.

    Maximizes the gauge-fixing functional:
        R = Sum_{links (x,y)} Re Tr(g(x) U_{xy} g(y)^dag) / N

    by iterative overrelaxation: at each site x, find the SU(N)
    element g(x) that maximizes the local contribution, then update.

    Parameters
    ----------
    lym : LatticeYM
        Lattice YM system (links are modified in-place).
    max_iter : int
        Maximum number of gauge-fixing sweeps.
    tol : float
        Convergence tolerance on the gauge-fixing functional change.
    omega : float
        Overrelaxation parameter (1.0 = no overrelaxation, 1.5-1.8 typical).

    Returns
    -------
    dict with:
        'converged' : bool
        'n_iter'    : int
        'functional': float (final value of R)
        'delta'     : float (last change)
    """
    N = lym.N
    n_verts = lym.lattice.vertex_count()
    edges = lym._edges

    # Build adjacency: for each vertex, list of (edge_idx, is_source)
    # If vertex x is the source of edge (x, y), link is U_{xy}
    # If vertex x is the target of edge (y, x), link is U_{yx}^dag
    adj = {v: [] for v in range(n_verts)}
    for idx, (i, j) in enumerate(edges):
        adj[i].append((idx, True, j))    # edge i->j, x=i is source
        adj[j].append((idx, False, i))   # edge i->j, x=j is target

    def compute_functional():
        """Compute R = Sum_links Re Tr(U) / N."""
        total = 0.0
        for idx in range(lym._n_links):
            total += np.real(np.trace(lym._links[idx])) / N
        return total

    R_old = compute_functional()

    for iteration in range(max_iter):
        # Sweep over all vertices
        for x in range(n_verts):
            # Compute the "staple" sum: W(x) = Sum_{neighbors y} U_{xy}
            # where U_{xy} is the link from x to y (forward or reverse)
            W = np.zeros((N, N), dtype=complex)
            for (idx, is_source, y) in adj[x]:
                if is_source:
                    # Link stored as (x, y): U_{xy}
                    W += lym._links[idx]
                else:
                    # Link stored as (y, x): need U_{xy} = U_{yx}^dag
                    W += lym._links[idx].conj().T

            # For SU(2): the optimal g(x) is W^dag / |W| projected to SU(2)
            # For general SU(N): use polar decomposition W = P * V,
            # optimal gauge is g = V^dag
            if np.linalg.norm(W) < 1e-14:
                continue

            # Polar decomposition: W = U_polar * H (U_polar unitary, H pos semi-def)
            # To maximize Re Tr(g W) = Re Tr(g U_polar H), set g = U_polar^dag
            try:
                U_polar, _ = la.polar(W)
            except Exception:
                continue

            # g = U_polar^dag, projected to SU(N)
            g = U_polar.conj().T
            det = np.linalg.det(g)
            if abs(det) < 1e-14:
                continue
            g = g / (det ** (1.0 / N))

            # Overrelaxation: g_eff = g^omega (approximate for omega != 1)
            # For SU(2), exact overrelaxation is possible
            if abs(omega - 1.0) > 1e-6:
                # Approximate: use matrix log/exp
                try:
                    log_g = la.logm(g)
                    g_eff = la.expm(omega * log_g)
                    # Re-project to SU(N)
                    det_eff = np.linalg.det(g_eff)
                    if abs(det_eff) > 1e-14:
                        g_eff = g_eff / (det_eff ** (1.0 / N))
                    else:
                        g_eff = g
                except Exception:
                    g_eff = g
            else:
                g_eff = g

            # Apply gauge transformation at site x:
            # U_{xy} -> g(x) U_{xy}   for all links leaving x
            # U_{yx} -> U_{yx} g(x)^dag  for all links arriving at x
            g_dag = g_eff.conj().T
            for (idx, is_source, y) in adj[x]:
                if is_source:
                    lym._links[idx] = g_eff @ lym._links[idx]
                else:
                    lym._links[idx] = lym._links[idx] @ g_dag

        R_new = compute_functional()
        delta = abs(R_new - R_old)
        R_old = R_new

        if delta < tol:
            return {
                'converged': True,
                'n_iter': iteration + 1,
                'functional': R_new,
                'delta': delta,
            }

    return {
        'converged': False,
        'n_iter': max_iter,
        'functional': R_old,
        'delta': delta if max_iter > 0 else 0.0,
    }


# ======================================================================
# Faddeev-Popov operator
# ======================================================================

def build_fp_operator(lym):
    """
    Build the lattice Faddeev-Popov operator M_FP.

    M_FP = -nabla . D where nabla is the lattice gradient and D is the
    gauge-covariant derivative.

    On the lattice, the matrix elements are:

        M_{xy}^{ab} = Sum_{z: neighbor of x} [
            delta_{xy} * delta^{ab} * Re Tr(U_{xz}) / N
            - delta_{y in neighbors(x)} * Re Tr(T^a U_{xy} T^b U_{xy}^dag) * 2
        ]

    More precisely, for each link (x, y):
        Diagonal (x=y):  M_{xx}^{ab} += delta^{ab} * (1/N) Re Tr(U_{xz}) for each neighbor z
        Off-diagonal:     M_{xy}^{ab} -= R^{ab}(U_{xy}) / 2  [adjoint rep of link]

    Wait — the standard lattice FP operator is:

        M_{xy}^{ab} = Sum_{mu} [
            2 delta_{xy} delta^{ab}  -  delta_{y,x+mu} R^{ab}(U_mu(x))
                                     -  delta_{y,x-mu} R^{ab}(U_mu(x-mu)^dag)
        ] / a^2

    For our irregular lattice with valence z_x = 12:

        M_{xx}^{ab} = z_x * delta^{ab}     (diagonal = valence)
        M_{xy}^{ab} = -R^{ab}(U_{xy})      (off-diagonal, for neighbors y of x)

    This gives the lattice Laplacian structure: M = z*I - A (adjacency).
    At vacuum (all U=I, R^{ab} = delta^{ab}): M = z*I - A = lattice Laplacian.

    The overall scale is set by the lattice spacing a ~ R * edge_length_fraction.

    Parameters
    ----------
    lym : LatticeYM
        Lattice YM system (should be gauge-fixed first).

    Returns
    -------
    M : (n_verts * dim_adj, n_verts * dim_adj) real array
        The FP operator matrix.
    """
    N = lym.N
    dim_adj = N * N - 1
    n_verts = lym.lattice.vertex_count()
    edges = lym._edges
    generators = _su_n_generators(N)

    dim_total = n_verts * dim_adj
    M = np.zeros((dim_total, dim_total))

    # Build adjacency with link info
    adj = {v: [] for v in range(n_verts)}
    for idx, (i, j) in enumerate(edges):
        adj[i].append((idx, True, j))
        adj[j].append((idx, False, i))

    for x in range(n_verts):
        n_neighbors = len(adj[x])

        # Diagonal block: M_{xx}^{ab} = n_neighbors * delta^{ab}
        for a in range(dim_adj):
            row = x * dim_adj + a
            M[row, row] = float(n_neighbors)

        # Off-diagonal blocks: M_{xy}^{ab} = -R^{ab}(U_{xy})
        for (idx, is_source, y) in adj[x]:
            if is_source:
                U_xy = lym._links[idx]
            else:
                U_xy = lym._links[idx].conj().T

            R_adj = _adjoint_rep(U_xy, generators)

            for a in range(dim_adj):
                for b in range(dim_adj):
                    row = x * dim_adj + a
                    col = y * dim_adj + b
                    M[row, col] -= R_adj[a, b]

    # Scale by 1/a^2 where a is the lattice spacing
    # For a regular lattice on S³(R), the lattice spacing is
    # a = R * (edge_length on unit S³)
    a = lym.lattice.lattice_spacing()
    if a > 1e-14:
        M /= a**2

    return M


def fp_eigenvalues(lym, n_evals=10, gauge_fix=True, max_gf_iter=300):
    """
    Compute the lowest eigenvalues of the Faddeev-Popov operator.

    Parameters
    ----------
    lym : LatticeYM
        Lattice YM system.
    n_evals : int
        Number of lowest eigenvalues to return.
    gauge_fix : bool
        Whether to gauge-fix first.
    max_gf_iter : int
        Max iterations for gauge fixing.

    Returns
    -------
    dict with:
        'eigenvalues'  : array of lowest n_evals eigenvalues
        'gf_result'    : gauge fixing result dict (if gauge_fix=True)
        'n_zero_modes' : count of eigenvalues < threshold
    """
    gf_result = None
    if gauge_fix:
        gf_result = lattice_gauge_fix(lym, max_iter=max_gf_iter)

    M = build_fp_operator(lym)

    # M should be symmetric — symmetrize to handle numerical noise
    M = (M + M.T) / 2.0

    # Full diagonalization (matrix is ~360x360 for SU(2) on 120-cell)
    eigenvalues = np.linalg.eigvalsh(M)

    # Sort ascending
    eigenvalues = np.sort(eigenvalues)

    # Count zero modes (eigenvalues below threshold)
    zero_threshold = 1e-6 * max(abs(eigenvalues[-1]), 1.0)
    n_zero = np.sum(eigenvalues < zero_threshold)

    return {
        'eigenvalues': eigenvalues[:n_evals],
        'all_eigenvalues': eigenvalues,
        'gf_result': gf_result,
        'n_zero_modes': int(n_zero),
    }


# ======================================================================
# Measurement functions
# ======================================================================

def measure_gribov_spectrum(R, N=2, beta=None, n_configs=50, n_therm=200,
                            n_evals=10, rng=None):
    """
    Measure the FP operator spectrum on thermalized lattice configurations.

    NUMERICAL: This is a Monte Carlo measurement, not a formal proof.

    For each thermalized configuration:
        1. Gauge fix to Coulomb gauge
        2. Build the FP operator
        3. Compute lowest eigenvalues

    Parameters
    ----------
    R : float
        Radius of S³.
    N : int
        Gauge group SU(N). Default 2.
    beta : float or None
        Lattice coupling. If None, uses asymptotic freedom estimate:
        beta = 2N / g²(R) with g²(R) from one-loop running.
    n_configs : int
        Number of independent configurations.
    n_therm : int
        Number of thermalization sweeps (total for initial thermalization).
    n_evals : int
        Number of lowest eigenvalues to compute per config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict with:
        'eigenvalues'        : (n_configs, n_evals) array
        'lambda_min_mean'    : mean of lowest eigenvalue
        'lambda_min_std'     : std of lowest eigenvalue
        'lambda_min_R2_mean' : mean of lambda_min * R²  (should be ~3 at vacuum)
        'n_zero_modes_mean'  : average number of near-zero modes
        'R'                  : radius used
        'beta'               : coupling used
        'label'              : 'NUMERICAL'
    """
    if rng is None:
        rng = np.random.default_rng()

    if beta is None:
        beta = _default_beta(R, N)

    lattice = S3Lattice(R=R)
    lym = LatticeYM(lattice, N=N, beta=beta)

    # Start from random config and thermalize
    lym.randomize_links(rng)
    lym.thermalize(n_sweeps=n_therm, epsilon=0.3, rng=rng)

    all_evals = []
    n_zeros = []

    for config_idx in range(n_configs):
        # Additional thermalization between measurements (decorrelation)
        lym.thermalize(n_sweeps=max(5, n_therm // 10), epsilon=0.3, rng=rng)

        # Save links (gauge fixing modifies them)
        saved_links = lym._links.copy()

        # Measure FP spectrum
        result = fp_eigenvalues(lym, n_evals=n_evals, gauge_fix=True)
        all_evals.append(result['eigenvalues'][:n_evals])
        n_zeros.append(result['n_zero_modes'])

        # Restore links (undo gauge fixing for next MC step)
        lym._links = saved_links

    all_evals = np.array(all_evals)

    # The lowest eigenvalue across configs (skip zero modes)
    # For each config, the lowest non-trivial eigenvalue
    dim_adj = N * N - 1
    # Expect dim_adj zero modes from global gauge invariance
    # Take the first eigenvalue after the zero modes
    lambda_mins = []
    for i in range(n_configs):
        evals_i = all_evals[i]
        # Find first eigenvalue above zero-mode threshold
        threshold = 1e-6 * max(abs(evals_i[-1]), 1.0) if len(evals_i) > 0 else 1e-6
        nonzero = evals_i[evals_i > threshold]
        if len(nonzero) > 0:
            lambda_mins.append(nonzero[0])
        else:
            lambda_mins.append(0.0)

    lambda_mins = np.array(lambda_mins)

    return {
        'eigenvalues': all_evals,
        'lambda_min_mean': float(np.mean(lambda_mins)),
        'lambda_min_std': float(np.std(lambda_mins)),
        'lambda_min_R2_mean': float(np.mean(lambda_mins) * R**2),
        'lambda_min_R2_std': float(np.std(lambda_mins) * R**2),
        'n_zero_modes_mean': float(np.mean(n_zeros)),
        'R': R,
        'beta': beta,
        'N': N,
        'n_configs': n_configs,
        'label': 'NUMERICAL',
    }


def gribov_vs_radius(R_values, N=2, n_configs=20, n_therm=100, rng=None):
    """
    Scan lambda_min(FP) vs R to test if the gap persists.

    NUMERICAL: The key measurement for the mass gap problem.

    If lambda_min(FP) * R² stays bounded away from 0, the Gribov region
    radius is R-independent and the mass gap persists as R → ∞.

    Parameters
    ----------
    R_values : list of float
        Radii to scan.
    N : int
        Gauge group SU(N).
    n_configs : int
        Configs per R value.
    n_therm : int
        Thermalization sweeps.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict with:
        'R_values'          : list of R
        'lambda_min_mean'   : list of mean lambda_min for each R
        'lambda_min_std'    : list of std lambda_min
        'lambda_min_R2'     : list of lambda_min * R² (key observable)
        'beta_values'       : list of beta used
        'gap_persists'      : bool (True if lambda_min * R² > 0 for all R)
        'label'             : 'NUMERICAL'
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results = {
        'R_values': [],
        'lambda_min_mean': [],
        'lambda_min_std': [],
        'lambda_min_R2': [],
        'beta_values': [],
    }

    for R in R_values:
        meas = measure_gribov_spectrum(
            R=R, N=N, n_configs=n_configs, n_therm=n_therm, rng=rng,
        )
        results['R_values'].append(R)
        results['lambda_min_mean'].append(meas['lambda_min_mean'])
        results['lambda_min_std'].append(meas['lambda_min_std'])
        results['lambda_min_R2'].append(meas['lambda_min_R2_mean'])
        results['beta_values'].append(meas['beta'])

    # Check if gap persists
    lam_R2 = np.array(results['lambda_min_R2'])
    results['gap_persists'] = bool(np.all(lam_R2 > 0.1))
    results['label'] = 'NUMERICAL'

    return results


def quick_gribov_check(R=1.0, N=2, n_configs=5, n_therm=50):
    """
    Quick diagnostic: run Gribov measurement with minimal configs.

    Useful for testing that everything works before a long run.

    Parameters
    ----------
    R : float
        Radius of S³.
    N : int
        Gauge group SU(N).
    n_configs : int
        Number of configs (keep small for speed).
    n_therm : int
        Thermalization sweeps.

    Returns
    -------
    dict with measurement results + diagnostic info.
    """
    rng = np.random.default_rng(42)

    # 1. Vacuum check
    lattice = S3Lattice(R=R)
    lym_vac = LatticeYM(lattice, N=N, beta=1.0)
    vac_result = fp_eigenvalues(lym_vac, n_evals=10, gauge_fix=False)

    # Expected: dim_adj zero modes, then 3/R² (with lattice corrections)
    dim_adj = N * N - 1
    vac_evals = vac_result['all_eigenvalues']
    # Get first non-zero eigenvalue
    threshold = 1e-6 * max(abs(vac_evals[-1]), 1.0)
    nonzero_vac = vac_evals[vac_evals > threshold]
    vac_lambda_min = float(nonzero_vac[0]) if len(nonzero_vac) > 0 else 0.0

    # 2. Thermalized check
    therm_result = measure_gribov_spectrum(
        R=R, N=N, n_configs=n_configs, n_therm=n_therm, rng=rng,
    )

    return {
        'vacuum': {
            'n_zero_modes': vac_result['n_zero_modes'],
            'expected_zero_modes': dim_adj,
            'lambda_min': vac_lambda_min,
            'lambda_min_theory': 3.0 / R**2,
            'ratio_to_theory': vac_lambda_min / (3.0 / R**2) if R > 0 else 0.0,
        },
        'thermalized': {
            'lambda_min_mean': therm_result['lambda_min_mean'],
            'lambda_min_std': therm_result['lambda_min_std'],
            'lambda_min_R2': therm_result['lambda_min_R2_mean'],
            'n_zero_modes_mean': therm_result['n_zero_modes_mean'],
        },
        'R': R,
        'N': N,
        'beta': therm_result['beta'],
        'label': 'NUMERICAL',
    }


# ======================================================================
# Helpers
# ======================================================================

def _default_beta(R, N):
    """
    Default lattice coupling from asymptotic freedom.

    At one loop: g²(mu) = g²(mu_0) / (1 + b_0 g²(mu_0) log(mu/mu_0) / (8pi²))
    where b_0 = 11N/3 for pure YM.

    We set mu = 1/(a) where a is the lattice spacing.
    For Lambda_QCD ~ 0.3 GeV and R in fm:
        g² ~ 4pi² / (b_0 * log(1/(a * Lambda_QCD)))

    For the 600-cell: a ~ R * 0.618 / pi (edge length on S³(R)).

    beta = 2N / g²

    Parameters
    ----------
    R : float
        Radius of S³.
    N : int
        N for SU(N).

    Returns
    -------
    float : beta value
    """
    # Edge length on unit S³ for 600-cell: 2*sin(pi/10) = 1/phi ~ 0.618
    a = R * 0.618  # approximate lattice spacing

    # Lambda_QCD scale (use Lambda ~ 0.3 GeV ~ 1/(0.66 fm))
    # In natural units where R is dimensionless, set Lambda_QCD * R ~ 1 at R ~ 2 fm
    # Simpler: use a reasonable beta that gives good thermalization
    # For SU(2) on coarse lattice: beta ~ 2-4 is typical
    b0 = 11.0 * N / 3.0

    # Simple asymptotic freedom formula
    # g² = 16 pi² / (b0 * log((1/a)² / Lambda²))
    # With Lambda ~ 1/(2*R) as a rough scale:
    log_arg = max(1.0 / (a * 0.5), 2.0)  # Avoid log(0)
    g_sq = 16.0 * np.pi**2 / (b0 * np.log(log_arg)**2 + 1.0)

    # Clamp to reasonable range
    g_sq = max(0.5, min(g_sq, 20.0))

    beta = 2.0 * N / g_sq

    # Clamp beta to a practical range
    return max(1.0, min(beta, 10.0))
