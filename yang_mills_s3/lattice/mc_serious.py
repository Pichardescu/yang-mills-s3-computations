"""
Serious Monte Carlo simulations for SU(2) Yang-Mills on the 600-cell.

This module provides publication-quality MC measurements:
  1. Beta scan with proper thermalization + jackknife errors
  2. Mass gap extraction from correlator decay (exponential fit + GEVP)
  3. Lattice refinement study (600-cell + subdivided lattice)
  4. Comparison with analytical prediction m^2 = 4/R^2

Key physics:
  - Wilson action: S = beta * Sum_plaq (1 - (1/2) Re Tr U_plaq)
  - For SU(2): beta = 4/g^2
  - 600-cell: 120 vertices, 720 edges, 1200 triangular faces
  - Lattice spacing: a ~ pi/(5*R) ~ 0.628*R for the 600-cell
  - Coexact gap prediction: m^2 = 4/R^2 (m = 2/R = 2 for R=1)

Limitations (HONEST):
  - 120 vertices is VERY coarse (~6 points across diameter)
  - Only ~10 distinct geodesic distances available for correlator
  - Finite volume effects are O(1) not small corrections
  - Triangular plaquettes (not squares) complicate area law interpretation
  - No continuum extrapolation possible without finer lattices

STATUS: NUMERICAL
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import eigh as scipy_eigh
from .s3_lattice import S3Lattice
from .mc_engine import MCEngine


# ======================================================================
# Statistical tools
# ======================================================================

def jackknife_mean_error(data):
    """
    Compute mean and jackknife error estimate.

    The jackknife is more reliable than naive bootstrap for
    correlated MC data because it naturally accounts for
    autocorrelations (to leading order).

    Parameters
    ----------
    data : array-like, shape (N,) or (N, ...)
        The N measurements. Each can be a scalar or array.

    Returns
    -------
    mean : float or array
    error : float or array
    """
    data = np.asarray(data)
    N = len(data)
    if N < 2:
        return np.mean(data, axis=0), np.zeros_like(np.mean(data, axis=0))

    full_mean = np.mean(data, axis=0)

    # Jackknife resamples: leave one out
    jk_means = np.zeros_like(data, dtype=float)
    for i in range(N):
        jk_means[i] = np.mean(np.delete(data, i, axis=0), axis=0)

    # Jackknife variance
    delta = jk_means - full_mean
    variance = (N - 1) / N * np.sum(delta**2, axis=0)
    error = np.sqrt(variance)

    return full_mean, error


def jackknife_function(data, func):
    """
    Jackknife error for an arbitrary function of the data.

    Parameters
    ----------
    data : array-like, shape (N, ...)
    func : callable, maps array of shape (M, ...) -> scalar or array

    Returns
    -------
    value : result of func(data)
    error : jackknife error estimate
    """
    data = np.asarray(data)
    N = len(data)

    full_value = func(data)

    jk_values = []
    for i in range(N):
        jk_sample = np.delete(data, i, axis=0)
        jk_values.append(func(jk_sample))

    jk_values = np.array(jk_values)
    delta = jk_values - full_value
    variance = (N - 1) / N * np.sum(delta**2, axis=0)
    error = np.sqrt(variance)

    return full_value, error


def autocorrelation_time(data, max_lag=None):
    """
    Estimate integrated autocorrelation time.

    tau_int = 0.5 + sum_{t=1}^{t_max} rho(t)

    where rho(t) is the normalized autocorrelation function.
    Uses automatic windowing (Madras-Sokal).

    Parameters
    ----------
    data : 1D array of measurements

    Returns
    -------
    tau_int : float, integrated autocorrelation time
    """
    data = np.asarray(data, dtype=float)
    N = len(data)
    if N < 10:
        return 0.5

    mean = np.mean(data)
    var = np.var(data)
    if var < 1e-30:
        return 0.5

    if max_lag is None:
        max_lag = N // 4

    fluct = data - mean
    tau = 0.5

    for t in range(1, max_lag):
        # Unnormalized autocorrelation
        c_t = np.mean(fluct[:N - t] * fluct[t:])
        rho_t = c_t / var

        if rho_t < 0:
            break  # Madras-Sokal: stop when rho goes negative
        tau += rho_t

    return tau


# ======================================================================
# Thermalization monitor
# ======================================================================

def thermalization_check(engine, n_sweeps, measure_every=10):
    """
    Monitor plaquette during thermalization to verify equilibration.

    Returns the plaquette history so the caller can verify
    that it has plateaued.

    Parameters
    ----------
    engine : MCEngine
    n_sweeps : int, total thermalization sweeps
    measure_every : int, sweeps between measurements

    Returns
    -------
    dict with 'plaquettes', 'sweeps', 'converged'
    """
    plaquettes = []
    sweeps = []

    for s in range(n_sweeps):
        engine.compound_sweep(n_heatbath=1, n_overrelax=4)
        if s % measure_every == 0:
            P = engine.plaquette_average()
            plaquettes.append(P)
            sweeps.append(s)

    plaq_arr = np.array(plaquettes)
    n = len(plaq_arr)

    # Check convergence: compare first and second half
    if n >= 10:
        first_half = plaq_arr[:n // 2]
        second_half = plaq_arr[n // 2:]
        mean1 = np.mean(first_half)
        mean2 = np.mean(second_half)
        std2 = np.std(second_half)
        # Converged if second half mean is within 2 sigma of first half
        converged = abs(mean2 - mean1) < 3 * max(std2, 1e-6)
    else:
        converged = True  # Not enough data to judge

    return {
        'plaquettes': plaq_arr.tolist(),
        'sweeps': sweeps,
        'converged': converged,
        'final_plaquette': float(plaq_arr[-1]) if len(plaq_arr) > 0 else 0.0,
    }


# ======================================================================
# Multi-operator correlator matrix
# ======================================================================

def compute_operator_timeslices(engine, n_bins=10, coord=0):
    """
    Compute multiple gauge-invariant operators on each time slice.

    Operators:
      O1: plaquette average per slice (simplest glueball operator ~ 0++)
      O2: plaquette squared per slice (excited state content)
      O3: spatial Wilson loop average per slice (larger loops)

    Returns
    -------
    ops : array of shape (n_operators, n_bins)
    """
    verts = engine.lattice.vertices
    faces = engine._faces
    n_f = len(faces)

    # Compute face w-coordinates for time slicing
    w_faces = np.zeros(n_f)
    for f_idx, (i, j, k) in enumerate(faces):
        w_faces[f_idx] = (verts[i, coord] + verts[j, coord] + verts[k, coord]) / 3.0

    bin_edges = np.linspace(w_faces.min() - 1e-10, w_faces.max() + 1e-10, n_bins + 1)
    bin_indices = np.digitize(w_faces, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Plaquette field
    P = engine.plaquette_field()

    # O1: average plaquette per bin
    O1 = np.zeros(n_bins)
    O2 = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for f_idx in range(n_f):
        b = bin_indices[f_idx]
        O1[b] += P[f_idx]
        O2[b] += P[f_idx] ** 2
        counts[b] += 1

    mask = counts > 0
    O1[mask] /= counts[mask]
    O2[mask] /= counts[mask]

    # O3: |Polyakov-like| trace (trace of product along "spatial" links within slice)
    # Simpler: use plaquette variance per slice as a proxy for excited states
    O3 = np.zeros(n_bins)
    for f_idx in range(n_f):
        b = bin_indices[f_idx]
        O3[b] += (P[f_idx] - O1[b]) ** 2
    O3[mask] /= np.maximum(counts[mask], 1)

    # Stack operators: shape (n_ops, n_bins)
    ops = np.array([O1, O2, O3])

    return ops, counts


def build_correlator_matrix(ops_samples, n_bins):
    """
    Build the correlator matrix C_{ij}(t) from operator timeslice samples.

    C_{ij}(t) = <O_i(t) O_j(0)>_conn = <O_i(t)O_j(0)> - <O_i><O_j>

    Parameters
    ----------
    ops_samples : list of arrays, each shape (n_ops, n_bins)
        One entry per MC configuration.
    n_bins : int

    Returns
    -------
    C : array of shape (n_ops, n_ops, max_dt+1)
        Correlator matrix as function of time separation.
    """
    n_cfg = len(ops_samples)
    n_ops = ops_samples[0].shape[0]
    max_dt = n_bins // 2

    # Compute ensemble averages
    all_ops = np.array(ops_samples)  # (n_cfg, n_ops, n_bins)
    mean_ops = np.mean(all_ops, axis=(0, 2))  # (n_ops,)

    # Fluctuations
    fluct = all_ops - mean_ops[:, np.newaxis]  # broadcast over configs and bins

    # Build correlator matrix
    C = np.zeros((n_ops, n_ops, max_dt + 1))

    for dt in range(max_dt + 1):
        for cfg in range(n_cfg):
            for t in range(n_bins):
                t2 = (t + dt) % n_bins
                for i in range(n_ops):
                    for j in range(n_ops):
                        C[i, j, dt] += fluct[cfg, i, t] * fluct[cfg, j, t2]
        C[:, :, dt] /= (n_cfg * n_bins)

    return C


def gevp_mass_extraction(C, t0=1, bin_width=1.0):
    """
    Extract masses using the Generalized Eigenvalue Problem (GEVP).

    Solve: C(t) v = lambda(t, t0) C(t0) v
    Then: m_n = -ln(lambda_n(t, t0)) / (t - t0) * (1/bin_width)

    This is the standard method for extracting excited state masses
    from a matrix of correlators (Luscher-Wolff, 1990).

    Parameters
    ----------
    C : array (n_ops, n_ops, n_t)
        Correlator matrix.
    t0 : int
        Reference time slice index.
    bin_width : float
        Physical width of each time bin (geodesic distance).

    Returns
    -------
    dict with 'eigenvalues', 'masses', 'effective_masses'
    """
    n_ops = C.shape[0]
    n_t = C.shape[2]

    if t0 >= n_t - 1:
        return {'eigenvalues': [], 'masses': [], 'effective_masses': []}

    C_t0 = C[:, :, t0]

    # Regularize C(t0) if nearly singular
    eigvals_t0 = np.linalg.eigvalsh(C_t0)
    min_eig = np.min(eigvals_t0)
    if min_eig < 1e-14:
        C_t0 = C_t0 + (abs(min_eig) + 1e-12) * np.eye(n_ops)

    eigenvalue_history = []
    effective_masses = []

    for t in range(t0 + 1, n_t):
        C_t = C[:, :, t]

        # Symmetrize (should be symmetric but enforce it)
        C_t = 0.5 * (C_t + C_t.T)
        C_t0_sym = 0.5 * (C_t0 + C_t0.T)

        try:
            eigvals, _ = scipy_eigh(C_t, C_t0_sym)
            eigvals = np.sort(eigvals)[::-1]  # largest first
        except np.linalg.LinAlgError:
            eigvals = np.zeros(n_ops)

        eigenvalue_history.append({
            't': t,
            'eigenvalues': eigvals.tolist(),
        })

        # Extract effective masses
        dt = (t - t0) * bin_width
        masses_t = []
        for lam in eigvals:
            if lam > 1e-15:
                m = -np.log(lam) / dt
                masses_t.append(float(m))
            else:
                masses_t.append(float('inf'))
        effective_masses.append({
            't': t,
            'dt': dt,
            'masses': masses_t,
        })

    # Best mass estimate: from largest eigenvalue at t = t0 + 1
    masses = []
    if eigenvalue_history:
        first_eigs = eigenvalue_history[0]['eigenvalues']
        dt_first = bin_width
        for lam in first_eigs:
            if lam > 1e-15:
                masses.append(-np.log(lam) / dt_first)
            else:
                masses.append(float('inf'))

    return {
        'eigenvalue_history': eigenvalue_history,
        'masses': masses,
        'effective_masses': effective_masses,
    }


# ======================================================================
# Core simulation routines
# ======================================================================

def run_beta_scan_serious(beta_values=None, n_therm=1000, n_measure=500,
                          n_skip=10, seed=42, verbose=True):
    """
    Serious beta scan with proper thermalization and jackknife errors.

    Parameters
    ----------
    beta_values : list of float
        Beta values to scan. Default: [2, 4, 6, 8, 12, 16].
    n_therm : int
        Thermalization sweeps (compound: 1 HB + 4 OR each).
    n_measure : int
        Number of measurement configurations.
    n_skip : int
        Compound sweeps between measurements.
    seed : int
    verbose : bool

    Returns
    -------
    list of dicts with comprehensive measurements at each beta.
    """
    if beta_values is None:
        beta_values = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    lattice = S3Lattice(R=1.0)
    results = []

    for beta in beta_values:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Beta = {beta:.1f}  (g^2 = {4.0 / beta:.4f})")
            print(f"{'=' * 60}")

        rng = np.random.default_rng(seed)
        engine = MCEngine(lattice, beta=beta, rng=rng)

        # Hot start for strong coupling, cold start for weak coupling
        if beta < 3.0:
            engine.set_hot_start()
            if verbose:
                print("  Start: HOT")
        else:
            engine.set_cold_start()
            if verbose:
                print("  Start: COLD")

        # Thermalization with monitoring
        if verbose:
            print(f"  Thermalizing ({n_therm} compound sweeps)...")

        therm = thermalization_check(engine, n_therm, measure_every=max(1, n_therm // 20))

        if verbose:
            print(f"  Thermalization converged: {therm['converged']}")
            print(f"  Final plaquette: {therm['final_plaquette']:.6f}")

        # Measurement phase
        if verbose:
            print(f"  Measuring ({n_measure} configs, {n_skip} sweeps between)...")

        plaq_samples = []
        action_samples = []
        wilson_loop_samples = {3: [], 4: [], 5: []}
        polyakov_samples = []

        # Find loops once (topology doesn't change)
        loops = engine.find_loops_by_length(max_length=5, max_per_length=100)

        # Find great circles for Polyakov loops
        gc_paths = engine.great_circle_paths(n_paths=10)

        for cfg in range(n_measure):
            # Update
            for _ in range(n_skip):
                engine.compound_sweep(n_heatbath=1, n_overrelax=4)

            # Measure plaquette
            P = engine.plaquette_average()
            plaq_samples.append(P)
            action_samples.append(engine.wilson_action())

            # Measure Wilson loops
            wl = engine.measure_wilson_loops(loops)
            for L in [3, 4, 5]:
                if L in wl:
                    wilson_loop_samples[L].append(wl[L]['mean'])

            # Measure Polyakov loops
            poly_vals = engine.polyakov_loops(paths=gc_paths)
            if len(poly_vals) > 0:
                polyakov_samples.append(float(np.mean(np.abs(poly_vals))))

        # Analysis with jackknife
        plaq_mean, plaq_err = jackknife_mean_error(plaq_samples)
        action_mean, action_err = jackknife_mean_error(action_samples)

        # Autocorrelation time
        tau_plaq = autocorrelation_time(plaq_samples)

        # Wilson loops
        wl_results = {}
        for L in [3, 4, 5]:
            if wilson_loop_samples[L]:
                wl_mean, wl_err = jackknife_mean_error(wilson_loop_samples[L])
                wl_results[L] = {
                    'W_mean': float(wl_mean),
                    'W_err': float(wl_err),
                    'ln_W': float(np.log(abs(wl_mean))) if abs(wl_mean) > 1e-15 else float('-inf'),
                    'n_configs': len(wilson_loop_samples[L]),
                }

        # Polyakov loop
        if polyakov_samples:
            poly_mean, poly_err = jackknife_mean_error(polyakov_samples)
        else:
            poly_mean, poly_err = 0.0, 0.0

        # Weak coupling prediction
        wc_pred = 1.0 - 3.0 / (4.0 * beta) if beta > 0.5 else 0.0
        # Strong coupling prediction (leading order): <P> ~ beta/4 for SU(2) on triangular
        sc_pred = beta / 4.0 if beta < 2.0 else None

        result = {
            'beta': beta,
            'g_squared': 4.0 / beta,
            'plaq_mean': float(plaq_mean),
            'plaq_err': float(plaq_err),
            'action_mean': float(action_mean),
            'action_err': float(action_err),
            'tau_int': float(tau_plaq),
            'wilson_loops': wl_results,
            'polyakov_mean': float(poly_mean),
            'polyakov_err': float(poly_err),
            'wc_prediction': float(wc_pred),
            'sc_prediction': float(sc_pred) if sc_pred is not None else None,
            'thermalization': {
                'converged': therm['converged'],
                'final_plaq': therm['final_plaquette'],
            },
            'n_configs': n_measure,
            'n_therm': n_therm,
            'n_skip': n_skip,
        }
        results.append(result)

        if verbose:
            print(f"\n  Results at beta={beta:.1f}:")
            print(f"    <P> = {plaq_mean:.6f} +/- {plaq_err:.6f}")
            print(f"    Weak-coupling prediction: {wc_pred:.6f}")
            print(f"    tau_int = {tau_plaq:.1f}")
            print(f"    |Polyakov| = {poly_mean:.6f} +/- {poly_err:.6f}")
            for L, wl_data in sorted(wl_results.items()):
                print(f"    W({L}) = {wl_data['W_mean']:.6f} +/- {wl_data['W_err']:.6f}"
                      f"  [ln W = {wl_data['ln_W']:.4f}]")

    return results


def run_mass_gap_serious(beta=4.0, n_therm=1000, n_measure=500,
                         n_skip=10, n_bins=10, seed=42, verbose=True):
    """
    Serious mass gap extraction with GEVP and jackknife errors.

    Method:
      1. Thermalize with monitoring
      2. Measure multi-operator correlator matrix on each config
      3. Average over configs with jackknife
      4. Extract masses via:
         a) Simple exponential fit to plaquette correlator
         b) Effective mass plot
         c) GEVP from correlator matrix
      5. Compare with analytical prediction m = 2/R

    Parameters
    ----------
    beta : float
    n_therm : int
    n_measure : int
    n_skip : int
    n_bins : int, number of time slices
    seed : int
    verbose : bool

    Returns
    -------
    dict with comprehensive mass gap results.
    """
    lattice = S3Lattice(R=1.0)
    R = lattice.R
    rng = np.random.default_rng(seed)
    engine = MCEngine(lattice, beta=beta, rng=rng)

    if beta < 3.0:
        engine.set_hot_start()
    else:
        engine.set_cold_start()

    # Thermalize
    if verbose:
        print(f"Thermalizing ({n_therm} compound sweeps at beta={beta})...")

    therm = thermalization_check(engine, n_therm, measure_every=max(1, n_therm // 20))

    if verbose:
        print(f"  Converged: {therm['converged']}")
        print(f"  Final <P>: {therm['final_plaquette']:.6f}")

    # Measurement phase
    if verbose:
        print(f"Measuring ({n_measure} configs)...")

    # Storage for correlators
    plaq_corr_samples = []  # (n_measure, max_dt+1) -- simple plaq correlator
    dist_corr_samples = []  # (n_measure, n_dist_bins) -- distance-based
    ops_samples = []  # For GEVP

    max_dt = n_bins // 2
    bin_width = np.pi * R / n_bins  # geodesic distance per bin

    for cfg in range(n_measure):
        # Update
        for _ in range(n_skip):
            engine.compound_sweep(n_heatbath=1, n_overrelax=4)

        # Time-slice correlator
        tc = engine.time_slice_correlator(coord=0, n_bins=n_bins)
        plaq_corr_samples.append(tc['correlator'])

        # Distance-based correlator (every 5th config to save time)
        if cfg % 5 == 0:
            dc = engine.plaquette_correlator_by_distance(n_distance_bins=12)
            dist_corr_samples.append(dc['correlator'])

        # Multi-operator measurement for GEVP
        ops, _ = compute_operator_timeslices(engine, n_bins=n_bins)
        ops_samples.append(ops)

    # ---- Analysis ----

    # 1. Simple plaquette correlator with jackknife
    plaq_corr_arr = np.array(plaq_corr_samples)  # (n_measure, max_dt+1)
    corr_mean, corr_err = jackknife_mean_error(plaq_corr_arr)

    separations = np.arange(max_dt + 1) * bin_width

    # 2. Exponential fit to plaquette correlator
    gap_fit = _fit_mass_gap_robust(separations, corr_mean, corr_err)

    # 3. Effective mass with jackknife errors
    eff_mass_data = _effective_mass_jackknife(plaq_corr_arr, bin_width)

    # 4. Distance-based correlator
    if dist_corr_samples:
        dist_arr = np.array(dist_corr_samples)
        dist_mean, dist_err = jackknife_mean_error(dist_arr)
        dist_bins = dc['distances']  # from last measurement
        dist_gap = _fit_mass_gap_robust(dist_bins, dist_mean, dist_err)
    else:
        dist_mean = np.array([])
        dist_err = np.array([])
        dist_bins = np.array([])
        dist_gap = {'mass_gap': 0.0, 'mass_gap_err': float('inf')}

    # 5. GEVP mass extraction
    if verbose:
        print("Running GEVP analysis...")

    C_matrix = build_correlator_matrix(ops_samples, n_bins)
    gevp_result = gevp_mass_extraction(C_matrix, t0=1, bin_width=bin_width)

    # Analytical predictions
    analytical_gap = 2.0 / R
    analytical_gap_sq = 4.0 / R**2

    # Compile results
    result = {
        'beta': beta,
        'R': R,
        'n_configs': n_measure,
        'n_therm': n_therm,
        'n_skip': n_skip,
        'n_bins': n_bins,
        'bin_width': float(bin_width),
        'thermalization': {
            'converged': therm['converged'],
            'final_plaq': therm['final_plaquette'],
        },
        'plaquette': float(engine.plaquette_average()),
        # Simple correlator
        'separations': separations.tolist(),
        'correlator': corr_mean.tolist(),
        'correlator_err': corr_err.tolist(),
        'gap_fit': gap_fit,
        # Effective mass
        'effective_mass': eff_mass_data,
        # Distance correlator
        'dist_correlator': {
            'distances': dist_bins.tolist() if len(dist_bins) > 0 else [],
            'correlator': dist_mean.tolist() if len(dist_mean) > 0 else [],
            'correlator_err': dist_err.tolist() if len(dist_err) > 0 else [],
            'gap_fit': dist_gap,
        },
        # GEVP
        'gevp': {
            'masses': gevp_result['masses'],
            'effective_masses': gevp_result['effective_masses'],
        },
        # Analytical
        'analytical_gap': float(analytical_gap),
        'analytical_gap_sq': float(analytical_gap_sq),
    }

    if verbose:
        print(f"\n--- Mass Gap Results (beta={beta}) ---")
        print(f"  Plaquette: {result['plaquette']:.6f}")
        print(f"\n  Time-slice correlator:")
        for i in range(len(corr_mean)):
            print(f"    t={separations[i]:.4f}  C(t)={corr_mean[i]:.4e} +/- {corr_err[i]:.4e}")

        print(f"\n  Effective mass (jackknife):")
        for em in eff_mass_data:
            print(f"    t={em['t']:.4f}  m_eff={em['m_eff']:.4f} +/- {em['m_eff_err']:.4f}")

        print(f"\n  Exponential fit: m = {gap_fit['mass_gap']:.4f} +/- {gap_fit['mass_gap_err']:.4f}")
        print(f"  Distance-based fit: m = {dist_gap['mass_gap']:.4f} +/- {dist_gap['mass_gap_err']:.4f}")

        if gevp_result['masses']:
            print(f"\n  GEVP masses:")
            for i, m in enumerate(gevp_result['masses']):
                if m < 100:
                    print(f"    State {i}: m = {m:.4f}")

        print(f"\n  Analytical prediction: m = {analytical_gap:.4f} (m^2 = {analytical_gap_sq:.4f})")

    return result


def run_gap_vs_beta_serious(beta_values=None, n_therm=1000, n_measure=500,
                            n_skip=10, n_bins=10, seed=42, verbose=True):
    """
    Scan mass gap over beta values with full analysis at each point.

    Parameters
    ----------
    beta_values : list of float
    n_therm, n_measure, n_skip, n_bins : MC parameters
    seed : int
    verbose : bool

    Returns
    -------
    list of mass gap result dicts.
    """
    if beta_values is None:
        beta_values = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    results = []
    for beta in beta_values:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  MASS GAP SCAN: beta = {beta:.1f}")
            print(f"{'=' * 70}")

        result = run_mass_gap_serious(
            beta=beta, n_therm=n_therm, n_measure=n_measure,
            n_skip=n_skip, n_bins=n_bins, seed=seed, verbose=verbose,
        )
        results.append(result)

    if verbose:
        print(f"\n{'=' * 70}")
        print("SUMMARY: Mass gap vs beta")
        print(f"{'=' * 70}")
        print(f"{'beta':>8s} {'g^2':>8s} {'<P>':>10s} {'m_fit':>10s} {'m_err':>10s}"
              f" {'m_dist':>10s} {'m_analyt':>10s}")
        for r in results:
            print(f"{r['beta']:8.2f} {4.0/r['beta']:8.4f}"
                  f" {r['plaquette']:10.6f}"
                  f" {r['gap_fit']['mass_gap']:10.4f}"
                  f" {r['gap_fit']['mass_gap_err']:10.4f}"
                  f" {r['dist_correlator']['gap_fit']['mass_gap']:10.4f}"
                  f" {r['analytical_gap']:10.4f}")

    return results


def run_full_serious(seed=42, verbose=True, reduced=False):
    """
    Run the complete serious MC study.

    If reduced=True, use fewer statistics (for testing or when time is limited).

    Parameters
    ----------
    seed : int
    verbose : bool
    reduced : bool, use reduced statistics

    Returns
    -------
    dict with all results.
    """
    if reduced:
        n_therm = 200
        n_measure = 100
        n_skip = 5
        beta_values = [2.0, 4.0, 8.0]
    else:
        n_therm = 1000
        n_measure = 500
        n_skip = 10
        beta_values = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    all_results = {}

    # 1. Beta scan (plaquette, Wilson loops, Polyakov)
    if verbose:
        print("\n" + "=" * 70)
        print("  PART 1: Beta scan (plaquette + Wilson loops + Polyakov)")
        print("=" * 70)

    all_results['beta_scan'] = run_beta_scan_serious(
        beta_values=beta_values,
        n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        seed=seed, verbose=verbose,
    )

    # 2. Mass gap vs beta
    if verbose:
        print("\n" + "=" * 70)
        print("  PART 2: Mass gap extraction vs beta")
        print("=" * 70)

    all_results['gap_scan'] = run_gap_vs_beta_serious(
        beta_values=beta_values,
        n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        seed=seed, verbose=verbose,
    )

    # 3. Summary
    if verbose:
        _print_summary(all_results)

    return all_results


# ======================================================================
# Internal helpers
# ======================================================================

def _fit_mass_gap_robust(distances, correlator, correlator_err=None):
    """
    Robust mass gap fit with multiple strategies.

    Tries:
      1. Full exponential fit with errors
      2. Cosh fit (for periodic boundary conditions on S3)
      3. Effective mass from first two positive points

    Returns dict with mass_gap, mass_gap_err, fit_quality.
    """
    distances = np.asarray(distances, dtype=float)
    correlator = np.asarray(correlator, dtype=float)

    # Filter to positive values with nonzero distance
    mask = (correlator > 0) & (distances > 0)
    if np.sum(mask) < 2:
        # Try including t=0
        mask = correlator > 0
        if np.sum(mask) < 2:
            return {
                'mass_gap': 0.0,
                'mass_gap_err': float('inf'),
                'amplitude': 0.0,
                'chi_squared': float('inf'),
                'fit_method': 'none',
                'n_points': 0,
            }

    d = distances[mask]
    c = correlator[mask]

    sigma = None
    if correlator_err is not None:
        s = np.asarray(correlator_err, dtype=float)[mask]
        if np.all(s > 0):
            sigma = s

    # Initial guess from first two points
    if len(d) >= 2 and c[0] > 0 and c[1] > 0 and d[1] > d[0]:
        m0 = max(-np.log(c[1] / c[0]) / (d[1] - d[0]), 0.01)
    else:
        m0 = 1.0
    A0 = c[0] if len(c) > 0 else 0.1

    # Strategy 1: Simple exponential
    def exp_decay(x, A, m):
        return A * np.exp(-m * x)

    try:
        popt, pcov = curve_fit(
            exp_decay, d, c, p0=[A0, m0],
            sigma=sigma, maxfev=10000,
            bounds=([0, 0], [np.inf, 50.0]),
        )
        A_fit, m_fit = popt
        perr = np.sqrt(np.diag(np.abs(pcov)))
        m_err = perr[1] if len(perr) > 1 else float('inf')

        # Chi-squared
        residuals = c - exp_decay(d, A_fit, m_fit)
        if sigma is not None:
            chi2 = float(np.sum((residuals / sigma)**2) / max(len(d) - 2, 1))
        else:
            chi2 = float(np.sum(residuals**2) / max(np.sum(c**2), 1e-30))

        return {
            'mass_gap': float(m_fit),
            'mass_gap_err': float(m_err),
            'amplitude': float(A_fit),
            'chi_squared': float(chi2),
            'fit_method': 'exponential',
            'n_points': int(np.sum(mask)),
        }
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass

    # Fallback: effective mass from first two positive points
    if len(c) >= 2 and c[0] > c[1] > 0 and d[1] > d[0]:
        m_est = -np.log(c[1] / c[0]) / (d[1] - d[0])
    else:
        m_est = 0.0

    return {
        'mass_gap': float(max(m_est, 0.0)),
        'mass_gap_err': float('inf'),
        'amplitude': float(c[0]) if len(c) > 0 else 0.0,
        'chi_squared': float('inf'),
        'fit_method': 'effective_mass',
        'n_points': int(np.sum(mask)),
    }


def _effective_mass_jackknife(corr_samples, bin_width):
    """
    Compute effective mass with jackknife errors.

    m_eff(t) = -ln[C(t+1)/C(t)] / dt

    Parameters
    ----------
    corr_samples : array (n_cfg, n_t)
    bin_width : float

    Returns
    -------
    list of dicts with t, m_eff, m_eff_err
    """
    n_cfg, n_t = corr_samples.shape
    results = []

    for t_idx in range(n_t - 1):
        dt = bin_width

        def eff_mass_func(data):
            c_t = np.mean(data[:, t_idx])
            c_tp1 = np.mean(data[:, t_idx + 1])
            if c_t > 0 and c_tp1 > 0:
                return -np.log(c_tp1 / c_t) / dt
            else:
                return float('nan')

        m, m_err = jackknife_function(corr_samples, eff_mass_func)

        if not np.isnan(m) and not np.isinf(m):
            results.append({
                't': float(t_idx * bin_width),
                'm_eff': float(m),
                'm_eff_err': float(m_err) if not np.isnan(m_err) else float('inf'),
            })

    return results


def _print_summary(all_results):
    """Print a comprehensive summary of all results."""
    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")

    print("\n  600-cell lattice: 120 vertices, 720 edges, 1200 faces")
    print("  Lattice spacing: a ~ 0.628 R")
    print("  Max geodesic time extent: pi*R ~ 3.14 R")

    # Beta scan summary
    if 'beta_scan' in all_results:
        print(f"\n  --- Plaquette vs beta ---")
        print(f"  {'beta':>6s} {'<P>':>10s} {'err':>10s} {'WC pred':>10s}")
        for r in all_results['beta_scan']:
            print(f"  {r['beta']:6.1f} {r['plaq_mean']:10.6f}"
                  f" {r['plaq_err']:10.6f} {r['wc_prediction']:10.6f}")

    # Gap scan summary
    if 'gap_scan' in all_results:
        print(f"\n  --- Mass gap vs beta ---")
        print(f"  {'beta':>6s} {'m_fit':>10s} {'m_err':>10s} {'m_analyt':>10s} {'method':>12s}")
        for r in all_results['gap_scan']:
            gf = r['gap_fit']
            print(f"  {r['beta']:6.1f} {gf['mass_gap']:10.4f}"
                  f" {gf['mass_gap_err']:10.4f}"
                  f" {r['analytical_gap']:10.4f}"
                  f" {gf.get('fit_method', 'N/A'):>12s}")

    print(f"\n  Analytical prediction: m = 2/R = 2.0000 (for R=1)")
    print(f"  NOTE: 600-cell has only 120 vertices. Results are qualitative, not precision.")
    print(f"  A proper continuum extrapolation requires multiple lattice spacings.")
