"""
Monte Carlo simulation runner for SU(2) Yang-Mills on the 600-cell.

Runs actual MC simulations and extracts physics:
  1. Plaquette averages vs beta (phase structure)
  2. Wilson loops and string tension
  3. Correlator decay and mass gap
  4. Comparison with analytical predictions

The 600-cell has 120 vertices, 720 edges, 1200 triangular faces.
At beta=4/g^2, the weak coupling regime is beta >> 1.

Analytical predictions (linearized, R=1):
  - Mass gap: m^2 = 4/R^2 = 4 (coexact spectrum)
  - Mass gap: m = 2/R = 2
  - Strong coupling (beta << 1): plaquette ~ beta/4
  - Weak coupling (beta >> 1): plaquette ~ 1 - 3/(4*beta)

STATUS: NUMERICAL
"""

import numpy as np
from scipy.optimize import curve_fit
from .s3_lattice import S3Lattice
from .mc_engine import MCEngine


def _do_sweep(engine, method='metropolis'):
    """Perform one MC update sweep using the specified method."""
    if method == 'heatbath':
        engine.compound_sweep(n_heatbath=1, n_overrelax=3)
    else:
        engine.metropolis_sweep(epsilon=0.3)


def run_plaquette_scan(beta_values, n_therm=200, n_measure=100, n_skip=5,
                        seed=42, verbose=True, method='metropolis'):
    """
    Scan plaquette average over a range of beta values.

    This is the basic diagnostic: plaquette <P> vs beta shows the transition
    from strong coupling (disordered) to weak coupling (ordered).

    For SU(2) on a finite lattice, there is no sharp phase transition
    (crossover only), but the plaquette should smoothly increase from
    ~0 to ~1 as beta increases.

    Analytical limits:
      beta -> 0: <P> -> 0 (random links, <Tr U> = 0 for Haar)
      beta -> inf: <P> -> 1 (ordered, all links = identity)

    Parameters
    ----------
    beta_values : list of float
    n_therm : int, thermalization sweeps
    n_measure : int, measurement sweeps
    n_skip : int, sweeps between measurements
    seed : int, random seed
    verbose : bool

    Returns
    -------
    list of dicts with beta, plaq_mean, plaq_err, action
    """
    lattice = S3Lattice(R=1.0)
    results = []

    for beta in beta_values:
        rng = np.random.default_rng(seed)
        engine = MCEngine(lattice, beta=beta, rng=rng)

        # Hot start for strong coupling, cold start for weak coupling
        if beta < 3.0:
            engine.set_hot_start()
        else:
            engine.set_cold_start()

        # Thermalize
        for _ in range(n_therm):
            _do_sweep(engine, method)

        # Measure
        plaq_samples = []
        for _ in range(n_measure):
            for _ in range(n_skip):
                _do_sweep(engine, method)
            plaq_samples.append(engine.plaquette_average())

        plaq_arr = np.array(plaq_samples)
        plaq_mean = float(np.mean(plaq_arr))
        plaq_err = float(np.std(plaq_arr) / np.sqrt(len(plaq_arr)))
        action = float(engine.wilson_action())

        result = {
            'beta': beta,
            'plaq_mean': plaq_mean,
            'plaq_err': plaq_err,
            'action': action,
            'n_measure': n_measure,
        }
        results.append(result)

        if verbose:
            # Analytical weak-coupling prediction for comparison
            wc_pred = 1.0 - 3.0 / (4.0 * beta) if beta > 0.5 else 0.0
            print(f"  beta={beta:6.2f}  <P>={plaq_mean:.6f} +/- {plaq_err:.6f}"
                  f"  (weak-coupling: {wc_pred:.6f})")

    return results


def run_wilson_loops(beta=4.0, n_therm=300, n_measure=200, n_skip=5,
                      max_loop_length=6, seed=42, verbose=True, method='metropolis'):
    """
    Measure Wilson loops of various sizes and extract string tension.

    On a confining theory: <W(C)> ~ exp(-sigma * Area(C))
    On a deconfined theory: <W(C)> ~ exp(-mu * Perimeter(C))

    The crossover between area and perimeter law gives the string tension.

    Parameters
    ----------
    beta : float
    n_therm : thermalization sweeps
    n_measure : measurement sweeps
    n_skip : sweeps between measurements
    max_loop_length : max loop length to search for
    seed : random seed
    verbose : bool

    Returns
    -------
    dict with Wilson loop averages and string tension estimate
    """
    lattice = S3Lattice(R=1.0)
    rng = np.random.default_rng(seed)
    engine = MCEngine(lattice, beta=beta, rng=rng)

    if beta < 3.0:
        engine.set_hot_start()
    else:
        engine.set_cold_start()

    if verbose:
        print(f"Finding loops up to length {max_loop_length}...")

    # Find loops (before thermalization -- topology doesn't change)
    loops = engine.find_loops_by_length(max_length=max_loop_length,
                                         max_per_length=100)

    if verbose:
        for L, paths in sorted(loops.items()):
            print(f"  Length {L}: {len(paths)} loops found")

    # Thermalize
    if verbose:
        print(f"Thermalizing ({n_therm} sweeps)...")

    for sweep in range(n_therm):
        _do_sweep(engine, method)

    # Measure
    if verbose:
        print(f"Measuring ({n_measure} configs, {n_skip} sweeps between)...")

    # Accumulate Wilson loop values
    wl_accum = {L: [] for L in loops}

    for _ in range(n_measure):
        for _ in range(n_skip):
            _do_sweep(engine, method)

        wl = engine.measure_wilson_loops(loops)
        for L, data in wl.items():
            wl_accum[L].append(data['mean'])

    # Average over configs
    wl_results = {}
    for L in sorted(wl_accum.keys()):
        if wl_accum[L]:
            arr = np.array(wl_accum[L])
            wl_results[L] = {
                'W_mean': float(np.mean(arr)),
                'W_err': float(np.std(arr) / np.sqrt(len(arr))),
                'n_configs': len(arr),
                'n_loops': len(loops[L]),
            }

    # Compute Creutz ratio / string tension
    sigma_result = engine.creutz_ratio_estimate(loops)

    if verbose:
        print("\nWilson loop results:")
        for L, data in sorted(wl_results.items()):
            ln_W = np.log(abs(data['W_mean'])) if abs(data['W_mean']) > 1e-10 else float('-inf')
            print(f"  W({L}) = {data['W_mean']:.6f} +/- {data['W_err']:.6f}"
                  f"  [ln W = {ln_W:.4f}]")

        if sigma_result['sigma_eff']:
            print(f"\nString tension estimates:")
            for s in sigma_result['sigma_eff']:
                print(f"  sigma({s['lengths']}) = {s['sigma']:.4f}")
            print(f"  Mean sigma = {sigma_result['sigma_mean']:.4f}")

    return {
        'beta': beta,
        'wilson_loops': wl_results,
        'string_tension': sigma_result,
        'loops_found': {L: len(p) for L, p in loops.items()},
    }


def run_mass_gap(beta=4.0, n_therm=300, n_measure=200, n_skip=5,
                  n_bins=10, seed=42, verbose=True, method='metropolis'):
    """
    Extract the mass gap from plaquette correlator decay.

    Method:
      1. Thermalize gauge configuration
      2. Measure plaquette field on each face
      3. Bin plaquettes by "time" coordinate (w-axis)
      4. Compute connected correlator C(t) = <O(t)O(0)> - <O>^2
      5. Fit C(t) ~ A * exp(-m * t) to extract mass gap m

    Analytical prediction (R=1): m = 2/R = 2 (linearized gap)

    Parameters
    ----------
    beta : float
    n_therm, n_measure, n_skip : MC parameters
    n_bins : time slices for correlator
    seed : random seed
    verbose : bool

    Returns
    -------
    dict with correlator data, mass gap, comparison with analytics
    """
    lattice = S3Lattice(R=1.0)
    rng = np.random.default_rng(seed)
    engine = MCEngine(lattice, beta=beta, rng=rng)

    if beta < 3.0:
        engine.set_hot_start()
    else:
        engine.set_cold_start()

    # Thermalize
    if verbose:
        print(f"Thermalizing ({n_therm} sweeps at beta={beta})...")

    for sweep in range(n_therm):
        _do_sweep(engine, method)

    if verbose:
        print(f"Thermalized. Plaquette = {engine.plaquette_average():.6f}")

    # Accumulate correlators
    if verbose:
        print(f"Measuring correlators ({n_measure} configs)...")

    max_dt = n_bins // 2
    corr_accum = np.zeros((n_measure, max_dt + 1))

    for cfg in range(n_measure):
        for _ in range(n_skip):
            _do_sweep(engine, method)

        tc = engine.time_slice_correlator(coord=0, n_bins=n_bins)
        corr_accum[cfg, :] = tc['correlator']

    # Average over configs
    corr_mean = np.mean(corr_accum, axis=0)
    corr_err = np.std(corr_accum, axis=0) / np.sqrt(n_measure)
    separations = np.arange(max_dt + 1, dtype=float) * (np.pi / n_bins)

    # Extract mass gap by fitting C(t) = A * exp(-m * t)
    # Use points where correlator is positive and significantly nonzero
    mask = corr_mean > 0
    if np.sum(mask) < 2:
        # Try using absolute values
        mask = np.abs(corr_mean) > 1e-15

    gap_result = _fit_mass_gap(separations, corr_mean, corr_err)

    # Also try effective mass: m_eff(t) = -ln[C(t+1)/C(t)] / dt
    eff_mass = []
    for i in range(len(corr_mean) - 1):
        dt = separations[i + 1] - separations[i]
        if corr_mean[i] > 0 and corr_mean[i + 1] > 0 and dt > 0:
            m = -np.log(corr_mean[i + 1] / corr_mean[i]) / dt
            eff_mass.append((float(separations[i]), float(m)))

    # Distance-based correlator (better for S3 geometry)
    if verbose:
        print("Measuring distance-based correlator...")

    dist_corr_accum = []
    # Reset for distance correlator
    for cfg in range(min(n_measure, 50)):  # fewer configs for this (more expensive)
        for _ in range(n_skip):
            _do_sweep(engine, method)
        dc = engine.plaquette_correlator_by_distance(n_distance_bins=12)
        dist_corr_accum.append(dc['correlator'])

    if dist_corr_accum:
        dist_corr_mean = np.mean(dist_corr_accum, axis=0)
        dist_corr_err = np.std(dist_corr_accum, axis=0) / np.sqrt(len(dist_corr_accum))
        # Use the distance bins from last measurement
        dist_bins = dc['distances']
        dist_gap = _fit_mass_gap(dist_bins, dist_corr_mean, dist_corr_err)
    else:
        dist_corr_mean = np.array([])
        dist_corr_err = np.array([])
        dist_bins = np.array([])
        dist_gap = {'mass_gap': 0.0, 'mass_gap_err': float('inf')}

    # Analytical prediction
    R = lattice.R
    analytical_gap = 2.0 / R  # m = 2/R from coexact spectrum
    analytical_gap_sq = 4.0 / R**2  # m^2 = 4/R^2

    result = {
        'beta': beta,
        'R': R,
        'separations': separations.tolist(),
        'correlator': corr_mean.tolist(),
        'correlator_err': corr_err.tolist(),
        'effective_mass': eff_mass,
        'gap_fit': gap_result,
        'dist_correlator': {
            'distances': dist_bins.tolist() if len(dist_bins) > 0 else [],
            'correlator': dist_corr_mean.tolist() if len(dist_corr_mean) > 0 else [],
            'correlator_err': dist_corr_err.tolist() if len(dist_corr_err) > 0 else [],
            'gap_fit': dist_gap,
        },
        'analytical_gap': analytical_gap,
        'analytical_gap_sq': analytical_gap_sq,
        'plaquette': float(engine.plaquette_average()),
    }

    if verbose:
        print(f"\n--- Mass Gap Results (beta={beta}) ---")
        print(f"  Plaquette average: {result['plaquette']:.6f}")
        print(f"  Time-slice correlator:")
        for i, (sep, c) in enumerate(zip(separations, corr_mean)):
            print(f"    t={sep:.4f}  C(t)={c:.2e} +/- {corr_err[i]:.2e}")
        if eff_mass:
            print(f"  Effective mass:")
            for (t, m) in eff_mass:
                print(f"    t={t:.4f}  m_eff={m:.4f}")
        print(f"  Fitted gap (time-slice): {gap_result['mass_gap']:.4f}"
              f" +/- {gap_result['mass_gap_err']:.4f}")
        print(f"  Fitted gap (distance): {dist_gap['mass_gap']:.4f}"
              f" +/- {dist_gap['mass_gap_err']:.4f}")
        print(f"  Analytical prediction: m = {analytical_gap:.4f} (m^2 = {analytical_gap_sq:.4f})")

    return result


def run_beta_scan_gap(beta_values=None, n_therm=200, n_measure=100,
                       n_skip=5, n_bins=10, seed=42, verbose=True):
    """
    Scan mass gap over a range of beta values.

    This tests the key physics question: does the gap persist as we
    vary the coupling?

    At weak coupling (large beta): gap should approach 2/R (free theory).
    At strong coupling (small beta): gap should be larger (confinement scale).
    For all beta > 0: gap should be positive (mass gap exists).

    Parameters
    ----------
    beta_values : list of float, or None for default scan
    n_therm, n_measure, n_skip, n_bins : MC parameters
    seed : random seed
    verbose : bool

    Returns
    -------
    list of dicts with beta and gap measurements
    """
    if beta_values is None:
        beta_values = [1.0, 2.0, 4.0, 8.0, 16.0]

    results = []

    for beta in beta_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Beta = {beta}")
            print(f"{'='*60}")

        result = run_mass_gap(
            beta=beta, n_therm=n_therm, n_measure=n_measure,
            n_skip=n_skip, n_bins=n_bins, seed=seed, verbose=verbose
        )
        results.append(result)

    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY: Mass gap vs beta")
        print(f"{'='*60}")
        print(f"{'beta':>8s} {'<P>':>10s} {'m_gap':>10s} {'m_err':>10s} {'m_analyt':>10s}")
        for r in results:
            print(f"{r['beta']:8.2f} {r['plaquette']:10.6f}"
                  f" {r['gap_fit']['mass_gap']:10.4f}"
                  f" {r['gap_fit']['mass_gap_err']:10.4f}"
                  f" {r['analytical_gap']:10.4f}")

    return results


def run_full_simulation(seed=42, verbose=True, fast=False):
    """
    Run the complete Monte Carlo study.

    Performs:
    1. Plaquette scan over beta
    2. Wilson loop measurement
    3. Mass gap extraction
    4. Beta scan for gap

    Parameters
    ----------
    seed : random seed
    verbose : print results
    fast : if True, use reduced statistics (for testing)

    Returns
    -------
    dict with all results
    """
    if fast:
        n_therm = 50
        n_measure = 30
        n_skip = 2
        beta_scan = [1.0, 2.0, 4.0, 8.0]
    else:
        n_therm = 200
        n_measure = 100
        n_skip = 5
        beta_scan = [1.0, 2.0, 4.0, 8.0, 16.0]

    all_results = {}

    # 1. Plaquette scan
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Plaquette average vs beta")
        print("=" * 60)

    all_results['plaquette_scan'] = run_plaquette_scan(
        beta_values=beta_scan,
        n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        seed=seed, verbose=verbose
    )

    # 2. Wilson loops at representative beta
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Wilson loops at beta=4.0")
        print("=" * 60)

    all_results['wilson_loops'] = run_wilson_loops(
        beta=4.0, n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        max_loop_length=5, seed=seed, verbose=verbose
    )

    # 3. Mass gap at representative beta
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 3: Mass gap at beta=4.0")
        print("=" * 60)

    all_results['mass_gap'] = run_mass_gap(
        beta=4.0, n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        seed=seed, verbose=verbose
    )

    # 4. Gap vs beta scan
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 4: Mass gap vs beta scan")
        print("=" * 60)

    all_results['gap_scan'] = run_beta_scan_gap(
        beta_values=beta_scan,
        n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        seed=seed, verbose=verbose
    )

    return all_results


# ==================================================================
# Internal helpers
# ==================================================================

def _fit_mass_gap(distances, correlator, correlator_err=None):
    """
    Fit C(d) = A * exp(-m * d) to extract mass gap m.

    Returns dict with mass_gap, mass_gap_err, amplitude, chi_squared.
    """
    # Filter to positive values
    mask = correlator > 0
    if np.sum(mask) < 2:
        return {
            'mass_gap': 0.0,
            'mass_gap_err': float('inf'),
            'amplitude': 0.0,
            'chi_squared': float('inf'),
        }

    d = np.asarray(distances)[mask]
    c = np.asarray(correlator)[mask]

    sigma = None
    if correlator_err is not None:
        s = np.asarray(correlator_err)[mask]
        if np.all(s > 0):
            sigma = s

    # Initial guess from first two points
    if len(d) >= 2 and c[0] > 0 and c[1] > 0 and d[1] > d[0]:
        m0 = max(-np.log(c[1] / c[0]) / (d[1] - d[0]), 0.1)
    else:
        m0 = 1.0
    A0 = c[0]

    def exp_decay(x, A, m):
        return A * np.exp(-m * x)

    try:
        popt, pcov = curve_fit(
            exp_decay, d, c, p0=[A0, m0],
            sigma=sigma, maxfev=10000,
            bounds=([0, 0], [np.inf, 100.0])
        )
        A_fit, m_fit = popt
        perr = np.sqrt(np.diag(pcov))
        m_err = perr[1] if len(perr) > 1 else float('inf')

        residuals = c - exp_decay(d, A_fit, m_fit)
        chi2 = float(np.sum(residuals**2) / max(np.sum(c**2), 1e-30))

        return {
            'mass_gap': float(m_fit),
            'mass_gap_err': float(m_err),
            'amplitude': float(A_fit),
            'chi_squared': chi2,
        }
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
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
        }
