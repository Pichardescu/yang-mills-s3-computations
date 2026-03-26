#!/usr/bin/env python3
"""Run a lattice Monte Carlo simulation of SU(2) Yang-Mills on S^3 (600-cell)."""

import argparse
import sys
import time
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Physical constants
HBAR_C = 197.3269804   # MeV*fm
LAMBDA_QCD = 200.0     # MeV
R_PHYS = 2.2           # fm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a lattice Monte Carlo simulation of SU(2) Yang-Mills "
            "on S^3 discretized as the 600-cell regular polytope."
        ),
        epilog=(
            "The 600-cell has 120 vertices, 720 edges, and 1200 triangular "
            "faces. It provides a natural, highly symmetric discretization of S^3."
        ),
    )
    parser.add_argument(
        "--beta", type=float, default=6.0,
        help="Inverse coupling beta = 4/g^2 for SU(2) (default: 6.0)"
    )
    parser.add_argument(
        "--sweeps", type=int, default=1000,
        help="Number of MC sweeps (default: 1000)"
    )
    parser.add_argument(
        "--thermalization", type=int, default=200,
        help="Number of thermalization sweeps (default: 200)"
    )
    parser.add_argument(
        "--algorithm", type=str, default="heatbath",
        choices=["metropolis", "heatbath", "compound"],
        help="MC algorithm (default: heatbath)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--start", type=str, default="cold",
        choices=["cold", "hot"],
        help="Starting configuration: cold (identity) or hot (random)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show a matplotlib plot of the MC history"
    )
    args = parser.parse_args()

    from yang_mills_s3.lattice.s3_lattice import S3Lattice
    from yang_mills_s3.lattice.mc_engine import MCEngine

    beta = args.beta
    n_sweeps = args.sweeps
    n_therm = args.thermalization
    g_sq = 4.0 / beta
    R = R_PHYS

    # ==================================================================
    # Setup
    # ==================================================================
    print()
    print("=" * 64)
    print("  SU(2) YANG-MILLS MONTE CARLO ON S^3 (600-CELL)")
    print("=" * 64)
    print()
    print(f"  Lattice: 600-cell (120 vertices, 720 edges, 1200 faces)")
    print(f"  Beta (4/g^2):        {beta:.2f}")
    print(f"  g^2:                 {g_sq:.4f}")
    print(f"  Algorithm:           {args.algorithm}")
    print(f"  Start:               {args.start}")
    print(f"  Thermalization:      {n_therm} sweeps")
    print(f"  Measurement sweeps:  {n_sweeps}")
    print(f"  Random seed:         {args.seed}")
    print()

    # Build lattice
    print("  Building 600-cell lattice...", end=" ", flush=True)
    t0 = time.time()
    lattice = S3Lattice(R=R)
    rng = np.random.default_rng(args.seed)
    engine = MCEngine(lattice, beta=beta, rng=rng)
    print(f"done ({time.time()-t0:.2f}s)")

    # Set starting configuration
    if args.start == "cold":
        engine.set_cold_start()
        print("  Initial plaquette (cold start): 1.0000")
    else:
        engine.set_hot_start()
        plaq0 = engine.plaquette_average()
        print(f"  Initial plaquette (hot start):  {plaq0:.4f}")

    # Choose sweep function
    if args.algorithm == "metropolis":
        def sweep():
            return engine.metropolis_sweep(epsilon=0.3)
    elif args.algorithm == "heatbath":
        def sweep():
            return engine.heatbath_sweep()
    else:  # compound
        def sweep():
            result = engine.compound_sweep(n_heatbath=1, n_overrelax=4)
            return result['acceptance_rate']

    # ==================================================================
    # Thermalization
    # ==================================================================
    print()
    print(f"  Thermalizing ({n_therm} sweeps)...", end=" ", flush=True)
    t0 = time.time()
    therm_plaq = []
    for i in range(n_therm):
        sweep()
        if (i + 1) % max(1, n_therm // 10) == 0:
            therm_plaq.append(engine.plaquette_average())
    t_therm = time.time() - t0
    print(f"done ({t_therm:.1f}s)")

    if therm_plaq:
        print(f"  Plaquette after thermalization: {therm_plaq[-1]:.6f}")

    # ==================================================================
    # Measurement
    # ==================================================================
    print(f"\n  Measuring ({n_sweeps} sweeps)...", flush=True)
    t0 = time.time()

    plaquettes = []
    actions = []
    acceptance_rates = []
    measure_interval = max(1, n_sweeps // 20)

    for i in range(n_sweeps):
        acc = sweep()
        acceptance_rates.append(acc if acc is not None else 1.0)

        plaq = engine.plaquette_average()
        plaquettes.append(plaq)
        actions.append(engine.wilson_action())

        if (i + 1) % measure_interval == 0:
            pct = 100 * (i + 1) / n_sweeps
            print(f"    sweep {i+1:6d}/{n_sweeps}  ({pct:5.1f}%)  "
                  f"<plaq> = {plaq:.6f}", flush=True)

    t_meas = time.time() - t0
    print(f"\n  Measurement complete ({t_meas:.1f}s)")

    # ==================================================================
    # Analysis
    # ==================================================================
    plaq_arr = np.array(plaquettes)
    action_arr = np.array(actions)
    acc_arr = np.array(acceptance_rates)

    plaq_mean = np.mean(plaq_arr)
    plaq_std = np.std(plaq_arr) / np.sqrt(len(plaq_arr))  # error on mean
    action_mean = np.mean(action_arr)

    # String tension estimate (rough, from plaquette):
    # On triangular lattice: <plaq> ~ 1 - (3/(8*beta)) for weak coupling (SU(2))
    # sigma * a^2 ~ -ln(<plaq>) for strong coupling
    # More careful: sigma ~ (1 - <plaq>) * beta / a^2
    a_lattice = lattice.lattice_spacing()
    sigma_lattice = (1.0 - plaq_mean) * beta / a_lattice**2 if plaq_mean < 1.0 else 0.0
    # Convert to physical units: sigma_phys = sigma_lattice / R^2
    # and sigma in MeV^2: sigma_MeV2 = sigma_phys * (hbar*c)^2
    sigma_phys = sigma_lattice / R**2
    sigma_MeV2 = sigma_phys * HBAR_C**2
    sqrt_sigma_MeV = np.sqrt(sigma_MeV2) if sigma_MeV2 > 0 else 0.0

    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print()
    print(f"  Average plaquette:     {plaq_mean:.6f} +/- {plaq_std:.6f}")
    print(f"  Average action:        {action_mean:.2f}")
    print(f"  Acceptance rate:       {np.mean(acc_arr):.4f}")
    print(f"  Lattice spacing:       {a_lattice:.4f} (units of R)")
    print()
    print(f"  String tension estimate (rough):")
    print(f"    sigma (lattice):     {sigma_lattice:.4f}")
    print(f"    sqrt(sigma):         {sqrt_sigma_MeV:.1f} MeV")
    print(f"    Experiment (QCD):    ~440 MeV")
    print()

    # Weak coupling comparison
    plaq_wc = 1.0 - 3.0 / (8.0 * beta)
    print(f"  Weak-coupling plaquette (1 - 3/(8*beta)): {plaq_wc:.6f}")
    if plaq_mean > 0:
        plaq_diff_pct = abs(plaq_mean - plaq_wc) / plaq_mean * 100
        print(f"  Deviation from weak coupling: {plaq_diff_pct:.2f}%")
    print()

    # Caveats
    print("  CAVEATS:")
    print("    - 600-cell is VERY coarse (120 sites, a ~ 0.63)")
    print("    - String tension is a rough estimate, not competitive with")
    print("      large-scale lattice QCD simulations")
    print("    - This is a proof-of-concept for YM on compact S^3")
    print("    - For quantitative results, use refined lattices")
    print()

    # ==================================================================
    # Optional plot
    # ==================================================================
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Plaquette history
            ax = axes[0]
            sweeps_x = np.arange(1, n_sweeps + 1)
            ax.plot(sweeps_x, plaq_arr, color='steelblue', linewidth=0.5, alpha=0.6)
            # Running average
            window = min(50, n_sweeps // 5)
            if window > 1:
                running_avg = np.convolve(plaq_arr, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(window, n_sweeps + 1), running_avg,
                        color='darkblue', linewidth=2, label=f'Running avg (w={window})')
            ax.axhline(y=plaq_mean, color='red', linestyle='--', linewidth=1,
                       label=f'Mean = {plaq_mean:.4f}')
            ax.axhline(y=plaq_wc, color='green', linestyle=':', linewidth=1,
                       label=f'Weak coupling = {plaq_wc:.4f}')
            ax.set_ylabel('Average Plaquette')
            ax.set_title(f'SU(2) Yang-Mills MC on 600-cell ($\\beta$ = {beta})')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Action history
            ax = axes[1]
            ax.plot(sweeps_x, action_arr, color='coral', linewidth=0.5, alpha=0.6)
            if window > 1:
                running_avg_a = np.convolve(action_arr, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(window, n_sweeps + 1), running_avg_a,
                        color='darkred', linewidth=2, label=f'Running avg (w={window})')
            ax.set_xlabel('MC Sweep')
            ax.set_ylabel('Wilson Action')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('mc_history.png', dpi=150)
            print(f"  Plot saved to mc_history.png")
            plt.show()

        except ImportError:
            print("  matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
