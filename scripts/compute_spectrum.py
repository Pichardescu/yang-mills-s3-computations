#!/usr/bin/env python3
"""Compute and display the Hodge spectrum on S^3."""

import argparse
import sys
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
        description="Compute and display the Hodge spectrum on S^3.",
        epilog=(
            "Eigenvalues of the Hodge-de Rham Laplacian on 0-forms and 1-forms, "
            "including exact and coexact branches, with physical mass gap in MeV."
        ),
    )
    parser.add_argument(
        "--R", type=float, default=R_PHYS,
        help=f"Radius of S^3 in femtometers (default: {R_PHYS})"
    )
    parser.add_argument(
        "--l-max", type=int, default=10,
        help="Maximum angular momentum quantum number (default: 10)"
    )
    parser.add_argument(
        "--gauge-group", type=str, default="SU(2)",
        choices=["SU(2)", "SU(3)"],
        help="Gauge group (default: SU(2))"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show a matplotlib plot of the spectrum"
    )
    args = parser.parse_args()

    R = args.R
    l_max = args.l_max
    gauge_group = args.gauge_group

    from yang_mills_s3.geometry.hodge_spectrum import HodgeSpectrum
    from yang_mills_s3.spectral.yang_mills_operator import YangMillsOperator

    dim_adj = YangMillsOperator.adjoint_dimension(gauge_group)

    # ==================================================================
    # Scalar Spectrum
    # ==================================================================
    print()
    print("=" * 72)
    print(f"  HODGE SPECTRUM ON S^3  (R = {R} fm, gauge group = {gauge_group})")
    print("=" * 72)

    print(f"\n  Scalar Laplacian Delta_0 on S^3(R = {R} fm)")
    print(f"  Eigenvalue = l(l+2)/R^2,  Multiplicity = (l+1)^2")
    print()
    print(f"  {'l':>4s}  {'Eigenvalue (fm^-2)':>18s}  {'Multiplicity':>12s}  {'Mass (MeV)':>12s}")
    print(f"  {'---':>4s}  {'------------------':>18s}  {'------------':>12s}  {'----------':>12s}")

    scalar_spec = HodgeSpectrum.scalar_eigenvalues(3, R, l_max=l_max)
    for l_val, (ev, mult) in enumerate(scalar_spec):
        mass = HBAR_C * np.sqrt(ev) if ev > 0 else 0.0
        print(f"  {l_val:4d}  {ev:18.8f}  {mult:12d}  {mass:12.2f}")

    # ==================================================================
    # Coexact 1-Form Spectrum (physical modes)
    # ==================================================================
    print()
    print("-" * 72)
    print(f"  Coexact 1-Form Laplacian on S^3(R = {R} fm)  [PHYSICAL MODES]")
    print(f"  Eigenvalue = (k+1)^2/R^2,  Multiplicity = 2k(k+2)")
    print(f"  YM multiplicity = Hodge mult x dim(adj({gauge_group})) = Hodge x {dim_adj}")
    print()
    print(f"  {'k':>4s}  {'Eigenvalue (fm^-2)':>18s}  {'Hodge mult':>10s}"
          f"  {'YM mult':>8s}  {'Mass (MeV)':>12s}  {'m/m_1':>8s}")
    print(f"  {'---':>4s}  {'------------------':>18s}  {'----------':>10s}"
          f"  {'--------':>8s}  {'----------':>12s}  {'-----':>8s}")

    coexact_spec = HodgeSpectrum.one_form_eigenvalues(3, R, l_max=l_max, mode='coexact')
    m1 = HBAR_C * np.sqrt(coexact_spec[0][0])

    coexact_evs = []
    coexact_mults = []
    for k_val, (ev, mult) in enumerate(coexact_spec, start=1):
        ym_mult = mult * dim_adj
        mass = HBAR_C * np.sqrt(ev)
        ratio = mass / m1
        coexact_evs.append(ev)
        coexact_mults.append(ym_mult)
        print(f"  {k_val:4d}  {ev:18.8f}  {mult:10d}  {ym_mult:8d}  {mass:12.2f}  {ratio:8.4f}")

    # ==================================================================
    # Exact 1-Form Spectrum (pure gauge, unphysical)
    # ==================================================================
    print()
    print("-" * 72)
    print(f"  Exact 1-Form Laplacian on S^3(R = {R} fm)  [PURE GAUGE, UNPHYSICAL]")
    print(f"  Eigenvalue = l(l+2)/R^2,  Multiplicity = (l+1)^2")
    print()
    print(f"  {'l':>4s}  {'Eigenvalue (fm^-2)':>18s}  {'Multiplicity':>12s}  {'Mass (MeV)':>12s}")
    print(f"  {'---':>4s}  {'------------------':>18s}  {'------------':>12s}  {'----------':>12s}")

    exact_spec = HodgeSpectrum.one_form_eigenvalues(3, R, l_max=l_max, mode='exact')
    exact_evs = []
    exact_mults = []
    for l_val, (ev, mult) in enumerate(exact_spec, start=1):
        mass = HBAR_C * np.sqrt(ev)
        exact_evs.append(ev)
        exact_mults.append(mult)
        print(f"  {l_val:4d}  {ev:18.8f}  {mult:12d}  {mass:12.2f}")

    # ==================================================================
    # Summary
    # ==================================================================
    gap_mev = YangMillsOperator.physical_mass_gap(gauge_group, R)

    print()
    print("=" * 72)
    print(f"  SUMMARY")
    print("=" * 72)
    print(f"  Coexact (physical) gap eigenvalue: 4/R^2 = {4/R**2:.8f} fm^-2")
    print(f"  Physical mass gap: 2*hbar*c/R = {gap_mev:.2f} MeV")
    print(f"  Ratio to Lambda_QCD: {gap_mev / LAMBDA_QCD:.2f}")
    print(f"  Total coexact modes up to l_max={l_max}: {sum(coexact_mults)}")
    print(f"  b_1(S^3) = 0: no zero modes, gap is topologically forced")
    print()

    # ==================================================================
    # Optional plot
    # ==================================================================
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: eigenvalue spectrum
            ax = axes[0]
            ks = list(range(1, len(coexact_evs) + 1))
            ls = list(range(1, len(exact_evs) + 1))
            ax.stem(ks, coexact_evs, linefmt='b-', markerfmt='bo', basefmt='k-',
                    label='Coexact (physical)')
            ax.stem(ls, exact_evs, linefmt='r--', markerfmt='r^', basefmt='k-',
                    label='Exact (pure gauge)')
            ax.set_xlabel('Quantum number k (coexact) / l (exact)')
            ax.set_ylabel('Eigenvalue (fm$^{-2}$)')
            ax.set_title(f'Hodge Spectrum on $S^3$ (R = {R} fm)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Right: mass spectrum in MeV
            ax = axes[1]
            coexact_masses = [HBAR_C * np.sqrt(ev) for ev in coexact_evs]
            ax.bar(ks, coexact_masses, color='steelblue', alpha=0.8,
                   label=f'{gauge_group} coexact modes')
            ax.axhline(y=LAMBDA_QCD, color='red', linestyle='--', linewidth=1.5,
                       label=f'$\\Lambda_{{QCD}}$ = {LAMBDA_QCD:.0f} MeV')
            ax.set_xlabel('Quantum number k')
            ax.set_ylabel('Mass (MeV)')
            ax.set_title(f'Physical Mass Spectrum ({gauge_group} on $S^3$)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('spectrum_s3.png', dpi=150)
            print(f"  Plot saved to spectrum_s3.png")
            plt.show()

        except ImportError:
            print("  matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
