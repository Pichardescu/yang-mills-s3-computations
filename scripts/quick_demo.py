#!/usr/bin/env python3
"""60-second tour of the Yang-Mills mass gap on S3."""

import argparse
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Physical constants
HBAR_C = 197.3269804   # MeV*fm
LAMBDA_QCD = 200.0     # MeV (approximate)
R_PHYS = 2.2           # fm (default physical radius)


def section(title):
    """Print a section header."""
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="60-second tour of the Yang-Mills mass gap on S^3.",
        epilog=(
            "This script demonstrates the key geometric and spectral results "
            "underlying the mass gap proof for Yang-Mills theory on S^3 x R."
        ),
    )
    parser.add_argument(
        "--R", type=float, default=R_PHYS,
        help=f"Radius of S^3 in femtometers (default: {R_PHYS})"
    )
    args = parser.parse_args()
    R = args.R

    # ---- Imports from package ----
    from yang_mills_s3.geometry.s3_coordinates import S3Coordinates
    from yang_mills_s3.geometry.hodge_spectrum import HodgeSpectrum
    from yang_mills_s3.geometry.ricci import RicciTensor
    from yang_mills_s3.geometry.weitzenboeck import Weitzenboeck
    from yang_mills_s3.spectral.yang_mills_operator import YangMillsOperator

    # ==================================================================
    # 1. S^3 Geometry Basics
    # ==================================================================
    section("1. S^3 GEOMETRY BASICS")

    vol = S3Coordinates.volume(R)
    ricci = RicciTensor.on_sphere(3, R)
    betti = HodgeSpectrum.betti_numbers(3)

    print(f"  Radius R           = {R} fm")
    print(f"  Volume Vol(S^3)    = 2 pi^2 R^3 = {vol:.4f} fm^3")
    print(f"  Ricci curvature    = {ricci['einstein_constant']:.6f} / fm^2")
    print(f"                     = (n-1)/R^2 = 2/R^2  [Einstein manifold]")
    print(f"  Scalar curvature   = {ricci['ricci_scalar']:.6f} / fm^2")
    print(f"  Betti numbers      = {betti}")
    print(f"  b_1(S^3) = {betti[1]}  -->  NO harmonic 1-forms  -->  SPECTRAL GAP")

    # ==================================================================
    # 2. Weitzenboeck Decomposition
    # ==================================================================
    section("2. WEITZENBOECK DECOMPOSITION")

    wb = Weitzenboeck.decomposition('S3', R)
    print(f"  {wb['formula']}")
    print(f"  Delta_1 = nabla*nabla + Ric")
    print(f"  On left-invariant 1-forms:")
    print(f"    nabla*nabla = 2/R^2 = {2/R**2:.6f}")
    print(f"    Ric         = 2/R^2 = {wb['ricci_term']:.6f}")
    print(f"    Total       = 4/R^2 = {4/R**2:.6f}  <-- coexact gap")

    # ==================================================================
    # 3. Hodge Spectrum on S^3
    # ==================================================================
    section("3. HODGE SPECTRUM ON S^3")

    print(f"\n  Scalar eigenvalues (Delta_0):")
    print(f"  {'l':>4s}  {'Eigenvalue':>14s}  {'Multiplicity':>12s}")
    print(f"  {'---':>4s}  {'-----------':>14s}  {'------------':>12s}")
    scalar_spec = HodgeSpectrum.scalar_eigenvalues(3, R, l_max=5)
    for l_val, (ev, mult) in enumerate(scalar_spec):
        print(f"  {l_val:4d}  {ev:14.6f}  {mult:12d}")

    print(f"\n  Coexact 1-form eigenvalues (Delta_1, physical modes):")
    print(f"  {'k':>4s}  {'Eigenvalue':>14s}  {'Multiplicity':>12s}  {'Formula':>20s}")
    print(f"  {'---':>4s}  {'-----------':>14s}  {'------------':>12s}  {'-------':>20s}")
    coexact_spec = HodgeSpectrum.one_form_eigenvalues(3, R, l_max=6, mode='coexact')
    for k_val, (ev, mult) in enumerate(coexact_spec, start=1):
        formula = f"(k+1)^2/R^2 = {(k_val+1)**2}/R^2"
        print(f"  {k_val:4d}  {ev:14.6f}  {mult:12d}  {formula:>20s}")

    print(f"\n  Exact 1-form eigenvalues (Delta_1, pure gauge -- unphysical):")
    print(f"  {'l':>4s}  {'Eigenvalue':>14s}  {'Multiplicity':>12s}")
    print(f"  {'---':>4s}  {'-----------':>14s}  {'------------':>12s}")
    exact_spec = HodgeSpectrum.one_form_eigenvalues(3, R, l_max=4, mode='exact')
    for l_val, (ev, mult) in enumerate(exact_spec, start=1):
        print(f"  {l_val:4d}  {ev:14.6f}  {mult:12d}")

    # ==================================================================
    # 4. The Mass Gap
    # ==================================================================
    section("4. THE YANG-MILLS MASS GAP")

    gap_ev = YangMillsOperator.mass_gap_eigenvalue('SU(2)', R)
    gap_nat = YangMillsOperator.mass_gap('SU(2)', R)
    gap_mev = YangMillsOperator.physical_mass_gap('SU(2)', R)
    mult = YangMillsOperator.multiplicity_lowest_mode('SU(2)')

    print(f"  Gauge group: SU(2), dim(adj) = 3")
    print(f"  Coexact gap eigenvalue: 4/R^2 = {gap_ev:.6f} fm^-2")
    print(f"  Mass gap (natural):     2/R   = {gap_nat:.6f} fm^-1")
    print(f"  Mass gap (physical):    hbar*c * 2/R = {gap_mev:.2f} MeV")
    print(f"  Multiplicity of gap mode: {mult}")
    print()

    # SU(3) for comparison
    gap_mev_su3 = YangMillsOperator.physical_mass_gap('SU(3)', R)
    mult_su3 = YangMillsOperator.multiplicity_lowest_mode('SU(3)')
    print(f"  Gauge group: SU(3), dim(adj) = 8")
    print(f"  Mass gap (physical): {gap_mev_su3:.2f} MeV")
    print(f"  Multiplicity: {mult_su3}")
    print(f"  NOTE: The gap eigenvalue 4/R^2 is the SAME for any gauge group.")

    # ==================================================================
    # 5. Comparison with Lambda_QCD
    # ==================================================================
    section("5. COMPARISON WITH QCD")

    ratio = gap_mev / LAMBDA_QCD
    R_from_gap = 2 * HBAR_C / LAMBDA_QCD

    print(f"  Mass gap at R = {R} fm:     {gap_mev:.2f} MeV")
    print(f"  Lambda_QCD (approximate):   {LAMBDA_QCD:.0f} MeV")
    print(f"  Ratio m_gap / Lambda_QCD:   {ratio:.2f}")
    print()
    print(f"  R that gives m_gap = Lambda_QCD:")
    print(f"    R = 2 * hbar*c / Lambda_QCD = {R_from_gap:.2f} fm")
    print()
    print(f"  At R = {R} fm:")
    if gap_mev > LAMBDA_QCD:
        print(f"    m_gap > Lambda_QCD  --> gap is {ratio:.1f}x the QCD scale")
    else:
        print(f"    m_gap < Lambda_QCD  --> gap is below the QCD scale")

    # ==================================================================
    # 6. Summary
    # ==================================================================
    section("SUMMARY")

    print(f"""
  The Yang-Mills mass gap on S^3 arises from three geometric facts:

  1. TOPOLOGY:  H^1(S^3) = 0
     --> No harmonic 1-forms --> spectral gap is forced

  2. CURVATURE: Ric(S^3) = 2/R^2 > 0
     --> Weitzenboeck identity lifts the spectrum:
         Delta_1 = nabla*nabla + 2/R^2 >= 2/R^2

  3. SPECTRUM:  Coexact 1-form gap = 4/R^2
     --> Physical (gauge-invariant) mass gap = 2 * hbar*c / R
     --> At R = {R} fm: mass gap = {gap_mev:.1f} MeV

  Key insight: On S^3, the mass gap is a TOPOLOGICAL CONSEQUENCE,
  not a dynamical mystery. The compact topology forces a discrete
  spectrum with no zero modes for 1-forms.
""")


if __name__ == "__main__":
    main()
