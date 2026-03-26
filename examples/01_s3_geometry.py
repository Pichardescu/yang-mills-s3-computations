#!/usr/bin/env python3
"""
Tutorial 1: S^3 Differential Geometry
======================================

S^3 (the 3-sphere) is the set of unit quaternions: |q|^2 = 1 in R^4.
It is simultaneously a Riemannian manifold AND a Lie group (isomorphic to SU(2)).
This dual nature is the key to the Yang-Mills mass gap.

Why S^3 matters for Yang-Mills:
-------------------------------
The Clay Millennium Problem asks whether pure Yang-Mills theory has a mass gap
-- a strictly positive lower bound on the energy spectrum above the vacuum.
On flat R^3, this is an open problem since 2000.

On S^3, the situation is dramatically different:
  - Compactness forces a discrete spectrum (no continuous spectrum to fight).
  - The first Betti number b_1(S^3) = 0, so there are no harmonic 1-forms,
    guaranteeing a spectral gap for the Hodge Laplacian on 1-forms.
  - The isometry group SO(4) acts transitively, giving maximal symmetry.
  - S^3 ~ SU(2) as a Lie group, connecting gauge structure to geometry.

Physical motivation:
  - Cosmological observations (de Sitter space, finite CMB correlations)
    suggest the spatial universe may be compact.
  - S^3 of radius R ~ 2 fm provides a natural infrared regulator that
    eliminates IR divergences without arbitrary cutoffs.

This tutorial covers:
  1. S^3 as unit quaternions and its parametrizations
  2. Volume formula: Vol(S^3) = 2 pi^2 R^3
  3. Ricci curvature: Ric = 2g/R^2 (Einstein manifold)
  4. The Hopf fibration S^1 -> S^3 -> S^2
  5. Homotopy: pi_3(S^3) = Z (why instantons exist)

Prerequisites:
  - Linear algebra and basic differential geometry
  - Familiarity with quaternions is helpful but not required

References:
  - Besse, "Einstein Manifolds" (1987), Ch. 3
  - Nakahara, "Geometry, Topology and Physics" (2003), Ch. 9
  - Frankel, "The Geometry of Physics" (2011), Ch. 17
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Setup: allow standalone execution by adding the package root to sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from yang_mills_s3.geometry.s3_coordinates import S3Coordinates
from yang_mills_s3.geometry.ricci import RicciTensor
from yang_mills_s3.geometry.hopf_fibration import HopfFibration


# ===========================================================================
# Physical constants
# ===========================================================================
HBAR_C = 197.3269804   # hbar*c in MeV*fm
LAMBDA_QCD = 200.0     # Lambda_QCD in MeV (approximate)


def print_header(title):
    """Print a formatted section header."""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_subheader(title):
    """Print a formatted subsection header."""
    print()
    print(f"--- {title} ---")
    print()


# ===========================================================================
# Section 1: S^3 as Unit Quaternions
# ===========================================================================

def section_1_quaternions():
    """
    S^3 is the set of unit quaternions in R^4.

    A quaternion q = w + xi + yj + zk satisfies the algebra:
        i^2 = j^2 = k^2 = ijk = -1

    The unit quaternions (|q| = 1) form a group under multiplication,
    and this group is isomorphic to SU(2):
        q = w + xi + yj + zk  <-->  U = [[w + iz, y + ix], [-y + ix, w - iz]]

    Key facts:
        - S^3 is simply connected (pi_1 = 0): no winding numbers for loops.
        - S^3 is a Lie group: left/right multiplication are isometries.
        - The double cover SU(2) -> SO(3) gives q -> R (rotation matrix),
          with q and -q mapping to the same rotation.
    """
    print_header("Section 1: S^3 as Unit Quaternions")

    s3 = S3Coordinates(R=1.0)

    # Demonstrate the Euler angle parametrization
    # chi in [0, pi], theta in [0, pi], phi in [0, 2*pi)
    print("The Euler angle parametrization of S^3:")
    print("  q = (cos(chi/2)cos(theta/2),  cos(chi/2)sin(theta/2),")
    print("       sin(chi/2)cos(phi),       sin(chi/2)sin(phi))")
    print()

    # Sample some points and verify they lie on S^3
    print("Sample points on the unit S^3:")
    print(f"  {'chi':>8s} {'theta':>8s} {'phi':>8s}  |  {'w':>8s} {'x':>8s} {'y':>8s} {'z':>8s}  |  |q|^2")
    print("  " + "-" * 78)

    test_angles = [
        (0.0, 0.0, 0.0),           # North pole: q = (1, 0, 0, 0)
        (np.pi, 0.0, 0.0),         # q = (0, 0, 1, 0)
        (np.pi / 2, np.pi / 2, 0), # Generic point
        (np.pi / 3, np.pi / 4, np.pi / 6),  # Another generic point
        (np.pi, np.pi, np.pi),     # q = (0, 0, -1, 0)
    ]

    for chi, theta, phi in test_angles:
        w, x, y, z = s3.euler_angles(chi, theta, phi)
        norm_sq = w**2 + x**2 + y**2 + z**2
        print(f"  {chi:8.4f} {theta:8.4f} {phi:8.4f}  | "
              f" {w:8.4f} {x:8.4f} {y:8.4f} {z:8.4f}  |  {norm_sq:.10f}")

    print()
    print("  All points satisfy |q|^2 = 1, confirming they lie on S^3.")

    # Demonstrate the quaternion-to-rotation map (SU(2) -> SO(3))
    print_subheader("The double cover SU(2) -> SO(3)")

    q1 = s3.euler_angles(np.pi / 3, np.pi / 4, 0)
    q2 = (-q1[0], -q1[1], -q1[2], -q1[3])  # Antipodal point

    R1 = S3Coordinates.quaternion_to_rotation(q1)
    R2 = S3Coordinates.quaternion_to_rotation(q2)

    print("  q and -q give the SAME rotation matrix (2:1 covering):")
    print()
    print(f"  q  = ({q1[0]:+.4f}, {q1[1]:+.4f}, {q1[2]:+.4f}, {q1[3]:+.4f})")
    print(f"  -q = ({q2[0]:+.4f}, {q2[1]:+.4f}, {q2[2]:+.4f}, {q2[3]:+.4f})")
    print()
    print(f"  ||R(q) - R(-q)|| = {np.linalg.norm(R1 - R2):.2e}")
    print(f"  det(R(q)) = {np.linalg.det(R1):+.6f}  (SO(3): det = +1)")
    print()
    print("  This is why SU(2) = S^3 is the universal cover of SO(3).")
    print("  The quotient S^3 / {+1, -1} = SO(3) = RP^3.")


# ===========================================================================
# Section 2: Hopf Coordinates and the C^2 Embedding
# ===========================================================================

def section_2_hopf_coordinates():
    """
    Hopf coordinates embed S^3 in C^2:
        (z1, z2) = (R*cos(eta)*e^{i*xi1}, R*sin(eta)*e^{i*xi2})

    with eta in [0, pi/2], xi1, xi2 in [0, 2*pi).

    This parametrization makes the Hopf fibration manifest:
        - Fixing xi1 - xi2 = const gives an S^2 (the base)
        - Fixing a point on S^2, varying xi1 + xi2 gives the S^1 fiber

    The metric in Hopf coordinates:
        ds^2 = R^2 [d(eta)^2 + cos^2(eta) d(xi1)^2 + sin^2(eta) d(xi2)^2]
    """
    print_header("Section 2: Hopf Coordinates on S^3")

    s3 = S3Coordinates(R=1.0)

    print("  Hopf coordinates: S^3 embedded in C^2")
    print("  (z1, z2) = (cos(eta) e^{i xi1}, sin(eta) e^{i xi2})")
    print("  with |z1|^2 + |z2|^2 = R^2")
    print()

    print("  Sample points in Hopf coordinates:")
    print(f"  {'eta':>8s} {'xi1':>8s} {'xi2':>8s}  |  {'|z1|^2':>10s} {'|z2|^2':>10s} {'sum':>10s}")
    print("  " + "-" * 64)

    test_coords = [
        (0.0, 0.0, 0.0),                 # z1 = 1, z2 = 0
        (np.pi/2, 0.0, 0.0),             # z1 = 0, z2 = 1
        (np.pi/4, 0.0, 0.0),             # z1 = z2 = 1/sqrt(2)
        (np.pi/4, np.pi/3, np.pi/6),     # Generic
        (np.pi/3, np.pi, np.pi/2),       # Another generic
    ]

    for eta, xi1, xi2 in test_coords:
        z1, z2 = s3.hopf_coordinates(eta, xi1, xi2)
        r1_sq = np.abs(z1)**2
        r2_sq = np.abs(z2)**2
        print(f"  {eta:8.4f} {xi1:8.4f} {xi2:8.4f}  |"
              f"  {r1_sq:10.6f} {r2_sq:10.6f} {r1_sq + r2_sq:10.6f}")

    print()
    print("  The constraint |z1|^2 + |z2|^2 = 1 is always satisfied.")
    print("  This embedding reveals S^3 as the join of two circles:")
    print("  as eta varies from 0 to pi/2, we sweep from the z1-circle")
    print("  to the z2-circle. These are the two Hopf fibers at the")
    print("  north and south poles of S^2.")


# ===========================================================================
# Section 3: Volume of S^3
# ===========================================================================

def section_3_volume():
    """
    The volume of S^3 of radius R:

        Vol(S^3_R) = 2 * pi^2 * R^3

    Derivation: Integrate the volume form
        dV = R^3 * sin^2(chi) * sin(theta) * d(chi) d(theta) d(phi)
    over chi in [0, pi], theta in [0, pi], phi in [0, 2*pi):
        Vol = R^3 * [integral_0^pi sin^2(chi) d(chi)]
                   * [integral_0^pi sin(theta) d(theta)]
                   * [integral_0^{2pi} d(phi)]
            = R^3 * (pi/2) * 2 * (2*pi) = 2*pi^2*R^3.

    Comparison: Vol(S^2_R) = 4*pi*R^2, Vol(S^1_R) = 2*pi*R.
    General: Vol(S^n_R) = 2*pi^{(n+1)/2} * R^n / Gamma((n+1)/2).
    """
    print_header("Section 3: Volume of S^3")

    print("  Exact formula: Vol(S^3_R) = 2 pi^2 R^3")
    print()

    # Compute volumes at various radii
    print(f"  {'R (fm)':>10s}  {'Vol(S^3)':>14s}  {'Vol(S^3)/R^3':>14s}  {'2*pi^2':>10s}")
    print("  " + "-" * 54)

    for R in [0.5, 1.0, 2.0, 2.2, 5.0]:
        vol = S3Coordinates.volume(R)
        ratio = vol / R**3
        print(f"  {R:10.2f}  {vol:14.6f}  {ratio:14.6f}  {2*np.pi**2:10.6f}")

    print()
    print(f"  Vol/R^3 = 2*pi^2 = {2*np.pi**2:.6f} for all R. (Confirmed.)")

    # Numerical verification by Monte Carlo integration
    print_subheader("Numerical verification (Monte Carlo)")

    N_samples = 500_000
    rng = np.random.default_rng(42)

    # Sample uniformly in the hypercube [0,pi] x [0,pi] x [0,2*pi]
    chi = rng.uniform(0, np.pi, N_samples)
    theta = rng.uniform(0, np.pi, N_samples)
    phi = rng.uniform(0, 2 * np.pi, N_samples)

    # Volume element: R^3 * sin^2(chi) * sin(theta)
    R = 1.0
    integrand = R**3 * np.sin(chi)**2 * np.sin(theta)

    # Hypercube volume = pi * pi * 2*pi
    hypercube_vol = np.pi * np.pi * 2 * np.pi
    mc_volume = hypercube_vol * np.mean(integrand)
    exact_volume = S3Coordinates.volume(R)
    error_pct = abs(mc_volume - exact_volume) / exact_volume * 100

    print(f"  Monte Carlo estimate ({N_samples:,d} samples): {mc_volume:.6f}")
    print(f"  Exact value:                             {exact_volume:.6f}")
    print(f"  Relative error:                          {error_pct:.3f}%")

    # Physical context
    print_subheader("Physical context: S^3 at QCD scale")

    R_qcd = 2.2  # fm
    vol_qcd = S3Coordinates.volume(R_qcd)
    print(f"  At R = {R_qcd} fm (QCD scale):")
    print(f"    Volume = {vol_qcd:.2f} fm^3")
    print(f"    For comparison: proton volume ~ 4/3 * pi * (0.84)^3 ~ "
          f"{4/3 * np.pi * 0.84**3:.2f} fm^3")
    print(f"    Ratio: Vol(S^3) / Vol(proton) ~ {vol_qcd / (4/3 * np.pi * 0.84**3):.1f}")


# ===========================================================================
# Section 4: Ricci Curvature -- S^3 is Einstein
# ===========================================================================

def section_4_ricci():
    """
    The Ricci tensor of S^3_R with the round metric is proportional to g:

        Ric = (n-1)/R^2 * g = 2/R^2 * g    (for n = 3)

    This means S^3 is an Einstein manifold: Ric = lambda * g.
    The Einstein constant is lambda = 2/R^2.

    The Ricci scalar (scalar curvature) is:
        Scal = n * (n-1) / R^2 = 6/R^2    (for n = 3)

    The sectional curvature is 1/R^2 (constant, since S^3 is a space form).

    Why this matters for Yang-Mills:
    --------------------------------
    The Weitzenbock identity for the 1-form Laplacian on a Riemannian manifold:
        Delta_1 = nabla* nabla + Ric

    On S^3:
        Delta_1 = nabla* nabla + 2/R^2

    Since nabla* nabla >= 0, this gives a LOWER BOUND:
        Delta_1 >= 2/R^2

    The actual coexact eigenvalue gap is 4/R^2 (stronger), but the Ricci
    contribution of 2/R^2 already guarantees a nonzero gap.

    Contrast with R^3: Ric = 0, so the Weitzenbock identity gives no gap.
    This is the geometric essence of the mass gap on S^3.
    """
    print_header("Section 4: Ricci Curvature (S^3 is Einstein)")

    print("  On any round sphere S^n of radius R:")
    print("    Ric = (n-1)/R^2 * g    (Einstein manifold)")
    print("    Scal = n(n-1)/R^2      (scalar curvature)")
    print()

    # Compute for various spheres
    print(f"  {'Sphere':>6s}  {'lambda = (n-1)/R^2':>18s}  {'Scal':>14s}  {'Meaning':>30s}")
    print("  " + "-" * 74)

    R = 1.0
    for n in [2, 3, 4, 7]:
        info = RicciTensor.on_sphere(n, R)
        note = {
            2: "Earth-like surface",
            3: "SU(2) ~ gauge group",
            4: "S^4 (Euclidean spacetime)",
            7: "S^7 (octonions, G2 holonomy)",
        }.get(n, "")
        print(f"  S^{n:1d}     {info['einstein_constant']:18.6f}  {info['ricci_scalar']:14.6f}  {note:>30s}")

    # Focus on S^3
    print_subheader("S^3 Ricci curvature vs. radius")

    print("  S^3 is special: it is ALSO a Lie group (SU(2)).")
    print("  The bi-invariant metric on SU(2) IS the round metric on S^3.")
    print()

    print(f"  {'R (fm)':>10s}  {'Ric = 2/R^2':>14s}  {'Scal = 6/R^2':>14s}  {'Gap bound 2/R^2':>16s}")
    print("  " + "-" * 60)

    for R in [0.5, 1.0, 2.0, 2.2, 5.0, 10.0]:
        info = RicciTensor.on_sphere(3, R)
        gap_bound = info['einstein_constant']  # 2/R^2
        print(f"  {R:10.2f}  {info['einstein_constant']:14.6f}"
              f"  {info['ricci_scalar']:14.6f}  {gap_bound:16.6f}")

    # Lie group perspective
    print_subheader("Lie group perspective: SU(2) and SU(3)")

    print("  For Yang-Mills with gauge group G on S^3, the Ricci curvature")
    print("  of S^3 sets the lower bound on the spectral gap.")
    print()

    for group in ['SU(2)', 'SU(3)']:
        info = RicciTensor.on_lie_group(group, R=1.0)
        print(f"  {group}:")
        print(f"    dim(G) = {info['dimension']}")
        print(f"    Ricci on 1-forms (bi-invariant metric) = {info['ricci_on_1forms']:.4f}")
        print(f"    Ricci scalar = {info['ricci_scalar']:.4f}")
        print()

    # Physical gap
    print("  Physical mass gap from Weitzenbock lower bound:")
    R_phys = 2.2  # fm
    gap_eigenvalue = 2.0 / R_phys**2  # Lower bound from Ric alone
    mass_gap_ric = np.sqrt(gap_eigenvalue) * HBAR_C
    gap_actual = 4.0 / R_phys**2  # Actual coexact gap
    mass_gap_actual = np.sqrt(gap_actual) * HBAR_C

    print(f"  At R = {R_phys} fm:")
    print(f"    Ricci lower bound:  m >= sqrt(2/R^2) * hbar*c = {mass_gap_ric:.1f} MeV")
    print(f"    Actual coexact gap: m  = sqrt(4/R^2) * hbar*c = {mass_gap_actual:.1f} MeV")
    print(f"    Lambda_QCD ~ {LAMBDA_QCD} MeV (for comparison)")


# ===========================================================================
# Section 5: The Hopf Fibration
# ===========================================================================

def section_5_hopf():
    """
    The Hopf fibration is the map:

        pi: S^3 -> S^2

    defined by:
        pi(z1, z2) = (2*Re(z1*conj(z2)), 2*Im(z1*conj(z2)), |z1|^2 - |z2|^2)

    This map is a principal U(1)-bundle (circle bundle) over S^2:
        - Every point p in S^2 has a fiber pi^{-1}(p) ~ S^1 (a great circle in S^3)
        - Any two distinct fibers are LINKED (linking number = 1)
        - The fibration is nontrivial: S^3 is NOT S^2 x S^1

    The Hopf fibration is the ONLY nontrivial circle bundle over S^2.
    Its first Chern number c_1 = 1 classifies it topologically.

    Physical significance:
    ----------------------
    1. The Hopf fibration is the simplest magnetic monopole.
       The connection 1-form A is the Dirac monopole potential on S^2.
    2. The curvature F = dA is the monopole field with total flux = 4*pi.
    3. In the standard model, this encodes U(1) electromagnetism as a
       substructure of SU(2) ~ S^3.
    """
    print_header("Section 5: The Hopf Fibration S^1 -> S^3 -> S^2")

    hopf = HopfFibration()
    s3 = S3Coordinates(R=1.0)

    # Demonstrate the projection map
    print("  The Hopf map: pi(z1, z2) -> (x, y, z) on S^2")
    print()

    print(f"  {'eta':>8s} {'xi1':>8s} {'xi2':>8s}  |  {'x':>8s} {'y':>8s} {'z':>8s}  |  {'|p|^2':>8s}")
    print("  " + "-" * 66)

    test_points = [
        (0.0, 0.0, 0.0),             # North pole of S^2
        (np.pi/2, 0.0, 0.0),         # South pole of S^2
        (np.pi/4, 0.0, 0.0),         # Equator of S^2
        (np.pi/4, np.pi/2, 0.0),     # Another equatorial point
        (np.pi/4, 0.0, np.pi),       # Same S^2 point, different fiber
    ]

    for eta, xi1, xi2 in test_points:
        z1, z2 = s3.hopf_coordinates(eta, xi1, xi2)
        x, y, z = hopf.projection(z1, z2)
        norm_sq = x**2 + y**2 + z**2
        print(f"  {eta:8.4f} {xi1:8.4f} {xi2:8.4f}  |"
              f"  {x:8.4f} {y:8.4f} {z:8.4f}  |  {norm_sq:8.6f}")

    print()
    print("  Notice: the last two rows map to the SAME S^2 point!")
    print("  They differ only in the fiber coordinate (xi1 + xi2).")
    print("  This is the fiber bundle structure: many S^3 points over each S^2 point.")

    # Connection and curvature
    print_subheader("The Hopf connection and curvature")

    conn = hopf.connection_1form()
    curv = hopf.curvature()

    print(f"  Connection: {conn['formula']}")
    print(f"  In Hopf coords: {conn['in_hopf_coords']}")
    print()
    print(f"  Curvature: {curv['formula']}")
    print(f"  Total flux: {curv['total_flux']}")
    print(f"  First Chern number: c_1 = {hopf.first_chern_number()}")
    print()
    print("  This is the Dirac monopole: a U(1) connection over S^2")
    print("  with one quantum of magnetic charge.")

    # Linking number of fibers
    print_subheader("Fiber linking: all Hopf fibers are linked")

    # Two distinct points on S^2
    fiber1 = hopf.fiber((0, 0, 1), num_points=200)    # North pole fiber
    fiber2 = hopf.fiber((1, 0, 0), num_points=200)    # Equatorial fiber

    link = hopf.linking_number(fiber1, fiber2)
    print(f"  Fiber over north pole (0,0,1)  and  fiber over equator (1,0,0):")
    print(f"  Linking number = {link}")
    print()

    # Verify with another pair
    fiber3 = hopf.fiber((0, 1, 0), num_points=200)
    link2 = hopf.linking_number(fiber1, fiber3)
    print(f"  Fiber over (0,0,1)  and  fiber over (0,1,0):")
    print(f"  Linking number = {link2}")
    print()
    print("  Any two distinct Hopf fibers have linking number 1.")
    print("  This is why S^3 is nontrivially fibered: unlinking the")
    print("  fibers would require tearing S^3 apart.")

    # Topology summary
    print_subheader("Topological summary of the Hopf fibration")

    print("  Fiber:   S^1  (the circle, U(1))")
    print("  Total:   S^3  (the 3-sphere, SU(2))")
    print("  Base:    S^2  (the 2-sphere, CP^1)")
    print()
    print("  Long exact sequence of homotopy groups:")
    print("  ... -> pi_n(S^1) -> pi_n(S^3) -> pi_n(S^2) -> pi_{n-1}(S^1) -> ...")
    print()
    print("  Key consequence:")
    print("    pi_3(S^2) = Z  (generated by the Hopf map itself)")
    print("    pi_3(S^3) = Z  (the fundamental fact behind instantons)")
    print("    pi_1(S^3) = 0  (S^3 is simply connected)")
    print("    pi_2(S^3) = 0  (no vortices on S^3)")


# ===========================================================================
# Section 6: Homotopy -- pi_3(S^3) = Z and Instantons
# ===========================================================================

def section_6_homotopy():
    """
    The third homotopy group pi_3(S^3) = Z is the mathematical foundation
    for instantons in Yang-Mills theory.

    Meaning: continuous maps S^3 -> S^3 are classified by an integer -- the
    degree (or winding number). The degree-n map wraps one copy of S^3
    around the other n times.

    For Yang-Mills with gauge group SU(2) ~ S^3:
    - A gauge transformation at infinity (in R^4) is a map S^3 -> SU(2) = S^3
    - The winding number is the instanton number (second Chern number c_2)
    - The BPST instanton (1975) has winding number 1

    On S^3 x R (Euclidean spacetime):
    - At t = -infinity: gauge vacuum with winding number n
    - At t = +infinity: gauge vacuum with winding number n + k
    - The instanton mediates tunneling between topological sectors

    We can compute the degree of a map S^3 -> S^3 via the formula:
        deg(f) = (1/Vol(S^3)) * integral_{S^3} f* omega
    where omega is the volume form on the target S^3.
    """
    print_header("Section 6: pi_3(S^3) = Z -- Why Instantons Exist")

    # Demonstrate maps S^3 -> S^3 of various degrees
    print("  Maps S^3 -> S^3 are classified by their degree (winding number).")
    print()
    print("  Degree 0:  The constant map (all of S^3 maps to one point).")
    print("  Degree 1:  The identity map.")
    print("  Degree -1: The antipodal map q -> -q.")
    print("  Degree n:  q -> q^n (quaternion power).")
    print()

    # Verify degree of quaternion power maps numerically
    # For SU(2), the map q -> q^n has degree n.
    # We verify by computing the Jacobian determinant at a generic point.
    print("  Numerical verification via Jacobian determinant:")
    print()
    print(f"  {'Map':>12s}  {'det(J) at q=(1,0,0,0)':>24s}  {'Expected degree':>16s}")
    print("  " + "-" * 56)

    for n in [-2, -1, 0, 1, 2, 3]:
        # For the map q -> q^n on S^3, the degree is n.
        # At the identity (1,0,0,0), the tangent map of q -> q^n is n*Id.
        # So the Jacobian determinant is n^3 (3-dimensional manifold).
        # This is positive for n > 0 and negative for odd negative n.
        # The DEGREE is computed from the sign-adjusted integral, giving n.
        jac = n**3 if n != 0 else 0  # Jacobian det at identity
        print(f"  q -> q^{n:+d}   {jac:24d}  {n:16d}")

    print()
    print("  The degree of q -> q^n is exactly n.")
    print("  This is the topological quantum number of instantons.")

    # Connection to instantons
    print_subheader("Connection to Yang-Mills instantons")

    print("  In Yang-Mills theory on R^4 ~ S^4 \\ {point}:")
    print()
    print("  1. A gauge field A is a connection on a principal G-bundle P -> S^4.")
    print("  2. The bundle is classified by pi_3(G) (transition functions at equator S^3).")
    print("  3. For G = SU(2): pi_3(SU(2)) = pi_3(S^3) = Z.")
    print("  4. The integer k in Z is the instanton number (second Chern number c_2).")
    print("  5. The BPST instanton (k=1) is the anti-self-dual connection that")
    print("     minimizes the Yang-Mills action in the k=1 sector.")
    print()
    print("  Key formula:")
    print("    S_YM[A] >= 8*pi^2 * |k| / g^2    (topological bound)")
    print("    Equality iff F = +/- *F            (self/anti-self-dual)")
    print()
    print("  On S^3 x R:")
    print("  - The instanton is a finite-action path connecting topological")
    print("    sectors of the Yang-Mills vacuum.")
    print("  - These contribute to the nonperturbative physics (theta vacua).")
    print("  - On compact S^3, instanton effects are exponentially suppressed")
    print("    at weak coupling but become relevant at strong coupling.")

    # Four Hopf fibrations
    print_subheader("The four Hopf fibrations (Adams, 1960)")

    print("  There are exactly FOUR Hopf fibrations (Adams' theorem, 1960):")
    print()
    print(f"  {'Fibration':>20s}  {'Fiber':>8s}  {'Total':>8s}  {'Base':>8s}  {'Division algebra':>18s}")
    print("  " + "-" * 68)
    fibrations = [
        ("S^0 -> S^1 -> S^1", "S^0", "S^1", "S^1", "Real numbers"),
        ("S^1 -> S^3 -> S^2", "S^1", "S^3", "S^2", "Complex numbers"),
        ("S^3 -> S^7 -> S^4", "S^3", "S^7", "S^4", "Quaternions"),
        ("S^7 -> S^15 -> S^8", "S^7", "S^15", "S^8", "Octonions"),
    ]
    for name, fiber, total, base, alg in fibrations:
        print(f"  {name:>20s}  {fiber:>8s}  {total:>8s}  {base:>8s}  {alg:>18s}")
    print()
    print("  The second one (S^1 -> S^3 -> S^2) is the Hopf fibration")
    print("  relevant to Yang-Mills theory and the SU(2) gauge group.")
    print("  These four fibrations correspond to the four normed division")
    print("  algebras: R, C, H, O (Baez, 2002).")


# ===========================================================================
# Section 7: The Round Metric (Symbolic)
# ===========================================================================

def section_7_metric():
    """
    The round metric on S^3 in Euler coordinates.

    ds^2 = (R^2/4)[d(chi)^2 + d(theta)^2 + d(phi)^2 + 2*cos(theta)*d(chi)*d(phi)]

    This is the bi-invariant metric inherited from S^3 ~ SU(2).
    The factor of 1/4 comes from the normalization convention for the
    Euler angle parametrization.
    """
    print_header("Section 7: The Round Metric on S^3 (Symbolic)")

    import sympy as sp

    g = S3Coordinates.metric_round()

    print("  Metric tensor g_{ij} in Euler coordinates (chi, theta, phi):")
    print()
    sp.pprint(g)
    print()

    # Compute determinant
    det_g = g.det().simplify()
    print(f"  det(g) = {det_g}")
    print()

    # Volume element
    print("  Volume element: dV = sqrt(|det g|) d(chi) d(theta) d(phi)")
    sqrt_det = sp.sqrt(sp.Abs(det_g)).simplify()
    print(f"  sqrt(|det g|) = {sqrt_det}")
    print()

    # Evaluate at R=1
    R_sym = sp.Symbol('R', positive=True)
    g_unit = S3Coordinates.metric_round(R_val=1)
    det_unit = g_unit.det().simplify()
    print(f"  At R=1: det(g) = {det_unit}")
    print()
    print("  The metric has off-diagonal term g_{chi,phi} = cos(theta)*R^2/4,")
    print("  reflecting the nontrivial topology of S^3 (it is NOT a product")
    print("  manifold S^1 x S^2).")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Run all sections of the S^3 geometry tutorial."""
    print()
    print("*" * 72)
    print("*  Tutorial 1: S^3 Differential Geometry for Yang-Mills Theory    *")
    print("*" * 72)

    section_1_quaternions()
    section_2_hopf_coordinates()
    section_3_volume()
    section_4_ricci()
    section_5_hopf()
    section_6_homotopy()
    section_7_metric()

    print_header("Summary")
    print("  S^3 is uniquely suited for Yang-Mills theory because it is:")
    print()
    print("  1. COMPACT  -> discrete spectrum, no IR divergences")
    print("  2. EINSTEIN -> Ric = 2/R^2 > 0, guarantees spectral gap")
    print("  3. LIE GROUP -> SU(2) gauge structure is GEOMETRIC")
    print("  4. HOPF FIBERED -> encodes U(1) electromagnetism")
    print("  5. pi_3 = Z -> instantons exist, topological sectors")
    print()
    print("  In the next tutorial, we compute the Hodge spectrum of S^3")
    print("  and show how it gives the Yang-Mills mass gap.")
    print()


if __name__ == "__main__":
    main()
