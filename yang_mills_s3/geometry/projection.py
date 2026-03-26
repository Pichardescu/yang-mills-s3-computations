"""
Projection Formalism: How 4D (S³) Reality Maps to 3D Observables

ONTOLOGICAL NOTE:
The universe IS 4D. Not "projects to 3D." We, as 3D observers with 3D
instruments, measure 3D cross-sections of 4D objects. Like a 2D being
measuring the shadow of a 3D ball — the ball doesn't "project", the
being's instruments are limited.

The mathematical question: given an eigenfunction ψ on S³ (4D reality),
what does a 3D observer MEASURE?

Key insight: the Hopf fibration π: S³ → S² is NOT the physical projection.
It's one specific way to decompose S³ into (base × fiber). The physical
"observation" is more nuanced — it depends on WHICH 3D slice the observer
occupies and HOW the measurement couples to the w direction.

TEMPERATURE HYPOTHESIS:
Temperature = amplitude/frequency of w-oscillation in an ensemble.
- Cold: particles have small, slow w-oscillation → localized in fiber
- Hot: particles have large, fast w-oscillation → spread across fiber
- Deconfinement: w-oscillation violent enough to break Hopf fiber linking
- Plasma: 4D structure becomes manifest (can't be explained in 3D alone)
- Tunneling increases with T because w-oscillation provides the "shortcut"
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


# Physical constants
HBAR_C = 197.3269804  # MeV·fm
K_BOLTZMANN = 8.617333e-2  # MeV/K


class Projection:
    """
    Mathematical framework for how S³ geometry maps to 3D observables.

    Three distinct concepts:
    1. Eigenvalue on S³ (the 4D reality)
    2. Fiber-averaged observable (what a 3D measurement integrates over)
    3. Measured quantity (what the detector reports)
    """

    @staticmethod
    def hopf_map(z1: complex, z2: complex) -> Tuple[float, float, float]:
        """The Hopf map π: S³ → S². Not 'the' projection, but a key decomposition."""
        norm_sq = abs(z1)**2 + abs(z2)**2
        x = 2 * (z1 * z2.conjugate()).real / norm_sq
        y = 2 * (z1 * z2.conjugate()).imag / norm_sq
        z = (abs(z1)**2 - abs(z2)**2) / norm_sq
        return (x, y, z)

    @staticmethod
    def fiber_integral_eigenfunction(l: int, R: float) -> Dict:
        """
        How a spherical eigenfunction on S³ looks when integrated over the fiber.

        An eigenfunction ψ_l on S³ with eigenvalue μ_l = l(l+2)/R² decomposes
        under the Hopf fibration as:

            ψ_l(η, ξ₁, ξ₂) = Σ_m f_{l,m}(η) · e^{im(ξ₁-ξ₂)}

        where η is the "base" coordinate (position on S²) and (ξ₁-ξ₂) is
        the fiber coordinate.

        Integrating over the fiber (averaging over w) projects onto the m=0 mode:

            ⟨ψ_l⟩_fiber = f_{l,0}(η)

        This is a function on S² — what the 3D observer "sees."

        KEY POINT: The m=0 mode has a DIFFERENT effective eigenvalue on S²
        than the full eigenvalue on S³. This is the source of the "factor."
        """
        # Full eigenvalue on S³
        eigenvalue_s3 = l * (l + 2) / R**2

        # The m=0 projection onto S² gives effective angular momentum l_eff
        # On S², the eigenvalue is l_eff(l_eff+1)/R²
        # The relationship: l_eff = l (same angular momentum quantum number)
        # But the eigenvalue changes: l(l+2)/R² on S³ vs l(l+1)/R² on S²
        eigenvalue_s2 = l * (l + 1) / R**2

        # The DIFFERENCE encodes what's "lost" in the fiber averaging
        fiber_contribution = eigenvalue_s3 - eigenvalue_s2  # = l/R²

        return {
            'l': l,
            'eigenvalue_s3': eigenvalue_s3,         # 4D reality
            'eigenvalue_s2_projected': eigenvalue_s2, # 3D observable (m=0 mode)
            'fiber_contribution': fiber_contribution,  # what the fiber adds
            'ratio_s3_to_s2': eigenvalue_s3 / eigenvalue_s2 if l > 0 else float('inf'),
            'note': (
                f'The 4D eigenvalue l(l+2)/R² = {eigenvalue_s3:.4f}/R² '
                f'projects to l(l+1)/R² = {eigenvalue_s2:.4f}/R² on S², '
                f'with fiber correction l/R² = {fiber_contribution:.4f}/R²'
            )
        }

    @staticmethod
    def mass_from_eigenvalue_4d(eigenvalue: float, R: float) -> float:
        """
        Mass from the FULL 4D eigenvalue. This is what we've been computing.
        m = ℏc · sqrt(eigenvalue)
        """
        return HBAR_C * np.sqrt(eigenvalue)

    @staticmethod
    def mass_from_eigenvalue_3d_projected(l: int, R: float) -> float:
        """
        Mass from the PROJECTED eigenvalue on S² (fiber-averaged).
        Uses l(l+1)/R² instead of l(l+2)/R².
        """
        eigenvalue = l * (l + 1) / R**2
        return HBAR_C * np.sqrt(eigenvalue)

    @staticmethod
    def yang_mills_gap_4d(R: float) -> Dict:
        """
        The Yang-Mills mass gap as seen in 4D vs 3D projection.

        4D: Δ_YM on 1-forms = (l(l+2)+2)/R² with l=1 → 5/R²

        For 1-forms, the Weitzenböck decomposition gives:
            Δ_YM = Δ_Hodge + Ric = nabla*nabla + 2/R²

        The Hodge part on 1-forms: l(l+2)/R² + 2/R² (this 2/R² is Ricci, already in S³)

        The question: what does a 3D observer measure?

        If the observer integrates over the fiber, the Ricci contribution
        (which comes from S³ curvature) is STILL present because it's
        intrinsic to S³. The observer doesn't "lose" the curvature by
        being 3D — they lose the fiber quantum number m.
        """
        # Full gap in 4D
        gap_4d_eigenvalue = 5 / R**2
        gap_4d_mass = HBAR_C * np.sqrt(gap_4d_eigenvalue)

        # If we naively project to S² (wrong — Ricci is intrinsic)
        # We'd get l(l+1)/R² + 2/R² = 1*2/R² + 2/R² = 4/R²
        gap_projected_naive = 4 / R**2
        gap_projected_mass = HBAR_C * np.sqrt(gap_projected_naive)

        # The Ricci term 2/R² is NOT a fiber effect — it's intrinsic curvature
        # A 3D observer STILL feels it (like a 2D being on a curved sheet
        # still feels the curvature even without seeing the 3rd dimension)
        # So the observed gap should still be 5/R²

        return {
            'gap_eigenvalue_4d': gap_4d_eigenvalue,
            'gap_mass_4d_MeV': gap_4d_mass,
            'gap_eigenvalue_projected_naive': gap_projected_naive,
            'gap_mass_projected_MeV': gap_projected_mass,
            'ricci_intrinsic': 2 / R**2,
            'conclusion': (
                'The gap 5/R² is INTRINSIC to S³. A 3D observer still '
                'measures it because the curvature affects geodesics in 3D. '
                'The fiber adds quantum numbers (m) but does not change the gap.'
            )
        }

    @staticmethod
    def observable_spectrum_table(R_fm: float, l_max: int = 6) -> List[Dict]:
        """
        Compare 4D eigenvalues, 3D projections, and experimental masses.

        The key question: which column matches experiment?
        If 4D matches → we're measuring the full 4D structure
        If 3D projected matches → we're only seeing the fiber-averaged part
        """
        hbar_c = HBAR_C

        rows = []
        for l in range(1, l_max + 1):
            # 1-form eigenvalues (Yang-Mills relevant)
            eig_4d = (l * (l + 2) + 2) / R_fm**2  # Hodge + Ricci
            eig_scalar_4d = l * (l + 2) / R_fm**2   # scalar (no Ricci)

            mass_4d = hbar_c * np.sqrt(eig_4d)
            mass_scalar = hbar_c * np.sqrt(eig_scalar_4d)

            # Ratio to ground state (l=1)
            eig_ground = 5 / R_fm**2
            ratio = np.sqrt(eig_4d / eig_ground)

            rows.append({
                'l': l,
                'eigenvalue_1form': eig_4d,
                'eigenvalue_scalar': eig_scalar_4d,
                'mass_1form_MeV': mass_4d,
                'mass_scalar_MeV': mass_scalar,
                'ratio_to_ground': ratio,
            })

        return rows


class TemperatureModel:
    """
    S³ Temperature: amplitude of w-oscillation.

    POSTULATE: Temperature in the S³ framework is the mean kinetic energy of
    oscillation in the w direction:

        T ∝ ⟨(dw/dt)²⟩

    This is speculative but motivated by:
    - Hot systems have more w-oscillation → more tunneling (shortcut in w)
    - Plasma requires 4D explanation (3D structure breaks down)
    - Deconfinement = Hopf fiber linking breaks from violent w-oscillation
    """

    @staticmethod
    def geometric_temperature(R: float) -> float:
        """
        T_geom = ℏc / (2π k_B R)

        This is the temperature scale of the S³ geometry itself.
        NOT the deconfinement temperature (which requires dynamics).

        Returns temperature in MeV (natural units where k_B = 1).
        """
        return HBAR_C / (2 * np.pi * R)

    @staticmethod
    def w_oscillation_energy(amplitude: float, frequency: float) -> float:
        """
        Energy of w-oscillation: E = ½ m ω² A²
        In natural units with m=1: E = ½ ω² A²

        If temperature ∝ this energy, then:
            T ∝ ω² A²
        """
        return 0.5 * frequency**2 * amplitude**2

    @staticmethod
    def deconfinement_estimate(R: float, linking_energy_factor: float = 12.0) -> Dict:
        """
        Estimate deconfinement temperature.

        The geometric temperature T_geom = ℏc/(2πR) is too low by factor ~12.

        HYPOTHESIS: Deconfinement requires enough energy to break the
        topological linking of Hopf fibers. The linking energy is:

            E_link ≈ factor × T_geom

        where the factor encodes the "strength" of the topological binding.

        For SU(3), experimental T_c ≈ 170 MeV, so:
            factor ≈ T_c / T_geom ≈ 170 / 14.3 ≈ 12

        This factor might be calculable from the topology of the fiber bundle.
        CONJECTURE: factor ∝ dim(adjoint) for the gauge group?
            SU(2): dim(adj) = 3,  T_c = 300 MeV, factor ≈ 21
            SU(3): dim(adj) = 8,  T_c = 170 MeV, factor ≈ 12

        Interesting: 21/12 ≈ 1.75 while 300/170 ≈ 1.76. Coincidence?
        The ratio T_c(SU(2))/T_c(SU(3)) ≈ 1.76 is known from lattice.
        But 3/8 × (21/12) ≠ 1.76, so the relationship isn't simply dim(adj).
        """
        T_geom = HBAR_C / (2 * np.pi * R)
        T_deconf = linking_energy_factor * T_geom

        return {
            'T_geometric_MeV': T_geom,
            'linking_factor': linking_energy_factor,
            'T_deconf_MeV': T_deconf,
            'T_deconf_SU3_exp': 170.0,
            'T_deconf_SU2_exp': 300.0,
            'factor_SU3': 170.0 / T_geom,
            'factor_SU2': 300.0 / T_geom,
            'ratio_factors': (300.0 / T_geom) / (170.0 / T_geom),
            'ratio_exp': 300.0 / 170.0,
        }


class MassOntology:
    """
    S³ Framework Mass: extension in w.

    POSTULATE: Mass is proportional to the object's extent in the w direction.

    For eigenmodes on S³:
    - A mode with eigenvalue μ has a characteristic "wavelength" λ = 2π/sqrt(μ)
    - This wavelength determines how the mode extends in ALL directions,
      including w (the fiber direction)
    - For the Hopf fiber S¹, the extension in w is bounded by the
      fiber circumference

    KEY QUESTION: Is mass = ℏc·sqrt(eigenvalue) (standard QFT relation),
    or is there a correction from the fiber geometry?
    """

    @staticmethod
    def standard_mass(eigenvalue: float) -> float:
        """Standard QFT: m = ℏc · sqrt(eigenvalue). Units: MeV if eigenvalue in fm⁻²."""
        return HBAR_C * np.sqrt(eigenvalue)

    @staticmethod
    def fiber_corrected_mass(l: int, R: float) -> Dict:
        """
        Explore whether the fiber geometry modifies the mass formula.

        On S³ with Hopf fibration, a mode of angular momentum l decomposes into
        fiber modes labeled by m (the "magnetic" quantum number for the U(1) fiber).

        For scalar harmonics: m ranges from -l to l in steps of 1.
        The fiber winding number m determines the "w-extension" of the mode.

        HYPOTHESIS: The observed mass depends on m:
            m_phys(l, m) = ℏc · sqrt((l(l+2) + 2 - m²/something) / R²)

        This is speculative. The key test is whether any choice of m gives
        the lattice glueball masses with a SINGLE value of R.
        """
        eigenvalue_full = (l * (l + 2) + 2) / R**2
        mass_standard = HBAR_C * np.sqrt(eigenvalue_full)

        # The fiber quantum number m for 1-forms on S³
        # For the Hopf fibration, m labels the U(1) charge
        # Physical modes have m = 0, ±1, ±2, ..., ±l
        masses_by_m = {}
        for m in range(-l, l + 1):
            # Explore: does the fiber quantum number modify the effective mass?
            # In standard math, the eigenvalue is the eigenvalue regardless of m.
            # But in the S³ framework, if mass = w-extension, then m (fiber winding) matters.
            #
            # HYPOTHESIS: effective eigenvalue = (l(l+2)+2)/R² for all m
            # (standard, no correction). But the 3D OBSERVABLE mass might
            # depend on m through the projection mechanism.
            #
            # For now: document this as OPEN, don't fake a formula.
            masses_by_m[m] = mass_standard  # No correction (standard)

        return {
            'l': l,
            'R_fm': R,
            'eigenvalue': eigenvalue_full,
            'mass_standard_MeV': mass_standard,
            'masses_by_fiber_m': masses_by_m,
            'status': 'OPEN — fiber correction to mass not yet derived',
            'note': (
                'In standard QFT, mass = ℏc·sqrt(eigenvalue) regardless of '
                'fiber quantum number. The S³ framework says mass = w-extension, which '
                'MIGHT depend on m. This needs a concrete calculation of how '
                'eigenfunctions distribute over the Hopf fiber.'
            )
        }

    @staticmethod
    def two_R_tension(R_gap: float = 2.2, R_glueball: float = 0.255) -> Dict:
        """
        Document the tension between two R values.

        R_gap = 2.2 fm gives mass gap = 200 MeV (= Λ_QCD)
        R_glueball = 0.255 fm gives 0++ glueball = 1730 MeV

        POSSIBLE RESOLUTIONS:
        1. Different R for different observables (ugly, ad hoc)
        2. Non-perturbative corrections enhance the mass by factor ~8.6
        3. The gap and glueball probe DIFFERENT geometric scales of S³
        4. Mass formula needs fiber correction (ontological)
        5. The gap IS 200 MeV but glueball masses need interaction terms
        """
        ratio = R_gap / R_glueball

        return {
            'R_gap_fm': R_gap,
            'R_glueball_fm': R_glueball,
            'ratio': ratio,
            'gap_MeV': HBAR_C * np.sqrt(5) / R_gap,
            'glueball_MeV': HBAR_C * np.sqrt(5) / R_glueball,
            'possible_resolutions': [
                '1. Non-perturbative enhancement factor ≈ 8.6 (standard physics)',
                '2. Gap and glueball probe different scales of S³ fiber structure',
                '3. Mass = w-extension (compact topology) gives different formula than m=ℏc√μ',
                '4. The factor 8.6 is geometric: R_S³ / circumference_fiber?',
                f'5. 2πR / R = 2π ≈ 6.28 (close but not 8.6)',
                f'6. R² factor: (R/R_gb)² = {ratio**2:.1f} (too large)',
                f'7. Need concrete calculation of fiber integral for bound states',
            ],
            'status': 'OPEN — central tension of the framework'
        }
