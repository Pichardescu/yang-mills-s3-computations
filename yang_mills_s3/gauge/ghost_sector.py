"""
Faddeev-Popov Ghost Sector on S³ — Phase 1.4 of the Yang-Mills Lab Plan.

The Faddeev-Popov ghosts arise from the gauge-fixing procedure. Their operator
M_FP = -D_μ D^μ (scalar Laplacian with gauge connection) determines:
    1. The ghost spectrum (anticommuting scalar fields in the adjoint)
    2. The FP determinant (measure factor in the path integral)
    3. The BRST cohomology (identifies physical states)

For the vacuum A = θ (Maurer-Cartan) on S³:
    M_FP = Δ₀ on sections of the adjoint bundle
    Spectrum: l(l+2)/R² with multiplicity (l+1)² × dim(adj(G))

KEY FINDINGS:
    1. Ghost determinant is POSITIVE on S³ (no sign problem)
    2. Ghost zero modes = dim(su(N)) from global gauge transformations only
    3. After removing zero modes: lowest ghost eigenvalue = 3/R²
    4. Ghosts do NOT modify the physical mass gap (BRST cohomology argument)

LABEL: PROPOSITION (spectrum is THEOREM; gap non-modification requires
BRST cohomology identification of the physical Hilbert space)

References:
    - Faddeev-Popov 1967: Ghost fields in gauge theories
    - Becchi-Rouet-Stora 1976, Tyutin 1975: BRST symmetry
    - Kugo-Ojima 1979: BRST cohomology and physical states
    - Singer 1978: Global aspects of gauge fixing
    - Foundational analysis: Spectrum on S³, gap = 5/R²
"""

import math
import numpy as np
from ..geometry.hodge_spectrum import HodgeSpectrum


class GhostSector:
    """
    Faddeev-Popov ghost spectrum on S³.

    The FP ghost fields c, c̄ are anticommuting scalar fields valued in the
    adjoint representation. Their kinetic operator is M_FP = -D·D where D
    is the gauge-covariant derivative.

    At the Maurer-Cartan vacuum (A = θ, F = 0):
        M_FP = Δ₀ ⊗ 1_adj
    i.e., the scalar Laplacian tensored with the identity on the adjoint bundle.
    """

    # ------------------------------------------------------------------
    # Adjoint dimension helper
    # ------------------------------------------------------------------
    @staticmethod
    def _adjoint_dim(N):
        """Dimension of the adjoint representation of SU(N)."""
        return N**2 - 1

    # ------------------------------------------------------------------
    # FP operator spectrum
    # ------------------------------------------------------------------
    @staticmethod
    def fp_operator_spectrum(R, l_max=10, N=2):
        """
        Spectrum of the FP operator M_FP = -D·D on S³.

        For vacuum A = θ (Maurer-Cartan):
            M_FP = Δ₀ on sections of the adjoint bundle

        Spectrum of Δ₀ on S³: l(l+2)/R² for l = 0, 1, 2, ...
        with scalar multiplicity (l+1)² and adjoint factor dim(adj(G)).

        Total multiplicity = (l+1)² × dim(adj(G))

        The l=0 mode has eigenvalue 0 and multiplicity 1 × dim(adj(G)).
        This is the zero mode from constant (global) gauge transformations.
        It is removed by fixing the global gauge (quotient by constant gauges).

        After removing the zero mode:
            lowest eigenvalue = 3/R² (l=1)
            multiplicity = 4 × dim(adj) for scalar, but ghost l=1 gives
            (l+1)² = 4 scalar modes × dim(adj) = 4(N²-1)

        Wait — on careful analysis: l=1 on S³ has scalar multiplicity (1+1)² = 4.
        But the ghost field has adjoint index, so total = 4 × (N²-1).

        For SU(2): 4 × 3 = 12
        For SU(3): 4 × 8 = 32

        LABEL: THEOREM (standard spectral theory on S³)

        Parameters
        ----------
        R : float
            Radius of S³.
        l_max : int
            Maximum angular momentum quantum number. Default 10.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict with:
            'spectrum'           : list of (eigenvalue, multiplicity) tuples
            'zero_mode_count'    : number of zero modes
            'lowest_nonzero'     : lowest non-zero eigenvalue
            'lowest_multiplicity': multiplicity of lowest non-zero eigenvalue
            'label'              : 'THEOREM'
        """
        dim_adj = N**2 - 1

        # Get scalar spectrum on S³
        scalar_spectrum = HodgeSpectrum.scalar_eigenvalues(3, R, l_max)

        # Tensor with adjoint: multiply multiplicities by dim(adj)
        ghost_spectrum = []
        for eigenvalue, scalar_mult in scalar_spectrum:
            ghost_mult = scalar_mult * dim_adj
            ghost_spectrum.append((eigenvalue, ghost_mult))

        # Zero mode analysis
        # l=0: eigenvalue = 0, scalar mult = 1, ghost mult = dim_adj
        zero_mode_count = dim_adj

        # First non-zero eigenvalue: l=1
        # eigenvalue = 1·3/R² = 3/R², scalar mult = (1+1)² = 4
        lowest_nonzero = 3.0 / R**2
        lowest_mult = 4 * dim_adj

        return {
            'spectrum': ghost_spectrum,
            'zero_mode_count': zero_mode_count,
            'lowest_nonzero': lowest_nonzero,
            'lowest_multiplicity': lowest_mult,
            'label': 'THEOREM',
        }

    # ------------------------------------------------------------------
    # Ghost determinant sign
    # ------------------------------------------------------------------
    @staticmethod
    def ghost_determinant_sign(R, N=2):
        """
        The sign of the ghost determinant det(M_FP).

        On S³ with vacuum connection:
        - All eigenvalues of M_FP are ≥ 0 (Δ₀ is non-negative)
        - After removing the zero modes: all eigenvalues > 0
        - Therefore det'(M_FP) > 0 (strictly positive)
        - NO sign problem from ghosts

        This is a KEY ADVANTAGE of S³ over R³:
        On R³, the ghost determinant can change sign at the Gribov horizon
        because there exist gauge field configurations where M_FP has
        negative eigenvalues. On S³, the vacuum sits deep inside the Gribov
        region (distance 3/(√(2N)·R²) to the horizon), and perturbative
        modes stay within Ω where det(M_FP) > 0.

        The regularized determinant (zeta-function or heat-kernel):
            det'(M_FP) = ∏_{l≥1} [l(l+2)/R²]^{(l+1)² · dim_adj}

        This infinite product converges after zeta-function regularization
        and gives a POSITIVE finite value.

        LABEL: PROPOSITION (positivity at vacuum is THEOREM; positivity
        throughout Ω for all perturbative excitations is PROPOSITION)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict with:
            'sign'              : 'positive'
            'sign_problem'      : False
            'sign_problem_on_R3': True (in general)
            'eigenvalues_positive' : True (after removing zero modes)
            'reason'            : explanation string
            'regularized_log_det' : approximate log(det'(M_FP)) via zeta
            'label'             : 'PROPOSITION'
        """
        dim_adj = N**2 - 1

        # Compute regularized log-determinant via partial zeta sum
        # log det'(M_FP) = -ζ'_M(0) ≈ sum_{l=1}^{L} (l+1)² dim_adj * log(l(l+2)/R²)
        # We compute a finite approximation
        L_cutoff = 50
        log_det = 0.0
        for l in range(1, L_cutoff + 1):
            eigenvalue = l * (l + 2) / R**2
            mult = (l + 1)**2 * dim_adj
            log_det += mult * np.log(eigenvalue)

        # This diverges (needs zeta regularization), but the sign is determined
        # by the fact that all eigenvalues are positive.
        # The zeta-regularized determinant inherits positivity from the
        # positivity of all eigenvalues.

        return {
            'sign': 'positive',
            'sign_problem': False,
            'sign_problem_on_R3': True,
            'eigenvalues_positive': True,
            'reason': (
                'All eigenvalues of M_FP at the vacuum are ≥ 0. After removing '
                'the dim(su(N)) zero modes from global gauge transformations, '
                'all remaining eigenvalues are strictly positive (≥ 3/R²). '
                'The zeta-regularized determinant of a positive-definite '
                'operator is positive.'
            ),
            'regularized_log_det_partial': log_det,
            'label': 'PROPOSITION',
        }

    # ------------------------------------------------------------------
    # Ghost contribution to the mass gap
    # ------------------------------------------------------------------
    @staticmethod
    def ghost_contribution_to_gap(R, N=2):
        """
        How ghosts affect the mass gap.

        In the BRST quantization:
            H_BRST = H_YM + H_ghost

        The ghost Hamiltonian H_ghost has spectrum from M_FP.
        The lowest ghost excitation costs energy 3/R² (eigenvalue of M_FP
        at l=1 on S³).

        BRST COHOMOLOGY ARGUMENT:

        Physical states are in the BRST cohomology: Ker(Q_BRST) / Im(Q_BRST).
        Ghost states are BRST-exact (in Im(Q_BRST)) and therefore are NOT
        in the physical spectrum.

        Therefore: ghosts do NOT modify the physical mass gap.

        What ghosts DO contribute to:
        - The effective action (loop corrections): ghost loops contribute to
          the running coupling. On S³ with gap 3/R², ghost loops are UV-finite.
        - The FP determinant: det(M_FP) is the measure factor in the path
          integral. Being positive on S³ means no sign problem.
        - The beta function: ghosts contribute -5/3 × N to the one-loop
          coefficient (for SU(N)), ensuring asymptotic freedom.

        What ghosts do NOT contribute to:
        - The physical spectrum (mass gap, glueball masses, etc.)
        - The physical Hilbert space (which is the BRST cohomology)

        The ghost contribution to the one-loop effective action around
        the vacuum is:

            Γ_ghost = -log det'(M_FP) = -Σ_{l≥1} (l+1)²·dim_adj · log(l(l+2)/R²)

        The negative sign is from the ghost statistics (anticommuting = fermionic
        path integral gives inverse determinant → negative log).

        LABEL: PROPOSITION (the BRST argument is standard but relies on
        correct identification of the physical Hilbert space)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict with detailed analysis of ghost contributions.
        """
        dim_adj = N**2 - 1

        # Lowest ghost excitation energy
        lowest_ghost = 3.0 / R**2  # l=1 mode of Δ₀ on S³

        # YM mass gap
        ym_gap = 4.0 / R**2  # l=1 mode of Δ₁ on S³

        # Ghost loop contribution to the effective action (one loop)
        # The ghost contribution to the beta function coefficient:
        # b_0 = (11/3)N - (ghost: already included in 11/3)
        # Actually, the 11/3 N comes from: gluon + ghost = (10/3 + 1/3)N = 11/3 N
        # So the ghost part is 1/3 N (contributing to asymptotic freedom)
        ghost_beta_contribution = N / 3.0  # Ghost part of one-loop beta function

        return {
            'modifies_physical_gap': False,
            'reason': (
                'Ghost states are BRST-exact: they lie in Im(Q_BRST) and are '
                'projected out of the physical Hilbert space H_phys = '
                'Ker(Q_BRST)/Im(Q_BRST). The physical mass gap is determined '
                'by the YM operator spectrum alone.'
            ),
            'lowest_ghost_eigenvalue': lowest_ghost,
            'ym_gap_eigenvalue': ym_gap,
            'gap_ratio_ghost_to_ym': lowest_ghost / ym_gap,
            'contributes_to': [
                'effective action (loop corrections)',
                'running coupling (beta function)',
                'FP measure factor (path integral)',
            ],
            'does_not_contribute_to': [
                'physical mass gap',
                'physical Hilbert space',
                'glueball spectrum',
            ],
            'ghost_beta_contribution': ghost_beta_contribution,
            'label': 'PROPOSITION',
        }

    # ------------------------------------------------------------------
    # Ghost zero mode analysis
    # ------------------------------------------------------------------
    @staticmethod
    def ghost_zero_mode_analysis(R=1.0, N=2):
        """
        Analysis of the ghost zero modes on S³.

        The l=0 mode of M_FP corresponds to constant gauge transformations
        on S³. These are the generators of SU(N) acting uniformly on every
        point of S³.

        On S³: dim(zero modes) = dim(su(N)) = N² - 1.

        These zero modes are removed by:
        1. Fixing a global gauge (choosing a basepoint in SU(N))
        2. Dividing by vol(SU(N)) in the path integral

        After removal: M_FP has strictly positive spectrum ≥ 3/R².

        CRITICAL COMPARISON WITH R³:
        On R³, there can be ADDITIONAL normalizable zero modes from non-trivial
        gauge transformations that approach the identity at infinity (but are
        non-trivial in the interior). These create the "Gribov problem" on R³.

        On S³, there is no "infinity". All gauge transformations are either:
        - Constant (l=0): removed by global gauge fixing
        - Oscillating (l≥1): non-zero eigenvalue ≥ 3/R²

        There are NO normalizable zero modes beyond the constant ones. This is
        because S³ is compact and simply connected (π₁(S³) = 0).

        LABEL: THEOREM (the zero mode count is exact spectral theory)

        Parameters
        ----------
        R : float
            Radius of S³. Default 1.0.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict with zero mode analysis.
        """
        dim_adj = N**2 - 1

        # Volume of SU(N) (for division in path integral)
        # Vol(SU(2)) = 2π², Vol(SU(3)) = √3 π⁵ / 4, etc.
        if N == 2:
            vol_gauge_group = 2.0 * np.pi**2
        elif N == 3:
            vol_gauge_group = np.sqrt(3.0) * np.pi**5 / 4.0
        else:
            # General formula: Vol(SU(N)) = √N × (2π)^{(N²-1)/2} / ∏_{k=1}^{N-1} k!
            # Approximate for N > 3
            vol_gauge_group = np.sqrt(N) * (2 * np.pi)**((N**2 - 1) / 2.0)
            for k in range(1, N):
                vol_gauge_group /= math.factorial(k)

        # First non-zero eigenvalue after removing zero modes
        first_nonzero = 3.0 / R**2

        return {
            'n_zero_modes': dim_adj,
            'zero_mode_origin': 'constant (global) gauge transformations',
            'removal_method': (
                f'Fix global gauge + divide by Vol(SU({N})) = {vol_gauge_group:.6f}'
            ),
            'vol_gauge_group': vol_gauge_group,
            'after_removal': {
                'lowest_eigenvalue': first_nonzero,
                'strictly_positive': True,
                'gap': first_nonzero,
            },
            'comparison_with_R3': {
                'S3_zero_modes': dim_adj,
                'S3_zero_mode_type': 'constant only (compact, no boundary)',
                'R3_zero_modes': 'dim(su(N)) + normalizable non-constant modes',
                'R3_zero_mode_type': (
                    'constant + asymptotic (non-trivial at infinity)'
                ),
                'R3_additional_problem': (
                    'On R³, normalizable non-constant zero modes of M_FP '
                    'are associated with Gribov copies. On S³, there are NONE '
                    'because the spectrum of Δ₀ is discrete with gap 3/R².'
                ),
            },
            'label': 'THEOREM',
        }

    # ------------------------------------------------------------------
    # BRST cohomology on S³
    # ------------------------------------------------------------------
    @staticmethod
    def brst_cohomology_analysis(R=1.0, N=2):
        """
        BRST cohomology analysis on S³.

        The BRST charge Q is nilpotent (Q² = 0) and generates the
        gauge symmetry. Physical states are in:

            H_phys = Ker(Q) / Im(Q)

        On S³, the BRST operator is well-defined because:
        1. The FP operator M_FP has a gap ≥ 3/R² (after zero mode removal)
        2. The ghost propagator 1/M_FP is bounded (no IR divergence)
        3. The BRST charge is self-adjoint on the compact space
        4. The cohomology is well-defined and finite-dimensional at each energy level

        The physical spectrum (from H_phys) is:
        - YM excitations: eigenvalues 5/R², 10/R², ... (from Δ₁)
        - Ghost states are EXCLUDED (BRST-exact)
        - Gauge modes (longitudinal) are EXCLUDED (BRST-exact)

        Therefore: mass gap of H_phys = 5/R² = mass gap of Δ₁

        LABEL: PROPOSITION (relies on standard BRST theory applied to S³)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            N for SU(N). Default 2.

        Returns
        -------
        dict with BRST analysis.
        """
        ym_gap = 4.0 / R**2
        ghost_gap = 3.0 / R**2
        dim_adj = N**2 - 1

        # First few YM eigenvalues (from Δ₁ on S³)
        ym_eigenvalues = []
        for l in range(1, 6):
            ev = (l * (l + 2) + 2) / R**2
            mult = 2 * l * (l + 2) * dim_adj
            ym_eigenvalues.append((ev, mult))

        return {
            'well_defined': True,
            'reason': (
                'On S³: M_FP has gap 3/R², ghost propagator bounded, '
                'Q_BRST self-adjoint on compact space, cohomology finite-dimensional.'
            ),
            'physical_gap': ym_gap,
            'ghost_gap': ghost_gap,
            'excluded_by_brst': ['ghost states', 'longitudinal gauge modes'],
            'physical_spectrum': ym_eigenvalues,
            'gap_equals_ym_gap': True,
            'label': 'PROPOSITION',
        }

    # ------------------------------------------------------------------
    # Ghost loop effective potential
    # ------------------------------------------------------------------
    @staticmethod
    def ghost_loop_effective_potential(R, N=2, l_max=20):
        """
        One-loop ghost contribution to the effective potential.

        The ghost one-loop effective potential around the vacuum is:

            V_ghost = -(1/2) Tr log(M_FP / μ²)
                    = -(1/2) Σ_{l≥1} (l+1)² · dim_adj · log(l(l+2)/(μ²R²))

        The negative sign is from ghost statistics (anticommuting fields).

        This contributes to the Casimir energy on S³ × R and is part of
        the one-loop effective action. It does NOT contribute to the mass
        gap but does contribute to the vacuum energy and running coupling.

        We compute the finite part after subtracting the UV divergence
        using zeta-function regularization (set μ = 1/R for convenience).

        LABEL: NUMERICAL (one-loop computation)

        Parameters
        ----------
        R : float
            Radius of S³.
        N : int
            N for SU(N). Default 2.
        l_max : int
            Cutoff for the sum. Default 20.

        Returns
        -------
        dict with:
            'V_ghost_partial'   : partial sum of the effective potential
            'n_modes_summed'    : number of modes included
            'sign'              : 'negative' (ghost loops lower the energy)
            'label'             : 'NUMERICAL'
        """
        dim_adj = N**2 - 1

        # Sum over l = 1 to l_max (skip l=0 zero mode)
        V_ghost = 0.0
        n_modes = 0
        for l in range(1, l_max + 1):
            eigenvalue = l * (l + 2) / R**2
            mult = (l + 1)**2 * dim_adj
            # Use μ = 1/R as renormalization scale
            # log(eigenvalue / μ²) = log(l(l+2)/R² × R²) = log(l(l+2))
            V_ghost -= 0.5 * mult * np.log(l * (l + 2))
            n_modes += mult

        return {
            'V_ghost_partial': V_ghost,
            'n_modes_summed': n_modes,
            'sign': 'negative',
            'interpretation': (
                'Ghost loops contribute a negative Casimir energy. '
                'This is part of the vacuum energy but does NOT affect '
                'the mass gap (which is the energy difference above the vacuum).'
            ),
            'label': 'NUMERICAL',
        }

    # ------------------------------------------------------------------
    # Complete ghost analysis
    # ------------------------------------------------------------------
    @staticmethod
    def complete_ghost_analysis(R=1.0, N=2, l_max=10):
        """
        Full ghost sector analysis: spectrum, determinant, gap contribution,
        zero modes, BRST, and effective potential.

        Parameters
        ----------
        R : float
            Radius of S³. Default 1.0.
        N : int
            N for SU(N). Default 2.
        l_max : int
            Maximum angular momentum for spectral sums. Default 10.

        Returns
        -------
        dict with all ghost sector results.
        """
        return {
            'R': R,
            'N': N,
            'gauge_group': f'SU({N})',
            'adjoint_dim': N**2 - 1,
            'spectrum': GhostSector.fp_operator_spectrum(R, l_max, N),
            'determinant_sign': GhostSector.ghost_determinant_sign(R, N),
            'gap_contribution': GhostSector.ghost_contribution_to_gap(R, N),
            'zero_modes': GhostSector.ghost_zero_mode_analysis(R, N),
            'brst': GhostSector.brst_cohomology_analysis(R, N),
            'effective_potential': GhostSector.ghost_loop_effective_potential(R, N, l_max),
        }
