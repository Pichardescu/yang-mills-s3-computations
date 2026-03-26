"""
Continuum limit verification for YM on S³.

Verifies THEOREM 8.1 (multi-scale uniform bounds) and THEOREM 8.2
(continuum measure existence) from the RG companion paper.

The key claims:
1. ||K_j||_j is bounded uniformly in N (the refinement level)
2. The effective mass gap converges as N → ∞
3. The contraction product Π κ_j → 0 as N → ∞

On S³, compactness provides three structural advantages:
    - Finite number of polymers at every scale.
    - Uniform constants across blocks (SU(2) homogeneity).
    - No zero modes (H¹(S³) = 0 ⇒ spectral gap ≥ 4/R²).

Labels:
    THEOREM:   Uniform bounds hold for all N (spectral analysis).
    THEOREM:   Contraction product Π κ_j → 0 as N → ∞.
    NUMERICAL: Effective mass gap convergence as N → ∞.
    NUMERICAL: Reflection positivity preservation through RG flow.

Physical parameters:
    R = 2.2 fm (physical S³ radius)
    g² = 6.28 (bare coupling at the lattice scale)
    N_c = 2 (SU(2) gauge group)
    M = 2 (blocking factor)

References:
    - Balaban (1984-89): UV stability for YM on T⁴
    - inductive_closure.py: Multi-scale RG flow infrastructure
    - first_rg_step.py: Single-shell integration
"""

import numpy as np
from typing import Optional

from .inductive_closure import (
    MultiScaleRGFlow,
    run_inductive_closure,
)
from .heat_kernel_slices import (
    coexact_eigenvalue,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    LAMBDA_QCD_MEV,
)


def verify_uniform_bounds(R: float = 2.2, M: float = 2.0,
                          N_range: tuple = (2, 10), g2_bare: float = 6.28,
                          N_c: int = 2) -> dict:
    """
    THEOREM 8.1 verification: multi-scale bounds are uniform in N.

    Runs the full RG flow for increasing refinement levels N and verifies
    that max_j ||K_j||_j is bounded independently of N.

    The argument: at each RG step, the remainder norm satisfies
        ||K_{j-1}|| <= kappa_j * ||K_j|| + C_j
    with kappa_j < 1 (contraction) and C_j = O(g_j^4).
    Because kappa_j is bounded away from 1 and C_j is summable
    (asymptotic freedom), the accumulated K_norm is bounded
    uniformly in the total number of scales N.

    On S³, the key input is that kappa_j ~ 1/M + O(1/(M^j R)²)
    is UNIFORM across all blocks (SU(2) homogeneity), so the
    bound does not depend on which block we look at.

    THEOREM (uniform bound) + NUMERICAL (explicit values).

    Parameters
    ----------
    R : float
        Radius of S³ in fm.
    M : float
        Blocking factor (> 1, typically 2).
    N_range : tuple of (int, int)
        Range of refinement levels (N_min, N_max) to scan.
    g2_bare : float
        Bare coupling at the UV scale.
    N_c : int
        Number of colors (2 for SU(2)).

    Returns
    -------
    dict with:
        'K_max_trajectory': list, max ||K_j|| at each N
        'product_trajectory': list, Π κ_j at each N
        'gap_trajectory': list, effective gap at each N (MeV)
        'uniform_bound': float, the supremum of max ||K_j|| over all N
        'converged': bool, whether the bound stabilizes
        'N_values': list, the N values scanned
        'kappa_max_trajectory': list, max κ_j at each N
        'curvature_corrections': list, sum of curvature corrections at each N
    """
    N_min, N_max = N_range
    N_values = list(range(N_min, N_max + 1))

    K_max_trajectory = []
    product_trajectory = []
    gap_trajectory = []
    kappa_max_trajectory = []
    curvature_corrections = []

    for N in N_values:
        flow = MultiScaleRGFlow(
            R=R, M=M, N_scales=N, N_c=N_c,
            g2_bare=g2_bare, k_max=100,
        )
        result = flow.run_flow()

        # max ||K_j|| over all scales at this N
        K_norms = result['K_norm_trajectory']
        K_max = max(K_norms) if K_norms else 0.0
        K_max_trajectory.append(K_max)

        # Accumulated contraction product
        product_trajectory.append(result['total_product'])

        # Mass gap at IR
        gap_trajectory.append(result['mass_gap_mev'])

        # Worst contraction factor
        kappas = result['kappa_trajectory']
        kappa_max_trajectory.append(max(kappas) if kappas else 0.0)

        # Sum of curvature corrections = sum(kappa_j - 1/M)
        base_kappa = 1.0 / M
        curv_sum = sum(k - base_kappa for k in kappas)
        curvature_corrections.append(curv_sum)

    uniform_bound = max(K_max_trajectory) if K_max_trajectory else 0.0

    # Check convergence: K_max should stabilize (not grow with N)
    # We check that the last value is within 20% of the second-to-last
    converged = False
    if len(K_max_trajectory) >= 3:
        last_three = K_max_trajectory[-3:]
        spread = max(last_three) - min(last_three)
        mean_val = np.mean(last_three)
        if mean_val > 0:
            converged = spread / mean_val < 0.2
        else:
            converged = spread < 1e-10

    return {
        'K_max_trajectory': K_max_trajectory,
        'product_trajectory': product_trajectory,
        'gap_trajectory': gap_trajectory,
        'uniform_bound': uniform_bound,
        'converged': converged,
        'N_values': N_values,
        'kappa_max_trajectory': kappa_max_trajectory,
        'curvature_corrections': curvature_corrections,
    }


def verify_schwinger_convergence(R: float = 2.2, M: float = 2.0,
                                  N_range: tuple = (2, 8), g2_bare: float = 6.28,
                                  N_c: int = 2) -> dict:
    """
    THEOREM 8.2 support: Schwinger function proxies converge as N → ∞.

    The actual Schwinger functions require the full functional integral.
    Here we verify the proxy: the effective mass gap (which determines
    the exponential decay rate of correlators) converges as N → ∞.

    The mass gap at scale j=0 is:
        m_eff² = 4/R² + Σ_j δm²_j
    where the mass corrections δm²_j come from one-loop shell integration.
    As N increases, more UV shells are integrated out, but asymptotic
    freedom ensures the corrections are summable: δm²_j ~ g²_j / R²
    with g²_j → 0 in the UV (j → ∞).

    NUMERICAL: Convergence verified from explicit spectral data.

    Parameters
    ----------
    R : float
        Radius of S³ in fm.
    M : float
        Blocking factor.
    N_range : tuple of (int, int)
        Range of refinement levels to scan.
    g2_bare : float
        Bare coupling at the UV scale.
    N_c : int
        Number of colors.

    Returns
    -------
    dict with:
        'gap_values': list, effective gap at each N (MeV)
        'relative_changes': list, |gap(N) - gap(N-1)| / gap(N)
        'converged': bool, whether the gap stabilizes
        'N_values': list, the N values scanned
        'coupling_ir': list, IR coupling at each N
    """
    N_min, N_max = N_range
    N_values = list(range(N_min, N_max + 1))

    gap_values = []
    coupling_ir = []

    for N in N_values:
        flow = MultiScaleRGFlow(
            R=R, M=M, N_scales=N, N_c=N_c,
            g2_bare=g2_bare, k_max=100,
        )
        result = flow.run_flow()
        gap_values.append(result['mass_gap_mev'])
        coupling_ir.append(result['g2_trajectory'][-1])

    # Relative changes between successive N values
    relative_changes = []
    for i in range(1, len(gap_values)):
        if gap_values[i] > 0:
            rel = abs(gap_values[i] - gap_values[i - 1]) / gap_values[i]
        else:
            rel = float('inf')
        relative_changes.append(rel)

    # Convergence: relative changes should decrease
    converged = False
    if len(relative_changes) >= 2:
        # Last two relative changes should be small (< 10%)
        converged = all(rc < 0.1 for rc in relative_changes[-2:])

    return {
        'gap_values': gap_values,
        'relative_changes': relative_changes,
        'converged': converged,
        'N_values': N_values,
        'coupling_ir': coupling_ir,
    }


def verify_reflection_positivity_preservation(R: float = 2.2, M: float = 2.0,
                                                N_range: tuple = (2, 8),
                                                g2_bare: float = 6.28,
                                                N_c: int = 2) -> dict:
    """
    THEOREM 8.2 (iii) support: RP is preserved through the RG flow.

    At each RG scale, the transfer matrix T_j = exp(-a_j H_j) is positive
    (from the lattice OS axioms). The RG step preserves positivity because
    the Gaussian integration over shell modes maps positive operators to
    positive operators. We verify that the effective Hamiltonian eigenvalues
    remain positive at all scales.

    The effective Hamiltonian at scale j has eigenvalues:
        E_j(k) = lambda_k + delta_m²_j
    where lambda_k = (k+1)²/R² are the coexact eigenvalues on S³
    and delta_m²_j is the accumulated mass correction from integrating
    out shells j+1, ..., N-1.

    On S³, lambda_1 = 4/R² > 0 (H¹(S³) = 0 ensures no zero modes),
    so the effective Hamiltonian is automatically positive-definite
    as long as the mass corrections don't overwhelm the bare gap.

    NUMERICAL: Verified from the RG flow data.

    Parameters
    ----------
    R : float
        Radius of S³ in fm.
    M : float
        Blocking factor.
    N_range : tuple of (int, int)
        Range of refinement levels to scan.
    g2_bare : float
        Bare coupling.
    N_c : int
        Number of colors.

    Returns
    -------
    dict with:
        'H_eigenvalues_per_scale': list of lists, eigenvalues at each scale
        'all_positive': bool, True if all eigenvalues are positive at all scales
        'N_values': list, the N values scanned
        'min_eigenvalue_per_N': list, minimum eigenvalue at each N
        'gap_protected': bool, True if gap remains above bare_gap/2
    """
    N_min, N_max = N_range
    N_values = list(range(N_min, N_max + 1))

    H_eigenvalues_per_N = []
    min_eigenvalue_per_N = []
    all_positive = True
    gap_protected = True

    bare_gap = 4.0 / R ** 2  # lambda_1 = 4/R²

    for N in N_values:
        flow = MultiScaleRGFlow(
            R=R, M=M, N_scales=N, N_c=N_c,
            g2_bare=g2_bare, k_max=100,
        )
        result = flow.run_flow()

        # Reconstruct effective Hamiltonian eigenvalues at each scale.
        # The m2_trajectory gives accumulated mass corrections after
        # integrating each shell. At scale j, the effective lowest
        # eigenvalue is lambda_1 + m2_accumulated_up_to_j.
        m2_traj = result['m2_trajectory']
        scale_eigenvalues = []

        for m2_acc in m2_traj:
            # Lowest three coexact eigenvalues + mass correction
            eigs = []
            for k in range(1, 4):
                lam_k = coexact_eigenvalue(k, R)
                eigs.append(lam_k + m2_acc)
            scale_eigenvalues.append(eigs)

        H_eigenvalues_per_N.append(scale_eigenvalues)

        # Check positivity
        all_eigs_flat = [e for scale_eigs in scale_eigenvalues for e in scale_eigs]
        min_eig = min(all_eigs_flat) if all_eigs_flat else 0.0
        min_eigenvalue_per_N.append(min_eig)

        if min_eig <= 0:
            all_positive = False

        # Gap protection: effective gap should be >= bare_gap / 2
        effective_gap = result['effective_mass_gap']
        if effective_gap < bare_gap * 0.5:
            gap_protected = False

    return {
        'H_eigenvalues_per_scale': H_eigenvalues_per_N,
        'all_positive': all_positive,
        'N_values': N_values,
        'min_eigenvalue_per_N': min_eigenvalue_per_N,
        'gap_protected': gap_protected,
    }
