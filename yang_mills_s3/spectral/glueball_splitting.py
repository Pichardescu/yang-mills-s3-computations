"""
Non-perturbative glueball mass splitting in the 9-DOF effective theory on S^3/I*.

LABEL: NUMERICAL (truncated-basis diagonalization, not a proof)

Physics summary:
    By THEOREM 11.1, all single-particle coexact eigenmodes on S^3 have J >= 1.
    A J^PC = 0++ glueball must be a two-particle composite (THEOREM 11.1a).
    The lightest 0++ state comes from two k=1 modes (each J=1) coupled to
    total J=0, with free-theory threshold M = 4/R.

    The 9-DOF effective Hamiltonian (THEOREM 7.1) has:
        H_eff = T + V_2 + V_4
    where V_2 = (2/R^2)|a|^2 and V_4 = (g^2/2)[(Tr S)^2 - Tr(S^2)].

    The EXISTING code in src/proofs/glueball_spectrum.py treats the 3 singular
    values as independent DOFs and diagonalizes their coupled Hamiltonian.
    That gives the "single-particle" excitation gap in the gauge-invariant sector.

    THIS MODULE constructs the explicit two-particle Hamiltonian in the 0++
    channel by:

    1. Building the single-particle Hilbert space from the 9-DOF Hamiltonian
       restricted to the J=1 sector (the 3 degenerate k=1 modes, each with
       3 color components, reduced via SVD to 3 singular values).

    2. Constructing the 2-particle tensor product space with Bose symmetry
       (symmetric under particle exchange).

    3. Adding the V_4 interaction between the two particles. The key: V_4
       couples pairs of modes via sigma_i^2 * sigma_j^2 cross-terms. When
       two particles are present, V_4 generates an effective attraction in
       the J=0 channel that can bind them below the 2-particle threshold.

    4. Projecting onto J=0 and diagonalizing to find the 0++ bound state.

    The interaction mechanism:
        The full V_4 = (g^2/2) sum_{i<j} sigma_i^2 sigma_j^2 acts on the
        FULL 9-DOF configuration space. For a two-particle state, the 9
        DOFs split into two groups: particle A (modes 1-3 with amplitudes
        a_{i,alpha}) and particle B (modes 1-3 with amplitudes b_{i,alpha}).

        Actually, the correct framework is: the 9-DOF Hamiltonian already
        contains the FULL dynamics of 3 coupled oscillators. The "two-particle"
        nature of the 0++ state emerges from the SPECTRUM of H_eff: the gap
        between the ground state (vacuum) and the first SYMMETRIC excited
        state gives the 0++ mass.

        In the singular-value reduction (sigma_1, sigma_2, sigma_3), the
        0++ sector corresponds to states that are TOTALLY SYMMETRIC under
        permutations of the three sigma_i AND are even under parity
        (sigma_i -> -sigma_i for all i).

    This module therefore:
    (a) Works directly in the 9-DOF configuration space
    (b) Constructs the Hamiltonian with V_2 + V_4
    (c) Projects onto the 0++ sector via symmetry projection
    (d) Extracts the mass splitting between the 0++ ground state and
        the J=1 single-particle state

    The mass SPLITTING is:
        Delta M = E(0++) - E(vacuum) - [E(J=1 first) - E(vacuum)]
                = M(0++) - m_gap

    If Delta M > m_gap, the 0++ is above the two-particle threshold
    (unbound resonance). If Delta M < m_gap, it's a bound state.

References:
    - THEOREM 7.1: Effective Hamiltonian (papers/final/modules/07_effective_theory.md)
    - THEOREM 11.1: J >= 1 for single-particle modes
    - THEOREM 11.1a: 0++ is two-particle composite
    - Luscher (1982): Symmetry breaking aspects of the Hamiltonian
    - van Baal (1988): Gauge theory in a finite volume
    - Koller & van Baal (1988): Non-perturbative mass gap
"""

import numpy as np
from scipy.linalg import eigh


# ======================================================================
# Physical constants
# ======================================================================

HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm

# Lattice QCD reference values (SU(2) pure gauge)
LATTICE_0PP_MEV = 1730.0     # Morningstar & Peardon 1999
LATTICE_2PP_MEV = 2400.0     # Morningstar & Peardon 1999
LATTICE_0MP_MEV = 2590.0     # Morningstar & Peardon 1999


# ======================================================================
# 1. Single-particle 1D operators
# ======================================================================

def _build_1d_ops(n_basis, omega):
    """
    Build 1D harmonic oscillator operators in truncated basis.

    Parameters
    ----------
    n_basis : int
        Number of basis states per DOF.
    omega : float
        Harmonic frequency omega = 2/R.

    Returns
    -------
    dict with keys 'x', 'x2', 'x4', 'H0', 'p2' as (n_basis, n_basis) arrays.
    """
    length_scale = 1.0 / np.sqrt(2.0 * omega)

    # Position operator: x = sqrt(1/(2*omega)) * (a + a^dag)
    x = np.zeros((n_basis, n_basis))
    for n in range(n_basis - 1):
        x[n, n + 1] = np.sqrt(n + 1) * length_scale
        x[n + 1, n] = np.sqrt(n + 1) * length_scale

    x2 = x @ x
    x4 = x2 @ x2

    # Free Hamiltonian: H0 = omega * (n + 1/2)
    H0 = np.diag([omega * (n + 0.5) for n in range(n_basis)])

    return {'x': x, 'x2': x2, 'x4': x4, 'H0': H0}


def _kron3(A, B, C):
    """Kronecker product of three matrices: A (x) B (x) C."""
    return np.kron(np.kron(A, B), C)


# ======================================================================
# 2. Full 9-DOF Hamiltonian with J^PC projection
# ======================================================================

def build_H_full_9dof(R, g_squared, n_basis):
    """
    Build the FULL 9-DOF effective Hamiltonian.

    H = sum_{i,alpha} [omega*(n_{i,alpha} + 1/2)]
        + (g^2/2) * sum_{(i,alpha)<(j,beta)} quartic_coupling

    The quartic potential V_4 = (g^2/2)[(Tr S)^2 - Tr(S^2)] where
    S = M^T M and M_{i,alpha} = a_{i,alpha}.

    Expanding: V_4 = (g^2/2) * [sum_{alpha} (sum_i a_{i,alpha}^2)
                                 * sum_{beta!=alpha} (sum_j a_{j,beta}^2)
                                 - ...]

    In the 3-singular-value reduction (after gauge fixing), this becomes:
        V_4 = (g^2/2) * sum_{i<j} sigma_i^2 sigma_j^2

    We work in the singular-value (gauge-invariant) space.

    Parameters
    ----------
    R : float
        Radius of S^3.
    g_squared : float
        Yang-Mills coupling g^2.
    n_basis : int
        Basis states per DOF. Matrix dimension = n_basis^3.

    Returns
    -------
    dict with:
        'H' : ndarray of shape (n_basis^3, n_basis^3)
        'dim' : int, matrix dimension
        'omega' : float, harmonic frequency 2/R
        'R' : float
        'g_squared' : float
    """
    omega = 2.0 / R
    dim = n_basis ** 3

    ops = _build_1d_ops(n_basis, omega)
    I = np.eye(n_basis)

    # Harmonic part: sum_i H0_i
    H = (_kron3(ops['H0'], I, I)
         + _kron3(I, ops['H0'], I)
         + _kron3(I, I, ops['H0']))

    # Quartic part: V_4 = (g^2/2) sum_{i<j} sigma_i^2 sigma_j^2
    V4_coeff = 0.5 * g_squared
    H += V4_coeff * _kron3(ops['x2'], ops['x2'], I)    # sigma_1^2 sigma_2^2
    H += V4_coeff * _kron3(ops['x2'], I, ops['x2'])    # sigma_1^2 sigma_3^2
    H += V4_coeff * _kron3(I, ops['x2'], ops['x2'])    # sigma_2^2 sigma_3^2

    return {
        'H': H,
        'dim': dim,
        'omega': omega,
        'R': R,
        'g_squared': g_squared,
    }


# ======================================================================
# 3. S_3 permutation symmetry projector
# ======================================================================

def build_S3_projector(n_basis):
    """
    Build the projector onto the TOTALLY SYMMETRIC subspace of the
    3-fold tensor product, under permutations of the 3 singular values.

    The 0++ channel requires total symmetry under sigma permutations
    (Bose symmetry of identical DOFs) AND even parity.

    The S_3 symmetry projector is P_sym = (1/6) sum_{pi in S_3} U_pi
    where U_pi permutes the tensor factors.

    Parameters
    ----------
    n_basis : int
        Basis size per DOF.

    Returns
    -------
    ndarray of shape (dim, n_sym) where dim = n_basis^3 and n_sym is
        the number of symmetric states.
        Columns are orthonormal symmetric basis vectors.
    """
    dim = n_basis ** 3

    # Build the S_3 group action on the tensor product basis
    # Basis state |n1, n2, n3> has index n1*n_basis^2 + n2*n_basis + n3

    def basis_index(n1, n2, n3):
        return n1 * n_basis**2 + n2 * n_basis + n3

    # Build symmetrization projector
    P_sym = np.zeros((dim, dim))

    for n1 in range(n_basis):
        for n2 in range(n_basis):
            for n3 in range(n_basis):
                idx = basis_index(n1, n2, n3)
                # All 6 permutations of (n1, n2, n3)
                perms = [
                    (n1, n2, n3),  # identity
                    (n1, n3, n2),  # (23)
                    (n2, n1, n3),  # (12)
                    (n2, n3, n1),  # (123)
                    (n3, n1, n2),  # (132)
                    (n3, n2, n1),  # (13)
                ]
                for (p1, p2, p3) in perms:
                    pidx = basis_index(p1, p2, p3)
                    P_sym[idx, pidx] += 1.0 / 6.0

    # Diagonalize P_sym to find the symmetric subspace
    # Eigenvalues are 1 (symmetric) and 0 (non-symmetric)
    evals, evecs = eigh(P_sym)

    # Select eigenvectors with eigenvalue ~1
    sym_mask = np.abs(evals - 1.0) < 1e-8
    sym_basis = evecs[:, sym_mask]

    return sym_basis


def count_symmetric_states(n_basis):
    """
    Count the number of totally symmetric states of 3 bosons
    with quantum numbers 0, 1, ..., n_basis-1.

    Formula: C(n_basis + 2, 3) = n_basis*(n_basis+1)*(n_basis+2)/6

    Parameters
    ----------
    n_basis : int

    Returns
    -------
    int : number of symmetric states
    """
    return n_basis * (n_basis + 1) * (n_basis + 2) // 6


# ======================================================================
# 4. Parity projector (even under sigma_i -> -sigma_i)
# ======================================================================

def build_parity_projector(n_basis):
    """
    Build the projector onto EVEN parity states in the HO basis.

    In the harmonic oscillator basis, parity acts as:
        P |n> = (-1)^n |n>

    For the 3-DOF system:
        P |n1, n2, n3> = (-1)^{n1+n2+n3} |n1, n2, n3>

    The 0++ sector has P = +1, requiring n1+n2+n3 = even.

    Parameters
    ----------
    n_basis : int

    Returns
    -------
    ndarray of shape (dim,) : diagonal of parity operator (+1 or -1)
    """
    dim = n_basis ** 3
    parity = np.zeros(dim)

    for n1 in range(n_basis):
        for n2 in range(n_basis):
            for n3 in range(n_basis):
                idx = n1 * n_basis**2 + n2 * n_basis + n3
                parity[idx] = (-1.0) ** (n1 + n2 + n3)

    return parity


# ======================================================================
# 5. 0++ projected Hamiltonian
# ======================================================================

def build_H_0pp(R, g_squared, n_basis):
    """
    Build the Hamiltonian projected onto the 0++ sector.

    The 0++ quantum numbers require:
    1. Total symmetry under S_3 permutations of sigma_i (Bose symmetry)
    2. Even parity P = +1 (even total oscillator quantum number)
    3. Charge conjugation C = +1 (automatic for gauge-invariant states)

    The projection is done in two steps:
    (a) Build S_3-symmetric basis vectors
    (b) Restrict to even parity (n1+n2+n3 even) within the symmetric basis

    Parameters
    ----------
    R : float
        Radius of S^3.
    g_squared : float
        Coupling constant g^2.
    n_basis : int
        Basis size per DOF.

    Returns
    -------
    dict with:
        'H_0pp' : ndarray, Hamiltonian in 0++ subspace
        'dim_0pp' : int, dimension of 0++ subspace
        'dim_full' : int, dimension of full space
        'projector' : ndarray, columns = 0++ basis vectors
        'omega' : float
        'R' : float
        'g_squared' : float
    """
    data = build_H_full_9dof(R, g_squared, n_basis)
    H_full = data['H']
    omega = data['omega']

    # Step 1: Get S_3-symmetric basis
    sym_basis = build_S3_projector(n_basis)

    # Step 2: Restrict to even parity within symmetric basis
    parity_diag = build_parity_projector(n_basis)
    P_even = np.diag((1.0 + parity_diag) / 2.0)  # projector onto even parity

    # Apply parity projector to symmetric basis vectors
    sym_even_raw = P_even @ sym_basis

    # Orthonormalize (some symmetric vectors may have odd parity -> removed)
    # Use SVD to find the rank and orthonormal basis
    U, s, Vt = np.linalg.svd(sym_even_raw, full_matrices=False)
    # Keep vectors with significant singular values
    tol = 1e-10
    rank = np.sum(s > tol)
    projector = U[:, :rank]

    # Project Hamiltonian
    H_0pp = projector.T @ H_full @ projector

    return {
        'H_0pp': H_0pp,
        'dim_0pp': rank,
        'dim_full': data['dim'],
        'projector': projector,
        'omega': omega,
        'R': R,
        'g_squared': g_squared,
    }


# ======================================================================
# 6. J=1 (single-particle) projected Hamiltonian
# ======================================================================

def build_H_J1(R, g_squared, n_basis):
    """
    Build the Hamiltonian in the J=1 sector for single-particle excitations.

    The J=1 sector consists of states with one quantum excited in one DOF
    and symmetric under the OTHER two DOFs. More precisely, the J=1
    (vector) representation of S_3 appears in states transforming as the
    "standard representation" of S_3.

    For the purpose of identifying the single-particle gap, we can use the
    FULL spectrum and identify levels by their quantum numbers. The
    single-particle gap is omega (harmonic) + V_4 correction for the
    first J=1 state.

    In practice, for the mass splitting we need:
    - E_vacuum = ground state of H (in 0++ sector by definition)
    - E_1particle = first excited state in the J >= 1 sector
    - E_0pp = first excited state in the 0++ sector

    The mass splitting is:
        Delta_split = (E_0pp - E_vacuum) - 2*(E_1particle - E_vacuum)

    If Delta_split < 0: the 0++ is BOUND (below two-particle threshold)
    If Delta_split > 0: the 0++ is a RESONANCE (above threshold)
    If Delta_split = 0: at threshold (free theory limit)

    Parameters
    ----------
    R : float
    g_squared : float
    n_basis : int

    Returns
    -------
    dict with:
        'E_vacuum' : float, ground state energy
        'E_1particle' : float, first single-particle excitation
        'm_1particle' : float, single-particle mass = E_1 - E_0
        'm_1particle_MeV' : float, in MeV
    """
    data = build_H_full_9dof(R, g_squared, n_basis)
    H = data['H']
    omega = data['omega']

    # Full diagonalization
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)

    E_vacuum = eigenvalues[0]
    # The first excited state is a J=1 excitation (3-fold degenerate in free
    # theory, split by V_4). It corresponds to exciting one sigma by one quantum.
    E_1particle = eigenvalues[1]
    m_1 = E_1particle - E_vacuum

    return {
        'E_vacuum': E_vacuum,
        'E_1particle': E_1particle,
        'm_1particle': m_1,
        'm_1particle_MeV': m_1 * HBAR_C_MEV_FM,
        'omega': omega,
    }


# ======================================================================
# 7. Full mass splitting computation
# ======================================================================

def compute_mass_splitting(R, g_squared, n_basis):
    """
    Compute the non-perturbative 0++ glueball mass splitting.

    Returns the full mass splitting analysis:
    - Vacuum energy E_0
    - Single-particle gap m_1 (from 0++ ground state this is the gap)
    - 0++ excited state mass M_0pp
    - Two-particle threshold 2*m_1
    - Binding energy B = 2*m_1 - M_0pp (positive = bound)
    - Comparison with lattice QCD

    LABEL: NUMERICAL

    Parameters
    ----------
    R : float
        Radius of S^3 in fm.
    g_squared : float
        Yang-Mills coupling g^2.
    n_basis : int
        Basis size per DOF. Total full-space dimension = n_basis^3.

    Returns
    -------
    dict with mass splitting results.
    """
    omega = 2.0 / R

    # --- Full spectrum for single-particle gap ---
    full_data = build_H_full_9dof(R, g_squared, n_basis)
    H_full = full_data['H']
    evals_full = np.sort(np.linalg.eigvalsh(H_full))

    E_vacuum = evals_full[0]
    E_1st = evals_full[1]  # first excited (J=1 single-particle)

    m_1particle = E_1st - E_vacuum  # single-particle mass
    two_particle_threshold = 2.0 * m_1particle

    # --- 0++ spectrum ---
    opp_data = build_H_0pp(R, g_squared, n_basis)
    H_0pp = opp_data['H_0pp']
    dim_0pp = opp_data['dim_0pp']

    evals_0pp = np.sort(np.linalg.eigvalsh(H_0pp))

    E_0pp_ground = evals_0pp[0]  # this should match E_vacuum
    E_0pp_1st = evals_0pp[1] if dim_0pp > 1 else evals_0pp[0]
    E_0pp_2nd = evals_0pp[2] if dim_0pp > 2 else evals_0pp[1]

    # The 0++ mass is the gap to the first 0++ excited state
    M_0pp = E_0pp_1st - E_0pp_ground

    # Mass splitting relative to two-particle threshold
    binding_energy = two_particle_threshold - M_0pp
    # binding_energy > 0 means BOUND state (below threshold)
    # binding_energy < 0 means RESONANCE (above threshold)

    # Convert to MeV
    m_1_MeV = m_1particle * HBAR_C_MEV_FM
    M_0pp_MeV = M_0pp * HBAR_C_MEV_FM
    threshold_MeV = two_particle_threshold * HBAR_C_MEV_FM
    binding_MeV = binding_energy * HBAR_C_MEV_FM
    omega_MeV = omega * HBAR_C_MEV_FM

    # Enhancement factors
    enhancement_over_free = M_0pp / omega if omega > 0 else 0.0
    enhancement_over_m1 = M_0pp / m_1particle if m_1particle > 0 else 0.0

    # Comparison with lattice
    ratio_to_lattice = M_0pp_MeV / LATTICE_0PP_MEV

    # Higher 0++ excitations (radial excitations of the glueball)
    opp_spectrum_MeV = (evals_0pp[:min(6, dim_0pp)] - E_0pp_ground) * HBAR_C_MEV_FM

    return {
        # Energies (natural units, 1/fm)
        'E_vacuum': E_vacuum,
        'E_1particle': E_1st,
        'E_0pp_ground': E_0pp_ground,
        'E_0pp_1st_excited': E_0pp_1st,
        # Masses
        'm_1particle': m_1particle,
        'M_0pp': M_0pp,
        'two_particle_threshold': two_particle_threshold,
        'binding_energy': binding_energy,
        # MeV conversions
        'omega_MeV': omega_MeV,
        'm_1particle_MeV': m_1_MeV,
        'M_0pp_MeV': M_0pp_MeV,
        'threshold_MeV': threshold_MeV,
        'binding_MeV': binding_MeV,
        # Ratios
        'enhancement_over_free': enhancement_over_free,
        'enhancement_over_m1': enhancement_over_m1,
        'ratio_to_lattice': ratio_to_lattice,
        # 0++ spectrum
        'opp_spectrum_MeV': opp_spectrum_MeV,
        'dim_0pp': dim_0pp,
        'dim_full': full_data['dim'],
        # Parameters
        'R': R,
        'g_squared': g_squared,
        'n_basis': n_basis,
        'omega': omega,
        # Classification
        'is_bound': binding_energy > 0,
        'label': 'NUMERICAL',
    }


# ======================================================================
# 8. J^PC channel decomposition
# ======================================================================

def jpc_channel_masses(R, g_squared, n_basis):
    """
    Compute masses in different J^PC channels of the 3-DOF system.

    The 3-fold tensor product decomposes under S_3 as:
        V^{x3} = Sym^3(V) + Mixed + Anti

    The symmetric part gives 0++ (even parity) and 0-+ (odd parity).
    The mixed representation gives J=1 states.
    The antisymmetric part (if it exists) gives J=2 or higher.

    We project H onto each sector and diagonalize.

    Parameters
    ----------
    R : float
    g_squared : float
    n_basis : int

    Returns
    -------
    dict with masses in each accessible J^PC channel.
    """
    omega = 2.0 / R
    data = build_H_full_9dof(R, g_squared, n_basis)
    H_full = data['H']

    # Full spectrum
    evals_full = np.sort(np.linalg.eigvalsh(H_full))
    E_vacuum = evals_full[0]

    # --- 0++ sector (symmetric, even parity) ---
    opp_data = build_H_0pp(R, g_squared, n_basis)
    evals_0pp = np.sort(np.linalg.eigvalsh(opp_data['H_0pp']))
    masses_0pp = (evals_0pp - E_vacuum) * HBAR_C_MEV_FM

    # --- 0-+ sector (symmetric, odd parity) ---
    sym_basis = build_S3_projector(n_basis)
    parity_diag = build_parity_projector(n_basis)
    P_odd = np.diag((1.0 - parity_diag) / 2.0)
    sym_odd_raw = P_odd @ sym_basis

    U, s, Vt = np.linalg.svd(sym_odd_raw, full_matrices=False)
    tol = 1e-10
    rank_0mp = np.sum(s > tol)

    masses_0mp = np.array([])
    if rank_0mp > 0:
        proj_0mp = U[:, :rank_0mp]
        H_0mp = proj_0mp.T @ H_full @ proj_0mp
        evals_0mp = np.sort(np.linalg.eigvalsh(H_0mp))
        masses_0mp = (evals_0mp - E_vacuum) * HBAR_C_MEV_FM

    # --- Full spectrum masses ---
    masses_full = (evals_full[:min(15, len(evals_full))] - E_vacuum) * HBAR_C_MEV_FM

    # Mass ratios (vs ground state 0++)
    m_0pp_ground = masses_0pp[1] if len(masses_0pp) > 1 else 0.0  # first excited in 0++
    m_0mp_ground = masses_0mp[0] if len(masses_0mp) > 0 else 0.0  # first in 0-+

    return {
        'masses_0pp_MeV': masses_0pp,
        'masses_0mp_MeV': masses_0mp,
        'masses_full_MeV': masses_full,
        'M_0pp_MeV': m_0pp_ground,
        'M_0mp_MeV': m_0mp_ground,
        'ratio_0mp_0pp': m_0mp_ground / m_0pp_ground if m_0pp_ground > 0 else 0.0,
        'dim_0pp': opp_data['dim_0pp'],
        'dim_0mp': rank_0mp,
        'omega_MeV': omega * HBAR_C_MEV_FM,
        'R': R,
        'g_squared': g_squared,
        'label': 'NUMERICAL',
    }


# ======================================================================
# 9. Convergence study for the splitting
# ======================================================================

def convergence_splitting(R, g_squared, n_basis_values=None):
    """
    Study convergence of the 0++ mass splitting with basis size.

    LABEL: NUMERICAL

    Parameters
    ----------
    R : float
    g_squared : float
    n_basis_values : list of int, or None for defaults

    Returns
    -------
    dict with convergence data.
    """
    if n_basis_values is None:
        n_basis_values = [4, 6, 8, 10, 12, 14]

    results = []
    for n in n_basis_values:
        try:
            r = compute_mass_splitting(R, g_squared, n)
            results.append({
                'n_basis': n,
                'dim_full': n**3,
                'dim_0pp': r['dim_0pp'],
                'M_0pp_MeV': r['M_0pp_MeV'],
                'm_1_MeV': r['m_1particle_MeV'],
                'binding_MeV': r['binding_MeV'],
                'is_bound': r['is_bound'],
            })
        except (MemoryError, np.linalg.LinAlgError):
            break

    if len(results) < 2:
        converged = False
        rel_change = float('inf')
    else:
        last = results[-1]['M_0pp_MeV']
        prev = results[-2]['M_0pp_MeV']
        rel_change = abs(last - prev) / abs(prev) if prev != 0 else float('inf')
        converged = rel_change < 0.01

    return {
        'results': results,
        'n_basis_values': [r['n_basis'] for r in results],
        'M_0pp_values': [r['M_0pp_MeV'] for r in results],
        'binding_values': [r['binding_MeV'] for r in results],
        'relative_change_last': rel_change,
        'converged': converged,
        'R': R,
        'g_squared': g_squared,
        'label': 'NUMERICAL',
    }


# ======================================================================
# 10. Coupling dependence
# ======================================================================

def splitting_vs_coupling(R, g_squared_values=None, n_basis=10):
    """
    Compute the 0++ mass and splitting as a function of coupling g^2.

    At g^2 = 0: 0++ and two-particle threshold are degenerate (no binding).
    As g^2 increases: V_4 lifts the degeneracy.

    LABEL: NUMERICAL

    Parameters
    ----------
    R : float
    g_squared_values : array-like or None
    n_basis : int

    Returns
    -------
    dict with coupling scan results.
    """
    if g_squared_values is None:
        g_squared_values = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 6.28, 10.0, 20.0])

    g2_arr = np.asarray(g_squared_values, dtype=float)
    M_0pp = np.zeros(len(g2_arr))
    m_1 = np.zeros(len(g2_arr))
    thresholds = np.zeros(len(g2_arr))
    bindings = np.zeros(len(g2_arr))

    for i, g2 in enumerate(g2_arr):
        r = compute_mass_splitting(R, g2, n_basis)
        M_0pp[i] = r['M_0pp_MeV']
        m_1[i] = r['m_1particle_MeV']
        thresholds[i] = r['threshold_MeV']
        bindings[i] = r['binding_MeV']

    omega_MeV = (2.0 / R) * HBAR_C_MEV_FM

    return {
        'g_squared_values': g2_arr,
        'M_0pp_MeV': M_0pp,
        'm_1_MeV': m_1,
        'threshold_MeV': thresholds,
        'binding_MeV': bindings,
        'omega_MeV': omega_MeV,
        'R': R,
        'n_basis': n_basis,
        'label': 'NUMERICAL',
    }


# ======================================================================
# 11. Physical prediction at R = 2.2 fm
# ======================================================================

def physical_splitting_prediction(R_fm=2.2, g_squared=6.28, n_basis=14):
    """
    Compute the physical 0++ mass splitting prediction.

    Parameters:
    - R = 2.2 fm (from Lambda_QCD ~ 200 MeV)
    - g^2 = 6.28 (alpha_s = g^2/(4*pi) ~ 0.5)
    - n_basis = 14 (for convergence)

    LABEL: NUMERICAL

    Parameters
    ----------
    R_fm : float
    g_squared : float
    n_basis : int

    Returns
    -------
    dict with physical prediction and detailed comparison.
    """
    # Main computation
    result = compute_mass_splitting(R_fm, g_squared, n_basis)

    # J^PC channels
    channels = jpc_channel_masses(R_fm, g_squared, min(n_basis, 12))

    # Convergence check
    conv = convergence_splitting(R_fm, g_squared,
                                  n_basis_values=[6, 8, 10, 12, 14])

    # Build assessment
    lines = []
    lines.append(f"R = {R_fm} fm, g^2 = {g_squared}")
    lines.append(f"omega = 2/R = {result['omega']:.4f} 1/fm = {result['omega_MeV']:.1f} MeV")
    lines.append(f"Single-particle gap m_1 = {result['m_1particle_MeV']:.1f} MeV")
    lines.append(f"Two-particle threshold = {result['threshold_MeV']:.1f} MeV")
    lines.append(f"0++ mass M(0++) = {result['M_0pp_MeV']:.1f} MeV")
    lines.append(f"Binding energy = {result['binding_MeV']:.1f} MeV")
    lines.append(f"Enhancement over free: {result['enhancement_over_free']:.3f}x")
    lines.append(f"0++ / m_1 ratio: {result['enhancement_over_m1']:.3f}")

    if result['is_bound']:
        lines.append("STATUS: 0++ is a BOUND STATE (below two-particle threshold)")
    else:
        lines.append("STATUS: 0++ is a RESONANCE (above two-particle threshold)")

    lines.append(f"\nLattice 0++ = {LATTICE_0PP_MEV:.0f} MeV")
    lines.append(f"Model / Lattice = {result['ratio_to_lattice']:.3f}")
    lines.append(
        "NOTE: The 9-DOF truncation captures only k=1 modes. "
        "The full 0++ mass requires all modes. "
        "The deficit is expected from the k=1 truncation."
    )

    result['channels'] = channels
    result['convergence'] = conv
    result['assessment'] = '\n'.join(lines)

    return result


# ======================================================================
# 12. Summary
# ======================================================================

def splitting_summary(R_fm=2.2, g_squared=6.28, n_basis=14):
    """
    Generate a human-readable summary of the glueball mass splitting.

    Parameters
    ----------
    R_fm : float
    g_squared : float
    n_basis : int

    Returns
    -------
    str : formatted summary
    """
    pred = physical_splitting_prediction(R_fm, g_squared, n_basis)

    lines = [
        "=" * 72,
        "NON-PERTURBATIVE 0++ GLUEBALL MASS SPLITTING (9-DOF on S^3/I*)",
        "=" * 72,
        "",
        f"Parameters: R = {R_fm} fm, g^2 = {g_squared}, n_basis = {n_basis}",
        f"Full basis dim: {n_basis}^3 = {n_basis**3}",
        f"0++ subspace dim: {pred['dim_0pp']}",
        "",
        "--- Single-particle spectrum ---",
        f"  omega (free gap) = {pred['omega_MeV']:.1f} MeV",
        f"  m_1 (interacting) = {pred['m_1particle_MeV']:.1f} MeV",
        "",
        "--- 0++ composite spectrum ---",
        f"  Two-particle threshold = 2*m_1 = {pred['threshold_MeV']:.1f} MeV",
        f"  0++ mass M(0++) = {pred['M_0pp_MeV']:.1f} MeV",
        f"  Binding energy B = 2*m_1 - M(0++) = {pred['binding_MeV']:.1f} MeV",
        f"  Status: {'BOUND' if pred['is_bound'] else 'RESONANCE'}",
        "",
        "--- 0++ excitation spectrum (MeV above vacuum) ---",
    ]

    for i, E in enumerate(pred['opp_spectrum_MeV'][:6]):
        lines.append(f"  Level {i}: {E:.1f} MeV")

    lines.extend([
        "",
        "--- Enhancement ratios ---",
        f"  M(0++) / omega = {pred['enhancement_over_free']:.3f}",
        f"  M(0++) / m_1 = {pred['enhancement_over_m1']:.3f}",
        "",
        "--- Lattice QCD comparison ---",
        f"  Lattice 0++ (SU(2)):   {LATTICE_0PP_MEV:.0f} MeV",
        f"  Model 0++:             {pred['M_0pp_MeV']:.1f} MeV",
        f"  Model / Lattice:       {pred['ratio_to_lattice']:.3f}",
    ])

    if 'channels' in pred and pred['channels']['M_0mp_MeV'] > 0:
        ch = pred['channels']
        lines.extend([
            "",
            "--- J^PC channel comparison ---",
            f"  0++ mass: {ch['M_0pp_MeV']:.1f} MeV",
            f"  0-+ mass: {ch['M_0mp_MeV']:.1f} MeV",
            f"  Ratio 0-+/0++: {ch['ratio_0mp_0pp']:.3f}",
            f"  Lattice ratio: {LATTICE_0MP_MEV/LATTICE_0PP_MEV:.3f}",
        ])

    conv = pred.get('convergence', {})
    if conv.get('results'):
        lines.extend([
            "",
            "--- Convergence ---",
        ])
        for r in conv['results']:
            lines.append(
                f"  n_basis={r['n_basis']:2d} (dim_0pp={r['dim_0pp']:4d}) "
                f"M(0++)={r['M_0pp_MeV']:.1f} MeV  "
                f"B={r['binding_MeV']:.1f} MeV"
            )
        lines.append(f"  Converged (1%): {conv.get('converged', False)}")

    lines.extend([
        "",
        "LABEL: NUMERICAL (truncated-basis diagonalization of 9-DOF model)",
        "THEOREM 11.1: J >= 1 for single-particle modes",
        "THEOREM 11.1a: 0++ is two-particle composite with threshold 4/R",
        "=" * 72,
    ])

    return "\n".join(lines)
