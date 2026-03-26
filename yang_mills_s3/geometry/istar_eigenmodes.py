"""
Explicit I*-invariant eigenmodes on S3 via Wigner D-matrix projection.

Constructs the m(k) I*-invariant scalar harmonics at each level k by
projecting the (k+1)^2-dimensional representation V_k onto the trivial
representation of the binary icosahedral group I* (120 elements).

The projection operator is:
    P_trivial = (1/120) * sum_{g in I*} D^k(g)

where D^k(g) is the (k+1) x (k+1) Wigner D-matrix for the SU(2) element g.

The eigenmodes are then evaluated at specific observer positions on S3/I*
to compute position-dependent CMB angular power spectra.

References:
    - Wigner (1931): Group Theory
    - Lehoucq, Weeks, Uzan, Gausmann, Luminet, CQG 19, 4683 (2002)
    - Aurich, Lustig, Steiner, CQG 22, 2061 (2005)
"""

import numpy as np
from scipy.special import factorial
from functools import lru_cache


def istar_quaternions():
    """
    Return the 120 unit quaternions of the binary icosahedral group I*.

    These are the vertices of the 600-cell. Format: array of shape (120, 4)
    with columns (w, x, y, z) representing q = w + xi + yj + zk.

    The 120 elements decompose as:
        8  axis-permutation quaternions (+-1, 0, 0, 0) and permutations
       16  half-integer quaternions (+-1/2, +-1/2, +-1/2, +-1/2)
       96  golden-ratio quaternions (even permutations of (0, +-1, +-phi, +-1/phi)/2)
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    iphi = phi - 1  # 1/phi = phi - 1

    verts = []

    # 8 axis quaternions: all permutations of (+-1, 0, 0, 0)
    for i in range(4):
        for sign in [1, -1]:
            q = [0, 0, 0, 0]
            q[i] = sign
            verts.append(q)

    # 16 half-integer quaternions: (+-1/2, +-1/2, +-1/2, +-1/2)
    for s0 in [0.5, -0.5]:
        for s1 in [0.5, -0.5]:
            for s2 in [0.5, -0.5]:
                for s3 in [0.5, -0.5]:
                    verts.append([s0, s1, s2, s3])

    # 96 golden-ratio quaternions: even permutations of (0, +-1, +-phi, +-1/phi)/2
    base_vals = [0, 1, phi, iphi]
    # Even permutations of 4 elements (12 total)
    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]
    for perm in even_perms:
        base = [base_vals[perm[i]] for i in range(4)]
        # Apply all sign combinations to the nonzero entries
        nonzero_idx = [i for i in range(4) if base[i] != 0]
        n_nonzero = len(nonzero_idx)
        for signs in range(2 ** n_nonzero):
            q = list(base)
            for bit, idx in enumerate(nonzero_idx):
                if signs & (1 << bit):
                    q[idx] = -q[idx]
            q = [x / 2 for x in q]
            verts.append(q)

    verts = np.array(verts)

    # Remove duplicates (some golden-ratio quats may repeat)
    unique = []
    for v in verts:
        is_dup = False
        for u in unique:
            if np.allclose(v, u, atol=1e-12):
                is_dup = True
                break
        if not is_dup:
            unique.append(v)

    result = np.array(unique)
    assert len(result) == 120, f"Expected 120 elements, got {len(result)}"

    # Verify unit quaternions
    norms = np.sqrt(np.sum(result ** 2, axis=1))
    assert np.allclose(norms, 1.0, atol=1e-12), "Not all unit quaternions"

    return result


def quaternion_to_su2(q):
    """
    Convert a unit quaternion (w, x, y, z) to a 2x2 SU(2) matrix.

    q = w + xi + yj + zk  ->  U = [[w + iz, y + ix], [-y + ix, w - iz]]
                                 = [[a, b], [-b*, a*]]
    where a = w + iz, b = y + ix.
    """
    w, x, y, z = q
    a = complex(w, z)
    b = complex(y, x)
    return np.array([[a, b], [-np.conj(b), np.conj(a)]])


@lru_cache(maxsize=256)
def _wigner_d_small(j, m1, m2, beta):
    """
    Wigner small d-matrix element d^j_{m1,m2}(beta).

    Uses the explicit sum formula (Wigner 1931).
    j can be integer or half-integer.
    """
    val = 0.0
    s_min = max(0, int(m2 - m1))
    s_max = min(int(j + m2), int(j - m1))

    for s in range(s_min, s_max + 1):
        num = factorial(j + m2, exact=True) * factorial(j - m2, exact=True) * \
              factorial(j + m1, exact=True) * factorial(j - m1, exact=True)
        den = factorial(j + m2 - s, exact=True) * factorial(s, exact=True) * \
              factorial(j - m1 - s, exact=True) * factorial(s + m1 - m2, exact=True)
        sign = (-1) ** (s + m1 - m2)
        cos_term = np.cos(beta / 2) ** (2 * j - 2 * s - m1 + m2)
        sin_term = np.sin(beta / 2) ** (2 * s + m1 - m2)
        val += sign * np.sqrt(float(num)) / float(den) * cos_term * sin_term

    return val


def wigner_D_matrix(l, q):
    """
    Compute the (2l+1) x (2l+1) Wigner D-matrix D^l(g) for SU(2) element g
    represented as a unit quaternion q = (w, x, y, z).

    For integer l, this is the representation matrix in V_l.
    For the scalar harmonics on S3 at level k, use l = k (dimension k+1).

    Wait -- on S3, the scalar harmonics at level k transform under SU(2)_R
    in the representation of spin j = k/2 (dimension k+1). So we need
    l = k/2 (half-integer for odd k).

    Actually, for the right-action of I* on S3 = SU(2), the scalar harmonics
    Q_{klm} at level k transform as the representation D^{k/2} under SU(2)_R.
    So the Wigner matrix we need is D^{k/2}_{m'm}(g) with dimension (k+1).

    Parameters
    ----------
    l : float
        Spin quantum number (integer or half-integer). Dimension = 2l+1.
    q : array-like
        Unit quaternion (w, x, y, z).

    Returns
    -------
    D : ndarray, shape (2l+1, 2l+1)
        Wigner D-matrix.
    """
    # Convert quaternion to Euler angles (ZYZ convention)
    # q = (w, x, y, z) -> U = [[a, b], [-b*, a*]]
    w, x, y, z = q
    a = complex(w, z)   # = cos(beta/2) * exp(i*(alpha+gamma)/2)
    b = complex(y, x)   # = sin(beta/2) * exp(i*(gamma-alpha)/2) ... actually this needs care

    # Extract Euler angles from SU(2) matrix
    # |a|^2 + |b|^2 = 1
    # a = cos(beta/2) * exp(i*(alpha+gamma)/2)
    # b = sin(beta/2) * exp(i*(gamma-alpha)/2)  ... CAREFUL with conventions

    # Actually, it's simpler to compute D directly from the SU(2) matrix.
    # For spin-j representation, D^j_{m'm}(U) can be computed from U = [[a,b],[-b*,a*]]
    # using the formula involving binomial sums.

    dim = int(2 * l + 1)
    D = np.zeros((dim, dim), dtype=complex)

    for m1_idx in range(dim):
        m1 = l - m1_idx  # m1 = l, l-1, ..., -l
        for m2_idx in range(dim):
            m2 = l - m2_idx

            # D^j_{m1,m2}(U) = sum_s C(j,m1,m2,s) * a^p * conj(a)^q * b^r * (-conj(b))^t
            # where the sum is over s and the exponents depend on j, m1, m2, s
            val = 0.0 + 0.0j
            s_min = int(max(0, m2 - m1))
            s_max = int(min(l + m2, l - m1))

            for s in range(s_min, s_max + 1):
                p = int(l + m2 - s)
                q = int(l - m1 - s)
                r = int(s + m1 - m2)
                t = s

                num = factorial(int(l + m1), exact=True) * \
                      factorial(int(l - m1), exact=True) * \
                      factorial(int(l + m2), exact=True) * \
                      factorial(int(l - m2), exact=True)
                den = factorial(p, exact=True) * factorial(q, exact=True) * \
                      factorial(r, exact=True) * factorial(t, exact=True)

                coeff = np.sqrt(float(num)) / float(den)
                val += coeff * (a ** p) * (np.conj(a) ** q) * \
                       (b ** r) * ((-np.conj(b)) ** t)

            D[m1_idx, m2_idx] = val

    return D


def wigner_D_row(l, q, m1_idx=0):
    """
    Compute a single row of the Wigner D-matrix D^l(g).

    This is O(dim) instead of O(dim^2) for the full matrix, giving a
    significant speedup when only the observer's angular contribution
    (typically row 0, i.e. m1 = l) is needed.

    Parameters
    ----------
    l : float
        Spin quantum number (integer or half-integer). Dimension = 2l+1.
    q : array-like
        Unit quaternion (w, x, y, z).
    m1_idx : int
        Row index to compute. Default 0 (m1 = l).

    Returns
    -------
    row : ndarray, shape (2l+1,), complex
        The m1_idx-th row of the Wigner D-matrix.
    """
    w, x, y, z = q
    a = complex(w, z)
    b = complex(y, x)

    dim = int(2 * l + 1)
    row = np.zeros(dim, dtype=complex)

    m1 = l - m1_idx

    for m2_idx in range(dim):
        m2 = l - m2_idx

        val = 0.0 + 0.0j
        s_min = int(max(0, m2 - m1))
        s_max = int(min(l + m2, l - m1))

        for s in range(s_min, s_max + 1):
            p = int(l + m2 - s)
            qq = int(l - m1 - s)
            r = int(s + m1 - m2)
            t = s

            num = factorial(int(l + m1), exact=True) * \
                  factorial(int(l - m1), exact=True) * \
                  factorial(int(l + m2), exact=True) * \
                  factorial(int(l - m2), exact=True)
            den = factorial(p, exact=True) * factorial(qq, exact=True) * \
                  factorial(r, exact=True) * factorial(t, exact=True)

            coeff = np.sqrt(float(num)) / float(den)
            val += coeff * (a ** p) * (np.conj(a) ** qq) * \
                   (b ** r) * ((-np.conj(b)) ** t)

        row[m2_idx] = val

    return row


def istar_projector(k, elements=None):
    """
    Compute the projection matrix onto the I*-invariant subspace of V_{k/2}.

    Parameters
    ----------
    k : int
        Level of scalar harmonics on S3. Dimension of V_{k/2} is k+1.
    elements : ndarray or None
        The 120 I* quaternions. If None, computed.

    Returns
    -------
    P : ndarray, shape (k+1, k+1)
        Projection matrix (Hermitian, P^2 = P, rank = m(k)).
    """
    if elements is None:
        elements = istar_quaternions()

    j = k / 2.0  # spin
    dim = k + 1

    P = np.zeros((dim, dim), dtype=complex)
    for g in elements:
        D = wigner_D_matrix(j, g)
        P += D

    P /= len(elements)  # = 120

    return P


def invariant_eigenmodes(k, elements=None):
    """
    Compute the m(k) I*-invariant eigenmodes at level k.

    Returns the invariant vectors as columns of a matrix.

    Parameters
    ----------
    k : int
        Level of scalar harmonics.
    elements : ndarray or None
        The 120 I* quaternions.

    Returns
    -------
    modes : ndarray, shape (k+1, m(k))
        Each column is an I*-invariant vector in V_{k/2}.
    mk : int
        Number of invariant modes (= m(k) from Molien).
    """
    P = istar_projector(k, elements)

    # Eigendecompose P: eigenvalues are 0 or 1 (it's a projector)
    eigenvalues, eigenvectors = np.linalg.eigh(P)

    # Select eigenvectors with eigenvalue ~ 1
    tol = 1e-6
    mask = np.abs(eigenvalues - 1.0) < tol
    mk = np.sum(mask)

    if mk == 0:
        return np.zeros((k + 1, 0)), 0

    modes = eigenvectors[:, mask]
    return modes, int(mk)
