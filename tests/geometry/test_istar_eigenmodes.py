"""
Tests for the I*-invariant eigenmode module on S3.

Verifies:
  1. istar_quaternions: 120 elements, unit norms, group closure
  2. quaternion_to_su2: unitarity, determinant, mapping properties
  3. wigner_D_matrix: unitarity, dimension, identity mapping
  4. istar_projector: Hermiticity, idempotence, correct rank from Molien
  5. invariant_eigenmodes: correct count m(k), orthonormality, I*-invariance

Mathematical ground truth:
  - I* (binary icosahedral group) has order 120, is a subgroup of SU(2)
  - Molien series: M(t) = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30))
  - m(k) = 0 for k = 1..11, m(0) = m(12) = m(20) = m(24) = m(30) = 1
  - Wigner D-matrices are unitary representations of SU(2)
"""

import pytest
import numpy as np
from yang_mills_s3.geometry.istar_eigenmodes import (
    istar_quaternions,
    quaternion_to_su2,
    wigner_D_matrix,
    istar_projector,
    invariant_eigenmodes,
)


# ==================================================================
# Helpers
# ==================================================================

def _quat_multiply(p, q):
    """
    Multiply two quaternions p, q in (w, x, y, z) format.

    pq = (p0q0 - p1q1 - p2q2 - p3q3,
          p0q1 + p1q0 + p2q3 - p3q2,
          p0q2 - p1q3 + p2q0 + p3q1,
          p0q3 + p1q2 - p2q1 + p3q0)
    """
    w = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    x = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    y = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    z = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return np.array([w, x, y, z])


def _is_in_group(q, elements, atol=1e-10):
    """Check if quaternion q matches any element in the group."""
    for e in elements:
        if np.allclose(q, e, atol=atol):
            return True
    return False


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="module")
def elements():
    """Compute I* quaternions once for the entire module."""
    return istar_quaternions()


# ==================================================================
# 1. istar_quaternions
# ==================================================================

class TestIstarQuaternions:
    """Tests for the binary icosahedral group I* quaternion generator."""

    def test_returns_120_elements(self, elements):
        """I* has exactly 120 elements (order of binary icosahedral group)."""
        assert elements.shape[0] == 120

    def test_shape_is_120x4(self, elements):
        """Each quaternion has 4 components (w, x, y, z)."""
        assert elements.shape == (120, 4)

    def test_all_unit_quaternions(self, elements):
        """Every element must have unit norm |q| = 1."""
        norms = np.linalg.norm(elements, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12,
                                   err_msg="Not all quaternions have unit norm")

    def test_identity_present(self, elements):
        """The identity quaternion (1, 0, 0, 0) must be in I*."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        assert _is_in_group(identity, elements), \
            "Identity quaternion (1,0,0,0) not found in I*"

    def test_negative_identity_present(self, elements):
        """The central element (-1, 0, 0, 0) must be in I*."""
        neg_id = np.array([-1.0, 0.0, 0.0, 0.0])
        assert _is_in_group(neg_id, elements), \
            "Central element (-1,0,0,0) not found in I*"

    def test_no_duplicates(self, elements):
        """All 120 elements must be distinct."""
        for i in range(120):
            for j in range(i + 1, 120):
                assert not np.allclose(elements[i], elements[j], atol=1e-10), \
                    f"Duplicate found: elements[{i}] == elements[{j}]"

    @pytest.mark.slow
    def test_closure_under_multiplication(self, elements):
        """
        Group closure: for all g, h in I*, the product g*h must also be in I*.

        This is the definitive test that we have an actual group.
        Uses a random sample of 500 pairs to keep runtime manageable.
        """
        rng = np.random.default_rng(42)
        n_pairs = 500
        indices = rng.integers(0, 120, size=(n_pairs, 2))

        for idx_g, idx_h in indices:
            g = elements[idx_g]
            h = elements[idx_h]
            gh = _quat_multiply(g, h)
            assert _is_in_group(gh, elements, atol=1e-8), \
                f"Product elements[{idx_g}]*elements[{idx_h}] = {gh} not in I*"

    def test_inverses_exist(self, elements):
        """
        Every element has an inverse in the group.

        For unit quaternions, q^{-1} = q* = (w, -x, -y, -z).
        """
        for i, g in enumerate(elements):
            g_inv = np.array([g[0], -g[1], -g[2], -g[3]])
            assert _is_in_group(g_inv, elements, atol=1e-10), \
                f"Inverse of elements[{i}] = {g_inv} not found in I*"

    def test_contains_axis_quaternions(self, elements):
        """The 8 axis quaternions (+-1, 0, 0, 0) and permutations are in I*."""
        for i in range(4):
            for sign in [1.0, -1.0]:
                q = np.zeros(4)
                q[i] = sign
                assert _is_in_group(q, elements, atol=1e-12), \
                    f"Axis quaternion {q} not in I*"

    def test_contains_half_integer_quaternions(self, elements):
        """The 16 half-integer quaternions (+-1/2, +-1/2, +-1/2, +-1/2) are in I*."""
        count = 0
        for s0 in [0.5, -0.5]:
            for s1 in [0.5, -0.5]:
                for s2 in [0.5, -0.5]:
                    for s3 in [0.5, -0.5]:
                        q = np.array([s0, s1, s2, s3])
                        assert _is_in_group(q, elements, atol=1e-12), \
                            f"Half-integer quaternion {q} not in I*"
                        count += 1
        assert count == 16


# ==================================================================
# 2. quaternion_to_su2
# ==================================================================

class TestQuaternionToSU2:
    """Tests for the quaternion-to-SU(2) matrix conversion."""

    def test_identity_quaternion(self):
        """The identity quaternion maps to the 2x2 identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        U = quaternion_to_su2(q)
        np.testing.assert_allclose(U, np.eye(2), atol=1e-14,
                                   err_msg="Identity quaternion should give I_2")

    def test_negative_identity_quaternion(self):
        """(-1, 0, 0, 0) maps to -I_2."""
        q = np.array([-1.0, 0.0, 0.0, 0.0])
        U = quaternion_to_su2(q)
        np.testing.assert_allclose(U, -np.eye(2), atol=1e-14,
                                   err_msg="(-1,0,0,0) should give -I_2")

    def test_returns_2x2_matrix(self):
        """Output shape must be (2, 2)."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        U = quaternion_to_su2(q)
        assert U.shape == (2, 2)

    def test_unitarity(self, elements):
        """U^dagger U = I for all I* elements (SU(2) matrices are unitary)."""
        for i, g in enumerate(elements):
            U = quaternion_to_su2(g)
            product = U.conj().T @ U
            np.testing.assert_allclose(product, np.eye(2), atol=1e-12,
                                       err_msg=f"U^dagger U != I for element {i}")

    def test_determinant_is_one(self, elements):
        """det(U) = 1 for all I* elements (special unitary)."""
        for i, g in enumerate(elements):
            U = quaternion_to_su2(g)
            det = np.linalg.det(U)
            np.testing.assert_allclose(det, 1.0, atol=1e-12,
                                       err_msg=f"det(U) != 1 for element {i}")

    def test_su2_form(self):
        """
        U = [[a, b], [-b*, a*]] for unit quaternion.

        Verify the matrix has the canonical SU(2) structure.
        """
        q = np.array([0.5, 0.5, 0.5, 0.5])
        U = quaternion_to_su2(q)
        a = U[0, 0]
        b = U[0, 1]
        np.testing.assert_allclose(U[1, 0], -np.conj(b), atol=1e-14,
                                   err_msg="U[1,0] != -b*")
        np.testing.assert_allclose(U[1, 1], np.conj(a), atol=1e-14,
                                   err_msg="U[1,1] != a*")

    def test_anti_homomorphism_property(self, elements):
        """
        quaternion_to_su2 is a group anti-homomorphism:
        U(g*h) = U(h) * U(g).

        This is the standard convention for the quaternion-to-SU(2) map
        q = w + xi + yj + zk -> [[w+iz, y+ix],[-y+ix, w-iz]].
        The reversal arises because quaternion multiplication and
        matrix multiplication use opposite conventions for the
        left/right action.

        Test on a random sample of pairs.
        """
        rng = np.random.default_rng(123)
        n_pairs = 50
        indices = rng.integers(0, 120, size=(n_pairs, 2))

        for idx_g, idx_h in indices:
            g = elements[idx_g]
            h = elements[idx_h]
            gh = _quat_multiply(g, h)

            U_g = quaternion_to_su2(g)
            U_h = quaternion_to_su2(h)
            U_gh = quaternion_to_su2(gh)

            np.testing.assert_allclose(
                U_gh, U_h @ U_g, atol=1e-10,
                err_msg=f"Anti-homomorphism failed: U(g*h) != U(h)*U(g) "
                        f"for elements[{idx_g}]*elements[{idx_h}]"
            )

    def test_i_quaternion_to_pauli(self):
        """(0, 1, 0, 0) -> i*sigma_x (up to convention)."""
        q = np.array([0.0, 1.0, 0.0, 0.0])
        U = quaternion_to_su2(q)
        # Should be unitary with det 1
        np.testing.assert_allclose(np.linalg.det(U), 1.0, atol=1e-14)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-14)


# ==================================================================
# 3. wigner_D_matrix
# ==================================================================

class TestWignerDMatrix:
    """Tests for the Wigner D-matrix computation."""

    def test_dimension_integer_spin(self):
        """D^l has dimension (2l+1) x (2l+1) for integer l."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for l in [0, 1, 2, 3, 5]:
            D = wigner_D_matrix(l, q)
            dim = 2 * l + 1
            assert D.shape == (dim, dim), \
                f"D^{l} has shape {D.shape}, expected ({dim}, {dim})"

    def test_dimension_half_integer_spin(self):
        """D^{k/2} has dimension (k+1) x (k+1) for half-integer spin."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for k in [1, 3, 5]:
            j = k / 2.0
            D = wigner_D_matrix(j, q)
            dim = k + 1
            assert D.shape == (dim, dim), \
                f"D^{j} has shape {D.shape}, expected ({dim}, {dim})"

    def test_identity_quaternion_gives_identity_matrix(self):
        """D^l(identity) = I_{2l+1}."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        for l in [0, 1, 2, 3, 5]:
            D = wigner_D_matrix(l, identity)
            dim = 2 * l + 1
            np.testing.assert_allclose(
                D, np.eye(dim), atol=1e-12,
                err_msg=f"D^{l}(identity) != I for l={l}"
            )

    def test_identity_quaternion_half_integer(self):
        """D^{k/2}(identity) = I_{k+1} for half-integer spin."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        for k in [1, 3, 5]:
            j = k / 2.0
            D = wigner_D_matrix(j, identity)
            dim = k + 1
            np.testing.assert_allclose(
                D, np.eye(dim), atol=1e-12,
                err_msg=f"D^{j}(identity) != I for k={k}"
            )

    def test_negative_identity_quaternion(self):
        """D^l(-I) = (-1)^{2l} I = I for integer l, -I for half-integer."""
        neg_id = np.array([-1.0, 0.0, 0.0, 0.0])
        # Integer l: D(-I) = (-1)^{2l} I = I
        for l in [0, 1, 2, 3]:
            D = wigner_D_matrix(l, neg_id)
            dim = 2 * l + 1
            np.testing.assert_allclose(
                D, np.eye(dim), atol=1e-10,
                err_msg=f"D^{l}(-I) != I for integer l={l}"
            )
        # Half-integer: D(-I) = -I
        for k in [1, 3, 5]:
            j = k / 2.0
            D = wigner_D_matrix(j, neg_id)
            dim = k + 1
            np.testing.assert_allclose(
                D, -np.eye(dim), atol=1e-10,
                err_msg=f"D^{j}(-I) != -I for half-integer j={j}"
            )

    def test_unitarity_integer_spin(self, elements):
        """D^l(g) is unitary: D^dagger D = I for integer l."""
        for l in [0, 1, 2, 3]:
            # Test a subset of elements
            for g in elements[:20]:
                D = wigner_D_matrix(l, g)
                product = D.conj().T @ D
                dim = 2 * l + 1
                np.testing.assert_allclose(
                    product, np.eye(dim), atol=1e-10,
                    err_msg=f"D^{l}(g) not unitary for g={g}"
                )

    @pytest.mark.slow
    def test_unitarity_all_elements_l2(self, elements):
        """D^2(g) is unitary for all 120 I* elements."""
        for i, g in enumerate(elements):
            D = wigner_D_matrix(2, g)
            product = D.conj().T @ D
            np.testing.assert_allclose(
                product, np.eye(5), atol=1e-10,
                err_msg=f"D^2(elements[{i}]) not unitary"
            )

    def test_unitarity_half_integer(self, elements):
        """D^{k/2}(g) is unitary for half-integer spins."""
        for k in [1, 3]:
            j = k / 2.0
            dim = k + 1
            for g in elements[:20]:
                D = wigner_D_matrix(j, g)
                product = D.conj().T @ D
                np.testing.assert_allclose(
                    product, np.eye(dim), atol=1e-10,
                    err_msg=f"D^{j}(g) not unitary"
                )

    def test_spin_0_is_trivial(self, elements):
        """D^0(g) = [[1]] for all g (trivial representation)."""
        for g in elements[:30]:
            D = wigner_D_matrix(0, g)
            assert D.shape == (1, 1)
            np.testing.assert_allclose(D[0, 0], 1.0, atol=1e-12)

    def test_representation_anti_homomorphism(self, elements):
        """
        D^l(g*h) = D^l(h) * D^l(g) (anti-homomorphism, consistent with
        the quaternion_to_su2 convention).

        The Wigner D-matrix inherits the anti-homomorphism from the
        quaternion-to-SU(2) map. This is the right-action convention:
        the representation acts on the right, so D(gh) = D(h)D(g).

        Test on a sample of pairs for l=1.
        """
        rng = np.random.default_rng(77)
        l = 1
        n_pairs = 30
        indices = rng.integers(0, 120, size=(n_pairs, 2))

        for idx_g, idx_h in indices:
            g = elements[idx_g]
            h = elements[idx_h]
            gh = _quat_multiply(g, h)

            D_g = wigner_D_matrix(l, g)
            D_h = wigner_D_matrix(l, h)
            D_gh = wigner_D_matrix(l, gh)

            np.testing.assert_allclose(
                D_gh, D_h @ D_g, atol=1e-9,
                err_msg=f"D^{l}(g*h) != D^{l}(h)*D^{l}(g) "
                        f"for elements[{idx_g}]*elements[{idx_h}]"
            )

    @pytest.mark.slow
    def test_representation_anti_homomorphism_l2(self, elements):
        """D^2(g*h) = D^2(h)*D^2(g) on a larger sample."""
        rng = np.random.default_rng(99)
        l = 2
        n_pairs = 100
        indices = rng.integers(0, 120, size=(n_pairs, 2))

        for idx_g, idx_h in indices:
            g = elements[idx_g]
            h = elements[idx_h]
            gh = _quat_multiply(g, h)

            D_g = wigner_D_matrix(l, g)
            D_h = wigner_D_matrix(l, h)
            D_gh = wigner_D_matrix(l, gh)

            np.testing.assert_allclose(
                D_gh, D_h @ D_g, atol=1e-9,
                err_msg=f"D^2(g*h) != D^2(h)*D^2(g)"
            )


# ==================================================================
# 4. istar_projector
# ==================================================================

class TestIstarProjector:
    """Tests for the I*-invariant projection operator P = (1/120) sum D^{k/2}(g)."""

    def test_hermitian_k0(self, elements):
        """P is Hermitian at k=0."""
        P = istar_projector(0, elements)
        np.testing.assert_allclose(P, P.conj().T, atol=1e-12,
                                   err_msg="P(k=0) not Hermitian")

    def test_hermitian_k12(self, elements):
        """P is Hermitian at k=12."""
        P = istar_projector(12, elements)
        np.testing.assert_allclose(P, P.conj().T, atol=1e-10,
                                   err_msg="P(k=12) not Hermitian")

    def test_idempotent_k0(self, elements):
        """P^2 = P at k=0 (projection property)."""
        P = istar_projector(0, elements)
        P2 = P @ P
        np.testing.assert_allclose(P2, P, atol=1e-12,
                                   err_msg="P(k=0) not idempotent")

    def test_idempotent_k12(self, elements):
        """P^2 = P at k=12 (projection property)."""
        P = istar_projector(12, elements)
        P2 = P @ P
        np.testing.assert_allclose(P2, P, atol=1e-10,
                                   err_msg="P(k=12) not idempotent")

    def test_idempotent_k20(self, elements):
        """P^2 = P at k=20."""
        P = istar_projector(20, elements)
        P2 = P @ P
        np.testing.assert_allclose(P2, P, atol=1e-9,
                                   err_msg="P(k=20) not idempotent")

    def test_trace_k0_is_m0(self, elements):
        """Tr(P) = m(0) = 1 at k=0."""
        P = istar_projector(0, elements)
        tr = np.real(np.trace(P))
        np.testing.assert_allclose(tr, 1.0, atol=1e-10,
                                   err_msg="Tr(P(k=0)) != 1")

    def test_trace_k1_through_k11_is_zero(self, elements):
        """Tr(P) = m(k) = 0 for k = 1..11 (Molien series)."""
        for k in range(1, 12):
            P = istar_projector(k, elements)
            tr = np.real(np.trace(P))
            np.testing.assert_allclose(
                tr, 0.0, atol=1e-8,
                err_msg=f"Tr(P(k={k})) = {tr}, expected 0"
            )

    def test_trace_k12_is_one(self, elements):
        """Tr(P) = m(12) = 1 (first nontrivial invariant scalar)."""
        P = istar_projector(12, elements)
        tr = np.real(np.trace(P))
        np.testing.assert_allclose(tr, 1.0, atol=1e-8,
                                   err_msg="Tr(P(k=12)) != 1")

    def test_trace_k20_is_one(self, elements):
        """Tr(P) = m(20) = 1."""
        P = istar_projector(20, elements)
        tr = np.real(np.trace(P))
        np.testing.assert_allclose(tr, 1.0, atol=1e-8,
                                   err_msg="Tr(P(k=20)) != 1")

    def test_trace_k24_is_one(self, elements):
        """Tr(P) = m(24) = 1."""
        P = istar_projector(24, elements)
        tr = np.real(np.trace(P))
        np.testing.assert_allclose(tr, 1.0, atol=1e-6,
                                   err_msg="Tr(P(k=24)) != 1")

    @pytest.mark.slow
    def test_trace_k30_is_one(self, elements):
        """Tr(P) = m(30) = 1."""
        P = istar_projector(30, elements)
        tr = np.real(np.trace(P))
        np.testing.assert_allclose(tr, 1.0, atol=1e-5,
                                   err_msg="Tr(P(k=30)) != 1")

    def test_eigenvalues_are_zero_or_one(self, elements):
        """All eigenvalues of the projector must be 0 or 1."""
        for k in [0, 2, 6, 12]:
            P = istar_projector(k, elements)
            eigenvalues = np.linalg.eigvalsh(P)
            for ev in eigenvalues:
                assert abs(ev) < 1e-8 or abs(ev - 1.0) < 1e-8, \
                    f"Eigenvalue {ev} at k={k} is neither 0 nor 1"

    def test_shape(self, elements):
        """P has shape (k+1, k+1)."""
        for k in [0, 1, 5, 12]:
            P = istar_projector(k, elements)
            assert P.shape == (k + 1, k + 1), \
                f"P(k={k}) has shape {P.shape}, expected ({k+1}, {k+1})"

    def test_computes_without_elements_arg(self):
        """Passing elements=None should auto-compute I* quaternions."""
        P = istar_projector(0, elements=None)
        assert P.shape == (1, 1)
        np.testing.assert_allclose(np.real(P[0, 0]), 1.0, atol=1e-12)


# ==================================================================
# 5. invariant_eigenmodes
# ==================================================================

class TestInvariantEigenmodes:
    """Tests for the I*-invariant eigenmode extraction."""

    def test_m0_is_one(self, elements):
        """m(0) = 1: the constant function is always I*-invariant."""
        modes, mk = invariant_eigenmodes(0, elements)
        assert mk == 1
        assert modes.shape == (1, 1)

    def test_m1_through_m11_are_zero(self, elements):
        """m(k) = 0 for k = 1..11 (all suppressed by I* symmetry)."""
        for k in range(1, 12):
            modes, mk = invariant_eigenmodes(k, elements)
            assert mk == 0, f"m({k}) = {mk}, expected 0"
            assert modes.shape[1] == 0, \
                f"modes at k={k} has {modes.shape[1]} columns, expected 0"

    def test_m12_is_one(self, elements):
        """m(12) = 1: first nontrivial I*-invariant scalar."""
        modes, mk = invariant_eigenmodes(12, elements)
        assert mk == 1
        assert modes.shape == (13, 1)

    def test_m20_is_one(self, elements):
        """m(20) = 1 from Molien series."""
        modes, mk = invariant_eigenmodes(20, elements)
        assert mk == 1
        assert modes.shape == (21, 1)

    def test_m24_is_one(self, elements):
        """m(24) = 1 from Molien series."""
        modes, mk = invariant_eigenmodes(24, elements)
        assert mk == 1
        assert modes.shape == (25, 1)

    @pytest.mark.slow
    def test_m30_is_one(self, elements):
        """m(30) = 1 from Molien series."""
        modes, mk = invariant_eigenmodes(30, elements)
        assert mk == 1

    def test_modes_are_orthonormal_k12(self, elements):
        """Invariant modes at k=12 are orthonormal (trivially 1 mode)."""
        modes, mk = invariant_eigenmodes(12, elements)
        if mk > 0:
            gram = modes.conj().T @ modes
            np.testing.assert_allclose(
                gram, np.eye(mk), atol=1e-10,
                err_msg="Invariant modes at k=12 not orthonormal"
            )

    def test_modes_are_orthonormal_k20(self, elements):
        """Invariant modes at k=20 are orthonormal."""
        modes, mk = invariant_eigenmodes(20, elements)
        if mk > 0:
            gram = modes.conj().T @ modes
            np.testing.assert_allclose(
                gram, np.eye(mk), atol=1e-10,
                err_msg="Invariant modes at k=20 not orthonormal"
            )

    def test_modes_are_istar_invariant_k12(self, elements):
        """
        Each invariant mode v at k=12 satisfies D^{k/2}(g) v = v for all g in I*.

        This is the core mathematical property: P projects onto the trivial
        representation of I*, so every mode in the image must be fixed by all
        group elements.
        """
        k = 12
        j = k / 2.0
        modes, mk = invariant_eigenmodes(k, elements)
        assert mk == 1

        for i, g in enumerate(elements):
            D = wigner_D_matrix(j, g)
            for col in range(mk):
                v = modes[:, col]
                Dv = D @ v
                np.testing.assert_allclose(
                    Dv, v, atol=1e-8,
                    err_msg=f"Mode {col} at k={k} not invariant under element {i}"
                )

    @pytest.mark.slow
    def test_modes_are_istar_invariant_k20(self, elements):
        """
        Each invariant mode v at k=20 satisfies D^{k/2}(g) v = v for all g in I*.
        """
        k = 20
        j = k / 2.0
        modes, mk = invariant_eigenmodes(k, elements)
        assert mk == 1

        for i, g in enumerate(elements):
            D = wigner_D_matrix(j, g)
            for col in range(mk):
                v = modes[:, col]
                Dv = D @ v
                np.testing.assert_allclose(
                    Dv, v, atol=1e-7,
                    err_msg=f"Mode {col} at k={k} not invariant under element {i}"
                )

    @pytest.mark.slow
    def test_modes_are_istar_invariant_k24(self, elements):
        """
        Each invariant mode v at k=24 satisfies D^{k/2}(g) v = v for all g in I*.
        """
        k = 24
        j = k / 2.0
        modes, mk = invariant_eigenmodes(k, elements)
        assert mk == 1

        for i, g in enumerate(elements):
            D = wigner_D_matrix(j, g)
            for col in range(mk):
                v = modes[:, col]
                Dv = D @ v
                np.testing.assert_allclose(
                    Dv, v, atol=1e-6,
                    err_msg=f"Mode {col} at k={k} not invariant under element {i}"
                )

    def test_mode_count_matches_projector_trace(self, elements):
        """m(k) from invariant_eigenmodes must match Tr(P) from istar_projector."""
        for k in [0, 1, 5, 12, 20]:
            P = istar_projector(k, elements)
            expected_mk = int(round(np.real(np.trace(P))))
            _, mk = invariant_eigenmodes(k, elements)
            assert mk == expected_mk, \
                f"k={k}: invariant_eigenmodes gives mk={mk}, but Tr(P)={expected_mk}"

    def test_zero_modes_shape(self, elements):
        """When m(k)=0, modes has shape (k+1, 0)."""
        for k in [1, 2, 3, 5, 7, 11]:
            modes, mk = invariant_eigenmodes(k, elements)
            assert mk == 0
            assert modes.shape == (k + 1, 0), \
                f"k={k}: zero-mode shape is {modes.shape}, expected ({k+1}, 0)"

    def test_mode_norms_are_one(self, elements):
        """Each invariant mode column has unit norm."""
        for k in [0, 12, 20, 24]:
            modes, mk = invariant_eigenmodes(k, elements)
            for col in range(mk):
                norm = np.linalg.norm(modes[:, col])
                np.testing.assert_allclose(
                    norm, 1.0, atol=1e-10,
                    err_msg=f"Mode {col} at k={k} has norm {norm}, expected 1"
                )

    def test_computes_without_elements_arg(self):
        """Passing elements=None should auto-compute I* quaternions."""
        modes, mk = invariant_eigenmodes(0, elements=None)
        assert mk == 1

    @pytest.mark.slow
    def test_molien_sequence_k0_to_k30(self, elements):
        """
        Verify full Molien sequence m(k) for k = 0..30.

        From M(t) = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30)):
        m(k) = 1 for k in {0, 12, 20, 24, 30}
        m(k) = 0 for all other k in [0, 30]
        """
        expected = {0: 1, 12: 1, 20: 1, 24: 1, 30: 1}
        for k in range(31):
            _, mk = invariant_eigenmodes(k, elements)
            exp = expected.get(k, 0)
            assert mk == exp, f"m({k}) = {mk}, expected {exp}"


# ==================================================================
# 6. Consistency across functions
# ==================================================================

class TestConsistency:
    """Cross-function consistency checks."""

    def test_projector_from_D_matrices_is_consistent(self, elements):
        """
        Verify that P = (1/120) sum D(g) computed by istar_projector
        satisfies P*v = v for each invariant mode v.
        """
        k = 12
        P = istar_projector(k, elements)
        modes, mk = invariant_eigenmodes(k, elements)
        assert mk >= 1

        for col in range(mk):
            v = modes[:, col]
            Pv = P @ v
            np.testing.assert_allclose(
                Pv, v, atol=1e-8,
                err_msg=f"P*v != v for invariant mode {col} at k={k}"
            )

    def test_projector_annihilates_non_invariant(self, elements):
        """
        For k where m(k)=0, P should annihilate all vectors: P*v ~ 0.
        """
        for k in [1, 2, 5]:
            P = istar_projector(k, elements)
            dim = k + 1
            # Test with random vectors
            rng = np.random.default_rng(42)
            for _ in range(5):
                v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
                v /= np.linalg.norm(v)
                Pv = P @ v
                np.testing.assert_allclose(
                    np.linalg.norm(Pv), 0.0, atol=1e-8,
                    err_msg=f"P did not annihilate vector at k={k}"
                )

    def test_su2_matrix_from_quaternion_matches_D_spin_half(self, elements):
        """
        The SU(2) matrix U(g) from quaternion_to_su2 should equal
        the Wigner D-matrix D^{1/2}(g) (the spin-1/2 representation).
        """
        for g in elements[:30]:
            U = quaternion_to_su2(g)
            D_half = wigner_D_matrix(0.5, g)
            np.testing.assert_allclose(
                D_half, U, atol=1e-12,
                err_msg=f"D^{{1/2}}(g) != quaternion_to_su2(g) for g={g}"
            )
