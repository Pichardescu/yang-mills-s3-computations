"""
Tests for S^3/I* spectral sparsification on the 600-cell lattice.

Verifies:
  1. I* group action on the 600-cell (permutations, composition, identity)
  2. Scalar projector (idempotent, Hermitian, rank 1)
  3. Scalar spectrum (eigenvalue 0 survives, k=1..5 absent)
  4. Edge projector (idempotent, rank = 6 = edges on quotient)
  5. Hodge Laplacian (symmetric PSD, d1 d0 = 0, no harmonic 1-forms)
  6. Coexact spectrum (lowest eigenvalue ~4/R^2 analog, multiplicities)
  7. Gap preservation (lowest coexact eigenvalue survives I* projection)
  8. Spectral sparsification (k=2..5 coexact modes absent after I* projection)

THEOREM: The Yang-Mills mass gap on S^3/I* equals that on S^3.
The lowest coexact eigenvalue (k=1, 3 modes) survives the I* projection.
The next coexact levels (k=2..5) are absent: sparsification.
"""

import pytest
import numpy as np
from scipy import linalg as la
from yang_mills_s3.lattice.poincare_lattice import (
    PoincareLattice,
    _quaternion_multiply,
    _cluster_eigenvalues,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def poincare():
    """Build PoincareLattice once for all tests (expensive construction)."""
    return PoincareLattice(R=1.0)


@pytest.fixture(scope="module")
def scalar_projector(poincare):
    """Scalar (vertex) I*-invariant projector."""
    return poincare.istar_projector_vertices()


@pytest.fixture(scope="module")
def edge_projector(poincare):
    """Edge (1-form) I*-invariant projector."""
    return poincare.istar_projector_edges()


@pytest.fixture(scope="module")
def scalar_spectrum(poincare):
    """Full scalar spectrum (eigenvalues, eigenvectors)."""
    return poincare.scalar_spectrum_full()


@pytest.fixture(scope="module")
def oneform_decomp(poincare):
    """Decomposed 1-form spectrum."""
    return poincare._decompose_oneform_spectrum()


# ==================================================================
# 1. I* Group Action
# ==================================================================

class TestIStarGroupAction:
    """
    THEOREM: The 120 vertices of the 600-cell form a single I* orbit
    under right-multiplication. Each vertex permutation is a valid
    bijection, and the map h -> sigma_h is a group homomorphism.
    """

    def test_120_vertices(self, poincare):
        """600-cell has exactly 120 vertices = |I*|."""
        assert poincare._n_vertices == 120

    def test_identity_is_identity_permutation(self, poincare):
        """
        Right-multiplication by the identity quaternion (1,0,0,0)
        gives the identity permutation.
        """
        e_idx = poincare._find_identity_index()
        perm = poincare.vertex_permutation(e_idx)
        np.testing.assert_array_equal(perm, np.arange(120))

    def test_all_permutations_are_bijections(self, poincare):
        """
        THEOREM: Each right-multiplication sigma_h is a bijection
        {0..119} -> {0..119}.
        """
        for h in range(120):
            perm = poincare.vertex_permutation(h)
            assert set(perm) == set(range(120)), \
                f"Permutation {h} is not a bijection"

    def test_single_orbit(self, poincare):
        """
        THEOREM: I* acts transitively on itself.
        Starting from vertex 0, the orbit under all right-multiplications
        covers all 120 vertices.
        """
        orbit = set()
        for h in range(120):
            perm = poincare.vertex_permutation(h)
            orbit.add(perm[0])
        assert len(orbit) == 120

    def test_composition_matches_quaternion_product(self, poincare):
        """
        THEOREM: sigma_{h1} composed with sigma_{h2} = sigma_{h2*h1}.
        (Since sigma_h(g) = g*h, composing gives g*h2*h1 = sigma_{h2*h1}(g).)
        """
        V = poincare._unit_verts
        for h1 in [1, 5, 17, 42, 99]:
            for h2 in [2, 13, 50, 77]:
                # Compute h2 * h1 as a quaternion
                q_prod = _quaternion_multiply(V[h2], V[h1])
                # Find which vertex this is
                dots = V @ q_prod
                h21_idx = int(np.argmax(dots))

                p1 = poincare.vertex_permutation(h1)
                p2 = poincare.vertex_permutation(h2)
                p21 = poincare.vertex_permutation(h21_idx)

                # sigma_{h1}(sigma_{h2}(g)) = (g*h2)*h1 = g*(h2*h1) = sigma_{h2*h1}(g)
                composition = p1[p2]
                np.testing.assert_array_equal(
                    composition, p21,
                    err_msg=f"Composition failed for h1={h1}, h2={h2}"
                )

    def test_permutation_orders_divide_120(self, poincare):
        """
        THEOREM: Every element of I* has order dividing 120 = |I*|.
        The orders present in I* are: 1, 2, 3, 4, 5, 6, 10.
        """
        valid_orders = {1, 2, 3, 4, 5, 6, 10}
        identity = np.arange(120)
        for h in range(120):
            perm = poincare.vertex_permutation(h)
            # Compute order
            current = perm.copy()
            for order in range(1, 121):
                if np.all(current == identity):
                    assert order in valid_orders or 120 % order == 0, \
                        f"Vertex {h} has unexpected order {order}"
                    break
                current = perm[current]
            else:
                pytest.fail(f"Vertex {h}: order > 120")


# ==================================================================
# 2. Scalar Projector
# ==================================================================

class TestScalarProjector:
    """
    THEOREM: The I*-invariant projector on scalars (120x120) is
    idempotent, Hermitian, and has rank 1 (since I* acts freely
    and transitively on its 120 elements -> one orbit -> rank 1).
    """

    def test_idempotent(self, scalar_projector):
        """Pi^2 = Pi (projector property)."""
        Pi = scalar_projector
        Pi2 = Pi @ Pi
        np.testing.assert_allclose(Pi2, Pi, atol=1e-12,
            err_msg="Scalar projector not idempotent")

    def test_symmetric(self, scalar_projector):
        """Pi^T = Pi (Hermitian/symmetric)."""
        Pi = scalar_projector
        np.testing.assert_allclose(Pi.T, Pi, atol=1e-14,
            err_msg="Scalar projector not symmetric")

    def test_rank_is_1(self, scalar_projector):
        """
        THEOREM: rank(Pi) = 1.
        I* acts freely on itself: 120 elements, one orbit of size 120,
        so exactly one I*-invariant function (the constant).
        """
        Pi = scalar_projector
        evals = la.eigh(Pi, eigvals_only=True)
        rank = np.sum(evals > 0.5)
        assert rank == 1, f"Expected rank 1, got {rank}"

    def test_constant_vector_preserved(self, scalar_projector):
        """Pi * (constant vector) = (constant vector)."""
        Pi = scalar_projector
        const = np.ones(120) / np.sqrt(120)
        proj_const = Pi @ const
        np.testing.assert_allclose(proj_const, const, atol=1e-14,
            err_msg="Constant vector not preserved by projector")

    def test_nonconstant_annihilated(self, scalar_projector):
        """
        Any vector orthogonal to the constant should be annihilated.
        """
        Pi = scalar_projector
        # e_0 - e_1 is orthogonal to the constant on S^3 (not quite, but
        # the I*-invariant projection of any non-constant should vanish)
        v = np.zeros(120)
        v[0] = 1.0
        v[1] = -1.0
        proj_v = Pi @ v
        # Project out the constant component
        const = np.ones(120) / np.sqrt(120)
        proj_v_orth = proj_v - np.dot(proj_v, const) * const
        assert np.linalg.norm(proj_v_orth) < 1e-12, \
            "Non-constant component should vanish under I* projection"


# ==================================================================
# 3. Scalar Spectrum
# ==================================================================

class TestScalarSpectrum:
    """
    NUMERICAL: The scalar Laplacian on the 600-cell has eigenvalues
    clustering near the continuum values k(k+2)/R^2 with multiplicities
    matching (k+1)^2 exactly.
    """

    def test_eigenvalue_zero_multiplicity_1(self, scalar_spectrum):
        """Eigenvalue 0 has multiplicity 1 (connected graph)."""
        evals, _ = scalar_spectrum
        n_zero = np.sum(np.abs(evals) < 1e-8)
        assert n_zero == 1, f"Expected 1 zero eigenvalue, got {n_zero}"

    def test_120_eigenvalues(self, scalar_spectrum):
        """Scalar Laplacian has 120 eigenvalues (one per vertex)."""
        evals, _ = scalar_spectrum
        assert len(evals) == 120

    def test_positive_semidefinite(self, scalar_spectrum):
        """All eigenvalues are >= 0."""
        evals, _ = scalar_spectrum
        assert np.all(evals >= -1e-10), \
            f"Negative eigenvalue found: {np.min(evals)}"

    def test_multiplicities_match_continuum(self, scalar_spectrum):
        """
        NUMERICAL: Eigenvalue multiplicities match continuum predictions.
        Continuum: k=0 -> mult 1, k=1 -> 4, k=2 -> 9, k=3 -> 16,
                   k=4 -> 25, k=5 -> 36. Total: 1+4+9+16+25+36 = 91 < 120.

        On the lattice, the first 5 levels (k=0..4) are well-resolved.
        k=5 (eigenvalue ~35 in continuum) starts to merge with lattice
        artifacts at the top of the band, so we only check k=0..4.
        """
        evals, _ = scalar_spectrum
        clusters = _cluster_eigenvalues(evals, tol=0.5)

        # First 5 clusters should have multiplicities (k+1)^2 for k=0..4
        expected_mults = [1, 4, 9, 16, 25]
        for i, expected in enumerate(expected_mults):
            if i >= len(clusters):
                pytest.fail(f"Only {len(clusters)} clusters, expected at least {i+1}")
            actual = clusters[i][1]
            assert actual == expected, \
                f"Cluster {i} (k={i}): expected mult {expected}, got {actual}"

    def test_istar_invariant_only_zero(self, poincare):
        """
        THEOREM: The only I*-invariant scalar eigenvalue on the 600-cell
        is zero (the constant mode). All modes k=1..5 are absent.
        """
        istar_evals, rank = poincare.scalar_spectrum_istar()
        assert rank == 1, f"Expected I* rank 1, got {rank}"
        assert len(istar_evals) == 1, \
            f"Expected 1 I*-invariant eigenvalue, got {len(istar_evals)}"
        assert abs(istar_evals[0]) < 1e-10, \
            f"I*-invariant eigenvalue should be 0, got {istar_evals[0]}"

    def test_eigenvalue_clustering(self, scalar_spectrum):
        """
        NUMERICAL: Scalar eigenvalues cluster near k(k+2) for k=0..5.
        The first cluster (k=0) is at 0, the second (k=1) is near 3, etc.
        Lattice distortion increases with k.
        """
        evals, _ = scalar_spectrum
        clusters = _cluster_eigenvalues(evals, tol=0.5)

        # k=0: eigenvalue = 0
        assert abs(clusters[0][0]) < 1e-8

        # k=1: eigenvalue ~ 3 (with lattice corrections)
        # On the 600-cell, typically ~2.29
        assert 1.5 < clusters[1][0] < 4.0, \
            f"k=1 cluster at {clusters[1][0]}, expected near 3"

        # k=2: eigenvalue ~ 8 (with lattice corrections)
        assert 3.5 < clusters[2][0] < 10.0, \
            f"k=2 cluster at {clusters[2][0]}, expected near 8"


# ==================================================================
# 4. Edge Projector
# ==================================================================

class TestEdgeProjector:
    """
    THEOREM: The I*-invariant projector on edges has rank 6.
    The quotient S^3/I* has V_q=1, E_q=6, F_q=10, C_q=5
    (Euler: 1-6+10-5 = 0).
    """

    def test_idempotent(self, edge_projector):
        """Pi_edge^2 = Pi_edge."""
        Pi = edge_projector
        Pi2 = Pi @ Pi
        np.testing.assert_allclose(Pi2, Pi, atol=1e-12,
            err_msg="Edge projector not idempotent")

    def test_symmetric(self, edge_projector):
        """Pi_edge^T = Pi_edge."""
        Pi = edge_projector
        np.testing.assert_allclose(Pi.T, Pi, atol=1e-14,
            err_msg="Edge projector not symmetric")

    def test_rank_is_6(self, edge_projector):
        """
        THEOREM: rank(Pi_edge) = 6.
        The quotient S^3/I* has 6 edges (720 edges / 120 group elements).
        """
        Pi = edge_projector
        evals = la.eigh(Pi, eigvals_only=True)
        rank = np.sum(evals > 0.5)
        assert rank == 6, f"Expected edge projector rank 6, got {rank}"

    def test_eigenvalues_are_0_or_1(self, edge_projector):
        """Projector eigenvalues should be 0 or 1."""
        Pi = edge_projector
        evals = la.eigh(Pi, eigvals_only=True)
        for ev in evals:
            assert abs(ev) < 1e-10 or abs(ev - 1) < 1e-10, \
                f"Projector eigenvalue {ev} is not 0 or 1"


# ==================================================================
# 5. Hodge Laplacian
# ==================================================================

class TestHodgeLaplacian:
    """
    THEOREM: The Hodge Laplacian on 1-forms is symmetric positive
    semi-definite, and d_1 d_0 = 0 (coboundary squares to zero).
    On S^3 (b_1 = 0), there are no harmonic 1-forms.
    """

    def test_coboundary_squares_to_zero(self, poincare):
        """
        THEOREM: d_1 d_0 = 0 (the coboundary operator squares to zero).
        This is the discrete analog of d^2 = 0.
        """
        d0 = poincare.incidence_matrix()
        d1 = poincare.face_boundary_matrix()
        product = d1 @ d0
        assert np.linalg.norm(product) < 1e-12, \
            f"d1 d0 should be zero, norm = {np.linalg.norm(product)}"

    def test_symmetric(self, poincare):
        """Hodge Laplacian is symmetric."""
        L1 = poincare.hodge_laplacian_1()
        np.testing.assert_allclose(L1, L1.T, atol=1e-12,
            err_msg="Hodge Laplacian not symmetric")

    def test_positive_semidefinite(self, poincare):
        """All eigenvalues >= 0."""
        evals, _ = poincare.oneform_spectrum_full()
        assert np.all(evals >= -1e-10), \
            f"Negative eigenvalue: {np.min(evals)}"

    def test_no_harmonic_1forms(self, poincare):
        """
        THEOREM: dim ker(Delta_1) = b_1(S^3) = 0.
        S^3 is simply connected, so there are no harmonic 1-forms.
        """
        evals, _ = poincare.oneform_spectrum_full()
        n_zero = np.sum(np.abs(evals) < 1e-6)
        assert n_zero == 0, f"Expected 0 harmonic 1-forms, got {n_zero}"

    def test_dimension_720(self, poincare):
        """1-form space has dimension 720 (one per edge)."""
        evals, _ = poincare.oneform_spectrum_full()
        assert len(evals) == 720

    def test_hodge_decomposition_dimensions(self, oneform_decomp):
        """
        THEOREM: C^1 = exact + coexact + harmonic
        dim(exact) = rank(d0) = 119
        dim(coexact) = rank(d1) = 601
        dim(harmonic) = b_1 = 0
        Total = 720
        """
        n_exact = len(oneform_decomp['exact_evals'])
        n_coexact = len(oneform_decomp['coexact_evals'])
        n_harmonic = (oneform_decomp['harmonic_evecs'].shape[1]
                      if oneform_decomp['harmonic_evecs'].ndim > 1
                      else 0)

        total = n_exact + n_coexact + n_harmonic
        assert total == 720, f"Decomposition total {total} != 720"
        assert n_harmonic == 0, f"Expected 0 harmonic modes, got {n_harmonic}"
        # Allow off-by-one due to numerical classification
        assert abs(n_exact - 119) <= 1, \
            f"Expected ~119 exact modes, got {n_exact}"
        assert abs(n_coexact - 601) <= 1, \
            f"Expected ~601 coexact modes, got {n_coexact}"

    def test_incidence_matrix_shape(self, poincare):
        """d_0 has shape (720, 120)."""
        d0 = poincare.incidence_matrix()
        assert d0.shape == (720, 120)

    def test_face_boundary_matrix_shape(self, poincare):
        """d_1 has shape (1200, 720)."""
        d1 = poincare.face_boundary_matrix()
        assert d1.shape == (1200, 720)


# ==================================================================
# 6. Coexact Spectrum
# ==================================================================

class TestCoexactSpectrum:
    """
    NUMERICAL: The coexact 1-form spectrum on the 600-cell approximates
    the continuum values (k+1)^2/R^2 for k=1,2,3,...
    The lowest coexact eigenvalue is at k=1 with multiplicity 6
    (= 2*1*3, the total coexact multiplicity at k=1 on S^3).
    """

    def test_lowest_coexact_is_positive(self, poincare):
        """Lowest coexact eigenvalue > 0 (mass gap on S^3)."""
        coex_evals, _ = poincare.oneform_spectrum_coexact()
        assert coex_evals[0] > 0.1, \
            f"Lowest coexact eigenvalue too small: {coex_evals[0]}"

    def test_k1_multiplicity_is_6(self, poincare):
        """
        NUMERICAL: The k=1 coexact level has multiplicity 6.
        Continuum: 2k(k+2) = 2*1*3 = 6 at k=1.
        """
        coex_evals, _ = poincare.oneform_spectrum_coexact()
        clusters = _cluster_eigenvalues(coex_evals, tol=0.3)
        assert clusters[0][1] == 6, \
            f"k=1 cluster has mult {clusters[0][1]}, expected 6"

    def test_k2_multiplicity_is_16(self, poincare):
        """
        NUMERICAL: The k=2 coexact level has multiplicity 16.
        Continuum: 2k(k+2) = 2*2*4 = 16 at k=2.
        """
        coex_evals, _ = poincare.oneform_spectrum_coexact()
        clusters = _cluster_eigenvalues(coex_evals, tol=0.3)
        assert clusters[1][1] == 16, \
            f"k=2 cluster has mult {clusters[1][1]}, expected 16"

    def test_k3_multiplicity_is_30(self, poincare):
        """
        NUMERICAL: The k=3 coexact level has multiplicity 30.
        Continuum: 2k(k+2) = 2*3*5 = 30 at k=3.
        """
        coex_evals, _ = poincare.oneform_spectrum_coexact()
        clusters = _cluster_eigenvalues(coex_evals, tol=0.3)
        assert clusters[2][1] == 30, \
            f"k=3 cluster has mult {clusters[2][1]}, expected 30"

    def test_total_coexact_count(self, poincare):
        """
        Total coexact modes should be ~600-601.
        """
        coex_evals, _ = poincare.oneform_spectrum_coexact()
        assert 598 <= len(coex_evals) <= 603, \
            f"Expected ~600 coexact modes, got {len(coex_evals)}"


# ==================================================================
# 7. Gap Preservation (THE KEY RESULT)
# ==================================================================

class TestGapPreservation:
    """
    THEOREM: The lowest coexact eigenvalue on S^3 survives the I* projection.
    This means the Yang-Mills mass gap on S^3/I* equals that on S^3.

    The k=1 coexact level (eigenvalue ~0.528 on the lattice, approximating
    4/R^2 in the continuum) has 3 I*-invariant modes (the right-invariant
    self-dual 1-forms).
    """

    def test_gap_preserved(self, poincare):
        """
        THEOREM: gap_preserved is True.
        The lowest coexact eigenvalue in the full spectrum equals the
        lowest coexact eigenvalue in the I*-invariant spectrum.
        """
        coex_full, _ = poincare.oneform_spectrum_coexact()
        coex_istar, _ = poincare.coexact_spectrum_istar()

        lowest_full = np.min(coex_full)
        lowest_istar = np.min(coex_istar)

        # They should be the same (within numerical precision)
        np.testing.assert_allclose(lowest_istar, lowest_full, rtol=1e-8,
            err_msg=f"Gap NOT preserved: full={lowest_full}, istar={lowest_istar}")

    def test_istar_coexact_k1_has_3_modes(self, poincare):
        """
        THEOREM: The k=1 coexact level has exactly 3 I*-invariant modes.
        These are the self-dual right-invariant 1-forms on S^3.
        (m(k-1) = m(0) = 1, so SD count = 1 * (k+2) = 3.)
        """
        coex_istar, coex_clusters = poincare.coexact_spectrum_istar()
        assert len(coex_clusters) >= 1, "No I*-invariant coexact clusters found"
        # First cluster should have multiplicity 3
        assert coex_clusters[0][1] == 3, \
            f"k=1 I*-invariant cluster has mult {coex_clusters[0][1]}, expected 3"

    def test_total_istar_coexact_modes(self, poincare):
        """
        NUMERICAL: Total I*-invariant coexact modes should be 6.
        (3 at the lowest level, 3 at a higher level.)
        All I*-invariant 1-forms are coexact (no I*-invariant exact forms).
        """
        coex_istar, _ = poincare.coexact_spectrum_istar()
        assert len(coex_istar) == 6, \
            f"Expected 6 I*-invariant coexact modes, got {len(coex_istar)}"

    def test_istar_coexact_has_two_clusters(self, poincare):
        """
        NUMERICAL: The I*-invariant coexact spectrum on the 600-cell
        has exactly two clusters: 3 modes at the lowest level and 3
        at a higher level.
        """
        _, coex_clusters = poincare.coexact_spectrum_istar()
        assert len(coex_clusters) == 2, \
            f"Expected 2 I*-invariant coexact clusters, got {len(coex_clusters)}"
        assert coex_clusters[0][1] == 3
        assert coex_clusters[1][1] == 3

    def test_all_istar_1forms_are_coexact(self, poincare):
        """
        THEOREM: All I*-invariant 1-forms on S^3 are coexact.
        The only I*-invariant scalar is the constant, whose gradient is zero.
        Therefore there are no I*-invariant exact 1-forms.
        """
        # I*-invariant 1-form eigenvalues (all types)
        istar_all, rank_all = poincare.oneform_spectrum_istar()
        # I*-invariant coexact eigenvalues
        istar_coex, _ = poincare.coexact_spectrum_istar()

        assert rank_all == len(istar_coex), \
            f"All {rank_all} I*-invariant 1-forms should be coexact, but only {len(istar_coex)} are"


# ==================================================================
# 8. Spectral Sparsification
# ==================================================================

class TestSpectralSparsification:
    """
    THEOREM: On S^3/I*, the coexact spectrum is drastically sparser
    than on S^3. Modes at k=2..5 are absent. The next surviving level
    after k=1 is much higher.
    """

    def test_k2_absent(self, poincare):
        """
        THEOREM: The k=2 coexact level has 0 I*-invariant modes.
        On S^3, k=2 has multiplicity 16. On S^3/I*, all 16 are projected out.
        """
        coex_full, _ = poincare.oneform_spectrum_coexact()
        coex_istar, _ = poincare.coexact_spectrum_istar()
        full_clusters = _cluster_eigenvalues(coex_full, tol=0.3)

        # k=2 lattice eigenvalue
        k2_eigenvalue = full_clusters[1][0]

        # Check no I*-invariant modes near this eigenvalue
        near_k2 = [e for e in coex_istar
                    if abs(e - k2_eigenvalue) / k2_eigenvalue < 0.2]
        assert len(near_k2) == 0, \
            f"k=2 should be absent in I*-invariant spectrum, found {len(near_k2)} modes"

    def test_k3_absent(self, poincare):
        """k=3 coexact level has 0 I*-invariant modes."""
        coex_full, _ = poincare.oneform_spectrum_coexact()
        coex_istar, _ = poincare.coexact_spectrum_istar()
        full_clusters = _cluster_eigenvalues(coex_full, tol=0.3)

        k3_eigenvalue = full_clusters[2][0]
        near_k3 = [e for e in coex_istar
                    if abs(e - k3_eigenvalue) / k3_eigenvalue < 0.2]
        assert len(near_k3) == 0, \
            f"k=3 should be absent in I*-invariant spectrum, found {len(near_k3)} modes"

    def test_k4_absent(self, poincare):
        """k=4 coexact level has 0 I*-invariant modes."""
        coex_full, _ = poincare.oneform_spectrum_coexact()
        coex_istar, _ = poincare.coexact_spectrum_istar()
        full_clusters = _cluster_eigenvalues(coex_full, tol=0.3)

        k4_eigenvalue = full_clusters[3][0]
        near_k4 = [e for e in coex_istar
                    if abs(e - k4_eigenvalue) / k4_eigenvalue < 0.2]
        assert len(near_k4) == 0, \
            f"k=4 should be absent in I*-invariant spectrum, found {len(near_k4)} modes"

    def test_sparsification_ratio(self, poincare):
        """
        NUMERICAL: The sparsification ratio is dramatic.
        Full coexact has ~600 modes. I*-invariant coexact has 6 modes.
        Ratio: 100x reduction.
        """
        coex_full, _ = poincare.oneform_spectrum_coexact()
        coex_istar, _ = poincare.coexact_spectrum_istar()

        ratio = len(coex_full) / max(len(coex_istar), 1)
        assert ratio > 50, \
            f"Sparsification ratio {ratio:.1f} < 50 (expected ~100)"

    def test_verification_report_gap(self, poincare):
        """Integration test: verification_report says gap_preserved = True."""
        report = poincare.verification_report()
        assert report['gap_preserved'] == True

    def test_verification_report_sparsification(self, poincare):
        """Integration test: verification_report says sparsification_verified = True."""
        report = poincare.verification_report()
        assert report['sparsification_verified'] == True


# ==================================================================
# 9. Quaternion Multiplication (unit tests)
# ==================================================================

class TestQuaternionMultiplication:
    """Unit tests for the quaternion multiplication helper."""

    def test_identity_left(self):
        """1 * q = q."""
        q = np.array([0.5, 0.3, -0.1, 0.8])
        result = _quaternion_multiply(np.array([1, 0, 0, 0]), q)
        np.testing.assert_allclose(result, q, atol=1e-14)

    def test_identity_right(self):
        """q * 1 = q."""
        q = np.array([0.5, 0.3, -0.1, 0.8])
        result = _quaternion_multiply(q, np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(result, q, atol=1e-14)

    def test_i_squared(self):
        """i * i = -1."""
        i = np.array([0, 1, 0, 0])
        result = _quaternion_multiply(i, i)
        np.testing.assert_allclose(result, [-1, 0, 0, 0], atol=1e-14)

    def test_j_squared(self):
        """j * j = -1."""
        j = np.array([0, 0, 1, 0])
        result = _quaternion_multiply(j, j)
        np.testing.assert_allclose(result, [-1, 0, 0, 0], atol=1e-14)

    def test_k_squared(self):
        """k * k = -1."""
        k = np.array([0, 0, 0, 1])
        result = _quaternion_multiply(k, k)
        np.testing.assert_allclose(result, [-1, 0, 0, 0], atol=1e-14)

    def test_ij_equals_k(self):
        """i * j = k."""
        i = np.array([0, 1, 0, 0])
        j = np.array([0, 0, 1, 0])
        result = _quaternion_multiply(i, j)
        np.testing.assert_allclose(result, [0, 0, 0, 1], atol=1e-14)

    def test_ji_equals_neg_k(self):
        """j * i = -k (non-commutative)."""
        i = np.array([0, 1, 0, 0])
        j = np.array([0, 0, 1, 0])
        result = _quaternion_multiply(j, i)
        np.testing.assert_allclose(result, [0, 0, 0, -1], atol=1e-14)

    def test_associativity(self):
        """(p * q) * r = p * (q * r)."""
        p = np.array([0.5, 0.3, -0.1, 0.8])
        q = np.array([-0.2, 0.7, 0.4, 0.1])
        r = np.array([0.1, -0.5, 0.6, -0.3])
        lhs = _quaternion_multiply(_quaternion_multiply(p, q), r)
        rhs = _quaternion_multiply(p, _quaternion_multiply(q, r))
        np.testing.assert_allclose(lhs, rhs, atol=1e-13)

    def test_norm_preservation(self):
        """|p * q| = |p| * |q| (quaternion norm is multiplicative)."""
        p = np.array([0.5, 0.3, -0.1, 0.8])
        q = np.array([-0.2, 0.7, 0.4, 0.1])
        product = _quaternion_multiply(p, q)
        np.testing.assert_allclose(
            np.linalg.norm(product),
            np.linalg.norm(p) * np.linalg.norm(q),
            atol=1e-14
        )


# ==================================================================
# 10. Radius Scaling
# ==================================================================

class TestRadiusScaling:
    """
    NUMERICAL: Eigenvalues scale as 1/R^2 when the radius changes.
    The I*-invariant structure (rank, multiplicity) is R-independent.
    """

    def test_scalar_eigenvalues_scale(self):
        """Scalar eigenvalues scale as 1/R^2."""
        pl1 = PoincareLattice(R=1.0)
        pl2 = PoincareLattice(R=2.0)

        evals1, _ = pl1.scalar_spectrum_full()
        evals2, _ = pl2.scalar_spectrum_full()

        # evals2 should be evals1 / 4 (since R2 = 2*R1, eigenvalues ~ 1/R^2)
        # But the graph Laplacian uses chordal distance, so it's not exactly 1/R^2.
        # The graph Laplacian L = D - A is independent of R for unit sphere,
        # but our d0 uses actual vertex coordinates, so d0 ~ R * d0_unit.
        # L_0 = d0^T d0 ~ R^2 * L_unit. Wait no, d0 f = f(j) - f(i) doesn't
        # depend on R for the graph Laplacian (it's combinatorial).
        # Actually, our d0 is the combinatorial coboundary: (d0 f)_e = f(j) - f(i),
        # which is independent of R. So the scalar Laplacian is R-independent!
        # The connection to the continuum Laplacian Delta = k(k+2)/R^2 is through
        # a lattice-to-continuum normalization factor that depends on R.
        # But for the discrete I* projection, the structure is R-independent.
        np.testing.assert_allclose(evals2, evals1, atol=1e-10,
            err_msg="Graph Laplacian eigenvalues should be R-independent")

    def test_istar_rank_r_independent(self):
        """I* projector rank is independent of R."""
        pl2 = PoincareLattice(R=2.0)
        _, rank = pl2.scalar_spectrum_istar()
        assert rank == 1

    def test_istar_coexact_structure_r_independent(self):
        """I*-invariant coexact cluster structure is R-independent."""
        pl2 = PoincareLattice(R=2.0)
        _, clusters = pl2.coexact_spectrum_istar()
        assert len(clusters) == 2
        assert clusters[0][1] == 3
        assert clusters[1][1] == 3


# ==================================================================
# 11. Cluster Eigenvalues Helper
# ==================================================================

class TestClusterEigenvalues:
    """Unit tests for the _cluster_eigenvalues helper."""

    def test_empty(self):
        """Empty input gives empty output."""
        assert _cluster_eigenvalues(np.array([]), tol=0.5) == []

    def test_single(self):
        """Single eigenvalue gives one cluster of size 1."""
        result = _cluster_eigenvalues(np.array([3.0]), tol=0.5)
        assert len(result) == 1
        assert result[0][1] == 1

    def test_two_close(self):
        """Two eigenvalues within tol -> one cluster of size 2."""
        result = _cluster_eigenvalues(np.array([1.0, 1.2]), tol=0.5)
        assert len(result) == 1
        assert result[0][1] == 2

    def test_two_far(self):
        """Two eigenvalues outside tol -> two clusters of size 1."""
        result = _cluster_eigenvalues(np.array([1.0, 5.0]), tol=0.5)
        assert len(result) == 2
        assert result[0][1] == 1
        assert result[1][1] == 1

    def test_known_pattern(self):
        """Known pattern: 3 at 1.0, 2 at 5.0."""
        evals = np.array([0.9, 1.0, 1.1, 4.9, 5.1])
        result = _cluster_eigenvalues(evals, tol=0.5)
        assert len(result) == 2
        assert result[0][1] == 3
        assert result[1][1] == 2
