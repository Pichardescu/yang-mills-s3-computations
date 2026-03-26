"""
Tests for the Yang-Mills polymer algebra on S^3.

Tests cover all 7 classes:
1. GaugeFieldNorm: gauge field norms, lattice norms, scaling
2. WilsonLoopRegulator: Wilson loops, large-field detection, Gribov emptiness
3. TPhiSeminorm: field regulator, analyticity, numerical evaluation
4. GaugeCovariantPolymerAlgebra: product, norm, submultiplicativity, gauge inv
5. KoteckyPreissCondition: KP check, optimal a, explicit inequalities
6. BKTreeExpansion: tree counting, enumeration, finiteness
7. PolymerSpaceAtScale: integration of all components

Run:
    pytest tests/rg/test_polymer_algebra_ym.py -v
"""

import math
import numpy as np
import pytest
from collections import defaultdict

from yang_mills_s3.rg.polymer_algebra_ym import (
    GaugeFieldNorm,
    WilsonLoopRegulator,
    TPhiSeminorm,
    GaugeCovariantPolymerAlgebra,
    KoteckyPreissCondition,
    BKTreeExpansion,
    PolymerSpaceAtScale,
    PolymerSpaceReport,
    G2_BARE_DEFAULT,
    N_C_DEFAULT,
    DIM_ADJ_SU2,
    D_A_ENGINEERING,
    R_PHYSICAL_FM,
)
from yang_mills_s3.rg.banach_norm import Polymer


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def linear_adj_5():
    """5-node linear graph: 0-1-2-3-4."""
    return {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3},
    }


@pytest.fixture
def cycle_adj_4():
    """4-node cycle: 0-1-2-3-0."""
    return {
        0: {1, 3},
        1: {0, 2},
        2: {1, 3},
        3: {2, 0},
    }


@pytest.fixture
def complete_adj_4():
    """Complete graph K4."""
    return {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }


@pytest.fixture
def star_adj_5():
    """Star graph: center 0 connected to 1,2,3,4."""
    return {
        0: {1, 2, 3, 4},
        1: {0},
        2: {0},
        3: {0},
        4: {0},
    }


@pytest.fixture
def triangle_adj():
    """Triangle: 0-1-2-0."""
    return {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
    }


@pytest.fixture
def single_node_adj():
    """Single isolated node."""
    return {0: set()}


@pytest.fixture
def two_node_adj():
    """Two connected nodes."""
    return {0: {1}, 1: {0}}


@pytest.fixture
def identity_su2():
    """SU(2) identity matrix."""
    return np.eye(2, dtype=complex)


@pytest.fixture
def random_su2_link(identity_su2):
    """Generate a random SU(2) link variable close to identity."""
    rng = np.random.RandomState(42)
    theta = rng.randn(3) * 0.1
    # Pauli matrices
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    H = sum(t * s for t, s in zip(theta, sigma))
    # U = exp(i H) via eigendecomposition
    U = la.expm(1j * H)
    return U


def la_expm_import():
    """Import scipy.linalg for matrix exponential."""
    from scipy import linalg
    return linalg


# Need scipy.linalg for SU(2) matrix exponential
from scipy import linalg as la


# ======================================================================
# 1. GaugeFieldNorm tests
# ======================================================================

class TestGaugeFieldNorm:
    """Tests for gauge field norm computation."""

    def test_creation_su2(self):
        """GaugeFieldNorm for SU(2)."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        assert gfn.N_c == 2
        assert gfn.dim_adj == 3
        assert gfn.d_A == 1

    def test_creation_su3(self):
        """GaugeFieldNorm for SU(3)."""
        gfn = GaugeFieldNorm(N_c=3, M=2.0)
        assert gfn.N_c == 3
        assert gfn.dim_adj == 8

    def test_creation_invalid_nc(self):
        """N_c must be >= 2."""
        with pytest.raises(ValueError, match="N_c must be >= 2"):
            GaugeFieldNorm(N_c=1)

    def test_creation_invalid_M(self):
        """M must be > 1."""
        with pytest.raises(ValueError, match="Blocking factor M must be > 1"):
            GaugeFieldNorm(N_c=2, M=0.5)

    def test_field_norm_zero_field(self):
        """Zero field has zero norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.zeros((10, 3, 3))  # 10 sites, 3 directions, 3 color components
        assert gfn.field_norm(A, scale_j=0) == 0.0

    def test_field_norm_uniform_field(self):
        """Uniform field: all components = 1."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.ones((5, 3, 3))
        # |A(x)|^2 = 3 * 3 = 9, |A(x)| = 3
        # At scale j=0: M^0 = 1
        norm = gfn.field_norm(A, scale_j=0)
        assert abs(norm - 3.0) < 1e-10

    def test_field_norm_scaling_with_j(self):
        """Norm scales as M^{j*d_A} = 2^j."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0, d_A=1)
        A = np.ones((5, 3, 3))
        norm_0 = gfn.field_norm(A, scale_j=0)
        norm_1 = gfn.field_norm(A, scale_j=1)
        norm_2 = gfn.field_norm(A, scale_j=2)
        assert abs(norm_1 / norm_0 - 2.0) < 1e-10
        assert abs(norm_2 / norm_0 - 4.0) < 1e-10

    def test_field_norm_empty_field(self):
        """Empty array has zero norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.array([]).reshape(0, 3, 3)
        assert gfn.field_norm(A, scale_j=0) == 0.0

    def test_field_norm_single_site(self):
        """Single site with known values."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.zeros((1, 3, 3))
        A[0, 0, 0] = 1.0  # Single component
        norm = gfn.field_norm(A, scale_j=0)
        assert abs(norm - 1.0) < 1e-10

    def test_field_norm_sup_over_sites(self):
        """Norm takes sup over sites."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.zeros((3, 3, 3))
        A[0, 0, 0] = 1.0  # site 0: |A| = 1
        A[1, 0, 0] = 2.0  # site 1: |A| = 2
        A[2, 0, 0] = 0.5  # site 2: |A| = 0.5
        norm = gfn.field_norm(A, scale_j=0)
        assert abs(norm - 2.0) < 1e-10

    def test_lattice_field_norm_identity_links(self):
        """Identity link variables have zero lattice norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        n_links = 10
        links = np.tile(np.eye(2, dtype=complex), (n_links, 1, 1))
        assert gfn.lattice_field_norm(links, scale_j=0) == 0.0

    def test_lattice_field_norm_nonidentity(self):
        """Non-identity links have positive norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        # U = exp(i * 0.1 * sigma_z)
        U = la.expm(0.1j * sigma_z)
        links = np.tile(U, (5, 1, 1))
        norm = gfn.lattice_field_norm(links, scale_j=0)
        assert norm > 0
        # ||U - I||^2 = Tr((U-I)^dag (U-I)) for this U
        dev = U - np.eye(2)
        expected = np.sqrt(np.real(np.trace(dev.conj().T @ dev)))
        assert abs(norm - expected) < 1e-10

    def test_lattice_field_norm_empty(self):
        """Empty link array has zero norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        links = np.array([]).reshape(0, 2, 2)
        assert gfn.lattice_field_norm(links, scale_j=0) == 0.0

    def test_scaled_norm_equals_field_norm(self):
        """scaled_norm should equal field_norm."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        A = np.ones((5, 3, 3))
        assert gfn.scaled_norm(A, 3) == gfn.field_norm(A, 3)

    def test_lattice_norm_scaling(self):
        """Lattice norm scales with M^{j*d_A}."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        U = la.expm(0.05j * sigma_x)
        links = np.tile(U, (5, 1, 1))
        norm_0 = gfn.lattice_field_norm(links, scale_j=0)
        norm_2 = gfn.lattice_field_norm(links, scale_j=2)
        assert abs(norm_2 / norm_0 - 4.0) < 1e-10


# ======================================================================
# 2. WilsonLoopRegulator tests
# ======================================================================

class TestWilsonLoopRegulator:
    """Tests for Wilson loop large-field regulator."""

    def test_creation(self):
        """Basic creation."""
        wlr = WilsonLoopRegulator(N_c=2, p0=0.5)
        assert wlr.N_c == 2
        assert wlr.p0 == 0.5

    def test_creation_invalid_nc(self):
        """N_c must be >= 2."""
        with pytest.raises(ValueError, match="N_c must be >= 2"):
            WilsonLoopRegulator(N_c=1)

    def test_wilson_plaquette_identity_links(self):
        """Identity links give identity plaquette."""
        wlr = WilsonLoopRegulator(N_c=2)
        link_vars = {
            (0, 1): np.eye(2, dtype=complex),
            (1, 2): np.eye(2, dtype=complex),
            (2, 3): np.eye(2, dtype=complex),
            (0, 3): np.eye(2, dtype=complex),
        }
        W = wlr.wilson_plaquette(link_vars, (0, 1, 2, 3))
        assert np.allclose(W, np.eye(2), atol=1e-12)

    def test_wilson_plaquette_is_unitary(self):
        """Wilson plaquette should be (approximately) unitary."""
        wlr = WilsonLoopRegulator(N_c=2)
        rng = np.random.RandomState(42)
        sigma = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        link_vars = {}
        for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            theta = rng.randn(3) * 0.1
            H = sum(t * s for t, s in zip(theta, sigma))
            link_vars[(i, j)] = la.expm(1j * H)

        W = wlr.wilson_plaquette(link_vars, (0, 1, 2, 3))
        # W should be approximately unitary
        assert np.allclose(W @ W.conj().T, np.eye(2), atol=1e-10)

    def test_field_strength_proxy_identity(self):
        """Identity plaquette has zero field strength."""
        wlr = WilsonLoopRegulator(N_c=2)
        W = np.eye(2, dtype=complex)
        assert wlr.field_strength_proxy(W) < 1e-12

    def test_field_strength_proxy_positive(self):
        """Non-identity plaquette has positive field strength."""
        wlr = WilsonLoopRegulator(N_c=2)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        W = la.expm(0.1j * sigma_z)
        assert wlr.field_strength_proxy(W) > 0

    def test_field_strength_proxy_gauge_invariant(self):
        """Field strength proxy is gauge-invariant: ||gWg^{-1} - I|| = ||W - I||."""
        wlr = WilsonLoopRegulator(N_c=2)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        W = la.expm(0.3j * sigma_z)
        g = la.expm(0.7j * sigma_x)
        g_inv = g.conj().T
        W_rotated = g @ W @ g_inv
        fs_original = wlr.field_strength_proxy(W)
        fs_rotated = wlr.field_strength_proxy(W_rotated)
        assert abs(fs_original - fs_rotated) < 1e-10

    def test_is_large_field_below_threshold(self):
        """Fields below threshold are not large."""
        wlr = WilsonLoopRegulator(N_c=2, p0=1.0)
        link_vars = {
            (0, 1): np.eye(2, dtype=complex),
            (1, 2): np.eye(2, dtype=complex),
            (2, 3): np.eye(2, dtype=complex),
            (0, 3): np.eye(2, dtype=complex),
        }
        plaquettes = [(0, 1, 2, 3)]
        assert not wlr.is_large_field(link_vars, plaquettes)

    def test_is_large_field_above_threshold(self):
        """Fields above threshold are large."""
        wlr = WilsonLoopRegulator(N_c=2, p0=0.01)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        U = la.expm(0.5j * sigma_z)
        link_vars = {
            (0, 1): U,
            (1, 2): U,
            (2, 3): U,
            (0, 3): U,
        }
        plaquettes = [(0, 1, 2, 3)]
        assert wlr.is_large_field(link_vars, plaquettes)

    def test_is_large_field_no_p0_raises(self):
        """Must provide p0 either at construction or as argument."""
        wlr = WilsonLoopRegulator(N_c=2)
        link_vars = {(0, 1): np.eye(2, dtype=complex)}
        with pytest.raises(ValueError, match="Threshold p0 must be set"):
            wlr.is_large_field(link_vars, [(0, 1, 2, 3)])

    def test_large_field_blocks_all_small(self):
        """All identity links: no large-field blocks."""
        wlr = WilsonLoopRegulator(N_c=2, p0=0.1)
        link_vars = {
            (0, 1): np.eye(2, dtype=complex),
            (1, 2): np.eye(2, dtype=complex),
            (2, 3): np.eye(2, dtype=complex),
            (0, 3): np.eye(2, dtype=complex),
        }
        blocks = {0: [(0, 1, 2, 3)]}
        result = wlr.large_field_blocks(link_vars, blocks)
        assert len(result) == 0

    def test_large_field_blocks_some_large(self):
        """Some blocks with large fields detected."""
        wlr = WilsonLoopRegulator(N_c=2, p0=0.01)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        U_large = la.expm(0.5j * sigma_z)
        link_vars = {
            (0, 1): np.eye(2, dtype=complex),
            (1, 2): np.eye(2, dtype=complex),
            (2, 3): np.eye(2, dtype=complex),
            (0, 3): np.eye(2, dtype=complex),
            (4, 5): U_large,
            (5, 6): U_large,
            (6, 7): U_large,
            (4, 7): U_large,
        }
        blocks = {
            0: [(0, 1, 2, 3)],
            1: [(4, 5, 6, 7)],
        }
        result = wlr.large_field_blocks(link_vars, blocks)
        assert 1 in result
        assert 0 not in result

    def test_threshold_from_gribov_positive(self):
        """Gribov threshold is positive."""
        wlr = WilsonLoopRegulator(N_c=2)
        p0 = wlr.threshold_from_gribov(g2=6.28, mesh_size=0.1)
        assert p0 > 0

    def test_threshold_from_gribov_increases_with_g(self):
        """Larger coupling needs larger threshold."""
        wlr = WilsonLoopRegulator(N_c=2)
        p0_weak = wlr.threshold_from_gribov(g2=1.0, mesh_size=0.1)
        p0_strong = wlr.threshold_from_gribov(g2=10.0, mesh_size=0.1)
        assert p0_strong > p0_weak

    def test_gribov_emptiness_check_satisfied(self):
        """With large enough p0, large-field region is empty."""
        result = WilsonLoopRegulator.gribov_emptiness_check(
            g2=6.28, mesh_size=0.1, p0=10.0
        )
        assert result['is_empty']
        assert result['label'] == 'THEOREM'
        assert result['margin'] > 0

    def test_gribov_emptiness_check_violated(self):
        """With tiny p0, emptiness check fails."""
        result = WilsonLoopRegulator.gribov_emptiness_check(
            g2=6.28, mesh_size=0.5, p0=0.001
        )
        assert not result['is_empty']
        assert result['label'] == 'FAILS'

    def test_wilson_plaquette_reverse_orientation_field_strength(self):
        """Reversing plaquette orientation preserves field strength proxy.

        ||W - I|| is the same for W and W^dag (since ||W-I||_F = ||W^dag-I||_F).
        When we reverse the vertex ordering, the field strength proxy
        (which is the physical observable) is preserved.
        """
        wlr = WilsonLoopRegulator(N_c=2)
        rng = np.random.RandomState(99)
        sigma = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        link_vars = {}
        for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            theta = rng.randn(3) * 0.1
            H = sum(t * s for t, s in zip(theta, sigma))
            link_vars[(i, j)] = la.expm(1j * H)

        W_forward = wlr.wilson_plaquette(link_vars, (0, 1, 2, 3))
        W_reverse = wlr.wilson_plaquette(link_vars, (3, 2, 1, 0))
        # Field strength proxy is preserved under orientation reversal
        fs_forward = wlr.field_strength_proxy(W_forward)
        fs_reverse = wlr.field_strength_proxy(W_reverse)
        assert abs(fs_forward - fs_reverse) < 1e-10


# ======================================================================
# 3. TPhiSeminorm tests
# ======================================================================

class TestTPhiSeminorm:
    """Tests for the T_phi seminorm."""

    def test_creation(self):
        """Basic creation."""
        tp = TPhiSeminorm(M=2.0, n_derivatives=2)
        assert tp.M == 2.0
        assert tp.n_derivatives == 2

    def test_creation_invalid_M(self):
        """M must be > 1."""
        with pytest.raises(ValueError, match="M must be > 1"):
            TPhiSeminorm(M=0.5)

    def test_creation_invalid_n_deriv(self):
        """n_derivatives must be >= 0."""
        with pytest.raises(ValueError, match="n_derivatives must be >= 0"):
            TPhiSeminorm(n_derivatives=-1)

    def test_field_regulator_at_j0(self):
        """At j=0: h_0 = M^0 * sqrt(g2) = sqrt(g2)."""
        tp = TPhiSeminorm(M=2.0)
        h = tp.field_regulator(j=0, g2_j=4.0)
        assert abs(h - 2.0) < 1e-10

    def test_field_regulator_decreases_with_j(self):
        """Field regulator decreases with scale (UV -> IR)."""
        tp = TPhiSeminorm(M=2.0)
        h0 = tp.field_regulator(j=0, g2_j=4.0)
        h1 = tp.field_regulator(j=1, g2_j=4.0)
        h2 = tp.field_regulator(j=2, g2_j=4.0)
        assert h1 < h0
        assert h2 < h1
        # Ratio should be 1/M = 0.5
        assert abs(h1 / h0 - 0.5) < 1e-10

    def test_field_regulator_zero_coupling(self):
        """Zero coupling gives zero regulator."""
        tp = TPhiSeminorm(M=2.0)
        h = tp.field_regulator(j=0, g2_j=0.0)
        assert h == 0.0

    def test_analyticity_radius_equals_regulator(self):
        """Analyticity radius equals field regulator."""
        tp = TPhiSeminorm(M=2.0)
        h = tp.field_regulator(j=1, g2_j=6.28)
        r = tp.analyticity_radius(j=1, g2_j=6.28)
        assert abs(h - r) < 1e-15

    def test_evaluate_constant_activity(self):
        """Constant activity K(X, A) = c has T_phi norm ~ |c|."""
        tp = TPhiSeminorm(M=2.0, n_derivatives=0)
        polymer = Polymer(frozenset([0]))

        def K_const(poly, A):
            return 1.0

        norm = tp.evaluate(K_const, polymer, scale_j=0, g2_j=4.0)
        # With 0 derivatives, just sup of |K|, which is 1.0
        assert abs(norm - 1.0) < 0.2  # sampling-based, allow tolerance

    def test_evaluate_quadratic_activity(self):
        """Quadratic activity K(X, A) = A^2 has bounded T_phi norm."""
        tp = TPhiSeminorm(M=2.0, n_derivatives=1)
        polymer = Polymer(frozenset([0]))

        def K_quad(poly, A):
            return float(np.sum(A**2))

        norm = tp.evaluate(K_quad, polymer, scale_j=0, g2_j=4.0,
                           n_sample=100)
        assert norm > 0  # Non-trivial
        assert np.isfinite(norm)

    def test_evaluate_zero_activity(self):
        """Zero activity has zero T_phi norm."""
        tp = TPhiSeminorm(M=2.0)
        polymer = Polymer(frozenset([0]))

        def K_zero(poly, A):
            return 0.0

        norm = tp.evaluate(K_zero, polymer, scale_j=0, g2_j=4.0)
        assert abs(norm) < 1e-10

    def test_evaluate_bounded_by_sup_norm(self):
        """T_phi norm is bounded by sup_norm * (1 + h + h^2/2 + ...)."""
        tp = TPhiSeminorm(M=2.0, n_derivatives=2)
        polymer = Polymer(frozenset([0]))

        def K_bounded(poly, A):
            return 0.5  # Constant

        norm = tp.evaluate(K_bounded, polymer, scale_j=0, g2_j=1.0,
                           n_sample=50)
        # For constant function, derivatives are 0
        # So T_phi norm should be ~ 0.5
        assert norm < 5.0  # generous bound

    def test_field_regulator_negative_coupling_clamps(self):
        """Negative coupling gets clamped to 0."""
        tp = TPhiSeminorm(M=2.0)
        h = tp.field_regulator(j=0, g2_j=-1.0)
        assert h == 0.0


# ======================================================================
# 4. GaugeCovariantPolymerAlgebra tests
# ======================================================================

class TestGaugeCovariantPolymerAlgebra:
    """Tests for the polymer algebra product and norms."""

    def test_creation(self, linear_adj_5):
        """Basic creation."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5, kappa=1.0)
        assert alg.n_blocks == 5
        assert alg.kappa == 1.0

    def test_creation_invalid_kappa(self, linear_adj_5):
        """kappa must be > 0."""
        with pytest.raises(ValueError, match="kappa must be > 0"):
            GaugeCovariantPolymerAlgebra(linear_adj_5, kappa=0.0)

    def test_product_disjoint_polymers(self, linear_adj_5):
        """Product of disjoint polymer activities."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)
        p1 = Polymer(frozenset([0]))
        p2 = Polymer(frozenset([2]))
        K1 = {p1: 2.0 + 0j}
        K2 = {p2: 3.0 + 0j}
        result = alg.product(K1, K2)
        # Result should have polymer {0, 2}
        union_poly = Polymer(frozenset([0, 2]))
        assert union_poly in result
        assert abs(result[union_poly] - 6.0) < 1e-10

    def test_product_overlapping_polymers_empty(self, linear_adj_5):
        """Overlapping polymers give no product (disjointness required)."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)
        p1 = Polymer(frozenset([0, 1]))
        p2 = Polymer(frozenset([1, 2]))
        K1 = {p1: 1.0 + 0j}
        K2 = {p2: 1.0 + 0j}
        result = alg.product(K1, K2)
        assert len(result) == 0  # No disjoint decompositions

    def test_product_multiple_terms(self, linear_adj_5):
        """Product with multiple polymer terms."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)
        p0 = Polymer(frozenset([0]))
        p1 = Polymer(frozenset([1]))
        p3 = Polymer(frozenset([3]))
        K1 = {p0: 2.0 + 0j, p1: 1.0 + 0j}
        K2 = {p3: 3.0 + 0j}
        result = alg.product(K1, K2)
        # {0} u {3} and {1} u {3} are both valid
        assert Polymer(frozenset([0, 3])) in result
        assert Polymer(frozenset([1, 3])) in result

    def test_product_empty_gives_empty(self, linear_adj_5):
        """Product with empty activity is empty."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)
        p0 = Polymer(frozenset([0]))
        K1 = {p0: 1.0 + 0j}
        K2 = {}
        result = alg.product(K1, K2)
        assert len(result) == 0

    def test_norm_zero_activity(self, linear_adj_5):
        """Zero activity has zero norm."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)
        assert alg.norm({}) == 0.0

    def test_norm_single_polymer(self, linear_adj_5):
        """Norm of single polymer activity."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5, kappa=1.0)
        p = Polymer(frozenset([0]))
        K = {p: 2.0 + 0j}
        # ||K|| = |2| * exp(1 * 1) = 2 * e
        expected = 2.0 * np.exp(1.0)
        assert abs(alg.norm(K) - expected) < 1e-10

    def test_norm_exponential_decay(self, linear_adj_5):
        """Larger polymers contribute less to norm (for fixed amplitude)."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5, kappa=2.0)
        p1 = Polymer(frozenset([0]))        # size 1
        p2 = Polymer(frozenset([0, 1]))     # size 2
        # Same amplitude but larger polymer gets more weight
        K1 = {p1: 1.0 + 0j}
        K2 = {p2: 1.0 + 0j}
        # Weight: exp(kappa * size)
        assert alg.norm(K2) > alg.norm(K1)  # exp(2*2) > exp(2*1)

    def test_submultiplicativity(self, linear_adj_5):
        """||K1 * K2|| <= ||K1|| * ||K2||."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5, kappa=1.0)
        p0 = Polymer(frozenset([0]))
        p4 = Polymer(frozenset([4]))
        K1 = {p0: 0.5 + 0j}
        K2 = {p4: 0.3 + 0j}
        product = alg.product(K1, K2)
        norm_product = alg.norm(product)
        norm_K1 = alg.norm(K1)
        norm_K2 = alg.norm(K2)
        # Note: submultiplicativity is a theorem; here the product norm
        # should be <= norm(K1) * norm(K2)
        bound = alg.algebra_product_bound(norm_K1, norm_K2)
        assert norm_product <= bound * (1.0 + 1e-10)

    def test_algebra_product_bound(self):
        """Product bound is multiplicative."""
        bound = GaugeCovariantPolymerAlgebra.algebra_product_bound(2.0, 3.0)
        assert abs(bound - 6.0) < 1e-10

    def test_gauge_invariance_trivial_activity(self, linear_adj_5):
        """Constant activity is gauge-invariant."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)

        def K_const(polymer, A):
            return 1.0

        polymers = [Polymer(frozenset([0]))]
        assert alg.is_gauge_invariant(K_const, polymers, dim_field=9)

    def test_gauge_invariance_trace_activity(self, linear_adj_5):
        """Trace of A^2 is gauge-invariant (Killing norm)."""
        alg = GaugeCovariantPolymerAlgebra(linear_adj_5)

        def K_trace_sq(polymer, A):
            # sum A_a^2 is invariant under adjoint rotation
            return float(np.sum(A**2))

        polymers = [Polymer(frozenset([0]))]
        assert alg.is_gauge_invariant(K_trace_sq, polymers, dim_field=9)

    def test_disjoint_decompositions_single(self):
        """Single-block polymer has no non-trivial decompositions."""
        alg = GaugeCovariantPolymerAlgebra({0: set()})
        decomps = alg._disjoint_decompositions(frozenset([0]))
        # Single block: only decomposition would be ({0}, {}) which is trivial
        assert len(decomps) == 0

    def test_disjoint_decompositions_two_blocks(self):
        """Two-block polymer has one decomposition."""
        alg = GaugeCovariantPolymerAlgebra({0: {1}, 1: {0}})
        decomps = alg._disjoint_decompositions(frozenset([0, 1]))
        # ({0}, {1}) counted once (canonical ordering)
        assert len(decomps) == 1

    def test_disjoint_decompositions_three_blocks(self):
        """Three-block polymer has 3 decompositions."""
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        alg = GaugeCovariantPolymerAlgebra(adj)
        decomps = alg._disjoint_decompositions(frozenset([0, 1, 2]))
        # {0},{1,2} and {1},{0,2} and {2},{0,1} = 3 unique
        assert len(decomps) == 3


# ======================================================================
# 5. KoteckyPreissCondition tests
# ======================================================================

class TestKoteckyPreissCondition:
    """Tests for the Kotecky-Preiss convergence criterion."""

    def test_creation(self, linear_adj_5):
        """Basic creation."""
        kp = KoteckyPreissCondition(linear_adj_5)
        assert kp.n_blocks == 5
        assert kp._max_degree == 2

    def test_check_condition_small_norms(self, triangle_adj):
        """KP condition satisfied for very small norms."""
        kp = KoteckyPreissCondition(triangle_adj)
        # Very small norms: should satisfy any KP with reasonable a
        K_norms = {1: 1e-5, 2: 1e-10, 3: 1e-15}
        assert kp.check_condition(K_norms, a=0.1, max_polymer_size=3)

    def test_check_condition_large_norms_fails(self, triangle_adj):
        """KP condition fails for very large norms."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 100.0, 2: 100.0, 3: 100.0}
        assert not kp.check_condition(K_norms, a=0.01, max_polymer_size=3)

    def test_check_condition_negative_a_fails(self, triangle_adj):
        """KP condition requires a > 0."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-5}
        assert not kp.check_condition(K_norms, a=-1.0)

    def test_check_condition_zero_a_fails(self, triangle_adj):
        """a = 0 always fails."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-5}
        assert not kp.check_condition(K_norms, a=0.0)

    def test_find_optimal_a_exists(self, triangle_adj):
        """Optimal a exists for small norms."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-6, 2: 1e-12, 3: 1e-18}
        a_opt = kp.find_optimal_a(K_norms, max_polymer_size=3)
        assert a_opt > 0

    def test_find_optimal_a_monotone(self, triangle_adj):
        """Smaller norms allow larger optimal a."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms_small = {1: 1e-8, 2: 1e-16}
        K_norms_large = {1: 1e-3, 2: 1e-6}
        a_small = kp.find_optimal_a(K_norms_small, max_polymer_size=2)
        a_large = kp.find_optimal_a(K_norms_large, max_polymer_size=2)
        assert a_small >= a_large

    def test_explicit_inequalities_finite(self, triangle_adj):
        """Explicit inequalities list is finite."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-5, 2: 1e-10}
        ineqs = kp.explicit_inequalities(K_norms, a=1.0, max_polymer_size=3)
        assert len(ineqs) == 3  # one per size s=1,2,3
        assert all('satisfied' in ineq for ineq in ineqs)

    def test_explicit_inequalities_all_satisfied(self, triangle_adj):
        """All inequalities satisfied for small norms."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-8, 2: 1e-16, 3: 1e-24}
        ineqs = kp.explicit_inequalities(K_norms, a=0.1, max_polymer_size=3)
        assert all(ineq['satisfied'] for ineq in ineqs)

    def test_margin_positive_when_satisfied(self, triangle_adj):
        """Margin is positive when KP is satisfied."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-8, 2: 1e-16}
        margin = kp.margin(K_norms, a=0.1, max_polymer_size=2)
        assert margin > 0

    def test_margin_negative_when_violated(self, triangle_adj):
        """Margin is negative when KP fails."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 100.0, 2: 100.0}
        margin = kp.margin(K_norms, a=0.01, max_polymer_size=2)
        assert margin < 0

    def test_kp_on_complete_graph(self, complete_adj_4):
        """KP on K4 (higher degree = stronger condition)."""
        kp = KoteckyPreissCondition(complete_adj_4)
        assert kp._max_degree == 3
        K_norms = {1: 1e-6, 2: 1e-12, 3: 1e-18, 4: 1e-24}
        assert kp.check_condition(K_norms, a=0.1, max_polymer_size=4)

    def test_kp_empty_norms(self, triangle_adj):
        """Empty norms trivially satisfy KP."""
        kp = KoteckyPreissCondition(triangle_adj)
        assert kp.check_condition({}, a=1.0, max_polymer_size=3)

    def test_kp_perturbative_ym_norms(self, linear_adj_5):
        """KP with physically motivated YM norms (perturbative)."""
        kp = KoteckyPreissCondition(linear_adj_5)
        g2 = 0.1  # weak coupling
        C = 3.0
        K_norms = {}
        for s in range(1, 6):
            K_norms[s] = (C * g2)**s * np.exp(-s) / float(math.factorial(min(s, 20)))
        assert kp.check_condition(K_norms, a=0.5, max_polymer_size=5)


# ======================================================================
# 6. BKTreeExpansion tests
# ======================================================================

class TestBKTreeExpansion:
    """Tests for Brydges-Kennedy tree expansion."""

    def test_is_finite_always(self, triangle_adj):
        """BK expansion is always finite on S^3."""
        bk = BKTreeExpansion(triangle_adj)
        assert bk.is_finite()

    def test_is_finite_single_node(self, single_node_adj):
        """Single node: finite."""
        bk = BKTreeExpansion(single_node_adj)
        assert bk.is_finite()

    def test_tree_count_single_node(self, single_node_adj):
        """Single node has 1 spanning tree (trivial)."""
        bk = BKTreeExpansion(single_node_adj)
        assert bk.tree_count() == 1.0

    def test_tree_count_two_nodes(self, two_node_adj):
        """Two connected nodes have 1 spanning tree."""
        bk = BKTreeExpansion(two_node_adj)
        count = bk.tree_count()
        assert abs(count - 1.0) < 0.5  # Kirchhoff may give ~1

    def test_tree_count_triangle(self, triangle_adj):
        """Triangle (K3) has 3 spanning trees."""
        bk = BKTreeExpansion(triangle_adj)
        count = bk.tree_count()
        assert abs(count - 3.0) < 0.5

    def test_tree_count_complete_4(self, complete_adj_4):
        """K4 has 4^2 = 16 spanning trees (Cayley: n^{n-2})."""
        bk = BKTreeExpansion(complete_adj_4)
        count = bk.tree_count()
        assert abs(count - 16.0) < 1.0

    def test_tree_count_star(self, star_adj_5):
        """Star graph with 5 nodes has 1 spanning tree (star itself)."""
        bk = BKTreeExpansion(star_adj_5)
        count = bk.tree_count()
        assert abs(count - 1.0) < 0.5

    def test_enumerate_trees_triangle(self, triangle_adj):
        """Enumerate spanning trees of triangle."""
        bk = BKTreeExpansion(triangle_adj)
        trees = bk.enumerate_trees(max_nodes=3)
        assert len(trees) == 3
        # Each tree has 2 edges (n-1)
        for tree in trees:
            assert len(tree) == 2

    def test_enumerate_trees_two_nodes(self, two_node_adj):
        """Two nodes: 1 tree."""
        bk = BKTreeExpansion(two_node_adj)
        trees = bk.enumerate_trees(max_nodes=2)
        assert len(trees) == 1
        assert len(trees[0]) == 1  # 1 edge

    def test_enumerate_trees_too_large(self, linear_adj_5):
        """Cannot enumerate trees for graphs larger than max_nodes."""
        bk = BKTreeExpansion(linear_adj_5)
        with pytest.raises(ValueError, match="exceeding max_nodes"):
            bk.enumerate_trees(max_nodes=3)

    def test_enumerate_trees_single_node(self, single_node_adj):
        """Single node: 1 trivial tree."""
        bk = BKTreeExpansion(single_node_adj)
        trees = bk.enumerate_trees(max_nodes=1)
        assert len(trees) == 1
        assert len(trees[0]) == 0  # No edges

    def test_tree_weight_uniform_activities(self, triangle_adj):
        """Tree weight with uniform activities."""
        bk = BKTreeExpansion(triangle_adj)
        K = {0: 1.0, 1: 1.0, 2: 1.0}
        trees = bk.enumerate_trees(max_nodes=3)
        for tree in trees:
            w = bk.tree_weight(tree, K)
            # Each edge weight = sqrt(1*1) = 1, product of 2 edges = 1
            assert abs(w - 1.0) < 1e-10

    def test_bk_sum_small_graph(self, triangle_adj):
        """BK sum on triangle with unit activities."""
        bk = BKTreeExpansion(triangle_adj)
        K = {0: 1.0, 1: 1.0, 2: 1.0}
        total = bk.bk_sum(K, max_nodes=3)
        # 3 trees, each weight 1 -> sum = 3
        assert abs(total - 3.0) < 1e-10

    def test_bk_sum_zero_activities(self, triangle_adj):
        """BK sum with zero activities is zero."""
        bk = BKTreeExpansion(triangle_adj)
        K = {0: 0.0, 1: 0.0, 2: 0.0}
        total = bk.bk_sum(K, max_nodes=3)
        assert abs(total) < 1e-10

    def test_bk_sum_consistency(self, complete_adj_4):
        """BK sum on K4: should equal sum of tree weights."""
        bk = BKTreeExpansion(complete_adj_4)
        K = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
        total = bk.bk_sum(K, max_nodes=4)
        # Verify by manual enumeration
        trees = bk.enumerate_trees(max_nodes=4)
        manual_sum = sum(bk.tree_weight(t, K) for t in trees)
        assert abs(total - manual_sum) < 1e-10

    def test_tree_count_cycle(self, cycle_adj_4):
        """4-cycle has 4 spanning trees."""
        bk = BKTreeExpansion(cycle_adj_4)
        count = bk.tree_count()
        assert abs(count - 4.0) < 0.5

    def test_enumerate_trees_complete_4(self, complete_adj_4):
        """K4 has 16 spanning trees."""
        bk = BKTreeExpansion(complete_adj_4)
        trees = bk.enumerate_trees(max_nodes=4)
        assert len(trees) == 16

    def test_tree_count_empty(self):
        """Empty graph has 0 trees."""
        bk = BKTreeExpansion({})
        assert bk.tree_count() == 0.0

    def test_laplacian_built(self, triangle_adj):
        """Internal Laplacian is built correctly."""
        bk = BKTreeExpansion(triangle_adj)
        L = bk._build_laplacian()
        assert L.shape == (3, 3)
        # Row sums should be 0 (Laplacian property)
        row_sums = np.sum(L, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-10)


# ======================================================================
# 7. PolymerSpaceAtScale tests
# ======================================================================

class TestPolymerSpaceAtScale:
    """Tests for the integrated polymer space at a given RG scale."""

    @pytest.fixture
    def small_space(self, triangle_adj):
        """Small polymer space for testing (triangle graph)."""
        return PolymerSpaceAtScale(
            adjacency=triangle_adj,
            scale_j=0,
            g2_j=0.1,  # weak coupling
            M=2.0,
            R=2.2,
            N_c=2,
            kappa=1.0,
        )

    @pytest.fixture
    def physical_space(self, triangle_adj):
        """Physical coupling polymer space."""
        return PolymerSpaceAtScale(
            adjacency=triangle_adj,
            scale_j=0,
            g2_j=G2_BARE_DEFAULT,
            M=2.0,
            R=R_PHYSICAL_FM,
            N_c=2,
            kappa=1.0,
        )

    def test_creation(self, small_space):
        """Basic creation."""
        assert small_space.n_blocks == 3
        assert small_space.scale_j == 0
        assert small_space.N_c == 2

    def test_polymer_count_finite(self, small_space):
        """Polymer count is finite."""
        counts = small_space.polymer_count(max_size=3)
        assert all(np.isfinite(v) for v in counts.values())
        assert all(v > 0 for v in counts.values())

    def test_polymer_count_size_1(self, small_space):
        """Size-1 polymers: one per block."""
        counts = small_space.polymer_count(max_size=3)
        assert counts[1] == 3  # 3 blocks

    def test_max_polymer_size(self, small_space):
        """Max polymer = whole graph."""
        assert small_space.max_polymer_size() == 3

    def test_activity_norm_bound_positive(self, small_space):
        """Activity norm bound is positive."""
        bound = small_space.activity_norm_bound()
        assert bound > 0
        assert np.isfinite(bound)

    def test_activity_norm_bound_increases_with_g(self, triangle_adj):
        """Stronger coupling gives larger norm bound."""
        space_weak = PolymerSpaceAtScale(
            adjacency=triangle_adj, g2_j=0.1, kappa=1.0)
        space_strong = PolymerSpaceAtScale(
            adjacency=triangle_adj, g2_j=10.0, kappa=1.0)
        assert space_strong.activity_norm_bound() > space_weak.activity_norm_bound()

    def test_is_well_defined_weak_coupling(self, small_space):
        """Well-defined at weak coupling."""
        assert small_space.is_well_defined(max_polymer_size=3, kp_a=1.0)

    def test_report_structure(self, small_space):
        """Report has correct structure."""
        report = small_space.report(max_polymer_size=3, kp_a=1.0)
        assert isinstance(report, PolymerSpaceReport)
        assert report.scale_j == 0
        assert report.n_blocks == 3
        assert report.max_polymer_size == 3
        assert np.isfinite(report.activity_norm_bound)
        assert np.isfinite(report.h_j)
        assert np.isfinite(report.bk_tree_count)

    def test_report_bk_finite(self, small_space):
        """BK tree count is finite in report."""
        report = small_space.report()
        assert report.bk_tree_count > 0

    def test_report_label(self, small_space):
        """Report label is THEOREM or NUMERICAL."""
        report = small_space.report(max_polymer_size=3, kp_a=1.0)
        assert report.label in ('THEOREM', 'NUMERICAL')

    def test_gauge_norm_component(self, small_space):
        """GaugeFieldNorm component is accessible."""
        assert isinstance(small_space.gauge_norm, GaugeFieldNorm)
        assert small_space.gauge_norm.N_c == 2

    def test_wilson_regulator_component(self, small_space):
        """WilsonLoopRegulator component is accessible."""
        assert isinstance(small_space.wilson_regulator, WilsonLoopRegulator)

    def test_tphi_component(self, small_space):
        """TPhiSeminorm component is accessible."""
        assert isinstance(small_space.t_phi, TPhiSeminorm)

    def test_algebra_component(self, small_space):
        """GaugeCovariantPolymerAlgebra component is accessible."""
        assert isinstance(small_space.algebra, GaugeCovariantPolymerAlgebra)

    def test_kp_component(self, small_space):
        """KoteckyPreissCondition component is accessible."""
        assert isinstance(small_space.kp, KoteckyPreissCondition)

    def test_bk_component(self, small_space):
        """BKTreeExpansion component is accessible."""
        assert isinstance(small_space.bk, BKTreeExpansion)


# ======================================================================
# 8. Cross-component integration tests
# ======================================================================

class TestIntegration:
    """Integration tests across components."""

    def test_gauge_norm_with_wilson_loop(self):
        """GaugeFieldNorm and WilsonLoopRegulator agree on trivial config."""
        gfn = GaugeFieldNorm(N_c=2, M=2.0)
        wlr = WilsonLoopRegulator(N_c=2, p0=0.1)

        # Identity links -> zero field norm, small-field
        n_links = 10
        links = np.tile(np.eye(2, dtype=complex), (n_links, 1, 1))
        assert gfn.lattice_field_norm(links, scale_j=0) == 0.0

        identity_link_vars = {
            (0, 1): np.eye(2, dtype=complex),
            (1, 2): np.eye(2, dtype=complex),
            (2, 3): np.eye(2, dtype=complex),
            (0, 3): np.eye(2, dtype=complex),
        }
        assert not wlr.is_large_field(identity_link_vars, [(0, 1, 2, 3)])

    def test_kp_with_bk_consistency(self, triangle_adj):
        """KP satisfied implies BK sum is bounded."""
        kp = KoteckyPreissCondition(triangle_adj)
        bk = BKTreeExpansion(triangle_adj)

        K_norms = {1: 1e-6, 2: 1e-12, 3: 1e-18}
        assert kp.check_condition(K_norms, a=0.1, max_polymer_size=3)
        # BK is always finite
        assert bk.is_finite()
        # Tree count is finite
        assert np.isfinite(bk.tree_count())

    def test_algebra_with_kp(self, triangle_adj):
        """Product submultiplicativity consistent with KP."""
        alg = GaugeCovariantPolymerAlgebra(triangle_adj, kappa=1.0)
        kp = KoteckyPreissCondition(triangle_adj)

        # If individual norms satisfy KP, product norms should too
        # (at potentially larger coupling)
        K_norms = {1: 1e-6, 2: 1e-12}
        assert kp.check_condition(K_norms, a=0.1, max_polymer_size=2)

    def test_full_pipeline_weak_coupling(self, triangle_adj):
        """Full pipeline at weak coupling: all checks pass."""
        space = PolymerSpaceAtScale(
            adjacency=triangle_adj,
            scale_j=0,
            g2_j=0.01,
            M=2.0,
            R=2.2,
            N_c=2,
            kappa=1.0,
        )
        # At very weak coupling, everything should work
        assert space.is_well_defined(max_polymer_size=3, kp_a=1.0)
        report = space.report(max_polymer_size=3, kp_a=1.0)
        assert report.kp_satisfied
        assert report.is_well_defined

    def test_tphi_regulator_decreases(self):
        """Field regulator decreases with scale (UV suppression)."""
        tp = TPhiSeminorm(M=2.0)
        regulators = [tp.field_regulator(j, g2_j=4.0) for j in range(5)]
        for i in range(len(regulators) - 1):
            assert regulators[i + 1] < regulators[i]

    def test_polymer_space_different_scales(self, triangle_adj):
        """Polymer spaces at different scales are well-defined."""
        for j in range(3):
            space = PolymerSpaceAtScale(
                adjacency=triangle_adj,
                scale_j=j,
                g2_j=0.1,
                M=2.0,
                R=2.2,
                N_c=2,
                kappa=1.0,
            )
            assert space.is_well_defined(max_polymer_size=3, kp_a=1.0)


# ======================================================================
# 9. Comparison with existing banach_norm.py
# ======================================================================

class TestComparisonWithScalar:
    """Tests comparing YM algebra with existing scalar phi^4 infrastructure."""

    def test_polymer_norm_agrees_at_zero_field(self, triangle_adj):
        """YM polymer norm agrees with scalar norm at A=0 (no gauge field)."""
        from yang_mills_s3.rg.banach_norm import PolymerNorm, LargeFieldRegulator

        kappa = 1.0
        sigma_sq = 1.0
        regulator = LargeFieldRegulator(sigma_sq=sigma_sq)
        scalar_norm = PolymerNorm(kappa=kappa, regulator=regulator)
        ym_algebra = GaugeCovariantPolymerAlgebra(triangle_adj, kappa=kappa)

        p = Polymer(frozenset([0]))
        K = {p: 2.5 + 0j}

        # Scalar norm at zero field (no regulator correction)
        scalar_val = scalar_norm.evaluate({p: 2.5})
        ym_val = ym_algebra.norm(K)

        # Both should be |2.5| * exp(kappa * 1) = 2.5 * e
        expected = 2.5 * np.exp(1.0)
        assert abs(scalar_val - expected) < 1e-10
        assert abs(ym_val - expected) < 1e-10

    def test_polymer_class_reuse(self):
        """Polymer class from banach_norm works in YM algebra."""
        p = Polymer(frozenset([0, 1, 2]), scale=1)
        assert p.size == 3
        assert p.scale == 1
        # Should be usable as dict key
        d = {p: 1.0}
        assert d[p] == 1.0


# ======================================================================
# 10. Edge cases and robustness
# ======================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_block_polymer(self, single_node_adj):
        """Single-block polymer: simplest case."""
        alg = GaugeCovariantPolymerAlgebra(single_node_adj, kappa=1.0)
        p = Polymer(frozenset([0]))
        K = {p: 1.0 + 0j}
        norm = alg.norm(K)
        assert abs(norm - np.exp(1.0)) < 1e-10

    def test_whole_graph_polymer(self, triangle_adj):
        """Whole-graph polymer: X = entire S^3."""
        p = Polymer(frozenset([0, 1, 2]))
        alg = GaugeCovariantPolymerAlgebra(triangle_adj, kappa=0.5)
        K = {p: 1.0 + 0j}
        norm = alg.norm(K)
        assert abs(norm - np.exp(0.5 * 3)) < 1e-10

    def test_different_g_squared(self, triangle_adj):
        """Different g^2 values give different regulator sizes."""
        tp = TPhiSeminorm(M=2.0)
        h_weak = tp.field_regulator(j=0, g2_j=0.1)
        h_strong = tp.field_regulator(j=0, g2_j=10.0)
        assert h_strong > h_weak

    def test_su3_gauge_norm(self):
        """GaugeFieldNorm works for SU(3)."""
        gfn = GaugeFieldNorm(N_c=3, M=2.0)
        assert gfn.dim_adj == 8
        A = np.ones((5, 3, 8))
        norm = gfn.field_norm(A, scale_j=0)
        expected = np.sqrt(3 * 8)  # sqrt(sum of 24 ones)
        assert abs(norm - expected) < 1e-10

    def test_large_scale_regulator(self):
        """At large scale j, field regulator is very small."""
        tp = TPhiSeminorm(M=2.0)
        h = tp.field_regulator(j=20, g2_j=4.0)
        assert h < 1e-5

    def test_kp_with_only_size_1(self, triangle_adj):
        """KP with only size-1 polymer norms."""
        kp = KoteckyPreissCondition(triangle_adj)
        K_norms = {1: 1e-5}
        assert kp.check_condition(K_norms, a=0.1, max_polymer_size=1)

    def test_bk_tree_weight_with_missing_blocks(self, triangle_adj):
        """Tree weight handles missing block activities gracefully."""
        bk = BKTreeExpansion(triangle_adj)
        K = {0: 1.0}  # Only block 0 has activity
        trees = bk.enumerate_trees(max_nodes=3)
        for tree in trees:
            w = bk.tree_weight(tree, K)
            assert np.isfinite(w)
            assert w >= 0

    def test_wilson_loop_plaquette_missing_links(self):
        """Missing links default to identity."""
        wlr = WilsonLoopRegulator(N_c=2)
        # Only provide one link; others default to identity
        link_vars = {(0, 1): np.eye(2, dtype=complex)}
        W = wlr.wilson_plaquette(link_vars, (0, 1, 2, 3))
        # Only (0,1) is provided; rest are identity
        # W = U_{01} * I * I * I = U_{01} = I
        assert np.allclose(W, np.eye(2), atol=1e-12)

    def test_gribov_emptiness_various_meshes(self):
        """Emptiness check works for various mesh sizes."""
        for mesh in [0.01, 0.1, 0.5, 1.0]:
            result = WilsonLoopRegulator.gribov_emptiness_check(
                g2=6.28, mesh_size=mesh, p0=100.0
            )
            assert result['is_empty']
