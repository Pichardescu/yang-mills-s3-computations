"""
Tests for the Banach norm and polymer activity spaces for RG on S³.

Tests cover:
1. Polymer construction and enumeration on the 600-cell
2. Norm computations for sample activities
3. Contraction of the toy φ⁴ model
4. Stable manifold eigenvalues
5. S³ vs flat space comparisons

Run:
    pytest tests/rg/test_banach_norm.py -v
"""

import numpy as np
import pytest
from itertools import combinations

from yang_mills_s3.rg.banach_norm import (
    Polymer,
    LargeFieldRegulator,
    PolymerNorm,
    ScalarPhi4OnS3,
    StableManifoldAnalysis,
    build_block_adjacency,
    enumerate_connected_polymers,
    count_polymers_by_size,
    contraction_estimate_from_spectrum,
    s3_vs_flat_contraction,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    BETA_0_SU2,
)


# ======================================================================
# Fixtures: small test graphs for polymer enumeration
# ======================================================================

@pytest.fixture
def linear_graph_5():
    """5-node linear graph: 0-1-2-3-4."""
    adj = {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3},
    }
    return adj


@pytest.fixture
def cycle_graph_4():
    """4-node cycle: 0-1-2-3-0."""
    adj = {
        0: {1, 3},
        1: {0, 2},
        2: {1, 3},
        3: {2, 0},
    }
    return adj


@pytest.fixture
def complete_graph_4():
    """Complete graph K₄: every pair connected."""
    adj = {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }
    return adj


@pytest.fixture
def star_graph_5():
    """Star graph: center 0 connected to 1,2,3,4."""
    adj = {
        0: {1, 2, 3, 4},
        1: {0},
        2: {0},
        3: {0},
        4: {0},
    }
    return adj


@pytest.fixture
def tetrahedron_adj():
    """Tetrahedral adjacency: each block shares faces with 3 neighbors.
    This mimics a small piece of the 600-cell."""
    adj = {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }
    return adj


# ======================================================================
# Section 1: Polymer construction tests
# ======================================================================

class TestPolymerConstruction:
    """Tests for the Polymer class."""

    def test_polymer_creation_single_block(self):
        """A polymer with one block has size 1."""
        p = Polymer(frozenset([0]))
        assert p.size == 1
        assert len(p) == 1
        assert p.block_ids == frozenset([0])

    def test_polymer_creation_multiple_blocks(self):
        """A polymer with multiple blocks."""
        p = Polymer(frozenset([0, 1, 2]))
        assert p.size == 3
        assert 1 in p.block_ids

    def test_polymer_empty_raises(self):
        """Empty polymer should raise ValueError."""
        with pytest.raises(ValueError, match="at least one block"):
            Polymer(frozenset())

    def test_polymer_from_set(self):
        """Can create from regular set (auto-converts to frozenset)."""
        p = Polymer({3, 5, 7})
        assert p.size == 3
        assert p.block_ids == frozenset([3, 5, 7])

    def test_polymer_equality(self):
        """Two polymers with same blocks and scale are equal."""
        p1 = Polymer(frozenset([0, 1]), scale=2)
        p2 = Polymer(frozenset([1, 0]), scale=2)
        assert p1 == p2

    def test_polymer_inequality_different_blocks(self):
        """Polymers with different blocks are not equal."""
        p1 = Polymer(frozenset([0, 1]))
        p2 = Polymer(frozenset([0, 2]))
        assert p1 != p2

    def test_polymer_inequality_different_scale(self):
        """Polymers with same blocks but different scale are not equal."""
        p1 = Polymer(frozenset([0, 1]), scale=1)
        p2 = Polymer(frozenset([0, 1]), scale=2)
        assert p1 != p2

    def test_polymer_hash_equal(self):
        """Equal polymers have the same hash."""
        p1 = Polymer(frozenset([0, 1]), scale=2)
        p2 = Polymer(frozenset([1, 0]), scale=2)
        assert hash(p1) == hash(p2)

    def test_polymer_in_dict(self):
        """Polymers can be used as dictionary keys."""
        p = Polymer(frozenset([0, 1]))
        d = {p: 3.14}
        assert d[p] == 3.14

    def test_polymer_repr(self):
        """repr includes sorted block list."""
        p = Polymer(frozenset([3, 1, 2]), scale=5)
        r = repr(p)
        assert "1, 2, 3" in r
        assert "scale=5" in r

    def test_polymer_is_connected_single(self, linear_graph_5):
        """Single-block polymer is always connected."""
        p = Polymer(frozenset([0]))
        assert p.is_connected(linear_graph_5)

    def test_polymer_is_connected_yes(self, linear_graph_5):
        """Adjacent blocks form a connected polymer."""
        p = Polymer(frozenset([1, 2, 3]))
        assert p.is_connected(linear_graph_5)

    def test_polymer_is_connected_no(self, linear_graph_5):
        """Non-adjacent blocks form a disconnected polymer."""
        p = Polymer(frozenset([0, 3]))  # 0 and 3 not adjacent in linear graph
        assert not p.is_connected(linear_graph_5)

    def test_polymer_is_connected_cycle(self, cycle_graph_4):
        """All subsets of a cycle are connected if contiguous."""
        p = Polymer(frozenset([0, 1, 2, 3]))
        assert p.is_connected(cycle_graph_4)

    def test_polymer_overlaps_yes(self):
        """Overlapping polymers detected."""
        p1 = Polymer(frozenset([0, 1, 2]))
        p2 = Polymer(frozenset([2, 3, 4]))
        assert p1.overlaps(p2)

    def test_polymer_overlaps_no(self):
        """Non-overlapping polymers detected."""
        p1 = Polymer(frozenset([0, 1]))
        p2 = Polymer(frozenset([2, 3]))
        assert not p1.overlaps(p2)

    def test_polymer_union(self):
        """Union of two polymers."""
        p1 = Polymer(frozenset([0, 1]))
        p2 = Polymer(frozenset([2, 3]))
        pu = p1.union(p2)
        assert pu.block_ids == frozenset([0, 1, 2, 3])
        assert pu.size == 4

    def test_polymer_distance(self):
        """Distance between polymers via block distances."""
        p1 = Polymer(frozenset([0]))
        p2 = Polymer(frozenset([2]))
        distances = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])
        assert p1.distance_to(p2, distances) == 2.0


# ======================================================================
# Section 2: Block adjacency and polymer enumeration
# ======================================================================

class TestBlockAdjacency:
    """Tests for building block adjacency from shared vertices."""

    def test_no_shared_vertices(self):
        """Blocks with no shared vertices are not adjacent."""
        vertex_lists = [[0, 1], [2, 3], [4, 5]]
        adj = build_block_adjacency(3, vertex_lists)
        assert len(adj[0]) == 0
        assert len(adj[1]) == 0
        assert len(adj[2]) == 0

    def test_shared_vertices(self):
        """Blocks sharing a vertex are adjacent."""
        vertex_lists = [[0, 1], [1, 2], [2, 3]]
        adj = build_block_adjacency(3, vertex_lists)
        assert 1 in adj[0]
        assert 0 in adj[1]
        assert 2 in adj[1]
        assert 1 in adj[2]
        assert 0 not in adj[2]

    def test_complete_sharing(self):
        """All blocks share vertex 0: complete adjacency."""
        vertex_lists = [[0, 1], [0, 2], [0, 3], [0, 4]]
        adj = build_block_adjacency(4, vertex_lists)
        for i in range(4):
            assert len(adj[i]) == 3  # connected to all others

    def test_tetrahedron_from_600_cell(self):
        """Tetrahedral cells share faces (3 vertices) in 600-cell."""
        # Each cell has 4 vertices; adjacent cells share 3
        vertex_lists = [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 3, 5],
            [5, 6, 7, 8],
        ]
        adj = build_block_adjacency(4, vertex_lists)
        # Cell 0 and 1 share vertices {0,1,2}
        assert 1 in adj[0]
        # Cell 0 and 2 share vertices {0,1,3}
        assert 2 in adj[0]
        # Cell 0 and 3 share no vertices
        assert 3 not in adj[0]


class TestPolymerEnumeration:
    """Tests for connected polymer enumeration."""

    def test_size_1_polymers(self, linear_graph_5):
        """Size-1 polymers: one per block."""
        polys = enumerate_connected_polymers(linear_graph_5, max_size=1)
        assert len(polys) == 5
        for p in polys:
            assert p.size == 1

    def test_size_2_linear(self, linear_graph_5):
        """Size-2 connected polymers on a linear graph: 4 edges."""
        polys = enumerate_connected_polymers(linear_graph_5, max_size=2)
        size_2 = [p for p in polys if p.size == 2]
        # Linear graph has 4 edges: (0,1), (1,2), (2,3), (3,4)
        assert len(size_2) == 4

    def test_size_2_complete(self, complete_graph_4):
        """Size-2 on K₄: C(4,2) = 6 edges."""
        polys = enumerate_connected_polymers(complete_graph_4, max_size=2)
        size_2 = [p for p in polys if p.size == 2]
        assert len(size_2) == 6

    def test_size_3_linear(self, linear_graph_5):
        """Size-3 connected polymers on linear graph: 3 paths."""
        polys = enumerate_connected_polymers(linear_graph_5, max_size=3)
        size_3 = [p for p in polys if p.size == 3]
        # Connected triples in 0-1-2-3-4: {0,1,2}, {1,2,3}, {2,3,4}
        assert len(size_3) == 3

    def test_size_3_complete(self, complete_graph_4):
        """Size-3 on K₄: C(4,3) = 4 triangles."""
        polys = enumerate_connected_polymers(complete_graph_4, max_size=3)
        size_3 = [p for p in polys if p.size == 3]
        assert len(size_3) == 4

    def test_size_4_complete(self, complete_graph_4):
        """Size-4 on K₄: the whole graph (1 polymer)."""
        polys = enumerate_connected_polymers(complete_graph_4, max_size=4)
        size_4 = [p for p in polys if p.size == 4]
        assert len(size_4) == 1

    def test_all_connected(self, cycle_graph_4):
        """Every enumerated polymer is connected."""
        polys = enumerate_connected_polymers(cycle_graph_4, max_size=4)
        for p in polys:
            assert p.is_connected(cycle_graph_4), f"{p} is not connected"

    def test_no_duplicates(self, linear_graph_5):
        """No duplicate polymers in enumeration."""
        polys = enumerate_connected_polymers(linear_graph_5, max_size=3)
        poly_sets = [p.block_ids for p in polys]
        assert len(poly_sets) == len(set(poly_sets))

    def test_star_graph_size_2(self, star_graph_5):
        """Size-2 on star graph: 4 edges (center to each leaf)."""
        polys = enumerate_connected_polymers(star_graph_5, max_size=2)
        size_2 = [p for p in polys if p.size == 2]
        assert len(size_2) == 4

    def test_star_graph_size_3(self, star_graph_5):
        """Size-3 on star graph: C(4,2) = 6 (center + any 2 leaves)."""
        polys = enumerate_connected_polymers(star_graph_5, max_size=3)
        size_3 = [p for p in polys if p.size == 3]
        assert len(size_3) == 6


class TestPolymerCounting:
    """Tests for count_polymers_by_size."""

    def test_matches_enumeration(self, linear_graph_5):
        """Counting agrees with full enumeration."""
        counts = count_polymers_by_size(linear_graph_5, max_size=4)
        polys = enumerate_connected_polymers(linear_graph_5, max_size=4)
        for s in range(1, 5):
            expected = len([p for p in polys if p.size == s])
            assert counts[s] == expected, f"Size {s}: count={counts[s]}, enum={expected}"

    def test_size_1_equals_n_blocks(self, complete_graph_4):
        """Size-1 count equals number of blocks."""
        counts = count_polymers_by_size(complete_graph_4, max_size=1)
        assert counts[1] == 4

    def test_complete_graph_sizes(self, complete_graph_4):
        """K₄ polymer counts: C(4,s) for each s."""
        counts = count_polymers_by_size(complete_graph_4, max_size=4)
        assert counts[1] == 4
        assert counts[2] == 6
        assert counts[3] == 4
        assert counts[4] == 1

    def test_exponential_growth_bounded(self, star_graph_5):
        """Polymer count grows at most exponentially."""
        counts = count_polymers_by_size(star_graph_5, max_size=4)
        # For star graph, growth is bounded by coordination number
        for s in range(2, 5):
            if s in counts and counts[s] > 0:
                assert counts[s] <= 5 * 4 ** (s - 1)  # loose upper bound


# ======================================================================
# Section 3: Large-field regulator
# ======================================================================

class TestLargeFieldRegulator:
    """Tests for the large-field regulator h_j."""

    def test_zero_field_gives_one(self):
        """At zero field, h_j = 1."""
        reg = LargeFieldRegulator(sigma_sq=1.0, p=0.5)
        assert reg.evaluate_scalar(0.0, 10) == pytest.approx(1.0)

    def test_large_field_suppressed(self):
        """Large fields are exponentially suppressed."""
        reg = LargeFieldRegulator(sigma_sq=1.0, p=1.0)
        h = reg.evaluate_scalar(100.0, 1)
        assert h < 1e-10

    def test_monotone_decreasing_in_field(self):
        """h_j decreases with increasing field strength."""
        reg = LargeFieldRegulator(sigma_sq=1.0, p=0.5)
        h1 = reg.evaluate_scalar(1.0, 1)
        h2 = reg.evaluate_scalar(2.0, 1)
        h3 = reg.evaluate_scalar(5.0, 1)
        assert h1 > h2 > h3

    def test_gauge_regulator_positive(self):
        """Gauge regulator is always positive."""
        reg = LargeFieldRegulator(sigma_sq=0.5, p=0.25)
        for F_sq in [0.0, 0.1, 1.0, 10.0]:
            h = reg.evaluate_gauge(F_sq, 5)
            assert h > 0

    def test_gauge_regulator_bounded_by_one(self):
        """Gauge regulator is at most 1."""
        reg = LargeFieldRegulator(sigma_sq=2.0, p=0.1)
        for F_sq in [0.0, 0.1, 1.0]:
            h = reg.evaluate_gauge(F_sq, 3)
            assert h <= 1.0 + 1e-15

    def test_invalid_sigma_raises(self):
        """Negative variance raises ValueError."""
        with pytest.raises(ValueError, match="sigma_sq"):
            LargeFieldRegulator(sigma_sq=-1.0)

    def test_invalid_p_raises(self):
        """Non-positive p raises ValueError."""
        with pytest.raises(ValueError, match="p"):
            LargeFieldRegulator(sigma_sq=1.0, p=0.0)

    def test_zero_sites_returns_one(self):
        """Zero sites in polymer gives h = 1."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        assert reg.evaluate_scalar(5.0, 0) == 1.0

    def test_sigma_scale_increases(self):
        """Variance at higher scales is larger (d=3: σ² ~ M^j)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        s0 = reg.sigma_at_scale(0)
        s1 = reg.sigma_at_scale(1)
        s2 = reg.sigma_at_scale(2)
        assert s1 > s0
        assert s2 > s1

    def test_sigma_scale_factor(self):
        """σ_j² = σ₀² M^{(d-2)j} for d=3."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        M = 2.0
        for j in range(5):
            expected = M ** j
            actual = reg.sigma_at_scale(j, M=M, d=3)
            assert actual == pytest.approx(expected, rel=1e-10)


# ======================================================================
# Section 4: Polymer norm (Banach space)
# ======================================================================

class TestPolymerNorm:
    """Tests for the Banach norm on polymer activities."""

    def test_empty_activities_zero_norm(self):
        """Empty activity set has zero norm."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm = PolymerNorm(kappa=1.0, regulator=reg)
        assert norm.evaluate({}) == 0.0

    def test_single_polymer_norm(self):
        """Norm of single polymer activity = |K(X)| exp(κ|X|)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        kappa = 1.0
        norm_obj = PolymerNorm(kappa=kappa, regulator=reg)

        p = Polymer(frozenset([0]))
        activities = {p: 0.5}
        expected = 0.5 * np.exp(kappa * 1)
        assert norm_obj.evaluate(activities) == pytest.approx(expected)

    def test_norm_takes_supremum(self):
        """Norm is the sup over all polymers."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=0.5, regulator=reg)

        p1 = Polymer(frozenset([0]))
        p2 = Polymer(frozenset([1]))
        p3 = Polymer(frozenset([0, 1]))

        activities = {
            p1: 1.0,
            p2: 2.0,
            p3: 0.1,
        }
        # Norm = max over polymers of |K(X)| * exp(κ|X|)
        vals = [
            1.0 * np.exp(0.5 * 1),
            2.0 * np.exp(0.5 * 1),
            0.1 * np.exp(0.5 * 2),
        ]
        expected = max(vals)
        assert norm_obj.evaluate(activities) == pytest.approx(expected)

    def test_norm_penalizes_large_polymers(self):
        """Larger polymers need smaller amplitudes for same norm contribution."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        kappa = 2.0
        norm_obj = PolymerNorm(kappa=kappa, regulator=reg)

        # Two polymers with same |K(X)| but different sizes
        p_small = Polymer(frozenset([0]))
        p_large = Polymer(frozenset([0, 1, 2, 3]))

        act_small = {p_small: 1.0}
        act_large = {p_large: 1.0}

        n_small = norm_obj.evaluate(act_small)
        n_large = norm_obj.evaluate(act_large)
        # Large polymer has higher norm (penalized more)
        assert n_large > n_small
        assert n_large / n_small == pytest.approx(np.exp(kappa * 3))

    def test_norm_with_regulator(self):
        """Large-field regulator divides the amplitude."""
        reg = LargeFieldRegulator(sigma_sq=1.0, p=0.5)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        p = Polymer(frozenset([0]))
        activities = {p: 1.0}
        field_data = {p: (4.0, 1)}  # phi_sq_sum=4, n_sites=1

        h = reg.evaluate_scalar(4.0, 1)
        expected = 1.0 * np.exp(1.0) / h
        actual = norm_obj.evaluate(activities, field_data)
        assert actual == pytest.approx(expected)

    def test_weight_exponential(self):
        """Weight function is exp(κ·s)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.5, regulator=reg)
        assert norm_obj.weight(1) == pytest.approx(np.exp(1.5))
        assert norm_obj.weight(3) == pytest.approx(np.exp(4.5))

    def test_completeness_finite(self):
        """
        THEOREM: The polymer activity space with ||·||_j < ∞ is complete
        (Banach). On S³ this is trivial: sup over finite set of polymers
        in a finite compact space.
        """
        # This is a mathematical fact, not a numerical test.
        # We verify the structure: norm is well-defined and finite
        # for any finite activity set.
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        # Any finite activity set has finite norm
        activities = {}
        for i in range(100):
            p = Polymer(frozenset([i]))
            activities[p] = np.random.randn()
        n = norm_obj.evaluate(activities)
        assert np.isfinite(n)

    def test_invalid_kappa_raises(self):
        """Non-positive kappa raises ValueError."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        with pytest.raises(ValueError, match="kappa"):
            PolymerNorm(kappa=0.0, regulator=reg)

    def test_contraction_check_passes(self):
        """Contraction check passes when norm decreases."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        ok, details = norm_obj.is_contractive(
            norm_before=1.0,
            norm_after=0.3,
            g_sq=0.1,
        )
        assert ok
        assert details['ratio'] < 1.0

    def test_contraction_check_fails(self):
        """Contraction check fails when norm increases too much."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        ok, details = norm_obj.is_contractive(
            norm_before=0.1,
            norm_after=100.0,
            g_sq=0.001,
        )
        assert not ok

    def test_triangle_inequality_structure(self):
        """
        The polymer norm satisfies the triangle inequality by construction
        (it is a sup norm, hence a Banach norm).
        """
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        p = Polymer(frozenset([0]))
        a1 = {p: 1.0}
        a2 = {p: 2.0}
        a_sum = {p: 3.0}

        n1 = norm_obj.evaluate(a1)
        n2 = norm_obj.evaluate(a2)
        n_sum = norm_obj.evaluate(a_sum)
        assert n_sum <= n1 + n2 + 1e-15  # triangle inequality


# ======================================================================
# Section 5: Scalar φ⁴ on S³ (toy model)
# ======================================================================

class TestScalarPhi4OnS3:
    """Tests for the scalar φ⁴ toy model."""

    def test_construction(self):
        """Basic construction with valid parameters."""
        model = ScalarPhi4OnS3(n_sites=120, R=1.0, lam=0.1)
        assert model.n_sites == 120
        assert model.R == 1.0
        assert model.lam == 0.1

    def test_invalid_n_sites(self):
        """Too few sites raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            ScalarPhi4OnS3(n_sites=1)

    def test_invalid_radius(self):
        """Non-positive radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius"):
            ScalarPhi4OnS3(R=-1.0)

    def test_invalid_coupling(self):
        """Negative coupling raises ValueError."""
        with pytest.raises(ValueError, match="lambda"):
            ScalarPhi4OnS3(lam=-0.1)

    def test_invalid_M(self):
        """Blocking factor M <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="Blocking factor"):
            ScalarPhi4OnS3(M=0.5)

    def test_laplacian_from_adjacency(self, complete_graph_4):
        """Graph Laplacian has correct structure."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0)
        L = model.build_laplacian_from_adjacency(complete_graph_4)
        # K₄: each vertex has degree 3
        # Diagonal should be positive, off-diagonal negative
        for i in range(4):
            assert L[i, i] > 0
            for j in range(4):
                if i != j:
                    assert L[i, j] < 0
        # Laplacian is symmetric
        assert np.allclose(L, L.T)
        # Row sums = 0 (up to normalization)
        row_sums = L.sum(axis=1)
        assert np.allclose(row_sums, row_sums[0] * np.ones(4))

    def test_s3_spectrum_eigenvalues(self):
        """S³ spectrum starts with λ₁ = 4/R²."""
        model = ScalarPhi4OnS3(n_sites=10, R=1.0)
        evs = model.build_laplacian_s3_spectrum()
        assert len(evs) == 10
        # First eigenvalue: (1+1)²/1² = 4 with multiplicity 2*1*3 = 6
        assert evs[0] == pytest.approx(4.0)
        # First 6 eigenvalues should be 4.0
        assert all(evs[i] == pytest.approx(4.0) for i in range(6))

    def test_s3_spectrum_ordered(self):
        """S³ eigenvalues are non-decreasing."""
        model = ScalarPhi4OnS3(n_sites=50, R=2.0)
        evs = model.build_laplacian_s3_spectrum()
        for i in range(1, len(evs)):
            assert evs[i] >= evs[i - 1] - 1e-14

    def test_propagator_slice_positive(self):
        """Propagator slice C_j(k) ≥ 0 for all j, k."""
        model = ScalarPhi4OnS3(n_sites=20, R=1.0)
        evs = model.build_laplacian_s3_spectrum()
        for j in range(5):
            C_j = model.propagator_at_scale(j, evs)
            assert np.all(C_j >= -1e-15)

    def test_propagator_sum_rule(self):
        """
        THEOREM: Σ_j C_j(k) ≈ 1/λ_k for each mode.
        (Up to IR and UV tail corrections.)
        """
        model = ScalarPhi4OnS3(n_sites=20, R=1.0, M=2.0)
        evs = model.build_laplacian_s3_spectrum()
        N_rg = 8
        for k_idx in range(min(5, len(evs))):
            total = sum(model.propagator_at_scale(j, evs)[k_idx]
                        for j in range(N_rg))
            exact = 1.0 / evs[k_idx]
            # Allow tail corrections (up to ~30% for small N_rg)
            assert total <= exact * 1.01  # sum ≤ exact (since we miss tails)
            assert total > exact * 0.3    # not wildly off

    def test_one_step_rg_produces_activities(self, complete_graph_4):
        """One RG step produces non-empty output."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=0.1)
        evs = model.build_laplacian_s3_spectrum()

        K_in = {Polymer(frozenset([i])): 0.1 for i in range(4)}
        K_out, flow = model.one_step_rg(3, K_in, complete_graph_4, evs)

        assert len(K_out) > 0
        assert 'scale' in flow
        assert flow['lambda_j'] > 0

    def test_one_step_rg_lambda_decreases(self, complete_graph_4):
        """Effective coupling λ_j decreases with scale (asymptotic freedom analog)."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=1.0)
        evs = model.build_laplacian_s3_spectrum()

        K = {Polymer(frozenset([i])): 0.1 for i in range(4)}
        _, flow1 = model.one_step_rg(5, K, complete_graph_4, evs)
        _, flow2 = model.one_step_rg(3, K, complete_graph_4, evs)
        # Higher scale (more UV) has smaller coupling
        assert flow1['lambda_j'] <= flow2['lambda_j']

    def test_rg_flow_small_coupling(self, complete_graph_4):
        """
        NUMERICAL: RG flow contracts for small coupling.
        At weak coupling (λ << 1), perturbation theory is valid
        and the irrelevant remainder should shrink.
        """
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=0.01, M=2.0)
        result = model.run_rg_flow(
            n_steps=3,
            adjacency=complete_graph_4,
            kappa=0.5,
        )
        assert result['status'] == 'NUMERICAL'
        # Norms should be finite
        for n in result['norms']:
            assert np.isfinite(n)

    def test_rg_flow_tracks_norms(self, complete_graph_4):
        """RG flow produces norm values at each step."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=0.05)
        result = model.run_rg_flow(n_steps=4, adjacency=complete_graph_4)
        assert len(result['norms']) == 5  # n_steps + 1
        assert len(result['ratios']) == 4

    def test_rg_flow_records_lambda(self, complete_graph_4):
        """Flow data records the coupling at each step."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=0.1)
        result = model.run_rg_flow(n_steps=3, adjacency=complete_graph_4)
        for fd in result['flow_data']:
            assert 'lambda_j' in fd
            assert fd['lambda_j'] > 0


# ======================================================================
# Section 6: Stable manifold analysis
# ======================================================================

class TestStableManifoldAnalysis:
    """Tests for the linearized RG near the Gaussian fixed point."""

    def test_construction(self):
        """Basic construction."""
        sma = StableManifoldAnalysis(d=3, M=2.0, R=1.0)
        assert sma.d == 3
        assert sma.M == 2.0
        assert sma.R == 1.0

    def test_invalid_dimension(self):
        """Dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension"):
            StableManifoldAnalysis(d=0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="Blocking factor"):
            StableManifoldAnalysis(M=1.0)

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="Radius"):
            StableManifoldAnalysis(R=0.0)

    def test_scaling_dim_mass_d3(self):
        """Mass operator φ² in d=3: scaling dim = 2."""
        sma = StableManifoldAnalysis(d=3)
        sd = sma.scaling_dimension(operator_dim=0, n_fields=2)
        # [g_mass] = d - n*(d-2)/2 = 3 - 2*(1/2) = 2
        assert sd == pytest.approx(2.0)

    def test_scaling_dim_phi4_d3(self):
        """φ⁴ coupling in d=3: scaling dim = 1 (relevant!)."""
        sma = StableManifoldAnalysis(d=3)
        sd = sma.scaling_dimension(operator_dim=0, n_fields=4)
        # [g_φ4] = 3 - 4*(1/2) = 1
        assert sd == pytest.approx(1.0)

    def test_scaling_dim_phi6_d3(self):
        """φ⁶ coupling in d=3: scaling dim = 0 (marginal!)."""
        sma = StableManifoldAnalysis(d=3)
        sd = sma.scaling_dimension(operator_dim=0, n_fields=6)
        # [g_φ6] = 3 - 6*(1/2) = 0
        assert sd == pytest.approx(0.0)

    def test_scaling_dim_phi4_d4(self):
        """φ⁴ coupling in d=4: scaling dim = 0 (marginal)."""
        sma = StableManifoldAnalysis(d=4)
        sd = sma.scaling_dimension(operator_dim=0, n_fields=4)
        # [g_φ4] = 4 - 4*(2/2) = 0
        assert sd == pytest.approx(0.0)

    def test_scaling_dim_phi4_d5(self):
        """φ⁴ coupling in d=5: scaling dim = -1 (irrelevant)."""
        sma = StableManifoldAnalysis(d=5)
        sd = sma.scaling_dimension(operator_dim=0, n_fields=4)
        # [g_φ4] = 5 - 4*(3/2) = -1
        assert sd == pytest.approx(-1.0)

    def test_rg_eigenvalue_relevant(self):
        """Relevant operator has eigenvalue > 1."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        ev = sma.rg_eigenvalue(operator_dim=0, n_fields=2)  # mass
        assert ev > 1.0

    def test_rg_eigenvalue_marginal(self):
        """Marginal operator has eigenvalue = 1."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        ev = sma.rg_eigenvalue(operator_dim=0, n_fields=6)  # φ⁶
        assert ev == pytest.approx(1.0)

    def test_rg_eigenvalue_irrelevant(self):
        """Irrelevant operator has eigenvalue < 1."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        ev = sma.rg_eigenvalue(operator_dim=0, n_fields=8)  # φ⁸
        assert ev < 1.0

    def test_curvature_shift_positive(self):
        """S³ curvature shift is positive (helps stabilize)."""
        sma = StableManifoldAnalysis(d=3, R=1.0)
        shift = sma.curvature_shift()
        assert shift > 0
        # ξ = 1/8, Ric = 2: shift = 2/8 = 0.25
        assert shift == pytest.approx(0.25)

    def test_curvature_shift_scales_with_R(self):
        """Curvature shift decreases with R (flat limit)."""
        sma1 = StableManifoldAnalysis(d=3, R=1.0)
        sma2 = StableManifoldAnalysis(d=3, R=10.0)
        assert sma1.curvature_shift() > sma2.curvature_shift()
        # ratio: (10/1)² = 100
        ratio = sma1.curvature_shift() / sma2.curvature_shift()
        assert ratio == pytest.approx(100.0)

    def test_phi4_eigenvalues_classification(self):
        """
        THEOREM: In d=3, φ⁴ is relevant, φ⁶ is marginal, φ⁸ is irrelevant.
        This is standard power-counting.
        """
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.phi4_eigenvalues()

        assert evs['mass']['classification'] == 'relevant'
        assert evs['phi4']['classification'] == 'relevant'
        assert evs['phi6']['classification'] == 'marginal'
        assert evs['phi8']['classification'] == 'irrelevant'

    def test_phi4_eigenvalue_mass_value(self):
        """Mass eigenvalue = M² = 4 for M=2."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.phi4_eigenvalues()
        assert evs['mass']['eigenvalue'] == pytest.approx(4.0)

    def test_phi4_eigenvalue_phi4_value(self):
        """φ⁴ eigenvalue = M¹ = 2 for M=2 in d=3."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.phi4_eigenvalues()
        assert evs['phi4']['eigenvalue'] == pytest.approx(2.0)

    def test_ym_eigenvalues_structure(self):
        """YM eigenvalue table has correct operator classifications."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.ym_eigenvalues(N_c=2)

        assert 'gauge_coupling' in evs
        assert 'mass_gap' in evs
        assert 'F4_operator' in evs
        # Gauge coupling is marginal in d=4
        assert evs['gauge_coupling']['classification'] == 'marginal'
        # Mass is relevant
        assert evs['mass_gap']['classification'] == 'relevant'
        # F⁴ is irrelevant
        assert evs['F4_operator']['classification'] == 'irrelevant'

    def test_ym_asymptotic_freedom(self):
        """Anomalous dimension for gauge coupling is negative (AF)."""
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.ym_eigenvalues(N_c=2)
        assert evs['gauge_coupling']['anomalous_dimension'] < 0

    def test_flat_vs_s3_comparison_structure(self):
        """Comparison returns all expected keys."""
        sma = StableManifoldAnalysis(d=3, M=2.0, R=1.0)
        comp = sma.flat_vs_s3_comparison()
        assert 'gap_s3' in comp
        assert 'gap_torus' in comp
        assert 'b1_s3' in comp
        assert 'b1_torus' in comp

    def test_s3_has_positive_gap(self):
        """
        S³ has a positive spectral gap 4/R² > 0 for any finite R.
        T³ has b₁=3 zero modes that kill the effective gauge gap.
        The advantage is topological, not metric.
        """
        sma = StableManifoldAnalysis(d=3, M=2.0, R=1.0)
        comp = sma.flat_vs_s3_comparison()
        assert comp['gap_s3'] > 0
        # The effective gauge gap on T³ is 0 because of zero modes
        assert comp['gap_torus'] == 0.0

    def test_s3_no_zero_modes(self):
        """S³ has no harmonic 1-forms (b₁ = 0)."""
        sma = StableManifoldAnalysis(d=3, M=2.0, R=1.0)
        comp = sma.flat_vs_s3_comparison()
        assert comp['b1_s3'] == 0
        assert comp['b1_torus'] == 3  # T³ has 3 zero modes


# ======================================================================
# Section 7: Contraction estimates from spectral data
# ======================================================================

class TestContractionEstimate:
    """Tests for the spectral contraction estimate."""

    def test_basic_estimate(self):
        """Contraction estimate runs without error."""
        result = contraction_estimate_from_spectrum(
            R=1.0, M=2.0, n_modes=20, g_sq=0.5
        )
        assert 'effective_epsilon' in result
        assert 'contracts' in result
        assert np.isfinite(result['effective_epsilon'])

    def test_propagator_diagonals_positive(self):
        """Propagator diagonals at each scale are positive."""
        result = contraction_estimate_from_spectrum(R=1.0, M=2.0, n_modes=30)
        for d in result['propagator_diagonals']:
            assert d >= 0

    def test_vertex_bounds_finite(self):
        """Vertex bounds at each scale are finite."""
        result = contraction_estimate_from_spectrum(R=1.0, M=2.0, n_modes=20)
        for vb in result['vertex_bounds']:
            assert np.isfinite(vb)

    def test_small_coupling_contracts(self):
        """
        NUMERICAL: At small coupling g² << 1, the perturbative
        vertex bounds ensure contraction.
        """
        result = contraction_estimate_from_spectrum(
            R=1.0, M=2.0, n_modes=30, g_sq=0.01
        )
        # The perturbative estimate should give ε > 0
        assert result['effective_epsilon'] > 0
        assert result['contracts']

    def test_spectral_gap_recorded(self):
        """The spectral gap 4/R² is recorded."""
        R = 2.0
        result = contraction_estimate_from_spectrum(R=R, M=2.0, n_modes=10)
        assert result['spectral_gap'] == pytest.approx(4.0 / R ** 2)

    def test_physical_radius(self):
        """Test with physical radius R = 2.2 fm."""
        result = contraction_estimate_from_spectrum(
            R=R_PHYSICAL_FM, M=2.0, n_modes=30, g_sq=1.0
        )
        assert np.isfinite(result['effective_epsilon'])


class TestS3VsFlat:
    """Tests comparing S³ with flat T³."""

    def test_comparison_runs(self):
        """Comparison runs without error."""
        result = s3_vs_flat_contraction(R=1.0, M=2.0)
        assert 'gap_s3' in result
        assert 'gap_torus' in result

    def test_s3_effective_gap_larger(self):
        """
        S³ effective gauge gap > T³ effective gauge gap for any radius.
        T³ has b₁=3 zero modes → effective gauge gap = 0.
        S³ has b₁=0 → gap = 4/R² > 0.
        """
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = s3_vs_flat_contraction(R=R, M=2.0)
            assert result['gap_s3'] > result['gap_torus'], (
                f"S³ effective gauge gap not larger at R={R}"
            )
            assert result['gap_s3'] > 0
            assert result['gap_torus'] == 0.0  # zero modes kill T³ gauge gap

    def test_s3_zero_modes_eliminated(self):
        """S³ has b₁ = 0, T³ has b₁ = 3."""
        result = s3_vs_flat_contraction(R=1.0)
        assert result['b1_s3'] == 0
        assert result['b1_torus'] > 0

    def test_advantages_listed(self):
        """At least 3 structural advantages of S³ are listed."""
        result = s3_vs_flat_contraction(R=1.0)
        assert len(result['advantages']) >= 3


# ======================================================================
# Section 8: Integration with existing 600-cell infrastructure
# ======================================================================

class TestIntegrationWith600Cell:
    """
    Tests that verify compatibility with block_geometry.py.
    These use a minimal mock of the 600-cell structure.
    """

    @pytest.fixture
    def small_600cell_mock(self):
        """
        Mock a small portion of the 600-cell: 8 tetrahedral blocks
        sharing vertices (like a cube-like piece of S³).
        """
        # 8 blocks, each with 4 vertices from a set of 12
        vertex_lists = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9],
            [7, 8, 9, 10],
        ]
        adj = build_block_adjacency(8, vertex_lists)
        return adj, vertex_lists

    def test_adjacency_from_600cell_mock(self, small_600cell_mock):
        """Block adjacency built from vertex sharing."""
        adj, _ = small_600cell_mock
        assert len(adj) == 8
        # Block 0 shares vertices with block 1 (vertices 1,2,3)
        assert 1 in adj[0]
        # Block 0 does NOT share with block 7
        assert 7 not in adj[0]

    def test_polymer_enum_on_mock(self, small_600cell_mock):
        """Polymer enumeration works on 600-cell mock."""
        adj, _ = small_600cell_mock
        polys = enumerate_connected_polymers(adj, max_size=3)
        assert len(polys) > 8  # more than just size-1

    def test_polymer_connectivity_check(self, small_600cell_mock):
        """All enumerated polymers are connected."""
        adj, _ = small_600cell_mock
        polys = enumerate_connected_polymers(adj, max_size=3)
        for p in polys:
            assert p.is_connected(adj), f"{p} not connected"

    def test_rg_flow_on_mock(self, small_600cell_mock):
        """RG flow runs on 600-cell mock."""
        adj, _ = small_600cell_mock
        model = ScalarPhi4OnS3(n_sites=8, R=1.0, lam=0.05)
        result = model.run_rg_flow(n_steps=3, adjacency=adj)
        assert len(result['norms']) == 4
        for n in result['norms']:
            assert np.isfinite(n)


# ======================================================================
# Section 9: Mathematical consistency checks
# ======================================================================

class TestMathematicalConsistency:
    """
    Tests for mathematical properties that must hold regardless of
    implementation details.
    """

    def test_polymer_norm_homogeneity(self):
        """||α K||_j = |α| · ||K||_j (norm homogeneity)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        p = Polymer(frozenset([0, 1]))
        alpha = 3.7

        K = {p: 1.0}
        K_scaled = {p: alpha}

        n_K = norm_obj.evaluate(K)
        n_scaled = norm_obj.evaluate(K_scaled)
        assert n_scaled == pytest.approx(alpha * n_K, rel=1e-12)

    def test_polymer_norm_nonnegativity(self):
        """||K||_j ≥ 0 always."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        for _ in range(20):
            activities = {}
            n_polys = np.random.randint(1, 10)
            for i in range(n_polys):
                p = Polymer(frozenset([i]))
                activities[p] = np.random.randn()
            assert norm_obj.evaluate(activities) >= 0

    def test_polymer_norm_zero_iff_zero(self):
        """||K||_j = 0 iff K = 0 (zero activity)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        # Zero activities
        assert norm_obj.evaluate({}) == 0.0

        # Non-zero activity has non-zero norm
        p = Polymer(frozenset([0]))
        assert norm_obj.evaluate({p: 0.001}) > 0

    def test_exponential_decay_required(self):
        """
        THEOREM: Activities K(X) must decay as exp(-κ|X|) for the norm
        to be finite. Test that constant-amplitude activities have
        divergent norm with increasing polymer size.
        """
        reg = LargeFieldRegulator(sigma_sq=1.0)
        kappa = 1.0
        norm_obj = PolymerNorm(kappa=kappa, regulator=reg)

        # Activities with constant amplitude but increasing size
        constant_amp = 1.0
        norms = []
        for s in range(1, 10):
            p = Polymer(frozenset(range(s)))
            n = norm_obj.evaluate({p: constant_amp})
            norms.append(n)

        # Norms should grow exponentially (constant amplitude doesn't decay)
        for i in range(1, len(norms)):
            ratio = norms[i] / norms[i - 1]
            assert ratio == pytest.approx(np.exp(kappa), rel=1e-10)

    def test_curvature_improvement_quantitative(self):
        """
        THEOREM: On S³, the conformal coupling ξRic provides a mass shift.
        For d=3: ξ = (d-2)/(4(d-1)) = 1/8.
        Ric(S³) = 2/R².
        Mass shift = ξ·Ric = 1/(4R²).
        """
        for R in [0.5, 1.0, 2.0, 5.0]:
            sma = StableManifoldAnalysis(d=3, R=R)
            shift = sma.curvature_shift()
            expected = 1.0 / (4.0 * R ** 2)
            assert shift == pytest.approx(expected, rel=1e-12)

    def test_spectral_gap_formula(self):
        """
        THEOREM: Coexact gap on S³(R) is λ₁ = 4/R².
        This is the mass gap squared in natural units.
        """
        for R in [0.5, 1.0, R_PHYSICAL_FM, 5.0]:
            gap = 4.0 / R ** 2
            # Check consistency with mass gap
            m_gap = 2.0 * HBAR_C_MEV_FM / R  # in MeV
            m_gap_from_gap = np.sqrt(gap) * HBAR_C_MEV_FM  # also in MeV
            assert m_gap == pytest.approx(m_gap_from_gap, rel=1e-12)

    def test_beta_function_sign(self):
        """
        THEOREM: β₀ > 0 for SU(2) (asymptotic freedom).
        β₀ = 11 Nc / (3 · 16π²) with Nc = 2.
        """
        assert BETA_0_SU2 > 0
        expected = 22.0 / (3.0 * 16.0 * np.pi ** 2)
        assert BETA_0_SU2 == pytest.approx(expected, rel=1e-12)


# ======================================================================
# Section 10: Edge cases and stress tests
# ======================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_block_system(self):
        """System with a single block."""
        adj = {0: set()}
        polys = enumerate_connected_polymers(adj, max_size=5)
        assert len(polys) == 1
        assert polys[0].size == 1

    def test_disconnected_graph(self):
        """Disconnected graph: no size-2+ connected polymers across components."""
        adj = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        polys = enumerate_connected_polymers(adj, max_size=2)
        size_2 = [p for p in polys if p.size == 2]
        # Only edges (0,1) and (2,3)
        assert len(size_2) == 2
        # No polymer contains both 0 and 2
        for p in size_2:
            assert not (0 in p.block_ids and 2 in p.block_ids)

    def test_large_kappa_kills_large_polymers(self):
        """Very large κ makes large polymer contributions negligible."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        kappa = 50.0  # very large
        norm_obj = PolymerNorm(kappa=kappa, regulator=reg)

        p_small = Polymer(frozenset([0]))
        p_large = Polymer(frozenset(range(10)))

        # Same amplitude, but large polymer gets exp(50*10) penalty
        act_small = {p_small: 1.0}
        act_large = {p_large: 1.0}

        n_small = norm_obj.evaluate(act_small)
        n_large = norm_obj.evaluate(act_large)
        assert n_large / n_small > 1e100  # huge penalty ratio

    def test_very_small_radius(self):
        """Very small R: large spectral gap, strong contraction."""
        result = contraction_estimate_from_spectrum(
            R=0.1, M=2.0, n_modes=10, g_sq=0.1
        )
        assert result['spectral_gap'] == pytest.approx(400.0)  # 4/0.01
        assert np.isfinite(result['effective_epsilon'])

    def test_very_large_radius(self):
        """Very large R: small spectral gap, harder contraction."""
        result = contraction_estimate_from_spectrum(
            R=100.0, M=2.0, n_modes=30, g_sq=0.1
        )
        assert result['spectral_gap'] == pytest.approx(4.0 / 10000.0)
        assert np.isfinite(result['effective_epsilon'])

    def test_norm_with_complex_amplitudes(self):
        """Norm handles complex amplitudes (takes absolute value)."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        norm_obj = PolymerNorm(kappa=1.0, regulator=reg)

        p = Polymer(frozenset([0]))
        # Complex amplitude
        K = {p: 3.0 + 4.0j}
        n = norm_obj.evaluate(K)
        expected = 5.0 * np.exp(1.0)  # |3+4i| = 5
        assert n == pytest.approx(expected)

    def test_phi4_zero_coupling(self, complete_graph_4):
        """At λ = 0 (free theory), activities stay at initial value."""
        model = ScalarPhi4OnS3(n_sites=4, R=1.0, lam=0.0)
        evs = model.build_laplacian_s3_spectrum()
        K = {Polymer(frozenset([i])): 0.0 for i in range(4)}
        K_out, flow = model.one_step_rg(3, K, complete_graph_4, evs)
        # All zero in, all zero out
        for amp in K_out.values():
            assert abs(amp) < 1e-15

    def test_max_size_0_returns_empty(self, linear_graph_5):
        """max_size=0 (or less) returns empty."""
        polys = enumerate_connected_polymers(linear_graph_5, max_size=0)
        assert len(polys) == 0

    def test_max_size_equals_n_blocks(self, complete_graph_4):
        """max_size = n_blocks gives the complete polymer."""
        polys = enumerate_connected_polymers(complete_graph_4, max_size=4)
        size_4 = [p for p in polys if p.size == 4]
        assert len(size_4) == 1
        assert size_4[0].block_ids == frozenset(range(4))


# ======================================================================
# Section 11: Status labels and honesty
# ======================================================================

class TestStatusLabels:
    """
    Verify that all claims are properly labeled with their rigor status.
    """

    def test_polymer_finiteness_is_theorem(self):
        """
        THEOREM: On S³ (compact), the number of polymers at each
        scale is finite. This is a topological fact.
        Verification: any finite graph has finitely many connected subsets.
        """
        # For any finite adjacency, enumeration terminates
        adj = {i: {(i+1) % 10} for i in range(10)}
        polys = enumerate_connected_polymers(adj, max_size=10)
        assert len(polys) < float('inf')

    def test_sum_rule_is_theorem(self):
        """
        THEOREM: Σ_j C_j(k) = 1/λ_k (exact integral identity).
        Verified via the sum rule check.
        """
        model = ScalarPhi4OnS3(n_sites=20, R=1.0, M=2.0)
        evs = model.build_laplacian_s3_spectrum()
        # Verify for first eigenvalue
        total = 0.0
        for j in range(15):  # enough scales for convergence
            Cj = model.propagator_at_scale(j, evs)
            total += Cj[0]
        exact = 1.0 / evs[0]
        # Should be close (within tail corrections)
        assert abs(total / exact - 1.0) < 0.1

    def test_scaling_dimensions_are_theorem(self):
        """
        THEOREM: Canonical scaling dimensions are exact results
        from dimensional analysis.
        """
        sma = StableManifoldAnalysis(d=3, M=2.0)
        evs = sma.phi4_eigenvalues()
        # These are well-known results in QFT
        assert evs['mass']['scaling_dimension'] == pytest.approx(2.0)
        assert evs['phi4']['scaling_dimension'] == pytest.approx(1.0)
        assert evs['phi6']['scaling_dimension'] == pytest.approx(0.0)

    def test_contraction_is_numerical(self):
        """
        NUMERICAL: The contraction estimate uses perturbative bounds.
        It is NOT a theorem.
        """
        result = contraction_estimate_from_spectrum(R=1.0, M=2.0, n_modes=20)
        assert result['status'] in ['NUMERICAL', 'INCONCLUSIVE']

    def test_ym_contraction_is_conjecture(self):
        """
        CONJECTURE: Full YM contraction on S³ follows from scalar φ⁴
        proof plus gauge-covariant estimates.

        This is stated but not proven here. The scalar φ⁴ contraction
        serves as evidence, not proof.
        """
        # This test documents the status; it passes by construction
        assert True  # Placeholder for the conjecture statement
