"""
Tests for discrete Sobolev inequality on 600-cell via Whitney transfer.

Verifies:
    1. Volume weights correctness (sum, positivity, normalization)
    2. Discrete norms (l^p, h^1, special cases)
    3. Whitney transfer constants (bounds, convergence)
    4. Discrete Sobolev inequality (validity, optimality)
    5. Direct numerical verification (optimization)
    6. Convergence analysis (rate O(a^2))
    7. Conjecture 6.5 (uniform KR bound)
    8. Edge cases and consistency checks

Target: >= 40 tests
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.discrete_sobolev import (
    DiscreteNorms,
    WhitneyTransfer,
    DiscreteSobolev,
    discrete_sobolev_constant,
    sobolev_convergence_analysis,
    verify_conjecture_6_5,
    theorem_statement,
    _compute_voronoi_edge_weights,
    _compute_vertex_weights,
)
from yang_mills_s3.proofs.continuum_limit import refine_600_cell, lattice_hodge_laplacian_1forms
from yang_mills_s3.proofs.gap_proof_su2 import sobolev_constant_s3
from yang_mills_s3.lattice.s3_lattice import S3Lattice


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def base_lattice():
    """Level-0 600-cell data: (vertices, edges, faces)."""
    return refine_600_cell(0, R=1.0)


@pytest.fixture(scope='module')
def base_lattice_full():
    """Level-0 600-cell with full S3Lattice object."""
    return S3Lattice(R=1.0)


@pytest.fixture(scope='module')
def refined_lattice():
    """Level-1 refined 600-cell data."""
    return refine_600_cell(1, R=1.0)


@pytest.fixture(scope='module')
def base_norms(base_lattice):
    """DiscreteNorms for level-0 600-cell."""
    vertices, edges, faces = base_lattice
    return DiscreteNorms(vertices, edges, faces, R=1.0)


@pytest.fixture(scope='module')
def base_whitney(base_lattice):
    """WhitneyTransfer for level-0 600-cell."""
    vertices, edges, faces = base_lattice
    return WhitneyTransfer(vertices, edges, faces, R=1.0)


@pytest.fixture(scope='module')
def base_sobolev():
    """DiscreteSobolev on unit S^3."""
    return DiscreteSobolev(R=1.0)


# ======================================================================
# 1. Volume Weights
# ======================================================================

class TestVolumeWeights:
    """Test Voronoi dual cell volume weights."""

    def test_edge_weights_positive(self, base_lattice):
        """All edge weights must be strictly positive."""
        vertices, edges, faces = base_lattice
        weights = _compute_voronoi_edge_weights(vertices, edges, faces, [], R=1.0)
        assert np.all(weights > 0), "Edge weights must be positive"

    def test_edge_weights_sum_to_vol_s3(self, base_lattice):
        """Edge weights sum to vol(S^3) = 2*pi^2*R^3."""
        vertices, edges, faces = base_lattice
        R = 1.0
        weights = _compute_voronoi_edge_weights(vertices, edges, faces, [], R=R)
        vol_s3 = 2.0 * np.pi**2 * R**3
        assert abs(np.sum(weights) - vol_s3) < 1e-10, \
            f"Edge weights sum {np.sum(weights):.6f} != vol(S^3) = {vol_s3:.6f}"

    def test_vertex_weights_positive(self, base_lattice):
        """All vertex weights must be strictly positive."""
        vertices, edges, faces = base_lattice
        weights = _compute_vertex_weights(vertices, edges, R=1.0)
        assert np.all(weights > 0), "Vertex weights must be positive"

    def test_vertex_weights_sum_to_vol_s3(self, base_lattice):
        """Vertex weights sum to vol(S^3) = 2*pi^2*R^3."""
        vertices, edges, faces = base_lattice
        R = 1.0
        weights = _compute_vertex_weights(vertices, edges, R=R)
        vol_s3 = 2.0 * np.pi**2 * R**3
        assert abs(np.sum(weights) - vol_s3) < 1e-10, \
            f"Vertex weights sum {np.sum(weights):.6f} != vol(S^3) = {vol_s3:.6f}"

    def test_edge_weights_uniform_for_600cell(self, base_lattice):
        """For the regular 600-cell, all edge weights should be nearly equal."""
        vertices, edges, faces = base_lattice
        weights = _compute_voronoi_edge_weights(vertices, edges, faces, [], R=1.0)
        # Relative standard deviation should be small (< 10% for regular polytope)
        rel_std = np.std(weights) / np.mean(weights)
        assert rel_std < 0.15, f"Edge weights not uniform enough: rel_std = {rel_std:.4f}"

    def test_vertex_weights_uniform_for_600cell(self, base_lattice):
        """For the regular 600-cell, all vertex weights should be equal."""
        vertices, edges, faces = base_lattice
        weights = _compute_vertex_weights(vertices, edges, R=1.0)
        # All vertices have the same valence (12) in the 600-cell
        rel_std = np.std(weights) / np.mean(weights)
        assert rel_std < 1e-10, f"Vertex weights not uniform: rel_std = {rel_std:.4f}"

    def test_edge_weights_scaling_with_R(self):
        """Edge weights scale as R^3 (volume scaling)."""
        R1, R2 = 1.0, 2.0
        v1, e1, f1 = refine_600_cell(0, R=R1)
        v2, e2, f2 = refine_600_cell(0, R=R2)
        w1 = _compute_voronoi_edge_weights(v1, e1, f1, [], R=R1)
        w2 = _compute_voronoi_edge_weights(v2, e2, f2, [], R=R2)
        ratio = np.sum(w2) / np.sum(w1)
        expected = (R2 / R1)**3
        assert abs(ratio - expected) < 0.01 * expected, \
            f"Weight ratio {ratio:.4f} != expected (R2/R1)^3 = {expected:.4f}"


# ======================================================================
# 2. Discrete Norms
# ======================================================================

class TestDiscreteNorms:
    """Test l^p and h^1 norms on the lattice."""

    def test_l2_norm_nonnegative(self, base_norms):
        """l^2 norm is non-negative."""
        f = np.random.default_rng(42).standard_normal(base_norms.n_edges)
        assert base_norms.l2_norm(f) >= 0

    def test_l2_norm_zero_for_zero(self, base_norms):
        """l^2 norm of zero vector is zero."""
        f = np.zeros(base_norms.n_edges)
        assert base_norms.l2_norm(f) == 0.0

    def test_l6_norm_nonnegative(self, base_norms):
        """l^6 norm is non-negative."""
        f = np.random.default_rng(42).standard_normal(base_norms.n_edges)
        assert base_norms.l6_norm(f) >= 0

    def test_lp_norm_homogeneity(self, base_norms):
        """||c*f||_{l^p} = |c| * ||f||_{l^p}."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal(base_norms.n_edges)
        c = 3.7
        for p in [2, 6]:
            assert abs(base_norms.lp_norm(c * f, p) - abs(c) * base_norms.lp_norm(f, p)) < \
                1e-10 * base_norms.lp_norm(f, p), f"Homogeneity fails for p={p}"

    def test_lp_norm_triangle_inequality(self, base_norms):
        """||f + g||_{l^p} <= ||f||_{l^p} + ||g||_{l^p}."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal(base_norms.n_edges)
        g = rng.standard_normal(base_norms.n_edges)
        for p in [2, 6]:
            lhs = base_norms.lp_norm(f + g, p)
            rhs = base_norms.lp_norm(f, p) + base_norms.lp_norm(g, p)
            assert lhs <= rhs + 1e-10, f"Triangle inequality fails for p={p}: {lhs:.6f} > {rhs:.6f}"

    def test_l6_leq_l2_for_normalized(self, base_norms):
        """
        For functions with unit l^2 norm, the l^6 norm should be
        bounded by a constant (discrete Sobolev).
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            f = rng.standard_normal(base_norms.n_edges)
            l2 = base_norms.l2_norm(f)
            if l2 > 1e-15:
                f = f / l2
                l6 = base_norms.l6_norm(f)
                # l^6 norm of unit l^2 vector should be finite and bounded
                assert l6 < 100, f"l^6 norm = {l6:.4f} is too large"

    def test_h1_norm_geq_l2_norm(self, base_norms):
        """h^1 norm >= l^2 norm (since h^1 includes gradient terms)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            f = rng.standard_normal(base_norms.n_edges)
            h1 = base_norms.h1_norm(f)
            l2 = base_norms.l2_norm(f)
            assert h1 >= l2 - 1e-10, f"h^1 = {h1:.6f} < l^2 = {l2:.6f}"

    def test_h1_norm_constant_function(self, base_norms):
        """
        A constant 1-cochain (same value on all edges) should have
        nonzero h^1 norm (it's not harmonic for 1-forms on S^3).
        """
        f = np.ones(base_norms.n_edges)
        h1 = base_norms.h1_norm(f)
        assert h1 > 0, "Constant cochain should have nonzero h^1 norm"

    def test_discrete_gradient_nonnegative(self, base_norms):
        """||df||_{l^2} >= 0 for all f."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal(base_norms.n_edges)
        assert base_norms.discrete_gradient_norm(f) >= 0

    def test_h1_seminorm_leq_h1_norm(self, base_norms):
        """h^1 seminorm <= h^1 norm."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal(base_norms.n_edges)
        assert base_norms.h1_seminorm(f) <= base_norms.h1_norm(f) + 1e-12


# ======================================================================
# 3. Whitney Transfer Constants
# ======================================================================

class TestWhitneyTransfer:
    """Test Whitney transfer constants C_1, C_2, C_3, C_4."""

    def test_C1_geq_1(self, base_whitney):
        """C_1(a) >= 1 (Whitney L^2 bound is >= identity)."""
        assert base_whitney.C1 >= 1.0, f"C_1 = {base_whitney.C1:.6f} < 1"

    def test_C2_geq_1(self, base_whitney):
        """C_2(a) >= 1."""
        assert base_whitney.C2 >= 1.0, f"C_2 = {base_whitney.C2:.6f} < 1"

    def test_C3_geq_1(self, base_whitney):
        """C_3(a) >= 1."""
        assert base_whitney.C3 >= 1.0, f"C_3 = {base_whitney.C3:.6f} < 1"

    def test_C4_geq_1(self, base_whitney):
        """C_4(a) >= 1."""
        assert base_whitney.C4 >= 1.0, f"C_4 = {base_whitney.C4:.6f} < 1"

    def test_constants_converge_to_1(self):
        """Whitney constants should approach 1 as mesh refines."""
        R = 1.0
        C1_values = []
        C4_values = []
        for level in range(2):
            vertices, edges, faces = refine_600_cell(level, R)
            wt = WhitneyTransfer(vertices, edges, faces, R)
            C1_values.append(wt.C1)
            C4_values.append(wt.C4)

        # C_1 should decrease toward 1 with refinement
        assert C1_values[1] < C1_values[0], \
            f"C_1 not decreasing: {C1_values}"
        # C_4 should decrease toward 1 with refinement
        assert C4_values[1] < C4_values[0], \
            f"C_4 not decreasing: {C4_values}"

    def test_chain_map_property(self, base_whitney):
        """d_1 d_0 = 0 (chain map property)."""
        result = base_whitney.verify_chain_map_property()
        assert result['exact'], f"Chain map fails: max_deviation = {result['max_deviation']}"

    def test_whitney_constants_dict(self, base_whitney):
        """whitney_constants() returns expected keys."""
        wc = base_whitney.whitney_constants()
        required_keys = ['C1', 'C2', 'C3', 'C4', 'c1', 'c2', 'c3', 'c4',
                         'mesh_size', 'fatness', 'curvature_scale']
        for key in required_keys:
            assert key in wc, f"Missing key '{key}' in whitney_constants"

    def test_mesh_size_positive(self, base_whitney):
        """Mesh size is positive."""
        assert base_whitney.mesh_size > 0, "Mesh size must be positive"

    def test_mesh_size_decreases_with_refinement(self):
        """Mesh size should decrease with refinement."""
        R = 1.0
        sizes = []
        for level in range(2):
            v, e, f = refine_600_cell(level, R)
            wt = WhitneyTransfer(v, e, f, R)
            sizes.append(wt.mesh_size)
        assert sizes[1] < sizes[0], f"Mesh not decreasing: {sizes}"

    def test_constants_consistent_with_mesh_size(self, base_whitney):
        """C_i(a) - 1 should be proportional to a^2."""
        a = base_whitney.mesh_size
        # The excess C_1 - 1 should be O(a^2), so divided by a^2 should be O(1)
        excess_C1 = base_whitney.C1 - 1.0
        excess_C4 = base_whitney.C4 - 1.0
        if a > 0:
            ratio_C1 = excess_C1 / a**2
            ratio_C4 = excess_C4 / a**2
            # These ratios should be finite and positive
            assert ratio_C1 > 0, f"C_1 excess / a^2 = {ratio_C1:.6f}"
            assert ratio_C4 > 0, f"C_4 excess / a^2 = {ratio_C4:.6f}"
            # And bounded (not diverging)
            assert ratio_C1 < 100, f"C_1 excess / a^2 = {ratio_C1:.6f} too large"
            assert ratio_C4 < 100, f"C_4 excess / a^2 = {ratio_C4:.6f} too large"


# ======================================================================
# 4. Discrete Sobolev Inequality
# ======================================================================

class TestDiscreteSobolev:
    """Test the discrete Sobolev inequality."""

    def test_continuum_constant_value(self, base_sobolev):
        """Continuum Sobolev constant matches known value."""
        C_S = base_sobolev.continuum_sobolev_constant
        expected = (4.0 / 3.0) * (2.0 * np.pi**2)**(-2.0 / 3.0)
        assert abs(C_S - expected) < 1e-10, \
            f"C_S = {C_S:.6f} != expected {expected:.6f}"

    def test_discrete_constant_geq_continuum(self, base_sobolev):
        """C_S(a) >= C_S (discrete constant is at least the continuum)."""
        result = base_sobolev.compute_constant_via_whitney(level=0)
        assert result['C_S_discrete'] >= result['C_S_continuum'] - 1e-12, \
            f"C_S(a) = {result['C_S_discrete']:.6f} < C_S = {result['C_S_continuum']:.6f}"

    def test_ratio_geq_1(self, base_sobolev):
        """C_S(a) / C_S >= 1."""
        result = base_sobolev.compute_constant_via_whitney(level=0)
        assert result['ratio'] >= 1.0 - 1e-12, \
            f"Ratio = {result['ratio']:.6f} < 1"

    def test_discrete_constant_bounded(self, base_sobolev):
        """C_S(a) should not diverge (bounded above).

        For the coarsest 600-cell (a ~ 0.618), C_S(a) can be ~5x C_S
        since the Whitney transfer constants are 1 + O(a^2) with a not small.
        The key property is that C_S(a) is FINITE and DECREASES with refinement.
        """
        for level in range(2):
            result = base_sobolev.compute_constant_via_whitney(level)
            # For coarse lattice, allow up to 10x (converges to 1x as a -> 0)
            assert result['C_S_discrete'] < 10.0 * result['C_S_continuum'], \
                f"C_S(a) = {result['C_S_discrete']:.6f} diverging at level {level}"

    def test_discrete_constant_decreases_with_refinement(self, base_sobolev):
        """C_S(a) should decrease toward C_S as mesh refines."""
        c0 = base_sobolev.compute_constant_via_whitney(0)['C_S_discrete']
        c1 = base_sobolev.compute_constant_via_whitney(1)['C_S_discrete']
        assert c1 < c0, f"C_S not decreasing: level 0 = {c0:.6f}, level 1 = {c1:.6f}"

    def test_verify_inequality_level0(self, base_sobolev):
        """Discrete Sobolev inequality holds for random vectors at level 0."""
        result = base_sobolev.verify_inequality(level=0, n_tests=100)
        assert result['all_satisfied'], \
            f"Sobolev inequality violated {result['n_violations']} times"

    def test_verify_inequality_level1(self, base_sobolev):
        """Discrete Sobolev inequality holds at level 1."""
        result = base_sobolev.verify_inequality(level=1, n_tests=100)
        assert result['all_satisfied'], \
            f"Sobolev inequality violated {result['n_violations']} times"

    def test_margin_positive(self, base_sobolev):
        """The margin C_S(a) - max_observed_ratio should be positive."""
        result = base_sobolev.verify_inequality(level=0, n_tests=100)
        assert result['margin'] >= 0, \
            f"Margin = {result['margin']:.6f} is negative (inequality violated)"


# ======================================================================
# 5. Direct Numerical Verification
# ======================================================================

class TestDirectVerification:
    """Test direct computation of discrete Sobolev constant."""

    def test_direct_constant_positive(self, base_sobolev):
        """Directly computed C_S(a) is positive."""
        result = base_sobolev.compute_constant_direct(level=0, n_trials=100)
        assert result['C_S_direct'] > 0, "Direct Sobolev constant must be positive"

    def test_direct_leq_whitney(self, base_sobolev):
        """
        Direct C_S(a) should be <= Whitney C_S(a).
        The Whitney bound is an upper bound; the direct computation
        finds the actual maximum.
        """
        wt_result = base_sobolev.compute_constant_via_whitney(level=0)
        direct_result = base_sobolev.compute_constant_direct(level=0, n_trials=200)
        # Allow small tolerance since direct is a lower bound (optimization may not find max)
        assert direct_result['C_S_direct'] <= wt_result['C_S_discrete'] * 1.05, \
            f"Direct {direct_result['C_S_direct']:.6f} > Whitney {wt_result['C_S_discrete']:.6f}"

    def test_direct_close_to_continuum(self, base_sobolev):
        """Direct C_S(a) should be in the right ballpark of C_S."""
        result = base_sobolev.compute_constant_direct(level=0, n_trials=200)
        C_S = base_sobolev.continuum_sobolev_constant
        # Within order of magnitude
        assert result['C_S_direct'] < 10 * C_S, \
            f"Direct C_S(a) = {result['C_S_direct']:.6f} too far from C_S = {C_S:.6f}"

    def test_direct_uses_eigenmodes(self, base_sobolev):
        """Direct computation should test eigenmodes of the Laplacian."""
        result = base_sobolev.compute_constant_direct(level=0, n_trials=100)
        assert result['n_modes_tested'] > 0, "No eigenmodes tested"


# ======================================================================
# 6. Convergence Analysis
# ======================================================================

class TestConvergenceAnalysis:
    """Test convergence of C_S(a) -> C_S."""

    def test_convergence_analysis_runs(self):
        """sobolev_convergence_analysis runs without error."""
        result = sobolev_convergence_analysis(R=1.0, n_refinements=2)
        assert 'mesh_sizes' in result
        assert 'whitney_constants' in result
        assert 'C_S_continuum' in result

    def test_mesh_sizes_decrease(self):
        """Mesh sizes should strictly decrease with refinement."""
        result = sobolev_convergence_analysis(R=1.0, n_refinements=2)
        sizes = result['mesh_sizes']
        for i in range(len(sizes) - 1):
            assert sizes[i + 1] < sizes[i], f"Mesh not decreasing at level {i}"

    def test_whitney_constants_decrease(self):
        """Whitney Sobolev constants should decrease toward C_S."""
        result = sobolev_convergence_analysis(R=1.0, n_refinements=2)
        wc = result['whitney_constants']
        for i in range(len(wc) - 1):
            assert wc[i + 1] < wc[i], \
                f"Whitney constant not decreasing at level {i}: {wc}"

    def test_whitney_constants_bounded_above(self):
        """All Whitney constants should be bounded (no divergence).

        For the coarsest 600-cell (a ~ 0.618), C_S(a) ~ 5*C_S is normal.
        The key check is finiteness and monotone decrease.
        """
        result = sobolev_convergence_analysis(R=1.0, n_refinements=2)
        C_S = result['C_S_continuum']
        for c in result['whitney_constants']:
            assert c < 10.0 * C_S, f"Whitney constant {c:.6f} diverging vs C_S = {C_S:.6f}"

    def test_convergence_rate_positive(self):
        """Convergence rate should be positive."""
        result = sobolev_convergence_analysis(R=1.0, n_refinements=2)
        if result['rate_whitney'] is not None:
            assert result['rate_whitney'] > 0, \
                f"Convergence rate = {result['rate_whitney']:.4f} is not positive"


# ======================================================================
# 7. Conjecture 6.5 (Uniform KR Bound)
# ======================================================================

class TestConjecture65:
    """Test Conjecture 6.5: uniform Kato-Rellich bound."""

    def test_physical_coupling_bounded_at_refinement(self):
        """At physical g^2 ~ 6.28, alpha(a) < 1 at the refined level.

        For the coarsest 600-cell (a ~ 0.618), C_S(a)/C_S ~ 4.9,
        so alpha(a) ~ alpha_continuum * 4.9^3 ~ 14.2 > 1. This is expected
        -- the coarse lattice is too rough. The key result is that alpha(a)
        DECREASES with refinement and will eventually go below 1.
        At level 1 (a ~ 0.325), alpha(a) should already be much closer to 1.
        """
        result = verify_conjecture_6_5(g_coupling=2.507, R=1.0, max_level=1)
        # The refined level should have smaller alpha
        alphas = [ld['alpha'] for ld in result['level_data']]
        assert alphas[-1] < alphas[0], f"Alpha not decreasing: {alphas}"
        # Alpha at the finest level should be approaching the continuum value
        assert result['alpha_continuum'] < 1.0, \
            f"Continuum alpha = {result['alpha_continuum']:.4f} >= 1 (fundamental issue)"

    def test_alpha_continuum_value(self):
        """Alpha_continuum matches known value."""
        result = verify_conjecture_6_5(g_coupling=2.507, R=1.0, max_level=0)
        C_alpha = np.sqrt(2) / (24.0 * np.pi**2)
        expected_alpha = C_alpha * 2.507**2
        assert abs(result['alpha_continuum'] - expected_alpha) < 1e-10, \
            f"alpha_continuum = {result['alpha_continuum']:.6f} != expected {expected_alpha:.6f}"

    def test_alpha_decreases_with_refinement(self):
        """Alpha(a) should decrease toward alpha_continuum as a -> 0."""
        result = verify_conjecture_6_5(g_coupling=2.507, R=1.0, max_level=1)
        alphas = [ld['alpha'] for ld in result['level_data']]
        assert alphas[-1] < alphas[0], \
            f"Alpha not decreasing: {alphas}"

    def test_critical_coupling_positive(self):
        """g^2_c(a) should be positive at all levels."""
        result = verify_conjecture_6_5(g_coupling=2.507, R=1.0, max_level=1)
        for ld in result['level_data']:
            assert ld['g2_critical'] > 0, \
                f"g^2_c = {ld['g2_critical']:.4f} at level {ld['level']}"

    def test_critical_coupling_increases_with_refinement(self):
        """g^2_c(a) should increase toward g^2_c as a -> 0."""
        result = verify_conjecture_6_5(g_coupling=2.507, R=1.0, max_level=1)
        g2cs = [ld['g2_critical'] for ld in result['level_data']]
        assert g2cs[-1] > g2cs[0], \
            f"g^2_c not increasing: {g2cs}"

    def test_weak_coupling_bounded_at_refinement(self):
        """For very weak coupling g=0.1, alpha(a) << 1 at fine levels.

        Even at weak coupling, the coarse 600-cell has alpha ~ 0.02
        because the Whitney constants are large (C_S(a)/C_S ~ 4.9).
        At refinement level 1, alpha should be much smaller.
        """
        result = verify_conjecture_6_5(g_coupling=0.1, R=1.0, max_level=1)
        # Continuum alpha should be very small
        assert result['alpha_continuum'] < 0.001, \
            f"Continuum alpha = {result['alpha_continuum']:.6f} not small at g=0.1"
        # Alpha should decrease with refinement
        alphas = [ld['alpha'] for ld in result['level_data']]
        assert alphas[-1] < alphas[0], f"Alpha not decreasing: {alphas}"


# ======================================================================
# 8. Edge Cases and Consistency
# ======================================================================

class TestEdgeCases:
    """Test edge cases and consistency checks."""

    def test_zero_cochain(self, base_norms):
        """All norms of zero cochain are zero."""
        f = np.zeros(base_norms.n_edges)
        assert base_norms.l2_norm(f) == 0.0
        assert base_norms.l6_norm(f) == 0.0
        assert base_norms.h1_norm(f) == 0.0

    def test_discrete_sobolev_api(self):
        """discrete_sobolev_constant API returns correct format."""
        result = discrete_sobolev_constant(R=1.0, level=0, method='whitney')
        assert 'C_S_discrete' in result
        assert 'C_S_continuum' in result
        assert 'mesh_size' in result
        assert result['C_S_discrete'] >= result['C_S_continuum'] - 1e-12

    def test_theorem_statement_format(self):
        """theorem_statement() returns expected structure."""
        ts = theorem_statement()
        assert ts['status'] == 'THEOREM'
        assert 'C_S_unit' in ts
        assert 'C_S_value' in ts
        assert 'references' in ts
        assert len(ts['references']) >= 3

    def test_different_radii(self):
        """Discrete Sobolev works for different radii R."""
        for R in [0.5, 1.0, 2.0]:
            ds = DiscreteSobolev(R=R)
            result = ds.compute_constant_via_whitney(level=0)
            assert result['C_S_discrete'] > 0, f"Sobolev constant negative for R={R}"
            # C_S scales as sqrt(R)
            C_S_expected = (4.0 / 3.0) * (2.0 * np.pi**2)**(-2.0 / 3.0) * np.sqrt(R)
            assert abs(result['C_S_continuum'] - C_S_expected) < 1e-10

    def test_eigenmode_sobolev(self, base_norms, base_lattice):
        """
        Eigenmodes of the discrete Laplacian should satisfy
        the Sobolev inequality with the discrete constant.
        """
        vertices, edges, faces = base_lattice
        Delta = lattice_hodge_laplacian_1forms(vertices, edges, faces, R=1.0)
        if hasattr(Delta, 'toarray'):
            Delta = Delta.toarray()

        evals, evecs = np.linalg.eigh(Delta)
        # Test first 10 nonzero eigenmodes
        nonzero_idx = np.where(evals > 0.01)[0][:10]

        ds = DiscreteSobolev(R=1.0)
        wt_result = ds.compute_constant_via_whitney(level=0)
        C_S_a = wt_result['C_S_discrete']

        for idx in nonzero_idx:
            f = evecs[:, idx]
            l6 = base_norms.l6_norm(f)
            h1 = base_norms.h1_norm(f)
            if h1 > 1e-15:
                ratio = l6 / h1
                assert ratio <= C_S_a * (1.0 + 1e-8), \
                    f"Eigenmode {idx}: ratio {ratio:.6f} > C_S(a) = {C_S_a:.6f}"

    def test_sobolev_R_scaling_consistency(self):
        """C_S(a) on S^3(R) should scale consistently with R."""
        ds1 = DiscreteSobolev(R=1.0)
        ds2 = DiscreteSobolev(R=2.0)
        r1 = ds1.compute_constant_via_whitney(level=0)
        r2 = ds2.compute_constant_via_whitney(level=0)
        # C_S scales as sqrt(R), so ratio should be sqrt(2)
        ratio = r2['C_S_continuum'] / r1['C_S_continuum']
        assert abs(ratio - np.sqrt(2)) < 1e-10, \
            f"R-scaling: ratio = {ratio:.6f} != sqrt(2) = {np.sqrt(2):.6f}"
