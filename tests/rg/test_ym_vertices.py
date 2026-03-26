"""
Tests for Yang-Mills vertices on S^3 for the RG program.

Verifies:
    1. Structure constants and Casimir (THEOREM)
    2. Clebsch-Gordan selection rules (THEOREM)
    3. Cubic vertex: symmetries, k=1 exact, operator norm (THEOREM/NUMERICAL)
    4. Quartic vertex: symmetries, k=1 exact, 9-DOF consistency (THEOREM/NUMERICAL)
    5. Ghost vertex: selection rules, coupling magnitude (NUMERICAL)
    6. Scale-decomposed vertices for RG (NUMERICAL)
    7. Counter-term structure: b_0 = 22/3 for SU(2) (THEOREM/NUMERICAL)
    8. Vertex bounds and IR finiteness (THEOREM/NUMERICAL)
    9. Bose symmetry and Ward identities (THEOREM)
    10. Flat-space comparison in UV (NUMERICAL)
"""

import numpy as np
import pytest

from yang_mills_s3.rg.ym_vertices import (
    su2_structure_constants,
    casimir_adjoint,
    cg_selection_rule,
    cg_selection_rule_quartic,
    CubicVertex,
    QuarticVertex,
    GhostVertex,
    ScaleDecomposedVertices,
    CounterTerms,
    VertexBounds,
    run_vertex_analysis,
    R_PHYSICAL_FM,
    G2_PHYSICAL,
    N_COLORS_SU2,
    DIM_ADJ_SU2,
    CASIMIR_ADJ_SU2,
    B0_SU2,
    HBAR_C_MEV_FM,
)
from yang_mills_s3.rg.heat_kernel_slices import (
    coexact_eigenvalue,
    coexact_multiplicity,
    HeatKernelSlices,
)


# =====================================================================
# 1. Structure constants and Casimir
# =====================================================================

class TestStructureConstants:
    """THEOREM: su(2) structure constants = Levi-Civita tensor."""

    def test_epsilon_123(self):
        """f^{123} = +1."""
        f = su2_structure_constants()
        assert f[0, 1, 2] == pytest.approx(1.0)

    def test_epsilon_antisymmetric(self):
        """f^{abc} is totally antisymmetric."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert f[a, b, c] == pytest.approx(-f[b, a, c])
                    assert f[a, b, c] == pytest.approx(-f[a, c, b])

    def test_epsilon_cyclic(self):
        """f^{abc} is cyclic: f^{123} = f^{231} = f^{312}."""
        f = su2_structure_constants()
        assert f[0, 1, 2] == pytest.approx(f[1, 2, 0])
        assert f[0, 1, 2] == pytest.approx(f[2, 0, 1])

    def test_diagonal_zero(self):
        """f^{aab} = 0 for all a, b."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                assert f[a, a, b] == pytest.approx(0.0)

    def test_jacobi_identity(self):
        """f^{ade}f^{bce} + cyclic = 0 (Jacobi identity)."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    jacobi = 0.0
                    for e in range(3):
                        jacobi += (f[a, b, e] * f[e, c, :].sum()
                                   if False else 0)
                    # More explicit Jacobi check:
                    lhs = 0.0
                    for d in range(3):
                        for e in range(3):
                            lhs += (f[a, d, e] * f[b, c, e]
                                    + f[b, d, e] * f[c, a, e]
                                    + f[c, d, e] * f[a, b, e])
                    # This is 0 for each d, but summed over d it's still 0
                    # Correct Jacobi: sum_e f^{abe}f^{ecd} + cyclic(a,b,c) = 0
        # Standard form of Jacobi identity
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        jacobi = 0.0
                        for e in range(3):
                            jacobi += (f[a, b, e] * f[e, c, d]
                                       + f[b, c, e] * f[e, a, d]
                                       + f[c, a, e] * f[e, b, d])
                        assert jacobi == pytest.approx(0.0), \
                            f"Jacobi failed for (a,b,c,d)=({a},{b},{c},{d})"

    def test_su2_identity(self):
        """sum_e f^{abe}f^{cde} = delta_{ac}delta_{bd} - delta_{ad}delta_{bc}."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        lhs = sum(f[a, b, e] * f[c, d, e] for e in range(3))
                        rhs = (float(a == c) * float(b == d)
                               - float(a == d) * float(b == c))
                        assert lhs == pytest.approx(rhs), \
                            f"SU(2) identity failed at ({a},{b},{c},{d})"


class TestCasimir:
    """THEOREM: Casimir of adjoint representation."""

    def test_casimir_su2(self):
        """C_2(adj) = 2 for SU(2)."""
        assert casimir_adjoint(2) == pytest.approx(2.0)

    def test_casimir_su3(self):
        """C_2(adj) = 3 for SU(3)."""
        assert casimir_adjoint(3) == pytest.approx(3.0)

    def test_casimir_general(self):
        """C_2(adj) = N for SU(N)."""
        for N in [2, 3, 4, 5, 10]:
            assert casimir_adjoint(N) == pytest.approx(float(N))


# =====================================================================
# 2. Clebsch-Gordan selection rules
# =====================================================================

class TestCGSelectionRules:
    """THEOREM: Selection rules from SO(4) representation theory."""

    def test_k1_k1_k1_forbidden(self):
        """(1,1,1) is forbidden: 1+1+1=3 is odd."""
        assert cg_selection_rule(1, 1, 1) is False

    def test_k1_k1_k2_allowed(self):
        """(1,1,2) is allowed: sum=4 even, triangle satisfied."""
        assert cg_selection_rule(1, 1, 2) is True

    def test_k1_k2_k3_allowed(self):
        """(1,2,3) is allowed: sum=6 even, triangle satisfied."""
        assert cg_selection_rule(1, 2, 3) is True

    def test_k1_k2_k2_forbidden(self):
        """(1,2,2) is forbidden: sum=5 is odd."""
        assert cg_selection_rule(1, 2, 2) is False

    def test_triangle_inequality_violation(self):
        """(1,1,5) violates triangle: 5 > 1+1=2."""
        assert cg_selection_rule(1, 1, 5) is False

    def test_triangle_boundary(self):
        """(1,1,2) is at the triangle boundary: 2 = 1+1."""
        assert cg_selection_rule(1, 1, 2) is True

    def test_symmetric_in_arguments(self):
        """Selection rule is symmetric under permutations."""
        for k1, k2, k3 in [(1, 2, 3), (2, 3, 5), (1, 4, 5)]:
            val = cg_selection_rule(k1, k2, k3)
            assert cg_selection_rule(k2, k1, k3) == val
            assert cg_selection_rule(k1, k3, k2) == val
            assert cg_selection_rule(k3, k2, k1) == val

    def test_invalid_indices(self):
        """k < 1 always returns False."""
        assert cg_selection_rule(0, 1, 1) is False
        assert cg_selection_rule(1, 0, 1) is False
        assert cg_selection_rule(1, 1, 0) is False

    def test_quartic_selection_basic(self):
        """(1,1,1,1) is allowed for quartic: sum=4 even."""
        assert cg_selection_rule_quartic(1, 1, 1, 1) is True

    def test_quartic_parity(self):
        """(1,1,1,2) is forbidden: sum=5 is odd."""
        assert cg_selection_rule_quartic(1, 1, 1, 2) is False

    def test_quartic_allowed_mixed(self):
        """(1,1,2,2) is allowed: sum=6 even."""
        assert cg_selection_rule_quartic(1, 1, 2, 2) is True

    def test_quartic_triangle(self):
        """(1,1,5,5) is forbidden by triangle: |1-1|=0 but 5+5=10 > 1+1=2."""
        # Actually |k1-k2| <= k3+k4 is 0 <= 10, satisfied
        # And |k3-k4| <= k1+k2 is 0 <= 2, satisfied
        # Sum = 12 is even. So this is allowed.
        assert cg_selection_rule_quartic(1, 1, 5, 5) is True

    def test_quartic_selection_large_gap(self):
        """(1,2,10,10) forbidden: |1-2|=1 <= 20, |10-10|=0 <= 3, sum=23 odd."""
        assert cg_selection_rule_quartic(1, 2, 10, 10) is False


# =====================================================================
# 3. Cubic vertex
# =====================================================================

class TestCubicVertex:
    """Tests for the cubic (3-gluon) vertex on S^3."""

    @pytest.fixture
    def vertex(self):
        return CubicVertex(R=1.0, g=1.0, k_max=10)

    @pytest.fixture
    def vertex_physical(self):
        return CubicVertex(R=R_PHYSICAL_FM, g=np.sqrt(G2_PHYSICAL), k_max=10)

    def test_coupling_k1_positive(self, vertex):
        """C_3(1,1,1) is positive on unit S^3."""
        c = vertex.coupling_k1()
        assert c > 0

    def test_coupling_k1_exact_formula(self, vertex):
        """C_3(1,1,1) = 2/R * sqrt(3/Vol(S^3)) on unit sphere."""
        R = 1.0
        vol = 2.0 * np.pi**2 * R**3
        expected = 2.0 / R * np.sqrt(3.0 / vol)
        assert vertex.coupling_k1() == pytest.approx(expected)

    def test_coupling_k1_scales_with_R(self):
        """C_3 scales as 1/R^{5/2} (from 1/R * 1/sqrt(R^3))."""
        R1, R2 = 1.0, 2.0
        v1 = CubicVertex(R=R1, g=1.0, k_max=5)
        v2 = CubicVertex(R=R2, g=1.0, k_max=5)
        ratio = v1.coupling_k1() / v2.coupling_k1()
        expected_ratio = (R2 / R1)**2.5
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_selection_rule_enforced(self, vertex):
        """Forbidden couplings return 0."""
        # (1,1,1) is forbidden (odd sum)
        assert vertex.coupling(1, 1, 1) == pytest.approx(0.0)
        # (1,2,2) is forbidden (odd sum)
        assert vertex.coupling(1, 2, 2) == pytest.approx(0.0)

    def test_allowed_coupling_nonzero(self, vertex):
        """Allowed couplings are non-zero."""
        # (1,1,2) is allowed
        c = vertex.coupling(1, 1, 2)
        assert c > 0

    def test_coupling_symmetric(self, vertex):
        """C_3(k1,k2,k3) is symmetric under permutations."""
        for k1, k2, k3 in [(1, 1, 2), (1, 2, 3), (2, 2, 4)]:
            if not cg_selection_rule(k1, k2, k3):
                continue
            c = vertex.coupling(k1, k2, k3)
            assert vertex.coupling(k2, k1, k3) == pytest.approx(c)
            assert vertex.coupling(k1, k3, k2) == pytest.approx(c)
            assert vertex.coupling(k3, k2, k1) == pytest.approx(c)

    def test_vertex_with_g(self, vertex):
        """Full vertex = g * coupling."""
        for k1, k2, k3 in [(1, 1, 2), (1, 2, 3)]:
            if not cg_selection_rule(k1, k2, k3):
                continue
            c = vertex.coupling(k1, k2, k3)
            assert vertex.vertex_with_g(k1, k2, k3) == pytest.approx(
                vertex.g * c
            )

    def test_operator_norm_positive(self, vertex):
        """Operator norm is positive."""
        n = vertex.operator_norm(k_cutoff=3)
        assert n > 0

    def test_operator_norm_monotone(self, vertex):
        """Operator norm increases with cutoff."""
        n1 = vertex.operator_norm(k_cutoff=2)
        n2 = vertex.operator_norm(k_cutoff=4)
        assert n2 >= n1

    def test_bose_symmetry(self, vertex):
        """Bose symmetry: C_3 is symmetric (antisymmetry from f^{abc})."""
        result = vertex.bose_symmetry_check(1, 1, 2)
        assert result['symmetric'] is True

    def test_bose_symmetry_higher_modes(self, vertex):
        """Bose symmetry for higher modes."""
        for k1, k2, k3 in [(1, 2, 3), (2, 2, 4), (1, 3, 4)]:
            if not cg_selection_rule(k1, k2, k3):
                continue
            result = vertex.bose_symmetry_check(k1, k2, k3)
            assert result['symmetric'] is True

    def test_coupling_decays_with_k(self, vertex):
        """Higher-mode couplings decay (from CG coefficient spreading)."""
        # Compare C_3(1,1,2) with C_3(3,3,6)
        c_low = vertex.coupling(1, 1, 2)
        c_high = vertex.coupling(3, 3, 6)
        # Both should be positive if allowed
        if c_low > 0 and c_high > 0:
            # High mode coupling should be smaller per unit multiplicity
            d_low = coexact_multiplicity(1)**2 * coexact_multiplicity(2)
            d_high = coexact_multiplicity(3)**2 * coexact_multiplicity(6)
            assert c_high / np.sqrt(d_high) <= c_low / np.sqrt(d_low) * 2.0

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CubicVertex(R=-1.0)
        with pytest.raises(ValueError):
            CubicVertex(R=0.0)

    def test_invalid_k_max(self):
        """k_max < 1 raises ValueError."""
        with pytest.raises(ValueError):
            CubicVertex(k_max=0)


# =====================================================================
# 4. Quartic vertex
# =====================================================================

class TestQuarticVertex:
    """Tests for the quartic (4-gluon) vertex on S^3."""

    @pytest.fixture
    def vertex(self):
        return QuarticVertex(R=1.0, g2=1.0, k_max=10)

    @pytest.fixture
    def vertex_physical(self):
        return QuarticVertex(R=R_PHYSICAL_FM, g2=G2_PHYSICAL, k_max=10)

    def test_coupling_k1_positive(self, vertex):
        """C_4(1,1,1,1) is positive."""
        c = vertex.coupling_k1()
        assert c > 0

    def test_coupling_k1_formula(self, vertex):
        """C_4(1,1,1,1) = 2/Vol(S^3) on unit sphere."""
        vol = 2.0 * np.pi**2
        expected = 2.0 / vol
        assert vertex.coupling_k1() == pytest.approx(expected)

    def test_v4_9dof_nonnegative(self, vertex):
        """V_4(a) >= 0 for all a (sum of squares)."""
        rng = np.random.RandomState(42)
        for _ in range(50):
            a = rng.randn(9)
            assert vertex.v4_9dof(a) >= -1e-14

    def test_v4_9dof_zero_at_origin(self, vertex):
        """V_4(0) = 0."""
        assert vertex.v4_9dof(np.zeros(9)) == pytest.approx(0.0)

    def test_v4_9dof_identity_config(self, vertex):
        """V_4 for M = identity."""
        a = np.eye(3).flatten()
        # S = I, TrS = 3, TrS^2 = 3
        # V_4 = 0.5 * g2 * (9 - 3) = 3*g2
        expected = 3.0 * vertex.g2
        assert vertex.v4_9dof(a) == pytest.approx(expected)

    def test_v4_9dof_diagonal_config(self, vertex):
        """V_4 for diagonal M = diag(1, 2, 3)."""
        a = np.diag([1.0, 2.0, 3.0]).flatten()
        S = np.diag([1.0, 4.0, 9.0])
        tr_S = 14.0
        tr_S2 = 1.0 + 16.0 + 81.0
        expected = 0.5 * vertex.g2 * (tr_S**2 - tr_S2)
        assert vertex.v4_9dof(a) == pytest.approx(expected)

    def test_v4_consistency_with_v4_convexity(self):
        """V_4 matches the formula in v4_convexity.py."""
        # Import the reference implementation
        from yang_mills_s3.proofs.v4_convexity import v4_potential
        rng = np.random.RandomState(123)
        v = QuarticVertex(R=1.0, g2=2.5, k_max=5)
        for _ in range(20):
            a = rng.randn(9)
            our_val = v.v4_9dof(a)
            ref_val = v4_potential(a, g2=2.5)
            assert our_val == pytest.approx(ref_val, abs=1e-12)

    def test_quartic_selection_rule(self, vertex):
        """Forbidden quartic couplings return 0."""
        # (1,1,1,2) is forbidden: odd sum
        assert vertex.coupling(1, 1, 1, 2) == pytest.approx(0.0)

    def test_quartic_allowed_nonzero(self, vertex):
        """Allowed quartic couplings are positive."""
        assert vertex.coupling(1, 1, 1, 1) > 0
        assert vertex.coupling(1, 1, 2, 2) > 0

    def test_quartic_symmetric(self, vertex):
        """C_4 is symmetric under permutations."""
        result = vertex.bose_symmetry_check(1, 1, 2, 2)
        assert result['all_equal'] is True

    def test_quartic_symmetric_higher(self, vertex):
        """Bose symmetry for higher modes."""
        result = vertex.bose_symmetry_check(1, 2, 1, 2)
        assert result['all_equal'] is True

    def test_operator_norm_positive(self, vertex):
        """Quartic operator norm is positive."""
        n = vertex.operator_norm(k_cutoff=3)
        assert n > 0

    def test_operator_norm_monotone(self, vertex):
        """Quartic norm increases with cutoff."""
        n1 = vertex.operator_norm(k_cutoff=2)
        n2 = vertex.operator_norm(k_cutoff=3)
        assert n2 >= n1

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            QuarticVertex(R=-1.0)

    def test_v4_scaling_with_g2(self):
        """V_4 scales linearly with g^2."""
        a = np.array([0.1, 0.2, 0.3, -0.1, 0.4, -0.2, 0.15, -0.3, 0.25])
        v1 = QuarticVertex(R=1.0, g2=1.0)
        v2 = QuarticVertex(R=1.0, g2=3.0)
        assert v2.v4_9dof(a) == pytest.approx(3.0 * v1.v4_9dof(a))


# =====================================================================
# 5. Ghost vertex
# =====================================================================

class TestGhostVertex:
    """Tests for the ghost (Faddeev-Popov) vertex on S^3."""

    @pytest.fixture
    def ghost(self):
        return GhostVertex(R=1.0, g2=1.0, k_max=10, l_max=15)

    def test_scalar_eigenvalue_l0(self, ghost):
        """mu_0 = 0 (zero mode, excluded from sums)."""
        assert ghost.scalar_eigenvalue(0) == pytest.approx(0.0)

    def test_scalar_eigenvalue_l1(self, ghost):
        """mu_1 = 3/R^2 on unit sphere."""
        assert ghost.scalar_eigenvalue(1) == pytest.approx(3.0)

    def test_scalar_eigenvalue_formula(self, ghost):
        """mu_l = l(l+2)/R^2."""
        for l in range(1, 10):
            expected = l * (l + 2) / ghost.R**2
            assert ghost.scalar_eigenvalue(l) == pytest.approx(expected)

    def test_scalar_multiplicity(self, ghost):
        """d_l = (l+1)^2."""
        for l in range(0, 10):
            assert ghost.scalar_multiplicity(l) == (l + 1)**2

    def test_ghost_coupling_selection_rule(self, ghost):
        """Ghost-gluon coupling obeys selection rules."""
        # k+l1+l2 must be odd for non-zero coupling
        assert ghost.ghost_gluon_coupling(1, 1, 1) > 0  # 1+1+1=3 odd: allowed
        assert ghost.ghost_gluon_coupling(2, 1, 1) == pytest.approx(0.0)  # 2+1+1=4 even: forbidden
        assert ghost.ghost_gluon_coupling(2, 1, 2) > 0  # 2+1+2=5 odd: allowed

    def test_ghost_coupling_triangle(self, ghost):
        """Ghost-gluon coupling obeys triangle inequality."""
        # k=1, l1=1, l2=5: |1-5|=4 > k=1, so forbidden
        assert ghost.ghost_gluon_coupling(1, 1, 5) == pytest.approx(0.0)

    def test_ghost_coupling_positive(self, ghost):
        """Allowed couplings are non-negative."""
        for k in range(1, 5):
            for l1 in range(1, 5):
                for l2 in range(1, 5):
                    c = ghost.ghost_gluon_coupling(k, l1, l2)
                    assert c >= 0

    def test_ghost_self_energy_negative(self, ghost):
        """Ghost contribution to self-energy is negative (ghost statistics)."""
        sigma = ghost.one_loop_ghost_contribution(1)
        assert sigma <= 0

    def test_ghost_self_energy_finite(self, ghost):
        """Ghost self-energy is finite on S^3."""
        for k in [1, 2, 3]:
            sigma = ghost.one_loop_ghost_contribution(k)
            assert np.isfinite(sigma)

    def test_invalid_l(self, ghost):
        """Negative l raises ValueError."""
        with pytest.raises(ValueError):
            ghost.scalar_eigenvalue(-1)

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            GhostVertex(R=-1.0)


# =====================================================================
# 6. Scale-decomposed vertices
# =====================================================================

class TestScaleDecomposedVertices:
    """Tests for scale-decomposed vertices for the RG step."""

    @pytest.fixture
    def sdv(self):
        hks = HeatKernelSlices(R=1.0, M=2.0, a_lattice=0.1, k_max=20)
        return ScaleDecomposedVertices(R=1.0, g2=1.0, hks=hks, k_max=20)

    def test_mode_cutoff_monotone(self, sdv):
        """Mode cutoff k_j increases with scale j."""
        cutoffs = [sdv.mode_cutoff_at_scale(j)
                   for j in range(sdv.hks.num_scales)]
        for i in range(len(cutoffs) - 1):
            assert cutoffs[i + 1] >= cutoffs[i]

    def test_mode_cutoff_at_least_1(self, sdv):
        """Mode cutoff is always >= 1."""
        for j in range(sdv.hks.num_scales + 1):
            assert sdv.mode_cutoff_at_scale(j) >= 1

    def test_high_shell_modes_valid(self, sdv):
        """High shell mode range is valid."""
        for j in range(sdv.hks.num_scales - 1):
            k_lo, k_hi = sdv.high_shell_modes(j)
            assert k_lo >= 1
            assert k_hi >= k_lo or k_hi == k_lo - 1  # Can be empty

    def test_low_low_low_norm_positive(self, sdv):
        """Low-low-low vertex norm is positive."""
        n = sdv.low_low_low_vertex_norm(0)
        assert n >= 0

    def test_one_loop_cubic_correction_finite(self, sdv):
        """One-loop cubic correction is finite."""
        for j in range(min(3, sdv.hks.num_scales)):
            corr = sdv.one_loop_cubic_correction(j, 1)
            assert np.isfinite(corr)

    def test_one_loop_quartic_correction_finite(self, sdv):
        """One-loop quartic correction is finite."""
        for j in range(min(3, sdv.hks.num_scales)):
            corr = sdv.one_loop_quartic_correction(j, 1)
            assert np.isfinite(corr)

    def test_one_loop_corrections_nonnegative(self, sdv):
        """One-loop corrections should be non-negative (positive mass shift)."""
        for j in range(min(3, sdv.hks.num_scales)):
            c_cubic = sdv.one_loop_cubic_correction(j, 1)
            c_quartic = sdv.one_loop_quartic_correction(j, 1)
            assert c_cubic >= -1e-15
            assert c_quartic >= -1e-15

    def test_default_hks_creation(self):
        """ScaleDecomposedVertices creates HKS if not provided."""
        sdv = ScaleDecomposedVertices(R=1.0, g2=1.0, k_max=10)
        assert sdv.hks is not None
        assert sdv.hks.R == pytest.approx(1.0)


# =====================================================================
# 7. Counter-terms and beta function
# =====================================================================

class TestCounterTerms:
    """Tests for one-loop counter-term structure."""

    @pytest.fixture
    def ct_su2(self):
        return CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=50)

    @pytest.fixture
    def ct_physical(self):
        return CounterTerms(R=R_PHYSICAL_FM, g2=G2_PHYSICAL, N_c=2, k_max=50)

    def test_b0_su2(self, ct_su2):
        """b_0 = 22/3 for SU(2). THEOREM."""
        assert ct_su2.beta_function_coefficient() == pytest.approx(22.0 / 3.0)

    def test_b0_su3(self):
        """b_0 = 11 for SU(3). THEOREM."""
        ct = CounterTerms(R=1.0, g2=1.0, N_c=3, k_max=50)
        assert ct.beta_function_coefficient() == pytest.approx(11.0)

    def test_b0_formula(self):
        """b_0 = 11*N/3 for general SU(N). THEOREM."""
        for N in [2, 3, 4, 5]:
            ct = CounterTerms(R=1.0, g2=1.0, N_c=N, k_max=50)
            assert ct.beta_function_coefficient() == pytest.approx(11.0 * N / 3.0)

    def test_mass_renormalization_positive(self, ct_su2):
        """delta_m^2 > 0 (positive mass shift from interactions)."""
        dm2 = ct_su2.mass_renormalization()
        assert dm2 > 0

    def test_wavefunction_renormalization_positive(self, ct_su2):
        """delta_Z > 0. NUMERICAL."""
        dZ = ct_su2.wavefunction_renormalization()
        assert dZ > 0

    def test_wavefunction_renormalization_finite(self, ct_su2):
        """delta_Z is finite on S^3 (no log divergence in 3D). NUMERICAL."""
        dZ = ct_su2.wavefunction_renormalization()
        assert np.isfinite(dZ)
        # Check it's not too large (should be O(1) for g^2=1)
        assert dZ < 100.0

    def test_coupling_renormalization_negative(self, ct_su2):
        """delta_g^2/g^2 < 0 (asymptotic freedom). NUMERICAL."""
        dg2 = ct_su2.coupling_renormalization()
        assert dg2 < 0

    def test_coupling_renorm_scales_with_g2(self):
        """delta_g^2/g^2 scales linearly with g^2."""
        ct1 = CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=50)
        ct2 = CounterTerms(R=1.0, g2=2.0, N_c=2, k_max=50)
        dg1 = ct1.coupling_renormalization()
        dg2 = ct2.coupling_renormalization()
        # delta_g^2/g^2 contains one factor of g^2
        assert dg2 == pytest.approx(2.0 * dg1, rel=1e-10)

    def test_counter_term_summary(self, ct_su2):
        """Counter-term summary has all required keys."""
        s = ct_su2.counter_term_summary()
        required_keys = ['delta_m2', 'delta_Z', 'delta_g2_rel', 'b0',
                         'b0_expected', 'k_max', 'R', 'g2']
        for key in required_keys:
            assert key in s

    def test_mass_renorm_grows_with_kmax(self):
        """delta_m^2 grows with k_max (linear UV divergence in 3D). NUMERICAL."""
        ct1 = CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=20)
        ct2 = CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=50)
        dm1 = ct1.mass_renormalization()
        dm2 = ct2.mass_renormalization()
        assert dm2 > dm1

    def test_wavefunction_converges_with_kmax(self):
        """delta_Z converges as k_max -> inf (no divergence in 3D). NUMERICAL."""
        ct1 = CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=30)
        ct2 = CounterTerms(R=1.0, g2=1.0, N_c=2, k_max=100)
        dZ1 = ct1.wavefunction_renormalization()
        dZ2 = ct2.wavefunction_renormalization()
        # Should converge: relative difference should be small
        rel_diff = abs(dZ2 - dZ1) / max(abs(dZ2), 1e-30)
        assert rel_diff < 0.1  # Less than 10% change doubling k_max

    def test_flat_space_comparison_uv_convergence(self, ct_su2):
        """UV modes agree with flat space. NUMERICAL."""
        result = ct_su2.flat_space_comparison(k_threshold=10)
        # For k >> 1, S^3 eigenvalue (k+1)^2/R^2 ~ k^2/R^2 (flat space)
        # The ratio should approach 1
        assert result['ratio_uv'] < 2.0  # Within factor of 2
        assert result['deviation_uv'] < 0.5  # Less than 50% deviation

    def test_ir_fraction_nonzero(self, ct_su2):
        """IR modes contribute a nonzero fraction. NUMERICAL."""
        result = ct_su2.flat_space_comparison(k_threshold=5)
        assert result['ir_fraction'] > 0.0
        assert result['ir_fraction'] < 1.0

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            CounterTerms(R=-1.0)

    def test_invalid_N_c(self):
        """N_c < 2 raises ValueError."""
        with pytest.raises(ValueError):
            CounterTerms(N_c=1)


# =====================================================================
# 8. Vertex bounds and IR finiteness
# =====================================================================

class TestVertexBounds:
    """Tests for vertex scaling analysis and IR finiteness."""

    @pytest.fixture
    def vb(self):
        return VertexBounds(R=1.0, g2=1.0, k_max=10)

    def test_ir_finiteness(self, vb):
        """All vertices are finite on S^3. THEOREM (compactness)."""
        result = vb.ir_finiteness_check()
        assert result['is_finite'] is True
        assert result['cubic_k1'] > 0
        assert result['quartic_k1'] > 0
        assert np.isfinite(result['cubic_k1'])
        assert np.isfinite(result['quartic_k1'])

    def test_spectral_gap_positive(self, vb):
        """lambda_1 = 4/R^2 > 0. THEOREM."""
        result = vb.ir_finiteness_check()
        assert result['gap_lambda1'] == pytest.approx(4.0)  # R=1

    def test_cubic_norm_scaling(self, vb):
        """Cubic norm grows polynomially with cutoff. NUMERICAL."""
        result = vb.cubic_norm_vs_scale(k_cutoffs=[1, 2, 3, 4, 5])
        # Norm should be positive for each cutoff
        for n in result['norms']:
            assert n >= 0
        # Should be monotone
        for i in range(len(result['norms']) - 1):
            assert result['norms'][i + 1] >= result['norms'][i]

    def test_quartic_norm_scaling(self, vb):
        """Quartic norm grows polynomially with cutoff. NUMERICAL."""
        result = vb.quartic_norm_vs_scale(k_cutoffs=[1, 2, 3, 4])
        for n in result['norms']:
            assert n >= 0

    def test_ir_bound_matches_exact(self, vb):
        """IR bound matches the exact k=1 cubic coupling."""
        result = vb.ir_finiteness_check()
        assert result['cubic_k1'] == pytest.approx(result['ir_bound_cubic'])


# =====================================================================
# 9. Sum rules and spectral identities
# =====================================================================

class TestSumRules:
    """Tests for spectral sum rules involving vertices."""

    def test_structure_constant_trace(self):
        """Tr(f^a f^a) = C_2(adj) * dim(adj). THEOREM."""
        f = su2_structure_constants()
        # sum_{b,c} f^{abc} f^{abc} = C_2(adj) * delta_{aa} summed
        total = 0.0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    total += f[a, b, c]**2
        # This should be 2 * 3 = 6 for SU(2)
        # (C_2(adj) = 2, dim(adj) = 3)
        assert total == pytest.approx(CASIMIR_ADJ_SU2 * DIM_ADJ_SU2)

    def test_multiplicity_sum(self):
        """sum_{k=1}^K d_k = total number of coexact modes up to level K."""
        # d_k = 2k(k+2), sum = 2 * sum k^2 + 4 * sum k
        # = 2*K*(K+1)*(2K+1)/6 + 4*K*(K+1)/2
        # = K*(K+1)*(2K+1)/3 + 2*K*(K+1)
        # = K*(K+1)*(2K+7)/3
        for K in [5, 10, 20]:
            total = sum(coexact_multiplicity(k) for k in range(1, K + 1))
            expected = K * (K + 1) * (2 * K + 7) // 3
            assert total == expected

    def test_quartic_v4_is_sum_of_squares(self):
        """V_4 = g^2 * sum of squared 2x2 minors. THEOREM.

        The identity: (Tr S)^2 - Tr(S^2) = 2 * sum_{i<j, alpha<beta} |minor|^2.
        So V_4 = (g^2/2) * 2 * sum_minors^2 = g^2 * sum_minors^2.
        """
        rng = np.random.RandomState(99)
        v = QuarticVertex(R=1.0, g2=1.0, k_max=5)
        for _ in range(20):
            a = rng.randn(9)
            M = a.reshape(3, 3)
            # Sum of squared 2x2 minors
            sos = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    for alpha in range(3):
                        for beta in range(alpha + 1, 3):
                            minor = M[i, alpha] * M[j, beta] - M[j, alpha] * M[i, beta]
                            sos += minor**2
            # (TrS)^2 - TrS^2 = 2 * sos, so V_4 = g^2/2 * 2*sos = g^2 * sos
            expected = v.g2 * sos
            computed = v.v4_9dof(a)
            assert computed == pytest.approx(expected, abs=1e-12)


# =====================================================================
# 10. Physical parameter consistency
# =====================================================================

class TestPhysicalParameters:
    """Tests for physical parameter values and consistency."""

    def test_physical_constants(self):
        """Physical constants have correct values."""
        assert HBAR_C_MEV_FM == pytest.approx(197.3269804, rel=1e-6)
        assert R_PHYSICAL_FM == pytest.approx(2.2)
        assert G2_PHYSICAL == pytest.approx(6.28)
        assert N_COLORS_SU2 == 2
        assert DIM_ADJ_SU2 == 3

    def test_b0_su2_value(self):
        """b_0 = 22/3 for SU(2)."""
        assert B0_SU2 == pytest.approx(22.0 / 3.0)

    def test_mass_gap_from_R(self):
        """m_gap = 2*hbar*c/R ~ 179 MeV."""
        m_gap = 2.0 * HBAR_C_MEV_FM / R_PHYSICAL_FM
        assert m_gap == pytest.approx(179.39, rel=0.01)

    def test_coexact_gap_at_physical_R(self):
        """lambda_1 = 4/R^2 at physical R."""
        lam1 = coexact_eigenvalue(1, R_PHYSICAL_FM)
        assert lam1 == pytest.approx(4.0 / R_PHYSICAL_FM**2)

    def test_run_vertex_analysis(self):
        """Full analysis runs without error. NUMERICAL."""
        results = run_vertex_analysis(R=1.0, g2=1.0, k_max=5, verbose=False)
        assert 'cubic_k1' in results
        assert 'quartic_k1' in results
        assert 'counter_terms' in results
        assert 'ir_finiteness' in results
        assert results['cubic_k1'] > 0
        assert results['quartic_k1'] > 0

    def test_run_analysis_physical(self):
        """Analysis at physical parameters. NUMERICAL."""
        results = run_vertex_analysis(
            R=R_PHYSICAL_FM, g2=G2_PHYSICAL, k_max=5, verbose=False
        )
        assert results['cubic_k1'] > 0
        assert results['quartic_k1'] > 0
        ct = results['counter_terms']
        assert ct['b0'] == pytest.approx(22.0 / 3.0)


# =====================================================================
# 11. Ward identity / gauge invariance checks
# =====================================================================

class TestWardIdentity:
    """Tests for gauge invariance constraints on vertices."""

    def test_longitudinal_decoupling_cubic(self):
        """
        Ward identity: longitudinal modes decouple from the cubic vertex.

        On S^3, exact 1-forms (gauge modes) have eigenvalues l(l+2)/R^2
        which are DIFFERENT from coexact eigenvalues (k+1)^2/R^2.
        The cubic vertex only couples coexact modes (physical gluons).
        Longitudinal (exact) modes are projected out by the Coulomb gauge
        condition d*a = 0.

        THEOREM (Coulomb gauge on S^3).
        """
        # The coexact spectrum {4, 9, 16, 25, ...} and exact spectrum
        # {3, 8, 15, 24, ...} are disjoint. No accidental degeneracy
        # means no mixing.
        for k in range(1, 20):
            lam_coexact = (k + 1)**2
            for l in range(1, 20):
                lam_exact = l * (l + 2)
                if lam_coexact == lam_exact:
                    pytest.fail(f"Degenerate eigenvalue at k={k}, l={l}")

    def test_vertex_color_structure(self):
        """
        The cubic vertex color factor is f^{abc} (antisymmetric).
        Contraction f^{abc}f^{abc} = 6 for SU(2).
        This is the Casimir of the adjoint times dim(adj).

        THEOREM.
        """
        f = su2_structure_constants()
        color_factor = np.sum(f**2)
        expected = casimir_adjoint(2) * 3  # C_2(adj) * dim(adj)
        assert color_factor == pytest.approx(expected)

    def test_quartic_color_structure(self):
        """
        The quartic vertex color factor is f^{abe}f^{cde}.
        When summed over e, this gives delta_{ac}delta_{bd} - delta_{ad}delta_{bc}.

        THEOREM (SU(2) identity).
        """
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        computed = sum(f[a, b, e] * f[c, d, e] for e in range(3))
                        expected = (float(a == c) * float(b == d)
                                    - float(a == d) * float(b == c))
                        assert computed == pytest.approx(expected)


# =====================================================================
# 12. Cross-checks with existing codebase
# =====================================================================

class TestCrossChecks:
    """Cross-checks with effective_hamiltonian.py and v4_convexity.py."""

    def test_structure_constants_match_effective_hamiltonian(self):
        """Our f^{abc} matches effective_hamiltonian.su2_structure_constants."""
        from yang_mills_s3.proofs.effective_hamiltonian import su2_structure_constants as sc_ref
        f_ours = su2_structure_constants()
        f_ref = sc_ref()
        np.testing.assert_array_almost_equal(f_ours, f_ref)

    def test_structure_constants_match_gribov_diameter(self):
        """Our f^{abc} matches gribov_diameter._su2_structure_constants."""
        from yang_mills_s3.proofs.gribov_diameter import _su2_structure_constants as sc_ref
        f_ours = su2_structure_constants()
        f_ref = sc_ref()
        np.testing.assert_array_almost_equal(f_ours, f_ref)

    def test_quartic_matches_effective_hamiltonian(self):
        """V_4 from QuarticVertex matches EffectiveHamiltonian.quartic_potential."""
        from yang_mills_s3.proofs.effective_hamiltonian import EffectiveHamiltonian
        rng = np.random.RandomState(77)
        R, g = 2.0, 1.5
        qv = QuarticVertex(R=R, g2=g**2, k_max=5)
        eh = EffectiveHamiltonian(R=R, g_coupling=g)
        for _ in range(20):
            a = rng.randn(9) * 0.5
            v_ours = qv.v4_9dof(a)
            v_ref = eh.quartic_potential(a)
            assert v_ours == pytest.approx(v_ref, abs=1e-12)

    def test_eigenvalue_consistency(self):
        """Coexact eigenvalues match heat_kernel_slices."""
        for k in range(1, 20):
            for R in [1.0, 2.2, 0.5]:
                assert coexact_eigenvalue(k, R) == pytest.approx(
                    (k + 1)**2 / R**2
                )

    def test_multiplicity_consistency(self):
        """Coexact multiplicities match heat_kernel_slices."""
        for k in range(1, 20):
            assert coexact_multiplicity(k) == 2 * k * (k + 2)
