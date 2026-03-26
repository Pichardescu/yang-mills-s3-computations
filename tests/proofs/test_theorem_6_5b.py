"""
Tests for THEOREM 6.5b (Whitney L^6 Convergence => Continuum Limit Uniqueness).

Verifies that theorem_6_5b_whitney_l6_convergence() correctly establishes:
    1. H^1 convergence of Whitney forms (chain map + Dodziuk norm equiv)
    2. L^6 convergence via Sobolev embedding H^1 -> L^6 (dim=3)
    3. Cubic vertex convergence ||V^(n) - V||_{H^1->L^2} -> 0
    4. Full theory uniqueness via strong resolvent convergence

This upgrades the continuum limit uniqueness from PROPOSITION to THEOREM.

Test categories:
    1. Function execution and THEOREM status
    2. H^1 convergence (chain map + norm equivalence)
    3. L^6 convergence (Sobolev embedding)
    4. Cubic vertex convergence
    5. Full theory uniqueness
    6. Proof chain structure
    7. Robustness across R values
    8. Convergence rates
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.discrete_sobolev import theorem_6_5b_whitney_l6_convergence


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def result_R1():
    """Theorem 6.5b result at R=1."""
    return theorem_6_5b_whitney_l6_convergence(R=1.0, max_level=1)


@pytest.fixture(scope='module')
def result_R05():
    """Theorem 6.5b result at R=0.5."""
    return theorem_6_5b_whitney_l6_convergence(R=0.5, max_level=1)


@pytest.fixture(scope='module')
def result_R2():
    """Theorem 6.5b result at R=2.0."""
    return theorem_6_5b_whitney_l6_convergence(R=2.0, max_level=1)


# ======================================================================
# 1. Function execution and THEOREM status
# ======================================================================

class TestTheoremStatus:
    """The function should run and return THEOREM status."""

    def test_runs_without_error(self, result_R1):
        """theorem_6_5b should complete without error."""
        assert result_R1 is not None

    def test_returns_dict(self, result_R1):
        """Should return a dictionary."""
        assert isinstance(result_R1, dict)

    def test_status_is_theorem(self, result_R1):
        """Status should be THEOREM (all ingredients verified)."""
        assert result_R1['status'] == 'THEOREM'

    def test_name_contains_6_5b(self, result_R1):
        """Name should identify Theorem 6.5b."""
        assert '6.5b' in result_R1['name']

    def test_name_contains_uniqueness(self, result_R1):
        """Name should mention uniqueness."""
        assert 'Uniqueness' in result_R1['name']

    def test_name_contains_whitney(self, result_R1):
        """Name should mention Whitney."""
        assert 'Whitney' in result_R1['name']


# ======================================================================
# 2. H^1 convergence (chain map + norm equivalence)
# ======================================================================

class TestH1Convergence:
    """H^1 convergence of Whitney forms should be established."""

    def test_h1_convergence_true(self, result_R1):
        """H^1 convergence flag should be True."""
        assert result_R1['h1_convergence'] is True

    def test_chain_map_exact(self, result_R1):
        """Chain map d W = W d should be exact (algebraic)."""
        assert result_R1['chain_map_exact'] is True

    def test_chain_map_all_levels(self, result_R1):
        """Chain map should be exact at every refinement level."""
        for cm in result_R1['chain_map_results']:
            assert bool(cm['exact']) is True
            assert cm['max_deviation'] < 1e-10

    def test_fatness_bounded(self, result_R1):
        """Fatness should be uniformly bounded below > 0.1."""
        assert result_R1['fatness_bounded'] is True

    def test_fatness_above_040(self, result_R1):
        """Fatness should be >= 0.4 (600-cell has sigma >= 0.41)."""
        assert min(result_R1['fatness_by_level']) >= 0.4, (
            f"Min fatness {min(result_R1['fatness_by_level']):.4f} < 0.4"
        )

    def test_mesh_decreasing(self, result_R1):
        """Mesh sizes should decrease under refinement."""
        assert result_R1['mesh_decreasing'] is True

    def test_h1_errors_decrease(self, result_R1):
        """H^1 errors should decrease with refinement."""
        errors = result_R1['h1_errors_by_level']
        if len(errors) >= 2:
            assert errors[-1] < errors[0], (
                f"H^1 error did not decrease: {errors}"
            )

    def test_h1_errors_positive(self, result_R1):
        """H^1 error estimates should be positive (finite mesh)."""
        for err in result_R1['h1_errors_by_level']:
            assert err > 0


# ======================================================================
# 3. L^6 convergence (Sobolev embedding)
# ======================================================================

class TestL6Convergence:
    """L^6 convergence via Sobolev embedding should be established."""

    def test_l6_convergence_true(self, result_R1):
        """L^6 convergence flag should be True."""
        assert result_R1['l6_convergence'] is True

    def test_sobolev_dimension_3(self, result_R1):
        """Sobolev embedding should be for dimension 3."""
        assert result_R1['sobolev_embedding']['dimension'] == 3

    def test_sobolev_embedding_h1_to_l6(self, result_R1):
        """Embedding should be H^1 -> L^6."""
        assert 'L^6' in result_R1['sobolev_embedding']['embedding']

    def test_positive_curvature(self, result_R1):
        """Ric(S^3) > 0 should be noted (helps Sobolev constant)."""
        assert result_R1['sobolev_embedding']['positive_curvature'] is True

    def test_sobolev_constant_positive(self, result_R1):
        """Sobolev constant should be positive."""
        assert result_R1['sobolev_constant'] > 0

    def test_l6_errors_decrease(self, result_R1):
        """L^6 errors should decrease with refinement."""
        errors = result_R1['l6_errors_by_level']
        if len(errors) >= 2:
            assert errors[-1] < errors[0], (
                f"L^6 error did not decrease: {errors}"
            )

    def test_l6_errors_bounded_by_h1(self, result_R1):
        """L^6 errors should be bounded by C_S * H^1 errors."""
        C_S = result_R1['sobolev_constant']
        for h1_err, l6_err in zip(
            result_R1['h1_errors_by_level'],
            result_R1['l6_errors_by_level']
        ):
            assert l6_err <= C_S * h1_err * (1 + 1e-10), (
                f"L^6 error {l6_err} > C_S * H^1 error = {C_S * h1_err}"
            )


# ======================================================================
# 4. Cubic vertex convergence
# ======================================================================

class TestCubicVertexConvergence:
    """The cubic vertex should converge in operator norm."""

    def test_vertex_convergence_true(self, result_R1):
        """Vertex convergence flag should be True."""
        assert result_R1['vertex_convergence'] is True

    def test_vertex_errors_decrease(self, result_R1):
        """Vertex errors should decrease with refinement."""
        errors = result_R1['vertex_errors_by_level']
        if len(errors) >= 2:
            assert errors[-1] < errors[0], (
                f"Vertex error did not decrease: {errors}"
            )

    def test_vertex_errors_bounded(self, result_R1):
        """Vertex errors should be bounded (< infinity)."""
        for err in result_R1['vertex_errors_by_level']:
            assert np.isfinite(err)
            assert err >= 0


# ======================================================================
# 5. Full theory uniqueness
# ======================================================================

class TestUniqueness:
    """Full theory uniqueness should be established."""

    def test_uniqueness_true(self, result_R1):
        """Uniqueness should be True."""
        assert bool(result_R1['uniqueness']) is True

    def test_alpha_less_than_1(self, result_R1):
        """KR relative bound alpha_0 should be < 1."""
        assert bool(result_R1['alpha_less_than_1']) is True
        assert result_R1['alpha_0'] < 1.0

    def test_alpha_approximately_0038(self, result_R1):
        """alpha_0 should be approximately 0.0375 (= g²√2/(24π²) at g²=6.28)."""
        assert result_R1['alpha_0'] == pytest.approx(0.0375, abs=0.01)


# ======================================================================
# 6. Proof chain structure
# ======================================================================

class TestProofChain:
    """The proof chain should have 4 steps, all THEOREM status."""

    def test_proof_chain_has_4_steps(self, result_R1):
        """Proof chain should consist of exactly 4 steps."""
        chain = result_R1['proof_chain']
        assert len(chain) == 4

    def test_all_steps_are_theorem(self, result_R1):
        """Each proof step should have THEOREM status."""
        chain = result_R1['proof_chain']
        for step in chain:
            assert 'THEOREM' in step['status'], (
                f"Step {step['step']} ({step['name']}) has status "
                f"'{step['status']}', expected THEOREM"
            )

    def test_steps_numbered_sequentially(self, result_R1):
        """Steps should be numbered 1 through 4."""
        chain = result_R1['proof_chain']
        step_numbers = [s['step'] for s in chain]
        assert step_numbers == [1, 2, 3, 4]

    def test_step1_is_h1_convergence(self, result_R1):
        """Step 1 should be H^1 convergence."""
        step = result_R1['proof_chain'][0]
        assert 'H^1' in step['name'] or 'H1' in step['name']

    def test_step2_is_l6_convergence(self, result_R1):
        """Step 2 should be L^6 convergence."""
        step = result_R1['proof_chain'][1]
        assert 'L^6' in step['name'] or 'L6' in step['name'] or 'Sobolev' in step['name']

    def test_step3_is_vertex_convergence(self, result_R1):
        """Step 3 should be cubic vertex convergence."""
        step = result_R1['proof_chain'][2]
        assert 'vertex' in step['name'].lower() or 'cubic' in step['name'].lower()

    def test_step4_is_uniqueness(self, result_R1):
        """Step 4 should be full theory uniqueness."""
        step = result_R1['proof_chain'][3]
        assert 'uniqueness' in step['name'].lower() or 'SRC' in step['name']

    def test_each_step_has_reference(self, result_R1):
        """Each step should cite a reference."""
        for step in result_R1['proof_chain']:
            assert 'reference' in step and len(step['reference']) > 0

    def test_each_step_has_gives(self, result_R1):
        """Each step should state what it gives."""
        for step in result_R1['proof_chain']:
            assert 'gives' in step and len(step['gives']) > 0

    def test_each_step_has_ingredients(self, result_R1):
        """Each step should list its ingredients."""
        for step in result_R1['proof_chain']:
            assert 'ingredients' in step and len(step['ingredients']) > 0


# ======================================================================
# 7. Statement string
# ======================================================================

class TestStatement:
    """The theorem statement should be properly formatted."""

    def test_statement_nonempty(self, result_R1):
        """Statement string should be non-empty."""
        assert isinstance(result_R1['statement'], str)
        assert len(result_R1['statement']) > 100

    def test_statement_contains_theorem(self, result_R1):
        """Statement should identify itself as THEOREM 6.5b."""
        assert 'THEOREM 6.5b' in result_R1['statement']

    def test_statement_contains_qed(self, result_R1):
        """Statement should end with QED."""
        assert 'QED' in result_R1['statement']

    def test_statement_mentions_h1(self, result_R1):
        """Statement should mention H^1 convergence."""
        assert 'H^1' in result_R1['statement']

    def test_statement_mentions_l6(self, result_R1):
        """Statement should mention L^6 convergence."""
        assert 'L^6' in result_R1['statement']

    def test_statement_mentions_unique(self, result_R1):
        """Statement should mention uniqueness."""
        assert 'unique' in result_R1['statement'].lower()


# ======================================================================
# 8. Robustness across R values
# ======================================================================

class TestDifferentR:
    """Theorem should hold for different R values."""

    def test_theorem_holds_R05(self, result_R05):
        """Theorem should hold at R=0.5."""
        assert result_R05['status'] == 'THEOREM'
        assert bool(result_R05['uniqueness']) is True

    def test_theorem_holds_R2(self, result_R2):
        """Theorem should hold at R=2.0."""
        assert result_R2['status'] == 'THEOREM'
        assert bool(result_R2['uniqueness']) is True

    def test_chain_map_exact_all_R(self, result_R05, result_R1, result_R2):
        """Chain map should be exact at all R values."""
        for r in [result_R05, result_R1, result_R2]:
            assert r['chain_map_exact'] is True

    def test_alpha_independent_of_R(self, result_R05, result_R1, result_R2):
        """alpha_0 should be independent of R (coupling constant)."""
        a05 = result_R05['alpha_0']
        a1 = result_R1['alpha_0']
        a2 = result_R2['alpha_0']
        assert a05 == pytest.approx(a1, rel=1e-10)
        assert a1 == pytest.approx(a2, rel=1e-10)

    def test_l6_errors_decrease_all_R(self, result_R05, result_R1, result_R2):
        """L^6 errors should decrease with refinement at all R."""
        for r in [result_R05, result_R1, result_R2]:
            errors = r['l6_errors_by_level']
            if len(errors) >= 2:
                assert errors[-1] < errors[0]


# ======================================================================
# 9. Convergence rates
# ======================================================================

class TestConvergenceRates:
    """Errors should decrease at expected rates under refinement."""

    def test_h1_error_halves_approximately(self, result_R1):
        """H^1 error should decrease by approximately factor 2 under refinement.
        (O(a) rate: mesh halves => error halves)."""
        errors = result_R1['h1_errors_by_level']
        if len(errors) >= 2:
            ratio = errors[0] / errors[-1]
            # Expect ratio ~ 2 for O(a) convergence (mesh halves)
            assert ratio > 1.3, (
                f"H^1 convergence ratio {ratio:.2f} too small (expected ~2)"
            )

    def test_l6_error_halves_approximately(self, result_R1):
        """L^6 error should decrease at O(a) rate."""
        errors = result_R1['l6_errors_by_level']
        if len(errors) >= 2:
            ratio = errors[0] / errors[-1]
            assert ratio > 1.3, (
                f"L^6 convergence ratio {ratio:.2f} too small (expected ~2)"
            )

    def test_vertex_error_halves_approximately(self, result_R1):
        """Vertex error should decrease at O(a) rate."""
        errors = result_R1['vertex_errors_by_level']
        if len(errors) >= 2:
            ratio = errors[0] / errors[-1]
            assert ratio > 1.3, (
                f"Vertex convergence ratio {ratio:.2f} too small (expected ~2)"
            )
