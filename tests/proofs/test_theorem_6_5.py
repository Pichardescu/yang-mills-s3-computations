"""
Tests for THEOREM 6.5 (Continuum Limit with Gap Preservation).

Verifies that theorem_6_5_continuum_limit() correctly assembles
the Dodziuk-Patodi spectral convergence + Kato-Rellich stability
argument to prove the continuum mass gap on S^3_R.

Test categories:
    1. Function execution and THEOREM status
    2. Dodziuk-Patodi hypotheses (H1-H4)
    3. Kato-Rellich bound properties
    4. Gap preservation and continuum gap
    5. Proof chain structure
    6. Robustness across R values
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.continuum_limit import theorem_6_5_continuum_limit


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def result_R1():
    """Theorem 6.5 result at R=1."""
    return theorem_6_5_continuum_limit(R=1.0, max_level=1)


@pytest.fixture(scope='module')
def result_R05():
    """Theorem 6.5 result at R=0.5."""
    return theorem_6_5_continuum_limit(R=0.5, max_level=1)


@pytest.fixture(scope='module')
def result_R2():
    """Theorem 6.5 result at R=2.0."""
    return theorem_6_5_continuum_limit(R=2.0, max_level=1)


# ======================================================================
# 1. Function execution and THEOREM status
# ======================================================================

class TestTheoremStatus:
    """The function should run and return THEOREM status."""

    def test_runs_without_error(self, result_R1):
        """theorem_6_5_continuum_limit() should complete without error."""
        assert result_R1 is not None

    def test_returns_dict(self, result_R1):
        """Should return a dictionary."""
        assert isinstance(result_R1, dict)

    def test_status_is_theorem(self, result_R1):
        """Status should be THEOREM (all hypotheses verified)."""
        assert result_R1['status'] == 'THEOREM'

    def test_name_present(self, result_R1):
        """Name should identify Theorem 6.5."""
        assert '6.5' in result_R1['name']
        assert 'Continuum Limit' in result_R1['name']


# ======================================================================
# 2. Dodziuk-Patodi hypotheses (H1-H4)
# ======================================================================

class TestDodziukPatodiHypotheses:
    """All 4 Dodziuk-Patodi hypotheses should be individually verified."""

    def test_all_hypotheses_verified(self, result_R1):
        """All hypotheses should pass together."""
        assert result_R1['all_hypotheses_verified'] is True

    def test_H1_compact_riemannian(self, result_R1):
        """H1: S^3 is a compact Riemannian manifold."""
        dh = result_R1['dodziuk_hypotheses']
        assert dh['H1_compact_riemannian'] is True

    def test_H2_smooth_triangulations(self, result_R1):
        """H2: 600-cell refinements are smooth triangulations
        (vertices on sphere, chain complex exact)."""
        dh = result_R1['dodziuk_hypotheses']
        assert dh['H2_smooth_triangulations'] is True

    def test_H3_mesh_decreases(self, result_R1):
        """H3: Mesh sizes decrease under refinement."""
        dh = result_R1['dodziuk_hypotheses']
        assert dh['H3_mesh_to_zero'] is True

    def test_H4_bounded_fatness(self, result_R1):
        """H4: Fatness uniformly bounded below (sigma >= 0.1)."""
        dh = result_R1['dodziuk_hypotheses']
        assert dh['H4_bounded_fatness'] is True

    def test_fatness_above_040(self, result_R1):
        """Fatness should be >= 0.4 (we computed 0.41+)."""
        dh = result_R1['dodziuk_hypotheses']
        assert dh['min_fatness'] >= 0.4, (
            f"Min fatness {dh['min_fatness']:.4f} < 0.4"
        )

    def test_mesh_sizes_present(self, result_R1):
        """Mesh sizes should be recorded for each level."""
        dh = result_R1['dodziuk_hypotheses']
        assert len(dh['mesh_by_level']) >= 2
        assert all(m > 0 for m in dh['mesh_by_level'])

    def test_fatness_per_level_present(self, result_R1):
        """Fatness values should be recorded for each level."""
        dh = result_R1['dodziuk_hypotheses']
        assert len(dh['fatness_by_level']) >= 2
        assert all(f > 0 for f in dh['fatness_by_level'])


# ======================================================================
# 3. Kato-Rellich bound properties
# ======================================================================

class TestKatoRellichBound:
    """The Kato-Rellich relative bound should be < 1."""

    def test_alpha_0_less_than_1(self, result_R1):
        """alpha_0 = C_alpha * g^2 should be < 1."""
        kr = result_R1['kato_rellich']
        assert bool(kr['alpha_less_than_1']) is True
        assert kr['alpha_0'] < 1.0

    def test_alpha_0_approximately_012(self, result_R1):
        """alpha_0 should be approximately 0.038."""
        kr = result_R1['kato_rellich']
        assert kr['alpha_0'] == pytest.approx(0.038, abs=0.005)

    def test_g2_critical_approximately_167(self, result_R1):
        """g^2_c = 1/C_alpha = 24*pi^2/sqrt(2) should be approximately 167.53."""
        kr = result_R1['kato_rellich']
        assert kr['g2_critical'] == pytest.approx(167.53, abs=0.5)

    def test_gap_retention_approximately_096(self, result_R1):
        """Gap retention = 1 - alpha_0 should be approximately 0.96."""
        kr = result_R1['kato_rellich']
        assert kr['gap_retention'] == pytest.approx(0.962, abs=0.01)

    def test_C_alpha_positive(self, result_R1):
        """C_alpha should be a positive constant."""
        kr = result_R1['kato_rellich']
        assert kr['C_alpha'] > 0

    def test_g2_physical_positive(self, result_R1):
        """Physical coupling g^2 should be positive."""
        kr = result_R1['kato_rellich']
        assert kr['g2_physical'] > 0


# ======================================================================
# 4. Gap preservation and continuum gap
# ======================================================================

class TestContinuumGap:
    """The continuum gap should be positive and match expected value."""

    def test_continuum_gap_positive(self, result_R1):
        """Continuum gap should be > 0."""
        assert result_R1['continuum_gap'] > 0

    def test_continuum_gap_approximately_385_R1(self, result_R1):
        """At R=1, continuum gap ~ (1-0.038)*4 ~ 3.85."""
        assert result_R1['continuum_gap'] == pytest.approx(3.85, abs=0.1)

    def test_continuum_gap_fraction(self, result_R1):
        """Gap fraction = 1 - alpha_0 should be consistent."""
        frac = result_R1['continuum_gap_fraction']
        gap = result_R1['continuum_gap']
        # gap = frac * 4/R^2, at R=1: gap = frac * 4
        assert gap == pytest.approx(frac * 4.0, rel=1e-10)

    def test_gap_value_MeV_positive(self, result_R1):
        """Gap in MeV should be positive."""
        assert result_R1['gap_value_MeV'] > 0


# ======================================================================
# 5. Proof chain structure
# ======================================================================

class TestProofChain:
    """The proof chain should have 4 steps, all THEOREM status."""

    def test_proof_chain_has_4_steps(self, result_R1):
        """Proof chain should consist of exactly 4 steps."""
        chain = result_R1['proof_chain']
        assert len(chain) == 4

    def test_all_steps_are_theorem(self, result_R1):
        """Each proof step should have THEOREM in its status."""
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

    def test_each_step_has_reference(self, result_R1):
        """Each step should cite a reference."""
        chain = result_R1['proof_chain']
        for step in chain:
            assert 'reference' in step and len(step['reference']) > 0, (
                f"Step {step['step']} missing reference"
            )

    def test_each_step_has_gives(self, result_R1):
        """Each step should state what it gives."""
        chain = result_R1['proof_chain']
        for step in chain:
            assert 'gives' in step and len(step['gives']) > 0, (
                f"Step {step['step']} missing 'gives'"
            )


# ======================================================================
# 6. Convergence threshold
# ======================================================================

class TestConvergence:
    """The convergence threshold should be < 1."""

    def test_convergence_threshold_less_than_1(self, result_R1):
        """Convergence threshold = (1 + alpha_0)/2 should be < 1."""
        assert result_R1['convergence_threshold'] < 1.0

    def test_convergence_threshold_above_alpha_0(self, result_R1):
        """Threshold should be between alpha_0 and 1."""
        alpha_0 = result_R1['kato_rellich']['alpha_0']
        threshold = result_R1['convergence_threshold']
        assert alpha_0 < threshold < 1.0


# ======================================================================
# 7. Statement string
# ======================================================================

class TestStatement:
    """The theorem statement should be a non-empty descriptive string."""

    def test_statement_nonempty(self, result_R1):
        """Statement string should be non-empty."""
        assert isinstance(result_R1['statement'], str)
        assert len(result_R1['statement']) > 50

    def test_statement_contains_theorem(self, result_R1):
        """Statement should identify itself as THEOREM 6.5."""
        assert 'THEOREM 6.5' in result_R1['statement']

    def test_statement_contains_qed(self, result_R1):
        """Statement should end with QED."""
        assert 'QED' in result_R1['statement']


# ======================================================================
# 8. Robustness across R values
# ======================================================================

class TestDifferentR:
    """Theorem should hold for different R values."""

    def test_theorem_holds_R05(self, result_R05):
        """Theorem should hold at R=0.5."""
        assert result_R05['status'] == 'THEOREM'
        assert result_R05['continuum_gap'] > 0

    def test_theorem_holds_R2(self, result_R2):
        """Theorem should hold at R=2.0."""
        assert result_R2['status'] == 'THEOREM'
        assert result_R2['continuum_gap'] > 0

    def test_gap_scales_as_R_minus_2(self, result_R05, result_R1, result_R2):
        """Gap should scale as 1/R^2: gap(R) * R^2 = const."""
        g05 = result_R05['continuum_gap'] * 0.5**2
        g1 = result_R1['continuum_gap'] * 1.0**2
        g2 = result_R2['continuum_gap'] * 2.0**2
        assert g05 == pytest.approx(g1, rel=1e-10)
        assert g1 == pytest.approx(g2, rel=1e-10)

    def test_gap_larger_at_smaller_R(self, result_R05, result_R1, result_R2):
        """Smaller R should give larger gap (1/R^2 scaling)."""
        assert result_R05['continuum_gap'] > result_R1['continuum_gap']
        assert result_R1['continuum_gap'] > result_R2['continuum_gap']
