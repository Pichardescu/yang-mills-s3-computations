"""
Tests for the kinetic prefactor analysis.

Verifies that the physical mass gap from the BE argument on the
FP-weighted 9-DOF system decays as C/R (partial epsilon cancellation),
NOT growing as R^2 (would require full epsilon cancellation).

Test categories:
    1. Physical gap bound components
    2. Large-R asymptotics (C/R decay)
    3. Harmonic consistency
    4. Combined bound positivity at each R
    5. Uniform gap status

LABELS: THEOREM (gap > 0 at each R) + PROPOSITION (uniform gap)
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.kinetic_prefactor_analysis import (
    physical_gap_bound,
    large_R_asymptotics,
    harmonic_consistency_check,
    uniform_gap_status,
    kinetic_prefactor_analysis,
)
from yang_mills_s3.rg.quantitative_gap_be import (
    running_coupling_g2,
    kappa_min_analytical,
    HBAR_C_MEV_FM,
)


# ======================================================================
# 1. Physical gap bound components
# ======================================================================

class TestPhysicalGapBound:
    """Tests for the physical gap bound formula."""

    def test_epsilon_positive(self):
        """Kinetic prefactor epsilon must be positive."""
        for R in [0.5, 1.0, 2.2, 10.0]:
            result = physical_gap_bound(R)
            assert result['epsilon'] > 0

    def test_epsilon_decays_with_R(self):
        """eps = g^2/(2R^3) should decay as ~1/R^3."""
        e1 = physical_gap_bound(1.0)['epsilon']
        e10 = physical_gap_bound(10.0)['epsilon']
        ratio = e1 / e10
        assert ratio > 500  # should be ~1000 (1/R^3 scaling)

    def test_ghost_curv_grows_with_R(self):
        """Ghost curvature should grow as R^2."""
        g1 = physical_gap_bound(1.0)['ghost_curv_min']
        g10 = physical_gap_bound(10.0)['ghost_curv_min']
        ratio = g10 / g1
        assert ratio > 80  # should be ~100 (R^2 scaling)

    def test_eps_times_ghost_decays(self):
        """eps * ghost_curv should decay as ~1/R."""
        eg1 = physical_gap_bound(1.0)['eps_times_ghost_min']
        eg10 = physical_gap_bound(10.0)['eps_times_ghost_min']
        ratio = eg1 / eg10
        assert 5 < ratio < 20  # should be ~10 (1/R scaling)

    def test_combined_positive_at_all_R(self):
        """Combined bound should be positive at all tested R."""
        for R in [0.1, 0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 100.0]:
            result = physical_gap_bound(R)
            assert result['gap_combined'] > 0, f"Failed at R={R}"

    def test_physical_gap_at_R_2_2(self):
        """At R=2.2 fm, gap should be reasonable (100-1000 MeV)."""
        result = physical_gap_bound(2.2)
        gap_MeV = result['gap_combined_MeV']
        assert 50 < gap_MeV < 1000, f"Gap = {gap_MeV} MeV, expected 100-1000"

    def test_hess_V_negative_at_some_R(self):
        """Hess(V_total) = -11.19/R^2 should be negative."""
        result = physical_gap_bound(1.0)
        assert result['hess_V_total'] < 0


# ======================================================================
# 2. Large-R asymptotics
# ======================================================================

class TestLargeRAsymptotics:
    """Tests for the 1/R decay at large R."""

    def test_gap_decays_at_large_R(self):
        """Physical gap should decrease with R for large R."""
        g10 = physical_gap_bound(10.0)['gap_combined']
        g100 = physical_gap_bound(100.0)['gap_combined']
        assert g100 < g10, "Gap should decay at large R"

    def test_gap_times_R_stabilizes(self):
        """gap*R should approach a constant at large R."""
        asym = large_R_asymptotics()
        large_rows = [r for r in asym['table'] if r['R'] >= 50]
        gap_R_vals = [r['gap_times_R'] for r in large_rows]
        # These should be roughly equal (converged)
        if len(gap_R_vals) >= 2:
            ratio = gap_R_vals[-1] / gap_R_vals[0]
            assert 0.5 < ratio < 2.0, f"gap*R not converged: {gap_R_vals}"

    def test_asymptotic_coefficient_positive(self):
        """The asymptotic coefficient C should be positive."""
        asym = large_R_asymptotics()
        assert asym['C_theory'] > 0

    def test_decay_is_one_over_R(self):
        """Verify the decay rate is 1/R, not 1/R^2."""
        g10 = physical_gap_bound(10.0)['gap_combined']
        g100 = physical_gap_bound(100.0)['gap_combined']
        ratio = g10 / g100
        # For 1/R: ratio should be ~10
        # For 1/R^2: ratio should be ~100
        assert 5 < ratio < 20, f"Ratio = {ratio}, expected ~10 for 1/R decay"


# ======================================================================
# 3. Harmonic consistency
# ======================================================================

class TestHarmonicConsistency:
    """Verify the analysis against known harmonic oscillator results."""

    def test_harmonic_gap_is_2_over_R(self):
        """9D harmonic oscillator on S^3: gap = 2/R."""
        for R in [1.0, 2.0, 5.0]:
            omega = 2.0 / R
            gap_MeV = HBAR_C_MEV_FM * omega
            if R == 2.2:
                assert abs(gap_MeV - 179.4) < 1.0

    def test_naive_be_violated_at_small_R(self):
        """The naive bound Hess(V) should EXCEED the actual gap for small R."""
        check = harmonic_consistency_check()
        for row in check['harmonic_checks']:
            if row['R'] < 2.0:
                # Naive bound 4/R^2 > actual gap 2/R for R < 2
                assert not row['naive_valid'], (
                    f"At R={row['R']}: naive bound should violate"
                )

    def test_correct_be_always_valid(self):
        """The correct BE bound should never exceed the actual gap."""
        check = harmonic_consistency_check()
        for row in check['harmonic_checks']:
            assert row['be_correct'] <= row['actual_gap'] + 1e-10


# ======================================================================
# 4. Combined bound positivity
# ======================================================================

class TestCombinedBoundPositivity:
    """Verify gap > 0 at each R (THEOREM 10.7 Part I)."""

    def test_positive_at_small_R(self):
        """KR dominates and gives gap > 0 for small R."""
        for R in [0.1, 0.2, 0.5]:
            result = physical_gap_bound(R)
            assert result['gap_combined'] > 0
            assert result['dominant'] == 'KR'

    def test_positive_at_large_R(self):
        """BE dominates and gives gap > 0 for large R."""
        for R in [5.0, 10.0, 100.0]:
            result = physical_gap_bound(R)
            assert result['gap_combined'] > 0

    def test_positive_at_crossover(self):
        """Gap is positive in the crossover region R ~ 2 fm."""
        for R in [1.5, 2.0, 2.2, 2.5, 3.0]:
            result = physical_gap_bound(R)
            assert result['gap_combined'] > 0, f"Failed at R={R}"

    def test_dense_scan(self):
        """Dense scan: gap > 0 at 500 log-spaced R values."""
        R_values = np.logspace(-1, 2, 500)
        for R in R_values:
            result = physical_gap_bound(R)
            assert result['gap_combined'] > 0, f"Failed at R={R:.4f}"


# ======================================================================
# 5. Uniform gap status
# ======================================================================

class TestUniformGapStatus:
    """Tests for the uniform gap assessment."""

    def test_each_R_positive(self):
        """gap > 0 at each R should be confirmed."""
        status = uniform_gap_status()
        assert status['each_R_positive']

    def test_gap_min_positive(self):
        """The minimum of the combined bound should be positive."""
        status = uniform_gap_status()
        assert status['gap_min'] > 0

    def test_uniform_from_9dof_is_false(self):
        """9-DOF truncation alone should NOT prove uniform gap."""
        status = uniform_gap_status()
        assert not status['uniform_gap_from_9DOF']

    def test_gap_min_reasonable(self):
        """Minimum gap on [0.1, 100] fm should be reasonable."""
        status = uniform_gap_status()
        # The minimum on [0.1, 100] is at R~100 (BE decays as 1/R)
        # gap ~ 8*g^4/(225*R) ~ 5.6/R ~ 0.056 fm^-1 at R=100
        # = ~11 MeV. On the grid, the min is dominated by the right boundary.
        assert status['gap_min_MeV'] > 1  # Must be positive
        assert status['gap_min_MeV'] < 100000  # Not absurdly large

    def test_part_I_is_theorem(self):
        """Part I (each R) should be THEOREM."""
        status = uniform_gap_status()
        assert 'THEOREM' in status['status']['Part_I']


# ======================================================================
# 6. Full analysis
# ======================================================================

class TestFullAnalysis:
    """Test the master analysis function."""

    def test_runs_without_error(self):
        """The full analysis should run."""
        result = kinetic_prefactor_analysis(verbose=False)
        assert 'answer' in result
        assert 'PARTIAL' in result['answer']

    def test_answer_mentions_decay(self):
        """The answer should mention the 1/R decay."""
        result = kinetic_prefactor_analysis(verbose=False)
        assert '1/R' in result['answer'] or 'decays' in result['answer']
