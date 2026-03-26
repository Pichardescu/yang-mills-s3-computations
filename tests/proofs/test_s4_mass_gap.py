"""
Tests for the S^4 mass gap proof and geometric Zwanziger interpretation.

Tests the S4MassGap class (direct gap proof on S^4 via spectral geometry,
Sobolev embedding, and Kato-Rellich) and the GeometricZwanziger class
(speculative M_FP <-> Jacobi connection).

Test categories:
    1. Linearized gap on S^4
    2. Sobolev embedding on S^4
    3. Kato-Rellich stability on S^4
    4. Gap enhancement over S^3
    5. Time foliation argument
    6. Path A summary
    7. Geometric Zwanziger (conjecture)
    8. Dependency map (honest accounting)
"""

import pytest
import numpy as np

from yang_mills_s3.proofs.s4_mass_gap import (
    S4MassGap,
    GeometricZwanziger,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
)


# ======================================================================
# 1. Linearized gap on S^4
# ======================================================================

class TestLinearizedGap:
    """The linearized coexact 1-form gap on S^4 is 6/R^2. THEOREM status."""

    def test_gap_is_6_over_r2_unit(self):
        """Delta_0 = 6 for R=1."""
        result = S4MassGap.linearized_gap(R=1.0)
        assert abs(result['gap'] - 6.0) < 1e-14

    def test_gap_is_6_over_r2_various(self):
        """Delta_0 = 6/R^2 for various R values."""
        for R in [0.5, 1.0, 2.0, 3.14, 10.0]:
            result = S4MassGap.linearized_gap(R=R)
            expected = 6.0 / R**2
            assert abs(result['gap'] - expected) < 1e-12, \
                f"R={R}: expected {expected}, got {result['gap']}"

    def test_bigger_than_s3(self):
        """S^4 gap (6/R^2) > S^3 gap (4/R^2) for all R."""
        for R in [0.1, 1.0, 5.0, 100.0]:
            s4_gap = S4MassGap.linearized_gap(R=R)['gap']
            s3_gap = 4.0 / R**2
            assert s4_gap > s3_gap, \
                f"R={R}: S^4 gap {s4_gap} should exceed S^3 gap {s3_gap}"

    def test_label_is_theorem(self):
        """The linearized gap is a THEOREM."""
        result = S4MassGap.linearized_gap()
        assert result['label'] == 'THEOREM'

    def test_proof_steps_nonempty(self):
        """Proof steps are provided."""
        result = S4MassGap.linearized_gap()
        assert len(result['proof_steps']) >= 3

    def test_positive_for_all_finite_R(self):
        """Gap is strictly positive for all finite R > 0."""
        for R in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            result = S4MassGap.linearized_gap(R=R)
            assert result['gap'] > 0


# ======================================================================
# 2. Sobolev embedding on S^4
# ======================================================================

class TestSobolevS4:
    """Sobolev embedding on compact S^4. THEOREM status."""

    def test_critical_exponent_is_4(self):
        """In 4D, critical Sobolev exponent p* = 2n/(n-2) = 4."""
        result = S4MassGap.sobolev_on_s4()
        assert result['critical_exponent'] == 4

    def test_compact_embedding(self):
        """On compact S^4, H^1 embeds into L^p for all finite p."""
        result = S4MassGap.sobolev_on_s4()
        assert result['compact_embedding'] is True

    def test_volume_formula(self):
        """Vol(S^4(R)) = 8*pi^2*R^4/3."""
        for R in [1.0, 2.0, 0.5]:
            result = S4MassGap.sobolev_on_s4(R=R)
            expected_vol = 8.0 * np.pi**2 * R**4 / 3.0
            assert abs(result['volume'] - expected_vol) < 1e-10, \
                f"R={R}: expected volume {expected_vol}, got {result['volume']}"

    def test_sobolev_constant_positive(self):
        """Sobolev constant is positive."""
        result = S4MassGap.sobolev_on_s4()
        assert result['sobolev_constant_l4'] > 0

    def test_label_is_theorem(self):
        """Sobolev embedding is a THEOREM."""
        result = S4MassGap.sobolev_on_s4()
        assert result['label'] == 'THEOREM'


# ======================================================================
# 3. Kato-Rellich stability on S^4
# ======================================================================

class TestKatoRellichS4:
    """Kato-Rellich perturbation stability on S^4. THEOREM status."""

    def test_alpha_positive(self):
        """alpha > 0 for g^2 > 0."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28)
        assert result['alpha'] > 0

    def test_gap_survives_physical(self):
        """At physical coupling g^2=6.28, R=2.2 fm, the gap survives."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28, R=2.2)
        assert result['gap_survives'] is True, \
            f"Gap should survive at physical coupling, alpha={result['alpha']}"

    def test_alpha_less_than_one(self):
        """At physical coupling, alpha < 1."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28, R=1.0)
        assert result['alpha'] < 1.0, \
            f"alpha={result['alpha']} should be < 1 at physical coupling"

    def test_gap_remaining_positive(self):
        """(1 - alpha) * 6/R^2 > 0 at physical coupling."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28, R=1.0)
        assert result['gap_remaining'] > 0, \
            f"gap_remaining={result['gap_remaining']} should be > 0"

    def test_critical_coupling_exists(self):
        """A critical coupling g^2_c exists and is positive."""
        result = S4MassGap.kato_rellich_bound(g_squared=1.0)
        assert result['g_squared_critical'] > 0

    def test_physical_below_critical(self):
        """Physical g^2=6.28 is below critical coupling."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28)
        assert 6.28 < result['g_squared_critical'], \
            f"g^2=6.28 should be below g^2_c={result['g_squared_critical']}"

    def test_label_is_theorem(self):
        """Kato-Rellich bound is a THEOREM."""
        result = S4MassGap.kato_rellich_bound(g_squared=6.28)
        assert result['label'] == 'THEOREM'

    def test_zero_coupling_full_gap(self):
        """At g^2=0, the full linearized gap is preserved."""
        result = S4MassGap.kato_rellich_bound(g_squared=0.0, R=1.0)
        assert result['alpha'] == 0.0
        assert abs(result['gap_remaining'] - 6.0) < 1e-12


# ======================================================================
# 4. Gap enhancement over S^3
# ======================================================================

class TestGapEnhancement:
    """Comparison of S^4 vs S^3 mass gap."""

    def test_linearized_ratio_is_1_5(self):
        """Linearized gap ratio S^4/S^3 = 6/4 = 3/2."""
        result = S4MassGap.gap_enhancement_over_s3()
        assert abs(result['linearized_ratio'] - 1.5) < 1e-14

    def test_both_gaps_positive_physical(self):
        """Both S^4 and S^3 gaps are positive at physical coupling."""
        result = S4MassGap.gap_enhancement_over_s3(R=1.0, g_squared=6.28)
        assert result['both_positive'] is True


# ======================================================================
# 5. Time foliation argument
# ======================================================================

class TestTimeFoliation:
    """Foliation of S^4 by S^3 slices. PROPOSITION status."""

    def test_equatorial_slice_is_s3(self):
        """At the equator chi=pi/2, the gap equals the S^3 gap 4/R^2."""
        result = S4MassGap.time_foliation_argument(R=1.0)
        assert abs(result['equatorial_gap'] - 4.0) < 1e-14

    def test_actual_gap_larger(self):
        """The actual S^4 gap 6/R^2 exceeds the equatorial S^3 gap 4/R^2."""
        result = S4MassGap.time_foliation_argument(R=1.0)
        assert result['actual_s4_gap'] > result['equatorial_gap']

    def test_gap_increases_away_from_equator(self):
        """Gap increases away from the equator (smaller effective radius)."""
        result = S4MassGap.time_foliation_argument(R=1.0)
        assert result['gap_increases_away'] is True

    def test_label_is_proposition(self):
        """Time foliation is a PROPOSITION (transfer matrix needs regularity)."""
        result = S4MassGap.time_foliation_argument()
        assert result['label'] == 'PROPOSITION'


# ======================================================================
# 6. Path A summary
# ======================================================================

class TestPathASummary:
    """Path A proof summary on S^4."""

    def test_has_three_steps(self):
        """The proof has 3 theorem steps."""
        result = S4MassGap.path_a_s4_summary()
        assert 'step_1' in result
        assert 'step_2' in result
        assert 'step_3' in result

    def test_all_steps_are_theorem(self):
        """Each individual step is a THEOREM."""
        result = S4MassGap.path_a_s4_summary()
        for key in ['step_1', 'step_2', 'step_3']:
            assert result[key]['label'] == 'THEOREM', \
                f"{key} should be THEOREM, got {result[key]['label']}"

    def test_overall_label(self):
        """Overall label is THEOREM modulo POSTULATE."""
        result = S4MassGap.path_a_s4_summary()
        assert 'THEOREM' in result['label']
        assert 'POSTULATE' in result['label']

    def test_gz_free(self):
        """Path A does NOT require Gribov-Zwanziger."""
        result = S4MassGap.path_a_s4_summary()
        exclusions = result['does_NOT_require']
        gz_excluded = any('Gribov' in item or 'Zwanziger' in item
                          for item in exclusions)
        assert gz_excluded, "GZ should be listed as not required"


# ======================================================================
# 7. Geometric Zwanziger (conjecture)
# ======================================================================

class TestGeometricZwanziger:
    """Speculative M_FP <-> Jacobi connection. CONJECTURE status."""

    def test_labeled_conjecture(self):
        """Geometric interpretation is a CONJECTURE."""
        result = GeometricZwanziger.geometric_interpretation()
        assert result['label'] == 'CONJECTURE'

    def test_has_supporting_evidence(self):
        """Supporting evidence is provided."""
        result = GeometricZwanziger.geometric_interpretation()
        assert len(result['supporting_evidence']) >= 2

    def test_has_open_problems(self):
        """Open problems are honestly listed."""
        result = GeometricZwanziger.geometric_interpretation()
        assert len(result['open_problems']) >= 2

    def test_jacobi_eigenvalues_positive(self):
        """All Jacobi eigenvalues of S^3 in R^4 are positive."""
        result = GeometricZwanziger.second_variation_connection(R=1.0)
        assert result['all_jacobi_positive'] is True

    def test_second_variation_labeled_conjecture(self):
        """Second variation connection is also a CONJECTURE."""
        result = GeometricZwanziger.second_variation_connection()
        assert result['label'] == 'CONJECTURE'

    def test_jacobi_gap_at_l1(self):
        """Jacobi gap at l=1 equals 6/R^2 (= S^4 coexact gap)."""
        result = GeometricZwanziger.second_variation_connection(R=1.0)
        assert abs(result['jacobi_gap_l1'] - 6.0) < 1e-12


# ======================================================================
# 8. Dependency map (honest accounting)
# ======================================================================

class TestDependencyMap:
    """Dependency map structure and honest labeling."""

    def test_has_required_keys(self):
        """Dependency map contains all expected entries."""
        dep = GeometricZwanziger.dependency_map()
        required = [
            's4_linearized_gap',
            's4_sobolev',
            's4_kato_rellich',
            'time_foliation',
            'geometric_zwanziger',
            'path_a_summary',
        ]
        for key in required:
            assert key in dep, f"Missing key: {key}"

    def test_gz_free_identified(self):
        """GZ-free claims are marked as such in does_NOT_use."""
        dep = GeometricZwanziger.dependency_map()
        gz_free_keys = [
            's4_linearized_gap', 's4_sobolev', 's4_kato_rellich',
            'path_a_summary',
        ]
        for key in gz_free_keys:
            exclusions = dep[key]['does_NOT_use']
            has_gz = any('Gribov' in item or 'Zwanziger' in item
                         for item in exclusions)
            assert has_gz, f"{key} should list GZ as not used"

    def test_conjecture_labeled(self):
        """geometric_zwanziger is labeled as CONJECTURE."""
        dep = GeometricZwanziger.dependency_map()
        assert dep['geometric_zwanziger']['label'] == 'CONJECTURE'

    def test_theorem_entries_labeled(self):
        """Theorem entries are correctly labeled."""
        dep = GeometricZwanziger.dependency_map()
        for key in ['s4_linearized_gap', 's4_sobolev', 's4_kato_rellich']:
            assert dep[key]['label'] == 'THEOREM', \
                f"{key} should be THEOREM, got {dep[key]['label']}"

    def test_proposition_labeled(self):
        """time_foliation is labeled PROPOSITION."""
        dep = GeometricZwanziger.dependency_map()
        assert dep['time_foliation']['label'] == 'PROPOSITION'

    def test_each_entry_has_structure(self):
        """Each dependency entry has name, label, proves, inputs."""
        dep = GeometricZwanziger.dependency_map()
        for key, entry in dep.items():
            assert 'name' in entry, f"{key} missing 'name'"
            assert 'label' in entry, f"{key} missing 'label'"
            assert 'proves' in entry, f"{key} missing 'proves'"
            assert 'inputs' in entry, f"{key} missing 'inputs'"
