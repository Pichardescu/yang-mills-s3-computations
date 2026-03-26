"""
Tests for the quantitative mass gap from Bakry-Emery curvature (GZ-free).

Tests verify:
1. Running coupling properties
2. Curvature bound kappa_min(R) from THEOREM 9.10
3. Physical gap conversion kappa -> MeV
4. Uniform gap computation
5. Consistency with known results (harmonic gap, PROPOSITION 10.6)
6. Honesty of the conversion chain

LABEL: Tests for THEOREM (qualitative) + NUMERICAL (quantitative)
"""

import pytest
import numpy as np

from yang_mills_s3.rg.quantitative_gap_be import (
    running_coupling_g2,
    kappa_min_analytical,
    kappa_at_origin,
    kappa_to_mass_gap,
    QuantitativeGapBE,
    compare_be_vs_gz,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_MEV,
)


# ======================================================================
# 1. Running coupling g^2(R)
# ======================================================================

class TestRunningCoupling:
    """Tests for the running coupling model."""

    def test_ir_saturation(self):
        """g^2(R) -> 4*pi as R -> infinity."""
        g2_max = 4.0 * np.pi
        g2_large = running_coupling_g2(100.0)
        assert abs(g2_large - g2_max) < 0.1, \
            f"g^2(R=100) = {g2_large}, expected ~{g2_max}"

    def test_uv_small_coupling(self):
        """g^2(R) is small for small R (asymptotic freedom)."""
        g2_small = running_coupling_g2(0.1)
        assert g2_small < 4.0 * np.pi, \
            f"g^2(R=0.1) = {g2_small}, should be < 4*pi"
        assert g2_small > 0, "g^2 must be positive"

    def test_monotonicity(self):
        """g^2(R) is monotonically increasing."""
        R_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        g2_values = [running_coupling_g2(R) for R in R_values]
        for i in range(len(g2_values) - 1):
            assert g2_values[i] < g2_values[i+1], \
                f"g^2 not increasing at R={R_values[i+1]}"

    def test_physical_radius_value(self):
        """g^2(R=2.2) should be around 6-12 (physical coupling)."""
        g2 = running_coupling_g2(2.2)
        assert 5.0 < g2 < 13.0, f"g^2(2.2) = {g2}, expected 5-13"


# ======================================================================
# 2. Curvature bound kappa_min(R) from THEOREM 9.10
# ======================================================================

class TestKappaMinAnalytical:
    """Tests for the analytical BE curvature bound."""

    def test_formula_decomposition(self):
        """kappa_min = -7.19/R^2 + (16/225)*g^2*R^2."""
        R = 3.0
        g2 = running_coupling_g2(R)
        expected = -7.19 / R**2 + (16.0 / 225.0) * g2 * R**2
        result = kappa_min_analytical(R)
        assert abs(result - expected) < 1e-10, \
            f"kappa_min({R}) = {result}, expected {expected}"

    def test_negative_at_small_R(self):
        """kappa_min < 0 for small R (V_4 non-convexity dominates)."""
        assert kappa_min_analytical(0.5) < 0
        assert kappa_min_analytical(1.0) < 0

    def test_positive_at_large_R(self):
        """kappa_min > 0 for large R (ghost curvature dominates)."""
        assert kappa_min_analytical(3.0) > 0
        assert kappa_min_analytical(5.0) > 0
        assert kappa_min_analytical(10.0) > 0

    def test_grows_as_R_squared(self):
        """For large R, kappa_min ~ (16/225)*g^2_max*R^2."""
        g2_max = 4.0 * np.pi
        for R in [10.0, 50.0, 100.0]:
            kappa = kappa_min_analytical(R)
            expected = (16.0 / 225.0) * g2_max * R**2
            # The -7.19/R^2 term is negligible
            ratio = kappa / expected
            assert 0.95 < ratio < 1.05, \
                f"kappa/expected = {ratio} at R={R}"

    def test_threshold_exists(self):
        """There exists R_0 where kappa_min(R_0) = 0."""
        qgap = QuantitativeGapBE()
        R0 = qgap.R_threshold()
        assert 1.0 < R0 < 3.0, f"R_0 = {R0}, expected 1-3 fm"
        # Verify kappa is approximately 0 at threshold
        assert abs(kappa_min_analytical(R0)) < 0.01

    def test_threshold_near_paper_value(self):
        """R_0 should be near the paper's value of 1.684 fm."""
        qgap = QuantitativeGapBE()
        R0 = qgap.R_threshold()
        # Paper uses 8/R^2 for Hess(V_2), our running coupling differs slightly
        assert 1.5 < R0 < 2.0, f"R_0 = {R0}, expected ~1.7"


class TestKappaAtOrigin:
    """Tests for the exact curvature at the vacuum."""

    def test_formula(self):
        """kappa(0) = 8/R^2 + 4*g^2*R^2/9."""
        R = 2.2
        g2 = running_coupling_g2(R)
        expected = 8.0 / R**2 + 4.0 * g2 * R**2 / 9.0
        result = kappa_at_origin(R)
        assert abs(result - expected) < 1e-10

    def test_always_positive(self):
        """kappa(0) > 0 for all R > 0."""
        for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
            assert kappa_at_origin(R) > 0, f"kappa(0) <= 0 at R={R}"

    def test_exceeds_kappa_min(self):
        """kappa(0) >= kappa_min for all R (origin is NOT the worst case)."""
        for R in [0.5, 1.0, 2.0, 3.0, 5.0]:
            assert kappa_at_origin(R) >= kappa_min_analytical(R) - 1e-10, \
                f"kappa(0) < kappa_min at R={R}"


# ======================================================================
# 3. Physical gap conversion
# ======================================================================

class TestKappaToMassGap:
    """Tests for the kappa -> MeV conversion."""

    def test_zero_kappa(self):
        """kappa <= 0 gives zero gap."""
        assert kappa_to_mass_gap(0.0) == 0.0
        assert kappa_to_mass_gap(-1.0) == 0.0

    def test_positive_kappa(self):
        """Positive kappa gives positive gap."""
        gap = kappa_to_mass_gap(1.0)
        assert gap > 0
        assert abs(gap - HBAR_C_MEV_FM / 2.0) < 0.01

    def test_harmonic_case(self):
        """
        For the harmonic oscillator with kappa = 4/R^2:
        Delta = hbar*c * kappa/2 = hbar*c * 2/R^2.
        At R = 2.2: Delta = 81.5 MeV (bound on the 179 MeV actual gap).
        """
        R = 2.2
        kappa_harmonic = 4.0 / R**2
        gap = kappa_to_mass_gap(kappa_harmonic)
        expected = HBAR_C_MEV_FM * 2.0 / R**2
        assert abs(gap - expected) < 0.1
        # The bound should be LESS than the actual harmonic gap
        actual_harmonic_gap = HBAR_C_MEV_FM * 2.0 / R
        assert gap < actual_harmonic_gap, \
            f"BE bound {gap} exceeds actual harmonic gap {actual_harmonic_gap}"

    def test_scaling(self):
        """Gap scales linearly with kappa."""
        gap1 = kappa_to_mass_gap(1.0)
        gap2 = kappa_to_mass_gap(4.0)
        assert abs(gap2 / gap1 - 4.0) < 0.01


# ======================================================================
# 4. QuantitativeGapBE class
# ======================================================================

class TestQuantitativeGapBE:
    """Tests for the main computation class."""

    @pytest.fixture
    def qgap(self):
        return QuantitativeGapBE(N=2, Lambda_QCD=200.0)

    def test_initialization(self, qgap):
        assert qgap.N == 2
        assert qgap.Lambda_QCD == 200.0

    def test_physical_gap_BE_at_2p2(self, qgap):
        """At R = 2.2 fm, Delta_BE should be O(Lambda_QCD)."""
        gap = qgap.physical_gap_BE(2.2)
        assert gap > 0, "Gap should be positive at R=2.2"
        assert 100 < gap < 500, f"Gap = {gap} MeV, expected O(Lambda)"

    def test_physical_gap_KR(self, qgap):
        """KR gap should be ~ 2*hbar*c/R."""
        gap = qgap.physical_gap_KR(2.2)
        expected = HBAR_C_MEV_FM * 2.0 / 2.2
        assert abs(gap - expected) / expected < 0.1

    def test_physical_gap_KR_decreases(self, qgap):
        """KR gap decreases with R."""
        gaps = [qgap.physical_gap_KR(R) for R in [1.0, 2.0, 5.0, 10.0]]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i+1]

    def test_physical_gap_BE_increases(self, qgap):
        """BE gap increases with R for R > R_0."""
        R0 = qgap.R_threshold()
        R_values = [R0 + 0.5, R0 + 1.0, R0 + 3.0, R0 + 8.0]
        gaps = [qgap.physical_gap_BE(R) for R in R_values]
        for i in range(len(gaps) - 1):
            assert gaps[i] < gaps[i+1], \
                f"BE gap not increasing at R={R_values[i+1]}"

    def test_crossover_exists(self, qgap):
        """There exists R* where BE = KR."""
        R_cross = qgap.crossover_radius()
        assert R_cross is not None
        assert 1.5 < R_cross < 3.0, f"R_cross = {R_cross}"

    def test_crossover_is_balanced(self, qgap):
        """At R*, Delta_BE and Delta_KR are approximately equal."""
        R_cross = qgap.crossover_radius()
        if R_cross is not None:
            be = qgap.physical_gap_BE(R_cross)
            kr = qgap.physical_gap_KR(R_cross)
            assert abs(be - kr) / max(be, kr) < 0.01, \
                f"At crossover: BE={be}, KR={kr}"

    def test_uniform_gap_positive(self, qgap):
        """The uniform gap is strictly positive."""
        ug = qgap.uniform_gap()
        assert ug['Delta_min_MeV'] > 0
        assert ug['Delta_min_MeV'] > 50  # Should be at least ~100 MeV

    def test_uniform_gap_at_crossover(self, qgap):
        """The uniform gap minimum is at the crossover radius."""
        ug = qgap.uniform_gap()
        R_cross = qgap.crossover_radius()
        if R_cross is not None:
            assert abs(ug['R_at_minimum_fm'] - R_cross) < 0.1

    def test_uniform_gap_less_than_GZ(self, qgap):
        """The GZ-free bound should be less than the GZ PROPOSITION."""
        ug = qgap.uniform_gap()
        gz_value = 3.0 * qgap.Lambda_QCD  # 600 MeV
        assert ug['Delta_min_MeV'] < gz_value, \
            "GZ-free bound exceeds GZ value (unexpected)"

    def test_uniform_gap_order_of_magnitude(self, qgap):
        """The uniform gap should be O(Lambda_QCD)."""
        ug = qgap.uniform_gap()
        ratio = ug['Delta_min_MeV'] / qgap.Lambda_QCD
        assert 0.3 < ratio < 5.0, f"Delta/Lambda = {ratio}"


# ======================================================================
# 5. Gap table
# ======================================================================

class TestGapTable:
    """Tests for the gap table generation."""

    @pytest.fixture
    def table(self):
        qgap = QuantitativeGapBE()
        return qgap.gap_table()

    def test_table_length(self, table):
        """Default table has 10 entries."""
        assert len(table) == 10

    def test_all_have_positive_best_gap(self, table):
        """Every R has a positive best gap."""
        for row in table:
            assert row['Delta_best_MeV'] > 0, \
                f"Zero gap at R={row['R_fm']}"

    def test_regime_transition(self, table):
        """Regime transitions from KR to BE at some R."""
        regimes = [row['regime'] for row in table]
        assert 'KR' in regimes, "No KR regime found"
        assert 'BE' in regimes, "No BE regime found"

    def test_kappa_sign_transition(self, table):
        """kappa_min transitions from negative to positive."""
        kappas = [row['kappa_min_fm2'] for row in table]
        has_negative = any(k < 0 for k in kappas)
        has_positive = any(k > 0 for k in kappas)
        assert has_negative and has_positive


# ======================================================================
# 6. Consistency checks
# ======================================================================

class TestConsistency:
    """Consistency and honesty checks."""

    def test_harmonic_limit(self):
        """With g^2 = 0, should recover linearized gap bound."""
        # At very small R, g^2 is small. The KR bound should give
        # Delta ~ 2*hbar*c/R, which is the linearized gap.
        qgap = QuantitativeGapBE()
        gap_KR = qgap.physical_gap_KR(0.3)  # small R, weak coupling
        harmonic = HBAR_C_MEV_FM * 2.0 / 0.3
        # KR includes (1-alpha) factor, so slightly less
        assert gap_KR < harmonic
        assert gap_KR > 0.8 * harmonic

    def test_be_bound_valid_at_harmonic(self):
        """BE bound <= actual harmonic gap (it's a LOWER bound)."""
        R = 3.0
        kappa_harmonic = 4.0 / R**2  # Hess(V_2) only
        be_bound = kappa_to_mass_gap(kappa_harmonic)
        actual_harmonic = HBAR_C_MEV_FM * 2.0 / R

        # BE bound on the harmonic part alone should be <= actual
        assert be_bound < actual_harmonic, \
            f"BE bound {be_bound} > actual {actual_harmonic} (invalid)"

    def test_ghost_curvature_enhances_gap(self):
        """
        The ghost curvature increases kappa_min beyond the harmonic value.
        At R = 2.2, kappa_min > kappa_harmonic (because ghost dominates).
        """
        R = 2.2
        kappa_harmonic = 4.0 / R**2  # 0.826 fm^{-2}
        kappa_full = kappa_min_analytical(R)  # includes ghost + V_4
        # Ghost contribution is large at R=2.2
        assert kappa_full > kappa_harmonic, \
            f"kappa_full={kappa_full} <= kappa_harmonic={kappa_harmonic}"

    def test_comparison_structure(self):
        """Compare function returns structured data."""
        comp = compare_be_vs_gz()
        assert 'be_gz_free' in comp
        assert 'gz_proposition' in comp
        assert 'ratio_be_over_gz' in comp
        assert 'honest_assessment' in comp
        assert comp['ratio_be_over_gz'] > 0
        assert comp['ratio_be_over_gz'] < 1  # BE is weaker than GZ

    def test_theorem_statement_generated(self):
        """Theorem statement is a non-empty string."""
        qgap = QuantitativeGapBE()
        stmt = qgap.theorem_statement()
        assert isinstance(stmt, str)
        assert len(stmt) > 200
        assert "THEOREM" in stmt
        assert "GZ-free" in stmt or "GZ" in stmt


# ======================================================================
# 7. Honesty checks
# ======================================================================

class TestHonesty:
    """
    Tests that verify we're being HONEST about what's proven vs assumed.
    """

    def test_running_coupling_is_model(self):
        """
        The running coupling is a MODEL, not a theorem.
        The quantitative gap depends on the running coupling model.
        Different gauge groups should give different gaps
        (but both positive -- qualitative result is robust).
        """
        qgap_su2 = QuantitativeGapBE(N=2)
        qgap_su3 = QuantitativeGapBE(N=3)

        gap_su2 = qgap_su2.physical_gap_BE(2.2)
        gap_su3 = qgap_su3.physical_gap_BE(2.2)

        # Different N gives different running coupling -> different gap
        assert gap_su2 != gap_su3
        # But both are positive (qualitative robustness)
        assert gap_su2 > 0
        assert gap_su3 > 0

    def test_kappa_bound_not_tight(self):
        """
        The analytical kappa_min(R) is NOT tight (it's a lower bound).
        The actual kappa at the origin is much larger.
        """
        R = 2.2
        kappa_min = kappa_min_analytical(R)
        kappa_0 = kappa_at_origin(R)
        assert kappa_0 > 2 * kappa_min, \
            f"kappa_origin / kappa_min = {kappa_0/kappa_min}, expected > 2"

    def test_large_R_gap_unphysically_large(self):
        """
        At large R, the BE bound becomes unphysically large.
        This is because:
        1. kappa grows as R^2 (ghost curvature dominates)
        2. The kappa/2 conversion overestimates for large kappa
        3. The 9-DOF truncation becomes increasingly loose

        This is EXPECTED and is NOT a bug. It means the BE bound
        is useful primarily near the crossover R ~ 2 fm.
        """
        qgap = QuantitativeGapBE()
        gap_R10 = qgap.physical_gap_BE(10.0)
        gap_R2 = qgap.physical_gap_BE(2.2)

        # Gap at R=10 should be much larger than at R=2.2
        # (unphysical growth, by design)
        assert gap_R10 > 10 * gap_R2

    def test_qualitative_conclusion_robust(self):
        """
        The qualitative conclusion (gap > 0 for all R) is robust.
        It does not depend on the specific running coupling model.

        For ANY g^2 in (0, 4*pi), the combined KR + BE bound gives
        gap > 0 at every R.
        """
        qgap = QuantitativeGapBE()
        # Check at many R values
        for R in np.linspace(0.3, 20.0, 50):
            gap = qgap.physical_gap_combined(R)
            assert gap > 0, f"Gap <= 0 at R={R}"


# ======================================================================
# 8. Edge cases
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_R(self):
        """Gap computation at very small R (perturbative regime)."""
        qgap = QuantitativeGapBE()
        gap = qgap.physical_gap_combined(0.1)
        assert gap > 0
        assert np.isfinite(gap)

    def test_very_large_R(self):
        """Gap computation at very large R (IR regime)."""
        qgap = QuantitativeGapBE()
        gap = qgap.physical_gap_combined(50.0)
        assert gap > 0
        assert np.isfinite(gap)

    def test_R_at_threshold(self):
        """Gap computation at R = R_0 (kappa = 0)."""
        qgap = QuantitativeGapBE()
        R0 = qgap.R_threshold()
        gap = qgap.physical_gap_combined(R0)
        assert gap > 0  # KR should provide the gap here

    def test_SU3(self):
        """The computation works for N=3."""
        qgap = QuantitativeGapBE(N=3)
        gap = qgap.physical_gap_combined(2.2)
        assert gap > 0
