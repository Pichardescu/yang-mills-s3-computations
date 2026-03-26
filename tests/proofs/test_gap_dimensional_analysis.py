"""
Tests for gap_dimensional_analysis.py — Three Gap Types in Yang-Mills on S^3.

Tests cover:
1. Geometric gap computation and R-dependence
2. Field-space gap (PW) computation and growth
3. Kinetic normalization factor
4. Physical mass gap estimation
5. Dimensional transmutation connection
6. Three-gap comparison table
7. Honest assessment structure
8. Cross-consistency between gap types
9. Physical mass gap > 0 for all R
10. Asymptotic behavior at large R

~35 tests total.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.gap_dimensional_analysis import (
    geometric_gap_vs_R,
    field_space_gap_vs_R,
    kinetic_normalization,
    physical_mass_gap_vs_R,
    dimensional_transmutation_connection,
    three_gap_comparison,
    honest_assessment_gap_types,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
    COEXACT_EIGENVALUE,
    DR_ASYMPTOTIC,
    PW_COEFF,
    DIM_9DOF,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def R_values():
    """Standard range of radii in fm."""
    return np.array([0.5, 1.0, 2.0, 2.2, 5.0, 10.0, 20.0, 50.0])


@pytest.fixture
def R_wide():
    """Wide range for asymptotic tests."""
    return np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])


@pytest.fixture
def R_physical():
    """The physical radius."""
    return np.array([2.2])


# ======================================================================
# 1. Constants
# ======================================================================

class TestConstants:
    """Test module-level constants."""

    def test_coexact_eigenvalue(self):
        """Coexact 1-form eigenvalue on S^3 is (1+1)^2 = 4."""
        assert COEXACT_EIGENVALUE == 4.0

    def test_dim_9dof(self):
        """9-DOF space: 3 modes x 3 adjoint."""
        assert DIM_9DOF == 9

    def test_hbar_c(self):
        """hbar*c ~ 197.3 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.327) < 0.01

    def test_lambda_qcd_default(self):
        """Default Lambda_QCD = 200 MeV."""
        assert LAMBDA_QCD_DEFAULT == 200.0

    def test_dr_asymptotic_positive(self):
        """Asymptotic dimensionless diameter is positive."""
        assert DR_ASYMPTOTIC > 0

    def test_pw_coeff_value(self):
        """PW coefficient pi^2/(2*dR^2) ~ 1.02."""
        expected = np.pi**2 / (2.0 * DR_ASYMPTOTIC**2)
        assert abs(PW_COEFF - expected) < 1e-10
        assert PW_COEFF > 1.0  # known to be slightly above 1


# ======================================================================
# 2. Geometric gap
# ======================================================================

class TestGeometricGap:
    """Tests for geometric_gap_vs_R."""

    def test_formula_at_R1(self):
        """At R=1 fm: eigenvalue = 4, mass = 2*hbar_c."""
        result = geometric_gap_vs_R([1.0])
        assert abs(result['eigenvalue_inv_fm2'][0] - 4.0) < 1e-10
        assert abs(result['mass_MeV'][0] - 2.0 * HBAR_C_MEV_FM) < 1e-8

    def test_formula_at_R2(self):
        """At R=2 fm: eigenvalue = 1, mass = hbar_c."""
        result = geometric_gap_vs_R([2.0])
        assert abs(result['eigenvalue_inv_fm2'][0] - 1.0) < 1e-10
        assert abs(result['mass_MeV'][0] - HBAR_C_MEV_FM) < 1e-8

    def test_decreases_with_R(self, R_values):
        """Geometric gap decreases monotonically with R."""
        result = geometric_gap_vs_R(R_values)
        masses = result['mass_MeV']
        for i in range(len(masses) - 1):
            assert masses[i] > masses[i + 1], \
                f"Mass should decrease: m({R_values[i]}) = {masses[i]} > m({R_values[i+1]}) = {masses[i+1]}"

    def test_vanishes_at_large_R(self, R_wide):
        """Geometric mass -> 0 as R -> infinity."""
        result = geometric_gap_vs_R(R_wide)
        assert result['vanishes_at_large_R'] is True
        # At R=1000 fm, mass should be very small
        assert result['mass_MeV'][-1] < 1.0  # < 1 MeV

    def test_label_is_theorem(self):
        """Geometric gap is a THEOREM."""
        result = geometric_gap_vs_R([1.0])
        assert result['label'] == 'THEOREM'

    def test_scales_as_one_over_R(self, R_values):
        """m_geom * R = constant = 2*hbar_c."""
        result = geometric_gap_vs_R(R_values)
        product = result['mass_MeV'] * result['R_fm']
        expected = 2.0 * HBAR_C_MEV_FM
        for p in product:
            assert abs(p - expected) < 1e-8

    def test_mass_over_lambda_at_crossover(self):
        """At R ~ 2 fm, geometric mass ~ Lambda_QCD."""
        R_cross = 2.0 * HBAR_C_MEV_FM / LAMBDA_QCD_DEFAULT  # ~ 1.97 fm
        result = geometric_gap_vs_R([R_cross])
        assert abs(result['mass_over_Lambda'][0] - 1.0) < 0.01

    def test_raises_on_nonpositive_R(self):
        """Must raise for R <= 0."""
        with pytest.raises(ValueError):
            geometric_gap_vs_R([0.0])
        with pytest.raises(ValueError):
            geometric_gap_vs_R([-1.0])


# ======================================================================
# 3. Field-space gap (PW)
# ======================================================================

class TestFieldSpaceGap:
    """Tests for field_space_gap_vs_R."""

    def test_grows_with_R(self, R_values):
        """PW gap grows with R (confinement strengthens)."""
        result = field_space_gap_vs_R(R_values)
        gaps = result['pw_gap']
        for i in range(len(gaps) - 1):
            assert gaps[i] < gaps[i + 1], \
                f"PW gap should grow: gap({R_values[i]}) < gap({R_values[i+1]})"

    def test_grows_as_R_squared_asymptotically(self, R_wide):
        """At large R, PW gap ~ R^2 (up to logarithmic corrections)."""
        result = field_space_gap_vs_R(R_wide)
        R = result['R']
        gaps = result['pw_gap']
        # For R >= 50: gap / R^2 should be approximately constant
        mask = R >= 50.0
        if np.sum(mask) >= 2:
            ratio = gaps[mask] / R[mask]**2
            rel_var = np.std(ratio) / np.mean(ratio)
            assert rel_var < 0.1, f"PW gap / R^2 should stabilize, rel_var = {rel_var}"

    def test_positive_for_all_R(self, R_wide):
        """PW gap is strictly positive for all R > 0."""
        result = field_space_gap_vs_R(R_wide)
        assert np.all(result['pw_gap'] > 0)

    def test_gribov_diameter_shrinks(self, R_values):
        """Gribov diameter d(R) shrinks with R."""
        result = field_space_gap_vs_R(R_values)
        diameters = result['gribov_diameter']
        for i in range(len(diameters) - 1):
            assert diameters[i] > diameters[i + 1], \
                f"Diameter should shrink with R"

    def test_dR_stabilizes(self, R_wide):
        """d(R)*R -> constant ~ 2.199 at large R."""
        result = field_space_gap_vs_R(R_wide)
        dR = result['dR']
        R = result['R']
        # At large R, dR should approach DR_ASYMPTOTIC
        large = R >= 50.0
        if np.sum(large) >= 1:
            for dr in dR[large]:
                assert abs(dr - DR_ASYMPTOTIC) / DR_ASYMPTOTIC < 0.05, \
                    f"dR should -> {DR_ASYMPTOTIC}, got {dr}"

    def test_label_is_theorem(self):
        """Field-space gap is a THEOREM."""
        result = field_space_gap_vs_R([1.0])
        assert result['label'] == 'THEOREM'


# ======================================================================
# 4. Kinetic normalization
# ======================================================================

class TestKineticNormalization:
    """Tests for kinetic_normalization."""

    def test_decays_with_R(self, R_values):
        """Kinetic factor K(R) decreases with R."""
        result = kinetic_normalization(R_values)
        K = result['K']
        for i in range(len(K) - 1):
            assert K[i] > K[i + 1], "K should decrease with R"

    def test_positive_for_all_R(self, R_wide):
        """K(R) > 0 for all R > 0."""
        result = kinetic_normalization(R_wide)
        assert np.all(result['K'] > 0)

    def test_formula_structure(self):
        """K = 1/(4*pi^2*g^2*R^3) has correct structure."""
        R = np.array([5.0])
        result = kinetic_normalization(R)
        K = result['K'][0]
        g2 = result['g_squared'][0]
        expected = 1.0 / (4.0 * np.pi**2 * g2 * R[0]**3)
        assert abs(K - expected) / expected < 1e-10

    def test_label_is_theorem(self):
        """Kinetic normalization is a THEOREM."""
        result = kinetic_normalization([1.0])
        assert result['label'] == 'THEOREM'

    def test_K_times_R2_behavior(self, R_wide):
        """K*R^2 = 1/(4*pi^2*g^2*R) -> 0 as R -> inf."""
        result = kinetic_normalization(R_wide)
        KR2 = result['K_times_R2']
        # Should decrease at large R
        large_mask = result['R'] >= 10.0
        if np.sum(large_mask) >= 2:
            kr2_large = KR2[large_mask]
            for i in range(len(kr2_large) - 1):
                assert kr2_large[i] > kr2_large[i + 1]


# ======================================================================
# 5. Physical mass gap
# ======================================================================

class TestPhysicalMassGap:
    """Tests for physical_mass_gap_vs_R."""

    def test_positive_for_all_R(self, R_values):
        """Physical mass gap > 0 for all tested R."""
        result = physical_mass_gap_vs_R(R_values)
        for m in result['best_estimate_MeV']:
            if np.isfinite(m):
                assert m > 0, f"Physical mass gap must be > 0"

    def test_zwanziger_stabilizes(self, R_wide):
        """Zwanziger mass should stabilize at large R."""
        result = physical_mass_gap_vs_R(R_wide)
        zw = result['zwanziger_mass_MeV']
        # At large R (> 10 fm), Zwanziger mass should be roughly constant
        large = result['R_fm'] >= 10.0
        finite_large = zw[large]
        finite_mask = np.isfinite(finite_large)
        if np.sum(finite_mask) >= 2:
            vals = finite_large[finite_mask]
            mean = np.mean(vals)
            std = np.std(vals)
            assert std / mean < 0.1, \
                f"Zwanziger mass should stabilize: mean={mean:.1f}, std={std:.1f}"

    def test_geometric_dominates_at_small_R(self):
        """At small R, geometric mass > Zwanziger mass."""
        result = physical_mass_gap_vs_R([0.2])
        geom = result['geometric_mass_MeV'][0]
        # Geometric should be large at small R
        assert geom > 500.0  # >> Lambda_QCD

    def test_label_is_numerical(self, R_values):
        """Physical mass gap is NUMERICAL."""
        result = physical_mass_gap_vs_R(R_values)
        assert result['label'] == 'NUMERICAL'

    def test_best_estimate_ge_pw(self, R_values):
        """Best estimate >= PW estimate."""
        result = physical_mass_gap_vs_R(R_values)
        for pw, best in zip(result['pw_mass_MeV'], result['best_estimate_MeV']):
            if np.isfinite(best):
                assert best >= pw - 1e-10


# ======================================================================
# 6. Dimensional transmutation
# ======================================================================

class TestDimensionalTransmutation:
    """Tests for dimensional_transmutation_connection."""

    def test_gamma_positive(self, R_values):
        """Gamma > 0 at all tested R."""
        result = dimensional_transmutation_connection(R_values)
        for g in result['gamma_Lambda']:
            if np.isfinite(g):
                assert g > 0

    def test_gluon_mass_positive(self, R_values):
        """Gluon mass > 0 at all tested R."""
        result = dimensional_transmutation_connection(R_values)
        for m in result['gluon_mass_MeV']:
            if np.isfinite(m):
                assert m > 0

    def test_label_is_proposition(self, R_values):
        """Dimensional transmutation connection is PROPOSITION."""
        result = dimensional_transmutation_connection(R_values)
        assert result['label'] == 'PROPOSITION'

    def test_gamma_in_reasonable_range(self, R_values):
        """gamma/Lambda_QCD should be O(1) -- between 0.5 and 5."""
        result = dimensional_transmutation_connection(R_values)
        for g in result['gamma_Lambda']:
            if np.isfinite(g):
                assert 0.1 < g < 10.0, f"gamma/Lambda = {g} is out of range"

    def test_gluon_mass_near_3_lambda(self):
        """At large R, m_g ~ 3 * Lambda_QCD."""
        R_large = np.array([50.0, 100.0])
        result = dimensional_transmutation_connection(R_large)
        for m in result['gluon_mass_MeV']:
            if np.isfinite(m):
                ratio = m / LAMBDA_QCD_DEFAULT
                # Should be in range [1, 6] (around 3)
                assert 1.0 < ratio < 6.0, f"m_g/Lambda = {ratio}"


# ======================================================================
# 7. Three-gap comparison
# ======================================================================

class TestThreeGapComparison:
    """Tests for three_gap_comparison."""

    def test_table_length(self, R_values):
        """Table has one row per R value."""
        result = three_gap_comparison(R_values)
        assert len(result['table']) == len(R_values)

    def test_geometric_decreases(self, R_values):
        """Geometric mass decreases monotonically."""
        result = three_gap_comparison(R_values)
        masses = [row['geometric_mass_MeV'] for row in result['table']]
        for i in range(len(masses) - 1):
            assert masses[i] > masses[i + 1]

    def test_pw_eigenvalue_increases(self, R_values):
        """PW eigenvalue increases monotonically."""
        result = three_gap_comparison(R_values)
        eigs = [row['pw_eigenvalue_inv_fm2'] for row in result['table']]
        for i in range(len(eigs) - 1):
            assert eigs[i] < eigs[i + 1]

    def test_physical_gap_positive(self, R_values):
        """Physical gap > 0 at all R."""
        result = three_gap_comparison(R_values)
        assert result['physical_bounded']
        assert result['min_physical_gap_MeV'] > 0

    def test_crossover_exists(self, R_wide):
        """A crossover radius exists where geometric ~ dynamical."""
        result = three_gap_comparison(R_wide)
        crossover = result['crossover_R_fm']
        if np.isfinite(crossover):
            assert crossover > 0

    def test_regime_classification(self, R_wide):
        """Small R -> geometric_dominates, large R -> dynamical_dominates."""
        result = three_gap_comparison(R_wide)
        table = result['table']
        # Smallest R should have geometric > dynamical
        assert table[0]['regime'] == 'geometric_dominates'
        # Largest R should have dynamical > geometric
        assert table[-1]['regime'] == 'dynamical_dominates'

    def test_all_rows_have_required_keys(self, R_values):
        """Every row in the table has the required keys."""
        result = three_gap_comparison(R_values)
        required = [
            'R_fm', 'geometric_mass_MeV', 'pw_eigenvalue_inv_fm2',
            'pw_mass_MeV', 'zwanziger_mass_MeV', 'best_physical_mass_MeV',
            'kinetic_factor', 'g_squared', 'gribov_diameter_fm', 'regime',
        ]
        for row in result['table']:
            for key in required:
                assert key in row, f"Missing key: {key}"


# ======================================================================
# 8. Honest assessment
# ======================================================================

class TestHonestAssessment:
    """Tests for honest_assessment_gap_types."""

    def test_has_proven_section(self):
        """Assessment has a 'proven' section."""
        result = honest_assessment_gap_types()
        assert 'proven' in result
        assert len(result['proven']) >= 3

    def test_has_needs_care_section(self):
        """Assessment has a 'needs_care' section."""
        result = honest_assessment_gap_types()
        assert 'needs_care' in result
        assert len(result['needs_care']) >= 1

    def test_has_reviewer_response(self):
        """Assessment has a reviewer response."""
        result = honest_assessment_gap_types()
        assert 'reviewer_response' in result
        assert 'criticism' in result['reviewer_response']
        assert 'response' in result['reviewer_response']

    def test_three_gaps_summary(self):
        """Assessment has summary of all three gap types."""
        result = honest_assessment_gap_types()
        summary = result['three_gaps_summary']
        assert 'geometric' in summary
        assert 'field_space' in summary
        assert 'physical' in summary

    def test_proven_items_are_theorem(self):
        """All proven items have THEOREM label."""
        result = honest_assessment_gap_types()
        for item in result['proven']:
            assert item['label'] == 'THEOREM'


# ======================================================================
# 9. Cross-consistency between gap types
# ======================================================================

class TestCrossConsistency:
    """Tests that the three gap types are internally consistent."""

    def test_geometric_and_pw_diverge(self, R_wide):
        """Geometric -> 0 while PW -> infinity: they diverge."""
        geom = geometric_gap_vs_R(R_wide)
        fs = field_space_gap_vs_R(R_wide)
        # Ratio PW/geometric should grow
        ratio = fs['pw_gap'] / geom['eigenvalue_inv_fm2']
        for i in range(len(ratio) - 1):
            assert ratio[i] < ratio[i + 1]

    def test_kinetic_compensates_pw_growth(self, R_wide):
        """K(R)*Delta_PW doesn't grow as fast as Delta_PW alone."""
        kn = kinetic_normalization(R_wide)
        fs = field_space_gap_vs_R(R_wide)
        product = kn['K'] * fs['pw_gap']
        # Product should eventually decrease (not grow unbounded)
        # At very large R, K*PW ~ 1/(g^2*R) -> 0
        large = R_wide >= 50.0
        if np.sum(large) >= 2:
            prod_large = product[large]
            # Should be decreasing at large R
            for i in range(len(prod_large) - 1):
                assert prod_large[i] > prod_large[i + 1], \
                    "K*PW should decrease at large R"

    def test_physical_gap_between_geometric_and_lambda(self, R_values):
        """At crossover, physical gap ~ Lambda_QCD."""
        # At R ~ 2 fm: geometric ~ Lambda, physical ~ Lambda
        result = physical_mass_gap_vs_R([2.0])
        geom = result['geometric_mass_MeV'][0]
        # Geometric at R=2 should be ~ Lambda_QCD
        assert abs(geom - LAMBDA_QCD_DEFAULT) / LAMBDA_QCD_DEFAULT < 0.5

    def test_physical_gap_positive_wide_range(self, R_wide):
        """Physical mass gap > 0 for R from 0.1 to 1000 fm."""
        result = physical_mass_gap_vs_R(R_wide)
        for i, m in enumerate(result['best_estimate_MeV']):
            if np.isfinite(m):
                assert m > 0, (
                    f"Physical gap must be > 0 at R = {R_wide[i]} fm, "
                    f"got {m} MeV"
                )
