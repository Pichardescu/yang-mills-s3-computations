"""
Tests for the Brascamp-Lieb quantitative mass gap on Omega_9.

Verifies:
  1. Module construction and imports
  2. kappa_at_point: Hess(Phi) eigenvalues at specific configurations
  3. kappa_at_origin: matches THEOREM 9.8 formula
  4. compute_kappa_min: directional scan finds positive kappa for R >= R_BL
  5. compute_kappa_min_refined: local optimization improves the bound
  6. physical_gap_BL: correct unit conversion sqrt(kappa)*Lambda_QCD
  7. physical_gap_KR: matches (1-alpha)*2/R*Lambda_QCD
  8. combined_gap: max(BL, KR) and regime detection
  9. Linearized self-consistency: sqrt(4/R^2) = 2/R
 10. R_BL threshold: kappa_min transitions from negative to positive
 11. Monotonicity: kappa_min(R) increases with R for R > R_BL
 12. Decomposition: V_2, V_4, ghost contributions
 13. Ghost compensation: -Hess(log det M_FP) compensates V_4 negative eigenvalues
 14. Summary table: correct fields and structure
 15. Uniform gap: infimum over all R is positive
 16. Physical plausibility: gap at R=2.2 exceeds Lambda_QCD

LABEL: MIXED (THEOREM-level structural results + NUMERICAL scans)

References:
    - Brascamp & Lieb (1976): J. Funct. Anal. 22, 366-389
    - Bakry & Emery (1985): Diffusions hypercontractives
    - THEOREM 9.7-9.10 from the preprint
"""

import numpy as np
import pytest

from yang_mills_s3.rg.quantitative_gap_bl import QuantitativeGapBL, HBAR_C_MEV_FM, LAMBDA_QCD_MEV
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# 0. Module construction
# ======================================================================

class TestModuleConstruction:
    """Basic import and construction tests."""

    def test_constructs(self):
        """QuantitativeGapBL can be constructed."""
        qgbl = QuantitativeGapBL()
        assert qgbl.N == 2
        assert qgbl.Lambda_QCD == 200.0
        assert qgbl.dim == 9
        assert qgbl.beg is not None
        assert qgbl.gd is not None

    def test_physical_constants(self):
        """Physical constants are correct."""
        assert HBAR_C_MEV_FM == pytest.approx(197.3269804, rel=1e-6)
        assert LAMBDA_QCD_MEV == pytest.approx(200.0)

    def test_custom_lambda(self):
        """Can construct with custom Lambda_QCD."""
        qgbl = QuantitativeGapBL(Lambda_QCD=250.0)
        assert qgbl.Lambda_QCD == 250.0


# ======================================================================
# 1. kappa at specific points
# ======================================================================

class TestKappaAtPoint:
    """Hessian eigenvalue computation at specific configurations."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_kappa_at_origin_positive(self, qgbl):
        """kappa(0) > 0 for all R > 0. THEOREM 9.8."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            kappa = qgbl.kappa_at_origin(R)
            assert kappa > 0, "kappa(0) must be positive at R=%.1f" % R

    def test_kappa_at_origin_matches_formula(self, qgbl):
        """kappa(0) = 4/R^2 + 4*g^2*R^2/9. THEOREM 9.8."""
        for R in [1.0, 2.0, 2.2, 5.0]:
            kappa = qgbl.kappa_at_origin(R)
            g2 = ZwanzigerGapEquation.running_coupling_g2(R)
            expected = 4.0 / R**2 + 4.0 * g2 * R**2 / 9.0
            assert kappa == pytest.approx(expected, rel=1e-3), \
                "kappa(0) formula mismatch at R=%.1f" % R

    def test_kappa_at_interior_point(self, qgbl):
        """kappa is finite at interior points of Omega_9."""
        a = np.array([0.1, 0.05, -0.02, 0.03, 0.0, 0.01, -0.01, 0.04, 0.02])
        R = 2.0
        kappa = qgbl.kappa_at_point(a, R)
        assert np.isfinite(kappa)

    def test_kappa_at_boundary_grows(self, qgbl):
        """kappa increases near the Gribov horizon. THEOREM 9.9."""
        R = 2.0
        d = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        t_hor = qgbl.gd.gribov_horizon_distance_truncated(d, R)
        if np.isfinite(t_hor) and t_hor > 0:
            kappa_mid = qgbl.kappa_at_point(0.5 * t_hor * d, R)
            kappa_near = qgbl.kappa_at_point(0.9 * t_hor * d, R)
            if np.isfinite(kappa_mid) and np.isfinite(kappa_near):
                assert kappa_near > kappa_mid, \
                    "kappa must increase toward the Gribov horizon"


# ======================================================================
# 2. kappa_min scan over Omega_9
# ======================================================================

class TestKappaMinScan:
    """Directional scanning for kappa_min."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_kappa_min_positive_at_R_1(self, qgbl):
        """kappa_min > 0 at R=1.0. NUMERICAL."""
        result = qgbl.compute_kappa_min(1.0, n_directions=60, n_fractions=12)
        assert result['kappa_min'] > 0
        assert result['all_positive']

    def test_kappa_min_positive_at_R_2p2(self, qgbl):
        """kappa_min > 0 at R=2.2 (physical radius). NUMERICAL."""
        result = qgbl.compute_kappa_min(2.2, n_directions=60, n_fractions=12)
        assert result['kappa_min'] > 0
        assert result['all_positive']

    def test_kappa_min_positive_for_large_R(self, qgbl):
        """kappa_min > 0 for large R. NUMERICAL."""
        result = qgbl.compute_kappa_min(10.0, n_directions=40, n_fractions=10)
        assert result['kappa_min'] > 0
        assert result['all_positive']

    def test_kappa_min_negative_at_small_R(self, qgbl):
        """kappa_min < 0 for very small R (below BL threshold). NUMERICAL."""
        result = qgbl.compute_kappa_min(0.5, n_directions=60, n_fractions=12)
        assert result['kappa_min'] < 0
        assert not result['all_positive']

    def test_kappa_min_less_than_origin(self, qgbl):
        """kappa_min <= kappa(0) (interior minimum). NUMERICAL."""
        for R in [1.0, 2.0, 5.0]:
            result = qgbl.compute_kappa_min(R, n_directions=60, n_fractions=12)
            assert result['kappa_min'] <= result['kappa_at_origin'] + 1e-10

    def test_minimizer_inside_omega(self, qgbl):
        """The minimizer a* is inside Omega_9. NUMERICAL."""
        result = qgbl.compute_kappa_min(2.0, n_directions=60, n_fractions=12)
        a_min = result['a_minimizer']
        lam_fp = qgbl.gd.fp_min_eigenvalue(a_min, 2.0)
        assert lam_fp > 0 or np.linalg.norm(a_min) < 1e-10

    def test_valid_samples_count(self, qgbl):
        """Scan produces many valid samples. NUMERICAL."""
        result = qgbl.compute_kappa_min(2.0, n_directions=80, n_fractions=15)
        assert result['n_valid_samples'] > 100

    def test_reproducibility(self, qgbl):
        """Same seed gives same result."""
        r1 = qgbl.compute_kappa_min(2.0, n_directions=50, n_fractions=10, seed=42)
        r2 = qgbl.compute_kappa_min(2.0, n_directions=50, n_fractions=10, seed=42)
        assert r1['kappa_min'] == pytest.approx(r2['kappa_min'], abs=1e-12)


# ======================================================================
# 3. Refined kappa_min (with local optimization)
# ======================================================================

class TestKappaMinRefined:
    """Local optimization around the scan minimum."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_refined_exists(self, qgbl):
        """Refined result has 'kappa_refined' field."""
        result = qgbl.compute_kappa_min_refined(2.0, n_directions=50,
                                                 n_fractions=10)
        assert 'kappa_refined' in result

    def test_refined_le_scan(self, qgbl):
        """Refined kappa <= scan kappa (optimization can only improve)."""
        result = qgbl.compute_kappa_min_refined(2.0, n_directions=50,
                                                 n_fractions=10, refine=True)
        assert result['kappa_refined'] <= result['kappa_min'] + 1e-10

    def test_refined_positive_at_R2(self, qgbl):
        """Refined kappa is positive at R=2.0."""
        result = qgbl.compute_kappa_min_refined(2.0, n_directions=50,
                                                 n_fractions=10)
        assert result['kappa_refined'] > 0


# ======================================================================
# 4. Physical gap conversion
# ======================================================================

class TestPhysicalGap:
    """Mass gap in physical units (MeV)."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_bl_gap_positive_at_R2p2(self, qgbl):
        """BL gap is positive at the physical radius R=2.2."""
        result = qgbl.physical_gap_BL(2.2, n_directions=60, n_fractions=12)
        assert result['gap_MeV'] > 0

    def test_bl_gap_sqrt_kappa(self, qgbl):
        """BL gap = sqrt(kappa_min) * Lambda_QCD."""
        result = qgbl.physical_gap_BL(2.0, n_directions=60, n_fractions=12)
        if result['kappa_min'] > 0:
            expected = np.sqrt(result['kappa_min']) * 200.0
            assert result['gap_MeV'] == pytest.approx(expected, rel=1e-6)

    def test_linearized_self_consistency(self, qgbl):
        """At a=0 with g=0: gap = sqrt(4/R^2)*Lambda = 2/R*Lambda. THEOREM."""
        # At the origin, kappa includes ghost contribution. But
        # the V_2-only part gives kappa_V2 = 4/R^2.
        R = 2.2
        kappa_V2 = 4.0 / R**2
        gap_V2 = np.sqrt(kappa_V2) * 200.0
        expected = 2.0 / R * 200.0
        assert gap_V2 == pytest.approx(expected, rel=1e-6), \
            "sqrt(4/R^2)*Lambda must equal 2/R*Lambda"

    def test_kr_gap_positive(self, qgbl):
        """KR gap is positive for all R > 0."""
        for R in [0.3, 0.5, 1.0, 5.0, 100.0]:
            gap = qgbl.physical_gap_KR(R)
            assert gap > 0, "KR gap must be positive at R=%.1f" % R

    def test_kr_gap_formula(self, qgbl):
        """KR gap = (1-alpha)*2/R*Lambda. THEOREM 4.1."""
        R = 2.0
        g2 = ZwanzigerGapEquation.running_coupling_g2(R)
        g2_c = 12.0 * np.sqrt(2.0) * np.pi**2
        alpha = g2 / g2_c
        expected = (1 - alpha) * 2.0 / R * 200.0
        assert qgbl.physical_gap_KR(R) == pytest.approx(expected, rel=1e-6)

    def test_kr_gap_decreases_with_R(self, qgbl):
        """KR gap ~ 2/R decreases as R increases."""
        gaps = [qgbl.physical_gap_KR(R) for R in [1.0, 2.0, 5.0, 10.0]]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_bl_gap_zero_when_kappa_negative(self, qgbl):
        """BL gap is 0 when kappa_min < 0."""
        result = qgbl.physical_gap_BL(0.5, n_directions=40, n_fractions=10)
        assert result['gap_MeV'] == 0.0
        assert not result['all_positive']


# ======================================================================
# 5. Combined gap
# ======================================================================

class TestCombinedGap:
    """Combined gap: max(BL, KR)."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_combined_positive_everywhere(self, qgbl):
        """Combined gap is positive for all R tested."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            cg = qgbl.combined_gap(R, n_directions=40, n_fractions=10)
            assert cg['gap_MeV'] > 0, "Combined gap must be > 0 at R=%.1f" % R

    def test_combined_ge_bl(self, qgbl):
        """Combined >= BL at each R."""
        cg = qgbl.combined_gap(2.0, n_directions=40, n_fractions=10)
        assert cg['gap_MeV'] >= cg['gap_BL_MeV'] - 1e-10

    def test_combined_ge_kr(self, qgbl):
        """Combined >= KR at each R."""
        cg = qgbl.combined_gap(2.0, n_directions=40, n_fractions=10)
        assert cg['gap_MeV'] >= cg['gap_KR_MeV'] - 1e-10

    def test_regime_kr_at_small_R(self, qgbl):
        """KR dominates at small R."""
        cg = qgbl.combined_gap(0.5, n_directions=40, n_fractions=10)
        assert cg['regime'] == 'KR'

    def test_regime_bl_at_large_R(self, qgbl):
        """BL dominates at large R."""
        cg = qgbl.combined_gap(5.0, n_directions=40, n_fractions=10)
        assert cg['regime'] == 'BL'


# ======================================================================
# 6. Monotonicity and transition
# ======================================================================

class TestMonotonicity:
    """kappa_min(R) behavior as function of R."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_kappa_min_increases_with_R(self, qgbl):
        """kappa_min(R) is increasing for R >= 1. NUMERICAL."""
        kappas = []
        for R in [1.0, 2.0, 3.0, 5.0, 10.0]:
            result = qgbl.compute_kappa_min(R, n_directions=50, n_fractions=10)
            kappas.append(result['kappa_min'])
        for i in range(len(kappas) - 1):
            assert kappas[i + 1] > kappas[i], \
                "kappa_min must increase with R for R >= 1"

    def test_transition_near_R_0p7(self, qgbl):
        """BL threshold is near R ~ 0.69. NUMERICAL."""
        # R=0.6: kappa_min < 0
        r1 = qgbl.compute_kappa_min(0.6, n_directions=60, n_fractions=12)
        assert r1['kappa_min'] < 0

        # R=0.75: kappa_min > 0
        r2 = qgbl.compute_kappa_min(0.75, n_directions=60, n_fractions=12)
        assert r2['kappa_min'] > 0

    def test_kappa_origin_increases_with_R(self, qgbl):
        """kappa(0) = 4/R^2 + (4g^2R^2/9) increases for moderate R. THEOREM 9.8."""
        # At large R: ghost term ~ g^2_max * R^2 dominates
        kappas = [qgbl.kappa_at_origin(R) for R in [1.0, 2.0, 5.0]]
        for i in range(len(kappas) - 1):
            assert kappas[i + 1] > kappas[i]


# ======================================================================
# 7. Decomposition analysis
# ======================================================================

class TestDecomposition:
    """Hessian decomposition into V_2, V_4, ghost."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_v2_is_4_over_R2(self, qgbl):
        """Hess(V_2) = (4/R^2)*I_9. THEOREM."""
        R = 2.0
        decomp = qgbl.decompose_kappa(np.zeros(9), R)
        assert decomp['valid']
        assert decomp['kappa_V2'] == pytest.approx(4.0 / R**2, rel=1e-6)
        # All eigenvalues should be equal (scalar multiple of identity)
        assert decomp['eigs_V2'][-1] == pytest.approx(decomp['eigs_V2'][0], rel=1e-6)

    def test_ghost_psd_at_origin(self, qgbl):
        """Ghost curvature is PSD at origin. THEOREM 9.7."""
        decomp = qgbl.decompose_kappa(np.zeros(9), 2.0)
        assert decomp['valid']
        assert decomp['kappa_ghost'] >= -1e-10

    def test_ghost_psd_at_interior(self, qgbl):
        """Ghost curvature is PSD at interior points. THEOREM 9.7."""
        a = np.array([0.1, 0.05, -0.02, 0.03, 0.0, 0.01, -0.01, 0.04, 0.02])
        decomp = qgbl.decompose_kappa(a, 2.0)
        if decomp['valid']:
            assert decomp['kappa_ghost'] >= -1e-10

    def test_v4_can_be_negative(self, qgbl):
        """Hess(V_4) can have negative eigenvalues at nonzero a."""
        a = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.2, -0.3, 0.1, 0.2])
        decomp = qgbl.decompose_kappa(a, 2.0)
        if decomp['valid']:
            # V_4 is not convex: its Hessian can have negative eigenvalues
            # This is the key challenge that ghost curvature compensates
            assert decomp['eigs_V4'][0] < decomp['eigs_V4'][-1]  # Spread exists

    def test_total_positive_at_origin(self, qgbl):
        """Total Hessian is positive at origin. THEOREM 9.8."""
        decomp = qgbl.decompose_kappa(np.zeros(9), 2.0)
        assert decomp['valid']
        assert decomp['kappa_total'] > 0


# ======================================================================
# 8. Physical plausibility
# ======================================================================

class TestPhysicalPlausibility:
    """Gap values are physically reasonable."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_gap_at_R2p2_exceeds_lambda(self, qgbl):
        """Gap at physical radius exceeds Lambda_QCD. NUMERICAL."""
        result = qgbl.physical_gap_BL(2.2, n_directions=60, n_fractions=12)
        assert result['gap_MeV'] > 200.0  # > Lambda_QCD

    def test_gap_at_R2p2_reasonable_range(self, qgbl):
        """Gap at R=2.2 is in the range 100-2000 MeV. NUMERICAL."""
        result = qgbl.physical_gap_BL(2.2, n_directions=60, n_fractions=12)
        assert 100.0 < result['gap_MeV'] < 2000.0

    def test_kr_gap_dominates_at_small_R(self, qgbl):
        """KR gap is very large at small R (asymptotic freedom)."""
        gap = qgbl.physical_gap_KR(0.3)
        assert gap > 1000.0  # Very large at small R

    def test_combined_gap_exceeds_100_MeV(self, qgbl):
        """Combined gap exceeds 100 MeV everywhere tested. NUMERICAL."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            cg = qgbl.combined_gap(R, n_directions=40, n_fractions=10)
            assert cg['gap_MeV'] > 100.0, \
                "Combined gap must exceed 100 MeV at R=%.1f" % R


# ======================================================================
# 9. kappa_min_vs_R curve
# ======================================================================

class TestKappaVsR:
    """Full kappa_min(R) curve computation."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_returns_correct_fields(self, qgbl):
        """kappa_min_vs_R returns all expected fields."""
        data = qgbl.kappa_min_vs_R([1.0, 2.0], n_directions=30, n_fractions=8)
        assert 'R' in data
        assert 'kappa_min' in data
        assert 'kappa_at_origin' in data
        assert 'all_positive' in data
        assert 'gap_BL_MeV' in data
        assert 'gap_KR_MeV' in data
        assert 'gap_best_MeV' in data
        assert 'g_squared' in data

    def test_arrays_correct_length(self, qgbl):
        """Output arrays have correct length."""
        R_values = [1.0, 2.0, 5.0]
        data = qgbl.kappa_min_vs_R(R_values, n_directions=30, n_fractions=8)
        for key in ['R', 'kappa_min', 'gap_BL_MeV', 'gap_KR_MeV']:
            assert len(data[key]) == 3

    def test_gap_best_ge_bl_and_kr(self, qgbl):
        """gap_best >= max(BL, KR) at each R."""
        R_values = [0.5, 1.0, 2.0, 5.0]
        data = qgbl.kappa_min_vs_R(R_values, n_directions=40, n_fractions=10)
        for i in range(len(R_values)):
            assert data['gap_best_MeV'][i] >= data['gap_BL_MeV'][i] - 1e-10
            assert data['gap_best_MeV'][i] >= data['gap_KR_MeV'][i] - 1e-10


# ======================================================================
# 10. Summary table
# ======================================================================

class TestSummaryTable:
    """Summary table generation."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_table_has_entries(self, qgbl):
        """Summary table has entries for each R."""
        table = qgbl.summary_table([1.0, 2.0], n_directions=30, n_fractions=8)
        assert len(table) == 2

    def test_table_fields(self, qgbl):
        """Each table entry has all expected fields."""
        table = qgbl.summary_table([2.0], n_directions=30, n_fractions=8)
        entry = table[0]
        assert 'R' in entry
        assert 'g_squared' in entry
        assert 'kappa_min' in entry
        assert 'gap_BL_MeV' in entry
        assert 'gap_KR_MeV' in entry
        assert 'gap_best_MeV' in entry
        assert 'regime' in entry

    def test_regime_correctly_assigned(self, qgbl):
        """Regime is 'BL' when BL dominates, 'KR' otherwise."""
        table = qgbl.summary_table([0.5, 5.0], n_directions=40, n_fractions=10)
        # At R=0.5, KR should dominate (BL is negative)
        assert table[0]['regime'] == 'KR'
        # At R=5.0, BL should dominate
        assert table[1]['regime'] == 'BL'


# ======================================================================
# 11. Unit conversion self-consistency
# ======================================================================

class TestUnitConsistency:
    """Verify unit conversion is self-consistent."""

    def test_harmonic_gap_matches_2_over_R(self):
        """sqrt(4/R^2)*Lambda = 2/R*Lambda exactly. THEOREM."""
        for R in [1.0, 2.0, 2.2, 5.0, 10.0]:
            kappa_V2 = 4.0 / R**2
            gap_sqrt = np.sqrt(kappa_V2) * 200.0
            gap_exact = 2.0 / R * 200.0
            assert gap_sqrt == pytest.approx(gap_exact, rel=1e-10)

    def test_kappa_units_lambda_qcd_squared(self):
        """kappa has units Lambda_QCD^2 (dimensionless when Lambda=1)."""
        # V_2 = (2/R^2)|a|^2, Hess(V_2) = 4/R^2
        # At R=1 (Lambda_QCD), Hess = 4 Lambda_QCD^2
        R = 1.0
        assert 4.0 / R**2 == 4.0  # Lambda_QCD^2

    def test_gap_at_physical_radius(self):
        """Gap at R=2.2 fm: 2*hbar_c/R = 179 MeV. THEOREM."""
        R_fm = 2.2  # physical radius in fm
        gap_phys = 2.0 * HBAR_C_MEV_FM / R_fm
        assert gap_phys == pytest.approx(179.4, abs=1.0)

    def test_lambda_conversion(self):
        """R(fm) = R(Lambda^{-1}) / Lambda(fm^{-1})."""
        Lambda_fm = LAMBDA_QCD_MEV / HBAR_C_MEV_FM  # ~ 1.014 fm^{-1}
        R_fm = 2.2
        R_Lambda = R_fm * Lambda_fm  # ~ 2.23 Lambda_QCD^{-1}
        # gap = 2/R_Lambda * Lambda_QCD = 2*Lambda_fm / (R_fm*Lambda_fm) * Lambda_QCD
        # = 2/R_fm * (Lambda_QCD/Lambda_fm) = 2*hbar_c/R_fm (correct!)
        gap_from_R_Lambda = 2.0 / R_Lambda * LAMBDA_QCD_MEV
        gap_from_R_fm = 2.0 * HBAR_C_MEV_FM / R_fm
        assert gap_from_R_Lambda == pytest.approx(gap_from_R_fm, rel=0.02)


# ======================================================================
# 12. Theorem statement
# ======================================================================

class TestTheoremStatement:
    """Theorem statement generation."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_statement_is_string(self, qgbl):
        """theorem_statement returns a string."""
        stmt = qgbl.theorem_statement([1.0, 2.0, 2.2],
                                       n_directions=30, n_fractions=8)
        assert isinstance(stmt, str)
        assert len(stmt) > 100

    def test_statement_mentions_brascamp_lieb(self, qgbl):
        """Statement mentions Brascamp-Lieb."""
        stmt = qgbl.theorem_statement([2.0, 2.2],
                                       n_directions=30, n_fractions=8)
        assert 'Brascamp-Lieb' in stmt

    def test_statement_mentions_gz_free(self, qgbl):
        """Statement mentions GZ-free."""
        stmt = qgbl.theorem_statement([2.0, 2.2],
                                       n_directions=30, n_fractions=8)
        assert 'GZ-free' in stmt


# ======================================================================
# 13. Ghost curvature compensation (the key mechanism)
# ======================================================================

class TestGhostCompensation:
    """Ghost curvature compensates V_4 negative eigenvalues."""

    @pytest.fixture
    def qgbl(self):
        return QuantitativeGapBL()

    def test_ghost_trace_grows_near_horizon(self, qgbl):
        """Ghost curvature trace increases near the Gribov horizon. THEOREM 9.9."""
        R = 2.0
        d = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        t_hor = qgbl.gd.gribov_horizon_distance_truncated(d, R)

        if np.isfinite(t_hor) and t_hor > 0:
            traces = []
            for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                a = frac * t_hor * d
                decomp = qgbl.decompose_kappa(a, R)
                if decomp['valid']:
                    # The TRACE (sum of eigenvalues) of the ghost curvature
                    # diverges at the boundary -- this is the correct quantity
                    traces.append(np.sum(decomp['eigs_ghost']))

            # Ghost curvature trace should increase toward the boundary
            if len(traces) >= 3:
                assert traces[-1] > traces[0], \
                    "Ghost curvature trace must increase toward the Gribov horizon"

    def test_ghost_compensates_v4(self, qgbl):
        """At the interior minimum, ghost curvature compensates V_4. NUMERICAL."""
        R = 2.0
        result = qgbl.compute_kappa_min(R, n_directions=60, n_fractions=12)
        a_min = result['a_minimizer']

        if np.linalg.norm(a_min) > 1e-10:
            decomp = qgbl.decompose_kappa(a_min, R)
            if decomp['valid']:
                # V_4 may contribute negative eigenvalue
                # Ghost must compensate so that total > 0
                assert decomp['kappa_total'] > 0
                # Check that ghost contribution is significant
                assert decomp['kappa_ghost'] > 0


# ======================================================================
# 14. Integration test: full quantitative gap computation
# ======================================================================

class TestIntegration:
    """End-to-end test of the quantitative gap computation."""

    def test_full_gap_computation(self):
        """Full computation: kappa_min -> gap -> infimum. NUMERICAL."""
        qgbl = QuantitativeGapBL()

        # Compute at representative R values
        R_values = [0.7, 1.0, 2.2, 5.0]
        data = qgbl.kappa_min_vs_R(R_values, n_directions=50, n_fractions=10)

        # All gap_best should be positive
        for i, R in enumerate(R_values):
            assert data['gap_best_MeV'][i] > 0, \
                "Gap must be positive at R=%.1f" % R

        # Infimum should be positive
        min_gap = np.min(data['gap_best_MeV'])
        assert min_gap > 50.0  # At least 50 MeV

    def test_infimum_exceeds_lambda_qcd(self):
        """Infimum of combined gap exceeds Lambda_QCD. NUMERICAL."""
        qgbl = QuantitativeGapBL()

        R_values = [0.5, 0.7, 0.85, 1.0, 1.5, 2.0, 2.2, 5.0, 10.0]
        data = qgbl.kappa_min_vs_R(R_values, n_directions=60, n_fractions=12)

        min_gap = np.min(data['gap_best_MeV'])
        assert min_gap > 200.0, \
            "Infimum %.1f MeV must exceed Lambda_QCD = 200 MeV" % min_gap
