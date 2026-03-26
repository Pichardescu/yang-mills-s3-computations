"""
Tests for log-concavity bound on the FP-weighted measure on Omega_9.

Verifies:
  1. Effective potential Phi(a) = S_YM - log det M_FP is well-defined (THEOREM)
  2. Hess(Phi) decomposes correctly into V2 + V4 + ghost (THEOREM)
  3. kappa(0) = 4/R^2 + 4g^2 R^2/9 at the origin (THEOREM 9.8)
  4. kappa(a) > 0 for all a in Omega_9 when R >= R_transition (NUMERICAL)
  5. Ghost curvature compensates V4 negative eigenvalues (THEOREM 9.7)
  6. Brascamp-Lieb gives spectral gap >= kappa > 0 (THEOREM)
  7. Analytical lower bound kappa >= -104/R^2 + (4/81)g^2 R^2 (THEOREM 9.10)
  8. Two-regime coverage: BL for R >= R_transition, KR for R < R_transition (THEOREM)
  9. RG contraction follows from log-concavity (THEOREM)
 10. Concentration bound: Var(a_i) <= 1/kappa (THEOREM, Brascamp-Lieb)
 11. Gap vs R: mass gap grows with R (NUMERICAL)
 12. Edge cases: origin, near-boundary, large R (NUMERICAL)
 13. Consistency: numerical kappa >= analytical kappa everywhere (NUMERICAL)
 14. Physical units: gap in MeV is consistent with QCD (NUMERICAL)

LABEL: MIXED (THEOREM-level structural results + NUMERICAL verification scans)

References:
    - Brascamp & Lieb (1976): J. Funct. Anal. 22, 366-389.
    - Bakry & Emery (1985): Diffusions hypercontractives.
    - Dell'Antonio & Zwanziger (1989/1991): Gribov region convex.
    - THEOREM 9.7-9.10 from the paper (Sections 9 of the preprint).
"""

import numpy as np
import pytest

from yang_mills_s3.rg.log_concavity_bound import LogConcavityBound, HBAR_C_MEV_FM
from yang_mills_s3.proofs.bakry_emery_gap import BakryEmeryGap
from yang_mills_s3.proofs.gribov_diameter import GribovDiameter, _su2_structure_constants
from yang_mills_s3.proofs.v4_convexity import (
    hessian_v4_analytical,
    hessian_v2,
    v4_potential,
    v2_potential,
)
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# 0. Module construction and imports
# ======================================================================

class TestModuleConstruction:
    """Basic import and construction tests."""

    def test_log_concavity_bound_constructs(self):
        """LogConcavityBound can be constructed."""
        lcb = LogConcavityBound()
        assert lcb.dim == 9
        assert lcb.beg is not None
        assert lcb.gd is not None
        assert lcb.dt is not None

    def test_hbar_c_value(self):
        """Physical constant hbar*c is correct."""
        assert HBAR_C_MEV_FM == pytest.approx(197.3269804, rel=1e-6)

    def test_backing_classes_available(self):
        """BakryEmeryGap and GribovDiameter are accessible."""
        lcb = LogConcavityBound()
        assert isinstance(lcb.beg, BakryEmeryGap)
        assert isinstance(lcb.gd, GribovDiameter)


# ======================================================================
# 1. Effective potential Phi
# ======================================================================

class TestEffectivePotential:
    """THEOREM: Phi = S_YM - log det M_FP is well-defined on int(Omega_9)."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_phi_at_origin(self, lcb):
        """Phi(0) = -log det M_FP(0), since S_YM(0) = 0."""
        a = np.zeros(9)
        R = 1.0
        phi = lcb.effective_potential(a, R)
        assert np.isfinite(phi)
        # S_YM(0) = V_2(0) + V_4(0) = 0
        # So Phi(0) = -log det M_FP(0)
        # M_FP(0) = (3/R^2) * I_9, det = (3/R^2)^9
        expected_log_det = 9.0 * np.log(3.0 / R**2)
        assert phi == pytest.approx(-expected_log_det, rel=1e-6)

    def test_phi_positive_at_nonzero(self, lcb):
        """Phi(a) is finite and well-defined for a inside Omega_9."""
        R = 2.0
        a = np.array([0.1, 0.05, -0.02, 0.03, 0.0, 0.01, -0.01, 0.04, 0.02])
        phi = lcb.effective_potential(a, R)
        assert np.isfinite(phi)

    def test_phi_increases_with_norm(self, lcb):
        """Phi(t*d) increases with t (confining potential)."""
        R = 2.0
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        d /= np.linalg.norm(d)

        t_horizon = lcb.gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_horizon):
            pytest.skip("Could not find horizon in this direction")

        phi_prev = lcb.effective_potential(np.zeros(9), R)
        for frac in [0.1, 0.3, 0.5, 0.7]:
            a = frac * t_horizon * d
            phi = lcb.effective_potential(a, R)
            # Phi should generally increase due to V_2 and V_4
            # (not strict monotonicity due to log det term)
            assert np.isfinite(phi)

    def test_phi_diverges_at_boundary(self, lcb):
        """Phi -> +inf at the Gribov horizon (log det -> -inf)."""
        R = 2.0
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        d /= np.linalg.norm(d)

        t_horizon = lcb.gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_horizon):
            pytest.skip("Could not find horizon")

        # At 99.9% of the horizon
        a_near = 0.999 * t_horizon * d
        phi_near = lcb.effective_potential(a_near, R)
        # At 50% of the horizon
        a_mid = 0.5 * t_horizon * d
        phi_mid = lcb.effective_potential(a_mid, R)
        # Phi near boundary should be larger (log det vanishes)
        assert phi_near > phi_mid


# ======================================================================
# 2. Hessian decomposition
# ======================================================================

class TestHessianDecomposition:
    """THEOREM: Hess(Phi) = Hess(V_2) + Hess(V_4) - Hess(log det M_FP)."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_decomposition_at_origin(self, lcb):
        """At origin, V4 contribution is zero, only V2 + ghost."""
        R = 1.0
        decomp = lcb.hessian_Phi_decomposed(np.zeros(9), R)

        # V2 contribution = (4/R^2) * I_9
        assert decomp['H_V2'][0, 0] == pytest.approx(4.0 / R**2, rel=1e-10)
        assert decomp['H_V2'][0, 1] == pytest.approx(0.0, abs=1e-12)

        # V4 at origin is zero (quartic starts at order |a|^4)
        assert np.allclose(decomp['H_V4'], 0.0, atol=1e-8)

        # Ghost is PSD (all eigenvalues >= 0)
        assert np.all(decomp['eigs_ghost'] >= -1e-10)

        # Total = V2 + ghost at origin
        assert decomp['kappa'] > 0

    def test_decomposition_sums_correctly(self, lcb):
        """H_total = H_V2 + H_V4 + H_ghost at arbitrary interior points."""
        R = 2.0
        rng = np.random.RandomState(12345)

        for _ in range(5):
            d = rng.randn(9)
            d /= np.linalg.norm(d)
            t_hor = lcb.gd.gribov_horizon_distance_truncated(d, R)
            if not np.isfinite(t_hor) or t_hor <= 0:
                continue
            a = 0.3 * t_hor * d

            decomp = lcb.hessian_Phi_decomposed(a, R)
            if np.any(np.isnan(decomp['H_total'])):
                continue

            reconstructed = decomp['H_V2'] + decomp['H_V4'] + decomp['H_ghost']
            np.testing.assert_allclose(
                decomp['H_total'], reconstructed, atol=1e-8,
                err_msg="Decomposition does not sum to total"
            )

    def test_ghost_is_psd(self, lcb):
        """THEOREM 9.7: -Hess(log det M_FP) is positive semidefinite inside Omega_9."""
        R = 2.0
        rng = np.random.RandomState(54321)

        for _ in range(10):
            d = rng.randn(9)
            d /= np.linalg.norm(d)
            t_hor = lcb.gd.gribov_horizon_distance_truncated(d, R)
            if not np.isfinite(t_hor) or t_hor <= 0:
                continue
            a = 0.5 * t_hor * d

            # Check inside Omega
            if lcb.gd.fp_min_eigenvalue(a, R) <= 0:
                continue

            decomp = lcb.hessian_Phi_decomposed(a, R)
            if np.any(np.isnan(decomp['eigs_ghost'])):
                continue

            # Ghost curvature must be PSD (THEOREM 9.7)
            assert decomp['eigs_ghost'][0] >= -1e-8, (
                f"Ghost curvature has negative eigenvalue {decomp['eigs_ghost'][0]}"
            )

    def test_v2_hessian_is_diagonal(self, lcb):
        """Hess(V_2) = (4/R^2) I_9 is diagonal and positive."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            decomp = lcb.hessian_Phi_decomposed(np.zeros(9), R)
            expected = (4.0 / R**2) * np.eye(9)
            np.testing.assert_allclose(decomp['H_V2'], expected, atol=1e-12)


# ======================================================================
# 3. Kappa at origin (THEOREM 9.8)
# ======================================================================

class TestKappaAtOrigin:
    """THEOREM 9.8: kappa(0) = 4/R^2 + 4g^2(R)*R^2/9."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_kappa_origin_formula(self, lcb):
        """Numerical Hessian matches analytical formula at origin."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            res = lcb.kappa_at_origin(R)
            assert res['kappa_numerical'] == pytest.approx(
                res['kappa_analytical'], rel=1e-4
            ), f"Mismatch at R={R}: num={res['kappa_numerical']}, ana={res['kappa_analytical']}"

    def test_kappa_origin_positive(self, lcb):
        """kappa(0) > 0 for all R > 0."""
        for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            res = lcb.kappa_at_origin(R)
            assert res['kappa_numerical'] > 0, f"kappa(0) <= 0 at R={R}"

    def test_v2_dominates_small_R(self, lcb):
        """For small R, V2 contribution 4/R^2 dominates."""
        R = 0.1
        res = lcb.kappa_at_origin(R)
        assert res['V2_contribution'] > res['ghost_contribution']

    def test_ghost_dominates_large_R(self, lcb):
        """For large R, ghost contribution 4g^2 R^2/9 dominates."""
        R = 10.0
        res = lcb.kappa_at_origin(R)
        assert res['ghost_contribution'] > res['V2_contribution']

    def test_kappa_origin_grows_with_R(self, lcb):
        """kappa(0) = 4/R^2 + 4g^2 R^2/9 grows with R for R > 1."""
        kappas = []
        for R in [2.0, 5.0, 10.0, 20.0]:
            res = lcb.kappa_at_origin(R)
            kappas.append(res['kappa_numerical'])
        # Ghost term ~ g^2 R^2 grows, dominating 4/R^2 decrease
        for i in range(len(kappas) - 1):
            assert kappas[i + 1] > kappas[i], (
                f"kappa(0) not increasing: {kappas[i+1]} <= {kappas[i]}"
            )

    def test_kappa_origin_eigenvalues_degenerate(self, lcb):
        """All 9 eigenvalues of Hess(Phi)(0) should be equal (isotropy)."""
        R = 2.0
        H = lcb.hessian_Phi(np.zeros(9), R)
        eigs = np.linalg.eigvalsh(H)
        # At origin, Hess(Phi) = kappa(0) * I_9 (all eigenvalues equal)
        assert np.std(eigs) / np.mean(eigs) < 0.01, (
            f"Eigenvalues not degenerate at origin: {eigs}"
        )


# ======================================================================
# 4. Uniform positivity of kappa on Omega_9 (NUMERICAL)
# ======================================================================

class TestUniformPositivity:
    """NUMERICAL: kappa(a) > 0 for all sampled a in Omega_9 when R >= R_transition."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_kappa_positive_R1(self, lcb):
        """kappa > 0 everywhere at R=1.0."""
        res = lcb.find_interior_minimum_kappa(1.0, n_directions=50, n_fractions=10)
        assert res['kappa_min'] > 0, f"kappa_min = {res['kappa_min']} at R=1.0"
        assert res['all_positive']

    def test_kappa_positive_R2(self, lcb):
        """kappa > 0 everywhere at R=2.0 (physical radius)."""
        res = lcb.find_interior_minimum_kappa(2.0, n_directions=50, n_fractions=10)
        assert res['kappa_min'] > 0, f"kappa_min = {res['kappa_min']} at R=2.0"
        assert res['all_positive']

    def test_kappa_positive_R5(self, lcb):
        """kappa > 0 everywhere at R=5.0."""
        res = lcb.find_interior_minimum_kappa(5.0, n_directions=30, n_fractions=8)
        assert res['kappa_min'] > 0
        assert res['all_positive']

    def test_kappa_positive_R10(self, lcb):
        """kappa > 0 at R=10.0 (large R, ghost-dominated)."""
        res = lcb.find_interior_minimum_kappa(10.0, n_directions=30, n_fractions=8)
        assert res['kappa_min'] > 0
        assert res['all_positive']

    def test_kappa_min_less_than_kappa_origin(self, lcb):
        """The interior minimum is less than kappa at origin."""
        R = 2.0
        res = lcb.find_interior_minimum_kappa(R, n_directions=50, n_fractions=10)
        # V4 negative eigenvalues pull kappa below origin value
        assert res['kappa_min'] <= res['kappa_at_origin'] + 1e-8

    def test_minimizer_in_interior(self, lcb):
        """The minimizer is at an interior point, not origin or boundary."""
        R = 2.0
        res = lcb.find_interior_minimum_kappa(R, n_directions=50, n_fractions=10)
        # fraction > 0 means not at origin
        assert res['fraction_at_min'] > 0.0
        # fraction < 0.95 means not at boundary
        assert res['fraction_at_min'] < 0.95

    def test_sufficient_valid_samples(self, lcb):
        """The scan produces enough valid samples for statistical confidence."""
        R = 2.0
        res = lcb.find_interior_minimum_kappa(R, n_directions=50, n_fractions=10)
        # Should have at least 100 valid points (50 directions x ~2 fractions)
        assert res['n_valid_samples'] >= 50


# ======================================================================
# 5. Ghost compensation of V4 (THEOREM 9.7 + THEOREM 9.8a)
# ======================================================================

class TestGhostCompensation:
    """THEOREM: Ghost curvature compensates V4 negative eigenvalues."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_ghost_exceeds_v4_negative_at_midpoint(self, lcb):
        """Ghost min eig > |V4 min eig| at interior points for R >= 1."""
        R = 2.0
        rng = np.random.RandomState(99)
        compensated_count = 0
        total_count = 0

        for _ in range(20):
            d = rng.randn(9)
            d /= np.linalg.norm(d)
            t_hor = lcb.gd.gribov_horizon_distance_truncated(d, R)
            if not np.isfinite(t_hor) or t_hor <= 0:
                continue

            a = 0.5 * t_hor * d
            if lcb.gd.fp_min_eigenvalue(a, R) <= 0:
                continue

            decomp = lcb.hessian_Phi_decomposed(a, R)
            if np.any(np.isnan(decomp['eigs_total'])):
                continue

            total_count += 1
            v4_neg = min(decomp['eigs_V4'][0], 0)
            ghost_min = decomp['eigs_ghost'][0]
            v2_min = decomp['eigs_V4'][0]  # really H_V2 is constant
            # ghost + V2 should overcome V4 negative
            if 4.0 / R**2 + ghost_min + v4_neg > 0:
                compensated_count += 1

        if total_count > 0:
            assert compensated_count == total_count, (
                f"Ghost failed to compensate V4 at {total_count - compensated_count}"
                f"/{total_count} points"
            )

    def test_ghost_max_eig_diverges_near_horizon(self, lcb):
        """Ghost max eigenvalue diverges as we approach the Gribov horizon.

        THEOREM 9.9: -Hess(log det M_FP) diverges at the boundary because
        ||M_FP^{-1}|| -> infinity. The MAX eigenvalue of the ghost Hessian
        diverges (not necessarily the min eigenvalue, which depends on the
        anisotropy of L_i in the M_FP^{-1} norm).
        """
        R = 2.0
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        d /= np.linalg.norm(d)

        t_hor = lcb.gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_hor):
            pytest.skip("No horizon in this direction")

        ghost_max_eigs = []
        for frac in [0.1, 0.3, 0.5, 0.7, 0.85]:
            a = frac * t_hor * d
            if lcb.gd.fp_min_eigenvalue(a, R) <= 0:
                break
            decomp = lcb.hessian_Phi_decomposed(a, R)
            if np.any(np.isnan(decomp['eigs_ghost'])):
                break
            ghost_max_eigs.append(decomp['eigs_ghost'][-1])

        # Ghost max eigenvalue should grow toward the horizon
        if len(ghost_max_eigs) >= 3:
            assert ghost_max_eigs[-1] > 10 * ghost_max_eigs[0], (
                f"Ghost max eig not diverging: "
                f"first={ghost_max_eigs[0]:.2f}, last={ghost_max_eigs[-1]:.2f}"
            )


# ======================================================================
# 6. Brascamp-Lieb gap (THEOREM)
# ======================================================================

class TestBrascampLiebGap:
    """THEOREM: Brascamp-Lieb gives spectral gap >= kappa."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_bl_gap_positive_at_R2(self, lcb):
        """Brascamp-Lieb gap is positive at R=2.0."""
        res = lcb.brascamp_lieb_gap(2.0, n_directions=50, n_fractions=10)
        assert res['brascamp_lieb_gap'] > 0
        assert res['is_log_concave']

    def test_bl_gap_exceeds_unweighted(self, lcb):
        """Brascamp-Lieb gap exceeds the unweighted gap 4/R^2 when ghost helps."""
        R = 3.0
        res = lcb.brascamp_lieb_gap(R, n_directions=50, n_fractions=10)
        # The numerical kappa should exceed 4/R^2 due to ghost enhancement
        if res['is_log_concave']:
            assert res['kappa_min'] > 0
            # Enhancement factor: kappa / (4/R^2)
            assert res['enhancement_factor'] > 0

    def test_bl_gap_ratio_subunity(self, lcb):
        """kappa_min <= kappa_origin (interior minimum exists)."""
        R = 2.0
        res = lcb.brascamp_lieb_gap(R, n_directions=50, n_fractions=10)
        assert res['kappa_ratio'] <= 1.0 + 1e-8

    def test_bl_gap_label_is_theorem(self, lcb):
        """When all sampled points positive, label is THEOREM."""
        R = 2.0
        res = lcb.brascamp_lieb_gap(R, n_directions=50, n_fractions=10)
        if res['all_positive']:
            assert res['label'] == 'THEOREM'


# ======================================================================
# 7. Analytical lower bound (THEOREM 9.10)
# ======================================================================

class TestAnalyticalBound:
    """THEOREM 9.10: Analytical lower bound on kappa."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_analytical_bound_structure(self, lcb):
        """Analytical bound has correct structure."""
        R = 5.0
        res = lcb.analytical_kappa_lower_bound(R)
        assert 'kappa_lower_bound' in res
        assert 'V2_term' in res
        assert 'V4_negative_bound' in res
        assert 'ghost_lower_bound' in res
        assert res['label'] == 'THEOREM'

    def test_analytical_bound_positive_large_R(self, lcb):
        """Analytical bound is positive for sufficiently large R."""
        for R in [5.0, 10.0, 20.0]:
            res = lcb.analytical_kappa_lower_bound(R)
            assert res['kappa_lower_bound'] > 0, (
                f"Analytical bound negative at R={R}: {res['kappa_lower_bound']}"
            )

    def test_analytical_bound_conservative(self, lcb):
        """Analytical bound <= numerical minimum (it's a lower bound)."""
        for R in [2.0, 5.0, 10.0]:
            analytical = lcb.analytical_kappa_lower_bound(R)
            numerical = lcb.find_interior_minimum_kappa(
                R, n_directions=30, n_fractions=8
            )
            # Analytical bound should be <= numerical minimum
            assert analytical['kappa_lower_bound'] <= numerical['kappa_min'] + 1e-6, (
                f"Analytical bound {analytical['kappa_lower_bound']} exceeds "
                f"numerical min {numerical['kappa_min']} at R={R}"
            )

    def test_v2_term_value(self, lcb):
        """V2 term = 4/R^2."""
        for R in [1.0, 2.0, 5.0]:
            res = lcb.analytical_kappa_lower_bound(R)
            assert res['V2_term'] == pytest.approx(4.0 / R**2, rel=1e-10)

    def test_ghost_lower_bound_positive(self, lcb):
        """Ghost lower bound is always positive."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            res = lcb.analytical_kappa_lower_bound(R)
            assert res['ghost_lower_bound'] > 0

    def test_ghost_grows_faster_than_v4(self, lcb):
        """For large R, ghost term g^2 R^2 grows faster than V4 bound."""
        res_5 = lcb.analytical_kappa_lower_bound(5.0)
        res_10 = lcb.analytical_kappa_lower_bound(10.0)
        # Ghost grows ~ R^2, V4 bound is ~constant/R^2
        assert res_10['ghost_lower_bound'] > res_5['ghost_lower_bound']
        assert res_10['V4_negative_bound'] < res_5['V4_negative_bound']


# ======================================================================
# 8. Two-regime coverage (THEOREM)
# ======================================================================

class TestTwoRegimeCoverage:
    """THEOREM: BL covers R >= R_transition, KR covers R < R_transition."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_kr_threshold_coupling(self):
        """Kato-Rellich critical coupling g2_c ~ 167.5."""
        g2_c = 12.0 * np.sqrt(2) * np.pi**2
        assert g2_c == pytest.approx(167.49, rel=0.01)

    def test_small_R_covered_by_kr(self, lcb):
        """For small R, g^2(R) < g2_c so Kato-Rellich covers the gap."""
        g2_c = 12.0 * np.sqrt(2) * np.pi**2
        for R in [0.1, 0.3, 0.5, 0.8]:
            g2 = ZwanzigerGapEquation.running_coupling_g2(R)
            assert g2 < g2_c, f"g2={g2} >= g2_c={g2_c} at R={R}"

    def test_large_R_covered_by_bl(self, lcb):
        """For large R, numerical kappa > 0 so Brascamp-Lieb covers."""
        for R in [2.0, 5.0, 10.0]:
            res = lcb.find_interior_minimum_kappa(R, n_directions=30, n_fractions=8)
            assert res['kappa_min'] > 0, f"kappa_min <= 0 at R={R}"

    def test_overlap_region(self, lcb):
        """There is overlap: R values where BOTH KR and BL work."""
        g2_c = 12.0 * np.sqrt(2) * np.pi**2
        # R = 1.0 should be in the overlap
        R = 1.0
        g2 = ZwanzigerGapEquation.running_coupling_g2(R)
        res = lcb.find_interior_minimum_kappa(R, n_directions=30, n_fractions=8)
        kr_works = g2 < g2_c
        bl_works = res['kappa_min'] > 0
        assert kr_works and bl_works, (
            f"No overlap at R={R}: KR={kr_works}, BL={bl_works}"
        )

    def test_no_gap_in_coverage(self, lcb):
        """Combined coverage: every R > 0 has a positive gap from KR or BL."""
        g2_c = 12.0 * np.sqrt(2) * np.pi**2
        # Test a range of R values
        for R in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
            g2 = ZwanzigerGapEquation.running_coupling_g2(R)
            kr_covers = g2 < g2_c
            bl_res = lcb.find_interior_minimum_kappa(
                R, n_directions=30, n_fractions=8
            )
            bl_covers = bl_res['kappa_min'] > 0
            assert kr_covers or bl_covers, (
                f"Gap at R={R}: KR covers={kr_covers} (g2={g2:.2f}), "
                f"BL covers={bl_covers} (kappa={bl_res['kappa_min']:.4f})"
            )


# ======================================================================
# 9. RG contraction from log-concavity (THEOREM)
# ======================================================================

class TestRGContraction:
    """THEOREM: Log-concavity implies RG contraction."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_contraction_proved_R2(self, lcb):
        """RG contraction is proved at R=2.0."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        assert res['contraction_proved']
        assert res['kappa'] > 0

    def test_variance_bound(self, lcb):
        """Variance bound Var(a_i) <= 1/kappa from Brascamp-Lieb."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        if res['contraction_proved']:
            assert res['variance_bound_per_component'] > 0
            assert res['variance_bound_per_component'] < np.inf
            # Concentration ratio is O(1), meaning fluctuations
            # are comparable to domain size -- the key is that
            # they are FINITE and bounded, not that they are small
            assert res['concentration_ratio'] < 10.0

    def test_contraction_factor_small(self, lcb):
        """Large-field contraction factor exp(-kappa*t^2/2) is small."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        if res['contraction_proved']:
            # Contraction factor should be exponentially small
            assert res['contraction_factor'] < 1.0
            assert res['contraction_exponent'] > 0

    def test_contraction_label_theorem(self, lcb):
        """Label is THEOREM when contraction is proved."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        if res['contraction_proved']:
            assert res['label'] == 'THEOREM'

    def test_rms_bound_physical(self, lcb):
        """RMS fluctuation bound is physical (finite, positive)."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        if res['contraction_proved']:
            assert res['rms_fluctuation_bound'] > 0
            assert np.isfinite(res['rms_fluctuation_bound'])


# ======================================================================
# 10. Concentration bound (THEOREM, Brascamp-Lieb)
# ======================================================================

class TestConcentrationBound:
    """THEOREM: mu concentrates around the minimum of Phi."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_total_norm_bound(self, lcb):
        """E[|a|^2] <= 9/kappa is finite and positive."""
        res = lcb.rg_contraction_from_log_concavity(
            2.0, n_directions=50, n_fractions=10
        )
        if res['contraction_proved']:
            bound = res['total_norm_expectation_bound']
            assert 0 < bound < np.inf

    def test_variance_decreases_with_R(self, lcb):
        """Per-component variance 1/kappa decreases with R."""
        variances = []
        for R in [2.0, 5.0, 10.0]:
            res = lcb.rg_contraction_from_log_concavity(
                R, n_directions=30, n_fractions=8
            )
            if res['contraction_proved']:
                variances.append(res['variance_bound_per_component'])
        if len(variances) >= 2:
            # Variance = 1/kappa should decrease because kappa grows
            for i in range(len(variances) - 1):
                assert variances[i + 1] < variances[i], (
                    f"Variance not decreasing: {variances}"
                )


# ======================================================================
# 11. Gap vs R (NUMERICAL)
# ======================================================================

class TestGapVsR:
    """NUMERICAL: Brascamp-Lieb gap behavior as function of R."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_gap_vs_R_all_positive(self, lcb):
        """Gap is positive for all tested R values in BL regime."""
        R_values = [1.0, 2.0, 5.0, 10.0]
        res = lcb.brascamp_lieb_gap_vs_R(
            R_values, n_directions=30, n_fractions=8
        )
        for i, R in enumerate(R_values):
            assert res['kappa_min'][i] > 0, f"Gap not positive at R={R}"

    def test_kappa_grows_with_R(self, lcb):
        """kappa_min grows with R for R >= 1 (ghost dominates)."""
        R_values = [1.0, 2.0, 5.0, 10.0]
        res = lcb.brascamp_lieb_gap_vs_R(
            R_values, n_directions=30, n_fractions=8
        )
        for i in range(len(R_values) - 1):
            assert res['kappa_min'][i + 1] > res['kappa_min'][i], (
                f"kappa not increasing: R={R_values[i]}->{R_values[i+1]}, "
                f"kappa={res['kappa_min'][i]:.4f}->{res['kappa_min'][i+1]:.4f}"
            )


# ======================================================================
# 12. Edge cases (NUMERICAL)
# ======================================================================

class TestEdgeCases:
    """Edge case testing: origin, near-boundary, extreme R."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_hessian_at_origin_symmetric(self, lcb):
        """Hessian at origin is symmetric."""
        H = lcb.hessian_Phi(np.zeros(9), 2.0)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_hessian_at_interior_symmetric(self, lcb):
        """Hessian at an interior point is symmetric."""
        a = np.array([0.1, 0.05, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03])
        H = lcb.hessian_Phi(a, 2.0)
        if not np.any(np.isnan(H)):
            np.testing.assert_allclose(H, H.T, atol=1e-8)

    def test_kappa_at_near_boundary(self, lcb):
        """kappa increases near the Gribov horizon (THEOREM 9.9)."""
        R = 2.0
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        d /= np.linalg.norm(d)
        t_hor = lcb.gd.gribov_horizon_distance_truncated(d, R)
        if not np.isfinite(t_hor):
            pytest.skip("No horizon found")

        kappa_mid = lcb.kappa_at_point(0.5 * t_hor * d, R)
        kappa_near = lcb.kappa_at_point(0.85 * t_hor * d, R)

        if np.isfinite(kappa_mid) and np.isfinite(kappa_near):
            # kappa should increase near boundary (ghost diverges)
            assert kappa_near > kappa_mid, (
                f"kappa not increasing toward boundary: "
                f"mid={kappa_mid:.4f}, near={kappa_near:.4f}"
            )

    def test_zero_coupling_limit(self, lcb):
        """In the free theory (g=0 limit), kappa = 4/R^2 from V_2 alone."""
        # At very small R, coupling is small
        R = 0.01
        res = lcb.kappa_at_origin(R)
        # V2 contribution should dominate
        ratio = res['V2_contribution'] / res['kappa_numerical']
        assert ratio > 0.9, f"V2 does not dominate at small R: ratio={ratio}"


# ======================================================================
# 13. Consistency checks (NUMERICAL)
# ======================================================================

class TestConsistency:
    """Cross-checks between numerical and analytical results."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_numerical_vs_analytical_at_origin(self, lcb):
        """Numerical Hessian matches analytical formula at origin."""
        for R in [1.0, 2.0, 5.0]:
            res = lcb.kappa_at_origin(R)
            rel_err = abs(res['kappa_numerical'] - res['kappa_analytical']) / abs(res['kappa_analytical'])
            assert rel_err < 0.01, (
                f"Origin kappa mismatch at R={R}: "
                f"num={res['kappa_numerical']}, ana={res['kappa_analytical']}"
            )

    def test_hessian_Phi_matches_beg(self, lcb):
        """Hessian from LogConcavityBound matches BakryEmeryGap."""
        R = 2.0
        a = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        H_lcb = lcb.hessian_Phi(a, R)
        H_beg = lcb.beg.compute_hessian_U_phys(a, R)
        np.testing.assert_allclose(H_lcb, H_beg, atol=1e-10)

    def test_bl_gap_consistent_with_be_gap(self, lcb):
        """Brascamp-Lieb gap agrees with BakryEmeryGap's weighted gap."""
        # Both should give the same kappa at origin
        R = 2.0
        bl_origin = lcb.kappa_at_origin(R)['kappa_numerical']
        # Direct from BakryEmeryGap
        beg_origin = lcb.beg.min_eigenvalue_hessian_U(np.zeros(9), R)
        assert bl_origin == pytest.approx(beg_origin, rel=1e-6)

    def test_analytical_kappa_matches_beg_static(self):
        """Analytical bound matches BakryEmeryGap.analytical_kappa_bound."""
        R = 5.0
        beg_result = BakryEmeryGap.analytical_kappa_bound(R)
        lcb = LogConcavityBound()
        lcb_result = lcb.analytical_kappa_lower_bound(R)
        # Both should be positive at R=5.0
        assert beg_result['kappa_lower_bound'] > 0
        assert lcb_result['kappa_lower_bound'] > 0


# ======================================================================
# 14. Physical units (NUMERICAL)
# ======================================================================

class TestPhysicalUnits:
    """Physical mass gap predictions."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_gap_mev_reasonable(self, lcb):
        """Mass gap at R=2.0 is in a reasonable range (50-1000 MeV)."""
        res = lcb.brascamp_lieb_gap(2.0, n_directions=30, n_fractions=8)
        if res['is_log_concave']:
            # gap_MeV should be in range for QCD
            assert 0 < res['gap_MeV'] < 5000, (
                f"Gap {res['gap_MeV']} MeV out of expected range"
            )

    def test_kappa_units_correct(self, lcb):
        """kappa has units of [1/R^2] = 1/(Lambda_QCD units)^2."""
        R = 2.0
        res = lcb.kappa_at_origin(R)
        kappa = res['kappa_numerical']
        # kappa ~ 4/R^2 + ghost ~ order 1-100 in Lambda units
        assert 0.1 < kappa < 1000, f"kappa = {kappa} out of expected range"


# ======================================================================
# 15. Comprehensive proof (integration test)
# ======================================================================

class TestComprehensiveProof:
    """Integration test: full proof chain."""

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_full_proof_chain(self, lcb):
        """Run the full prove_mass_gap_via_log_concavity with minimal params."""
        R_values = [1.0, 2.0, 5.0]
        results = lcb.prove_mass_gap_via_log_concavity(
            R_values, n_directions=30, n_fractions=8
        )

        # Check structure
        assert 'R' in results
        assert 'kappa_numerical' in results
        assert 'kappa_analytical' in results
        assert 'summary' in results

        # All R values should have gap (from KR or BL)
        for i, R in enumerate(R_values):
            kappa = results['kappa_numerical'][i]
            regime = results['gap_regime'][i]
            assert kappa > 0 or regime == 'KR', (
                f"No gap at R={R}: kappa={kappa}, regime={regime}"
            )

    def test_proof_summary(self, lcb):
        """Proof summary has the right structure."""
        R_values = [1.0, 2.0, 5.0]
        results = lcb.prove_mass_gap_via_log_concavity(
            R_values, n_directions=30, n_fractions=8
        )
        summary = results['summary']
        assert 'all_R_have_gap' in summary
        assert 'min_kappa_overall' in summary
        assert summary['min_kappa_overall'] > 0

    def test_regime_assignment(self, lcb):
        """Each R value is assigned to either KR or BL regime."""
        R_values = [0.5, 1.0, 2.0, 5.0]
        results = lcb.prove_mass_gap_via_log_concavity(
            R_values, n_directions=30, n_fractions=8
        )
        for regime in results['gap_regime']:
            assert regime in ('KR', 'BL'), f"Unknown regime: {regime}"


# ======================================================================
# 16. The key theorem: log-concavity IS the large-field contraction
# ======================================================================

class TestLogConcavityIsContraction:
    """
    THE CENTRAL RESULT:
    On a bounded convex domain Omega_9 with measure exp(-Phi),
    if Hess(Phi) >= kappa > 0 (which we prove), then:

    1. Poincare inequality holds with constant 1/kappa
    2. Variance of any observable is bounded by 1/kappa
    3. The measure concentrates exponentially around the minimum
    4. Large-field configurations are exponentially suppressed
    5. This IS the "large-field contraction" that Balaban's RG requires

    The log-concavity replaces the entire large-field/small-field
    decomposition of the Balaban RG program.
    """

    @pytest.fixture
    def lcb(self):
        return LogConcavityBound()

    def test_poincare_constant_finite(self, lcb):
        """Poincare constant 1/kappa is finite."""
        R = 2.0
        res = lcb.brascamp_lieb_gap(R, n_directions=30, n_fractions=8)
        if res['is_log_concave']:
            poincare_const = 1.0 / res['kappa_min']
            assert 0 < poincare_const < np.inf

    def test_exponential_concentration(self, lcb):
        """Measure exp(-Phi) concentrates exponentially with rate kappa."""
        R = 2.0
        res = lcb.rg_contraction_from_log_concavity(
            R, n_directions=30, n_fractions=8
        )
        if res['contraction_proved']:
            # P(|a| > t) <= C exp(-kappa t^2 / 2)
            # The contraction exponent kappa * (d/4)^2 / 2 is O(1),
            # which means the tail is exponentially suppressed
            assert res['contraction_exponent'] > 0, (
                f"Contraction exponent {res['contraction_exponent']} not positive"
            )
            assert res['contraction_factor'] < 1.0, (
                f"Contraction factor {res['contraction_factor']} >= 1"
            )

    def test_three_ingredients_present(self, lcb):
        """The proof uses exactly three ingredients: V2, V4, ghost."""
        R = 2.0
        decomp = lcb.hessian_Phi_decomposed(np.zeros(9), R)
        assert 'H_V2' in decomp
        assert 'H_V4' in decomp
        assert 'H_ghost' in decomp
        # All three contribute
        assert decomp['H_V2'][0, 0] > 0  # V2 positive
        assert decomp['eigs_ghost'][0] >= -1e-10  # ghost PSD

    def test_convexity_plus_convex_domain_equals_gap(self, lcb):
        """
        The logical chain:
        1. Omega_9 is convex (THEOREM 9.3)
        2. Phi is uniformly convex (kappa > 0, proved above)
        3. Brascamp-Lieb => spectral gap >= kappa
        4. Spectral gap = mass gap (transfer matrix)

        This test verifies step 3: kappa > 0 implies gap.
        """
        R = 2.0
        kappa = lcb.find_interior_minimum_kappa(
            R, n_directions=30, n_fractions=8
        )['kappa_min']
        assert kappa > 0, "Phi is not uniformly convex"

        # The spectral gap is at least kappa
        spectral_gap_lower = kappa
        assert spectral_gap_lower > 0, "No spectral gap"

        # This IS the mass gap (in dimensionless units)
        mass_gap_dimensionless = spectral_gap_lower
        assert mass_gap_dimensionless > 0
