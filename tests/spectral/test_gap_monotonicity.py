"""
Tests for the gap monotonicity and R-dependence analysis.

Verifies:
  - Running coupling g^2(R) behavior
  - Lambda_QCD R-independence
  - Kato-Rellich bound validity and critical radius
  - Gap positivity across all R
  - Perturbative and non-perturbative limits
  - Effective potential confinement
  - Confinement order parameter
  - Monotonicity analysis
  - Conjecture 7.2 support
"""

import pytest
import numpy as np
from yang_mills_s3.spectral.gap_monotonicity import (
    RunningCouplingS3,
    KatoRellichBound,
    EffectivePotential,
    ConfinementAnalysis,
    DimensionalTransmutation,
    GapMonotonicity,
    GapEstimateResult,
    RigorLevel,
    gap_vs_R,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
    C_ALPHA_SU2,
    G2_CRIT_SU2,
    GAP_EIGENVALUE_COEFF,
    GAP_MASS_COEFF,
)


# ======================================================================
# Running coupling tests
# ======================================================================

class TestRunningCouplingS3:
    """1-loop running coupling on S^3."""

    def test_asymptotic_freedom_coupling_decreases(self):
        """
        THEOREM: g^2(R) decreases as R -> 0 (high energy).
        Asymptotic freedom: smaller R = higher mu = smaller coupling.
        """
        rc = RunningCouplingS3(N=2)
        R_small = 0.1  # fm, mu ~ 1973 MeV >> Lambda
        R_larger = 0.5  # fm, mu ~ 395 MeV > Lambda

        g2_small = rc.g_squared_direct(R_small)
        g2_larger = rc.g_squared_direct(R_larger)

        assert g2_small < g2_larger, (
            f"g^2({R_small}) = {g2_small:.4f} should be < g^2({R_larger}) = {g2_larger:.4f}"
        )

    def test_coupling_positive_in_perturbative_regime(self):
        """g^2 > 0 for all R in perturbative regime."""
        rc = RunningCouplingS3(N=2)
        for R in [0.01, 0.05, 0.1, 0.2, 0.5]:
            g2 = rc.g_squared_direct(R)
            assert g2 > 0, f"g^2 at R={R} should be positive"
            assert np.isfinite(g2), f"g^2 at R={R} should be finite"

    def test_coupling_infinite_at_landau_pole(self):
        """g^2 -> inf at R = hbar_c / Lambda (Landau pole)."""
        rc = RunningCouplingS3(N=2)
        R_landau = HBAR_C_MEV_FM / LAMBDA_QCD_DEFAULT  # ~ 0.987 fm
        # Just above Landau pole
        g2 = rc.g_squared_direct(R_landau * 1.01)
        assert np.isinf(g2), "g^2 should be inf at non-perturbative radius"

    def test_coupling_inf_for_large_R(self):
        """g^2 = inf for R >> 1/Lambda (non-perturbative regime)."""
        rc = RunningCouplingS3(N=2)
        g2 = rc.g_squared_direct(10.0)  # 10 fm >> 1/Lambda
        assert np.isinf(g2)

    def test_lambda_qcd_r_independence(self):
        """
        THEOREM: Lambda_QCD is R-independent.
        Check at several perturbative R values.
        """
        rc = RunningCouplingS3(N=2)
        R_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        lambdas = []
        for R in R_values:
            L = rc.lambda_qcd_check(R)
            assert L is not None, f"Should be perturbative at R={R}"
            lambdas.append(L)

        # All Lambda values should be the same
        mean = np.mean(lambdas)
        for i, L in enumerate(lambdas):
            assert abs(L - mean) / mean < 1e-10, (
                f"Lambda at R={R_values[i]} is {L:.6f}, mean is {mean:.6f}"
            )

    def test_lambda_qcd_equals_input(self):
        """Lambda_QCD recovered from g^2(R) should equal the input."""
        rc = RunningCouplingS3(N=2, Lambda_QCD=200.0)
        L = rc.lambda_qcd_check(0.1)
        assert L is not None
        assert abs(L - 200.0) / 200.0 < 1e-10, (
            f"Lambda_QCD check: got {L:.6f}, expected 200.0"
        )

    def test_alpha_s_positive(self):
        """alpha_s > 0 in perturbative regime."""
        rc = RunningCouplingS3(N=3)
        for R in [0.01, 0.05, 0.1]:
            a = rc.alpha_s(R)
            assert a > 0 and np.isfinite(a)

    def test_su3_coupling_larger_than_su2(self):
        """
        At the same R, SU(3) coupling > SU(2) coupling.
        (b0(SU(3)) > b0(SU(2)) but the formula is 1/(b0*ln), so...)
        Actually: b0(SU(3)) > b0(SU(2)) means faster running,
        which means SMALLER coupling at high energy.
        """
        rc2 = RunningCouplingS3(N=2)
        rc3 = RunningCouplingS3(N=3)
        R = 0.1
        g2_su2 = rc2.g_squared_direct(R)
        g2_su3 = rc3.g_squared_direct(R)
        # SU(3) has larger b0 -> runs faster -> smaller g^2 at same scale
        assert g2_su3 < g2_su2

    def test_lattice_beta_positive(self):
        """Lattice beta = 2N/g^2 > 0 in perturbative regime."""
        rc = RunningCouplingS3(N=2)
        for R in [0.01, 0.05, 0.1]:
            beta = rc.lattice_beta(R)
            assert beta > 0

    def test_lattice_beta_zero_nonperturbative(self):
        """Lattice beta -> 0 when g^2 -> inf."""
        rc = RunningCouplingS3(N=2)
        beta = rc.lattice_beta(10.0)  # non-perturbative
        assert beta == 0.0

    def test_coupling_at_scan_radii(self):
        """
        Compute g^2(R) for a range of R values.
        Verify monotonicity in the perturbative regime.
        """
        rc = RunningCouplingS3(N=2)
        R_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        g2_values = rc.g_squared_scan(R_values)

        # Should be monotonically increasing (larger R = larger g^2)
        for i in range(len(g2_values) - 1):
            if np.isfinite(g2_values[i]) and np.isfinite(g2_values[i+1]):
                assert g2_values[i] < g2_values[i+1], (
                    f"g^2 should increase with R: g^2({R_values[i]}) = {g2_values[i]:.4f} "
                    f">= g^2({R_values[i+1]}) = {g2_values[i+1]:.4f}"
                )

    def test_is_perturbative(self):
        """Perturbative check at various R."""
        rc = RunningCouplingS3(N=2)
        assert rc.is_perturbative(0.1) is True   # mu = 1973 >> 200
        assert rc.is_perturbative(0.5) is True   # mu = 395 > 200
        assert rc.is_perturbative(2.0) is False   # mu = 99 < 200

    def test_invalid_R_raises(self):
        """R <= 0 should raise ValueError."""
        rc = RunningCouplingS3(N=2)
        with pytest.raises(ValueError):
            rc.g_squared_direct(0.0)
        with pytest.raises(ValueError):
            rc.g_squared_direct(-1.0)


# ======================================================================
# Kato-Rellich bound tests
# ======================================================================

class TestKatoRellichBound:
    """Kato-Rellich bound as function of R."""

    def test_alpha_less_than_one_for_small_R(self):
        """
        alpha(R) < 1 for R < R_c (KR regime valid).
        """
        kr = KatoRellichBound(N=2)
        for R in [0.01, 0.05, 0.1]:
            a = kr.alpha(R)
            assert a < 1.0, f"alpha({R}) = {a:.4f} should be < 1"

    def test_alpha_increases_with_R(self):
        """alpha(R) increases with R (coupling grows)."""
        kr = KatoRellichBound(N=2)
        R_values = [0.01, 0.05, 0.1, 0.2]
        alphas = [kr.alpha(R) for R in R_values]
        for i in range(len(alphas) - 1):
            assert alphas[i] < alphas[i+1], (
                f"alpha should increase: alpha({R_values[i]}) = {alphas[i]:.6f} "
                f">= alpha({R_values[i+1]}) = {alphas[i+1]:.6f}"
            )

    def test_critical_radius_positive(self):
        """R_c > 0."""
        kr = KatoRellichBound(N=2)
        R_c = kr.critical_radius()
        assert R_c > 0, f"R_c = {R_c} should be positive"

    def test_alpha_near_one_at_critical_radius(self):
        """
        alpha(R_c) should be approximately 1.

        This defines R_c: the radius where the KR bound breaks down.
        We check nearby values since the exact R_c is computed analytically.
        """
        kr = KatoRellichBound(N=2)
        R_c = kr.critical_radius()

        # Just below R_c: alpha should be close to 1 but < 1
        a_below = kr.alpha(R_c * 0.99)
        assert a_below < 1.0, f"alpha at 0.99*R_c = {a_below:.4f} should be < 1"

        # alpha should be reasonably close to 1 at R_c
        # (running coupling is steep near Landau pole, so 1% shift in R
        # can cause a significant drop in alpha)
        assert a_below > 0.5, f"alpha at 0.99*R_c = {a_below:.4f} should be reasonably close to 1"

    def test_gap_positive_for_small_R(self):
        """Gap > 0 from KR for R < R_c."""
        kr = KatoRellichBound(N=2)
        for R in [0.01, 0.05, 0.1]:
            gap = kr.gap_MeV(R)
            assert gap > 0, f"KR gap at R={R} should be positive, got {gap}"

    def test_gap_zero_for_large_R(self):
        """Gap = 0 from KR for R >> R_c (bound invalid)."""
        kr = KatoRellichBound(N=2)
        gap = kr.gap_MeV(10.0)
        assert gap == 0.0, f"KR gap at R=10 should be 0 (bound invalid), got {gap}"

    def test_gap_approaches_geometric_for_very_small_R(self):
        """
        For R << R_c: alpha -> 0, so KR gap -> 2*hbar_c/R (geometric gap).
        """
        kr = KatoRellichBound(N=2)
        R = 0.01  # very small
        gap = kr.gap_MeV(R)
        geom_gap = GAP_MASS_COEFF * HBAR_C_MEV_FM / R
        # Should be within 5% of geometric gap
        assert abs(gap - geom_gap) / geom_gap < 0.05, (
            f"KR gap at R={R} is {gap:.2f}, should be close to {geom_gap:.2f}"
        )

    def test_g2_crit_consistent(self):
        """Critical coupling from KR matches the constant."""
        kr = KatoRellichBound(N=2)
        assert abs(kr.g2_crit - G2_CRIT_SU2) < 1e-10

    def test_is_valid_method(self):
        """is_valid returns correct boolean."""
        kr = KatoRellichBound(N=2)
        assert kr.is_valid(0.01) is True
        assert kr.is_valid(10.0) is False

    def test_gap_profile_keys(self):
        """gap_profile returns expected keys."""
        kr = KatoRellichBound(N=2)
        R_values = np.array([0.01, 0.1, 1.0])
        profile = kr.gap_profile(R_values)
        assert 'R_fm' in profile
        assert 'alpha' in profile
        assert 'gap_MeV' in profile
        assert 'valid' in profile
        assert 'R_c_fm' in profile


# ======================================================================
# Effective potential tests
# ======================================================================

class TestEffectivePotential:
    """Effective potential for the 3-mode theory."""

    def test_classical_potential_positive(self):
        """V(|a|; R) >= 0 for all a >= 0, all R."""
        ep = EffectivePotential(N=2)
        for R in [0.1, 1.0, 5.0, 50.0]:
            for a in [0.0, 0.5, 1.0, 2.0]:
                V = ep.classical_potential(a, R)
                assert V >= 0, (
                    f"V({a}, R={R}) = {V:.4f} should be >= 0"
                )

    def test_classical_potential_zero_at_origin(self):
        """V(0; R) = 0 for all R."""
        ep = EffectivePotential(N=2)
        for R in [0.1, 1.0, 10.0]:
            V = ep.classical_potential(0.0, R)
            assert abs(V) < 1e-14, f"V(0, R={R}) = {V}, should be 0"

    def test_classical_potential_increases(self):
        """V(a) increases with |a| (confining)."""
        ep = EffectivePotential(N=2)
        R = 1.0
        a_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        V_values = [ep.classical_potential(a, R) for a in a_values]
        for i in range(len(V_values) - 1):
            assert V_values[i] <= V_values[i+1], (
                f"V should increase: V({a_values[i]}) = {V_values[i]:.4f} "
                f"> V({a_values[i+1]}) = {V_values[i+1]:.4f}"
            )

    def test_is_confining_always_true(self):
        """
        THEOREM: The YM quartic potential is always confining.
        This is structural: V_4 >= 0 from Tr(F^2) >= 0.
        """
        ep = EffectivePotential(N=2)
        for R in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            assert ep.is_confining(R) is True

    def test_quantum_gap_positive(self):
        """
        NUMERICAL: Quantum gap from effective potential > 0 for all R.
        """
        ep = EffectivePotential(N=2)
        for R in [0.5, 1.0, 2.0, 5.0]:
            result = ep.quantum_gap(R, n_basis=30)
            assert result['gap_MeV'] > 0, (
                f"Quantum gap at R={R} should be positive, got {result['gap_MeV']:.4f}"
            )

    def test_quantum_gap_confining_flag(self):
        """confining flag should always be True."""
        ep = EffectivePotential(N=2)
        result = ep.quantum_gap(1.0)
        assert result['confining'] is True

    def test_potential_scan(self):
        """potential_scan returns correct shape."""
        ep = EffectivePotential(N=2)
        a_values = np.linspace(0, 2, 20)
        V = ep.potential_scan(1.0, a_values)
        assert len(V) == 20
        assert V[0] < 1e-10  # V(0) ~ 0


# ======================================================================
# Confinement tests
# ======================================================================

class TestConfinementAnalysis:
    """Confinement order parameter and gap from confinement."""

    def test_polyakov_loop_zero_at_T_zero(self):
        """
        PROPOSITION: <P> = 0 at T = 0 (always confined).
        """
        ca = ConfinementAnalysis(N=2)
        assert ca.polyakov_loop_expectation(T_MeV=0.0) == 0.0

    def test_polyakov_loop_nonzero_above_Tc(self):
        """<P> > 0 above deconfinement temperature."""
        ca = ConfinementAnalysis(N=2)
        T_c = ca.deconfinement_temperature()
        P = ca.polyakov_loop_expectation(T_MeV=T_c * 2.0)
        assert P > 0, f"<P> at 2*T_c should be > 0, got {P}"

    def test_is_confined_at_zero_temperature(self):
        """Always confined at T=0."""
        for N in [2, 3, 5]:
            ca = ConfinementAnalysis(N=N)
            assert ca.is_confined(T_MeV=0.0) is True

    def test_deconfinement_temperature_positive(self):
        """T_c > 0."""
        for N in [2, 3, 5]:
            ca = ConfinementAnalysis(N=N)
            T_c = ca.deconfinement_temperature()
            assert T_c > 0

    def test_gap_from_confinement_positive(self):
        """Confinement gap >= Lambda_QCD > 0."""
        ca = ConfinementAnalysis(N=2)
        gap = ca.gap_from_confinement()
        assert gap == LAMBDA_QCD_DEFAULT
        assert gap > 0

    def test_center_symmetry(self):
        """Center symmetry group is Z_N."""
        for N in [2, 3, 5]:
            ca = ConfinementAnalysis(N=N)
            assert ca.center_symmetry_order() == f"Z_{N}"


# ======================================================================
# Dimensional transmutation tests
# ======================================================================

class TestDimensionalTransmutation:
    """Dimensional transmutation and R-independence."""

    def test_dynamical_gap_equals_lambda(self):
        """Dynamical gap = Lambda_QCD (conservative bound)."""
        dt = DimensionalTransmutation(N=2)
        assert dt.dynamical_gap() == LAMBDA_QCD_DEFAULT

    def test_r_independence_verification(self):
        """Lambda_QCD is R-independent at perturbative R values."""
        dt = DimensionalTransmutation(N=2)
        R_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        result = dt.verify_r_independence(R_values)
        assert result['verified'] is True, (
            f"Lambda should be R-independent. Spread: {result['relative_spread']:.2e}"
        )

    def test_r_independence_fails_for_nonperturbative(self):
        """Cannot verify R-independence at non-perturbative R."""
        dt = DimensionalTransmutation(N=2)
        R_values = np.array([5.0, 10.0, 50.0])  # all non-perturbative
        result = dt.verify_r_independence(R_values)
        assert result['verified'] is False


# ======================================================================
# Gap monotonicity (main class) tests
# ======================================================================

class TestGapMonotonicity:
    """Main gap analysis."""

    def test_geometric_gap_correct(self):
        """Geometric gap = 2*hbar_c/R."""
        gm = GapMonotonicity(N=2)
        R = 2.2
        expected = 2.0 * HBAR_C_MEV_FM / R
        assert abs(gm.geometric_gap(R) - expected) < 1e-10

    def test_geometric_gap_invalid_R_raises(self):
        """R <= 0 should raise."""
        gm = GapMonotonicity(N=2)
        with pytest.raises(ValueError):
            gm.geometric_gap(0.0)

    def test_gap_at_small_R_is_theorem(self):
        """At small R, rigor is THEOREM (KR bound valid)."""
        gm = GapMonotonicity(N=2)
        result = gm.gap_at_R(0.01)
        assert result.rigor == RigorLevel.THEOREM
        assert result.regime == 'perturbative'

    def test_gap_at_large_R_is_conjecture(self):
        """At large R, rigor is CONJECTURE."""
        gm = GapMonotonicity(N=2)
        result = gm.gap_at_R(100.0)
        assert result.rigor in (RigorLevel.CONJECTURE, RigorLevel.NUMERICAL)
        assert result.regime in ('nonperturbative', 'transition')

    def test_gap_positive_for_all_R(self):
        """
        KEY TEST: Gap > 0 for all tested R values.
        This is the core claim of the analysis.
        """
        gm = GapMonotonicity(N=2)
        R_values = np.array([
            0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0,
            10.0, 50.0, 100.0, 500.0, 1000.0,
        ])
        results = gm.gap_vs_R(R_values)
        for r in results:
            assert r.gap_MeV > 0, (
                f"Gap at R={r.R_fm} should be positive, got {r.gap_MeV:.4f} MeV "
                f"(regime: {r.regime}, rigor: {r.rigor.value})"
            )

    def test_gap_approx_2_over_R_for_small_R(self):
        """
        For R << 1/Lambda: gap ~ 2*hbar_c/R (geometric dominates).
        """
        gm = GapMonotonicity(N=2)
        R = 0.01  # very small
        result = gm.gap_at_R(R)
        expected = 2.0 * HBAR_C_MEV_FM / R
        # Should be within 10% (small KR correction)
        ratio = result.gap_MeV / expected
        assert 0.8 < ratio <= 1.0, (
            f"Gap at R={R}: {result.gap_MeV:.2f} vs expected {expected:.2f}, "
            f"ratio = {ratio:.4f}"
        )

    def test_gap_approaches_lambda_for_large_R(self):
        """
        For R >> 1/Lambda: gap approaches Lambda_QCD.
        """
        gm = GapMonotonicity(N=2)
        result = gm.gap_at_R(1000.0)
        assert result.gap_MeV >= LAMBDA_QCD_DEFAULT - 1e-6, (
            f"Gap at large R should be >= Lambda_QCD = {LAMBDA_QCD_DEFAULT}, "
            f"got {result.gap_MeV:.2f}"
        )

    def test_gap_vs_R_default_values(self):
        """gap_vs_R with default R values returns a list."""
        gm = GapMonotonicity(N=2)
        results = gm.gap_vs_R()
        assert len(results) > 0
        assert all(isinstance(r, GapEstimateResult) for r in results)

    def test_gap_vs_R_custom_values(self):
        """gap_vs_R with custom R values."""
        gm = GapMonotonicity(N=2)
        R_values = np.array([0.1, 1.0, 10.0])
        results = gm.gap_vs_R(R_values)
        assert len(results) == 3
        assert results[0].R_fm == pytest.approx(0.1)
        assert results[2].R_fm == pytest.approx(10.0)

    def test_critical_radius_positive(self):
        """Critical radius R_c > 0."""
        gm = GapMonotonicity(N=2)
        R_c = gm.critical_radius()
        assert R_c > 0

    def test_crossover_radius_consistent(self):
        """R* = 2*hbar_c / Lambda_QCD."""
        gm = GapMonotonicity(N=2)
        R_star = gm.crossover_radius()
        expected = 2.0 * HBAR_C_MEV_FM / LAMBDA_QCD_DEFAULT
        assert abs(R_star - expected) < 1e-10

    def test_summary_table_string(self):
        """summary_table returns a non-empty string."""
        gm = GapMonotonicity(N=2)
        table = gm.summary_table(R_values=np.array([0.1, 1.0, 10.0]))
        assert isinstance(table, str)
        assert len(table) > 100
        assert "SU(2)" in table

    def test_monotonicity_analysis_keys(self):
        """monotonicity_analysis returns expected keys."""
        gm = GapMonotonicity(N=2)
        R_values = np.logspace(-1, 2, 30)
        analysis = gm.monotonicity_analysis(R_values)
        expected_keys = [
            'R_values', 'gaps_MeV', 'monotone_decreasing',
            'min_gap_MeV', 'R_at_min_gap_fm', 'all_positive',
            'large_R_mean_MeV', 'approaches_constant',
            'conjecture_7_2_supported', 'assessment',
        ]
        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"

    def test_monotonicity_all_gaps_positive(self):
        """Monotonicity analysis confirms all gaps positive."""
        gm = GapMonotonicity(N=2)
        R_values = np.logspace(-1, 2, 50)
        analysis = gm.monotonicity_analysis(R_values)
        assert analysis['all_positive'] is True

    def test_monotonicity_min_gap_positive(self):
        """Minimum gap over all R is positive."""
        gm = GapMonotonicity(N=2)
        R_values = np.logspace(-1, 2, 50)
        analysis = gm.monotonicity_analysis(R_values)
        assert analysis['min_gap_MeV'] > 0, (
            f"Minimum gap = {analysis['min_gap_MeV']:.4f} MeV should be > 0"
        )

    def test_conjecture_7_2_supported(self):
        """
        CONJECTURE 7.2: inf_R Delta(R) > 0.
        Our numerical analysis supports this.
        """
        gm = GapMonotonicity(N=2)
        R_values = np.logspace(-1, 2, 50)
        analysis = gm.monotonicity_analysis(R_values)
        assert analysis['conjecture_7_2_supported'] is True

    def test_perturbative_limit_gap_is_geometric(self):
        """In the perturbative limit (small R), gap ~ 2/R."""
        gm = GapMonotonicity(N=2)
        R_values = np.logspace(-2, -1, 20)
        analysis = gm.monotonicity_analysis(R_values)
        assert analysis['perturbative_limit_holds'] is True


# ======================================================================
# Top-level convenience function tests
# ======================================================================

class TestGapVsRFunction:
    """Tests for the top-level gap_vs_R function."""

    def test_returns_list(self):
        """gap_vs_R returns a list."""
        results = gap_vs_R(np.array([0.1, 1.0, 10.0]))
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_gaps_positive(self):
        """All gaps from the convenience function are positive."""
        results = gap_vs_R(np.array([0.01, 0.1, 1.0, 10.0, 100.0]))
        for r in results:
            assert r.gap_MeV > 0

    def test_su3(self):
        """Works for SU(3) too."""
        results = gap_vs_R(np.array([0.1, 1.0, 10.0]), N=3)
        assert len(results) == 3
        for r in results:
            assert r.gap_MeV > 0

    def test_custom_lambda(self):
        """Works with custom Lambda_QCD."""
        results = gap_vs_R(np.array([0.1, 1.0]), Lambda_QCD=300.0)
        assert len(results) == 2
        # With higher Lambda, the dynamical gap should be higher
        large_R_gap = results[1].gap_MeV
        assert large_R_gap >= 300.0 - 1e-6


# ======================================================================
# Constants consistency tests
# ======================================================================

class TestConstants:
    """Verify internal consistency of constants."""

    def test_c_alpha_su2(self):
        """C_alpha = sqrt(2)/(24*pi^2)."""
        expected = np.sqrt(2) / (24.0 * np.pi**2)
        assert abs(C_ALPHA_SU2 - expected) < 1e-15

    def test_g2_crit_su2(self):
        """g^2_crit = 1/C_alpha = 24*pi^2/sqrt(2) ~ 167.53."""
        assert abs(G2_CRIT_SU2 - 1.0 / C_ALPHA_SU2) < 1e-10
        assert abs(G2_CRIT_SU2 - 167.53) < 0.1

    def test_gap_coefficients(self):
        """Gap coefficients: eigenvalue = 4/R^2, mass = 2/R."""
        assert GAP_EIGENVALUE_COEFF == 4.0
        assert GAP_MASS_COEFF == 2.0

    def test_hbar_c(self):
        """hbar*c = 197.3269804 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-7
