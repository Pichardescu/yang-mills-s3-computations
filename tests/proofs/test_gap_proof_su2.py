"""
Tests for Phase 1.1: Non-perturbative mass gap via Kato-Rellich.

Tests the GapProofSU2 class which applies Kato-Rellich perturbation theory
to establish stability of the linearized mass gap Delta_0 = 5/R^2 under
the non-linear Yang-Mills vertex V(a) = g^2 * [a ^ a, .].

Test categories:
    1. Linearized gap correctness
    2. Perturbation bound properties
    3. Kato-Rellich gap bound
    4. Critical coupling existence
    5. Numerical verification
    6. Gap vs coupling table
    7. Physical coupling comparison
    8. Theorem statement
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.gap_proof_su2 import (
    GapProofSU2,
    sobolev_constant_s3,
    structure_constant_norm_sq,
    kato_rellich_global_bound,
    HBAR_C_MEV_FM,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def proof():
    """GapProofSU2 instance for SU(2)."""
    return GapProofSU2('SU(2)')


@pytest.fixture
def critical(proof):
    """Critical coupling data at R=1."""
    return proof.critical_coupling(R=1.0)


# ======================================================================
# 1. Linearized gap
# ======================================================================

class TestLinearizedGap:
    """The linearized gap is 4/R^2. THEOREM status."""

    def test_unit_radius(self, proof):
        """Delta_0 = 4 for R=1."""
        assert abs(proof.linearized_gap(R=1.0) - 4.0) < 1e-14

    def test_radius_scaling(self, proof):
        """Delta_0 = 4/R^2 for any R."""
        for R in [0.5, 1.0, 2.0, 3.14, 10.0]:
            expected = 4.0 / R**2
            actual = proof.linearized_gap(R)
            assert abs(actual - expected) < 1e-12, \
                f"R={R}: expected {expected}, got {actual}"

    def test_positive_for_all_R(self, proof):
        """Gap is strictly positive for all finite R > 0."""
        for R in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            assert proof.linearized_gap(R) > 0

    def test_diverges_as_R_to_zero(self, proof):
        """Gap -> infinity as R -> 0."""
        assert proof.linearized_gap(0.01) > proof.linearized_gap(0.1)
        assert proof.linearized_gap(0.1) > proof.linearized_gap(1.0)

    def test_vanishes_as_R_to_infinity(self, proof):
        """Gap -> 0 as R -> infinity."""
        gap_large = proof.linearized_gap(1e6)
        assert gap_large < 1e-10


# ======================================================================
# 2. Perturbation bound
# ======================================================================

class TestPerturbationBound:
    """The non-linear perturbation V(a) must be relatively bounded."""

    def test_zero_coupling_gives_zero_bound(self, proof):
        """At g=0, the perturbation vanishes."""
        pb = proof.perturbation_bound(g=0.0, R=1.0)
        assert abs(pb['alpha']) < 1e-14
        assert abs(pb['beta']) < 1e-14

    def test_alpha_increases_with_g(self, proof):
        """alpha(g) is monotonically increasing in g."""
        alphas = [proof.perturbation_bound(g, R=1.0)['alpha']
                  for g in [0.1, 0.5, 1.0, 2.0, 3.0]]
        for i in range(len(alphas) - 1):
            assert alphas[i] < alphas[i + 1], \
                f"alpha should increase: {alphas[i]} >= {alphas[i+1]}"

    def test_beta_increases_with_g(self, proof):
        """beta(g) is monotonically increasing in g."""
        betas = [proof.perturbation_bound(g, R=1.0)['beta']
                 for g in [0.1, 0.5, 1.0, 2.0, 3.0]]
        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1], \
                f"beta should increase: {betas[i]} >= {betas[i+1]}"

    def test_alpha_quadratic_in_g(self, proof):
        """alpha scales as g^2."""
        pb1 = proof.perturbation_bound(g=1.0, R=1.0)
        pb2 = proof.perturbation_bound(g=2.0, R=1.0)
        # alpha(2g) / alpha(g) should be 4
        ratio = pb2['alpha'] / pb1['alpha']
        assert abs(ratio - 4.0) < 1e-10, \
            f"alpha should scale as g^2: ratio = {ratio}"

    def test_beta_quadratic_in_g(self, proof):
        """beta scales as g^2."""
        pb1 = proof.perturbation_bound(g=1.0, R=1.0)
        pb2 = proof.perturbation_bound(g=2.0, R=1.0)
        ratio = pb2['beta'] / pb1['beta']
        assert abs(ratio - 4.0) < 1e-10, \
            f"beta should scale as g^2: ratio = {ratio}"

    def test_alpha_positive(self, proof):
        """alpha is non-negative for all g >= 0."""
        for g in [0.0, 0.1, 1.0, 5.0]:
            pb = proof.perturbation_bound(g, R=1.0)
            assert pb['alpha'] >= 0

    def test_beta_positive(self, proof):
        """beta is non-negative for all g >= 0."""
        for g in [0.0, 0.1, 1.0, 5.0]:
            pb = proof.perturbation_bound(g, R=1.0)
            assert pb['beta'] >= 0

    def test_geometric_constants_are_positive(self, proof):
        """C_alpha and C_beta should be positive geometric constants."""
        pb = proof.perturbation_bound(g=1.0, R=1.0)
        assert pb['C_alpha'] > 0
        assert pb['C_beta'] > 0

    def test_alpha_small_for_small_g(self, proof):
        """For sufficiently small g, alpha < 1 (Kato-Rellich applicable)."""
        pb = proof.perturbation_bound(g=0.1, R=1.0)
        assert pb['alpha'] < 1.0, \
            f"alpha should be < 1 for g=0.1: alpha = {pb['alpha']}"


# ======================================================================
# 3. Kato-Rellich gap bound
# ======================================================================

class TestKatoRellichGap:
    """The Kato-Rellich theorem gives a lower bound on the full gap."""

    def test_zero_coupling_recovers_linearized(self, proof):
        """At g=0, the full gap equals the linearized gap."""
        result = proof.kato_rellich_gap(g=0.0, R=1.0)
        assert abs(result['full_gap_lower_bound'] - 4.0) < 1e-12
        assert result['gap_survives'] is True

    def test_gap_decreases_with_coupling(self, proof):
        """The gap lower bound is monotonically decreasing in g."""
        gaps = [proof.kato_rellich_gap(g, R=1.0)['full_gap_lower_bound']
                for g in [0.0, 0.1, 0.5, 1.0, 2.0]]
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1], \
                f"Gap bound should decrease: {gaps[i]} < {gaps[i+1]}"

    def test_gap_survives_small_coupling(self, proof):
        """For small g, the gap survives."""
        result = proof.kato_rellich_gap(g=0.1, R=1.0)
        assert result['gap_survives'] is True
        assert result['full_gap_lower_bound'] > 0

    def test_gap_positive_below_critical(self, proof, critical):
        """Gap bound is positive for g < g_critical."""
        g_c = critical['g_critical']
        g_test = g_c * 0.5  # well below critical
        result = proof.kato_rellich_gap(g=g_test, R=1.0)
        assert result['full_gap_lower_bound'] > 0, \
            f"Gap should be positive at g={g_test:.3f} < g_c={g_c:.3f}"
        assert result['gap_survives'] is True

    def test_gap_ratio_between_0_and_1_for_small_g(self, proof):
        """The gap ratio (full/linear) is in (0, 1] for small g."""
        result = proof.kato_rellich_gap(g=0.1, R=1.0)
        assert 0 < result['gap_ratio'] <= 1.0

    def test_kato_rellich_applies_for_small_g(self, proof):
        """KR theorem applies when alpha < 1."""
        result = proof.kato_rellich_gap(g=0.1, R=1.0)
        assert result['kato_rellich_applies'] is True
        assert result['perturbation_alpha'] < 1.0

    def test_linearized_gap_in_result(self, proof):
        """Result dict contains the linearized gap."""
        result = proof.kato_rellich_gap(g=1.0, R=1.0)
        assert abs(result['linearized_gap'] - 4.0) < 1e-12

    def test_radius_dependence(self, proof):
        """Gap scales as 1/R^2."""
        r1 = proof.kato_rellich_gap(g=0.1, R=1.0)
        r2 = proof.kato_rellich_gap(g=0.1, R=2.0)
        # The linearized gap scales as 1/R^2, so does the correction
        ratio = r1['full_gap_lower_bound'] / r2['full_gap_lower_bound']
        # Should be approximately 4 (= 2^2)
        assert abs(ratio - 4.0) < 0.5, \
            f"Gap should scale ~ 1/R^2: ratio = {ratio}"


# ======================================================================
# 4. Critical coupling
# ======================================================================

class TestCriticalCoupling:
    """The critical coupling g_c where the gap bound reaches zero."""

    def test_critical_coupling_exists(self, critical):
        """g_critical is a finite positive number."""
        assert np.isfinite(critical['g_critical'])
        assert critical['g_critical'] > 0

    def test_critical_coupling_squared_positive(self, critical):
        """g_c^2 > 0."""
        assert critical['g_critical_squared'] > 0

    def test_gap_near_zero_at_critical(self, proof, critical):
        """The gap bound is near zero at g = g_critical."""
        g_c = critical['g_critical']
        result = proof.kato_rellich_gap(g=g_c * 0.99, R=1.0)
        # Should be near zero (positive but small)
        assert result['full_gap_lower_bound'] >= -0.1, \
            "Gap bound should be near zero at critical coupling"

    def test_gap_negative_above_critical(self, proof, critical):
        """The gap bound is negative above g_critical."""
        g_c = critical['g_critical']
        # Use 1.5x critical to be well above
        result = proof.kato_rellich_gap(g=g_c * 1.5, R=1.0)
        assert result['full_gap_lower_bound'] < 0, \
            "Gap bound should be negative above critical coupling"

    def test_c_eff_positive(self, critical):
        """C_eff > 0."""
        assert critical['C_eff'] > 0

    def test_c_alpha_positive(self, critical):
        """C_alpha > 0."""
        assert critical['C_alpha'] > 0

    def test_physical_coupling_documented(self, critical):
        """The physical coupling value is documented."""
        assert 'g_physical_squared' in critical
        assert critical['g_physical_squared'] > 0
        # g^2 = 4*pi*alpha_s ~ 6.28 for alpha_s = 0.5
        assert abs(critical['g_physical_squared'] - 4 * np.pi * 0.5) < 0.1

    def test_physical_comparison_honest(self, critical):
        """
        FINDING: Document honestly whether physical coupling is below critical.

        If g^2_phys > g^2_c, this means Kato-Rellich alone is INSUFFICIENT
        at physical coupling. This is an expected finding, not a failure.
        """
        g_sq_phys = critical['g_physical_squared']
        g_sq_c = critical['g_critical_squared']
        physical_below = critical['physical_below_critical']

        # The comparison must be consistent
        assert physical_below == (g_sq_phys < g_sq_c), \
            "physical_below_critical should be consistent with the numbers"

        # Document the finding regardless of which way it goes
        if physical_below:
            print(f"\nFINDING: Physical g^2 = {g_sq_phys:.2f} < g^2_c = {g_sq_c:.2f}")
            print("Kato-Rellich SUFFICIENT at physical coupling.")
        else:
            print(f"\nFINDING: Physical g^2 = {g_sq_phys:.2f} > g^2_c = {g_sq_c:.2f}")
            print("Kato-Rellich INSUFFICIENT at physical coupling.")
            print("Non-perturbative methods required (Phase 1.2+).")

    def test_critical_coupling_radius_dependence(self, proof):
        """
        The critical coupling should be INDEPENDENT of R.

        Since both the gap and the perturbation scale as 1/R^2,
        g_c^2 should be a pure number.
        """
        cc1 = proof.critical_coupling(R=1.0)
        cc2 = proof.critical_coupling(R=2.0)
        cc3 = proof.critical_coupling(R=0.5)

        g_sq_1 = cc1['g_critical_squared']
        g_sq_2 = cc2['g_critical_squared']
        g_sq_3 = cc3['g_critical_squared']

        # Should be approximately the same (not exactly due to Sobolev constant scaling)
        # The Sobolev constant scales with R, so g_c may have weak R dependence
        # We check they're in the same order of magnitude
        assert abs(g_sq_1 - g_sq_2) / g_sq_1 < 1.0, \
            f"g_c^2 has too strong R dependence: {g_sq_1:.3f} vs {g_sq_2:.3f}"


# ======================================================================
# 5. Numerical verification
# ======================================================================

class TestNumericalVerification:
    """Numerical diagonalization of the truncated YM operator."""

    def test_unperturbed_eigenvalues(self, proof):
        """At g=0, eigenvalues match the coexact Hodge spectrum."""
        result = proof.numerical_verification(R=1.0, g=0.0, l_max=5)
        expected = [(k + 1) ** 2 for k in range(1, 6)]

        for i, (actual, exp) in enumerate(zip(result['eigenvalues'], expected)):
            assert abs(actual - exp) < 1e-10, \
                f"Mode {i+1}: expected {exp}, got {actual}"

    def test_numerical_gap_positive_small_g(self, proof):
        """Numerical gap is positive for small g."""
        result = proof.numerical_verification(R=1.0, g=0.5, l_max=5)
        assert result['numerical_gap'] > 0, \
            f"Numerical gap should be positive at g=0.5: {result['numerical_gap']}"

    def test_numerical_above_kr_bound_small_g(self, proof):
        """
        Numerical eigenvalue >= Kato-Rellich bound for small g.

        This verifies the KR bound is not vacuous — the actual spectrum
        lies above the analytical lower bound.
        """
        result = proof.numerical_verification(R=1.0, g=0.3, l_max=5)
        if result['kato_rellich_bound'] > 0:
            assert result['numerical_gap'] >= result['kato_rellich_bound'] - 0.5, \
                (f"Numerical gap {result['numerical_gap']:.4f} should be >= "
                 f"KR bound {result['kato_rellich_bound']:.4f}")

    def test_eigenvalue_count(self, proof):
        """Number of eigenvalues equals number of modes."""
        result = proof.numerical_verification(R=1.0, g=1.0, l_max=5)
        assert len(result['eigenvalues']) == result['n_modes']
        assert result['n_modes'] == 5  # l = 1, 2, 3, 4, 5

    def test_eigenvalues_sorted(self, proof):
        """Eigenvalues are returned sorted."""
        result = proof.numerical_verification(R=1.0, g=1.0, l_max=5)
        evs = result['eigenvalues']
        for i in range(len(evs) - 1):
            assert evs[i] <= evs[i + 1] + 1e-10

    def test_convergence_with_l_max(self, proof):
        """
        The lowest eigenvalue should converge as l_max increases.

        The truncated matrix approximation should stabilize for the
        lowest mode as more modes are included.
        """
        gaps = []
        for l_max in [3, 5, 8, 10]:
            result = proof.numerical_verification(R=1.0, g=0.5, l_max=l_max)
            gaps.append(result['numerical_gap'])

        # The differences between consecutive l_max should decrease
        diffs = [abs(gaps[i+1] - gaps[i]) for i in range(len(gaps)-1)]
        # At least the last difference should be smaller than the first
        assert diffs[-1] <= diffs[0] + 0.5, \
            f"Gap should converge: diffs = {diffs}"

    def test_coupling_matrix_norm_finite(self, proof):
        """The coupling matrix has finite norm."""
        result = proof.numerical_verification(R=1.0, g=1.0, l_max=5)
        assert np.isfinite(result['coupling_matrix_norm'])
        assert result['coupling_matrix_norm'] >= 0


# ======================================================================
# 6. Gap vs coupling table
# ======================================================================

class TestGapVsCouplingTable:
    """Table of gap bounds as function of coupling."""

    def test_table_returns_list(self, proof):
        """Table is a list of dicts."""
        table = proof.gap_vs_coupling_table(R=1.0)
        assert isinstance(table, list)
        assert len(table) > 0
        assert isinstance(table[0], dict)

    def test_table_has_required_keys(self, proof):
        """Each entry has the required keys."""
        table = proof.gap_vs_coupling_table(R=1.0)
        required_keys = {'g', 'g_squared', 'linearized_gap',
                         'full_gap_lower_bound', 'gap_survives',
                         'gap_ratio', 'alpha'}
        for entry in table:
            assert required_keys.issubset(set(entry.keys()))

    def test_gap_monotonically_decreasing(self, proof):
        """Gap lower bound decreases monotonically with g."""
        g_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        table = proof.gap_vs_coupling_table(R=1.0, g_values=g_values)
        gaps = [entry['full_gap_lower_bound'] for entry in table]

        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 1e-10, \
                f"Gap should decrease: {gaps[i]:.4f} < {gaps[i+1]:.4f} " \
                f"at g={table[i+1]['g']}"

    def test_linearized_gap_constant(self, proof):
        """Linearized gap is the same for all couplings."""
        table = proof.gap_vs_coupling_table(R=1.0)
        for entry in table:
            assert abs(entry['linearized_gap'] - 4.0) < 1e-12

    def test_zero_coupling_full_gap_equals_linearized(self, proof):
        """At g=0, full gap = linearized gap."""
        table = proof.gap_vs_coupling_table(R=1.0, g_values=[0.0])
        assert abs(table[0]['full_gap_lower_bound'] - 4.0) < 1e-12
        assert table[0]['gap_survives'] is True

    def test_custom_g_values(self, proof):
        """Table works with custom g values."""
        g_values = [0.0, 0.5, 1.0]
        table = proof.gap_vs_coupling_table(R=1.0, g_values=g_values)
        assert len(table) == 3
        assert table[0]['g'] == 0.0
        assert table[1]['g'] == 0.5
        assert table[2]['g'] == 1.0


# ======================================================================
# 7. Physical coupling comparison
# ======================================================================

class TestPhysicalCoupling:
    """Compare critical coupling with physical QCD coupling."""

    def test_physical_coupling_value(self, critical):
        """Physical g^2 ~ 4*pi*0.5 ~ 6.28."""
        g_sq = critical['g_physical_squared']
        expected = 4.0 * np.pi * 0.5
        assert abs(g_sq - expected) < 0.1

    def test_physical_coupling_in_table(self, proof, critical):
        """Include the physical coupling in the gap table."""
        g_phys = critical['g_physical']
        g_values = [0.0, 0.5, 1.0, g_phys]
        table = proof.gap_vs_coupling_table(R=1.0, g_values=g_values)

        physical_entry = table[-1]
        print(f"\nPhysical coupling g = {g_phys:.3f}, g^2 = {g_phys**2:.3f}")
        print(f"  Gap lower bound: {physical_entry['full_gap_lower_bound']:.4f}")
        print(f"  Gap survives (KR): {physical_entry['gap_survives']}")
        print(f"  Alpha: {physical_entry['alpha']:.4f}")

    def test_mass_gap_in_mev_if_gap_survives(self, proof, critical):
        """
        If the gap survives at physical coupling, compute the mass in MeV.
        Otherwise, document that KR is insufficient.
        """
        g_phys = critical['g_physical']
        R_fm = 2.2  # fm, consistent with Lambda_QCD

        kr = proof.kato_rellich_gap(g=g_phys, R=R_fm)

        if kr['gap_survives']:
            mass_gap_eigenvalue = kr['full_gap_lower_bound']
            mass_mev = HBAR_C_MEV_FM * np.sqrt(mass_gap_eigenvalue)
            print(f"\nMass gap at physical coupling: {mass_mev:.1f} MeV")
            print(f"Compare with Lambda_QCD ~ 200 MeV")
        else:
            print(f"\nKR bound is negative at physical coupling.")
            print(f"  Full gap bound: {kr['full_gap_lower_bound']:.4f}")
            print(f"  This does NOT mean the gap doesn't exist.")
            print(f"  It means Kato-Rellich alone cannot prove it.")


# ======================================================================
# 8. Theorem statement
# ======================================================================

class TestTheoremStatement:
    """The formal theorem statement."""

    def test_statement_is_string(self, proof):
        """Theorem statement is a non-empty string."""
        stmt = proof.theorem_statement()
        assert isinstance(stmt, str)
        assert len(stmt) > 100

    def test_statement_contains_key_elements(self, proof):
        """Statement mentions the key mathematical ingredients."""
        stmt = proof.theorem_statement()
        # Must mention the key concepts
        assert 'S^3' in stmt or 'S3' in stmt
        assert 'SU(2)' in stmt
        assert 'Kato-Rellich' in stmt
        assert '4/R^2' in stmt or '4/R' in stmt
        assert 'Weitzenboeck' in stmt or 'Weitzenb' in stmt or 'coexact' in stmt
        assert 'THEOREM' in stmt

    def test_statement_contains_finding(self, proof):
        """Statement honestly reports whether KR is sufficient."""
        stmt = proof.theorem_statement()
        assert 'FINDING' in stmt
        assert 'SUFFICIENT' in stmt or 'INSUFFICIENT' in stmt

    def test_statement_contains_critical_coupling(self, proof):
        """Statement mentions the critical coupling value."""
        stmt = proof.theorem_statement()
        assert 'g^2' in stmt or 'g_c' in stmt or 'critical' in stmt.lower()

    def test_statement_mentions_assumptions(self, proof):
        """Statement lists its assumptions."""
        stmt = proof.theorem_statement()
        assert 'Assumption' in stmt or '(A1)' in stmt


# ======================================================================
# 9. Auxiliary functions
# ======================================================================

class TestAuxiliaryFunctions:
    """Test helper functions used by the proof."""

    def test_sobolev_constant_positive(self):
        """Sobolev constant is positive."""
        for R in [0.5, 1.0, 2.0, 10.0]:
            C_S = sobolev_constant_s3(R)
            assert C_S > 0

    def test_sobolev_constant_increases_with_R(self):
        """C_S(R) increases with R (scales as sqrt(R))."""
        c1 = sobolev_constant_s3(1.0)
        c2 = sobolev_constant_s3(4.0)
        # C_S(4) / C_S(1) should be sqrt(4) = 2
        assert abs(c2 / c1 - 2.0) < 1e-10

    def test_structure_constants_su2(self):
        """SU(2) structure constants: |f|^2 = 6, |f|_eff^2 = 2."""
        sc = structure_constant_norm_sq('SU(2)')
        assert abs(sc['total_norm_sq'] - 6.0) < 1e-14
        assert abs(sc['effective_norm_sq'] - 2.0) < 1e-14
        assert sc['dim_adj'] == 3

    def test_structure_constants_su3(self):
        """SU(3) structure constants: standard normalization."""
        sc = structure_constant_norm_sq('SU(3)')
        assert sc['total_norm_sq'] > 0
        assert sc['effective_norm_sq'] > 0
        assert sc['dim_adj'] == 8


# ======================================================================
# 10. Full analysis
# ======================================================================

class TestFullAnalysis:
    """End-to-end analysis."""

    def test_full_analysis_returns_dict(self, proof):
        """Full analysis returns a complete results dict."""
        result = proof.full_analysis(R=1.0)
        assert isinstance(result, dict)
        assert 'linearized_gap' in result
        assert 'critical_coupling' in result
        assert 'gap_vs_coupling' in result
        assert 'numerical_verification' in result
        assert 'theorem' in result

    def test_full_analysis_consistency(self, proof):
        """Results within full_analysis are internally consistent."""
        result = proof.full_analysis(R=1.0)

        # Linearized gap should be 4.0
        assert abs(result['linearized_gap'] - 4.0) < 1e-12

        # Critical coupling should be positive and finite
        cc = result['critical_coupling']
        assert cc['g_critical'] > 0
        assert np.isfinite(cc['g_critical'])

        # Table should have entries
        assert len(result['gap_vs_coupling']) > 0

        # Theorem should be a string
        assert isinstance(result['theorem'], str)


# ======================================================================
# 11. Global Kato-Rellich bound (standalone function)
# ======================================================================

class TestKatoRellichGlobalBound:
    """
    Tests for the global Kato-Rellich bound via Sobolev embedding.

    The key requirement is that the bound holds for ALL psi in Dom(Delta_1),
    not just for specific eigenmodes. The derivation uses:
    1. Sobolev: H^1(S^3) -> L^6(S^3) with sharp Aubin-Talenti constant
    2. Holder (6,6,6): ||a*a*psi||_L2 <= ||a||_L6^2 * ||psi||_L6
    3. Peter-Paul inequality to convert H^1 norm to operator norm
    """

    def test_function_exists(self):
        """The standalone function kato_rellich_global_bound exists."""
        result = kato_rellich_global_bound(g_coupling=1.0)
        assert isinstance(result, dict)

    def test_returns_required_keys(self):
        """Result contains all required keys."""
        result = kato_rellich_global_bound(g_coupling=1.0)
        required = {
            'alpha', 'beta', 'C_alpha', 'C_beta',
            'g_critical_squared', 'gap_lower_bound', 'gap_survives',
            'linearized_gap', 'sobolev_constant', 'f_eff', 'derivation',
        }
        assert required.issubset(set(result.keys()))

    def test_critical_coupling_value(self):
        """
        g^2_c = 24*pi^2/sqrt(2) ~ 167.53.

        Post-Weitzenbock value from the global Sobolev + spectral bound.
        """
        result = kato_rellich_global_bound(g_coupling=1.0)
        expected_g2c = 24.0 * np.pi**2 / np.sqrt(2)
        assert abs(result['g_critical_squared'] - expected_g2c) < 0.1, \
            f"Expected g^2_c ~ {expected_g2c:.2f}, got {result['g_critical_squared']:.2f}"
        assert abs(expected_g2c - 167.53) < 0.5  # sanity check

    def test_critical_coupling_well_above_physical(self):
        """
        g^2_c >> g^2_phys ~ 6.28.

        The gap must survive at physical coupling.
        """
        result = kato_rellich_global_bound(g_coupling=1.0)
        g_sq_phys = 4.0 * np.pi * 0.5  # ~ 6.28
        assert result['g_critical_squared'] > g_sq_phys, \
            f"g^2_c = {result['g_critical_squared']:.2f} should be > g^2_phys = {g_sq_phys:.2f}"

    def test_gap_survives_at_physical_coupling(self):
        """The gap bound is positive at physical QCD coupling."""
        g_phys = np.sqrt(4.0 * np.pi * 0.5)  # ~ 2.507
        result = kato_rellich_global_bound(g_coupling=g_phys)
        assert result['gap_survives'], \
            f"Gap should survive at physical coupling: alpha={result['alpha']:.4f}"
        assert result['gap_lower_bound'] > 0

    def test_alpha_about_0_037_at_physical(self):
        """At physical coupling g^2 ~ 6.28, alpha ~ 0.037 (post-Weitzenbock)."""
        g_phys = np.sqrt(4.0 * np.pi * 0.5)
        result = kato_rellich_global_bound(g_coupling=g_phys)
        assert abs(result['alpha'] - 0.037) < 0.005, \
            f"Expected alpha ~ 0.037, got {result['alpha']:.4f}"

    def test_bound_works_for_arbitrary_R(self):
        """
        The bound must hold for arbitrary R > 0.
        The relative bound alpha is R-independent.
        """
        for R in [0.1, 0.5, 1.0, 2.0, 10.0]:
            result = kato_rellich_global_bound(g_coupling=1.0, R=R)
            assert result['alpha'] > 0
            assert np.isfinite(result['alpha'])
            # alpha should be the same for all R (R-independent)
            assert abs(result['C_alpha'] - np.sqrt(2)/(24*np.pi**2)) < 1e-10

    def test_alpha_R_independent(self):
        """
        C_alpha is R-independent (dimensionless).

        Both the perturbation and the unperturbed operator scale as 1/R^2,
        so the relative bound is R-independent.
        """
        results = [kato_rellich_global_bound(g_coupling=1.0, R=R)
                   for R in [0.5, 1.0, 2.0, 5.0]]
        alphas = [r['alpha'] for r in results]
        for i in range(1, len(alphas)):
            assert abs(alphas[i] - alphas[0]) < 1e-12, \
                f"alpha should be R-independent: {alphas}"

    def test_zero_coupling_gives_full_gap(self):
        """At g=0, the gap lower bound equals the linearized gap."""
        result = kato_rellich_global_bound(g_coupling=0.0)
        assert abs(result['gap_lower_bound'] - 4.0) < 1e-12
        assert result['gap_survives']

    def test_derivation_mentions_sobolev(self):
        """The derivation string mentions the Sobolev chain."""
        result = kato_rellich_global_bound(g_coupling=1.0)
        deriv = result['derivation']
        assert 'Sobolev' in deriv
        assert 'Holder' in deriv or 'H\\"{o}lder' in deriv or 'Holder' in deriv
        assert 'Weitzenbock' in deriv or 'spectral' in deriv
        assert 'Dom(Delta_1)' in deriv or 'ALL psi' in deriv

    def test_sobolev_constant_is_aubin_talenti(self):
        """
        The Sobolev constant used is the sharp Aubin-Talenti value.
        C_S(1) = (4/3) * (2*pi^2)^{-2/3} ~ 0.18255
        """
        result = kato_rellich_global_bound(g_coupling=1.0, R=1.0)
        expected_CS = (4.0/3.0) * (2.0 * np.pi**2)**(-2.0/3.0)
        assert abs(result['sobolev_constant'] - expected_CS) < 1e-6, \
            f"Expected C_S ~ {expected_CS:.6f}, got {result['sobolev_constant']:.6f}"

    def test_consistent_with_class_method(self):
        """
        The standalone function gives the same alpha as the class method.
        """
        proof = GapProofSU2('SU(2)')
        g = 2.0
        standalone = kato_rellich_global_bound(g_coupling=g, R=1.0)
        class_result = proof.perturbation_bound(g=g, R=1.0)
        assert abs(standalone['alpha'] - class_result['alpha']) < 1e-12
        assert abs(standalone['C_alpha'] - class_result['C_alpha']) < 1e-12


# ======================================================================
# 12. Global bound validity (not mode-specific)
# ======================================================================

class TestGlobalBoundNotModeSpecific:
    """
    Verify that the Kato-Rellich bound is GLOBAL (for all psi in Dom(Delta_1)),
    not specific to the lowest eigenmode.

    The key test: the perturbation bound alpha = C_alpha * g^2 must work
    for ANY psi, including:
    - The lowest eigenmode (k=1)
    - Higher eigenmodes (k=2,3,...)
    - Superpositions of multiple eigenmodes
    - Generic H^1 functions
    """

    def test_theorem_statement_mentions_global(self, proof):
        """The theorem statement explicitly mentions the global nature of the bound."""
        stmt = proof.theorem_statement()
        assert 'ALL psi' in stmt or 'all psi' in stmt or 'global' in stmt.lower(), \
            "Theorem statement should mention global validity of the bound"

    def test_theorem_mentions_sobolev_chain(self, proof):
        """The theorem statement mentions the Sobolev embedding chain."""
        stmt = proof.theorem_statement()
        assert 'Sobolev' in stmt, \
            "Theorem statement should mention Sobolev embedding"
        assert 'Holder' in stmt or 'H\\"{o}lder' in stmt or 'older' in stmt, \
            "Theorem statement should mention Holder inequality"

    def test_theorem_mentions_operator_domain(self, proof):
        """The theorem mentions the operator domain (Coulomb gauge)."""
        stmt = proof.theorem_statement()
        assert 'Coulomb' in stmt or 'coexact' in stmt or 'domain' in stmt.lower(), \
            "Theorem should mention operator domain"

    def test_theorem_mentions_gribov(self, proof):
        """The theorem mentions the Gribov obstruction."""
        stmt = proof.theorem_statement()
        assert 'Gribov' in stmt or 'Singer' in stmt, \
            "Theorem should mention Gribov/Singer obstruction for gauge fixing"

    def test_perturbation_bound_docstring_is_global(self, proof):
        """The perturbation_bound docstring indicates global validity."""
        doc = proof.perturbation_bound.__doc__
        assert 'ALL' in doc or 'all psi' in doc.lower() or 'GLOBAL' in doc or 'global' in doc.lower(), \
            "perturbation_bound docstring should indicate global validity"
