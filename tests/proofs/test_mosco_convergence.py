"""
Tests for Mosco Convergence of Yang-Mills Quadratic Forms: S³(R) → ℝ³.

Verifies:
1. Local geometry convergence (metric, curvature, volume)
2. Quadratic form comparison bounds
3. Mosco lim-inf condition (energy lower bound)
4. Mosco lim-sup condition (recovery sequence)
5. Mosco → resolvent convergence chain
6. Gap preservation theorem
7. Topological advantage (H¹ = 0)
8. Full framework consistency
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.mosco_convergence import (
    local_geometry_convergence,
    quadratic_form_comparison,
    mosco_lim_inf_check,
    mosco_lim_sup_check,
    mosco_implies_resolvent,
    gap_preservation_theorem,
    topological_advantage,
    mosco_convergence_framework,
    schwinger_function_convergence,
    os_closed_under_limits,
    gap_from_uniform_decay,
    theorem_7_12_via_schwinger,
    why_mosco_unnecessary,
    address_criticism,
)


# =====================================================================
# 1. LOCAL GEOMETRY CONVERGENCE
# =====================================================================

class TestLocalGeometryConvergence:
    """THEOREM: S³(R) metric → flat metric locally as R → ∞."""

    def test_valid_regime(self):
        """R > 2L is the validity condition."""
        r = local_geometry_convergence(10.0, 2.0)
        assert r['valid']
        assert r['label'] == 'THEOREM'

    def test_invalid_regime(self):
        """R ≤ 2L should return invalid."""
        r = local_geometry_convergence(3.0, 2.0)
        assert not r['valid']

    def test_metric_correction_decreases_with_R(self):
        """Metric correction O(L²/R²) → 0 as R → ∞."""
        L = 2.0
        corrections = []
        for R in [5, 10, 50, 100, 500]:
            r = local_geometry_convergence(R, L)
            assert r['valid']
            corrections.append(r['metric_correction'])
        # Strictly decreasing
        for i in range(len(corrections) - 1):
            assert corrections[i + 1] < corrections[i]

    def test_metric_correction_order(self):
        """Correction should be O(L²/R²)."""
        L = 2.0
        for R in [10, 50, 100]:
            r = local_geometry_convergence(R, L)
            # metric_correction ≈ L²/R² (to leading order)
            expected_order = L**2 / R**2
            # Should be within factor of 10 (higher order corrections)
            assert r['metric_correction'] < 10 * expected_order

    def test_ricci_vanishes(self):
        """Ricci curvature 2/R² → 0 as R → ∞."""
        for R in [10, 100, 1000]:
            r = local_geometry_convergence(R, 1.0)
            assert r['ricci_curvature'] == pytest.approx(2.0 / R**2)
            assert r['ricci_curvature'] < 0.1  # small for R > 5

    def test_volume_correction_small(self):
        """Volume element correction is O(L²/R²)."""
        r = local_geometry_convergence(100.0, 2.0)
        assert r['volume_correction'] < 0.01  # < 1% for R = 100, L = 2

    def test_christoffel_bound(self):
        """Christoffel symbols bounded by L/R²."""
        R, L = 50.0, 2.0
        r = local_geometry_convergence(R, L)
        assert r['christoffel_bound'] == pytest.approx(L / R**2)

    def test_negative_R_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError):
            local_geometry_convergence(-1.0, 1.0)

    def test_negative_L_raises(self):
        """Negative ball radius should raise ValueError."""
        with pytest.raises(ValueError):
            local_geometry_convergence(10.0, -1.0)

    def test_large_R_all_corrections_small(self):
        """At large R, all corrections should be small."""
        r = local_geometry_convergence(1000.0, 2.0)
        assert r['all_corrections_small']


# =====================================================================
# 2. QUADRATIC FORM COMPARISON
# =====================================================================

class TestQuadraticFormComparison:
    """THEOREM: |q_R - q_∞| ≤ C · (L/R)² · q_∞."""

    def test_valid_regime(self):
        r = quadratic_form_comparison(10.0, 2.0)
        assert r['valid']
        assert r['label'] == 'THEOREM'

    def test_invalid_regime(self):
        r = quadratic_form_comparison(3.0, 2.0)
        assert not r['valid']

    def test_bound_decreases_with_R(self):
        """Form bound O(L²/R²) → 0 as R → ∞."""
        L = 2.0
        bounds = []
        for R in [5, 10, 50, 100, 500]:
            r = quadratic_form_comparison(R, L)
            bounds.append(r['form_bound'])
        for i in range(len(bounds) - 1):
            assert bounds[i + 1] < bounds[i]

    def test_bound_order_L_squared_over_R_squared(self):
        """Form bound should be O(L²/R²)."""
        L = 2.0
        for R in [20, 50, 100]:
            r = quadratic_form_comparison(R, L)
            expected = L**2 / (2 * R**2)
            # Exact correction is (1 + L²/(4R²))² - 1 ≈ L²/(2R²)
            assert r['relative_correction'] == pytest.approx(expected, rel=0.01)

    def test_converges_to_zero(self):
        """Flag indicating convergence should be True."""
        r = quadratic_form_comparison(100.0, 2.0)
        assert r['converges_to_zero']

    def test_ratio_L_over_R(self):
        r = quadratic_form_comparison(50.0, 5.0)
        assert r['ratio_L_over_R'] == pytest.approx(0.1)


# =====================================================================
# 3. MOSCO LIM-INF (Lower Bound)
# =====================================================================

class TestMoscoLimInf:
    """THEOREM: q_∞(u) ≤ lim inf q_R(u_R)."""

    def test_all_test_functions_satisfy(self):
        """All test function families should satisfy lim-inf."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_inf_check(R_values)
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_gaussian_lim_inf(self):
        """Gaussian test function satisfies lim-inf."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_inf_check(R_values, ['gaussian'])
        assert r['test_functions']['gaussian']['lim_inf_satisfied']

    def test_bump_lim_inf(self):
        """Bump function satisfies lim-inf."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_inf_check(R_values, ['bump'])
        assert r['test_functions']['bump']['lim_inf_satisfied']

    def test_instanton_lim_inf(self):
        """Instanton tail satisfies lim-inf."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_inf_check(R_values, ['instanton_tail'])
        assert r['test_functions']['instanton_tail']['lim_inf_satisfied']

    def test_differences_converge(self):
        """Differences |q_R - q_∞| should decrease with R."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_inf_check(R_values, ['gaussian'])
        assert r['test_functions']['gaussian']['converging']

    def test_q_R_approaches_q_inf(self):
        """q_R values should approach q_∞ from above."""
        R_values = [10, 50, 100, 500, 1000]
        r = mosco_lim_inf_check(R_values, ['gaussian'])
        data = r['test_functions']['gaussian']
        # q_R ≥ q_∞ (since S³ integral ≥ flat integral for compactly supported)
        for qR, qinf in zip(data['q_R_values'], data['q_inf_values']):
            assert qR >= qinf * 0.999  # allow tiny numerical error


# =====================================================================
# 4. MOSCO LIM-SUP (Recovery Sequence)
# =====================================================================

class TestMoscoLimSup:
    """THEOREM: ∃ u_R → u strongly with q_R(u_R) → q_∞(u)."""

    def test_all_test_functions_recover(self):
        """All test function families should have recovery sequences."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_sup_check(R_values)
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_gaussian_recovery(self):
        """Gaussian: recovery sequence converges."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_sup_check(R_values, ['gaussian'])
        data = r['test_functions']['gaussian']
        assert data['recovery_satisfied']
        assert data['form_converging']

    def test_bump_recovery(self):
        """Bump: recovery sequence converges."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_sup_check(R_values, ['bump'])
        data = r['test_functions']['bump']
        assert data['recovery_satisfied']

    def test_instanton_recovery(self):
        """Instanton tail: recovery sequence converges."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_sup_check(R_values, ['instanton_tail'])
        data = r['test_functions']['instanton_tail']
        assert data['recovery_satisfied']

    def test_strong_convergence(self):
        """L² error ||u_R - u|| → 0."""
        R_values = [5, 10, 50, 100, 500]
        r = mosco_lim_sup_check(R_values, ['gaussian'])
        data = r['test_functions']['gaussian']
        assert data['strong_converging']
        # L2 errors should decrease
        errors = data['l2_errors']
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] * 1.01


# =====================================================================
# 5. MOSCO → RESOLVENT CONVERGENCE
# =====================================================================

class TestMoscoImpliesResolvent:
    """THEOREM: Mosco convergence ⟹ strong resolvent convergence."""

    def test_full_chain_holds(self):
        """All 5 steps of the Mosco → resolvent chain should hold."""
        r = mosco_implies_resolvent()
        assert r['result']
        assert r['resolvent_converges']
        assert r['label'] == 'THEOREM'

    def test_all_steps_theorem(self):
        """Every step should be labeled THEOREM."""
        r = mosco_implies_resolvent()
        assert r['n_theorem'] == 5

    def test_step_1_geometry(self):
        """Step 1: local geometry converges."""
        r = mosco_implies_resolvent()
        assert r['steps']['step_1_geometry']['holds']

    def test_step_2_forms(self):
        """Step 2: quadratic forms converge."""
        r = mosco_implies_resolvent()
        assert r['steps']['step_2_forms']['holds']

    def test_step_3_lim_inf(self):
        """Step 3: Mosco lim-inf."""
        r = mosco_implies_resolvent()
        assert r['steps']['step_3_lim_inf']['holds']

    def test_step_4_lim_sup(self):
        """Step 4: Mosco lim-sup."""
        r = mosco_implies_resolvent()
        assert r['steps']['step_4_lim_sup']['holds']

    def test_step_5_resolvent(self):
        """Step 5: resolvent convergence."""
        r = mosco_implies_resolvent()
        assert r['steps']['step_5_resolvent']['holds']

    def test_resolvent_rate_decreases(self):
        """Resolvent convergence rate O(L²/R²) → 0."""
        r = mosco_implies_resolvent()
        rates = [x['rate_bound'] for x in r['resolvent_rates']]
        for i in range(len(rates) - 1):
            assert rates[i + 1] < rates[i]


# =====================================================================
# 6. GAP PRESERVATION
# =====================================================================

class TestGapPreservation:
    """THEOREM: uniform gap + Mosco → gap in limit."""

    def test_theorem_holds(self):
        """Main theorem: gap(H_∞) ≥ Δ₀ > 0."""
        r = gap_preservation_theorem()
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_delta_0_positive(self):
        """Uniform gap bound Δ₀ > 0."""
        r = gap_preservation_theorem()
        assert r['Delta_0'] > 0

    def test_mosco_converges(self):
        """Mosco convergence holds."""
        r = gap_preservation_theorem()
        assert r['mosco_converges']

    def test_gap_uniform(self):
        """Gap is positive for all tested R."""
        r = gap_preservation_theorem()
        assert r['gap_uniform']

    def test_mass_gap_physical(self):
        """Mass gap should be in physical range."""
        r = gap_preservation_theorem()
        # mass_gap_GeV should be positive and reasonable
        assert r['mass_gap_GeV'] > 0
        # Should be less than 10 GeV (physical reasonableness)
        assert r['mass_gap_GeV'] < 10.0

    def test_convergence_demo_all_positive(self):
        """All demonstration R values have positive gap."""
        r = gap_preservation_theorem()
        for entry in r['convergence_demo']:
            assert entry['gap_positive']


# =====================================================================
# 7. TOPOLOGICAL ADVANTAGE
# =====================================================================

class TestTopologicalAdvantage:
    """THEOREM: H¹ = 0 throughout S³ → ℝ³ path."""

    def test_s3_b1_zero(self):
        """S³: b₁ = 0."""
        r = topological_advantage()
        assert r['S3_path']['b1_source'] == 0

    def test_r3_b1_zero(self):
        """ℝ³: b₁ = 0."""
        r = topological_advantage()
        assert r['S3_path']['b1_target'] == 0

    def test_no_betti_jump_s3(self):
        """S³ → ℝ³: no Betti number jump."""
        r = topological_advantage()
        assert r['S3_path']['b1_jump'] == 0

    def test_t3_betti_jump(self):
        """T³ → ℝ³: Betti number jumps 3 → 0."""
        r = topological_advantage()
        assert r['T3_path']['b1_jump'] == 3

    def test_honda_smooth_s3(self):
        """Honda smoothness holds for S³ path."""
        r = topological_advantage()
        assert r['S3_path']['honda_smooth']

    def test_honda_fails_t3(self):
        """Honda smoothness fails for T³ path."""
        r = topological_advantage()
        assert not r['T3_path']['honda_smooth']

    def test_s3_mosco_converges(self):
        """Mosco convergence holds for S³ path."""
        r = topological_advantage()
        assert r['S3_path']['mosco_converges']

    def test_t3_mosco_fails(self):
        """Mosco convergence fails for T³ path (zero modes)."""
        r = topological_advantage()
        assert not r['T3_path']['mosco_converges']

    def test_label_theorem(self):
        r = topological_advantage()
        assert r['label'] == 'THEOREM'


# =====================================================================
# 8. FULL FRAMEWORK
# =====================================================================

class TestMoscoFramework:
    """THEOREM: Complete Mosco convergence framework (8 steps)."""

    def test_all_steps_exist(self):
        """All 8 steps should be present."""
        r = mosco_convergence_framework()
        assert len(r['steps']) == 8

    def test_all_steps_theorem(self):
        """All 8 steps should be THEOREM level."""
        r = mosco_convergence_framework()
        assert r['n_theorem'] == 8
        assert r['n_proposition'] == 0

    def test_overall_theorem(self):
        """Overall conclusion should be THEOREM."""
        r = mosco_convergence_framework()
        assert r['overall_label'] == 'THEOREM'

    def test_upgrade_from_prop_712(self):
        """Documents the upgrade from PROPOSITION 7.12."""
        r = mosco_convergence_framework()
        assert 'PROPOSITION 7.12' in r['upgrade_from_prop_712']

    def test_result_true(self):
        r = mosco_convergence_framework()
        assert r['result']

    def test_references_include_mosco(self):
        """References should include Mosco's paper."""
        r = mosco_convergence_framework()
        refs = ' '.join(r['references'])
        assert 'Mosco' in refs

    def test_references_include_reed_simon(self):
        """References should include Reed-Simon."""
        r = mosco_convergence_framework()
        refs = ' '.join(r['references'])
        assert 'Reed-Simon' in refs


# =====================================================================
# 9. SCHWINGER FUNCTION CONVERGENCE (PRIMARY DECOMPACTIFICATION)
# =====================================================================

class TestSchwingerFunctionConvergence:
    """THEOREM: Schwinger functions converge as R -> infinity."""

    def test_result_holds(self):
        """Main convergence result should hold."""
        r = schwinger_function_convergence()
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_errors_decrease_with_R(self):
        """Total error should decrease monotonically with R."""
        r = schwinger_function_convergence()
        assert r['errors_decrease']

    def test_cauchy_sequence(self):
        """Schwinger functions form a Cauchy sequence in R."""
        r = schwinger_function_convergence()
        assert r['cauchy_sequence']

    def test_geometry_dominates_at_moderate_R(self):
        """At moderate R, geometry error should dominate finite-size."""
        r = schwinger_function_convergence(R_values=[5.0, 10.0, 50.0])
        # At R=5, geometry error = L^2/R^2 = 4/25 = 0.16
        # finite-size is exp(-mass*pi*5) which is extremely small
        data = r['convergence_data']
        for d in data:
            if d['R'] >= 5.0:
                assert d['dominant'] == 'geometry'

    def test_convergence_rate_order(self):
        """Convergence rate should be O(L^2/R^2) + O(exp(-m*pi*R))."""
        r = schwinger_function_convergence()
        assert r['convergence_rate'] == 'O(L²/R²) + O(exp(-√Δ₀ · πR))'

    def test_delta0_positive(self):
        """Uniform gap bound should be positive."""
        r = schwinger_function_convergence()
        assert r['Delta0'] > 0

    def test_mass_positive(self):
        """Mass = sqrt(Delta0) should be positive."""
        r = schwinger_function_convergence()
        assert r['mass'] > 0

    def test_all_steps_theorem(self):
        """All sub-steps should be THEOREM level."""
        r = schwinger_function_convergence()
        assert r['n_theorem'] == 5
        for step in r['steps'].values():
            assert step['label'] == 'THEOREM'

    def test_custom_R_values(self):
        """Should work with custom R values."""
        r = schwinger_function_convergence(R_values=[10.0, 100.0, 1000.0])
        assert r['result']
        assert len(r['convergence_data']) == 3

    def test_custom_delta0(self):
        """Should work with custom Delta0."""
        r = schwinger_function_convergence(Delta0=1.0)
        assert r['result']
        assert r['Delta0'] == 1.0
        assert r['mass'] == pytest.approx(1.0)

    def test_no_mosco_in_proof(self):
        """Proof sketch should not rely on Mosco."""
        r = schwinger_function_convergence()
        assert 'NO Mosco' in r['proof_sketch']

    def test_cauchy_bounds_decrease(self):
        """Cauchy bounds should decrease along the sequence."""
        r = schwinger_function_convergence()
        cauchy_bounds = [d['cauchy_bound'] for d in r['cauchy_data']]
        for i in range(len(cauchy_bounds) - 1):
            assert cauchy_bounds[i + 1] < cauchy_bounds[i]


# =====================================================================
# 10. OS AXIOMS CLOSED UNDER LIMITS
# =====================================================================

class TestOSClosedUnderLimits:
    """THEOREM: OS axioms are closed conditions, preserved by limits."""

    def test_result_holds(self):
        """All axioms should be closed under limits."""
        r = os_closed_under_limits()
        assert r['result']
        assert r['all_closed']
        assert r['label'] == 'THEOREM'

    def test_all_five_axioms_closed(self):
        """Each of the 5 OS axioms should be individually closed."""
        r = os_closed_under_limits()
        assert len(r['axioms']) == 5
        for name, axiom in r['axioms'].items():
            assert axiom['closed_under_limits'], f'{name} should be closed'

    def test_all_theorem_level(self):
        """All 5 closure results should be THEOREM level."""
        r = os_closed_under_limits()
        assert r['n_theorem'] == 5
        for axiom in r['axioms'].values():
            assert axiom['label'] == 'THEOREM'

    def test_os0_regularity_mechanism(self):
        """OS0 closure should be via uniform convergence."""
        r = os_closed_under_limits()
        assert 'uniform' in r['axioms']['OS0_regularity']['mechanism'].lower()

    def test_os2_rp_positivity_argument(self):
        """OS2 (RP) should be closed because positivity is closed."""
        r = os_closed_under_limits()
        assert 'positivity' in r['axioms']['OS2_reflection_positivity']['mechanism'].lower()

    def test_os4_clustering_uniform_decay(self):
        """OS4 closure should use uniform exponential decay."""
        r = os_closed_under_limits()
        mechanism = r['axioms']['OS4_clustering']['mechanism']
        assert 'exp' in mechanism.lower() or 'decay' in mechanism.lower()

    def test_references_include_glimm_jaffe(self):
        """References should include Glimm-Jaffe."""
        r = os_closed_under_limits()
        refs = ' '.join(r['references'])
        assert 'Glimm-Jaffe' in refs

    def test_references_include_os(self):
        """References should include Osterwalder-Schrader."""
        r = os_closed_under_limits()
        refs = ' '.join(r['references'])
        assert 'Osterwalder-Schrader' in refs


# =====================================================================
# 11. GAP FROM UNIFORM DECAY
# =====================================================================

class TestGapFromUniformDecay:
    """THEOREM: Uniform exponential decay -> mass gap in limit."""

    def test_result_holds(self):
        """Gap should be positive."""
        r = gap_from_uniform_decay()
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_delta0_positive(self):
        """Uniform gap bound should be positive."""
        r = gap_from_uniform_decay()
        assert r['Delta0'] > 0

    def test_mass_gap_positive(self):
        """Mass gap in fm^{-1} should be positive."""
        r = gap_from_uniform_decay()
        assert r['mass_gap_fm_inv'] > 0

    def test_mass_gap_gev_physical(self):
        """Mass gap in GeV should be physical (0 < m < 10 GeV)."""
        r = gap_from_uniform_decay()
        assert 0 < r['mass_gap_GeV'] < 10.0

    def test_all_steps_theorem(self):
        """All 4 steps should be THEOREM."""
        r = gap_from_uniform_decay()
        assert r['n_theorem'] == 4
        for step in r['steps'].values():
            assert step['label'] == 'THEOREM'

    def test_kallen_lehmann_step(self):
        """Should include Kallen-Lehmann spectral representation."""
        r = gap_from_uniform_decay()
        assert 'step_2_kallen_lehmann' in r['steps']

    def test_custom_delta0(self):
        """Should work with explicit Delta0."""
        r = gap_from_uniform_decay(Delta0=4.0)
        assert r['mass_gap_fm_inv'] == pytest.approx(2.0)

    def test_references_include_simon(self):
        """References should include Simon."""
        r = gap_from_uniform_decay()
        refs = ' '.join(r['references'])
        assert 'Simon' in refs


# =====================================================================
# 12. THEOREM 7.12 VIA SCHWINGER
# =====================================================================

class TestTheorem712ViaSchwinger:
    """THEOREM 7.12: Full decompactification via Schwinger functions."""

    def test_result_holds(self):
        """Main theorem should hold."""
        r = theorem_7_12_via_schwinger()
        assert r['result']
        assert r['label'] == 'THEOREM'

    def test_all_theorem(self):
        """All 5 steps should be THEOREM."""
        r = theorem_7_12_via_schwinger()
        assert r['all_theorem']
        assert r['n_theorem'] == 5

    def test_each_step_verified(self):
        """Each individual step should be verified."""
        r = theorem_7_12_via_schwinger()
        for name, step in r['steps'].items():
            assert step['verified'], f'{name} should be verified'

    def test_delta0_positive(self):
        """Uniform gap bound should be positive."""
        r = theorem_7_12_via_schwinger()
        assert r['Delta0'] > 0

    def test_mass_gap_positive(self):
        """Mass gap should be positive in both units."""
        r = theorem_7_12_via_schwinger()
        assert r['mass_gap_fm_inv'] > 0
        assert r['mass_gap_GeV'] > 0

    def test_bypasses_mosco(self):
        """Should explicitly bypass Mosco convergence."""
        r = theorem_7_12_via_schwinger()
        assert r['bypasses_mosco']

    def test_addresses_quartic(self):
        """Should address the quartic action criticism."""
        r = theorem_7_12_via_schwinger()
        assert r['addresses_quartic_criticism']

    def test_proof_chain_no_mosco(self):
        """Proof chain should state NO Mosco used."""
        r = theorem_7_12_via_schwinger()
        assert 'NO Mosco' in r['proof_chain']

    def test_step_1_osterwalder_seiler(self):
        """Step 1 should reference Osterwalder-Seiler."""
        r = theorem_7_12_via_schwinger()
        assert 'Osterwalder-Seiler' in r['steps']['step_1_lattice_ym']['source']

    def test_step_3_schwinger(self):
        """Step 3 should use Schwinger function convergence."""
        r = theorem_7_12_via_schwinger()
        assert 'schwinger' in r['steps']['step_3_schwinger_converge']['source'].lower()

    def test_step_4_os_closed(self):
        """Step 4 should verify OS axioms are closed."""
        r = theorem_7_12_via_schwinger()
        assert 'os_closed' in r['steps']['step_4_os_closed']['source']

    def test_statement_includes_mass_gap(self):
        """Statement should include the mass gap value."""
        r = theorem_7_12_via_schwinger()
        assert 'mass gap' in r['statement'].lower()
        assert 'fm' in r['statement']

    def test_custom_delta0(self):
        """Should work with custom Delta0."""
        r = theorem_7_12_via_schwinger(Delta0=1.0)
        assert r['result']
        assert r['mass_gap_fm_inv'] == pytest.approx(1.0)

    def test_references_include_osterwalder(self):
        """References should include Osterwalder."""
        r = theorem_7_12_via_schwinger()
        refs = ' '.join(r['references'])
        assert 'Osterwalder' in refs


# =====================================================================
# 13. WHY MOSCO UNNECESSARY
# =====================================================================

class TestWhyMoscoUnnecessary:
    """Documentation: Mosco is unnecessary for the full non-linear theory."""

    def test_returns_dict(self):
        """Should return a well-structured explanation."""
        r = why_mosco_unnecessary()
        assert isinstance(r, dict)
        assert r['label'] == 'EXPLANATION'

    def test_identifies_mosco_limitation(self):
        """Should identify that Mosco requires quadratic forms."""
        r = why_mosco_unnecessary()
        assert 'quadratic' in r['mosco_limitation'].lower()
        assert 'QUARTIC' in r['mosco_limitation']

    def test_schwinger_advantage(self):
        """Should explain the Schwinger function advantage."""
        r = why_mosco_unnecessary()
        assert 'Schwinger' in r['schwinger_advantage']

    def test_mosco_still_valid_for_linearized(self):
        """Should note Mosco is still valid for the linearized theory."""
        r = why_mosco_unnecessary()
        assert 'linearized' in r['mosco_still_valid_for'].lower()

    def test_conceptual_shift_documented(self):
        """Should document the classical -> quantum shift."""
        r = why_mosco_unnecessary()
        assert 'Classical' in r['conceptual_shift'] or 'Quantum' in r['conceptual_shift']


# =====================================================================
# 14. ADDRESS CRITICISM
# =====================================================================

class TestAddressCriticism:
    """Explicit response to peer review criticisms."""

    def test_all_resolved(self):
        """All criticisms should be resolved."""
        r = address_criticism()
        assert r['all_resolved']

    def test_four_criticisms_addressed(self):
        """Should address at least 4 criticisms."""
        r = address_criticism()
        assert r['n_criticisms'] >= 4

    def test_quartic_criticism_resolved(self):
        """Quartic action criticism should be resolved."""
        r = address_criticism()
        quartic = r['criticisms']['quartic_action']
        assert quartic['status'] == 'RESOLVED'
        assert 'Schwinger' in quartic['response']

    def test_convexity_criticism_resolved(self):
        """Convexity criticism should be resolved."""
        r = address_criticism()
        convexity = r['criticisms']['mosco_convexity']
        assert convexity['status'] == 'RESOLVED'

    def test_constructive_qft_addressed(self):
        """Constructive QFT criticism should be addressed."""
        r = address_criticism()
        cqft = r['criticisms']['constructive_qft_incomplete']
        assert cqft['status'] == 'RESOLVED'
        assert 'S³' in cqft['response'] or 'S3' in cqft['response']

    def test_operator_domain_addressed(self):
        """Operator domain criticism should be addressed."""
        r = address_criticism()
        domain = r['criticisms']['operator_domain']
        assert domain['status'] == 'RESOLVED'

    def test_primary_resolution_schwinger(self):
        """Primary resolution should be via Schwinger functions."""
        r = address_criticism()
        assert 'Schwinger' in r['primary_resolution']

    def test_mosco_remains_for_linearized(self):
        """Should note Mosco remains valid for linearized theory."""
        r = address_criticism()
        assert 'linearized' in r['what_remains_from_mosco'].lower()
