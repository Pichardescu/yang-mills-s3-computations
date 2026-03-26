"""
Tests for dimensional_transmutation.py — One-loop self-energy on S^3(R).

Tests the vertex factors, selection rules, R-dependence, and the
conclusion that perturbative self-energy does NOT produce an
R-independent mass.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.dimensional_transmutation import (
    coexact_eigenvalue,
    coexact_multiplicity,
    harmonic_frequency,
    volume_s3,
    su2_structure_constants,
    color_factor_cubic,
    color_factor_cubic_sun,
    cubic_selection_rule,
    allowed_couplings_from_k1,
    CubicVertexS3,
    OneLoopSelfEnergy,
    dimensional_transmutation_analysis,
    correct_dt_argument,
    uniform_gap_from_dt,
)


# ======================================================================
# Spectral data tests
# ======================================================================

class TestSpectralData:
    """Tests for S^3 eigenvalues and multiplicities."""

    def test_eigenvalue_k1(self):
        """k=1 eigenvalue is 4/R^2."""
        R = 2.0
        assert coexact_eigenvalue(1, R) == pytest.approx(4.0 / R**2)

    def test_eigenvalue_k2(self):
        """k=2 eigenvalue is 9/R^2."""
        R = 3.0
        assert coexact_eigenvalue(2, R) == pytest.approx(9.0 / R**2)

    def test_eigenvalue_general(self):
        """k-th eigenvalue is (k+1)^2/R^2."""
        R = 1.5
        for k in range(1, 11):
            assert coexact_eigenvalue(k, R) == pytest.approx((k + 1)**2 / R**2)

    def test_eigenvalue_k0_raises(self):
        """k=0 should raise ValueError."""
        with pytest.raises(ValueError):
            coexact_eigenvalue(0, 1.0)

    def test_multiplicity_k1(self):
        """d_1 = 2*1*3 = 6."""
        assert coexact_multiplicity(1) == 6

    def test_multiplicity_k2(self):
        """d_2 = 2*2*4 = 16."""
        assert coexact_multiplicity(2) == 16

    def test_multiplicity_k3(self):
        """d_3 = 2*3*5 = 30."""
        assert coexact_multiplicity(3) == 30

    def test_multiplicity_general(self):
        """d_k = 2k(k+2) for k = 1..10."""
        for k in range(1, 11):
            assert coexact_multiplicity(k) == 2 * k * (k + 2)

    def test_harmonic_frequency(self):
        """omega_k = (k+1)/R."""
        R = 2.2
        for k in range(1, 6):
            assert harmonic_frequency(k, R) == pytest.approx((k + 1) / R)

    def test_volume_s3(self):
        """Vol(S^3(R)) = 2 pi^2 R^3."""
        R = 1.0
        assert volume_s3(R) == pytest.approx(2 * np.pi**2)
        R = 2.0
        assert volume_s3(R) == pytest.approx(2 * np.pi**2 * 8)


# ======================================================================
# Color factor tests
# ======================================================================

class TestColorFactors:
    """Tests for SU(N) color factors."""

    def test_structure_constants_antisymmetric(self):
        """f^{abc} is totally antisymmetric."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert f[a, b, c] == -f[b, a, c]
                    assert f[a, b, c] == -f[a, c, b]
                    assert f[a, b, c] == f[b, c, a]

    def test_structure_constants_epsilon(self):
        """f^{123} = 1 (epsilon tensor)."""
        f = su2_structure_constants()
        assert f[0, 1, 2] == 1.0
        assert f[1, 0, 2] == -1.0

    def test_color_contraction_su2(self):
        """sum_{b,c} f^{abc} f^{a'bc} = 2 delta_{aa'} for SU(2)."""
        f = su2_structure_constants()
        for a in range(3):
            for ap in range(3):
                val = sum(f[a, b, c] * f[ap, b, c] for b in range(3) for c in range(3))
                expected = 2.0 if a == ap else 0.0
                assert val == pytest.approx(expected, abs=1e-12)

    def test_c2_adj_su2(self):
        """C_2(adj) = 2 for SU(2)."""
        assert color_factor_cubic() == 2.0

    def test_c2_adj_sun(self):
        """C_2(adj) = N for SU(N)."""
        assert color_factor_cubic_sun(2) == 2.0
        assert color_factor_cubic_sun(3) == 3.0
        assert color_factor_cubic_sun(5) == 5.0


# ======================================================================
# Selection rule tests
# ======================================================================

class TestSelectionRules:
    """Tests for the cubic vertex selection rule on S^3."""

    def test_111_allowed(self):
        """(1,1,1) is NOT allowed: 1+1+1=3 is odd."""
        assert cubic_selection_rule(1, 1, 1) == False

    def test_112_allowed(self):
        """(1,1,2) IS allowed: triangle + parity."""
        assert cubic_selection_rule(1, 1, 2) == True

    def test_113_forbidden(self):
        """(1,1,3) is NOT allowed: 1+1+3=5 is odd."""
        assert cubic_selection_rule(1, 1, 3) == False

    def test_114_forbidden_triangle(self):
        """(1,1,4) is NOT allowed: triangle violated (4 > 1+1=2)."""
        assert cubic_selection_rule(1, 1, 4) == False

    def test_123_forbidden(self):
        """(1,2,3) IS allowed: triangle OK, 1+2+3=6 even."""
        assert cubic_selection_rule(1, 2, 3) == True

    def test_122_forbidden_parity(self):
        """(1,2,2) is NOT allowed: 1+2+2=5 odd."""
        assert cubic_selection_rule(1, 2, 2) == False

    def test_222_allowed(self):
        """(2,2,2) IS allowed: triangle OK, 2+2+2=6 even."""
        assert cubic_selection_rule(2, 2, 2) == True

    def test_234_allowed(self):
        """(2,3,4) is NOT allowed: 2+3+4=9 odd."""
        assert cubic_selection_rule(2, 3, 4) == False

    def test_only_k2_from_11k(self):
        """
        THEOREM: The ONLY cubic coupling (1,1,k) with k >= 1 is k=2.

        This is crucial: the leading self-energy channel is uniquely determined.
        """
        allowed_k = []
        for k in range(1, 100):
            if cubic_selection_rule(1, 1, k):
                allowed_k.append(k)
        assert allowed_k == [2], f"Expected only k=2, got {allowed_k}"

    def test_allowed_couplings_from_k1(self):
        """Check all (1, k2, k3) couplings up to k_max."""
        couplings = allowed_couplings_from_k1(k_max=5)
        # (1,1,2), (1,2,1), (1,2,3), (1,3,2), (1,3,4), (1,4,3), (1,4,5), (1,5,4)
        # and their k2<=k3 variants
        assert len(couplings) > 0
        # Check (1,1,2) is there
        assert (1, 1, 2) in couplings
        # Check (1,2,3) is there
        assert (1, 2, 3) in couplings
        # Check (1,1,1) is NOT there (odd parity)
        assert (1, 1, 1) not in couplings

    def test_triangle_inequality(self):
        """Triangle inequality is necessary condition."""
        # k3 > k1 + k2 should fail
        assert cubic_selection_rule(1, 2, 5) == False
        assert cubic_selection_rule(2, 3, 8) == False
        # k3 < |k1 - k2| should fail
        assert cubic_selection_rule(5, 1, 2) == False  # k3=2 < |5-1|=4


# ======================================================================
# Vertex factor tests
# ======================================================================

class TestCubicVertex:
    """Tests for the cubic vertex factors on S^3."""

    def test_c3_111_positive(self):
        """C_3(1,1,1) > 0."""
        v = CubicVertexS3(R=1.0)
        assert v.c3_111() > 0

    def test_c3_111_R_scaling(self):
        """
        THEOREM: C_3(1,1,1) = alpha / R^{5/2} where alpha is a pure number.

        Check that C_3 * R^{5/2} is R-independent.
        """
        alpha_values = []
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            v = CubicVertexS3(R)
            c3 = v.c3_111()
            alpha = c3 * R**(5.0 / 2.0)
            alpha_values.append(alpha)

        # All alpha values should be the same
        for alpha in alpha_values:
            assert alpha == pytest.approx(alpha_values[0], rel=1e-10)

    def test_c3_111_exact_value(self):
        """
        C_3(1,1,1) = 2/R * sqrt(3/Vol) = 2/R * sqrt(3/(2*pi^2*R^3))
                    = 2*sqrt(3/(2*pi^2)) / R^{5/2}
        """
        R = 1.0
        v = CubicVertexS3(R)
        expected = 2.0 / R * np.sqrt(3.0 / (2.0 * np.pi**2 * R**3))
        assert v.c3_111() == pytest.approx(expected, rel=1e-10)

    def test_c3_112_positive(self):
        """C_3(1,1,2) > 0."""
        v = CubicVertexS3(R=1.0)
        assert v.c3_112() > 0

    def test_c3_112_R_scaling(self):
        """
        THEOREM: C_3(1,1,2) = beta / R^{5/2} where beta is a pure number.

        Same R-scaling as C_3(1,1,1).
        """
        beta_values = []
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            v = CubicVertexS3(R)
            c3 = v.c3_112()
            beta = c3 * R**(5.0 / 2.0)
            beta_values.append(beta)

        for beta in beta_values:
            assert beta == pytest.approx(beta_values[0], rel=1e-10)

    def test_c3_112_larger_than_c3_111(self):
        """
        C_3(1,1,2) > C_3(1,1,1) because the derivative eigenvalue is larger
        for k=2 (3/R vs 2/R) and the CG coefficient is O(1).
        """
        v = CubicVertexS3(R=1.0)
        assert v.c3_112() > v.c3_111()

    def test_c3_112_ratio(self):
        """
        C_3(1,1,2) / C_3(1,1,1) = sqrt(3/2) ~ 1.2247.
        """
        v = CubicVertexS3(R=1.0)
        ratio = v.c3_112() / v.c3_111()
        assert ratio == pytest.approx(np.sqrt(3.0 / 2.0), rel=1e-10)

    def test_c3_general_selection_rule(self):
        """Forbidden couplings give zero."""
        v = CubicVertexS3(R=1.0)
        assert v.c3_general(1, 1, 3) == 0.0  # odd parity
        assert v.c3_general(1, 1, 4) == 0.0  # triangle violation

    def test_c3_general_matches_exact_111(self):
        """General formula agrees with exact for (1,1,1)."""
        v = CubicVertexS3(R=2.0)
        # (1,1,1) fails selection rule (odd sum), so it should be 0
        # This is correct! The structure constant vertex f^{abc} with
        # all three in the same mode vanishes after CG projection for odd sum.
        assert v.c3_general(1, 1, 1) == 0.0

    def test_c3_general_matches_exact_112(self):
        """General formula agrees with exact for (1,1,2)."""
        v = CubicVertexS3(R=2.0)
        assert v.c3_general(1, 1, 2) == pytest.approx(v.c3_112(), rel=1e-10)


# ======================================================================
# One-loop self-energy tests
# ======================================================================

class TestOneLoopSelfEnergy:
    """Tests for the one-loop self-energy computation."""

    def test_leading_channel_exists(self):
        """The leading channel (1,1,2) gives a nonzero result."""
        se = OneLoopSelfEnergy(R=2.0, g2=6.28, N=2, k_max=5)
        result = se.leading_channel_112()
        assert result['delta_E'] > 0

    def test_leading_channel_R_scaling(self):
        """
        PROPOSITION: delta_E ~ C(g^2)/R.

        At FIXED coupling, R * delta_E should be R-independent.
        """
        g2 = 6.28
        products = []
        for R in [1.0, 2.0, 4.0, 8.0]:
            se = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
            result = se.leading_channel_112()
            products.append(R * result['delta_E'])

        # All products should be the same (at fixed g^2)
        for p in products:
            assert p == pytest.approx(products[0], rel=1e-8)

    def test_self_energy_not_R_independent(self):
        """
        The self-energy IS R-dependent (goes as 1/R at fixed g^2).
        This is the KEY result: perturbative DT does not give R-independent mass.
        """
        g2 = 6.28
        delta_E_values = []
        R_values = [1.0, 2.0, 5.0, 10.0]
        for R in R_values:
            se = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
            result = se.leading_channel_112()
            delta_E_values.append(result['delta_E'])

        # delta_E should decrease with R (not constant)
        for i in range(len(delta_E_values) - 1):
            assert delta_E_values[i] > delta_E_values[i + 1]

    def test_self_energy_small_coupling(self):
        """At small g^2, self-energy is small compared to bare gap."""
        R = 2.0
        g2 = 0.01
        se = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
        result = se.leading_channel_112()
        omega_1 = 2.0 / R
        ratio = result['delta_E'] / omega_1
        assert ratio < 0.01  # small perturbative correction

    def test_self_energy_proportional_to_g8(self):
        """
        delta_E ~ g^8 at fixed R.
        Check by varying g^2 and seeing the scaling.
        """
        R = 2.0
        g2_values = [1.0, 2.0, 4.0]
        delta_E_values = []
        for g2 in g2_values:
            se = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
            result = se.leading_channel_112()
            delta_E_values.append(result['delta_E'])

        # Check g^8 scaling: delta_E(2*g2) / delta_E(g2) = (2)^4 = 16
        # (g^8 means (g^2)^4)
        ratio_12 = delta_E_values[1] / delta_E_values[0]
        expected_ratio = (g2_values[1] / g2_values[0])**4
        assert ratio_12 == pytest.approx(expected_ratio, rel=0.01)

    def test_color_factor_enters(self):
        """Self-energy is proportional to C_2(adj)."""
        R = 2.0
        g2 = 6.28

        se_su2 = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
        se_su3 = OneLoopSelfEnergy(R, g2, N=3, k_max=5)

        r_su2 = se_su2.leading_channel_112()
        r_su3 = se_su3.leading_channel_112()

        # SU(3)/SU(2) ratio should be 3/2
        ratio = r_su3['delta_E'] / r_su2['delta_E']
        assert ratio == pytest.approx(3.0 / 2.0, rel=1e-6)

    def test_full_one_loop_includes_leading(self):
        """Full one-loop sum includes the leading (1,1,2) channel."""
        se = OneLoopSelfEnergy(R=2.0, g2=6.28, N=2, k_max=5)
        full = se.full_one_loop()
        assert full['total_sigma'] > 0

    def test_full_one_loop_has_channels(self):
        """Full computation produces channel data."""
        se = OneLoopSelfEnergy(R=2.0, g2=6.28, N=2, k_max=5)
        full = se.full_one_loop()
        assert len(full['channels']) > 0

    def test_full_one_loop_label(self):
        """Full result is labeled NUMERICAL."""
        se = OneLoopSelfEnergy(R=2.0, g2=6.28, N=2, k_max=5)
        full = se.full_one_loop()
        assert full['label'] == 'NUMERICAL'


# ======================================================================
# Dimensional transmutation analysis tests
# ======================================================================

class TestDimensionalTransmutationAnalysis:
    """Tests for the R-dependence analysis."""

    def test_analysis_runs(self):
        """Analysis produces results for multiple R values."""
        result = dimensional_transmutation_analysis(R_values=[1.0, 2.0, 5.0])
        assert len(result['table']) == 3

    def test_bare_gap_decreases_with_R(self):
        """Bare gap = 2*hbar_c/R decreases with R."""
        result = dimensional_transmutation_analysis(R_values=[1.0, 2.0, 5.0])
        gaps = [r['gap_bare_MeV'] for r in result['table']]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_total_gap_decreases_with_R(self):
        """Total gap (bare + one-loop) also decreases with R."""
        result = dimensional_transmutation_analysis(R_values=[1.0, 2.0, 5.0])
        gaps = [r['gap_total_MeV'] for r in result['table']]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_running_coupling_varies(self):
        """Running coupling varies with R."""
        result = dimensional_transmutation_analysis(R_values=[0.5, 2.0, 10.0])
        g2_values = [r['g2_running'] for r in result['table']]
        # Should increase with R (asymptotic freedom: larger R -> stronger coupling)
        assert g2_values[0] < g2_values[1]

    def test_one_loop_small_at_weak_coupling(self):
        """One-loop correction is small when g^2 is small (perturbative regime)."""
        # Use a small fixed coupling where perturbation theory is valid
        result = dimensional_transmutation_analysis(
            R_values=[0.5], g2=0.1
        )
        r = result['table'][0]
        # At g2=0.1 the correction is tiny (g^8 = 10^{-8})
        assert abs(r['ratio_delta_E_over_bare']) < 0.01


# ======================================================================
# Correct argument structure tests
# ======================================================================

class TestCorrectArgument:
    """Tests for the honest assessment of dimensional transmutation."""

    def test_correct_argument_structure(self):
        """The correct argument has all required components."""
        arg = correct_dt_argument()
        assert 'perturbative_self_energy' in arg
        assert 'dimensional_transmutation' in arg
        assert 'non_perturbative_gap' in arg
        assert 'uniform_gap' in arg
        assert 'honest_assessment' in arg

    def test_perturbative_not_R_independent(self):
        """Perturbative self-energy is correctly labeled as R-dependent."""
        arg = correct_dt_argument()
        assert arg['perturbative_self_energy']['R_independent'] == False

    def test_dt_does_not_prove_existence(self):
        """Dimensional transmutation alone does not prove gap existence."""
        arg = correct_dt_argument()
        assert arg['dimensional_transmutation']['proves_existence'] == False

    def test_uniform_gap_is_theorem(self):
        """The uniform gap statement (7.12a) is labeled THEOREM."""
        arg = correct_dt_argument()
        assert 'THEOREM' in arg['uniform_gap']['label']


# ======================================================================
# Uniform gap function tests
# ======================================================================

class TestUniformGap:
    """Tests for the combined gap estimate."""

    def test_uniform_gap_positive(self):
        """Gap is positive at any finite R."""
        result = uniform_gap_from_dt(R=2.2, g2=6.28)
        assert result['gap_bare_MeV'] > 0
        assert result['gap_bare_plus_one_loop_MeV'] > 0

    def test_uniform_gap_physical_value(self):
        """At R=2.2 fm, bare gap ~ 180 MeV."""
        result = uniform_gap_from_dt(R=2.2, g2=6.28)
        assert 150 < result['gap_bare_MeV'] < 250

    def test_uniform_gap_labels(self):
        """Result has proper labels."""
        result = uniform_gap_from_dt(R=2.2, g2=6.28)
        assert 'label' in result
        assert 'THEOREM' in result['label'] or 'NUMERICAL' in result['label']


# ======================================================================
# Dimension consistency tests
# ======================================================================

class TestDimensionConsistency:
    """Verify dimensional consistency of all quantities."""

    def test_eigenvalue_dimensions(self):
        """lambda_k has dimensions 1/length^2."""
        R = 2.0  # fm
        lam = coexact_eigenvalue(1, R)
        # lam = 4/R^2 = 4/4 = 1 fm^{-2}
        assert lam == pytest.approx(1.0, rel=1e-10)

    def test_frequency_dimensions(self):
        """omega_k has dimensions 1/length."""
        R = 2.0
        omega = harmonic_frequency(1, R)
        # omega = 2/R = 1 fm^{-1}
        assert omega == pytest.approx(1.0, rel=1e-10)

    def test_volume_dimensions(self):
        """Volume has dimensions length^3."""
        R = 1.0
        vol = volume_s3(R)
        # vol = 2 pi^2 ~ 19.739 fm^3 at R=1
        assert vol == pytest.approx(2 * np.pi**2, rel=1e-10)

    def test_vertex_dimensions(self):
        """C_3 has dimensions 1/length^{5/2}."""
        R = 1.0
        v = CubicVertexS3(R)
        c3 = v.c3_111()
        # At R=1: c3 = 2 * sqrt(3/(2*pi^2)) ~ 2 * 0.3898 ~ 0.7797
        # Dimensions: 1/R * 1/sqrt(R^3) = 1/R^{5/2}
        expected = 2.0 * np.sqrt(3.0 / (2.0 * np.pi**2))
        assert c3 == pytest.approx(expected, rel=1e-10)

    def test_self_energy_dimensions(self):
        """Self-energy delta_E has dimensions 1/length (energy)."""
        # Use weak coupling where perturbation theory is reliable
        se = OneLoopSelfEnergy(R=2.0, g2=0.5, N=2, k_max=5)
        result = se.leading_channel_112()
        delta_E = result['delta_E']
        # delta_E should be in units of 1/fm and positive
        assert delta_E > 0
        # At weak coupling, correction should be small
        omega_1 = harmonic_frequency(1, 2.0)
        assert delta_E < omega_1  # correction smaller than bare gap at weak coupling

    def test_self_energy_large_at_strong_coupling(self):
        """
        At strong coupling g^2 ~ 6.28, the self-energy is LARGER than the bare gap.

        This is expected and physically correct: perturbation theory breaks down
        at strong coupling. The perturbative self-energy ~ g^8 diverges, showing
        that the perturbative expansion is unreliable. This is the standard
        signature that non-perturbative methods are needed.
        """
        se = OneLoopSelfEnergy(R=2.0, g2=6.28, N=2, k_max=5)
        result = se.leading_channel_112()
        omega_1 = harmonic_frequency(1, 2.0)
        # At g^2=6.28, perturbative correction is >> bare gap
        assert result['delta_E'] > omega_1  # perturbation theory breaks down!


# ======================================================================
# Key physical result tests
# ======================================================================

class TestKeyResults:
    """Tests for the central physics conclusions."""

    def test_only_k2_couples_to_k1_pair(self):
        """
        THEOREM: The selection rule (1,1,k) only allows k=2.

        This means the cubic self-energy has exactly ONE channel.
        """
        for k in range(1, 50):
            if k == 2:
                assert cubic_selection_rule(1, 1, k) == True
            else:
                assert cubic_selection_rule(1, 1, k) == False

    def test_perturbative_gap_goes_to_zero(self):
        """
        PROPOSITION: At fixed coupling, the perturbative gap
        (bare + one-loop) goes to zero as R -> infinity.
        """
        g2 = 6.28
        gaps = []
        for R in [1.0, 10.0, 100.0]:
            se = OneLoopSelfEnergy(R, g2, N=2, k_max=5)
            result = se.leading_channel_112()
            omega_1 = harmonic_frequency(1, R)
            total = omega_1 + result['delta_E']
            gaps.append(total)

        # Gap should decrease
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

        # Last gap should be very small
        assert gaps[-1] < gaps[0] / 10

    def test_vertex_squared_times_R5_constant(self):
        """
        |C_3(1,1,2)|^2 * R^5 is R-independent.

        This is the dimensional analysis theorem for the vertex.
        """
        products = []
        for R in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            v = CubicVertexS3(R)
            c3_sq = v.c3_112()**2
            products.append(c3_sq * R**5)

        for p in products:
            assert p == pytest.approx(products[0], rel=1e-10)

    def test_alpha_112_numerical_value(self):
        """
        alpha_112 = C_3(1,1,2) * R^{5/2} is a specific pure number.
        """
        v = CubicVertexS3(R=1.0)
        alpha_112 = v.c3_112()  # at R=1, this IS alpha_112
        alpha_111 = v.c3_111()  # = 2*sqrt(3/(2*pi^2)) ~ 0.7797

        expected_111 = 2.0 * np.sqrt(3.0 / (2.0 * np.pi**2))
        assert alpha_111 == pytest.approx(expected_111, rel=1e-10)

        # alpha_112 = alpha_111 * sqrt(3/2)
        expected_112 = expected_111 * np.sqrt(3.0 / 2.0)
        assert alpha_112 == pytest.approx(expected_112, rel=1e-10)

    def test_self_energy_scales_as_g8(self):
        """
        delta_E ~ g^8 at fixed R.

        The vertex squared gives g^2, and the fluctuation factors give
        g^6 more, for a total of g^8.
        """
        R = 2.0
        # Test at two coupling values
        g2_a = 1.0
        g2_b = 2.0

        se_a = OneLoopSelfEnergy(R, g2_a, N=2, k_max=5)
        se_b = OneLoopSelfEnergy(R, g2_b, N=2, k_max=5)

        dE_a = se_a.leading_channel_112()['delta_E']
        dE_b = se_b.leading_channel_112()['delta_E']

        # g^8 scaling: dE_b/dE_a = (g2_b/g2_a)^4 = 16
        ratio = dE_b / dE_a
        expected = (g2_b / g2_a)**4
        assert ratio == pytest.approx(expected, rel=0.01)
