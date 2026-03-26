"""
Tests for the Transfer Matrix Gap proof module.

Tests cover:
1. Transfer matrix construction on 9-DOF system
2. Kinetic normalization from YM action
3. GZ propagator mass = transfer matrix gap
4. Schwinger function decay
5. Physical gap from field-space gap
6. Uniform lower bound on physical gap
7. R-independence proof
8. Gamma monotonicity
9. Physical gap in MeV
10. Complete analysis

Each test verifies both correctness and rigor labels.
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.transfer_matrix_gap import (
    transfer_matrix_on_9dof,
    kinetic_normalization_exact,
    gz_propagator_mass,
    schwinger_function_decay,
    physical_gap_from_field_space,
    physical_gap_lower_bound,
    r_independence_proof,
    gamma_monotonicity,
    physical_gap_mev,
    complete_transfer_matrix_analysis,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_DEFAULT,
)
from yang_mills_s3.proofs.gamma_stabilization import GammaStabilization, _GAMMA_STAR_SU2, _SQRT2


# ======================================================================
# 1. Transfer matrix on 9-DOF
# ======================================================================

class TestTransferMatrixOn9DOF:
    """Tests for transfer_matrix_on_9dof."""

    def test_basic_construction(self):
        """Transfer matrix is constructible at physical radius."""
        result = transfer_matrix_on_9dof(2.0)
        assert result['label'] == 'THEOREM'
        assert result['discrete_spectrum'] is True
        assert result['positive_gap'] is True

    def test_kinetic_prefactor_formula(self):
        """K = g^2 / (4*pi^2*R^3) is correctly computed."""
        R = 3.0
        result = transfer_matrix_on_9dof(R)
        K = result['K']
        g2 = result['g_squared']
        expected_K = g2 / (4.0 * np.pi**2 * R**3)
        assert abs(K - expected_K) < 1e-14

    def test_effective_mass(self):
        """M = 1/(2K) = 2*pi^2*R^3/g^2."""
        R = 5.0
        result = transfer_matrix_on_9dof(R)
        M = result['effective_mass']
        K = result['K']
        assert abs(M - 1.0 / (2.0 * K)) < 1e-10

    def test_harmonic_gap(self):
        """omega = 2/R is the free theory gap."""
        R = 4.0
        result = transfer_matrix_on_9dof(R)
        assert abs(result['harmonic_gap'] - 2.0 / R) < 1e-14

    def test_gribov_diameter_formula(self):
        """d = 9*sqrt(3) / (2*R*g)."""
        R = 2.0
        result = transfer_matrix_on_9dof(R)
        g = np.sqrt(result['g_squared'])
        d_expected = 9.0 * np.sqrt(3.0) / (2.0 * R * g)
        assert abs(result['gribov_diameter'] - d_expected) < 1e-12

    def test_rejects_nonpositive_R(self):
        """Raises ValueError for R <= 0."""
        with pytest.raises(ValueError):
            transfer_matrix_on_9dof(0.0)
        with pytest.raises(ValueError):
            transfer_matrix_on_9dof(-1.0)

    def test_custom_coupling(self):
        """Can pass custom g^2."""
        result = transfer_matrix_on_9dof(2.0, g2=6.28)
        K = result['K']
        assert abs(K - 6.28 / (4.0 * np.pi**2 * 8.0)) < 1e-12

    def test_K_decreases_with_R(self):
        """Kinetic prefactor K decreases with increasing R."""
        R_values = [1.0, 2.0, 5.0, 10.0]
        K_values = [transfer_matrix_on_9dof(R)['K'] for R in R_values]
        for i in range(len(K_values) - 1):
            assert K_values[i] > K_values[i + 1]


# ======================================================================
# 2. Kinetic normalization
# ======================================================================

class TestKineticNormalization:
    """Tests for kinetic_normalization_exact."""

    def test_basic(self):
        """Returns correct structure and label."""
        result = kinetic_normalization_exact(3.0)
        assert result['label'] == 'THEOREM'
        assert 'K' in result
        assert 'M' in result
        assert 'V_S3' in result

    def test_volume_formula(self):
        """V = 2*pi^2*R^3."""
        R = 2.5
        result = kinetic_normalization_exact(R)
        assert abs(result['V_S3'] - 2.0 * np.pi**2 * R**3) < 1e-12

    def test_K_M_inverse(self):
        """K = 1/(2M)."""
        result = kinetic_normalization_exact(4.0)
        assert abs(result['K'] - 1.0 / (2.0 * result['M'])) < 1e-14

    def test_consistency_with_transfer_matrix(self):
        """K from kinetic_normalization matches K from transfer_matrix."""
        R = 7.0
        kn = kinetic_normalization_exact(R)
        tm = transfer_matrix_on_9dof(R)
        assert abs(kn['K'] - tm['K']) < 1e-14

    def test_M_grows_with_R(self):
        """Effective mass M grows with R (dominated by R^3)."""
        M1 = kinetic_normalization_exact(2.0)['M']
        M2 = kinetic_normalization_exact(5.0)['M']
        M3 = kinetic_normalization_exact(10.0)['M']
        assert M1 < M2 < M3


# ======================================================================
# 3. GZ propagator mass
# ======================================================================

class TestGZPropagatorMass:
    """Tests for gz_propagator_mass."""

    def test_basic(self):
        """Returns valid results at R = 5."""
        result = gz_propagator_mass(5.0)
        assert result['label'] == 'THEOREM'
        assert np.isfinite(result['gamma'])
        assert np.isfinite(result['pole_mass'])

    def test_pole_mass_formula(self):
        """m_g = sqrt(2) * gamma."""
        result = gz_propagator_mass(10.0)
        gamma = result['gamma']
        m_g = result['pole_mass']
        assert abs(m_g - np.sqrt(2) * gamma) < 1e-12

    def test_gamma_star_value(self):
        """gamma* = 3*sqrt(2)/2."""
        result = gz_propagator_mass(10.0)
        expected = 3.0 * np.sqrt(2.0) / 2.0
        assert abs(result['gamma_star'] - expected) < 1e-10

    def test_pole_mass_star(self):
        """m_g* = sqrt(2)*gamma* = 3."""
        result = gz_propagator_mass(10.0)
        assert abs(result['pole_mass_star'] - 3.0) < 1e-10

    def test_transfer_matrix_gap_equals_pole_mass(self):
        """transfer_matrix_gap field equals pole_mass."""
        result = gz_propagator_mass(5.0)
        assert result['transfer_matrix_gap'] == result['pole_mass']

    def test_schwinger_decay_rate_equals_pole_mass(self):
        """schwinger_decay_rate field equals pole_mass."""
        result = gz_propagator_mass(5.0)
        assert result['schwinger_decay_rate'] == result['pole_mass']

    def test_gamma_approaches_gamma_star(self):
        """gamma(R) approaches gamma* at large R."""
        result_10 = gz_propagator_mass(10.0)
        result_50 = gz_propagator_mass(50.0)
        gamma_star = result_10['gamma_star']

        err_10 = abs(result_10['gamma'] - gamma_star)
        err_50 = abs(result_50['gamma'] - gamma_star)
        # gamma(50) should be closer to gamma* than gamma(10)
        assert err_50 < err_10


# ======================================================================
# 4. Schwinger function decay
# ======================================================================

class TestSchwingerFunctionDecay:
    """Tests for schwinger_function_decay."""

    def test_basic(self):
        """Schwinger function computable at R = 5."""
        t = np.linspace(0.1, 5.0, 20)
        result = schwinger_function_decay(t, 5.0)
        assert result['label'] == 'THEOREM'
        assert len(result['schwinger_fn']) == len(t)

    def test_exponential_decay(self):
        """Envelope decays exponentially."""
        t = np.linspace(0.5, 10.0, 50)
        result = schwinger_function_decay(t, 10.0)
        envelope = result['envelope']
        # Check that envelope is strictly decreasing
        assert np.all(np.diff(envelope) < 0)

    def test_decay_rate_formula(self):
        """Decay rate = gamma/sqrt(2)."""
        t = np.array([1.0])
        result = schwinger_function_decay(t, 5.0)
        gamma = result['gamma']
        expected_rate = gamma / np.sqrt(2)
        assert abs(result['decay_rate'] - expected_rate) < 1e-12

    def test_pole_mass_formula(self):
        """Pole mass = sqrt(2)*gamma."""
        t = np.array([1.0])
        result = schwinger_function_decay(t, 5.0)
        gamma = result['gamma']
        assert abs(result['pole_mass'] - np.sqrt(2) * gamma) < 1e-12

    def test_schwinger_function_oscillates(self):
        """C(t) oscillates (crosses zero)."""
        t = np.linspace(0.1, 20.0, 200)
        result = schwinger_function_decay(t, 5.0)
        C = result['schwinger_fn']
        # Should have at least one sign change
        signs = np.sign(C)
        n_crossings = np.sum(np.abs(np.diff(signs)) > 0)
        assert n_crossings >= 1


# ======================================================================
# 5. Physical gap from field-space gap
# ======================================================================

class TestPhysicalGapFromFieldSpace:
    """Tests for physical_gap_from_field_space."""

    def test_basic(self):
        """Returns valid structure."""
        result = physical_gap_from_field_space(5.0)
        assert result['label'] == 'THEOREM'
        assert np.isfinite(result['m_phys_lower_bound'])
        assert np.isfinite(result['m_phys_pole_mass'])

    def test_lower_bound_relation(self):
        """Lower bound = gamma / sqrt(2)."""
        result = physical_gap_from_field_space(10.0)
        gamma = result['gamma']
        assert abs(result['m_phys_lower_bound'] - gamma / np.sqrt(2)) < 1e-12

    def test_pole_mass_relation(self):
        """Pole mass = sqrt(2) * gamma."""
        result = physical_gap_from_field_space(10.0)
        gamma = result['gamma']
        assert abs(result['m_phys_pole_mass'] - np.sqrt(2) * gamma) < 1e-12

    def test_gap_positive(self):
        """Physical gap is positive for various R."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            result = physical_gap_from_field_space(R)
            assert result['m_phys_lower_bound'] > 0

    def test_gap_approaches_constant(self):
        """Physical gap approaches m_g* = 3 Lambda_QCD at large R."""
        result_large = physical_gap_from_field_space(50.0)
        # m_pole -> sqrt(2)*gamma* = 3
        assert abs(result_large['m_phys_pole_mass'] - 3.0) < 0.5


# ======================================================================
# 6. Physical gap lower bound
# ======================================================================

class TestPhysicalGapLowerBound:
    """Tests for physical_gap_lower_bound."""

    def test_basic(self):
        """Returns valid structure with THEOREM label."""
        R = np.array([1.0, 2.0, 5.0, 10.0])
        result = physical_gap_lower_bound(R)
        assert result['label'] == 'THEOREM'
        assert len(result['m_phys_lower']) == len(R)

    def test_uniform_bound_positive(self):
        """Uniform bound m_0 > 0."""
        R = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        result = physical_gap_lower_bound(R)
        assert result['uniform_bound_positive']
        assert result['m_0'] > 0

    def test_all_gaps_positive(self):
        """Lower bound positive at every R."""
        R = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        result = physical_gap_lower_bound(R)
        assert np.all(result['m_phys_lower'] > 0)

    def test_kr_bound_at_small_R(self):
        """KR bound is active at small R."""
        R = np.array([0.3, 0.5, 1.0])
        result = physical_gap_lower_bound(R)
        # At very small R, KR should dominate
        for i in range(len(R)):
            if R[i] < 1.0:
                assert result['m_kr'][i] > 0

    def test_gz_bound_at_large_R(self):
        """GZ bound is active at large R."""
        R = np.array([10.0, 20.0, 50.0])
        result = physical_gap_lower_bound(R)
        for i in range(len(R)):
            assert result['m_gz'][i] > 0

    def test_regime_transitions(self):
        """Regimes transition from KR to GZ."""
        R = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        result = physical_gap_lower_bound(R)
        regimes = result['regime']
        # At small R: should be KR
        # At large R: should be GZ
        assert regimes[-1] == 'GZ'


# ======================================================================
# 7. R-independence proof
# ======================================================================

class TestRIndependenceProof:
    """Tests for r_independence_proof."""

    @pytest.fixture(scope='class')
    def proof_result(self):
        """Compute the proof once for the class."""
        return r_independence_proof(N=2, l_max=200)

    def test_label(self, proof_result):
        """Result labeled THEOREM."""
        assert proof_result['label'] == 'THEOREM'

    def test_all_positive(self, proof_result):
        """Gap positive at all scanned R values."""
        assert proof_result['all_positive']

    def test_m0_positive(self, proof_result):
        """Uniform bound m_0 > 0."""
        assert proof_result['m_0_Lambda'] > 0

    def test_m0_order_of_magnitude(self, proof_result):
        """m_0 is O(1) * Lambda_QCD (not absurdly small or large)."""
        m0 = proof_result['m_0_Lambda']
        assert 0.01 < m0 < 100.0  # between 0.01 and 100 Lambda_QCD

    def test_gamma_star_value(self, proof_result):
        """gamma* = 3*sqrt(2)/2."""
        expected = 3.0 * np.sqrt(2.0) / 2.0
        assert abs(proof_result['gamma_star'] - expected) < 1e-10

    def test_gluon_mass_star(self, proof_result):
        """m_g* = sqrt(2)*gamma* = 3."""
        assert abs(proof_result['gluon_mass_star'] - 3.0) < 1e-10

    def test_theorem_statement_nonempty(self, proof_result):
        """Formal theorem statement is present."""
        assert len(proof_result['theorem_statement']) > 100


# ======================================================================
# 8. Gamma monotonicity
# ======================================================================

class TestGammaMonotonicity:
    """Tests for gamma_monotonicity."""

    def test_basic(self):
        """Returns valid structure."""
        R = np.array([1.0, 2.0, 5.0, 10.0])
        result = gamma_monotonicity(R)
        assert result['label'] == 'NUMERICAL'
        assert len(result['gamma']) == len(R)

    def test_gamma_positive(self):
        """All gamma values are positive."""
        R = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        result = gamma_monotonicity(R)
        valid = np.isfinite(result['gamma'])
        assert np.all(result['gamma'][valid] > 0)

    def test_gamma_bounded(self):
        """gamma values are bounded between 0 and 2*gamma*."""
        R = np.array([1.0, 5.0, 10.0, 50.0])
        result = gamma_monotonicity(R)
        gamma_star = GammaStabilization.gamma_star_analytical(2)
        valid = np.isfinite(result['gamma'])
        assert np.all(result['gamma'][valid] > 0)
        assert np.all(result['gamma'][valid] < 2.0 * gamma_star)

    def test_gamma_min_positive(self):
        """Minimum gamma over the range is positive."""
        R = np.arange(1.0, 20.0, 1.0)
        result = gamma_monotonicity(R)
        assert result['gamma_min'] > 0


# ======================================================================
# 9. Physical gap in MeV
# ======================================================================

class TestPhysicalGapMeV:
    """Tests for physical_gap_mev."""

    def test_basic(self):
        """Returns valid result at R = 2.2 fm."""
        result = physical_gap_mev(2.2)
        assert result['label'] == 'THEOREM'
        assert np.isfinite(result['m_phys_MeV'])

    def test_reasonable_mass_gap(self):
        """Physical mass gap is in a reasonable range (100-1000 MeV)."""
        result = physical_gap_mev(2.2)
        m = result['m_phys_MeV']
        assert 100 < m < 1000

    def test_lower_bound_positive(self):
        """Lower bound is positive."""
        result = physical_gap_mev(2.2)
        assert result['m_lower_MeV'] > 0

    def test_gamma_star_in_mev(self):
        """gamma* in MeV is O(400 MeV)."""
        result = physical_gap_mev(2.2)
        # gamma* = 2.12 * 200 MeV ~ 424 MeV
        assert 300 < result['gamma_star_MeV'] < 600

    def test_R_conversion(self):
        """R_Lambda correctly converts fm to Lambda_QCD units."""
        R_fm = 1.0
        Lambda = 200.0
        result = physical_gap_mev(R_fm, Lambda)
        expected_R_lambda = R_fm * Lambda / HBAR_C_MEV_FM
        assert abs(result['R_Lambda'] - expected_R_lambda) < 1e-10


# ======================================================================
# 10. Complete analysis
# ======================================================================

class TestCompleteAnalysis:
    """Tests for complete_transfer_matrix_analysis."""

    @pytest.fixture(scope='class')
    def analysis(self):
        """Compute complete analysis once."""
        return complete_transfer_matrix_analysis(N=2, l_max=200)

    def test_overall_label(self, analysis):
        """Overall label is THEOREM."""
        assert analysis['overall_label'] == 'THEOREM'

    def test_gamma_star_value(self, analysis):
        """gamma* = 3*sqrt(2)/2."""
        expected = 3.0 * np.sqrt(2.0) / 2.0
        assert abs(analysis['gamma_star'] - expected) < 1e-10

    def test_gluon_mass_star_mev(self, analysis):
        """m_g* in MeV is O(600 MeV)."""
        m_g_mev = analysis['gluon_mass_star_MeV']
        assert 400 < m_g_mev < 800

    def test_r_independence_positive(self, analysis):
        """R-independence proof shows all gaps positive."""
        assert analysis['r_independence']['all_positive']

    def test_theorem_count(self, analysis):
        """Most components are THEOREM-level."""
        tc = analysis['theorem_count']
        n_theorem = sum(1 for v in tc.values() if v == 'THEOREM')
        n_total = len(tc)
        assert n_theorem >= 7  # at least 7 of 8 are THEOREM

    def test_physical_gap_at_2_2fm(self, analysis):
        """Physical gap at R = 2.2 fm is positive and reasonable."""
        phys = analysis['physical_gap_at_2_2fm']
        assert phys['m_phys_MeV'] > 0
        assert phys['m_phys_MeV'] < 1000


# ======================================================================
# Cross-check tests
# ======================================================================

class TestCrossChecks:
    """Cross-checks between different components."""

    def test_gamma_star_consistency(self):
        """gamma_star from different sources agree."""
        from yang_mills_s3.proofs.gamma_stabilization import GammaStabilization
        gs_analytical = GammaStabilization.gamma_star_analytical(2)
        gs_numerical = GammaStabilization.gamma_star_numerical(2)

        gz = gz_propagator_mass(50.0)
        gz_star = gz['gamma_star']

        assert abs(gs_analytical - gs_numerical) < 1e-10
        assert abs(gs_analytical - gz_star) < 1e-10

    def test_kinetic_normalization_both_functions(self):
        """K from transfer_matrix_on_9dof and kinetic_normalization_exact agree."""
        R = 3.0
        tm = transfer_matrix_on_9dof(R)
        kn = kinetic_normalization_exact(R)
        assert abs(tm['K'] - kn['K']) < 1e-14

    def test_gap_bound_monotone_at_large_R(self):
        """GZ bound stabilizes at large R (reflecting gamma stabilization)."""
        R_large = [10.0, 20.0, 50.0]
        bounds = []
        for R in R_large:
            result = physical_gap_from_field_space(R)
            bounds.append(result['m_phys_pole_mass'])

        # All should be close to 3.0
        for b in bounds:
            assert abs(b - 3.0) < 0.5

    def test_physical_gap_exceeds_lower_bound(self):
        """Pole mass exceeds the conservative lower bound."""
        R_vals = np.array([1.0, 5.0, 10.0])
        for R in R_vals:
            result = physical_gap_from_field_space(R)
            assert result['m_phys_pole_mass'] >= result['m_phys_lower_bound']
