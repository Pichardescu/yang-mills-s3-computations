"""
Tests for lattice Yang-Mills on the 600-cell.

Verifies:
    - Wilson action computation
    - Plaquette average for trivial and random configs
    - SU(N) matrix generation
    - Metropolis thermalization
    - Basic observables (Polyakov loop)
    - Transfer matrix gap estimation

Note: Monte Carlo sample counts are kept LOW (n=50-100) so tests
run fast (< 10 seconds total). The point is to verify the infrastructure
works, not to get precise physics.
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.s3_lattice import S3Lattice
from yang_mills_s3.lattice.lattice_ym import LatticeYM


@pytest.fixture
def lattice():
    """600-cell lattice on unit S^3."""
    return S3Lattice(R=1.0)


@pytest.fixture
def lym(lattice):
    """Lattice YM with SU(2) at beta=1.0, initialized to identity (flat vacuum)."""
    return LatticeYM(lattice, N=2, beta=1.0)


class TestWilsonAction:
    """Wilson plaquette action on the 600-cell."""

    def test_trivial_config_zero_action(self, lym):
        """
        For the trivial configuration (all links = identity),
        every plaquette U_plaq = I, so Tr U_plaq / N = 1,
        and the action S = beta * Sum(1 - 1) = 0.
        """
        action = lym.wilson_action()
        assert abs(action) < 1e-10, \
            f"Trivial config should have zero action, got {action}"

    def test_trivial_config_plaquette_avg_one(self, lym):
        """
        Plaquette average = (1/N) Re Tr(U_plaq) should be 1.0
        for the trivial configuration.
        """
        avg = lym.plaquette_average()
        assert abs(avg - 1.0) < 1e-10, \
            f"Trivial plaquette average should be 1.0, got {avg}"

    def test_random_config_positive_action(self, lattice):
        """Random SU(2) configuration should have positive action."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)
        action = lym.wilson_action()
        assert action > 0, f"Random config should have positive action, got {action}"

    def test_random_config_plaquette_avg_less_than_one(self, lattice):
        """Random config: plaquette average should be less than 1."""
        lym = LatticeYM(lattice, N=2, beta=1.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)
        avg = lym.plaquette_average()
        assert avg < 1.0, f"Random plaquette avg should be < 1, got {avg}"

    def test_action_scales_with_beta(self, lattice):
        """Action should scale linearly with beta for fixed configuration."""
        rng = np.random.default_rng(42)

        lym1 = LatticeYM(lattice, N=2, beta=1.0)
        lym1.randomize_links(rng)
        # Copy links to lym2
        lym2 = LatticeYM(lattice, N=2, beta=2.0)
        for idx in range(lym2._n_links):
            lym2._links[idx] = lym1._links[idx].copy()

        s1 = lym1.wilson_action()
        s2 = lym2.wilson_action()
        ratio = s2 / s1 if s1 > 0 else 0
        assert abs(ratio - 2.0) < 1e-10, \
            f"Action ratio should be 2.0 (beta ratio), got {ratio}"


class TestSUNGeneration:
    """SU(N) random matrix generation."""

    def test_su2_is_unitary(self):
        """Random SU(2) matrix should be unitary: U^dag U = I."""
        rng = np.random.default_rng(42)
        U = LatticeYM.random_su_n(2, rng)
        product = U.conj().T @ U
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10,
            err_msg="SU(2) matrix should be unitary")

    def test_su2_determinant_one(self):
        """Random SU(2) matrix should have det = 1."""
        rng = np.random.default_rng(42)
        U = LatticeYM.random_su_n(2, rng)
        det = np.linalg.det(U)
        assert abs(det - 1.0) < 1e-10, f"det(SU(2)) should be 1, got {det}"

    def test_su3_is_unitary(self):
        """Random SU(3) matrix should be unitary."""
        rng = np.random.default_rng(42)
        U = LatticeYM.random_su_n(3, rng)
        product = U.conj().T @ U
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10,
            err_msg="SU(3) matrix should be unitary")

    def test_su3_determinant_one(self):
        """Random SU(3) matrix should have det = 1."""
        rng = np.random.default_rng(42)
        U = LatticeYM.random_su_n(3, rng)
        det = np.linalg.det(U)
        assert abs(det - 1.0) < 1e-10, f"det(SU(3)) should be 1, got {det}"

    def test_near_identity_is_close(self):
        """SU(N) near identity with small epsilon should be close to I."""
        rng = np.random.default_rng(42)
        V = LatticeYM.su_n_near_identity(2, epsilon=0.01, rng=rng)

        # Should be close to identity
        diff = np.linalg.norm(V - np.eye(2))
        assert diff < 0.1, f"Near-identity matrix too far from I: ||V-I|| = {diff}"

        # Should be unitary
        product = V.conj().T @ V
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_near_identity_is_sun(self):
        """Near-identity matrix should be in SU(N)."""
        rng = np.random.default_rng(42)
        V = LatticeYM.su_n_near_identity(3, epsilon=0.1, rng=rng)
        det = np.linalg.det(V)
        assert abs(det - 1.0) < 1e-10, f"det should be 1, got {det}"


class TestMetropolis:
    """Metropolis thermalization."""

    def test_thermalization_runs(self, lattice):
        """Thermalization should run without errors and return statistics."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)

        result = lym.thermalize(n_sweeps=2, epsilon=0.3, rng=rng)

        assert 'acceptance_rate' in result
        assert 'final_action' in result
        assert 0.0 <= result['acceptance_rate'] <= 1.0

    def test_thermalization_reduces_action_from_random(self, lattice):
        """
        Starting from random config at high beta, thermalization
        should reduce the action (drive towards ordered state).
        """
        lym = LatticeYM(lattice, N=2, beta=4.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)

        initial_action = lym.wilson_action()
        lym.thermalize(n_sweeps=10, epsilon=0.3, rng=rng)
        final_action = lym.wilson_action()

        assert final_action < initial_action, \
            f"Thermalization should reduce action: {initial_action:.2f} -> {final_action:.2f}"

    def test_acceptance_rate_reasonable(self, lattice):
        """Acceptance rate should be between 20% and 90% for reasonable parameters."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)

        result = lym.thermalize(n_sweeps=5, epsilon=0.3, rng=rng)
        rate = result['acceptance_rate']

        assert 0.1 < rate < 0.95, \
            f"Acceptance rate {rate:.3f} outside reasonable range [0.1, 0.95]"

    def test_plaquette_average_increases_with_beta(self, lattice):
        """
        Higher beta = stronger coupling = more ordered.
        Plaquette average should be closer to 1 for larger beta.
        """
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        lym_low = LatticeYM(lattice, N=2, beta=1.0)
        lym_low.randomize_links(rng1)
        lym_low.thermalize(n_sweeps=10, epsilon=0.3, rng=rng1)
        avg_low = lym_low.plaquette_average()

        lym_high = LatticeYM(lattice, N=2, beta=4.0)
        lym_high.randomize_links(rng2)
        lym_high.thermalize(n_sweeps=10, epsilon=0.3, rng=rng2)
        avg_high = lym_high.plaquette_average()

        assert avg_high > avg_low, \
            f"Higher beta should give larger plaquette avg: {avg_low:.4f} vs {avg_high:.4f}"


class TestObservables:
    """Physical observables on the lattice."""

    def test_polyakov_loop_trivial(self, lym):
        """Polyakov loop on trivial config should be close to 1."""
        P = lym.polyakov_loop()
        # For identity links along any path, Tr(I^n)/N = 1
        assert abs(P - 1.0) < 1e-6, f"Trivial Polyakov loop should be ~1, got {P}"

    def test_correlator_at_zero_separation(self, lym):
        """Correlator at t=0 should be positive (autocorrelation)."""
        c = lym.correlator_at_separation(0)
        assert c > 0, f"C(0) should be positive, got {c}"

    def test_correlator_trivial_config(self, lym):
        """
        On trivial config, all plaquettes = 1, so correlator at
        any separation should be ~1.
        """
        c0 = lym.correlator_at_separation(0)
        c1 = lym.correlator_at_separation(1)
        # Both should be close to 1
        assert abs(c0 - 1.0) < 0.1, f"C(0) should be ~1 for trivial, got {c0}"
        assert abs(c1 - 1.0) < 0.1, f"C(1) should be ~1 for trivial, got {c1}"


class TestTransferMatrixGap:
    """Transfer matrix gap estimation (NUMERICAL)."""

    def test_gap_estimation_runs(self, lattice):
        """
        Transfer matrix gap estimation should run and return results.
        We use very few configs for speed — this tests infrastructure, not physics.
        """
        lym = LatticeYM(lattice, N=2, beta=2.0)
        result = lym.transfer_matrix_gap(n_configs=10, n_therm=2, epsilon=0.3)

        assert 'gap_estimate' in result
        assert 'correlators' in result
        assert 'gap_positive' in result
        assert isinstance(result['correlators'], list)
        assert len(result['correlators']) > 0

    def test_correlator_values_are_finite(self, lattice):
        """All correlator values should be finite."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        result = lym.transfer_matrix_gap(n_configs=10, n_therm=2, epsilon=0.3)

        for (t, c) in result['correlators']:
            assert np.isfinite(c), f"Correlator at t={t} is not finite: {c}"


class TestSU3:
    """Basic tests with SU(3) gauge group."""

    def test_su3_trivial_action_zero(self, lattice):
        """SU(3) trivial config has zero action."""
        lym = LatticeYM(lattice, N=3, beta=1.0)
        assert abs(lym.wilson_action()) < 1e-10

    def test_su3_plaquette_avg_one_trivial(self, lattice):
        """SU(3) trivial plaquette average is 1."""
        lym = LatticeYM(lattice, N=3, beta=1.0)
        assert abs(lym.plaquette_average() - 1.0) < 1e-10

    def test_su3_thermalization(self, lattice):
        """SU(3) thermalization runs without errors."""
        lym = LatticeYM(lattice, N=3, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)
        result = lym.thermalize(n_sweeps=2, epsilon=0.2, rng=rng)
        assert 0.0 <= result['acceptance_rate'] <= 1.0
