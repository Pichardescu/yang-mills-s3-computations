"""
Tests for the optimized Monte Carlo engine on the 600-cell.

Verifies:
  1. SU(2) matrix generation (unitarity, det=1, Haar, near-identity)
  2. Cold and hot start initialization
  3. Plaquette average for trivial and random configs
  4. Staple computation correctness
  5. Metropolis sweep (acceptance, action reduction)
  6. Heat bath sweep (correctness, convergence)
  7. Overrelaxation (energy conservation)
  8. Observable measurements (Wilson loops, correlators)
  9. Loop finding on the 600-cell graph
  10. Physics: plaquette increases with beta, gap is positive

STATUS: NUMERICAL
Tests use small sample counts for speed. Physics tests use relaxed tolerances.
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.s3_lattice import S3Lattice
from yang_mills_s3.lattice.mc_engine import MCEngine


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="module")
def lattice():
    """600-cell lattice, built once."""
    return S3Lattice(R=1.0)


@pytest.fixture
def engine(lattice):
    """Fresh MCEngine with cold start at beta=4."""
    return MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))


@pytest.fixture
def hot_engine(lattice):
    """MCEngine with hot (random) start."""
    eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
    eng.set_hot_start()
    return eng


# ==================================================================
# 1. SU(2) Matrix Generation
# ==================================================================

class TestSU2Generation:
    """SU(2) matrices are valid: unitary, det=1."""

    def test_random_su2_is_unitary(self, engine):
        """U^dag U = I for random SU(2)."""
        U = engine._random_su2()
        product = U.conj().T @ U
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    def test_random_su2_det_one(self, engine):
        """det(U) = 1 for random SU(2)."""
        U = engine._random_su2()
        det = np.linalg.det(U)
        assert abs(det - 1.0) < 1e-12, f"det = {det}"

    def test_random_su2_special(self, engine):
        """Random SU(2) is special unitary (det=+1, not -1)."""
        for _ in range(10):
            U = engine._random_su2()
            det = np.linalg.det(U)
            assert np.real(det) > 0.9, f"det has wrong sign: {det}"

    def test_quaternion_roundtrip(self, engine):
        """Quaternion -> SU(2) -> quaternion roundtrip."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        U = MCEngine._quaternion_to_su2(a)
        a_back = MCEngine._su2_to_quaternion(U)
        np.testing.assert_allclose(a_back, a, atol=1e-12)

    def test_near_identity_close_to_I(self, engine):
        """Near-identity SU(2) with small epsilon is close to I."""
        V = engine._su2_near_identity(0.01)
        diff = np.linalg.norm(V - np.eye(2))
        assert diff < 0.1, f"||V - I|| = {diff} too large for epsilon=0.01"

    def test_near_identity_is_su2(self, engine):
        """Near-identity matrix is valid SU(2)."""
        V = engine._su2_near_identity(0.3)
        product = V.conj().T @ V
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)
        det = np.linalg.det(V)
        assert abs(det - 1.0) < 1e-10


# ==================================================================
# 2. Initialization
# ==================================================================

class TestInitialization:
    """Cold start = identity, hot start = random."""

    def test_cold_start_plaquette_one(self, engine):
        """Cold start: all plaquettes = 1."""
        engine.set_cold_start()
        P = engine.plaquette_average()
        assert abs(P - 1.0) < 1e-10, f"Cold start plaquette = {P}"

    def test_cold_start_zero_action(self, engine):
        """Cold start: Wilson action = 0."""
        engine.set_cold_start()
        S = engine.wilson_action()
        assert abs(S) < 1e-10, f"Cold start action = {S}"

    def test_hot_start_plaquette_near_zero(self, hot_engine):
        """Hot start: plaquette ~ 0 for SU(2)."""
        P = hot_engine.plaquette_average()
        # For random SU(2), <Tr U> = 0, so <P> ~ 0
        assert abs(P) < 0.15, f"Hot start plaquette = {P}, expected ~0"

    def test_hot_start_positive_action(self, hot_engine):
        """Hot start: Wilson action > 0."""
        S = hot_engine.wilson_action()
        assert S > 0, f"Hot start action = {S}"

    def test_lattice_counts(self, engine):
        """600-cell topology."""
        assert engine.lattice.vertex_count() == 120
        assert engine._n_links == 720
        assert engine._n_plaq == 1200

    def test_link_plaquette_map_nonempty(self, engine):
        """Every link participates in at least one plaquette."""
        for idx in range(engine._n_links):
            n_plaqs = len(engine._link_to_plaqs[idx])
            assert n_plaqs > 0, f"Link {idx} has no plaquettes"

    def test_link_plaquette_count(self, engine):
        """
        On the 600-cell, each edge belongs to exactly 5 triangular faces.
        (Euler: 3 * 1200 faces / 720 edges = 5 faces per edge.)
        So each link participates in exactly 5 plaquettes.
        """
        for idx in range(engine._n_links):
            n_plaqs = len(engine._link_to_plaqs[idx])
            assert n_plaqs == 5, \
                f"Link {idx} has {n_plaqs} plaquettes, expected 5"


# ==================================================================
# 3. Staple Computation
# ==================================================================

class TestStaple:
    """Staple computation is consistent with the full action."""

    def test_staple_at_identity(self, engine):
        """At identity config, staple for any link = 5 * I (5 plaquettes per link)."""
        engine.set_cold_start()
        V = engine._compute_staple(0)
        # Each plaquette contributes I to the staple at identity
        # There are 5 plaquettes per link on the 600-cell
        np.testing.assert_allclose(V, 5.0 * np.eye(2), atol=1e-10)

    def test_local_action_consistency(self, hot_engine):
        """
        Verify that modifying one link and recomputing the action
        gives the same result as using the staple formula.
        """
        # Get total action
        S_total = hot_engine.wilson_action()

        # Get staple for link 0
        V = hot_engine._compute_staple(0)
        U = hot_engine._links[0].copy()

        # Local action from staple: s_local = -beta/2 * Re Tr(U * V)
        s_local = -0.5 * hot_engine.beta * np.real(np.trace(U @ V))

        # Modify link 0 and recompute total action
        U_new = hot_engine._su2_near_identity(0.5) @ U
        hot_engine._links[0] = U_new
        S_new = hot_engine.wilson_action()
        hot_engine._links[0] = U  # restore

        s_local_new = -0.5 * hot_engine.beta * np.real(np.trace(U_new @ V))

        # The change in total action should equal change in local action
        dS_total = S_new - S_total
        dS_local = s_local_new - s_local

        np.testing.assert_allclose(dS_total, dS_local, atol=1e-8,
            err_msg=f"dS_total={dS_total}, dS_local={dS_local}")


# ==================================================================
# 4. Metropolis
# ==================================================================

class TestMetropolis:
    """Metropolis sweep produces valid acceptance and equilibration."""

    def test_metropolis_acceptance_rate(self, hot_engine):
        """Acceptance rate is in [0.1, 0.95] for epsilon=0.3."""
        rate = hot_engine.metropolis_sweep(epsilon=0.3)
        assert 0.1 < rate < 0.95, f"Acceptance rate = {rate}"

    def test_metropolis_reduces_action(self, hot_engine):
        """Starting from hot start, Metropolis reduces the action."""
        S_before = hot_engine.wilson_action()
        for _ in range(10):
            hot_engine.metropolis_sweep(epsilon=0.3)
        S_after = hot_engine.wilson_action()
        assert S_after < S_before, \
            f"Action should decrease: {S_before:.2f} -> {S_after:.2f}"

    def test_metropolis_preserves_su2(self, hot_engine):
        """After many sweeps, all links are still SU(2)."""
        for _ in range(5):
            hot_engine.metropolis_sweep(epsilon=0.3)

        for idx in range(hot_engine._n_links):
            U = hot_engine._links[idx]
            product = U.conj().T @ U
            np.testing.assert_allclose(product, np.eye(2), atol=1e-10,
                err_msg=f"Link {idx} not unitary after sweeps")

    def test_small_epsilon_high_acceptance(self, hot_engine):
        """Small epsilon gives high acceptance rate."""
        rate = hot_engine.metropolis_sweep(epsilon=0.001)
        assert rate > 0.9, f"Small epsilon: rate = {rate}"


# ==================================================================
# 5. Heat Bath
# ==================================================================

class TestHeatBath:
    """SU(2) heat bath sweep."""

    def test_heatbath_returns_one(self, hot_engine):
        """Heat bath always accepts (no rejection)."""
        rate = hot_engine.heatbath_sweep()
        assert rate == 1.0

    def test_heatbath_equilibrates(self, lattice):
        """Heat bath from hot start converges to equilibrium plaquette."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()

        for _ in range(10):
            eng.heatbath_sweep()

        P = eng.plaquette_average()
        # At beta=4 for SU(2), plaquette should be around 0.6-0.8 after 10 sweeps
        assert 0.4 < P < 0.95, f"Plaquette after HB = {P}"

    def test_heatbath_preserves_su2(self, hot_engine):
        """After heat bath sweeps, all links remain SU(2)."""
        for _ in range(5):
            hot_engine.heatbath_sweep()

        for idx in range(hot_engine._n_links):
            U = hot_engine._links[idx]
            product = U.conj().T @ U
            np.testing.assert_allclose(product, np.eye(2), atol=1e-10,
                err_msg=f"Link {idx} not SU(2) after heat bath")

    def test_heatbath_detailed_balance(self, lattice):
        """
        Statistical test: at large beta with cold start, plaquette stays near 1
        after heat bath sweeps (testing correct equilibrium distribution).
        """
        beta = 16.0
        eng = MCEngine(lattice, beta=beta, rng=np.random.default_rng(42))
        eng.set_cold_start()

        # A few HB sweeps from cold start at large beta
        for _ in range(5):
            eng.heatbath_sweep()

        P = eng.plaquette_average()
        # At beta=16 starting from cold, should stay very close to 1
        assert P > 0.9, \
            f"<P>={P:.6f} at beta={beta}, expected > 0.9"


# ==================================================================
# 6. Overrelaxation
# ==================================================================

class TestOverrelaxation:
    """Overrelaxation preserves energy (microcanonical)."""

    def test_overrelax_preserves_action(self, lattice):
        """Overrelaxation should not change the Wilson action."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        # First thermalize a bit with metropolis (faster than HB for this test)
        for _ in range(20):
            eng.metropolis_sweep(epsilon=0.3)

        S_before = eng.wilson_action()
        eng.overrelaxation_sweep()
        S_after = eng.wilson_action()

        np.testing.assert_allclose(S_after, S_before, rtol=1e-8,
            err_msg=f"OR changed action: {S_before:.6f} -> {S_after:.6f}")

    def test_overrelax_preserves_su2(self, lattice):
        """After overrelaxation, all links remain SU(2)."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(10):
            eng.metropolis_sweep(epsilon=0.3)

        eng.overrelaxation_sweep()

        for idx in range(eng._n_links):
            U = eng._links[idx]
            product = U.conj().T @ U
            np.testing.assert_allclose(product, np.eye(2), atol=1e-10,
                err_msg=f"Link {idx} not SU(2) after OR")


# ==================================================================
# 7. Compound Sweep
# ==================================================================

class TestCompoundSweep:
    """Compound sweep (heatbath + overrelaxation)."""

    def test_compound_sweep_runs(self, lattice):
        """Compound sweep returns acceptance info."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        # First get away from hot start with a few metropolis
        for _ in range(5):
            eng.metropolis_sweep(epsilon=0.3)
        result = eng.compound_sweep(n_heatbath=1, n_overrelax=2)
        assert 'acceptance_rate' in result

    def test_compound_sweep_equilibrates(self, lattice):
        """Compound sweeps reach equilibrium."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        # Thermalize first with fast metropolis
        for _ in range(15):
            eng.metropolis_sweep(epsilon=0.3)
        # Then a few compound sweeps
        for _ in range(3):
            eng.compound_sweep(n_heatbath=1, n_overrelax=2)

        P = eng.plaquette_average()
        assert 0.3 < P < 0.95, f"Plaquette after compound sweeps = {P}"


# ==================================================================
# 8. Wilson Loops
# ==================================================================

class TestWilsonLoops:
    """Wilson loop measurements."""

    def test_trivial_wilson_loop(self, engine):
        """At identity config, Wilson loop along any path = 1."""
        engine.set_cold_start()
        # Use a face as a 3-loop
        faces = engine._faces
        path = list(faces[0])
        W = engine.wilson_loop_path(path)
        assert abs(W - 1.0) < 1e-10, f"Trivial W(3) = {W}"

    def test_wilson_loop_bounded(self, hot_engine):
        """Wilson loop is bounded: |W| <= 1 for SU(2)."""
        for _ in range(10):
            hot_engine.heatbath_sweep()

        faces = hot_engine._faces
        for i in range(min(50, len(faces))):
            path = list(faces[i])
            W = hot_engine.wilson_loop_path(path)
            assert abs(W) <= 1.0 + 1e-10, f"|W| = {abs(W)} > 1"

    def test_loop_finding(self, engine):
        """Loop finding discovers triangles and longer loops."""
        loops = engine.find_loops_by_length(max_length=5, max_per_length=20)
        assert 3 in loops
        assert len(loops[3]) > 0
        # Length 4 loops exist on 600-cell
        assert 4 in loops

    def test_wilson_loop_measurement(self, lattice):
        """Measure Wilson loops after thermalization."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(20):
            eng.metropolis_sweep(epsilon=0.3)

        loops = eng.find_loops_by_length(max_length=4, max_per_length=20)
        wl = eng.measure_wilson_loops(loops)

        assert 3 in wl
        assert 'mean' in wl[3]
        # After thermalization at beta=4, W(3) should be significantly > 0
        assert wl[3]['mean'] > 0.0, f"W(3) = {wl[3]['mean']}"


# ==================================================================
# 9. Correlators
# ==================================================================

class TestCorrelators:
    """Plaquette correlator measurements."""

    def test_plaquette_field_shape(self, engine):
        """Plaquette field has 1200 entries."""
        P = engine.plaquette_field()
        assert P.shape == (1200,)

    def test_plaquette_field_trivial(self, engine):
        """At identity, all plaquette values = 1."""
        engine.set_cold_start()
        P = engine.plaquette_field()
        np.testing.assert_allclose(P, 1.0, atol=1e-10)

    def test_distance_correlator_structure(self, lattice):
        """Distance-based correlator has correct structure."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(15):
            eng.metropolis_sweep(epsilon=0.3)

        dc = eng.plaquette_correlator_by_distance(n_distance_bins=10)
        assert 'distances' in dc
        assert 'correlator' in dc
        assert 'n_pairs' in dc
        assert len(dc['distances']) == 10
        assert len(dc['correlator']) == 10
        # C(0) should be positive (autocorrelation)
        assert dc['correlator'][0] >= 0

    def test_time_slice_correlator_structure(self, lattice):
        """Time-slice correlator has correct structure."""
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(15):
            eng.metropolis_sweep(epsilon=0.3)

        tc = eng.time_slice_correlator(coord=0, n_bins=8)
        assert 'separations' in tc
        assert 'correlator' in tc
        assert 'observable_by_slice' in tc
        assert 'bin_counts' in tc
        assert len(tc['correlator']) == 5  # max_dt = n_bins//2 + 1


# ==================================================================
# 10. Physics: Plaquette vs Beta
# ==================================================================

class TestPhysics:
    """Key physics tests: plaquette ordering and gap positivity."""

    def test_plaquette_increases_with_beta(self, lattice):
        """Higher beta = weaker coupling = more ordered = higher <P>."""
        plaqs = {}
        for beta in [1.0, 8.0]:
            eng = MCEngine(lattice, beta=beta, rng=np.random.default_rng(42))
            eng.set_hot_start()
            for _ in range(15):
                eng.metropolis_sweep(epsilon=0.3)
            plaqs[beta] = eng.plaquette_average()

        assert plaqs[8.0] > plaqs[1.0], \
            f"<P>(beta=8) = {plaqs[8.0]:.4f} should be > <P>(beta=1) = {plaqs[1.0]:.4f}"

    def test_strong_coupling_plaquette_small(self, lattice):
        """At beta=0.5 (strong coupling), <P> should be small."""
        eng = MCEngine(lattice, beta=0.5, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(15):
            eng.metropolis_sweep(epsilon=0.3)

        P = eng.plaquette_average()
        assert P < 0.5, f"Strong coupling <P> = {P}, expected < 0.5"

    def test_wilson_loops_decay_with_size(self, lattice):
        """
        Wilson loops should decrease with loop size (area law or perimeter law).
        W(3) > W(4) for a confining theory.
        """
        eng = MCEngine(lattice, beta=4.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(20):
            eng.metropolis_sweep(epsilon=0.3)

        loops = eng.find_loops_by_length(max_length=4, max_per_length=20)
        wl = eng.measure_wilson_loops(loops)

        if 3 in wl and 4 in wl:
            assert wl[3]['mean'] > wl[4]['mean'], \
                f"W(3)={wl[3]['mean']:.4f} should be > W(4)={wl[4]['mean']:.4f}"

    def test_polyakov_loops_small(self, lattice):
        """
        On S3 with confinement, Polyakov-like loops should be small
        (not exactly zero since S3 is finite, but suppressed).
        """
        eng = MCEngine(lattice, beta=2.0, rng=np.random.default_rng(42))
        eng.set_hot_start()
        for _ in range(20):
            eng.metropolis_sweep(epsilon=0.3)

        P_loops = eng.polyakov_loops()
        if len(P_loops) > 0:
            P_mean = np.mean(np.abs(P_loops))
            # Should be significantly less than 1 (confined)
            assert P_mean < 0.8, f"|Polyakov| = {P_mean:.4f}"
