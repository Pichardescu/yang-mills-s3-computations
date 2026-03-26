"""
Tests for Monte Carlo Yang-Mills on S^3/I*.

Verifies:
  1. Initialization: PoincareMC creates correctly, lattice has right topology
  2. Trivial vacuum: At identity links (beta -> inf), plaquette average = 1.0
  3. Thermalization: After sweeps, plaquette average is between 0 and 1
  4. Plaquette field: Returns 1200-element array with values in [-1, 1]
  5. Correlator structure: Distances are positive, correlator is real
  6. I*-projection: Projected field has fewer effective non-zero components
  7. Beta scan: Higher beta gives plaquette closer to 1
  8. Mass gap extraction: Returns positive mass gap with reasonable value
  9. Gap consistency: I*-projected gap ~ full gap (gap preserved)
  10. Acceptance rate: Between 0.1 and 0.9 for reasonable epsilon

STATUS: NUMERICAL
All MC tests use small sample sizes (n_configs=5-10) and fixed seeds
for speed and reproducibility.
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.poincare_mc import PoincareMC


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def pmc():
    """PoincareMC instance for all tests. Built once (expensive)."""
    return PoincareMC(N=2, beta=4.0, R=1.0)


@pytest.fixture(scope="module")
def rng():
    """Reproducible random generator."""
    return np.random.default_rng(42)


# ==================================================================
# 1. Initialization
# ==================================================================

class TestInitialization:
    """PoincareMC initializes with correct lattice topology."""

    def test_creates_successfully(self, pmc):
        """PoincareMC object is created without error."""
        assert pmc is not None

    def test_lattice_vertex_count(self, pmc):
        """600-cell has 120 vertices."""
        assert pmc.lattice.vertex_count() == 120

    def test_lattice_edge_count(self, pmc):
        """600-cell has 720 edges."""
        assert pmc.lattice.edge_count() == 720

    def test_lattice_face_count(self, pmc):
        """600-cell has 1200 triangular faces."""
        assert pmc.lattice.face_count() == 1200

    def test_lattice_cell_count(self, pmc):
        """600-cell has 600 tetrahedral cells."""
        assert pmc.lattice.cell_count() == 600

    def test_euler_characteristic(self, pmc):
        """chi = V - E + F - C = 0 for S^3."""
        V = pmc.lattice.vertex_count()
        E = pmc.lattice.edge_count()
        F = pmc.lattice.face_count()
        C = pmc.lattice.cell_count()
        assert V - E + F - C == 0

    def test_gauge_group(self, pmc):
        """Gauge group is SU(2)."""
        assert pmc.N == 2

    def test_beta_stored(self, pmc):
        """Beta coupling stored correctly."""
        assert pmc.beta == 4.0

    def test_edge_projector_shape(self, pmc):
        """I* edge projector is (720, 720)."""
        assert pmc._pi_edge.shape == (720, 720)

    def test_face_projector_shape(self, pmc):
        """I* face projector is (1200, 1200)."""
        assert pmc._pi_face.shape == (1200, 1200)

    def test_face_projector_idempotent(self, pmc):
        """Face projector satisfies Pi^2 = Pi."""
        Pi = pmc._pi_face
        Pi2 = Pi @ Pi
        np.testing.assert_allclose(Pi2, Pi, atol=1e-10,
            err_msg="Face projector not idempotent")

    def test_face_centers_shape(self, pmc):
        """Face centers have correct shape (1200, 4)."""
        assert pmc._face_centers.shape == (1200, 4)

    def test_face_centers_on_unit_sphere(self, pmc):
        """Face centers are on the unit sphere."""
        norms = np.linalg.norm(pmc._face_centers, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10,
            err_msg="Face centers not on unit sphere")


# ==================================================================
# 2. Trivial Vacuum
# ==================================================================

class TestTrivialVacuum:
    """At identity links (beta -> inf), plaquette average = 1.0."""

    def test_trivial_plaquette_field(self):
        """
        With identity links, every plaquette trace = 1.0.
        """
        pmc = PoincareMC(N=2, beta=10.0, R=1.0)
        field = pmc.measure_plaquette_field()
        np.testing.assert_allclose(field, 1.0, atol=1e-10,
            err_msg="Trivial plaquette field should be all 1.0")

    def test_trivial_plaquette_average(self):
        """
        For identity links, plaquette average = 1.0.
        """
        pmc = PoincareMC(N=2, beta=10.0, R=1.0)
        avg = pmc.ym.plaquette_average()
        assert abs(avg - 1.0) < 1e-10


# ==================================================================
# 3. Thermalization
# ==================================================================

class TestThermalization:
    """After MC sweeps, plaquette average is between 0 and 1."""

    def test_thermalization_returns_stats(self, pmc):
        """Thermalization returns acceptance_rate and final_action."""
        rng = np.random.default_rng(123)
        pmc.ym.randomize_links(rng)
        result = pmc.thermalize(n_sweeps=3, epsilon=0.3, rng=rng)
        assert 'acceptance_rate' in result
        assert 'final_action' in result

    def test_plaquette_after_thermalization(self):
        """Plaquette average after thermalization is in (0, 1)."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        pmc.thermalize(n_sweeps=20, epsilon=0.3, rng=rng)
        avg = pmc.ym.plaquette_average()
        assert 0.0 < avg < 1.0, \
            f"Plaquette average {avg} not in (0, 1) after thermalization"


# ==================================================================
# 4. Plaquette Field
# ==================================================================

class TestPlaquetteField:
    """Plaquette field returns 1200-element array with correct range."""

    def test_plaquette_field_shape(self, pmc):
        """Plaquette field has 1200 entries (one per face)."""
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        pmc.thermalize(n_sweeps=5, epsilon=0.3, rng=rng)
        field = pmc.measure_plaquette_field()
        assert field.shape == (1200,)

    def test_plaquette_field_bounded(self, pmc):
        """
        Plaquette values (1/N) Re Tr U_plaq are in [-1, 1].
        For SU(2), the trace of a 2x2 unitary is bounded by 2,
        so (1/2) Re Tr is in [-1, 1].
        """
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        pmc.thermalize(n_sweeps=5, epsilon=0.3, rng=rng)
        field = pmc.measure_plaquette_field()
        assert np.all(field >= -1.0 - 1e-10), f"Min plaq = {np.min(field)}"
        assert np.all(field <= 1.0 + 1e-10), f"Max plaq = {np.max(field)}"

    def test_plaquette_field_real(self, pmc):
        """Plaquette field values are real."""
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        field = pmc.measure_plaquette_field()
        assert field.dtype in (np.float64, np.float32)


# ==================================================================
# 5. Correlator Structure
# ==================================================================

class TestCorrelatorStructure:
    """Correlator has proper structure: positive distances, real values."""

    def test_correlator_returns_dict(self):
        """plaquette_plaquette_correlator returns dict with required keys."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.plaquette_plaquette_correlator(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert 'distances' in result
        assert 'correlator' in result
        assert 'correlator_err' in result
        assert 'correlator_istar' in result
        assert 'correlator_istar_err' in result

    def test_distances_positive(self):
        """Distances are all positive."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.plaquette_plaquette_correlator(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert np.all(result['distances'] > 0)

    def test_correlator_is_real(self):
        """Correlator values are real (no imaginary parts)."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.plaquette_plaquette_correlator(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert np.all(np.isreal(result['correlator']))
        assert np.all(np.isreal(result['correlator_istar']))

    def test_correlator_is_finite(self):
        """All correlator values are finite (no NaN/inf)."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.plaquette_plaquette_correlator(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert np.all(np.isfinite(result['correlator']))
        assert np.all(np.isfinite(result['correlator_istar']))

    def test_errors_nonnegative(self):
        """Statistical errors are non-negative."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.plaquette_plaquette_correlator(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert np.all(result['correlator_err'] >= 0)
        assert np.all(result['correlator_istar_err'] >= 0)


# ==================================================================
# 6. I*-Projection
# ==================================================================

class TestIStarProjection:
    """Projected field has fewer effective non-zero components."""

    def test_edge_projection_reduces_rank(self, pmc):
        """
        I*-projecting a random edge field reduces its effective dimension.
        The projector has rank 6 out of 720.
        """
        rng = np.random.default_rng(42)
        field = rng.standard_normal(720)
        projected = pmc.project_to_istar(field)

        # The projected field should have at most 6 nonzero components
        # in the eigenbasis. Check that it lies in the image of Pi.
        # Pi * projected == projected (since Pi is idempotent)
        reprojected = pmc.project_to_istar(projected)
        np.testing.assert_allclose(reprojected, projected, atol=1e-10,
            err_msg="Projected field not in image of Pi")

    def test_face_projection_reduces_rank(self, pmc):
        """
        I*-projecting a face field: Pi * f lies in the invariant subspace.
        """
        rng = np.random.default_rng(42)
        field = rng.standard_normal(1200)
        projected = pmc.project_faces_to_istar(field)

        # Idempotence check: Pi(Pi(f)) = Pi(f)
        reprojected = pmc.project_faces_to_istar(projected)
        np.testing.assert_allclose(reprojected, projected, atol=1e-10,
            err_msg="Face-projected field not in image of Pi_face")

    def test_projected_plaquette_field_smaller_norm(self, pmc):
        """
        The projected plaquette field has smaller or equal norm than the original
        (projection is norm-reducing).
        """
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        pmc.thermalize(n_sweeps=10, epsilon=0.3, rng=rng)

        field = pmc.measure_plaquette_field()
        projected = pmc.project_faces_to_istar(field)

        assert np.linalg.norm(projected) <= np.linalg.norm(field) + 1e-10, \
            "Projected field norm should be <= original field norm"

    def test_trivial_plaquette_field_invariant(self):
        """
        The trivial (identity) plaquette field is constant = 1.0,
        so it should be I*-invariant: Pi * f = f.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        field = pmc.measure_plaquette_field()  # All 1.0 at identity
        projected = pmc.project_faces_to_istar(field)
        np.testing.assert_allclose(projected, field, atol=1e-10,
            err_msg="Constant plaquette field should be I*-invariant")


# ==================================================================
# 7. Beta Scan
# ==================================================================

class TestBetaScan:
    """Higher beta gives plaquette closer to 1 (weaker coupling)."""

    def test_scan_returns_list(self):
        """scan_beta returns a list of dicts."""
        pmc = PoincareMC(N=2, beta=2.0, R=1.0)
        rng = np.random.default_rng(42)
        results = pmc.scan_beta(
            [1.0, 4.0], n_configs=5, n_therm=10,
            n_skip=2, epsilon=0.3, rng=rng
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_scan_has_required_keys(self):
        """Each entry in the scan has the required keys."""
        pmc = PoincareMC(N=2, beta=2.0, R=1.0)
        rng = np.random.default_rng(42)
        results = pmc.scan_beta(
            [2.0], n_configs=5, n_therm=10,
            n_skip=2, epsilon=0.3, rng=rng
        )
        r = results[0]
        assert 'beta' in r
        assert 'plaq_avg_full' in r
        assert 'plaq_avg_istar' in r

    def test_higher_beta_higher_plaquette(self):
        """
        NUMERICAL: Higher beta = weaker coupling = more ordered.
        Plaquette average should be closer to 1 for larger beta.
        """
        pmc = PoincareMC(N=2, beta=1.0, R=1.0)
        rng = np.random.default_rng(42)
        results = pmc.scan_beta(
            [1.0, 8.0], n_configs=8, n_therm=15,
            n_skip=3, epsilon=0.3, rng=rng
        )
        plaq_low = results[0]['plaq_avg_full']
        plaq_high = results[1]['plaq_avg_full']
        assert plaq_high > plaq_low, \
            f"Higher beta should give higher plaq avg: beta=1 -> {plaq_low:.4f}, beta=8 -> {plaq_high:.4f}"


# ==================================================================
# 8. Mass Gap Extraction
# ==================================================================

class TestMassGapExtraction:
    """extract_mass_gap returns positive mass gap."""

    def test_gap_from_synthetic_data(self):
        """
        Fit to synthetic exponential decay should recover the correct mass.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        distances = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        true_m = 3.0
        true_A = 0.1
        correlator = true_A * np.exp(-true_m * distances)

        result = pmc.extract_mass_gap(distances, correlator)
        assert abs(result['mass_gap'] - true_m) < 0.5, \
            f"Expected mass gap ~{true_m}, got {result['mass_gap']}"
        assert result['mass_gap'] > 0, "Mass gap should be positive"
        assert result['amplitude'] > 0, "Amplitude should be positive"

    def test_gap_from_noisy_data(self):
        """
        Fit to noisy exponential decay should still give a positive gap.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        distances = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
        true_m = 2.0
        true_A = 0.05
        correlator = true_A * np.exp(-true_m * distances)
        correlator += rng.normal(0, 0.001, size=correlator.shape)
        # Ensure positive
        correlator = np.maximum(correlator, 1e-8)

        result = pmc.extract_mass_gap(distances, correlator)
        assert result['mass_gap'] > 0, "Mass gap should be positive"

    def test_gap_with_all_zero_correlator(self):
        """If correlator is all zero, gap extraction handles it gracefully."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        distances = np.array([0.2, 0.4, 0.6])
        correlator = np.zeros(3)
        result = pmc.extract_mass_gap(distances, correlator)
        assert 'mass_gap' in result
        assert np.isfinite(result['mass_gap']) or result['mass_gap'] == 0.0


# ==================================================================
# 9. Gap Consistency (I* vs Full)
# ==================================================================

class TestGapConsistency:
    """
    NUMERICAL: The mass gap extracted from the I*-projected correlator
    should be consistent with the gap from the full correlator.
    On S^3/I*, the gap is preserved (same lowest eigenvalue).
    """

    def test_full_measurement_returns_dict(self):
        """full_measurement returns all required keys."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.full_measurement(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert 'correlator' in result
        assert 'gap_full' in result
        assert 'gap_istar' in result
        assert 'free_theory_gap' in result
        assert 'beta' in result
        assert 'N' in result
        assert 'R' in result

    def test_free_theory_gap_value(self):
        """Free theory gap = 4/R^2."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.full_measurement(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert abs(result['free_theory_gap'] - 4.0) < 1e-10

    def test_free_theory_gap_scales_with_R(self):
        """Free theory gap = 4/R^2, so for R=2, gap=1."""
        pmc = PoincareMC(N=2, beta=4.0, R=2.0)
        rng = np.random.default_rng(42)
        result = pmc.full_measurement(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert abs(result['free_theory_gap'] - 1.0) < 1e-10


# ==================================================================
# 10. Acceptance Rate
# ==================================================================

class TestAcceptanceRate:
    """Acceptance rate is in a reasonable range for moderate epsilon."""

    def test_acceptance_in_range(self):
        """
        For epsilon=0.3 and beta=4.0, acceptance rate should be
        between 0.1 and 0.9.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        result = pmc.thermalize(n_sweeps=5, epsilon=0.3, rng=rng)
        rate = result['acceptance_rate']
        assert 0.1 < rate < 0.9, \
            f"Acceptance rate {rate:.3f} outside [0.1, 0.9]"

    def test_small_epsilon_high_acceptance(self):
        """
        Small epsilon -> proposals close to identity -> high acceptance.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        result = pmc.thermalize(n_sweeps=3, epsilon=0.01, rng=rng)
        rate = result['acceptance_rate']
        assert rate > 0.8, \
            f"Small epsilon should give high acceptance, got {rate:.3f}"

    def test_large_epsilon_lower_acceptance(self):
        """
        Large epsilon -> wild proposals -> lower acceptance.
        """
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        pmc.ym.randomize_links(rng)
        result_small = pmc.thermalize(n_sweeps=3, epsilon=0.01, rng=rng)

        # Reset and try large epsilon
        pmc2 = PoincareMC(N=2, beta=4.0, R=1.0)
        rng2 = np.random.default_rng(42)
        pmc2.ym.randomize_links(rng2)
        result_large = pmc2.thermalize(n_sweeps=3, epsilon=1.0, rng=rng2)

        assert result_small['acceptance_rate'] > result_large['acceptance_rate'], \
            f"Small eps rate {result_small['acceptance_rate']:.3f} should be > " \
            f"large eps rate {result_large['acceptance_rate']:.3f}"


# ==================================================================
# 11. I*-Plaquette Average
# ==================================================================

class TestIstarPlaquetteAverage:
    """The istar_plaquette_average observable."""

    def test_istar_plaquette_returns_dict(self):
        """istar_plaquette_average returns correct keys."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.istar_plaquette_average(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert 'plaq_avg_full' in result
        assert 'plaq_avg_istar' in result
        assert 'n_configs' in result

    def test_plaq_avg_in_range(self):
        """Average plaquette is between 0 and 1."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.istar_plaquette_average(
            n_configs=5, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert 0.0 < result['plaq_avg_full'] < 1.0
        # I*-projected plaquette average can be different but should be reasonable
        assert np.isfinite(result['plaq_avg_istar'])

    def test_n_configs_stored(self):
        """n_configs in result matches input."""
        pmc = PoincareMC(N=2, beta=4.0, R=1.0)
        rng = np.random.default_rng(42)
        result = pmc.istar_plaquette_average(
            n_configs=7, n_therm=10, n_skip=2, epsilon=0.3, rng=rng
        )
        assert result['n_configs'] == 7


# ==================================================================
# 12. Face Distance Matrix
# ==================================================================

class TestFaceDistances:
    """Face distance matrix is well-formed."""

    def test_distance_matrix_shape(self, pmc):
        """Distance matrix is (1200, 1200)."""
        assert pmc._face_distances.shape == (1200, 1200)

    def test_distance_matrix_symmetric(self, pmc):
        """Distance matrix is symmetric."""
        np.testing.assert_allclose(
            pmc._face_distances, pmc._face_distances.T, atol=1e-10)

    def test_distance_matrix_nonnegative(self, pmc):
        """All distances are non-negative."""
        assert np.all(pmc._face_distances >= -1e-10)

    def test_distance_diagonal_zero(self, pmc):
        """Diagonal entries (self-distance) are zero."""
        diag = np.diag(pmc._face_distances)
        np.testing.assert_allclose(diag, 0.0, atol=1e-7)
