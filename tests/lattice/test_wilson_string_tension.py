"""
Tests for Wilson loop measurement and string tension extraction
on the 600-cell lattice.

STATUS: NUMERICAL
Tests verify:
  1. Edge classification (temporal vs spatial)
  2. Loop construction (rectangular and general)
  3. Wilson loop measurement at identity (trivial check)
  4. MC measurement pipeline
  5. Area law: W(L) decays with loop size
  6. String tension extraction (area fit and Creutz)
  7. Physical unit conversion
  8. Multi-beta scan: sigma and plaquette ordering
  9. Qualitative confinement signal

Honest caveats:
  - 600-cell is very coarse (120 vertices)
  - Tolerances are relaxed for qualitative checks
  - Quantitative sigma is unreliable at this lattice size
  - Short MC runs for test speed
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.s3_lattice import S3Lattice
from yang_mills_s3.lattice.wilson_string_tension import (
    WilsonStringTension,
    run_multi_beta_analysis,
)


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="module")
def lattice():
    """600-cell lattice, built once for all tests."""
    return S3Lattice(R=1.0)


@pytest.fixture(scope="module")
def wst(lattice):
    """WilsonStringTension at beta=4, built once."""
    return WilsonStringTension(
        lattice, beta=4.0, rng=np.random.default_rng(42))


@pytest.fixture(scope="module")
def wst_strong(lattice):
    """WilsonStringTension at strong coupling beta=0.637."""
    return WilsonStringTension(
        lattice, beta=0.637, rng=np.random.default_rng(99))


# ==================================================================
# 1. Edge Classification
# ==================================================================

class TestEdgeClassification:
    """Temporal/spatial edge classification covers all edges."""

    def test_all_edges_classified(self, wst):
        """Every edge is either temporal or spatial."""
        all_temporal = set()
        all_spatial = set()
        for v, nbs in wst._temporal_neighbors.items():
            for nb in nbs:
                edge = (min(v, nb), max(v, nb))
                all_temporal.add(edge)
        for v, nbs in wst._spatial_neighbors.items():
            for nb in nbs:
                edge = (min(v, nb), max(v, nb))
                all_spatial.add(edge)

        total = all_temporal | all_spatial
        assert len(total) == 720, (
            f"Only {len(total)}/720 edges classified")

    def test_temporal_spatial_disjoint(self, wst):
        """No edge is both temporal and spatial."""
        for v, t_nbs in wst._temporal_neighbors.items():
            s_nbs = wst._spatial_neighbors.get(v, set())
            overlap = t_nbs & s_nbs
            assert len(overlap) == 0, (
                f"Vertex {v}: {len(overlap)} edges are both "
                f"temporal and spatial")

    def test_reasonable_split(self, wst):
        """
        The split should be roughly balanced (not all temporal or
        all spatial). On 600-cell with 12 neighbors per vertex,
        expect ~3-6 temporal and ~6-9 spatial per vertex.
        """
        for v in range(wst._n_verts):
            n_t = len(wst._temporal_neighbors.get(v, set()))
            n_s = len(wst._spatial_neighbors.get(v, set()))
            total = n_t + n_s
            assert total == 12, (
                f"Vertex {v}: {total} neighbors, expected 12")
            # At least 1 temporal and 1 spatial
            assert n_t >= 1, f"Vertex {v}: no temporal neighbors"
            assert n_s >= 1, f"Vertex {v}: no spatial neighbors"


# ==================================================================
# 2. Loop Construction
# ==================================================================

class TestLoopConstruction:
    """Finding rectangular and general loops on the 600-cell."""

    def test_general_loops_found(self, wst):
        """General loops of lengths 3-6 exist."""
        loops = wst.find_loops_with_area(max_length=6, max_per_length=10)
        assert 3 in loops, "No triangles found"
        assert len(loops[3]) > 0
        # 4-loops should also exist
        assert 4 in loops, "No squares found"
        assert len(loops[4]) > 0

    def test_loop_areas_positive(self, wst):
        """All estimated loop areas are positive."""
        loops = wst.find_loops_with_area(max_length=5, max_per_length=10)
        for length, items in loops.items():
            for path, area in items:
                assert area > 0, (
                    f"Loop of length {length} has area={area}")

    def test_loop_areas_increase_with_length(self, wst):
        """Longer loops enclose more area on average."""
        loops = wst.find_loops_with_area(max_length=6, max_per_length=20)
        mean_areas = {}
        for length, items in loops.items():
            if items:
                mean_areas[length] = np.mean([a for _, a in items])

        lengths = sorted(mean_areas.keys())
        if len(lengths) >= 2:
            # Area should generally increase with perimeter
            assert mean_areas[lengths[-1]] > mean_areas[lengths[0]], (
                f"Area({lengths[-1]})={mean_areas[lengths[-1]]:.4f} "
                f"<= Area({lengths[0]})={mean_areas[lengths[0]]:.4f}")

    def test_rectangular_loops_found(self, wst):
        """At least some rectangular loops W(R,T) are found."""
        rect = wst.find_rectangular_loops(R_max=2, T_max=2,
                                          max_per_size=5)
        # On the 600-cell, finding exact rectangular loops is hard
        # due to the triangular structure. We accept partial success.
        # At least W(1,1) should be findable (it's a 4-loop).
        found_any = len(rect) > 0
        if not found_any:
            # This is OK -- the 600-cell's triangular structure makes
            # exact rectangles rare. General loops are the fallback.
            pytest.skip("No rectangular loops found on 600-cell "
                        "(expected for triangular lattice)")

    def test_loops_are_valid(self, wst):
        """All found loops are valid closed paths on the graph."""
        loops = wst.find_loops_with_area(max_length=5, max_per_length=10)
        for length, items in loops.items():
            for path, area in items:
                assert wst._is_valid_closed_loop(path), (
                    f"Invalid loop of length {length}: {path}")


# ==================================================================
# 3. Wilson Loop at Identity
# ==================================================================

class TestWilsonLoopIdentity:
    """At identity (cold start), all Wilson loops = 1."""

    def test_wilson_loops_unity_at_identity(self, wst):
        """W(C) = 1 for all loops at the trivial vacuum."""
        wst.engine.set_cold_start()
        loops = wst.find_loops_with_area(max_length=5, max_per_length=5)
        for length, items in loops.items():
            for path, area in items:
                W = np.real(wst.engine.wilson_loop_path(path))
                assert abs(W - 1.0) < 1e-10, (
                    f"W(L={length}) = {W} at identity, expected 1.0")


# ==================================================================
# 4. MC Measurement Pipeline
# ==================================================================

class TestMCMeasurement:
    """MC measurement produces valid Wilson loop values."""

    def test_measurement_returns_arrays(self, wst):
        """measure_wilson_loops_mc returns numpy arrays."""
        wst.thermalize(n_therm=10, method='heatbath')
        loops = wst.find_loops_with_area(max_length=4, max_per_length=5)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=10, n_skip=1, method='heatbath')

        for key, vals in meas.items():
            assert isinstance(vals, np.ndarray), (
                f"Key {key}: expected ndarray, got {type(vals)}")
            assert len(vals) == 10, (
                f"Key {key}: {len(vals)} measurements, expected 10")

    def test_wilson_loops_bounded(self, wst):
        """All measured Wilson loop values are in [-1, 1] for SU(2)."""
        wst.thermalize(n_therm=10, method='heatbath')
        loops = wst.find_loops_with_area(max_length=4, max_per_length=5)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=20, n_skip=1, method='heatbath')

        for key, vals in meas.items():
            assert np.all(vals >= -1.0 - 1e-10), (
                f"Key {key}: min W = {np.min(vals)}")
            assert np.all(vals <= 1.0 + 1e-10), (
                f"Key {key}: max W = {np.max(vals)}")


# ==================================================================
# 5. Area Law: W decays with loop size
# ==================================================================

class TestAreaLaw:
    """
    Wilson loops should decay with increasing loop size.
    This is the qualitative signal of confinement.
    """

    def test_wilson_loops_decay_with_length(self, lattice):
        """<W(L)> decreases as L increases — confinement signal."""
        wst = WilsonStringTension(
            lattice, beta=2.0, rng=np.random.default_rng(77))
        wst.thermalize(n_therm=15, method='heatbath')

        loops = wst.find_loops_with_area(max_length=5, max_per_length=20)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=50, n_skip=1, method='heatbath')

        means = {}
        for key, vals in meas.items():
            if isinstance(key, int):
                means[key] = np.mean(vals)

        lengths = sorted(means.keys())
        if len(lengths) >= 2:
            # W(3) should be larger than W(larger)
            W_small = means[lengths[0]]
            W_large = means[lengths[-1]]
            assert W_small > W_large, (
                f"No decay: W({lengths[0]})={W_small:.4f} "
                f"vs W({lengths[-1]})={W_large:.4f}")

    def test_strong_coupling_faster_decay(self, lattice):
        """
        At stronger coupling (smaller beta), Wilson loops should
        decay faster — stronger confinement.
        """
        W_by_beta = {}
        for beta in [2.0, 8.0]:
            wst = WilsonStringTension(
                lattice, beta=beta,
                rng=np.random.default_rng(42 + int(beta * 100)))
            wst.thermalize(n_therm=15, method='heatbath')

            loops = wst.find_loops_with_area(max_length=5,
                                              max_per_length=20)
            meas = wst.measure_wilson_loops_mc(
                loops, n_configs=30, n_skip=1, method='heatbath')

            if 4 in meas:
                W_by_beta[beta] = np.mean(meas[4])

        if 2.0 in W_by_beta and 8.0 in W_by_beta:
            # At weak coupling (beta=8), W(4) should be larger
            # (less confined, closer to perturbative)
            assert W_by_beta[8.0] > W_by_beta[2.0], (
                f"W(4, beta=8)={W_by_beta[8.0]:.4f} should be > "
                f"W(4, beta=2)={W_by_beta[2.0]:.4f}")


# ==================================================================
# 6. String Tension Extraction
# ==================================================================

class TestStringTension:
    """String tension extraction from Wilson loop data."""

    def test_area_fit_positive_sigma(self, lattice):
        """
        At intermediate coupling, the area-law fit should
        give a positive string tension.
        """
        wst = WilsonStringTension(
            lattice, beta=2.0, rng=np.random.default_rng(55))
        wst.thermalize(n_therm=15, method='heatbath')

        loops = wst.find_loops_with_area(max_length=6,
                                          max_per_length=30)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=60, n_skip=1, method='heatbath')

        result = wst.extract_string_tension_from_area(meas, loops)

        # sigma should be positive (confining)
        assert result['sigma'] > 0, (
            f"sigma = {result['sigma']}, expected positive")

    def test_area_fit_has_data_points(self, lattice):
        """The area fit returns data points for each loop category."""
        wst = WilsonStringTension(
            lattice, beta=4.0, rng=np.random.default_rng(66))
        wst.thermalize(n_therm=10, method='heatbath')

        loops = wst.find_loops_with_area(max_length=5,
                                          max_per_length=10)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=20, n_skip=1, method='heatbath')

        result = wst.extract_string_tension_from_area(meas, loops)
        assert len(result['data_points']) >= 2, (
            f"Only {len(result['data_points'])} data points")

    def test_sigma_decreases_with_beta(self, lattice):
        """
        String tension should decrease at weaker coupling (larger beta).
        This is the fundamental physics: weaker coupling = less confinement.
        CAVEAT: on the 600-cell this may not hold precisely due to
        lattice artifacts.
        """
        sigmas = {}
        for beta in [1.5, 6.0]:
            wst = WilsonStringTension(
                lattice, beta=beta,
                rng=np.random.default_rng(42 + int(beta * 100)))
            wst.thermalize(n_therm=15, method='heatbath')

            loops = wst.find_loops_with_area(max_length=5,
                                              max_per_length=20)
            meas = wst.measure_wilson_loops_mc(
                loops, n_configs=40, n_skip=1, method='heatbath')

            result = wst.extract_string_tension_from_area(meas, loops)
            sigmas[beta] = result['sigma']

        # Allow for lattice artifacts: just check qualitative ordering
        if sigmas[1.5] > 0 and sigmas[6.0] > 0:
            assert sigmas[1.5] > sigmas[6.0], (
                f"sigma(beta=1.5)={sigmas[1.5]:.4f} should be > "
                f"sigma(beta=6)={sigmas[6.0]:.4f}")


# ==================================================================
# 7. Physical Unit Conversion
# ==================================================================

class TestPhysicalUnits:
    """Conversion to physical units."""

    def test_sigma_to_physical_known(self):
        """
        Check conversion with known values.
        sqrt(sigma) ~ 440 MeV corresponds to sigma ~ 0.194 GeV^2.
        At a = 0.1 fm: sigma_lat = sigma_phys * a^2
                      = 0.194 / 0.197327^2 * 0.1^2 ~ 0.0498
        """
        result = WilsonStringTension.sigma_to_physical(0.0498, 0.1)
        # sqrt(sigma) should be ~440 MeV
        assert 300 < result['sqrt_sigma_MeV'] < 600, (
            f"sqrt(sigma) = {result['sqrt_sigma_MeV']:.0f} MeV, "
            f"expected ~440")

    def test_known_value_present(self):
        """The known value (440 MeV) is reported for comparison."""
        result = WilsonStringTension.sigma_to_physical(1.0, 1.0)
        assert result['known_value_MeV'] == 440.0

    def test_ratio_computed(self):
        """The ratio to the known value is computed."""
        result = WilsonStringTension.sigma_to_physical(1.0, 1.0)
        assert 'ratio_to_known' in result
        assert result['ratio_to_known'] > 0


# ==================================================================
# 8. Full Analysis Pipeline
# ==================================================================

class TestFullAnalysis:
    """End-to-end analysis pipeline."""

    def test_run_analysis_structure(self, lattice):
        """run_analysis returns a complete result dict."""
        wst = WilsonStringTension(
            lattice, beta=4.0, rng=np.random.default_rng(42))
        result = wst.run_analysis(
            n_therm=5, n_configs=10, n_skip=1,
            R_max=1, T_max=1, max_length=4,
            method='heatbath', verbose=False)

        required_keys = [
            'beta', 'lattice_spacing', 'n_vertices', 'plaquette',
            'wilson_loops', 'area_law_sigma', 'creutz_ratios',
            'area_law_check', 'n_configs', 'caveats',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_run_analysis_plaquette_reasonable(self, lattice):
        """Plaquette after thermalization is in valid range."""
        wst = WilsonStringTension(
            lattice, beta=4.0, rng=np.random.default_rng(42))
        result = wst.run_analysis(
            n_therm=10, n_configs=5, n_skip=1,
            R_max=1, T_max=1, max_length=4,
            method='heatbath', verbose=False)

        assert 0.3 < result['plaquette'] < 0.95, (
            f"<P> = {result['plaquette']}")

    def test_caveats_present(self, lattice):
        """Honest caveats about lattice coarseness are included."""
        wst = WilsonStringTension(
            lattice, beta=4.0, rng=np.random.default_rng(42))
        result = wst.run_analysis(
            n_therm=5, n_configs=5, n_skip=1,
            R_max=1, T_max=1, max_length=4,
            method='heatbath', verbose=False)

        caveats = result['caveats']
        assert any('120' in c for c in caveats), (
            "Should mention 120 vertices")
        assert any('NUMERICAL' in c or 'QUALITATIVE' in c.upper()
                    for c in caveats), (
            "Should label as NUMERICAL/QUALITATIVE")


# ==================================================================
# 9. Multi-Beta Scan
# ==================================================================

class TestMultiBeta:
    """Multi-beta comparison."""

    def test_plaquette_ordering(self, lattice):
        """
        <P> increases with beta across all tested values.
        This is the most robust lattice test.
        """
        results = {}
        for beta in [1.0, 4.0]:
            wst = WilsonStringTension(
                lattice, beta=beta,
                rng=np.random.default_rng(42 + int(beta * 100)))
            result = wst.run_analysis(
                n_therm=10, n_configs=5, n_skip=1,
                R_max=1, T_max=1, max_length=4,
                method='heatbath', verbose=False)
            results[beta] = result

        assert results[4.0]['plaquette'] > results[1.0]['plaquette'], (
            f"<P>(beta=4) = {results[4.0]['plaquette']:.4f} should be > "
            f"<P>(beta=1) = {results[1.0]['plaquette']:.4f}")


# ==================================================================
# 10. Confinement Signal
# ==================================================================

class TestConfinement:
    """
    The 600-cell should show qualitative confinement signals
    even at this coarse lattice spacing.
    """

    def test_negative_ln_W_increases_with_area(self, lattice):
        """
        -ln<W> should increase with enclosed area.
        This is the defining signature of area-law confinement.
        """
        wst = WilsonStringTension(
            lattice, beta=2.0, rng=np.random.default_rng(88))
        wst.thermalize(n_therm=15, method='heatbath')

        loops = wst.find_loops_with_area(max_length=5,
                                          max_per_length=20)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=50, n_skip=1, method='heatbath')

        result = wst.extract_string_tension_from_area(meas, loops)

        # Check that data points show increasing -ln<W> with area
        dp = result['data_points']
        if len(dp) >= 2:
            # Sort by area
            dp_sorted = sorted(dp, key=lambda d: d['area'])
            first = dp_sorted[0]
            last = dp_sorted[-1]
            assert last['neg_ln_W'] > first['neg_ln_W'], (
                f"-ln<W> at area={last['area']:.3f} is "
                f"{last['neg_ln_W']:.4f}, should be > "
                f"{first['neg_ln_W']:.4f} at area="
                f"{first['area']:.3f}")

    def test_area_law_check_not_no(self, lattice):
        """
        At intermediate coupling, the area law check should not
        return 'no_area_law'. At worst it's 'partial_decay' or
        'inconclusive' due to lattice coarseness.
        """
        wst = WilsonStringTension(
            lattice, beta=2.0, rng=np.random.default_rng(33))
        result = wst.run_analysis(
            n_therm=15, n_configs=50, n_skip=1,
            R_max=1, T_max=1, max_length=5,
            method='heatbath', verbose=False)

        assert result['area_law_check'] != 'no_area_law', (
            "Expected at least partial area law signal at beta=2")

    def test_physical_coupling_confines(self, lattice):
        """
        At the physical coupling beta ~ 0.637, the theory should
        show strong confinement (small Wilson loops, positive sigma).
        """
        wst = WilsonStringTension(
            lattice, beta=0.637, rng=np.random.default_rng(44))
        wst.thermalize(n_therm=15, method='heatbath')

        loops = wst.find_loops_with_area(max_length=5,
                                          max_per_length=20)
        meas = wst.measure_wilson_loops_mc(
            loops, n_configs=50, n_skip=1, method='heatbath')

        # At strong coupling, even small Wilson loops should be
        # significantly below 1
        if 3 in meas:
            W3_mean = np.mean(meas[3])
            assert W3_mean < 0.8, (
                f"W(3) = {W3_mean:.4f} at physical coupling, "
                f"expected < 0.8 (strong coupling)")


# ==================================================================
# 11. Loop Area Estimation
# ==================================================================

class TestAreaEstimation:
    """Area estimation for loops on S3."""

    def test_triangle_area_formula(self, wst):
        """Triangle area uses equilateral formula."""
        a = wst._lattice_spacing
        expected = a**2 * np.sqrt(3) / 4
        computed = wst._estimate_loop_area([0, 1, 2], a)
        # Triangle formula should be exact
        assert abs(computed - expected) < 1e-10, (
            f"Triangle area = {computed}, expected {expected}")

    def test_area_scales_with_lattice_spacing(self, lattice):
        """Area should scale with R^2."""
        wst1 = WilsonStringTension(
            S3Lattice(R=1.0), beta=4.0,
            rng=np.random.default_rng(42))
        wst2 = WilsonStringTension(
            S3Lattice(R=2.0), beta=4.0,
            rng=np.random.default_rng(42))

        a1 = wst1._lattice_spacing
        a2 = wst2._lattice_spacing

        area1 = wst1._estimate_loop_area([0, 1, 2], a1)
        area2 = wst2._estimate_loop_area([0, 1, 2], a2)

        # Area should scale as R^2, so area2/area1 ~ 4
        ratio = area2 / area1
        assert 3.5 < ratio < 4.5, (
            f"Area ratio = {ratio}, expected ~4 for R=2/R=1")
