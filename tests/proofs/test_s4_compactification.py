"""
Tests for the S^4 Compactification Argument.

Tests the conformal maps, YM invariance, point removal theorem,
instanton correspondence, and the bridge from S^3 x R to R^4.

Test categories:
    1. Stereographic projection S^4 <-> R^4
    2. Cylinder map S^4\\{2pts} <-> S^3 x R
    3. Composition S^3 x R -> R^4
    4. Conformal factors
    5. Conformal invariance of YM in 4D
    6. Point removal (capacity argument)
    7. Uhlenbeck removable singularity
    8. Instanton correspondence
    9. Bridge theorem structure and honesty
   10. Full analysis integration
"""

import pytest
import numpy as np
from scipy import integrate

from yang_mills_s3.proofs.s4_compactification import (
    ConformalMaps,
    ConformalYM,
    PointRemoval,
    InstantonCorrespondence,
    BridgeTheorem,
    S4CompactificationAnalysis,
    ClaimStatus,
    _t_hooft_eta,
    _levi_civita_3,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def analysis():
    """S4CompactificationAnalysis with unit S^4."""
    return S4CompactificationAnalysis(R_s4=1.0)


@pytest.fixture
def analysis_R2():
    """S4CompactificationAnalysis with R=2 S^4."""
    return S4CompactificationAnalysis(R_s4=2.0)


@pytest.fixture
def rng():
    """Fixed-seed random number generator."""
    return np.random.RandomState(123)


def _random_points_on_s4(n, R=1.0, rng=None):
    """Generate n random points on S^4 of radius R."""
    if rng is None:
        rng = np.random.RandomState(42)
    raw = rng.randn(n, 5)
    norms = np.sqrt(np.sum(raw**2, axis=1, keepdims=True))
    return R * raw / norms


# ======================================================================
# 1. Stereographic projection S^4 <-> R^4
# ======================================================================

class TestStereographic:
    """Stereographic projection maps S^4\\{north pole} to R^4."""

    def test_south_pole_maps_to_origin(self):
        """South pole (X_0 = -R) maps to the origin in R^4."""
        R = 1.0
        south_pole = np.array([-R, 0, 0, 0, 0])
        y = ConformalMaps.stereographic_s4_to_r4(south_pole, R)
        assert np.allclose(y, np.zeros(4), atol=1e-14)

    def test_equator_maps_correctly(self):
        """A point on the equator (X_0=0) maps to a specific point."""
        R = 1.0
        # X_0=0, X_1=R, X_2=X_3=X_4=0 is on S^4 equator
        equator_pt = np.array([0.0, R, 0, 0, 0])
        y = ConformalMaps.stereographic_s4_to_r4(equator_pt, R)
        expected = np.array([1.0, 0, 0, 0])  # R * (R,0,0,0) / (R-0) = (1,0,0,0)
        assert np.allclose(y, expected, atol=1e-14)

    def test_north_pole_raises(self):
        """North pole cannot be projected (maps to infinity)."""
        R = 1.0
        north_pole = np.array([R, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="Cannot project the north pole"):
            ConformalMaps.stereographic_s4_to_r4(north_pole, R)

    def test_roundtrip_single_point(self):
        """Inverse(forward(X)) = X for a point on S^4."""
        R = 1.0
        X = np.array([-0.5, 0.3, 0.4, 0.5, np.sqrt(1 - 0.25 - 0.09 - 0.16 - 0.25)])
        X = R * X / np.linalg.norm(X)
        y = ConformalMaps.stereographic_s4_to_r4(X, R)
        X_back = ConformalMaps.stereographic_r4_to_s4(y, R)
        assert np.allclose(X, X_back, atol=1e-12)

    def test_roundtrip_batch(self, rng):
        """Roundtrip works for a batch of points."""
        R = 1.0
        pts = _random_points_on_s4(50, R, rng)
        # Exclude points near north pole
        pts = pts[pts[:, 0] < 0.95 * R]
        y = ConformalMaps.stereographic_s4_to_r4(pts, R)
        pts_back = ConformalMaps.stereographic_r4_to_s4(y, R)
        assert np.allclose(pts, pts_back, atol=1e-10)

    def test_inverse_maps_origin_to_south_pole(self):
        """Origin in R^4 maps to south pole of S^4."""
        R = 2.0
        y = np.zeros(4)
        X = ConformalMaps.stereographic_r4_to_s4(y, R)
        expected = np.array([-R, 0, 0, 0, 0])
        assert np.allclose(X, expected, atol=1e-14)

    def test_points_on_s4_after_inverse(self, rng):
        """Points produced by inverse stereographic lie on S^4."""
        R = 1.5
        y = rng.randn(30, 4)
        X = ConformalMaps.stereographic_r4_to_s4(y, R)
        norms = np.sqrt(np.sum(X**2, axis=1))
        assert np.allclose(norms, R, atol=1e-12)

    def test_radius_scaling(self):
        """Changing R scales the projection appropriately."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            south = np.array([-R, 0, 0, 0, 0])
            y = ConformalMaps.stereographic_s4_to_r4(south, R)
            assert np.allclose(y, np.zeros(4), atol=1e-14)


# ======================================================================
# 2. Cylinder map S^4\\{2pts} <-> S^3 x R
# ======================================================================

class TestCylinderMap:
    """Cylinder map between S^4 minus poles and S^3 x R."""

    def test_equator_maps_to_t_zero(self):
        """The equator of S^4 (X_0=0) maps to t=0 on the cylinder."""
        R = 1.0
        # equator: X_0=0, |X_rest|=R
        eq_pt = np.array([0.0, R, 0, 0, 0])
        omega, t = ConformalMaps.cylinder_map_s4_to_s3xR(eq_pt, R)
        assert abs(t) < 1e-10, f"Equator should map to t=0, got t={t}"

    def test_omega_is_unit_vector(self, rng):
        """The S^3 component omega should be a unit vector."""
        R = 1.0
        pts = _random_points_on_s4(50, R, rng)
        # Exclude poles
        pts = pts[np.abs(pts[:, 0]) < 0.95 * R]
        omega, t = ConformalMaps.cylinder_map_s4_to_s3xR(pts, R)
        norms = np.sqrt(np.sum(omega**2, axis=1))
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_roundtrip_single(self):
        """Inverse(forward(X)) = X for a point on S^4."""
        R = 1.0
        X = np.array([0.3, 0.5, 0.4, 0.3, np.sqrt(1 - 0.09 - 0.25 - 0.16 - 0.09)])
        X = R * X / np.linalg.norm(X)
        omega, t = ConformalMaps.cylinder_map_s4_to_s3xR(X, R)
        X_back = ConformalMaps.cylinder_map_s3xR_to_s4(omega, t, R)
        assert np.allclose(X, X_back, atol=1e-10)

    def test_roundtrip_batch(self, rng):
        """Roundtrip for a batch of points."""
        R = 1.0
        pts = _random_points_on_s4(80, R, rng)
        pts = pts[np.abs(pts[:, 0]) < 0.95 * R]
        omega, t = ConformalMaps.cylinder_map_s4_to_s3xR(pts, R)
        pts_back = ConformalMaps.cylinder_map_s3xR_to_s4(omega, t, R)
        assert np.allclose(pts, pts_back, atol=1e-10)

    def test_t_range(self, rng):
        """t ranges from -inf to +inf as theta goes from 0 to pi."""
        R = 1.0
        # Points near south pole (X_0 ~ -R) have large positive t
        south_ish = np.array([-0.99, 0.1, 0.0, 0.0, np.sqrt(1 - 0.9801 - 0.01)])
        south_ish = R * south_ish / np.linalg.norm(south_ish)
        _, t_south = ConformalMaps.cylinder_map_s4_to_s3xR(south_ish, R)

        # Points near north pole have large negative t
        north_ish = np.array([0.99, 0.1, 0.0, 0.0, np.sqrt(1 - 0.9801 - 0.01)])
        north_ish = R * north_ish / np.linalg.norm(north_ish)
        _, t_north = ConformalMaps.cylinder_map_s4_to_s3xR(north_ish, R)

        # South should give positive t, north should give negative t
        # Actually: our convention has cos(theta) = X_0/R
        # theta near 0 => X_0 ~ R (north) => tan(theta/2) ~ 0 => t ~ -inf
        # theta near pi => X_0 ~ -R (south) => tan(theta/2) ~ inf => t ~ +inf
        assert t_north < -1.0, f"Near north pole, t should be negative, got {t_north}"
        assert t_south > 1.0, f"Near south pole, t should be positive, got {t_south}"

    def test_inverse_produces_s4_points(self):
        """Points from the inverse map lie on S^4."""
        R = 2.0
        omega = np.array([1.0, 0, 0, 0])
        t_values = np.linspace(-3.0, 3.0, 20)
        for t in t_values:
            X = ConformalMaps.cylinder_map_s3xR_to_s4(omega, t, R)
            norm = np.linalg.norm(X)
            assert abs(norm - R) < 1e-10, f"Point should be on S^4(R={R}), norm={norm}"


# ======================================================================
# 3. Composition S^3 x R -> R^4
# ======================================================================

class TestComposition:
    """The composed map from S^3 x R to R^4."""

    def test_equator_maps_to_finite_point(self):
        """The equator (t=0) on the cylinder maps to a finite point in R^4."""
        R = 1.0
        omega = np.array([1.0, 0, 0, 0])
        t = 0.0
        y = ConformalMaps.s3xR_to_r4(omega, t, R)
        assert np.all(np.isfinite(y))

    def test_consistency_with_two_step(self, rng):
        """Direct composition equals two-step (cylinder + stereo)."""
        R = 1.0
        omega = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        t = np.array([-1.0, 0.0, 0.5, 1.5])

        y_direct = ConformalMaps.s3xR_to_r4(omega, t, R)
        X_s4 = ConformalMaps.cylinder_map_s3xR_to_s4(omega, t, R)
        y_two_step = ConformalMaps.stereographic_s4_to_r4(X_s4, R)

        assert np.allclose(y_direct, y_two_step, atol=1e-10)


# ======================================================================
# 4. Conformal factors
# ======================================================================

class TestConformalFactors:
    """Conformal factors for the various maps."""

    def test_stereo_factor_at_south_pole(self):
        """At south pole (X_0=-R), Omega^2 = (R-(-R))^2/(4R^2) = 1."""
        R = 1.0
        south = np.array([-R, 0, 0, 0, 0])
        omega_sq = ConformalMaps.stereographic_conformal_factor(south, R)
        assert abs(omega_sq - 1.0) < 1e-14

    def test_stereo_factor_at_equator(self):
        """At equator (X_0=0), Omega^2 = R^2/(4R^2) = 1/4."""
        R = 1.0
        equator = np.array([0.0, R, 0, 0, 0])
        omega_sq = ConformalMaps.stereographic_conformal_factor(equator, R)
        assert abs(omega_sq - 0.25) < 1e-14

    def test_cylinder_factor_at_t_zero(self):
        """At t=0 (equator), Omega^2 = sech^2(0) = 1."""
        R = 1.0
        omega_sq = ConformalMaps.cylinder_conformal_factor(0.0, R)
        assert abs(omega_sq - 1.0) < 1e-14

    def test_cylinder_factor_symmetric(self):
        """Conformal factor is symmetric: Omega(t) = Omega(-t)."""
        R = 1.0
        for t in [0.5, 1.0, 2.0, 5.0]:
            f_pos = ConformalMaps.cylinder_conformal_factor(t, R)
            f_neg = ConformalMaps.cylinder_conformal_factor(-t, R)
            assert abs(f_pos - f_neg) < 1e-14

    def test_cylinder_factor_decays_to_zero(self):
        """Omega^2 -> 0 as t -> +-inf (poles shrink to points)."""
        R = 1.0
        for t in [10.0, 20.0, 50.0]:
            f = ConformalMaps.cylinder_conformal_factor(t, R)
            assert f < 1e-4, f"At t={t}, Omega^2 should be small, got {f}"

    def test_cylinder_factor_positive(self):
        """Conformal factor is always positive."""
        R = 1.0
        t_vals = np.linspace(-10, 10, 100)
        factors = ConformalMaps.cylinder_conformal_factor(t_vals, R)
        assert np.all(factors > 0)

    def test_total_conformal_factor_finite(self):
        """Total conformal factor S^3xR -> R^4 is finite at generic points."""
        R = 1.0
        t_vals = np.linspace(-5, 5, 50)
        factors = ConformalMaps.s3xR_to_r4_conformal_factor(t_vals, R)
        assert np.all(np.isfinite(factors))
        assert np.all(factors > 0)


# ======================================================================
# 5. Conformal invariance of YM in 4D
# ======================================================================

class TestConformalYM:
    """Yang-Mills action is conformally invariant only in 4D."""

    def test_weight_zero_in_4d(self):
        """Conformal weight of |F|^2 dvol is 0 in 4D."""
        weight = ConformalYM.ym_action_conformal_weight(4)
        assert weight == 0

    def test_weight_nonzero_in_other_dims(self):
        """Conformal weight is non-zero in dim != 4."""
        for dim in [2, 3, 5, 6, 7, 8]:
            weight = ConformalYM.ym_action_conformal_weight(dim)
            assert weight != 0, f"Weight should be non-zero in dim {dim}"

    def test_is_invariant_only_4d(self):
        """is_conformally_invariant returns True only for dim=4."""
        for dim in range(1, 8):
            result = ConformalYM.is_conformally_invariant(dim)
            if dim == 4:
                assert result is True
            else:
                assert result is False

    def test_action_ratio_is_1_in_4d(self):
        """Under conformal change, S_YM is unchanged in 4D."""
        for omega_sq in [0.1, 0.5, 1.0, 2.0, 10.0]:
            ratio = ConformalYM.action_ratio_under_conformal(omega_sq, dim=4)
            assert abs(ratio - 1.0) < 1e-14, \
                f"Action ratio should be 1 in 4D, got {ratio} for Omega^2={omega_sq}"

    def test_action_ratio_not_1_in_3d(self):
        """In 3D, conformal change DOES affect the action."""
        ratio = ConformalYM.action_ratio_under_conformal(2.0, dim=3)
        assert abs(ratio - 1.0) > 0.1, "Action should change in 3D"

    def test_hodge_star_invariant_2forms_4d(self):
        """Hodge star on 2-forms in 4D has weight 0 (conformally invariant)."""
        weight = ConformalYM.hodge_star_conformal_weight(4, 2)
        assert weight == 0

    def test_hodge_star_weight_general(self):
        """Hodge star weight is n - 2p."""
        assert ConformalYM.hodge_star_conformal_weight(3, 1) == 1  # 3 - 2
        assert ConformalYM.hodge_star_conformal_weight(4, 2) == 0  # 4 - 4
        assert ConformalYM.hodge_star_conformal_weight(6, 3) == 0  # 6 - 6


# ======================================================================
# 6. Point removal (capacity argument)
# ======================================================================

class TestPointRemoval:
    """Removing a point in dim >= 3 does not affect W^{1,2}."""

    def test_capacity_decreases_with_epsilon(self):
        """Capacity of B_eps decreases as eps -> 0 in dim >= 3."""
        for dim in [3, 4, 5]:
            caps = [PointRemoval.capacity_of_point(dim, eps) for eps in [1e-2, 1e-4, 1e-6, 1e-8]]
            for i in range(len(caps) - 1):
                assert caps[i] > caps[i + 1], \
                    f"Capacity should decrease in dim {dim}: {caps}"

    def test_capacity_zero_in_dim4(self):
        """At very small epsilon, capacity is very small in dim 4."""
        cap = PointRemoval.capacity_of_point(4, epsilon=1e-10)
        assert cap < 1e-10, f"Capacity should be negligible in dim 4, got {cap}"

    def test_capacity_vanishes_dim_ge_2(self):
        """Capacity vanishes for dim >= 2."""
        assert PointRemoval.capacity_vanishes(1) is False
        for dim in [2, 3, 4, 5, 6]:
            assert PointRemoval.capacity_vanishes(dim) is True

    def test_sobolev_unchanged_dim_ge_3(self):
        """W^{1,2} unchanged by point removal in dim >= 3."""
        assert PointRemoval.sobolev_space_unchanged(1) is False
        assert PointRemoval.sobolev_space_unchanged(2) is False
        for dim in [3, 4, 5, 6, 7]:
            assert PointRemoval.sobolev_space_unchanged(dim) is True

    def test_capacity_power_law_dim4(self):
        """In dim 4, cap(B_eps) ~ eps^2."""
        eps1 = 1e-3
        eps2 = 1e-6
        cap1 = PointRemoval.capacity_of_point(4, eps1)
        cap2 = PointRemoval.capacity_of_point(4, eps2)
        # Ratio should be (eps1/eps2)^2 = 10^6
        ratio = cap1 / cap2
        expected_ratio = (eps1 / eps2)**(4 - 2)  # eps^{n-2}
        assert abs(ratio / expected_ratio - 1.0) < 0.01, \
            f"Capacity should scale as eps^2 in dim 4, ratio={ratio}, expected={expected_ratio}"

    def test_capacity_log_in_dim2(self):
        """In dim 2, capacity decays logarithmically."""
        cap1 = PointRemoval.capacity_of_point(2, epsilon=1e-3)
        cap2 = PointRemoval.capacity_of_point(2, epsilon=1e-6)
        # Both should be positive but small
        assert cap1 > 0
        assert cap2 > 0
        assert cap2 < cap1  # smaller eps => smaller cap

    def test_invalid_dim_raises(self):
        """Dimension < 1 raises ValueError."""
        with pytest.raises(ValueError):
            PointRemoval.capacity_of_point(0)

    def test_capacity_positive_in_dim1(self):
        """In dim 1, a point has positive capacity (disconnects the line)."""
        cap = PointRemoval.capacity_of_point(1)
        assert cap > 0


# ======================================================================
# 7. Uhlenbeck removable singularity
# ======================================================================

class TestUhlenbeck:
    """Uhlenbeck's removable singularity theorem."""

    def test_theorem_in_dim4(self):
        """In dim 4, it's a THEOREM."""
        claim = PointRemoval.uhlenbeck_removable_singularity(4)
        assert claim.label == 'THEOREM'

    def test_proposition_in_other_dims(self):
        """In other dims, it's a PROPOSITION (partial results)."""
        for dim in [3, 5, 6]:
            claim = PointRemoval.uhlenbeck_removable_singularity(dim)
            assert claim.label == 'PROPOSITION'

    def test_spectrum_unchanged_dim4(self):
        """In dim 4, spectrum unchanged by point removal is THEOREM."""
        claim = PointRemoval.spectrum_unchanged_by_point_removal(4)
        assert claim.label == 'THEOREM'

    def test_spectrum_may_change_dim2(self):
        """In dim 2, spectrum CAN change — not a theorem."""
        claim = PointRemoval.spectrum_unchanged_by_point_removal(2)
        assert claim.label != 'THEOREM'  # Should be lower status


# ======================================================================
# 8. Instanton correspondence
# ======================================================================

class TestInstantonCorrespondence:
    """BPST on R^4 corresponds to Hopf maps on S^3."""

    def test_action_scale_invariant(self):
        """BPST action = 8*pi^2, independent of scale rho."""
        expected = 8.0 * np.pi**2
        for rho in [0.01, 0.1, 1.0, 10.0, 100.0]:
            S = InstantonCorrespondence.bpst_action_integral(rho)
            assert abs(S - expected) < 1e-10

    def test_charge_equals_degree(self):
        """Instanton charge = Hopf degree."""
        for k in range(-5, 6):
            assert InstantonCorrespondence.instanton_charge_from_hopf_degree(k) == k

    def test_instanton_action_formula(self):
        """S = 8*pi^2 * |k| / g^2."""
        g = 1.0
        for k in [1, 2, -1, -3]:
            S = InstantonCorrespondence.instanton_action(k, g)
            expected = 8.0 * np.pi**2 * abs(k) / g**2
            assert abs(S - expected) < 1e-10

    def test_action_negative_coupling_raises(self):
        """Negative coupling raises ValueError."""
        with pytest.raises(ValueError):
            InstantonCorrespondence.instanton_action(1, -1.0)

    def test_field_strength_peaked_at_origin(self):
        """BPST |F|^2 is maximal at the origin."""
        rho = 1.0
        f_origin = InstantonCorrespondence.bpst_field_strength_sq(np.zeros(4), rho)
        f_far = InstantonCorrespondence.bpst_field_strength_sq(np.array([10.0, 0, 0, 0]), rho)
        assert f_origin > f_far

    def test_field_strength_at_origin(self):
        """At origin: F^a_mn F^a_mn = 192/rho^4."""
        for rho in [0.5, 1.0, 2.0]:
            f = InstantonCorrespondence.bpst_field_strength_sq(np.zeros(4), rho)
            expected = 192.0 / rho**4
            assert abs(f - expected) < 1e-10

    def test_bpst_action_numerical_integration(self):
        """Numerically integrate (1/4)|F|^2 over R^4 to get 8*pi^2."""
        rho = 1.0

        def integrand(r):
            # S_YM = (1/4) * F^a_mn F^a_mn * r^3 * 2*pi^2
            f_sq = 192.0 * rho**4 / (r**2 + rho**2)**4
            return 0.25 * f_sq * r**3 * 2.0 * np.pi**2

        result, error = integrate.quad(integrand, 0, np.inf)
        expected = 8.0 * np.pi**2
        assert abs(result - expected) / expected < 1e-8, \
            f"Numerical action = {result}, expected {expected}"

    def test_bpst_gauge_potential_shape(self):
        """BPST gauge potential has shape (4, 3)."""
        x = np.array([0.5, 0.3, 0.1, 0.2])
        A = InstantonCorrespondence.bpst_instanton_r4(x, rho=1.0)
        assert A.shape == (4, 3)

    def test_bpst_gauge_potential_at_origin(self):
        """At the origin, A = 0 (by symmetry, all x_nu = 0)."""
        A = InstantonCorrespondence.bpst_instanton_r4(np.zeros(4), rho=1.0)
        assert np.allclose(A, 0.0, atol=1e-14)


# ======================================================================
# 9. Bridge theorem structure and honesty
# ======================================================================

class TestBridgeTheorem:
    """The bridge theorem connecting S^3 x R to R^4."""

    def test_chain_has_seven_steps(self):
        """The chain of equivalences has 7 steps."""
        chain = BridgeTheorem.chain_of_equivalences()
        assert len(chain) == 7

    def test_chain_starts_with_theorems(self):
        """First five steps are THEOREM."""
        chain = BridgeTheorem.chain_of_equivalences()
        for i in range(5):
            assert chain[i].label == 'THEOREM', \
                f"Step {i} should be THEOREM, got {chain[i].label}"

    def test_bridge_is_proposition(self):
        """Step 6 (the bridge) is PROPOSITION, not THEOREM."""
        chain = BridgeTheorem.chain_of_equivalences()
        assert chain[5].label == 'PROPOSITION'

    def test_full_persistence_is_conjecture(self):
        """Step 7 (full non-perturbative persistence) is CONJECTURE."""
        chain = BridgeTheorem.chain_of_equivalences()
        assert chain[6].label == 'CONJECTURE'

    def test_all_claims_have_evidence(self):
        """Every claim in the chain has non-empty evidence."""
        chain = BridgeTheorem.chain_of_equivalences()
        for claim in chain:
            assert len(claim.evidence) > 0

    def test_all_claims_have_caveats(self):
        """Every claim has caveats (honesty)."""
        chain = BridgeTheorem.chain_of_equivalences()
        for claim in chain:
            assert len(claim.caveats) > 0

    def test_one_point_gap_result(self):
        """The one-point-gap analysis returns correct structure."""
        result = BridgeTheorem.the_one_point_gap()
        assert result['capacity_of_point_in_4d'] == 0.0
        assert result['sobolev_unchanged'] is True
        assert result['uhlenbeck_applies'] is True
        assert result['status'].label == 'PROPOSITION'

    def test_honest_assessment_categories(self):
        """Honest assessment has proven, propositions, and conjectures."""
        assessment = BridgeTheorem.honest_assessment()
        assert len(assessment['proven']) > 0
        assert len(assessment['propositions']) > 0
        assert len(assessment['conjectures']) > 0

    def test_honest_assessment_proven_are_theorems(self):
        """All 'proven' claims are labeled THEOREM."""
        assessment = BridgeTheorem.honest_assessment()
        for claim in assessment['proven']:
            assert claim.label == 'THEOREM'

    def test_honest_assessment_propositions_labeled(self):
        """All 'propositions' are labeled PROPOSITION."""
        assessment = BridgeTheorem.honest_assessment()
        for claim in assessment['propositions']:
            assert claim.label == 'PROPOSITION'

    def test_honest_assessment_conjectures_labeled(self):
        """All 'conjectures' are labeled CONJECTURE."""
        assessment = BridgeTheorem.honest_assessment()
        for claim in assessment['conjectures']:
            assert claim.label == 'CONJECTURE'

    def test_honest_assessment_has_summary(self):
        """Assessment includes a summary."""
        assessment = BridgeTheorem.honest_assessment()
        assert 'summary' in assessment
        assert len(assessment['summary']) > 0

    def test_honest_assessment_mentions_quantum_measure(self):
        """Assessment honestly mentions the quantum measure issue."""
        assessment = BridgeTheorem.honest_assessment()
        remaining = assessment['what_remains']
        assert 'measure' in remaining.lower() or 'quantum' in remaining.lower()


# ======================================================================
# 10. Full analysis integration
# ======================================================================

class TestFullAnalysis:
    """Integration tests for the complete analysis."""

    def test_conformal_maps_verification(self, analysis):
        """Conformal maps pass numerical verification."""
        result = analysis.verify_conformal_maps()
        assert result['all_passed'], \
            f"Conformal maps failed: stereo_err={result['stereo_roundtrip_error']}, " \
            f"cyl_err={result['cylinder_roundtrip_error']}"

    def test_ym_invariance_only_4d(self, analysis):
        """YM is invariant only in 4D."""
        result = analysis.verify_ym_conformal_invariance()
        assert result['only_4d_invariant']

    def test_point_removal_dim4(self, analysis):
        """Dim 4 is safe for point removal."""
        result = analysis.verify_point_removal()
        assert result['dim_4_safe']

    def test_instanton_scale_independence(self, analysis):
        """Instanton action is scale-independent."""
        result = analysis.verify_instanton_correspondence()
        assert result['action_scale_independence']

    def test_instanton_charge_degree(self, analysis):
        """Charge = degree for instantons."""
        result = analysis.verify_instanton_correspondence()
        assert result['charge_equals_degree']

    def test_instanton_action_numerical(self, analysis):
        """Numerical integration of BPST action matches 8*pi^2."""
        result = analysis.verify_instanton_correspondence()
        assert result['action_numerical_error'] < 1e-6

    def test_full_analysis_runs(self, analysis):
        """Full analysis completes without error."""
        result = analysis.full_analysis()
        assert 'conformal_maps' in result
        assert 'ym_invariance' in result
        assert 'point_removal' in result
        assert 'instanton_correspondence' in result
        assert 'bridge_theorem' in result
        assert 'one_point_gap' in result
        assert 'honest_assessment' in result

    def test_full_analysis_with_different_radius(self, analysis_R2):
        """Analysis works with non-unit radius."""
        result = analysis_R2.verify_conformal_maps()
        assert result['all_passed']


# ======================================================================
# 11. Helper functions
# ======================================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_levi_civita_even_permutations(self):
        """Even permutations of (0,1,2) give +1."""
        assert _levi_civita_3(0, 1, 2) == 1.0
        assert _levi_civita_3(1, 2, 0) == 1.0
        assert _levi_civita_3(2, 0, 1) == 1.0

    def test_levi_civita_odd_permutations(self):
        """Odd permutations of (0,1,2) give -1."""
        assert _levi_civita_3(0, 2, 1) == -1.0
        assert _levi_civita_3(2, 1, 0) == -1.0
        assert _levi_civita_3(1, 0, 2) == -1.0

    def test_levi_civita_repeated_index(self):
        """Repeated indices give 0."""
        assert _levi_civita_3(0, 0, 1) == 0.0
        assert _levi_civita_3(1, 1, 1) == 0.0

    def test_t_hooft_eta_antisymmetric(self):
        """eta^a_{mu nu} = -eta^a_{nu mu}."""
        for a in range(3):
            for mu in range(4):
                for nu in range(4):
                    assert abs(_t_hooft_eta(a, mu, nu) + _t_hooft_eta(a, nu, mu)) < 1e-14

    def test_t_hooft_eta_diagonal_zero(self):
        """eta^a_{mu mu} = 0."""
        for a in range(3):
            for mu in range(4):
                assert _t_hooft_eta(a, mu, mu) == 0.0

    def test_t_hooft_eta_specific_values(self):
        """Check specific known values of the 't Hooft symbol."""
        # eta^0_{01} = delta_{0,0} = 1
        assert _t_hooft_eta(0, 0, 1) == 1.0
        # eta^1_{02} = delta_{1,1} = 1
        assert _t_hooft_eta(1, 0, 2) == 1.0
        # eta^2_{03} = delta_{2,2} = 1
        assert _t_hooft_eta(2, 0, 3) == 1.0
        # eta^0_{02} = delta_{0,1} = 0
        assert _t_hooft_eta(0, 0, 2) == 0.0

    def test_claim_status_repr(self):
        """ClaimStatus has a readable repr."""
        claim = ClaimStatus(
            label='THEOREM',
            statement='Test',
            evidence='Some evidence',
            caveats='Some caveats',
        )
        s = repr(claim)
        assert 'THEOREM' in s
        assert 'Test' in s
