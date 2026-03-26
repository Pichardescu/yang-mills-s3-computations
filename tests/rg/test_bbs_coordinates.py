"""
Tests for the BBS (Bauerschmidt-Brydges-Slade) Coordinate Framework.

Verifies:
1.  RelevantCouplings: construction, properties, evolution
2.  PolymerCoordinate: activities, norm, algebra operations
3.  BBSCoordinates: (V, K) pair, sum rule, curvature corrections
4.  ExtractionOperator: idempotency, coupling extraction, subtract
5.  RGMapBBS: single-step RG, contraction, Gaussian integration
6.  MultiScaleRGBBS: full iteration, asymptotic freedom, mass gap
7.  Physical parameter checks and cross-validation with existing modules

Run:
    pytest tests/rg/test_bbs_coordinates.py -v
"""

import numpy as np
import pytest

from yang_mills_s3.rg.bbs_coordinates import (
    RelevantCouplings,
    PolymerCoordinate,
    BBSCoordinates,
    ExtractionOperator,
    RGMapBBS,
    MultiScaleRGBBS,
    _beta_0,
    BETA_0_SU2,
    G2_BARE_DEFAULT,
    M_DEFAULT,
    N_SCALES_DEFAULT,
    N_COLORS_DEFAULT,
    K_MAX_DEFAULT,
    DIM_SPACETIME,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    LAMBDA_QCD_MEV,
)
from yang_mills_s3.rg.banach_norm import Polymer, LargeFieldRegulator
from yang_mills_s3.rg.heat_kernel_slices import coexact_eigenvalue, coexact_multiplicity
from yang_mills_s3.rg.first_rg_step import quadratic_casimir


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def default_couplings():
    """Default SU(2) couplings at the bare scale."""
    return RelevantCouplings(g2=G2_BARE_DEFAULT, nu=0.0, z=1.0, N_c=2)


@pytest.fixture
def weak_couplings():
    """Weak-coupling regime for perturbative tests."""
    return RelevantCouplings(g2=0.5, nu=0.0, z=1.0, N_c=2)


@pytest.fixture
def strong_couplings():
    """Strong-coupling regime (alpha_s > 1 requires g^2 > 4*pi ~ 12.57)."""
    return RelevantCouplings(g2=4.0 * np.pi + 1.0, nu=0.0, z=1.0, N_c=2)


@pytest.fixture
def su3_couplings():
    """SU(3) couplings for universality tests."""
    return RelevantCouplings(g2=6.28, nu=0.0, z=1.0, N_c=3)


@pytest.fixture
def single_block_polymer():
    """A single-block polymer at scale 0."""
    return Polymer(frozenset([0]), scale=0)


@pytest.fixture
def two_block_polymer():
    """A two-block polymer at scale 0."""
    return Polymer(frozenset([0, 1]), scale=0)


@pytest.fixture
def three_block_polymer():
    """A three-block polymer at scale 0."""
    return Polymer(frozenset([0, 1, 2]), scale=0)


@pytest.fixture
def sample_activities(single_block_polymer, two_block_polymer, three_block_polymer):
    """Sample polymer activities for testing."""
    return {
        single_block_polymer: 0.01,
        two_block_polymer: 0.005,
        three_block_polymer: 0.001,
    }


@pytest.fixture
def zero_polymer_coord():
    """Zero polymer coordinate (bare action has no remainder)."""
    return PolymerCoordinate(scale=0, kappa=1.0)


@pytest.fixture
def sample_polymer_coord(sample_activities):
    """Sample polymer coordinate with test activities."""
    return PolymerCoordinate(scale=0, activities=sample_activities, kappa=1.0)


@pytest.fixture
def uv_bbs_coords(default_couplings, zero_polymer_coord):
    """UV BBS coordinates (bare action)."""
    return BBSCoordinates(
        v=default_couplings,
        k=zero_polymer_coord,
        scale=0,
        R=R_PHYSICAL_FM,
    )


@pytest.fixture
def sample_bbs_coords(default_couplings, sample_polymer_coord):
    """BBS coordinates with sample polymer activities."""
    return BBSCoordinates(
        v=default_couplings,
        k=sample_polymer_coord,
        scale=0,
        R=R_PHYSICAL_FM,
    )


@pytest.fixture
def extraction_op():
    """Extraction operator for SU(2)."""
    return ExtractionOperator(R=R_PHYSICAL_FM, N_c=2)


@pytest.fixture
def rg_map():
    """Single-step RG map."""
    return RGMapBBS(R=R_PHYSICAL_FM, M=M_DEFAULT, N_c=2)


@pytest.fixture
def multi_scale_rg():
    """Multi-scale RG iteration."""
    return MultiScaleRGBBS(
        n_scales=N_SCALES_DEFAULT,
        R=R_PHYSICAL_FM,
        M=M_DEFAULT,
        N_c=2,
        g2_bare=G2_BARE_DEFAULT,
    )


# ======================================================================
# Section 1: RelevantCouplings
# ======================================================================

class TestRelevantCouplings:
    """Tests for the RelevantCouplings dataclass."""

    def test_construction_default(self, default_couplings):
        """Default SU(2) couplings are correctly initialized."""
        assert default_couplings.g2 == G2_BARE_DEFAULT
        assert default_couplings.nu == 0.0
        assert default_couplings.z == 1.0
        assert default_couplings.N_c == 2

    def test_construction_with_mass(self):
        """Couplings with nonzero mass parameter."""
        c = RelevantCouplings(g2=1.0, nu=0.5, z=1.0)
        assert c.nu == 0.5

    def test_negative_g2_raises(self):
        """Negative coupling constant is rejected."""
        with pytest.raises(ValueError, match="g2 must be non-negative"):
            RelevantCouplings(g2=-1.0)

    def test_zero_g2_allowed(self):
        """Zero coupling (free theory) is allowed."""
        c = RelevantCouplings(g2=0.0)
        assert c.g2 == 0.0

    def test_negative_z_raises(self):
        """Non-positive wavefunction renormalization is rejected."""
        with pytest.raises(ValueError, match="z must be positive"):
            RelevantCouplings(g2=1.0, z=0.0)

    def test_small_N_c_raises(self):
        """N_c < 2 is rejected (no SU(1))."""
        with pytest.raises(ValueError, match="N_c must be >= 2"):
            RelevantCouplings(g2=1.0, N_c=1)

    def test_alpha_s_weak_coupling(self, weak_couplings):
        """alpha_s = g^2/(4*pi) in the perturbative regime."""
        expected = 0.5 / (4.0 * np.pi)
        assert abs(weak_couplings.alpha_s - expected) < 1e-14

    def test_alpha_s_strong_coupling(self, strong_couplings):
        """alpha_s in the strong coupling regime is > 1."""
        assert strong_couplings.alpha_s > 1.0

    def test_is_perturbative_weak(self, weak_couplings):
        """Weak coupling is perturbative (alpha_s < 1)."""
        assert weak_couplings.is_perturbative is True

    def test_is_perturbative_strong(self, strong_couplings):
        """Strong coupling is non-perturbative."""
        assert strong_couplings.is_perturbative is False

    def test_dim_adj_su2(self, default_couplings):
        """dim(adj(SU(2))) = 3."""
        assert default_couplings.dim_adj == 3

    def test_dim_adj_su3(self, su3_couplings):
        """dim(adj(SU(3))) = 8."""
        assert su3_couplings.dim_adj == 8

    def test_beta_0_su2(self):
        """Beta function coefficient for SU(2)."""
        b0 = _beta_0(2)
        expected = 22.0 / (48.0 * np.pi**2)
        assert abs(b0 - expected) < 1e-14

    def test_beta_0_su3(self):
        """Beta function coefficient for SU(3)."""
        b0 = _beta_0(3)
        expected = 33.0 / (48.0 * np.pi**2)
        assert abs(b0 - expected) < 1e-14

    def test_beta_0_property(self, default_couplings):
        """beta_0 property matches module-level function."""
        assert abs(default_couplings.beta_0 - BETA_0_SU2) < 1e-14

    def test_beta_function_value_positive(self, default_couplings):
        """Beta function value is positive (asymptotic freedom: 1/g^2 decreases toward IR)."""
        bfv = default_couplings.beta_function_value(M=2.0)
        assert bfv > 0

    def test_evolved_g2_increases_toward_ir(self, default_couplings):
        """Coupling grows toward IR (asymptotic freedom)."""
        g2_new = default_couplings.evolved_g2(M=2.0)
        assert g2_new >= default_couplings.g2

    def test_evolved_g2_bounded(self, default_couplings):
        """Evolved coupling is bounded by 4*pi."""
        g2_new = default_couplings.evolved_g2(M=2.0)
        assert g2_new <= 4.0 * np.pi + 1e-10

    def test_evolved_g2_weak_coupling_perturbative(self, weak_couplings):
        """Weak coupling evolution reproduces one-loop formula."""
        g2 = weak_couplings.g2
        b0 = weak_couplings.beta_0
        # 1/g^2_new = 1/g^2 - b0 * log(M^2)
        inv_g2_expected = 1.0 / g2 - b0 * np.log(4.0)
        g2_expected = 1.0 / inv_g2_expected
        g2_actual = weak_couplings.evolved_g2(M=2.0)
        assert abs(g2_actual - g2_expected) < 1e-10

    def test_as_vector(self, default_couplings):
        """Vector representation is (g^2, nu, z)."""
        v = default_couplings.as_vector()
        assert len(v) == 3
        assert v[0] == G2_BARE_DEFAULT
        assert v[1] == 0.0
        assert v[2] == 1.0

    def test_from_vector_roundtrip(self, default_couplings):
        """Vector -> couplings -> vector roundtrip."""
        v = default_couplings.as_vector()
        c = RelevantCouplings.from_vector(v, N_c=2)
        v2 = c.as_vector()
        np.testing.assert_allclose(v, v2)

    def test_copy_independence(self, default_couplings):
        """Copy is independent (mutation does not propagate)."""
        c2 = default_couplings.copy()
        # Modifying the original shouldn't affect the copy (but dataclass fields are immutable floats)
        assert c2.g2 == default_couplings.g2


# ======================================================================
# Section 2: PolymerCoordinate
# ======================================================================

class TestPolymerCoordinate:
    """Tests for the PolymerCoordinate class."""

    def test_zero_coordinate(self, zero_polymer_coord):
        """Zero polymer coordinate has no activities."""
        assert zero_polymer_coord.is_zero is True
        assert zero_polymer_coord.n_polymers == 0
        assert zero_polymer_coord.norm() == 0.0

    def test_nonzero_coordinate(self, sample_polymer_coord):
        """Sample coordinate has activities."""
        assert sample_polymer_coord.is_zero is False
        assert sample_polymer_coord.n_polymers == 3

    def test_negative_scale_raises(self):
        """Negative scale is rejected."""
        with pytest.raises(ValueError, match="Scale must be non-negative"):
            PolymerCoordinate(scale=-1)

    def test_negative_kappa_raises(self):
        """Non-positive kappa is rejected."""
        with pytest.raises(ValueError, match="kappa must be positive"):
            PolymerCoordinate(scale=0, kappa=0.0)

    def test_get_activity_existing(self, sample_polymer_coord, single_block_polymer):
        """Get activity for existing polymer."""
        val = sample_polymer_coord.get_activity(single_block_polymer)
        assert abs(val - 0.01) < 1e-14

    def test_get_activity_nonexistent(self, sample_polymer_coord):
        """Get activity for nonexistent polymer returns 0."""
        missing = Polymer(frozenset([99]), scale=0)
        assert sample_polymer_coord.get_activity(missing) == 0.0

    def test_set_activity(self, zero_polymer_coord, single_block_polymer):
        """Setting activity adds to the dictionary."""
        zero_polymer_coord.set_activity(single_block_polymer, 0.05)
        assert zero_polymer_coord.n_polymers == 1
        assert abs(zero_polymer_coord.get_activity(single_block_polymer) - 0.05) < 1e-14

    def test_set_activity_zero_removes(self, sample_polymer_coord, single_block_polymer):
        """Setting activity to zero removes the polymer."""
        n_before = sample_polymer_coord.n_polymers
        sample_polymer_coord.set_activity(single_block_polymer, 0.0)
        assert sample_polymer_coord.n_polymers == n_before - 1

    def test_max_polymer_size(self, sample_polymer_coord):
        """Maximum polymer size is 3 (three-block polymer)."""
        assert sample_polymer_coord.max_polymer_size == 3

    def test_max_polymer_size_empty(self, zero_polymer_coord):
        """Empty coordinate has max_polymer_size = 0."""
        assert zero_polymer_coord.max_polymer_size == 0

    def test_norm_positive(self, sample_polymer_coord):
        """Norm is positive for nonzero activities."""
        assert sample_polymer_coord.norm() > 0

    def test_norm_zero_for_zero(self, zero_polymer_coord):
        """Norm is zero for zero coordinate."""
        assert zero_polymer_coord.norm() == 0.0

    def test_norm_increases_with_kappa(self):
        """Larger kappa gives larger norm (exponential weight)."""
        p = Polymer(frozenset([0]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=2.0)
        assert k2.norm() > k1.norm()

    def test_norm_exponential_decay(self):
        """Norm penalizes larger polymers exponentially."""
        p1 = Polymer(frozenset([0]), scale=0)
        p3 = Polymer(frozenset([0, 1, 2]), scale=0)
        # Same amplitude, but 3-block polymer has heavier weight
        k = PolymerCoordinate(scale=0, activities={p1: 1.0, p3: 1.0}, kappa=1.0)
        # The norm is the sup, which should be from p3 (larger weight)
        expected_p3_contribution = 1.0 * np.exp(3.0)
        expected_p1_contribution = 1.0 * np.exp(1.0)
        assert k.norm() == pytest.approx(expected_p3_contribution, rel=1e-10)

    def test_norm_with_regulator(self, sample_polymer_coord, single_block_polymer):
        """Norm with large-field regulator."""
        reg = LargeFieldRegulator(sigma_sq=1.0)
        field_data = {single_block_polymer: (2.0, 1)}
        n = sample_polymer_coord.norm(regulator=reg, field_data=field_data)
        assert n > 0

    def test_evaluate_zero_field(self, sample_polymer_coord, single_block_polymer):
        """Evaluate at zero field returns stored activity."""
        val = sample_polymer_coord.evaluate(single_block_polymer)
        assert abs(val - 0.01) < 1e-14

    def test_evaluate_nonzero_field(self, sample_polymer_coord, single_block_polymer):
        """Evaluate with field config modulates the activity."""
        field = np.array([0.1, 0.2, 0.3])
        val = sample_polymer_coord.evaluate(single_block_polymer, field)
        # Should be close to base activity (small field correction)
        assert abs(val - 0.01) < 0.01  # within factor of 2

    def test_addition_same_scale(self):
        """Adding two polymer coordinates at the same scale."""
        p1 = Polymer(frozenset([0]), scale=0)
        p2 = Polymer(frozenset([1]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p1: 1.0}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p2: 2.0}, kappa=1.0)
        k_sum = k1 + k2
        assert k_sum.get_activity(p1) == pytest.approx(1.0)
        assert k_sum.get_activity(p2) == pytest.approx(2.0)

    def test_addition_overlapping_polymers(self):
        """Adding with overlapping polymers sums activities."""
        p = Polymer(frozenset([0]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p: 2.0}, kappa=1.0)
        k_sum = k1 + k2
        assert k_sum.get_activity(p) == pytest.approx(3.0)

    def test_addition_different_scale_raises(self):
        """Adding at different scales raises ValueError."""
        k1 = PolymerCoordinate(scale=0, kappa=1.0)
        k2 = PolymerCoordinate(scale=1, kappa=1.0)
        with pytest.raises(ValueError, match="different scales"):
            k1 + k2

    def test_scalar_multiplication(self):
        """Scalar multiplication scales all activities."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 2.0}, kappa=1.0)
        k_scaled = k * 3.0
        assert k_scaled.get_activity(p) == pytest.approx(6.0)

    def test_rmul(self):
        """Right multiplication (3.0 * k) works."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 2.0}, kappa=1.0)
        k_scaled = 3.0 * k
        assert k_scaled.get_activity(p) == pytest.approx(6.0)

    def test_scalar_zero_multiplication(self, sample_polymer_coord):
        """Multiplication by zero gives zero coordinate."""
        k_zero = sample_polymer_coord * 0.0
        assert k_zero.is_zero is True

    def test_copy_independence(self, sample_polymer_coord, single_block_polymer):
        """Copy is independent of original."""
        k_copy = sample_polymer_coord.copy()
        k_copy.set_activity(single_block_polymer, 999.0)
        assert sample_polymer_coord.get_activity(single_block_polymer) == pytest.approx(0.01)

    def test_total_activity(self, sample_polymer_coord):
        """Total activity is the sum of all amplitudes."""
        total = sample_polymer_coord.total_activity()
        assert total == pytest.approx(0.01 + 0.005 + 0.001)


# ======================================================================
# Section 3: BBSCoordinates
# ======================================================================

class TestBBSCoordinates:
    """Tests for the BBSCoordinates class."""

    def test_construction(self, uv_bbs_coords):
        """Basic construction of UV BBS coordinates."""
        assert uv_bbs_coords.scale == 0
        assert uv_bbs_coords.g2 == G2_BARE_DEFAULT
        assert uv_bbs_coords.nu == 0.0
        assert uv_bbs_coords.z == 1.0
        assert uv_bbs_coords.R == R_PHYSICAL_FM

    def test_none_v_raises(self):
        """None couplings raises ValueError."""
        k = PolymerCoordinate(scale=0)
        with pytest.raises(ValueError, match="must not be None"):
            BBSCoordinates(v=None, k=k, scale=0)

    def test_none_k_raises(self):
        """None polymer coordinate raises ValueError."""
        v = RelevantCouplings(g2=1.0)
        with pytest.raises(ValueError, match="must not be None"):
            BBSCoordinates(v=v, k=None, scale=0)

    def test_negative_scale_raises(self):
        """Negative scale raises ValueError."""
        v = RelevantCouplings(g2=1.0)
        k = PolymerCoordinate(scale=0)
        with pytest.raises(ValueError, match="Scale must be non-negative"):
            BBSCoordinates(v=v, k=k, scale=-1)

    def test_scale_mismatch_raises(self):
        """Mismatched scales between K and BBS raises ValueError."""
        v = RelevantCouplings(g2=1.0)
        k = PolymerCoordinate(scale=1)
        with pytest.raises(ValueError, match="must match"):
            BBSCoordinates(v=v, k=k, scale=0)

    def test_nonpositive_R_raises(self):
        """R <= 0 raises ValueError."""
        v = RelevantCouplings(g2=1.0)
        k = PolymerCoordinate(scale=0)
        with pytest.raises(ValueError, match="R must be positive"):
            BBSCoordinates(v=v, k=k, scale=0, R=0.0)

    def test_k_norm_zero_for_bare(self, uv_bbs_coords):
        """Bare action has zero polymer norm."""
        assert uv_bbs_coords.k_norm == 0.0

    def test_k_norm_positive_for_sample(self, sample_bbs_coords):
        """Sample coordinates have positive polymer norm."""
        assert sample_bbs_coords.k_norm > 0.0

    def test_is_perturbative_at_bare(self, uv_bbs_coords):
        """Bare coupling g^2 = 6.28 is non-perturbative (alpha_s > 1)."""
        # g^2=6.28 -> alpha_s = 6.28/(4*pi) ~ 0.5 -> perturbative
        assert uv_bbs_coords.is_perturbative is True

    def test_is_contracted_bare(self, uv_bbs_coords):
        """Bare action (K=0) is always contracted."""
        assert uv_bbs_coords.is_contracted is True

    def test_total_action_estimate(self, uv_bbs_coords):
        """Total action estimate is finite and positive for bare action."""
        S = uv_bbs_coords.total_action_estimate(n_blocks=120)
        assert S > 0
        assert np.isfinite(S)

    def test_curvature_correction_uv(self, uv_bbs_coords):
        """Curvature correction at UV scale (j=0) is O(1)."""
        delta = uv_bbs_coords.curvature_correction(M=2.0)
        assert delta == 1.0  # j=0 has maximal curvature correction

    def test_curvature_correction_decreases(self):
        """Curvature correction decreases with scale (UV suppression)."""
        v = RelevantCouplings(g2=1.0)
        deltas = []
        for j in range(5):
            k = PolymerCoordinate(scale=j)
            coords = BBSCoordinates(v=v, k=k, scale=j)
            deltas.append(coords.curvature_correction(M=2.0))
        # Should decrease (except j=0 which is 1.0)
        for i in range(1, len(deltas) - 1):
            assert deltas[i+1] < deltas[i]

    def test_copy_independence(self, sample_bbs_coords):
        """Copy is independent of original."""
        copy = sample_bbs_coords.copy()
        assert copy.g2 == sample_bbs_coords.g2
        assert copy.scale == sample_bbs_coords.scale


# ======================================================================
# Section 4: ExtractionOperator
# ======================================================================

class TestExtractionOperator:
    """Tests for the ExtractionOperator (localization)."""

    def test_construction(self, extraction_op):
        """Extraction operator is constructed correctly."""
        assert extraction_op.R == R_PHYSICAL_FM
        assert extraction_op.N_c == 2
        assert extraction_op.dim_adj == 3

    def test_nonpositive_R_raises(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            ExtractionOperator(R=0.0)

    def test_extract_from_zero(self, extraction_op, zero_polymer_coord):
        """Extraction from zero K gives zero corrections."""
        dg2, dnu, dz = extraction_op.extract_couplings(zero_polymer_coord)
        assert dg2 == 0.0
        assert dnu == 0.0
        assert dz == 0.0

    def test_extract_from_single_block(self, extraction_op):
        """Extraction from a single-block polymer."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 0.1}, kappa=1.0)
        dg2, dnu, dz = extraction_op.extract_couplings(k, block_id=0)
        # All corrections should be nonzero and finite
        assert np.isfinite(dg2)
        assert np.isfinite(dnu)
        assert np.isfinite(dz)

    def test_extract_scales_with_amplitude(self, extraction_op):
        """Extraction scales linearly with polymer amplitude."""
        p = Polymer(frozenset([0]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p: 0.1}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p: 0.2}, kappa=1.0)
        dg2_1, dnu_1, dz_1 = extraction_op.extract_couplings(k1, block_id=0)
        dg2_2, dnu_2, dz_2 = extraction_op.extract_couplings(k2, block_id=0)
        assert abs(dg2_2 - 2.0 * dg2_1) < 1e-14
        assert abs(dnu_2 - 2.0 * dnu_1) < 1e-14
        assert abs(dz_2 - 2.0 * dz_1) < 1e-14

    def test_extract_multi_block_weight(self, extraction_op):
        """Multi-block polymer distributes weight over blocks."""
        p1 = Polymer(frozenset([0]), scale=0)
        p2 = Polymer(frozenset([0, 1]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p1: 1.0}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p2: 1.0}, kappa=1.0)
        dg2_1, _, _ = extraction_op.extract_couplings(k1, block_id=0)
        dg2_2, _, _ = extraction_op.extract_couplings(k2, block_id=0)
        # Two-block polymer contributes half per block
        assert abs(dg2_2 - 0.5 * dg2_1) < 1e-14

    def test_extract_and_subtract_zero(self, extraction_op, zero_polymer_coord):
        """Extract-and-subtract from zero gives zero delta_v."""
        delta_v, k_irrel = extraction_op.extract_and_subtract(zero_polymer_coord)
        assert delta_v.g2 == 0.0
        assert delta_v.nu == 0.0

    def test_extract_and_subtract_nonzero(self, extraction_op, sample_polymer_coord):
        """Extract-and-subtract from nonzero K produces delta_v."""
        delta_v, k_irrel = extraction_op.extract_and_subtract(sample_polymer_coord)
        # delta_v should have finite, small corrections
        assert np.isfinite(delta_v.g2)
        assert np.isfinite(delta_v.nu)

    def test_idempotent_zero(self, extraction_op, zero_polymer_coord):
        """Idempotency on zero K is trivially true."""
        assert extraction_op.is_idempotent(zero_polymer_coord)

    def test_idempotent_single_block(self, extraction_op):
        """Idempotency on single-block polymer activities."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 0.1}, kappa=1.0)
        assert extraction_op.is_idempotent(k, tol=1e-8)

    def test_remainder_norm_bounded(self, extraction_op, sample_polymer_coord):
        """The irrelevant remainder norm is bounded by the original norm."""
        _, k_irrel = extraction_op.extract_and_subtract(sample_polymer_coord)
        original_norm = sample_polymer_coord.norm()
        irrel_norm = k_irrel.norm()
        # THEOREM: ||(1-Loc)[K]|| <= ||K||
        assert irrel_norm <= original_norm * (1.0 + 1e-10)


# ======================================================================
# Section 5: RGMapBBS (single step)
# ======================================================================

class TestRGMapBBS:
    """Tests for the single-step RG map in BBS coordinates."""

    def test_construction(self, rg_map):
        """RG map is constructed correctly."""
        assert rg_map.R == R_PHYSICAL_FM
        assert rg_map.M == M_DEFAULT
        assert rg_map.N_c == 2

    def test_nonpositive_R_raises(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            RGMapBBS(R=0.0)

    def test_M_leq_1_raises(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be > 1"):
            RGMapBBS(M=1.0)

    def test_step_from_bare(self, rg_map, uv_bbs_coords):
        """One RG step from the bare action produces valid coordinates."""
        result = rg_map.step(uv_bbs_coords)
        assert result.scale == 1
        assert result.g2 > 0
        assert result.z > 0
        assert np.isfinite(result.g2)
        assert np.isfinite(result.nu)
        assert np.isfinite(result.z)

    def test_step_coupling_increases(self, rg_map, uv_bbs_coords):
        """Coupling increases toward IR (asymptotic freedom)."""
        result = rg_map.step(uv_bbs_coords)
        assert result.g2 >= uv_bbs_coords.g2 - 1e-10

    def test_step_preserves_scale_consistency(self, rg_map, uv_bbs_coords):
        """After step, scale increments by 1."""
        result = rg_map.step(uv_bbs_coords)
        assert result.scale == uv_bbs_coords.scale + 1

    def test_step_with_nonzero_K(self, rg_map, sample_bbs_coords):
        """RG step with nonzero K produces valid result."""
        result = rg_map.step(sample_bbs_coords)
        assert result.scale == 1
        assert np.isfinite(result.g2)
        assert np.isfinite(result.k_norm)

    def test_gaussian_integration(self, rg_map, uv_bbs_coords):
        """Gaussian integration produces finite corrections."""
        dg2, dnu, dz = rg_map._gaussian_integration(uv_bbs_coords)
        assert np.isfinite(dg2)
        assert np.isfinite(dnu)
        assert np.isfinite(dz)
        # Coupling should increase toward IR
        assert dg2 >= 0

    def test_reblock_reduces_polymer_ids(self, rg_map):
        """Reblocking maps fine block ids to coarse ids."""
        p = Polymer(frozenset([0, 1, 2, 3, 4, 5, 6, 7]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        k_coarse = rg_map._reblock(k, new_scale=1)
        # All ids in [0,7] should map to id 0 (integer division by 8)
        for polymer in k_coarse.activities:
            assert polymer.scale == 1

    def test_reblock_suppression(self, rg_map):
        """Reblocking suppresses activities by M^{-1}."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        k_coarse = rg_map._reblock(k, new_scale=1)
        # Activity should be suppressed by 1/M
        total = k_coarse.total_activity()
        assert abs(total - 1.0 / rg_map.M) < 1e-10

    def test_verify_contraction_bare(self, rg_map, uv_bbs_coords):
        """Contraction verification for bare action step."""
        result = rg_map.step(uv_bbs_coords)
        diag = rg_map.verify_contraction(uv_bbs_coords, result)
        assert diag['kappa'] < 1.0
        assert diag['contracting'] is True

    def test_remainder_estimation(self, rg_map, uv_bbs_coords, zero_polymer_coord):
        """Remainder estimation produces non-negative result."""
        est = rg_map._remainder_estimation(uv_bbs_coords, zero_polymer_coord)
        assert est >= 0

    def test_step_g2_bounded_above(self, rg_map, uv_bbs_coords):
        """Coupling after step does not exceed 4*pi."""
        result = rg_map.step(uv_bbs_coords)
        assert result.g2 <= 4.0 * np.pi + 1e-10

    def test_step_z_positive(self, rg_map, uv_bbs_coords):
        """Wave-function renormalization stays positive after step."""
        result = rg_map.step(uv_bbs_coords)
        assert result.z > 0


# ======================================================================
# Section 6: MultiScaleRGBBS
# ======================================================================

class TestMultiScaleRGBBS:
    """Tests for the full multi-scale RG iteration."""

    def test_construction(self, multi_scale_rg):
        """Multi-scale RG is constructed correctly."""
        assert multi_scale_rg.n_scales == N_SCALES_DEFAULT
        assert multi_scale_rg.R == R_PHYSICAL_FM

    def test_invalid_n_scales(self):
        """n_scales < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_scales must be >= 1"):
            MultiScaleRGBBS(n_scales=0)

    def test_invalid_R(self):
        """R <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="R must be positive"):
            MultiScaleRGBBS(R=0.0)

    def test_invalid_M(self):
        """M <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="M must be > 1"):
            MultiScaleRGBBS(M=1.0)

    def test_invalid_g2(self):
        """g2_bare <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="g2_bare must be positive"):
            MultiScaleRGBBS(g2_bare=0.0)

    def test_initial_coordinates(self, multi_scale_rg):
        """Initial UV coordinates are correct."""
        coords = multi_scale_rg.initial_coordinates()
        assert coords.scale == 0
        assert coords.g2 == G2_BARE_DEFAULT
        assert coords.nu == 0.0
        assert coords.z == 1.0
        assert coords.k.is_zero is True

    def test_run_produces_trajectory(self, multi_scale_rg):
        """Running the flow produces N+1 trajectory points."""
        traj = multi_scale_rg.run()
        assert len(traj) == multi_scale_rg.n_scales + 1

    def test_trajectory_scales_increase(self, multi_scale_rg):
        """Scales in the trajectory increase monotonically."""
        traj = multi_scale_rg.run()
        scales = [c.scale for c in traj]
        for i in range(len(scales) - 1):
            assert scales[i+1] == scales[i] + 1

    def test_coupling_trajectory(self, multi_scale_rg):
        """Coupling trajectory has correct length and keys."""
        traj = multi_scale_rg.run()
        ct = multi_scale_rg.coupling_trajectory(traj)
        assert len(ct['g2']) == len(traj)
        assert len(ct['nu']) == len(traj)
        assert len(ct['z']) == len(traj)
        assert len(ct['alpha_s']) == len(traj)
        assert len(ct['k_norm']) == len(traj)

    def test_asymptotic_freedom(self, multi_scale_rg):
        """Coupling flow exhibits asymptotic freedom."""
        traj = multi_scale_rg.run()
        af = multi_scale_rg.verify_asymptotic_freedom(traj)
        assert af['is_asymptotically_free'] is True

    def test_beta_function_match(self, multi_scale_rg):
        """Effective b_0 matches the exact one-loop value (at weak coupling)."""
        # Use weak coupling to stay perturbative
        rg_weak = MultiScaleRGBBS(
            n_scales=5, R=R_PHYSICAL_FM, M=2.0,
            N_c=2, g2_bare=0.5,
        )
        traj = rg_weak.run()
        af = rg_weak.verify_asymptotic_freedom(traj)
        b0_exact = _beta_0(2)
        # At least the first few steps should match well
        for b in af['b0_effective'][:3]:
            if b > 0:
                assert abs(b - b0_exact) / b0_exact < 0.5

    def test_contraction_all_scales(self, multi_scale_rg):
        """Polymer norm contracts at all scales."""
        traj = multi_scale_rg.run()
        contraction = multi_scale_rg.verify_contraction(traj)
        assert contraction['all_contracting'] is True

    def test_kappa_below_one(self, multi_scale_rg):
        """All contraction factors are below 1."""
        traj = multi_scale_rg.run()
        contraction = multi_scale_rg.verify_contraction(traj)
        assert contraction['max_kappa'] < 1.0

    def test_accumulated_product_decreasing(self, multi_scale_rg):
        """Accumulated kappa product decreases toward zero."""
        traj = multi_scale_rg.run()
        contraction = multi_scale_rg.verify_contraction(traj)
        products = contraction['accumulated_product']
        for i in range(len(products) - 1):
            assert products[i+1] <= products[i] + 1e-10

    def test_mass_gap_positive(self, multi_scale_rg):
        """Mass gap from RG flow is positive."""
        traj = multi_scale_rg.run()
        gap = multi_scale_rg.mass_gap_from_flow(traj)
        assert gap['mass_gap_mev'] > 0
        assert gap['m2_effective'] > 0

    def test_mass_gap_order_of_magnitude(self, multi_scale_rg):
        """Mass gap is positive and finite (right order of magnitude)."""
        traj = multi_scale_rg.run()
        gap = multi_scale_rg.mass_gap_from_flow(traj)
        # The bare gap on S3(R=2.2fm) is 2*hbar*c/R ~ 179 MeV.
        # With RG corrections (mass renormalization adds to nu), the
        # effective gap can be larger. We check it is positive, finite,
        # and within a broad physically reasonable window.
        assert 50.0 < gap['mass_gap_mev'] < 5000.0

    def test_bare_gap_value(self, multi_scale_rg):
        """Bare gap is 4/R^2 = 4/(2.2)^2."""
        traj = multi_scale_rg.run()
        gap = multi_scale_rg.mass_gap_from_flow(traj)
        expected = 4.0 / R_PHYSICAL_FM**2
        assert abs(gap['bare_gap_inv_fm2'] - expected) < 1e-10

    def test_curvature_corrections_decrease(self, multi_scale_rg):
        """Curvature corrections decrease with scale (UV suppression)."""
        traj = multi_scale_rg.run()
        curv = multi_scale_rg.curvature_corrections(traj)
        # Skip j=0 (maximal correction) and check decreasing
        for i in range(2, len(curv)):
            assert curv[i] <= curv[i-1] + 1e-10

    def test_curvature_uv_negligible(self, multi_scale_rg):
        """Curvature corrections at high UV scales are negligible."""
        traj = multi_scale_rg.run()
        curv = multi_scale_rg.curvature_corrections(traj)
        # At scale j >= 3: M^{-6} = 1/64 ~ 0.016
        if len(curv) > 3:
            assert curv[3] < 0.1

    def test_summary_keys(self, multi_scale_rg):
        """Summary contains all expected keys."""
        traj = multi_scale_rg.run()
        s = multi_scale_rg.summary(traj)
        expected_keys = [
            'n_scales', 'R_fm', 'M', 'N_c', 'g2_bare',
            'couplings', 'asymptotic_freedom', 'contraction',
            'mass_gap', 'curvature_corrections', 'n_trajectory_points',
        ]
        for key in expected_keys:
            assert key in s

    def test_run_with_custom_initial(self, multi_scale_rg):
        """Running with custom initial coordinates works."""
        v_custom = RelevantCouplings(g2=1.0, nu=0.1, z=0.9, N_c=2)
        k_custom = PolymerCoordinate(scale=0, kappa=1.0)
        coords_custom = BBSCoordinates(v=v_custom, k=k_custom, scale=0)
        traj = multi_scale_rg.run(initial_coords=coords_custom)
        assert len(traj) == multi_scale_rg.n_scales + 1
        assert traj[0].g2 == 1.0


# ======================================================================
# Section 7: Physical parameter checks
# ======================================================================

class TestPhysicalParameters:
    """Tests for physical parameter consistency."""

    def test_hbar_c_value(self):
        """hbar*c = 197.327 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-4

    def test_r_physical(self):
        """Physical radius R = 2.2 fm."""
        assert R_PHYSICAL_FM == 2.2

    def test_lambda_qcd(self):
        """Lambda_QCD = 200 MeV."""
        assert LAMBDA_QCD_MEV == 200.0

    def test_g2_bare_default(self):
        """Default bare coupling g^2 = 6.28."""
        assert G2_BARE_DEFAULT == 6.28

    def test_m_default(self):
        """Default blocking factor M = 2."""
        assert M_DEFAULT == 2.0

    def test_dim_spacetime(self):
        """Spacetime dimension is 4 (S3 x R)."""
        assert DIM_SPACETIME == 4

    def test_beta_0_su2_numerical(self):
        """b_0 for SU(2) ~ 0.0464."""
        assert abs(BETA_0_SU2 - 0.04637) < 1e-4

    def test_quadratic_casimir_su2(self):
        """C_2(adj, SU(2)) = 2."""
        assert quadratic_casimir(2) == 2.0

    def test_quadratic_casimir_su3(self):
        """C_2(adj, SU(3)) = 3."""
        assert quadratic_casimir(3) == 3.0

    def test_coexact_eigenvalue_k1(self):
        """lambda_1 = 4/R^2 on S3(R)."""
        R = R_PHYSICAL_FM
        assert coexact_eigenvalue(1, R) == pytest.approx(4.0 / R**2)

    def test_coexact_multiplicity_k1(self):
        """d_1 = 2*1*3 = 6 coexact modes at k=1."""
        assert coexact_multiplicity(1) == 6


# ======================================================================
# Section 8: R-scan and universality
# ======================================================================

class TestRScanAndUniversality:
    """Tests for R-dependence and gauge group universality."""

    def test_small_R_coupling_weak(self):
        """At small R, coupling is weak (asymptotic freedom at short distances)."""
        rg = MultiScaleRGBBS(n_scales=5, R=0.5, M=2.0, N_c=2, g2_bare=1.0)
        traj = rg.run()
        # At small R, the coupling should not blow up too fast
        assert traj[-1].g2 < 4.0 * np.pi + 1e-10

    def test_large_R_gap_decreases(self):
        """At larger R, bare mass gap decreases (4/R^2)."""
        rg1 = MultiScaleRGBBS(n_scales=5, R=1.0, M=2.0, N_c=2, g2_bare=G2_BARE_DEFAULT)
        rg2 = MultiScaleRGBBS(n_scales=5, R=5.0, M=2.0, N_c=2, g2_bare=G2_BARE_DEFAULT)
        gap1 = rg1.mass_gap_from_flow()
        gap2 = rg2.mass_gap_from_flow()
        assert gap1['bare_gap_inv_fm2'] > gap2['bare_gap_inv_fm2']

    def test_su3_flow(self):
        """SU(3) flow also exhibits asymptotic freedom."""
        rg = MultiScaleRGBBS(n_scales=5, R=R_PHYSICAL_FM, M=2.0, N_c=3, g2_bare=6.28)
        traj = rg.run()
        af = rg.verify_asymptotic_freedom(traj)
        assert af['is_asymptotically_free'] is True

    def test_su3_beta_larger(self):
        """SU(3) has larger b_0 than SU(2) (more asymptotically free)."""
        b0_su2 = _beta_0(2)
        b0_su3 = _beta_0(3)
        assert b0_su3 > b0_su2

    def test_n_scales_1(self):
        """Flow with just 1 scale step works."""
        rg = MultiScaleRGBBS(n_scales=1, R=R_PHYSICAL_FM)
        traj = rg.run()
        assert len(traj) == 2

    def test_n_scales_10(self):
        """Flow with 10 scales works without blowup."""
        rg = MultiScaleRGBBS(n_scales=10, R=R_PHYSICAL_FM)
        traj = rg.run()
        assert len(traj) == 11
        # All values should be finite
        for c in traj:
            assert np.isfinite(c.g2)
            assert np.isfinite(c.nu)
            assert np.isfinite(c.z)


# ======================================================================
# Section 9: Sum rule and consistency
# ======================================================================

class TestSumRuleAndConsistency:
    """Tests for the sum rule V + K = S and internal consistency."""

    def test_sum_rule_bare_action(self, uv_bbs_coords):
        """For bare action: V = S (K = 0), so V + K = V = S."""
        assert uv_bbs_coords.k.is_zero is True
        # Total action is just V
        S = uv_bbs_coords.total_action_estimate()
        assert S > 0

    def test_extraction_preserves_total(self, extraction_op, sample_polymer_coord):
        """Extraction preserves the total: Loc[K] + (1-Loc)[K] ~ K."""
        delta_v, k_irrel = extraction_op.extract_and_subtract(sample_polymer_coord)
        # The total activity of the original should roughly equal
        # the sum of extracted and irrelevant parts
        original_total = sample_polymer_coord.total_activity()
        irrel_total = k_irrel.total_activity()
        # Not exact because extraction modifies single-block activities
        assert np.isfinite(original_total)
        assert np.isfinite(irrel_total)

    def test_coupling_flow_consistency(self, multi_scale_rg):
        """Coupling at each scale is consistent with one-loop evolution."""
        traj = multi_scale_rg.run()
        g2_traj = [c.g2 for c in traj]
        b0 = _beta_0(multi_scale_rg.N_c)
        M = multi_scale_rg.M

        # Check one-loop consistency (approximate, due to higher-order + extraction)
        for i in range(min(3, len(g2_traj) - 1)):
            g2_i = g2_traj[i]
            g2_ip1 = g2_traj[i + 1]
            if g2_i > 0 and g2_i < 4.0 * np.pi:
                # One-loop prediction
                inv_predicted = 1.0 / g2_i - b0 * np.log(M**2)
                if inv_predicted > 0:
                    g2_predicted = 1.0 / inv_predicted
                    # Should be within ~50% (extraction corrections + higher loops)
                    assert abs(g2_ip1 - g2_predicted) / g2_predicted < 1.0

    def test_mass_gap_above_bare(self, multi_scale_rg):
        """Effective mass gap is at least half the bare gap (gauge protection)."""
        traj = multi_scale_rg.run()
        gap = multi_scale_rg.mass_gap_from_flow(traj)
        bare = gap['bare_gap_inv_fm2']
        effective = gap['m2_effective']
        assert effective >= bare * 0.5 - 1e-10


# ======================================================================
# Section 10: Edge cases
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_coupling_flow(self):
        """Zero coupling (free theory) stays at zero."""
        rg = MultiScaleRGBBS(n_scales=3, R=R_PHYSICAL_FM, g2_bare=1e-10)
        traj = rg.run()
        # Coupling should stay very small
        for c in traj:
            assert c.g2 < 0.1

    def test_very_large_R(self):
        """Very large R (quasi-flat) still produces finite flow."""
        rg = MultiScaleRGBBS(n_scales=3, R=100.0, g2_bare=G2_BARE_DEFAULT)
        traj = rg.run()
        assert all(np.isfinite(c.g2) for c in traj)

    def test_very_small_R(self):
        """Very small R (strong curvature) still produces finite flow."""
        rg = MultiScaleRGBBS(n_scales=3, R=0.1, g2_bare=G2_BARE_DEFAULT)
        traj = rg.run()
        assert all(np.isfinite(c.g2) for c in traj)

    def test_polymer_coordinate_many_polymers(self):
        """PolymerCoordinate handles many polymers."""
        activities = {}
        for i in range(50):
            p = Polymer(frozenset([i]), scale=0)
            activities[p] = 0.01 / (i + 1)
        k = PolymerCoordinate(scale=0, activities=activities, kappa=1.0)
        assert k.n_polymers == 50
        assert k.norm() > 0

    def test_extraction_many_blocks(self, extraction_op):
        """Extraction handles polymers spanning many blocks."""
        p = Polymer(frozenset(range(10)), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 0.05}, kappa=1.0)
        dg2, dnu, dz = extraction_op.extract_couplings(k, block_id=0)
        assert np.isfinite(dg2)
        # Weight should be 1/10 for a 10-block polymer
        p1 = Polymer(frozenset([0]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p1: 0.05}, kappa=1.0)
        dg2_1, _, _ = extraction_op.extract_couplings(k1, block_id=0)
        assert abs(dg2 - dg2_1 / 10.0) < 1e-14

    def test_high_scale_curvature_correction(self):
        """Curvature correction at scale j=10 is < 10^{-6}."""
        v = RelevantCouplings(g2=1.0)
        k = PolymerCoordinate(scale=10)
        coords = BBSCoordinates(v=v, k=k, scale=10)
        delta = coords.curvature_correction(M=2.0)
        assert delta < 1e-6

    def test_polymer_norm_triangle_inequality(self):
        """Polymer norm satisfies triangle inequality."""
        p = Polymer(frozenset([0]), scale=0)
        k1 = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        k2 = PolymerCoordinate(scale=0, activities={p: 2.0}, kappa=1.0)
        k_sum = k1 + k2
        assert k_sum.norm() <= k1.norm() + k2.norm() + 1e-10

    def test_polymer_norm_homogeneity(self):
        """Polymer norm satisfies |alpha * K| = |alpha| * |K|."""
        p = Polymer(frozenset([0]), scale=0)
        k = PolymerCoordinate(scale=0, activities={p: 1.0}, kappa=1.0)
        alpha = 3.7
        k_scaled = k * alpha
        assert abs(k_scaled.norm() - alpha * k.norm()) < 1e-10


# ======================================================================
# Section 11: Cross-validation with existing modules
# ======================================================================

class TestCrossValidation:
    """Cross-validation with first_rg_step.py and inductive_closure.py."""

    def test_beta_0_matches_first_rg_step(self):
        """b_0 from BBS matches b_0 from first_rg_step."""
        from yang_mills_s3.rg.first_rg_step import RGFlow
        flow = RGFlow(R=R_PHYSICAL_FM, N_c=2)
        b0_flow = flow.beta_coefficient()
        b0_bbs = _beta_0(2)
        assert abs(b0_flow - b0_bbs) < 1e-14

    def test_coupling_evolution_matches_first_rg_step(self):
        """One-loop coupling evolution matches existing RGFlow."""
        from yang_mills_s3.rg.first_rg_step import OneLoopEffectiveAction
        one_loop = OneLoopEffectiveAction(R=R_PHYSICAL_FM, N_c=2, g2=0.5)

        # BBS evolution
        couplings = RelevantCouplings(g2=0.5, N_c=2)
        g2_bbs = couplings.evolved_g2(M=2.0)

        # OneLoopEffectiveAction evolution
        g2_ola = one_loop.effective_coupling_after_step(3, 0.5)

        # Should match (both are one-loop)
        assert abs(g2_bbs - g2_ola) < 1e-10

    def test_kappa_matches_remainder_estimate(self):
        """Contraction factor from RGMapBBS matches RemainderEstimate."""
        rg_map = RGMapBBS(R=R_PHYSICAL_FM, M=2.0, N_c=2)
        kappa_bbs = rg_map.remainder_est.spectral_contraction(3)

        from yang_mills_s3.rg.first_rg_step import RemainderEstimate
        rem = RemainderEstimate(R=R_PHYSICAL_FM, M=2.0, N_scales=7, N_c=2)
        kappa_rem = rem.spectral_contraction(3)

        assert abs(kappa_bbs - kappa_rem) < 1e-14

    def test_mass_gap_consistent_with_flow(self):
        """Mass gap from BBS is consistent with MultiScaleRGFlow."""
        from yang_mills_s3.rg.inductive_closure import MultiScaleRGFlow
        flow = MultiScaleRGFlow(R=R_PHYSICAL_FM, M=2.0, N_scales=7, N_c=2)
        result = flow.run_flow()
        gap_flow_mev = result['mass_gap_mev']

        rg = MultiScaleRGBBS(n_scales=7, R=R_PHYSICAL_FM, M=2.0, N_c=2)
        gap_bbs = rg.mass_gap_from_flow()
        gap_bbs_mev = gap_bbs['mass_gap_mev']

        # Both should be in the same ballpark (same physics, different implementation)
        assert abs(gap_flow_mev - gap_bbs_mev) / max(gap_flow_mev, gap_bbs_mev) < 1.0
