"""
Tests for BBS T_z Seminorm -- Definition 7.1.1 from Bauerschmidt-Brydges-Slade.

Tests cover all 8 classes + comparison utility:
1. FrechetDerivative: exact polynomial, numerical, symmetry, operator norm
2. BBSTzSeminorm: polynomial/numerical evaluation, algebra property, quartic
3. NormParameter: ell_j, Pythagorean inflation, scale evolution, g_bar
4. BBSRegulator: evaluate, sub-multiplicativity, convolution bound, stability
5. WeightedPolymerNorm: evaluation, contraction check
6. ProductPropertyVerifier: polynomial, numerical, exponential
7. GaussianConvolutionBound: ell_plus, verify bounds
8. GaugeCovariantTaylor: covariant derivatives, extraction, seminorm
9. Comparison with legacy TPhiSeminorm

Run:
    pytest tests/rg/test_bbs_seminorm.py -v
"""

import math
import numpy as np
import pytest

from yang_mills_s3.rg.bbs_seminorm import (
    FrechetDerivative,
    BBSTzSeminorm,
    NormParameter,
    BBSRegulator,
    WeightedPolymerNorm,
    ProductPropertyVerifier,
    GaussianConvolutionBound,
    GaugeCovariantTaylor,
    compare_with_legacy,
    G2_BARE_DEFAULT,
    N_C_DEFAULT,
    DIM_ADJ_SU2,
    SPACETIME_DIM,
    P_N_DEFAULT,
    M_DEFAULT,
    BETA_0_SU2,
    R_PHYSICAL_FM,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def zero_3d():
    """Zero vector in R^3."""
    return np.zeros(3)


@pytest.fixture
def unit_3d():
    """Unit vector (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))."""
    v = np.ones(3)
    return v / np.linalg.norm(v)


@pytest.fixture
def random_3d():
    """Random vector in R^3 with norm ~ 1."""
    rng = np.random.RandomState(42)
    v = rng.randn(3)
    return v / np.linalg.norm(v)


@pytest.fixture
def random_9d():
    """Random vector in R^9 (SU(2) gauge field dimension)."""
    rng = np.random.RandomState(42)
    return rng.randn(9) * 0.5


@pytest.fixture
def quartic_coeffs():
    """Coefficients for V = g * |phi|^4 with g = G2_BARE_DEFAULT."""
    return np.array([0.0, 0.0, G2_BARE_DEFAULT])


@pytest.fixture
def quadratic_coeffs():
    """Coefficients for F = |phi|^2."""
    return np.array([0.0, 1.0])


@pytest.fixture
def constant_coeffs():
    """Coefficients for F = 1."""
    return np.array([1.0])


@pytest.fixture
def norm_param():
    """Default NormParameter."""
    return NormParameter()


@pytest.fixture
def bbs_seminorm():
    """Default BBSTzSeminorm with p_N=5."""
    return BBSTzSeminorm(p_N=5, ell=1.0)


@pytest.fixture
def bbs_seminorm_natural():
    """BBSTzSeminorm at natural ell for g^2 = 6.28."""
    ell = G2_BARE_DEFAULT ** (-0.25)
    return BBSTzSeminorm(p_N=5, ell=ell)


@pytest.fixture
def regulator():
    """Default BBSRegulator."""
    return BBSRegulator(c_G=0.1)


# ======================================================================
# 1. FrechetDerivative tests
# ======================================================================

class TestFrechetDerivative:
    """Tests for FrechetDerivative class."""

    def test_init_positive_eps(self):
        """eps must be positive."""
        fd = FrechetDerivative(eps=1e-4)
        assert fd.eps == 1e-4

    def test_init_negative_eps_raises(self):
        """Negative eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be positive"):
            FrechetDerivative(eps=-1e-5)

    def test_init_zero_eps_raises(self):
        """Zero eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be positive"):
            FrechetDerivative(eps=0)

    def test_polynomial_derivative_constant(self, zero_3d):
        """D^0 of constant = constant, D^k of constant = 0 for k >= 1."""
        coeffs = np.array([5.0])  # F = 5
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 0) == 5.0
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 1) == 0.0
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 2) == 0.0

    def test_polynomial_derivative_quadratic_at_origin(self, zero_3d):
        """D^k of |phi|^2 at origin."""
        coeffs = np.array([0.0, 1.0])  # F = |phi|^2
        # D^0 at origin: |0|^2 = 0
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 0) == 0.0
        # D^1 at origin: 2|z| * (z_hat . h) -> 0 at z=0
        # Actually for |phi|^2, D^1 = 2*phi, so at z=0 it's 0
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 1) == 0.0
        # D^2 at origin: constant = 2 (falling factorial: 2!/0! = 2)
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 2) == pytest.approx(2.0)

    def test_polynomial_derivative_quadratic_at_unit(self, unit_3d):
        """D^k of |phi|^2 at unit vector."""
        coeffs = np.array([0.0, 1.0])  # F = |phi|^2
        z_norm = np.linalg.norm(unit_3d)
        # D^0: |z|^2 = 1
        assert FrechetDerivative.polynomial_derivative(coeffs, unit_3d, 0) == pytest.approx(1.0)
        # D^1: 2 * |z|^{2-1} = 2
        assert FrechetDerivative.polynomial_derivative(coeffs, unit_3d, 1) == pytest.approx(2.0)
        # D^2: 2!/0! * |z|^0 = 2
        assert FrechetDerivative.polynomial_derivative(coeffs, unit_3d, 2) == pytest.approx(2.0)
        # D^3: 0 (power 2 < k=3)
        assert FrechetDerivative.polynomial_derivative(coeffs, unit_3d, 3) == 0.0

    def test_polynomial_derivative_quartic_at_origin(self, zero_3d):
        """D^k of g*|phi|^4 at origin."""
        g = 2.0
        coeffs = np.array([0.0, 0.0, g])  # F = g * |phi|^4
        # D^0 = 0 at origin
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 0) == 0.0
        # D^4 at origin: g * 4!/0! = 24g
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 4) == pytest.approx(24.0 * g)
        # D^5 of |phi|^4 = 0 (power 4 < k=5)
        assert FrechetDerivative.polynomial_derivative(coeffs, zero_3d, 5) == 0.0

    def test_polynomial_derivative_quartic_at_nonzero(self):
        """D^k of g*|phi|^4 at nonzero z."""
        g = 1.0
        coeffs = np.array([0.0, 0.0, g])
        z = np.array([1.0, 0.0, 0.0])
        z_norm = 1.0
        # D^0: g * |z|^4 = g
        assert FrechetDerivative.polynomial_derivative(coeffs, z, 0) == pytest.approx(g)
        # D^1: g * 4 * |z|^3 = 4g
        assert FrechetDerivative.polynomial_derivative(coeffs, z, 1) == pytest.approx(4.0 * g)
        # D^2: g * 4*3 * |z|^2 = 12g
        assert FrechetDerivative.polynomial_derivative(coeffs, z, 2) == pytest.approx(12.0 * g)
        # D^3: g * 4*3*2 * |z|^1 = 24g
        assert FrechetDerivative.polynomial_derivative(coeffs, z, 3) == pytest.approx(24.0 * g)
        # D^4: g * 4! = 24g (same as at origin for k=power)
        assert FrechetDerivative.polynomial_derivative(coeffs, z, 4) == pytest.approx(24.0 * g)

    def test_polynomial_derivative_negative_k_raises(self, zero_3d):
        """Negative derivative order raises ValueError."""
        coeffs = np.array([1.0])
        with pytest.raises(ValueError, match="k must be >= 0"):
            FrechetDerivative.polynomial_derivative(coeffs, zero_3d, -1)

    def test_numerical_derivative_k0_constant(self, unit_3d):
        """Numerical D^0 = |F(z)|."""
        fd = FrechetDerivative()
        F = lambda phi: 3.0
        assert fd.numerical_derivative(F, unit_3d, 0) == pytest.approx(3.0)

    def test_numerical_derivative_k1_quadratic(self, unit_3d):
        """Numerical D^1 of |phi|^2 at unit vector should be close to 2.

        NUMERICAL: Random direction sampling gives a lower bound on the
        operator norm. The true sup is 2|phi| = 2 along z-hat, but
        random directions may not align perfectly.
        """
        fd = FrechetDerivative(eps=1e-5)
        F = lambda phi: float(np.sum(phi**2))
        val = fd.numerical_derivative(F, unit_3d, 1, n_directions=50)
        # Random sampling gives a lower bound; accept within 20%
        assert val == pytest.approx(2.0, rel=0.2)

    def test_numerical_derivative_k2_quadratic(self, unit_3d):
        """Numerical D^2 of |phi|^2 should be ~ 2 (constant Hessian)."""
        fd = FrechetDerivative(eps=1e-4)
        F = lambda phi: float(np.sum(phi**2))
        val = fd.numerical_derivative(F, unit_3d, 2, n_directions=50)
        assert val == pytest.approx(2.0, rel=0.15)

    def test_numerical_derivative_k0_zero_func(self, unit_3d):
        """D^0 of zero function is 0."""
        fd = FrechetDerivative()
        F = lambda phi: 0.0
        assert fd.numerical_derivative(F, unit_3d, 0) == 0.0

    def test_numerical_derivative_negative_k_raises(self, unit_3d):
        """Negative k raises ValueError."""
        fd = FrechetDerivative()
        with pytest.raises(ValueError, match="k must be >= 0"):
            fd.numerical_derivative(lambda phi: 0.0, unit_3d, -1)

    def test_symmetry_trivially_true_k0(self, unit_3d):
        """k=0 and k=1 are trivially symmetric."""
        assert FrechetDerivative.is_symmetric(
            lambda z, hs: 0.0, unit_3d, 0, 3
        )
        assert FrechetDerivative.is_symmetric(
            lambda z, hs: 0.0, unit_3d, 1, 3
        )

    def test_symmetry_quadratic_form(self, unit_3d):
        """D^2 of |phi|^2 is symmetric (it's the identity bilinear form)."""
        def D2_phi_sq(z, h_list):
            # D^2 |phi|^2 [h1, h2] = 2 * h1 . h2
            return 2.0 * float(np.dot(h_list[0], h_list[1]))

        assert FrechetDerivative.is_symmetric(D2_phi_sq, unit_3d, 2, 3)


# ======================================================================
# 2. BBSTzSeminorm tests
# ======================================================================

class TestBBSTzSeminorm:
    """Tests for BBSTzSeminorm class (BBS Definition 7.1.1)."""

    def test_init_default(self):
        """Default parameters: p_N=5, ell=1.0."""
        sn = BBSTzSeminorm()
        assert sn.p_N == P_N_DEFAULT
        assert sn.ell == 1.0

    def test_init_negative_pN_raises(self):
        """Negative p_N raises ValueError."""
        with pytest.raises(ValueError, match="p_N must be >= 0"):
            BBSTzSeminorm(p_N=-1)

    def test_init_negative_ell_raises(self):
        """Negative ell raises ValueError."""
        with pytest.raises(ValueError, match="ell must be >= 0"):
            BBSTzSeminorm(ell=-0.1)

    def test_constant_polynomial(self, zero_3d):
        """T_z norm of constant c is |c|."""
        sn = BBSTzSeminorm(p_N=5, ell=1.0)
        coeffs = np.array([3.14])
        assert sn.evaluate_polynomial(coeffs, zero_3d) == pytest.approx(3.14)

    def test_zero_polynomial(self, zero_3d):
        """T_z norm of zero polynomial is 0."""
        sn = BBSTzSeminorm(p_N=5, ell=1.0)
        coeffs = np.array([0.0])
        assert sn.evaluate_polynomial(coeffs, zero_3d) == 0.0

    def test_quadratic_at_origin(self, zero_3d):
        """||phi|^2||_{T_0(ell)} = ell^2/2! * 2 = ell^2 (at origin)."""
        ell = 2.0
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        coeffs = np.array([0.0, 1.0])  # |phi|^2
        # At origin: D^0 = 0, D^1 = 0, D^2 = 2, rest = 0
        # T_z = 0 + 0 + (ell^2/2)*2 = ell^2
        assert sn.evaluate_polynomial(coeffs, zero_3d) == pytest.approx(ell**2)

    def test_quartic_at_origin(self, zero_3d):
        """||g*|phi|^4||_{T_0(ell)} = g * ell^4 / 4! * 24 = g * ell^4 at origin."""
        g = 2.0
        ell = 1.5
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        coeffs = np.array([0.0, 0.0, g])
        # At origin: D^4(g|phi|^4) = g * 24, others are 0
        # T_z = (ell^4/4!) * 24g = g * ell^4
        expected = g * ell**4
        assert sn.evaluate_polynomial(coeffs, zero_3d) == pytest.approx(expected)

    def test_quartic_natural_scale(self, zero_3d):
        """At ell = g^{-1/4}, ||V|| = O(1) -- the BBS normalization."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        coeffs = np.array([0.0, 0.0, g])
        val = sn.evaluate_polynomial(coeffs, zero_3d)
        # At origin: g * ell^4 = g * g^{-1} = 1
        assert val == pytest.approx(1.0, rel=1e-10)

    def test_quartic_natural_scale_nonzero_z(self):
        """||V||_{T_z(ell)} at nonzero z is still finite and moderate."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        coeffs = np.array([0.0, 0.0, g])
        z = np.array([0.5, 0.3, 0.1])  # small field
        val = sn.evaluate_polynomial(coeffs, z)
        # Should be finite and positive
        assert val > 0
        assert np.isfinite(val)

    def test_p_N_dependence(self, zero_3d):
        """Higher p_N includes more derivative terms."""
        coeffs = np.array([0.0, 0.0, 1.0])  # |phi|^4
        ell = 1.0
        # With p_N=3: misses D^4 term
        sn3 = BBSTzSeminorm(p_N=3, ell=ell)
        val3 = sn3.evaluate_polynomial(coeffs, zero_3d)
        # With p_N=5: includes D^4 term (= 24*ell^4/24 = 1)
        sn5 = BBSTzSeminorm(p_N=5, ell=ell)
        val5 = sn5.evaluate_polynomial(coeffs, zero_3d)
        assert val5 >= val3

    def test_ell_zero_gives_just_F_value(self, unit_3d):
        """With ell=0, only D^0 contributes: ||F||_{T_z(0)} = |F(z)|."""
        sn = BBSTzSeminorm(p_N=5, ell=0.0)
        coeffs = np.array([1.0, 2.0])  # 1 + 2|phi|^2
        val = sn.evaluate_polynomial(coeffs, unit_3d)
        expected = 1.0 + 2.0 * np.sum(unit_3d**2)  # F(z)
        assert val == pytest.approx(expected)

    def test_is_algebra_always_true(self, bbs_seminorm):
        """T_z seminorm is always a normed algebra (THEOREM)."""
        assert bbs_seminorm.is_algebra is True

    def test_evaluate_dispatches_polynomial(self, zero_3d):
        """evaluate() accepts ndarray as polynomial coefficients."""
        sn = BBSTzSeminorm(p_N=5, ell=1.0)
        coeffs = np.array([1.0, 0.0, 1.0])
        val = sn.evaluate(coeffs, zero_3d)
        assert val > 0

    def test_evaluate_dispatches_callable(self, unit_3d):
        """evaluate() accepts callable for numerical evaluation."""
        sn = BBSTzSeminorm(p_N=3, ell=1.0)
        F = lambda phi: float(np.sum(phi**2))
        val = sn.evaluate(F, unit_3d)
        assert val > 0

    def test_evaluate_raises_on_bad_type(self, unit_3d):
        """evaluate() raises TypeError for unsupported types."""
        sn = BBSTzSeminorm(p_N=5, ell=1.0)
        with pytest.raises(TypeError):
            sn.evaluate("not a function", unit_3d)

    def test_quartic_seminorm_method(self, zero_3d):
        """quartic_seminorm convenience method matches manual computation."""
        g = 3.0
        ell = g ** (-0.25)
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        val = sn.quartic_seminorm(g, zero_3d)
        assert val == pytest.approx(1.0, rel=1e-10)

    def test_numerical_matches_polynomial_approx(self, unit_3d):
        """Numerical evaluation should approximate polynomial evaluation."""
        sn = BBSTzSeminorm(p_N=3, ell=1.0)
        coeffs = np.array([0.0, 1.0])  # |phi|^2
        F = lambda phi: float(np.sum(phi**2))

        val_poly = sn.evaluate_polynomial(coeffs, unit_3d)
        val_num = sn.evaluate_numerical(F, unit_3d, n_directions=50)

        # Numerical should be close but may underestimate
        assert val_num > 0
        assert val_num <= val_poly * 2.0  # Within factor 2


# ======================================================================
# 3. NormParameter tests
# ======================================================================

class TestNormParameter:
    """Tests for NormParameter class."""

    def test_init_default(self):
        """Default c_ell=1.0, beta_0 > 0."""
        np_obj = NormParameter()
        assert np_obj.c_ell == 1.0
        assert np_obj.beta_0 > 0

    def test_init_negative_c_ell_raises(self):
        """Negative c_ell raises ValueError."""
        with pytest.raises(ValueError, match="c_ell must be positive"):
            NormParameter(c_ell=-1.0)

    def test_init_negative_beta0_raises(self):
        """Negative beta_0 raises ValueError."""
        with pytest.raises(ValueError, match="beta_0 must be positive"):
            NormParameter(beta_0=-0.01)

    def test_g_bar_at_scale_0(self, norm_param):
        """g_bar at j=0 is g2_bare."""
        g2 = G2_BARE_DEFAULT
        assert norm_param.g_bar(0, g2) == pytest.approx(g2)

    def test_g_bar_decreases_with_j(self, norm_param):
        """g_bar decreases with j (asymptotic freedom)."""
        g2 = G2_BARE_DEFAULT
        g_bars = [norm_param.g_bar(j, g2) for j in range(10)]
        for i in range(len(g_bars) - 1):
            assert g_bars[i + 1] < g_bars[i]

    def test_g_bar_one_loop_formula(self, norm_param):
        """g_bar_j = g2 / (1 + beta_0 * g2 * j)."""
        g2 = G2_BARE_DEFAULT
        j = 3
        expected = g2 / (1 + norm_param.beta_0 * g2 * j)
        assert norm_param.g_bar(j, g2) == pytest.approx(expected)

    def test_ell_j_at_scale_0(self, norm_param):
        """ell_0 = c_ell * g_bar_0^{-1/4} = g2^{-1/4}."""
        g2 = G2_BARE_DEFAULT
        expected = g2 ** (-0.25)
        assert norm_param.at_scale(0, g2) == pytest.approx(expected)

    def test_ell_j_physical_value(self, norm_param):
        """ell_0 ~ 0.631 for g^2 = 6.28."""
        g2 = G2_BARE_DEFAULT
        ell_0 = norm_param.at_scale(0, g2)
        assert ell_0 == pytest.approx(0.631, abs=0.01)

    def test_ell_j_increases_with_j(self, norm_param):
        """ell_j increases with j (coupling decreases -> ell grows)."""
        g2 = G2_BARE_DEFAULT
        ells = [norm_param.at_scale(j, g2) for j in range(10)]
        for i in range(len(ells) - 1):
            assert ells[i + 1] > ells[i]

    def test_inflate_pythagorean(self, norm_param):
        """ell_+ = sqrt(ell^2 + w^2)."""
        ell = 3.0
        w = 4.0
        assert norm_param.inflate(ell, w) == pytest.approx(5.0)

    def test_inflate_zero_w(self, norm_param):
        """Zero fluctuation leaves ell unchanged."""
        ell = 2.5
        assert norm_param.inflate(ell, 0.0) == pytest.approx(ell)

    def test_inflate_zero_ell(self, norm_param):
        """ell=0 inflated by w gives w."""
        w = 1.5
        assert norm_param.inflate(0.0, w) == pytest.approx(w)

    def test_deflate_ratio(self, norm_param):
        """Ratio ell_+/ell computed correctly."""
        ell = 3.0
        ell_plus = 5.0
        assert norm_param.deflate_ratio(ell_plus, ell) == pytest.approx(5.0 / 3.0)

    def test_deflate_ratio_zero_ell(self, norm_param):
        """Zero ell gives inf ratio."""
        assert norm_param.deflate_ratio(1.0, 0.0) == float('inf')

    def test_scale_evolution(self, norm_param):
        """scale_evolution returns (ell_j, ell_plus)."""
        g2 = G2_BARE_DEFAULT
        w = 0.1
        ell_j, ell_plus = norm_param.scale_evolution(0, g2, w)
        assert ell_j > 0
        assert ell_plus > ell_j
        assert ell_plus == pytest.approx(np.sqrt(ell_j**2 + w**2))

    def test_ell_j_zero_coupling(self, norm_param):
        """Zero coupling gives infinite ell (no interaction)."""
        ell = norm_param.at_scale(0, 0.0)
        assert ell == float('inf')


# ======================================================================
# 4. BBSRegulator tests
# ======================================================================

class TestBBSRegulator:
    """Tests for BBSRegulator class (BBS Section 8.2.2)."""

    def test_init_default(self):
        """Default construction."""
        reg = BBSRegulator()
        assert reg.c_G == 0.1

    def test_init_negative_cG_raises(self):
        """Negative c_G raises ValueError."""
        with pytest.raises(ValueError, match="c_G must be positive"):
            BBSRegulator(c_G=-0.1)

    def test_evaluate_zero_field(self, regulator):
        """G(0) = exp(0) = 1 for zero field."""
        phi = np.zeros(3)
        assert regulator.evaluate(phi, 1.0) == pytest.approx(1.0)

    def test_evaluate_positive(self, regulator):
        """G >= 1 for all field configurations."""
        phi = np.array([1.0, 2.0, 3.0])
        val = regulator.evaluate(phi, 1.0)
        assert val >= 1.0

    def test_evaluate_increases_with_phi(self, regulator):
        """G increases with |phi|."""
        phi_small = np.array([0.1, 0.0, 0.0])
        phi_large = np.array([1.0, 0.0, 0.0])
        assert regulator.evaluate(phi_large, 1.0) > regulator.evaluate(phi_small, 1.0)

    def test_evaluate_decreases_with_ell(self, regulator):
        """G decreases with ell (for fixed phi)."""
        phi = np.array([1.0, 1.0, 1.0])
        assert regulator.evaluate(phi, 2.0) < regulator.evaluate(phi, 1.0)

    def test_evaluate_zero_ell_raises(self, regulator):
        """ell_j = 0 raises ValueError."""
        with pytest.raises(ValueError, match="ell_j must be positive"):
            regulator.evaluate(np.array([1.0]), 0.0)

    def test_evaluate_negative_ell_raises(self, regulator):
        """Negative ell_j raises ValueError."""
        with pytest.raises(ValueError, match="ell_j must be positive"):
            regulator.evaluate(np.array([1.0]), -1.0)

    def test_sub_multiplicativity_disjoint(self, regulator):
        """G(X union Y) = G(X) * G(Y) for disjoint polymers."""
        phi = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ell = 1.0
        assert regulator.is_sub_multiplicative(phi, ell, split_index=3)

    def test_sub_multiplicativity_various_splits(self, regulator):
        """Sub-multiplicativity holds for various split points."""
        rng = np.random.RandomState(42)
        phi = rng.randn(10) * 2
        ell = 1.0
        for split in [1, 3, 5, 7, 9]:
            assert regulator.is_sub_multiplicative(phi, ell, split)

    def test_evaluate_per_site(self, regulator):
        """Per-site evaluation matches flat evaluation."""
        phi_sites = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]
        phi_flat = np.concatenate(phi_sites)
        ell = 1.0
        val_per_site = regulator.evaluate_per_site(phi_sites, ell)
        val_flat = regulator.evaluate(phi_flat, ell)
        assert val_per_site == pytest.approx(val_flat)

    def test_convolution_bound_parameters(self, regulator):
        """Convolution bound returns correct Pythagorean ell."""
        ell_j = 1.0
        C_norm = 0.5
        result = regulator.convolution_bound(ell_j, C_norm)
        expected_ell = np.sqrt(1.0 + 0.1 * 0.5)
        assert result['ell_pythagorean'] == pytest.approx(expected_ell)
        assert result['inflation_sq'] == pytest.approx(0.1 * 0.5)

    def test_stability_with_V_bounded(self, regulator):
        """G * e^{-V} is bounded for moderate fields."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        phi = np.array([0.5, 0.3, 0.1])
        result = regulator.stability_with_V(g, ell, phi)
        assert result['is_stable']
        assert np.isfinite(result['combined_exponent'])

    def test_stability_critical_field(self, regulator):
        """Critical field value gives maximum of G*e^{-V}."""
        g = 1.0
        ell = 1.0
        result = regulator.stability_with_V(g, ell, np.zeros(3))
        assert result['phi_sq_critical'] == pytest.approx(
            regulator.c_G / (2 * g * ell**2)
        )


# ======================================================================
# 5. WeightedPolymerNorm tests
# ======================================================================

class TestWeightedPolymerNorm:
    """Tests for WeightedPolymerNorm class."""

    def test_init_default(self):
        """Default construction with no arguments."""
        wpn = WeightedPolymerNorm()
        assert wpn.seminorm is not None
        assert wpn.regulator is not None
        assert wpn.norm_param is not None

    def test_evaluate_zero_activity(self):
        """Zero activity has zero norm."""
        wpn = WeightedPolymerNorm()
        K = lambda phi: 0.0
        val = wpn.evaluate(K, 0, G2_BARE_DEFAULT, dim=3, n_samples=10)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_positive_for_nonzero_activity(self):
        """Nonzero activity has positive norm."""
        wpn = WeightedPolymerNorm()
        K = lambda phi: float(np.sum(phi**2))
        val = wpn.evaluate(K, 0, G2_BARE_DEFAULT, dim=3, n_samples=20)
        assert val > 0

    def test_evaluate_at_config(self):
        """evaluate_at_config returns finite value."""
        wpn = WeightedPolymerNorm()
        K = lambda phi: float(np.sum(phi**2))
        phi = np.array([0.1, 0.2, 0.3])
        val = wpn.evaluate_at_config(K, phi, 0, G2_BARE_DEFAULT)
        assert np.isfinite(val)
        assert val >= 0

    def test_contraction_check_contracting(self):
        """Contraction detected when ratio < 1."""
        wpn = WeightedPolymerNorm()
        result = wpn.is_contracting(1.0, 0.5, 1.0)
        assert result['is_contracting'] is True
        assert result['ratio'] == pytest.approx(0.5)

    def test_contraction_check_not_contracting(self):
        """Non-contraction detected when ratio > 1."""
        wpn = WeightedPolymerNorm()
        result = wpn.is_contracting(0.5, 1.0, 1.0)
        assert result['is_contracting'] is False
        assert result['ratio'] == pytest.approx(2.0)

    def test_contraction_check_zero_norm(self):
        """Zero norm at scale j gives is_contracting only if K_{j+1} = 0."""
        wpn = WeightedPolymerNorm()
        result = wpn.is_contracting(0.0, 0.0, 1.0)
        assert result['is_contracting'] is True
        result2 = wpn.is_contracting(0.0, 1.0, 1.0)
        assert result2['is_contracting'] is False


# ======================================================================
# 6. ProductPropertyVerifier tests
# ======================================================================

class TestProductPropertyVerifier:
    """Tests for ProductPropertyVerifier (BBS Proposition 7.1.2)."""

    def test_product_constant_times_constant(self, zero_3d):
        """||c1 * c2|| <= ||c1|| * ||c2|| for constants."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=1.0))
        result = ppv.verify_product_polynomial(
            np.array([2.0]), np.array([3.0]), zero_3d
        )
        assert result['holds']
        assert result['norm_FG'] == pytest.approx(6.0)
        assert result['product_bound'] == pytest.approx(6.0)

    def test_product_quadratic_times_quadratic(self, zero_3d):
        """||phi^2 * phi^2|| <= ||phi^2|| * ||phi^2||."""
        ell = 1.0
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=ell))
        coeffs = np.array([0.0, 1.0])  # |phi|^2
        result = ppv.verify_product_polynomial(coeffs, coeffs, zero_3d)
        assert result['holds']

    def test_product_quartic_times_constant(self, zero_3d):
        """||g*phi^4 * c|| <= ||g*phi^4|| * ||c||."""
        g = 1.0
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=1.0))
        coeffs_V = np.array([0.0, 0.0, g])
        coeffs_c = np.array([5.0])
        result = ppv.verify_product_polynomial(coeffs_V, coeffs_c, zero_3d)
        assert result['holds']
        assert result['norm_FG'] == pytest.approx(5.0 * g)
        assert result['product_bound'] == pytest.approx(5.0 * g)

    def test_product_quadratic_times_quartic(self, zero_3d):
        """||phi^2 * phi^4|| <= ||phi^2|| * ||phi^4||."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=6, ell=1.0))
        coeffs_2 = np.array([0.0, 1.0])
        coeffs_4 = np.array([0.0, 0.0, 1.0])
        result = ppv.verify_product_polynomial(coeffs_2, coeffs_4, zero_3d)
        assert result['holds']

    def test_product_at_nonzero_basepoint(self):
        """Product property holds at nonzero basepoint."""
        z = np.array([1.0, 0.5, 0.2])
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=1.0))
        coeffs_F = np.array([1.0, 2.0])
        coeffs_G = np.array([0.5, 0.0, 1.0])
        result = ppv.verify_product_polynomial(coeffs_F, coeffs_G, z)
        assert result['holds']

    def test_product_with_natural_ell(self, zero_3d):
        """Product property holds at natural ell = g^{-1/4}."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=ell))
        coeffs_V = np.array([0.0, 0.0, g])
        coeffs_c = np.array([1.0])
        result = ppv.verify_product_polynomial(coeffs_V, coeffs_c, zero_3d)
        assert result['holds']

    def test_product_numerical_simple(self, unit_3d):
        """Numerical product property for simple functions."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=2, ell=0.5))
        F = lambda phi: 1.0 + 0.1 * float(np.sum(phi**2))
        G = lambda phi: 2.0
        result = ppv.verify_product_numerical(F, G, unit_3d, n_directions=30)
        assert result['holds']

    def test_product_label_theorem(self, zero_3d):
        """Polynomial verification labeled as THEOREM."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=1.0))
        result = ppv.verify_product_polynomial(
            np.array([1.0]), np.array([1.0]), zero_3d
        )
        assert result['label'] == 'THEOREM'

    def test_product_label_numerical(self, unit_3d):
        """Numerical verification labeled as NUMERICAL."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=2, ell=1.0))
        result = ppv.verify_product_numerical(
            lambda phi: 1.0, lambda phi: 1.0, unit_3d
        )
        assert result['label'] == 'NUMERICAL'

    def test_exponential_bound(self, zero_3d):
        """||e^F|| <= e^{||F||} (from iterated product property)."""
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=0.5))
        coeffs_F = np.array([0.0, 0.1])  # small quadratic
        result = ppv.verify_exponential(coeffs_F, zero_3d, n_terms=6)
        assert result['holds']

    def test_exponential_bound_quartic(self, zero_3d):
        """Exponential bound for quartic interaction at natural scale."""
        g = 0.5  # Moderate coupling
        ell = g ** (-0.25)
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=ell))
        coeffs = np.array([0.0, 0.0, g])
        result = ppv.verify_exponential(coeffs, zero_3d, n_terms=5)
        # ||V|| = 1 at natural scale, so e^{||V||} = e
        assert result['norm_F'] == pytest.approx(1.0, rel=1e-6)


# ======================================================================
# 7. GaussianConvolutionBound tests
# ======================================================================

class TestGaussianConvolutionBound:
    """Tests for GaussianConvolutionBound (BBS Section 7.3)."""

    def test_compute_ell_plus_pythagorean(self):
        """ell_+ = sqrt(ell^2 + C_norm)."""
        gcb = GaussianConvolutionBound()
        assert gcb.compute_ell_plus(3.0, 16.0) == pytest.approx(5.0)

    def test_compute_ell_plus_zero_covariance(self):
        """Zero covariance leaves ell unchanged."""
        gcb = GaussianConvolutionBound()
        assert gcb.compute_ell_plus(2.5, 0.0) == pytest.approx(2.5)

    def test_compute_ell_plus_zero_ell(self):
        """ell=0 gives ell_+ = sqrt(C_norm)."""
        gcb = GaussianConvolutionBound()
        assert gcb.compute_ell_plus(0.0, 4.0) == pytest.approx(2.0)

    def test_verify_bound_constant(self):
        """Convolution bound trivial for constant F."""
        gcb = GaussianConvolutionBound(
            seminorm=BBSTzSeminorm(p_N=3, ell=1.0)
        )
        coeffs = np.array([5.0])  # constant
        z = np.array([0.0, 0.0, 0.0])
        result = gcb.verify_bound_polynomial(coeffs, z, 1.0, 0.1, n_samples=100)
        assert result['holds']
        # For constant: LHS = RHS = 5.0
        assert result['lhs_expectation'] == pytest.approx(5.0, rel=0.05)

    def test_verify_bound_quadratic(self):
        """Convolution bound for |phi|^2.

        NUMERICAL: The BBS Gaussian convolution bound is:
            E_C[||F(. + zeta)||_{T_z(ell)}] <= C(d) * ||F||_{T_z(ell_+)}
        where C(d) is a dimension-dependent constant. For d=3, C(d) ~ 3-4
        because the D^0 term involves E[|zeta|^2] = d * w^2.

        The ratio LHS/RHS should be bounded by a moderate constant
        depending on the field dimension d.
        """
        gcb = GaussianConvolutionBound(
            seminorm=BBSTzSeminorm(p_N=3, ell=1.0)
        )
        coeffs = np.array([0.0, 1.0])  # |phi|^2
        z = np.zeros(3)
        result = gcb.verify_bound_polynomial(coeffs, z, 1.0, 0.5, n_samples=1000)
        # The ratio includes a dimension-dependent constant C(d)
        # For d=3: ratio should be O(d) ~ 3-5
        assert result['ratio'] < 10.0  # bounded by moderate constant
        assert result['lhs_expectation'] > 0  # nontrivial
        assert result['rhs_bound'] > 0

    def test_s3_fluctuation_scale(self):
        """w_j^2 correct on S^3."""
        w2 = GaussianConvolutionBound.s3_fluctuation_scale(0, R_PHYSICAL_FM, M_DEFAULT)
        expected = 1.0 / (4 * np.pi**2 * R_PHYSICAL_FM**2)
        assert w2 == pytest.approx(expected)

    def test_s3_fluctuation_decreases_with_j(self):
        """Fluctuation scale decreases exponentially with j."""
        w2_0 = GaussianConvolutionBound.s3_fluctuation_scale(0)
        w2_3 = GaussianConvolutionBound.s3_fluctuation_scale(3)
        assert w2_3 < w2_0
        assert w2_3 == pytest.approx(w2_0 * M_DEFAULT**(-6))

    def test_ell_plus_increases_monotonically(self):
        """ell_+ increases with C_norm."""
        gcb = GaussianConvolutionBound()
        ell = 1.0
        vals = [gcb.compute_ell_plus(ell, c) for c in [0.0, 0.1, 0.5, 1.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] > vals[i]

    def test_verify_bound_numerical_simple(self):
        """Numerical convolution bound for simple function."""
        gcb = GaussianConvolutionBound(
            seminorm=BBSTzSeminorm(p_N=2, ell=1.0)
        )
        F = lambda phi: 1.0 + 0.1 * float(np.sum(phi**2))
        z = np.zeros(3)
        result = gcb.verify_bound_numerical(
            F, z, 1.0, 0.1, n_mc_samples=50, n_directions=10
        )
        assert result['holds']


# ======================================================================
# 8. GaugeCovariantTaylor tests
# ======================================================================

class TestGaugeCovariantTaylor:
    """Tests for GaugeCovariantTaylor (gauge theory adaptation)."""

    def test_init_default(self):
        """Default N_c=2, p_N=5."""
        gct = GaugeCovariantTaylor()
        assert gct.N_c == N_C_DEFAULT
        assert gct.dim_adj == DIM_ADJ_SU2
        assert gct.p_N == P_N_DEFAULT

    def test_init_invalid_Nc_raises(self):
        """N_c < 2 raises ValueError."""
        with pytest.raises(ValueError, match="N_c must be >= 2"):
            GaugeCovariantTaylor(N_c=1)

    def test_adjoint_action_su2(self):
        """[A, X] = A x X for su(2) (cross product)."""
        A = np.array([1.0, 0.0, 0.0])
        X = np.array([0.0, 1.0, 0.0])
        result = GaugeCovariantTaylor.adjoint_action(A, X)
        expected = np.cross(A, X)  # [0, 0, 1]
        np.testing.assert_allclose(result, expected)

    def test_adjoint_action_antisymmetric(self):
        """[A, X] = -[X, A] (antisymmetry of Lie bracket)."""
        rng = np.random.RandomState(42)
        A = rng.randn(3)
        X = rng.randn(3)
        AX = GaugeCovariantTaylor.adjoint_action(A, X)
        XA = GaugeCovariantTaylor.adjoint_action(X, A)
        np.testing.assert_allclose(AX, -XA, atol=1e-14)

    def test_adjoint_action_jacobi(self):
        """Jacobi identity: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0."""
        rng = np.random.RandomState(42)
        A = rng.randn(3)
        B = rng.randn(3)
        C = rng.randn(3)
        adj = GaugeCovariantTaylor.adjoint_action
        t1 = adj(A, adj(B, C))
        t2 = adj(B, adj(C, A))
        t3 = adj(C, adj(A, B))
        np.testing.assert_allclose(t1 + t2 + t3, np.zeros(3), atol=1e-14)

    def test_covariant_derivative_k0(self):
        """D_A^0 F = F(A)."""
        gct = GaugeCovariantTaylor()
        F = lambda A: float(np.sum(A**2))
        A = np.array([1.0, 2.0, 3.0])
        h = np.array([0.1, 0.2, 0.3])
        assert gct.covariant_derivative(F, A, h, k=0) == pytest.approx(14.0)

    def test_covariant_derivative_k1_linear(self):
        """D_A^1 of linear function is constant."""
        gct = GaugeCovariantTaylor()
        c = np.array([1.0, 2.0, 3.0])
        F = lambda A: float(np.dot(c, A))
        A = np.array([0.0, 0.0, 0.0])
        h = c / np.linalg.norm(c)
        val = gct.covariant_derivative(F, A, h, k=1)
        assert val == pytest.approx(np.linalg.norm(c), rel=1e-3)

    def test_covariant_derivative_k2_quadratic(self):
        """D_A^2 of |A|^2 is constant = 2."""
        gct = GaugeCovariantTaylor()
        F = lambda A: float(np.sum(A**2))
        A = np.array([1.0, 0.0, 0.0])
        h = np.array([1.0, 0.0, 0.0])
        val = gct.covariant_derivative(F, A, h, k=2)
        assert val == pytest.approx(2.0, rel=1e-2)

    def test_operator_norm_quadratic(self):
        """||D^1(|A|^2)|| at A=(1,0,0) should be ~ 2."""
        gct = GaugeCovariantTaylor()
        F = lambda A: float(np.sum(A**2))
        A = np.array([1.0, 0.0, 0.0])
        val = gct.covariant_derivative_operator_norm(F, A, k=1, n_directions=50)
        assert val == pytest.approx(2.0, rel=0.15)

    def test_gauge_covariant_seminorm_positive(self):
        """Gauge-covariant seminorm is positive for nonzero F."""
        gct = GaugeCovariantTaylor(p_N=3)
        F = lambda A: 1.0 + float(np.sum(A**2))
        A = np.array([0.5, 0.3, 0.1])
        val = gct.gauge_covariant_seminorm(F, A, ell=1.0)
        assert val > 0

    def test_gauge_covariant_seminorm_zero_func(self):
        """Zero function has zero seminorm."""
        gct = GaugeCovariantTaylor(p_N=3)
        F = lambda A: 0.0
        A = np.zeros(3)
        val = gct.gauge_covariant_seminorm(F, A, ell=1.0)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_gauge_invariant_extraction_constant(self):
        """Loc of constant F is the constant itself."""
        gct = GaugeCovariantTaylor()
        F = lambda A: 5.0
        A = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = gct.gauge_invariant_extraction(F, A, dim=9)
        assert result['constant'] == pytest.approx(5.0)
        assert abs(result['remainder']) < 0.1

    def test_gauge_invariant_extraction_quadratic(self):
        """Loc of |A|^2 extracts the quadratic term."""
        gct = GaugeCovariantTaylor()
        F = lambda A: float(np.sum(A**2))
        A = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = gct.gauge_invariant_extraction(F, A, dim=9)
        assert result['constant'] == pytest.approx(0.0, abs=1e-6)
        # The quadratic should capture most of |A|^2
        assert result['quadratic'] > 0


# ======================================================================
# 9. Comparison with legacy TPhiSeminorm
# ======================================================================

class TestCompareWithLegacy:
    """Tests for compare_with_legacy utility."""

    def test_comparison_returns_all_keys(self, unit_3d):
        """Comparison returns expected keys."""
        K = lambda phi: float(np.sum(phi**2))
        result = compare_with_legacy(K, unit_3d, j=0, g2=G2_BARE_DEFAULT)
        assert 'bbs_ell_j' in result
        assert 'legacy_h_j' in result
        assert 'bbs_norm' in result
        assert 'ell_over_h_ratio' in result

    def test_ell_j_differs_from_h_j(self, unit_3d):
        """BBS ell_j != legacy h_j (different formulas)."""
        K = lambda phi: float(np.sum(phi**2))
        result = compare_with_legacy(K, unit_3d, j=0, g2=G2_BARE_DEFAULT)
        # ell_0 = g^{-1/4} ~ 0.631, h_0 = M^0 * sqrt(g) ~ 2.506
        assert result['bbs_ell_j'] != pytest.approx(result['legacy_h_j'], rel=0.1)

    def test_ell_less_than_h_at_scale_0(self, unit_3d):
        """At scale 0, ell_0 < h_0 (BBS uses smaller field scale)."""
        K = lambda phi: float(np.sum(phi**2))
        result = compare_with_legacy(K, unit_3d, j=0, g2=G2_BARE_DEFAULT)
        assert result['bbs_ell_j'] < result['legacy_h_j']

    def test_bbs_norm_positive(self, unit_3d):
        """BBS norm is positive for nonzero activity."""
        K = lambda phi: float(np.sum(phi**2))
        result = compare_with_legacy(K, unit_3d, j=0, g2=G2_BARE_DEFAULT)
        assert result['bbs_norm'] > 0


# ======================================================================
# 10. Edge cases and physical parameter tests
# ======================================================================

class TestEdgeCases:
    """Edge cases: ell -> 0, ell -> inf, g -> 0, different p_N."""

    def test_ell_approaches_zero(self, zero_3d):
        """As ell -> 0, seminorm -> |F(z)| (only D^0 survives)."""
        coeffs = np.array([1.0, 2.0, 3.0])  # 1 + 2|phi|^2 + 3|phi|^4
        vals = []
        for ell in [1.0, 0.1, 0.01, 0.001]:
            sn = BBSTzSeminorm(p_N=5, ell=ell)
            vals.append(sn.evaluate_polynomial(coeffs, zero_3d))
        # As ell -> 0, val -> F(0) = 1.0
        assert vals[-1] == pytest.approx(1.0, abs=0.01)

    def test_large_ell(self, zero_3d):
        """Large ell makes higher derivatives dominate."""
        coeffs = np.array([0.0, 0.0, 1.0])  # |phi|^4
        sn_small = BBSTzSeminorm(p_N=5, ell=0.1)
        sn_large = BBSTzSeminorm(p_N=5, ell=10.0)
        # With ell=10: term at k=4 is 10^4/24*24 = 10^4
        assert sn_large.evaluate_polynomial(coeffs, zero_3d) > \
               sn_small.evaluate_polynomial(coeffs, zero_3d)

    def test_g_approaches_zero(self, norm_param):
        """As g -> 0, ell_j -> infinity (free theory)."""
        ells = [norm_param.at_scale(0, g) for g in [6.28, 1.0, 0.1, 0.01]]
        for i in range(len(ells) - 1):
            assert ells[i + 1] > ells[i]

    def test_p_N_zero(self, zero_3d):
        """p_N=0 gives just |F(z)| (no derivatives)."""
        sn = BBSTzSeminorm(p_N=0, ell=1.0)
        coeffs = np.array([1.0, 2.0, 3.0])
        # At z=0: F(0) = 1.0 (only constant survives)
        assert sn.evaluate_polynomial(coeffs, zero_3d) == pytest.approx(1.0)

    def test_p_N_1(self, zero_3d):
        """p_N=1 gives D^0 + D^1 terms only."""
        sn = BBSTzSeminorm(p_N=1, ell=1.0)
        coeffs = np.array([0.0, 1.0])  # |phi|^2
        # At z=0: D^0 = 0, D^1 = 0 (|phi|^2 has zero gradient at origin)
        assert sn.evaluate_polynomial(coeffs, zero_3d) == pytest.approx(0.0)

    def test_large_p_N(self, zero_3d):
        """Large p_N doesn't change result for polynomial of bounded degree."""
        coeffs = np.array([0.0, 0.0, 1.0])  # |phi|^4, degree 4
        sn5 = BBSTzSeminorm(p_N=5, ell=1.0)
        sn20 = BBSTzSeminorm(p_N=20, ell=1.0)
        # D^k = 0 for k > 4, so p_N=5 and p_N=20 give same result
        assert sn5.evaluate_polynomial(coeffs, zero_3d) == \
               pytest.approx(sn20.evaluate_polynomial(coeffs, zero_3d))


# ======================================================================
# 11. Physical consistency tests
# ======================================================================

class TestPhysicalConsistency:
    """Tests for physical parameter consistency."""

    def test_ell_0_numerical_value(self):
        """ell_0 = g^{-1/4} ~ 0.631 for g^2 = 6.28."""
        np_obj = NormParameter()
        ell_0 = np_obj.at_scale(0, G2_BARE_DEFAULT)
        assert ell_0 == pytest.approx(0.631, abs=0.01)

    def test_quartic_order_one_at_natural_scale(self):
        """||V||_{T_0(ell_0)} = 1 at origin (the BBS normalization)."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        sn = BBSTzSeminorm(p_N=5, ell=ell)
        val = sn.quartic_seminorm(g, np.zeros(3))
        assert val == pytest.approx(1.0, rel=1e-10)

    def test_beta_0_matches_known_value(self):
        """beta_0 for SU(2) matches 22/(3*16*pi^2)."""
        np_obj = NormParameter()
        expected = 22.0 / (3.0 * 16.0 * np.pi**2)
        assert np_obj.beta_0 == pytest.approx(expected)

    def test_regulator_constant_reasonable(self):
        """c_G = 0.1 ~ 1/(2*p_N) for p_N=5."""
        reg = BBSRegulator()
        assert reg.c_G == pytest.approx(1.0 / (2 * P_N_DEFAULT))

    def test_fluctuation_scale_j0_on_s3(self):
        """w_0^2 on S^3 is O(1/R^2)."""
        w2 = GaussianConvolutionBound.s3_fluctuation_scale(0)
        assert w2 > 0
        expected_order = 1.0 / R_PHYSICAL_FM**2
        assert w2 == pytest.approx(expected_order / (4 * np.pi**2))

    def test_inflation_small_at_uv(self):
        """At UV scales (large j), inflation ell_+ ~ ell_j (w_j -> 0)."""
        np_obj = NormParameter()
        g2 = G2_BARE_DEFAULT
        j = 6  # UV scale
        ell_j = np_obj.at_scale(j, g2)
        w2_j = GaussianConvolutionBound.s3_fluctuation_scale(j)
        ell_plus = np_obj.inflate(ell_j, np.sqrt(w2_j))
        ratio = ell_plus / ell_j
        # Inflation should be tiny at UV
        assert ratio < 1.001

    def test_product_property_at_physical_coupling(self):
        """Product property verified at g^2 = 6.28."""
        g = G2_BARE_DEFAULT
        ell = g ** (-0.25)
        ppv = ProductPropertyVerifier(BBSTzSeminorm(p_N=5, ell=ell))
        coeffs_V = np.array([0.0, 0.0, g])
        coeffs_c = np.array([1.0])
        result = ppv.verify_product_polynomial(
            coeffs_V, coeffs_c, np.zeros(3)
        )
        assert result['holds']

    def test_spacetime_dim_is_4(self):
        """d=4 for YM on S^3 x R."""
        assert SPACETIME_DIM == 4

    def test_p_N_at_least_5_for_d4(self):
        """BBS requires p_N >= 5 for d=4."""
        assert P_N_DEFAULT >= 5
