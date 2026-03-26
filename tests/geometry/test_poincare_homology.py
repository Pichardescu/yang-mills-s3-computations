"""
Tests for the Poincare homology sphere spectrum module.

Verifies the corrected I*-invariant spectrum of Laplacians on S^3/I*
with proper coexact/exact split (Session 2+ corrections).

Mathematical ground truth:
  - Molien series: M(t) = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30))
  - First nontrivial scalar invariant: l=12, eigenvalue 168/R^2
  - Coexact 1-form gap: k=1, eigenvalue 4/R^2 (SAME as S^3, corrected from 5)
  - Coexact 1-form second mode: k=11, eigenvalue 144/R^2
  - CMB: l=2 through l=11 all suppressed
"""

import pytest
import numpy as np
from yang_mills_s3.geometry.poincare_homology import PoincareHomology, HBAR_C_MEV_FM


@pytest.fixture
def ph():
    """Create a PoincareHomology instance."""
    return PoincareHomology()


# ======================================================================
# Conjugacy class validation
# ======================================================================

class TestConjugacyClasses:
    """Verify the conjugacy class structure of I*."""

    def test_group_order(self, ph):
        """I* has order 120."""
        assert ph.group_order == 120

    def test_nine_conjugacy_classes(self, ph):
        """I* has 9 conjugacy classes."""
        assert len(ph.conjugacy_classes) == 9

    def test_class_sizes_sum_to_120(self, ph):
        """Class sizes must sum to |I*| = 120."""
        total = sum(size for size, _ in ph.conjugacy_classes)
        assert total == 120

    def test_class_sizes(self, ph):
        """Verify individual class sizes: 1, 1, 12, 12, 12, 12, 20, 20, 30."""
        sizes = sorted(size for size, _ in ph.conjugacy_classes)
        assert sizes == [1, 1, 12, 12, 12, 12, 20, 20, 30]

    def test_identity_angle(self, ph):
        """Identity has theta = 0."""
        identity_classes = [(s, t) for s, t in ph.conjugacy_classes if s == 1 and abs(t) < 1e-10]
        assert len(identity_classes) == 1

    def test_central_element_angle(self, ph):
        """Central element (-I) has theta = 2*pi."""
        central_classes = [(s, t) for s, t in ph.conjugacy_classes
                           if s == 1 and abs(t - 2 * np.pi) < 1e-10]
        assert len(central_classes) == 1


# ======================================================================
# Character formula tests
# ======================================================================

class TestCharacterSU2:
    """Test the SU(2) character formula chi_l(theta)."""

    def test_identity_gives_dimension(self, ph):
        """chi_l(0) = l+1 (dimension of V_l)."""
        for l in range(25):
            assert abs(ph.character_su2(l, 0.0) - (l + 1)) < 1e-12

    def test_central_element(self, ph):
        """chi_l(2*pi) = (-1)^l * (l+1)."""
        for l in range(25):
            expected = ((-1) ** l) * (l + 1)
            assert abs(ph.character_su2(l, 2 * np.pi) - expected) < 1e-10

    def test_pi_rotation(self, ph):
        """chi_l(pi) = sin((l+1)*pi/2)."""
        for l in range(25):
            expected = np.sin((l + 1) * np.pi / 2)
            assert abs(ph.character_su2(l, np.pi) - expected) < 1e-12

    def test_trivial_rep_is_one(self, ph):
        """chi_0(theta) = 1 for all theta."""
        angles = [0.0, np.pi / 5, np.pi / 3, np.pi / 2, np.pi, 4 * np.pi / 3, 2 * np.pi]
        for theta in angles:
            assert abs(ph.character_su2(0, theta) - 1.0) < 1e-12

    def test_fundamental_rep(self, ph):
        """chi_1(theta) = 2*cos(theta/2)."""
        angles = [np.pi / 5, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
        for theta in angles:
            expected = 2 * np.cos(theta / 2)
            assert abs(ph.character_su2(1, theta) - expected) < 1e-12

    def test_clebsch_gordan(self, ph):
        """chi_l * chi_1 = chi_{l+1} + chi_{l-1} (CG for V_l x V_1)."""
        angles = [np.pi / 5, np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 5, np.pi]
        for l in range(1, 15):
            for theta in angles:
                product = ph.character_su2(l, theta) * ph.character_su2(1, theta)
                cg_sum = ph.character_su2(l + 1, theta) + ph.character_su2(l - 1, theta)
                assert abs(product - cg_sum) < 1e-10, \
                    f"CG failed at l={l}, theta={theta:.4f}"

    def test_adjoint_character(self, ph):
        """chi_2(theta) = 2*cos(theta) + 1 (adjoint of SU(2))."""
        angles = [np.pi / 5, np.pi / 3, 2 * np.pi / 3, np.pi]
        for theta in angles:
            expected = 2 * np.cos(theta) + 1
            assert abs(ph.character_su2(2, theta) - expected) < 1e-12


# ======================================================================
# Scalar multiplicity tests
# ======================================================================

class TestTrivialMultiplicity:
    """Test the I*-invariant multiplicity m(l)."""

    def test_m0_is_one(self, ph):
        """m(0) = 1: the constant function is always invariant."""
        assert ph.trivial_multiplicity(0) == 1

    def test_always_nonnegative_integer(self, ph):
        """m(l) must be a non-negative integer for all l."""
        for l in range(100):
            m = ph.trivial_multiplicity(l)
            assert isinstance(m, int)
            assert m >= 0, f"m({l}) = {m} < 0"

    def test_first_nontrivial_at_l12(self, ph):
        """First nontrivial I*-invariant scalar is at l=12."""
        for l in range(1, 12):
            assert ph.trivial_multiplicity(l) == 0, \
                f"m({l}) = {ph.trivial_multiplicity(l)}, expected 0"
        assert ph.trivial_multiplicity(12) == 1

    def test_known_invariant_levels(self, ph):
        """Verify m(l) for all nonzero levels up to l=60."""
        expected_nonzero = {
            0: 1, 12: 1, 20: 1, 24: 1, 30: 1, 32: 1, 36: 1,
            40: 1, 42: 1, 44: 1, 48: 1, 50: 1, 52: 1, 54: 1,
            56: 1, 60: 2
        }
        for l in range(61):
            m = ph.trivial_multiplicity(l)
            if l in expected_nonzero:
                assert m == expected_nonzero[l], \
                    f"m({l}) = {m}, expected {expected_nonzero[l]}"
            else:
                assert m == 0, f"m({l}) = {m}, expected 0"

    def test_negative_l_returns_zero(self, ph):
        """m(l) = 0 for l < 0."""
        assert ph.trivial_multiplicity(-1) == 0
        assert ph.trivial_multiplicity(-10) == 0


# ======================================================================
# Molien series verification
# ======================================================================

class TestMolienSeries:
    """Verify character computation against closed-form Molien series."""

    def test_molien_matches_character(self, ph):
        """Character computation must match Molien series for all l <= 60."""
        ok, mismatches = ph.verify_against_molien(60)
        if not ok:
            details = ", ".join(
                f"l={l}: char={mc}, mol={mm}" for l, mc, mm in mismatches
            )
            pytest.fail(f"Molien series mismatch: {details}")

    def test_molien_coefficients_nonnegative(self, ph):
        """Molien series coefficients must be non-negative."""
        molien = ph.molien_series_coefficients(100)
        for l, m in enumerate(molien):
            assert m >= 0, f"Molien coefficient at l={l} is {m} < 0"

    def test_molien_m0_is_one(self, ph):
        """Molien series starts with m(0) = 1."""
        molien = ph.molien_series_coefficients(10)
        assert molien[0] == 1

    def test_molien_first_nonzero_at_12(self, ph):
        """After l=0, next nonzero Molien coefficient is at l=12."""
        molien = ph.molien_series_coefficients(15)
        for l in range(1, 12):
            assert molien[l] == 0
        assert molien[12] == 1

    def test_molien_extended(self, ph):
        """Extended Molien verification up to l=100."""
        ok, mismatches = ph.verify_against_molien(100)
        assert ok, f"Extended Molien verification failed at l={[x[0] for x in mismatches]}"


# ======================================================================
# Scalar spectrum on S^3/I*
# ======================================================================

class TestScalarSpectrum:
    """Test the scalar Laplacian spectrum on S^3/I*."""

    def test_zero_mode_exists(self, ph):
        """l=0 mode (constants) always survives."""
        spectrum = ph.scalar_spectrum(l_max=60, R=1.0)
        assert spectrum[0] == (0.0, 1)

    def test_scalar_gap_is_168(self, ph):
        """First nontrivial scalar eigenvalue on S^3/I* is 168/R^2."""
        spectrum = ph.scalar_spectrum(l_max=60, R=1.0)
        nontrivial = [(ev, m) for ev, m in spectrum if ev > 0]
        assert len(nontrivial) > 0
        assert abs(nontrivial[0][0] - 168.0) < 1e-10, \
            f"Scalar gap should be 168, got {nontrivial[0][0]}"

    def test_scalar_gap_ratio_56(self, ph):
        """Scalar gap on S^3/I* is 56x larger than on S^3."""
        # S^3: gap = 3/R^2 (l=1)
        # S^3/I*: gap = 168/R^2 (l=12)
        spectrum = ph.scalar_spectrum(l_max=60, R=1.0)
        nontrivial = [(ev, m) for ev, m in spectrum if ev > 0]
        gap_poincare = nontrivial[0][0]
        gap_s3 = 3.0
        assert abs(gap_poincare / gap_s3 - 56.0) < 1e-10

    def test_radius_scaling(self, ph):
        """Eigenvalues scale as 1/R^2."""
        R = 3.7
        spectrum = ph.scalar_spectrum(l_max=20, R=R)
        nontrivial = [(ev, m) for ev, m in spectrum if ev > 0]
        assert abs(nontrivial[0][0] - 168.0 / R**2) < 1e-10


# ======================================================================
# Coexact 1-form spectrum tests (CORRECTED)
# ======================================================================

class TestCoexactSpectrum:
    """Test the CORRECTED coexact 1-form spectrum on S^3/I*."""

    def test_k1_survives_with_mult_3(self, ph):
        """
        THEOREM: k=1 coexact mode survives on S^3/I* with multiplicity 3.

        This comes from the anti-self-dual sector: V_0 under SU(2)_L.
        m(0) = 1, and dim(V_{k+1}) = dim(V_2) = 3.
        So n_ASD = m(0) * (1+2) = 1 * 3 = 3.
        """
        n = ph.coexact_invariant_multiplicity(1)
        assert n == 3, f"Expected 3 surviving k=1 modes, got {n}"

    def test_k1_is_anti_self_dual_only(self, ph):
        """At k=1, only anti-self-dual modes survive (SD has m(2)=0)."""
        detail = ph.coexact_invariant_detail(1)
        assert detail['n_sd'] == 0, "Self-dual should be 0 at k=1"
        assert detail['n_asd'] == 3, "Anti-self-dual should be 3 at k=1"

    def test_k2_through_k10_vanish(self, ph):
        """No coexact modes survive for k=2,...,10."""
        for k in range(2, 11):
            n = ph.coexact_invariant_multiplicity(k)
            assert n == 0, f"k={k}: expected 0 modes, got {n}"

    def test_k11_survives_with_mult_11(self, ph):
        """
        THEOREM: k=11 is the second surviving coexact level, with mult 11.

        Self-dual: V_{12} under SU(2)_L, m(12) = 1, times dim(V_{10}) = 11.
        Anti-self-dual: V_{10} under SU(2)_L, m(10) = 0, times dim(V_{12}) = 13.
        Total: 1*11 + 0*13 = 11.
        """
        n = ph.coexact_invariant_multiplicity(11)
        assert n == 11, f"Expected 11 surviving k=11 modes, got {n}"

    def test_k11_is_self_dual_only(self, ph):
        """At k=11, only self-dual modes survive."""
        detail = ph.coexact_invariant_detail(11)
        assert detail['n_sd'] == 11, "Self-dual should be 11 at k=11"
        assert detail['n_asd'] == 0, "Anti-self-dual should be 0 at k=11"

    def test_k13_survives_with_mult_15(self, ph):
        """k=13: anti-self-dual from m(12)=1 in V_{12}, times dim(V_{14})=15."""
        n = ph.coexact_invariant_multiplicity(13)
        assert n == 15, f"Expected 15 modes at k=13, got {n}"
        detail = ph.coexact_invariant_detail(13)
        assert detail['n_sd'] == 0
        assert detail['n_asd'] == 15

    def test_coexact_gap_eigenvalue_4(self, ph):
        """Coexact gap on S^3/I* is 4/R^2 (same as S^3)."""
        spectrum = ph.coexact_spectrum(k_max=30, R=1.0)
        assert len(spectrum) > 0
        ev, mult, k = spectrum[0]
        assert abs(ev - 4.0) < 1e-10, f"Coexact gap should be 4, got {ev}"
        assert k == 1

    def test_second_coexact_eigenvalue_144(self, ph):
        """Second coexact eigenvalue on S^3/I* is 144/R^2 (k=11)."""
        spectrum = ph.coexact_spectrum(k_max=30, R=1.0)
        assert len(spectrum) >= 2
        ev, mult, k = spectrum[1]
        assert abs(ev - 144.0) < 1e-10, f"Second coexact should be 144, got {ev}"
        assert k == 11

    def test_gap_ratio_second_to_first(self, ph):
        """Ratio of second to first coexact eigenvalue: 144/4 = 36."""
        spectrum = ph.coexact_spectrum(k_max=30, R=1.0)
        ratio = spectrum[1][0] / spectrum[0][0]
        assert abs(ratio - 36.0) < 1e-10

    def test_mass_ratio_second_to_first(self, ph):
        """Mass ratio of second to first mode: 12/2 = 6."""
        levels = ph.invariant_levels_coexact(30)
        k1 = levels[0][0]
        k2 = levels[1][0]
        mass_ratio = (k2 + 1) / (k1 + 1)
        assert abs(mass_ratio - 6.0) < 1e-10

    def test_first_five_coexact_levels(self, ph):
        """Verify the first five surviving coexact levels."""
        levels = ph.invariant_levels_coexact(60)
        expected_k = [1, 11, 13, 19, 21]
        for i, expected in enumerate(expected_k):
            assert levels[i][0] == expected, \
                f"Level {i}: expected k={expected}, got k={levels[i][0]}"

    def test_multiplicities_nonnegative(self, ph):
        """All coexact multiplicities are non-negative integers."""
        for k in range(1, 100):
            n = ph.coexact_invariant_multiplicity(k)
            assert n >= 0 and isinstance(n, (int, np.integer))

    def test_invariant_leq_s3(self, ph):
        """I*-invariant count <= S^3 total for all k."""
        for k in range(1, 60):
            n_inv = ph.coexact_invariant_multiplicity(k)
            n_s3 = 2 * k * (k + 2)
            assert n_inv <= n_s3, f"k={k}: {n_inv} > {n_s3}"

    def test_radius_scaling_coexact(self, ph):
        """Coexact eigenvalues scale as 1/R^2."""
        R = 2.5
        spectrum = ph.coexact_spectrum(k_max=20, R=R)
        for ev, mult, k in spectrum:
            expected = (k + 1)**2 / R**2
            assert abs(ev - expected) < 1e-10


# ======================================================================
# Exact 1-form spectrum tests
# ======================================================================

class TestExactSpectrum:
    """Test the exact 1-form spectrum on S^3/I*."""

    def test_first_exact_at_l12(self, ph):
        """First surviving exact 1-form is at l=12, eigenvalue 168/R^2."""
        exact = ph.exact_spectrum(l_max=30, R=1.0)
        assert len(exact) > 0
        ev, mult, l = exact[0]
        assert abs(ev - 168.0) < 1e-10
        assert l == 12

    def test_exact_multiplicity_at_l12(self, ph):
        """At l=12: m(12)=1, dim(V_12 under SU(2)_R)=13, so 13 exact modes."""
        n = ph.exact_invariant_multiplicity(12)
        assert n == 13

    def test_no_exact_for_l1_to_l11(self, ph):
        """No exact 1-forms survive for l=1,...,11."""
        for l in range(1, 12):
            assert ph.exact_invariant_multiplicity(l) == 0, \
                f"l={l}: expected 0 exact modes"


# ======================================================================
# Yang-Mills spectrum tests
# ======================================================================

class TestYMSpectrum:
    """Test the Yang-Mills spectrum on S^3/I*."""

    def test_ym_gap_is_4(self, ph):
        """YM gap on S^3/I* is 4/R^2 (CORRECTED from 5)."""
        ym = ph.ym_spectrum(k_max=30, R=1.0, gauge_group='SU(2)')
        assert len(ym) > 0
        ev, mult, k = ym[0]
        assert abs(ev - 4.0) < 1e-10, f"YM gap should be 4, got {ev}"

    def test_ym_gap_multiplicity_su2(self, ph):
        """YM gap multiplicity for SU(2): 3 (coexact) x 3 (adjoint) = 9."""
        ym = ph.ym_spectrum(k_max=5, R=1.0, gauge_group='SU(2)')
        ev, mult, k = ym[0]
        assert mult == 9, f"Expected multiplicity 9, got {mult}"

    def test_ym_gap_multiplicity_su3(self, ph):
        """YM gap multiplicity for SU(3): 3 (coexact) x 8 (adjoint) = 24."""
        ym = ph.ym_spectrum(k_max=5, R=1.0, gauge_group='SU(3)')
        ev, mult, k = ym[0]
        assert mult == 24, f"Expected multiplicity 24, got {mult}"

    def test_ym_gap_same_as_s3(self, ph):
        """YM gap on S^3/I* equals that on S^3."""
        gap = ph.ym_gap(R=1.0)
        assert abs(gap['gap_eigenvalue'] - 4.0) < 1e-10
        assert gap['same_as_s3'] is True

    def test_ym_second_mode_at_k11(self, ph):
        """Second YM mode on S^3/I* is at k=11."""
        ym = ph.ym_spectrum(k_max=30, R=1.0, gauge_group='SU(2)')
        assert len(ym) >= 2
        ev, mult, k = ym[1]
        assert k == 11
        assert abs(ev - 144.0) < 1e-10


# ======================================================================
# Gap comparison tests (CORRECTED)
# ======================================================================

class TestGapComparison:
    """Test the gap comparison between S^3 and S^3/I*."""

    def test_scalar_gap_ratio_56(self, ph):
        """Scalar gap ratio S^3/I* to S^3 is 56."""
        comp = ph.gap_comparison(R_fm=2.2)
        assert abs(comp['scalar']['ratio'] - 56.0) < 1e-10

    def test_coexact_gap_ratio_1(self, ph):
        """Coexact (YM physical) gap ratio is 1 (CORRECTED: was using 5/R^2)."""
        comp = ph.gap_comparison(R_fm=2.2)
        assert abs(comp['coexact']['ratio'] - 1.0) < 1e-10, \
            f"Coexact gap ratio should be 1, got {comp['coexact']['ratio']}"

    def test_coexact_gap_uses_4_not_5(self, ph):
        """CORRECTED: Coexact gap coefficient is 4, not 5."""
        comp = ph.gap_comparison(R_fm=2.2)
        assert comp['coexact']['s3']['eigenvalue_coeff'] == 4
        assert comp['coexact']['poincare']['eigenvalue_coeff'] == 4

    def test_second_coexact_ratio(self, ph):
        """Second coexact mode: 144/9 = 16x ratio."""
        comp = ph.gap_comparison(R_fm=2.2)
        assert abs(comp['second_coexact']['ratio'] - 16.0) < 1e-10

    def test_physical_coexact_gap_mev(self, ph):
        """Coexact gap at R=2.2 fm: sqrt(4)*hbar_c/R = 2*197.33/2.2 = 179.4 MeV."""
        comp = ph.gap_comparison(R_fm=2.2)
        expected = np.sqrt(4) * HBAR_C_MEV_FM / 2.2
        actual = comp['coexact']['s3']['gap_mev']
        assert abs(actual - expected) < 0.1

    def test_physical_scalar_gap_s3(self, ph):
        """Scalar gap on S^3 at R=2.2 fm: sqrt(3)*hbar_c/R."""
        comp = ph.gap_comparison(R_fm=2.2)
        expected = np.sqrt(3) * HBAR_C_MEV_FM / 2.2
        actual = comp['scalar']['s3']['gap_mev']
        assert abs(actual - expected) < 0.1


# ======================================================================
# adjoint scenario tests
# ======================================================================

class TestAdjointSpectrum:
    """Test the adjoint scenario where I* acts on gauge fiber too."""

    def test_adjoint_gap_at_k1(self, ph):
        """In the compact topology framework, the first level is k=1."""
        levels = ph.invariant_levels_coexact_adjoint(30)
        assert len(levels) > 0
        assert levels[0][0] == 1

    def test_adjoint_k1_multiplicity(self, ph):
        """Adjoint: k=1 has m_adj(1) = 1 mode."""
        levels = ph.invariant_levels_coexact_adjoint(5)
        assert levels[0] == (1, 1)

    def test_adjoint_second_level_at_k3(self, ph):
        """In the compact topology framework, the second level is k=3."""
        levels = ph.invariant_levels_coexact_adjoint(10)
        assert len(levels) >= 2
        assert levels[1][0] == 3

    def test_adjoint_sparser_than_standard(self, ph):
        """Adjoint spectrum is sparser at low k than standard."""
        adjoint = ph.invariant_levels_coexact_adjoint(30)
        standard = ph.invariant_levels_coexact(30)
        # Standard has k=1 (mult 3), k=11, k=13, ...
        # Adjoint has k=1 (mult 1), k=3, k=9, k=11, ...
        # Adjoint has fewer modes at k=1 but more total surviving levels
        assert adjoint[0][1] < standard[0][1]  # 1 < 3 at k=1


# ======================================================================
# CMB predictions
# ======================================================================

class TestCMBPredictions:
    """Test CMB multipole predictions from S^3/I* topology."""

    def test_monopole_present(self, ph):
        """l=0 (monopole) is present."""
        cmb = ph.cmb_multipole_prediction(5)
        assert cmb[0]['multiplicity'] == 1
        assert not cmb[0]['suppressed']

    def test_quadrupole_suppressed(self, ph):
        """l=2 (quadrupole) suppressed — consistent with Planck anomaly."""
        cmb = ph.cmb_multipole_prediction(5)
        assert cmb[2]['suppressed']
        assert cmb[2]['multiplicity'] == 0

    def test_octupole_suppressed(self, ph):
        """l=3 (octupole) suppressed."""
        cmb = ph.cmb_multipole_prediction(5)
        assert cmb[3]['suppressed']

    def test_all_low_multipoles_suppressed(self, ph):
        """All multipoles l=1 through l=11 are suppressed."""
        cmb = ph.cmb_multipole_prediction(15)
        for l in range(1, 12):
            assert cmb[l]['suppressed'], \
                f"l={l} should be suppressed but has m={cmb[l]['multiplicity']}"

    def test_l12_present(self, ph):
        """l=12 is the first nontrivial multipole that is NOT suppressed."""
        cmb = ph.cmb_multipole_prediction(15)
        assert not cmb[12]['suppressed']
        assert cmb[12]['multiplicity'] == 1


# ======================================================================
# Physical predictions
# ======================================================================

class TestPhysicalPredictions:
    """Test the distinguishable physical predictions."""

    def test_gap_preserved(self, ph):
        """Gap is preserved on quotient (k=1 survives)."""
        pred = ph.physical_predictions(R_fm=2.2)
        assert pred['mass_gap']['status'] == 'THEOREM'

    def test_second_excitation_at_k11(self, ph):
        """Second excitation is at k=11 (DISTINGUISHABLE)."""
        pred = ph.physical_predictions(R_fm=2.2)
        assert pred['second_excitation']['poincare']['k'] == 11

    def test_mass_ratio_m2_m1_is_6(self, ph):
        """Mass ratio m2/m1 on S^3/I* is 6.0 (vs 1.5 on S^3)."""
        pred = ph.physical_predictions(R_fm=2.2)
        assert abs(pred['mass_ratio_m2_m1']['poincare'] - 6.0) < 1e-10
        assert abs(pred['mass_ratio_m2_m1']['s3'] - 1.5) < 1e-10

    def test_sparsification_reasonable(self, ph):
        """Sparsification fraction is small but nonzero."""
        pred = ph.physical_predictions(R_fm=2.2)
        frac = pred['spectrum_sparsification']['fraction_surviving']
        assert 0 < frac < 0.1  # should be a few percent

    def test_five_surviving_levels(self, ph):
        """At least 5 coexact levels survive up to k=60."""
        levels = ph.invariant_levels_coexact(60)
        assert len(levels) >= 5


# ======================================================================
# Consistency checks
# ======================================================================

class TestConsistency:
    """Cross-checks and consistency tests."""

    def test_coexact_sum_check(self, ph):
        """Sum of I*-invariant coexact modes should be reasonable fraction of S^3."""
        inv_total = sum(ph.coexact_invariant_multiplicity(k) for k in range(1, 61))
        s3_total = sum(2 * k * (k + 2) for k in range(1, 61))
        ratio = inv_total / s3_total
        # Asymptotically should approach 1/120 ~ 0.0083
        assert 0.001 < ratio < 0.05, f"Ratio {ratio} out of expected range"

    def test_volume_ratio(self):
        """Vol(S^3/I*) = Vol(S^3) / 120."""
        R = 1.0
        vol_s3 = 2 * np.pi**2 * R**3
        vol_poincare = vol_s3 / 120
        assert abs(vol_poincare - np.pi**2 / 60) < 1e-12

    def test_coexact_detail_consistent(self, ph):
        """Detail breakdown matches total multiplicity."""
        for k in range(1, 50):
            detail = ph.coexact_invariant_detail(k)
            assert detail['total'] == detail['n_sd'] + detail['n_asd']
            assert detail['total'] == ph.coexact_invariant_multiplicity(k)
            assert detail['s3_total'] == 2 * k * (k + 2)

    def test_print_summary_runs(self, ph, capsys):
        """print_summary should run without errors."""
        ph.print_summary(k_max=15, R_fm=2.2)
        captured = capsys.readouterr()
        assert "POINCARE HOMOLOGY SPHERE" in captured.out
        assert "COEXACT 1-FORM SPECTRUM" in captured.out
        assert "GAP PRESERVED" in captured.out
        assert "Molien series verification: PASSED" in captured.out

    def test_coexact_parity_pattern(self, ph):
        """
        The surviving levels follow a pattern related to I* invariants.

        At level k, self-dual survives iff m(k+1) > 0, i.e., k+1 in {12, 20, 24, ...}
        So SD survives at k in {11, 19, 23, 29, 31, ...}

        Anti-self-dual survives iff m(k-1) > 0, i.e., k-1 in {0, 12, 20, 24, ...}
        So ASD survives at k in {1, 13, 21, 25, ...}
        """
        # k=1: ASD from m(0)=1
        assert ph.coexact_invariant_detail(1)['n_asd'] > 0
        assert ph.coexact_invariant_detail(1)['n_sd'] == 0

        # k=11: SD from m(12)=1
        assert ph.coexact_invariant_detail(11)['n_sd'] > 0
        assert ph.coexact_invariant_detail(11)['n_asd'] == 0

        # k=13: ASD from m(12)=1
        assert ph.coexact_invariant_detail(13)['n_asd'] > 0
        assert ph.coexact_invariant_detail(13)['n_sd'] == 0

        # k=19: SD from m(20)=1
        assert ph.coexact_invariant_detail(19)['n_sd'] > 0
        assert ph.coexact_invariant_detail(19)['n_asd'] == 0

        # k=21: ASD from m(20)=1
        assert ph.coexact_invariant_detail(21)['n_asd'] > 0
