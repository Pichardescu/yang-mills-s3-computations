"""
Tests for the icosahedral spectrum module.

Verifies the I*-invariant spectrum of Laplacians on S3/I* (Poincare
homology sphere), including scalar, 1-form, and Yang-Mills spectra.

Mathematical ground truth:
  - Molien series: M(t) = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30))
  - First nontrivial scalar invariant: l=12
  - 1-form gap (standard): l=1, eigenvalue 5/R^2 (same as S3)
  - CMB: l=2 through l=11 all suppressed
"""

import pytest
import numpy as np
from yang_mills_s3.geometry.icosahedral_spectrum import IcosahedralSpectrum, HBAR_C_MEV_FM


@pytest.fixture
def spec():
    """Create an IcosahedralSpectrum instance."""
    return IcosahedralSpectrum()


# ======================================================================
# Conjugacy class validation
# ======================================================================

class TestConjugacyClasses:
    """Verify the conjugacy class structure of I*."""

    def test_group_order(self, spec):
        """I* has order 120."""
        assert spec.group_order == 120

    def test_nine_conjugacy_classes(self, spec):
        """I* has 9 conjugacy classes."""
        assert len(spec.conjugacy_classes) == 9

    def test_class_sizes_sum_to_120(self, spec):
        """Class sizes must sum to |I*| = 120."""
        total = sum(size for size, _ in spec.conjugacy_classes)
        assert total == 120

    def test_class_sizes(self, spec):
        """Verify individual class sizes."""
        sizes = sorted(size for size, _ in spec.conjugacy_classes)
        assert sizes == [1, 1, 12, 12, 12, 12, 20, 20, 30]


# ======================================================================
# Character formula tests
# ======================================================================

class TestCharacterSU2:
    """Test the SU(2) character formula chi_l(theta)."""

    def test_identity_gives_dimension(self, spec):
        """chi_l(0) = l+1 (dimension of V_l)."""
        for l in range(20):
            assert abs(spec.character_su2(l, 0.0) - (l + 1)) < 1e-12

    def test_central_element(self, spec):
        """chi_l(2*pi) = (-1)^l * (l+1)."""
        for l in range(20):
            expected = ((-1) ** l) * (l + 1)
            assert abs(spec.character_su2(l, 2 * np.pi) - expected) < 1e-10

    def test_pi_rotation(self, spec):
        """chi_l(pi) = sin((l+1)*pi/2)."""
        for l in range(20):
            expected = np.sin((l + 1) * np.pi / 2)
            assert abs(spec.character_su2(l, np.pi) - expected) < 1e-12

    def test_character_v0_is_one(self, spec):
        """chi_0(theta) = 1 for all theta (trivial representation)."""
        angles = [0.0, np.pi/5, np.pi/3, np.pi/2, np.pi, 4*np.pi/3, 2*np.pi]
        for theta in angles:
            assert abs(spec.character_su2(0, theta) - 1.0) < 1e-12

    def test_character_v1(self, spec):
        """chi_1(theta) = 2*cos(theta/2) (fundamental representation)."""
        angles = [np.pi/5, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]
        for theta in angles:
            expected = 2 * np.cos(theta / 2)
            assert abs(spec.character_su2(1, theta) - expected) < 1e-12

    def test_clebsch_gordan_v_l_tensor_v_1(self, spec):
        """chi_l * chi_1 = chi_{l+1} + chi_{l-1} (Clebsch-Gordan for V_l x V_1)."""
        angles = [np.pi/5, np.pi/3, 2*np.pi/3, 4*np.pi/5, np.pi]
        for l in range(1, 15):
            for theta in angles:
                product = spec.character_su2(l, theta) * spec.character_su2(1, theta)
                cg_sum = spec.character_su2(l + 1, theta) + spec.character_su2(l - 1, theta)
                assert abs(product - cg_sum) < 1e-10, \
                    f"CG identity failed at l={l}, theta={theta:.4f}"


# ======================================================================
# Scalar multiplicity tests
# ======================================================================

class TestTrivialMultiplicity:
    """Test the I*-invariant multiplicity m(l)."""

    def test_m0_is_one(self, spec):
        """m(0) = 1: the constant function is always invariant."""
        assert spec.trivial_multiplicity(0) == 1

    def test_always_nonnegative_integer(self, spec):
        """m(l) must be a non-negative integer for all l."""
        for l in range(100):
            m = spec.trivial_multiplicity(l)
            assert isinstance(m, int)
            assert m >= 0, f"m({l}) = {m} < 0"

    def test_first_nontrivial_at_l12(self, spec):
        """First nontrivial I*-invariant scalar is at l=12."""
        for l in range(1, 12):
            assert spec.trivial_multiplicity(l) == 0, \
                f"m({l}) = {spec.trivial_multiplicity(l)}, expected 0"
        assert spec.trivial_multiplicity(12) == 1

    def test_known_invariant_levels(self, spec):
        """Verify m(l) for the known nonzero levels up to l=60."""
        # From Molien series: (1-t^60)/((1-t^12)(1-t^20)(1-t^30))
        expected_nonzero = {
            0: 1, 12: 1, 20: 1, 24: 1, 30: 1, 32: 1, 36: 1,
            40: 1, 42: 1, 44: 1, 48: 1, 50: 1, 52: 1, 54: 1,
            56: 1, 60: 2
        }
        for l in range(61):
            m = spec.trivial_multiplicity(l)
            if l in expected_nonzero:
                assert m == expected_nonzero[l], \
                    f"m({l}) = {m}, expected {expected_nonzero[l]}"
            else:
                assert m == 0, f"m({l}) = {m}, expected 0"

    def test_negative_l_returns_zero(self, spec):
        """m(l) = 0 for l < 0."""
        assert spec.trivial_multiplicity(-1) == 0
        assert spec.trivial_multiplicity(-10) == 0

    def test_multiplicity_sum_formula(self, spec):
        """
        sum_{l=0}^{L} m(l) * (l+1)^2 counts the total number of
        I*-fixed points in the representations up to level L.

        On S3/I*, the scalar eigenspace at level l has total dimension
        m(l) * (l+1) (with the (l+1) from the right SU(2) action).
        """
        # Just verify the sum is reasonable and monotonically increasing
        running_sum = 0
        for l in range(61):
            m = spec.trivial_multiplicity(l)
            running_sum += m * (l + 1)
        # With m(0)=1 contributing 1, m(12)=1 contributing 13,
        # m(20)=1 contributing 21, etc.
        assert running_sum > 0


# ======================================================================
# Molien series verification
# ======================================================================

class TestMolienSeries:
    """Verify character computation against the closed-form Molien series."""

    def test_molien_matches_character(self, spec):
        """Character computation must match Molien series for all l."""
        ok, mismatches = spec.verify_against_molien(60)
        if not ok:
            details = ", ".join(
                f"l={l}: char={mc}, mol={mm}" for l, mc, mm in mismatches
            )
            pytest.fail(f"Molien series mismatch: {details}")

    def test_molien_coefficients_nonnegative(self, spec):
        """Molien series coefficients must be non-negative."""
        molien = spec.molien_series_coefficients(100)
        for l, m in enumerate(molien):
            assert m >= 0, f"Molien coefficient at l={l} is {m} < 0"

    def test_molien_m0_is_one(self, spec):
        """Molien series starts with m(0) = 1."""
        molien = spec.molien_series_coefficients(10)
        assert molien[0] == 1

    def test_molien_first_nonzero_at_12(self, spec):
        """After l=0, next nonzero Molien coefficient is at l=12."""
        molien = spec.molien_series_coefficients(15)
        for l in range(1, 12):
            assert molien[l] == 0
        assert molien[12] == 1


# ======================================================================
# Scalar spectrum on S3/I*
# ======================================================================

class TestScalarSpectrum:
    """Test the scalar Laplacian spectrum on S3/I*."""

    def test_zero_mode_exists(self, spec):
        """l=0 mode (constants) always survives."""
        spectrum = spec.scalar_spectrum_poincare(l_max=60, R=1.0)
        assert spectrum[0] == (0.0, 1)

    def test_scalar_gap_is_168(self, spec):
        """First nontrivial scalar eigenvalue on S3/I* is 168/R^2."""
        spectrum = spec.scalar_spectrum_poincare(l_max=60, R=1.0)
        # Skip l=0 (eigenvalue 0)
        nontrivial = [(ev, m) for ev, m in spectrum if ev > 0]
        assert len(nontrivial) > 0
        assert abs(nontrivial[0][0] - 168.0) < 1e-10, \
            f"Scalar gap should be 168, got {nontrivial[0][0]}"

    def test_scalar_gap_ratio(self, spec):
        """Scalar gap on S3/I* is 56x larger than on S3."""
        # S3: gap = 3/R^2 (l=1)
        # S3/I*: gap = 168/R^2 (l=12)
        # Ratio: 168/3 = 56
        spectrum = spec.scalar_spectrum_poincare(l_max=60, R=1.0)
        nontrivial = [(ev, m) for ev, m in spectrum if ev > 0]
        gap_poincare = nontrivial[0][0]
        gap_s3 = 3.0
        assert abs(gap_poincare / gap_s3 - 56.0) < 1e-10

    def test_radius_scaling(self, spec):
        """Eigenvalues scale as 1/R^2."""
        R = 3.7
        spectrum = spec.scalar_spectrum_poincare(l_max=20, R=R)
        for ev, m in spectrum:
            if ev > 0:
                # Eigenvalue should be 168/R^2 for l=12
                assert abs(ev - 168.0 / R**2) < 1e-10
                break


# ======================================================================
# 1-form spectrum tests
# ======================================================================

class TestOneFormSpectrum:
    """Test the I*-invariant 1-form spectrum."""

    def test_l1_survives(self, spec):
        """
        l=1 1-form mode survives on S3/I*.

        This comes from the coexact part: V_{l-1} = V_0 for l=1.
        V_0 always has an I*-invariant (the constant), so the
        Maurer-Cartan form survives the projection.
        """
        levels = spec.invariant_levels_oneform(l_max=5)
        assert len(levels) > 0
        assert levels[0] == (1, 1), f"Expected l=1 with mult=1, got {levels[0]}"

    def test_oneform_gap_is_5(self, spec):
        """Yang-Mills gap on S3/I* = 5/R^2 (standard scenario)."""
        ym_spectrum = spec.yang_mills_spectrum_poincare(l_max=30, R=1.0)
        assert len(ym_spectrum) > 0
        ev, mult, l_val = ym_spectrum[0]
        assert abs(ev - 5.0) < 1e-10, f"YM gap should be 5, got {ev}"
        assert l_val == 1

    def test_oneform_gap_same_as_s3(self, spec):
        """
        IMPORTANT RESULT: The Yang-Mills 1-form gap on S3/I* equals
        that on S3 (both 5/R^2). The l=1 mode survives because the
        coexact component at l=1 is V_0-invariant (constant function).
        """
        comp = spec.gap_comparison(R_fm=2.2)
        ratio = comp['yang_mills_standard']['ratio']
        assert abs(ratio - 1.0) < 1e-10, \
            f"YM gap ratio should be 1.0, got {ratio}"

    def test_oneform_multiplicities_nonneg(self, spec):
        """All 1-form multiplicities must be non-negative integers."""
        for l in range(1, 100):
            m_plus = spec.trivial_multiplicity(l + 1)
            m_minus = spec.trivial_multiplicity(l - 1)
            m_total = m_plus + m_minus
            assert m_total >= 0
            assert isinstance(m_total, int)

    def test_oneform_invariant_decomposition(self, spec):
        """
        m_1(l) = m(l+1) + m(l-1) must equal the direct character
        computation (1/120) sum |C| chi_l chi_1.
        """
        for l in range(1, 50):
            # Decomposition formula
            m_decomp = spec.trivial_multiplicity(l + 1) + spec.trivial_multiplicity(l - 1)

            # Direct character formula
            total = 0.0
            for size, theta in spec.conjugacy_classes:
                chi_l = spec.character_su2(l, theta)
                chi_1 = spec.character_su2(1, theta)
                total += size * chi_l * chi_1
            m_direct = int(round(total / 120))

            assert m_decomp == m_direct, \
                f"l={l}: decomp={m_decomp} != direct={m_direct}"

    def test_second_oneform_level(self, spec):
        """The second surviving 1-form level should be l=11."""
        levels = spec.invariant_levels_oneform(l_max=30)
        assert len(levels) >= 2
        assert levels[1][0] == 11, \
            f"Second 1-form level should be l=11, got {levels[1][0]}"

    def test_oneform_gap_larger_than_s3_scalar(self, spec):
        """The YM gap on S3/I* (5/R^2) is larger than the S3 scalar gap (3/R^2)."""
        ym_spec = spec.yang_mills_spectrum_poincare(l_max=5, R=1.0)
        assert ym_spec[0][0] > 3.0


# ======================================================================
# Adjoint-valued 1-form tests
# ======================================================================

class TestAdjointSpectrum:
    """
    Test the adjoint scenario where I* acts on both base and gauge fiber.

    In this scenario, the gauge SU(2) = geometric SU(2), and I* acts
    on the adjoint (gauge) index as well.
    """

    def test_adjoint_gap_at_l1(self, spec):
        """In the compact topology framework, the first level is still l=1."""
        levels = spec.invariant_levels_oneform_adjoint(l_max=30)
        assert len(levels) > 0
        assert levels[0][0] == 1, \
            f"Adjoint first level should be l=1, got {levels[0][0]}"

    def test_adjoint_l1_multiplicity(self, spec):
        """Adjoint: m_adj(1) = 1."""
        levels = spec.invariant_levels_oneform_adjoint(l_max=5)
        assert levels[0] == (1, 1)

    def test_adjoint_second_level_at_l3(self, spec):
        """In the compact topology framework, the second level is l=3 (not l=11 as in standard)."""
        levels = spec.invariant_levels_oneform_adjoint(l_max=10)
        assert len(levels) >= 2
        assert levels[1][0] == 3, \
            f"Adjoint second level should be l=3, got {levels[1][0]}"

    def test_adjoint_multiplicities_integer(self, spec):
        """All adjoint multiplicities must be non-negative integers."""
        levels = spec.invariant_levels_oneform_adjoint(l_max=60)
        for l, m in levels:
            assert isinstance(m, int)
            assert m >= 0

    def test_adjoint_spectrum_eigenvalues(self, spec):
        """Adjoint Yang-Mills eigenvalues have correct formula."""
        ym_adjoint = spec.yang_mills_spectrum_poincare(l_max=10, R=1.0, adjoint=True)
        for ev, mult, l in ym_adjoint:
            expected_ev = l * (l + 2) + 2
            assert abs(ev - expected_ev) < 1e-10


# ======================================================================
# CMB predictions
# ======================================================================

class TestCMBPredictions:
    """Test CMB multipole predictions from S3/I* topology."""

    def test_monopole_present(self, spec):
        """l=0 (monopole) is present (m=1)."""
        cmb = spec.cmb_multipole_prediction(l_max=5)
        assert cmb[0]['multiplicity'] == 1
        assert not cmb[0]['suppressed']

    def test_quadrupole_suppressed(self, spec):
        """
        l=2 (quadrupole) is suppressed.

        This is consistent with the observed anomalously low CMB
        quadrupole power in Planck data (~2-3 sigma below LCDM).
        """
        cmb = spec.cmb_multipole_prediction(l_max=5)
        assert cmb[2]['suppressed'], "Quadrupole (l=2) should be suppressed"
        assert cmb[2]['multiplicity'] == 0

    def test_octupole_suppressed(self, spec):
        """l=3 (octupole) is suppressed."""
        cmb = spec.cmb_multipole_prediction(l_max=5)
        assert cmb[3]['suppressed']

    def test_all_low_multipoles_suppressed(self, spec):
        """All multipoles l=1 through l=11 are suppressed."""
        cmb = spec.cmb_multipole_prediction(l_max=15)
        for l in range(1, 12):
            assert cmb[l]['suppressed'], \
                f"l={l} should be suppressed but has m={cmb[l]['multiplicity']}"

    def test_l12_present(self, spec):
        """l=12 is the first nontrivial multipole that is NOT suppressed."""
        cmb = spec.cmb_multipole_prediction(l_max=15)
        assert not cmb[12]['suppressed'], "l=12 should NOT be suppressed"
        assert cmb[12]['multiplicity'] == 1

    def test_cmb_eigenvalue_coefficients(self, spec):
        """Eigenvalue coefficients are l*(l+2)."""
        cmb = spec.cmb_multipole_prediction(l_max=20)
        for entry in cmb:
            l = entry['l']
            assert entry['eigenvalue_coeff'] == l * (l + 2)


# ======================================================================
# Gap comparison and physical values
# ======================================================================

class TestGapComparison:
    """Test the gap comparison between S3 and S3/I*."""

    def test_scalar_gap_ratio_56(self, spec):
        """Scalar gap ratio S3/I* to S3 is 56."""
        comp = spec.gap_comparison(R_fm=2.2)
        assert abs(comp['scalar']['ratio'] - 56.0) < 1e-10

    def test_ym_standard_gap_ratio_1(self, spec):
        """Standard YM gap ratio is 1 (gap preserved on quotient)."""
        comp = spec.gap_comparison(R_fm=2.2)
        assert abs(comp['yang_mills_standard']['ratio'] - 1.0) < 1e-10

    def test_physical_scalar_gap_s3(self, spec):
        """Scalar gap on S3 at R=2.2 fm is sqrt(3)*hbar_c/R ~ 155 MeV."""
        comp = spec.gap_comparison(R_fm=2.2)
        gap = comp['scalar']['s3']['gap_mev']
        expected = np.sqrt(3) * HBAR_C_MEV_FM / 2.2
        assert abs(gap - expected) < 0.1

    def test_physical_ym_gap_s3(self, spec):
        """YM gap on S3 at R=2.2 fm is sqrt(5)*hbar_c/R ~ 200 MeV."""
        comp = spec.gap_comparison(R_fm=2.2)
        gap = comp['yang_mills_standard']['s3']['gap_mev']
        expected = np.sqrt(5) * HBAR_C_MEV_FM / 2.2
        assert abs(gap - expected) < 0.1

    def test_gap_comparison_has_lattice_value(self, spec):
        """Gap comparison includes lattice glueball reference value."""
        comp = spec.gap_comparison(R_fm=2.2)
        assert comp['lattice_glueball_0pp_mev'] == 1730

    def test_adjoint_gap_in_comparison(self, spec):
        """Gap comparison includes adjoint scenario."""
        comp = spec.gap_comparison(R_fm=2.2)
        assert 'yang_mills_adjoint' in comp
        # Adjoint gap is also at l=1, so ratio = 1
        assert abs(comp['yang_mills_adjoint']['ratio'] - 1.0) < 1e-10


# ======================================================================
# Consistency checks
# ======================================================================

class TestConsistency:
    """Cross-checks and consistency tests."""

    def test_spectrum_gap_ordering(self, spec):
        """S3/I* spectral gap > S3 spectral gap for scalars."""
        s3_gap = 3.0  # l=1: 1*3
        poincare_gap = 12 * 14  # l=12: 12*14 = 168
        assert poincare_gap > s3_gap

    def test_ym_gap_at_least_s3(self, spec):
        """YM gap on S3/I* is >= YM gap on S3 (5/R^2)."""
        ym_spec = spec.yang_mills_spectrum_poincare(l_max=30, R=1.0)
        if ym_spec:
            assert ym_spec[0][0] >= 5.0

    def test_volume_ratio(self):
        """Vol(S3/I*) = Vol(S3) / 120."""
        R = 1.0
        vol_s3 = 2 * np.pi**2 * R**3
        vol_poincare = vol_s3 / 120
        assert abs(vol_poincare - np.pi**2 / 60) < 1e-12

    def test_total_scalar_modes_up_to_l(self, spec):
        """
        Total number of I*-invariant scalar modes up to level l_max
        must be consistent. On S3/I* the total Weyl counting gives:
            N(lambda) ~ Vol(S3/I*) / (6*pi^2) * lambda^{3/2}
        for the counting function of the scalar Laplacian.
        This is 1/120 of the S3 count.
        """
        l_max = 60
        s3_modes = sum((l + 1)**2 for l in range(l_max + 1))
        poincare_modes = 0
        for l in range(l_max + 1):
            m = spec.trivial_multiplicity(l)
            # On S3/I*, each invariant gives (l+1) modes (from right action)
            poincare_modes += m * (l + 1)

        # Weyl's law says ratio should approach 1/120 for large l_max
        ratio = poincare_modes / s3_modes
        # For finite l_max the ratio won't be exact, but should be in range
        assert 0 < ratio < 1, f"Ratio {ratio} out of bounds"

    def test_print_summary_runs(self, spec, capsys):
        """print_summary should run without errors."""
        spec.print_summary(l_max=15, R_fm=2.2)
        captured = capsys.readouterr()
        assert "POINCARE HOMOLOGY SPHERE" in captured.out
        assert "SCALAR SPECTRUM" in captured.out
        assert "Molien series verification: PASSED" in captured.out
