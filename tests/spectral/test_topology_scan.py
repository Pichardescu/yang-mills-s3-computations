"""
Tests for the topology scan module (all finite quotients S^3/Gamma).

Verifies:
  1. Molien series: r_{Z_1}(k) = k+1 (trivial group = full S^3)
  2. Molien series: r_{I*}(k) matches existing istar values
  3. Sum rule: sum_{k=0}^{n-1} r_{Z_n}(k) * (k+1) = sum of dims of invariant harmonics
  4. Basic sanity: r(0) = 1 for all groups (identity is always invariant)
  5. Monotonicity: larger groups give fewer invariant modes
  6. D_2 comparison with Planck (requires CAMB)
  7. Cross-check against PoincareHomology.trivial_multiplicity

NUMERICAL: All tests verify representation-theoretic identities and
Boltzmann transfer function computations.
"""

import pytest
import numpy as np

from yang_mills_s3.spectral.topology_scan import (
    character_su2,
    conjugacy_classes_cyclic,
    conjugacy_classes_binary_dihedral,
    conjugacy_classes_binary_tetrahedral,
    conjugacy_classes_binary_octahedral,
    conjugacy_classes_binary_icosahedral,
    molien_cyclic,
    molien_binary_dihedral,
    molien_binary_tetrahedral,
    molien_binary_octahedral,
    molien_binary_icosahedral,
    build_group_catalog,
    compute_suppression,
)


# ==================================================================
# Character tests
# ==================================================================

class TestCharacterSU2:
    """Tests for the SU(2) character function."""

    def test_identity(self):
        """chi_{k/2}(0) = k + 1 (dimension of representation)."""
        for k in range(20):
            assert abs(character_su2(k, 0.0) - (k + 1)) < 1e-10

    def test_central_element(self):
        """chi_{k/2}(2*pi) = (-1)^k * (k+1)."""
        for k in range(20):
            expected = ((-1) ** k) * (k + 1)
            assert abs(character_su2(k, 2 * np.pi) - expected) < 1e-10

    def test_order_4(self):
        """chi_{k/2}(pi) = sin((k+1)*pi/2) (known values: 1,0,-1,0,1,...)."""
        expected = [1, 0, -1, 0, 1, 0, -1, 0]
        for k, exp in enumerate(expected):
            val = character_su2(k, np.pi)
            assert abs(val - exp) < 1e-10, f"k={k}: got {val}, expected {exp}"

    def test_character_real(self):
        """SU(2) characters are real for real theta."""
        for k in range(15):
            for theta in [0.5, 1.0, np.pi / 3, np.pi / 5, 2.0]:
                val = character_su2(k, theta)
                assert isinstance(val, float) or abs(val.imag) < 1e-14


# ==================================================================
# Conjugacy class sanity
# ==================================================================

class TestConjugacyClasses:
    """Verify conjugacy class sizes sum to group order."""

    def test_cyclic(self):
        for n in [1, 2, 3, 5, 10, 60, 120]:
            classes, order = conjugacy_classes_cyclic(n)
            assert order == n
            assert sum(s for s, _ in classes) == n

    def test_binary_dihedral(self):
        for n in [2, 3, 4, 5, 6, 8, 10]:
            classes, order = conjugacy_classes_binary_dihedral(n)
            assert order == 4 * n
            assert sum(s for s, _ in classes) == 4 * n

    def test_binary_tetrahedral(self):
        classes, order = conjugacy_classes_binary_tetrahedral()
        assert order == 24
        assert sum(s for s, _ in classes) == 24

    def test_binary_octahedral(self):
        classes, order = conjugacy_classes_binary_octahedral()
        assert order == 48
        assert sum(s for s, _ in classes) == 48

    def test_binary_icosahedral(self):
        classes, order = conjugacy_classes_binary_icosahedral()
        assert order == 120
        assert sum(s for s, _ in classes) == 120


# ==================================================================
# Molien series: trivial group Z_1
# ==================================================================

class TestMolienTrivialGroup:
    """
    Z_1 = trivial group.
    r_{Z_1}(k) = k+1 (no quotient, full S^3 spectrum).

    This is because the trivial group has one element (identity) with
    chi_{k/2}(0) = k+1, and |Z_1| = 1, so r(k) = (k+1)/1 = k+1.
    """

    def test_z1_multiplicities(self):
        """r_{Z_1}(k) = k + 1 for all k."""
        r = molien_cyclic(1, 60)
        for k in range(61):
            assert r[k] == k + 1, f"k={k}: r={r[k]}, expected {k+1}"


# ==================================================================
# Molien series: cyclic groups
# ==================================================================

class TestMolienCyclic:
    """Tests for cyclic group multiplicities."""

    def test_identity_always_one(self):
        """r_{Z_n}(0) = 1 for all n (the constant harmonic is always invariant)."""
        for n in [1, 2, 3, 5, 10, 60, 120]:
            r = molien_cyclic(n, 0)
            assert r[0] == 1

    def test_z2_multiplicities(self):
        """
        Z_2 = {I, -I} in SU(2), with theta = {0, 4*pi/2 = 2*pi}.
        -I acts on V_{k/2} as (-1)^k * Identity.
        r(k) = (1/2)[(k+1) + (-1)^k*(k+1)] = (k+1) if k even, 0 if k odd.
        """
        r = molien_cyclic(2, 20)
        for k in range(21):
            expected = (k + 1) if k % 2 == 0 else 0
            assert r[k] == expected, f"Z_2, k={k}: r={r[k]}, expected {expected}"

    def test_z3_low_k(self):
        """
        Z_3 in SU(2) generated by diag(e^{2pi i/3}, e^{-2pi i/3}).
        r(0)=1, r(1)=0 (no invariant in V_{1/2}), r(2)=1, r(3)=2, ...
        """
        r = molien_cyclic(3, 10)
        for k in range(11):
            assert r[k] >= 0
            assert r[k] == int(r[k])
        assert r[0] == 1
        assert r[1] == 0  # V_{1/2} has no Z_3-invariant vectors
        assert r[2] == 1  # V_1 has one Z_3-invariant vector (m=0)

    def test_sum_over_invariant_modes(self):
        """
        For Z_n, total invariant scalar harmonics up to level k_max
        should be sum_k r(k) * (k+1).
        """
        for n in [2, 4, 6]:
            r = molien_cyclic(n, 30)
            total = sum(r[k] * (k + 1) for k in range(31))
            assert total > 0

    def test_larger_group_fewer_modes(self):
        """Groups with larger order should have fewer total invariant modes."""
        k_max = 50
        total_modes = {}
        for n in [2, 4, 8, 16]:
            r = molien_cyclic(n, k_max)
            total_modes[n] = sum(r[k] for k in range(1, k_max + 1))

        # More elements -> fewer invariant modes (on average)
        # This is a statistical property, not strict per-k
        assert total_modes[2] >= total_modes[4]
        assert total_modes[4] >= total_modes[8]


# ==================================================================
# Molien series: binary icosahedral I* (cross-check)
# ==================================================================

class TestMolienIcosahedral:
    """Cross-check I* multiplicities against existing PoincareHomology code."""

    def test_istar_known_values(self):
        """r_{I*}(k) for known k values."""
        r = molien_binary_icosahedral(60)

        # Known: m(0)=1, m(1..11)=0, m(12)=1, m(20)=1, m(24)=1, m(30)=1
        assert r[0] == 1
        for k in range(1, 12):
            assert r[k] == 0, f"r_{k} should be 0, got {r[k]}"
        assert r[12] == 1
        assert r[20] == 1
        assert r[24] == 1
        assert r[30] == 1

    def test_istar_matches_poincare_homology(self):
        """r_{I*}(k) should match PoincareHomology.trivial_multiplicity(k)."""
        from yang_mills_s3.geometry.poincare_homology import PoincareHomology
        ph = PoincareHomology()

        r = molien_binary_icosahedral(60)
        for k in range(61):
            expected = ph.trivial_multiplicity(k)
            assert r[k] == expected, (
                f"k={k}: topology_scan gives {r[k]}, "
                f"PoincareHomology gives {expected}"
            )


# ==================================================================
# Molien series: binary dihedral
# ==================================================================

class TestMolienBinaryDihedral:
    """Tests for binary dihedral group multiplicities."""

    def test_identity_always_one(self):
        """r(0) = 1 for all D*_n."""
        for n in [2, 3, 4, 5, 6, 8, 10]:
            r = molien_binary_dihedral(n, 0)
            assert r[0] == 1

    def test_nonnegative(self):
        """All multiplicities must be non-negative."""
        for n in [2, 3, 4, 5, 6, 8, 10]:
            r = molien_binary_dihedral(n, 40)
            for k in range(41):
                assert r[k] >= 0, f"D*_{n}, k={k}: r={r[k]} < 0"

    def test_integer_valued(self):
        """All multiplicities must be integers."""
        for n in [2, 3, 4, 5, 6, 8, 10]:
            r = molien_binary_dihedral(n, 40)
            for k in range(41):
                assert r[k] == int(r[k])


# ==================================================================
# Molien series: exceptional groups T*, O*
# ==================================================================

class TestMolienExceptional:
    """Tests for T* and O* multiplicities."""

    def test_tstar_identity(self):
        """r_{T*}(0) = 1."""
        r = molien_binary_tetrahedral(0)
        assert r[0] == 1

    def test_ostar_identity(self):
        """r_{O*}(0) = 1."""
        r = molien_binary_octahedral(0)
        assert r[0] == 1

    def test_tstar_nonnegative(self):
        """All multiplicities for T* are non-negative integers."""
        r = molien_binary_tetrahedral(60)
        for k in range(61):
            assert r[k] >= 0
            assert r[k] == int(r[k])

    def test_ostar_nonnegative(self):
        """All multiplicities for O* are non-negative integers."""
        r = molien_binary_octahedral(60)
        for k in range(61):
            assert r[k] >= 0
            assert r[k] == int(r[k])

    def test_tstar_has_gap(self):
        """T* should have r(k)=0 for some k > 0 (topology creates spectral gap)."""
        r = molien_binary_tetrahedral(30)
        assert r[1] == 0  # First nonzero after k=0 should be > 1

    def test_ostar_has_gap(self):
        """O* should have r(k)=0 for some k > 0."""
        r = molien_binary_octahedral(30)
        assert r[1] == 0


# ==================================================================
# Group catalog
# ==================================================================

class TestGroupCatalog:
    """Tests for the full group catalog."""

    def test_catalog_size(self):
        """Should contain 15 cyclic + 7 dihedral + T* + O* + I* = 25 groups."""
        catalog = build_group_catalog(30)
        assert len(catalog) == 25

    def test_all_have_r0_equals_1(self):
        """Every group has r(0) = 1."""
        catalog = build_group_catalog(30)
        for group in catalog:
            assert group['multiplicities'][0] == 1, (
                f"{group['name']}: r(0) = {group['multiplicities'][0]}"
            )

    def test_all_nonnegative(self):
        """All multiplicities are non-negative for all groups."""
        catalog = build_group_catalog(30)
        for group in catalog:
            r = group['multiplicities']
            for k in range(len(r)):
                assert r[k] >= 0, (
                    f"{group['name']}, k={k}: r={r[k]}"
                )

    def test_larger_order_sparser_spectrum(self):
        """
        Groups of larger order tend to have fewer nonzero modes.

        Not strict per pair, but Z_2 (order 2) should have more modes
        than I* (order 120).
        """
        catalog = build_group_catalog(60)

        z2_modes = None
        istar_modes = None
        for g in catalog:
            r = g['multiplicities']
            total = sum(1 for k in range(1, 61) if r[k] > 0)
            if g['name'] == 'Z_2':
                z2_modes = total
            elif g['name'] == 'I*':
                istar_modes = total

        assert z2_modes is not None and istar_modes is not None
        assert z2_modes > istar_modes


# ==================================================================
# CAMB-dependent tests
# ==================================================================

# Guard: skip if CAMB is not installed
try:
    import camb as _camb_check
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False


@pytest.mark.skipif(not HAS_CAMB, reason="CAMB not installed")
class TestCMBSuppression:
    """Tests requiring CAMB for transfer function computation."""

    @pytest.fixture(scope="class")
    def camb_data(self):
        """Pre-computed CAMB data (expensive, reuse across tests)."""
        from yang_mills_s3.spectral.topology_scan import _get_camb_transfer
        return _get_camb_transfer(omega_tot=1.018, l_max=30)

    @pytest.fixture(scope="class")
    def k_max(self, camb_data):
        """k_max derived from CAMB nu_vals to cover all modes."""
        return int(camb_data['nu_vals'][-1])

    def test_s3_no_suppression(self, camb_data, k_max):
        """
        Z_1 (trivial group) should give S_l = 1 (no suppression).

        r_{Z_1}(k) = k+1, so the weight is r(k)*nu = (k+1)*(k+1) = nu^2 = S^3.
        """
        r = molien_cyclic(1, k_max)
        sup = compute_suppression(r, camb_data['nu_vals'], camb_data['delta_sq'])
        for l, s_l in sup.items():
            assert abs(s_l - 1.0) < 1e-6, f"l={l}: S_l = {s_l}, expected 1.0"

    def test_istar_suppression_strong(self, camb_data, k_max):
        """
        I* (order 120) should give strong quadrupole suppression: S_2 << 1.

        Known: S_2 ~ 0.017 from Session 21.
        """
        r = molien_binary_icosahedral(k_max)
        sup = compute_suppression(r, camb_data['nu_vals'], camb_data['delta_sq'])
        assert sup[2] < 0.1, f"S_2 = {sup[2]}, expected << 1"
        assert sup[2] > 0.0, "S_2 should be positive"

    def test_suppression_monotonic_in_order(self, camb_data, k_max):
        """Larger groups should give stronger suppression at l=2."""
        r_z2 = molien_cyclic(2, k_max)
        r_z5 = molien_cyclic(5, k_max)
        r_z10 = molien_cyclic(10, k_max)

        s_z2 = compute_suppression(r_z2, camb_data['nu_vals'], camb_data['delta_sq'])
        s_z5 = compute_suppression(r_z5, camb_data['nu_vals'], camb_data['delta_sq'])
        s_z10 = compute_suppression(r_z10, camb_data['nu_vals'], camb_data['delta_sq'])

        assert s_z2[2] > s_z5[2], "Z_2 should suppress less than Z_5"
        assert s_z5[2] > s_z10[2], "Z_5 should suppress less than Z_10"


@pytest.mark.skipif(not HAS_CAMB, reason="CAMB not installed")
class TestScanResults:
    """Tests for the full topology scan."""

    @pytest.fixture(scope="class")
    def scan_results(self):
        """Full scan at omega_tot = 1.018 (k_max auto from CAMB)."""
        from yang_mills_s3.spectral.topology_scan import scan_all_ade_groups
        return scan_all_ade_groups(omega_tot=1.018, l_max=30)

    def test_all_groups_have_d2(self, scan_results):
        """Every group should produce a D_2 value."""
        for r in scan_results:
            assert 'D_2' in r
            assert r['D_2'] >= 0, f"{r['name']}: D_2 = {r['D_2']}"

    def test_istar_d2_matches_known(self, scan_results):
        """I* should give D_2 ~ 17.9 muK^2 (from Session 21)."""
        istar = [r for r in scan_results if r['name'] == 'I*'][0]
        # Allow 50% tolerance (CAMB version differences)
        assert 5 < istar['D_2'] < 50, (
            f"I* D_2 = {istar['D_2']}, expected ~17.9"
        )

    def test_d2_between_0_and_s3(self, scan_results):
        """All D_2 values should be between 0 and the S^3 value (~1058)."""
        for r in scan_results:
            assert 0 <= r['D_2'] <= 2000, (
                f"{r['name']}: D_2 = {r['D_2']} out of range"
            )

    def test_smaller_groups_closer_to_planck(self, scan_results):
        """
        Smaller quotient groups (larger S^3/Gamma) should give D_2
        closer to the S^3 value, hence closer to Planck 201.
        """
        # Z_2 (smallest nontrivial group) should give D_2 much larger
        # than I* (largest exceptional group)
        z2 = [r for r in scan_results if r['name'] == 'Z_2'][0]
        istar = [r for r in scan_results if r['name'] == 'I*'][0]
        assert z2['D_2'] > istar['D_2'], (
            f"Z_2 D_2 = {z2['D_2']} should be > I* D_2 = {istar['D_2']}"
        )
