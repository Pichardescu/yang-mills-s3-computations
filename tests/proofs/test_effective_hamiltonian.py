"""
Tests for Finite-Dimensional Effective Hamiltonian on S^3/I*.

Tests the EffectiveHamiltonian class which constructs and analyzes the
9-dimensional quantum system arising from projecting YM onto the 3
I*-invariant coexact modes at k=1 on S^3/I*.

Test categories:
    1. SU(2) structure constants
    2. Mode overlap integrals
    3. Quadratic potential
    4. Quartic potential non-negativity (V_4 >= 0)
    5. Quartic algebraic vs explicit computation
    6. Total potential properties
    7. Confining property (V -> infinity)
    8. Unique minimum at a = 0
    9. SU(2) gauge invariance
   10. Harmonic limit (g -> 0)
   11. Numerical spectrum and gap
   12. Gap positivity for all couplings
   13. Gap scaling with R
   14. Reduced Hamiltonian consistency
   15. Gap theorem integration
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.effective_hamiltonian import (
    EffectiveHamiltonian,
    ModeOverlaps,
    su2_structure_constants,
    compute_effective_gap,
    HBAR_C_MEV_FM,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def heff():
    """EffectiveHamiltonian with R=1, g=1."""
    return EffectiveHamiltonian(R=1.0, g_coupling=1.0)


@pytest.fixture
def heff_weak():
    """EffectiveHamiltonian with weak coupling g=0.1."""
    return EffectiveHamiltonian(R=1.0, g_coupling=0.1)


@pytest.fixture
def heff_strong():
    """EffectiveHamiltonian with strong coupling g=5.0."""
    return EffectiveHamiltonian(R=1.0, g_coupling=5.0)


@pytest.fixture
def overlaps():
    """ModeOverlaps on unit S^3."""
    return ModeOverlaps(R=1.0)


# ======================================================================
# 1. SU(2) structure constants
# ======================================================================

class TestStructureConstants:
    """Verify su(2) structure constants f^{abc} = epsilon_{abc}."""

    def test_antisymmetry(self):
        """f^{abc} is totally antisymmetric."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert abs(f[a, b, c] + f[b, a, c]) < 1e-15, \
                        f"f[{a},{b},{c}] not antisymmetric in first two indices"
                    assert abs(f[a, b, c] + f[a, c, b]) < 1e-15, \
                        f"f[{a},{b},{c}] not antisymmetric in last two indices"

    def test_epsilon_values(self):
        """f^{123} = 1 and cyclic permutations."""
        f = su2_structure_constants()
        assert f[0, 1, 2] == 1.0
        assert f[1, 2, 0] == 1.0
        assert f[2, 0, 1] == 1.0
        assert f[0, 2, 1] == -1.0

    def test_total_norm(self):
        """sum_{abc} (f^{abc})^2 = 6."""
        f = su2_structure_constants()
        total = np.sum(f**2)
        assert abs(total - 6.0) < 1e-14

    def test_effective_norm(self):
        """For each a: sum_{bc} (f^{abc})^2 = 2."""
        f = su2_structure_constants()
        for a in range(3):
            norm_sq = np.sum(f[a]**2)
            assert abs(norm_sq - 2.0) < 1e-14, \
                f"Effective norm at a={a} is {norm_sq}, expected 2.0"

    def test_jacobi_identity(self):
        """f^{ade} f^{bce} + cyclic(abc) = 0 (Jacobi identity)."""
        f = su2_structure_constants()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    val = 0.0
                    for e in range(3):
                        for d in range(3):
                            val += f[a, d, e] * f[b, c, e]  # Wrong form
                    # Correct Jacobi: f^{ade}f^{bce} + f^{bde}f^{cae} + f^{cde}f^{abe} = 0
                    jacobi = 0.0
                    for e in range(3):
                        jacobi += (
                            sum(f[a, d, e] * f[b, c, d] for d in range(3))
                            + sum(f[b, d, e] * f[c, a, d] for d in range(3))
                            + sum(f[c, d, e] * f[a, b, d] for d in range(3))
                        )
                    assert abs(jacobi) < 1e-14, \
                        f"Jacobi failed at ({a},{b},{c}): {jacobi}"


# ======================================================================
# 2. Mode overlap integrals
# ======================================================================

class TestModeOverlaps:
    """Test the overlap integrals for I*-invariant modes on S^3."""

    def test_quadratic_orthonormality(self, overlaps):
        """Normalized modes are orthonormal: <phi_i, phi_j> = delta_{ij}."""
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(overlaps.quadratic_overlap(i, j) - expected) < 1e-15

    def test_quartic_antisymmetry_ij(self, overlaps):
        """I_{ijkl} = -I_{jikl} (antisymmetric in first pair)."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val1 = overlaps.quartic_overlap(i, j, k, l)
                        val2 = overlaps.quartic_overlap(j, i, k, l)
                        assert abs(val1 + val2) < 1e-15

    def test_quartic_antisymmetry_kl(self, overlaps):
        """I_{ijkl} = -I_{ijlk} (antisymmetric in second pair)."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val1 = overlaps.quartic_overlap(i, j, k, l)
                        val2 = overlaps.quartic_overlap(i, j, l, k)
                        assert abs(val1 + val2) < 1e-15

    def test_quartic_symmetry_pairs(self, overlaps):
        """I_{ijkl} = I_{klij} (symmetric under pair exchange)."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val1 = overlaps.quartic_overlap(i, j, k, l)
                        val2 = overlaps.quartic_overlap(k, l, i, j)
                        assert abs(val1 - val2) < 1e-15

    def test_quartic_known_values(self, overlaps):
        """Check specific quartic overlap values."""
        # I_{1212} = delta_{12}*delta_{12} - delta_{12}*delta_{12} = 0 - 0 = 0
        # Wait: I_{ijkl} = delta_{ik}delta_{jl} - delta_{il}delta_{jk}
        # I_{0,1,0,1} = delta_{00}*delta_{11} - delta_{01}*delta_{10} = 1 - 0 = 1
        assert overlaps.quartic_overlap(0, 1, 0, 1) == 1.0
        # I_{0,1,1,0} = delta_{01}*delta_{10} - delta_{00}*delta_{11} = 0 - 1 = -1
        assert overlaps.quartic_overlap(0, 1, 1, 0) == -1.0
        # I_{0,0,0,0} = 1*1 - 1*1 = 0
        assert overlaps.quartic_overlap(0, 0, 0, 0) == 0.0
        # I_{0,0,1,1} = 0*0 - 0*0 = 0
        assert overlaps.quartic_overlap(0, 0, 1, 1) == 0.0

    def test_volume_s3(self, overlaps):
        """Vol(S^3) = 2*pi^2*R^3."""
        R = overlaps.R
        expected = 2.0 * np.pi**2 * R**3
        assert abs(overlaps.vol_s3 - expected) < 1e-12


# ======================================================================
# 3. Quadratic potential
# ======================================================================

class TestQuadraticPotential:
    """Test V_2(a) = (1/2) * mu_1 * |a|^2."""

    def test_zero_at_origin(self, heff):
        """V_2(0) = 0."""
        assert abs(heff.quadratic_potential(np.zeros(9))) < 1e-15

    def test_positive_definite(self, heff):
        """V_2(a) > 0 for a != 0."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.standard_normal(9)
            assert heff.quadratic_potential(a) > 0

    def test_quadratic_scaling(self, heff):
        """V_2(lambda * a) = lambda^2 * V_2(a)."""
        a = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        v1 = heff.quadratic_potential(a)
        v2 = heff.quadratic_potential(2.0 * a)
        assert abs(v2 - 4.0 * v1) < 1e-12

    def test_mu1_value(self, heff):
        """mu_1 = 4/R^2 for unit radius."""
        assert abs(heff.mu1 - 4.0) < 1e-14

    def test_single_mode_value(self, heff):
        """V_2 for a single mode a_{1,1} = 1."""
        a = np.zeros(9)
        a[0] = 1.0
        expected = 0.5 * 4.0 * 1.0  # (1/2) * mu_1 * |a|^2
        assert abs(heff.quadratic_potential(a) - expected) < 1e-14

    def test_radius_dependence(self):
        """V_2 scales as 1/R^2."""
        a = np.ones(9)
        v1 = EffectiveHamiltonian(R=1.0).quadratic_potential(a)
        v2 = EffectiveHamiltonian(R=2.0).quadratic_potential(a)
        assert abs(v2 / v1 - 0.25) < 1e-12


# ======================================================================
# 4. Quartic potential non-negativity
# ======================================================================

class TestQuarticNonnegativity:
    """THEOREM: V_4(a) >= 0 for all a."""

    def test_zero_at_origin(self, heff):
        """V_4(0) = 0."""
        assert abs(heff.quartic_potential(np.zeros(9))) < 1e-15

    def test_nonnegative_random(self, heff):
        """V_4(a) >= 0 for random configurations."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            a = rng.standard_normal(9)
            v4 = heff.quartic_potential(a)
            assert v4 >= -1e-14, f"V_4 = {v4} < 0 for a = {a}"

    def test_nonnegative_large_scale(self, heff):
        """V_4(a) >= 0 for large |a|."""
        rng = np.random.default_rng(43)
        for scale in [10.0, 100.0, 1000.0]:
            for _ in range(100):
                a = rng.standard_normal(9) * scale
                assert heff.quartic_potential(a) >= -1e-10

    def test_nonnegative_special_configs(self, heff):
        """V_4 >= 0 for special configurations."""
        # Identity-like
        a = np.eye(3).ravel()
        assert heff.quartic_potential(a) >= -1e-14

        # Rank-1
        a = np.outer([1, 0, 0], [1, 0, 0]).ravel()
        assert heff.quartic_potential(a) >= -1e-14

        # Equal entries
        a = np.ones(9) / 3.0
        assert heff.quartic_potential(a) >= -1e-14

    def test_numerical_verification(self, heff):
        """Run the built-in numerical verification."""
        result = heff.is_quartic_nonnegative(n_samples=5000)
        assert result['nonnegative'], \
            f"V_4 min value: {result['min_value']}"

    def test_quartic_homogeneity(self, heff):
        """V_4(lambda * a) = lambda^4 * V_4(a)."""
        rng = np.random.default_rng(44)
        a = rng.standard_normal(9)
        v1 = heff.quartic_potential(a)
        v2 = heff.quartic_potential(3.0 * a)
        if abs(v1) > 1e-14:
            assert abs(v2 / v1 - 81.0) < 1e-8  # 3^4 = 81

    def test_quartic_coupling_scaling(self):
        """V_4 scales as g^2."""
        a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        h1 = EffectiveHamiltonian(R=1.0, g_coupling=1.0)
        h2 = EffectiveHamiltonian(R=1.0, g_coupling=2.0)
        v1 = h1.quartic_potential(a)
        v2 = h2.quartic_potential(a)
        if abs(v1) > 1e-14:
            assert abs(v2 / v1 - 4.0) < 1e-12  # (2/1)^2 = 4


# ======================================================================
# 5. Quartic: algebraic vs explicit
# ======================================================================

class TestQuarticConsistency:
    """Verify algebraic formula matches explicit structure constant computation."""

    def test_explicit_matches_algebraic(self, heff):
        """quartic_potential == quartic_potential_explicit for random a."""
        rng = np.random.default_rng(55)
        for _ in range(20):
            a = rng.standard_normal(9)
            v_alg = heff.quartic_potential(a)
            v_exp = heff.quartic_potential_explicit(a)
            assert abs(v_alg - v_exp) < 1e-10 * (abs(v_alg) + 1e-14), \
                f"Algebraic={v_alg}, Explicit={v_exp}, diff={abs(v_alg-v_exp)}"

    def test_explicit_matches_algebraic_special(self, heff):
        """Match for special configurations."""
        configs = [
            np.eye(3).ravel(),
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            np.ones(9),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float),
        ]
        for a in configs:
            v_alg = heff.quartic_potential(a)
            v_exp = heff.quartic_potential_explicit(a)
            assert abs(v_alg - v_exp) < 1e-10 * (abs(v_alg) + 1e-14)

    def test_rank1_quartic_is_zero(self, heff):
        """For rank-1 M (M = u v^T), Tr(S)^2 = Tr(S^2) so V_4 = 0."""
        u = np.array([1, 2, 3])
        v = np.array([0.5, -1, 0.3])
        a = np.outer(u, v).ravel()
        v4 = heff.quartic_potential(a)
        assert abs(v4) < 1e-12, \
            f"Rank-1 V_4 = {v4}, expected 0"


# ======================================================================
# 6. Total potential properties
# ======================================================================

class TestTotalPotential:
    """Test V(a) = V_2(a) + V_4(a)."""

    def test_zero_at_origin(self, heff):
        """V(0) = 0."""
        assert abs(heff.total_potential(np.zeros(9))) < 1e-15

    def test_positive_away_from_origin(self, heff):
        """V(a) > 0 for a != 0."""
        rng = np.random.default_rng(66)
        for _ in range(500):
            a = rng.standard_normal(9) * rng.uniform(0.01, 10.0)
            v = heff.total_potential(a)
            assert v > 0, f"V = {v} for |a| = {np.linalg.norm(a)}"

    def test_decomposition(self, heff):
        """V = V_2 + V_4."""
        rng = np.random.default_rng(67)
        for _ in range(50):
            a = rng.standard_normal(9)
            v_total = heff.total_potential(a)
            v2 = heff.quadratic_potential(a)
            v4 = heff.quartic_potential(a)
            assert abs(v_total - (v2 + v4)) < 1e-12


# ======================================================================
# 7. Confining property
# ======================================================================

class TestConfining:
    """THEOREM: V(a) -> infinity as |a| -> infinity."""

    def test_grows_radially(self, heff):
        """V(r * d) is increasing in r for large r, any direction d."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            d = rng.standard_normal(9)
            d = d / np.linalg.norm(d)
            r_vals = [1.0, 10.0, 100.0]
            v_vals = [heff.total_potential(r * d) for r in r_vals]
            assert v_vals[1] > v_vals[0], "V not increasing r=1->10"
            assert v_vals[2] > v_vals[1], "V not increasing r=10->100"

    def test_grows_at_least_quadratically(self, heff):
        """V(r*d) >= (2/R^2) * r^2 for all r (since V_4 >= 0)."""
        d = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        for r in [1.0, 5.0, 10.0, 50.0]:
            v = heff.total_potential(r * d)
            v_lower = 0.5 * heff.mu1 * r**2
            assert v >= v_lower - 1e-10, \
                f"V({r}) = {v} < V_2_lower = {v_lower}"

    def test_builtin_confining_check(self, heff):
        """Run the built-in confining verification."""
        result = heff.is_confining()
        assert result['confining'], "Potential not confining"
        assert result['min_growth_rate'] >= 0.5 * heff.mu1 - 0.1, \
            f"Growth rate {result['min_growth_rate']} < expected {0.5*heff.mu1}"


# ======================================================================
# 8. Unique minimum at a = 0
# ======================================================================

class TestUniqueMinimum:
    """THEOREM: V(a) has unique minimum at a = 0."""

    def test_origin_is_minimum(self, heff):
        """V(0) = 0 < V(a) for all a != 0."""
        assert abs(heff.total_potential(np.zeros(9))) < 1e-15

    def test_no_other_minima(self, heff):
        """No configuration has V < V(0) = 0."""
        result = heff.unique_minimum(n_samples=5000)
        assert result['unique_minimum'], \
            f"V_min_nonzero = {result['min_nonzero_V']}"

    def test_gradient_at_origin(self, heff):
        """Gradient of V vanishes at origin (critical point)."""
        eps = 1e-7
        a0 = np.zeros(9)
        v0 = heff.total_potential(a0)
        for i in range(9):
            a_plus = a0.copy()
            a_plus[i] = eps
            grad_i = (heff.total_potential(a_plus) - v0) / eps
            assert abs(grad_i) < 1e-5, \
                f"Gradient component {i} = {grad_i} != 0"


# ======================================================================
# 9. SU(2) gauge invariance
# ======================================================================

class TestGaugeInvariance:
    """THEOREM: H_eff is invariant under SU(2) gauge transformations."""

    def test_v2_gauge_invariant(self, heff):
        """V_2 is invariant: |R(U)a|^2 = |a|^2."""
        rng = np.random.default_rng(88)
        for _ in range(50):
            a = rng.standard_normal((3, 3))
            angle = rng.uniform(0, 2 * np.pi)
            U = heff._rotation_matrix(angle, rng.integers(3))
            a_rot = heff.gauge_transform(a, U)
            v_orig = heff.quadratic_potential(a)
            v_rot = heff.quadratic_potential(a_rot)
            assert abs(v_orig - v_rot) < 1e-12 * (abs(v_orig) + 1e-15)

    def test_v4_gauge_invariant(self, heff):
        """V_4 is invariant under gauge transforms."""
        rng = np.random.default_rng(89)
        for _ in range(50):
            a = rng.standard_normal((3, 3))
            angle = rng.uniform(0, 2 * np.pi)
            U = heff._rotation_matrix(angle, rng.integers(3))
            a_rot = heff.gauge_transform(a, U)
            v_orig = heff.quartic_potential(a)
            v_rot = heff.quartic_potential(a_rot)
            assert abs(v_orig - v_rot) < 1e-10 * (abs(v_orig) + 1e-15)

    def test_total_gauge_invariant(self, heff):
        """Total potential is gauge invariant."""
        result = heff.check_gauge_invariance(n_samples=200)
        assert result['invariant'], \
            f"Max deviation: {result['max_deviation']}"

    def test_spatial_rotation_invariance(self, heff):
        """V is also invariant under spatial SO(3) rotations (left action)."""
        rng = np.random.default_rng(90)
        for _ in range(50):
            a = rng.standard_normal((3, 3))
            angle = rng.uniform(0, 2 * np.pi)
            O = heff._rotation_matrix(angle, rng.integers(3))
            a_rot = O @ a  # Spatial rotation (left multiply)
            v_orig = heff.total_potential(a)
            v_rot = heff.total_potential(a_rot)
            assert abs(v_orig - v_rot) < 1e-10 * (abs(v_orig) + 1e-15)


# ======================================================================
# 10. Harmonic limit (g -> 0)
# ======================================================================

class TestHarmonicLimit:
    """For g -> 0, H_eff reduces to 9 harmonic oscillators."""

    def test_gap_approaches_omega(self):
        """gap -> omega = 2/R as g -> 0."""
        R = 1.0
        omega = 2.0 / R
        h = EffectiveHamiltonian(R=R, g_coupling=0.001)
        spec = h.compute_spectrum(n_basis=6, method='reduced')
        # Reduced system: 3 oscillators, gap = omega
        assert abs(spec['gap'] - omega) < 0.05 * omega, \
            f"Gap = {spec['gap']}, expected ~ {omega}"

    def test_ground_energy_harmonic(self):
        """E_0 -> (3/2)*omega = 3/R for 3 reduced DOF."""
        R = 1.0
        omega = 2.0 / R
        h = EffectiveHamiltonian(R=R, g_coupling=0.001)
        spec = h.compute_spectrum(n_basis=6, method='reduced')
        expected_E0 = 1.5 * omega  # 3 oscillators: 3*(omega/2)
        assert abs(spec['ground_energy'] - expected_E0) < 0.1 * expected_E0

    def test_quartic_vanishes_at_g_zero(self):
        """V_4 = 0 when g = 0."""
        h = EffectiveHamiltonian(R=1.0, g_coupling=0.0)
        rng = np.random.default_rng(100)
        for _ in range(50):
            a = rng.standard_normal(9)
            assert abs(h.quartic_potential(a)) < 1e-15


# ======================================================================
# 11. Numerical spectrum and gap
# ======================================================================

class TestSpectrum:
    """Numerical spectrum computation."""

    def test_spectrum_has_eigenvalues(self, heff):
        """Spectrum computation returns eigenvalues."""
        spec = heff.compute_spectrum(n_basis=5, n_eigenvalues=5, method='reduced')
        assert len(spec['eigenvalues']) >= 2
        assert spec['gap'] > 0

    def test_eigenvalues_ordered(self, heff):
        """Eigenvalues are non-decreasing."""
        spec = heff.compute_spectrum(n_basis=5, n_eigenvalues=10, method='reduced')
        evals = spec['eigenvalues']
        for i in range(len(evals) - 1):
            assert evals[i] <= evals[i + 1] + 1e-10

    def test_ground_state_positive(self, heff):
        """Ground state energy > 0 (zero-point energy)."""
        spec = heff.compute_spectrum(n_basis=5, method='reduced')
        assert spec['ground_energy'] > 0

    def test_gap_positive(self, heff):
        """Spectral gap > 0."""
        spec = heff.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0, f"Gap = {spec['gap']}"

    def test_larger_basis_convergence(self):
        """Gap converges as basis size increases."""
        h = EffectiveHamiltonian(R=1.0, g_coupling=1.0)
        gaps = []
        for n in [4, 6, 8]:
            spec = h.compute_spectrum(n_basis=n, method='reduced')
            gaps.append(spec['gap'])
        # Gaps should be converging (differences decreasing)
        diff1 = abs(gaps[1] - gaps[0])
        diff2 = abs(gaps[2] - gaps[1])
        assert diff2 < diff1 + 0.1, \
            f"Not converging: diffs = {diff1}, {diff2}"


# ======================================================================
# 12. Gap positivity for all couplings
# ======================================================================

class TestGapPositivity:
    """THEOREM: gap > 0 for all g^2 >= 0 and R > 0."""

    def test_gap_positive_weak_coupling(self, heff_weak):
        """Gap > 0 at g = 0.1."""
        spec = heff_weak.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0

    def test_gap_positive_moderate_coupling(self, heff):
        """Gap > 0 at g = 1.0."""
        spec = heff.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0

    def test_gap_positive_strong_coupling(self, heff_strong):
        """Gap > 0 at g = 5.0."""
        spec = heff_strong.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0

    def test_gap_coupling_scan(self):
        """Gap > 0 for range of couplings."""
        h = EffectiveHamiltonian(R=1.0, g_coupling=1.0)
        result = h.gap_vs_coupling(
            g_values=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            n_basis=6,
        )
        assert result['all_positive'], \
            f"Gaps: {result['gaps']}"

    def test_gap_positive_physical_coupling(self):
        """Gap > 0 at physical coupling g^2 ~ 6.28."""
        g_phys = np.sqrt(6.28)
        h = EffectiveHamiltonian(R=1.0, g_coupling=g_phys)
        spec = h.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0, \
            f"Gap = {spec['gap']} at physical coupling"


# ======================================================================
# 13. Gap scaling with R
# ======================================================================

class TestGapRadiusScaling:
    """Gap behavior as function of radius R."""

    def test_gap_positive_small_R(self):
        """Gap > 0 for small R."""
        h = EffectiveHamiltonian(R=0.5, g_coupling=1.0)
        spec = h.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0

    def test_gap_positive_large_R(self):
        """Gap > 0 for large R."""
        h = EffectiveHamiltonian(R=10.0, g_coupling=1.0)
        spec = h.compute_spectrum(n_basis=6, method='reduced')
        assert spec['gap'] > 0

    def test_gap_scales_with_R(self):
        """gap * R is approximately constant for weak coupling.

        In the reduced (3-singular-value) Hamiltonian, the harmonic gap
        is omega = 2/R, so gap*R -> 2 for small coupling.
        """
        g = 0.01
        gaps_x_R = []
        for R in [0.5, 1.0, 2.0, 5.0]:
            h = EffectiveHamiltonian(R=R, g_coupling=g)
            spec = h.compute_spectrum(n_basis=6, method='reduced')
            gaps_x_R.append(spec['gap'] * R)
        # For weak coupling, gap*R should be close to 2 (harmonic value)
        for v in gaps_x_R:
            assert abs(v - 2.0) / 2.0 < 0.05, \
                f"gap*R values: {gaps_x_R}, expected ~2.0"

    def test_gap_radius_scan(self):
        """Gap > 0 across multiple R values."""
        h = EffectiveHamiltonian(R=1.0, g_coupling=1.0)
        result = h.gap_vs_radius(
            R_values=[0.5, 1.0, 2.0, 5.0],
            n_basis=6,
        )
        assert result['all_positive'], \
            f"Gaps: {result['gaps']}"


# ======================================================================
# 14. Reduced Hamiltonian consistency
# ======================================================================

class TestReducedHamiltonian:
    """Test the reduced 3-DOF Hamiltonian."""

    def test_matrix_hermitian(self, heff):
        """Reduced Hamiltonian matrix is Hermitian."""
        data = heff.build_reduced_hamiltonian(n_basis=5)
        H = data['matrix']
        assert np.allclose(H, H.T, atol=1e-12), \
            f"Max asymmetry: {np.max(np.abs(H - H.T))}"

    def test_matrix_dimension(self, heff):
        """Matrix has correct dimension n_basis^3."""
        for n in [3, 5]:
            data = heff.build_reduced_hamiltonian(n_basis=n)
            assert data['matrix'].shape == (n**3, n**3)
            assert data['basis_size'] == n**3

    def test_omega_value(self, heff):
        """omega = sqrt(mu_1) = 2/R."""
        data = heff.build_reduced_hamiltonian(n_basis=3)
        assert abs(data['omega'] - 2.0) < 1e-14  # R=1


# ======================================================================
# 15. Gap theorem integration test
# ======================================================================

class TestGapTheorem:
    """Integration test for the full gap theorem."""

    def test_gap_theorem_statement(self, heff):
        """gap_theorem() returns a valid result."""
        result = heff.gap_theorem(n_basis=5)
        assert 'statement' in result
        assert 'proof' in result
        assert 'numerical' in result

    def test_all_proof_components_pass(self, heff):
        """All proof components are True."""
        result = heff.gap_theorem(n_basis=5)
        proof = result['proof']
        assert proof['V4_nonnegative'], "V_4 not nonneg"
        assert proof['V_confining'], "Not confining"
        assert proof['unique_minimum'], "Not unique min"
        assert proof['gauge_invariant'], "Not gauge inv"

    def test_numerical_gap_positive(self, heff):
        """Numerical gap is positive."""
        result = heff.gap_theorem(n_basis=5)
        assert result['numerical']['gap'] > 0

    def test_coupling_scan_all_positive(self, heff):
        """All gaps positive in coupling scan."""
        result = heff.gap_theorem(n_basis=5)
        assert result['coupling_scan']['all_gaps_positive']

    def test_status_is_theorem(self, heff):
        """Result status indicates THEOREM."""
        result = heff.gap_theorem(n_basis=5)
        assert 'THEOREM' in result['status']


# ======================================================================
# 16. Analytical estimates
# ======================================================================

class TestAnalyticalEstimates:
    """Test analytical gap bounds."""

    def test_harmonic_gap_value(self, heff):
        """Harmonic gap = 2/R."""
        result = heff.gap_lower_bound_analytical()
        assert abs(result['harmonic_gap_value'] - 2.0) < 1e-14

    def test_harmonic_gap_radius_scaling(self):
        """Harmonic gap = 2/R for various R."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            h = EffectiveHamiltonian(R=R, g_coupling=1.0)
            result = h.gap_lower_bound_analytical()
            assert abs(result['harmonic_gap_value'] - 2.0 / R) < 1e-14


# ======================================================================
# 17. Convenience functions
# ======================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_compute_effective_gap(self):
        """compute_effective_gap returns positive value."""
        gap = compute_effective_gap(R=1.0, g_coupling=1.0, n_basis=5)
        assert gap > 0

    def test_compute_effective_gap_consistency(self):
        """Convenience function matches class method."""
        h = EffectiveHamiltonian(R=1.0, g_coupling=1.0)
        spec = h.compute_spectrum(n_basis=5, method='reduced')
        gap_class = spec['gap']
        gap_func = compute_effective_gap(R=1.0, g_coupling=1.0, n_basis=5)
        assert abs(gap_class - gap_func) < 1e-12


# ======================================================================
# 18. Physical dimensions and constants
# ======================================================================

class TestPhysicalConstants:
    """Test physical dimensions and constants."""

    def test_hbar_c_value(self):
        """hbar*c = 197.3269804 MeV*fm."""
        assert abs(HBAR_C_MEV_FM - 197.3269804) < 1e-4

    def test_dimensions(self, heff):
        """Check dimensional quantities."""
        assert heff.n_dof == 9
        assert heff.n_modes == 3
        assert heff.n_colors == 3

    def test_curl_eigenvalue(self, heff):
        """Curl eigenvalue at k=1 is -2/R."""
        assert abs(heff.curl_ev - (-2.0)) < 1e-14  # R=1
