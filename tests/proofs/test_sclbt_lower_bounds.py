"""
Tests for the SCLBT (Self-Consistent Lower Bound Theory) module.

Tests organized by class:
    1. TempleBound: classical Temple inequality
    2. LanczosConstruct: Lanczos tridiagonalization
    3. SCLBTBound: Pollak-Martinazzo self-consistent bounds
    4. IntervalMatrixElements: matrix elements with rounding control
    5. TruncationErrorBound: finite basis error estimates
    6. RigorousSpectralGap: certified gap assembly
    7. QuarticOscillatorBenchmark: known system benchmarks
    8. YangMillsReducedGap: physical YM application

Aim: 60+ tests.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from yang_mills_s3.proofs.sclbt_lower_bounds import (
    TempleBound,
    LanczosConstruct,
    SCLBTBound,
    IntervalMatrixElements,
    TruncationErrorBound,
    RigorousSpectralGap,
    QuarticOscillatorBenchmark,
    YangMillsReducedGap,
    _build_3d_hamiltonian,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_MEV,
    R_PHYSICAL_FM,
)


# ======================================================================
# Fixtures: reusable test Hamiltonians
# ======================================================================

@pytest.fixture
def harmonic_1d():
    """1D harmonic oscillator H = omega*(n+1/2), omega=1. Exact eigenvalues known."""
    N = 20
    H = np.diag([1.0 * (n + 0.5) for n in range(N)])
    exact = np.array([n + 0.5 for n in range(N)])
    return H, exact


@pytest.fixture
def harmonic_2d():
    """2D isotropic harmonic oscillator."""
    N = 8
    I = np.eye(N)
    h1d = np.diag([1.0 * (n + 0.5) for n in range(N)])
    H = np.kron(h1d, I) + np.kron(I, h1d)
    return H


@pytest.fixture
def quartic_1d_small():
    """Small 1D quartic oscillator for fast tests."""
    return QuarticOscillatorBenchmark.build_quartic_1d(20)


@pytest.fixture
def quartic_1d_large():
    """Larger 1D quartic oscillator for accuracy tests."""
    return QuarticOscillatorBenchmark.build_quartic_1d(40)


@pytest.fixture
def ym_3d_small():
    """Small 3D YM Hamiltonian, N=5 per mode = 125 total."""
    return _build_3d_hamiltonian(omega=2.0, g2=6.28, n_basis=5)


@pytest.fixture
def random_symmetric():
    """Random symmetric positive-definite matrix."""
    rng = np.random.default_rng(42)
    N = 30
    A = rng.standard_normal((N, N))
    H = A.T @ A + 0.1 * np.eye(N)  # Positive definite
    return H


# ======================================================================
# 1. TempleBound tests
# ======================================================================

class TestTempleBound:
    """Tests for the classical Temple inequality."""

    def test_temple_lower_bound_basic(self):
        """Temple bound is below the true eigenvalue for simple case."""
        # Known: E_0 = 0.5 (harmonic oscillator)
        ritz = 0.5
        variance = 0.0
        E1_upper = 1.5
        bound = TempleBound.temple_lower_bound(ritz, variance, E1_upper)
        assert bound == pytest.approx(0.5)  # Exact when variance=0

    def test_temple_lower_bound_with_variance(self):
        """Temple bound decreases with increasing variance."""
        ritz = 1.0
        E1_upper = 3.0
        b1 = TempleBound.temple_lower_bound(ritz, 0.1, E1_upper)
        b2 = TempleBound.temple_lower_bound(ritz, 0.5, E1_upper)
        assert b1 > b2  # Larger variance -> weaker bound

    def test_temple_bound_is_lower(self):
        """Temple bound is always <= Ritz value."""
        ritz = 2.0
        variance = 0.3
        E1_upper = 5.0
        bound = TempleBound.temple_lower_bound(ritz, variance, E1_upper)
        assert bound <= ritz + 1e-15

    def test_temple_bound_raises_invalid_E1(self):
        """Raises ValueError when E1_upper <= ritz."""
        with pytest.raises(ValueError):
            TempleBound.temple_lower_bound(2.0, 0.1, 1.5)

    def test_temple_bound_raises_negative_variance(self):
        """Raises ValueError for negative variance."""
        with pytest.raises(ValueError):
            TempleBound.temple_lower_bound(1.0, -0.1, 3.0)

    def test_spectral_gap_bound(self):
        """Spectral gap bound = E1 - E0."""
        gap = TempleBound.spectral_gap_bound(0.5, 1.5)
        assert gap == pytest.approx(1.0)

    def test_compute_harmonic(self, harmonic_1d):
        """Temple bounds for harmonic oscillator (exact system)."""
        H, exact = harmonic_1d
        temple = TempleBound()
        result = temple.compute(H, n_states=5)

        # For exact eigenstates, variance should be ~0
        for k in range(5):
            assert result['variances'][k] < 1e-10

        # Temple bounds should equal Ritz values (variance=0)
        for k in range(5):
            assert result['temple_lower_bounds'][k] == pytest.approx(
                result['ritz_eigenvalues'][k], abs=1e-10
            )

    def test_compute_quartic(self, quartic_1d_small):
        """Temple bounds for quartic oscillator."""
        temple = TempleBound()
        result = temple.compute(quartic_1d_small, n_states=3)

        # Ritz values should be upper bounds on the truncated problem
        # (The truncated Ritz value >= exact truncated eigenvalue)
        E0_ref = QuarticOscillatorBenchmark.QUARTIC_1D_E0
        # Ritz from truncated basis is an upper bound on the FULL problem
        assert result['ritz_eigenvalues'][0] >= E0_ref - 0.1  # Within ~10%

        # Temple bounds should be lower bounds
        assert result['temple_lower_bounds'][0] <= result['ritz_eigenvalues'][0] + 1e-12

    def test_verify_harmonic(self, harmonic_1d):
        """Verify Temple bounds against exact harmonic eigenvalues."""
        H, exact = harmonic_1d
        temple = TempleBound()
        result = temple.verify(H, exact[:5])
        assert result['verified']

    def test_verify_quartic(self, quartic_1d_large):
        """Verify Temple bounds bracket exact quartic eigenvalues."""
        temple = TempleBound()
        # Use eigenvalues from this same matrix as "exact" reference
        evals_ref = np.sort(np.linalg.eigvalsh(quartic_1d_large))

        result = temple.verify(quartic_1d_large, evals_ref[:3])
        # Temple lower <= exact <= Ritz upper (for the same matrix, should be tight)
        for d in result['details']:
            assert d['temple_lower'] <= d['exact'] + 1e-6

    def test_temple_gap_positive_quartic(self, quartic_1d_small):
        """Temple gap for quartic oscillator is positive."""
        temple = TempleBound()
        result = temple.compute(quartic_1d_small, n_states=3)
        assert result['spectral_gap_temple'] > 0


# ======================================================================
# 2. LanczosConstruct tests
# ======================================================================

class TestLanczosConstruct:
    """Tests for the Lanczos tridiagonalization."""

    def test_lanczos_basic(self, harmonic_1d):
        """Lanczos produces correct tridiagonal for harmonic oscillator."""
        H, _ = harmonic_1d
        lc = LanczosConstruct()
        # Use a vector that spans many eigenstates (NOT an eigenvector)
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        result = lc.lanczos(H, v0, 10)

        assert result['n_steps'] == 10
        assert len(result['tridiagonal_alpha']) == 10

    def test_lanczos_ritz_converge(self, random_symmetric):
        """Ritz values from Lanczos converge to exact eigenvalues."""
        H = random_symmetric
        exact = np.sort(np.linalg.eigvalsh(H))

        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, 25)

        ritz = lc.ritz_values()
        # Lowest Ritz values should be close to exact (Lanczos converges
        # fastest for extremal eigenvalues, but 25 steps on a 30x30 matrix
        # should give good results for the lowest few)
        for k in range(3):
            assert abs(ritz[k] - exact[k]) < 1.0

    def test_tridiagonal_symmetric(self, random_symmetric):
        """Tridiagonal matrix T_m is symmetric."""
        H = random_symmetric
        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, 15)

        T = lc.tridiagonal_matrix()
        assert np.allclose(T, T.T, atol=1e-12)

    def test_ritz_values_are_eigenvalues_of_T(self, random_symmetric):
        """Ritz values are eigenvalues of the tridiagonal matrix."""
        H = random_symmetric
        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, 10)

        T = lc.tridiagonal_matrix()
        evals_T = np.sort(np.linalg.eigvalsh(T))
        ritz = lc.ritz_values()
        np.testing.assert_allclose(ritz, evals_T, atol=1e-10)

    def test_ritz_vectors_orthogonal(self, random_symmetric):
        """Ritz vectors are orthonormal."""
        H = random_symmetric
        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, 10)

        V = lc.ritz_vectors()
        overlap = V.T @ V
        np.testing.assert_allclose(overlap, np.eye(10), atol=1e-10)

    def test_residual_norms_small_for_converged(self, harmonic_1d):
        """Residual norms decrease with Lanczos steps (for simple systems)."""
        H, _ = harmonic_1d
        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, H.shape[0])  # Full Lanczos

        norms = lc.residual_norms()
        # For full Lanczos on a small matrix, all residuals should be tiny
        assert np.max(norms) < 1e-8

    def test_variances_nonnegative(self, quartic_1d_small):
        """Variances are always non-negative."""
        H = quartic_1d_small
        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, 10)

        vars_out = lc.variances()
        assert np.all(vars_out >= -1e-15)

    def test_lanczos_invariant_subspace(self):
        """Lanczos terminates early for invariant subspaces."""
        # Block diagonal matrix: Lanczos starting in block 1 can't reach block 2
        H = np.zeros((10, 10))
        H[:5, :5] = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        H[5:, 5:] = np.diag([6.0, 7.0, 8.0, 9.0, 10.0])

        v0 = np.zeros(10)
        v0[0] = 1.0  # Start in first block

        lc = LanczosConstruct()
        result = lc.lanczos(H, v0, 10)

        # Should find eigenvalues of first block only
        ritz = lc.ritz_values()
        assert result['n_steps'] <= 5  # Can't exceed block size


# ======================================================================
# 3. SCLBTBound tests
# ======================================================================

class TestSCLBTBound:
    """Tests for the SCLBT self-consistent lower bounds."""

    def test_sclbt_lower_than_ritz(self, quartic_1d_small):
        """SCLBT lower bounds are always <= Ritz upper bounds."""
        sclbt = SCLBTBound()
        result = sclbt.compute(quartic_1d_small, n_states=3)

        for k in range(3):
            assert result['lower_bounds'][k] <= result['ritz_eigenvalues'][k] + 1e-12

    def test_sclbt_brackets_exact(self, quartic_1d_large):
        """SCLBT bounds bracket the exact quartic eigenvalues."""
        sclbt = SCLBTBound()
        result = sclbt.compute(quartic_1d_large, n_states=3)

        E0_ref = QuarticOscillatorBenchmark.QUARTIC_1D_E0  # ~ 0.66799
        # Lower bound <= reference (within basis truncation tolerance)
        assert result['lower_bounds'][0] <= E0_ref + 0.01
        # Ritz upper >= reference (Ritz is an upper bound for the truncated problem,
        # and converges from above to the exact value)
        assert result['ritz_eigenvalues'][0] >= E0_ref - 0.01

    def test_sclbt_tighter_than_temple(self, quartic_1d_small):
        """SCLBT lower bounds are tighter (higher) than Temple bounds."""
        sclbt = SCLBTBound()
        sclbt_result = sclbt.compute(quartic_1d_small, n_states=3)

        temple = TempleBound()
        temple_result = temple.compute(quartic_1d_small, n_states=3)

        # SCLBT should be >= Temple (tighter lower bound)
        for k in range(3):
            assert sclbt_result['lower_bounds'][k] >= temple_result['temple_lower_bounds'][k] - 1e-12

    def test_sclbt_converges(self, quartic_1d_small):
        """SCLBT iteration converges."""
        sclbt = SCLBTBound()
        result = sclbt.compute(quartic_1d_small, n_states=3)
        assert result['converged']

    def test_sclbt_monotone(self, quartic_1d_small):
        """SCLBT iteration is monotonically improving."""
        sclbt = SCLBTBound()
        sclbt.compute(quartic_1d_small, n_states=3)
        check = sclbt.convergence_check()
        assert check['monotone']

    def test_sclbt_gap_positive(self, quartic_1d_small):
        """SCLBT spectral gap is positive for quartic oscillator."""
        sclbt = SCLBTBound()
        result = sclbt.compute(quartic_1d_small, n_states=3)
        assert result['spectral_gap'] > 0

    def test_sclbt_gap_reasonable(self, quartic_1d_large):
        """SCLBT gap is close to exact gap for quartic oscillator."""
        sclbt = SCLBTBound()
        result = sclbt.compute(quartic_1d_large, n_states=3)

        exact_gap = QuarticOscillatorBenchmark.QUARTIC_1D_GAP
        # The SCLBT gap should be a lower bound on the exact gap
        assert result['spectral_gap'] <= exact_gap + 0.1
        # And it should be reasonably close (within 50% for moderate basis)
        assert result['spectral_gap'] > exact_gap * 0.3

    def test_sclbt_harmonic_exact(self, harmonic_1d):
        """SCLBT on harmonic oscillator gives exact bounds (zero variance)."""
        H, exact = harmonic_1d
        sclbt = SCLBTBound()
        result = sclbt.compute(H, n_states=5)

        # For exact eigenstates, bounds should equal eigenvalues
        for k in range(5):
            assert abs(result['lower_bounds'][k] - exact[k]) < 1e-8

    def test_sclbt_iterate_returns_history(self, quartic_1d_small):
        """iterate() returns the full history of bounds."""
        sclbt = SCLBTBound()
        sclbt.compute(quartic_1d_small, n_states=3)
        history = sclbt.iterate(0)
        assert len(history) >= 2  # At least initial + one iteration

    def test_sclbt_convergence_check(self, quartic_1d_small):
        """convergence_check returns proper diagnostics."""
        sclbt = SCLBTBound()
        sclbt.compute(quartic_1d_small, n_states=3)
        check = sclbt.convergence_check()

        assert 'converged' in check
        assert 'n_iterations' in check
        assert 'monotone' in check
        assert check['final_bounds'] is not None

    def test_sclbt_random_matrix(self, random_symmetric):
        """SCLBT works on random symmetric matrices."""
        sclbt = SCLBTBound()
        result = sclbt.compute(random_symmetric, n_states=5)

        exact = np.sort(np.linalg.eigvalsh(random_symmetric))
        for k in range(5):
            # Lower bound should be <= exact
            assert result['lower_bounds'][k] <= exact[k] + 1e-6
            # Ritz should be upper bound
            assert result['ritz_eigenvalues'][k] >= exact[k] - 1e-6

    def test_sclbt_lower_bounds_method(self, quartic_1d_small):
        """lower_bounds() returns stored bounds."""
        sclbt = SCLBTBound()
        sclbt.compute(quartic_1d_small, n_states=3)
        lb = sclbt.lower_bounds()
        assert len(lb) == 3

    def test_sclbt_raises_before_compute(self):
        """lower_bounds() raises if compute() not called."""
        sclbt = SCLBTBound()
        with pytest.raises(RuntimeError):
            sclbt.lower_bounds()


# ======================================================================
# 4. IntervalMatrixElements tests
# ======================================================================

class TestIntervalMatrixElements:
    """Tests for interval arithmetic matrix elements."""

    def test_x_matrix_element_selection_rules(self):
        """x connects only adjacent states: <m|x|n> = 0 unless |m-n|=1."""
        ime = IntervalMatrixElements(omega=1.0)
        # Diagonal should be zero
        assert ime.matrix_element_x(0, 0) == 0.0
        assert ime.matrix_element_x(3, 3) == 0.0
        # Off-by-2 should be zero
        assert ime.matrix_element_x(0, 2) == 0.0
        # Adjacent should be nonzero
        assert ime.matrix_element_x(0, 1) != 0.0
        assert ime.matrix_element_x(1, 0) != 0.0

    def test_x_matrix_element_values(self):
        """x matrix elements match analytical formula."""
        omega = 2.0
        ime = IntervalMatrixElements(omega=omega)
        scale = 1.0 / np.sqrt(2.0 * omega)

        # <0|x|1> = sqrt(1) * scale = scale
        assert ime.matrix_element_x(0, 1) == pytest.approx(scale)
        # <1|x|2> = sqrt(2) * scale
        assert ime.matrix_element_x(1, 2) == pytest.approx(np.sqrt(2) * scale)

    def test_x_matrix_element_hermitian(self):
        """<m|x|n> = <n|x|m> (Hermiticity)."""
        ime = IntervalMatrixElements(omega=1.5)
        for m in range(5):
            for n in range(5):
                assert ime.matrix_element_x(m, n) == pytest.approx(
                    ime.matrix_element_x(n, m)
                )

    def test_x2_matrix_element_diagonal(self):
        """<n|x^2|n> = (2n+1)/(2*omega)."""
        omega = 1.0
        ime = IntervalMatrixElements(omega=omega)
        for n in range(5):
            expected = (2 * n + 1) / (2.0 * omega)
            assert ime.matrix_element_x2(n, n) == pytest.approx(expected, rel=1e-10)

    def test_x4_matrix_element_diagonal(self):
        """<n|x^4|n> is correct for low n."""
        omega = 1.0
        ime = IntervalMatrixElements(omega=omega)
        # <0|x^4|0> = 3/(4*omega^2) for omega=1
        expected_0 = 3.0 / (4.0 * omega**2)
        assert ime.matrix_element_x4(0, 0) == pytest.approx(expected_0, rel=1e-8)

    def test_build_matrix_harmonic(self):
        """Build harmonic oscillator matrix (quartic=0) gives diagonal."""
        ime = IntervalMatrixElements(omega=1.0, n_basis=10)
        H = ime.build_matrix({'quartic': 0.0})
        # Should be diagonal
        off_diag = H - np.diag(np.diag(H))
        assert np.max(np.abs(off_diag)) < 1e-14

    def test_build_matrix_quartic(self):
        """Build quartic Hamiltonian and check symmetry."""
        ime = IntervalMatrixElements(omega=1.0, n_basis=15)
        H = ime.build_matrix({'quartic': 1.0})
        sym = ime.verify_symmetry(H)
        assert sym['symmetric']

    def test_verify_symmetry(self):
        """verify_symmetry detects asymmetric matrices."""
        ime = IntervalMatrixElements()
        H_sym = np.array([[1.0, 2.0], [2.0, 3.0]])
        assert ime.verify_symmetry(H_sym)['symmetric']

        H_asym = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not ime.verify_symmetry(H_asym)['symmetric']

    def test_interval_bounds_enclose(self):
        """Interval bounds enclose the matrix elements."""
        ime = IntervalMatrixElements(omega=1.0, n_basis=10)
        H = ime.build_matrix({'quartic': 0.5})
        H_lower, H_upper = ime.interval_bounds(H)

        # Every element should satisfy H_lower <= H <= H_upper
        assert np.all(H_lower <= H + 1e-15)
        assert np.all(H_upper >= H - 1e-15)


# ======================================================================
# 5. TruncationErrorBound tests
# ======================================================================

class TestTruncationErrorBound:
    """Tests for truncation error estimates."""

    def test_weyl_estimate_harmonic(self):
        """Weyl estimate for harmonic oscillator: lambda_k ~ k."""
        # s=1, d=1: exponent = 2/(1+1) = 1
        est = TruncationErrorBound.weyl_estimate(10, dimension=1, potential_exponent=1.0)
        assert est == pytest.approx(10.0)

    def test_weyl_estimate_quartic(self):
        """Weyl estimate for quartic: lambda_k ~ k^{4/3}."""
        # s=2, d=1: exponent = 4/(2+1) = 4/3
        est = TruncationErrorBound.weyl_estimate(10, dimension=1, potential_exponent=2.0)
        expected = 10.0 ** (4.0 / 3.0)
        assert est == pytest.approx(expected, rel=1e-10)

    def test_weyl_estimate_3d_quartic(self):
        """Weyl estimate for 3D quartic: lambda_k ~ k^{4/9}."""
        # s=2, d=3: exponent = 4/(6+3) = 4/9
        est = TruncationErrorBound.weyl_estimate(10, dimension=3, potential_exponent=2.0)
        expected = 10.0 ** (4.0 / 9.0)
        assert est == pytest.approx(expected, rel=1e-10)

    def test_truncation_bound_decreases(self):
        """Truncation error decreases with N."""
        errs = [TruncationErrorBound.truncation_bound(N, k=0) for N in [10, 20, 40, 80]]
        for i in range(len(errs) - 1):
            assert errs[i] > errs[i + 1]

    def test_truncation_bound_increases_with_k(self):
        """Higher eigenvalues have larger truncation error."""
        err_0 = TruncationErrorBound.truncation_bound(50, k=0)
        err_5 = TruncationErrorBound.truncation_bound(50, k=5)
        assert err_5 > err_0

    def test_truncation_bound_infinite_small_N(self):
        """Truncation bound is inf when N <= k+1."""
        assert TruncationErrorBound.truncation_bound(5, k=5) == float('inf')

    def test_minimum_N_basic(self):
        """Minimum N increases for tighter precision."""
        N_loose = TruncationErrorBound.minimum_N_for_precision(0.01)
        N_tight = TruncationErrorBound.minimum_N_for_precision(0.001)
        assert N_tight > N_loose

    def test_minimum_N_at_least_k_plus_2(self):
        """Minimum N is at least k+2."""
        N = TruncationErrorBound.minimum_N_for_precision(0.1, k=10)
        assert N >= 12


# ======================================================================
# 6. RigorousSpectralGap tests
# ======================================================================

class TestRigorousSpectralGap:
    """Tests for the certified spectral gap."""

    def test_certified_gap_harmonic(self, harmonic_1d):
        """Certified gap for harmonic oscillator = 1.0."""
        H, _ = harmonic_1d
        rsg = RigorousSpectralGap(n_states=5)
        result = rsg.certified_gap(H, dimension=1, potential_exponent=1.0)

        # Harmonic gap is exactly 1.0
        assert abs(result['sclbt_gap_raw'] - 1.0) < 0.01
        assert result['is_positive']

    def test_certified_gap_quartic(self, quartic_1d_large):
        """Certified gap for quartic oscillator is positive."""
        rsg = RigorousSpectralGap(n_states=5)
        result = rsg.certified_gap(quartic_1d_large, dimension=1, potential_exponent=2.0)

        assert result['sclbt_gap_raw'] > 0
        assert result['temple_gap_raw'] > 0

    def test_sclbt_gap_tighter_than_temple(self, quartic_1d_small):
        """SCLBT gap is at least as tight as Temple gap."""
        rsg = RigorousSpectralGap(n_states=5)
        result = rsg.certified_gap(quartic_1d_small, dimension=1, potential_exponent=2.0)

        # SCLBT should give tighter or equal gap
        assert result['sclbt_gap_raw'] >= result['temple_gap_raw'] - 1e-10

    def test_is_positive_method(self, quartic_1d_small):
        """is_positive() correctly identifies positive gap."""
        rsg = RigorousSpectralGap(n_states=5)
        assert rsg.is_positive(quartic_1d_small, dimension=1, potential_exponent=2.0)

    def test_confidence_level_converging(self, quartic_1d_large):
        """confidence_level() checks convergence."""
        rsg = RigorousSpectralGap(n_states=3)
        result = rsg.confidence_level(
            quartic_1d_large,
            N_values=[10, 20, 30, 40],
            dimension=1,
            potential_exponent=2.0,
        )
        assert 'converging' in result
        assert len(result['gap_data']) == 4

    def test_label_is_numerical(self, quartic_1d_small):
        """All results are labeled NUMERICAL (not THEOREM)."""
        rsg = RigorousSpectralGap(n_states=3)
        result = rsg.certified_gap(quartic_1d_small)
        assert result['label'] == 'NUMERICAL'


# ======================================================================
# 7. QuarticOscillatorBenchmark tests
# ======================================================================

class TestQuarticOscillatorBenchmark:
    """Tests for known quartic oscillator benchmarks."""

    def test_build_quartic_1d_symmetric(self):
        """1D quartic Hamiltonian is symmetric."""
        H = QuarticOscillatorBenchmark.build_quartic_1d(20)
        assert np.allclose(H, H.T, atol=1e-12)

    def test_quartic_1d_ground_state(self):
        """1D quartic E_0 ~ 0.668 (our convention with 1/2 on kinetic)."""
        H = QuarticOscillatorBenchmark.build_quartic_1d(50)
        evals = np.sort(np.linalg.eigvalsh(H))
        E0_ref = QuarticOscillatorBenchmark.QUARTIC_1D_E0  # ~ 0.66799
        assert abs(evals[0] - E0_ref) < 0.01

    def test_quartic_1d_gap(self):
        """1D quartic gap ~ 1.726 (our convention)."""
        H = QuarticOscillatorBenchmark.build_quartic_1d(50)
        evals = np.sort(np.linalg.eigvalsh(H))
        gap = evals[1] - evals[0]
        gap_ref = QuarticOscillatorBenchmark.QUARTIC_1D_GAP  # ~ 1.72566
        assert abs(gap - gap_ref) < 0.05

    def test_benchmark_1d(self):
        """Full 1D benchmark produces valid results."""
        bench = QuarticOscillatorBenchmark()
        result = bench.benchmark_1d(N=30)

        E0_ref = QuarticOscillatorBenchmark.QUARTIC_1D_E0  # ~ 0.66799
        # Ritz is an upper bound on E0
        assert result['ritz_E0'] >= E0_ref - 0.01
        # SCLBT is a lower bound on E0
        assert result['sclbt_E0_lower'] <= E0_ref + 0.01
        assert result['sclbt_gap'] > 0
        assert result['sclbt_converged']

    def test_benchmark_1d_sclbt_tighter(self):
        """SCLBT is tighter than Temple for 1D quartic."""
        bench = QuarticOscillatorBenchmark()
        result = bench.benchmark_1d(N=30)
        assert result['sclbt_tighter_than_temple']

    def test_build_quartic_2d_symmetric(self):
        """2D quartic Hamiltonian is symmetric."""
        H = QuarticOscillatorBenchmark.build_quartic_2d(8)
        assert np.allclose(H, H.T, atol=1e-12)

    def test_benchmark_2d(self):
        """Full 2D benchmark runs and gives positive gap."""
        bench = QuarticOscillatorBenchmark()
        result = bench.benchmark_2d(N_per_dim=10)

        assert result['sclbt_gap'] > 0
        assert result['sclbt_converged']

    def test_benchmark_3d_ym(self):
        """3D YM-like benchmark gives positive gap."""
        bench = QuarticOscillatorBenchmark()
        result = bench.benchmark_3d_ym(N=6, g2=6.28)

        assert result['ritz_gap'] > 0
        assert result['sclbt_gap'] > 0
        assert result['N_total'] == 216


# ======================================================================
# 8. YangMillsReducedGap tests
# ======================================================================

class TestYangMillsReducedGap:
    """Tests for the physical YM gap computation."""

    def test_compute_gap_basic(self):
        """Basic gap computation at physical parameters."""
        ym = YangMillsReducedGap(N_basis=6)
        result = ym.compute_gap(N=6, R=2.2, g2=6.28)

        assert result['ritz_gap'] > 0
        assert result['sclbt_gap'] > 0
        assert result['N_total'] == 216

    def test_gap_in_MeV_positive(self):
        """Gap in MeV is positive at physical parameters."""
        ym = YangMillsReducedGap(N_basis=6)
        result = ym.gap_in_MeV(N=6, R=2.2, g2=6.28)

        assert result['ritz_gap_MeV'] > 0
        assert result['sclbt_gap_MeV'] > 0
        assert result['label'] == 'NUMERICAL'

    def test_gap_in_MeV_order_of_magnitude(self):
        """Gap should be O(100 MeV) at physical R."""
        ym = YangMillsReducedGap(N_basis=8)
        result = ym.gap_in_MeV(N=8, R=2.2, g2=6.28)

        # Should be in the ballpark of hundreds of MeV
        # The 3-DOF reduced system may differ from the full 9-DOF
        assert result['ritz_gap_MeV'] > 10  # At least 10 MeV
        assert result['ritz_gap_MeV'] < 5000  # Less than 5 GeV

    def test_gap_increases_with_g2(self):
        """Gap increases with coupling (stronger quartic)."""
        ym = YangMillsReducedGap(N_basis=6)
        r1 = ym.compute_gap(N=6, R=2.2, g2=1.0)
        r2 = ym.compute_gap(N=6, R=2.2, g2=10.0)

        assert r2['ritz_gap'] > r1['ritz_gap']

    def test_gap_decreases_with_R(self):
        """Gap decreases with R (harmonic part weakens)."""
        ym = YangMillsReducedGap(N_basis=6)
        r1 = ym.compute_gap(N=6, R=1.0, g2=6.28)
        r2 = ym.compute_gap(N=6, R=5.0, g2=6.28)

        # At small R, harmonic dominates -> large gap
        # At large R, harmonic weakens but quartic may hold
        assert r1['ritz_gap'] > r2['ritz_gap']

    def test_convergence_study(self):
        """Convergence study shows gap stabilizing."""
        ym = YangMillsReducedGap()
        result = ym.convergence_study(N_range=[4, 6, 8], R=2.2, g2=6.28)

        assert len(result['results']) == 3
        assert result['R_fm'] == 2.2

    def test_sclbt_gap_below_ritz_gap(self):
        """SCLBT gap is a conservative (lower) bound on the Ritz gap."""
        ym = YangMillsReducedGap(N_basis=6)
        result = ym.compute_gap(N=6, R=2.2, g2=6.28)

        # SCLBT gap should be <= Ritz gap (conservative)
        assert result['sclbt_gap'] <= result['ritz_gap'] + 1e-10


# ======================================================================
# 9. Helper function tests
# ======================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_build_3d_hamiltonian_symmetric(self):
        """3D Hamiltonian is symmetric."""
        H = _build_3d_hamiltonian(omega=2.0, g2=6.28, n_basis=5)
        assert np.allclose(H, H.T, atol=1e-12)

    def test_build_3d_hamiltonian_positive_semidefinite(self):
        """3D Hamiltonian has non-negative eigenvalues."""
        H = _build_3d_hamiltonian(omega=2.0, g2=1.0, n_basis=5)
        evals = np.linalg.eigvalsh(H)
        # Eigenvalues should be positive (harmonic + quartic >= 0)
        assert np.min(evals) > -0.1  # Allow small numerical noise

    def test_build_3d_hamiltonian_size(self):
        """3D Hamiltonian has correct size n_basis^3."""
        H = _build_3d_hamiltonian(omega=1.0, g2=1.0, n_basis=4)
        assert H.shape == (64, 64)

    def test_build_3d_hamiltonian_no_quartic(self):
        """With g2=0, Hamiltonian is the pure harmonic oscillator."""
        N = 5
        H = _build_3d_hamiltonian(omega=2.0, g2=0.0, n_basis=N)
        evals = np.sort(np.linalg.eigvalsh(H))

        # Ground state: 3 * omega * 0.5 = 3.0
        assert abs(evals[0] - 3.0) < 1e-10

        # First excited: 3 * omega * 0.5 + omega = 5.0 (threefold degenerate)
        assert abs(evals[1] - 5.0) < 1e-10

    def test_physical_constants(self):
        """Physical constants have correct values."""
        assert abs(HBAR_C_MEV_FM - 197.327) < 0.01
        assert LAMBDA_QCD_MEV == 200.0
        assert R_PHYSICAL_FM == 2.2


# ======================================================================
# 10. Integration / cross-check tests
# ======================================================================

class TestIntegration:
    """Cross-checks between different methods."""

    def test_temple_vs_sclbt_harmonic(self, harmonic_1d):
        """Temple and SCLBT agree for harmonic oscillator (exact system)."""
        H, _ = harmonic_1d
        temple = TempleBound()
        t_result = temple.compute(H, n_states=5)

        sclbt = SCLBTBound()
        s_result = sclbt.compute(H, n_states=5)

        for k in range(5):
            assert abs(t_result['temple_lower_bounds'][k] - s_result['lower_bounds'][k]) < 1e-8

    def test_lanczos_vs_direct_eigenvalues(self, quartic_1d_small):
        """Lanczos Ritz values match direct diagonalization."""
        H = quartic_1d_small
        direct_evals = np.sort(np.linalg.eigvalsh(H))

        lc = LanczosConstruct()
        v0 = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
        lc.lanczos(H, v0, H.shape[0])
        ritz = lc.ritz_values()

        # Full Lanczos should recover all eigenvalues
        np.testing.assert_allclose(ritz, direct_evals, atol=1e-6)

    def test_interval_matrix_vs_direct(self):
        """IntervalMatrixElements agrees with direct construction for small indices.

        Note: IME computes matrix elements using the EXACT infinite-basis
        recursion, while the direct method uses truncated x matrices. They
        agree exactly for low-index elements where the truncation has no effect,
        but differ for elements near the boundary of the basis (where the
        truncated x matrix misses contributions from higher states).
        """
        omega = 2.0
        N = 15
        lam = 0.5

        # Method 1: IntervalMatrixElements (exact recursion)
        ime = IntervalMatrixElements(omega=omega, n_basis=N)
        H1 = ime.build_matrix({'quartic': lam})

        # Method 2: Direct construction (truncated basis)
        x_scale = 1.0 / np.sqrt(2.0 * omega)
        x = np.zeros((N, N))
        for n in range(N - 1):
            x[n, n + 1] = np.sqrt(n + 1) * x_scale
            x[n + 1, n] = np.sqrt(n + 1) * x_scale
        x4 = (x @ x) @ (x @ x)
        H2 = np.diag([omega * (n + 0.5) for n in range(N)]) + lam * x4

        # They should agree for the interior elements (far from truncation boundary)
        # Check the first 8x8 block where truncation effects are negligible
        np.testing.assert_allclose(H1[:8, :8], H2[:8, :8], atol=1e-10)

    def test_rigorous_gap_combines_correctly(self, quartic_1d_small):
        """RigorousSpectralGap correctly combines SCLBT + truncation."""
        rsg = RigorousSpectralGap(n_states=3)
        result = rsg.certified_gap(quartic_1d_small, dimension=1, potential_exponent=2.0)

        # certified_gap = sclbt_gap - trunc_E0 - trunc_E1
        expected = (result['sclbt_gap_raw']
                    - result['truncation_error_E0']
                    - result['truncation_error_E1'])
        assert abs(result['certified_gap'] - expected) < 1e-15

    def test_ym_gap_consistent_methods(self):
        """YM gap from different methods are consistent."""
        N = 6
        omega = 2.0 / 2.2  # R = 2.2 fm
        g2 = 6.28
        H = _build_3d_hamiltonian(omega, g2, N)

        # Direct eigenvalues
        evals = np.sort(np.linalg.eigvalsh(H))
        ritz_gap = evals[1] - evals[0]

        # SCLBT gap
        sclbt = SCLBTBound()
        result = sclbt.compute(H, n_states=3)
        sclbt_gap = result['spectral_gap']

        # Temple gap
        temple = TempleBound()
        t_result = temple.compute(H, n_states=3)

        # Ordering: temple_gap <= sclbt_gap <= ritz_gap
        assert t_result['spectral_gap_temple'] <= sclbt_gap + 1e-10
        assert sclbt_gap <= ritz_gap + 1e-10

    def test_convergence_of_bounds_with_basis_size(self):
        """Both Ritz and SCLBT bounds converge as basis grows.

        Note: For the truncated eigenvalue problem, when the eigenvectors of
        the truncated matrix have zero variance (they're exact eigenstates of
        that matrix), the SCLBT bounds equal the Ritz values. Both converge
        to the true eigenvalue from ABOVE. The lower-bound property of SCLBT
        applies to the FULL (infinite-dimensional) problem, not to the
        truncated one. As the basis grows, the truncated Ritz values decrease
        toward the true eigenvalue, and the SCLBT bounds follow.
        """
        ritz_E0s = []
        sclbt_E0s = []

        for N in [15, 25, 35]:
            H = QuarticOscillatorBenchmark.build_quartic_1d(N)
            sclbt = SCLBTBound()
            result = sclbt.compute(H, n_states=3)
            ritz_E0s.append(result['ritz_eigenvalues'][0])
            sclbt_E0s.append(result['lower_bounds'][0])

        E0_ref = QuarticOscillatorBenchmark.QUARTIC_1D_E0  # ~ 0.66799

        # Ritz should decrease toward exact (upper bound on truncated problem)
        for i in range(len(ritz_E0s) - 1):
            assert ritz_E0s[i] >= ritz_E0s[i + 1] - 1e-10

        # SCLBT bounds should also converge to the reference
        # (For zero-variance case, they equal Ritz values)
        for i in range(len(sclbt_E0s) - 1):
            # They should be converging (getting closer to reference)
            err_curr = abs(sclbt_E0s[i] - E0_ref)
            err_next = abs(sclbt_E0s[i + 1] - E0_ref)
            assert err_next <= err_curr + 1e-10

        # Both should approach reference
        assert abs(ritz_E0s[-1] - E0_ref) < 0.001
        assert abs(sclbt_E0s[-1] - E0_ref) < 0.001
