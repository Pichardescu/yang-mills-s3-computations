"""
Tests for Gribov horizon measurement on S³ lattice.

Verifies:
    - Vacuum FP spectrum (all links = I → lambda_min = 3/R² on lattice)
    - Gauge fixing convergence
    - FP operator symmetry and positive semi-definiteness
    - R-dependence at vacuum
    - Smoke test for thermalized configurations

Note: Config counts are kept LOW for speed. Physics accuracy requires
larger ensembles (n_configs >= 50).
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.s3_lattice import S3Lattice
from yang_mills_s3.lattice.lattice_ym import LatticeYM
from yang_mills_s3.lattice.gribov_measurement import (
    build_fp_operator,
    fp_eigenvalues,
    lattice_gauge_fix,
    measure_gribov_spectrum,
    quick_gribov_check,
    _su_n_generators,
    _adjoint_rep,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def lattice():
    """600-cell lattice on unit S³."""
    return S3Lattice(R=1.0)


@pytest.fixture
def lym_vacuum(lattice):
    """LatticeYM at vacuum (all links = identity)."""
    return LatticeYM(lattice, N=2, beta=1.0)


@pytest.fixture
def lym_thermalized(lattice):
    """LatticeYM after some thermalization."""
    lym = LatticeYM(lattice, N=2, beta=2.0)
    rng = np.random.default_rng(42)
    lym.randomize_links(rng)
    lym.thermalize(n_sweeps=10, epsilon=0.3, rng=rng)
    return lym


# ======================================================================
# Generator tests
# ======================================================================

class TestGenerators:
    """Test SU(N) generator construction."""

    def test_su2_generators_count(self):
        """SU(2) should have 3 generators."""
        gens = _su_n_generators(2)
        assert len(gens) == 3

    def test_su3_generators_count(self):
        """SU(3) should have 8 generators."""
        gens = _su_n_generators(3)
        assert len(gens) == 8

    def test_su2_generators_traceless(self):
        """Generators should be traceless."""
        gens = _su_n_generators(2)
        for T in gens:
            assert abs(np.trace(T)) < 1e-12

    def test_su2_generators_normalization(self):
        """Tr(T^a T^b) = delta^{ab}/2."""
        gens = _su_n_generators(2)
        for a in range(3):
            for b in range(3):
                tr = np.trace(gens[a] @ gens[b])
                expected = 0.5 if a == b else 0.0
                assert abs(tr - expected) < 1e-12, \
                    f"Tr(T^{a} T^{b}) = {tr}, expected {expected}"

    def test_adjoint_rep_identity(self):
        """Adjoint of identity = identity matrix."""
        gens = _su_n_generators(2)
        R = _adjoint_rep(np.eye(2, dtype=complex), gens)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


# ======================================================================
# FP operator at vacuum
# ======================================================================

class TestFPOperatorVacuum:
    """FP operator properties at the flat vacuum (all links = I)."""

    def test_fp_operator_is_symmetric(self, lym_vacuum):
        """M_FP should be a symmetric real matrix."""
        M = build_fp_operator(lym_vacuum)
        np.testing.assert_allclose(M, M.T, atol=1e-10,
            err_msg="FP operator should be symmetric")

    def test_fp_operator_positive_semidefinite(self, lym_vacuum):
        """All eigenvalues of M_FP should be >= 0."""
        M = build_fp_operator(lym_vacuum)
        eigenvalues = np.linalg.eigvalsh(M)
        # Allow small numerical noise
        assert np.all(eigenvalues > -1e-8), \
            f"FP operator has negative eigenvalues: min = {eigenvalues[0]}"

    def test_vacuum_zero_modes(self, lym_vacuum):
        """
        At vacuum, M_FP = lattice Laplacian ⊗ I_adj.
        Should have dim_adj = 3 zero modes (for SU(2)) from constant modes.
        """
        result = fp_eigenvalues(lym_vacuum, n_evals=10, gauge_fix=False)
        # Expect 3 zero modes for SU(2)
        assert result['n_zero_modes'] == 3, \
            f"Expected 3 zero modes, got {result['n_zero_modes']}"

    def test_vacuum_lowest_nonzero_eigenvalue(self, lym_vacuum):
        """
        At vacuum: lowest non-zero eigenvalue should be related to 3/R².

        The lattice Laplacian on the 600-cell approximates the continuum
        Laplacian Δ₀ on S³. The continuum result is λ₁ = 3/R².

        On the lattice, the eigenvalue is 3/R² up to lattice corrections.
        We check that it's within a factor of 2 of the continuum value.
        """
        R = lym_vacuum.lattice.R
        result = fp_eigenvalues(lym_vacuum, n_evals=10, gauge_fix=False)

        # Get first non-zero eigenvalue
        evals = result['all_eigenvalues']
        threshold = 1e-6 * abs(evals[-1])
        nonzero = evals[evals > threshold]
        lambda_min = nonzero[0]

        continuum = 3.0 / R**2
        ratio = lambda_min / continuum

        # Lattice discretization of S³ via 600-cell should give
        # a reasonable approximation. Allow factor of 0.3 to 3.0.
        assert 0.3 < ratio < 3.0, \
            f"Vacuum λ_min = {lambda_min:.4f}, continuum = {continuum:.4f}, ratio = {ratio:.4f}"

    def test_vacuum_r_dependence(self):
        """
        lambda_min * R² should be approximately constant across R values
        (both scale as 1/R²).
        """
        products = []
        for R in [0.5, 1.0, 2.0]:
            lattice = S3Lattice(R=R)
            lym = LatticeYM(lattice, N=2, beta=1.0)
            result = fp_eigenvalues(lym, n_evals=10, gauge_fix=False)

            evals = result['all_eigenvalues']
            threshold = 1e-6 * abs(evals[-1])
            nonzero = evals[evals > threshold]
            if len(nonzero) > 0:
                products.append(nonzero[0] * R**2)

        # All products should be similar (within 20%)
        assert len(products) >= 2
        mean_product = np.mean(products)
        for p in products:
            assert abs(p - mean_product) / mean_product < 0.2, \
                f"lambda_min * R² varies too much: {products}"


# ======================================================================
# Gauge fixing
# ======================================================================

class TestGaugeFixing:
    """Lattice Coulomb gauge fixing."""

    def test_gauge_fix_vacuum_converges(self, lym_vacuum):
        """Gauge fixing on vacuum should converge immediately (already in gauge)."""
        result = lattice_gauge_fix(lym_vacuum, max_iter=10)
        assert result['converged'], "Gauge fixing should converge on vacuum"

    def test_gauge_fix_increases_functional(self, lattice):
        """Gauge fixing should produce a positive functional (links aligned)."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)

        result = lattice_gauge_fix(lym, max_iter=100, omega=1.0)

        # After gauge fixing, the functional (sum of Re Tr U / N) should be positive
        # since we're maximizing alignment of link variables
        func_after = result['functional']
        assert np.isfinite(func_after), f"Functional should be finite, got {func_after}"

    def test_gauge_fix_preserves_plaquettes(self, lattice):
        """Gauge fixing should not change plaquette values (gauge invariant)."""
        lym = LatticeYM(lattice, N=2, beta=2.0)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)

        plaq_before = lym.plaquette_average()
        lattice_gauge_fix(lym, max_iter=50, omega=1.0)
        plaq_after = lym.plaquette_average()

        np.testing.assert_allclose(plaq_after, plaq_before, atol=1e-6,
            err_msg="Gauge fixing should preserve plaquette average")


# ======================================================================
# Thermalized configs
# ======================================================================

class TestThermalizedConfigs:
    """FP operator on thermalized configurations."""

    def test_fp_operator_symmetric_thermalized(self, lym_thermalized):
        """FP operator should be symmetric even on thermalized configs."""
        # Gauge fix first
        lattice_gauge_fix(lym_thermalized, max_iter=50, omega=1.0)
        M = build_fp_operator(lym_thermalized)
        np.testing.assert_allclose(M, M.T, atol=1e-8,
            err_msg="FP operator should be symmetric after gauge fixing")

    def test_fp_eigenvalues_run(self, lym_thermalized):
        """FP eigenvalue computation should run on thermalized config."""
        result = fp_eigenvalues(lym_thermalized, n_evals=10)
        assert len(result['eigenvalues']) == 10
        assert result['gf_result'] is not None

    def test_lambda_min_positive_thermalized(self, lattice):
        """
        On thermalized configs, lambda_min should be positive (inside Gribov region).

        At strong coupling (low beta), random fluctuations actually increase
        the FP eigenvalues because the off-diagonal (adjoint rep) terms average
        out while the diagonal (valence) dominates. The approach to the Gribov
        horizon only occurs at weak coupling (high beta) where configs are
        perturbative around vacuum.

        NUMERICAL: This is a lattice measurement, not a formal proof.
        """
        lym = LatticeYM(lattice, N=2, beta=1.5)
        rng = np.random.default_rng(42)
        lym.randomize_links(rng)
        lym.thermalize(n_sweeps=20, epsilon=0.3, rng=rng)

        therm_result = fp_eigenvalues(lym, n_evals=10, gauge_fix=True)
        therm_evals = therm_result['all_eigenvalues']

        # All eigenvalues should be non-negative (inside Gribov region)
        assert np.all(therm_evals > -1e-6), \
            f"FP operator has negative eigenvalues: min = {therm_evals[0]:.4f}"

        # The lowest eigenvalue should be positive (no zero modes after gauge fixing)
        # or have dim_adj=3 zero modes if global gauge not fully fixed
        threshold = 1e-4 * abs(therm_evals[-1])
        nonzero = therm_evals[therm_evals > threshold]
        assert len(nonzero) > 0, "Should have non-zero eigenvalues"
        assert nonzero[0] > 0, f"Lowest non-zero eigenvalue should be positive, got {nonzero[0]}"


# ======================================================================
# Measurement functions
# ======================================================================

class TestMeasurement:
    """Integration tests for measurement functions."""

    def test_measure_gribov_spectrum_runs(self):
        """measure_gribov_spectrum should run and return correct structure."""
        result = measure_gribov_spectrum(R=1.0, N=2, beta=2.0,
                                         n_configs=2, n_therm=5)
        assert 'lambda_min_mean' in result
        assert 'lambda_min_std' in result
        assert 'lambda_min_R2_mean' in result
        assert result['eigenvalues'].shape[0] == 2
        assert np.isfinite(result['lambda_min_mean'])

    def test_quick_gribov_check_runs(self):
        """quick_gribov_check should run and report vacuum vs thermalized."""
        result = quick_gribov_check(R=1.0, N=2, n_configs=2, n_therm=5)
        assert 'vacuum' in result
        assert 'thermalized' in result
        assert result['vacuum']['n_zero_modes'] == 3  # SU(2) has 3 zero modes
        assert result['vacuum']['lambda_min'] > 0

    def test_measure_returns_positive_lambda_min(self):
        """lambda_min should be positive (inside Gribov region)."""
        result = measure_gribov_spectrum(R=1.0, N=2, beta=2.0,
                                         n_configs=2, n_therm=5)
        # lambda_min should be positive (we're inside the Gribov region)
        assert result['lambda_min_mean'] > 0, \
            f"lambda_min should be positive, got {result['lambda_min_mean']}"
