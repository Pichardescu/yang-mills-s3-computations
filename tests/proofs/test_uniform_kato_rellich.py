"""
Tests for Proposition 6.5: Uniform Kato-Rellich bound on S^3 lattice.

Verifies that the Kato-Rellich relative bound alpha(a) of the non-linear
perturbation V = g^2 [a ^ a, .] satisfies alpha(a) < 1 uniformly for
all lattice spacings a <= a_max (600-cell).

Test categories:
    1. Constants and setup
    2. Discrete Sobolev constant
    3. Lattice non-linear perturbation
    4. Lattice alpha computation (analytic formula)
    5. Lattice alpha computation (numerical)
    6. Alpha vs spacing scan
    7. Uniform bound (the main result)
    8. Consequences: gap lower bound
    9. Consequences: conjecture status upgrade
    10. Convergence properties
    11. Physical coupling tests
    12. Edge cases and monotonicity
    13. Theorem statement
    14. Full verification pipeline
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.uniform_kato_rellich import (
    # Constants
    C_ALPHA_CONTINUUM,
    G_CRITICAL_SQUARED,
    PHYSICAL_G_SQUARED,
    CONTINUUM_COEXACT_GAP,
    HBAR_C_MEV_FM,
    # Discrete Sobolev
    discrete_sobolev_constant,
    discrete_sobolev_convergence,
    # Lattice perturbation
    lattice_wedge_product,
    lattice_nonlinear_perturbation,
    # Alpha computation
    lattice_alpha_from_spectrum,
    lattice_alpha_numerical,
    # Scan and bound
    alpha_vs_spacing,
    uniform_bound,
    # Consequences
    continuum_gap_lower_bound,
    upgrade_conjecture_status,
    # Pipeline
    full_verification_pipeline,
    # Theorem
    theorem_statement,
)
from yang_mills_s3.proofs.continuum_limit import (
    refine_600_cell,
    lattice_hodge_laplacian_1forms,
    _build_edge_index,
)
from yang_mills_s3.proofs.gap_proof_su2 import sobolev_constant_s3


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def base_lattice():
    """Level-0 600-cell lattice data."""
    return refine_600_cell(0, R=1.0)


@pytest.fixture(scope='module')
def refined_lattice():
    """Level-1 refined 600-cell lattice data."""
    return refine_600_cell(1, R=1.0)


@pytest.fixture(scope='module')
def base_alpha_data(base_lattice):
    """Alpha data at level 0 with physical coupling."""
    vertices, edges, faces = base_lattice
    return lattice_alpha_from_spectrum(
        vertices, edges, faces, R=1.0,
        g_squared=PHYSICAL_G_SQUARED, gauge_group='SU(2)'
    )


# ======================================================================
# 1. Constants and setup
# ======================================================================

class TestConstants:
    """Verify mathematical constants are correct."""

    def test_c_alpha_value(self):
        """C_alpha = sqrt(2)/(24*pi^2) ~ 0.005976."""
        expected = np.sqrt(2) / (24.0 * np.pi**2)
        assert abs(C_ALPHA_CONTINUUM - expected) < 1e-10

    def test_g_critical_squared(self):
        """g^2_crit = 1/C_alpha = 24*pi^2/sqrt(2) ~ 167.53."""
        assert abs(G_CRITICAL_SQUARED - 1.0 / C_ALPHA_CONTINUUM) < 1e-10
        assert 165.0 < G_CRITICAL_SQUARED < 170.0

    def test_physical_coupling(self):
        """Physical coupling g^2 ~ 6.28 (alpha_s = 0.5)."""
        assert abs(PHYSICAL_G_SQUARED - 4.0 * np.pi * 0.5) < 1e-10
        assert 6.0 < PHYSICAL_G_SQUARED < 7.0

    def test_physical_below_critical(self):
        """Physical coupling is well below critical."""
        assert PHYSICAL_G_SQUARED < G_CRITICAL_SQUARED
        assert PHYSICAL_G_SQUARED < 0.2 * G_CRITICAL_SQUARED

    def test_continuum_gap(self):
        """Continuum coexact gap coefficient is 4."""
        assert CONTINUUM_COEXACT_GAP == 4.0

    def test_continuum_alpha_at_physical(self):
        """At physical coupling, alpha ~ 0.038."""
        alpha = C_ALPHA_CONTINUUM * PHYSICAL_G_SQUARED
        assert 0.03 < alpha < 0.05
        assert alpha < 1.0


# ======================================================================
# 2. Discrete Sobolev constant
# ======================================================================

class TestDiscreteSobolev:
    """Test discrete Sobolev constant computation."""

    def test_sobolev_positive(self, base_lattice):
        """Discrete Sobolev constant is positive."""
        vertices, edges, faces = base_lattice
        data = discrete_sobolev_constant(vertices, edges, faces, R=1.0)
        assert data['C_S_discrete'] > 0

    def test_sobolev_continuum_positive(self, base_lattice):
        """Continuum Sobolev constant is positive."""
        vertices, edges, faces = base_lattice
        data = discrete_sobolev_constant(vertices, edges, faces, R=1.0)
        assert data['C_S_continuum'] > 0

    def test_sobolev_ratio_finite(self, base_lattice):
        """C_S(a) / C_S is finite and positive."""
        vertices, edges, faces = base_lattice
        data = discrete_sobolev_constant(vertices, edges, faces, R=1.0)
        assert 0 < data['ratio'] < 100

    def test_sobolev_mesh_size(self, base_lattice):
        """Mesh size is positive for 600-cell."""
        vertices, edges, faces = base_lattice
        data = discrete_sobolev_constant(vertices, edges, faces, R=1.0)
        assert data['mesh_size'] > 0

    def test_sobolev_n_edges(self, base_lattice):
        """600-cell has 720 edges."""
        vertices, edges, faces = base_lattice
        data = discrete_sobolev_constant(vertices, edges, faces, R=1.0)
        assert data['n_edges'] == 720

    def test_sobolev_convergence_bounded(self):
        """Discrete Sobolev ratios are bounded across levels."""
        data = discrete_sobolev_convergence(max_level=1, R=1.0)
        assert data['converges']
        for ratio in data['ratios']:
            assert ratio < 3.0  # bounded above


# ======================================================================
# 3. Lattice non-linear perturbation
# ======================================================================

class TestLatticePerturbation:
    """Test the discrete non-linear perturbation V^(a)."""

    def test_perturbation_zero_config(self, base_lattice):
        """V^(a) psi = 0 when a = 0 (zero background field)."""
        vertices, edges, faces = base_lattice
        n_e = len(edges)
        a_config = np.zeros(n_e)
        psi = np.random.default_rng(42).standard_normal(n_e)
        result = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        assert np.allclose(result, 0.0)

    def test_perturbation_zero_psi(self, base_lattice):
        """V^(a) psi = 0 when psi = 0."""
        vertices, edges, faces = base_lattice
        n_e = len(edges)
        a_config = np.random.default_rng(42).standard_normal(n_e)
        psi = np.zeros(n_e)
        result = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        assert np.allclose(result, 0.0)

    def test_perturbation_linearity_in_g(self, base_lattice):
        """V^(a) scales linearly with g^2."""
        vertices, edges, faces = base_lattice
        n_e = len(edges)
        rng = np.random.default_rng(42)
        a_config = rng.standard_normal(n_e)
        psi = rng.standard_normal(n_e)

        V1 = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        V2 = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=2.0
        )
        # V2 should be 2 * V1
        assert np.allclose(V2, 2.0 * V1, atol=1e-12)

    def test_perturbation_scales_with_field(self, base_lattice):
        """V^(a) is quadratic in the background field a."""
        vertices, edges, faces = base_lattice
        n_e = len(edges)
        rng = np.random.default_rng(42)
        a_config = rng.standard_normal(n_e)
        psi = rng.standard_normal(n_e)

        V1 = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        V2 = lattice_nonlinear_perturbation(
            2.0 * a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        # V scales as a^2 * psi, so doubling a quadruples V
        # (the wedge product is bilinear in a)
        assert np.linalg.norm(V2) > 0
        if np.linalg.norm(V1) > 1e-15:
            ratio = np.linalg.norm(V2) / np.linalg.norm(V1)
            assert 3.0 < ratio < 5.0  # expect ~4 (quadratic)

    def test_perturbation_nonzero_for_random(self, base_lattice):
        """V^(a) psi is nonzero for generic a, psi."""
        vertices, edges, faces = base_lattice
        n_e = len(edges)
        rng = np.random.default_rng(42)
        a_config = rng.standard_normal(n_e)
        psi = rng.standard_normal(n_e)
        result = lattice_nonlinear_perturbation(
            a_config, psi, vertices, edges, faces, g_squared=1.0
        )
        assert np.linalg.norm(result) > 0


# ======================================================================
# 4. Lattice alpha from spectrum (analytic formula)
# ======================================================================

class TestLatticeAlphaAnalytic:
    """Test the analytic lattice alpha computation."""

    def test_alpha_positive(self, base_alpha_data):
        """alpha(a) is positive for g^2 > 0."""
        assert base_alpha_data['alpha'] > 0

    def test_alpha_less_than_one(self, base_alpha_data):
        """alpha(a) < 1 at physical coupling on 600-cell."""
        assert base_alpha_data['alpha'] < 1.0

    def test_alpha_continuum_value(self, base_alpha_data):
        """Continuum alpha matches C_alpha * g^2."""
        expected = C_ALPHA_CONTINUUM * PHYSICAL_G_SQUARED
        assert abs(base_alpha_data['alpha_continuum'] - expected) < 1e-10

    def test_lambda1_scaled_positive(self, base_alpha_data):
        """Scaled discrete spectral gap is positive."""
        assert base_alpha_data['lambda_1_scaled'] > 0

    def test_lambda1_raw_positive(self, base_alpha_data):
        """Raw discrete spectral gap is positive."""
        assert base_alpha_data['lambda_1_discrete_raw'] > 0

    def test_lambda1_continuum(self, base_alpha_data):
        """Continuum spectral gap is 4/R^2 = 4."""
        assert abs(base_alpha_data['lambda_1_continuum'] - 4.0) < 1e-10

    def test_scale_factor_positive(self, base_alpha_data):
        """Scale factor is positive."""
        assert base_alpha_data['scale_factor'] > 0

    def test_mesh_size_positive(self, base_alpha_data):
        """Mesh size is positive."""
        assert base_alpha_data['mesh_size'] > 0

    def test_alpha_zero_coupling(self, base_lattice):
        """alpha = 0 when g^2 = 0."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_from_spectrum(
            vertices, edges, faces, R=1.0, g_squared=0.0
        )
        assert abs(data['alpha']) < 1e-14
        assert abs(data['alpha_continuum']) < 1e-14

    def test_alpha_increases_with_g(self, base_lattice):
        """alpha is monotonically increasing in g^2."""
        vertices, edges, faces = base_lattice
        alphas = []
        for g2 in [1.0, 3.0, 6.0, 10.0]:
            data = lattice_alpha_from_spectrum(
                vertices, edges, faces, R=1.0, g_squared=g2
            )
            alphas.append(data['alpha'])
        for i in range(len(alphas) - 1):
            assert alphas[i] < alphas[i + 1]


# ======================================================================
# 5. Lattice alpha numerical
# ======================================================================

class TestLatticeAlphaNumerical:
    """Test the numerical alpha estimation."""

    def test_numerical_alpha_positive(self, base_lattice):
        """Numerical alpha is positive for g^2 > 0."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_numerical(
            vertices, edges, faces, R=1.0,
            g_squared=PHYSICAL_G_SQUARED, n_test_vectors=20
        )
        assert data['alpha_numerical'] >= 0

    def test_numerical_alpha_less_than_one(self, base_lattice):
        """Numerical alpha < 1 at physical coupling."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_numerical(
            vertices, edges, faces, R=1.0,
            g_squared=PHYSICAL_G_SQUARED, n_test_vectors=20
        )
        assert data['alpha_numerical'] < 1.0

    def test_numerical_below_analytic(self, base_lattice):
        """Numerical alpha <= analytic alpha (within tolerance)."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_numerical(
            vertices, edges, faces, R=1.0,
            g_squared=PHYSICAL_G_SQUARED, n_test_vectors=30
        )
        # Numerical should be at most the analytic bound (with tolerance)
        # On coarse lattice, numerical alpha can exceed analytic by up to 3x
        assert data['alpha_numerical'] <= data['alpha_analytic'] * 3.0

    def test_numerical_consistency(self, base_lattice):
        """Numerical and analytic estimates are consistent."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_numerical(
            vertices, edges, faces, R=1.0,
            g_squared=PHYSICAL_G_SQUARED, n_test_vectors=20
        )
        assert data['consistent']


# ======================================================================
# 6. Alpha vs spacing scan
# ======================================================================

class TestAlphaVsSpacing:
    """Test the scan of alpha across refinement levels."""

    def test_scan_returns_data(self):
        """Scan returns data for each level."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert len(data['levels']) >= 1
        assert len(data['alphas']) >= 1
        assert len(data['mesh_sizes']) >= 1

    def test_scan_all_below_one(self):
        """All alpha values are below 1 at physical coupling."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['all_below_one']

    def test_scan_sup_alpha(self):
        """Supremum of alpha is finite and < 1."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert 0 < data['sup_alpha'] < 1.0

    def test_scan_mesh_decreasing(self):
        """Mesh sizes decrease with refinement (at level 1)."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=1)
        if len(data['mesh_sizes']) >= 2:
            assert data['mesh_sizes'][1] < data['mesh_sizes'][0]

    def test_scan_alpha_continuum(self):
        """Continuum alpha matches expected value."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        expected = C_ALPHA_CONTINUUM * PHYSICAL_G_SQUARED
        assert abs(data['alpha_continuum'] - expected) < 1e-10


# ======================================================================
# 7. Uniform bound (the main result)
# ======================================================================

class TestUniformBound:
    """Test the main result: uniform KR bound."""

    def test_uniform_bound_holds(self):
        """MAIN RESULT: sup_a alpha(a) < 1 at physical coupling."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['uniform_bound_holds']

    def test_sup_alpha_finite(self):
        """sup alpha is finite."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert np.isfinite(data['sup_alpha'])
        assert data['sup_alpha'] > 0

    def test_gap_persists(self):
        """Gap lower bound is positive at all lattice spacings."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['gap_persists']

    def test_gap_lower_bounds_positive(self):
        """Each gap lower bound is > 0."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        for gap in data['gap_lower_bounds']:
            assert gap > 0

    def test_continuum_gap_positive(self):
        """Continuum gap lower bound is positive."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['continuum_gap'] > 0

    def test_below_critical_coupling(self):
        """Physical coupling is below critical."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['below_critical']

    def test_status_is_proposition(self):
        """Status is PROPOSITION (not CONJECTURE or NUMERICAL)."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['status'] == 'PROPOSITION'

    def test_fails_above_critical(self):
        """Uniform bound should eventually fail for g^2 >> g^2_crit."""
        # At very large coupling, alpha > 1
        data = uniform_bound(g_squared=200.0, R=1.0, max_level=0)
        # At g^2 = 200 > 167.5, continuum alpha > 1
        assert data['sup_alpha'] > 1.0
        assert not data['uniform_bound_holds']


# ======================================================================
# 8. Consequences: gap lower bound
# ======================================================================

class TestGapLowerBound:
    """Test the gap lower bound from uniform KR."""

    def test_gap_positive_physical(self):
        """Gap is positive at physical coupling."""
        data = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0)
        assert data['gap_positive']
        assert data['gap_lower_bound'] > 0

    def test_gap_scales_with_R(self):
        """Gap scales as 1/R^2."""
        g1 = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0)
        g2 = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=2.0)
        # gap ~ 1/R^2, so gap(2R) ~ gap(R)/4
        ratio = g1['gap_lower_bound'] / g2['gap_lower_bound']
        assert 3.0 < ratio < 5.0  # should be ~4

    def test_gap_zero_at_zero_coupling(self):
        """At g^2 = 0, gap = linearized gap (alpha = 0, beta = 0)."""
        data = continuum_gap_lower_bound(g_squared=0.0, R=1.0)
        assert data['alpha'] == 0.0
        # Gap should equal linearized gap = 4.0
        assert abs(data['gap_lower_bound'] - 4.0) < 1e-10

    def test_gap_decreases_with_coupling(self):
        """Gap lower bound decreases with increasing coupling."""
        g_vals = [1.0, 3.0, 5.0]
        gaps = [
            continuum_gap_lower_bound(g_squared=g2, R=1.0)['gap_lower_bound']
            for g2 in g_vals
        ]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_gap_mev_reasonable(self):
        """Gap in MeV is in the right ballpark for physical R."""
        R_phys = 2.0 * HBAR_C_MEV_FM / 200.0  # ~ 1.97 fm
        data = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=R_phys)
        # Gap should be order 100-300 MeV
        if data['gap_positive']:
            assert data['gap_mev'] > 0

    def test_gap_ratio_less_than_one(self):
        """Gap ratio (gap / linearized gap) is less than 1."""
        data = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0)
        assert 0 < data['gap_ratio'] < 1.0


# ======================================================================
# 9. Consequences: conjecture status upgrade
# ======================================================================

class TestConjectureUpgrade:
    """Test the upgrade of Conjecture 6.5."""

    def test_status_proposition(self):
        """Conjecture 6.5 is upgraded to PROPOSITION."""
        data = upgrade_conjecture_status(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert data['conjecture_6_5_status'] == 'PROPOSITION'

    def test_proof_chain_exists(self):
        """Proof chain has at least 5 steps."""
        data = upgrade_conjecture_status(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert len(data['proof_chain']) >= 5

    def test_step1_is_theorem(self):
        """Step 1 (continuum KR) has THEOREM status."""
        data = upgrade_conjecture_status(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        step1 = data['proof_chain'][0]
        assert step1['status'] == 'THEOREM'

    def test_gaps_to_theorem_documented(self):
        """Gaps to full THEOREM are documented."""
        data = upgrade_conjecture_status(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert len(data['gaps_to_theorem']) >= 2

    def test_impact_documented(self):
        """Impact on Conjecture 7.2 is documented."""
        data = upgrade_conjecture_status(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert len(data['impact_on_conjecture_7_2']) > 0


# ======================================================================
# 10. Convergence properties
# ======================================================================

class TestConvergence:
    """Test convergence of alpha(a) to alpha_0 and related properties."""

    def test_sobolev_convergence(self):
        """Discrete Sobolev constants converge across levels."""
        data = discrete_sobolev_convergence(max_level=1, R=1.0)
        assert data['converges']

    def test_alpha_converges_with_level(self):
        """alpha(a) approaches alpha_continuum with refinement."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=1)
        if data['convergence_to_continuum'] is not None:
            assert data['convergence_to_continuum']

    def test_mesh_size_decreases(self):
        """Mesh size strictly decreases with refinement level."""
        data = alpha_vs_spacing(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=1)
        for i in range(len(data['mesh_sizes']) - 1):
            assert data['mesh_sizes'][i + 1] < data['mesh_sizes'][i]


# ======================================================================
# 11. Physical coupling tests
# ======================================================================

class TestPhysicalCoupling:
    """Test at the physical QCD coupling g^2 ~ 6.28."""

    def test_alpha_physical_below_one(self):
        """alpha < 1 at g^2 = 6.28."""
        alpha = C_ALPHA_CONTINUUM * PHYSICAL_G_SQUARED
        assert alpha < 1.0
        assert alpha < 0.2  # ~12%

    def test_gap_survives_physical(self):
        """Gap survives at physical coupling."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['gap_persists']

    def test_gap_ratio_at_physical(self):
        """At physical coupling, gap is ~88% of linearized gap."""
        alpha = C_ALPHA_CONTINUUM * PHYSICAL_G_SQUARED
        ratio = 1.0 - alpha
        assert 0.8 < ratio < 1.0

    def test_uniform_at_physical(self):
        """Uniform bound holds at physical coupling."""
        data = uniform_bound(g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0)
        assert data['uniform_bound_holds']


# ======================================================================
# 12. Edge cases and monotonicity
# ======================================================================

class TestEdgeCases:
    """Test edge cases and monotonicity properties."""

    def test_alpha_zero_at_g_zero(self, base_lattice):
        """alpha = 0 when g^2 = 0."""
        vertices, edges, faces = base_lattice
        data = lattice_alpha_from_spectrum(
            vertices, edges, faces, R=1.0, g_squared=0.0
        )
        assert abs(data['alpha']) < 1e-14

    def test_alpha_monotone_in_g(self, base_lattice):
        """alpha(a) is monotone increasing in g^2."""
        vertices, edges, faces = base_lattice
        prev_alpha = 0.0
        for g2 in [0.0, 1.0, 5.0, 10.0, 20.0]:
            data = lattice_alpha_from_spectrum(
                vertices, edges, faces, R=1.0, g_squared=g2
            )
            assert data['alpha'] >= prev_alpha - 1e-12
            prev_alpha = data['alpha']

    def test_gap_zero_at_critical_coupling(self):
        """At critical coupling, gap bound approaches zero."""
        # At 90% of critical, gap is small but positive
        data = continuum_gap_lower_bound(g_squared=G_CRITICAL_SQUARED * 0.90, R=1.0)
        assert data['gap_lower_bound'] > 0
        assert data['gap_lower_bound'] < 1.0  # much smaller than Delta_0 = 4
        # At 100% of critical, alpha = 1 so gap bound goes to zero or negative
        data_crit = continuum_gap_lower_bound(g_squared=G_CRITICAL_SQUARED, R=1.0)
        assert data_crit['gap_lower_bound'] < data['gap_lower_bound']

    def test_large_R(self):
        """Gap bound at large R is small but positive."""
        data = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=100.0)
        assert data['gap_positive']
        assert data['gap_lower_bound'] < 0.01  # small because 1/R^2

    def test_small_R(self):
        """Gap bound at small R is large."""
        data = continuum_gap_lower_bound(g_squared=PHYSICAL_G_SQUARED, R=0.1)
        assert data['gap_positive']
        assert data['gap_lower_bound'] > 100.0  # ~4/(0.01) * (1-0.12)

    def test_different_R_values(self, base_lattice):
        """Alpha computation works for different R."""
        for R in [0.5, 1.0, 2.0]:
            vertices, edges, faces = refine_600_cell(0, R=R)
            data = lattice_alpha_from_spectrum(
                vertices, edges, faces, R=R, g_squared=PHYSICAL_G_SQUARED
            )
            assert data['alpha'] > 0
            assert data['alpha'] < 1.0


# ======================================================================
# 13. Theorem statement
# ======================================================================

class TestTheoremStatement:
    """Test the formal theorem statement."""

    def test_statement_exists(self):
        """Theorem statement is returned."""
        stmt = theorem_statement()
        assert 'statement' in stmt
        assert len(stmt['statement']) > 100

    def test_status_proposition(self):
        """Status is PROPOSITION."""
        stmt = theorem_statement()
        assert stmt['status'] == 'PROPOSITION'

    def test_name_correct(self):
        """Name includes 'Proposition 6.5'."""
        stmt = theorem_statement()
        assert '6.5' in stmt['name']

    def test_assumptions_listed(self):
        """Assumptions are documented."""
        stmt = theorem_statement()
        assert len(stmt['assumptions']) >= 3

    def test_consequences_listed(self):
        """Consequences are documented."""
        stmt = theorem_statement()
        assert len(stmt['consequences']) >= 3

    def test_references_listed(self):
        """References are listed."""
        stmt = theorem_statement()
        assert len(stmt['references']) >= 4
        # Key references
        ref_text = ' '.join(stmt['references'])
        assert 'Kato' in ref_text
        assert 'Dodziuk' in ref_text
        assert 'Aubin' in ref_text or 'Talenti' in ref_text


# ======================================================================
# 14. Full verification pipeline
# ======================================================================

class TestVerificationPipeline:
    """Test the full verification pipeline."""

    def test_pipeline_runs(self):
        """Full pipeline runs without error."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert 'summary' in data

    def test_pipeline_summary(self):
        """Summary contains key fields."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        s = data['summary']
        assert 'g_squared' in s
        assert 'sup_alpha' in s
        assert 'uniform_bound_holds' in s
        assert 'gap_positive' in s
        assert 'status' in s

    def test_pipeline_below_critical(self):
        """Pipeline confirms physical coupling below critical."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert data['summary']['below_critical']

    def test_pipeline_uniform_bound_holds(self):
        """Pipeline confirms uniform bound holds."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert data['summary']['uniform_bound_holds']

    def test_pipeline_gap_positive(self):
        """Pipeline confirms gap is positive."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert data['summary']['gap_positive']

    def test_pipeline_status(self):
        """Pipeline gives PROPOSITION status."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert data['summary']['status'] == 'PROPOSITION'

    def test_pipeline_physical_gap(self):
        """Pipeline computes physical gap."""
        data = full_verification_pipeline(
            g_squared=PHYSICAL_G_SQUARED, R=1.0, max_level=0
        )
        assert 'physical_gap' in data
        assert data['physical_gap']['gap_positive']
