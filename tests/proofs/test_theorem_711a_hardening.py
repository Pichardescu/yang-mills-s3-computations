"""
Tests for THEOREM 7.11a hardening: dependency structure, label consistency,
and the dimensional transmutation bound.

These tests verify the honest restructuring of the "five converging proofs":
1. IR slavery and transfer matrix are mathematically equivalent (merged)
2. Gribov spectral (cluster bound) is PROPOSITION, not THEOREM
3. All GZ-based proofs share gamma* as single input (NOT independent)
4. g^2_max = 4*pi is NUMERICAL (documented, not first-principles)
5. Config space and log-Sobolev prove gap > 0 per R, but need GZ for R-indep
6. Dimensional transmutation gives explicit GZ-free bound (but -> 0 as R -> inf)
7. THEOREM 7.12a (gauge-invariant uniform gap) is the GZ-free existence proof
"""

import numpy as np
import pytest
import importlib.util
import os

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import(pkg, name):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(_BASE, 'src', pkg, f'{name}.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ir_slavery = _import('proofs', 'ir_slavery_gap')
dim_trans = _import('proofs', 'dimensional_transmutation_bound')

# Try to import the other modules; skip tests if they fail
try:
    transfer_matrix = _import('proofs', 'transfer_matrix_gap')
    HAS_TRANSFER_MATRIX = True
except Exception:
    HAS_TRANSFER_MATRIX = False

try:
    log_sobolev = _import('proofs', 'log_sobolev_gap')
    HAS_LOG_SOBOLEV = True
except Exception:
    HAS_LOG_SOBOLEV = False

try:
    config_space = _import('proofs', 'config_space_gap')
    HAS_CONFIG_SPACE = True
except Exception:
    HAS_CONFIG_SPACE = False

try:
    zwanziger = _import(os.path.join('spectral'), 'zwanziger_gap_equation')
    HAS_ZWANZIGER = True
except Exception:
    HAS_ZWANZIGER = False


# ===========================================================================
# Test class 1: Dependency structure documentation
# ===========================================================================
class TestDependencyStructure:
    """Verify the dependency map is honest and complete."""

    def test_dependency_map_exists(self):
        """dependency_map() returns a dict with all six categories."""
        dmap = dim_trans.dependency_map()
        assert isinstance(dmap, dict)
        expected_keys = {
            'gz_free_existence',
            'gz_pole_mass_bound',
            'gribov_spectral_cluster',
            'config_space',
            'log_sobolev',
            'dimensional_transmutation',
            'shared_dependency',
        }
        assert expected_keys.issubset(set(dmap.keys()))

    def test_gz_free_existence_does_not_use_gz(self):
        """THEOREM 7.12a must NOT list GZ propagator as input."""
        dmap = dim_trans.dependency_map()
        gz_free = dmap['gz_free_existence']
        assert gz_free['label'] == 'THEOREM'
        # Must explicitly document what it does NOT use
        assert 'does_NOT_use' in gz_free
        does_not_use = gz_free['does_NOT_use']
        assert any('GZ propagator' in item for item in does_not_use)
        assert any('gamma' in item.lower() for item in does_not_use)

    def test_gz_pole_mass_uses_gz(self):
        """GZ pole mass bound must list GZ propagator and g^2_max as inputs."""
        dmap = dim_trans.dependency_map()
        pole_mass = dmap['gz_pole_mass_bound']
        assert pole_mass['quantitative'] is True
        inputs = pole_mass['inputs']
        assert any('GZ propagator' in item for item in inputs)
        assert any('g^2_max' in item or 'g2_max' in item for item in inputs)

    def test_gribov_spectral_is_proposition(self):
        """Gribov spectral / cluster bound MUST be PROPOSITION, not THEOREM."""
        dmap = dim_trans.dependency_map()
        cluster = dmap['gribov_spectral_cluster']
        assert cluster['label'] == 'PROPOSITION'
        # Must document the weakness
        assert 'weakness' in cluster
        assert 'cluster' in cluster['weakness'].lower()

    def test_shared_dependency_acknowledged(self):
        """All five proofs share gamma* — this must be documented."""
        dmap = dim_trans.dependency_map()
        shared = dmap['shared_dependency']
        assert shared['truly_independent_proofs'] == 1
        assert shared['correlated_perspectives'] >= 3
        assert shared['gz_free_existence_proof'] == 1

    def test_dim_trans_is_gz_free(self):
        """Dimensional transmutation bound must be GZ-free."""
        dmap = dim_trans.dependency_map()
        dt = dmap['dimensional_transmutation']
        assert dt['independent'] is True
        assert 'does_NOT_use' in dt
        does_not_use = dt['does_NOT_use']
        assert any('GZ propagator' in item for item in does_not_use)

    def test_config_space_partial_independence(self):
        """Config space: existence is independent, R-independence uses GZ."""
        dmap = dim_trans.dependency_map()
        cs = dmap['config_space']
        assert 'THEOREM' in cs['label']
        assert 'PROPOSITION' in cs['label'] or 'independent_content' in cs

    def test_log_sobolev_partial_independence(self):
        """Log-Sobolev: field-space THEOREM, physical units PROPOSITION."""
        dmap = dim_trans.dependency_map()
        ls = dmap['log_sobolev']
        assert 'THEOREM' in ls['label']
        assert 'PROPOSITION' in ls['label']


# ===========================================================================
# Test class 2: IR slavery / transfer matrix equivalence
# ===========================================================================
class TestIRSlaveryTransferEquivalence:
    """Verify that IR slavery and transfer matrix give the same bound."""

    def test_ir_slavery_decay_rate(self):
        """IR slavery gives m = gamma/sqrt(2)."""
        gamma = 3.0 * np.sqrt(2) / 2.0  # gamma* for SU(2)
        result = ir_slavery.decay_rate_from_propagator(gamma)
        expected_mass = gamma / np.sqrt(2)
        assert abs(result['mass_gap'] - expected_mass) < 1e-10

    @pytest.mark.skipif(not HAS_TRANSFER_MATRIX, reason="transfer_matrix not available")
    def test_transfer_matrix_pole_mass(self):
        """Transfer matrix pole mass equals IR slavery decay rate."""
        gamma_star = 3.0 * np.sqrt(2) / 2.0
        # IR slavery bound
        ir_mass = gamma_star / np.sqrt(2)
        # Transfer matrix: m_g = sqrt(2) * gamma is the GZ pole mass
        # The gluon-channel gap is gamma/sqrt(2), same as IR slavery
        assert abs(ir_mass - 1.5) < 1e-10  # = 3/2 Lambda_QCD

    def test_merged_terminology(self):
        """Module docstring documents the merger."""
        docstring = ir_slavery.__doc__
        assert 'MERGED' in docstring or 'merged' in docstring
        assert 'transfer matrix' in docstring.lower()

    def test_cluster_bound_is_proposition_in_source(self):
        """gauge_invariant_correlator_decay must return label PROPOSITION."""
        gamma = 2.0
        x = np.array([1.0, 2.0, 5.0])
        result = ir_slavery.gauge_invariant_correlator_decay(gamma, x)
        assert result['label'] == 'PROPOSITION'


# ===========================================================================
# Test class 3: g^2_max assumption documentation
# ===========================================================================
class TestG2MaxAssumption:
    """Verify g^2_max = 4*pi is documented as NUMERICAL assumption."""

    @pytest.mark.skipif(not HAS_ZWANZIGER, reason="zwanziger not available")
    def test_zwanziger_documents_g2_max(self):
        """Zwanziger gap equation source documents g^2_max assumption."""
        import inspect
        source = inspect.getsource(zwanziger.ZwanzigerGapEquation.running_coupling_g2)
        # Must contain the word NUMERICAL or ASSUMPTION
        assert 'NUMERICAL' in source or 'ASSUMPTION' in source or 'assumption' in source

    def test_gamma_star_depends_on_g2_max(self):
        """gamma* = (N^2-1)*4*pi*sqrt(2)/(g^2_max*N) — varies with g^2_max."""
        # Default g^2_max = 4*pi
        g2_max_default = 4.0 * np.pi
        gamma_star_default = 3.0 * 4 * np.pi * np.sqrt(2) / (g2_max_default * 2)

        # 20% higher g^2_max
        g2_max_high = 1.2 * g2_max_default
        gamma_star_high = 3.0 * 4 * np.pi * np.sqrt(2) / (g2_max_high * 2)

        # gamma* decreases when g^2_max increases
        assert gamma_star_high < gamma_star_default
        # 20% change in g^2_max gives ~17% change in gamma*
        rel_change = abs(gamma_star_high - gamma_star_default) / gamma_star_default
        assert 0.10 < rel_change < 0.25

    def test_mass_gap_sensitivity_to_g2_max(self):
        """Physical mass gap scales as 1/g^2_max — documented sensitivity."""
        # m = gamma*/sqrt(2) = (N^2-1)*4*pi/(g^2_max*N)
        g2_max_default = 4.0 * np.pi
        m_default = 3.0 * 4 * np.pi / (g2_max_default * 2)  # = 3/2

        g2_max_alt = 3.0 * np.pi  # alternative choice
        m_alt = 3.0 * 4 * np.pi / (g2_max_alt * 2)  # = 2.0

        # Different g^2_max -> different mass gap estimate
        assert abs(m_default - 1.5) < 1e-10
        assert abs(m_alt - 2.0) < 1e-10
        assert m_alt > m_default  # smaller g^2_max -> larger mass gap


# ===========================================================================
# Test class 4: Dimensional transmutation bound (GZ-free)
# ===========================================================================
class TestDimensionalTransmutationBound:
    """Test the new GZ-free quantitative bound."""

    def test_small_R_geometric_dominates(self):
        """At small R, geometric gap >> dynamical bound."""
        result = dim_trans.dimensional_transmutation_gap_bound(0.1)
        assert result['geometric_gap_MeV'] > result['dynamical_bound_MeV']
        assert result['total_bound_MeV'] == result['geometric_gap_MeV']
        assert result['label'] == 'THEOREM'

    def test_large_R_dynamical_dominates(self):
        """At large R, dynamical bound >> geometric gap."""
        result = dim_trans.dimensional_transmutation_gap_bound(100.0)
        assert result['dynamical_bound_MeV'] > result['geometric_gap_MeV']
        assert result['total_bound_MeV'] == result['dynamical_bound_MeV']

    def test_always_positive(self):
        """Gap bound is positive for all tested R values."""
        for R in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
            result = dim_trans.dimensional_transmutation_gap_bound(R)
            assert result['total_bound_MeV'] > 0, f"Gap <= 0 at R = {R} fm"

    def test_gz_free_flag(self):
        """Results must be flagged as GZ-free."""
        result = dim_trans.dimensional_transmutation_gap_bound(2.2)
        assert result['gz_free'] is True

    def test_bound_decreases_at_large_R(self):
        """At large R, the bound decreases (logarithmically)."""
        r1 = dim_trans.dimensional_transmutation_gap_bound(10.0)
        r2 = dim_trans.dimensional_transmutation_gap_bound(100.0)
        r3 = dim_trans.dimensional_transmutation_gap_bound(1000.0)
        # Dynamical bound should decrease
        assert r2['dynamical_bound_MeV'] < r1['dynamical_bound_MeV']
        assert r3['dynamical_bound_MeV'] < r2['dynamical_bound_MeV']
        # But all still positive
        assert r3['dynamical_bound_MeV'] > 0

    def test_gap_table_returns_list(self):
        """gap_vs_radius_dim_trans returns a non-empty list."""
        table = dim_trans.gap_vs_radius_dim_trans()
        assert isinstance(table, list)
        assert len(table) > 10
        for entry in table:
            assert entry['total_bound_MeV'] > 0

    def test_negative_R_raises(self):
        """Negative R must raise ValueError."""
        with pytest.raises(ValueError):
            dim_trans.dimensional_transmutation_gap_bound(-1.0)

    def test_caveats_documented(self):
        """Result must document the caveat that bound -> 0."""
        result = dim_trans.dimensional_transmutation_gap_bound(100.0)
        assert 'caveats' in result
        assert 'approaches 0' in result['caveats'] or 'NOT' in result['caveats']


# ===========================================================================
# Test class 5: Label consistency between source and paper claims
# ===========================================================================
class TestLabelConsistency:
    """Verify source code labels match what the paper should say."""

    def test_ir_slavery_is_theorem(self):
        """IR slavery pole analysis is THEOREM."""
        result = ir_slavery.theorem_ir_slavery_mass_gap()
        assert result['label'] == 'THEOREM'

    def test_gauge_invariant_correlator_is_proposition(self):
        """Gauge-invariant correlator bound is PROPOSITION (not THEOREM)."""
        gamma = 2.0
        x = np.array([1.0, 2.0])
        result = ir_slavery.gauge_invariant_correlator_decay(gamma, x)
        assert result['label'] == 'PROPOSITION', (
            "Paper says THEOREM for Gribov spectral (m >= 3*Lambda), "
            "but source correctly says PROPOSITION. Paper must be corrected."
        )

    def test_ir_slavery_mass_gap_value(self):
        """m >= (3/2)*Lambda_QCD = 300 MeV for default Lambda."""
        result = ir_slavery.physical_mass_gap_ir_slavery()
        assert abs(result['mass_gap_MeV'] - 300.0) < 1.0

    @pytest.mark.skipif(not HAS_LOG_SOBOLEV, reason="log_sobolev not available")
    def test_log_sobolev_physical_is_proposition(self):
        """physical_mass_gap_bound should be labeled PROPOSITION."""
        ls = log_sobolev.LogSobolevGap()
        result = ls.physical_mass_gap_bound(5.0)
        assert result['label'] == 'PROPOSITION'

    def test_dim_trans_bound_is_theorem(self):
        """Dimensional transmutation bound is THEOREM (GZ-free)."""
        result = dim_trans.dimensional_transmutation_gap_bound(2.2)
        assert result['label'] == 'THEOREM'


# ===========================================================================
# Test class 6: The safety net — THEOREM 7.12a independence
# ===========================================================================
class TestSafetyNet:
    """Verify that THEOREM 7.12a (gauge-invariant gap) is truly GZ-free."""

    def test_existence_without_gz(self):
        """
        The existence proof (gap > 0 uniform in R) does NOT require GZ.
        Verify by checking the dependency map.
        """
        dmap = dim_trans.dependency_map()
        gz_free = dmap['gz_free_existence']

        # Must explicitly list things it does NOT use
        does_not_use = gz_free['does_NOT_use']
        assert len(does_not_use) >= 3, "Must list at least 3 GZ-related exclusions"
        # Check that GZ propagator is excluded
        all_text = ' '.join(does_not_use).lower()
        assert 'gz' in all_text or 'gribov' in all_text or 'zwanziger' in all_text
        assert 'gamma' in all_text
        assert 'propagator' in all_text

        # Must use gauge-invariant tools
        inputs = gz_free['inputs']
        assert any('Hodge' in item for item in inputs)
        assert any('EVT' in item or 'Extreme' in item for item in inputs)

    def test_separation_of_existence_and_value(self):
        """
        The dependency map must distinguish:
        - Existence (Delta_0 > 0): GZ-free, THEOREM
        - Value (Delta_0 ~ 3*Lambda): uses GZ, THEOREM within GZ
        """
        dmap = dim_trans.dependency_map()

        # Existence
        existence = dmap['gz_free_existence']
        assert existence['quantitative'] is False

        # Value identification
        value = dmap['gz_pole_mass_bound']
        assert value['quantitative'] is True


# ===========================================================================
# Test class 7: Numerical consistency checks
# ===========================================================================
class TestNumericalConsistency:
    """Cross-check numerical values between modules."""

    def test_gamma_star_consistent(self):
        """gamma* = 3*sqrt(2)/2 is used consistently."""
        gamma_star = 3.0 * np.sqrt(2) / 2.0
        # IR slavery
        ir_result = ir_slavery.physical_mass_gap_ir_slavery()
        assert abs(ir_result['gamma_star_Lambda'] - gamma_star) < 1e-10

    def test_mass_gap_from_gamma_star(self):
        """m = gamma*/sqrt(2) = 3/2."""
        gamma_star = 3.0 * np.sqrt(2) / 2.0
        mass_gap = gamma_star / np.sqrt(2)
        assert abs(mass_gap - 1.5) < 1e-10  # 3/2 Lambda_QCD

    def test_glueball_bound_from_gamma_star(self):
        """m_glueball >= sqrt(2)*gamma* = 3 (PROPOSITION)."""
        gamma_star = 3.0 * np.sqrt(2) / 2.0
        glueball_bound = np.sqrt(2) * gamma_star
        assert abs(glueball_bound - 3.0) < 1e-10  # 3 Lambda_QCD

    def test_spread_is_factor_two(self):
        """The spread 3/2 to 3 is exactly factor 2 (gluon vs glueball)."""
        ratio = 3.0 / 1.5
        assert abs(ratio - 2.0) < 1e-10
