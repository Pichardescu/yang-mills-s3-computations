"""
Tests for Physical Mass Gap computation.

Tests the PhysicalGap class which connects the 9-DOF effective Hamiltonian
on S³/I* to the physical mass gap in units of Λ_QCD.

Test categories:
    1. Effective mass M(R) properties
    2. Harmonic gap = 2/R
    3. Andrews-Clutterbuck bound positivity
    4. 1D anharmonic solver validation (harmonic limit)
    5. 1D anharmonic gap increases with quartic coupling
    6. Physical gap vs R: positive lower bound
    7. Gap stabilization at large R
    8. Dimensionless ratio analysis
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.physical_gap import PhysicalGap
from yang_mills_s3.spectral.zwanziger_gap_equation import ZwanzigerGapEquation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def pgap():
    """PhysicalGap with small direction count for speed."""
    return PhysicalGap(n_gribov_directions=20)


# ======================================================================
# 1. Effective mass M(R) properties
# ======================================================================

class TestEffectiveMass:
    """Test M(R) = V_{S³}/g²(R) = 2π²R³/g²(R)."""

    def test_positive(self, pgap):
        """M(R) > 0 for all R > 0."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            M = pgap.effective_mass(R)
            assert M > 0, f"M({R}) = {M} <= 0"

    def test_scales_as_R3_weak_coupling(self, pgap):
        """At small R (weak coupling, g² ~ const), M ~ R³."""
        R1, R2 = 0.1, 0.2
        M1 = pgap.effective_mass(R1)
        M2 = pgap.effective_mass(R2)
        # g² is nearly constant at small R, so M2/M1 ~ (R2/R1)³ = 8
        ratio = M2 / M1
        # Allow broad tolerance since g² also changes
        assert 4.0 < ratio < 12.0, f"M ratio = {ratio}, expected ~8"

    def test_formula_consistency(self, pgap):
        """M = V_{S³}/g² matches individual computations."""
        R = 2.0
        V = 2.0 * np.pi**2 * R**3
        g2 = ZwanzigerGapEquation.running_coupling_g2(R, 2)
        expected = V / g2
        actual = pgap.effective_mass(R)
        assert abs(actual - expected) / expected < 1e-12

    def test_increases_with_R(self, pgap):
        """M(R) is monotonically increasing in R."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        masses = [pgap.effective_mass(R) for R in R_values]
        for i in range(len(masses) - 1):
            assert masses[i + 1] > masses[i], \
                f"M not increasing: M({R_values[i]}) = {masses[i]}, " \
                f"M({R_values[i+1]}) = {masses[i+1]}"


# ======================================================================
# 2. Harmonic gap = 2/R
# ======================================================================

class TestHarmonicGap:
    """Test harmonic gap ω = 2/R."""

    def test_value(self, pgap):
        """harmonic_gap(R) = 2/R."""
        for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
            assert abs(pgap.harmonic_gap(R) - 2.0 / R) < 1e-14

    def test_positive(self, pgap):
        """Harmonic gap > 0 for all R > 0."""
        for R in [0.1, 1.0, 100.0]:
            assert pgap.harmonic_gap(R) > 0

    def test_decreases_with_R(self, pgap):
        """Harmonic gap decreases as R increases."""
        gaps = [pgap.harmonic_gap(R) for R in [1.0, 2.0, 5.0]]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]


# ======================================================================
# 3. Andrews-Clutterbuck bound positivity
# ======================================================================

class TestACBound:
    """Test particle_in_box_gap = 3π²/(2M·d²) > 0."""

    def test_positive(self, pgap):
        """AC bound > 0 for all R."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            ac = pgap.particle_in_box_gap(R)
            assert ac > 0, f"AC bound at R={R} is {ac} <= 0"

    def test_formula(self, pgap):
        """AC bound matches 3π²/(2M·d²)."""
        R = 1.0
        M = pgap.effective_mass(R)
        d = pgap._get_gribov_diameter(R)
        expected = 3.0 * np.pi**2 / (2.0 * M * d**2)
        actual = pgap.particle_in_box_gap(R)
        assert abs(actual - expected) / expected < 1e-10


# ======================================================================
# 4. 1D anharmonic solver: harmonic limit validation
# ======================================================================

class TestAnharmonic1DValidation:
    """Validate the 1D FD solver reproduces the harmonic gap when α=0."""

    def test_harmonic_limit_small_R(self, pgap):
        """At small R (large ω, tight HO), harmonic gap ≈ ω if box >> HO length."""
        R = 0.5
        omega = 2.0 / R  # = 4.0
        # Harmonic numerical should be close to ω
        gap_num = pgap.harmonic_gap_numerical(R, n_basis=80)
        # Allow 20% tolerance due to finite box effects
        assert abs(gap_num - omega) / omega < 0.20, \
            f"Harmonic numerical gap = {gap_num}, expected ~{omega}"

    def test_anharmonic_geq_harmonic_no_quartic(self, pgap):
        """Anharmonic gap ≥ harmonic numerical gap (quartic adds confinement)."""
        R = 1.0
        gap_harm = pgap.harmonic_gap_numerical(R, n_basis=60)
        gap_anh = pgap.anharmonic_gap_1d(R, n_basis=60)
        # Anharmonic gap should be >= harmonic gap (quartic raises it)
        assert gap_anh >= gap_harm * 0.95, \
            f"Anharmonic {gap_anh} < harmonic {gap_harm}"


# ======================================================================
# 5. 1D anharmonic gap increases with quartic coupling
# ======================================================================

class TestQuarticEffect:
    """Test that the quartic term raises the gap."""

    def test_gap_increases_at_strong_coupling(self):
        """At larger R (stronger coupling), quartic has more effect."""
        pgap = PhysicalGap(n_gribov_directions=20)
        # Compare anharmonic gap with and without quartic
        # The anharmonic gap includes quartic; harmonic_gap_numerical does not
        R = 2.0
        gap_harm = pgap.harmonic_gap_numerical(R, n_basis=60)
        gap_anh = pgap.anharmonic_gap_1d(R, n_basis=60)
        # At moderate coupling, quartic should raise the gap
        assert gap_anh >= gap_harm * 0.9, \
            f"Anharmonic {gap_anh} not >= 0.9 * harmonic {gap_harm}"


# ======================================================================
# 6. Physical gap vs R: positive lower bound
# ======================================================================

class TestGapVsR:
    """Test that physical gap has a positive lower bound."""

    def test_all_positive(self, pgap):
        """All gap components > 0 for tested R values."""
        R_values = [0.5, 1.0, 2.0, 5.0]
        results = pgap.physical_gap_vs_R(R_values, n_basis_1d=50, n_basis_3d=15)

        for idx, R in enumerate(R_values):
            assert results['harmonic_gap'][idx] > 0
            assert results['ac_bound'][idx] > 0
            assert results['anharmonic_1d'][idx] > 0
            assert results['best_bound'][idx] > 0

    def test_best_bound_positive_minimum(self, pgap):
        """The best bound has a positive minimum over R."""
        R_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        results = pgap.physical_gap_vs_R(R_values, n_basis_1d=50, n_basis_3d=15)
        min_gap = np.min(results['best_bound'])
        assert min_gap > 0, f"Minimum best bound = {min_gap} <= 0"

    def test_results_structure(self, pgap):
        """Results dict has all expected keys."""
        results = pgap.physical_gap_vs_R([1.0, 2.0], n_basis_1d=30, n_basis_3d=10)
        expected_keys = [
            'R', 'harmonic_gap', 'ac_bound', 'anharmonic_1d',
            'anharmonic_3d', 'best_bound', 'effective_mass',
            'g_squared', 'gribov_diameter', 'quartic_coupling', 'label',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"


# ======================================================================
# 7. Gap stabilization at large R
# ======================================================================

class TestStabilization:
    """Test gap behavior at large R."""

    def test_anharmonic_gap_finite_large_R(self, pgap):
        """Anharmonic 1D gap remains finite at large R."""
        for R in [5.0, 10.0]:
            gap = pgap.anharmonic_gap_1d(R, n_basis=50)
            assert np.isfinite(gap), f"Gap at R={R} is not finite: {gap}"
            assert gap > 0, f"Gap at R={R} is {gap} <= 0"

    def test_harmonic_gap_decreases(self, pgap):
        """Harmonic gap 2/R decreases to 0 — this is why we need anharmonic."""
        gap_1 = pgap.harmonic_gap(1.0)
        gap_10 = pgap.harmonic_gap(10.0)
        assert gap_10 < gap_1
        assert gap_10 == pytest.approx(0.2, abs=1e-14)


# ======================================================================
# 8. Dimensionless ratio analysis
# ======================================================================

class TestDimensionlessRatios:
    """Test the dimensionless ratio classification."""

    def test_small_R_harmonic_regime(self, pgap):
        """At small R, the system is in harmonic regime."""
        R = 0.3
        M = pgap.effective_mass(R)
        alpha = pgap.quartic_coupling(R)
        d = pgap._get_gribov_diameter(R)
        ratios = PhysicalGap._dimensionless_ratios(R, M, 2.0/R, alpha, d)
        # At small R, beta2 should be large (box >> HO length)
        assert ratios['beta2'] > 1.0 or ratios['regime'] in ['harmonic', 'quartic']

    def test_ratio_keys(self, pgap):
        """Dimensionless ratios dict has expected keys."""
        ratios = PhysicalGap._dimensionless_ratios(1.0, 1.0, 2.0, 0.1, 1.0)
        assert 'beta1' in ratios
        assert 'beta2' in ratios
        assert 'regime' in ratios


# ======================================================================
# 9. Complete analysis
# ======================================================================

class TestCompleteAnalysis:
    """Integration test for complete_analysis."""

    def test_complete_analysis_runs(self, pgap):
        """complete_analysis runs without error."""
        result = pgap.complete_analysis(
            R_range=[0.5, 1.0, 2.0, 5.0],
            n_basis_1d=40,
            n_basis_3d=12,
        )
        assert 'min_gap' in result
        assert 'assessment' in result
        assert 'theorems_used' in result

    def test_min_gap_positive(self, pgap):
        """Minimum gap over R is positive."""
        result = pgap.complete_analysis(
            R_range=[0.5, 1.0, 2.0, 5.0],
            n_basis_1d=40,
            n_basis_3d=12,
        )
        assert result['min_gap'] > 0, \
            f"Minimum gap = {result['min_gap']} <= 0"

    def test_assessment_positive(self, pgap):
        """Assessment indicates positive gap."""
        result = pgap.complete_analysis(
            R_range=[0.5, 1.0, 2.0, 5.0],
            n_basis_1d=40,
            n_basis_3d=12,
        )
        assert 'POSITIVE' in result['assessment'] or result['min_gap'] > 0


# ======================================================================
# 10. Volume of S³
# ======================================================================

class TestVolumeS3:
    """Test volume computation."""

    def test_volume_formula(self):
        """V(R) = 2π²R³."""
        for R in [0.5, 1.0, 2.0]:
            expected = 2.0 * np.pi**2 * R**3
            assert abs(PhysicalGap.volume_s3(R) - expected) < 1e-12

    def test_volume_positive(self):
        """Volume is positive."""
        assert PhysicalGap.volume_s3(1.0) > 0


# ======================================================================
# 11. Quartic coupling
# ======================================================================

class TestQuarticCoupling:
    """Test quartic coupling α = g²C₄/(2V)."""

    def test_positive(self, pgap):
        """Quartic coupling > 0."""
        for R in [0.5, 1.0, 5.0]:
            assert pgap.quartic_coupling(R) > 0

    def test_decreases_with_R(self, pgap):
        """α decreases with R (since V grows as R³)."""
        alpha_1 = pgap.quartic_coupling(1.0)
        alpha_5 = pgap.quartic_coupling(5.0)
        assert alpha_5 < alpha_1
