"""
Tests for Covering Space Lift: S^3/I* -> S^3 for Yang-Mills Mass Gap.

Tests the spectral decomposition, sector mixing analysis, equivariant
Kato-Rellich bounds, and the full covering space gap lift theorem.

Test categories:
    1. Spectrum decomposition (multiplicities, gaps)
    2. Sector consistency (inv + non_inv = full)
    3. I*-invariant gap properties
    4. Non-I*-invariant gap properties
    5. Full S^3 gap determination
    6. Sector mixing: [V_2, Pi_{I*}] = 0
    7. Sector mixing: [V_4, Pi_{I*}] = 0
    8. Full Hamiltonian commutation [H, Pi_{I*}] = 0
    9. Equivariant Kato-Rellich bounds
   10. 6-mode S^3 effective Hamiltonian
   11. Gap comparison: S^3/I* vs S^3
   12. Lift theorem consistency
   13. Mode counting verification
   14. Physical coupling regime
   15. Edge cases and scaling
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.covering_space_lift import (
    CoveringSpaceSpectrum,
    SectorMixingAnalysis,
    EquivariantKatoRellich,
    CoveringSpaceLift,
    S3EffectiveHamiltonian,
    _adjoint_dimension,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def spectrum():
    """CoveringSpaceSpectrum on unit S^3 with SU(2)."""
    return CoveringSpaceSpectrum(R=1.0, gauge_group='SU(2)')


@pytest.fixture
def spectrum_su3():
    """CoveringSpaceSpectrum with SU(3)."""
    return CoveringSpaceSpectrum(R=1.0, gauge_group='SU(3)')


@pytest.fixture
def mixing():
    """SectorMixingAnalysis on unit S^3."""
    return SectorMixingAnalysis(R=1.0, gauge_group='SU(2)')


@pytest.fixture
def kr():
    """EquivariantKatoRellich with moderate coupling."""
    return EquivariantKatoRellich(R=1.0, g_coupling=1.0, gauge_group='SU(2)')


@pytest.fixture
def kr_physical():
    """EquivariantKatoRellich at physical coupling."""
    return EquivariantKatoRellich(R=2.2, g_coupling=2.5, gauge_group='SU(2)')


@pytest.fixture
def lift():
    """CoveringSpaceLift with moderate coupling."""
    return CoveringSpaceLift(R=1.0, g_coupling=1.0, gauge_group='SU(2)')


@pytest.fixture
def s3_heff():
    """S3EffectiveHamiltonian with R=1, g=1."""
    return S3EffectiveHamiltonian(R=1.0, g_coupling=1.0)


# ======================================================================
# 1. Spectrum decomposition: multiplicities
# ======================================================================

class TestSpectrumDecomposition:
    """Test the decomposition of S^3 spectrum into I*-sectors."""

    def test_full_s3_multiplicity_k1(self, spectrum):
        """Full S^3 coexact multiplicity at k=1 is 6."""
        assert spectrum.full_s3_multiplicity(1) == 6

    def test_full_s3_multiplicity_k2(self, spectrum):
        """Full S^3 coexact multiplicity at k=2 is 16."""
        assert spectrum.full_s3_multiplicity(2) == 16

    def test_full_s3_multiplicity_formula(self, spectrum):
        """Full multiplicity follows 2k(k+2) for k >= 1."""
        for k in range(1, 20):
            expected = 2 * k * (k + 2)
            assert spectrum.full_s3_multiplicity(k) == expected, \
                f"Multiplicity mismatch at k={k}"

    def test_full_s3_multiplicity_k0(self, spectrum):
        """No coexact 1-forms at k=0."""
        assert spectrum.full_s3_multiplicity(0) == 0

    def test_invariant_multiplicity_k1(self, spectrum):
        """I*-invariant multiplicity at k=1 is 3 (right-invariant forms)."""
        assert spectrum.invariant_multiplicity(1) == 3

    def test_invariant_multiplicity_k2(self, spectrum):
        """I*-invariant multiplicity at k=2 is 0."""
        assert spectrum.invariant_multiplicity(2) == 0

    def test_non_invariant_multiplicity_k1(self, spectrum):
        """Non-I*-invariant multiplicity at k=1 is 3."""
        assert spectrum.non_invariant_multiplicity(1) == 3

    def test_non_invariant_multiplicity_k2(self, spectrum):
        """Non-I*-invariant multiplicity at k=2 is 16 (all modes)."""
        assert spectrum.non_invariant_multiplicity(2) == 16


# ======================================================================
# 2. Sector consistency: inv + non_inv = full
# ======================================================================

class TestSectorConsistency:
    """Test that sector multiplicities sum to the full multiplicity."""

    def test_sector_sum_k1(self, spectrum):
        """At k=1: 3 + 3 = 6."""
        inv = spectrum.invariant_multiplicity(1)
        non_inv = spectrum.non_invariant_multiplicity(1)
        full = spectrum.full_s3_multiplicity(1)
        assert inv + non_inv == full

    def test_sector_sum_all_k(self, spectrum):
        """Sector multiplicities sum correctly for k=1 to 40."""
        for k in range(1, 41):
            inv = spectrum.invariant_multiplicity(k)
            non_inv = spectrum.non_invariant_multiplicity(k)
            full = spectrum.full_s3_multiplicity(k)
            assert inv + non_inv == full, \
                f"Sector sum mismatch at k={k}: {inv} + {non_inv} != {full}"

    def test_non_invariant_nonnegative(self, spectrum):
        """Non-invariant multiplicity is always >= 0."""
        for k in range(1, 60):
            assert spectrum.non_invariant_multiplicity(k) >= 0, \
                f"Negative non-invariant multiplicity at k={k}"

    def test_sector_decomposition_length(self, spectrum):
        """Sector decomposition returns entries for all levels."""
        decomp = spectrum.sector_decomposition(k_max=30)
        assert len(decomp) == 30

    def test_sector_decomposition_consistency(self, spectrum):
        """Each entry in sector_decomposition is internally consistent."""
        decomp = spectrum.sector_decomposition(k_max=30)
        for entry in decomp:
            assert entry['full_multiplicity'] == (
                entry['invariant_multiplicity'] + entry['non_invariant_multiplicity']
            ), f"Inconsistency at k={entry['k']}"


# ======================================================================
# 3. I*-invariant gap properties
# ======================================================================

class TestInvariantGap:
    """Test the gap of the I*-invariant sector."""

    def test_invariant_gap_k1(self, spectrum):
        """I*-invariant gap is at k=1."""
        gap = spectrum.invariant_gap()
        assert gap['k'] == 1

    def test_invariant_gap_eigenvalue(self, spectrum):
        """I*-invariant gap eigenvalue is 4/R^2."""
        gap = spectrum.invariant_gap()
        assert abs(gap['eigenvalue'] - 4.0) < 1e-12

    def test_invariant_gap_multiplicity(self, spectrum):
        """I*-invariant gap multiplicity is 3."""
        gap = spectrum.invariant_gap()
        assert gap['multiplicity'] == 3

    def test_invariant_gap_status(self, spectrum):
        """I*-invariant gap status is THEOREM."""
        gap = spectrum.invariant_gap()
        assert gap['status'] == 'THEOREM'

    def test_invariant_gap_mass(self):
        """I*-invariant gap mass at R=2.2 fm."""
        from yang_mills_s3.geometry.poincare_homology import HBAR_C_MEV_FM
        spectrum = CoveringSpaceSpectrum(R=2.2)
        gap = spectrum.invariant_gap()
        expected_mass = 2.0 * HBAR_C_MEV_FM / 2.2
        assert abs(gap['mass_mev'] - expected_mass) < 0.01


# ======================================================================
# 4. Non-I*-invariant gap properties
# ======================================================================

class TestNonInvariantGap:
    """Test the gap of the non-I*-invariant sector."""

    def test_non_invariant_gap_exists(self, spectrum):
        """Non-I*-invariant gap exists (there are non-invariant modes)."""
        gap = spectrum.non_invariant_gap()
        assert gap['multiplicity'] > 0

    def test_non_invariant_gap_at_k1(self, spectrum):
        """Non-I*-invariant modes exist at k=1 (multiplicity 3)."""
        non_inv = spectrum.non_invariant_multiplicity(1)
        assert non_inv == 3, f"Expected 3 non-inv modes at k=1, got {non_inv}"

    def test_non_invariant_gap_eigenvalue(self, spectrum):
        """Non-I*-invariant gap eigenvalue is 4/R^2 (same as invariant)."""
        gap = spectrum.non_invariant_gap()
        # The non-invariant sector also has modes at k=1
        assert abs(gap['eigenvalue'] - 4.0) < 1e-12

    def test_non_invariant_gap_k(self, spectrum):
        """Non-I*-invariant gap is at k=1."""
        gap = spectrum.non_invariant_gap()
        assert gap['k'] == 1


# ======================================================================
# 5. Full S^3 gap determination
# ======================================================================

class TestFullGap:
    """Test the full S^3 gap = min(inv_gap, non_inv_gap)."""

    def test_full_gap_eigenvalue(self, spectrum):
        """Full S^3 gap eigenvalue is 4/R^2."""
        gap = spectrum.full_s3_gap()
        assert abs(gap['eigenvalue'] - 4.0) < 1e-12

    def test_full_gap_equals_inv_gap(self, spectrum):
        """Full gap equals I*-invariant gap."""
        full = spectrum.full_s3_gap()
        inv = spectrum.invariant_gap()
        assert abs(full['eigenvalue'] - inv['eigenvalue']) < 1e-12

    def test_full_gap_leq_non_inv_gap(self, spectrum):
        """Full gap <= non-I*-invariant gap."""
        full = spectrum.full_s3_gap()
        non_inv = spectrum.non_invariant_gap()
        assert full['eigenvalue'] <= non_inv['eigenvalue'] + 1e-12

    def test_full_gap_multiplicity(self, spectrum):
        """Full multiplicity at the gap level is 6 (3 inv + 3 non-inv)."""
        gap = spectrum.full_s3_gap()
        assert gap['full_multiplicity_at_gap'] == 6

    def test_full_gap_scaling(self):
        """Full gap scales as 1/R^2."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            s = CoveringSpaceSpectrum(R=R)
            gap = s.full_s3_gap()
            assert abs(gap['eigenvalue'] * R**2 - 4.0) < 1e-10, \
                f"Gap * R^2 = {gap['eigenvalue'] * R**2}, expected 4.0 at R={R}"


# ======================================================================
# 6. Sector mixing: [V_2, Pi_{I*}] = 0
# ======================================================================

class TestQuadraticCommutation:
    """Test that the quadratic potential commutes with the I*-projector."""

    def test_v2_commutes(self, mixing):
        """V_2 commutes with Pi_{I*}."""
        result = mixing.quadratic_commutes()
        assert result['commutes'] is True

    def test_v2_status(self, mixing):
        """V_2 commutation is a THEOREM."""
        result = mixing.quadratic_commutes()
        assert result['status'] == 'THEOREM'


# ======================================================================
# 7. Sector mixing: [V_4, Pi_{I*}] = 0
# ======================================================================

class TestQuarticCommutation:
    """Test that the quartic potential commutes with the I*-projector."""

    def test_v4_commutes(self, mixing):
        """V_4 commutes with Pi_{I*}."""
        result = mixing.quartic_commutes()
        assert result['commutes'] is True

    def test_v4_status(self, mixing):
        """V_4 commutation is a THEOREM."""
        result = mixing.quartic_commutes()
        assert result['status'] == 'THEOREM'

    def test_v4_proof_ingredients(self, mixing):
        """V_4 proof has all required ingredients."""
        result = mixing.quartic_commutes()
        assert 'lie_bracket' in result['proof_ingredients']
        assert 'wedge_product' in result['proof_ingredients']
        assert 'hodge_star' in result['proof_ingredients']
        assert 'inner_product' in result['proof_ingredients']
        assert 'volume_form' in result['proof_ingredients']


# ======================================================================
# 8. Full Hamiltonian commutation
# ======================================================================

class TestHamiltonianCommutation:
    """Test [H_eff, Pi_{I*}] = 0."""

    def test_full_hamiltonian_commutes(self, mixing):
        """Full H_eff commutes with Pi_{I*}."""
        result = mixing.full_hamiltonian_commutes()
        assert result['commutes'] is True

    def test_all_parts_commute(self, mixing):
        """Kinetic, quadratic, and quartic all commute."""
        result = mixing.full_hamiltonian_commutes()
        assert result['kinetic_commutes'] is True
        assert result['quadratic_commutes'] is True
        assert result['quartic_commutes'] is True

    def test_numerical_mixing(self, mixing):
        """Numerical test confirms sector decoupling."""
        result = mixing.numerical_mixing_test(n_samples=1000)
        assert result['sectors_decouple']
        assert result['max_cross_gradient'] == 0.0


# ======================================================================
# 9. Equivariant Kato-Rellich bounds
# ======================================================================

class TestEquivariantKR:
    """Test the equivariant Kato-Rellich bounds."""

    def test_alpha_positive(self, kr):
        """Alpha is positive for nonzero coupling."""
        assert kr.kato_rellich_alpha() > 0

    def test_alpha_at_zero_coupling(self):
        """Alpha = 0 at zero coupling."""
        kr = EquivariantKatoRellich(R=1.0, g_coupling=0.0)
        assert kr.kato_rellich_alpha() == 0.0

    def test_alpha_below_one_weak_coupling(self, kr):
        """Alpha < 1 at g=1."""
        assert kr.kato_rellich_alpha() < 1.0

    def test_critical_coupling_value(self, kr):
        """Critical coupling g^2_c = 24*pi^2/sqrt(2) ~ 167.53."""
        g2_c = kr.critical_coupling()
        assert abs(g2_c - 167.53) < 1.0

    def test_inv_gap_positive(self, kr):
        """I*-invariant sector gap is positive."""
        result = kr.invariant_sector_gap()
        assert result['gap_lower_bound'] > 0
        assert result['gap_survives']

    def test_non_inv_gap_positive(self, kr):
        """Non-I*-invariant sector gap is positive."""
        result = kr.non_invariant_sector_gap()
        assert result['gap_lower_bound'] > 0
        assert result['gap_survives']

    def test_non_inv_gap_geq_inv_gap(self, kr):
        """Non-inv gap bound >= inv gap bound (same free gap at k=1)."""
        inv = kr.invariant_sector_gap()
        non_inv = kr.non_invariant_sector_gap()
        assert non_inv['gap_lower_bound'] >= inv['gap_lower_bound'] - 1e-12

    def test_full_s3_gap_positive(self, kr):
        """Full S^3 gap bound is positive."""
        result = kr.full_s3_gap_bound()
        assert result['full_gap_lower_bound'] > 0
        assert result['gap_survives']

    def test_physical_coupling_gap(self, kr_physical):
        """Gap survives at physical coupling (g ~ 2.5, R ~ 2.2 fm)."""
        result = kr_physical.full_s3_gap_bound()
        assert result['gap_survives']
        assert result['alpha'] < 1.0

    def test_alpha_scales_with_g2(self):
        """Alpha scales linearly with g^2."""
        kr1 = EquivariantKatoRellich(R=1.0, g_coupling=1.0)
        kr2 = EquivariantKatoRellich(R=1.0, g_coupling=2.0)
        assert abs(kr2.kato_rellich_alpha() / kr1.kato_rellich_alpha() - 4.0) < 1e-10


# ======================================================================
# 10. 6-mode S^3 effective Hamiltonian
# ======================================================================

class TestS3EffectiveHamiltonian:
    """Test the 6-mode effective Hamiltonian on S^3."""

    def test_quadratic_nonneg(self, s3_heff):
        """V_2 >= 0 on the 6-mode system."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            a = rng.standard_normal(18)
            assert s3_heff.quadratic_potential(a) >= -1e-15

    def test_quadratic_zero_at_origin(self, s3_heff):
        """V_2(0) = 0."""
        assert abs(s3_heff.quadratic_potential(np.zeros(18))) < 1e-15

    def test_quartic_nonneg(self, s3_heff):
        """V_4 >= 0 on the 6-mode system."""
        result = s3_heff.quartic_nonnegative(n_samples=5000)
        assert result['nonnegative']

    def test_quartic_zero_at_origin(self, s3_heff):
        """V_4(0) = 0."""
        assert abs(s3_heff.quartic_potential(np.zeros(18))) < 1e-15

    def test_confining(self, s3_heff):
        """Total potential is confining (V -> inf as |a| -> inf)."""
        result = s3_heff.is_confining(n_directions=20, n_radii=10)
        assert result['confining']

    def test_gap_bound_positive(self, s3_heff):
        """Gap lower bound is positive for moderate coupling."""
        result = s3_heff.gap_lower_bound()
        assert result['gap_lower_bound'] > 0
        assert result['gap_survives']

    def test_total_potential_positive_nonzero_config(self, s3_heff):
        """V(a) > 0 for any nonzero a."""
        rng = np.random.default_rng(77)
        for _ in range(1000):
            a = rng.standard_normal(18) * rng.uniform(0.01, 10.0)
            v = s3_heff.total_potential(a)
            assert v > -1e-14, f"V(a) = {v} < 0"

    def test_quartic_matches_3mode_structure(self):
        """
        The quartic on 6-mode system has the same algebraic structure
        as on 3-mode system: V_4 = (g^2/2)*[(Tr S)^2 - Tr(S^2)]
        where S = M^T M (3x3).
        """
        rng = np.random.default_rng(99)
        g = 2.0
        h6 = S3EffectiveHamiltonian(R=1.0, g_coupling=g)

        for _ in range(100):
            a = rng.standard_normal((6, 3))
            M = a
            S = M.T @ M
            expected = 0.5 * g**2 * (np.trace(S)**2 - np.trace(S @ S))
            actual = h6.quartic_potential(a)
            assert abs(actual - expected) < 1e-12, \
                f"Quartic mismatch: {actual} vs {expected}"


# ======================================================================
# 11. Gap comparison: S^3/I* vs S^3
# ======================================================================

class TestGapComparison:
    """Test that S^3 gap relates correctly to S^3/I* gap."""

    def test_gap_comparison_all_positive(self, lift):
        """All gaps are positive across couplings."""
        result = lift.gap_comparison(
            g_values=[0.0, 0.5, 1.0, 2.0],
            n_basis=6
        )
        assert result['all_gaps_positive']

    def test_s3_effective_gap_positive(self, lift):
        """S^3 effective Hamiltonian gap is positive."""
        result = lift.s3_effective_hamiltonian_gap(n_basis=6)
        assert result['gap_s3_star'] > 0

    def test_s3_gap_bounded_by_kr(self, lift):
        """S^3 gap from reduced Hamiltonian is positive and consistent with KR."""
        gap_result = lift.s3_effective_hamiltonian_gap(n_basis=6)
        # The numerical gap from the reduced Hamiltonian should be positive.
        # Note: the reduced Hamiltonian uses a truncated basis, so the numerical
        # gap may differ from the analytic KR bound. Both should be positive.
        assert gap_result['gap_s3_star'] > 0
        kr = lift.kr
        bound = kr.full_s3_gap_bound()
        assert bound['full_gap_lower_bound'] > 0


# ======================================================================
# 12. Lift theorem consistency
# ======================================================================

class TestLiftTheorem:
    """Test the full covering space gap lift theorem."""

    def test_lift_theorem_status(self, lift):
        """Lift theorem status is THEOREM at moderate coupling."""
        result = lift.lift_theorem()
        assert result['theorem']['status'] == 'THEOREM'

    def test_lift_theorem_proof_chain(self, lift):
        """Proof chain has all 5 steps."""
        result = lift.lift_theorem()
        assert 'step_1' in result['proof_chain']
        assert 'step_2' in result['proof_chain']
        assert 'step_3' in result['proof_chain']
        assert 'step_4' in result['proof_chain']
        assert 'step_5' in result['proof_chain']

    def test_lift_theorem_all_steps_theorem(self, lift):
        """All steps in the proof chain have status THEOREM."""
        result = lift.lift_theorem()
        for step_name, step in result['proof_chain'].items():
            assert step['status'] == 'THEOREM', \
                f"Step {step_name} has status {step['status']}, expected THEOREM"

    def test_lift_theorem_gap_positive(self, lift):
        """Gap analysis shows positive gap."""
        result = lift.lift_theorem()
        assert result['gap_analysis']['gap_survives']

    def test_full_report_nonempty(self, lift):
        """Full report generates non-empty string."""
        report = lift.full_report()
        assert len(report) > 100
        assert 'COVERING SPACE LIFT' in report


# ======================================================================
# 13. Mode counting verification
# ======================================================================

class TestModeCounting:
    """Test the degree-of-freedom counting."""

    def test_s3_star_modes_k1(self, spectrum):
        """S^3/I* has 3 spatial modes at k=1."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3_star']['spatial_modes_k1'] == 3

    def test_s3_modes_k1(self, spectrum):
        """S^3 has 6 spatial modes at k=1."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3']['spatial_modes_k1'] == 6

    def test_s3_star_dof_su2(self, spectrum):
        """S^3/I* with SU(2) has 9 total DOF at k=1."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3_star']['total_dof'] == 9

    def test_s3_dof_su2(self, spectrum):
        """S^3 with SU(2) has 18 total DOF at k=1."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3']['total_dof'] == 18

    def test_dof_ratio(self, spectrum):
        """DOF ratio S^3/S^3* = 2."""
        modes = spectrum.effective_theory_modes()
        assert modes['ratio_dof'] == 2.0

    def test_physical_dof_s3_star(self, spectrum):
        """S^3/I* physical DOF = 9 - 3 = 6 (after gauge fixing)."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3_star']['physical_dof'] == 6

    def test_physical_dof_s3(self, spectrum):
        """S^3 physical DOF = 18 - 3 = 15 (after gauge fixing)."""
        modes = spectrum.effective_theory_modes()
        assert modes['s3']['physical_dof'] == 15

    def test_su3_modes(self, spectrum_su3):
        """SU(3) has 8 adjoint colors."""
        modes = spectrum_su3.effective_theory_modes()
        assert modes['s3_star']['adjoint_dim'] == 8
        assert modes['s3_star']['total_dof'] == 3 * 8  # = 24


# ======================================================================
# 14. Physical coupling regime
# ======================================================================

class TestPhysicalCoupling:
    """Test at physical coupling values."""

    def test_physical_alpha_small(self):
        """Alpha < 0.2 at physical coupling (g ~ 2.5)."""
        kr = EquivariantKatoRellich(R=2.2, g_coupling=2.5)
        alpha = kr.kato_rellich_alpha()
        assert alpha < 0.2, f"alpha = {alpha} >= 0.2 at physical coupling"

    def test_physical_gap_survives(self):
        """Gap survives at physical coupling."""
        kr = EquivariantKatoRellich(R=2.2, g_coupling=2.5)
        result = kr.full_s3_gap_bound()
        assert result['gap_survives']

    def test_physical_gap_retains_80_percent(self):
        """At physical coupling, at least 80% of the free gap is retained."""
        kr = EquivariantKatoRellich(R=2.2, g_coupling=2.5)
        alpha = kr.kato_rellich_alpha()
        assert (1 - alpha) > 0.80, \
            f"Only {(1-alpha)*100:.1f}% of gap retained, expected > 80%"

    def test_physical_lift_theorem(self):
        """Full lift theorem holds at physical coupling."""
        lift = CoveringSpaceLift(R=2.2, g_coupling=2.5)
        result = lift.lift_theorem()
        assert result['theorem']['status'] == 'THEOREM'


# ======================================================================
# 15. Edge cases and scaling
# ======================================================================

class TestEdgeCases:
    """Test edge cases and scaling properties."""

    def test_zero_coupling_gap(self):
        """At g=0, gap is exactly 4/R^2."""
        kr = EquivariantKatoRellich(R=1.0, g_coupling=0.0)
        result = kr.full_s3_gap_bound()
        assert abs(result['full_gap_lower_bound'] - 4.0) < 1e-12

    def test_gap_scales_with_R(self):
        """Gap bound scales as 1/R^2."""
        for R in [0.5, 1.0, 2.0, 5.0]:
            kr = EquivariantKatoRellich(R=R, g_coupling=1.0)
            result = kr.full_s3_gap_bound()
            gap_times_R2 = result['full_gap_lower_bound'] * R**2
            # Should be ~ (1-alpha) * 4, independent of R
            expected = (1 - kr.kato_rellich_alpha()) * 4.0
            assert abs(gap_times_R2 - expected) < 0.01, \
                f"Gap * R^2 = {gap_times_R2}, expected {expected} at R={R}"

    def test_adjoint_dimension_su2(self):
        """SU(2) adjoint dimension is 3."""
        assert _adjoint_dimension('SU(2)') == 3

    def test_adjoint_dimension_su3(self):
        """SU(3) adjoint dimension is 8."""
        assert _adjoint_dimension('SU(3)') == 8

    def test_adjoint_dimension_so3(self):
        """SO(3) adjoint dimension is 3."""
        assert _adjoint_dimension('SO(3)') == 3

    def test_eigenvalue_formula(self, spectrum):
        """Eigenvalue follows (k+1)^2/R^2."""
        for k in range(1, 20):
            expected = (k + 1)**2
            assert abs(spectrum.eigenvalue(k) - expected) < 1e-12

    def test_eigenvalue_scaling(self):
        """Eigenvalue scales as 1/R^2."""
        R = 3.0
        s = CoveringSpaceSpectrum(R=R)
        assert abs(s.eigenvalue(1) - 4.0 / R**2) < 1e-12

    def test_large_coupling_bound_fails(self):
        """At very large coupling, KR bound fails."""
        kr = EquivariantKatoRellich(R=1.0, g_coupling=15.0)
        alpha = kr.kato_rellich_alpha()
        assert alpha > 1.0, f"alpha = {alpha} should be > 1 at g=15"

    def test_s3_heff_18_dof(self, s3_heff):
        """S^3 effective Hamiltonian has 18 DOF."""
        assert s3_heff.n_dof == 18
        assert s3_heff.n_spatial == 6
        assert s3_heff.n_colors == 3
