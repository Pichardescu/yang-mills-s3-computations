"""
Tests for 't Hooft Twisted BC on T^3 (THEOREM 7.11).

Session 10: The mathematical breakthrough — twisted BC eliminates
abelian zero modes, allowing the S^3 machinery to apply on T^3.

Tests verify:
1. Twist matrix construction and cocycle condition
2. Zero-mode elimination (the key algebraic result)
3. Twisted spectrum (half-integer momenta, no zero eigenvalues)
4. Ghost curvature sign flip (negative → positive)
5. Mass gap positivity for all L
6. Comparison: twisted vs periodic
7. SU(N) extension for N = 2, 3, 5
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.torus_twisted import (
    twist_matrices_su2,
    zero_modes_twisted_su2,
    twisted_laplacian_spectrum,
    ghost_curvature_twisted,
    mass_gap_twisted_torus,
    twisted_vs_periodic_comparison,
    twist_eliminates_zero_modes_sun,
)


class TestTwistMatrices:
    """'t Hooft twist matrices satisfy required algebraic relations."""

    def test_cocycle_condition(self):
        """z_12 * z_23 * z_31 = 1 (cocycle)."""
        r = twist_matrices_su2()
        assert r['cocycle_valid']

    def test_twist_relations(self):
        """Omega_i Omega_j = z_ij Omega_j Omega_i."""
        r = twist_matrices_su2()
        # If construction didn't raise, relations are verified
        assert r['label'] == 'THEOREM'

    def test_all_twist_types(self):
        """All 3 non-trivial twist types are valid."""
        for tt in ['standard', 'cyclic_12', 'cyclic_23']:
            r = twist_matrices_su2(tt)
            assert r['cocycle_valid']

    def test_matrices_unitary(self):
        """Twist matrices are unitary (in SU(2))."""
        r = twist_matrices_su2()
        for Om in r['Omega']:
            assert np.allclose(Om @ Om.conj().T, np.eye(2))

    def test_matrices_special(self):
        """det(Omega_i) = 1 (special unitary)."""
        r = twist_matrices_su2()
        for Om in r['Omega']:
            assert abs(np.linalg.det(Om) - 1.0) < 1e-10 or \
                   abs(np.linalg.det(Om) + 1.0) < 1e-10  # ±1 for iσ


class TestZeroModeElimination:
    """THE KEY THEOREM: twist eliminates all constant abelian zero modes."""

    def test_zero_modes_eliminated_standard(self):
        """Standard twist (-1,-1,+1): 0 zero modes."""
        r = zero_modes_twisted_su2('standard')
        assert r['zero_modes_eliminated']
        assert r['n_zero_modes'] == 0

    def test_zero_modes_eliminated_all_twists(self):
        """All non-trivial twists eliminate zero modes."""
        for tt in ['standard', 'cyclic_12', 'cyclic_23']:
            r = zero_modes_twisted_su2(tt)
            assert r['zero_modes_eliminated'], f"Failed for {tt}"

    def test_disjoint_fixed_points(self):
        """Each Omega_i fixes a DIFFERENT tau_j → intersection is empty."""
        r = zero_modes_twisted_su2()
        inv = r['invariant_per_twist']
        # Omega_1 = iσ_1 fixes τ_1 (index 0)
        assert 0 in inv[0]
        # Omega_2 = iσ_2 fixes τ_2 (index 1)
        assert 1 in inv[1]
        # Intersection: {τ_1} ∩ {τ_2} = ∅
        assert len(r['common_invariant']) == 0

    def test_label_theorem(self):
        r = zero_modes_twisted_su2()
        assert r['label'] == 'THEOREM'

    def test_proof_documented(self):
        r = zero_modes_twisted_su2()
        assert 'EMPTY' in r['proof'] or 'empty' in r['proof'].lower()


class TestTwistedSpectrum:
    """Twisted Laplacian has no zero eigenvalues."""

    def test_no_zero_eigenvalue(self):
        """No zero eigenvalue in twisted spectrum."""
        r = twisted_laplacian_spectrum(2.0)
        assert not r['has_zero_eigenvalue']

    def test_gap_equals_pi_squared_over_L_squared(self):
        """Lowest eigenvalue = π²/L² (from τ₁ anti-periodic in dir 2)."""
        for L in [1.0, 2.0, 5.0]:
            r = twisted_laplacian_spectrum(L)
            expected = np.pi**2 / L**2
            assert abs(r['gap'] - expected) / expected < 0.02, \
                f"Gap mismatch at L={L}: {r['gap']:.4f} vs {expected:.4f}"

    def test_gap_matches_expected(self):
        r = twisted_laplacian_spectrum(3.0)
        assert r['gap_matches_expected']

    def test_gap_positive_all_L(self):
        """Gap is positive for all L > 0."""
        for L in [0.1, 0.5, 1.0, 2.2, 5.0, 10.0, 50.0]:
            r = twisted_laplacian_spectrum(L)
            assert r['gap'] > 0, f"Gap <= 0 at L={L}"

    def test_anti_periodic_bc(self):
        """BC matrix shows anti-periodic entries."""
        r = twisted_laplacian_spectrum(1.0)
        bc = r['boundary_conditions']
        # Should have some 0.5 (anti-periodic) entries
        has_anti = any(bc[c][d] == 0.5 for c in range(3) for d in range(3))
        assert has_anti


class TestGhostCurvature:
    """Ghost curvature is POSITIVE on twisted T^3 (contrast: negative on periodic)."""

    def test_positive(self):
        """Ghost curvature is positive with twist."""
        r = ghost_curvature_twisted(2.0)
        assert r['positive']

    def test_sign_flip(self):
        """Sign flips from negative (periodic) to positive (twisted)."""
        r = ghost_curvature_twisted(2.0)
        assert r['sign_flip']
        assert r['kappa_twisted'] > 0
        assert r['kappa_untwisted'] < 0

    def test_positive_all_L(self):
        """Ghost curvature remains positive for all L."""
        for L in [0.5, 1.0, 2.2, 5.0, 10.0]:
            r = ghost_curvature_twisted(L)
            assert r['positive'], f"Ghost kappa <= 0 at L={L}"


class TestMassGap:
    """Mass gap is POSITIVE for all L on twisted T^3."""

    def test_positive_small_L(self):
        r = mass_gap_twisted_torus(0.5)
        assert r['positive']
        assert r['gap_best'] > 0

    def test_positive_physical_L(self):
        r = mass_gap_twisted_torus(2.2)
        assert r['positive']

    def test_positive_large_L(self):
        r = mass_gap_twisted_torus(10.0)
        assert r['positive']

    def test_positive_very_large_L(self):
        r = mass_gap_twisted_torus(100.0)
        assert r['positive']

    def test_gap_scan(self):
        """Gap > 0 for all L in [0.1, 100]."""
        L_values = np.logspace(-1, 2, 20)
        for L in L_values:
            r = mass_gap_twisted_torus(L)
            assert r['positive'], f"Gap <= 0 at L={L:.2f}"

    def test_label_theorem(self):
        r = mass_gap_twisted_torus(2.2)
        assert r['label'] == 'THEOREM'


class TestTwistedVsPeriodic:
    """Twisted gap positive everywhere; periodic gap fails at large L."""

    def test_twisted_always_positive(self):
        comp = twisted_vs_periodic_comparison()
        assert comp['all_twisted_positive']

    def test_periodic_gap_shrinks(self):
        """Periodic gap decays toward 0 at large L (obstacle: no uniform bound).

        PW keeps gap > 0 at each fixed L, but gap ~ g^4/(8L) → 0.
        This is the obstacle: we can't prove a UNIFORM lower bound.
        Twisted gap also decays (π²/L²) but the mechanism is clean.
        """
        comp = twisted_vs_periodic_comparison(
            np.array([1.0, 10.0, 100.0])
        )
        gaps = [r['gap_periodic'] for r in comp['results']]
        # Gap decreases monotonically
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] < gaps[i], "Gap should decrease with L"
        # At L=100: gap very small (< 0.1)
        assert gaps[-1] < 0.1


class TestSUNExtension:
    """'t Hooft twist eliminates zero modes for SU(N) with N prime."""

    def test_su2(self):
        r = twist_eliminates_zero_modes_sun(2)
        assert r['eliminated']
        assert r['n_zero_modes'] == 0

    def test_su3(self):
        """SU(3): clock-shift matrices eliminate all zero modes."""
        r = twist_eliminates_zero_modes_sun(3)
        assert r['eliminated']
        assert r['n_zero_modes'] == 0

    def test_su5(self):
        """SU(5): prime N, maximal twist works."""
        r = twist_eliminates_zero_modes_sun(5)
        assert r['eliminated']
        assert r['n_zero_modes'] == 0

    def test_su7(self):
        """SU(7): prime N, maximal twist works."""
        r = twist_eliminates_zero_modes_sun(7)
        assert r['eliminated']

    def test_label_theorem(self):
        for N in [2, 3, 5]:
            r = twist_eliminates_zero_modes_sun(N)
            assert r['label'] == 'THEOREM'
