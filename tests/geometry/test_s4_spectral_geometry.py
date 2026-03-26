"""
Tests for S4 Spectral Geometry.

Verifies eigenvalues, multiplicities, curvature data, and physical mass gap
for the round 4-sphere S^4(R). All values are standard results from
differential geometry on constant-curvature spaces.
"""

import pytest
import numpy as np
from yang_mills_s3.geometry.s4_spectral_geometry import S4SpectralGeometry


# ======================================================================
# Scalar eigenvalues: l(l+3)/R^2, mult = (l+1)(l+2)(2l+3)/6
# ======================================================================
class TestScalarEigenvaluesS4:
    """Delta_0 on S^4: eigenvalue = l(l+3)/R^2."""

    def test_zero_mode(self):
        """l=0: eigenvalue = 0, multiplicity = 1 (constants)."""
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=1.0, l_max=0)
        ev, mult = spectrum[0]
        assert ev == 0.0
        assert mult == 1

    def test_first_nonzero(self):
        """l=1: eigenvalue = 4/R^2, multiplicity = 5."""
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=1.0, l_max=1)
        ev, mult = spectrum[1]
        assert abs(ev - 4.0) < 1e-12
        assert mult == 5

    def test_second(self):
        """l=2: eigenvalue = 10/R^2, multiplicity = 14."""
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=1.0, l_max=2)
        ev, mult = spectrum[2]
        assert abs(ev - 10.0) < 1e-12
        assert mult == 14

    def test_third(self):
        """l=3: eigenvalue = 18/R^2, multiplicity = 30."""
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=1.0, l_max=3)
        ev, mult = spectrum[3]
        assert abs(ev - 18.0) < 1e-12
        assert mult == 30

    def test_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R = 3.0
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=R, l_max=5)
        for l, (ev, mult) in enumerate(spectrum):
            expected = l * (l + 3) / R**2
            assert abs(ev - expected) < 1e-12, \
                f"l={l}: eigenvalue {ev} != {expected}"

    def test_multiplicity_formula(self):
        """Verify (l+1)(l+2)(2l+3)/6 for l=0..10."""
        spectrum = S4SpectralGeometry.scalar_eigenvalues(R=1.0, l_max=10)
        for l, (ev, mult) in enumerate(spectrum):
            expected = (l + 1) * (l + 2) * (2 * l + 3) // 6
            assert mult == expected, \
                f"l={l}: multiplicity {mult} != {expected}"


# ======================================================================
# Coexact 1-form eigenvalues: (k+1)(k+2)/R^2, k >= 1
# ======================================================================
class TestCoexact1FormS4:
    """Delta_1 coexact on S^4: eigenvalue = (k+1)(k+2)/R^2."""

    def test_first_coexact_is_6(self):
        """First coexact eigenvalue is 6/R^2 (the mass gap)."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=1)
        ev, mult = spectrum[0]
        assert abs(ev - 6.0) < 1e-12

    def test_second_coexact(self):
        """k=2: eigenvalue = 12/R^2."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=2)
        ev, mult = spectrum[1]
        assert abs(ev - 12.0) < 1e-12

    def test_third_coexact(self):
        """k=3: eigenvalue = 20/R^2."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=3)
        ev, mult = spectrum[2]
        assert abs(ev - 20.0) < 1e-12

    def test_no_zero_eigenvalue(self):
        """All coexact eigenvalues are strictly positive."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=20)
        for ev, mult in spectrum:
            assert ev > 0

    def test_gap_bigger_than_s3(self):
        """S^4 coexact gap (6/R^2) > S^3 coexact gap (4/R^2)."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=1)
        s4_gap = spectrum[0][0]
        s3_gap = 4.0  # known S^3 result
        assert s4_gap > s3_gap

    def test_multiplicity_k1(self):
        """k=1: multiplicity = 10."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=1)
        ev, mult = spectrum[0]
        assert mult == 10

    def test_multiplicity_k2(self):
        """k=2: multiplicity = (2*2+3)(2+1)(2+2)/3 = 7*3*4/3 = 28."""
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=2)
        ev, mult = spectrum[1]
        assert mult == 28


# ======================================================================
# Exact 1-form eigenvalues
# ======================================================================
class TestExact1FormS4:
    """Exact 1-forms on S^4 share scalar eigenvalues (l >= 1)."""

    def test_first_exact(self):
        """l=1: eigenvalue = 4/R^2, multiplicity = 5."""
        spectrum = S4SpectralGeometry.exact_1form_eigenvalues(R=1.0, l_max=1)
        ev, mult = spectrum[0]
        assert abs(ev - 4.0) < 1e-12
        assert mult == 5

    def test_no_l0_mode(self):
        """Exact 1-forms start at l=1 (df=0 for constants)."""
        spectrum = S4SpectralGeometry.exact_1form_eigenvalues(R=1.0, l_max=5)
        # First entry should be l=1, not l=0
        ev_first = spectrum[0][0]
        assert ev_first > 0, "No zero eigenvalue for exact 1-forms"


# ======================================================================
# Weitzenboeck decomposition
# ======================================================================
class TestWeitzenboeckS4:
    """Weitzenboeck identity: Delta_1 = nabla*nabla + Ric on S^4."""

    def test_ricci_term(self):
        """Ricci term is 3/R^2."""
        w = S4SpectralGeometry.weitzenboeck_decomposition(R=1.0)
        assert abs(w['ricci_term'] - 3.0) < 1e-12

    def test_ricci_term_scaled(self):
        """Ricci term scales as 1/R^2."""
        R = 2.0
        w = S4SpectralGeometry.weitzenboeck_decomposition(R=R)
        assert abs(w['ricci_term'] - 3.0 / R**2) < 1e-12

    def test_gap_lower_bound(self):
        """Weitzenboeck lower bound for 1-form gap is 3/R^2."""
        w = S4SpectralGeometry.weitzenboeck_decomposition(R=1.0)
        assert abs(w['gap_lower_bound'] - 3.0) < 1e-12

    def test_actual_gap_is_6(self):
        """Actual coexact gap (6/R^2) exceeds Weitzenboeck bound (3/R^2)."""
        w = S4SpectralGeometry.weitzenboeck_decomposition(R=1.0)
        spectrum = S4SpectralGeometry.coexact_1form_eigenvalues(R=1.0, l_max=1)
        actual_gap = spectrum[0][0]
        assert actual_gap >= w['gap_lower_bound']
        assert abs(actual_gap - 6.0) < 1e-12


# ======================================================================
# Gap comparison S^3 vs S^4
# ======================================================================
class TestGapComparison:
    """S^4 coexact gap is 3/2 times S^3 coexact gap."""

    def test_ratio_is_three_halves(self):
        """Gap ratio S^4/S^3 = 6/4 = 3/2."""
        cmp = S4SpectralGeometry.gap_comparison_s3_vs_s4(R=1.0)
        assert abs(cmp['ratio'] - 1.5) < 1e-12

    def test_s4_bigger(self):
        """S^4 gap is strictly larger than S^3 gap."""
        cmp = S4SpectralGeometry.gap_comparison_s3_vs_s4(R=1.0)
        assert cmp['s4_gap'] > cmp['s3_gap']

    def test_enhancement_50_percent(self):
        """Enhancement is exactly 50%."""
        cmp = S4SpectralGeometry.gap_comparison_s3_vs_s4(R=1.0)
        assert abs(cmp['enhancement_percent'] - 50.0) < 1e-12

    def test_radius_independent_ratio(self):
        """Ratio 3/2 holds for any R."""
        for R in [0.5, 1.0, 2.2, 10.0]:
            cmp = S4SpectralGeometry.gap_comparison_s3_vs_s4(R=R)
            assert abs(cmp['ratio'] - 1.5) < 1e-12


# ======================================================================
# Betti numbers
# ======================================================================
class TestBettiNumbers:
    """Betti numbers of S^4."""

    def test_b0(self):
        """b_0 = 1 (connected)."""
        b = S4SpectralGeometry.betti_numbers()
        assert b[0] == 1

    def test_b1(self):
        """b_1 = 0 (no harmonic 1-forms = topological protection)."""
        b = S4SpectralGeometry.betti_numbers()
        assert b[1] == 0

    def test_b2(self):
        """b_2 = 0."""
        b = S4SpectralGeometry.betti_numbers()
        assert b[2] == 0

    def test_b3(self):
        """b_3 = 0."""
        b = S4SpectralGeometry.betti_numbers()
        assert b[3] == 0

    def test_b4(self):
        """b_4 = 1 (oriented)."""
        b = S4SpectralGeometry.betti_numbers()
        assert b[4] == 1

    def test_length(self):
        """5 Betti numbers for a 4-manifold."""
        b = S4SpectralGeometry.betti_numbers()
        assert len(b) == 5


# ======================================================================
# Ricci curvature
# ======================================================================
class TestRicciCurvatureS4:
    """Curvature data for S^4(R)."""

    def test_ricci_constant(self):
        """Ric = 3/R^2 on unit S^4."""
        c = S4SpectralGeometry.ricci_curvature(R=1.0)
        assert abs(c['ricci_constant'] - 3.0) < 1e-12

    def test_scalar_curvature(self):
        """Scalar curvature = 12/R^2."""
        c = S4SpectralGeometry.ricci_curvature(R=1.0)
        assert abs(c['scalar_curvature'] - 12.0) < 1e-12

    def test_sectional_curvature(self):
        """Sectional curvature = 1/R^2."""
        c = S4SpectralGeometry.ricci_curvature(R=1.0)
        assert abs(c['sectional'] - 1.0) < 1e-12

    def test_scaling(self):
        """All curvatures scale as 1/R^2."""
        R = 5.0
        c = S4SpectralGeometry.ricci_curvature(R=R)
        assert abs(c['ricci_constant'] - 3.0 / R**2) < 1e-12
        assert abs(c['scalar_curvature'] - 12.0 / R**2) < 1e-12
        assert abs(c['sectional'] - 1.0 / R**2) < 1e-12


# ======================================================================
# Physical mass gap
# ======================================================================
class TestPhysicalMass:
    """Physical mass gap from linearized YM on S^4."""

    def test_mass_at_physical_R(self):
        """At R = 2.2 fm, mass ~ 219.6 MeV."""
        R = 2.2
        result = S4SpectralGeometry.mass_gap_linearized(R)
        expected = np.sqrt(6.0) * 197.3269804 / R
        assert abs(result['mass_MeV'] - expected) < 0.1

    def test_mass_ratio_s4_vs_s3(self):
        """Mass ratio = sqrt(3/2) ~ 1.225."""
        result = S4SpectralGeometry.mass_gap_linearized(R=2.2)
        assert abs(result['ratio'] - np.sqrt(1.5)) < 1e-12

    def test_s4_mass_larger_than_s3(self):
        """S^4 mass gap is always larger than S^3."""
        result = S4SpectralGeometry.mass_gap_linearized(R=2.2)
        assert result['mass_MeV'] > result['s3_mass_MeV']

    def test_gap_squared(self):
        """gap_squared = 6/R^2."""
        R = 1.5
        result = S4SpectralGeometry.mass_gap_linearized(R)
        assert abs(result['gap_squared'] - 6.0 / R**2) < 1e-12
