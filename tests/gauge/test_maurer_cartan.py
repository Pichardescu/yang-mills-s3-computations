"""
Tests for Maurer-Cartan form on SU(2) ≅ S³.

Verifies:
  • MC form components are correctly defined
  • Curvature = 0  (flat connection)
  • MC form is the YM vacuum (action = 0)
"""

import pytest
from yang_mills_s3.gauge.maurer_cartan import MaurerCartan


class TestMaurerCartan:

    def setup_method(self):
        self.mc = MaurerCartan()

    # ------------------------------------------------------------------
    # Form components
    # ------------------------------------------------------------------
    def test_form_su2_returns_three_components(self):
        form = self.mc.form_su2()
        assert 'theta1' in form
        assert 'theta2' in form
        assert 'theta3' in form

    def test_form_su2_components_have_four_coefficients(self):
        form = self.mc.form_su2()
        for key in ('theta1', 'theta2', 'theta3'):
            coeffs = form[key]
            assert set(coeffs.keys()) == {'dw', 'dx', 'dy', 'dz'}

    # ------------------------------------------------------------------
    # Flatness: dθ + θ∧θ = 0
    # ------------------------------------------------------------------
    def test_is_flat(self):
        assert self.mc.is_flat() is True

    # ------------------------------------------------------------------
    # Curvature = 0
    # ------------------------------------------------------------------
    def test_curvature_is_zero(self):
        assert self.mc.curvature() == 0

    # ------------------------------------------------------------------
    # MC is the YM vacuum
    # ------------------------------------------------------------------
    def test_verify_vacuum_action_zero(self):
        vac = self.mc.verify_vacuum()
        assert vac['action'] == 0

    def test_verify_vacuum_field_strength_zero(self):
        vac = self.mc.verify_vacuum()
        assert vac['field_strength'] == 0

    def test_verify_vacuum_gauge_invariant(self):
        vac = self.mc.verify_vacuum()
        assert vac['is_gauge_invariant'] is True

    def test_verify_vacuum_maximally_symmetric(self):
        vac = self.mc.verify_vacuum()
        assert vac['maximally_symmetric'] is True
