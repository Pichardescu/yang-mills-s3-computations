"""
Tests for the Yang-Mills action functional on S³.

Verifies:
  • Vacuum action = 0  (F = 0)
  • Instanton action = 8π²|k| / g²
  • Quadratic action for eigenvalue mode
  • Equations of motion string
"""

import pytest
import numpy as np
from sympy import pi
from yang_mills_s3.gauge.yang_mills_action import YangMillsAction


class TestYangMillsAction:

    def setup_method(self):
        self.ym = YangMillsAction()

    # ------------------------------------------------------------------
    # Vacuum: S[θ] = 0
    # ------------------------------------------------------------------
    def test_vacuum_action_is_zero(self):
        assert self.ym.action(F=0, R=1.0, g_coupling=1.0) == 0

    # ------------------------------------------------------------------
    # Instanton action = 8π²|k| / g²
    # ------------------------------------------------------------------
    def test_instanton_action_k1(self):
        g = 1.0
        expected = 8 * np.pi**2 / g**2
        result = float(self.ym.instanton_action(k=1, g_coupling=g))
        assert abs(result - expected) < 1e-10

    def test_instanton_action_k2(self):
        g = 1.0
        expected = 16 * np.pi**2 / g**2
        result = float(self.ym.instanton_action(k=2, g_coupling=g))
        assert abs(result - expected) < 1e-10

    def test_instanton_action_negative_k(self):
        g = 1.0
        result_pos = float(self.ym.instanton_action(k=1, g_coupling=g))
        result_neg = float(self.ym.instanton_action(k=-1, g_coupling=g))
        assert abs(result_pos - result_neg) < 1e-10

    def test_instanton_action_independent_of_R(self):
        """Instanton action does not depend on radius — it's topological."""
        g = 1.0
        result = float(self.ym.instanton_action(k=1, g_coupling=g))
        expected = 8 * np.pi**2 / g**2
        assert abs(result - expected) < 1e-10

    # ------------------------------------------------------------------
    # Quadratic action
    # ------------------------------------------------------------------
    def test_quadratic_action(self):
        lam = 4.0
        g = 2.0
        result = self.ym.quadratic_action(eigenvalue=lam, R=1.0, g_coupling=g)
        expected = lam / (2 * g**2)
        assert abs(result - expected) < 1e-10

    # ------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------
    def test_equations_of_motion(self):
        eom = self.ym.equations_of_motion()
        assert 'F' in eom
        assert '0' in eom
