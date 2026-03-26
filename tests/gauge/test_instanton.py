"""
Tests for Instanton module.

Verifies:
  • Topological charge = integer
  • BPST transition function = identity (Hopf map)
  • Action = 8π²|k| / g²
  • c₂ = 1
  • Moduli space compact on S³
  • Moduli space dimension = 4Nk
"""

import pytest
import numpy as np
from yang_mills_s3.gauge.instanton import Instanton


class TestInstanton:

    def setup_method(self):
        self.inst = Instanton()

    # ------------------------------------------------------------------
    # Topological charge
    # ------------------------------------------------------------------
    def test_topological_charge(self):
        assert self.inst.topological_charge(1) == 1
        assert self.inst.topological_charge(-2) == -2
        assert self.inst.topological_charge(0) == 0

    # ------------------------------------------------------------------
    # BPST transition function = identity (= Hopf map)
    # ------------------------------------------------------------------
    def test_bpst_transition_is_identity(self):
        """g(q) = q  for the charge-1 BPST instanton."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = self.inst.bpst_transition_function(q)
        np.testing.assert_array_equal(result, q)

    def test_bpst_transition_arbitrary_quaternion(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = self.inst.bpst_transition_function(q)
        np.testing.assert_array_equal(result, q)

    # ------------------------------------------------------------------
    # Action = 8π²|k| / g²  for k = 1
    # ------------------------------------------------------------------
    def test_instanton_action_k1(self):
        g = 1.0
        expected = 8 * np.pi**2 / g**2
        result = float(self.inst.action(k=1, g_coupling=g))
        assert abs(result - expected) < 1e-10

    # ------------------------------------------------------------------
    # Second Chern number c₂ = 1
    # ------------------------------------------------------------------
    def test_second_chern_number(self):
        assert self.inst.second_chern_number() == 1

    # ------------------------------------------------------------------
    # Moduli space compact on S³
    # ------------------------------------------------------------------
    def test_moduli_compact_on_s3(self):
        assert self.inst.moduli_compact_on_s3() is True

    # ------------------------------------------------------------------
    # Moduli space dimension = 4Nk
    # ------------------------------------------------------------------
    def test_moduli_space_dimension_su2_k1(self):
        """SU(2), k=1: dim = 4 × 2 × 1 = 8."""
        assert self.inst.moduli_space_dimension(k=1, N=2) == 8

    def test_moduli_space_dimension_su3_k2(self):
        """SU(3), k=2: dim = 4 × 3 × 2 = 24."""
        assert self.inst.moduli_space_dimension(k=2, N=3) == 24

    # ------------------------------------------------------------------
    # Chern-Simons functional returns symbolic expression
    # ------------------------------------------------------------------
    def test_chern_simons_functional_symbolic(self):
        import sympy as sp
        A = sp.Symbol('A')
        cs = self.inst.chern_simons_functional(A)
        # Should be a sympy expression (not zero, not None)
        assert cs is not None
        assert isinstance(cs, sp.Basic)
