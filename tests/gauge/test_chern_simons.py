"""
Tests for Chern-Simons theory on S³.

Verifies:
  • CS functional is symbolic
  • Level quantization (k ∈ Z)
  • Partition function matches Witten's exact formula for k = 1..5
  • Relation to 4D Yang-Mills instantons
"""

import pytest
import numpy as np
import sympy as sp
from yang_mills_s3.gauge.chern_simons import ChernSimons


class TestChernSimons:

    def setup_method(self):
        self.cs = ChernSimons()

    # ------------------------------------------------------------------
    # CS functional
    # ------------------------------------------------------------------
    def test_functional_returns_symbolic(self):
        f = self.cs.functional(level_k=1)
        assert isinstance(f, sp.Basic)

    def test_functional_scales_with_k(self):
        f1 = self.cs.functional(level_k=1)
        f3 = self.cs.functional(level_k=3)
        # f3 / f1 should simplify to 3
        ratio = sp.simplify(f3 / f1)
        assert ratio == 3

    # ------------------------------------------------------------------
    # Level quantization
    # ------------------------------------------------------------------
    def test_level_quantization(self):
        lq = self.cs.level_quantization()
        assert 'integer' in lq['statement']
        assert 'π₃' in lq['homotopy_group'] or 'pi_3' in lq['homotopy_group']

    # ------------------------------------------------------------------
    # Partition function  Z = sqrt(2/(k+2)) * sin(π/(k+2))
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_partition_function_witten(self, k):
        """
        Witten's exact result (1989):
            Z(S³, SU(2), k) = sqrt(2/(k+2)) * sin(π/(k+2))
        """
        expected = np.sqrt(2.0 / (k + 2)) * np.sin(np.pi / (k + 2))
        result = self.cs.partition_function_su2(k)
        assert abs(result - expected) < 1e-12, (
            f"k={k}: got {result}, expected {expected}"
        )

    def test_partition_function_k1_value(self):
        """k=1: Z = sqrt(2/3) * sin(π/3) = sqrt(2/3)*sqrt(3)/2 = 1/sqrt(2)."""
        result = self.cs.partition_function_su2(1)
        expected = np.sqrt(2.0 / 3.0) * np.sin(np.pi / 3.0)
        assert abs(result - expected) < 1e-12

    # ------------------------------------------------------------------
    # Relation to 4D Yang-Mills
    # ------------------------------------------------------------------
    def test_relation_to_yang_mills(self):
        rel = self.cs.relation_to_yang_mills()
        assert 'identity' in rel
        assert 'CS' in rel['identity']
        assert 'explanation' in rel
        assert 'consequence' in rel
