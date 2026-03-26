"""
Chern-Simons Theory — Topological QFT on S³.

The Chern-Simons action at level k:

    S_CS = (k / 4π) ∫_{S³} Tr(A ∧ dA + 2/3 A ∧ A ∧ A)

Key results (foundational analysis):
  • k must be an integer  (topological quantization via π₃(SU(2)) = Z)
  • Partition function Z(S³, SU(2), k) = √(2/(k+2)) · sin(π/(k+2))
    — Witten's exact result (1989)
  • 4D YM instanton number = CS(A₊) - CS(A₋) on the boundary S³
"""

import numpy as np
import sympy as sp
from sympy import symbols, pi, sqrt, sin, Rational


class ChernSimons:
    """Chern-Simons theory on S³ with gauge group SU(2)."""

    # ------------------------------------------------------------------
    # CS functional at level k
    # ------------------------------------------------------------------
    @staticmethod
    def functional(level_k):
        """
        Symbolic Chern-Simons action at level k.

            S_CS = (k / 4π) ∫_{S³} Tr(A ∧ dA + 2/3 A ∧ A ∧ A)

        Parameters
        ----------
        level_k : int or sympy expression — the level

        Returns
        -------
        Symbolic expression (level_k / 4π) × ∫ Tr(…)
        """
        A = sp.Symbol('A')
        integral = sp.Symbol(
            r'\\int Tr(A \\wedge dA + 2/3 A \\wedge A \\wedge A)')
        return (level_k / (4 * pi)) * integral

    # ------------------------------------------------------------------
    # Level quantization
    # ------------------------------------------------------------------
    @staticmethod
    def level_quantization():
        """
        Explain why the level k must be an integer.

        Returns
        -------
        dict with explanation and mathematical reason
        """
        return {
            'statement': 'k must be an integer',
            'reason': (
                'Under a gauge transformation that wraps around the '
                'non-trivial cycle of π₃(SU(2)) = Z, the CS action '
                'shifts by 2πk.  Single-valuedness of exp(iS_CS) '
                'requires k ∈ Z.'
            ),
            'homotopy_group': 'π₃(SU(2)) = Z',
        }

    # ------------------------------------------------------------------
    # Partition function  Z(S³, SU(2), k)
    # ------------------------------------------------------------------
    @staticmethod
    def partition_function_su2(k):
        """
        Exact partition function of SU(2) Chern-Simons theory on S³.

        Witten (1989):

            Z(S³, SU(2), k) = √(2 / (k+2)) · sin(π / (k+2))

        Parameters
        ----------
        k : int — Chern-Simons level  (k ≥ 1)

        Returns
        -------
        float  partition function value (numerical)
        """
        return np.sqrt(2.0 / (k + 2)) * np.sin(np.pi / (k + 2))

    # ------------------------------------------------------------------
    # Relation to 4D Yang-Mills
    # ------------------------------------------------------------------
    @staticmethod
    def relation_to_yang_mills():
        """
        Explain how 3D Chern-Simons relates to 4D Yang-Mills instantons.

        The key identity is:
            (1/8π²) ∫_{S⁴} Tr(F ∧ F) = CS(A₊) - CS(A₋)

        where A₊ and A₋ are the gauge-field restrictions to the two
        S³ hemispheres of the boundary.

        Returns
        -------
        dict with the relation and its physical content
        """
        return {
            'identity': (
                '(1/8π²) ∫_{S⁴} Tr(F∧F) = CS(A₊) - CS(A₋)'
            ),
            'explanation': (
                'The instanton number in 4D equals the difference of '
                'Chern-Simons invariants evaluated on the two S³ '
                'boundaries obtained by cutting S⁴ along the equator.'
            ),
            'consequence': (
                'Instantons interpolate between CS vacua. The vacuum '
                'structure of YM theory on S³ is labelled by CS values.'
            ),
        }
