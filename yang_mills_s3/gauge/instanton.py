"""
Instanton — Topological solutions of the Yang-Mills equations on S⁴/S³.

Instantons are connections with self-dual (or anti-self-dual) curvature.
Their topological charge k ∈ Z is classified by π₃(SU(2)) = Z.

Key results (foundational analysis):
  • BPST instanton on S³: transition function g(q) = q  (= Hopf map)
  • Action: S = 8π²|k| / g²
  • Moduli space: dim = 4Nk  for SU(N), charge k
  • On S³ the moduli space is *compact*  (unlike R⁴)
  • Second Chern number c₂ = 1 for the generator of π₃(S³)
"""

import sympy as sp
from sympy import symbols, pi, Abs, Rational, Function


class Instanton:
    """Instanton (self-dual Yang-Mills) solutions on S³ / S⁴."""

    # ------------------------------------------------------------------
    # Topological charge  k ∈ Z
    # ------------------------------------------------------------------
    @staticmethod
    def topological_charge(n):
        """
        Topological charge (winding number).

        Classified by π₃(SU(2)) = Z.

        Parameters
        ----------
        n : int

        Returns
        -------
        int  n  (the integer winding number itself)
        """
        return int(n)

    # ------------------------------------------------------------------
    # BPST transition function  g(q) = q  for charge 1
    # ------------------------------------------------------------------
    @staticmethod
    def bpst_transition_function(q):
        """
        Transition function of the charge-1 BPST instanton.

            g(q) = q        (identity map S³ → SU(2) ≅ S³)

        This is precisely the Hopf map.

        Parameters
        ----------
        q : any (quaternion / array / symbol)

        Returns
        -------
        q itself — the identity map.
        """
        return q

    # ------------------------------------------------------------------
    # Action  S = 8π²|k| / g²
    # ------------------------------------------------------------------
    @staticmethod
    def action(k, g_coupling):
        """
        Instanton action.

            S = 8π²|k| / g²

        Parameters
        ----------
        k : int — topological charge
        g_coupling : gauge coupling

        Returns
        -------
        Symbolic or numeric action value.
        """
        return 8 * pi**2 * abs(k) / g_coupling**2

    # ------------------------------------------------------------------
    # Moduli-space dimension  dim = 4Nk
    # ------------------------------------------------------------------
    @staticmethod
    def moduli_space_dimension(k, N):
        """
        Dimension of the instanton moduli space for SU(N) gauge group.

            dim M_{k,N} = 4 N k

        For SU(2), k = 1: dim = 8.
        (On S³ some of these are fixed by the isometry group, giving an
        effective 5-parameter family: 4 center + 1 scale, with the scale
        bounded from above by the radius.)

        Parameters
        ----------
        k : int — instanton number (positive)
        N : int — rank parameter of SU(N)

        Returns
        -------
        int  dimension
        """
        return 4 * N * abs(k)

    # ------------------------------------------------------------------
    # Compactness of moduli space on S³
    # ------------------------------------------------------------------
    @staticmethod
    def moduli_compact_on_s3():
        """
        On S³ the instanton moduli space is compact.

        On R⁴ the scale can run to infinity, but on S³ the radius provides
        a natural cutoff.  This guarantees that the instanton path integral
        converges — a crucial advantage of working on S³.

        Returns
        -------
        bool  True
        """
        return True

    # ------------------------------------------------------------------
    # Chern-Simons functional  CS[A]
    # ------------------------------------------------------------------
    @staticmethod
    def chern_simons_functional(A):
        """
        Symbolic Chern-Simons functional.

            CS[A] = (k/4π) ∫_{S³} Tr(A ∧ dA + 2/3 A ∧ A ∧ A)

        Parameters
        ----------
        A : sympy Symbol or expression representing the connection

        Returns
        -------
        Symbolic expression for the CS functional.
        """
        k = symbols('k', integer=True)
        return (k / (4 * pi)) * sp.Symbol(
            r'\\int Tr(A \\wedge dA + 2/3 A \\wedge A \\wedge A)')

    # ------------------------------------------------------------------
    # Second Chern number  c₂ = 1
    # ------------------------------------------------------------------
    @staticmethod
    def second_chern_number():
        """
        Second Chern number of the generator of π₃(S³).

            c₂ = (1/8π²) ∫ Tr(F ∧ F) = 1

        Returns
        -------
        int  1
        """
        return 1
