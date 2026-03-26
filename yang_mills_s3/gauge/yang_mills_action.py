"""
Yang-Mills Action — The gauge-theory action functional on S³.

    S[A] = -(1 / 2g²) ∫_{S³} Tr(F ∧ ★F)

For the vacuum (F = 0) the action vanishes.  For a perturbation A = θ + a
the quadratic part is  S₂ ≈ (1/2g²) ∫ Tr(D_θ a ∧ ★D_θ a), whose spectrum
is controlled by the Laplacian eigenvalues on S³.

Key formulae (foundational analysis):
  • S_vacuum = 0
  • S_instanton(k) = 8π²|k| / g²   (topological, independent of R)
  • Equations of motion: D★F = 0
"""

import sympy as sp
from sympy import symbols, pi, Abs, Rational


class YangMillsAction:
    """Yang-Mills action functional on S³."""

    # ------------------------------------------------------------------
    # Full action  S[A] = -(1/2g²) ∫ Tr(F ∧ ★F)
    # ------------------------------------------------------------------
    @staticmethod
    def action(F, R, g_coupling):
        """
        Evaluate the Yang-Mills action for a given field strength.

        For F = 0 (vacuum):  S = 0.
        For a perturbation mode with field-strength eigenvalue λ:
            S ≈ λ / (2g²)

        Parameters
        ----------
        F : numeric or sympy expression
            Effective field-strength measure.  F = 0 for the vacuum.
        R : radius of S³  (not used when F = 0)
        g_coupling : gauge coupling constant

        Returns
        -------
        Action value (numeric or symbolic).
        """
        if F == 0:
            return 0
        # For a perturbation mode, F encodes the eigenvalue contribution
        return F / (2 * g_coupling**2)

    # ------------------------------------------------------------------
    # Quadratic action for a single Laplacian mode
    # ------------------------------------------------------------------
    @staticmethod
    def quadratic_action(eigenvalue, R, g_coupling):
        """
        Action contribution of a single Laplacian eigenmode on S³(R).

            S_mode = eigenvalue / (2g²)

        Parameters
        ----------
        eigenvalue : Laplacian eigenvalue (numeric or symbolic)
        R : radius  (information only — eigenvalue already encodes geometry)
        g_coupling : gauge coupling constant

        Returns
        -------
        Quadratic action for this mode.
        """
        return eigenvalue / (2 * g_coupling**2)

    # ------------------------------------------------------------------
    # Instanton action  S = 8π²|k| / g²
    # ------------------------------------------------------------------
    @staticmethod
    def instanton_action(k, g_coupling):
        """
        Action of a (anti-)instanton with topological charge k.

            S_inst = 8π²|k| / g²

        This is exact, topological, and *independent* of R.

        Parameters
        ----------
        k : int — topological charge (winding number)
        g_coupling : gauge coupling constant

        Returns
        -------
        Instanton action (symbolic or numeric).
        """
        return 8 * pi**2 * abs(k) / g_coupling**2

    # ------------------------------------------------------------------
    # Equations of motion  D★F = 0
    # ------------------------------------------------------------------
    @staticmethod
    def equations_of_motion():
        """
        Symbolic Yang-Mills equations of motion.

            D★F = 0     (covariant divergence of the dual field strength)

        Returns
        -------
        str  LaTeX-style representation of the equation.
        """
        return "D★F = 0"
