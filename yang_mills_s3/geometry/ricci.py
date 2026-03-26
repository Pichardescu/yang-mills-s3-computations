"""
Ricci Tensor — Ricci curvature for symmetric spaces.

Computes Ricci tensors and Einstein constants for round spheres
and compact Lie groups with bi-invariant metrics.
"""

import re


class RicciTensor:
    """Ricci curvature computations for symmetric spaces."""

    @staticmethod
    def on_sphere(n, R):
        """
        Ricci tensor for S^n of radius R with the round metric.

        Ric = (n-1)/R^2 * g

        Parameters
        ----------
        n : int, dimension of the sphere (n >= 2)
        R : float or sympy expression, radius

        Returns
        -------
        dict with:
            'einstein_constant' : (n-1)/R^2
            'ricci_scalar'      : n*(n-1)/R^2
            'dimension'         : n
            'formula'           : string description
        """
        lam = (n - 1) / R**2
        scalar = n * (n - 1) / R**2
        return {
            'einstein_constant': lam,
            'ricci_scalar': scalar,
            'dimension': n,
            'formula': f'Ric = {n-1}/R^2 * g  (Einstein manifold)',
        }

    @staticmethod
    def on_lie_group(group_name, R):
        """
        Ricci tensor for compact Lie groups with bi-invariant metric
        scaled so the total space has "radius" R.

        Parameters
        ----------
        group_name : str, e.g. 'SU(2)', 'SU(3)', 'SU(N)'
        R          : float or sympy expression

        Returns
        -------
        dict with ricci_scalar, ricci_on_1forms (= einstein constant lambda),
        and dimension.

        Normalisation conventions:
            - SU(2) ~ S^3 of radius R  =>  Ric = 2/R^2 * g
            - SU(N):  dim = N^2-1,  Ric = N/(4*R^2) * g
              (bi-invariant metric normalised so sectional curvatures scale as 1/R^2)
        """
        name = group_name.upper().replace(' ', '')

        if name == 'SU(2)':
            dim = 3
            lam = 2 / R**2
            scalar = dim * lam
            return {
                'ricci_on_1forms': lam,
                'ricci_scalar': scalar,
                'dimension': dim,
            }

        m = re.match(r'SU\((\d+)\)', name)
        if m:
            N = int(m.group(1))
            dim = N**2 - 1
            lam = N / (4 * R**2)
            scalar = dim * lam
            return {
                'ricci_on_1forms': lam,
                'ricci_scalar': scalar,
                'dimension': dim,
            }

        raise ValueError(f"Unknown group: {group_name}")

    @staticmethod
    def einstein_constant(n, R):
        """
        Einstein constant lambda where Ric = lambda * g on S^n of radius R.

        Returns (n-1)/R^2.
        """
        return (n - 1) / R**2
