"""
Weitzenboeck Decomposition — Bochner-Weitzenboeck identities for
the Yang-Mills Laplacian on symmetric spaces.

Provides spectral gaps and mass gap predictions.

CORRECTION (2026-03-10): The 1-form gap on S^3 is 4/R^2 (coexact),
not 5/R^2. The mass gap is 2*hbar*c/R, not sqrt(5)*hbar*c/R.
See hodge_spectrum.py for the full corrected spectrum.
"""

import numpy as np


# Physical constant: hbar * c in MeV * fm
HBAR_C_MEV_FM = 197.3269804


class Weitzenboeck:
    """
    Weitzenboeck / Bochner decomposition for Laplacians on S^n
    and mass-gap predictions for Yang-Mills theory.
    """

    @staticmethod
    def decomposition(manifold='S3', R=1.0):
        """
        Weitzenboeck decomposition of the Yang-Mills Laplacian.

        For S^3 with a *flat* gauge connection:
            Delta_YM = nabla^* nabla  +  Ric
                     = nabla^* nabla  +  2/R^2

        Parameters
        ----------
        manifold : str ('S3' supported)
        R        : float, radius

        Returns
        -------
        dict describing the decomposition terms
        """
        if manifold.upper() in ('S3', 'S^3'):
            ric_value = 2 / R**2
            return {
                'manifold': 'S^3',
                'radius': R,
                'connection_laplacian': 'nabla^* nabla  (non-negative)',
                'ricci_term': ric_value,
                'curvature_endomorphism': 0,
                'formula': f'Delta_YM = nabla^* nabla + {ric_value}',
                'note': 'F=0 for flat gauge connection => [F,.] = 0',
            }
        raise ValueError(f"Manifold '{manifold}' not implemented")

    @staticmethod
    def spectral_gap_0forms(n, R, l=1):
        """
        Eigenvalue of the scalar (Hodge) Laplacian on S^n for mode l.

        lambda_l = l*(l + n - 1) / R^2

        For S^3 (n=3): lambda_l = l*(l+2)/R^2
        """
        return l * (l + n - 1) / R**2

    @staticmethod
    def spectral_gap_1forms_exact(n, R, l=1):
        """
        Eigenvalue of exact 1-forms (df) on S^n for mode l.

        These are the same as scalar eigenvalues:
            lambda_l = l*(l + n - 1) / R^2   for l >= 1

        For S^3: lambda_1 = 3/R^2
        """
        return l * (l + n - 1) / R**2

    @staticmethod
    def spectral_gap_1forms_coexact(n, R, k=1):
        """
        Eigenvalue of coexact (physical, divergence-free) 1-forms on S^n.

        For S^3 (n=3):
            lambda_k = (k+1)^2 / R^2  for k = 1, 2, 3, ...
            Lowest: lambda_1 = 4/R^2

        For general S^n:
            lambda_k = (k + n - 2)*(k + 1) / R^2

        These are the PHYSICAL modes relevant for Yang-Mills.
        """
        if n == 3:
            return (k + 1)**2 / R**2
        return (k + n - 2) * (k + 1) / R**2

    @staticmethod
    def spectral_gap_1forms(n, R, l=1):
        """
        Eigenvalue of the coexact (physical) 1-form Laplacian on S^n.

        For S^3:  lambda_k = (k+1)^2 / R^2  with k >= 1
        The lowest eigenvalue (k=1): 4/R^2

        This returns the COEXACT (physical) eigenvalue, which is
        the relevant one for the Yang-Mills mass gap.
        """
        return Weitzenboeck.spectral_gap_1forms_coexact(n, R, k=l)

    @staticmethod
    def mass_gap_yang_mills(R):
        """
        Mass gap prediction from the spectral gap of the coexact
        1-form Laplacian on S^3 of radius R.

        mass_gap = 2 * hbar * c / R

        (Since the coexact gap eigenvalue = 4/R^2, mass = sqrt(4/R^2) = 2/R.)

        Parameters
        ----------
        R : float, radius in fm

        Returns
        -------
        float : mass gap in MeV
        """
        return 2.0 * HBAR_C_MEV_FM / R

    @staticmethod
    def spectrum_0forms(n, R, l_max=10):
        """List of eigenvalues of Delta_0 on S^n for l = 0, 1, ..., l_max."""
        return [l * (l + n - 1) / R**2 for l in range(l_max + 1)]

    @staticmethod
    def spectrum_1forms(n, R, l_max=10):
        """List of coexact 1-form eigenvalues on S^n for k = 1, 2, ..., l_max."""
        if n == 3:
            return [(k + 1)**2 / R**2 for k in range(1, l_max + 1)]
        return [(k + n - 2) * (k + 1) / R**2 for k in range(1, l_max + 1)]
