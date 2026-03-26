"""
S4 Spectral Geometry — Complete spectral data for the round 4-sphere S^4(R).

Computes eigenvalues, multiplicities, and curvature data for the Hodge-de Rham
Laplacian on S^4, with emphasis on the coexact 1-form spectrum relevant to
Yang-Mills theory.

Key results:
  - Scalar eigenvalues: l(l+3)/R^2, multiplicity (l+1)(l+2)(2l+3)/6
  - Coexact 1-form eigenvalues: (k+1)(k+2)/R^2 for k >= 1
  - First coexact eigenvalue: 6/R^2 (50% larger than S^3's 4/R^2)
  - Ricci curvature: 3/R^2 (Einstein manifold)
  - Weitzenboeck: Delta_1 = nabla*nabla + 3/R^2

All results are standard differential geometry on the round sphere.
References: Berger-Gauduchon-Mazet (1971), Ikeda-Taniguchi (1978).
"""

import numpy as np
from math import comb


HBAR_C_MEV_FM = 197.3269804


class S4SpectralGeometry:
    """Complete spectral data for S^4(R)."""

    # ------------------------------------------------------------------
    # Scalar Laplacian Delta_0
    # ------------------------------------------------------------------
    @staticmethod
    def scalar_eigenvalues(R=1.0, l_max=20):
        """
        Eigenvalues of the scalar Laplacian Delta_0 on S^4(R).

        LABEL: THEOREM

        Formula:
            eigenvalue   = l(l+3) / R^2,  l = 0, 1, 2, ...
            multiplicity = (l+1)(l+2)(2l+3) / 6

        This is the standard result for spherical harmonics on S^n with n=4.
        The general formula eigenvalue = l(l+n-1)/R^2 gives l(l+3)/R^2.

        Verification:
            l=0: (0, 1)
            l=1: (4/R^2, 5)
            l=2: (10/R^2, 14)
            l=3: (18/R^2, 30)

        Parameters
        ----------
        R     : radius of S^4
        l_max : maximum angular momentum quantum number

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples
        """
        result = []
        for l in range(0, l_max + 1):
            ev = l * (l + 3) / R**2
            mult = (l + 1) * (l + 2) * (2 * l + 3) // 6
            result.append((ev, mult))
        return result

    # ------------------------------------------------------------------
    # Coexact 1-form Laplacian
    # ------------------------------------------------------------------
    @staticmethod
    def coexact_1form_eigenvalues(R=1.0, l_max=20):
        """
        Eigenvalues of the Hodge Laplacian on coexact 1-forms on S^4(R).

        LABEL: THEOREM

        Coexact 1-forms are divergence-free (delta omega = 0, omega != df).
        These are the physical (transverse) modes in Coulomb gauge.

        Formula (from S^n general: (l+p)(l+n-1-p)/R^2 with p=1, n=4):
            eigenvalue   = (k+1)(k+2) / R^2,  k = 1, 2, 3, ...
            multiplicity = (2k+3)(k+1)(k+2) / 3

        The multiplicity comes from SO(5) representation theory for
        traceless vector harmonics on S^4.

        Verification:
            k=1: (6/R^2, 10)   -- (2*1+3)(1+1)(1+2)/3 = 5*2*3/3 = 10
            k=2: (12/R^2, 28)  -- (2*2+3)(2+1)(2+2)/3 = 7*3*4/3 = 28
            k=3: (20/R^2, 60)  -- (2*3+3)(3+1)(3+2)/3 = 9*4*5/3 = 60

        KEY: First coexact eigenvalue is 6/R^2 -- 50% larger than S^3's 4/R^2.

        Parameters
        ----------
        R     : radius of S^4
        l_max : maximum quantum number (k ranges from 1 to l_max)

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples
        """
        result = []
        for k in range(1, l_max + 1):
            ev = (k + 1) * (k + 2) / R**2
            mult = (2 * k + 3) * (k + 1) * (k + 2) // 3
            result.append((ev, mult))
        return result

    # ------------------------------------------------------------------
    # Exact 1-form Laplacian
    # ------------------------------------------------------------------
    @staticmethod
    def exact_1form_eigenvalues(R=1.0, l_max=20):
        """
        Eigenvalues of the Hodge Laplacian on exact 1-forms on S^4(R).

        LABEL: THEOREM

        Exact 1-forms are df where f is a scalar eigenfunction.
        They inherit the scalar eigenvalues (excluding the constant l=0
        mode, since df=0 for constants).

        Formula:
            eigenvalue   = l(l+3) / R^2,  l = 1, 2, 3, ...
            multiplicity = (l+1)(l+2)(2l+3) / 6

        Parameters
        ----------
        R     : radius of S^4
        l_max : maximum quantum number

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples
        """
        result = []
        for l in range(1, l_max + 1):
            ev = l * (l + 3) / R**2
            mult = (l + 1) * (l + 2) * (2 * l + 3) // 6
            result.append((ev, mult))
        return result

    # ------------------------------------------------------------------
    # Betti numbers
    # ------------------------------------------------------------------
    @staticmethod
    def betti_numbers():
        """
        Betti numbers of S^4.

        LABEL: THEOREM

        b_k(S^4) = [1, 0, 0, 0, 1]

        b_1 = 0 implies H^1(S^4) = 0, so there are no harmonic 1-forms.
        This provides topological protection for the 1-form spectral gap:
        the kernel of Delta_1 is trivial.

        Returns
        -------
        list of int: [b_0, b_1, b_2, b_3, b_4]
        """
        return [1, 0, 0, 0, 1]

    # ------------------------------------------------------------------
    # Weitzenboeck decomposition
    # ------------------------------------------------------------------
    @staticmethod
    def weitzenboeck_decomposition(R=1.0):
        """
        Weitzenboeck decomposition for 1-forms on S^4(R).

        LABEL: THEOREM

        On any Riemannian manifold:
            Delta_1 = nabla*nabla + Ric

        On S^4(R), Ric = 3/R^2 * g (Einstein manifold with n-1=3), so:
            Delta_1 = nabla*nabla + 3/R^2

        Since nabla*nabla >= 0, this gives the Weitzenboeck lower bound:
            lambda_1(Delta_1) >= 3/R^2

        The actual coexact gap is 6/R^2, which exceeds this bound.

        Parameters
        ----------
        R : radius of S^4

        Returns
        -------
        dict with keys:
            connection_laplacian_formula : str
            ricci_term                  : float (3/R^2)
            total_formula               : str
            gap_lower_bound             : float (3/R^2)
        """
        ricci = 3.0 / R**2
        return {
            'connection_laplacian_formula': '\\nabla^*\\nabla',
            'ricci_term': ricci,
            'total_formula': '\\Delta_1 = \\nabla^*\\nabla + 3/R^2',
            'gap_lower_bound': ricci,
        }

    # ------------------------------------------------------------------
    # Gap comparison S^3 vs S^4
    # ------------------------------------------------------------------
    @staticmethod
    def gap_comparison_s3_vs_s4(R=1.0):
        """
        Compare coexact 1-form spectral gaps on S^3 and S^4.

        LABEL: THEOREM

        S^3 coexact gap: 4/R^2 (from (k+1)^2/R^2 at k=1)
        S^4 coexact gap: 6/R^2 (from (k+1)(k+2)/R^2 at k=1)
        Ratio: 6/4 = 3/2

        The 50% enhancement comes from the higher Ricci curvature in S^4:
        Ric(S^4) = 3/R^2 vs Ric(S^3) = 2/R^2.

        Parameters
        ----------
        R : radius (same for both spheres)

        Returns
        -------
        dict with keys:
            s3_gap             : float (4/R^2)
            s4_gap             : float (6/R^2)
            ratio              : float (1.5)
            enhancement_percent : float (50.0)
        """
        s3_gap = 4.0 / R**2
        s4_gap = 6.0 / R**2
        return {
            's3_gap': s3_gap,
            's4_gap': s4_gap,
            'ratio': s4_gap / s3_gap,
            'enhancement_percent': (s4_gap / s3_gap - 1.0) * 100.0,
        }

    # ------------------------------------------------------------------
    # Ricci curvature
    # ------------------------------------------------------------------
    @staticmethod
    def ricci_curvature(R=1.0):
        """
        Curvature data for S^4(R).

        LABEL: THEOREM

        S^4 is an Einstein manifold with constant sectional curvature 1/R^2.

        Ricci tensor:    Ric = (n-1)/R^2 * g = 3/R^2 * g
        Scalar curvature: S  = n(n-1)/R^2     = 12/R^2
        Sectional:        K  = 1/R^2

        Parameters
        ----------
        R : radius of S^4

        Returns
        -------
        dict with keys:
            ricci_constant    : float (3/R^2)
            scalar_curvature  : float (12/R^2)
            sectional         : float (1/R^2)
        """
        return {
            'ricci_constant': 3.0 / R**2,
            'scalar_curvature': 12.0 / R**2,
            'sectional': 1.0 / R**2,
        }

    # ------------------------------------------------------------------
    # Physical mass gap (linearized)
    # ------------------------------------------------------------------
    @staticmethod
    def mass_gap_linearized(R):
        """
        Physical mass gap from the linearized Yang-Mills operator on S^4(R).

        LABEL: THEOREM

        The coexact 1-form gap on S^4(R) is 6/R^2, so:
            m = sqrt(6/R^2) * hbar*c = sqrt(6) * 197.3269804 / R  [MeV]

        For comparison, on S^3(R):
            m_S3 = sqrt(4/R^2) * hbar*c = 2 * 197.3269804 / R  [MeV]

        Ratio: m_S4 / m_S3 = sqrt(6)/2 = sqrt(3/2) ~ 1.225

        At R = 2.2 fm:
            m_S4 ~ 219.6 MeV
            m_S3 ~ 179.4 MeV

        Parameters
        ----------
        R : radius in fm

        Returns
        -------
        dict with keys:
            gap_squared  : float (6/R^2)
            mass_MeV     : float
            s3_mass_MeV  : float
            ratio        : float (sqrt(3/2))
        """
        gap_sq = 6.0 / R**2
        mass = np.sqrt(gap_sq) * HBAR_C_MEV_FM
        s3_gap_sq = 4.0 / R**2
        s3_mass = np.sqrt(s3_gap_sq) * HBAR_C_MEV_FM
        return {
            'gap_squared': gap_sq,
            'mass_MeV': mass,
            's3_mass_MeV': s3_mass,
            'ratio': np.sqrt(3.0 / 2.0),
        }
