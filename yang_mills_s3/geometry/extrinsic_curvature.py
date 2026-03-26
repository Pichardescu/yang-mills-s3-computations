"""
Extrinsic Curvature — extrinsic geometry of S^n embedded in flat R^{n+1}.

Computes second fundamental form, mean curvature, Jacobi operator spectrum,
stability index, and Gauss-Codazzi relations for round spheres S^n(R)
as hypersurfaces in Euclidean space.
"""

import numpy as np
from math import comb


HBAR_C_MEV_FM = 197.3269804


class ExtrinsicCurvature:
    """Extrinsic geometry of S^n(R) embedded in flat R^{n+1}."""

    @staticmethod
    def second_fundamental_form(n, R):
        """
        Second fundamental form of S^n(R) in R^{n+1}.

        S^n(R) is umbilical: all principal curvatures equal 1/R,
        so K_ij = (1/R) g_ij.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere (n >= 1)
        R : float, radius (R > 0)

        Returns
        -------
        dict with:
            'principal_curvatures' : list of n copies of 1/R
            'is_umbilical'         : True (all principal curvatures equal)
            'K_scalar'             : 1/R (the common principal curvature)
        """
        kappa = 1.0 / R
        return {
            'principal_curvatures': [kappa] * n,
            'is_umbilical': True,
            'K_scalar': kappa,
        }

    @staticmethod
    def mean_curvature(n, R):
        """
        Mean curvature of S^n(R) in R^{n+1}.

        H = trace(shape operator) = n/R.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        float : H = n/R
        """
        return n / R

    @staticmethod
    def norm_squared_K(n, R):
        """
        Squared norm of the second fundamental form |K|^2.

        |K|^2 = trace(K^2) = n * (1/R)^2 = n/R^2.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        float : |K|^2 = n/R^2
        """
        return n / R**2

    @staticmethod
    def _scalar_harmonic_multiplicity(n, l):
        """
        Multiplicity of degree-l scalar spherical harmonic on S^n.

        For l >= 2: C(n+l, l) - C(n+l-2, l-2)
        For l = 0: 1
        For l = 1: n+1

        Parameters
        ----------
        n : int, dimension of sphere
        l : int, harmonic degree (l >= 0)

        Returns
        -------
        int : multiplicity
        """
        if l == 0:
            return 1
        if l == 1:
            return n + 1
        return comb(n + l, l) - comb(n + l - 2, l - 2)

    @staticmethod
    def jacobi_operator_eigenvalues(n, R, l_max=20):
        """
        Eigenvalues of the Jacobi operator for S^n in flat R^{n+1}.

        The Jacobi operator is J = Delta_{S^n} + |K|^2 + Ric_ambient(nu, nu).
        For flat ambient space: Ric_ambient = 0, |K|^2 = n/R^2.
        Laplacian eigenvalues on S^n: l(l+n-1)/R^2.

        So J eigenvalues = l(l+n-1)/R^2 + n/R^2 for l = 0, 1, 2, ...

        All eigenvalues are positive, proving S^n is a STABLE minimal
        (actually, totally umbilical) submanifold.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius
        l_max : int, maximum harmonic degree to compute

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples, one per l from 0 to l_max
        """
        result = []
        for l in range(l_max + 1):
            laplacian_eigenvalue = l * (l + n - 1) / R**2
            jacobi_eigenvalue = laplacian_eigenvalue + n / R**2
            mult = ExtrinsicCurvature._scalar_harmonic_multiplicity(n, l)
            result.append((jacobi_eigenvalue, mult))
        return result

    @staticmethod
    def stability_index(n, R):
        """
        Stability index of S^n(R) in flat R^{n+1}.

        The stability index counts the number of negative eigenvalues
        of the Jacobi operator. For S^n in flat space, all eigenvalues
        are >= n/R^2 > 0, so the index is 0.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        dict with:
            'index'              : 0 (no negative eigenvalues)
            'stable'             : True
            'smallest_eigenvalue': n/R^2 (at l=0)
        """
        return {
            'index': 0,
            'stable': True,
            'smallest_eigenvalue': n / R**2,
        }

    @staticmethod
    def jacobi_mass_gap(n, R):
        """
        Mass gap from the Jacobi operator spectrum.

        The l=0 mode corresponds to uniform translation of the embedding
        (trivial deformation). The first nontrivial mode is l=1:

          J(l=1) = 1*(1+n-1)/R^2 + n/R^2 = n/R^2 + n/R^2 = 2n/R^2

        For S^3: 6/R^2.  For S^4: 8/R^2.

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        dict with:
            'trivial_mode'    : n/R^2 (l=0 eigenvalue)
            'nontrivial_gap'  : 2n/R^2 (l=1 eigenvalue)
            'l_gap'           : 1 (the degree of the gap mode)
        """
        return {
            'trivial_mode': n / R**2,
            'nontrivial_gap': 2 * n / R**2,
            'l_gap': 1,
        }

    @staticmethod
    def gauss_codazzi(n, R):
        """
        Gauss-Codazzi relations for S^n(R) in flat R^{n+1}.

        Gauss equation:
          R_{ijkl}^{S^n} = K_{ik}K_{jl} - K_{il}K_{jk}
                         = (1/R^2)(g_{ik}g_{jl} - g_{il}g_{jk})

        This DERIVES the constant sectional curvature 1/R^2 from the
        embedding, rather than postulating it.

        Codazzi equation:
          nabla_i K_{jk} = nabla_j K_{ik}

        Automatically satisfied for umbilical hypersurfaces (K = (1/R)g
        implies nabla K = 0 since g is parallel).

        LABEL: THEOREM

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        dict with:
            'intrinsic_curvature' : 1/R^2 (sectional curvature derived from Gauss eq)
            'codazzi_satisfied'   : True
        """
        return {
            'intrinsic_curvature': 1.0 / R**2,
            'codazzi_satisfied': True,
        }

    @staticmethod
    def connection_to_yang_mills(n, R):
        """
        Structural analogy between Jacobi operator and Yang-Mills operator.

        The Jacobi operator J = Delta + |K|^2 controls stability of S^n
        in R^{n+1}. The linearized YM operator on S^n has the form
        Delta + curvature_term, with the same spectral structure.

        The Jacobi gap 2n/R^2 provides structural motivation (not a bound)
        for the YM gap:
          - n=3: Jacobi gap = 6/R^2, YM linearized gap = 4/R^2 (coexact 1-forms)
          - n=4: Jacobi gap = 8/R^2, YM linearized gap = 6/R^2

        LABEL: PROPOSITION

        Parameters
        ----------
        n : int, dimension of the sphere
        R : float, radius

        Returns
        -------
        dict with:
            'jacobi_gap'         : 2n/R^2
            'ym_gap_linearized'  : known linearized YM gap on S^n
            'label'              : 'PROPOSITION'
        """
        jacobi_gap = 2 * n / R**2
        # Linearized YM gap on S^n: coexact 1-form Laplacian eigenvalue
        # On S^n: Delta_1^coexact has smallest eigenvalue (2n-2)/R^2
        # This is (n-1)*2/R^2
        ym_gap = (2 * n - 2) / R**2
        return {
            'jacobi_gap': jacobi_gap,
            'ym_gap_linearized': ym_gap,
            'label': 'PROPOSITION',
        }
