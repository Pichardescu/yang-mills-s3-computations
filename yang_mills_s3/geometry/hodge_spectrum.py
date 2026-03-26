"""
Hodge Spectrum — Eigenvalues of Laplacians on compact manifolds.

Computes the spectrum of the Hodge-de Rham Laplacian Delta_p acting on
p-forms on the round n-sphere S^n of radius R.

Key mathematical facts:
  - Delta_0 on S^n: eigenvalue = l(l+n-1)/R^2,  l = 0,1,2,...
  - On S^3 specifically: eigenvalue = l(l+2)/R^2, multiplicity = (l+1)^2
  - Delta_1 on S^3 has TWO families of eigenvalues:
    (a) Exact 1-forms (df for scalar eigenfunction f):
        eigenvalue = l(l+2)/R^2 for l = 1, 2, 3, ...
        multiplicity = (l+1)^2
        Values on unit S^3: 3, 8, 15, 24, 35, ...
    (b) Coexact 1-forms (divergence-free, physical/transverse):
        eigenvalue = (k+1)^2/R^2 for k = 1, 2, 3, ...
        multiplicity = 2k(k+2)
        Values on unit S^3: 4, 9, 16, 25, 36, ...
  - The PHYSICAL mass gap (coexact modes in Coulomb gauge) is 4/R^2.
  - The overall first nonzero eigenvalue of Delta_1 is 3/R^2 (exact, l=1).

CORRECTION (2026-03-10): The previous code used the WRONG formula
(l(l+2)+2)/R^2 for all 1-form eigenvalues. This assumed nabla*nabla
on 1-forms has the same eigenvalues as Delta_0, then added Ric=2/R^2.
But nabla*nabla on 1-forms != Delta_0. The correct spectrum has two
branches: exact and coexact. The Weitzenbock identity Delta_1 = nabla*nabla + Ric
is correct, but nabla*nabla on 1-forms has eigenvalue 2/R^2 (not 3/R^2) for
left-invariant 1-forms on unit S^3.

Verification: On unit S^3, left-invariant 1-forms theta^a satisfy:
  d(theta^1) = -2 theta^2 ^ theta^3  (Maurer-Cartan)
  *d(theta^1) = -2 theta^1            (curl eigenvalue = -2)
  Delta_1(theta^1) = (*d)^2 theta^1 = 4 theta^1
  Weitzenbock check: Delta_1 = nabla*nabla + 2 => nabla*nabla = 2, not 3.
"""

import numpy as np
from math import comb


class HodgeSpectrum:
    """Spectra of Hodge Laplacians on round spheres S^n."""

    # ------------------------------------------------------------------
    # Scalar Laplacian Delta_0
    # ------------------------------------------------------------------
    @staticmethod
    def scalar_eigenvalues(n: int, R: float, l_max: int = 20):
        """
        Eigenvalues of the scalar Laplacian Delta_0 on S^n of radius R.

        Parameters
        ----------
        n     : dimension of the sphere (2 or 3 supported with exact multiplicities)
        R     : radius of the sphere
        l_max : maximum angular momentum quantum number

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples, sorted by eigenvalue.

        Formulae
        --------
        S^n general:
            eigenvalue   = l(l + n - 1) / R^2
            multiplicity = C(n+l, l) - C(n+l-2, l-2)
                         = (2l+n-1)(n+l-2)! / (l!(n-1)!)  for l >= 1
                         and 1 for l = 0

        Special cases:
            S^2: eigenvalue = l(l+1)/R^2, multiplicity = 2l+1
            S^3: eigenvalue = l(l+2)/R^2, multiplicity = (l+1)^2
        """
        result = []
        for l in range(0, l_max + 1):
            ev = l * (l + n - 1) / R**2
            mult = HodgeSpectrum._scalar_multiplicity(n, l)
            result.append((ev, mult))
        return result

    # ------------------------------------------------------------------
    # 1-form Laplacian Delta_1
    # ------------------------------------------------------------------
    @staticmethod
    def one_form_eigenvalues(n: int, R: float, l_max: int = 20,
                             mode: str = 'coexact'):
        """
        Eigenvalues of the 1-form Laplacian Delta_1 on S^n of radius R.

        On S^3 (n=3), Delta_1 has TWO branches:

        EXACT 1-forms (df for scalar eigenfunction f, pure gauge):
            eigenvalue = l(l+2)/R^2 for l = 1, 2, 3, ...
            multiplicity = (l+1)^2
            Values on unit S^3: 3, 8, 15, 24, ...

        COEXACT 1-forms (divergence-free, physical/transverse):
            eigenvalue = (k+1)^2/R^2 for k = 1, 2, 3, ...
            multiplicity = 2k(k+2)
            Values on unit S^3: 4, 9, 16, 25, ...

        Parameters
        ----------
        n     : dimension of sphere
        R     : radius
        l_max : max quantum number
        mode  : 'coexact' (physical), 'exact', or 'all'

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples, sorted by eigenvalue
        """
        if n == 3:
            return HodgeSpectrum._one_form_eigenvalues_s3(R, l_max, mode)
        elif n == 2:
            return HodgeSpectrum._one_form_eigenvalues_s2(R, l_max)
        else:
            return HodgeSpectrum._one_form_eigenvalues_general(n, R, l_max, mode)

    # ------------------------------------------------------------------
    # General p-form Laplacian
    # ------------------------------------------------------------------
    @staticmethod
    def p_form_eigenvalues(n: int, p: int, R: float, l_max: int = 20):
        """
        Eigenvalues of the p-form Laplacian Delta_p on S^n of radius R.

        Uses Hodge duality: Delta_p on S^n is isospectral to Delta_{n-p} on S^n
        (up to zero modes determined by Betti numbers).

        Parameters
        ----------
        n     : dimension of sphere
        p     : form degree (0 <= p <= n)
        R     : radius
        l_max : max angular momentum

        Returns
        -------
        list of (eigenvalue, multiplicity) tuples
        """
        if p < 0 or p > n:
            raise ValueError(f"Form degree p={p} out of range for S^{n}")

        # Use Hodge duality: if p > n/2, compute for n-p instead
        # The spectra are the same (Hodge star is an isometry on eigenspaces)
        p_eff = min(p, n - p)

        if p_eff == 0:
            return HodgeSpectrum.scalar_eigenvalues(n, R, l_max)
        elif p_eff == 1:
            # For the full p-form spectrum, return all eigenvalues
            return HodgeSpectrum.one_form_eigenvalues(n, R, l_max, mode='all')
        else:
            # General p-forms on S^n
            return HodgeSpectrum._general_p_form(n, p_eff, R, l_max)

    # ------------------------------------------------------------------
    # First nonzero eigenvalue (the spectral gap)
    # ------------------------------------------------------------------
    @staticmethod
    def first_nonzero_eigenvalue(n: int, p: int, R: float,
                                 mode: str = 'coexact'):
        """
        The spectral gap: first nonzero eigenvalue of Delta_p on S^n(R).

        For 1-forms on S^3:
            Exact gap   = 3/R^2  (l=1 exact eigenvalue)
            Coexact gap = 4/R^2  (k=1 coexact eigenvalue)

        The PHYSICAL (Yang-Mills) mass gap uses the COEXACT gap = 4/R^2,
        because exact 1-forms are pure gauge (df) and unphysical.

        Parameters
        ----------
        n    : dimension of sphere
        p    : form degree
        R    : radius
        mode : 'coexact' (default, physical), 'exact', or 'all'

        Returns
        -------
        float : the first nonzero eigenvalue
        """
        if p == 0:
            # Scalar: first nonzero is l=1
            return n / R**2  # l=1: 1*(1+n-1)/R^2 = n/R^2

        # For p-forms, get the spectrum and return the first entry
        if p == 1 and n == 3:
            if mode == 'coexact':
                return 4.0 / R**2  # (k+1)^2 with k=1
            elif mode == 'exact':
                return 3.0 / R**2  # l(l+2) with l=1
            else:  # 'all'
                return 3.0 / R**2  # smallest of exact and coexact

        spectrum = HodgeSpectrum.p_form_eigenvalues(n, p, R, l_max=5)
        if spectrum:
            return spectrum[0][0]

        raise ValueError(f"Could not compute gap for Delta_{p} on S^{n}")

    # ------------------------------------------------------------------
    # Betti numbers of S^n
    # ------------------------------------------------------------------
    @staticmethod
    def betti_numbers(n: int):
        """
        Betti numbers of S^n.

        b_k(S^n) = 1 if k = 0 or k = n, else 0.

        The fact that b_1(S^3) = 0 is crucial: it means there are NO
        harmonic 1-forms, so the 1-form Laplacian has a spectral gap.
        This gap is the geometric origin of the Yang-Mills mass gap.

        Parameters
        ----------
        n : dimension of sphere

        Returns
        -------
        list : [b_0, b_1, ..., b_n]
        """
        betti = [0] * (n + 1)
        betti[0] = 1
        betti[n] = 1
        return betti

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _scalar_multiplicity(n: int, l: int):
        """
        Multiplicity of eigenvalue l(l+n-1)/R^2 for Delta_0 on S^n.

        Formula: dim of degree-l spherical harmonics on S^n
            = C(n+l, l) - C(n+l-2, l-2)

        Special cases:
            S^2: 2l+1
            S^3: (l+1)^2
        """
        if l == 0:
            return 1

        if n == 2:
            return 2 * l + 1
        elif n == 3:
            return (l + 1) ** 2

        # General formula using binomial coefficients
        # Guard: comb(a, b) with b < 0 is 0 by convention
        if l < 2:
            return comb(n + l, l)
        return comb(n + l, l) - comb(n + l - 2, l - 2)

    @staticmethod
    def _one_form_eigenvalues_s3(R: float, l_max: int, mode: str = 'coexact'):
        """
        Delta_1 on S^3 of radius R.

        TWO families:

        EXACT 1-forms (pure gauge, df):
            eigenvalue = l(l+2)/R^2 for l = 1, 2, 3, ...
            multiplicity = (l+1)^2
            On unit S^3: 3, 8, 15, 24, 35, ...

        COEXACT 1-forms (physical, divergence-free):
            eigenvalue = (k+1)^2/R^2 for k = 1, 2, 3, ...
            multiplicity = 2k(k+2)
            On unit S^3: 4, 9, 16, 25, 36, ...

        The coexact eigenvalues come from the curl operator on S^3:
        the curl has eigenvalues +/-(k+1)/R for k=1,2,..., so
        Delta_1 on coexact forms = (curl)^2 = (k+1)^2/R^2.
        """
        result = []

        if mode == 'exact':
            for l in range(1, l_max + 1):
                ev = l * (l + 2) / R**2
                mult = (l + 1) ** 2
                result.append((ev, mult))
        elif mode == 'coexact':
            for k in range(1, l_max + 1):
                ev = (k + 1) ** 2 / R**2
                mult = 2 * k * (k + 2)
                result.append((ev, mult))
        else:  # 'all'
            # Combine both branches and sort
            for l in range(1, l_max + 1):
                ev = l * (l + 2) / R**2
                mult = (l + 1) ** 2
                result.append((ev, mult))
            for k in range(1, l_max + 1):
                ev = (k + 1) ** 2 / R**2
                mult = 2 * k * (k + 2)
                result.append((ev, mult))
            result.sort(key=lambda x: x[0])

        return result

    @staticmethod
    def _one_form_eigenvalues_s2(R: float, l_max: int):
        """
        Delta_1 on S^2 of radius R.

        On S^2, exact and coexact 1-forms have the SAME eigenvalues
        (Hodge duality maps 1-forms to 1-forms on S^2):
            eigenvalue = l(l+1)/R^2 for l = 1, 2, 3, ...
            multiplicity = 2*(2l+1)
        """
        result = []
        for l in range(1, l_max + 1):
            ev = l * (l + 1) / R**2
            mult = 2 * (2 * l + 1)
            result.append((ev, mult))
        return result

    @staticmethod
    def _one_form_eigenvalues_general(n: int, R: float, l_max: int,
                                      mode: str = 'coexact'):
        """
        General 1-form eigenvalues on S^n.

        Exact 1-forms: eigenvalue = l(l+n-1)/R^2 for l = 1, 2, ...
        (same as scalar eigenvalues, since exact 1-forms = df)

        Coexact 1-forms: eigenvalue depends on the curl spectrum of S^n.
        For S^n, the coexact eigenvalues of Delta_1 are:
            (l+n-1)^2 / ((n-1)*R^2) ... (complex formula for general n)

        For n=3, this simplifies to (k+1)^2/R^2 as above.
        For general n, we use the exact formula from representation theory.
        """
        result = []

        if mode == 'exact' or mode == 'all':
            for l in range(1, l_max + 1):
                ev = l * (l + n - 1) / R**2
                mult = HodgeSpectrum._scalar_multiplicity(n, l)
                result.append((ev, mult))

        if mode == 'coexact' or mode == 'all':
            for k in range(1, l_max + 1):
                # For general S^n, coexact 1-form eigenvalues:
                # lambda_k = (k + n - 2)*(k + 1) / R^2 for n >= 3
                # (This reduces to (k+1)^2/R^2 for n=3.)
                ev = (k + n - 2) * (k + 1) / R**2
                mult = HodgeSpectrum._one_form_multiplicity_general(n, k)
                result.append((ev, mult))

        if mode == 'all':
            result.sort(key=lambda x: x[0])

        return result

    @staticmethod
    def _one_form_multiplicity_general(n: int, l: int):
        """
        Multiplicity of coexact 1-form eigenvalue on S^n for quantum number l.

        For coexact 1-forms on S^n, the multiplicity at level l is the dimension
        of the SO(n+1) representation with highest weight (l, 1, 0, ..., 0).
        This equals the dimension of traceless symmetric-vector harmonics:

            mult = (n-1) * (2l+n-1) * C(l+n-2, l) / (n-1)
                 = (2l+n-1) * C(l+n-2, l)

        Special cases:
            S^2 (n=2): 2l+1
            S^3 (n=3): 2l(l+2)
            S^4 (n=4): (2l+3)(l+1)(l+2)/3
        """
        if n == 3:
            return 2 * l * (l + 2)
        elif n == 2:
            return 2 * l + 1
        elif n == 4:
            return (2 * l + 3) * (l + 1) * (l + 2) // 3
        # General formula for S^n coexact 1-form multiplicity
        # dim of traceless vector harmonics at level l on S^n
        return (2 * l + n - 1) * comb(l + n - 2, l)

    @staticmethod
    def _general_p_form(n: int, p: int, R: float, l_max: int):
        """
        General p-form eigenvalues on S^n using Weitzenbock.

        For p-forms on S^n, the Weitzenbock curvature term is p(n-p)/R^2.
        """
        result = []
        ricci_correction = p * (n - p)
        l_start = 1 if HodgeSpectrum.betti_numbers(n)[p] == 0 else 0

        for l in range(l_start, l_max + 1):
            ev = (l * (l + n - 1) + ricci_correction) / R**2
            if p == 0:
                mult = HodgeSpectrum._scalar_multiplicity(n, l)
            elif p == 1:
                mult = HodgeSpectrum._one_form_multiplicity_general(n, l)
            else:
                scalar_mult = HodgeSpectrum._scalar_multiplicity(n, l)
                mult = comb(n, p) * scalar_mult
            result.append((ev, mult))
        return result
