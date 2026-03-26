"""
Poincare Homology Sphere S^3/I* — Corrected spectrum with coexact/exact split.

The Poincare homology sphere is S^3 quotiented by the binary icosahedral group
I* (order 120). It is the unique 3-manifold that is a homology sphere (H_1 = 0)
but not simply connected (pi_1 = I*).

KEY MATHEMATICAL FACTS:
  - H^1(S^3/I*) = 0  ->  no harmonic 1-forms  ->  mass gap persists
  - The spectrum on S^3/I* is the I*-invariant subspace of the S^3 spectrum
  - Since only I*-invariant modes survive, the spectrum is SPARSER
  - The gap may be LARGER (if the lowest S^3 modes don't survive)

SPECTRUM DECOMPOSITION (CORRECTED, Session 2):

  Scalars (Delta_0):
    Eigenvalue l(l+2)/R^2 on S^3 with multiplicity (l+1)^2
    On S^3/I*: only levels with I*-invariant in V_l survive
    I*-invariant levels: l = 0, 12, 20, 24, 30, 32, 36, 40, ...
    Gap: l=12, eigenvalue 168/R^2 (vs 3/R^2 on S^3) -> 56x enhancement

  Coexact 1-forms (physical, divergence-free):
    Eigenvalue (k+1)^2/R^2 on S^3 with multiplicity 2k(k+2)
    The eigenspace at level k decomposes under SO(4) = SU(2)_L x SU(2)_R:
      Self-dual (+ curl):    V_{k+1} under SU(2)_L, V_{k-1} under SU(2)_R
      Anti-self-dual (- curl): V_{k-1} under SU(2)_L, V_{k+1} under SU(2)_R

    I* acts on SU(2)_R (right multiplication on S^3 = SU(2), standard Poincare quotient).
    I*-invariant coexact modes at level k:
      Self-dual:     m(k-1) * (k+2)   modes  [m(k-1) invariants in V_{k-1} under SU(2)_R, times dim V_{k+1}]
      Anti-self-dual: m(k+1) * k      modes  [m(k+1) invariants in V_{k+1} under SU(2)_R, times dim V_{k-1}]
      Total: m(k-1)*(k+2) + m(k+1)*k

    At k=1: m(0)*3 + m(2)*1 = 3 + 0 = 3 modes (the right-invariant forms survive the right-action quotient!)
    At k=2: m(3)*2 + m(1)*4 = 0 + 0 = 0 (no modes)
    Next surviving: k=11, eigenvalue 144/R^2 -> 36x gap enhancement for 2nd mode

  Exact 1-forms (pure gauge, df):
    Eigenvalue l(l+2)/R^2 for l >= 1, multiplicity (l+1)^2
    The exact eigenspace at level l is d(V_l) which transforms as V_l under SU(2)_L
    I*-invariant exact modes at level l: m(l) * (l+1) modes
    First nonzero: l=12, eigenvalue 168/R^2

THEOREM (COEXACT GAP PRESERVATION):
  The lowest coexact eigenvalue 4/R^2 (k=1) survives on S^3/I* with
  geometric multiplicity 3 (from the self-dual right-invariant forms;
  I* acts on SU(2)_R trivially on V_0). The Yang-Mills mass gap on
  S^3/I* equals that on S^3. For g-valued forms, physical multiplicity
  = 3 * dim(g).

THEOREM (SPECTRAL SPARSIFICATION):
  The second coexact eigenvalue on S^3/I* is at k=11 (eigenvalue 144/R^2),
  compared to k=2 (eigenvalue 9/R^2) on S^3. The ratio is 16x.

References:
  - Luminet, Weeks, Riazuelo, Lehoucq, Uzan, Nature 425, 593 (2003)
  - Ikeda & Taniguchi (1978): Spectra on spherical space forms
  - Molien (1897): Invariant theory of finite groups
  - McKay (1980): Graphs, singularities, and finite groups (I* <-> E8)
"""

import numpy as np
from typing import Optional


# Physical constants
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm


class PoincareHomology:
    """
    Spectrum of Laplacian operators on S^3/I* (Poincare homology sphere).

    Uses the CORRECTED coexact/exact split from Session 2.
    The binary icosahedral group I* has order 120 and 9 conjugacy classes.
    McKay correspondence: I* <-> E8 in the ADE classification.
    """

    def __init__(self):
        """
        Set up the conjugacy classes of I* in SU(2).

        Each element g in SU(2) has eigenvalues e^{i*theta/2}, e^{-i*theta/2}.

        Conjugacy classes (size, rotation_angle_theta):
        1. Identity:          1 element,  theta = 0
        2. Central (-I):      1 element,  theta = 2*pi
        3. Order 10 (type A): 12 elements, theta = 2*pi/5
        4. Order 10 (type B): 12 elements, theta = 4*pi/5
        5. Order 10 (type C): 12 elements, theta = 6*pi/5
        6. Order 10 (type D): 12 elements, theta = 8*pi/5
        7. Order 6 (type A):  20 elements, theta = 2*pi/3
        8. Order 6 (type B):  20 elements, theta = 4*pi/3
        9. Order 4:           30 elements, theta = pi

        Total: 1+1+12+12+12+12+20+20+30 = 120
        """
        self.conjugacy_classes = [
            (1,  0.0),                      # Identity
            (1,  2 * np.pi),                # -Identity
            (12, 2 * np.pi / 5),            # Order 10, angle 2*pi/5
            (12, 4 * np.pi / 5),            # Order 10, angle 4*pi/5
            (12, 6 * np.pi / 5),            # Order 10, angle 6*pi/5
            (12, 8 * np.pi / 5),            # Order 10, angle 8*pi/5
            (20, 2 * np.pi / 3),            # Order 6, angle 2*pi/3
            (20, 4 * np.pi / 3),            # Order 6, angle 4*pi/3
            (30, np.pi),                    # Order 4, angle pi
        ]
        self.group_order = 120

        # Verify total count
        total = sum(size for size, _ in self.conjugacy_classes)
        assert total == self.group_order, \
            f"Conjugacy class sizes sum to {total}, expected {self.group_order}"

    # ==================================================================
    # SU(2) character theory
    # ==================================================================

    def character_su2(self, l: int, theta: float) -> float:
        """
        Character of SU(2) irrep V_l at rotation angle theta.

        chi_l(theta) = sin((l+1)*theta/2) / sin(theta/2)

        This is the trace of the (l+1)-dimensional irreducible
        representation of SU(2) evaluated at diag(e^{i*theta/2}, e^{-i*theta/2}).

        Special cases:
          theta = 0:    chi_l = l + 1
          theta = 2*pi: chi_l = (-1)^l * (l + 1)
          theta = pi:   chi_l = sin((l+1)*pi/2)

        Parameters
        ----------
        l     : non-negative integer, labels the (l+1)-dim irrep
        theta : rotation angle in [0, 2*pi]

        Returns
        -------
        float : character value chi_l(theta)
        """
        half = theta / 2.0
        sin_denom = np.sin(half)

        if abs(sin_denom) < 1e-14:
            # L'Hopital at theta = 0 or 2*pi
            cos_num = np.cos((l + 1) * half)
            cos_denom = np.cos(half)
            if abs(cos_denom) < 1e-14:
                raise ValueError(f"Degenerate character at l={l}, theta={theta}")
            return (l + 1) * cos_num / cos_denom

        return np.sin((l + 1) * half) / sin_denom

    # ==================================================================
    # I*-invariant multiplicities
    # ==================================================================

    def trivial_multiplicity(self, l: int) -> int:
        """
        Number of I*-invariant vectors in V_l (the (l+1)-dim irrep of SU(2)).

        m(l) = (1/|I*|) * sum_{classes C} |C| * chi_l(theta_C)

        This is the multiplicity of the trivial rep of I* in V_l|_{I*}.

        NUMERICAL: Verified against the Molien series
            M(t) = (1 - t^60) / ((1 - t^12)(1 - t^20)(1 - t^30))

        First few nonzero values: m(0)=1, m(12)=1, m(20)=1, m(24)=1,
        m(30)=1, m(32)=1, m(36)=1, m(40)=1, m(42)=1, m(44)=1, ...

        Parameters
        ----------
        l : non-negative integer

        Returns
        -------
        int : number of I*-invariant vectors (>= 0)
        """
        if l < 0:
            return 0

        total = 0.0
        for size, theta in self.conjugacy_classes:
            chi = self.character_su2(l, theta)
            total += size * chi

        m = total / self.group_order
        m_int = int(round(m))

        if m_int < 0:
            raise ValueError(
                f"Negative multiplicity m({l}) = {m_int} (raw: {m:.6f})"
            )
        if abs(m - m_int) > 1e-6:
            raise ValueError(
                f"Non-integer multiplicity m({l}) = {m:.10f}"
            )
        return m_int

    def molien_series_coefficients(self, l_max: int = 60) -> list[int]:
        """
        Compute Molien series coefficients from the closed-form generating function.

        M(t) = (1 - t^60) / ((1 - t^12)(1 - t^20)(1 - t^30))

        The degrees 12, 20, 30 are the Klein invariant polynomial degrees.
        The t^60 correction is the syzygy relation.

        Returns
        -------
        list : m_molien[l] for l = 0, ..., l_max
        """
        molien = [0] * (l_max + 1)
        for l in range(l_max + 1):
            rhs = 0
            if l == 0:
                rhs = 1
            if l == 60:
                rhs = -1

            val = rhs
            if l >= 12:
                val += molien[l - 12]
            if l >= 20:
                val += molien[l - 20]
            if l >= 30:
                val += molien[l - 30]
            if l >= 32:
                val -= molien[l - 32]
            if l >= 42:
                val -= molien[l - 42]
            if l >= 50:
                val -= molien[l - 50]
            if l >= 62:
                val += molien[l - 62]

            molien[l] = val
        return molien

    def verify_against_molien(self, l_max: int = 60) -> tuple[bool, list]:
        """
        Verify character computation against the closed-form Molien series.

        Returns
        -------
        tuple : (all_match: bool, mismatches: list of (l, m_char, m_molien))
        """
        molien = self.molien_series_coefficients(l_max)
        mismatches = []
        for l in range(l_max + 1):
            m_char = self.trivial_multiplicity(l)
            m_mol = molien[l]
            if m_char != m_mol:
                mismatches.append((l, m_char, m_mol))
        return (len(mismatches) == 0, mismatches)

    # ==================================================================
    # Scalar spectrum on S^3/I*
    # ==================================================================

    def invariant_levels_scalar(self, l_max: int = 60) -> list[tuple[int, int]]:
        """
        Angular momentum levels l (0 to l_max) with I*-invariant scalars.

        The first nontrivial I*-invariant scalar is at l=12.
        Scalar gap on S^3/I*: 168/R^2  (vs 3/R^2 on S^3) -> 56x enhancement.

        Returns
        -------
        list of (l, multiplicity) for levels with m(l) > 0
        """
        result = []
        for l in range(l_max + 1):
            m = self.trivial_multiplicity(l)
            if m > 0:
                result.append((l, m))
        return result

    def scalar_spectrum(self, l_max: int = 60, R: float = 1.0) -> list[tuple[float, int]]:
        """
        Scalar eigenvalues of Delta_0 on S^3/I*.

        Eigenvalue at level l: l(l+2)/R^2 (same local formula as S^3).
        Multiplicity: m(l) = dim(V_l^{I*}).

        Returns
        -------
        list of (eigenvalue, multiplicity) for surviving modes
        """
        invariant = self.invariant_levels_scalar(l_max)
        return [(l * (l + 2) / R**2, mult) for l, mult in invariant]

    # ==================================================================
    # Coexact 1-form spectrum on S^3/I* (CORRECTED)
    # ==================================================================

    def coexact_invariant_multiplicity(self, k: int) -> int:
        """
        Number of I*-invariant coexact 1-forms at spectral level k.

        THEOREM: At level k, the coexact 1-forms on S^3 have eigenvalue
        (k+1)^2/R^2 and transform under SO(4) = SU(2)_L x SU(2)_R as:

          Self-dual (+ curl):     V_{k+1} under SU(2)_L, V_{k-1} under SU(2)_R
          Anti-self-dual (- curl): V_{k-1} under SU(2)_L, V_{k+1} under SU(2)_R

        I* acts on SU(2)_R (right multiplication g -> gh for h in I*,
        standard Poincare homology sphere quotient).
        The I*-invariant count is:

          n_SD(k)  = m(k-1) * dim(V_{k+1}) = m(k-1) * (k+2)
          n_ASD(k) = m(k+1) * dim(V_{k-1}) = m(k+1) * k
          total    = m(k-1)*(k+2) + m(k+1)*k

        Key values:
          k=1:  m(0)*3 + m(2)*1 = 3 + 0 = 3   (right-invariant forms)
          k=2:  m(1)*4 + m(3)*2 = 0 + 0 = 0
          k=11: m(10)*13 + m(12)*11 = 0 + 11 = 11
          k=12: m(11)*14 + m(13)*12 = 0 + 0 = 0
          k=13: m(12)*15 + m(14)*13 = 15 + 0 = 15

        Parameters
        ----------
        k : int, spectral level (k >= 1)

        Returns
        -------
        int : number of I*-invariant coexact modes at level k
        """
        if k < 1:
            return 0

        m_plus = self.trivial_multiplicity(k + 1)
        m_minus = self.trivial_multiplicity(k - 1)

        n_sd = m_plus * k             # self-dual contribution
        n_asd = m_minus * (k + 2)     # anti-self-dual contribution

        return n_sd + n_asd

    def coexact_invariant_detail(self, k: int) -> dict:
        """
        Detailed breakdown of I*-invariant coexact modes at level k.

        Returns
        -------
        dict with:
            'k'          : spectral level
            'eigenvalue' : (k+1)^2 (units R=1)
            'mass'       : k+1 (units 1/R)
            'm_left_sd'  : m(k+1) = I*-invariants in V_{k+1} (self-dual left rep)
            'm_left_asd' : m(k-1) = I*-invariants in V_{k-1} (anti-self-dual left rep)
            'n_sd'       : self-dual I*-invariant count
            'n_asd'      : anti-self-dual I*-invariant count
            'total'      : total I*-invariant coexact modes
            's3_total'   : total coexact modes on S^3 (for comparison)
        """
        m_sd = self.trivial_multiplicity(k + 1)
        m_asd = self.trivial_multiplicity(k - 1)

        n_sd = m_sd * k
        n_asd = m_asd * (k + 2)
        total = n_sd + n_asd
        s3_total = 2 * k * (k + 2)

        return {
            'k': k,
            'eigenvalue': (k + 1) ** 2,
            'mass': k + 1,
            'm_left_sd': m_sd,
            'm_left_asd': m_asd,
            'n_sd': n_sd,
            'n_asd': n_asd,
            'total': total,
            's3_total': s3_total,
        }

    def invariant_levels_coexact(self, k_max: int = 60) -> list[tuple[int, int]]:
        """
        All spectral levels k (1 to k_max) with surviving coexact 1-forms.

        Returns
        -------
        list of (k, multiplicity) for levels with nonzero I*-invariant count
        """
        result = []
        for k in range(1, k_max + 1):
            n = self.coexact_invariant_multiplicity(k)
            if n > 0:
                result.append((k, n))
        return result

    def coexact_spectrum(self, k_max: int = 60, R: float = 1.0) -> list[tuple[float, int, int]]:
        """
        Coexact 1-form eigenvalues on S^3/I* (CORRECTED spectrum).

        Eigenvalue at level k: (k+1)^2/R^2
        Multiplicity: coexact_invariant_multiplicity(k)

        Returns
        -------
        list of (eigenvalue, multiplicity, k_value) for surviving modes
        """
        invariant = self.invariant_levels_coexact(k_max)
        return [((k + 1)**2 / R**2, mult, k) for k, mult in invariant]

    # ==================================================================
    # Exact 1-form spectrum on S^3/I*
    # ==================================================================

    def exact_invariant_multiplicity(self, l: int) -> int:
        """
        Number of I*-invariant exact 1-forms at level l.

        Exact 1-forms at level l on S^3 have eigenvalue l(l+2)/R^2.
        They are df for scalar eigenfunctions f in V_l.
        Under SU(2)_L, they transform as V_l.
        I*-invariant count: m(l) * (l+1), where (l+1) = dim(V_l under SU(2)_R).

        Wait -- exact 1-forms df at level l come from scalars in the eigenspace
        with eigenvalue l(l+2)/R^2. The scalar eigenspace at level l on S^3
        carries (V_l, V_l) under SU(2)_L x SU(2)_R (the Peter-Weyl theorem
        on SU(2) gives left-regular representation = sum_l V_l tensor V_l*).

        Actually for scalars: at level l the eigenspace is V_l (left) tensor V_l (right),
        dimension (l+1)^2. The exact 1-forms df inherit the same quantum numbers
        (applying d doesn't change the SU(2) representation).

        I*-invariant: we need I*-invariants in V_l under SU(2)_L, times the full
        right factor V_l. So: m(l) * (l+1) exact modes survive.

        Parameters
        ----------
        l : int, angular momentum level (l >= 1)

        Returns
        -------
        int : number of I*-invariant exact 1-forms
        """
        if l < 1:
            return 0
        return self.trivial_multiplicity(l) * (l + 1)

    def exact_spectrum(self, l_max: int = 60, R: float = 1.0) -> list[tuple[float, int, int]]:
        """
        Exact 1-form eigenvalues on S^3/I*.

        Eigenvalue at level l: l(l+2)/R^2
        Multiplicity: m(l) * (l+1)

        Returns
        -------
        list of (eigenvalue, multiplicity, l_value) for surviving modes
        """
        result = []
        for l in range(1, l_max + 1):
            n = self.exact_invariant_multiplicity(l)
            if n > 0:
                result.append((l * (l + 2) / R**2, n, l))
        return result

    # ==================================================================
    # Yang-Mills spectrum on S^3/I*
    # ==================================================================

    def ym_spectrum(self, k_max: int = 60, R: float = 1.0,
                    gauge_group: str = 'SU(2)') -> list[tuple[float, int, int]]:
        """
        Yang-Mills spectrum on S^3/I* (physical, coexact modes only).

        The linearized YM operator around the Maurer-Cartan vacuum acts on
        adjoint-valued coexact 1-forms. The spectrum is:

          eigenvalue = (k+1)^2/R^2
          multiplicity = coexact_invariant_multiplicity(k) * dim(adj(G))

        For SU(2): dim(adj) = 3.

        Parameters
        ----------
        k_max       : maximum spectral level
        R           : radius
        gauge_group : 'SU(2)', 'SU(3)', etc.

        Returns
        -------
        list of (eigenvalue, total_multiplicity, k_value)
        """
        dim_adj = _adjoint_dimension(gauge_group)
        invariant = self.invariant_levels_coexact(k_max)
        return [((k + 1)**2 / R**2, mult * dim_adj, k) for k, mult in invariant]

    def ym_gap(self, R: float = 1.0) -> dict:
        """
        The Yang-Mills mass gap on S^3/I*.

        THEOREM: The coexact gap eigenvalue on S^3/I* is 4/R^2 (same as S^3).
        This is because the k=1 coexact mode survives: m(0)*(1+2) = 3 modes.

        Returns
        -------
        dict with gap information
        """
        k1_mult = self.coexact_invariant_multiplicity(1)
        assert k1_mult == 3, f"Expected 3 surviving k=1 modes, got {k1_mult}"

        return {
            'gap_eigenvalue': 4.0 / R**2,
            'gap_mass': 2.0 / R,
            'gap_multiplicity_coexact': k1_mult,
            'gap_k': 1,
            'same_as_s3': True,
            'note': (
                "THEOREM: The k=1 coexact gap 4/R^2 survives on S^3/I* "
                "because the self-dual sector has m(0)=1 invariant "
                "in (V_0)_R, giving (1+2)*1=3 modes. These are the "
                "right-invariant forms that survive the right-action quotient."
            ),
        }

    # ==================================================================
    # Adjoint-valued spectrum (I* acts on gauge fiber too)
    # ==================================================================

    def coexact_invariant_multiplicity_adjoint(self, k: int) -> int:
        """
        I*-invariant adjoint-valued coexact 1-forms (adjoint scenario).

        In the compact topology framework, gauge SU(2) = geometric SU(2), so I*
        acts on the adjoint (gauge) index too. The adjoint is V_2.

        The coexact eigenspace at level k carries:
          Self-dual:     V_{k+1} (left) x V_{k-1} (right)
          Anti-self-dual: V_{k-1} (left) x V_{k+1} (right)

        Tensored with adjoint V_2:
          SD:  V_{k+1} x V_2 (left) x V_{k-1} (right)
          ASD: V_{k-1} x V_2 (left) x V_{k+1} (right)

        I* invariants in V_a x V_b = sum_c N^c_{ab} * m(c) where N^c_{ab}
        are Clebsch-Gordan multiplicities. For SU(2): V_a x V_b = V_{|a-b|} + ... + V_{a+b}.

        For the self-dual part at level k (I* acts on V_{k+1} x V_2 on the left):
          V_{k+1} x V_2 = V_{k-1} + V_{k+1} + V_{k+3}  (for k >= 1)
          I*-invariant count = m(k-1) + m(k+1) + m(k+3)
          Multiplied by dim(V_{k-1}) = k on the right

        For the anti-self-dual part (I* acts on V_{k-1} x V_2):
          V_{k-1} x V_2 = V_{k-3} + V_{k-1} + V_{k+1}  (for k >= 2)
          For k=1: V_0 x V_2 = V_2, so invariants = m(2)
          I*-invariant count = m(k-3) + m(k-1) + m(k+1)  (for k >= 2)
          Multiplied by dim(V_{k+1}) = k+2 on the right

        Parameters
        ----------
        k : int, spectral level (k >= 1)

        Returns
        -------
        int : number of I*-invariant adjoint-valued coexact modes
        """
        if k < 1:
            return 0

        # Self-dual: V_{k+1} x V_2 under SU(2)_L
        # CG: V_a x V_2 = V_{a-2} + V_a + V_{a+2} for a >= 2
        #     V_1 x V_2 = V_1 + V_3 (since |1-2|=1, 1+2=3, step by 2: 1,3)
        #     V_0 x V_2 = V_2
        a_sd = k + 1  # SU(2)_L label for self-dual
        if a_sd >= 2:
            m_sd_left = (self.trivial_multiplicity(a_sd - 2) +
                         self.trivial_multiplicity(a_sd) +
                         self.trivial_multiplicity(a_sd + 2))
        elif a_sd == 1:
            m_sd_left = (self.trivial_multiplicity(1) +
                         self.trivial_multiplicity(3))
        else:  # a_sd == 0
            m_sd_left = self.trivial_multiplicity(2)

        n_sd = m_sd_left * k  # times dim(V_{k-1})

        # Anti-self-dual: V_{k-1} x V_2 under SU(2)_L
        a_asd = k - 1
        if a_asd >= 2:
            m_asd_left = (self.trivial_multiplicity(a_asd - 2) +
                          self.trivial_multiplicity(a_asd) +
                          self.trivial_multiplicity(a_asd + 2))
        elif a_asd == 1:
            m_asd_left = (self.trivial_multiplicity(1) +
                          self.trivial_multiplicity(3))
        else:  # a_asd == 0 (k=1)
            m_asd_left = self.trivial_multiplicity(2)

        n_asd = m_asd_left * (k + 2)  # times dim(V_{k+1})

        return n_sd + n_asd

    def invariant_levels_coexact_adjoint(self, k_max: int = 60) -> list[tuple[int, int]]:
        """
        Coexact levels with surviving modes in adjoint scenario.

        Returns
        -------
        list of (k, multiplicity)
        """
        result = []
        for k in range(1, k_max + 1):
            n = self.coexact_invariant_multiplicity_adjoint(k)
            if n > 0:
                result.append((k, n))
        return result

    # ==================================================================
    # Gap comparison: S^3 vs S^3/I*
    # ==================================================================

    def gap_comparison(self, R_fm: float = 2.2) -> dict:
        """
        Compare spectral gaps between S^3 and S^3/I*.

        Uses CORRECTED eigenvalues:
          S^3 scalar gap:  3/R^2  (l=1)
          S^3 coexact gap: 4/R^2  (k=1)
          S^3/I* scalar gap: 168/R^2  (l=12)
          S^3/I* coexact gap: 4/R^2  (k=1, survives)

        Parameters
        ----------
        R_fm : radius in femtometers

        Returns
        -------
        dict with comparison data
        """
        def to_mev(coeff):
            if coeff is None:
                return None
            return np.sqrt(coeff) * HBAR_C_MEV_FM / R_fm

        # Scalar gaps
        scalar_inv = self.invariant_levels_scalar(60)
        nontrivial_scalar = [(l, m) for l, m in scalar_inv if l > 0]
        l_scalar_gap = nontrivial_scalar[0][0] if nontrivial_scalar else None

        scalar_gap_s3 = 3        # l=1: 1*3
        scalar_gap_poincare = (
            l_scalar_gap * (l_scalar_gap + 2) if l_scalar_gap else None
        )

        # Coexact 1-form gaps (CORRECTED: use (k+1)^2/R^2)
        coexact_inv = self.invariant_levels_coexact(60)
        k_coexact_gap = coexact_inv[0][0] if coexact_inv else None

        coexact_gap_s3 = 4       # k=1: (1+1)^2 = 4
        coexact_gap_poincare = (
            (k_coexact_gap + 1)**2 if k_coexact_gap else None
        )

        # Second coexact mode
        k_second = coexact_inv[1][0] if len(coexact_inv) >= 2 else None
        second_coexact_poincare = (k_second + 1)**2 if k_second else None
        second_coexact_s3 = 9    # k=2: (2+1)^2 = 9

        # Adjoint coexact gaps
        adjoint_inv = self.invariant_levels_coexact_adjoint(60)
        k_adjoint_gap = adjoint_inv[0][0] if adjoint_inv else None
        adjoint_gap = (k_adjoint_gap + 1)**2 if k_adjoint_gap else None

        return {
            'R_fm': R_fm,
            'scalar': {
                's3': {
                    'l': 1,
                    'eigenvalue_coeff': scalar_gap_s3,
                    'gap_mev': to_mev(scalar_gap_s3),
                },
                'poincare': {
                    'l': l_scalar_gap,
                    'eigenvalue_coeff': scalar_gap_poincare,
                    'gap_mev': to_mev(scalar_gap_poincare),
                },
                'ratio': (scalar_gap_poincare / scalar_gap_s3
                          if scalar_gap_poincare else None),
            },
            'coexact': {
                's3': {
                    'k': 1,
                    'eigenvalue_coeff': coexact_gap_s3,
                    'gap_mev': to_mev(coexact_gap_s3),
                },
                'poincare': {
                    'k': k_coexact_gap,
                    'eigenvalue_coeff': coexact_gap_poincare,
                    'gap_mev': to_mev(coexact_gap_poincare),
                },
                'ratio': (coexact_gap_poincare / coexact_gap_s3
                          if coexact_gap_poincare else None),
            },
            'second_coexact': {
                's3': {
                    'k': 2,
                    'eigenvalue_coeff': second_coexact_s3,
                    'gap_mev': to_mev(second_coexact_s3),
                },
                'poincare': {
                    'k': k_second,
                    'eigenvalue_coeff': second_coexact_poincare,
                    'gap_mev': to_mev(second_coexact_poincare),
                },
                'ratio': (second_coexact_poincare / second_coexact_s3
                          if second_coexact_poincare else None),
            },
            'adjoint_coexact': {
                's3': {
                    'k': 1,
                    'eigenvalue_coeff': coexact_gap_s3,
                    'gap_mev': to_mev(coexact_gap_s3),
                },
                'poincare': {
                    'k': k_adjoint_gap,
                    'eigenvalue_coeff': adjoint_gap,
                    'gap_mev': to_mev(adjoint_gap),
                },
                'ratio': (adjoint_gap / coexact_gap_s3
                          if adjoint_gap else None),
            },
            'lattice_glueball_0pp_mev': 1730,
        }

    # ==================================================================
    # CMB predictions
    # ==================================================================

    def cmb_multipole_prediction(self, l_max: int = 30) -> list[dict]:
        """
        CMB prediction: which multipoles are suppressed on S^3/I*?

        The scalar harmonics Y_l on S^3 contribute to CMB multipole l.
        If m(l) = 0 for low l, the corresponding CMB power is zero.

        Planck observations:
        - Quadrupole (l=2): anomalously low (~2-3 sigma below LCDM)
        - Octupole (l=3): somewhat low

        Luminet et al. (2003) predicted this from S^3/I* topology.

        Returns
        -------
        list of dicts with l, multiplicity, eigenvalue, suppression status
        """
        result = []
        for l in range(l_max + 1):
            m = self.trivial_multiplicity(l)
            ev = l * (l + 2)
            result.append({
                'l': l,
                'multiplicity': m,
                'eigenvalue_coeff': ev,
                'suppressed': (m == 0),
            })
        return result

    # ==================================================================
    # Physical predictions
    # ==================================================================

    def physical_predictions(self, R_fm: float = 2.2) -> dict:
        """
        Distinguishable physical predictions from S^3/I* vs S^3 vs R^3.

        These are predictions that could, in principle, distinguish
        the three spatial topologies observationally.

        Parameters
        ----------
        R_fm : radius of the fundamental domain in femtometers

        Returns
        -------
        dict with predictions for each topology
        """
        comp = self.gap_comparison(R_fm)

        # Coexact spectrum details
        coexact_levels = self.invariant_levels_coexact(60)
        first_5 = coexact_levels[:5] if len(coexact_levels) >= 5 else coexact_levels

        # Spectrum sparsification ratios
        s3_modes_up_to_k60 = sum(2 * k * (k + 2) for k in range(1, 61))
        poincare_modes_up_to_k60 = sum(n for _, n in coexact_levels)
        sparsification = poincare_modes_up_to_k60 / s3_modes_up_to_k60

        # CMB
        cmb = self.cmb_multipole_prediction(20)
        suppressed_multipoles = [e['l'] for e in cmb if e['suppressed'] and e['l'] >= 1]

        # Mass ratios on S^3/I*
        if len(coexact_levels) >= 2:
            k1 = coexact_levels[0][0]
            k2 = coexact_levels[1][0]
            mass_ratio_21 = (k2 + 1) / (k1 + 1)
        else:
            mass_ratio_21 = None

        # Glueball spectrum: only I*-invariant composites
        # Since k=1 survives, the two-particle 0++ is still available
        # But higher glueballs need k values that survive
        glueball_note = (
            "The 0++ glueball (two k=1 composites) exists on S^3/I* "
            "with the same threshold mass 4/R as on S^3. However, "
            "excited glueballs require k > 1 modes. The next available "
            f"single-particle mode is at k={coexact_levels[1][0] if len(coexact_levels) > 1 else '?'}, "
            f"giving mass {(coexact_levels[1][0]+1) if len(coexact_levels) > 1 else '?'}/R. "
            "The excited glueball spectrum is MUCH sparser on S^3/I* than on S^3."
        )

        return {
            'mass_gap': {
                's3': {
                    'eigenvalue': f'4/R^2',
                    'mass_mev': comp['coexact']['s3']['gap_mev'],
                },
                'poincare': {
                    'eigenvalue': f'4/R^2 (same!)',
                    'mass_mev': comp['coexact']['poincare']['gap_mev'],
                },
                'r3': {
                    'eigenvalue': 'nonperturbative (conjectured)',
                    'mass_mev': '~200 MeV (lattice)',
                },
                'status': 'THEOREM',
                'note': 'Gap preserved on quotient because k=1 mode survives.',
            },
            'second_excitation': {
                's3': {
                    'k': 2,
                    'eigenvalue_coeff': 9,
                    'mass_mev': comp['second_coexact']['s3']['gap_mev'],
                },
                'poincare': {
                    'k': coexact_levels[1][0] if len(coexact_levels) > 1 else None,
                    'eigenvalue_coeff': comp['second_coexact']['poincare']['eigenvalue_coeff'],
                    'mass_mev': comp['second_coexact']['poincare']['gap_mev'],
                },
                'ratio': comp['second_coexact']['ratio'],
                'status': 'THEOREM',
                'note': 'DISTINGUISHABLE PREDICTION: second mode at k=11 vs k=2.',
            },
            'mass_ratio_m2_m1': {
                's3': 3.0 / 2.0,     # m_2/m_1 = 3/2
                'poincare': mass_ratio_21,
                'status': 'THEOREM',
                'note': 'The mass ratio of first two single-particle modes.',
            },
            'spectrum_sparsification': {
                'fraction_surviving': sparsification,
                'poincare_modes': poincare_modes_up_to_k60,
                's3_modes': s3_modes_up_to_k60,
                'status': 'THEOREM',
                'note': 'Fraction of coexact modes surviving I* projection.',
            },
            'coexact_surviving_levels': first_5,
            'cmb_suppressed_multipoles': suppressed_multipoles,
            'glueball': glueball_note,
        }

    # ==================================================================
    # J^PC on S^3/I*
    # ==================================================================

    def jpc_spectrum(self, k_max: int = 60) -> list[dict]:
        """
        J^PC quantum numbers of I*-invariant coexact modes.

        On S^3, coexact modes at level k have J = 1, ..., k.
        On S^3/I*, the J content depends on which SU(2)_L representations
        survive the I* projection.

        For the anti-self-dual part at level k (V_{k-1} under SU(2)_L):
          I*-invariants in V_{k-1} exist only if m(k-1) > 0.
          Each invariant gives modes with J = 1, ..., k from the diagonal
          SU(2) restriction of (V_{k-1}, V_{k+1}).
          Wait -- this is more subtle. The SU(2)_L invariant means
          j_L = 0 in V_{k-1}, and j_R ranges over V_{k+1}.
          So J = j_R = 0, 1, ..., k+1?

          No. The I*-invariant in V_{k-1} under SU(2)_L is a specific vector,
          not j_L = 0. I* is not a continuous subgroup, so we can't use
          continuous quantum numbers. The I*-invariant vectors in V_{k-1}
          are specific linear combinations.

          Actually, for the J^PC assignment, what matters is the diagonal
          SU(2)_diag action on the surviving modes. The I* invariance is
          about the LEFT action; the diagonal SU(2) involves BOTH left
          and right. So the J content on S^3/I* is not simply inherited
          from S^3 -- it requires a more careful analysis.

        For now, we report which k levels survive with their total
        multiplicities. A full J^PC decomposition on S^3/I* would
        require computing the I*-invariant subspace of each (j_L, j_R)
        representation, which is a non-trivial representation theory
        computation.

        STATUS: NUMERICAL for surviving levels, CONJECTURE for J content.

        Returns
        -------
        list of dicts with k, eigenvalue, surviving multiplicity, and
        available J range from S^3 (upper bound on J content on S^3/I*)
        """
        result = []
        for k in range(1, k_max + 1):
            detail = self.coexact_invariant_detail(k)
            if detail['total'] > 0:
                result.append({
                    'k': k,
                    'eigenvalue': (k + 1)**2,
                    'mass_ratio_to_gap': (k + 1) / 2.0,
                    'total_multiplicity': detail['total'],
                    's3_total': detail['s3_total'],
                    'j_range_s3': list(range(1, k + 1)),
                    'j_max': k,
                    'n_sd': detail['n_sd'],
                    'n_asd': detail['n_asd'],
                    'status': 'NUMERICAL',
                })
        return result

    # ==================================================================
    # Summary output
    # ==================================================================

    def print_summary(self, k_max: int = 30, R_fm: float = 2.2) -> str:
        """Print a comprehensive summary of the corrected I*-invariant spectrum."""
        lines = []
        lines.append("=" * 76)
        lines.append("YANG-MILLS SPECTRUM ON S^3/I* (POINCARE HOMOLOGY SPHERE)")
        lines.append("Corrected with coexact/exact split (Session 2+)")
        lines.append("=" * 76)

        # Scalar spectrum
        lines.append("")
        lines.append("--- SCALAR SPECTRUM (Delta_0) ---")
        lines.append(f"{'l':>4} {'m(l)':>6} {'ev_coeff':>10} {'S3_mult':>10}")
        lines.append("-" * 35)
        for l in range(min(k_max, 35)):
            m = self.trivial_multiplicity(l)
            ev_coeff = l * (l + 2)
            s3_mult = (l + 1)**2
            marker = " <-- SURVIVES" if m > 0 else ""
            lines.append(f"{l:4d} {m:6d} {ev_coeff:10d} {s3_mult:10d}{marker}")

        # Coexact 1-form spectrum (CORRECTED)
        lines.append("")
        lines.append("--- COEXACT 1-FORM SPECTRUM (physical, (k+1)^2/R^2) ---")
        coexact = self.invariant_levels_coexact(k_max)
        lines.append(f"{'k':>4} {'n_inv':>8} {'ev=(k+1)^2':>12} {'S3_mult':>10} {'ratio':>8}")
        lines.append("-" * 46)
        for k, mult in coexact:
            ev = (k + 1)**2
            s3_mult = 2 * k * (k + 2)
            ratio = mult / s3_mult
            lines.append(f"{k:4d} {mult:8d} {ev:12d} {s3_mult:10d} {ratio:8.4f}")

        # Exact 1-form spectrum
        lines.append("")
        lines.append("--- EXACT 1-FORM SPECTRUM (pure gauge, l(l+2)/R^2) ---")
        exact = self.exact_spectrum(l_max=k_max, R=1.0)
        lines.append(f"{'l':>4} {'n_inv':>8} {'ev=l(l+2)':>12}")
        lines.append("-" * 28)
        for ev, mult, l in exact[:10]:
            lines.append(f"{l:4d} {mult:8d} {l*(l+2):12d}")

        # Gap comparison
        lines.append("")
        lines.append("--- GAP COMPARISON (CORRECTED) ---")
        comp = self.gap_comparison(R_fm)
        lines.append(f"Radius R = {R_fm} fm")

        s = comp['scalar']
        lines.append(f"Scalar gap:")
        lines.append(f"  S^3:    l={s['s3']['l']}, coeff={s['s3']['eigenvalue_coeff']}, "
                      f"gap = {s['s3']['gap_mev']:.1f} MeV")
        lines.append(f"  S^3/I*: l={s['poincare']['l']}, "
                      f"coeff={s['poincare']['eigenvalue_coeff']}, "
                      f"gap = {s['poincare']['gap_mev']:.1f} MeV")
        lines.append(f"  Ratio: {s['ratio']:.1f}x")

        c = comp['coexact']
        lines.append(f"Coexact 1-form gap (YM physical):")
        lines.append(f"  S^3:    k={c['s3']['k']}, coeff={c['s3']['eigenvalue_coeff']}, "
                      f"gap = {c['s3']['gap_mev']:.1f} MeV")
        lines.append(f"  S^3/I*: k={c['poincare']['k']}, "
                      f"coeff={c['poincare']['eigenvalue_coeff']}, "
                      f"gap = {c['poincare']['gap_mev']:.1f} MeV")
        lines.append(f"  Ratio: {c['ratio']:.1f}x  <-- GAP PRESERVED")

        sc = comp['second_coexact']
        lines.append(f"Second coexact mode:")
        lines.append(f"  S^3:    k={sc['s3']['k']}, coeff={sc['s3']['eigenvalue_coeff']}, "
                      f"mass = {sc['s3']['gap_mev']:.1f} MeV")
        lines.append(f"  S^3/I*: k={sc['poincare']['k']}, "
                      f"coeff={sc['poincare']['eigenvalue_coeff']}, "
                      f"mass = {sc['poincare']['gap_mev']:.1f} MeV")
        lines.append(f"  Ratio: {sc['ratio']:.1f}x  <-- DISTINGUISHABLE PREDICTION")

        # CMB
        lines.append("")
        lines.append("--- CMB MULTIPOLE PREDICTIONS ---")
        cmb = self.cmb_multipole_prediction(min(k_max, 20))
        cmb_notes = {
            0: "monopole (unobservable)",
            1: "dipole (Doppler)",
            2: "quadrupole (LOW in Planck!)",
            3: "octupole (somewhat low)",
        }
        lines.append(f"{'l':>4} {'m(l)':>6} {'Status':>12} {'Note':>30}")
        lines.append("-" * 56)
        for entry in cmb:
            l = entry['l']
            m_val = entry['multiplicity']
            status = "SUPPRESSED" if entry['suppressed'] else "present"
            note = cmb_notes.get(l, "")
            lines.append(f"{l:4d} {m_val:6d} {status:>12} {note:>30}")

        # Physical predictions
        lines.append("")
        lines.append("--- DISTINGUISHABLE PREDICTIONS ---")
        pred = self.physical_predictions(R_fm)

        lines.append(f"1. Mass gap: S^3/I* = S^3 = 2/R = {pred['mass_gap']['poincare']['mass_mev']:.1f} MeV")
        lines.append(f"   Status: {pred['mass_gap']['status']}")

        lines.append(f"2. Second excitation: S^3/I* at k={pred['second_excitation']['poincare']['k']}, "
                      f"mass = {pred['second_excitation']['poincare']['mass_mev']:.1f} MeV")
        lines.append(f"   vs S^3 at k=2, mass = {pred['second_excitation']['s3']['mass_mev']:.1f} MeV")
        lines.append(f"   Ratio: {pred['second_excitation']['ratio']:.1f}x")
        lines.append(f"   Status: {pred['second_excitation']['status']}")

        lines.append(f"3. Mass ratio m2/m1:")
        lines.append(f"   S^3:    {pred['mass_ratio_m2_m1']['s3']:.3f}")
        lines.append(f"   S^3/I*: {pred['mass_ratio_m2_m1']['poincare']:.3f}")

        lines.append(f"4. Spectrum sparsification: {pred['spectrum_sparsification']['fraction_surviving']:.4f}")
        lines.append(f"   ({pred['spectrum_sparsification']['poincare_modes']} of "
                      f"{pred['spectrum_sparsification']['s3_modes']} coexact modes survive)")

        # Molien verification
        lines.append("")
        ok, mismatches = self.verify_against_molien(min(k_max, 60))
        if ok:
            lines.append(f"Molien series verification: PASSED (all l=0..{min(k_max, 60)})")
        else:
            lines.append(f"Molien series verification: FAILED at l={[x[0] for x in mismatches]}")

        lines.append("=" * 76)

        output = "\n".join(lines)
        print(output)
        return output


# ======================================================================
# Helper: adjoint dimension (standalone to avoid circular import)
# ======================================================================

def _adjoint_dimension(gauge_group: str) -> int:
    """Dimension of the adjoint representation."""
    group = gauge_group.strip().upper().replace(' ', '')
    if group.startswith('SU(') and group.endswith(')'):
        N = int(group[3:-1])
        return N**2 - 1
    elif group.startswith('SO(') and group.endswith(')'):
        N = int(group[3:-1])
        return N * (N - 1) // 2
    elif group in ('G2', 'G(2)'):
        return 14
    elif group in ('E6', 'E(6)'):
        return 78
    elif group in ('E7', 'E(7)'):
        return 133
    elif group in ('E8', 'E(8)'):
        return 248
    else:
        raise ValueError(f"Unknown gauge group: {gauge_group}")


# ======================================================================
# Convenience function
# ======================================================================

def compute_poincare_spectrum(k_max: int = 30, R_fm: float = 2.2,
                              verbose: bool = True) -> PoincareHomology:
    """Compute and optionally display the Poincare homology sphere spectrum."""
    ph = PoincareHomology()
    if verbose:
        ph.print_summary(k_max=k_max, R_fm=R_fm)
    return ph


if __name__ == "__main__":
    compute_poincare_spectrum(k_max=30, R_fm=2.2)
