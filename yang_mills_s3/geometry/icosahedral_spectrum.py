"""
Icosahedral Spectrum -- I*-invariant spectrum of Laplacians on S3/I*.

Computes the spectrum of the Hodge-de Rham Laplacian restricted to
I*-invariant modes, where I* is the binary icosahedral group (order 120).

The quotient S3/I* is the Poincare homology sphere -- same local geometry
as S3 but only I*-invariant eigenmodes survive.

Key results (NUMERICAL, verified against known Molien series):
  - First nontrivial scalar invariant: l=12, eigenvalue 168/R^2
  - Scalar gap ratio: S3/I* vs S3 = 168/3 = 56x
  - Yang-Mills 1-form gap: 5/R^2 at l=1 (SAME as S3)
  - CMB: all multipoles l=1..11 suppressed (l=2 quadrupole in particular)

The Molien series generating function is:
  sum_l m(l) t^l = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30))

where 12, 20, 30 are the degrees of the Klein invariant polynomials
of I* acting on C[x,y], and the t^60 corrects for the syzygy.

Mathematical method:
  For V_l the (l+1)-dimensional irrep of SU(2), the multiplicity of
  the trivial representation of I* in V_l restricted to I* is:
      m(l) = (1/|I*|) * sum_{conjugacy classes C} |C| * chi_l(theta_C)
  where chi_l(theta) = sin((l+1)*theta/2) / sin(theta/2).

For 1-forms, the eigenspace at level l decomposes under SU(2)_L as
V_{l+1} + V_{l-1} (exact + coexact). The I*-invariant count is:
      m_1(l) = m(l+1) + m(l-1) = (1/120) * sum |C| * chi_l * chi_1

STATUS: NUMERICAL (rigorous computation of representation-theoretic data)
"""

import numpy as np


# Physical constants (natural units: hbar = c = 1, lengths in fm, energies in MeV)
HBAR_C_MEV_FM = 197.3269804  # hbar*c in MeV*fm


class IcosahedralSpectrum:
    """
    Spectrum of Laplacian operators on S3/I* (Poincare homology sphere).

    Computes which angular momentum levels l survive the I*-invariant
    projection, and their multiplicities.

    The binary icosahedral group I* c SU(2) has order 120.
    Its elements as unit quaternions correspond to the 120 vertices
    of the 600-cell regular polytope.

    McKay correspondence: I* <-> E8 in the ADE classification.
    """

    def __init__(self):
        """
        Set up the conjugacy classes of I* c SU(2).

        I* has 9 conjugacy classes. Each element g in SU(2) has eigenvalues
        e^{i*theta/2}, e^{-i*theta/2} for some rotation angle theta in [0, 2*pi].

        Conjugacy classes (size, rotation_angle_theta):
        1. Identity:         1 element,  theta = 0
        2. Central element:  1 element,  theta = 2*pi
        3. Order 10 (+):    12 elements, theta = 2*pi/5
        4. Order 10 (+):    12 elements, theta = 4*pi/5
        5. Order 10 (-):    12 elements, theta = 6*pi/5
        6. Order 10 (-):    12 elements, theta = 8*pi/5
        7. Order 6 (+):     20 elements, theta = 2*pi/3
        8. Order 6 (-):     20 elements, theta = 4*pi/3
        9. Order 4:         30 elements, theta = pi

        Total: 1+1+12+12+12+12+20+20+30 = 120
        """
        # Each entry: (class_size, rotation_angle)
        self.conjugacy_classes = [
            (1,  0.0),                      # Identity
            (1,  2 * np.pi),                # -Identity (central element)
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

    def character_su2(self, l, theta):
        """
        Character of SU(2) irrep V_l at rotation angle theta.

        chi_l(theta) = sin((l+1)*theta/2) / sin(theta/2)

        This is the character of the (l+1)-dimensional irreducible
        representation of SU(2), evaluated at an element with eigenvalues
        e^{i*theta/2} and e^{-i*theta/2}.

        Special cases:
          theta = 0:    chi_l = l + 1  (dimension of the representation)
          theta = 2*pi: chi_l = (-1)^l * (l + 1)
          theta = pi:   chi_l = sin((l+1)*pi/2) / 1

        Parameters
        ----------
        l     : non-negative integer, angular momentum quantum number
        theta : rotation angle in [0, 2*pi]

        Returns
        -------
        float : character value chi_l(theta)
        """
        half_theta = theta / 2.0

        # Handle degenerate cases where sin(theta/2) = 0
        sin_denom = np.sin(half_theta)
        if abs(sin_denom) < 1e-14:
            # theta ~ 0 or theta ~ 2*pi
            # L'Hopital: lim = (l+1)*cos((l+1)*theta/2) / cos(theta/2)
            cos_num = np.cos((l + 1) * half_theta)
            cos_denom = np.cos(half_theta)
            if abs(cos_denom) < 1e-14:
                raise ValueError(
                    f"Degenerate character at l={l}, theta={theta}"
                )
            return (l + 1) * cos_num / cos_denom

        sin_num = np.sin((l + 1) * half_theta)
        return sin_num / sin_denom

    def trivial_multiplicity(self, l):
        """
        Number of I*-invariant vectors in V_l.

        m(l) = (1/|I*|) sum_{classes C} |C| * chi_l(theta_C)

        This is the multiplicity of the trivial representation of I*
        in the restriction of V_l from SU(2) to I*.

        Verified against the Molien series:
            M(t) = (1 - t^60) / ((1 - t^12)(1 - t^20)(1 - t^30))

        The first few nonzero values: m(0)=1, m(12)=1, m(20)=1, m(24)=1,
        m(30)=1, m(32)=1, m(36)=1, m(40)=1, m(42)=1, m(44)=1, ...

        Parameters
        ----------
        l : non-negative integer

        Returns
        -------
        int : number of I*-invariant vectors (always >= 0)
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
                f"Negative multiplicity m({l}) = {m_int} (raw: {m:.6f}). "
                f"Error in conjugacy class data."
            )

        if abs(m - m_int) > 1e-6:
            raise ValueError(
                f"Non-integer multiplicity m({l}) = {m:.10f}. "
                f"Error in conjugacy class data."
            )

        return m_int

    def invariant_levels_scalar(self, l_max=60):
        """
        All angular momentum levels l (0 to l_max) with I*-invariant scalars.

        These are the scalar harmonics that survive on S3/I*.

        The first nontrivial I*-invariant scalar is at l=12 (verified).
        This corresponds to the degree-12 Klein invariant polynomial.
        The scalar spectral gap on S3/I* is 168/R^2, compared to 3/R^2 on S3.

        Returns
        -------
        list of (l, multiplicity) : levels with m(l) > 0
        """
        result = []
        for l in range(l_max + 1):
            m = self.trivial_multiplicity(l)
            if m > 0:
                result.append((l, m))
        return result

    def scalar_spectrum_poincare(self, l_max=60, R=1.0):
        """
        Scalar eigenvalues of the Laplacian on S3/I*.

        Only l values with m(l) > 0 appear.
        Eigenvalue at level l: l(l+2)/R^2 (same local formula as S3)
        Multiplicity: m(l) = dim(V_l^{I*})

        Parameters
        ----------
        l_max : maximum l to check
        R     : radius of S3

        Returns
        -------
        list of (eigenvalue, multiplicity) for surviving modes
        """
        invariant = self.invariant_levels_scalar(l_max)
        result = []
        for l, mult in invariant:
            ev = l * (l + 2) / R**2
            result.append((ev, mult))
        return result

    def invariant_levels_oneform(self, l_max=60):
        """
        I*-invariant 1-form modes on S3.

        The 1-form eigenspace at level l on S3 decomposes under SU(2)_L as:
            V_{l+1} (exact part) + V_{l-1} (coexact part)

        This follows from the Peter-Weyl theorem on SU(2):
        - Exact eigenforms at eigenvalue (l(l+2)+2)/R^2 come from scalars
          at level l, giving representation V_{l+1} on the left
        - Coexact eigenforms come from 2-forms, giving V_{l-1} on the left

        The CG identity V_{l+1} + V_{l-1} = V_l tensor V_1 means:
            m_1(l) = m(l+1) + m(l-1) = (1/120) sum |C| chi_l chi_1

        At l=1: m_1(1) = m(2) + m(0) = 0 + 1 = 1.
        This surviving mode is the coexact component coming from the
        constant function (V_0), corresponding to the Maurer-Cartan form.
        It is physically real (not pure gauge) and gives the YM gap.

        Returns
        -------
        list of (l, multiplicity) : levels with m_1(l) > 0
        """
        result = []
        for l in range(1, l_max + 1):
            # Equivalent formulas:
            # (a) m(l+1) + m(l-1)
            # (b) (1/120) sum |C| chi_l chi_1
            # We use (a) since it's simpler and already verified:
            m_plus = self.trivial_multiplicity(l + 1)
            m_minus = self.trivial_multiplicity(l - 1)
            m_total = m_plus + m_minus

            if m_total > 0:
                result.append((l, m_total))
        return result

    def invariant_levels_oneform_adjoint(self, l_max=60):
        """
        I*-invariant adjoint-valued 1-form modes on S3 (adjoint scenario).

        In the compact topology framework where gauge SU(2) = geometric SU(2),
        the binary icosahedral group I* acts on both the base manifold
        and the gauge fiber. The gauge index carries V_2 (adjoint).

        The I*-invariant count uses:
            m_adj(l) = (1/120) sum |C| chi_l * chi_1 * chi_2

        This is the relevant formula when the gauge group is IDENTIFIED
        with the isometry group (the compact topology premise: space = gauge group).

        Returns
        -------
        list of (l, multiplicity) : levels with m_adj(l) > 0
        """
        result = []
        for l in range(1, l_max + 1):
            total = 0.0
            for size, theta in self.conjugacy_classes:
                chi_l = self.character_su2(l, theta)
                chi_1 = self.character_su2(1, theta)
                chi_2 = self.character_su2(2, theta)
                total += size * chi_l * chi_1 * chi_2

            m = total / self.group_order
            m_int = int(round(m))

            if abs(m - m_int) > 1e-6:
                raise ValueError(
                    f"Non-integer adjoint 1-form multiplicity at l={l}: {m:.10f}"
                )
            if m_int < 0:
                raise ValueError(
                    f"Negative adjoint 1-form multiplicity at l={l}: {m_int}"
                )

            if m_int > 0:
                result.append((l, m_int))
        return result

    def yang_mills_spectrum_poincare(self, l_max=60, R=1.0, adjoint=False):
        """
        Yang-Mills eigenvalues on S3/I*.

        Eigenvalue at level l: (l(l+2) + 2)/R^2
        (the +2 is the Weitzenboeck/Ricci correction on S3)

        Parameters
        ----------
        l_max : maximum l to check
        R     : radius of S3
        adjoint : if True, use the adjoint scenario where I* acts on gauge fiber too

        Returns
        -------
        list of (eigenvalue, multiplicity, l_value) for surviving modes
        """
        if adjoint:
            invariant = self.invariant_levels_oneform_adjoint(l_max)
        else:
            invariant = self.invariant_levels_oneform(l_max)
        result = []
        for l, mult in invariant:
            ev = (l * (l + 2) + 2) / R**2
            result.append((ev, mult, l))
        return result

    def gap_comparison(self, R_fm=2.2):
        """
        Compare spectral gaps between S3 and S3/I*.

        Energy from eigenvalue coefficient c: E = sqrt(c) * hbar_c / R

        Parameters
        ----------
        R_fm : radius of S3 in femtometers

        Returns
        -------
        dict with comparison data including both standard and adjoint scenarios
        """
        def to_mev(coeff):
            if coeff is None:
                return None
            return np.sqrt(coeff) * HBAR_C_MEV_FM / R_fm

        # Scalar gaps
        scalar_inv = self.invariant_levels_scalar(60)
        nontrivial_scalar = [(l, m) for l, m in scalar_inv if l > 0]
        l_scalar_gap = nontrivial_scalar[0][0] if nontrivial_scalar else None
        scalar_gap_coeff_s3 = 3  # l=1: 1*3
        scalar_gap_coeff_poincare = (
            l_scalar_gap * (l_scalar_gap + 2) if l_scalar_gap else None
        )

        # 1-form gaps (standard: I* acts on base only)
        oneform_inv = self.invariant_levels_oneform(60)
        l_ym_gap = oneform_inv[0][0] if oneform_inv else None
        ym_gap_coeff_s3 = 5  # l=1: 1*3+2
        ym_gap_coeff_poincare = (
            (l_ym_gap * (l_ym_gap + 2) + 2) if l_ym_gap else None
        )

        # 1-form gaps (I* acts on base + gauge fiber)
        adjoint_inv = self.invariant_levels_oneform_adjoint(60)
        l_adjoint_gap = adjoint_inv[0][0] if adjoint_inv else None
        adjoint_gap_coeff = (
            (l_adjoint_gap * (l_adjoint_gap + 2) + 2) if l_adjoint_gap else None
        )

        return {
            'R_fm': R_fm,
            'scalar': {
                's3': {
                    'l': 1,
                    'eigenvalue_coeff': scalar_gap_coeff_s3,
                    'gap_mev': to_mev(scalar_gap_coeff_s3),
                },
                'poincare': {
                    'l': l_scalar_gap,
                    'eigenvalue_coeff': scalar_gap_coeff_poincare,
                    'gap_mev': to_mev(scalar_gap_coeff_poincare),
                },
                'ratio': (scalar_gap_coeff_poincare / scalar_gap_coeff_s3
                          if scalar_gap_coeff_poincare else None),
            },
            'yang_mills_standard': {
                's3': {
                    'l': 1,
                    'eigenvalue_coeff': ym_gap_coeff_s3,
                    'gap_mev': to_mev(ym_gap_coeff_s3),
                },
                'poincare': {
                    'l': l_ym_gap,
                    'eigenvalue_coeff': ym_gap_coeff_poincare,
                    'gap_mev': to_mev(ym_gap_coeff_poincare),
                },
                'ratio': (ym_gap_coeff_poincare / ym_gap_coeff_s3
                          if ym_gap_coeff_poincare else None),
            },
            'yang_mills_adjoint': {
                's3': {
                    'l': 1,
                    'eigenvalue_coeff': ym_gap_coeff_s3,
                    'gap_mev': to_mev(ym_gap_coeff_s3),
                },
                'poincare': {
                    'l': l_adjoint_gap,
                    'eigenvalue_coeff': adjoint_gap_coeff,
                    'gap_mev': to_mev(adjoint_gap_coeff),
                },
                'ratio': (adjoint_gap_coeff / ym_gap_coeff_s3
                          if adjoint_gap_coeff else None),
            },
            'lattice_glueball_0pp_mev': 1730,
        }

    def cmb_multipole_prediction(self, l_max=30):
        """
        CMB prediction: which multipoles are suppressed on S3/I*?

        The scalar harmonics Y_l on S3 contribute to CMB multipole l.
        If m(l) = 0 for low l, the corresponding CMB multipole is suppressed.

        Compare with Planck observations:
        - Quadrupole (l=2): anomalously low (~2-3 sigma below LCDM)
        - Octupole (l=3): somewhat low

        Luminet et al. (2003) predicted this from S3/I* topology.

        Returns
        -------
        list of dicts with l, multiplicity, eigenvalue, suppression status
        """
        result = []
        for l in range(l_max + 1):
            m = self.trivial_multiplicity(l)
            ev = l * (l + 2)
            suppressed = (m == 0)
            result.append({
                'l': l,
                'multiplicity': m,
                'eigenvalue_coeff': ev,
                'suppressed': suppressed,
            })
        return result

    def molien_series_coefficients(self, l_max=60):
        """
        Compute Molien series coefficients from the known closed form.

        The Molien series for I* acting on Sym^l(C^2) is:
            M(t) = (1 - t^60) / ((1-t^12)(1-t^20)(1-t^30))

        The degrees 12, 20, 30 are the degrees of the Klein invariant
        polynomials of I*. The t^60 is the syzygy relation.

        Returns
        -------
        list : m_molien[l] for l = 0, ..., l_max
        """
        molien = [0] * (l_max + 1)

        # (1-t^12)(1-t^20)(1-t^30) expanded:
        # = 1 - t^12 - t^20 - t^30 + t^32 + t^42 + t^50 - t^62
        # Numerator: 1 - t^60
        # Recurrence: m[l] = rhs[l] + sum of contributions from denominator
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

    def verify_against_molien(self, l_max=60):
        """
        Verify the character computation against the closed-form Molien series.

        Returns
        -------
        tuple : (all_match: bool, mismatches: list of (l, m_character, m_molien))
        """
        molien = self.molien_series_coefficients(l_max)
        mismatches = []
        for l in range(l_max + 1):
            m_char = self.trivial_multiplicity(l)
            m_mol = molien[l]
            if m_char != m_mol:
                mismatches.append((l, m_char, m_mol))
        return (len(mismatches) == 0, mismatches)

    def print_summary(self, l_max=30, R_fm=2.2):
        """Print a comprehensive summary of the I*-invariant spectrum."""
        lines = []
        lines.append("=" * 72)
        lines.append("I*-INVARIANT SPECTRUM ON S3/I* (POINCARE HOMOLOGY SPHERE)")
        lines.append("=" * 72)

        # Scalar spectrum
        lines.append("")
        lines.append("--- SCALAR SPECTRUM (Delta_0) ---")
        lines.append(f"{'l':>4} {'m(l)':>6} {'ev_coeff':>10} {'S3_mult':>10}")
        lines.append("-" * 35)
        for l in range(l_max + 1):
            m = self.trivial_multiplicity(l)
            ev_coeff = l * (l + 2)
            s3_mult = (l + 1) ** 2
            marker = " <-- SURVIVES" if m > 0 else ""
            lines.append(
                f"{l:4d} {m:6d} {ev_coeff:10d} {s3_mult:10d}{marker}"
            )

        # 1-form spectrum (standard)
        lines.append("")
        lines.append("--- 1-FORM SPECTRUM (Delta_1, standard YM) ---")
        oneform = self.invariant_levels_oneform(l_max)
        lines.append(f"{'l':>4} {'m_1(l)':>8} {'ev_coeff':>10}")
        lines.append("-" * 26)
        for l, mult in oneform:
            ev_coeff = l * (l + 2) + 2
            lines.append(f"{l:4d} {mult:8d} {ev_coeff:10d}")

        # 1-form spectrum (compact topology)
        lines.append("")
        lines.append("--- 1-FORM SPECTRUM (Delta_1, adjoint: gauge=geometry) ---")
        adjoint = self.invariant_levels_oneform_adjoint(l_max)
        lines.append(f"{'l':>4} {'m_adj(l)':>8} {'ev_coeff':>10}")
        lines.append("-" * 26)
        for l, mult in adjoint:
            ev_coeff = l * (l + 2) + 2
            lines.append(f"{l:4d} {mult:8d} {ev_coeff:10d}")

        # Gap comparison
        lines.append("")
        lines.append("--- GAP COMPARISON ---")
        comp = self.gap_comparison(R_fm)

        lines.append(f"Radius R = {R_fm} fm")

        s = comp['scalar']
        lines.append(f"Scalar Laplacian gap:")
        lines.append(
            f"  S3:    l={s['s3']['l']}, coeff={s['s3']['eigenvalue_coeff']}, "
            f"gap = {s['s3']['gap_mev']:.1f} MeV"
        )
        lines.append(
            f"  S3/I*: l={s['poincare']['l']}, "
            f"coeff={s['poincare']['eigenvalue_coeff']}, "
            f"gap = {s['poincare']['gap_mev']:.1f} MeV"
        )
        lines.append(f"  Ratio: {s['ratio']:.1f}x")

        y = comp['yang_mills_standard']
        lines.append(f"Yang-Mills 1-form gap (standard):")
        lines.append(
            f"  S3:    l={y['s3']['l']}, coeff={y['s3']['eigenvalue_coeff']}, "
            f"gap = {y['s3']['gap_mev']:.1f} MeV"
        )
        lines.append(
            f"  S3/I*: l={y['poincare']['l']}, "
            f"coeff={y['poincare']['eigenvalue_coeff']}, "
            f"gap = {y['poincare']['gap_mev']:.1f} MeV"
        )
        lines.append(f"  Ratio: {y['ratio']:.1f}x")

        m = comp['yang_mills_adjoint']
        lines.append(f"Yang-Mills 1-form gap (adjoint: gauge=geometry):")
        lines.append(
            f"  S3:    l={m['s3']['l']}, coeff={m['s3']['eigenvalue_coeff']}, "
            f"gap = {m['s3']['gap_mev']:.1f} MeV"
        )
        lines.append(
            f"  S3/I*: l={m['poincare']['l']}, "
            f"coeff={m['poincare']['eigenvalue_coeff']}, "
            f"gap = {m['poincare']['gap_mev']:.1f} MeV"
        )
        lines.append(f"  Ratio: {m['ratio']:.1f}x")

        lines.append(f"  Lattice glueball 0++: {comp['lattice_glueball_0pp_mev']} MeV")

        # CMB predictions
        lines.append("")
        lines.append("--- CMB MULTIPOLE PREDICTIONS ---")
        cmb = self.cmb_multipole_prediction(min(l_max, 20))
        cmb_notes = {
            0: "monopole (unobservable)",
            1: "dipole (Doppler, unobservable)",
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

        # Molien verification
        lines.append("")
        ok, mismatches = self.verify_against_molien(l_max)
        if ok:
            lines.append(
                f"Molien series verification: PASSED (all l=0..{l_max})"
            )
        else:
            lines.append(
                f"Molien series verification: FAILED at l="
                f"{[x[0] for x in mismatches]}"
            )

        lines.append("=" * 72)

        output = "\n".join(lines)
        print(output)
        return output


def compute_icosahedral_spectrum(l_max=30, R_fm=2.2, verbose=True):
    """Compute and optionally display the I*-invariant spectrum."""
    spec = IcosahedralSpectrum()
    if verbose:
        spec.print_summary(l_max=l_max, R_fm=R_fm)
    return spec


if __name__ == "__main__":
    compute_icosahedral_spectrum(l_max=30, R_fm=2.2)
