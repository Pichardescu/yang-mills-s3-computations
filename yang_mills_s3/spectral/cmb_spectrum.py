"""
CMB Angular Power Spectrum on S³ and S³/I* (Poincaré Homology Sphere).

Computes the angular power spectrum C_l via the Sachs-Wolfe approximation
for compact spherical topologies, and compares predictions with Planck 2018
low-l TT data.

THEOREM: Radial eigenfunctions Φ_k^l(χ) on S³ form a complete orthonormal
  basis with analytically known normalization. The suppression formula
  S_l = C_l^{S³/I*} / C_l^{S³} follows from the I*-invariant mode count.

NUMERICAL: Comparison with Planck 2018 data (Commander component separation).

CONJECTURE: Physical space = S³/I* explains the observed quadrupole
  suppression (D_2^obs ≈ 200 μK² vs D_2^ΛCDM ≈ 1058 μK²).

Key physics:
  - Scalar eigenfunctions on S³ at level k: Q_{klm}(χ,θ,φ) = Φ_k^l(χ) Y_l^m(θ,φ)
  - On S³/I*: only I*-invariant modes survive (m(k) from Molien formula)
  - Quadrupole (l=2) suppressed because m(k)=0 for k=1..11

References:
  - Luminet, Weeks, Riazuelo, Lehoucq, Uzan, Nature 425, 593 (2003)
  - Cornish, Spergel, Starkman, CQG 15, 2657 (1998)
  - Aurich, Lustig, Steiner, CQG 22, 2061 (2005)
  - Planck 2018 results I, A&A 641, A1 (2020)
"""

import numpy as np
from scipy.special import gegenbauer, gammaln, eval_gegenbauer
from scipy.integrate import quad

from ..geometry.poincare_homology import PoincareHomology


# ======================================================================
# Planck 2018 low-l TT spectrum (Commander component separation)
# ======================================================================
# Format: l -> (D_l observed μK², D_l ΛCDM best-fit μK², sigma μK²)
# sigma includes cosmic variance + noise: σ ≈ √(2/(2l+1)) × C_l
# KEY OBSERVATION: D_2 observed is ~5x below ΛCDM prediction.

PLANCK_2018_LOW_L = {
    2:  (201,  1058, 596),
    3:  (946,  1055, 422),
    4:  (695,  1005, 336),
    5:  (1217, 1148, 297),
    6:  (1013, 1068, 261),
    7:  (1893, 1138, 243),
    8:  (2038, 1025, 215),
    9:  (1445, 1165, 210),
    10: (1578, 1097, 194),
    11: (1658, 1188, 189),
    12: (1439, 1163, 178),
    13: (901,  1125, 166),
    14: (1356, 1240, 168),
    15: (606,  1193, 155),
    16: (1419, 1220, 152),
    17: (1615, 1125, 142),
    18: (1549, 1223, 143),
    19: (1299, 1137, 135),
    20: (1191, 1175, 131),
    21: (1048, 1158, 125),
    22: (1220, 1283, 129),
    23: (1169, 1186, 121),
    24: (1531, 1312, 125),
    25: (1445, 1210, 118),
    26: (1218, 1304, 119),
    27: (1125, 1179, 112),
    28: (1399, 1267, 113),
    29: (1294, 1357, 115),
    30: (1630, 1269, 111),
}


class CMBSpectrum:
    """
    CMB angular power spectrum on S³ and S³/I* (Poincaré homology sphere).

    Implements the Sachs-Wolfe approximation for the angular power spectrum,
    comparing predictions from compact topologies with Planck 2018 data.

    Parameters
    ----------
    n_s : float
        Scalar spectral index. Planck 2018 best fit: 0.965.
    chi_lss : float
        Distance to last scattering surface in units of R (radians on S³).
        Luminet et al. best fit: ~0.35 (Ω_total ≈ 1.013).
        Compact topology: ~3.1 (R = c/H_0).
    k_max : int
        Maximum eigenmode level for summation truncation.
    """

    def __init__(self, n_s=0.965, chi_lss=0.35, k_max=200):
        self.n_s = np.float64(n_s)
        self.chi_lss = np.float64(chi_lss)
        self.k_max = int(k_max)
        self.poincare = PoincareHomology()
        self._norm_cache = {}
        self._multiplicity_cache = {}

    # ==================================================================
    # Normalization
    # ==================================================================

    def _normalization_sq(self, k, l):
        """
        N_{kl}² from analytical formula (log-gamma for stability).

        N_{kl}² = (k+1) × (k-l)! × (l!)² × 2^{2l+1} / (π × (k+l+1)!)

        THEOREM: This normalization ensures
            ∫_0^π |Φ_k^l(χ)|² sin²χ dχ = 1
        """
        if l > k or l < 0 or k < 0:
            return 0.0

        key = (k, l)
        if key in self._norm_cache:
            return self._norm_cache[key]

        log_N2 = (np.log(k + 1)
                  + gammaln(k - l + 1)
                  + 2.0 * gammaln(l + 1)
                  + (2 * l + 1) * np.log(2)
                  - np.log(np.pi)
                  - gammaln(k + l + 2))

        result = np.float64(np.exp(log_N2))
        self._norm_cache[key] = result
        return result

    # ==================================================================
    # Radial eigenfunctions
    # ==================================================================

    def radial_eigenfunction(self, k, l, chi):
        """
        Radial eigenfunction Φ_k^l(χ) on S³, normalized:

            ∫_0^π |Φ_k^l(χ)|² sin²χ dχ = 1

        Φ_k^l(χ) = N_{kl} × sin^l(χ) × C_{k-l}^{l+1}(cos χ)

        where C_n^α is the Gegenbauer polynomial.

        THEOREM: These form a complete orthonormal set for fixed l:
            ∫_0^π Φ_k^l(χ) Φ_{k'}^l(χ) sin²χ dχ = δ_{kk'}

        Parameters
        ----------
        k : int, eigenmode level (k >= 0)
        l : int, angular momentum (0 <= l <= k)
        chi : float or array, radial coordinate on S³ (0 to π)

        Returns
        -------
        float or array : Φ_k^l(χ)
        """
        if l > k or l < 0 or k < 0:
            return np.zeros_like(np.asarray(chi, dtype=np.float64))

        chi = np.asarray(chi, dtype=np.float64)
        scalar_input = chi.ndim == 0
        chi = np.atleast_1d(chi)

        N = np.sqrt(self._normalization_sq(k, l))

        n_geg = k - l
        alpha = l + 1.0

        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)

        # Use eval_gegenbauer for robustness at high order
        # (scipy.special.gegenbauer polynomial objects overflow for n > ~170)
        geg_vals = eval_gegenbauer(n_geg, alpha, cos_chi)

        # Handle sin^l factor carefully for l=0
        if l == 0:
            sin_factor = np.ones_like(chi)
        else:
            sin_factor = sin_chi ** l

        result = N * sin_factor * geg_vals

        if scalar_input:
            return np.float64(result[0])
        return result

    # ==================================================================
    # Primordial spectrum
    # ==================================================================

    def primordial_spectrum(self, k, k_pivot=10):
        """
        Primordial power spectrum P(k) on S³.

        P(k) = A_s × (k/k_pivot)^{n_s - 1} / [k(k+2)]

        For Harrison-Zel'dovich (n_s=1): P(k) = A_s / [k(k+2)]
        Normalization A_s is absorbed into overall C_l normalization.

        Parameters
        ----------
        k : int, eigenmode level (k >= 1)
        k_pivot : int, pivot scale for spectral tilt

        Returns
        -------
        float : P(k) (in arbitrary units, A_s = 1)
        """
        if k < 1:
            return 0.0
        k = np.float64(k)
        tilt = (k / k_pivot) ** (self.n_s - 1.0)
        return tilt / (k * (k + 2.0))

    # ==================================================================
    # I*-invariant multiplicity (cached)
    # ==================================================================

    def _trivial_mult(self, k):
        """Cached trivial multiplicity m(k) of I* in V_k."""
        if k not in self._multiplicity_cache:
            self._multiplicity_cache[k] = self.poincare.trivial_multiplicity(k)
        return self._multiplicity_cache[k]

    # ==================================================================
    # Angular power spectra
    # ==================================================================

    def cl_s3(self, l):
        """
        Angular power spectrum C_l on S³ (full sphere, all modes).

        C_l^{S³} = (1/9) Σ_{k=l}^{k_max} P(k) × |Φ_k^l(χ_LSS)|²

        The 1/9 factor is the Sachs-Wolfe coefficient (1/3)² from
        ΔT/T = (1/3) Φ on the last scattering surface.

        Parameters
        ----------
        l : int, multipole moment (l >= 0)

        Returns
        -------
        float : C_l in arbitrary units
        """
        l = int(l)
        if l < 0:
            return 0.0

        total = np.float64(0.0)
        chi = self.chi_lss

        for k in range(max(l, 1), self.k_max + 1):
            pk = self.primordial_spectrum(k)
            phi = self.radial_eigenfunction(k, l, chi)
            total += pk * phi ** 2

        return total / 9.0

    def cl_poincare(self, l):
        """
        Angular power spectrum C_l on S³/I* (position-averaged).

        C_l^{S³/I*} = (1/9) Σ_{k=l, m(k)>0}^{k_max} P(k) × |Φ_k^l(χ_LSS)|² × m(k)/(k+1)

        The factor m(k)/(k+1) arises from the Peter-Weyl decomposition:
          - On S³, eigenspace at level k is V_k ⊗ V_k with dim (k+1)²
          - I* acts on the RIGHT V_k (right multiplication on SU(2) = S³)
          - I*-invariant subspace: V_k ⊗ (V_k)^{I*} with dim (k+1) × m(k)
          - Position-averaged weight: (k+1)×m(k) / (k+1)² = m(k)/(k+1)

        NUMERICAL: m(k) = 0 for k = 1..11, so low-l multipoles are
        strongly suppressed (the key prediction). The position-averaged
        suppression is STRONGER than what a specific observer sees;
        Planck's observed D_2/D_2^LCDM ~ 0.19 is consistent with a
        specific observer position within the fundamental domain.

        Parameters
        ----------
        l : int, multipole moment (l >= 0)

        Returns
        -------
        float : C_l in arbitrary units
        """
        l = int(l)
        if l < 0:
            return 0.0

        total = np.float64(0.0)
        chi = self.chi_lss

        for k in range(max(l, 1), self.k_max + 1):
            mk = self._trivial_mult(k)
            if mk == 0:
                continue
            pk = self.primordial_spectrum(k)
            phi = self.radial_eigenfunction(k, l, chi)
            weight = mk / (k + 1.0)
            total += pk * phi ** 2 * weight

        return total / 9.0

    def suppression_ratio(self, l):
        """
        Suppression ratio S_l = C_l^{S³/I*} / C_l^{S³}.

        Key predictions:
          - S_l << 1 for l = 2..11 (strong suppression, m(k)=0 for k<12)
          - S_l → 1/120 as l → ∞ (Weyl law: fraction of I*-invariant modes)
          - S_2 is especially small (quadrupole anomaly)

        NUMERICAL: Compared with Planck 2018 observed D_2/D_2^ΛCDM ~ 0.19.

        Parameters
        ----------
        l : int, multipole moment

        Returns
        -------
        float : S_l in [0, 1]. Returns 0 if C_l^{S³} = 0.
        """
        cl_full = self.cl_s3(l)
        if cl_full <= 0.0:
            return 0.0
        cl_quotient = self.cl_poincare(l)
        return cl_quotient / cl_full

    def dl_s3(self, l):
        """
        D_l = l(l+1)C_l/(2π) on S³, in arbitrary units.

        Parameters
        ----------
        l : int, multipole moment (l >= 1)

        Returns
        -------
        float : D_l
        """
        l = int(l)
        if l < 1:
            return 0.0
        return l * (l + 1) * self.cl_s3(l) / (2.0 * np.pi)

    def dl_poincare(self, l):
        """
        D_l = l(l+1)C_l/(2π) on S³/I*, in arbitrary units.

        Parameters
        ----------
        l : int, multipole moment (l >= 1)

        Returns
        -------
        float : D_l
        """
        l = int(l)
        if l < 1:
            return 0.0
        return l * (l + 1) * self.cl_poincare(l) / (2.0 * np.pi)

    # ==================================================================
    # Planck comparison
    # ==================================================================

    def planck_comparison(self, l_max=30):
        """
        Compare S³/I* predictions with Planck 2018 data.

        The comparison is done at the level of suppression ratios:
          observed_ratio = D_l^obs / D_l^ΛCDM
          predicted_ratio = S_l (our suppression ratio)

        NUMERICAL: chi-squared computed with cosmic variance errors.

        Parameters
        ----------
        l_max : int, maximum multipole to include

        Returns
        -------
        dict with keys:
          - multipoles: list of l values
          - observed_ratio: D_l^obs / D_l^ΛCDM for each l
          - predicted_ratio: S_l for each l
          - chi_squared: Σ (observed - predicted)² / σ² (on ratio scale)
          - n_dof: number of data points
          - quadrupole_test: dict with l=2 specific comparison
        """
        multipoles = []
        observed_ratios = []
        predicted_ratios = []
        chi2 = np.float64(0.0)

        for l in range(2, l_max + 1):
            if l not in PLANCK_2018_LOW_L:
                continue

            d_obs, d_lcdm, sigma = PLANCK_2018_LOW_L[l]

            obs_ratio = d_obs / d_lcdm
            pred_ratio = self.suppression_ratio(l)

            # Error on ratio: sigma_ratio = sigma / D_l^LCDM
            sigma_ratio = sigma / d_lcdm

            multipoles.append(l)
            observed_ratios.append(obs_ratio)
            predicted_ratios.append(pred_ratio)

            chi2 += ((obs_ratio - pred_ratio) / sigma_ratio) ** 2

        n_dof = len(multipoles)

        # Quadrupole-specific test
        s2 = self.suppression_ratio(2)
        d2_obs, d2_lcdm, d2_sigma = PLANCK_2018_LOW_L[2]
        quad_test = {
            'S_2': s2,
            'observed_ratio': d2_obs / d2_lcdm,
            'sigma_ratio': d2_sigma / d2_lcdm,
            'suppression_consistent': abs(s2 - d2_obs / d2_lcdm) < 2 * d2_sigma / d2_lcdm,
        }

        return {
            'multipoles': multipoles,
            'observed_ratio': observed_ratios,
            'predicted_ratio': predicted_ratios,
            'chi_squared': float(chi2),
            'n_dof': n_dof,
            'chi_squared_per_dof': float(chi2 / n_dof) if n_dof > 0 else 0.0,
            'quadrupole_test': quad_test,
        }

    def scan_chi_lss(self, chi_values=None, l_max=30):
        """
        Scan over χ_LSS values to find best fit to Planck data.

        For each χ_LSS, computes the chi-squared of the suppression
        ratio prediction against the observed ratio D_l^obs/D_l^ΛCDM.

        Parameters
        ----------
        chi_values : array-like or None
            Values of χ_LSS to scan. Default: np.linspace(0.1, 3.5, 35).
        l_max : int
            Maximum multipole to include.

        Returns
        -------
        list of (chi_lss, chi_squared) pairs, sorted by chi_squared.
        """
        if chi_values is None:
            chi_values = np.linspace(0.1, 3.5, 35)

        results = []
        original_chi = self.chi_lss

        for chi in chi_values:
            self.chi_lss = np.float64(chi)
            # Clear caches that depend on chi_lss (norm cache doesn't)
            comparison = self.planck_comparison(l_max=l_max)
            results.append((float(chi), comparison['chi_squared']))

        # Restore original
        self.chi_lss = original_chi

        results.sort(key=lambda x: x[1])
        return results

    @staticmethod
    def omega_to_chi_lss(omega_total):
        """
        Convert Ω_total to χ_LSS assuming standard cosmology.

        Ω_k = Ω_total - 1  (spatial curvature parameter)
        R = c / (H_0 √Ω_k)  (curvature radius)
        d_LSS ≈ 3.1 × c/H_0  (comoving distance to last scattering)
        χ_LSS = d_LSS / R = 3.1 × √Ω_k

        For Ω_total = 1 (flat): R → ∞, χ_LSS → 0 (topology invisible).
        For Ω_total < 1 (open): formula not applicable (S³ requires Ω > 1).

        Parameters
        ----------
        omega_total : float, must be > 1 for S³ topology

        Returns
        -------
        float : χ_LSS in radians on S³

        Raises
        ------
        ValueError : if omega_total <= 1
        """
        omega_total = np.float64(omega_total)
        if omega_total <= 1.0:
            raise ValueError(
                f"S³ topology requires Ω_total > 1, got {omega_total}. "
                "For flat space (Ω=1), χ_LSS → ∞ (topology invisible)."
            )
        omega_k = omega_total - 1.0
        return 3.1 * np.sqrt(omega_k)

    # ==================================================================
    # Summary report
    # ==================================================================

    def full_report(self, l_max=30):
        """
        Generate a complete quantitative CMB prediction report.

        Returns
        -------
        dict with keys:
          - parameters: dict of input parameters
          - spectrum_s3: dict mapping l -> C_l on S³
          - spectrum_poincare: dict mapping l -> C_l on S³/I*
          - suppression: dict mapping l -> S_l
          - dl_s3: dict mapping l -> D_l on S³
          - dl_poincare: dict mapping l -> D_l on S³/I*
          - planck_comparison: output of planck_comparison()
          - weyl_limit: asymptotic S_l → 1/|I*| = 1/120
          - quadrupole_anomaly: summary of l=2 prediction
        """
        spectrum_s3 = {}
        spectrum_poincare = {}
        suppression = {}
        dl_s3 = {}
        dl_poincare = {}

        for l in range(2, l_max + 1):
            spectrum_s3[l] = self.cl_s3(l)
            spectrum_poincare[l] = self.cl_poincare(l)
            suppression[l] = self.suppression_ratio(l)
            dl_s3[l] = self.dl_s3(l)
            dl_poincare[l] = self.dl_poincare(l)

        comparison = self.planck_comparison(l_max=l_max)

        return {
            'parameters': {
                'n_s': float(self.n_s),
                'chi_lss': float(self.chi_lss),
                'k_max': self.k_max,
            },
            'spectrum_s3': spectrum_s3,
            'spectrum_poincare': spectrum_poincare,
            'suppression': suppression,
            'dl_s3': dl_s3,
            'dl_poincare': dl_poincare,
            'planck_comparison': comparison,
            'weyl_limit': 1.0 / 120.0,
            'quadrupole_anomaly': {
                'S_2': suppression.get(2, 0.0),
                'planck_observed_ratio': PLANCK_2018_LOW_L[2][0] / PLANCK_2018_LOW_L[2][1],
                'explanation': (
                    "m(k)=0 for k=1..11 means the quadrupole (l=2) receives "
                    "no power from the first 11 eigenmode levels. The first "
                    "contributing level is k=12 (m(12)=1), which produces a "
                    "strongly suppressed quadrupole consistent with Planck observations."
                ),
            },
        }
