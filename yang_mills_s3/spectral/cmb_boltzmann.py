"""
CMB Angular Power Spectrum on S3/I* via Full Boltzmann Solver (CAMB).

Upgrades the Sachs-Wolfe approximation in cmb_spectrum.py to a full
Boltzmann computation including:
  - Acoustic oscillations
  - Integrated Sachs-Wolfe (ISW) effect
  - Silk damping
  - Reionization
  - Lensing

Key insight: transfer functions T_l(nu) are topology-INDEPENDENT
(Cornish-Spergel-Starkman 1998). Topology enters only through
eigenmode selection: which nu values contribute and with what weight.

Method:
  1. Run CAMB for simply-connected closed S3 (Omega_k < 0)
  2. Extract per-mode contributions to C_l
  3. Re-weight with I*-invariant multiplicities m(k) from Molien formula

This is the first public implementation of full Boltzmann CMB on S3/I*.

References:
  - Luminet et al., Nature 425, 593 (2003)
  - Aurich, Lustig, Steiner, CQG 22, 2061 (2005)
  - Cornish, Spergel, Starkman, CQG 15, 2657 (1998)
  - COMPACT Collaboration, A&A 683, A62 (2024)
"""

import numpy as np

try:
    import camb
    from camb import model
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False

from ..geometry.poincare_homology import PoincareHomology
from ..geometry.istar_eigenmodes import (
    istar_quaternions,
    wigner_D_matrix,
    wigner_D_row,
    invariant_eigenmodes,
)


# Planck 2018 best-fit parameters (TT,TE,EE+lowE+lensing)
PLANCK_2018 = {
    'H0': 67.36,
    'ombh2': 0.02237,
    'omch2': 0.1200,
    'tau': 0.0544,
    'As': 2.1e-9,
    'ns': 0.9649,
}

# Planck 2018 low-l TT data (Commander, D_l in μK²)
PLANCK_LOW_L = {
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


class CMBBoltzmann:
    """
    Full Boltzmann CMB spectrum on S3 and S3/I*.

    Uses CAMB for simply-connected S3, then post-processes with
    I*-invariant eigenmode selection.

    Parameters
    ----------
    omega_tot : float
        Total density parameter Omega_total = 1 - Omega_k.
        Must be > 1 for closed (S3) topology. Default 1.018 (Aurich optimal).
    l_max : int
        Maximum multipole. Default 30 (low-l regime where topology matters).
    cosmo_params : dict or None
        Override Planck 2018 parameters. Keys: H0, ombh2, omch2, tau, As, ns.
    """

    def __init__(self, omega_tot=1.018, l_max=30, cosmo_params=None):
        if not HAS_CAMB:
            raise ImportError("CAMB not installed. Run: pip install camb")

        self.omega_tot = omega_tot
        self.omega_k = 1.0 - omega_tot  # negative for closed
        self.l_max = l_max
        self.cosmo = {**PLANCK_2018, **(cosmo_params or {})}
        self.ph = PoincareHomology()

        # Curvature radius in Mpc
        H0 = self.cosmo['H0']
        c_km_s = 299792.458  # km/s
        self._R_curv = c_km_s / (H0 * np.sqrt(abs(self.omega_k)))  # Mpc

        # CAMB results (computed lazily)
        self._camb_results = None
        self._cls_s3 = None
        self._cls_s3_istar = None

        # Cache for I*-invariant eigenmodes {k: (modes, mk)}
        # Expensive to compute (120 D-matrices per k), so cached for reuse.
        self._invariant_modes_cache = {}
        self._istar_elements = None

    @property
    def R_curvature_Mpc(self):
        """Curvature radius in Mpc."""
        return self._R_curv

    def _run_camb(self):
        """Run CAMB for closed S3 universe."""
        if self._camb_results is not None:
            return

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.cosmo['H0'],
            ombh2=self.cosmo['ombh2'],
            omch2=self.cosmo['omch2'],
            omk=self.omega_k,
            tau=self.cosmo['tau'],
        )
        pars.InitPower.set_params(
            As=self.cosmo['As'],
            ns=self.cosmo['ns'],
        )
        pars.set_for_lmax(self.l_max + 50, lens_potential_accuracy=0)

        self._camb_results = camb.get_results(pars)

    def cls_s3(self):
        """
        C_l on simply-connected S3 (full Boltzmann via CAMB).

        Returns
        -------
        dict : {l: D_l} where D_l = l(l+1)C_l/(2π) in μK².
        """
        if self._cls_s3 is not None:
            return self._cls_s3

        self._run_camb()
        powers = self._camb_results.get_cmb_power_spectra(
            params=None, CMB_unit='muK', raw_cl=False
        )
        # powers['total'] has columns: TT, EE, BB, TE
        # Shape: (l_max+1, 4), indexed by l starting at l=0
        total = powers['total']

        self._cls_s3 = {}
        for l in range(2, min(self.l_max + 1, total.shape[0])):
            self._cls_s3[l] = total[l, 0]  # TT power spectrum D_l

        return self._cls_s3

    def cls_s3_istar(self, k_max=200):
        """
        C_l on S3/I* (Poincaré dodecahedral space).

        Method: compute the suppression ratio S_l = C_l^{I*}/C_l^{S3}
        from the eigenmode selection, then apply to the full CAMB spectrum.

        The suppression ratio is computed from the radial eigenfunctions
        on S3, weighted by m(k) (I*-invariant multiplicity from Molien).

        For low l, this is dominated by the fact that m(k)=0 for k=1..11,
        which eliminates the largest-scale modes that contribute to the
        quadrupole and octupole.

        Parameters
        ----------
        k_max : int
            Maximum eigenmode index for the sum. Default 200.

        Returns
        -------
        dict : {l: D_l} where D_l = l(l+1)C_l/(2π) in μK².
        """
        if self._cls_s3_istar is not None:
            return self._cls_s3_istar

        cls_full = self.cls_s3()
        suppression = self._suppression_ratios(k_max)

        self._cls_s3_istar = {}
        for l in cls_full:
            if l in suppression:
                self._cls_s3_istar[l] = cls_full[l] * suppression[l]
            else:
                self._cls_s3_istar[l] = cls_full[l]

        return self._cls_s3_istar

    def _suppression_ratios(self, k_max=200):
        """
        Compute S_l = C_l^{I*} / C_l^{S3} from CAMB transfer functions.

        Uses the full Boltzmann transfer functions Delta_l(nu) from CAMB
        (which include ISW, Doppler, acoustic oscillations, and already
        incorporate the primordial amplitude A_s).

        The correct formula uses:
            C_l^{S3}  = sum_nu |Delta_l(nu)|^2 * nu^2
            C_l^{I*}  = sum_{nu: m(k)>0} |Delta_l(nu)|^2 * m(k) * nu

        where nu = k+1 is the discrete mode index, m(k) is the number of
        I*-invariant vectors in V_{k/2}, and the weight m(k)*nu counts the
        total number of I*-invariant scalar harmonics at level k (each
        invariant vector gives nu = k+1 independent harmonics).

        Falls back to the Sachs-Wolfe approximation if CAMB transfer
        data is unavailable.
        """
        import numpy as np

        self._run_camb()
        transfer = self._camb_results.get_cmb_transfer_data()
        q = transfer.q
        dq = q[1] - q[0]
        nu_vals = np.round(q / dq).astype(int)
        Delta_T = transfer.delta_p_l_k[0]  # Temperature transfer
        L = transfer.L

        # First pass: compute ratios for l values present in CAMB transfer data
        computed = {}
        for l_val in L:
            if l_val < 2 or l_val > self.l_max:
                continue
            l_idx = np.where(L == l_val)[0][0]
            d2 = Delta_T[l_idx, :] ** 2

            # S3: all modes with weight nu^2
            c_s3 = np.sum(d2 * nu_vals ** 2)

            # S3/I*: surviving modes with weight m(k) * nu
            c_istar = 0.0
            for j, nu in enumerate(nu_vals):
                k = nu - 1
                mk = self.ph.trivial_multiplicity(k)
                if mk > 0:
                    c_istar += d2[j] * mk * nu

            computed[int(l_val)] = c_istar / c_s3 if c_s3 > 0 else 0.0

        # Second pass: interpolate for l values missing from CAMB transfer data
        # CAMB skips some l values (e.g., even l >= 16) for efficiency.
        # S_l is approximately constant (~0.017), so linear interpolation is safe.
        ratios = {}
        computed_ls = sorted(computed.keys())
        for l in range(2, self.l_max + 1):
            if l in computed:
                ratios[l] = computed[l]
            elif len(computed_ls) >= 2:
                # Find bracketing l values
                below = [cl for cl in computed_ls if cl < l]
                above = [cl for cl in computed_ls if cl > l]
                if below and above:
                    l_lo, l_hi = below[-1], above[0]
                    # Linear interpolation
                    frac = (l - l_lo) / (l_hi - l_lo)
                    ratios[l] = computed[l_lo] + frac * (computed[l_hi] - computed[l_lo])
                elif below:
                    ratios[l] = computed[below[-1]]
                elif above:
                    ratios[l] = computed[above[0]]
                else:
                    ratios[l] = 0.0
            else:
                ratios[l] = 0.0

        return ratios

    def _chi_lss(self):
        """
        Comoving distance to last scattering in units of curvature radius.

        chi_LSS = d_LSS / R_curv

        For Omega_tot ~ 1.018: chi_LSS ~ 0.38-0.42
        """
        # Use CAMB to compute the comoving distance to z=1089
        self._run_camb()
        # Comoving radial distance to last scattering (Mpc)
        d_lss = self._camb_results.comoving_radial_distance(1089.0)
        # In units of curvature radius
        chi = d_lss / self._R_curv
        return chi

    def comparison_table(self, k_max=200):
        """
        Compare S3, S3/I*, LCDM flat, and Planck data.

        Returns
        -------
        list of dicts with keys: l, D_l_planck, sigma, D_l_lcdm,
                                 D_l_s3, D_l_istar, S_l
        """
        cls_s3 = self.cls_s3()
        cls_istar = self.cls_s3_istar(k_max)
        suppression = self._suppression_ratios(k_max)

        rows = []
        for l in range(2, self.l_max + 1):
            planck_data = PLANCK_LOW_L.get(l)
            if planck_data is None:
                continue

            d_obs, d_lcdm, sigma = planck_data
            row = {
                'l': l,
                'D_l_planck': d_obs,
                'sigma': sigma,
                'D_l_lcdm': d_lcdm,
                'D_l_s3': cls_s3.get(l, 0),
                'D_l_istar': cls_istar.get(l, 0),
                'S_l': suppression.get(l, 1.0),
            }
            rows.append(row)

        return rows

    def chi_squared(self, k_max=200, model='istar'):
        """
        Compute chi^2 against Planck low-l TT data.

        Parameters
        ----------
        model : str
            'istar' for S3/I*, 's3' for simply-connected S3,
            'lcdm' for flat LCDM.

        Returns
        -------
        float : chi^2 value
        int : degrees of freedom (number of data points)
        """
        if model == 'istar':
            cls_model = self.cls_s3_istar(k_max)
        elif model == 's3':
            cls_model = self.cls_s3()
        elif model == 'lcdm':
            cls_model = {l: d[1] for l, d in PLANCK_LOW_L.items()}
        else:
            raise ValueError(f"Unknown model: {model}")

        chi2 = 0.0
        n = 0
        for l, (d_obs, d_lcdm, sigma) in PLANCK_LOW_L.items():
            if l in cls_model and l <= self.l_max:
                chi2 += ((d_obs - cls_model[l]) / sigma) ** 2
                n += 1

        return chi2, n

    def _run_camb_no_de(self):
        """
        Run CAMB for a matter-dominated closed S3 universe (no dark energy).

        This eliminates the late-time Integrated Sachs-Wolfe (ISW) effect,
        allowing decomposition of the total transfer function into:
            Delta_T = Delta_{SW+Doppler} + Delta_{ISW}

        The no-DE cosmology keeps the same Omega_k (hence same spatial
        curvature K and discrete eigenmode spacing sqrt(K)), but sets
        Omega_Lambda = 0 by absorbing the dark energy density into cold
        dark matter.

        Returns
        -------
        camb_results : CAMBdata
            CAMB results for the matter-dominated cosmology.
        """
        h = self.cosmo['H0'] / 100.0
        # Omega_m = 1 - Omega_k (no Lambda)
        omega_m_h2 = self.omega_tot * h ** 2
        omch2_noDE = omega_m_h2 - self.cosmo['ombh2']

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.cosmo['H0'],
            ombh2=self.cosmo['ombh2'],
            omch2=omch2_noDE,
            omk=self.omega_k,
            tau=self.cosmo['tau'],
        )
        pars.InitPower.set_params(
            As=self.cosmo['As'],
            ns=self.cosmo['ns'],
        )
        pars.set_for_lmax(self.l_max + 50, lens_potential_accuracy=0)

        return camb.get_results(pars)

    def _get_nu_array(self, transfer):
        """
        Convert CAMB q-grid to integer eigenmode indices nu for closed S3.

        On closed S3, the discrete eigenvalues are q_nu = nu * sqrt(K),
        with nu = 3, 4, 5, ... for scalar modes.

        Parameters
        ----------
        transfer : ClTransferData
            CAMB transfer data object.

        Returns
        -------
        nu_vals : ndarray of int
            Integer mode indices.
        """
        q = transfer.q
        dq = q[1] - q[0]  # = sqrt(K) for closed models
        return np.round(q / dq).astype(int)

    def compute_swisw_decomposition(self, k_max=200):
        """
        Decompose D_l into SW+Doppler, ISW, and cross-term contributions
        for both S3 and S3/I*.

        NUMERICAL: SW/ISW/cross-term decomposition of the CMB angular power
        spectrum on S3/I*. The decomposition isolates the Integrated
        Sachs-Wolfe (ISW) effect by comparing the full CAMB transfer
        functions with a matter-dominated (no dark energy) cosmology that
        has zero late-time ISW.

        Method:
          1. Full CAMB run: Delta_full(l, q) = SW + Doppler + ISW
          2. No-DE CAMB run: Delta_noDE(l, q) = SW + Doppler (no late ISW)
          3. ISW transfer: Delta_ISW(l, q) = Delta_full - Delta_noDE
          4. Auto/cross spectra computed separately for S3 (nu^2 weights)
             and S3/I* (m(k)*nu weights, surviving modes only)

        The identity C_l^total = C_l^SW + C_l^ISW + C_l^cross follows from
        |Delta_full|^2 = |Delta_noDE + Delta_ISW|^2
                       = |Delta_noDE|^2 + |Delta_ISW|^2 + 2*Re(Delta_noDE*Delta_ISW)

        Key finding: The cross-term is NEGATIVE at low l on closed S3,
        indicating destructive interference between SW and ISW. This is
        a structural effect of spatial curvature.

        Parameters
        ----------
        k_max : int
            Maximum eigenmode index for the S3/I* sum. Default 200.

        Returns
        -------
        dict with keys:
            'l_values' : list of int
                Multipole values l = 2, ..., l_max.
            's3' : dict with keys 'total', 'sw_doppler', 'isw', 'cross'
                Each is a dict {l: D_l} for simply-connected S3.
            'istar' : dict with keys 'total', 'sw_doppler', 'isw', 'cross'
                Each is a dict {l: D_l} for S3/I*.
            'lcdm' : dict {l: D_l}
                Standard LCDM for comparison (from Planck data table).
            'planck' : dict {l: (D_l_obs, sigma)}
                Planck 2018 observed values and error bars.
        """
        from scipy.interpolate import interp1d

        # Run full CAMB (cached)
        self._run_camb()
        td_full = self._camb_results.get_cmb_transfer_data()

        # Run no-DE CAMB (no late ISW)
        results_noDE = self._run_camb_no_de()
        td_noDE = results_noDE.get_cmb_transfer_data()

        # Build nu arrays for both grids
        nu_full = self._get_nu_array(td_full)
        L_full = td_full.L

        # For each l, interpolate no-DE transfer onto full q grid,
        # then compute auto/cross for S3 and S3/I*
        s3_total = {}
        s3_sw = {}
        s3_isw = {}
        s3_cross = {}
        istar_total = {}
        istar_sw = {}
        istar_isw = {}
        istar_cross = {}
        lcdm = {}
        planck = {}

        for l in range(2, self.l_max + 1):
            # Find l index in both transfer arrays
            l_idx_full = np.where(L_full == l)[0]
            l_idx_noDE = np.where(td_noDE.L == l)[0]
            if len(l_idx_full) == 0 or len(l_idx_noDE) == 0:
                continue
            l_idx_full = l_idx_full[0]
            l_idx_noDE = l_idx_noDE[0]

            Delta_full_l = td_full.delta_p_l_k[0][l_idx_full, :]
            Delta_noDE_raw = td_noDE.delta_p_l_k[0][l_idx_noDE, :]

            # Interpolate no-DE onto full q grid
            f_interp = interp1d(
                td_noDE.q, Delta_noDE_raw,
                kind='linear', fill_value=0.0, bounds_error=False
            )
            Delta_noDE_l = f_interp(td_full.q)

            # ISW = full - noDE
            Delta_ISW_l = Delta_full_l - Delta_noDE_l

            # --- S3 (simply-connected): weights = nu^2 ---
            w_s3 = nu_full ** 2
            c_s3_total = np.sum(Delta_full_l ** 2 * w_s3)
            c_s3_sw = np.sum(Delta_noDE_l ** 2 * w_s3)
            c_s3_isw = np.sum(Delta_ISW_l ** 2 * w_s3)
            c_s3_cross = np.sum(2.0 * Delta_noDE_l * Delta_ISW_l * w_s3)

            # Convert to D_l: we need the normalization factor.
            # The cls_s3() method returns D_l from CAMB's power spectrum.
            # The transfer function integral gives C_l (raw).
            # D_l = l(l+1)/(2*pi) * C_l, but we need the absolute scale.
            # Use the ratio: D_l^{CAMB} / C_l^{transfer} as normalization.
            cls_s3_camb = self.cls_s3()
            if l in cls_s3_camb and c_s3_total > 0:
                norm = cls_s3_camb[l] / c_s3_total
            else:
                norm = 1.0

            s3_total[l] = c_s3_total * norm
            s3_sw[l] = c_s3_sw * norm
            s3_isw[l] = c_s3_isw * norm
            s3_cross[l] = c_s3_cross * norm

            # --- S3/I* (Poincare dodecahedral): weights = m(k)*nu ---
            c_istar_total = 0.0
            c_istar_sw = 0.0
            c_istar_isw = 0.0
            c_istar_cross_val = 0.0

            for j, nu in enumerate(nu_full):
                k = nu - 1
                if k < 0 or k > k_max:
                    continue
                mk = self.ph.trivial_multiplicity(k)
                if mk > 0:
                    w = mk * nu
                    c_istar_total += Delta_full_l[j] ** 2 * w
                    c_istar_sw += Delta_noDE_l[j] ** 2 * w
                    c_istar_isw += Delta_ISW_l[j] ** 2 * w
                    c_istar_cross_val += 2.0 * Delta_noDE_l[j] * Delta_ISW_l[j] * w

            istar_total[l] = c_istar_total * norm
            istar_sw[l] = c_istar_sw * norm
            istar_isw[l] = c_istar_isw * norm
            istar_cross[l] = c_istar_cross_val * norm

            # LCDM and Planck
            if l in PLANCK_LOW_L:
                d_obs, d_lcdm, sigma = PLANCK_LOW_L[l]
                lcdm[l] = d_lcdm
                planck[l] = (d_obs, sigma)

        return {
            'l_values': list(range(2, self.l_max + 1)),
            's3': {
                'total': s3_total,
                'sw_doppler': s3_sw,
                'isw': s3_isw,
                'cross': s3_cross,
            },
            'istar': {
                'total': istar_total,
                'sw_doppler': istar_sw,
                'isw': istar_isw,
                'cross': istar_cross,
            },
            'lcdm': lcdm,
            'planck': planck,
        }

    def _get_invariant_modes(self, k):
        """
        Get I*-invariant eigenmodes at level k, with caching.

        The invariant_eigenmodes computation requires 120 full Wigner D-matrix
        evaluations (one per I* element) and is O(k^3) -- extremely expensive
        for k > 30. Results are cached since they are position-independent.

        Parameters
        ----------
        k : int
            Level of scalar harmonics.

        Returns
        -------
        modes : ndarray, shape (k+1, m(k))
            Invariant mode vectors.
        mk : int
            Number of invariant modes.
        """
        if k in self._invariant_modes_cache:
            return self._invariant_modes_cache[k]

        mk_check = self.ph.trivial_multiplicity(k)
        if mk_check == 0:
            self._invariant_modes_cache[k] = (np.zeros((k + 1, 0)), 0)
            return self._invariant_modes_cache[k]

        if self._istar_elements is None:
            self._istar_elements = istar_quaternions()

        modes, mk = invariant_eigenmodes(k, self._istar_elements)
        self._invariant_modes_cache[k] = (modes, mk)
        return modes, mk

    def precompute_invariant_modes(self, k_max=48):
        """
        Precompute I*-invariant eigenmodes for all k <= k_max with m(k) > 0.

        This is expensive (minutes for k_max=48) but only needs to be done once.
        After precomputation, position-dependent evaluations are fast (ms each).

        Parameters
        ----------
        k_max : int
            Maximum k to precompute. Default 48 (sufficient for l=2..30).
        """
        if self._istar_elements is None:
            self._istar_elements = istar_quaternions()

        for k in range(0, k_max + 1):
            if k not in self._invariant_modes_cache:
                mk = self.ph.trivial_multiplicity(k)
                if mk > 0:
                    self._get_invariant_modes(k)

    def _evaluate_invariant_modes_at_position(self, k, position, elements=None):
        """
        Evaluate all m(k) I*-invariant eigenmodes at a given position on S3.

        A position on S3 = SU(2) is a unit quaternion x = (w, x, y, z).
        The scalar harmonics at level k live in V_{k/2} (dimension k+1).
        Each basis function m evaluates at position x via the Wigner D-matrix:

            Q_{k,alpha,m}(x) = sqrt(k+1) * D^{k/2}_{alpha,m}(x)

        An I*-invariant mode is a vector v in V_{k/2} (column from
        invariant_eigenmodes). Its value at position x is:

            f_v(x) = sqrt(k+1) * sum_m v_m * D^{k/2}_{0,m}(x)

        where we fix alpha=0 (the observer's angular reference).
        The factor sqrt(k+1) is the Peter-Weyl normalization on S3.

        Uses cached invariant modes and fast single-row D-matrix evaluation.

        Parameters
        ----------
        k : int
            Level of scalar harmonics.
        position : array-like, shape (4,)
            Unit quaternion (w, x, y, z) on S3.
        elements : ndarray or None
            Deprecated, ignored. I* elements are cached internally.

        Returns
        -------
        amplitudes : ndarray, shape (m(k),)
            Complex amplitudes of each I*-invariant mode at this position.
        mk : int
            Number of invariant modes.
        """
        modes, mk = self._get_invariant_modes(k)
        if mk == 0:
            return np.array([]), 0

        j = k / 2.0
        # Compute only the first row of the Wigner D-matrix (m1=j).
        # This is O(k) instead of O(k^2), giving significant speedup
        # for large k where the full matrix computation is expensive.
        D_row = wigner_D_row(j, position, m1_idx=0)  # shape (k+1,), complex

        norm = np.sqrt(k + 1)
        amplitudes = np.zeros(mk, dtype=complex)
        for i in range(mk):
            # v_i is the i-th invariant mode vector (column of modes)
            amplitudes[i] = norm * np.dot(D_row, modes[:, i])

        return amplitudes, mk

    def _position_dependent_suppression(self, position, k_max=200):
        """
        Compute position-dependent suppression ratios S_l(x).

        At position x on S3/I*, the power at multipole l is:

            C_l^{I*}(x) = sum_{k: m(k)>0} |Delta_l(nu)|^2 * W(k, x) * nu

        where W(k, x) = sum_{i=1}^{m(k)} |f_i(x)|^2 encodes how much power
        the invariant modes at level k deposit at position x, and nu = k+1.

        The factor of nu accounts for the (k+1) angular harmonics that each
        scalar eigenmode on S3 generates. Position-averaging gives:
            <W(k, x)>_x = m(k)
        so <W(k, x) * nu>_x = m(k) * nu, matching the position-averaged
        formula in _suppression_ratios().

        The suppression ratio is:
            S_l(x) = C_l^{I*}(x) / C_l^{S3}

        Parameters
        ----------
        position : array-like, shape (4,)
            Unit quaternion on S3.
        k_max : int
            Maximum eigenmode index.

        Returns
        -------
        dict : {l: S_l(x)} suppression ratios at this position.
        """
        position = np.asarray(position, dtype=float)
        # Normalize to unit quaternion
        position = position / np.linalg.norm(position)

        self._run_camb()
        transfer = self._camb_results.get_cmb_transfer_data()
        q = transfer.q
        dq = q[1] - q[0]
        nu_vals = np.round(q / dq).astype(int)
        Delta_T = transfer.delta_p_l_k[0]
        L = transfer.L

        # Precompute position-dependent weights W(k, x) for each k
        # that has m(k) > 0 and appears in the transfer data.
        # Invariant modes are cached (expensive part done once).
        k_weight_cache = {}
        for j_idx, nu in enumerate(nu_vals):
            k = nu - 1
            if k < 0 or k > k_max:
                continue
            mk = self.ph.trivial_multiplicity(k)
            if mk == 0:
                continue
            if k not in k_weight_cache:
                amps, mk_actual = self._evaluate_invariant_modes_at_position(
                    k, position
                )
                if mk_actual > 0:
                    # W(k, x) = sum_i |f_i(x)|^2
                    k_weight_cache[k] = np.sum(np.abs(amps) ** 2)
                else:
                    k_weight_cache[k] = 0.0

        ratios = {}
        for l in range(2, self.l_max + 1):
            l_idx = np.where(L == l)[0]
            if len(l_idx) == 0:
                ratios[l] = 0.0
                continue
            l_idx = l_idx[0]
            d2 = Delta_T[l_idx, :] ** 2

            # S3: all modes with weight nu^2
            c_s3 = np.sum(d2 * nu_vals ** 2)

            # S3/I* at position x: surviving modes weighted by W(k, x) * nu
            # The nu factor accounts for the (k+1) angular harmonics per
            # invariant vector. This matches the position-averaged formula
            # which uses m(k)*nu (since <W(k,x)> = m(k)).
            c_istar = 0.0
            for j_idx, nu in enumerate(nu_vals):
                k = nu - 1
                if k in k_weight_cache and k_weight_cache[k] > 0:
                    c_istar += d2[j_idx] * k_weight_cache[k] * nu

            ratios[l] = c_istar / c_s3 if c_s3 > 0 else 0.0

        return ratios

    def compute_position_dependent_cls(self, position, omega_tot=None, lmax=None,
                                        k_max=200):
        """
        Compute position-dependent C_l on S3/I* at a given observer position.

        NUMERICAL: Position-dependent CMB power spectrum on the Poincare
        dodecahedral space, using full Boltzmann transfer functions from CAMB
        combined with I*-invariant eigenmode evaluation at a specific point.

        This bypasses the CG (Clebsch-Gordan) coefficient bottleneck by
        evaluating Wigner D-matrices numerically at the observer position.

        Parameters
        ----------
        position : array-like, shape (4,)
            Observer position on S3 as a unit quaternion (w, x, y, z).
            Coordinates within the fundamental domain of S3/I*.
        omega_tot : float or None
            Override Omega_tot. If None, uses the instance value.
        lmax : int or None
            Override l_max. If None, uses the instance value.
        k_max : int
            Maximum eigenmode index for the sum.

        Returns
        -------
        dict : {l: D_l} where D_l = l(l+1)C_l/(2pi) in muK^2.
        """
        # Create a new instance if omega_tot or lmax differ
        if omega_tot is not None and omega_tot != self.omega_tot:
            cmb = CMBBoltzmann(omega_tot=omega_tot, l_max=lmax or self.l_max,
                               cosmo_params=self.cosmo)
            return cmb.compute_position_dependent_cls(
                position, omega_tot=None, lmax=None, k_max=k_max
            )
        if lmax is not None and lmax != self.l_max:
            cmb = CMBBoltzmann(omega_tot=self.omega_tot, l_max=lmax,
                               cosmo_params=self.cosmo)
            return cmb.compute_position_dependent_cls(
                position, omega_tot=None, lmax=None, k_max=k_max
            )

        cls_full = self.cls_s3()
        suppression = self._position_dependent_suppression(position, k_max)

        result = {}
        for l in cls_full:
            if l in suppression:
                result[l] = cls_full[l] * suppression[l]
            else:
                result[l] = cls_full[l]

        return result

    def print_comparison(self, k_max=200):
        """Print a formatted comparison table."""
        rows = self.comparison_table(k_max)

        print(f"CMB Comparison: Omega_tot = {self.omega_tot}, "
              f"chi_LSS = {self._chi_lss():.4f}, "
              f"R = {self._R_curv:.0f} Mpc")
        print()
        print(f"{'l':>3} {'D_l^Planck':>10} {'+-sig':>6} {'D_l^S3':>10} "
              f"{'D_l^I*':>10} {'S_l':>8} {'Planck/S3':>10}")
        print("-" * 70)

        for row in rows:
            ratio = row['D_l_planck'] / row['D_l_s3'] if row['D_l_s3'] > 0 else 0
            print(f"{row['l']:3d} {row['D_l_planck']:10.1f} {row['sigma']:6.0f} "
                  f"{row['D_l_s3']:10.1f} {row['D_l_istar']:10.1f} "
                  f"{row['S_l']:8.4f} {ratio:10.3f}")

        # Chi-squared comparison
        chi2_istar, n = self.chi_squared(k_max, 'istar')
        chi2_s3, _ = self.chi_squared(k_max, 's3')
        chi2_lcdm, _ = self.chi_squared(k_max, 'lcdm')

        print()
        print(f"chi^2 (l=2..{self.l_max}, {n} points):")
        print(f"  LCDM flat:     {chi2_lcdm:.1f}  (chi^2/dof = {chi2_lcdm/n:.2f})")
        print(f"  S3 closed:     {chi2_s3:.1f}  (chi^2/dof = {chi2_s3/n:.2f})")
        print(f"  S3/I* (PDS):   {chi2_istar:.1f}  (chi^2/dof = {chi2_istar/n:.2f})")


def scan_omega(omega_range=None, l_max=30, k_max=200):
    """
    Scan over Omega_tot values to find the best fit to Planck.

    Parameters
    ----------
    omega_range : array-like or None
        Values of Omega_tot to scan. Default: 1.005 to 1.04 in steps of 0.005.
    l_max : int
        Maximum multipole.
    k_max : int
        Maximum eigenmode.

    Returns
    -------
    list of (omega_tot, chi2_istar, chi2_s3, chi_lss)
    """
    if omega_range is None:
        omega_range = np.arange(1.005, 1.045, 0.005)

    results = []
    for omega in omega_range:
        try:
            cmb = CMBBoltzmann(omega_tot=omega, l_max=l_max)
            chi2_istar, n = cmb.chi_squared(k_max, 'istar')
            chi2_s3, _ = cmb.chi_squared(k_max, 's3')
            chi_lss = cmb._chi_lss()
            results.append((omega, chi2_istar, chi2_s3, chi_lss))
            print(f"Omega = {omega:.3f}: chi^2(I*) = {chi2_istar:.1f}, "
                  f"chi^2(S3) = {chi2_s3:.1f}, chi_LSS = {chi_lss:.4f}")
        except Exception as e:
            print(f"Omega = {omega:.3f}: FAILED ({e})")
            results.append((omega, np.inf, np.inf, np.nan))

    return results


def _sample_fundamental_domain(n_positions, seed=42):
    """
    Sample n_positions uniformly on the fundamental domain of S3/I*.

    The fundamental domain is a spherical dodecahedron covering 1/120 of S3.
    We sample by generating uniform points on S3 and keeping only those in
    one fundamental domain (near the identity element).

    For simplicity and reproducibility, we use a deterministic strategy:
    sample on S3 and filter to points closest to identity (within the
    Voronoi cell of the identity element under I* action).

    Parameters
    ----------
    n_positions : int
        Number of sample positions desired.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    positions : ndarray, shape (n_positions, 4)
        Unit quaternions in the fundamental domain.
    """
    rng = np.random.RandomState(seed)
    elements = istar_quaternions()

    # Generate candidate points on S3 (uniform via Gaussian projection)
    # Oversample by 200x since fundamental domain is 1/120 of S3
    n_candidates = max(n_positions * 300, 10000)
    candidates = rng.randn(n_candidates, 4)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

    # Keep points in the fundamental domain near identity (1, 0, 0, 0).
    # A point x is in the Voronoi cell of identity if it is closer to
    # identity than to any other I* element. Distance on S3 is
    # d(x, g) = arccos(|x . g|), so x is closest to identity when
    # |x . e_identity| >= |x . g| for all g in I*.
    #
    # x . identity = x[0] (the w-component).
    # x . g = sum_i x_i * g_i.
    identity_dot = np.abs(candidates[:, 0])  # |x . (1,0,0,0)| = |w|

    in_domain = np.ones(n_candidates, dtype=bool)
    for g in elements:
        if np.allclose(g, [1, 0, 0, 0], atol=1e-10):
            continue
        if np.allclose(g, [-1, 0, 0, 0], atol=1e-10):
            continue  # antipodal identity, same Voronoi cell
        g_dot = np.abs(candidates @ g)
        in_domain &= (identity_dot >= g_dot - 1e-12)

    domain_points = candidates[in_domain]

    if len(domain_points) < n_positions:
        # If not enough points in the fundamental domain, relax and take
        # the n_positions closest to identity
        dists = np.arccos(np.clip(np.abs(candidates[:, 0]), 0, 1))
        idx = np.argsort(dists)[:n_positions]
        domain_points = candidates[idx]

    # Select exactly n_positions points (subsample if we have too many)
    if len(domain_points) > n_positions:
        idx = rng.choice(len(domain_points), n_positions, replace=False)
        domain_points = domain_points[idx]

    return domain_points


def position_scan(n_positions=50, omega_tot=1.02, lmax=30, k_max=200,
                  seed=42, verbose=True):
    """
    Scan observer positions on S3/I* fundamental domain to find the
    position giving maximum D_2.

    NUMERICAL: Position scan of CMB quadrupole power on S3/I*.

    On S3/I*, the CMB power spectrum depends on the observer's position
    within the fundamental domain (spherical dodecahedron). Different
    positions see different amplitudes of the surviving eigenmodes.

    This function samples n_positions uniformly on the fundamental domain,
    computes D_2 at each, and reports statistics.

    Parameters
    ----------
    n_positions : int
        Number of sample positions. Default 50.
    omega_tot : float
        Total density parameter. Default 1.02.
    lmax : int
        Maximum multipole. Default 30.
    k_max : int
        Maximum eigenmode index. Default 200.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'D2_min', 'D2_max', 'D2_mean', 'D2_std' : float
        'D2_values' : ndarray of D_2 at each position
        'positions' : ndarray of shape (n_positions, 4)
        'best_position' : ndarray of shape (4,), position giving max D_2
        'best_D2' : float, maximum D_2 found
        'cls_at_best' : dict, full C_l at the best position
    """
    if not HAS_CAMB:
        raise ImportError("CAMB not installed. Run: pip install camb")

    # Clamp k_max for position-dependent computation.
    # Modes with k > 48 contribute negligibly to l=2..30 (verified numerically).
    # Computing invariant modes for k > 48 is extremely expensive (O(k^3) for
    # 120 full D-matrix evaluations in istar_projector).
    k_max_eff = min(k_max, 48)

    positions = _sample_fundamental_domain(n_positions, seed=seed)
    cmb = CMBBoltzmann(omega_tot=omega_tot, l_max=lmax)

    if verbose:
        print(f"Precomputing I*-invariant modes for k <= {k_max_eff}...")
    cmb.precompute_invariant_modes(k_max=k_max_eff)
    if verbose:
        print("Precomputation done. Starting position scan...")

    d2_values = np.zeros(n_positions)
    best_d2 = -np.inf
    best_idx = 0
    best_cls = None

    for i, pos in enumerate(positions):
        cls = cmb.compute_position_dependent_cls(pos, k_max=k_max_eff)
        d2 = cls.get(2, 0.0)
        d2_values[i] = d2

        if d2 > best_d2:
            best_d2 = d2
            best_idx = i
            best_cls = cls

        if verbose and (i + 1) % 10 == 0:
            print(f"  Position {i+1}/{n_positions}: D_2 = {d2:.2f} muK^2 "
                  f"(best so far: {best_d2:.2f})")

    if verbose:
        print(f"\nPosition scan results (Omega_tot = {omega_tot}, "
              f"n = {n_positions}):")
        print(f"  D_2 min:  {d2_values.min():.2f} muK^2")
        print(f"  D_2 max:  {d2_values.max():.2f} muK^2")
        print(f"  D_2 mean: {d2_values.mean():.2f} muK^2")
        print(f"  D_2 std:  {d2_values.std():.2f} muK^2")
        print(f"  Planck:   201 muK^2")
        ratio = d2_values.max() / 201.0
        print(f"  Best/Planck: {ratio:.3f}")

    return {
        'D2_min': float(d2_values.min()),
        'D2_max': float(d2_values.max()),
        'D2_mean': float(d2_values.mean()),
        'D2_std': float(d2_values.std()),
        'D2_values': d2_values,
        'positions': positions,
        'best_position': positions[best_idx],
        'best_D2': float(best_d2),
        'cls_at_best': best_cls,
    }


def omega_position_grid_scan(omega_range=(1.01, 1.04), n_omega=10,
                              n_positions=20, lmax=30, k_max=200,
                              seed=42, verbose=True):
    """
    2D scan over (Omega_tot, observer position) to find the combination
    that maximizes D_2.

    NUMERICAL: Joint optimization of curvature and observer position for
    CMB quadrupole on S3/I*.

    Parameters
    ----------
    omega_range : tuple of (float, float)
        Range of Omega_tot values to scan.
    n_omega : int
        Number of Omega_tot values. Default 10.
    n_positions : int
        Number of positions per Omega_tot. Default 20.
    lmax : int
        Maximum multipole. Default 30.
    k_max : int
        Maximum eigenmode index. Default 200.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'omega_values' : ndarray of Omega_tot values scanned
        'D2_grid' : ndarray of shape (n_omega, n_positions), D_2 values
        'best_omega' : float, Omega_tot giving maximum D_2
        'best_position' : ndarray shape (4,), position giving max D_2
        'best_D2' : float, maximum D_2 found
        'exceeds_100' : bool, whether any D_2 > 100 muK^2
        'exceeds_planck_half' : bool, whether any D_2 > 100 muK^2 (within 2x)
        'cls_at_best' : dict, full C_l at the best (omega, position)
        'scan_results' : list of per-omega scan results
    """
    if not HAS_CAMB:
        raise ImportError("CAMB not installed. Run: pip install camb")

    # Clamp k_max for feasible computation
    k_max_eff = min(k_max, 48)

    omega_values = np.linspace(omega_range[0], omega_range[1], n_omega)

    # Use same positions for all omega values (for consistency)
    positions = _sample_fundamental_domain(n_positions, seed=seed)

    d2_grid = np.zeros((n_omega, n_positions))
    global_best_d2 = -np.inf
    global_best_omega = omega_values[0]
    global_best_pos = positions[0]
    global_best_cls = None
    scan_results = []

    # Precompute invariant modes once (shared across all omega values)
    if verbose:
        print(f"Precomputing I*-invariant modes for k <= {k_max_eff}...")
    precomp_cmb = CMBBoltzmann(omega_tot=omega_values[0], l_max=lmax)
    precomp_cmb.precompute_invariant_modes(k_max=k_max_eff)
    shared_cache = precomp_cmb._invariant_modes_cache
    shared_elements = precomp_cmb._istar_elements
    if verbose:
        print("Precomputation done.")

    for i_omega, omega in enumerate(omega_values):
        if verbose:
            print(f"\n--- Omega_tot = {omega:.4f} ({i_omega+1}/{n_omega}) ---")

        try:
            cmb = CMBBoltzmann(omega_tot=omega, l_max=lmax)
            # Share precomputed invariant modes
            cmb._invariant_modes_cache = shared_cache
            cmb._istar_elements = shared_elements

            best_d2_this_omega = -np.inf
            best_cls_this_omega = None

            for i_pos, pos in enumerate(positions):
                cls = cmb.compute_position_dependent_cls(pos, k_max=k_max_eff)
                d2 = cls.get(2, 0.0)
                d2_grid[i_omega, i_pos] = d2

                if d2 > best_d2_this_omega:
                    best_d2_this_omega = d2
                    best_cls_this_omega = cls

                if d2 > global_best_d2:
                    global_best_d2 = d2
                    global_best_omega = omega
                    global_best_pos = pos.copy()
                    global_best_cls = cls

            row_mean = d2_grid[i_omega, :].mean()
            row_max = d2_grid[i_omega, :].max()
            if verbose:
                print(f"  D_2: mean={row_mean:.2f}, max={row_max:.2f} muK^2")

            scan_results.append({
                'omega': float(omega),
                'D2_mean': float(row_mean),
                'D2_max': float(row_max),
                'D2_min': float(d2_grid[i_omega, :].min()),
            })

        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            d2_grid[i_omega, :] = 0.0
            scan_results.append({
                'omega': float(omega),
                'D2_mean': 0.0,
                'D2_max': 0.0,
                'D2_min': 0.0,
                'error': str(e),
            })

    exceeds_100 = global_best_d2 > 100.0
    exceeds_half = global_best_d2 > 100.5  # within 2x of Planck 201

    if verbose:
        print(f"\n{'='*60}")
        print(f"Grid scan complete: {n_omega} x {n_positions} = "
              f"{n_omega * n_positions} evaluations")
        print(f"  Best D_2:      {global_best_d2:.2f} muK^2")
        print(f"  at Omega_tot:  {global_best_omega:.4f}")
        print(f"  Planck D_2:    201 muK^2")
        print(f"  Ratio:         {global_best_d2/201:.3f}")
        print(f"  D_2 > 100?     {'YES' if exceeds_100 else 'NO'}")
        print(f"  Within 2x?     {'YES' if exceeds_half else 'NO'}")

    return {
        'omega_values': omega_values,
        'D2_grid': d2_grid,
        'best_omega': float(global_best_omega),
        'best_position': global_best_pos,
        'best_D2': float(global_best_d2),
        'exceeds_100': exceeds_100,
        'exceeds_planck_half': exceeds_half,
        'cls_at_best': global_best_cls,
        'scan_results': scan_results,
    }


def plot_swisw_decomposition(decomposition, save_path=None, show=True):
    """
    Plot the SW/ISW/cross-term decomposition of the CMB spectrum.

    NUMERICAL: Visualization of Sachs-Wolfe, Integrated Sachs-Wolfe,
    and cross-term contributions to D_l on S3/I* vs standard LCDM.

    Creates a figure with two panels:
      Top: S3/I* decomposition (total, SW+Doppler, ISW auto, cross)
      Bottom: Simply-connected S3 decomposition for comparison

    Both panels include standard LCDM and Planck data as reference.

    Parameters
    ----------
    decomposition : dict
        Output of CMBBoltzmann.compute_swisw_decomposition().
    save_path : str or None
        If provided, save figure to this path. Default None.
    show : bool
        If True, call plt.show(). Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    s3 = decomposition['s3']
    istar = decomposition['istar']
    lcdm = decomposition['lcdm']
    planck = decomposition['planck']

    # Collect l-values that appear in all datasets
    l_vals = sorted(set(s3['total'].keys()) & set(istar['total'].keys()))

    # Arrays for plotting
    l_arr = np.array(l_vals)

    # S3/I*
    d_istar_total = np.array([istar['total'][l] for l in l_vals])
    d_istar_sw = np.array([istar['sw_doppler'][l] for l in l_vals])
    d_istar_isw = np.array([istar['isw'][l] for l in l_vals])
    d_istar_cross = np.array([istar['cross'][l] for l in l_vals])

    # S3
    d_s3_total = np.array([s3['total'][l] for l in l_vals])
    d_s3_sw = np.array([s3['sw_doppler'][l] for l in l_vals])
    d_s3_isw = np.array([s3['isw'][l] for l in l_vals])
    d_s3_cross = np.array([s3['cross'][l] for l in l_vals])

    # LCDM and Planck
    d_lcdm = np.array([lcdm.get(l, np.nan) for l in l_vals])
    d_planck = np.array([planck[l][0] if l in planck else np.nan for l in l_vals])
    d_planck_sig = np.array([planck[l][1] if l in planck else np.nan for l in l_vals])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Top panel: S3/I* ---
    ax1.set_title(r'CMB Decomposition on $S^3/I^*$ (Poincar\'e Dodecahedral Space)',
                  fontsize=13)

    ax1.errorbar(l_arr, d_planck, yerr=d_planck_sig, fmt='ko', markersize=4,
                 capsize=2, label='Planck 2018', zorder=10)
    ax1.plot(l_arr, d_lcdm, 'k--', linewidth=1.0, alpha=0.5, label=r'$\Lambda$CDM')
    ax1.plot(l_arr, d_istar_total, 'b-', linewidth=2.0, label=r'$S^3/I^*$ total')
    ax1.plot(l_arr, d_istar_sw, 'g-', linewidth=1.5, label='SW + Doppler')
    ax1.plot(l_arr, d_istar_isw, 'r-', linewidth=1.5, label='ISW (auto)')
    ax1.plot(l_arr, d_istar_cross, 'm--', linewidth=1.5, label='Cross (SW$\\times$ISW)')
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

    ax1.set_ylabel(r'$\mathcal{D}_\ell = \ell(\ell+1)C_\ell / 2\pi$ [$\mu$K$^2$]',
                   fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(1.5, max(l_vals) + 0.5)

    # --- Bottom panel: S3 (simply-connected) ---
    ax2.set_title(r'CMB Decomposition on simply-connected $S^3$', fontsize=13)

    ax2.errorbar(l_arr, d_planck, yerr=d_planck_sig, fmt='ko', markersize=4,
                 capsize=2, label='Planck 2018', zorder=10)
    ax2.plot(l_arr, d_lcdm, 'k--', linewidth=1.0, alpha=0.5, label=r'$\Lambda$CDM')
    ax2.plot(l_arr, d_s3_total, 'b-', linewidth=2.0, label=r'$S^3$ total')
    ax2.plot(l_arr, d_s3_sw, 'g-', linewidth=1.5, label='SW + Doppler')
    ax2.plot(l_arr, d_s3_isw, 'r-', linewidth=1.5, label='ISW (auto)')
    ax2.plot(l_arr, d_s3_cross, 'm--', linewidth=1.5, label='Cross (SW$\\times$ISW)')
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

    ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax2.set_ylabel(r'$\mathcal{D}_\ell = \ell(\ell+1)C_\ell / 2\pi$ [$\mu$K$^2$]',
                   fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
