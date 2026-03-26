"""
Angular two-point correlation function C(theta) and S(1/2) statistic.

Implements the S(60) angular correlation statistic used by Aurich et al.
(2005, 2007) for comparing CMB topological predictions with Planck data.

NUMERICAL: S(60) quantifies large-angle CMB correlations. Planck data
  shows anomalously low S(60) compared to flat LCDM predictions.
  S3/I* naturally produces low S(60) because the spectral desert
  (m(k)=0 for k=1..11) suppresses large-angle correlations.

Definitions:
  C(theta) = sum_l (2l+1)/(4*pi) * C_l * P_l(cos theta)
  S(theta_min) = integral_{theta_min}^{180} [C(theta)]^2 sin(theta) d(theta)
  S(1/2) = S(60 degrees)

where C_l is the raw angular power spectrum (NOT D_l = l(l+1)C_l/(2pi)),
P_l are Legendre polynomials, and the integral is over the angle range
where topology has the strongest effect.

References:
  - Spergel et al., ApJ 583, 553 (2003): S(1/2) definition
  - Aurich, Lustig, Steiner, CQG 22, 2061 (2005): S3/I* fits with S(60)
  - Copi, Huterer, Schwarz, Starkman, PRD 75, 023507 (2007): S(1/2) anomaly
  - Planck 2018 results VII, A&A 641, A7 (2020): low-l anomalies
"""

import numpy as np
from scipy.special import legendre
from scipy.integrate import quad


def d_l_to_c_l(d_l_dict):
    """
    Convert D_l = l(l+1)C_l/(2*pi) to raw C_l.

    Parameters
    ----------
    d_l_dict : dict
        {l: D_l} where D_l = l(l+1)*C_l/(2*pi) in muK^2.

    Returns
    -------
    dict : {l: C_l} in muK^2.
    """
    c_l = {}
    for l, d_l in d_l_dict.items():
        if l >= 2:
            c_l[l] = d_l * 2.0 * np.pi / (l * (l + 1))
    return c_l


def angular_correlation_function(C_l, theta_array, l_max=None):
    """
    Compute the angular two-point correlation function C(theta).

    C(theta) = sum_{l=2}^{l_max} (2l+1)/(4*pi) * C_l * P_l(cos theta)

    Parameters
    ----------
    C_l : dict or array-like
        If dict: {l: C_l_value} (raw C_l, NOT D_l).
        If array: C_l indexed by l (C_l[0]=monopole, C_l[1]=dipole, ...).
    theta_array : array-like
        Angles in degrees at which to evaluate C(theta).
    l_max : int or None
        Maximum multipole to include. If None, uses max key in C_l.

    Returns
    -------
    numpy.ndarray
        C(theta) values at each angle, in muK^2.
    """
    theta_rad = np.deg2rad(np.asarray(theta_array, dtype=float))
    cos_theta = np.cos(theta_rad)

    # Convert to dict if array-like
    if isinstance(C_l, dict):
        cl_dict = C_l
    else:
        cl_arr = np.asarray(C_l)
        cl_dict = {l: cl_arr[l] for l in range(len(cl_arr)) if l >= 2}

    if l_max is None:
        l_max = max(cl_dict.keys()) if cl_dict else 2

    result = np.zeros_like(cos_theta, dtype=float)

    for l in range(2, l_max + 1):
        if l not in cl_dict:
            continue
        c_l_val = cl_dict[l]
        if c_l_val == 0:
            continue
        weight = (2 * l + 1) / (4.0 * np.pi)
        # Evaluate Legendre polynomial at all cos(theta) values
        P_l = legendre(l)
        result += weight * c_l_val * P_l(cos_theta)

    return result


def s_half_statistic(C_l, theta_min=60, l_max=None, n_points=500):
    """
    Compute the S(theta_min) statistic (S(1/2) when theta_min=60).

    S(theta_min) = integral_{theta_min}^{180} [C(theta)]^2 sin(theta) d(theta)

    This is the integral of the squared angular correlation function
    over the large-angle range, as defined by Spergel et al. (2003).

    Parameters
    ----------
    C_l : dict
        {l: C_l_value} (raw C_l, NOT D_l). Units: muK^2.
    theta_min : float
        Minimum angle in degrees. Default 60 (the S(1/2) statistic).
    l_max : int or None
        Maximum multipole. If None, uses max key in C_l.
    n_points : int
        Number of quadrature points for the integration. Default 500.

    Returns
    -------
    float
        S(theta_min) in muK^4 * sr (steradians).
    """
    # Use scipy.integrate.quad for accurate integration
    theta_min_rad = np.deg2rad(theta_min)
    theta_max_rad = np.pi

    def integrand(theta_rad):
        theta_deg = np.rad2deg(theta_rad)
        c_theta = angular_correlation_function(C_l, [theta_deg], l_max=l_max)
        return c_theta[0] ** 2 * np.sin(theta_rad)

    result, error = quad(integrand, theta_min_rad, theta_max_rad,
                         limit=100, epsrel=1e-8)
    return result


def s_half_from_d_l(D_l, theta_min=60, l_max=None, n_points=500):
    """
    Compute S(theta_min) directly from D_l = l(l+1)C_l/(2*pi).

    Convenience wrapper that converts D_l to C_l first.

    Parameters
    ----------
    D_l : dict
        {l: D_l_value} in muK^2.
    theta_min : float
        Minimum angle in degrees. Default 60.
    l_max : int or None
        Maximum multipole.
    n_points : int
        Quadrature points.

    Returns
    -------
    float
        S(theta_min) in muK^4 * sr.
    """
    c_l = d_l_to_c_l(D_l)
    return s_half_statistic(c_l, theta_min=theta_min, l_max=l_max,
                            n_points=n_points)


def s_half_comparison(C_l_model, C_l_lcdm, theta_min=60, l_max=None):
    """
    Compare S(theta_min) between a topological model and flat LCDM.

    Parameters
    ----------
    C_l_model : dict
        {l: C_l} for the topological model (e.g., S3/I*). Raw C_l.
    C_l_lcdm : dict
        {l: C_l} for flat LCDM. Raw C_l.
    theta_min : float
        Minimum angle in degrees. Default 60.
    l_max : int or None
        Maximum multipole.

    Returns
    -------
    dict with keys:
        's_model': S(theta_min) for the model
        's_lcdm': S(theta_min) for LCDM
        'ratio': S_model / S_lcdm
        'theta_min': the angle used
    """
    s_model = s_half_statistic(C_l_model, theta_min=theta_min, l_max=l_max)
    s_lcdm = s_half_statistic(C_l_lcdm, theta_min=theta_min, l_max=l_max)

    ratio = s_model / s_lcdm if s_lcdm > 0 else float('inf')

    return {
        's_model': s_model,
        's_lcdm': s_lcdm,
        'ratio': ratio,
        'theta_min': theta_min,
    }


def plot_angular_correlation(C_l_model, C_l_lcdm, C_l_planck=None,
                             l_max=None, theta_range=(0, 180),
                             n_points=200, save_path=None):
    """
    Plot the angular correlation function C(theta) for comparison.

    Parameters
    ----------
    C_l_model : dict
        {l: C_l} for the topological model (raw C_l).
    C_l_lcdm : dict
        {l: C_l} for flat LCDM (raw C_l).
    C_l_planck : dict or None
        {l: C_l} from Planck observed data (raw C_l). Optional.
    l_max : int or None
        Maximum multipole.
    theta_range : tuple
        (theta_min, theta_max) in degrees. Default (0, 180).
    n_points : int
        Number of theta points for the plot.
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    c_theta_model = angular_correlation_function(C_l_model, theta, l_max=l_max)
    c_theta_lcdm = angular_correlation_function(C_l_lcdm, theta, l_max=l_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: C(theta)
    ax1.plot(theta, c_theta_model, 'b-', linewidth=2, label=r'S$^3$/I*')
    ax1.plot(theta, c_theta_lcdm, 'r--', linewidth=2, label=r'$\Lambda$CDM')
    if C_l_planck is not None:
        c_theta_planck = angular_correlation_function(
            C_l_planck, theta, l_max=l_max
        )
        ax1.plot(theta, c_theta_planck, 'ko', markersize=3, alpha=0.5,
                 label='Planck data')

    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axvline(x=60, color='gray', linestyle=':', linewidth=0.8,
                label=r'$\theta = 60°$')
    ax1.set_xlabel(r'$\theta$ [degrees]', fontsize=12)
    ax1.set_ylabel(r'$C(\theta)$ [$\mu$K$^2$]', fontsize=12)
    ax1.set_title('Angular Correlation Function', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(theta_range)

    # Right panel: [C(theta)]^2 * sin(theta) (the S(1/2) integrand)
    theta_rad = np.deg2rad(theta)
    integrand_model = c_theta_model ** 2 * np.sin(theta_rad)
    integrand_lcdm = c_theta_lcdm ** 2 * np.sin(theta_rad)

    ax2.plot(theta, integrand_model, 'b-', linewidth=2, label=r'S$^3$/I*')
    ax2.plot(theta, integrand_lcdm, 'r--', linewidth=2, label=r'$\Lambda$CDM')
    ax2.axvline(x=60, color='gray', linestyle=':', linewidth=0.8,
                label=r'$\theta = 60°$')

    # Shade the S(60) integration region
    mask = theta >= 60
    ax2.fill_between(theta[mask], integrand_model[mask], alpha=0.15, color='b')
    ax2.fill_between(theta[mask], integrand_lcdm[mask], alpha=0.15, color='r')

    ax2.set_xlabel(r'$\theta$ [degrees]', fontsize=12)
    ax2.set_ylabel(r'$[C(\theta)]^2 \sin\theta$ [$\mu$K$^4$]', fontsize=12)
    ax2.set_title(r'$S(1/2)$ Integrand', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(theta_range)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ======================================================================
# Convenience: Planck observed C_l as raw C_l (from D_l data)
# ======================================================================

def planck_observed_c_l():
    """
    Return Planck 2018 observed C_l (raw, NOT D_l) for l=2..30.

    Converts from D_l = l(l+1)C_l/(2*pi) stored in cmb_boltzmann.PLANCK_LOW_L.

    Returns
    -------
    dict : {l: C_l} in muK^2.
    """
    from .cmb_boltzmann import PLANCK_LOW_L
    d_l = {l: vals[0] for l, vals in PLANCK_LOW_L.items()}
    return d_l_to_c_l(d_l)


def planck_lcdm_c_l():
    """
    Return Planck 2018 best-fit LCDM C_l (raw) for l=2..30.

    Returns
    -------
    dict : {l: C_l} in muK^2.
    """
    from .cmb_boltzmann import PLANCK_LOW_L
    d_l = {l: vals[1] for l, vals in PLANCK_LOW_L.items()}
    return d_l_to_c_l(d_l)
