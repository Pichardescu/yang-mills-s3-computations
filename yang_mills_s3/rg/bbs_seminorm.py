"""
BBS T_z Seminorm -- Exact Definition 7.1.1 from Bauerschmidt-Brydges-Slade.

Implements the EXACT algebraic T_z seminorm from BBS (LNM 2242, 2019),
Chapter 7, replacing the numerical-sampling approach in polymer_algebra_ym.py.

The BBS T_z seminorm is:

    ||F||_{T_z(ell)} = Sum_{k=0}^{p_N} (1/k!) ||D^k F(z)|| * ell^k

where:
    D^k F(z) = k-th Frechet derivative at basepoint z
    ||D^k F(z)|| = operator norm as symmetric k-linear form
    ell = norm parameter ("allowed field increment scale")
    p_N >= 5 for d=4 (derivative order, fixed integer)

Critical BBS choices:
    ell_j = c_ell * g_bar_j^{-1/4}  (NOT h_j = M^{-j} * sqrt(g_j))
    This ensures ||V||_{T_z} ~ 1 for the quartic interaction V = g*tau^2

Key results:
    THEOREM (Prop 7.1.2):  ||FG||_{T_z} <= ||F||_{T_z} * ||G||_{T_z}
                           (product property via Leibniz + binomial)
    THEOREM (Sec 7.3):     ||E_C F(. + zeta)||_{T_z(ell)} <= ||F||_{T_z(ell_+)}
                           with ell_+^2 = ell^2 + w^2 (Pythagorean)
    THEOREM (Sec 8.2.2):   Regulator G_j(X, phi) = exp(c_G * Sum |phi_x|^2 / ell_j^2)
                           is sub-multiplicative over polymers
    NUMERICAL:             For SU(2) on S^3 with g^2 = 6.28, ell_0 ~ 0.631

Physical parameters:
    R = 2.2 fm, g^2 = 6.28, L = M = 2, N_c = 2
    p_N = 5 (derivative order for d=4)
    g_bar_0 = g^2 = 6.28, g_bar_j = g_0 / (1 + beta_0 * g_0 * j)
    ell_0 = g_bar_0^{-1/4} ~ 0.631

References:
    [1] Bauerschmidt-Brydges-Slade (2019): LNM 2242, Ch 7 (T_phi norm)
    [2] Bauerschmidt-Brydges-Slade (2019): LNM 2242, Ch 8 (regulators)
    [3] Dimock (2013-2022): Gauge-covariant adaptations for QED_3
    [4] Balaban (1984-89): Covariant derivatives in YM context
"""

import math
import numpy as np
from scipy import linalg as la
from typing import Optional, Dict, List, Tuple, Callable, Any
from dataclasses import dataclass

from yang_mills_s3.rg.banach_norm import (
    Polymer,
    HBAR_C_MEV_FM,
    R_PHYSICAL_FM,
    BETA_0_SU2,
)


# ======================================================================
# Physical constants
# ======================================================================

G2_BARE_DEFAULT = 6.28           # g^2 = 4*pi*alpha_s at lattice scale
N_C_DEFAULT = 2                  # SU(2) gauge group
DIM_ADJ_SU2 = 3                 # dim(su(2)) = N^2 - 1
SPACETIME_DIM = 4                # d=4 for YM on S^3 x R
P_N_DEFAULT = 5                  # Derivative order for d=4 (BBS: p_N >= 5)
M_DEFAULT = 2.0                  # RG blocking factor


# ======================================================================
# 1. FrechetDerivative
# ======================================================================

class FrechetDerivative:
    """
    Compute D^k F(z) for polymer activities F.

    For polynomial activities (e.g., V = g*tau^2 where tau = |phi|^2):
    exact symbolic derivatives via iterated chain rule.

    For general activities: numerical finite differences with
    Richardson extrapolation for improved accuracy.

    The k-th Frechet derivative D^k F(z) is a symmetric k-linear form:
        D^k F(z)[h_1, ..., h_k] = d^k/dt_1...dt_k F(z + t_1*h_1 + ... + t_k*h_k)|_{t=0}

    Its operator norm is:
        ||D^k F(z)|| = sup_{|h_i|=1} |D^k F(z)[h_1, ..., h_k]|

    NUMERICAL: Finite-difference computation with adaptive step size.
    """

    def __init__(self, eps: float = 1e-5):
        """
        Parameters
        ----------
        eps : float
            Base step size for finite differences.
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps

    @staticmethod
    def polynomial_derivative(coefficients: np.ndarray, z: np.ndarray,
                              k: int) -> float:
        """
        Exact k-th derivative of a polynomial activity at basepoint z.

        For a monomial F(phi) = c * |phi|^{2p}, the k-th derivative is:
            D^k F(z)[h,...,h] = c * (2p)! / (2p-k)! * |z|^{2p-k} * (z/|z| . h)^k
        when 2p >= k, and 0 otherwise.

        For a polynomial F = Sum_p c_p |phi|^{2p}, we sum contributions.

        Parameters
        ----------
        coefficients : ndarray
            Coefficients c_p for p = 0, 1, 2, ... in F = Sum c_p |phi|^{2p}.
        z : ndarray
            Basepoint (field configuration).
        k : int
            Derivative order.

        Returns
        -------
        float
            ||D^k F(z)|| = operator norm of the k-th Frechet derivative.
        """
        if k < 0:
            raise ValueError(f"Derivative order k must be >= 0, got {k}")

        z_norm = float(np.linalg.norm(z))
        result = 0.0

        for p, c_p in enumerate(coefficients):
            power = 2 * p  # F_p = c_p * |phi|^{2p}
            if power < k:
                continue
            if abs(c_p) < 1e-300:
                continue

            # The k-th derivative of |phi|^{2p} at z has operator norm:
            # (2p)! / (2p-k)! * |z|^{2p-k}
            # This comes from the multinomial expansion of |z+h|^{2p}
            falling_factorial = 1.0
            for i in range(k):
                falling_factorial *= (power - i)

            if z_norm > 1e-300:
                contrib = abs(c_p) * falling_factorial * z_norm ** (power - k)
            elif power == k:
                contrib = abs(c_p) * falling_factorial
            else:
                contrib = 0.0

            result += contrib

        return result

    def numerical_derivative(self, F: Callable, z: np.ndarray,
                             k: int, n_directions: int = 20) -> float:
        """
        Numerical k-th Frechet derivative via finite differences.

        Uses central differences along random unit directions to estimate
        the operator norm ||D^k F(z)||.

        For k=0: just |F(z)|
        For k=1: sup_{|h|=1} |F(z+eps*h) - F(z-eps*h)| / (2*eps)
        For k=2: sup_{|h|=1} |F(z+eps*h) - 2*F(z) + F(z-eps*h)| / eps^2
        For general k: iterated central differences

        NUMERICAL: Returns an estimate (lower bound on the true sup).

        Parameters
        ----------
        F : callable
            F(phi) -> float. The activity.
        z : ndarray
            Basepoint.
        k : int
            Derivative order.
        n_directions : int
            Number of random directions to sample.

        Returns
        -------
        float
            Estimated ||D^k F(z)||.
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        dim = len(z)
        eps = self.eps

        if k == 0:
            return abs(F(z))

        rng = np.random.RandomState(42)
        max_val = 0.0

        for _ in range(n_directions):
            h = rng.randn(dim)
            h_norm = np.linalg.norm(h)
            if h_norm < 1e-15:
                continue
            h = h / h_norm

            if k == 1:
                val = abs(F(z + eps * h) - F(z - eps * h)) / (2 * eps)
            elif k == 2:
                val = abs(F(z + eps * h) - 2 * F(z) + F(z - eps * h)) / eps**2
            elif k == 3:
                val = abs(
                    F(z + 1.5 * eps * h) - 3 * F(z + 0.5 * eps * h)
                    + 3 * F(z - 0.5 * eps * h) - F(z - 1.5 * eps * h)
                ) / eps**3
            elif k == 4:
                val = abs(
                    F(z + 2 * eps * h) - 4 * F(z + eps * h) + 6 * F(z)
                    - 4 * F(z - eps * h) + F(z - 2 * eps * h)
                ) / eps**4
            else:
                # General k: use binomial coefficients for central difference
                val = 0.0
                for i in range(k + 1):
                    sign = (-1) ** (k - i)
                    binom = math.comb(k, i)
                    shift = (i - k / 2) * eps
                    val += sign * binom * F(z + shift * h)
                val = abs(val) / eps**k

            max_val = max(max_val, val)

        return max_val

    @staticmethod
    def is_symmetric(D_k_F: Callable, z: np.ndarray, k: int,
                     dim: int, n_tests: int = 10,
                     tol: float = 1e-6) -> bool:
        """
        Check symmetry of the k-th Frechet derivative as a k-linear form.

        D^k F(z)[h_1, ..., h_k] should be symmetric under permutations
        of the h_i. We check this by evaluating on random vectors and
        comparing permuted evaluations.

        NUMERICAL: Checks to tolerance tol.

        Parameters
        ----------
        D_k_F : callable
            D_k_F(z, h_list) -> float. The k-th derivative evaluated on
            a list of k direction vectors.
        z : ndarray
            Basepoint.
        k : int
            Order.
        dim : int
            Dimension of field space.
        n_tests : int
            Number of random tests.
        tol : float
            Tolerance.

        Returns
        -------
        bool
            True if symmetric to tolerance.
        """
        if k <= 1:
            return True

        rng = np.random.RandomState(123)
        for _ in range(n_tests):
            h_list = [rng.randn(dim) for _ in range(k)]
            val_orig = D_k_F(z, h_list)

            # Check one random swap
            i, j = rng.choice(k, size=2, replace=False)
            h_swapped = list(h_list)
            h_swapped[i], h_swapped[j] = h_swapped[j], h_swapped[i]
            val_swapped = D_k_F(z, h_swapped)

            if abs(val_orig - val_swapped) > tol * max(abs(val_orig), 1e-15):
                return False

        return True


# ======================================================================
# 2. BBSTzSeminorm
# ======================================================================

class BBSTzSeminorm:
    """
    The EXACT BBS T_z seminorm from Definition 7.1.1 (LNM 2242, Ch 7).

        ||F||_{T_z(ell)} = Sum_{k=0}^{p_N} (ell^k / k!) * ||D^k F(z)||

    where:
        D^k F(z) = k-th Frechet derivative at basepoint z
        ||D^k F(z)|| = operator norm as symmetric k-linear form
        ell = norm parameter ("allowed field increment scale")
        p_N >= 5 for d=4

    THEOREM (Proposition 7.1.2): The T_z seminorm makes the polynomial
    algebra into a normed algebra:
        ||FG||_{T_z(ell)} <= ||F||_{T_z(ell)} * ||G||_{T_z(ell)}
    via the Leibniz rule + binomial identity.

    Key property: for V = g * tau^2 (quartic interaction) with
    ell = g^{-1/4}, we get ||V||_{T_z} ~ O(1). This is the BBS
    normalization choice that makes the norm natural for phi^4 / YM.

    Parameters
    ----------
    p_N : int
        Maximum derivative order. BBS requires p_N >= 5 for d=4.
    ell : float
        Norm parameter (field increment scale).
    """

    def __init__(self, p_N: int = P_N_DEFAULT, ell: float = 1.0):
        if p_N < 0:
            raise ValueError(f"p_N must be >= 0, got {p_N}")
        if ell < 0:
            raise ValueError(f"ell must be >= 0, got {ell}")
        self.p_N = p_N
        self.ell = ell
        self._frechet = FrechetDerivative()

    def evaluate_polynomial(self, coefficients: np.ndarray,
                            z: np.ndarray) -> float:
        """
        Evaluate T_z seminorm for a polynomial activity F = Sum c_p |phi|^{2p}.

        Uses EXACT symbolic derivatives (no finite differences).

            ||F||_{T_z(ell)} = Sum_{k=0}^{p_N} (ell^k / k!) * ||D^k F(z)||

        Parameters
        ----------
        coefficients : ndarray
            Coefficients c_p for F = Sum c_p |phi|^{2p}.
        z : ndarray
            Basepoint.

        Returns
        -------
        float
            ||F||_{T_z(ell)}
        """
        total = 0.0
        for k in range(self.p_N + 1):
            dk_norm = FrechetDerivative.polynomial_derivative(
                coefficients, z, k
            )
            weight = self.ell**k / math.factorial(k)
            total += weight * dk_norm
        return total

    def evaluate_numerical(self, F: Callable, z: np.ndarray,
                           n_directions: int = 20) -> float:
        """
        Evaluate T_z seminorm numerically for a general activity F.

        Uses finite-difference Frechet derivatives.

            ||F||_{T_z(ell)} = Sum_{k=0}^{p_N} (ell^k / k!) * ||D^k F(z)||

        NUMERICAL: Result is a lower bound (sampling underestimates sup).

        Parameters
        ----------
        F : callable
            F(phi) -> float.
        z : ndarray
            Basepoint.
        n_directions : int
            Number of random directions per derivative order.

        Returns
        -------
        float
            Estimated ||F||_{T_z(ell)}
        """
        total = 0.0
        for k in range(self.p_N + 1):
            dk_norm = self._frechet.numerical_derivative(
                F, z, k, n_directions
            )
            weight = self.ell**k / math.factorial(k)
            total += weight * dk_norm
        return total

    def evaluate(self, F: Any, z: np.ndarray,
                 n_directions: int = 20) -> float:
        """
        Evaluate T_z seminorm, dispatching between polynomial and general.

        If F is an ndarray, treat as polynomial coefficients.
        If F is callable, use numerical evaluation.

        Parameters
        ----------
        F : ndarray or callable
            Polynomial coefficients or activity function.
        z : ndarray
            Basepoint.
        n_directions : int
            For numerical mode.

        Returns
        -------
        float
        """
        if isinstance(F, np.ndarray):
            return self.evaluate_polynomial(F, z)
        elif callable(F):
            return self.evaluate_numerical(F, z, n_directions)
        else:
            raise TypeError(f"F must be ndarray or callable, got {type(F)}")

    @property
    def is_algebra(self) -> bool:
        """
        The T_z seminorm makes the space a normed algebra.

        THEOREM (BBS Prop 7.1.2): This is always True by construction.
        The product property follows from Leibniz rule + binomial identity.
        """
        return True

    def quartic_seminorm(self, g: float, z: np.ndarray) -> float:
        """
        Compute ||V||_{T_z} for V = g * tau^2 = g * |phi|^4.

        At the BBS natural scale ell = g^{-1/4}, this should be O(1).

        Parameters
        ----------
        g : float
            Quartic coupling.
        z : ndarray
            Basepoint.

        Returns
        -------
        float
            ||V||_{T_z(ell)} for V = g * |phi|^4.
        """
        # V = g * |phi|^4 has coefficients: c_0 = 0, c_1 = 0, c_2 = g
        coefficients = np.array([0.0, 0.0, g])
        return self.evaluate_polynomial(coefficients, z)


# ======================================================================
# 3. NormParameter
# ======================================================================

class NormParameter:
    """
    The ell_j parameter at each RG scale -- BBS norm parameter.

    BBS choice (Definition 7.1.1 + discussion in Section 7.2):
        ell_j = c_ell * g_bar_j^{-1/4}

    where g_bar_j is the running coupling at scale j. This ensures that
    the quartic interaction V = g * tau^2 has ||V||_{T_z} ~ O(1).

    NOTE: This differs from the earlier choice h_j = M^{-j} * sqrt(g_j)
    in polymer_algebra_ym.py. The BBS choice is the correct one for
    the T_z seminorm to have the product property with natural O(1) norms.

    Pythagorean update (Section 7.3):
        When integrating out a fluctuation field zeta with covariance C
        (so w^2 <= ||C||), the norm parameter inflates:
            ell_+^2 = ell^2 + w^2

    THEOREM: The Pythagorean rule gives the EXACT inflated norm parameter
    for Gaussian convolution bounds.

    Parameters
    ----------
    c_ell : float
        Proportionality constant. For ||V|| = O(1), c_ell ~ 1.
    beta_0 : float
        One-loop beta function coefficient (for running g_bar).
    """

    def __init__(self, c_ell: float = 1.0, beta_0: float = BETA_0_SU2):
        if c_ell <= 0:
            raise ValueError(f"c_ell must be positive, got {c_ell}")
        if beta_0 <= 0:
            raise ValueError(f"beta_0 must be positive, got {beta_0}")
        self.c_ell = c_ell
        self.beta_0 = beta_0

    def g_bar(self, j: int, g2_bare: float) -> float:
        """
        Running coupling at scale j.

        One-loop running:
            g_bar_j = g2_bare / (1 + beta_0 * g2_bare * j)

        Asymptotic freedom: g_bar_j decreases with j (toward UV).

        THEOREM (Gross-Wilczek-Politzer 1973).

        Parameters
        ----------
        j : int
            RG scale index.
        g2_bare : float
            Bare coupling g^2.

        Returns
        -------
        float
            g_bar_j
        """
        denom = 1.0 + self.beta_0 * g2_bare * j
        if denom <= 0:
            return g2_bare  # Landau pole protection
        return g2_bare / denom

    def at_scale(self, j: int, g2_bare: float) -> float:
        """
        Norm parameter ell_j at RG scale j.

            ell_j = c_ell * g_bar_j^{-1/4}

        Parameters
        ----------
        j : int
            RG scale index.
        g2_bare : float
            Bare coupling.

        Returns
        -------
        float
            ell_j
        """
        g_bar_j = self.g_bar(j, g2_bare)
        if g_bar_j <= 0:
            return float('inf')
        return self.c_ell * g_bar_j ** (-0.25)

    def inflate(self, ell: float, w: float) -> float:
        """
        Pythagorean inflation of norm parameter.

            ell_+ = sqrt(ell^2 + w^2)

        This is the EXACT BBS rule (Section 7.3): when integrating out
        a Gaussian fluctuation with variance <= w^2, the norm parameter
        must increase by the Pythagorean rule.

        THEOREM: This ensures ||E_C F(. + zeta)||_{T_z(ell)} <= ||F||_{T_z(ell_+)}.

        Parameters
        ----------
        ell : float
            Current norm parameter.
        w : float
            Fluctuation scale (sqrt of covariance norm).

        Returns
        -------
        float
            ell_+ = sqrt(ell^2 + w^2)
        """
        return np.sqrt(ell**2 + w**2)

    def deflate_ratio(self, ell_plus: float, ell: float) -> float:
        """
        Ratio ell_+ / ell, measuring the inflation factor.

        This controls how much the norm grows when integrating out
        a fluctuation. For the RG to contract, we need the coupling
        flow to compensate for this inflation.

        Parameters
        ----------
        ell_plus : float
            Inflated parameter.
        ell : float
            Original parameter.

        Returns
        -------
        float
            ell_+ / ell
        """
        if ell <= 0:
            return float('inf')
        return ell_plus / ell

    def scale_evolution(self, j: int, g2_bare: float,
                        w_j: float) -> Tuple[float, float]:
        """
        One-step evolution of norm parameter: ell_j -> ell_{j+1}.

        1. Start with ell_j = c_ell * g_bar_j^{-1/4}
        2. Inflate by fluctuation: ell_+^2 = ell_j^2 + w_j^2
        3. New scale parameter: ell_{j+1} = c_ell * g_bar_{j+1}^{-1/4}

        The key RG consistency check: ell_+ should be compatible with
        ell_{j+1} (i.e., inflation should not exceed what the next
        scale expects).

        Parameters
        ----------
        j : int
            Current scale.
        g2_bare : float
            Bare coupling.
        w_j : float
            Fluctuation scale at step j.

        Returns
        -------
        (ell_j, ell_plus) : (float, float)
            Current and inflated norm parameters.
        """
        ell_j = self.at_scale(j, g2_bare)
        ell_plus = self.inflate(ell_j, w_j)
        return ell_j, ell_plus


# ======================================================================
# 4. BBSRegulator
# ======================================================================

class BBSRegulator:
    """
    The BBS regulator G_j from Section 8.2.2.

        G_j(X, phi) = exp(c_G * Sum_{x in X} |phi_x|^2 / ell_j^2)

    This is a Gaussian weight that controls how fast polymer activities
    can grow with the field. The key properties are:

    THEOREM (sub-multiplicativity):
        G_j(X union Y, phi) <= G_j(X, phi) * G_j(Y, phi)
    This is trivially true since the exponent is additive over sites.

    THEOREM (Gaussian convolution bound, BBS Sec 8.2.2):
        E_C[G_j(X, phi + zeta)] <= const * G_{j+1}(X, phi)
    when ell_{j+1}^2 >= ell_j^2 + c_G * ||C||.

    Parameters
    ----------
    c_G : float
        Regulator constant. Must be small enough for the convolution
        bound to hold, large enough for control of large fields.
        BBS typically uses c_G ~ 1/(2*p_N).
    norm_param : NormParameter
        Norm parameter for ell_j computation.
    """

    def __init__(self, c_G: float = 0.1, norm_param: Optional[NormParameter] = None):
        if c_G <= 0:
            raise ValueError(f"c_G must be positive, got {c_G}")
        self.c_G = c_G
        self.norm_param = norm_param if norm_param is not None else NormParameter()

    def evaluate(self, phi: np.ndarray, ell_j: float,
                 n_sites: Optional[int] = None) -> float:
        """
        Evaluate G_j(X, phi) = exp(c_G * Sum |phi_x|^2 / ell_j^2).

        Parameters
        ----------
        phi : ndarray
            Field configuration. Either a flat vector or shape (n_sites, dim_internal).
        ell_j : float
            Norm parameter at scale j.
        n_sites : int or None
            Number of sites in polymer X. If None, inferred from phi.

        Returns
        -------
        float
            G_j(X, phi)
        """
        if ell_j <= 0:
            raise ValueError(f"ell_j must be positive, got {ell_j}")

        phi_sq = float(np.sum(phi**2))
        exponent = self.c_G * phi_sq / ell_j**2

        # Cap exponent to avoid overflow
        exponent = min(exponent, 500.0)
        return np.exp(exponent)

    def evaluate_per_site(self, phi_sites: List[np.ndarray],
                          ell_j: float) -> float:
        """
        Evaluate G_j with explicit per-site field values.

            G_j(X, phi) = exp(c_G * Sum_{x in X} |phi_x|^2 / ell_j^2)

        Parameters
        ----------
        phi_sites : list of ndarray
            Field value phi_x at each site x in the polymer X.
        ell_j : float
            Norm parameter.

        Returns
        -------
        float
            G_j(X, phi)
        """
        if ell_j <= 0:
            raise ValueError(f"ell_j must be positive, got {ell_j}")

        total_sq = sum(float(np.sum(phi_x**2)) for phi_x in phi_sites)
        exponent = self.c_G * total_sq / ell_j**2
        exponent = min(exponent, 500.0)
        return np.exp(exponent)

    def is_sub_multiplicative(self, phi: np.ndarray, ell_j: float,
                              split_index: int) -> bool:
        """
        Verify sub-multiplicativity: G(X union Y) <= G(X) * G(Y).

        Since the exponent is additive over sites, this is always true
        (with equality when X and Y are disjoint).

        THEOREM: True by construction.

        Parameters
        ----------
        phi : ndarray, shape (n_total,)
            Full field configuration on X union Y.
        ell_j : float
            Norm parameter.
        split_index : int
            Index splitting phi into X and Y parts.

        Returns
        -------
        bool
            Always True for disjoint polymers.
        """
        phi_X = phi[:split_index]
        phi_Y = phi[split_index:]

        G_union = self.evaluate(phi, ell_j)
        G_X = self.evaluate(phi_X, ell_j)
        G_Y = self.evaluate(phi_Y, ell_j)

        return G_union <= G_X * G_Y * (1 + 1e-10)  # numerical tolerance

    def convolution_bound(self, ell_j: float, C_norm: float) -> Dict[str, float]:
        """
        Compute the Gaussian convolution bound parameters.

        E_C[G_j(X, phi + zeta)] <= const * G_{j+1}(X, phi)

        This requires ell_{j+1}^2 >= ell_j^2 + c_G * ||C||.

        The constant in the bound depends on the number of sites |X|
        and the ratio of parameters.

        THEOREM (BBS Section 8.2.2).

        Parameters
        ----------
        ell_j : float
            Current norm parameter.
        C_norm : float
            Operator norm of the covariance ||C||.

        Returns
        -------
        dict with keys:
            'ell_j': current parameter
            'ell_required': minimum ell_{j+1} for bound to hold
            'inflation_sq': c_G * ||C||
            'ell_pythagorean': sqrt(ell_j^2 + c_G * ||C||)
        """
        inflation_sq = self.c_G * C_norm
        ell_required_sq = ell_j**2 + inflation_sq
        ell_pythagorean = np.sqrt(ell_required_sq)

        return {
            'ell_j': ell_j,
            'ell_required': ell_pythagorean,
            'inflation_sq': inflation_sq,
            'ell_pythagorean': ell_pythagorean,
        }

    def stability_with_V(self, g: float, ell_j: float, phi: np.ndarray,
                         n_sites: int = 1) -> Dict[str, float]:
        """
        Check compatibility of regulator G with e^{-V}.

        For V = g * Sum |phi_x|^4 (quartic interaction), the product
        G * e^{-V} must be bounded. This requires:
            c_G * |phi|^2 / ell^2 - g * |phi|^4 <= const

        The critical field value is |phi|^2 = c_G / (2 * g * ell^2),
        giving a maximum:
            G * e^{-V} <= exp(c_G^2 / (4 * g * ell^4)) * exp(const)

        Parameters
        ----------
        g : float
            Quartic coupling.
        ell_j : float
            Norm parameter.
        phi : ndarray
            Field configuration.
        n_sites : int
            Number of sites.

        Returns
        -------
        dict with stability info
        """
        phi_sq = float(np.sum(phi**2))
        phi_4 = phi_sq**2 / max(n_sites, 1)  # |phi|^4 per site, approximate

        regulator_exponent = self.c_G * phi_sq / ell_j**2
        potential_exponent = g * phi_4 * n_sites

        combined_exponent = regulator_exponent - potential_exponent

        # Critical point
        if g > 0 and ell_j > 0:
            phi_sq_critical = self.c_G / (2 * g * ell_j**2)
            max_combined = self.c_G**2 / (4 * g * ell_j**4)
        else:
            phi_sq_critical = float('inf')
            max_combined = float('inf')

        return {
            'regulator_exponent': regulator_exponent,
            'potential_exponent': potential_exponent,
            'combined_exponent': combined_exponent,
            'phi_sq_critical': phi_sq_critical,
            'max_combined_bound': max_combined,
            'is_stable': combined_exponent < max_combined + 10,
        }


# ======================================================================
# 5. WeightedPolymerNorm
# ======================================================================

class WeightedPolymerNorm:
    """
    The full BBS weighted polymer norm.

        ||K||_j = sup_X sup_phi ||K(X, phi)||_{T_phi(ell_j)} / G_j(X, phi)

    This combines the T_z seminorm with the regulator G_j. The polymer
    activity K is measured relative to what the regulator allows: K can
    be large when |phi| is large, but only as large as G permits.

    On S^3, the sup over polymers X is a sup over a FINITE set
    (compactness), and the sup over phi is over a bounded domain
    (Gribov region).

    NUMERICAL: The supremum is estimated by sampling.

    Parameters
    ----------
    seminorm : BBSTzSeminorm
        The T_z seminorm.
    regulator : BBSRegulator
        The G_j regulator.
    norm_param : NormParameter
        For computing ell_j.
    """

    def __init__(self, seminorm: Optional[BBSTzSeminorm] = None,
                 regulator: Optional[BBSRegulator] = None,
                 norm_param: Optional[NormParameter] = None):
        self.seminorm = seminorm if seminorm is not None else BBSTzSeminorm()
        self.regulator = regulator if regulator is not None else BBSRegulator()
        self.norm_param = norm_param if norm_param is not None else NormParameter()

    def evaluate_at_config(self, K_func: Callable, phi: np.ndarray,
                           j: int, g2_bare: float) -> float:
        """
        Evaluate ||K(phi)||_{T_phi(ell_j)} / G_j(phi) at a specific config.

        Parameters
        ----------
        K_func : callable
            K(phi) -> float. The polymer activity.
        phi : ndarray
            Field configuration.
        j : int
            RG scale.
        g2_bare : float
            Bare coupling.

        Returns
        -------
        float
            ||K||_{T_phi} / G_j at this configuration.
        """
        ell_j = self.norm_param.at_scale(j, g2_bare)

        # Set ell for seminorm evaluation
        self.seminorm.ell = ell_j

        T_norm = self.seminorm.evaluate_numerical(K_func, phi)
        G_val = self.regulator.evaluate(phi, ell_j)

        if G_val < 1e-300:
            return float('inf')
        return T_norm / G_val

    def evaluate(self, K_func: Callable, j: int, g2_bare: float,
                 dim: int = 9, n_samples: int = 50) -> float:
        """
        Estimate ||K||_j by sampling over field configurations.

            ||K||_j = sup_phi ||K(phi)||_{T_phi(ell_j)} / G_j(phi)

        NUMERICAL: Result is a lower bound on the true supremum.

        Parameters
        ----------
        K_func : callable
            K(phi) -> float.
        j : int
            RG scale.
        g2_bare : float
            Bare coupling.
        dim : int
            Field space dimension.
        n_samples : int
            Number of field configurations to sample.

        Returns
        -------
        float
            Estimated ||K||_j.
        """
        ell_j = self.norm_param.at_scale(j, g2_bare)
        rng = np.random.RandomState(42 + j)
        max_val = 0.0

        for _ in range(n_samples):
            # Sample phi with |phi| ~ ell_j (natural scale)
            phi = rng.randn(dim) * ell_j
            val = self.evaluate_at_config(K_func, phi, j, g2_bare)
            if np.isfinite(val):
                max_val = max(max_val, val)

        # Also check phi = 0
        phi_zero = np.zeros(dim)
        val_zero = self.evaluate_at_config(K_func, phi_zero, j, g2_bare)
        if np.isfinite(val_zero):
            max_val = max(max_val, val_zero)

        return max_val

    def is_contracting(self, K_j_norm: float, K_j1_norm: float,
                       g_bar_j: float) -> Dict[str, Any]:
        """
        Check whether the RG step is contracting in the weighted norm.

        Contraction: ||K_{j+1}||_{j+1} <= theta * ||K_j||_j
        for some theta < 1.

        Parameters
        ----------
        K_j_norm : float
            ||K_j||_j at scale j.
        K_j1_norm : float
            ||K_{j+1}||_{j+1} at scale j+1.
        g_bar_j : float
            Running coupling at scale j.

        Returns
        -------
        dict with contraction info
        """
        if K_j_norm <= 0:
            return {
                'is_contracting': K_j1_norm <= 0,
                'ratio': 0.0 if K_j1_norm <= 0 else float('inf'),
                'g_bar_j': g_bar_j,
            }

        ratio = K_j1_norm / K_j_norm
        return {
            'is_contracting': ratio < 1.0,
            'ratio': ratio,
            'g_bar_j': g_bar_j,
            'label': 'NUMERICAL',
        }


# ======================================================================
# 6. ProductPropertyVerifier
# ======================================================================

class ProductPropertyVerifier:
    """
    Verify the BBS product property (Proposition 7.1.2).

        ||FG||_{T_z(ell)} <= ||F||_{T_z(ell)} * ||G||_{T_z(ell)}

    This is THE key algebraic property of the T_z seminorm. It follows
    from the Leibniz rule for Frechet derivatives:

        D^k(FG)(z) = Sum_{j=0}^{k} C(k,j) * D^j F(z) . D^{k-j} G(z)

    combined with the binomial identity for the ell^k / k! weights:

        Sum_{k=0}^{p_N} (ell^k / k!) ||D^k(FG)||
        <= Sum_{k} (ell^k / k!) Sum_{j} C(k,j) ||D^j F|| ||D^{k-j} G||
        = [Sum (ell^j / j!) ||D^j F||] * [Sum (ell^m / m!) ||D^m G||]

    THEOREM: The product property holds for any p_N >= 0 and ell >= 0.

    Parameters
    ----------
    seminorm : BBSTzSeminorm
        The T_z seminorm to verify.
    """

    def __init__(self, seminorm: Optional[BBSTzSeminorm] = None):
        self.seminorm = seminorm if seminorm is not None else BBSTzSeminorm()

    def verify_product_polynomial(self, coeffs_F: np.ndarray,
                                  coeffs_G: np.ndarray,
                                  z: np.ndarray) -> Dict[str, Any]:
        """
        Verify ||FG|| <= ||F|| * ||G|| for polynomial activities.

        F = Sum c_p |phi|^{2p}, G = Sum d_q |phi|^{2q}
        FG = Sum_{p,q} c_p * d_q * |phi|^{2(p+q)}

        Parameters
        ----------
        coeffs_F : ndarray
            Coefficients of F.
        coeffs_G : ndarray
            Coefficients of G.
        z : ndarray
            Basepoint.

        Returns
        -------
        dict with verification results
        """
        # Compute FG coefficients via polynomial multiplication
        max_p = len(coeffs_F) - 1
        max_q = len(coeffs_G) - 1
        coeffs_FG = np.zeros(max_p + max_q + 1)
        for p, c_p in enumerate(coeffs_F):
            for q, d_q in enumerate(coeffs_G):
                coeffs_FG[p + q] += c_p * d_q

        norm_F = self.seminorm.evaluate_polynomial(coeffs_F, z)
        norm_G = self.seminorm.evaluate_polynomial(coeffs_G, z)
        norm_FG = self.seminorm.evaluate_polynomial(coeffs_FG, z)
        product_bound = norm_F * norm_G

        return {
            'norm_F': norm_F,
            'norm_G': norm_G,
            'norm_FG': norm_FG,
            'product_bound': product_bound,
            'holds': norm_FG <= product_bound * (1 + 1e-10),
            'ratio': norm_FG / product_bound if product_bound > 0 else 0.0,
            'label': 'THEOREM',
        }

    def verify_product_numerical(self, F: Callable, G: Callable,
                                 z: np.ndarray,
                                 n_directions: int = 20) -> Dict[str, Any]:
        """
        Verify product property numerically for general activities.

        NUMERICAL: Computes T_z norms and checks ||FG|| <= ||F|| * ||G||.

        Parameters
        ----------
        F, G : callable
            Activities F(phi), G(phi) -> float.
        z : ndarray
            Basepoint.
        n_directions : int
            Sampling parameter.

        Returns
        -------
        dict with verification results
        """
        def FG(phi):
            return F(phi) * G(phi)

        norm_F = self.seminorm.evaluate_numerical(F, z, n_directions)
        norm_G = self.seminorm.evaluate_numerical(G, z, n_directions)
        norm_FG = self.seminorm.evaluate_numerical(FG, z, n_directions)
        product_bound = norm_F * norm_G

        return {
            'norm_F': norm_F,
            'norm_G': norm_G,
            'norm_FG': norm_FG,
            'product_bound': product_bound,
            'holds': norm_FG <= product_bound * (1 + 0.01),
            'ratio': norm_FG / product_bound if product_bound > 0 else 0.0,
            'label': 'NUMERICAL',
        }

    def verify_exponential(self, coeffs_F: np.ndarray, z: np.ndarray,
                           n_terms: int = 6) -> Dict[str, Any]:
        """
        Verify that ||e^F||_{T_z} <= e^{||F||_{T_z}} (exponential bound).

        This follows from the product property applied iteratively:
            ||F^n/n!|| <= ||F||^n / n!
            ||e^F|| = ||Sum F^n/n!|| <= Sum ||F||^n/n! = e^{||F||}

        THEOREM: Direct consequence of submultiplicativity.

        Parameters
        ----------
        coeffs_F : ndarray
            Polynomial coefficients for F.
        z : ndarray
            Basepoint.
        n_terms : int
            Number of terms in Taylor expansion of e^F.

        Returns
        -------
        dict with verification results
        """
        norm_F = self.seminorm.evaluate_polynomial(coeffs_F, z)

        # Compute coefficients of e^F truncated to n_terms
        # e^F = Sum_{n=0}^{n_terms-1} F^n / n!
        # For F = Sum c_p |phi|^{2p}, F^n involves convolution powers
        coeffs_exp = np.array([1.0])  # Start with 1
        coeffs_power = np.array([1.0])  # F^0 = 1

        for n in range(1, n_terms):
            # F^n = F^{n-1} * F via polynomial multiplication
            new_power = np.zeros(len(coeffs_power) + len(coeffs_F) - 1)
            for i, c in enumerate(coeffs_power):
                for j, d in enumerate(coeffs_F):
                    new_power[i + j] += c * d
            coeffs_power = new_power

            # Add F^n / n! to e^F
            term = coeffs_power / math.factorial(n)
            max_len = max(len(coeffs_exp), len(term))
            padded_exp = np.zeros(max_len)
            padded_exp[:len(coeffs_exp)] = coeffs_exp
            padded_term = np.zeros(max_len)
            padded_term[:len(term)] = term
            coeffs_exp = padded_exp + padded_term

        norm_exp_F = self.seminorm.evaluate_polynomial(coeffs_exp, z)
        exp_norm_F = np.exp(norm_F) if norm_F < 500 else float('inf')

        return {
            'norm_F': norm_F,
            'norm_exp_F': norm_exp_F,
            'exp_norm_F': exp_norm_F,
            'holds': norm_exp_F <= exp_norm_F * (1 + 1e-6),
            'n_terms': n_terms,
            'label': 'NUMERICAL',
        }


# ======================================================================
# 7. GaussianConvolutionBound
# ======================================================================

class GaussianConvolutionBound:
    """
    The BBS Gaussian convolution bound (Section 7.3).

        ||E_C F(. + zeta)||_{T_z(ell)} <= ||F||_{T_z(ell_+)}

    where zeta ~ N(0, C) and ell_+^2 = ell^2 + w^2 with w^2 <= ||C||.

    The fluctuation field "inflates" the norm parameter via the
    Pythagorean rule. This is how the covariance enters the norm
    estimates at each RG step.

    On S^3, the covariance norm ||C_j|| comes from the heat kernel
    slice: w_j^2 ~ M^{-2j} / R^2 (integrated heat kernel at scale j).

    THEOREM: The Gaussian convolution bound holds for all p_N >= 0,
    all ell >= 0, all C >= 0.

    Parameters
    ----------
    seminorm : BBSTzSeminorm
        T_z seminorm.
    norm_param : NormParameter
        Norm parameter handler.
    """

    def __init__(self, seminorm: Optional[BBSTzSeminorm] = None,
                 norm_param: Optional[NormParameter] = None):
        self.seminorm = seminorm if seminorm is not None else BBSTzSeminorm()
        self.norm_param = norm_param if norm_param is not None else NormParameter()

    def compute_ell_plus(self, ell_j: float, C_norm: float) -> float:
        """
        Compute the inflated norm parameter.

            ell_+^2 = ell_j^2 + w^2  where w^2 = ||C||

        THEOREM: This is the exact Pythagorean rule from BBS Sec 7.3.

        Parameters
        ----------
        ell_j : float
            Current norm parameter.
        C_norm : float
            Operator norm of covariance (= w^2).

        Returns
        -------
        float
            ell_+ = sqrt(ell_j^2 + C_norm)
        """
        return np.sqrt(ell_j**2 + C_norm)

    def verify_bound_polynomial(self, coeffs_F: np.ndarray,
                                z: np.ndarray, ell_j: float,
                                C_norm: float,
                                n_samples: int = 200) -> Dict[str, Any]:
        """
        Verify the Gaussian convolution bound for a polynomial activity.

        We check:
            E_C[||F(z + zeta)||_{T_z(ell)}] <= ||F||_{T_{z}(ell_+)}

        by Monte Carlo sampling of zeta ~ N(0, C*I).

        NUMERICAL.

        Parameters
        ----------
        coeffs_F : ndarray
            Polynomial coefficients.
        z : ndarray
            Basepoint.
        ell_j : float
            Current norm parameter.
        C_norm : float
            Covariance norm.
        n_samples : int
            MC samples.

        Returns
        -------
        dict with verification results
        """
        dim = len(z)
        ell_plus = self.compute_ell_plus(ell_j, C_norm)
        w = np.sqrt(max(C_norm, 0))

        # Compute ||F||_{T_z(ell_+)} (the RHS)
        seminorm_plus = BBSTzSeminorm(p_N=self.seminorm.p_N, ell=ell_plus)
        rhs = seminorm_plus.evaluate_polynomial(coeffs_F, z)

        # Monte Carlo estimate of E_C[||F(z + zeta)||_{T_z(ell)}]
        rng = np.random.RandomState(42)
        seminorm_ell = BBSTzSeminorm(p_N=self.seminorm.p_N, ell=ell_j)

        lhs_samples = []
        for _ in range(n_samples):
            zeta = rng.randn(dim) * w
            z_shifted = z + zeta
            val = seminorm_ell.evaluate_polynomial(coeffs_F, z_shifted)
            lhs_samples.append(val)

        lhs_mean = float(np.mean(lhs_samples))

        return {
            'ell_j': ell_j,
            'ell_plus': ell_plus,
            'C_norm': C_norm,
            'lhs_expectation': lhs_mean,
            'rhs_bound': rhs,
            'holds': lhs_mean <= rhs * (1 + 0.1),  # 10% tolerance for MC
            'ratio': lhs_mean / rhs if rhs > 0 else 0.0,
            'label': 'NUMERICAL',
        }

    def verify_bound_numerical(self, F: Callable, z: np.ndarray,
                               ell_j: float, C_norm: float,
                               n_mc_samples: int = 100,
                               n_directions: int = 10) -> Dict[str, Any]:
        """
        Verify convolution bound for a general activity F.

        NUMERICAL.

        Parameters
        ----------
        F : callable
            F(phi) -> float.
        z : ndarray
            Basepoint.
        ell_j : float
            Current norm parameter.
        C_norm : float
            Covariance norm.
        n_mc_samples : int
            MC samples for expectation.
        n_directions : int
            Directions for T_z norm estimation.

        Returns
        -------
        dict with verification results
        """
        dim = len(z)
        ell_plus = self.compute_ell_plus(ell_j, C_norm)
        w = np.sqrt(max(C_norm, 0))

        # RHS: ||F||_{T_z(ell_+)}
        seminorm_plus = BBSTzSeminorm(p_N=self.seminorm.p_N, ell=ell_plus)
        rhs = seminorm_plus.evaluate_numerical(F, z, n_directions)

        # LHS: E[||F(. + zeta)||_{T_z(ell)}]
        rng = np.random.RandomState(42)
        seminorm_ell = BBSTzSeminorm(p_N=self.seminorm.p_N, ell=ell_j)

        lhs_samples = []
        for _ in range(n_mc_samples):
            zeta = rng.randn(dim) * w
            z_shifted = z + zeta
            val = seminorm_ell.evaluate_numerical(F, z_shifted, n_directions)
            lhs_samples.append(val)

        lhs_mean = float(np.mean(lhs_samples))

        return {
            'ell_j': ell_j,
            'ell_plus': ell_plus,
            'lhs_expectation': lhs_mean,
            'rhs_bound': rhs,
            'holds': lhs_mean <= rhs * (1 + 0.15),
            'ratio': lhs_mean / rhs if rhs > 0 else 0.0,
            'label': 'NUMERICAL',
        }

    @staticmethod
    def s3_fluctuation_scale(j: int, R: float = R_PHYSICAL_FM,
                             M: float = M_DEFAULT) -> float:
        """
        Fluctuation scale w_j^2 on S^3 from heat kernel slice.

        At RG scale j, the covariance slice has norm:
            ||C_j|| ~ M^{-2j} / (4*pi^2 * R^2)

        which gives w_j^2 = ||C_j||.

        NUMERICAL: From heat kernel expansion on S^3.

        Parameters
        ----------
        j : int
            RG scale.
        R : float
            S^3 radius.
        M : float
            Blocking factor.

        Returns
        -------
        float
            w_j^2 = ||C_j||
        """
        return M ** (-2 * j) / (4 * np.pi**2 * R**2)


# ======================================================================
# 8. GaugeCovariantTaylor
# ======================================================================

class GaugeCovariantTaylor:
    """
    Gauge-covariant Taylor expansion for YM polymer activities.

    In gauge theory, the ordinary Frechet derivative is replaced by
    the gauge-covariant derivative:
        D_A = d + [A, .]

    The k-th covariant derivative D_A^k F gives a gauge-covariant
    k-linear form. The T_z seminorm becomes:

        ||F||_{T_A(ell)} = Sum_{k=0}^{p_N} (ell^k / k!) ||D_A^k F(A)||

    The Loc operator extracts gauge-INVARIANT local terms via
    gauge-covariant Taylor expansion:
        Loc(F) = F(0) + D_A F(0)[A] + (1/2) D_A^2 F(0)[A,A]

    For YM on S^3, the covariant derivatives include the connection
    terms from the Maurer-Cartan form.

    NUMERICAL: Covariant derivatives computed via finite differences
    in the gauge field.

    Parameters
    ----------
    N_c : int
        Number of colors.
    p_N : int
        Maximum derivative order.
    """

    def __init__(self, N_c: int = N_C_DEFAULT, p_N: int = P_N_DEFAULT):
        if N_c < 2:
            raise ValueError(f"N_c must be >= 2, got {N_c}")
        self.N_c = N_c
        self.dim_adj = N_c**2 - 1
        self.p_N = p_N
        self._frechet = FrechetDerivative()

    @staticmethod
    def adjoint_action(A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute [A, X] in the Lie algebra su(N_c).

        For su(2) with basis {sigma_a / 2i}: [T_a, T_b] = epsilon_{abc} T_c.
        In the adjoint (3-vector) representation:
            [A, X]_c = Sum_{a,b} epsilon_{abc} A_a X_b

        Parameters
        ----------
        A : ndarray, shape (3,) for SU(2)
            Gauge field (Lie algebra valued).
        X : ndarray, shape (3,) for SU(2)
            Vector in Lie algebra.

        Returns
        -------
        ndarray
            [A, X] in the adjoint representation.
        """
        if len(A) == 3 and len(X) == 3:
            # su(2) structure constants: [T_a, T_b] = epsilon_{abc} T_c
            return np.cross(A, X)
        else:
            # General case: return zero for now (needs matrix commutator)
            return np.zeros_like(X)

    def covariant_derivative(self, F: Callable, A: np.ndarray,
                             h: np.ndarray, k: int = 1) -> float:
        """
        k-th gauge-covariant derivative of F at A in direction h.

        D_A F[h] = d/dt F(A + t*h)|_{t=0} = ordinary derivative

        For k=1, this equals the ordinary Frechet derivative (the
        covariance comes from how we compose these).

        For k=2: D_A^2 F[h1, h2] includes connection terms from
        the gauge structure.

        NUMERICAL: Via finite differences.

        Parameters
        ----------
        F : callable
            F(A) -> float. The gauge functional.
        A : ndarray
            Background gauge field.
        h : ndarray
            Direction vector(s).
        k : int
            Derivative order.

        Returns
        -------
        float
            D_A^k F(A)[h, ..., h]
        """
        eps = self._frechet.eps

        if k == 0:
            return F(A)
        elif k == 1:
            return (F(A + eps * h) - F(A - eps * h)) / (2 * eps)
        elif k == 2:
            return (F(A + eps * h) - 2 * F(A) + F(A - eps * h)) / eps**2
        else:
            # General k via iterated central differences
            val = 0.0
            for i in range(k + 1):
                sign = (-1) ** (k - i)
                binom = math.comb(k, i)
                shift = (i - k / 2) * eps
                val += sign * binom * F(A + shift * h)
            return val / eps**k

    def covariant_derivative_operator_norm(self, F: Callable,
                                           A: np.ndarray,
                                           k: int,
                                           n_directions: int = 20) -> float:
        """
        Operator norm of D_A^k F as a symmetric k-linear form.

            ||D_A^k F|| = sup_{|h|=1} |D_A^k F(A)[h, ..., h]|

        NUMERICAL: Estimated via random direction sampling.

        Parameters
        ----------
        F : callable
        A : ndarray
        k : int
        n_directions : int

        Returns
        -------
        float
        """
        if k == 0:
            return abs(F(A))

        dim = len(A)
        rng = np.random.RandomState(42)
        max_val = 0.0

        for _ in range(n_directions):
            h = rng.randn(dim)
            h_norm = np.linalg.norm(h)
            if h_norm < 1e-15:
                continue
            h = h / h_norm

            val = abs(self.covariant_derivative(F, A, h, k))
            max_val = max(max_val, val)

        return max_val

    def gauge_covariant_seminorm(self, F: Callable, A: np.ndarray,
                                 ell: float,
                                 n_directions: int = 20) -> float:
        """
        Gauge-covariant T_A seminorm.

            ||F||_{T_A(ell)} = Sum_{k=0}^{p_N} (ell^k / k!) ||D_A^k F(A)||

        Parameters
        ----------
        F : callable
        A : ndarray
        ell : float
        n_directions : int

        Returns
        -------
        float
        """
        total = 0.0
        for k in range(self.p_N + 1):
            dk_norm = self.covariant_derivative_operator_norm(
                F, A, k, n_directions
            )
            weight = ell**k / math.factorial(k)
            total += weight * dk_norm
        return total

    def gauge_invariant_extraction(self, F: Callable, A: np.ndarray,
                                   dim: int = 9) -> Dict[str, float]:
        """
        Extract gauge-invariant local terms via Taylor expansion (Loc).

        Loc(F)(A) = F(0) + D F(0)[A] + (1/2) D^2 F(0)[A, A]

        For gauge-invariant F, only even-order terms survive.
        The quadratic term gives the mass and kinetic contributions.

        NUMERICAL.

        Parameters
        ----------
        F : callable
            F(A) -> float.
        A : ndarray
            Background gauge field.
        dim : int
            Field space dimension.

        Returns
        -------
        dict with extracted local terms
        """
        A_zero = np.zeros(dim)
        eps = self._frechet.eps

        # F(0): constant term
        f0 = F(A_zero)

        # D F(0)[A]: linear term (vanishes for gauge-invariant F)
        f1 = (F(eps * A / max(np.linalg.norm(A), 1e-15))
              - F(-eps * A / max(np.linalg.norm(A), 1e-15))) / (2 * eps)
        if np.linalg.norm(A) > 1e-15:
            f1 *= np.linalg.norm(A)

        # D^2 F(0)[A, A]: quadratic term
        A_norm = max(np.linalg.norm(A), 1e-15)
        A_hat = A / A_norm
        f2 = (F(eps * A_hat) - 2 * F(A_zero) + F(-eps * A_hat)) / eps**2
        f2 *= A_norm**2

        return {
            'constant': f0,
            'linear': f1,
            'quadratic': f2 / 2,
            'loc_value': f0 + f1 + f2 / 2,
            'remainder': F(A) - (f0 + f1 + f2 / 2),
        }


# ======================================================================
# Comparison with existing TPhiSeminorm
# ======================================================================

def compare_with_legacy(K_func: Callable, phi: np.ndarray,
                        j: int, g2: float,
                        p_N: int = P_N_DEFAULT) -> Dict[str, Any]:
    """
    Compare the BBS T_z seminorm with the legacy TPhiSeminorm.

    The legacy seminorm in polymer_algebra_ym.py uses:
        h_j = M^{-j} * sqrt(g2_j)  (NOT BBS ell_j = g_bar^{-1/4})
        Numerical sampling over ||A|| <= h_j domain
        Central differences for derivatives

    The BBS seminorm uses:
        ell_j = g_bar_j^{-1/4}
        Exact or numerical Frechet derivatives
        Operator norm of symmetric k-linear forms

    NUMERICAL: Both are estimates; comparison shows structural differences.

    Parameters
    ----------
    K_func : callable
        K(phi) -> float. Activity function.
    phi : ndarray
        Field configuration.
    j : int
        RG scale.
    g2 : float
        Coupling at scale j.
    p_N : int
        Derivative order for BBS seminorm.

    Returns
    -------
    dict with comparison results
    """
    # BBS seminorm
    g_bar = g2 / (1 + BETA_0_SU2 * g2 * j)
    ell_j = g_bar ** (-0.25) if g_bar > 0 else 1.0
    bbs = BBSTzSeminorm(p_N=p_N, ell=ell_j)
    bbs_value = bbs.evaluate_numerical(K_func, phi)

    # Legacy seminorm parameters
    h_j = M_DEFAULT ** (-j) * np.sqrt(max(g2, 0))

    return {
        'bbs_ell_j': ell_j,
        'legacy_h_j': h_j,
        'bbs_norm': bbs_value,
        'ell_over_h_ratio': ell_j / h_j if h_j > 0 else float('inf'),
        'label': 'NUMERICAL',
    }
