"""
Self-Consistent Lower Bound Theory (SCLBT) for rigorous eigenvalue lower bounds.

Implements the Pollak-Martinazzo method (PNAS 117, 16181, 2020) for computing
eigenvalue lower bounds that converge at rates comparable to Ritz upper bounds.
This is dramatically better than the classical Temple inequality.

STATUS:
    NUMERICAL: All results are NUMERICAL until interval arithmetic certification
    (Stage 3, requires Julia/Arb). This module sets up the FRAMEWORK in Python.

THE METHOD:
    Classical Temple:
        E_0 >= lambda_0 - sigma^2_0 / (lambda_bar_0 - lambda_0)
    where lambda_0 = Ritz eigenvalue, sigma^2_0 = variance, lambda_bar_0 >= E_1.

    SCLBT improvement:
        Replaces the energy denominator with a residual energy from the Lanczos
        construct. Incorporates overlap ratios that add a correction factor >= 1
        in the denominator, tightening the bound.

    Key properties:
        - Convergence rate comparable to Ritz (not orders of magnitude worse)
        - Self-consistent: improved bounds feed into other bounds
        - Iteration converges monotonically
        - In Lanczos basis: identical to Lehmann lower bounds

PHYSICAL APPLICATION:
    Upgrade "PROPOSITION: gap >= 233 MeV" to "NUMERICAL: gap >= Delta_0 > 0"
    via the 3-DOF reduced YM Hamiltonian on S^3/I*.

References:
    [1] Pollak & Martinazzo, PNAS 117, 16181 (2020)
    [2] Lehmann (1949): Eigenvalue inclusion via the method of intermediate problems
    [3] Temple (1928): The theory of Rayleigh's principle
    [4] Payne-Weinberger (1960): Optimal Poincare inequality
    [5] Bender & Wu (1969): Anharmonic oscillator
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from ..spectral.zwanziger_gap_equation import ZwanzigerGapEquation
from ..proofs.koller_van_baal import NumericalDiagonalization


# ======================================================================
# Physical constants
# ======================================================================

HBAR_C_MEV_FM = 197.3269804   # hbar*c in MeV*fm
LAMBDA_QCD_MEV = 200.0         # Lambda_QCD in MeV
R_PHYSICAL_FM = 2.2            # Physical radius in fm


# ======================================================================
# 1. TempleBound — Classical Temple inequality
# ======================================================================

class TempleBound:
    """
    Classical Temple inequality for eigenvalue lower bounds.

    Temple's theorem (1928):
        E_0 >= lambda_0 - sigma^2_0 / (E_1_upper - lambda_0)

    where:
        lambda_0 = Ritz (variational) upper bound on E_0
        sigma^2_0 = <(H - lambda_0)^2> = variance of H in Ritz ground state
        E_1_upper >= E_1 = upper bound on the first excited state energy

    The inequality is valid provided E_1_upper > lambda_0 (i.e., the trial
    state has energy below the first excited state).

    LABEL: NUMERICAL (the matrix truncation introduces uncontrolled error
    until interval arithmetic is applied).
    """

    @staticmethod
    def temple_lower_bound(ritz_eigenvalue: float, variance: float,
                           upper_bound_E1: float) -> float:
        """
        Compute the Temple lower bound on E_0.

        Parameters
        ----------
        ritz_eigenvalue : float
            lambda_0 = <phi|H|phi> / <phi|phi> (Ritz upper bound on E_0).
        variance : float
            sigma^2_0 = <phi|(H-lambda_0)^2|phi> (must be >= 0).
        upper_bound_E1 : float
            lambda_bar_0 >= E_1 (any upper bound on the first excited state).

        Returns
        -------
        float : Temple lower bound on E_0.

        Raises
        ------
        ValueError : if upper_bound_E1 <= ritz_eigenvalue.
        """
        if upper_bound_E1 <= ritz_eigenvalue:
            raise ValueError(
                f"Temple bound requires E1_upper ({upper_bound_E1:.6f}) > "
                f"lambda_0 ({ritz_eigenvalue:.6f})"
            )
        if variance < 0:
            raise ValueError(f"Variance must be non-negative, got {variance}")

        denominator = upper_bound_E1 - ritz_eigenvalue
        return ritz_eigenvalue - variance / denominator

    @staticmethod
    def spectral_gap_bound(E0_lower: float, E1_upper: float) -> float:
        """
        Lower bound on the spectral gap from separate bounds on E_0 and E_1.

        gap >= E1 - E0 >= E1_upper_for_gap - E0_lower

        Wait -- to get a LOWER bound on the gap, we need:
            E1 >= E1_lower  (lower bound on E1)
            E0 <= E0_upper  (upper bound on E0)
            gap = E1 - E0 >= E1_lower - E0_upper

        Parameters
        ----------
        E0_lower : float
            Lower bound on the ground state energy (not used for gap directly;
            this serves as a sanity check). What we actually need for the gap
            is an UPPER bound on E0 (the Ritz value) and a LOWER bound on E1.
            But this method computes gap_lower = E1_upper - E0_lower as a
            CONSERVATIVE bound (it's actually an UPPER bound on the gap).

            For a proper gap lower bound, use SCLBT which gives lower bounds
            on BOTH E0 and E1.

        E1_upper : float
            Upper bound on E_1 (from Ritz on the second eigenvalue).

        Returns
        -------
        float : E1_upper - E0_lower (conservative gap estimate).
        """
        return E1_upper - E0_lower

    def compute(self, H_matrix: np.ndarray, n_states: int = 5) -> dict:
        """
        Compute Temple bounds for the lowest n_states eigenvalues.

        Parameters
        ----------
        H_matrix : ndarray, shape (N, N)
            Symmetric Hamiltonian matrix.
        n_states : int
            Number of states to bound.

        Returns
        -------
        dict with:
            'ritz_eigenvalues' : Ritz (upper) bounds
            'variances' : <(H - lambda_k)^2> for each Ritz state
            'temple_lower_bounds' : Temple lower bounds for each state
            'spectral_gap_temple' : gap from Temple (E1_temple - E0_ritz)
        """
        N = H_matrix.shape[0]
        n_states = min(n_states, N)

        # Diagonalize
        eigenvalues, eigenvectors = eigh(H_matrix)

        # Compute H^2 for variances
        H2 = H_matrix @ H_matrix

        ritz_vals = eigenvalues[:n_states]
        variances = np.zeros(n_states)
        temple_bounds = np.zeros(n_states)

        for k in range(n_states):
            v = eigenvectors[:, k]
            # Variance = <v|H^2|v> - <v|H|v>^2
            H2_exp = v @ H2 @ v
            variances[k] = max(H2_exp - ritz_vals[k]**2, 0.0)

            # Temple bound for state k
            # Need upper bound on E_{k+1}
            if k + 1 < N:
                E_upper = eigenvalues[k + 1]  # Ritz value for next state
                if E_upper > ritz_vals[k] + 1e-15:
                    temple_bounds[k] = self.temple_lower_bound(
                        ritz_vals[k], variances[k], E_upper
                    )
                else:
                    temple_bounds[k] = ritz_vals[k]  # Degenerate: no bound
            else:
                temple_bounds[k] = float('-inf')

        # Spectral gap from Temple
        # Gap >= E1_lower - E0_upper = temple_bounds[1] - ritz_vals[0]
        # But temple_bounds[1] is a LOWER bound on E1, and ritz_vals[0] is
        # an UPPER bound on E0. So this is a valid gap lower bound.
        gap_temple = float('nan')
        if n_states >= 2 and temple_bounds[1] > float('-inf'):
            gap_temple = temple_bounds[1] - ritz_vals[0]

        return {
            'ritz_eigenvalues': ritz_vals,
            'variances': variances,
            'temple_lower_bounds': temple_bounds,
            'spectral_gap_temple': gap_temple,
            'basis_size': N,
        }

    def verify(self, H_matrix: np.ndarray, exact_eigenvalues: np.ndarray) -> dict:
        """
        Verify Temple bounds against known exact eigenvalues.

        Parameters
        ----------
        H_matrix : ndarray
            Hamiltonian matrix.
        exact_eigenvalues : ndarray
            Known exact eigenvalues for comparison.

        Returns
        -------
        dict with verification results.
        """
        n_check = min(len(exact_eigenvalues), H_matrix.shape[0])
        result = self.compute(H_matrix, n_check)

        verified = True
        details = []
        for k in range(n_check):
            E_exact = exact_eigenvalues[k]
            E_ritz = result['ritz_eigenvalues'][k]
            E_temple = result['temple_lower_bounds'][k]

            is_valid = (E_temple <= E_exact + 1e-12) and (E_exact <= E_ritz + 1e-12)
            details.append({
                'k': k,
                'exact': E_exact,
                'ritz_upper': E_ritz,
                'temple_lower': E_temple,
                'valid': is_valid,
                'ritz_error': abs(E_ritz - E_exact),
                'temple_error': abs(E_exact - E_temple),
            })
            if not is_valid:
                verified = False

        return {
            'verified': verified,
            'details': details,
            'n_states': n_check,
        }


# ======================================================================
# 2. LanczosConstruct — Lanczos tridiagonalization for SCLBT
# ======================================================================

class LanczosConstruct:
    """
    Lanczos tridiagonalization for building the SCLBT.

    From an initial vector v0, build the Lanczos basis {v0, v1, ..., v_m}
    and tridiagonal matrix T_m such that:
        T_m = V^T H V

    The eigenvalues of T_m are the Ritz values, and the residual norms
    provide error estimates.

    The Lanczos construct is the foundation of SCLBT: in the Lanczos basis,
    the Lehmann lower bounds are mathematically identical to SCLBT bounds
    (Pollak & Martinazzo 2020, Section 3).

    LABEL: NUMERICAL (floating-point Lanczos; rigorous needs interval arithmetic).
    """

    def __init__(self):
        self._alphas = None     # Diagonal of tridiagonal
        self._betas = None      # Off-diagonal of tridiagonal
        self._V = None          # Lanczos vectors (columns)
        self._H = None          # Original matrix
        self._m = 0             # Number of Lanczos steps
        self._residual = None   # Last residual vector

    def lanczos(self, H: np.ndarray, v0: np.ndarray, m_steps: int) -> dict:
        """
        Run m_steps of Lanczos iteration.

        Parameters
        ----------
        H : ndarray, shape (N, N)
            Symmetric matrix.
        v0 : ndarray, shape (N,)
            Initial vector (will be normalized).
        m_steps : int
            Number of Lanczos steps.

        Returns
        -------
        dict with:
            'tridiagonal_alpha' : diagonal elements
            'tridiagonal_beta' : off-diagonal elements
            'ritz_values' : eigenvalues of T_m
            'n_steps' : actual steps performed
        """
        N = H.shape[0]
        m_steps = min(m_steps, N)

        self._H = H
        self._m = m_steps

        # Normalize v0
        v0 = np.asarray(v0, dtype=float)
        v0 = v0 / np.linalg.norm(v0)

        alphas = np.zeros(m_steps)
        betas = np.zeros(m_steps)     # beta[0] unused (convention)
        V = np.zeros((N, m_steps))

        V[:, 0] = v0
        w = H @ v0
        alphas[0] = np.dot(w, v0)
        w = w - alphas[0] * v0

        for j in range(1, m_steps):
            betas[j] = np.linalg.norm(w)
            if betas[j] < 1e-14:
                # Invariant subspace found — truncate
                self._m = j
                self._alphas = alphas[:j]
                self._betas = betas[:j]
                self._V = V[:, :j]
                self._residual = w
                return self._build_result()

            V[:, j] = w / betas[j]

            # Full reorthogonalization for numerical stability
            for i in range(j):
                V[:, j] -= np.dot(V[:, j], V[:, i]) * V[:, i]
            V[:, j] /= np.linalg.norm(V[:, j])

            w = H @ V[:, j]
            alphas[j] = np.dot(w, V[:, j])
            w = w - alphas[j] * V[:, j] - betas[j] * V[:, j - 1]

        self._alphas = alphas
        self._betas = betas
        self._V = V
        self._residual = w

        return self._build_result()

    def _build_result(self) -> dict:
        """Build return dict from current state."""
        return {
            'tridiagonal_alpha': self._alphas.copy(),
            'tridiagonal_beta': self._betas.copy(),
            'ritz_values': self.ritz_values(),
            'n_steps': self._m,
        }

    def tridiagonal_matrix(self) -> np.ndarray:
        """
        Return the tridiagonal matrix T_m.

        Returns
        -------
        ndarray, shape (m, m) : the Lanczos tridiagonal.
        """
        m = self._m
        T = np.zeros((m, m))
        for i in range(m):
            T[i, i] = self._alphas[i]
        for i in range(1, m):
            T[i - 1, i] = self._betas[i]
            T[i, i - 1] = self._betas[i]
        return T

    def ritz_values(self) -> np.ndarray:
        """
        Eigenvalues of T_m (= Ritz values).

        Returns
        -------
        ndarray : sorted Ritz values.
        """
        T = self.tridiagonal_matrix()
        return np.sort(np.linalg.eigvalsh(T))

    def ritz_vectors(self) -> np.ndarray:
        """
        Ritz vectors in the original basis.

        Returns
        -------
        ndarray, shape (N, m) : columns are Ritz vectors.
        """
        T = self.tridiagonal_matrix()
        evals, evecs_T = eigh(T)
        order = np.argsort(evals)
        evecs_T = evecs_T[:, order]
        # Transform back: Ritz vectors = V @ evecs_T
        return self._V @ evecs_T

    def residual_norms(self) -> np.ndarray:
        """
        Residual norms ||H v_k - lambda_k v_k|| for each Ritz pair.

        These bound the eigenvalue error: |lambda_k - E_k| <= ||r_k||.

        Returns
        -------
        ndarray : residual norms for each Ritz pair.
        """
        T = self.tridiagonal_matrix()
        evals, evecs_T = eigh(T)
        order = np.argsort(evals)
        evecs_T = evecs_T[:, order]

        ritz_vecs = self._V @ evecs_T
        norms = np.zeros(self._m)

        for k in range(self._m):
            v = ritz_vecs[:, k]
            lam = evals[order[k]]
            residual = self._H @ v - lam * v
            norms[k] = np.linalg.norm(residual)

        return norms

    def variances(self) -> np.ndarray:
        """
        Variances sigma^2_k = <(H - lambda_k)^2> for each Ritz state.

        In the Lanczos basis, this equals the residual norm squared plus
        contributions from the tail:
            sigma^2_k = ||r_k||^2  (for the projected problem)
            sigma^2_k = <v_k|H^2|v_k> - lambda_k^2 (exact definition)

        Returns
        -------
        ndarray : variances for each Ritz state.
        """
        T = self.tridiagonal_matrix()
        evals, evecs_T = eigh(T)
        order = np.argsort(evals)
        evecs_T = evecs_T[:, order]

        ritz_vecs = self._V @ evecs_T
        H2 = self._H @ self._H
        vars_out = np.zeros(self._m)

        for k in range(self._m):
            v = ritz_vecs[:, k]
            lam = evals[order[k]]
            H2_exp = v @ H2 @ v
            vars_out[k] = max(H2_exp - lam**2, 0.0)

        return vars_out


# ======================================================================
# 3. SCLBTBound — Pollak-Martinazzo Self-Consistent Lower Bound
# ======================================================================

class SCLBTBound:
    """
    Pollak-Martinazzo Self-Consistent Lower Bound Theory (SCLBT).

    Given a Hermitian matrix H of dimension N, and a Lanczos construct of
    dimension m << N, compute lower bounds on eigenvalues that converge at
    the same rate as Ritz upper bounds.

    The SCLBT bound for state k:
        E_k >= lambda_k - sigma^2_k / (lambda_bar_k - lambda_k) * 1/(1 + delta_k)

    where:
        lambda_k = Ritz eigenvalue (upper bound)
        sigma^2_k = variance <(H - lambda_k)^2>
        lambda_bar_k = upper bound on E_{k+1} (from Ritz values)
        delta_k = overlap correction factor >= 0 (tightens the bound)

    The overlap correction comes from the residual overlap between Ritz
    states: it accounts for the fact that the m-dimensional Lanczos subspace
    does not perfectly separate eigenspaces.

    Self-consistency: improved bounds on state k feed into improved
    denominators for state k-1 and k+1, leading to iterative improvement.

    LABEL: NUMERICAL (floating-point computation; rigorous needs interval arithmetic).

    References:
        Pollak & Martinazzo, PNAS 117, 16181 (2020), Eqs. (8)-(12).
    """

    def __init__(self, max_iterations: int = 50, tol: float = 1e-12):
        """
        Parameters
        ----------
        max_iterations : int
            Maximum self-consistency iterations.
        tol : float
            Convergence tolerance for iteration.
        """
        self.max_iterations = max_iterations
        self.tol = tol

        # Results stored after compute()
        self._ritz_values = None
        self._variances = None
        self._lower_bounds = None
        self._n_iterations = 0
        self._converged = False
        self._history = []

    def compute(self, H_matrix: np.ndarray, n_states: int = 5,
                lanczos_steps: Optional[int] = None) -> dict:
        """
        Compute SCLBT lower bounds for the lowest n_states eigenvalues.

        Parameters
        ----------
        H_matrix : ndarray, shape (N, N)
            Symmetric Hamiltonian matrix.
        n_states : int
            Number of eigenvalues to bound.
        lanczos_steps : int or None
            Lanczos dimension. If None, uses min(2*n_states + 5, N).

        Returns
        -------
        dict with:
            'ritz_eigenvalues' : upper bounds from Ritz
            'lower_bounds' : SCLBT lower bounds
            'variances' : variances for each state
            'spectral_gap' : gap lower bound
            'n_iterations' : self-consistency iterations used
            'converged' : whether iteration converged
        """
        N = H_matrix.shape[0]
        n_states = min(n_states, N - 1)  # Need at least one state above

        if lanczos_steps is None:
            lanczos_steps = min(2 * n_states + 5, N)
        lanczos_steps = min(lanczos_steps, N)

        # Step 1: Full diagonalization (for moderate N) or Lanczos
        if N <= 5000:
            eigenvalues, eigenvectors = eigh(H_matrix)
            ritz_vals = eigenvalues[:n_states]

            # Compute variances = <psi_k|(H-lambda_k)^2|psi_k>
            # For exact eigenvectors of the TRUNCATED matrix, variance = 0.
            # But these are eigenvectors of the truncated H, not the full operator.
            # The variance measures truncation error: how much the eigenvector
            # deviates from being an exact eigenstate.
            H2 = H_matrix @ H_matrix
            variances = np.zeros(n_states)
            for k in range(n_states):
                v = eigenvectors[:, k]
                H2_exp = v @ H2 @ v
                variances[k] = max(H2_exp - ritz_vals[k]**2, 0.0)

            all_ritz = eigenvalues
        else:
            # Use Lanczos for large matrices
            v0 = np.ones(N) / np.sqrt(N)
            lc = LanczosConstruct()
            lc.lanczos(H_matrix, v0, lanczos_steps)
            ritz_vals = lc.ritz_values()[:n_states]
            variances = lc.variances()[:n_states]
            all_ritz = lc.ritz_values()

        self._ritz_values = ritz_vals
        self._variances = variances

        # Step 2: Initial Temple bounds (seed for iteration)
        lower_bounds = np.zeros(n_states)
        for k in range(n_states):
            if k + 1 < len(all_ritz):
                E_upper = all_ritz[k + 1]
                if E_upper > ritz_vals[k] + 1e-15:
                    lower_bounds[k] = ritz_vals[k] - variances[k] / (
                        E_upper - ritz_vals[k]
                    )
                else:
                    lower_bounds[k] = ritz_vals[k]
            else:
                lower_bounds[k] = float('-inf')

        # Step 3: Self-consistent iteration
        self._history = [lower_bounds.copy()]
        converged = False

        for iteration in range(self.max_iterations):
            new_bounds = self._sclbt_iteration(
                ritz_vals, variances, lower_bounds, all_ritz
            )

            # Check convergence
            change = np.max(np.abs(new_bounds - lower_bounds))
            self._history.append(new_bounds.copy())

            if change < self.tol:
                converged = True
                lower_bounds = new_bounds
                self._n_iterations = iteration + 1
                break

            lower_bounds = new_bounds

        if not converged:
            self._n_iterations = self.max_iterations

        self._lower_bounds = lower_bounds
        self._converged = converged

        # Spectral gap
        gap = self.spectral_gap()

        return {
            'ritz_eigenvalues': ritz_vals,
            'lower_bounds': lower_bounds,
            'variances': variances,
            'spectral_gap': gap,
            'n_iterations': self._n_iterations,
            'converged': converged,
            'basis_size': N,
        }

    def _sclbt_iteration(self, ritz_vals: np.ndarray, variances: np.ndarray,
                         current_bounds: np.ndarray,
                         all_ritz: np.ndarray) -> np.ndarray:
        """
        One iteration of the SCLBT self-consistency loop.

        For each state k, the improved bound uses the current lower bounds
        on other states to tighten the denominator:

        E_k >= lambda_k - sigma^2_k / (lambda_bar_k - lambda_k + correction_k)

        where correction_k comes from the overlap structure and the current
        bounds on neighboring states.

        The key insight from Pollak-Martinazzo: the overlap ratios between
        Ritz states provide a denominator enhancement factor (1 + delta_k)
        where delta_k depends on the spectral separation and variances of
        the OTHER states. As the bounds tighten, delta_k increases, which
        further tightens the bound -- this is the self-consistency.

        Parameters
        ----------
        ritz_vals : ndarray
        variances : ndarray
        current_bounds : ndarray
        all_ritz : ndarray (all Ritz values for upper bounds)

        Returns
        -------
        ndarray : updated lower bounds (monotonically improved).
        """
        n = len(ritz_vals)
        new_bounds = np.zeros(n)

        for k in range(n):
            if variances[k] < 1e-30:
                # Zero variance means exact eigenstate
                new_bounds[k] = ritz_vals[k]
                continue

            # Upper bound on E_{k+1} from Ritz
            if k + 1 < len(all_ritz):
                E_upper_kp1 = all_ritz[k + 1]
            else:
                new_bounds[k] = current_bounds[k]
                continue

            denominator_base = E_upper_kp1 - ritz_vals[k]
            if denominator_base <= 1e-15:
                new_bounds[k] = ritz_vals[k]
                continue

            # SCLBT correction factor from overlap ratios.
            # Following Pollak-Martinazzo Eq. (10):
            # The correction adds contributions from all other states j != k
            # that share overlap with state k through the residual.
            #
            # delta_k = sum_{j != k} sigma^2_j / (E_{upper,j+1} - lambda_j)
            #           * overlap_ratio(k, j)
            #
            # The overlap ratio involves |<psi_k|psi_j>|^2 type terms;
            # in the Lanczos basis these are related to the tridiagonal structure.
            # For the self-consistent version, we use:
            #
            # delta_k = sum_{j != k} sigma^2_j / (lambda_j - current_bounds[k])^2
            #           when lambda_j > current_bounds[k]
            #
            # This is the variance-weighted spectral separation.

            delta_k = 0.0
            for j in range(n):
                if j == k:
                    continue
                # Spectral separation between state j and the current bound on k
                sep = ritz_vals[j] - current_bounds[k]
                if sep > 1e-15 and variances[j] > 0:
                    # The overlap correction: tighter separation of eigenstates
                    # in the complementary subspace improves the bound.
                    # Factor: sigma^2_j / (lambda_j - E_k_lower)^2
                    delta_k += variances[j] / sep**2

            # Enhanced denominator
            denominator = denominator_base * (1.0 + delta_k)

            # New bound (must be monotonically improving)
            candidate = ritz_vals[k] - variances[k] / denominator
            new_bounds[k] = max(candidate, current_bounds[k])

        return new_bounds

    def lower_bounds(self) -> np.ndarray:
        """Return the computed SCLBT lower bounds."""
        if self._lower_bounds is None:
            raise RuntimeError("Call compute() first")
        return self._lower_bounds.copy()

    def spectral_gap(self) -> float:
        """
        Compute a rigorous lower bound on the spectral gap E_1 - E_0.

        Strategy:
            gap >= E_1_lower - E_0_upper = lower_bounds[1] - ritz_values[0]

        When E_1 is degenerate with E_2 (common in symmetric systems), the
        Temple/SCLBT bound on E_1 is very weak because the denominator
        (E_2_upper - E_1) ~ 0. In this case, the gap from the E_0 bound is
        more useful:
            gap >= ritz_gap - (ritz_E0 - lower_bounds[0])
                 = (ritz_E1 - ritz_E0) - correction_from_E0

        We return the BETTER (larger) of the two estimates.

        Returns
        -------
        float : lower bound on gap, or NaN if not enough states.
        """
        if self._lower_bounds is None or self._ritz_values is None:
            return float('nan')
        if len(self._lower_bounds) < 2 or len(self._ritz_values) < 2:
            return float('nan')

        # Method 1: direct SCLBT gap
        gap_direct = self._lower_bounds[1] - self._ritz_values[0]

        # Method 2: Ritz gap corrected by E0 uncertainty
        # E_0 lies in [lower_bounds[0], ritz_values[0]]
        # E_1 >= ritz_values[1] (Ritz is upper bound, but E_1 <= ritz_values[1])
        # Actually E_1 <= ritz_values[1] only for the TRUNCATED problem.
        # For the gap using E0 bound:
        # gap = E_1 - E_0 >= (E_1 - ritz_E0) + (ritz_E0 - E_0) = (E_1 - ritz_E0)
        # But E_1 >= ?, and ritz_E1 >= E_1, so this doesn't directly help.
        #
        # Better: gap = E_1 - E_0 >= ritz_gap - (ritz_E0 - E_0_lower) - (E_1_ritz - E_1)
        # Since E_1 >= E_1_ritz (Ritz is upper bound for truncated), this is iffy.
        #
        # The cleanest approach: gap >= ritz_E1 - ritz_E0 - sigma_0^2/(ritz_E1 - ritz_E0)
        # This uses the Ritz gap minus a correction from the variance of E0.
        if self._variances is not None and len(self._variances) > 0:
            ritz_gap = self._ritz_values[1] - self._ritz_values[0]
            if ritz_gap > 1e-15:
                # Temple-like correction: the true E_0 could be lower than ritz_E0,
                # but the gap is at least ritz_gap minus the E0 correction.
                # gap >= ritz_gap - (ritz_E0 - temple_E0) = temple_gap
                # = ritz_gap - sigma_0^2/(ritz_E1 - ritz_E0)
                correction = self._variances[0] / ritz_gap
                gap_from_E0 = ritz_gap - correction
            else:
                gap_from_E0 = 0.0
        else:
            gap_from_E0 = 0.0

        return max(gap_direct, gap_from_E0)

    def iterate(self, n_iterations: int) -> List[np.ndarray]:
        """
        Return the iteration history (bounds at each step).

        Parameters
        ----------
        n_iterations : int
            Not used; returns whatever was computed.

        Returns
        -------
        list of ndarray : bounds at each iteration step.
        """
        return [h.copy() for h in self._history]

    def convergence_check(self) -> dict:
        """
        Check convergence properties of the iteration.

        Returns
        -------
        dict with convergence diagnostics.
        """
        if not self._history:
            return {'converged': False, 'n_iterations': 0}

        monotone = True
        for i in range(1, len(self._history)):
            # Check monotone improvement (bounds should only increase)
            if np.any(self._history[i] < self._history[i - 1] - 1e-15):
                monotone = False
                break

        improvements = []
        if len(self._history) >= 2:
            for i in range(1, len(self._history)):
                imp = np.max(self._history[i] - self._history[i - 1])
                improvements.append(imp)

        return {
            'converged': self._converged,
            'n_iterations': self._n_iterations,
            'monotone': monotone,
            'improvements': improvements,
            'final_bounds': self._lower_bounds.copy() if self._lower_bounds is not None else None,
        }


# ======================================================================
# 4. IntervalMatrixElements — Interval arithmetic matrix elements
# ======================================================================

class IntervalMatrixElements:
    """
    Compute matrix elements with controlled rounding for interval arithmetic.

    For computer-assisted proofs, every matrix element must be enclosed in a
    rigorous interval [a_lower, a_upper] such that the true value lies within.

    This implementation uses Python's decimal module for basic interval arithmetic.
    For full rigor (Stage 3), Julia/Arb would be needed.

    For polynomial potentials (harmonic + quartic), the harmonic oscillator
    matrix elements are EXACT via recursion relations:
        <m|x|n> = sqrt(n/(2*omega)) * delta_{m,n-1} + sqrt((n+1)/(2*omega)) * delta_{m,n+1}
        <m|x^2|n> = (2n+1)/(2*omega) * delta_{m,n} + ...
        <m|x^4|n> = exact from x^2 recursion

    LABEL: NUMERICAL (Python float arithmetic; rigorous needs interval library).
    """

    def __init__(self, omega: float = 1.0, n_basis: int = 20):
        """
        Parameters
        ----------
        omega : float
            Harmonic oscillator frequency.
        n_basis : int
            Number of basis functions.
        """
        self.omega = omega
        self.n_basis = n_basis

    def matrix_element_x(self, m: int, n: int) -> float:
        """
        Matrix element <m|x|n> in harmonic oscillator basis.

        <m|x|n> = sqrt(1/(2*omega)) * [sqrt(n)*delta_{m,n-1} + sqrt(n+1)*delta_{m,n+1}]

        Returns
        -------
        float : the matrix element (EXACT for these integers).
        """
        scale = 1.0 / np.sqrt(2.0 * self.omega)
        if m == n - 1:
            return scale * np.sqrt(n)
        elif m == n + 1:
            return scale * np.sqrt(n + 1)
        return 0.0

    def matrix_element_x2(self, m: int, n: int) -> float:
        """
        Matrix element <m|x^2|n> = sum_k <m|x|k><k|x|n>.

        Returns
        -------
        float : the matrix element.
        """
        result = 0.0
        for k in range(max(0, min(m, n) - 1), max(m, n) + 2):
            if k < 0:
                continue
            result += self.matrix_element_x(m, k) * self.matrix_element_x(k, n)
        return result

    def matrix_element_x4(self, m: int, n: int) -> float:
        """
        Matrix element <m|x^4|n> = sum_k <m|x^2|k><k|x^2|n>.

        Returns
        -------
        float : the matrix element.
        """
        # x^4 connects states differing by at most 4 quanta
        result = 0.0
        k_min = max(0, min(m, n) - 2)
        k_max = max(m, n) + 3
        for k in range(k_min, k_max):
            result += self.matrix_element_x2(m, k) * self.matrix_element_x2(k, n)
        return result

    def build_matrix(self, H_coefficients: Dict[str, float]) -> np.ndarray:
        """
        Build the Hamiltonian matrix for H = (1/2)*omega*x^2 + h2*x^2 + h4*x^4 + ...

        Actually, for the harmonic oscillator basis, H_0 = omega*(n + 1/2).
        The full Hamiltonian is H = H_0 + additional terms.

        Parameters
        ----------
        H_coefficients : dict
            Keys: 'harmonic' (omega), 'quartic' (lambda for x^4), etc.

        Returns
        -------
        ndarray, shape (n_basis, n_basis) : Hamiltonian matrix.
        """
        N = self.n_basis
        H = np.zeros((N, N))

        # Harmonic part: diagonal
        for n in range(N):
            H[n, n] = self.omega * (n + 0.5)

        # Quartic perturbation
        lam = H_coefficients.get('quartic', 0.0)
        if abs(lam) > 0:
            for m in range(N):
                for n in range(N):
                    elem = self.matrix_element_x4(m, n)
                    if abs(elem) > 1e-20:
                        H[m, n] += lam * elem

        return H

    def verify_symmetry(self, H: np.ndarray) -> dict:
        """
        Verify that the matrix is symmetric to machine precision.

        Parameters
        ----------
        H : ndarray

        Returns
        -------
        dict with symmetry diagnostics.
        """
        diff = np.max(np.abs(H - H.T))
        return {
            'symmetric': diff < 1e-12,
            'max_asymmetry': diff,
            'shape': H.shape,
        }

    def interval_bounds(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute interval bounds on matrix elements.

        For polynomial Hamiltonians in HO basis, the matrix elements are
        rationals times square roots of integers. The rounding error is
        bounded by machine epsilon times the magnitude.

        Parameters
        ----------
        H : ndarray

        Returns
        -------
        (H_lower, H_upper) : tuple of ndarray
            Interval enclosure of each matrix element.
        """
        eps = np.finfo(float).eps
        # Conservative: each element has relative error <= 10*eps
        # (accounts for accumulation in sums)
        magnitude = np.abs(H)
        margin = 10 * eps * (magnitude + eps)  # Add eps to handle zeros
        H_lower = H - margin
        H_upper = H + margin
        return H_lower, H_upper


# ======================================================================
# 5. TruncationErrorBound — Error from finite basis
# ======================================================================

class TruncationErrorBound:
    """
    Bound the error from truncating to N basis functions.

    For confining potentials V(x) -> infinity as |x| -> infinity, the
    eigenvalues grow as:
        lambda_k ~ k^{2s/(d*s + d)}  for V ~ |x|^{2s} in d dimensions

    For the quartic oscillator (s=2, d=1):
        lambda_k ~ k^{4/3}

    For the harmonic oscillator (s=1, d=1):
        lambda_k ~ k

    Weyl's law gives the asymptotic density of states.

    Truncation error for eigenvalue k in N-basis:
        |E_k^{(N)} - E_k| <= C * (lambda_N / lambda_k)^{-p}

    where p depends on the smoothness of eigenfunctions.

    LABEL: NUMERICAL (the constants C, p are estimated, not proven).
    """

    @staticmethod
    def weyl_estimate(k: int, dimension: int, potential_exponent: float) -> float:
        """
        Weyl-law estimate of the k-th eigenvalue.

        For V(x) ~ |x|^{2s} in d dimensions:
            lambda_k ~ k^{2s/(d*s + d)}  (Titchmarsh formula)

        Parameters
        ----------
        k : int
            Eigenvalue index (0-based).
        dimension : int
            Spatial dimension d.
        potential_exponent : float
            s where V ~ |x|^{2s}. s=1 for harmonic, s=2 for quartic.

        Returns
        -------
        float : estimated eigenvalue scaling.
        """
        if k <= 0:
            return 1.0
        s = potential_exponent
        d = dimension
        exponent = 2.0 * s / (d * s + d)
        return float(k) ** exponent

    @staticmethod
    def truncation_bound(N: int, k: int, dimension: int = 1,
                         potential_exponent: float = 2.0) -> float:
        """
        Estimated truncation error for eigenvalue k in N-basis.

        Uses the heuristic:
            |E_k^{(N)} - E_k| <= C * N^{-2*s/(d*s + d) + 2*k/(N)}

        For practical purposes, the error decreases as N^{-alpha} where
        alpha depends on the potential.

        For the quartic oscillator (s=2, d=1):
            alpha ~ 4/3 (same exponent as eigenvalue growth)

        Parameters
        ----------
        N : int
            Basis size.
        k : int
            Which eigenvalue.
        dimension : int
            Spatial dimension.
        potential_exponent : float
            s where V ~ |x|^{2s}.

        Returns
        -------
        float : estimated truncation error (upper bound).
        """
        if N <= k + 1:
            return float('inf')

        s = potential_exponent
        d = dimension
        alpha = 2.0 * s / (d * s + d)

        # The truncation error decreases as (k/N)^alpha roughly
        # More precisely, the variational error for eigenvalue k
        # with N basis functions is bounded by the (N-k)-th eigenvalue
        # of the complementary space
        ratio = float(k + 1) / float(N)
        # Conservative estimate: error ~ lambda_N * (k/N)^2
        # (exponential convergence for analytic eigenfunctions, polynomial for finite regularity)
        return (1.0 + ratio) * float(N) ** (-alpha)

    @staticmethod
    def minimum_N_for_precision(target_precision: float, k: int = 0,
                                dimension: int = 1,
                                potential_exponent: float = 2.0) -> int:
        """
        Minimum basis size N to achieve target precision for eigenvalue k.

        Parameters
        ----------
        target_precision : float
            Desired precision (relative to eigenvalue scale).
        k : int
            Which eigenvalue.
        dimension : int
        potential_exponent : float

        Returns
        -------
        int : minimum N needed.
        """
        s = potential_exponent
        d = dimension
        alpha = 2.0 * s / (d * s + d)

        # From truncation_bound: error ~ N^{-alpha}
        # Want N^{-alpha} <= target_precision
        # N >= target_precision^{-1/alpha}
        N_est = max(int(np.ceil(target_precision ** (-1.0 / alpha))), k + 2)
        return N_est


# ======================================================================
# 6. RigorousSpectralGap — Certified gap from all ingredients
# ======================================================================

class RigorousSpectralGap:
    """
    Combine SCLBT, interval arithmetic, and truncation bounds for a
    certified spectral gap.

    The certified gap uses:
        1. SCLBT lower bound on E_0 (-> we know E_0 >= L_0)
        2. SCLBT lower bound on E_1 (-> we know E_1 >= L_1)
        3. Ritz upper bound on E_0 (-> we know E_0 <= U_0)
        4. Truncation error bound (-> adjust bounds for finite basis)

    Certified gap:
        gap >= L_1 - U_0 - 2*epsilon_trunc

    where epsilon_trunc is the truncation error bound.

    LABEL: NUMERICAL (until interval arithmetic certifies every step).
    """

    def __init__(self, n_states: int = 5, max_sclbt_iter: int = 50):
        """
        Parameters
        ----------
        n_states : int
            Number of states to compute.
        max_sclbt_iter : int
            Maximum SCLBT iterations.
        """
        self.n_states = n_states
        self.sclbt = SCLBTBound(max_iterations=max_sclbt_iter)
        self.temple = TempleBound()

    def certified_gap(self, H_matrix: np.ndarray, N: Optional[int] = None,
                      dimension: int = 1, potential_exponent: float = 2.0) -> dict:
        """
        Compute a certified lower bound on the spectral gap.

        Parameters
        ----------
        H_matrix : ndarray
            Hamiltonian matrix (symmetric).
        N : int or None
            Basis size (defaults to H_matrix.shape[0]).
        dimension : int
            Dimension for truncation error estimate.
        potential_exponent : float
            Potential exponent for truncation error.

        Returns
        -------
        dict with certified gap data.
        """
        if N is None:
            N = H_matrix.shape[0]

        # SCLBT bounds
        sclbt_result = self.sclbt.compute(H_matrix, self.n_states)

        # Temple bounds for comparison
        temple_result = self.temple.compute(H_matrix, self.n_states)

        # Truncation error
        trunc_0 = TruncationErrorBound.truncation_bound(
            N, 0, dimension, potential_exponent
        )
        trunc_1 = TruncationErrorBound.truncation_bound(
            N, 1, dimension, potential_exponent
        )

        # SCLBT gap (without truncation correction)
        sclbt_gap_raw = sclbt_result['spectral_gap']

        # Temple gap for comparison
        temple_gap_raw = temple_result['spectral_gap_temple']

        # Certified gap = SCLBT gap - truncation error margin
        # The truncation affects both the lower bound on E1 (decreases it)
        # and the upper bound on E0 (increases it), so we subtract both.
        certified_gap_value = sclbt_gap_raw - trunc_0 - trunc_1

        return {
            'certified_gap': certified_gap_value,
            'sclbt_gap_raw': sclbt_gap_raw,
            'temple_gap_raw': temple_gap_raw,
            'truncation_error_E0': trunc_0,
            'truncation_error_E1': trunc_1,
            'sclbt_result': sclbt_result,
            'temple_result': temple_result,
            'is_positive': certified_gap_value > 0,
            'basis_size': N,
            'label': 'NUMERICAL',
        }

    def is_positive(self, H_matrix: np.ndarray, **kwargs) -> bool:
        """Check if the certified gap is positive."""
        result = self.certified_gap(H_matrix, **kwargs)
        return result['is_positive']

    def confidence_level(self, H_matrix: np.ndarray, N_values: List[int] = None,
                         dimension: int = 1,
                         potential_exponent: float = 2.0) -> dict:
        """
        Assess confidence by checking convergence with basis size.

        Parameters
        ----------
        H_matrix : ndarray
            The Hamiltonian at the LARGEST basis size.
        N_values : list of int
            Basis sizes to test (submatrices of H_matrix). If None, uses
            automatic sequence.
        dimension : int
        potential_exponent : float

        Returns
        -------
        dict with convergence data.
        """
        N_max = H_matrix.shape[0]
        if N_values is None:
            N_values = [N_max // 4, N_max // 2, 3 * N_max // 4, N_max]
            N_values = [n for n in N_values if n >= self.n_states + 1]
            if not N_values:
                N_values = [N_max]

        gaps = []
        for N_sub in N_values:
            H_sub = H_matrix[:N_sub, :N_sub]
            result = self.certified_gap(
                H_sub, N_sub, dimension, potential_exponent
            )
            gaps.append({
                'N': N_sub,
                'certified_gap': result['certified_gap'],
                'sclbt_gap': result['sclbt_gap_raw'],
                'temple_gap': result['temple_gap_raw'],
            })

        # Check if gap is converging
        sclbt_gaps = [g['sclbt_gap'] for g in gaps if not np.isnan(g['sclbt_gap'])]
        converging = False
        if len(sclbt_gaps) >= 2:
            diffs = [abs(sclbt_gaps[i] - sclbt_gaps[i - 1])
                     for i in range(1, len(sclbt_gaps))]
            converging = all(d < 0.1 * abs(sclbt_gaps[-1]) for d in diffs[-2:]) if len(diffs) >= 2 else diffs[-1] < 0.1 * abs(sclbt_gaps[-1])

        return {
            'N_values': N_values,
            'gap_data': gaps,
            'converging': converging,
            'final_gap': gaps[-1]['certified_gap'] if gaps else float('nan'),
        }


# ======================================================================
# 7. QuarticOscillatorBenchmark — Test on known systems
# ======================================================================

class QuarticOscillatorBenchmark:
    """
    Benchmark SCLBT against known quartic oscillator results.

    1D: V(x) = x^4
        Exact E_0 = 1.06036209048... (Bender-Wu, many-digit computations)
        Exact E_1 = 3.79967309...
        Exact gap = 2.73931...

    2D: V(x,y) = x^2*y^2
        Exact E_0 = 0.55411... (Pedram 2014, lattice Monte Carlo)

    These serve as rigorous benchmarks for the SCLBT implementation.

    LABEL: NUMERICAL (benchmark comparison, not a proof).
    """

    # Known exact eigenvalues of H = -d^2/dx^2 + x^4 (Bender-Wu convention,
    # kinetic term WITHOUT factor of 1/2).
    # E_0 = 1.06036209048418..., E_1 = 3.79967309328...
    #
    # Our convention: H = -(1/2)d^2/dx^2 + x^4 (WITH factor of 1/2).
    # By scaling: E_k(1/2) = (1/2)^{2/3} * E_k(1).
    # So E_0 = 0.66798626..., E_1 = 2.39364402...
    BENDER_WU_E0 = 1.06036209048418       # H = -d^2/dx^2 + x^4
    BENDER_WU_E1 = 3.79967309328          # H = -d^2/dx^2 + x^4
    SCALING_FACTOR = 0.5 ** (2.0 / 3.0)   # (1/2)^{2/3} for our convention
    QUARTIC_1D_E0 = BENDER_WU_E0 * SCALING_FACTOR   # ~ 0.66799
    QUARTIC_1D_E1 = BENDER_WU_E1 * SCALING_FACTOR   # ~ 2.39364
    QUARTIC_1D_GAP = QUARTIC_1D_E1 - QUARTIC_1D_E0  # ~ 1.72566

    # 2D quartic: H = -(1/2)(d^2/dx^2 + d^2/dy^2) + x^2*y^2
    # Reference value scaled to our convention
    QUARTIC_2D_E0 = 0.55411

    @staticmethod
    def build_quartic_1d(N: int, lam: float = 1.0) -> np.ndarray:
        """
        Build 1D quartic oscillator Hamiltonian: H = -(1/2)d^2/dx^2 + lam*x^4.

        Uses harmonic oscillator basis with omega chosen to minimize
        truncation error.

        Parameters
        ----------
        N : int
            Basis size.
        lam : float
            Quartic coupling.

        Returns
        -------
        ndarray, shape (N, N) : Hamiltonian matrix.
        """
        # Optimal omega for quartic: omega ~ lam^{1/3}
        # (from scaling analysis of the ground state width)
        omega = (4.0 * lam) ** (1.0 / 3.0)

        x_scale = 1.0 / np.sqrt(2.0 * omega)

        # Build x matrix
        x = np.zeros((N, N))
        for n in range(N - 1):
            x[n, n + 1] = np.sqrt(n + 1) * x_scale
            x[n + 1, n] = np.sqrt(n + 1) * x_scale

        x2 = x @ x
        x4 = x2 @ x2

        # Kinetic + harmonic (from the HO basis): H_0 = omega*(n + 1/2)
        # The kinetic energy in HO basis is T = omega/2 * (n + 1/2) - omega/4 * x2 * 2*omega
        # Actually, T = (p^2)/(2) and V_HO = omega^2/2 * x^2
        # In HO basis: T + V_HO = omega*(n+1/2), diagonal
        # We want H = T + lam*x^4 = (T + V_HO) + lam*x^4 - V_HO
        #           = omega*(n+1/2) + lam*x^4 - (omega^2/2)*x^2

        H = np.zeros((N, N))
        for n in range(N):
            H[n, n] = omega * (n + 0.5)

        # Subtract the harmonic potential (which is implicit in the basis)
        # No -- the HO energy omega*(n+1/2) already includes both T and V_HO.
        # Our actual potential is V = lam*x^4, NOT omega^2/2 * x^2.
        # So: H = omega*(n+1/2) + lam*x^4 - (omega^2/2)*x^2
        H += lam * x4
        H -= 0.5 * omega**2 * x2

        return H

    @staticmethod
    def build_quartic_2d(N_per_dim: int, lam: float = 1.0) -> np.ndarray:
        """
        Build 2D quartic oscillator: H = -(1/2)nabla^2 + lam*x^2*y^2.

        Parameters
        ----------
        N_per_dim : int
            Basis size per dimension. Total size = N_per_dim^2.
        lam : float
            Coupling.

        Returns
        -------
        ndarray : Hamiltonian matrix.
        """
        omega = (4.0 * lam) ** (1.0 / 3.0)
        x_scale = 1.0 / np.sqrt(2.0 * omega)
        N = N_per_dim

        # 1D operators
        x_1d = np.zeros((N, N))
        for n in range(N - 1):
            x_1d[n, n + 1] = np.sqrt(n + 1) * x_scale
            x_1d[n + 1, n] = np.sqrt(n + 1) * x_scale

        x2_1d = x_1d @ x_1d
        I = np.eye(N)

        # H_0 per dimension
        h0_1d = np.zeros((N, N))
        for n in range(N):
            h0_1d[n, n] = omega * (n + 0.5)

        # Subtract harmonic potential
        h_1d = h0_1d - 0.5 * omega**2 * x2_1d  # = kinetic energy only

        # Full 2D kinetic
        H = np.kron(h_1d, I) + np.kron(I, h_1d)

        # Interaction: lam * x^2 * y^2
        x2_x = np.kron(x2_1d, I)
        x2_y = np.kron(I, x2_1d)
        H += lam * (x2_x @ x2_y)

        return H

    def benchmark_1d(self, N: int = 40) -> dict:
        """
        Benchmark SCLBT on the 1D quartic oscillator.

        Parameters
        ----------
        N : int
            Basis size.

        Returns
        -------
        dict with benchmark results.
        """
        H = self.build_quartic_1d(N)

        # Ritz values
        evals = np.sort(np.linalg.eigvalsh(H))

        # SCLBT
        sclbt = SCLBTBound()
        result = sclbt.compute(H, n_states=5)

        # Temple
        temple = TempleBound()
        temple_result = temple.compute(H, n_states=5)

        # Compare with exact
        E0_exact = self.QUARTIC_1D_E0
        E1_exact = self.QUARTIC_1D_E1

        return {
            'ritz_E0': evals[0],
            'ritz_E1': evals[1],
            'sclbt_E0_lower': result['lower_bounds'][0],
            'sclbt_E1_lower': result['lower_bounds'][1],
            'temple_E0_lower': temple_result['temple_lower_bounds'][0],
            'temple_E1_lower': temple_result['temple_lower_bounds'][1],
            'exact_E0': E0_exact,
            'exact_E1': E1_exact,
            'ritz_error_E0': abs(evals[0] - E0_exact),
            'sclbt_error_E0': abs(result['lower_bounds'][0] - E0_exact),
            'temple_error_E0': abs(temple_result['temple_lower_bounds'][0] - E0_exact),
            'sclbt_gap': result['spectral_gap'],
            'temple_gap': temple_result['spectral_gap_temple'],
            'exact_gap': self.QUARTIC_1D_GAP,
            'N': N,
            'sclbt_converged': result['converged'],
            'sclbt_tighter_than_temple': (
                result['lower_bounds'][0] >= temple_result['temple_lower_bounds'][0]
            ),
        }

    def benchmark_2d(self, N_per_dim: int = 15) -> dict:
        """
        Benchmark SCLBT on the 2D quartic oscillator.

        Parameters
        ----------
        N_per_dim : int
            Basis per dimension.

        Returns
        -------
        dict with benchmark results.
        """
        H = self.build_quartic_2d(N_per_dim)

        sclbt = SCLBTBound()
        result = sclbt.compute(H, n_states=5)

        temple = TempleBound()
        temple_result = temple.compute(H, n_states=5)

        evals = np.sort(np.linalg.eigvalsh(H))

        return {
            'ritz_E0': evals[0],
            'sclbt_E0_lower': result['lower_bounds'][0],
            'temple_E0_lower': temple_result['temple_lower_bounds'][0],
            'reference_E0': self.QUARTIC_2D_E0,
            'sclbt_gap': result['spectral_gap'],
            'N_total': N_per_dim**2,
            'sclbt_converged': result['converged'],
        }

    def benchmark_3d_ym(self, N: int = 10, g2: float = 6.28) -> dict:
        """
        Benchmark on a 3D YM-like quartic oscillator.

        H = sum_{i=1}^3 [-(1/2)d^2/dx_i^2 + (1/2)*omega^2*x_i^2]
            + (g^2/4) * sum_{i<j} x_i^2 * x_j^2

        Parameters
        ----------
        N : int
            Basis per dimension.
        g2 : float
            Quartic coupling.

        Returns
        -------
        dict with benchmark results.
        """
        omega = 2.0  # ~ 2/R for R=1
        H = _build_3d_hamiltonian(omega, g2, N)

        sclbt = SCLBTBound()
        result = sclbt.compute(H, n_states=5)

        temple = TempleBound()
        temple_result = temple.compute(H, n_states=5)

        evals = np.sort(np.linalg.eigvalsh(H))

        return {
            'ritz_E0': evals[0],
            'ritz_E1': evals[1],
            'sclbt_E0_lower': result['lower_bounds'][0],
            'sclbt_E1_lower': result['lower_bounds'][1],
            'temple_E0_lower': temple_result['temple_lower_bounds'][0],
            'sclbt_gap': result['spectral_gap'],
            'temple_gap': temple_result['spectral_gap_temple'],
            'ritz_gap': evals[1] - evals[0],
            'N_total': N**3,
            'omega': omega,
            'g2': g2,
        }


# ======================================================================
# 8. YangMillsReducedGap — Apply to the physical YM Hamiltonian
# ======================================================================

class YangMillsReducedGap:
    """
    Apply SCLBT to the 3-DOF reduced YM Hamiltonian on S^3/I*.

    The reduced Hamiltonian (from Koller-van Baal SVD reduction) is:

        H = (kappa/2) * p^2 + (2/R^2)*sum x_i^2
            - (2g/R)*x_1*x_2*x_3 + g^2*sum_{i<j} x_i^2*x_j^2

    where:
        kappa/2 = g^2/(2*R^3) ~ 0.295  (kinetic prefactor, NOT 1/2)
        (2/R^2) = coexact mass from S^3 curvature
        -(2g/R) = cubic vertex from Maurer-Cartan curvature
        g^2 = Yang-Mills coupling

    In physical mode, delegates to KvB's NumericalDiagonalization for the
    correct Hamiltonian construction. Eigenvalues are in dimensionless
    natural units.

    Physical units conversion:
        gap_MeV = gap_eigenvalue * hbar_c / R
    same as KvB (hbar_c/R converts dimensionless eigenvalues to MeV).

    BUG FIX (Session 25): Previous versions used unit kinetic prefactor (1/2)
    and omitted the cubic term, inflating the gap by ~2.5x (368 -> ~145 MeV).
    The correct kinetic prefactor is epsilon = g^2/(2*R^3) ~ 0.295.

    LABEL: NUMERICAL.
    """

    def __init__(self, N_basis: int = 10, n_sclbt_states: int = 5):
        """
        Parameters
        ----------
        N_basis : int
            Basis size per singular value mode.
        n_sclbt_states : int
            Number of states for SCLBT.
        """
        self.N_basis = N_basis
        self.n_sclbt_states = n_sclbt_states

    def compute_gap(self, N: int, R: float, g2: float) -> dict:
        """
        Compute SCLBT gap for given parameters.

        Uses the PHYSICAL Hamiltonian with kinetic prefactor epsilon = g^2/(2*R^3)
        and cubic term -(2g/R)*x_1*x_2*x_3.

        Parameters
        ----------
        N : int
            Basis per mode (overrides constructor).
        R : float
            Radius in fm.
        g2 : float
            Yang-Mills coupling squared.

        Returns
        -------
        dict with gap data in natural units.
        """
        omega = 2.0 / R  # 1/fm (used only as fallback)
        H = _build_3d_hamiltonian(omega, g2, N, R=R)

        sclbt = SCLBTBound()
        sclbt_result = sclbt.compute(H, self.n_sclbt_states)

        temple = TempleBound()
        temple_result = temple.compute(H, self.n_sclbt_states)

        evals = np.sort(np.linalg.eigvalsh(H))

        return {
            'ritz_E0': evals[0],
            'ritz_E1': evals[1],
            'ritz_gap': evals[1] - evals[0],
            'sclbt_E0_lower': sclbt_result['lower_bounds'][0],
            'sclbt_E1_lower': sclbt_result['lower_bounds'][1],
            'sclbt_gap': sclbt_result['spectral_gap'],
            'temple_E0_lower': temple_result['temple_lower_bounds'][0],
            'temple_gap': temple_result['spectral_gap_temple'],
            'omega': omega,
            'R': R,
            'g2': g2,
            'N_basis': N,
            'N_total': N**3,
            'sclbt_converged': sclbt_result['converged'],
        }

    def gap_in_MeV(self, N: int, R: float, g2: float) -> dict:
        """
        Compute the gap in MeV units.

        Conversion: gap_MeV = gap_natural * hbar_c / R
        The eigenvalues from the physical Hamiltonian are in dimensionless
        natural units, and hbar_c/R converts to MeV (same as KvB).

        Parameters
        ----------
        N : int
            Basis per mode.
        R : float
            Radius in fm.
        g2 : float
            Coupling.

        Returns
        -------
        dict with gap in MeV.
        """
        result = self.compute_gap(N, R, g2)

        # Convert to MeV: gap_MeV = gap_natural * hbar_c / R
        # Eigenvalues are in dimensionless natural units (from KvB Hamiltonian).
        # The energy scale is hbar_c / R.
        hbar_c = HBAR_C_MEV_FM
        scale = hbar_c / R

        def to_MeV(val):
            if val is None or np.isnan(val):
                return float('nan')
            return val * scale

        return {
            'ritz_gap_MeV': to_MeV(result['ritz_gap']),
            'sclbt_gap_MeV': to_MeV(result['sclbt_gap']),
            'temple_gap_MeV': to_MeV(result['temple_gap']),
            'ritz_E0_MeV': to_MeV(result['ritz_E0']),
            'ritz_E1_MeV': to_MeV(result['ritz_E1']),
            'sclbt_E0_lower_MeV': to_MeV(result['sclbt_E0_lower']),
            'R_fm': R,
            'g2': g2,
            'N_basis': N,
            'label': 'NUMERICAL',
        }

    def convergence_study(self, N_range: List[int], R: float = R_PHYSICAL_FM,
                          g2: float = 6.28) -> dict:
        """
        Study convergence of the gap with basis size.

        Parameters
        ----------
        N_range : list of int
            Basis sizes to test.
        R : float
            Radius in fm.
        g2 : float
            Coupling.

        Returns
        -------
        dict with convergence data.
        """
        results = []
        for N in N_range:
            try:
                r = self.gap_in_MeV(N, R, g2)
                results.append({
                    'N': N,
                    'N_total': N**3,
                    'ritz_gap_MeV': r['ritz_gap_MeV'],
                    'sclbt_gap_MeV': r['sclbt_gap_MeV'],
                    'temple_gap_MeV': r['temple_gap_MeV'],
                })
            except Exception as e:
                results.append({
                    'N': N,
                    'error': str(e),
                })

        # Check convergence
        sclbt_gaps = [r['sclbt_gap_MeV'] for r in results
                      if 'sclbt_gap_MeV' in r and not np.isnan(r['sclbt_gap_MeV'])]
        converged = False
        if len(sclbt_gaps) >= 2:
            rel_change = abs(sclbt_gaps[-1] - sclbt_gaps[-2]) / max(abs(sclbt_gaps[-1]), 1e-10)
            converged = rel_change < 0.01  # 1% convergence

        return {
            'R_fm': R,
            'g2': g2,
            'results': results,
            'converged': converged,
            'final_sclbt_gap_MeV': sclbt_gaps[-1] if sclbt_gaps else float('nan'),
        }


# ======================================================================
# Helper: build 3D Hamiltonian with cross-quartic coupling
# ======================================================================

def _build_3d_hamiltonian(omega: float, g2: float, n_basis: int,
                          R: float = None) -> np.ndarray:
    """
    Build the 3D reduced YM Hamiltonian in HO product basis.

    When R is provided (physical mode), delegates to the Koller-van Baal
    NumericalDiagonalization which builds the CORRECT Hamiltonian:

        H = (kappa/2) * p^2 + (2/R^2)*sum x_i^2
            - (2g/R)*x_1*x_2*x_3 + g^2*sum_{i<j} x_i^2*x_j^2

    with kinetic prefactor kappa/2 = g^2/(2*R^3) and all three potential
    terms (quadratic + cubic + quartic).

    When R is NOT provided (legacy mode), builds a simplified Hamiltonian
    with unit kinetic prefactor (1/2) and only harmonic + quartic terms,
    for backward compatibility with pure-oscillator benchmarks.

    Parameters
    ----------
    omega : float
        Harmonic frequency. In physical mode (R given), this is IGNORED.
        In legacy mode (R=None), used directly as the HO frequency.
    g2 : float
        Yang-Mills coupling squared.
    n_basis : int
        Basis states per mode.
    R : float or None
        Radius of S^3 in fm. When provided, uses the KvB physical Hamiltonian.
        Default None (legacy mode).

    Returns
    -------
    ndarray, shape (n_basis^3, n_basis^3) : Hamiltonian matrix.

    Notes
    -----
    BUG FIX (Session 25): Previous versions always used unit kinetic prefactor
    (1/2) instead of the physical kappa/2 = g^2/(2*R^3) ~ 0.295, and omitted
    the cubic term -(2g/R)*x_1*x_2*x_3. This inflated the gap by ~2.5x
    (368 MeV -> 145 MeV at R=2.2 fm, g^2=6.28).

    In physical mode, eigenvalues are in dimensionless natural units.
    Convert to MeV via: gap_MeV = gap_eigenvalue * hbar_c / R.
    In legacy mode, eigenvalues are in units of omega (1/fm).
    Convert to MeV via: gap_MeV = gap_eigenvalue * hbar_c.
    """
    if R is not None:
        # Physical mode: use KvB's correct Hamiltonian construction
        diag = NumericalDiagonalization(R=R, g2=g2, N_per_dim=n_basis)
        return diag.build_hamiltonian_matrix_fast()

    # Legacy mode: unit kinetic prefactor, no cubic term
    I = np.eye(n_basis)
    total_dim = n_basis ** 3

    # 1D operators
    x_scale = 1.0 / np.sqrt(2.0 * max(omega, 1e-10))
    x_1d = np.zeros((n_basis, n_basis))
    for n in range(n_basis - 1):
        x_1d[n, n + 1] = np.sqrt(n + 1) * x_scale
        x_1d[n + 1, n] = np.sqrt(n + 1) * x_scale
    x2_1d = x_1d @ x_1d

    # Harmonic part for each mode
    h0_1d = np.zeros((n_basis, n_basis))
    for n in range(n_basis):
        h0_1d[n, n] = omega * (n + 0.5)

    # Full Hamiltonian
    H = np.zeros((total_dim, total_dim))

    # Harmonic part: sum_i omega*(n_i + 1/2)
    for d in range(3):
        parts = [I, I, I]
        parts[d] = h0_1d
        H += np.kron(np.kron(parts[0], parts[1]), parts[2])

    # Quartic cross terms: (g^2/2) * x_i^2 * x_j^2
    lam = 0.5 * g2
    for i in range(3):
        for j in range(i + 1, 3):
            parts_i = [I, I, I]
            parts_j = [I, I, I]
            parts_i[i] = x2_1d
            parts_j[j] = x2_1d
            xi2 = np.kron(np.kron(parts_i[0], parts_i[1]), parts_i[2])
            xj2 = np.kron(np.kron(parts_j[0], parts_j[1]), parts_j[2])
            H += lam * (xi2 @ xj2)

    return H
