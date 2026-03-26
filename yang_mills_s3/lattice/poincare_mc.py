"""
Monte Carlo Yang-Mills on S^3/I* -- Non-perturbative gap measurement.

Performs standard lattice YM Monte Carlo on the 600-cell, then projects
observables onto the I*-invariant subspace to measure the gap on S^3/I*.

The key observable: the plaquette-plaquette correlator.
Its exponential decay rate gives the mass gap on S^3/I*.

STRATEGY:
1. Thermalize SU(2) gauge field on the 600-cell at coupling beta
2. Measure plaquette field on each face
3. Compute plaquette-plaquette correlators binned by geodesic distance
4. Project correlators onto I*-invariant subspace
5. Extract mass gap from exponential decay
6. Compare with free-theory prediction: m_gap = 2/R

STATUS: NUMERICAL
"""

import numpy as np
from scipy.optimize import curve_fit
from .s3_lattice import S3Lattice
from .lattice_ym import LatticeYM
from .poincare_lattice import PoincareLattice


class PoincareMC:
    """
    Monte Carlo measurement of Yang-Mills observables on S^3/I*.
    """

    def __init__(self, N=2, beta=4.0, R=1.0):
        """
        Parameters
        ----------
        N : gauge group SU(N)
        beta : lattice coupling (beta = 2N/g^2)
        R : radius of S^3
        """
        self.N = N
        self.beta = beta
        self.R = R
        self.poincare = PoincareLattice(R)
        self.lattice = self.poincare.lattice
        self.ym = LatticeYM(self.lattice, N=N, beta=beta)

        # I* projector on edges (720 x 720)
        self._pi_edge = self.poincare.istar_projector_edges()

        # Precompute face centers and distance matrix for correlator binning
        self._face_centers = self._compute_face_centers()
        self._face_distances = self._compute_face_distances()

        # Precompute I* face projector for correlator projection
        self._pi_face = self._build_face_projector()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _compute_face_centers(self):
        """
        Compute the center of each triangular face on S^3.

        Returns (n_faces, 4) array of unit vectors (projected to S^3).
        """
        verts = self.lattice.vertices / self.R  # unit sphere
        faces = self.lattice.faces()
        centers = np.zeros((len(faces), 4))
        for f_idx, (i, j, k) in enumerate(faces):
            c = (verts[i] + verts[j] + verts[k]) / 3.0
            norm = np.linalg.norm(c)
            if norm > 1e-12:
                centers[f_idx] = c / norm
            else:
                centers[f_idx] = c
        return centers

    def _compute_face_distances(self):
        """
        Compute geodesic distances between all face centers on unit S^3.

        Returns (n_faces, n_faces) distance matrix.
        """
        centers = self._face_centers
        # Geodesic distance = arccos(dot product) on S^3
        dots = centers @ centers.T
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        return np.arccos(dots)

    def _build_face_projector(self):
        """
        Build I*-invariant projector on face space.

        The I* action on vertices induces an action on faces.
        Pi_face = (1/|I*|) sum_h P_h^face.

        Returns (n_faces, n_faces) projector.
        """
        n_f = self.poincare._n_faces
        n_v = self.poincare._n_vertices
        face_list = self.poincare._face_list

        # Build face index lookup
        face_set = {}
        for f_idx, face in enumerate(face_list):
            face_set[tuple(sorted(face))] = f_idx

        Pi = np.zeros((n_f, n_f))

        for h_idx in range(n_v):
            sigma = self.poincare._perms[h_idx]
            P_h = np.zeros((n_f, n_f))
            for f_idx, (i, j, k) in enumerate(face_list):
                new_face = tuple(sorted([sigma[i], sigma[j], sigma[k]]))
                if new_face in face_set:
                    P_h[f_idx, face_set[new_face]] = 1.0
            Pi += P_h

        Pi /= n_v
        return Pi

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def thermalize(self, n_sweeps=100, epsilon=0.3, rng=None):
        """Thermalize the gauge field configuration."""
        return self.ym.thermalize(n_sweeps=n_sweeps, epsilon=epsilon, rng=rng)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def measure_plaquette_field(self):
        """
        Measure the plaquette field: P_f = (1/N) Re Tr U_plaq for each face f.

        Returns (n_faces,) array of plaquette values in [-1, 1].
        """
        plaquettes = self.ym._plaquettes
        n_faces = len(plaquettes)
        field = np.zeros(n_faces)
        for f_idx, plaq in enumerate(plaquettes):
            field[f_idx] = self.ym.plaquette_trace(plaq)
        return field

    def project_to_istar(self, field_on_edges):
        """
        Project an edge-based field onto the I*-invariant subspace.

        Parameters
        ----------
        field_on_edges : (720,) or (720, ...) array

        Returns
        -------
        Projected field (same shape)
        """
        if field_on_edges.ndim == 1:
            return self._pi_edge @ field_on_edges
        else:
            # Matrix @ matrix for batch projection
            return self._pi_edge @ field_on_edges

    def project_faces_to_istar(self, field_on_faces):
        """
        Project a face-based field onto the I*-invariant subspace.

        Parameters
        ----------
        field_on_faces : (n_faces,) array

        Returns
        -------
        Projected field (same shape)
        """
        return self._pi_face @ field_on_faces

    def plaquette_plaquette_correlator(self, n_configs=50, n_therm=50,
                                       n_skip=5, epsilon=0.3, rng=None):
        """
        Measure the plaquette-plaquette correlator.

        C(d) = <P(x) P(y)>_connected averaged over pairs at geodesic distance d.

        The exponential decay of C(d) gives the mass gap:
        C(d) ~ exp(-m_gap * d) for large d.

        Parameters
        ----------
        n_configs : number of gauge configurations to average over
        n_therm : thermalization sweeps before measurement
        n_skip : sweeps between measurements (decorrelation)
        epsilon : Metropolis step size
        rng : numpy random generator

        Returns
        -------
        dict with:
            'distances' : array of unique geodesic distances
            'correlator' : C(d) values
            'correlator_err' : statistical errors
            'correlator_istar' : I*-projected correlator
            'correlator_istar_err' : errors on projected correlator
        """
        if rng is None:
            rng = np.random.default_rng()

        n_faces = len(self.lattice.faces())
        dist_matrix = self._face_distances

        # Bin distances: use discrete bins
        # Get unique distances (rounded for binning)
        all_dists = dist_matrix[np.triu_indices(n_faces, k=1)]
        n_bins = min(20, max(5, int(np.sqrt(len(all_dists)) / 5)))
        bin_edges = np.linspace(0.0, np.max(all_dists) + 1e-10, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Assign face pairs to distance bins
        pair_bins = np.digitize(dist_matrix, bin_edges) - 1
        # Clip to valid range
        pair_bins = np.clip(pair_bins, 0, n_bins - 1)

        # Randomize and thermalize
        self.ym.randomize_links(rng)
        self.ym.thermalize(n_sweeps=n_therm, epsilon=epsilon, rng=rng)

        # Collect per-config correlators
        corr_samples = np.zeros((n_configs, n_bins))
        corr_istar_samples = np.zeros((n_configs, n_bins))

        for cfg in range(n_configs):
            # MC sweeps for decorrelation
            self.ym.thermalize(n_sweeps=n_skip, epsilon=epsilon, rng=rng)

            # Measure plaquette field
            plaq_field = self.measure_plaquette_field()
            plaq_mean = np.mean(plaq_field)
            plaq_fluct = plaq_field - plaq_mean

            # I*-projected plaquette field
            plaq_istar = self.project_faces_to_istar(plaq_field)
            plaq_istar_mean = np.mean(plaq_istar)
            plaq_istar_fluct = plaq_istar - plaq_istar_mean

            # Compute correlator for each distance bin
            for b in range(n_bins):
                mask = (pair_bins == b)
                # Exclude diagonal
                np.fill_diagonal(mask, False)
                if np.sum(mask) == 0:
                    continue

                # Full correlator
                # C(d) = mean over pairs of P(x)*P(y) - <P>^2
                # Use outer product of fluctuations
                pairs_i, pairs_j = np.where(mask)
                corr_val = np.mean(plaq_fluct[pairs_i] * plaq_fluct[pairs_j])
                corr_samples[cfg, b] = corr_val

                # I*-projected correlator
                corr_istar_val = np.mean(plaq_istar_fluct[pairs_i] * plaq_istar_fluct[pairs_j])
                corr_istar_samples[cfg, b] = corr_istar_val

        # Average over configs
        correlator = np.mean(corr_samples, axis=0)
        correlator_err = np.std(corr_samples, axis=0) / np.sqrt(max(n_configs, 1))

        correlator_istar = np.mean(corr_istar_samples, axis=0)
        correlator_istar_err = np.std(corr_istar_samples, axis=0) / np.sqrt(max(n_configs, 1))

        return {
            'distances': bin_centers,
            'correlator': correlator,
            'correlator_err': correlator_err,
            'correlator_istar': correlator_istar,
            'correlator_istar_err': correlator_istar_err,
        }

    def extract_mass_gap(self, distances, correlator, correlator_err=None):
        """
        Extract mass gap from correlator decay.

        Fit C(d) = A * exp(-m * d) to the correlator data.

        Parameters
        ----------
        distances : array of distances
        correlator : array of correlator values
        correlator_err : optional array of errors

        Returns
        -------
        dict with 'mass_gap', 'amplitude', 'chi_squared', 'mass_gap_err'
        """
        # Filter to positive correlator values for log fit
        mask = correlator > 0
        if np.sum(mask) < 2:
            # Not enough points for a fit, use simple ratio
            if np.sum(mask) >= 1:
                return {
                    'mass_gap': 0.0,
                    'amplitude': float(correlator[mask][0]) if np.any(mask) else 0.0,
                    'chi_squared': float('inf'),
                    'mass_gap_err': float('inf'),
                }
            return {
                'mass_gap': 0.0,
                'amplitude': 0.0,
                'chi_squared': float('inf'),
                'mass_gap_err': float('inf'),
            }

        d_fit = distances[mask]
        c_fit = correlator[mask]

        # Try nonlinear fit: C(d) = A * exp(-m * d)
        def exp_decay(d, A, m):
            return A * np.exp(-m * d)

        try:
            sigma = None
            if correlator_err is not None:
                sigma_fit = correlator_err[mask]
                # Only use sigma if all positive and nonzero
                if np.all(sigma_fit > 0):
                    sigma = sigma_fit

            # Initial guesses
            A0 = c_fit[0]
            if len(c_fit) >= 2 and c_fit[0] > 0 and c_fit[1] > 0 and d_fit[1] > d_fit[0]:
                m0 = -np.log(c_fit[1] / c_fit[0]) / (d_fit[1] - d_fit[0])
                m0 = max(m0, 0.1)
            else:
                m0 = 1.0

            popt, pcov = curve_fit(exp_decay, d_fit, c_fit, p0=[A0, m0],
                                   sigma=sigma, maxfev=5000,
                                   bounds=([0, 0], [np.inf, np.inf]))
            A_fit, m_fit = popt
            perr = np.sqrt(np.diag(pcov))
            m_err = perr[1] if len(perr) > 1 else float('inf')

            # Chi-squared
            residuals = c_fit - exp_decay(d_fit, A_fit, m_fit)
            if sigma is not None:
                chi2 = np.sum((residuals / sigma) ** 2)
            else:
                chi2 = np.sum(residuals ** 2) / np.mean(c_fit ** 2) if np.any(c_fit != 0) else 0.0

            return {
                'mass_gap': float(m_fit),
                'amplitude': float(A_fit),
                'chi_squared': float(chi2),
                'mass_gap_err': float(m_err),
            }
        except (RuntimeError, ValueError):
            # Fallback: log-ratio estimate
            if len(c_fit) >= 2 and c_fit[0] > 0 and c_fit[1] > 0:
                m_est = -np.log(c_fit[1] / c_fit[0]) / (d_fit[1] - d_fit[0]) if d_fit[1] > d_fit[0] else 0.0
                m_est = max(m_est, 0.0)
            else:
                m_est = 0.0
            return {
                'mass_gap': float(m_est),
                'amplitude': float(c_fit[0]),
                'chi_squared': float('inf'),
                'mass_gap_err': float('inf'),
            }

    def istar_plaquette_average(self, n_configs=50, n_therm=50,
                                 n_skip=5, epsilon=0.3, rng=None):
        """
        Measure the average plaquette in the I*-invariant sector.

        This is a simpler observable than the full correlator.

        Parameters
        ----------
        n_configs : number of configurations
        n_therm : thermalization sweeps
        n_skip : sweeps between measurements
        epsilon : Metropolis step size
        rng : random generator

        Returns
        -------
        dict with:
            'plaq_avg_full' : average plaquette on full S^3
            'plaq_avg_istar' : average projected plaquette
            'n_configs' : number of configurations
        """
        if rng is None:
            rng = np.random.default_rng()

        # Randomize and thermalize
        self.ym.randomize_links(rng)
        self.ym.thermalize(n_sweeps=n_therm, epsilon=epsilon, rng=rng)

        plaq_sum_full = 0.0
        plaq_sum_istar = 0.0

        for cfg in range(n_configs):
            self.ym.thermalize(n_sweeps=n_skip, epsilon=epsilon, rng=rng)
            plaq_field = self.measure_plaquette_field()
            plaq_istar = self.project_faces_to_istar(plaq_field)

            plaq_sum_full += np.mean(plaq_field)
            plaq_sum_istar += np.mean(plaq_istar)

        return {
            'plaq_avg_full': plaq_sum_full / n_configs,
            'plaq_avg_istar': plaq_sum_istar / n_configs,
            'n_configs': n_configs,
        }

    def scan_beta(self, beta_values, n_configs=30, n_therm=30,
                   n_skip=3, epsilon=0.3, rng=None):
        """
        Scan coupling beta and measure observables at each value.

        Parameters
        ----------
        beta_values : list/array of beta values
        n_configs : configs per beta
        n_therm : thermalization sweeps
        n_skip : sweeps between measurements
        epsilon : Metropolis step size
        rng : random generator

        Returns list of dicts with beta, plaq_avg, plaq_istar, etc.
        """
        if rng is None:
            rng = np.random.default_rng()

        results = []
        for beta in beta_values:
            self.ym.beta = beta
            result = self.istar_plaquette_average(
                n_configs=n_configs, n_therm=n_therm,
                n_skip=n_skip, epsilon=epsilon, rng=rng
            )
            result['beta'] = beta
            results.append(result)

        return results

    def full_measurement(self, n_configs=50, n_therm=100,
                          n_skip=5, epsilon=0.3, rng=None):
        """
        Complete measurement: correlator + gap extraction + I* comparison.

        Returns comprehensive results dict.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Measure correlator
        corr_result = self.plaquette_plaquette_correlator(
            n_configs=n_configs, n_therm=n_therm,
            n_skip=n_skip, epsilon=epsilon, rng=rng
        )

        # Extract mass gap from full correlator
        gap_full = self.extract_mass_gap(
            corr_result['distances'],
            corr_result['correlator'],
            corr_result['correlator_err']
        )

        # Extract mass gap from I*-projected correlator
        gap_istar = self.extract_mass_gap(
            corr_result['distances'],
            corr_result['correlator_istar'],
            corr_result['correlator_istar_err']
        )

        # Free-theory prediction
        free_gap = 4.0 / (self.R ** 2)  # eigenvalue 4/R^2

        return {
            'correlator': corr_result,
            'gap_full': gap_full,
            'gap_istar': gap_istar,
            'free_theory_gap': free_gap,
            'beta': self.beta,
            'N': self.N,
            'R': self.R,
            'n_configs': n_configs,
        }
