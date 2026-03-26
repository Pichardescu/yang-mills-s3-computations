"""
Hopf Fibration — The principal U(1)-bundle S^3 -> S^2.

Provides the projection map, fiber reconstruction, connection 1-form,
curvature, first Chern number, and linking number computation.
"""

import numpy as np


class HopfFibration:
    """The Hopf fibration  pi: S^3 -> S^2."""

    @staticmethod
    def projection(z1, z2):
        """
        Hopf map  pi: S^3 (subset C^2) -> S^2 (subset R^3).

        Parameters
        ----------
        z1, z2 : complex numbers with |z1|^2 + |z2|^2 = R^2

        Returns
        -------
        (x, y, z) : point on unit S^2
                     x = 2*Re(z1 * conj(z2))
                     y = 2*Im(z1 * conj(z2))
                     z = |z1|^2 - |z2|^2
                     all divided by (|z1|^2 + |z2|^2)
        """
        product = z1 * np.conj(z2)
        x = 2 * np.real(product)
        y = 2 * np.imag(product)
        z = np.abs(z1)**2 - np.abs(z2)**2
        norm = np.abs(z1)**2 + np.abs(z2)**2
        if norm > 0:
            x, y, z = x / norm, y / norm, z / norm
        return (x, y, z)

    @staticmethod
    def fiber(point_on_s2, num_points=100):
        """
        Compute the S^1 fiber over a point on the unit S^2.

        Parameters
        ----------
        point_on_s2 : (x, y, z) on unit S^2
        num_points  : int

        Returns
        -------
        points : (num_points, 4) array of (Re z1, Im z1, Re z2, Im z2) on unit S^3
        """
        x, y, z = point_on_s2
        norm = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / norm, y / norm, z / norm

        cos_eta = np.sqrt(np.clip((1 + z) / 2, 0, 1))
        sin_eta = np.sqrt(np.clip((1 - z) / 2, 0, 1))

        if sin_eta < 1e-14:
            z1_0 = complex(1, 0)
            z2_0 = complex(0, 0)
        elif cos_eta < 1e-14:
            z1_0 = complex(0, 0)
            z2_0 = complex(1, 0)
        else:
            z1_0 = complex(cos_eta, 0)
            z2_0 = complex(x, -y) / (2 * cos_eta)
            z2_0 = sin_eta * z2_0 / abs(z2_0) if abs(z2_0) > 1e-14 else complex(sin_eta, 0)

        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        phase = np.exp(1j * t)
        z1_arr = phase * z1_0
        z2_arr = phase * z2_0

        points = np.column_stack([
            np.real(z1_arr),
            np.imag(z1_arr),
            np.real(z2_arr),
            np.imag(z2_arr),
        ])
        return points

    @staticmethod
    def connection_1form():
        """
        The canonical connection 1-form on the Hopf bundle.

        A = Im( conj(z1)*dz1 + conj(z2)*dz2 )

        Returns
        -------
        dict with symbolic expression and description
        """
        return {
            'formula': 'A = Im(conj(z1)*dz1 + conj(z2)*dz2)',
            'description': (
                'The connection 1-form of the canonical U(1) connection '
                'on the Hopf bundle S^3 -> S^2. '
                'Restricts to d(theta) on each S^1 fiber.'
            ),
            'in_hopf_coords': 'A = d(xi1)*cos^2(eta) + d(xi2)*sin^2(eta)',
        }

    @staticmethod
    def curvature():
        """
        Curvature 2-form F = dA of the Hopf connection.

        Returns
        -------
        dict with symbolic expression and description
        """
        return {
            'formula': 'F = dA = sin(2*eta) * d(eta) ^ (d(xi1) - d(xi2))',
            'description': (
                'The curvature of the Hopf connection. '
                'Equals the pullback of the area form on S^2.'
            ),
            'total_flux': '4*pi',
            'chern_integral': '(1/(2*pi)) * integral(F) = 1',
        }

    @staticmethod
    def first_chern_number():
        """
        First Chern number of the Hopf bundle.

        c_1 = (1/(2*pi)) * integral_{S^2} F = 1

        Returns
        -------
        int : 1  (topological invariant)
        """
        return 1

    @staticmethod
    def linking_number(fiber1, fiber2):
        """
        Compute the linking number of two fibers of the Hopf fibration
        via the Gauss linking integral in R^3.

        For any two *distinct* Hopf fibers the linking number is 1.

        Parameters
        ----------
        fiber1 : (N, 4) array -- points on S^3 (Re z1, Im z1, Re z2, Im z2)
        fiber2 : (M, 4) array -- points on S^3

        Returns
        -------
        int : linking number (rounded to nearest integer)
        """
        def stereo(pts, pole):
            """Stereographic projection from a chosen pole in R^4.

            Projects from `pole` on S^3 to R^3.  For a point p on the
            unit S^3 the projection is:

                pi(p) = (p - (p . pole) * pole) / (1 - p . pole)

            keeping only 3 of the 4 orthogonal components.
            """
            # Build an ON basis for the hyperplane orthogonal to pole
            # using Gram-Schmidt on the standard basis vectors.
            basis = []
            for k in range(4):
                e = np.zeros(4)
                e[k] = 1.0
                v = e - np.dot(e, pole) * pole
                for b in basis:
                    v = v - np.dot(v, b) * b
                nv = np.linalg.norm(v)
                if nv > 1e-10:
                    basis.append(v / nv)
            basis = np.array(basis[:3])  # (3, 4)

            dot = pts @ pole  # (N,)
            denom = 1.0 - dot
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

            # Project pts into the 3D subspace
            coords = (pts - np.outer(dot, pole)) / denom[:, None]
            return coords @ basis.T  # (N, 3)

        # Choose a projection pole that is far from both fibers.
        # Try a few candidates and pick the one maximising the minimum
        # distance to all fiber points.
        candidates = np.array([
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
        ], dtype=float)
        # normalise candidates
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        all_pts = np.vstack([fiber1, fiber2])  # (N+M, 4)
        best_pole = None
        best_min_dist = -1
        for c in candidates:
            dots = all_pts @ c
            min_dist = 1.0 - np.max(dots)  # minimum chord-ish distance to pole
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pole = c

        c1 = stereo(fiber1, best_pole)
        c2 = stereo(fiber2, best_pole)

        N = len(c1)

        c1_closed = np.vstack([c1, c1[0:1]])
        c2_closed = np.vstack([c2, c2[0:1]])

        dc1 = np.diff(c1_closed, axis=0)  # (N, 3)
        dc2 = np.diff(c2_closed, axis=0)  # (M, 3)

        # Gauss linking integral:
        # Lk = 1/(4*pi) * sum_{i,j} (r_ij / |r_ij|^3) . (dc1_i x dc2_j)
        link = 0.0
        for i in range(N):
            r = c1[i] - c2          # (M, 3)
            r_norm = np.linalg.norm(r, axis=1, keepdims=True)
            r_norm = np.maximum(r_norm, 1e-14)
            r_hat = r / r_norm**3

            cross = np.cross(dc1[i], dc2)  # (M, 3)
            link += np.sum(r_hat * cross)

        link /= (4 * np.pi)
        return int(np.round(link))
