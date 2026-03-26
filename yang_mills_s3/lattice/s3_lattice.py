"""
S3 Lattice -- Discretization of S^3 using the 600-cell regular polytope.

The 600-cell is the finest regular polytope in 4D:
    - 120 vertices on S^3
    - 720 edges
    - 1200 triangular faces
    - 600 tetrahedral cells
    - Symmetry group of order 14400

This gives a natural lattice for S^3 with high symmetry, ideal for
lattice gauge theory regularization.

The Euler characteristic chi = V - E + F - C = 120 - 720 + 1200 - 600 = 0,
consistent with chi(S^3) = 0 (odd-dimensional sphere).

THEOREM: The 600-cell provides a regular discretization of S^3 with
all vertices equivalent under the symmetry group. This is the finest
regular polytope available in 4D.
"""

import numpy as np
from itertools import combinations


# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # = phi - 1


class S3Lattice:
    """
    Discretization of S^3 using the 600-cell regular polytope.

    The 600-cell has:
        - 120 vertices, all at equal distance from the origin on S^3
        - 720 edges connecting nearest neighbors
        - 1200 triangular faces
        - 600 tetrahedral cells

    All vertices have the same valence (12 nearest neighbors each),
    making this an ideal lattice for gauge theory.
    """

    def __init__(self, R=1.0):
        """
        Construct the 600-cell lattice on S^3 of radius R.

        Parameters
        ----------
        R : float
            Radius of S^3. Default 1.0.
        """
        self.R = R
        self._vertices = None
        self._edges = None
        self._faces = None
        self._cells = None
        self._build_600_cell()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_600_cell(self):
        """
        Construct the 120 vertices of the 600-cell on S^3 of radius R.

        The vertices (on the unit S^3) are:
            Group 1: 8 vertices -- all permutations of (+-1, 0, 0, 0)
            Group 2: 16 vertices -- (+-1/2, +-1/2, +-1/2, +-1/2)
            Group 3: 96 vertices -- even permutations of
                     (0, +-1/2, +-phi/2, +-1/(2*phi))
                     where phi = golden ratio = (1+sqrt(5))/2

        Total: 8 + 16 + 96 = 120 vertices.
        """
        vertices = []

        # Group 1: permutations of (+-1, 0, 0, 0) -- 8 vertices
        for i in range(4):
            for sign in [1, -1]:
                v = [0.0, 0.0, 0.0, 0.0]
                v[i] = sign
                vertices.append(v)

        # Group 2: (+-1/2, +-1/2, +-1/2, +-1/2) -- 16 vertices
        for s0 in [0.5, -0.5]:
            for s1 in [0.5, -0.5]:
                for s2 in [0.5, -0.5]:
                    for s3 in [0.5, -0.5]:
                        vertices.append([s0, s1, s2, s3])

        # Group 3: even permutations of (0, +-1/2, +-phi/2, +-1/(2*phi))
        # = 96 vertices
        base_values = [0.0, 0.5, PHI / 2, INV_PHI / 2]

        # The even permutations of 4 elements: there are 12 even permutations
        even_perms = _even_permutations_of_4()

        for perm in even_perms:
            # Apply the permutation to base_values
            # Then apply all sign combinations to the non-zero entries
            permuted = [base_values[perm[j]] for j in range(4)]

            # Find which positions have non-zero values
            nonzero_positions = [j for j in range(4) if abs(permuted[j]) > 1e-12]

            # Apply all sign combinations to nonzero entries
            n_nonzero = len(nonzero_positions)
            for sign_combo in range(2**n_nonzero):
                v = list(permuted)
                for k, pos in enumerate(nonzero_positions):
                    if sign_combo & (1 << k):
                        v[pos] = -v[pos]
                vertices.append(v)

        # Convert to numpy and normalize to unit sphere
        raw = np.array(vertices)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Filter out any duplicates and keep unique vertices on unit S^3
        norms = np.where(norms < 1e-12, 1.0, norms)
        normalized = raw / norms

        # Remove duplicates (within tolerance)
        unique = _unique_rows(normalized, tol=1e-8)

        # Scale to radius R
        self._vertices = unique * self.R

        # Build edges from nearest-neighbor distances
        self._build_edges()
        # Build faces from edge triangles
        self._build_faces()
        # Build cells from face tetrahedra
        self._build_cells()

    def _build_edges(self):
        """
        Build edges by connecting nearest neighbors.

        On the unit 600-cell, the nearest-neighbor distance is 1/phi
        (the edge length). Two vertices are connected iff their
        inner product is >= cos(pi/5) = phi/2 (approx 0.809).

        More precisely, the dot product between adjacent vertices on
        the unit 600-cell is phi/2.
        """
        n = len(self._vertices)
        # Compute pairwise dot products (on unit sphere)
        unit_verts = self._vertices / self.R
        dots = unit_verts @ unit_verts.T

        # Nearest-neighbor dot product for 600-cell on unit S^3
        # Edge length = 1/phi on unit S^3, so dot product = 1 - (1/phi)^2/2
        # Actually: |v1 - v2|^2 = 2 - 2*dot => dot = 1 - |v1-v2|^2/2
        # Edge length of 600-cell on unit sphere: 2*sin(pi/10) = 1/phi
        # Wait -- let me compute directly.
        # For unit 600-cell, nearest neighbor distance squared = (1/phi)^2
        # So dot = 1 - (1/phi)^2/2 = 1 - (phi-1)^2/2 = 1 - (3-sqrt(5))/2/2
        # = 1 - (3-2.236)/4 = 1 - 0.191 = 0.809 = phi/2

        # The edge threshold: dots above this threshold (minus epsilon) are edges
        # But we should just find the nearest-neighbor distance and use that
        if n < 2:
            self._edges = []
            return

        # Find the maximum non-self dot product (nearest neighbor)
        np.fill_diagonal(dots, -2.0)  # Exclude self
        max_dot = np.max(dots)

        # Threshold: anything within 1% of max_dot
        threshold = max_dot - 0.01 * (1.0 - max_dot)

        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if dots[i, j] > threshold:
                    edges.add((i, j))

        self._edges = sorted(edges)

    def _build_faces(self):
        """
        Build triangular faces of the 600-cell.

        A face exists when three vertices are mutually connected by edges.
        The 600-cell has 1200 triangular faces.
        """
        # Build adjacency structure for efficiency
        n = len(self._vertices)
        adj = {i: set() for i in range(n)}
        for (i, j) in self._edges:
            adj[i].add(j)
            adj[j].add(i)

        faces = set()
        for (i, j) in self._edges:
            # Find common neighbors
            common = adj[i] & adj[j]
            for k in common:
                face = tuple(sorted([i, j, k]))
                faces.add(face)

        self._faces = sorted(faces)

    def _build_cells(self):
        """
        Build tetrahedral cells of the 600-cell.

        A cell exists when four vertices are mutually connected by edges.
        The 600-cell has 600 tetrahedral cells.
        """
        # Build adjacency structure
        n = len(self._vertices)
        adj = {i: set() for i in range(n)}
        for (i, j) in self._edges:
            adj[i].add(j)
            adj[j].add(i)

        cells = set()
        for (i, j, k) in self._faces:
            # Find vertices adjacent to all three
            common = adj[i] & adj[j] & adj[k]
            for l in common:
                cell = tuple(sorted([i, j, k, l]))
                cells.add(cell)

        self._cells = sorted(cells)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def vertices(self):
        """Return (N, 4) array of vertex coordinates on S^3(R)."""
        return self._vertices.copy()

    def vertex_count(self):
        """Number of vertices. Should be 120 for the 600-cell."""
        return len(self._vertices)

    def edge_count(self):
        """Number of edges. Should be 720 for the 600-cell."""
        return len(self._edges)

    def face_count(self):
        """Number of triangular faces. Should be 1200 for the 600-cell."""
        return len(self._faces)

    def cell_count(self):
        """Number of tetrahedral cells. Should be 600 for the 600-cell."""
        return len(self._cells)

    def edges(self):
        """Return list of (i, j) edge pairs."""
        return list(self._edges)

    def faces(self):
        """Return list of (i, j, k) face triples."""
        return list(self._faces)

    def cells(self):
        """Return list of (i, j, k, l) cell quadruples."""
        return list(self._cells)

    # ------------------------------------------------------------------
    # Geometric properties
    # ------------------------------------------------------------------
    def lattice_spacing(self):
        """
        Average edge length (lattice spacing) on S^3(R).

        For the 600-cell on unit S^3, the nearest-neighbor geodesic
        distance is related to the chordal distance.

        Returns
        -------
        float : average chordal distance between connected vertices
        """
        if not self._edges:
            return 0.0

        total = 0.0
        for (i, j) in self._edges:
            diff = self._vertices[i] - self._vertices[j]
            total += np.linalg.norm(diff)

        return total / len(self._edges)

    def valence(self):
        """
        Vertex valence (number of edges per vertex).

        For the 600-cell, every vertex has valence 12.

        Returns
        -------
        dict : {vertex_index: valence}
        """
        val = {i: 0 for i in range(len(self._vertices))}
        for (i, j) in self._edges:
            val[i] += 1
            val[j] += 1
        return val

    def is_regular(self):
        """
        Check if the lattice is regular (all vertices have same valence).

        Returns
        -------
        bool
        """
        val = self.valence()
        if not val:
            return True
        vals = set(val.values())
        return len(vals) == 1

    # ------------------------------------------------------------------
    # Topological verification
    # ------------------------------------------------------------------
    def verify_topology(self):
        """
        Verify the discretized S^3 has the correct topology.

        Checks:
            1. Euler characteristic chi = V - E + F - C = 0
               (correct for odd-dimensional manifolds)
            2. Exact counts: V=120, E=720, F=1200, C=600
            3. All vertices have the same valence (regularity)

        Returns
        -------
        dict with:
            'euler_characteristic': int (should be 0)
            'counts_correct': bool
            'is_regular': bool
            'all_checks_pass': bool
        """
        V = self.vertex_count()
        E = self.edge_count()
        F = self.face_count()
        C = self.cell_count()

        euler = V - E + F - C
        counts_ok = (V == 120 and E == 720 and F == 1200 and C == 600)
        regular = self.is_regular()

        return {
            'euler_characteristic': euler,
            'counts_correct': counts_ok,
            'is_regular': regular,
            'V': V, 'E': E, 'F': F, 'C': C,
            'all_checks_pass': (euler == 0 and counts_ok and regular),
        }

    def plaquettes(self):
        """
        Return list of minimal plaquettes (closed loops).

        On the 600-cell, the minimal plaquettes are the triangular faces
        (3-edge loops). Each face (i, j, k) defines a plaquette
        with ordered links: (i->j, j->k, k->i).

        Returns
        -------
        list of lists: each element is [i, j, k] defining a triangular plaquette
        """
        return [list(f) for f in self._faces]

    def great_circles(self, max_circles=10):
        """
        Find great circles through the 600-cell vertices.

        A great circle on S^3 passes through vertices that lie on a
        2D plane through the origin. We find paths of connected vertices
        that form (approximate) great circles.

        Returns
        -------
        list of lists: each element is a list of vertex indices forming
                       a closed path along a great circle
        """
        circles = []
        used_starts = set()
        n = len(self._vertices)

        # Build adjacency
        adj = {i: set() for i in range(n)}
        for (i, j) in self._edges:
            adj[i].add(j)
            adj[j].add(i)

        unit_verts = self._vertices / self.R

        for start in range(n):
            if start in used_starts or len(circles) >= max_circles:
                break

            # Try each neighbor as second vertex
            for second in sorted(adj[start]):
                if second in used_starts:
                    continue

                # Direction from start to second
                path = [start, second]
                current = second
                prev = start

                # Follow the path: at each step, pick the neighbor
                # most aligned with the great circle direction
                for _ in range(20):  # max steps
                    # Great circle direction: project out radial component
                    direction = unit_verts[current] - unit_verts[prev]

                    best_next = None
                    best_dot = -2.0

                    for nb in adj[current]:
                        if nb == prev:
                            continue
                        fwd = unit_verts[nb] - unit_verts[current]
                        d = np.dot(fwd, direction)
                        if d > best_dot:
                            best_dot = d
                            best_next = nb

                    if best_next is None or best_next == start:
                        if best_next == start and len(path) > 3:
                            circles.append(path)
                            used_starts.update(path)
                        break

                    if best_next in path:
                        break

                    path.append(best_next)
                    prev = current
                    current = best_next

                if len(circles) >= max_circles:
                    break

        return circles


# ======================================================================
# Module-level helpers
# ======================================================================

def _even_permutations_of_4():
    """
    Return the 12 even permutations of (0, 1, 2, 3).

    An even permutation has an even number of transpositions.
    """
    all_perms = []
    indices = [0, 1, 2, 3]
    _generate_perms(indices, 0, all_perms)

    even = []
    for p in all_perms:
        if _parity(p) == 0:
            even.append(p)
    return even


def _generate_perms(arr, start, result):
    """Generate all permutations of arr by swapping."""
    if start == len(arr):
        result.append(list(arr))
        return
    for i in range(start, len(arr)):
        arr[start], arr[i] = arr[i], arr[start]
        _generate_perms(arr, start + 1, result)
        arr[start], arr[i] = arr[i], arr[start]


def _parity(perm):
    """Return 0 for even permutation, 1 for odd."""
    n = len(perm)
    visited = [False] * n
    parity = 0
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len > 1:
                parity += cycle_len - 1
    return parity % 2


def _unique_rows(arr, tol=1e-8):
    """Remove duplicate rows from a 2D array within tolerance."""
    if len(arr) == 0:
        return arr

    unique = [arr[0]]
    for i in range(1, len(arr)):
        is_dup = False
        for u in unique:
            if np.linalg.norm(arr[i] - u) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(arr[i])

    return np.array(unique)
