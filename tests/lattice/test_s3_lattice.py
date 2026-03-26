"""
Tests for the S3 lattice (600-cell) discretization.

Verifies:
    - Correct vertex/edge/face/cell counts
    - Euler characteristic chi(S^3) = 0
    - Regularity (all vertices have same valence)
    - Geometric properties (lattice spacing, radius)
    - Topology of the discretized S^3
"""

import pytest
import numpy as np
from yang_mills_s3.lattice.s3_lattice import S3Lattice


class TestVertexConstruction:
    """600-cell vertex construction on S^3."""

    def test_vertex_count_is_120(self):
        """The 600-cell has exactly 120 vertices."""
        lat = S3Lattice(R=1.0)
        assert lat.vertex_count() == 120

    def test_all_vertices_on_sphere(self):
        """All vertices lie on S^3 of radius R."""
        R = 2.5
        lat = S3Lattice(R=R)
        verts = lat.vertices
        norms = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(norms, R, atol=1e-10,
            err_msg="All vertices should lie on S^3(R)")

    def test_vertices_are_4d(self):
        """Vertices are 4-dimensional (embedded in R^4)."""
        lat = S3Lattice(R=1.0)
        verts = lat.vertices
        assert verts.shape == (120, 4)

    def test_vertex_scaling_with_radius(self):
        """Vertices scale linearly with R."""
        lat1 = S3Lattice(R=1.0)
        lat2 = S3Lattice(R=3.0)
        # Vertices should scale by factor 3
        # Sort to ensure consistent ordering
        v1_sorted = np.sort(lat1.vertices.ravel())
        v2_sorted = np.sort(lat2.vertices.ravel())
        np.testing.assert_allclose(v2_sorted, 3.0 * v1_sorted, atol=1e-10)


class TestCombinatorics:
    """Combinatorial properties of the 600-cell."""

    @pytest.fixture
    def lattice(self):
        return S3Lattice(R=1.0)

    def test_edge_count_is_720(self, lattice):
        """The 600-cell has exactly 720 edges."""
        assert lattice.edge_count() == 720

    def test_face_count_is_1200(self, lattice):
        """The 600-cell has exactly 1200 triangular faces."""
        assert lattice.face_count() == 1200

    def test_cell_count_is_600(self, lattice):
        """The 600-cell has exactly 600 tetrahedral cells."""
        assert lattice.cell_count() == 600

    def test_euler_characteristic_is_zero(self, lattice):
        """
        Euler characteristic: chi = V - E + F - C = 0.

        For odd-dimensional manifolds, chi = 0.
        120 - 720 + 1200 - 600 = 0. Verified.
        """
        V = lattice.vertex_count()
        E = lattice.edge_count()
        F = lattice.face_count()
        C = lattice.cell_count()
        chi = V - E + F - C
        assert chi == 0, f"Euler characteristic should be 0, got {chi}"

    def test_edges_are_pairs(self, lattice):
        """Each edge is a pair (i, j) with i < j."""
        for (i, j) in lattice.edges():
            assert i < j, f"Edge ({i}, {j}) should have i < j"

    def test_faces_are_triples(self, lattice):
        """Each face is a triple (i, j, k) with i < j < k."""
        for (i, j, k) in lattice.faces():
            assert i < j < k, f"Face ({i}, {j}, {k}) should be sorted"

    def test_cells_are_quadruples(self, lattice):
        """Each cell is a quadruple (i, j, k, l) with i < j < k < l."""
        for (i, j, k, l) in lattice.cells():
            assert i < j < k < l


class TestRegularity:
    """Regularity properties of the 600-cell."""

    @pytest.fixture
    def lattice(self):
        return S3Lattice(R=1.0)

    def test_is_regular(self, lattice):
        """All vertices have the same valence."""
        assert lattice.is_regular()

    def test_valence_is_12(self, lattice):
        """Each vertex in the 600-cell has exactly 12 neighbors."""
        val = lattice.valence()
        for v, degree in val.items():
            assert degree == 12, f"Vertex {v} has valence {degree}, expected 12"

    def test_each_edge_in_multiple_faces(self, lattice):
        """
        Each edge should appear in multiple triangular faces.
        In the 600-cell, each edge is shared by exactly 5 triangles.
        """
        edge_face_count = {}
        for (i, j) in lattice.edges():
            edge_face_count[(i, j)] = 0

        for (a, b, c) in lattice.faces():
            for edge in [(a, b), (a, c), (b, c)]:
                if edge in edge_face_count:
                    edge_face_count[edge] += 1

        vals = set(edge_face_count.values())
        # All edges should have the same number of adjacent faces
        assert len(vals) == 1, f"Expected uniform face adjacency, got {vals}"
        # In the 600-cell, each edge is in 5 triangles
        assert vals == {5}, f"Each edge should be in 5 faces, got {vals}"


class TestGeometry:
    """Geometric properties of the lattice."""

    def test_lattice_spacing_unit_sphere(self):
        """
        Lattice spacing on unit S^3.

        For the 600-cell, nearest-neighbor distance = 1/phi where
        phi = (1+sqrt(5))/2 is the golden ratio.
        """
        lat = S3Lattice(R=1.0)
        spacing = lat.lattice_spacing()
        expected = 1.0 / ((1 + np.sqrt(5)) / 2)  # 1/phi ≈ 0.618
        assert abs(spacing - expected) < 1e-6, \
            f"Spacing {spacing:.6f} should be 1/phi = {expected:.6f}"

    def test_lattice_spacing_scales_with_R(self):
        """Lattice spacing scales linearly with R."""
        lat1 = S3Lattice(R=1.0)
        lat2 = S3Lattice(R=2.0)
        ratio = lat2.lattice_spacing() / lat1.lattice_spacing()
        assert abs(ratio - 2.0) < 1e-10, f"Spacing ratio should be 2.0, got {ratio}"

    def test_plaquettes_are_triangular(self):
        """Plaquettes on the 600-cell are triangular (3 vertices each)."""
        lat = S3Lattice(R=1.0)
        for plaq in lat.plaquettes():
            assert len(plaq) == 3, f"Plaquette has {len(plaq)} vertices, expected 3"

    def test_plaquette_count_equals_face_count(self):
        """Number of plaquettes equals number of faces (1200)."""
        lat = S3Lattice(R=1.0)
        assert len(lat.plaquettes()) == lat.face_count()


class TestTopology:
    """Full topology verification."""

    def test_verify_topology_passes(self):
        """The verify_topology method should report all checks passing."""
        lat = S3Lattice(R=1.0)
        result = lat.verify_topology()
        assert result['all_checks_pass'], f"Topology check failed: {result}"

    def test_verify_topology_euler(self):
        """Euler characteristic should be 0."""
        lat = S3Lattice(R=1.0)
        result = lat.verify_topology()
        assert result['euler_characteristic'] == 0

    def test_verify_topology_counts(self):
        """V=120, E=720, F=1200, C=600."""
        lat = S3Lattice(R=1.0)
        result = lat.verify_topology()
        assert result['V'] == 120
        assert result['E'] == 720
        assert result['F'] == 1200
        assert result['C'] == 600

    def test_different_radii_same_topology(self):
        """Topology should be independent of R."""
        for R in [0.5, 1.0, 2.0, 10.0]:
            lat = S3Lattice(R=R)
            result = lat.verify_topology()
            assert result['all_checks_pass'], f"Failed for R={R}: {result}"


class TestGreatCircles:
    """Great circle detection on the 600-cell."""

    def test_finds_at_least_one_great_circle(self):
        """Should find at least one great circle."""
        lat = S3Lattice(R=1.0)
        circles = lat.great_circles(max_circles=3)
        assert len(circles) >= 1, "Should find at least one great circle"

    def test_great_circle_is_closed_path(self):
        """
        Each great circle should be a sequence of connected vertices.
        Consecutive vertices in the path should be connected by edges.
        """
        lat = S3Lattice(R=1.0)
        circles = lat.great_circles(max_circles=1)
        if not circles:
            pytest.skip("No great circles found")

        edge_set = set()
        for (i, j) in lat.edges():
            edge_set.add((i, j))
            edge_set.add((j, i))

        path = circles[0]
        for step in range(len(path) - 1):
            assert (path[step], path[step + 1]) in edge_set, \
                f"Vertices {path[step]} and {path[step+1]} not connected"
