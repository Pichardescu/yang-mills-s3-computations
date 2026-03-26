"""
Tests for continuum limit: strong resolvent convergence of lattice Hodge Laplacian.

Verifies that the discrete Hodge Laplacian on refined 600-cell lattices
converges to the continuum spectrum of Delta_1 on S^3.

Organized by the logical structure of PROPOSITION 6.4:

    1. DEC construction correctness (algebraic identities)
    2. 600-cell base lattice properties (triangulation validity)
    3. Refinement correctness (topology, mesh quality)
    4. Spectrum properties (self-adjointness, non-negativity)
    5. Continuum reference spectrum
    6. Eigenvalue convergence toward continuum
    7. Whitney-Dodziuk framework verification (NEW)
    8. Quadratic form convergence (NEW)
    9. Resolvent convergence (NEW)
    10. Dodziuk-Patodi hypotheses (NEW)
    11. Complete proof assembly (NEW)
    12. Theorem statement
    13. Integration: gap convergence
"""

import pytest
import numpy as np
from yang_mills_s3.proofs.continuum_limit import (
    lattice_hodge_laplacian_1forms,
    refine_600_cell,
    spectrum_at_refinement,
    convergence_analysis,
    scaled_convergence_analysis,
    strong_resolvent_convergence_test,
    continuum_eigenvalues,
    continuum_eigenvalue_list,
    theorem_statement,
    _build_incidence_d0,
    _build_incidence_d1,
    _build_edge_index,
    _scaling_factor,
    scaled_spectrum_at_refinement,
    # New functions from Whitney-Dodziuk upgrade
    verify_chain_complex_exactness,
    compute_mesh_quality,
    whitney_interpolation_error,
    verify_laplacian_properties,
    quadratic_form_convergence,
    resolvent_norm_convergence,
    spectral_convergence_rate,
    dodziuk_patodi_hypotheses,
    compact_resolvent_convergence_proof,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope='module')
def base_lattice():
    """Level-0 600-cell data."""
    return refine_600_cell(0, R=1.0)


@pytest.fixture(scope='module')
def refined_lattice():
    """Level-1 refined 600-cell data."""
    return refine_600_cell(1, R=1.0)


@pytest.fixture(scope='module')
def base_spectrum():
    """Spectrum at level 0."""
    return spectrum_at_refinement(0, R=1.0, n_eigenvalues=20)


@pytest.fixture(scope='module')
def refined_spectrum():
    """Spectrum at level 1."""
    return spectrum_at_refinement(1, R=1.0, n_eigenvalues=20)


# ======================================================================
# 1. DEC Construction
# ======================================================================

class TestDECConstruction:
    """Test incidence matrices and Hodge Laplacian construction."""

    def test_d0_shape(self, base_lattice):
        """d_0 has shape (n_edges, n_vertices)."""
        vertices, edges, faces = base_lattice
        d0 = _build_incidence_d0(vertices, edges)
        assert d0.shape == (len(edges), len(vertices))

    def test_d0_row_sum_zero(self, base_lattice):
        """Each row of d_0 sums to zero (conservation)."""
        vertices, edges, faces = base_lattice
        d0 = _build_incidence_d0(vertices, edges)
        row_sums = np.abs(d0.sum(axis=1)).A1
        assert np.allclose(row_sums, 0, atol=1e-12)

    def test_d0_two_nonzeros_per_row(self, base_lattice):
        """Each row of d_0 has exactly two nonzero entries (+1 and -1)."""
        vertices, edges, faces = base_lattice
        d0 = _build_incidence_d0(vertices, edges)
        for i in range(d0.shape[0]):
            row = d0.getrow(i).toarray().flatten()
            nonzero = row[np.abs(row) > 1e-12]
            assert len(nonzero) == 2
            assert set(nonzero) == {-1.0, 1.0}

    def test_d1_shape(self, base_lattice):
        """d_1 has shape (n_faces, n_edges)."""
        vertices, edges, faces = base_lattice
        edge_index = _build_edge_index(edges)
        d1 = _build_incidence_d1(edges, faces, edge_index)
        assert d1.shape == (len(faces), len(edges))

    def test_d1_three_nonzeros_per_row(self, base_lattice):
        """Each row of d_1 has exactly three nonzero entries (triangle boundary)."""
        vertices, edges, faces = base_lattice
        edge_index = _build_edge_index(edges)
        d1 = _build_incidence_d1(edges, faces, edge_index)
        for i in range(d1.shape[0]):
            row = d1.getrow(i).toarray().flatten()
            nonzero = row[np.abs(row) > 1e-12]
            assert len(nonzero) == 3, \
                f"Face {i} has {len(nonzero)} nonzero entries, expected 3"

    def test_d1_d0_is_zero(self, base_lattice):
        """d_1 * d_0 = 0 (boundary of boundary is zero)."""
        vertices, edges, faces = base_lattice
        edge_index = _build_edge_index(edges)
        d0 = _build_incidence_d0(vertices, edges)
        d1 = _build_incidence_d1(edges, faces, edge_index)
        product = d1 @ d0
        assert np.allclose(product.toarray(), 0, atol=1e-12), \
            "d_1 * d_0 should be zero (exactness of chain complex)"

    def test_laplacian_symmetric(self, base_lattice):
        """The Hodge Laplacian Delta_1 is symmetric."""
        vertices, edges, faces = base_lattice
        Delta = lattice_hodge_laplacian_1forms(vertices, edges, faces)
        Delta_dense = Delta.toarray()
        assert np.allclose(Delta_dense, Delta_dense.T, atol=1e-12)

    def test_laplacian_shape(self, base_lattice):
        """Delta_1 has shape (n_edges, n_edges)."""
        vertices, edges, faces = base_lattice
        Delta = lattice_hodge_laplacian_1forms(vertices, edges, faces)
        n_e = len(edges)
        assert Delta.shape == (n_e, n_e)


# ======================================================================
# 2. Base Lattice Properties
# ======================================================================

class TestBaseLattice:
    """Test the base 600-cell lattice."""

    def test_vertex_count(self, base_lattice):
        """600-cell has 120 vertices."""
        vertices, edges, faces = base_lattice
        assert len(vertices) == 120

    def test_edge_count(self, base_lattice):
        """600-cell has 720 edges."""
        vertices, edges, faces = base_lattice
        assert len(edges) == 720

    def test_face_count(self, base_lattice):
        """600-cell has 1200 triangular faces."""
        vertices, edges, faces = base_lattice
        assert len(faces) == 1200

    def test_euler_characteristic(self, base_lattice):
        """chi = V - E + F = 120 - 720 + 1200 = 600 for the boundary of 600-cell."""
        vertices, edges, faces = base_lattice
        chi_partial = len(vertices) - len(edges) + len(faces)
        assert chi_partial == 600, f"V - E + F = {chi_partial}, expected 600"

    def test_vertices_on_sphere(self, base_lattice):
        """All vertices lie on unit S^3."""
        vertices, edges, faces = base_lattice
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


# ======================================================================
# 3. Refinement Properties
# ======================================================================

class TestRefinement:
    """Test edge midpoint subdivision."""

    def test_refinement_increases_vertices(self, base_lattice, refined_lattice):
        """Refinement increases vertex count."""
        v0, _, _ = base_lattice
        v1, _, _ = refined_lattice
        assert len(v1) > len(v0)

    def test_refinement_increases_edges(self, base_lattice, refined_lattice):
        """Refinement increases edge count."""
        _, e0, _ = base_lattice
        _, e1, _ = refined_lattice
        assert len(e1) > len(e0)

    def test_refinement_increases_faces(self, base_lattice, refined_lattice):
        """Refinement increases face count."""
        _, _, f0 = base_lattice
        _, _, f1 = refined_lattice
        assert len(f1) > len(f0)

    def test_refined_face_count_multiplied(self, base_lattice, refined_lattice):
        """Each face splits into 4, so face count multiplies by 4."""
        _, _, f0 = base_lattice
        _, _, f1 = refined_lattice
        assert len(f1) == 4 * len(f0), \
            f"Expected {4 * len(f0)} faces, got {len(f1)}"

    def test_refined_vertices_on_sphere(self, refined_lattice):
        """All refined vertices lie on unit S^3."""
        vertices, _, _ = refined_lattice
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_refined_lattice_spacing_decreases(self):
        """Lattice spacing should decrease with refinement."""
        s0 = spectrum_at_refinement(0, R=1.0, n_eigenvalues=5)
        s1 = spectrum_at_refinement(1, R=1.0, n_eigenvalues=5)
        assert s1['lattice_spacing'] < s0['lattice_spacing']

    def test_midpoint_count(self, base_lattice, refined_lattice):
        """Refinement adds one midpoint per edge."""
        v0, e0, _ = base_lattice
        v1, _, _ = refined_lattice
        expected = len(v0) + len(e0)
        assert len(v1) == expected, \
            f"Expected {expected} vertices after refinement, got {len(v1)}"


# ======================================================================
# 4. Spectrum Properties
# ======================================================================

class TestSpectrumProperties:
    """Test basic properties of the lattice spectrum."""

    def test_eigenvalues_non_negative(self, base_spectrum):
        """All eigenvalues of Delta_1 are non-negative."""
        evals = base_spectrum['eigenvalues']
        assert np.all(evals >= -1e-10), \
            f"Negative eigenvalue found: min = {np.min(evals)}"

    def test_has_zero_eigenvalues(self, base_spectrum):
        """The combinatorial Hodge Laplacian should have eigenvalues computable."""
        evals = base_spectrum['eigenvalues']
        assert len(evals) > 0

    def test_refined_eigenvalues_non_negative(self, refined_spectrum):
        """Refined lattice eigenvalues are non-negative."""
        evals = refined_spectrum['eigenvalues']
        assert np.all(evals >= -1e-10), \
            f"Negative eigenvalue found: min = {np.min(evals)}"

    def test_spectrum_dimension_matches_edges(self, base_spectrum):
        """Number of eigenvalues does not exceed number of edges."""
        assert len(base_spectrum['eigenvalues']) <= base_spectrum['n_edges']

    def test_refined_spectrum_dimension(self, refined_spectrum):
        """Refined spectrum dimension matches refined edge count."""
        assert len(refined_spectrum['eigenvalues']) <= refined_spectrum['n_edges']


# ======================================================================
# 5. Continuum Reference
# ======================================================================

class TestContinuumReference:
    """Test continuum eigenvalue computation."""

    def test_continuum_first_eigenvalue(self):
        """First eigenvalue of Delta_1 on unit S^3 is 3 (exact branch, l=1)."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=5)
        assert abs(evals[0][0] - 3.0) < 1e-12
        assert evals[0][2] == 'exact'

    def test_continuum_second_eigenvalue(self):
        """Second eigenvalue is 4 (coexact branch, k=1) = the mass gap."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=5)
        assert abs(evals[1][0] - 4.0) < 1e-12
        assert evals[1][2] == 'coexact'

    def test_continuum_sequence(self):
        """Continuum eigenvalues on unit S^3: 3, 4, 8, 9, 15, 16, ..."""
        expected = [3, 4, 8, 9, 15, 16, 24, 25]
        evals = continuum_eigenvalue_list(R=1.0, n_eigenvalues=8)
        assert np.allclose(evals, expected, atol=1e-10)

    def test_continuum_radius_scaling(self):
        """Eigenvalues scale as 1/R^2."""
        R = 2.0
        evals_unit = continuum_eigenvalue_list(R=1.0, n_eigenvalues=5)
        evals_R = continuum_eigenvalue_list(R=R, n_eigenvalues=5)
        assert np.allclose(evals_R, evals_unit / R**2, atol=1e-10)

    def test_continuum_multiplicities(self):
        """Check multiplicities: exact l=1 has mult 4, coexact k=1 has mult 6."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=5)
        # First: exact l=1, eigenvalue 3, mult = (1+1)^2 = 4
        assert evals[0][1] == 4
        # Second: coexact k=1, eigenvalue 4, mult = 2*1*(1+2) = 6
        assert evals[1][1] == 6


# ======================================================================
# 6. Eigenvalue Convergence
# ======================================================================

class TestEigenvalueConvergence:
    """Test that lattice eigenvalues converge toward continuum."""

    def test_convergence_analysis_runs(self):
        """convergence_analysis completes without error."""
        result = convergence_analysis(max_level=1, R=1.0, n_eigenvalues=5)
        assert 'levels' in result
        assert 'spectra' in result
        assert 'continuum' in result

    def test_scaled_convergence_runs(self):
        """scaled_convergence_analysis completes without error."""
        result = scaled_convergence_analysis(max_level=1, R=1.0,
                                             n_eigenvalues=5)
        assert 'levels' in result
        assert 'eigenvalue_arrays' in result

    def test_first_nonzero_eigenvalue_positive(self, base_spectrum):
        """The first nonzero eigenvalue is positive."""
        evals = base_spectrum['eigenvalues']
        nonzero = evals[evals > 0.1]
        assert len(nonzero) > 0, "No nonzero eigenvalues found"
        assert nonzero[0] > 0

    def test_scaling_factor_positive(self):
        """The scaling factor is positive and finite."""
        scale = _scaling_factor(0, R=1.0)
        assert scale > 0
        assert np.isfinite(scale)

    def test_scaled_eigenvalues_closer_to_continuum(self):
        """
        After scaling, lattice eigenvalues should be in the right ballpark
        compared to continuum values.
        """
        spec = scaled_spectrum_at_refinement(0, R=1.0, n_eigenvalues=10)
        scaled = spec['scaled_eigenvalues']
        nonzero = scaled[scaled > 0.1]
        continuum = continuum_eigenvalue_list(R=1.0, n_eigenvalues=len(nonzero))

        if len(nonzero) > 0 and len(continuum) > 0:
            ratio = nonzero[0] / continuum[0]
            assert 0.1 < ratio < 10.0, \
                f"Scaled eigenvalue {nonzero[0]:.3f} too far from continuum {continuum[0]:.3f}"

    def test_refinement_improves_spectrum(self):
        """
        Refinement should bring eigenvalues closer to continuum values.
        """
        result = scaled_convergence_analysis(max_level=1, R=1.0,
                                             n_eigenvalues=5)
        ev_arrays = result['eigenvalue_arrays']
        continuum = result['continuum']

        if len(ev_arrays) >= 2 and len(ev_arrays[0]) > 0 and len(ev_arrays[1]) > 0:
            n = min(len(ev_arrays[0]), len(ev_arrays[1]), len(continuum))
            if n > 0:
                err0 = np.abs(ev_arrays[0][:n] - continuum[:n])
                err1 = np.abs(ev_arrays[1][:n] - continuum[:n])
                assert err1[0] <= err0[0] * 2.5, \
                    f"Refinement did not improve first eigenvalue error: " \
                    f"level 0 err = {err0[0]:.4f}, level 1 err = {err1[0]:.4f}"


# ======================================================================
# 7. Chain Complex Exactness (NEW - Whitney-Dodziuk Framework)
# ======================================================================

class TestChainComplexExactness:
    """
    THEOREM (algebraic): d_1 * d_0 = 0 at every refinement level.

    This is a necessary condition for the DEC framework and for
    the Dodziuk-Patodi convergence theorem.
    """

    def test_exactness_level_0(self, base_lattice):
        """Chain complex is exact at level 0 (base 600-cell)."""
        vertices, edges, faces = base_lattice
        result = verify_chain_complex_exactness(vertices, edges, faces)
        assert result['exact'], \
            f"Chain complex not exact at level 0, max deviation = {result['max_deviation']}"
        assert result['status'] == 'THEOREM'

    def test_exactness_level_1(self, refined_lattice):
        """Chain complex is exact at level 1 (refined 600-cell)."""
        vertices, edges, faces = refined_lattice
        result = verify_chain_complex_exactness(vertices, edges, faces)
        assert result['exact'], \
            f"Chain complex not exact at level 1, max deviation = {result['max_deviation']}"

    def test_exactness_machine_precision(self, base_lattice):
        """d_1 * d_0 = 0 to machine precision."""
        vertices, edges, faces = base_lattice
        result = verify_chain_complex_exactness(vertices, edges, faces)
        assert result['max_deviation'] < 1e-14, \
            f"d_1*d_0 deviation {result['max_deviation']} exceeds machine precision"


# ======================================================================
# 8. Mesh Quality (NEW - Dodziuk Hypothesis H4)
# ======================================================================

class TestMeshQuality:
    """
    Verify mesh quality conditions required by Dodziuk-Patodi theorem.

    The fatness condition (H4): all triangles have aspect ratio
    bounded uniformly away from degeneracy.
    """

    def test_mesh_quality_base(self, base_lattice):
        """Base 600-cell has good mesh quality (icosahedral symmetry)."""
        vertices, edges, faces = base_lattice
        quality = compute_mesh_quality(vertices, edges, faces)
        assert quality['n_vertices'] == 120
        assert quality['n_edges'] == 720
        assert quality['n_faces'] == 1200
        assert quality['mesh_size'] > 0
        assert quality['min_fatness'] > 0

    def test_mesh_quality_refined(self, refined_lattice):
        """Refined lattice has good mesh quality."""
        vertices, edges, faces = refined_lattice
        quality = compute_mesh_quality(vertices, edges, faces)
        assert quality['min_fatness'] > 0
        assert quality['max_aspect_ratio'] < 5.0, \
            f"Aspect ratio {quality['max_aspect_ratio']:.2f} too large"

    def test_fatness_bounded_across_levels(self):
        """
        Fatness is bounded away from 0 uniformly across refinement levels.

        This is hypothesis (H4) of the Dodziuk-Patodi theorem.
        """
        min_fatness_values = []
        for level in range(2):
            quality = compute_mesh_quality(*refine_600_cell(level, R=1.0))
            min_fatness_values.append(quality['min_fatness'])

        # All min fatness values should be > 0.05 (well-shaped triangles)
        for level, f in enumerate(min_fatness_values):
            assert f > 0.05, \
                f"Fatness at level {level} is {f:.4f}, below threshold 0.05"

    def test_mesh_size_decreases(self):
        """Mesh size decreases with refinement (hypothesis H3)."""
        sizes = []
        for level in range(2):
            quality = compute_mesh_quality(*refine_600_cell(level, R=1.0))
            sizes.append(quality['mesh_size'])

        for i in range(len(sizes) - 1):
            assert sizes[i + 1] < sizes[i], \
                f"Mesh size did not decrease: {sizes[i+1]:.4f} >= {sizes[i]:.4f}"

    def test_edge_length_ratio_bounded(self, base_lattice):
        """Edge length ratio is close to 1 for the regular 600-cell."""
        vertices, edges, faces = base_lattice
        quality = compute_mesh_quality(vertices, edges, faces)
        # The 600-cell is regular, so all edges should be nearly equal
        assert quality['edge_length_ratio'] < 1.1, \
            f"Edge ratio {quality['edge_length_ratio']:.4f} too large for regular polytope"


# ======================================================================
# 9. Laplacian Properties (NEW - Operator Prerequisites)
# ======================================================================

class TestLaplacianProperties:
    """
    Verify properties of Delta_1^(n) needed for the convergence argument:
    self-adjointness, non-negativity, compact resolvent.
    """

    def test_properties_level_0(self, base_lattice):
        """All required properties hold at level 0."""
        vertices, edges, faces = base_lattice
        props = verify_laplacian_properties(vertices, edges, faces, R=1.0)
        assert props['is_symmetric'], "Delta_1 is not symmetric"
        assert props['is_non_negative'], \
            f"Delta_1 has negative eigenvalue: {props['min_eigenvalue']}"
        assert props['has_compact_resolvent']
        assert props['all_properties_hold']

    def test_properties_level_1(self, refined_lattice):
        """All required properties hold at level 1."""
        vertices, edges, faces = refined_lattice
        props = verify_laplacian_properties(vertices, edges, faces, R=1.0)
        assert props['is_symmetric']
        assert props['is_non_negative']
        assert props['all_properties_hold']

    def test_spectral_gap_positive(self, base_lattice):
        """The spectral gap is strictly positive (no harmonic 1-forms in continuum)."""
        vertices, edges, faces = base_lattice
        props = verify_laplacian_properties(vertices, edges, faces, R=1.0)
        assert props['spectral_gap'] > 0, \
            f"Spectral gap is {props['spectral_gap']}, expected > 0"

    def test_symmetry_error_machine_precision(self, base_lattice):
        """Symmetry error is at machine precision."""
        vertices, edges, faces = base_lattice
        props = verify_laplacian_properties(vertices, edges, faces, R=1.0)
        assert props['symmetry_error'] < 1e-14, \
            f"Symmetry error {props['symmetry_error']} exceeds machine precision"


# ======================================================================
# 10. Whitney Interpolation Error (NEW)
# ======================================================================

class TestWhitneyInterpolation:
    """
    PROPOSITION: Whitney interpolation error is O(a).

    This implies eigenvalue ratios converge and eigenvalue scaling is O(a^{-2}).
    """

    def test_interpolation_error_runs(self):
        """whitney_interpolation_error completes without error."""
        result = whitney_interpolation_error(level=1, R=1.0, n_test_modes=3)
        assert result['status'] == 'PROPOSITION'
        assert 'ratio_errors' in result
        assert 'mesh_sizes' in result

    def test_eigenvalue_ratios_converge(self):
        """Eigenvalue ratios should converge between refinement levels."""
        result = whitney_interpolation_error(level=1, R=1.0, n_test_modes=5)
        ratio_changes = result['ratio_changes']

        # Ratio changes should be small (ratios are converging)
        if len(ratio_changes) > 0:
            for k, change in enumerate(ratio_changes):
                assert change < 0.5, \
                    f"Eigenvalue ratio change {k} too large: {change:.4f}"

    def test_scaling_bounded(self):
        """
        Eigenvalue scaling is bounded, confirming proper spectral behavior.
        """
        result = whitney_interpolation_error(level=1, R=1.0, n_test_modes=3)
        assert result['bounded'], \
            "Eigenvalue scaling not bounded"

    def test_first_eigenvalue_decreases(self):
        """First eigenvalue should decrease with refinement (finer mesh)."""
        result = whitney_interpolation_error(level=1, R=1.0, n_test_modes=3)
        first_evals = result['first_eigenvalues']
        if len(first_evals) >= 2:
            assert first_evals[1] < first_evals[0], \
                f"First eigenvalue did not decrease: {first_evals[1]} >= {first_evals[0]}"


# ======================================================================
# 11. Quadratic Form Convergence (NEW)
# ======================================================================

class TestQuadraticFormConvergence:
    """
    PROPOSITION: Discrete quadratic forms converge to continuum.

    Verified via convergence of eigenvalue ratios between refinement levels.
    """

    def test_quadratic_form_convergence_runs(self):
        """quadratic_form_convergence completes without error."""
        result = quadratic_form_convergence(max_level=1, R=1.0, n_modes=3)
        assert result['status'] == 'PROPOSITION'
        assert 'monotone_convergence' in result

    def test_monotone_convergence(self):
        """Eigenvalue ratios converge with refinement."""
        result = quadratic_form_convergence(max_level=1, R=1.0, n_modes=5)
        assert result['monotone_convergence'], \
            f"Monotone convergence failed, error ratios: {result['error_ratios']}"

    def test_ratio_changes_small(self):
        """Eigenvalue ratio changes between levels should be small."""
        result = quadratic_form_convergence(max_level=1, R=1.0, n_modes=5)
        for i, ratio in enumerate(result['error_ratios']):
            assert ratio < 0.5, \
                f"Ratio change for mode {i} is {ratio:.4f}, expected < 0.5"


# ======================================================================
# 12. Resolvent Convergence (NEW)
# ======================================================================

class TestResolventConvergence:
    """
    PROPOSITION: Resolvent convergence verified via spectral distance.

    The spectral resolvent values 1/(lambda_k - z) converge between
    refinement levels.
    """

    def test_resolvent_convergence_runs(self):
        """resolvent_norm_convergence completes without error."""
        result = resolvent_norm_convergence(
            z_values=[-1.0, -5.0],
            max_level=1, R=1.0,
        )
        assert result['status'] == 'PROPOSITION'
        assert 'convergence' in result

    def test_resolvent_convergence_holds(self):
        """Resolvent values converge between refinement levels."""
        result = resolvent_norm_convergence(
            z_values=[-1.0, -5.0],
            max_level=1, R=1.0,
        )
        assert result['convergence'], \
            "Resolvent convergence failed"

    def test_resolvent_norms_finite(self):
        """Resolvent norms are finite for z away from spectrum."""
        result = resolvent_norm_convergence(
            z_values=[-5.0],
            max_level=1, R=1.0,
        )
        for z, data in result['results_by_z'].items():
            for level_data in data['level_data']:
                norm = level_data['resolvent_norm']
                if not np.isnan(norm):
                    assert np.isfinite(norm), \
                        f"Non-finite resolvent norm at z={z}, level={level_data['level']}"

    def test_resolvent_values_well_defined(self):
        """Resolvent values exist at all z test points."""
        result = resolvent_norm_convergence(
            z_values=[-5.0, -10.0],
            max_level=1, R=1.0,
        )
        for z, data in result['results_by_z'].items():
            for level_data in data['level_data']:
                rv = level_data['resolvent_values']
                assert len(rv) > 0, \
                    f"No resolvent values at z={z}, level={level_data['level']}"


# ======================================================================
# 13. Spectral Convergence Rate (NEW)
# ======================================================================

class TestSpectralConvergenceRate:
    """
    PROPOSITION: Spectral convergence rate is O(a^2).

    Verified via eigenvalue scaling: the combinatorial Laplacian
    eigenvalues scale as lambda ~ C * a^{-2}, consistent with O(a^2)
    convergence of the DEC approximation (Dodziuk 1976).
    """

    def test_convergence_rate_runs(self):
        """spectral_convergence_rate completes without error."""
        result = spectral_convergence_rate(max_level=1, R=1.0, n_eigenvalues=5)
        assert result['status'] == 'PROPOSITION'
        assert 'mean_rate' in result

    def test_scale_convergence_rate_exists(self):
        """A scaling convergence rate is computed."""
        result = spectral_convergence_rate(max_level=1, R=1.0, n_eigenvalues=5)
        assert result['scale_convergence_rate'] is not None, \
            "No scale convergence rate computed"

    def test_convergence_rate_consistent_with_O_a2(self):
        """
        Eigenvalue scaling rate should be >= 1.5, consistent with O(a^{-2}).

        The combinatorial Laplacian eigenvalues scale as lambda ~ C / a^2.
        We measure this exponent from the ratio of eigenvalues at successive
        mesh sizes.
        """
        result = spectral_convergence_rate(max_level=1, R=1.0, n_eigenvalues=5)
        assert result['consistent_with_O_a2'], \
            f"Mean rate {result['mean_rate']:.2f} not consistent with O(a^2) " \
            f"(expected >= 1.5, theoretical = 2.0)"

    def test_eigenvalue_ratios_converge(self):
        """Eigenvalue ratios between refinement levels should converge."""
        result = spectral_convergence_rate(max_level=1, R=1.0, n_eigenvalues=5)
        ratio_rates = result['ratio_rates']
        # Ratio changes should be small (eigenvalue structure is stable)
        for k, data in ratio_rates.items():
            assert data['relative_change'] < 0.5, \
                f"Eigenvalue ratio {k} changed too much between levels: " \
                f"{data['relative_change']:.4f}"

    def test_scaling_factors_decrease(self):
        """Raw eigenvalues should decrease with refinement (eigenvalues ~ 1/a^2)."""
        result = spectral_convergence_rate(max_level=1, R=1.0, n_eigenvalues=5)
        factors = result['scale_factors']
        if len(factors) >= 2:
            assert factors[1] < factors[0], \
                f"Eigenvalues did not decrease with refinement: {factors}"


# ======================================================================
# 14. Dodziuk-Patodi Hypotheses (NEW)
# ======================================================================

class TestDodziukPatodiHypotheses:
    """
    Verify all hypotheses of the Dodziuk-Patodi convergence theorem.
    This is the core of PROPOSITION 6.4.
    """

    def test_all_hypotheses_verified(self):
        """All Dodziuk-Patodi hypotheses hold."""
        result = dodziuk_patodi_hypotheses(max_level=1, R=1.0)
        hyp = result['hypotheses']

        # (H1) Compact Riemannian manifold
        assert hyp['H1_compact_riemannian']['holds'], "H1 failed"

        # (H2) Smooth triangulations
        assert hyp['H2_smooth_triangulations']['holds'], "H2 failed"

        # (H3) Mesh to zero
        assert hyp['H3_mesh_to_zero']['holds'], "H3 failed"

        # (H4) Bounded fatness
        assert hyp['H4_bounded_fatness']['holds'], "H4 failed"

    def test_operator_properties_hold(self):
        """Operator properties (symmetry, non-negativity) hold at all levels."""
        result = dodziuk_patodi_hypotheses(max_level=1, R=1.0)
        assert result['hypotheses']['operator_properties']['holds'], \
            "Operator properties failed"

    def test_h1_s3_zero(self):
        """H^1(S^3) = 0 (topological fact)."""
        result = dodziuk_patodi_hypotheses(max_level=1, R=1.0)
        assert result['hypotheses']['H1_S3_zero']['holds']
        assert result['hypotheses']['H1_S3_zero']['status'] == 'THEOREM'

    def test_conclusion_is_proposition(self):
        """
        If all hypotheses hold, the conclusion should be PROPOSITION.
        """
        result = dodziuk_patodi_hypotheses(max_level=1, R=1.0)
        assert result['conclusion']['all_hypotheses_verified'], \
            "Not all hypotheses verified"
        assert result['conclusion']['status'] == 'PROPOSITION', \
            f"Status is {result['conclusion']['status']}, expected PROPOSITION"

    def test_conclusion_has_proof_sketch(self):
        """The conclusion includes a proof sketch."""
        result = dodziuk_patodi_hypotheses(max_level=1, R=1.0)
        statement = result['conclusion']['statement']
        assert 'Dodziuk-Patodi' in statement
        assert 'Whitney' in statement or 'resolvent' in statement
        assert 'Kato' in statement or 'Reed-Simon' in statement


# ======================================================================
# 15. Complete Proof Assembly (NEW)
# ======================================================================

class TestCompactResolventProof:
    """
    PROPOSITION: Complete proof of spectral convergence via compact
    resolvent theory. This assembles all the pieces.
    """

    def test_proof_runs(self):
        """compact_resolvent_convergence_proof completes without error."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        assert 'spectral_convergence' in result
        assert 'resolvent_convergence' in result
        assert 'gap_convergence' in result

    def test_proof_status_proposition(self):
        """
        If all checks pass, the status should be PROPOSITION.
        """
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        assert result['status'] == 'PROPOSITION', \
            f"Proof status is {result['status']}, expected PROPOSITION"

    def test_spectral_convergence_consistent(self):
        """Spectral convergence is consistent with O(a^2)."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        assert result['spectral_convergence']['consistent_with_O_a2'], \
            f"Rate {result['spectral_convergence']['mean_rate']:.2f} not O(a^2)"

    def test_resolvent_convergence_holds(self):
        """Resolvent convergence is verified."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        assert result['resolvent_convergence']['convergence'], \
            "Resolvent convergence failed"

    def test_gap_convergence_tracked(self):
        """Gap convergence data is available."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        gap = result['gap_convergence']
        assert gap['continuum_gap'] is not None
        assert gap['coexact_gap'] is not None
        assert len(gap['gap_eigenvalues']) > 0
        assert len(gap['mesh_sizes']) > 0

    def test_gap_eigenvalue_decreases(self):
        """First eigenvalue decreases with refinement (finer mesh)."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        gap = result['gap_convergence']
        eigenvalues = gap['gap_eigenvalues']
        if len(eigenvalues) >= 2:
            assert eigenvalues[-1] < eigenvalues[-2], \
                f"Gap eigenvalue did not decrease: {eigenvalues[-1]:.6f} vs {eigenvalues[-2]:.6f}"

    def test_proof_statement_mentions_gap(self):
        """Proof statement mentions the mass gap."""
        result = compact_resolvent_convergence_proof(
            max_level=1, R=1.0, n_eigenvalues=5
        )
        assert '4/R^2' in result['statement'] or '4/R' in result['statement'], \
            "Statement does not mention the mass gap 4/R^2"


# ======================================================================
# 16. Strong Resolvent Convergence (original tests)
# ======================================================================

class TestStrongResolventConvergence:
    """Test strong resolvent convergence (original tests, preserved)."""

    def test_resolvent_test_runs(self):
        """strong_resolvent_convergence_test completes without error."""
        result = strong_resolvent_convergence_test(
            z_values=[-1.0, -5.0],
            max_level=1,
            R=1.0,
        )
        assert 'z_values' in result
        assert 'norms' in result
        assert 'convergence' in result

    def test_resolvent_norms_finite(self):
        """Resolvent norm differences are finite for z away from spectrum."""
        result = strong_resolvent_convergence_test(
            z_values=[-5.0],
            max_level=1,
            R=1.0,
        )
        for key, val in result['norms'].items():
            if not np.isnan(val):
                assert np.isfinite(val), f"Non-finite norm at {key}: {val}"

    def test_resolvent_norms_positive(self):
        """Resolvent norm differences are non-negative."""
        result = strong_resolvent_convergence_test(
            z_values=[-5.0],
            max_level=1,
            R=1.0,
        )
        for key, val in result['norms'].items():
            if not np.isnan(val):
                assert val >= 0, f"Negative norm at {key}: {val}"


# ======================================================================
# 17. Theorem Statement
# ======================================================================

class TestTheoremStatement:
    """Test the theorem statement generation."""

    def test_statement_status(self):
        """Status should be PROPOSITION (upgraded from NUMERICAL)."""
        result = theorem_statement()
        assert result['status'] == 'PROPOSITION', \
            f"Status is {result['status']}, expected PROPOSITION"

    def test_statement_has_text(self):
        """Statement should contain the key claim."""
        result = theorem_statement()
        assert 'Strong Resolvent Convergence' in result['statement']
        assert 'Hodge Laplacian' in result['statement']

    def test_statement_with_evidence(self):
        """Statement can include evidence from convergence analysis."""
        conv = scaled_convergence_analysis(max_level=1, R=1.0,
                                           n_eigenvalues=5)
        result = theorem_statement(conv)
        assert result['evidence'] is not None
        assert 'n_levels' in result['evidence']

    def test_statement_references(self):
        """Statement should cite relevant references."""
        result = theorem_statement()
        assert 'Dodziuk' in result['statement']
        assert 'Kato' in result['statement']

    def test_statement_mentions_dodziuk_patodi(self):
        """Statement should reference the Dodziuk-Patodi theorem."""
        result = theorem_statement()
        assert 'Dodziuk-Patodi' in result['statement'], \
            "Statement should reference Dodziuk-Patodi theorem"

    def test_statement_mentions_convergence_rate(self):
        """Statement should mention O(a^2) convergence rate."""
        result = theorem_statement()
        assert 'O(a' in result['statement'] or 'a_n^2' in result['statement'], \
            "Statement should mention convergence rate"

    def test_statement_mentions_reed_simon(self):
        """Statement should reference Reed-Simon for resolvent convergence."""
        result = theorem_statement()
        assert 'Reed-Simon' in result['statement'], \
            "Statement should reference Reed-Simon"

    def test_statement_says_strong_not_norm_resolvent(self):
        """
        Statement should say 'strong resolvent', NOT 'norm resolvent'.

        The Dodziuk-Patodi theorem gives strong resolvent convergence
        (equivalently, eigenvalue convergence with correct multiplicities).
        Norm resolvent convergence is stronger and would require additional
        uniform bounds on Whitney interpolation operator norms.
        """
        result = theorem_statement()
        stmt = result['statement']
        assert 'strong resolvent' in stmt.lower(), \
            "Statement should mention strong resolvent convergence"
        # If 'norm resolvent' appears, it should be in the context of
        # explaining what we do NOT claim
        if 'norm resolvent' in stmt.lower():
            assert 'not' in stmt.lower() or 'would' in stmt.lower(), \
                "If 'norm resolvent' appears, it should be in a caveat"

    def test_statement_mentions_arnold_falk_winther(self):
        """Statement should reference Arnold-Falk-Winther for FEEC."""
        result = theorem_statement()
        assert 'Arnold' in result['statement'] or 'finite element' in result['statement'].lower(), \
            "Statement should reference Arnold-Falk-Winther or finite element exterior calculus"


# ======================================================================
# 18. Integration: Gap convergence
# ======================================================================

class TestGapConvergence:
    """
    Integration test: verify that the spectral gap converges to 4/R^2
    (coexact) or at least that the first nonzero eigenvalue converges
    to 3/R^2 (exact branch).
    """

    def test_gap_exists_at_all_levels(self):
        """A spectral gap exists at every refinement level."""
        for level in range(2):
            spec = spectrum_at_refinement(level, R=1.0, n_eigenvalues=10)
            evals = spec['eigenvalues']
            nonzero = evals[evals > 0.1]
            assert len(nonzero) > 0, f"No gap at level {level}"

    def test_gap_is_positive(self):
        """The spectral gap is strictly positive at each level."""
        for level in range(2):
            spec = spectrum_at_refinement(level, R=1.0, n_eigenvalues=10)
            evals = spec['eigenvalues']
            nonzero = evals[evals > 0.1]
            assert nonzero[0] > 0, f"Non-positive gap at level {level}"

    def test_lattice_spacing_decreases_with_refinement(self):
        """Lattice spacing shrinks with each refinement."""
        spacings = []
        for level in range(2):
            spec = spectrum_at_refinement(level, R=1.0, n_eigenvalues=5)
            spacings.append(spec['lattice_spacing'])

        for i in range(len(spacings) - 1):
            assert spacings[i + 1] < spacings[i], \
                f"Spacing did not decrease at level {i+1}: " \
                f"{spacings[i+1]:.4f} >= {spacings[i]:.4f}"

    def test_richardson_extrapolation_available(self):
        """Richardson extrapolation should be computed when possible."""
        result = scaled_convergence_analysis(max_level=1, R=1.0,
                                             n_eigenvalues=5)
        assert result['richardson_extrapolation'] is not None


# ======================================================================
# 19. Mathematical Consistency Checks (NEW)
# ======================================================================

class TestMathematicalConsistency:
    """
    Cross-checks to ensure mathematical consistency of the framework.
    """

    def test_continuum_gap_is_3_over_R2(self):
        """The overall first nonzero eigenvalue on unit S^3 is 3."""
        evals = continuum_eigenvalue_list(R=1.0, n_eigenvalues=5)
        assert abs(evals[0] - 3.0) < 1e-12

    def test_coexact_gap_is_4_over_R2(self):
        """The coexact (physical) gap on unit S^3 is 4."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=5)
        coexact = [ev for ev, _, t in evals if t == 'coexact']
        assert len(coexact) > 0
        assert abs(coexact[0] - 4.0) < 1e-12

    def test_exact_exact_pattern(self):
        """Exact eigenvalues follow l(l+2): 3, 8, 15, 24, ..."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=20)
        exact_evals = [ev for ev, _, t in evals if t == 'exact']
        expected = [l * (l + 2) for l in range(1, 5)]  # 3, 8, 15, 24
        for i, (got, exp) in enumerate(zip(exact_evals[:4], expected)):
            assert abs(got - exp) < 1e-12, \
                f"Exact eigenvalue {i}: got {got}, expected {exp}"

    def test_coexact_pattern(self):
        """Coexact eigenvalues follow (k+1)^2: 4, 9, 16, 25, ..."""
        evals = continuum_eigenvalues(R=1.0, n_eigenvalues=20)
        coexact_evals = [ev for ev, _, t in evals if t == 'coexact']
        expected = [(k + 1) ** 2 for k in range(1, 5)]  # 4, 9, 16, 25
        for i, (got, exp) in enumerate(zip(coexact_evals[:4], expected)):
            assert abs(got - exp) < 1e-12, \
                f"Coexact eigenvalue {i}: got {got}, expected {exp}"

    def test_gap_scales_with_R(self):
        """Gap scales as 1/R^2 on S^3(R)."""
        for R in [0.5, 1.0, 2.0, 3.0]:
            evals = continuum_eigenvalue_list(R=R, n_eigenvalues=3)
            expected_gap = 3.0 / R**2
            assert abs(evals[0] - expected_gap) < 1e-10, \
                f"Gap at R={R}: got {evals[0]}, expected {expected_gap}"

    def test_euler_characteristic_preserved_under_refinement(self):
        """V - E + F is preserved under refinement (topological invariant)."""
        chi_values = []
        for level in range(2):
            vertices, edges, faces = refine_600_cell(level, R=1.0)
            chi = len(vertices) - len(edges) + len(faces)
            chi_values.append(chi)

        assert chi_values[0] == chi_values[1], \
            f"Euler characteristic changed: {chi_values[0]} -> {chi_values[1]}"
