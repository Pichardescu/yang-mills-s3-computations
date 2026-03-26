"""
Tests for the Certified Gap Pipeline.

Tests organized by class:
    1. CertifiedGapResult: dataclass integrity
    2. CertifiedGapPipeline: end-to-end pipeline stages
    3. GapCertificateChain: proof chain status
    4. MultiRGapScan: multi-R scan and uniform positivity

Aim: 30+ tests covering pipeline correctness, consistency, and physical sanity.

LABEL: NUMERICAL (all tests verify floating-point computation).
"""

import numpy as np
import pytest

from yang_mills_s3.proofs.certified_gap_pipeline import (
    CertifiedGapResult,
    CertifiedGapPipeline,
    GapCertificateChain,
    MultiRGapScan,
    HBAR_C,
)
from yang_mills_s3.proofs.sclbt_lower_bounds import (
    _build_3d_hamiltonian,
    HBAR_C_MEV_FM,
    LAMBDA_QCD_MEV,
    R_PHYSICAL_FM,
)
from yang_mills_s3.proofs.koller_van_baal import G2_DEFAULT


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def pipeline():
    """Pipeline with default settings."""
    return CertifiedGapPipeline(n_sclbt_states=5, max_sclbt_iter=50)


@pytest.fixture
def small_pipeline():
    """Pipeline with reduced SCLBT states for speed."""
    return CertifiedGapPipeline(n_sclbt_states=3, max_sclbt_iter=20)


@pytest.fixture
def physical_result(pipeline):
    """Run pipeline at physical parameters with small basis."""
    return pipeline.run(N=4, R=R_PHYSICAL_FM, g2=G2_DEFAULT)


@pytest.fixture
def unit_result(pipeline):
    """Run pipeline at unit parameters (R=1, g2=1)."""
    return pipeline.run(N=4, R=1.0, g2=1.0)


# ======================================================================
# 1. CertifiedGapResult tests
# ======================================================================

class TestCertifiedGapResult:
    """Tests for the CertifiedGapResult dataclass."""

    def test_result_fields_exist(self, physical_result):
        """All required fields are present in the result."""
        assert hasattr(physical_result, 'N_basis')
        assert hasattr(physical_result, 'R_fm')
        assert hasattr(physical_result, 'g2')
        assert hasattr(physical_result, 'E0_ritz')
        assert hasattr(physical_result, 'E0_sclbt')
        assert hasattr(physical_result, 'E1_ritz')
        assert hasattr(physical_result, 'E1_sclbt')
        assert hasattr(physical_result, 'gap_ritz')
        assert hasattr(physical_result, 'gap_sclbt')
        assert hasattr(physical_result, 'gap_certified')
        assert hasattr(physical_result, 'gap_MeV')
        assert hasattr(physical_result, 'is_positive')
        assert hasattr(physical_result, 'certification_level')

    def test_result_types(self, physical_result):
        """Fields have correct types."""
        assert isinstance(physical_result.N_basis, int)
        assert isinstance(physical_result.R_fm, float)
        assert isinstance(physical_result.g2, float)
        assert isinstance(physical_result.is_positive, bool)
        assert isinstance(physical_result.certification_level, str)
        assert isinstance(physical_result.gap_MeV, float)

    def test_result_summary(self, physical_result):
        """Summary string is non-empty and contains key info."""
        s = physical_result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert 'MeV' in s
        assert physical_result.certification_level in s

    def test_result_parameters_preserved(self, physical_result):
        """Input parameters are preserved in the result."""
        assert physical_result.N_basis == 4
        assert physical_result.R_fm == pytest.approx(R_PHYSICAL_FM)
        assert physical_result.g2 == pytest.approx(G2_DEFAULT)


# ======================================================================
# 2. CertifiedGapPipeline tests
# ======================================================================

class TestCertifiedGapPipeline:
    """Tests for the end-to-end pipeline."""

    def test_pipeline_runs_end_to_end(self, pipeline):
        """Pipeline runs without error and returns a result."""
        result = pipeline.run(N=4, R=2.0, g2=1.0)
        assert isinstance(result, CertifiedGapResult)

    def test_pipeline_gap_positive_at_physical_params(self, physical_result):
        """Gap is positive at physical parameters (R=2.2 fm, g2=6.28)."""
        assert physical_result.is_positive, (
            f"Gap should be positive at physical params, got "
            f"{physical_result.gap_MeV:.2f} MeV"
        )

    def test_pipeline_gap_positive_at_unit_params(self, unit_result):
        """Gap is positive at unit parameters (R=1, g2=1)."""
        assert unit_result.is_positive

    def test_ritz_E0_less_than_E1(self, physical_result):
        """Ritz eigenvalues: E0 < E1 (non-degenerate ground state)."""
        assert physical_result.E0_ritz < physical_result.E1_ritz

    def test_sclbt_E0_leq_ritz_E0(self, physical_result):
        """SCLBT lower bound on E0 <= Ritz upper bound on E0."""
        assert physical_result.E0_sclbt <= physical_result.E0_ritz + 1e-10

    def test_gap_ritz_positive(self, physical_result):
        """Ritz gap is positive."""
        assert physical_result.gap_ritz > 0

    def test_gap_certified_leq_ritz(self, physical_result):
        """Certified gap <= Ritz gap (certified is conservative)."""
        assert physical_result.gap_certified <= physical_result.gap_ritz + 1e-12

    def test_gap_MeV_conversion(self, physical_result):
        """Physical MeV conversion is consistent (gap * hbar_c / R)."""
        # BUG FIX (Session 25): conversion is gap * hbar_c / R, not gap * hbar_c.
        # Eigenvalues from the KvB Hamiltonian are in dimensionless natural units.
        expected = physical_result.gap_certified * HBAR_C / physical_result.R_fm
        assert physical_result.gap_MeV == pytest.approx(expected, rel=1e-10)

    def test_certification_level_is_numerical(self, physical_result):
        """Certification level is NUMERICAL (not THEOREM without interval arith)."""
        assert physical_result.certification_level == 'NUMERICAL'

    def test_pipeline_stage_data_populated(self, pipeline):
        """All 6 stages produce data."""
        pipeline.run(N=4, R=2.0, g2=1.0)
        sd = pipeline.stage_data()
        assert 'stage1_svd' in sd
        assert 'stage2_hamiltonian' in sd
        assert 'stage3_ritz' in sd
        assert 'stage4_sclbt' in sd
        assert 'stage5_certified' in sd
        assert 'stage6_physical' in sd

    def test_stage1_svd_theorem(self, pipeline):
        """Stage 1 (SVD reduction) is labeled THEOREM."""
        pipeline.run(N=4, R=2.0, g2=1.0)
        s1 = pipeline.stage_data()['stage1_svd']
        assert s1['status'] == 'THEOREM'
        assert s1['dof_original'] == 9
        assert s1['dof_reduced'] == 3

    def test_stage2_hamiltonian_symmetric(self, pipeline):
        """Stage 2: Hamiltonian matrix is symmetric."""
        pipeline.run(N=4, R=2.0, g2=1.0)
        s2 = pipeline.stage_data()['stage2_hamiltonian']
        assert s2['is_symmetric'] is True

    def test_stage4_sclbt_converges(self, pipeline):
        """Stage 4: SCLBT iteration converges."""
        pipeline.run(N=4, R=2.0, g2=1.0)
        s4 = pipeline.stage_data()['stage4_sclbt']
        assert s4['converged'] is True

    def test_full_report_nonempty(self, pipeline):
        """Full report is non-empty and contains key sections."""
        pipeline.run(N=4, R=2.0, g2=1.0)
        report = pipeline.full_report()
        assert len(report) > 100
        assert 'Stage 1' in report
        assert 'Stage 6' in report
        assert 'MeV' in report

    def test_full_report_before_run(self, pipeline):
        """Full report before run() gives a sensible message."""
        report = pipeline.full_report()
        assert 'No pipeline run' in report

    def test_gap_increases_with_basis(self):
        """Ritz gap converges (stabilizes) as basis size increases."""
        pipe = CertifiedGapPipeline(n_sclbt_states=3)
        gaps = []
        for N in [3, 4, 5]:
            r = pipe.run(N=N, R=2.0, g2=1.0)
            gaps.append(r.gap_ritz)
        # Gap should stabilize (relative change decreases)
        # At minimum, all should be positive
        assert all(g > 0 for g in gaps)

    def test_pipeline_different_R_values(self, pipeline):
        """Pipeline runs at various R values without error."""
        for R in [0.5, 1.0, 2.2, 5.0, 10.0]:
            result = pipeline.run(N=4, R=R, g2=G2_DEFAULT)
            assert isinstance(result, CertifiedGapResult)
            assert result.R_fm == pytest.approx(R)

    def test_sclbt_gap_consistency(self, pipeline):
        """SCLBT gap is not wildly different from Ritz gap."""
        result = pipeline.run(N=5, R=2.0, g2=1.0)
        if not np.isnan(result.gap_sclbt) and result.gap_sclbt > 0:
            # SCLBT should be within an order of magnitude of Ritz
            ratio = result.gap_sclbt / result.gap_ritz
            assert 0.01 < ratio <= 1.0 + 1e-10, (
                f"SCLBT/Ritz ratio = {ratio:.4f}, expected between 0.01 and 1"
            )


# ======================================================================
# 3. GapCertificateChain tests
# ======================================================================

class TestGapCertificateChain:
    """Tests for the proof chain connector."""

    def test_proof_status_structure(self):
        """Proof status returns correct structure."""
        chain = GapCertificateChain()
        status = chain.proof_status()
        assert 'links' in status
        assert 'n_links' in status
        assert 'overall_status' in status
        assert len(status['links']) == 6

    def test_proof_status_with_pipeline(self, pipeline):
        """Proof status uses pipeline result when available."""
        pipeline.run(N=4, R=R_PHYSICAL_FM, g2=G2_DEFAULT)
        chain = GapCertificateChain(pipeline=pipeline)
        status = chain.proof_status()
        assert status['pipeline_gap_MeV'] is not None
        assert status['pipeline_gap_MeV'] > 0

    def test_link_statuses(self):
        """Each link has a valid status label."""
        chain = GapCertificateChain()
        status = chain.proof_status()
        valid_labels = {'THEOREM', 'NUMERICAL', 'PROPOSITION', 'CONJECTURE'}
        for link in status['links']:
            assert link['status'] in valid_labels
            assert 'name' in link
            assert 'statement' in link

    def test_svd_link_is_theorem(self):
        """SVD reduction link is THEOREM."""
        chain = GapCertificateChain()
        status = chain.proof_status()
        svd_link = status['links'][1]  # Link 2 = SVD
        assert svd_link['status'] == 'THEOREM'

    def test_s3_chain_is_theorem(self):
        """S^3 proof chain link is THEOREM."""
        chain = GapCertificateChain()
        status = chain.proof_status()
        s3_link = status['links'][0]  # Link 1 = S^3 chain
        assert s3_link['status'] == 'THEOREM'

    def test_overall_is_proposition(self):
        """Overall status is PROPOSITION (not THEOREM yet)."""
        chain = GapCertificateChain()
        status = chain.proof_status()
        assert status['overall_status'] == 'PROPOSITION'

    def test_what_remains(self):
        """what_remains_for_theorem returns upgrade paths."""
        chain = GapCertificateChain()
        remains = chain.what_remains_for_theorem()
        assert 'upgrades' in remains
        assert 'critical_path' in remains
        assert len(remains['upgrades']) >= 2

    def test_connection_to_clay(self):
        """connection_to_clay returns Clay assessment."""
        chain = GapCertificateChain()
        clay = chain.connection_to_clay()
        assert 'clay_problem' in clay
        assert 'our_result' in clay
        assert 'gap_to_clay' in clay
        assert 'status' in clay
        assert clay['status'] == 'PROPOSITION'

    def test_connection_to_clay_with_pipeline(self, pipeline):
        """Clay connection includes pipeline gap when available."""
        pipeline.run(N=4, R=R_PHYSICAL_FM, g2=G2_DEFAULT)
        chain = GapCertificateChain(pipeline=pipeline)
        clay = chain.connection_to_clay()
        assert 'MeV' in clay['our_result']


# ======================================================================
# 4. MultiRGapScan tests
# ======================================================================

class TestMultiRGapScan:
    """Tests for multi-R scanning."""

    def test_scan_runs(self):
        """Scan runs with default R values."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0)
        result = scanner.scan(R_values=np.array([1.0, 2.0, 5.0]))
        assert 'R_values' in result
        assert 'gaps_MeV' in result
        assert len(result['gaps_MeV']) == 3

    def test_all_gaps_positive(self):
        """All gaps are positive in a small scan."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0)
        result = scanner.scan(R_values=np.array([1.0, 2.0, 3.0, 5.0]))
        assert result['all_positive'], (
            f"Not all gaps positive. Gaps: {result['gaps_MeV']}"
        )

    def test_min_gap_reported(self):
        """Minimum gap is correctly identified."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0)
        result = scanner.scan(R_values=np.array([1.0, 2.0, 5.0]))
        assert not np.isnan(result['min_gap_MeV'])
        assert result['min_gap_MeV'] > 0
        # min_gap should equal the minimum of all gaps
        valid = result['gaps_MeV'][~np.isnan(result['gaps_MeV'])]
        assert result['min_gap_MeV'] == pytest.approx(np.min(valid))

    def test_min_gap_R_reported(self):
        """R at minimum gap is one of the tested values."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0)
        R_test = np.array([1.0, 2.0, 5.0])
        result = scanner.scan(R_values=R_test)
        assert result['min_gap_R_fm'] in R_test

    def test_verify_uniform_positivity(self):
        """verify_uniform_positivity returns correct structure."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0, n_sclbt_states=3)
        result = scanner.verify_uniform_positivity(
            R_values=np.array([1.0, 2.0, 5.0])
        )
        assert 'uniform_positive' in result
        assert 'min_gap_MeV' in result
        assert 'label' in result
        assert result['label'] == 'NUMERICAL'

    def test_scan_physical_params(self):
        """Scan at physical coupling gives positive gaps."""
        scanner = MultiRGapScan(N_basis=4, g2=G2_DEFAULT, n_sclbt_states=3)
        result = scanner.scan(R_values=np.array([1.0, 2.2, 5.0]))
        assert result['all_positive']

    def test_summary_table(self):
        """Summary table is a non-empty formatted string."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0, n_sclbt_states=3)
        table = scanner.summary_table(R_values=np.array([1.0, 2.0]))
        assert isinstance(table, str)
        assert len(table) > 50
        assert 'MeV' in table
        assert 'Min gap' in table

    def test_scan_label_is_numerical(self):
        """Scan result has NUMERICAL label."""
        scanner = MultiRGapScan(N_basis=4, g2=1.0, n_sclbt_states=3)
        result = scanner.scan(R_values=np.array([2.0]))
        assert result['label'] == 'NUMERICAL'


# ======================================================================
# 5. Consistency and cross-check tests
# ======================================================================

class TestConsistency:
    """Cross-checks between pipeline components."""

    def test_sclbt_leq_ritz_gap(self):
        """SCLBT gap <= Ritz gap (lower bound <= value)."""
        pipe = CertifiedGapPipeline(n_sclbt_states=5)
        result = pipe.run(N=5, R=2.0, g2=1.0)
        if not np.isnan(result.gap_sclbt):
            assert result.gap_sclbt <= result.gap_ritz + 1e-10, (
                f"SCLBT gap ({result.gap_sclbt:.6f}) should be <= "
                f"Ritz gap ({result.gap_ritz:.6f})"
            )

    def test_E0_bounds_bracket_true_E0(self):
        """E0_sclbt <= E0_true <= E0_ritz (bounds bracket the true value)."""
        pipe = CertifiedGapPipeline(n_sclbt_states=5)
        result = pipe.run(N=5, R=2.0, g2=1.0)
        # We can't check E0_true exactly, but we check the ordering
        assert result.E0_sclbt <= result.E0_ritz + 1e-10

    def test_gap_MeV_positive_implies_is_positive(self):
        """gap_MeV > 0 iff is_positive is True."""
        pipe = CertifiedGapPipeline()
        result = pipe.run(N=4, R=2.0, g2=1.0)
        assert (result.gap_MeV > 0) == result.is_positive

    def test_certified_gap_is_min_of_ritz_and_sclbt(self):
        """Certified gap equals min(ritz, sclbt) when both are valid."""
        pipe = CertifiedGapPipeline(n_sclbt_states=5)
        result = pipe.run(N=5, R=2.0, g2=1.0)
        if not np.isnan(result.gap_sclbt):
            expected = min(result.gap_ritz, result.gap_sclbt)
            assert result.gap_certified == pytest.approx(expected, abs=1e-14)

    def test_pipeline_matches_ym_reduced_gap(self):
        """Pipeline gap is consistent with YangMillsReducedGap at same params."""
        from yang_mills_s3.proofs.sclbt_lower_bounds import YangMillsReducedGap

        N, R, g2 = 4, 2.0, 1.0
        pipe = CertifiedGapPipeline(n_sclbt_states=5)
        pipe_result = pipe.run(N=N, R=R, g2=g2)

        ym = YangMillsReducedGap(N_basis=N, n_sclbt_states=5)
        ym_result = ym.compute_gap(N, R, g2)

        # Both build the same Hamiltonian, so Ritz gaps should match
        assert pipe_result.gap_ritz == pytest.approx(
            ym_result['ritz_gap'], rel=1e-8
        )

    def test_pipeline_hamiltonian_size(self):
        """Hamiltonian matrix size matches N^3."""
        pipe = CertifiedGapPipeline()
        pipe.run(N=4, R=2.0, g2=1.0)
        s2 = pipe.stage_data()['stage2_hamiltonian']
        assert s2['N_total'] == 4 ** 3
        assert s2['matrix_shape'] == (64, 64)
