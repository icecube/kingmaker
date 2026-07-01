"""
Unit tests for KingPDF.

Covers initialization, PDF/CDF correctness and consistency, normalization,
angular cutoff enforcement, array broadcasting, sampling, and marginalization.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kingmaker.pdf import KingPDF, MarginalizedKingPDF


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

PARAM_CASES = [
    pytest.param(np.radians(0.5), 2.0, id="narrow-moderate"),
    pytest.param(np.radians(2.0), 2.0, id="wide-moderate"),
    pytest.param(np.radians(1.0), 5.0, id="moderate-heavy"),
    pytest.param(np.radians(1.0), 100.0, id="near-gaussian"),
]

CUTOFF_CASES = [
    pytest.param(np.pi, id="full-sphere"),
    pytest.param(np.pi / 2, id="half-sphere"),
    pytest.param(0.2, id="small-cutoff"),
]


def _sphere_integral(pdf_obj, alpha, beta, n=5000):
    """Numerically integrate PDF over the sphere: 2pi * int PDF(theta) sin(theta) dtheta."""
    theta = np.linspace(0, pdf_obj.angular_cutoff, n)
    vals = pdf_obj.pdf(theta, alpha, beta)
    return 2 * np.pi * np.trapezoid(vals * np.sin(theta), theta)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestKingPDFInit:
    def test_default_cutoff(self):
        king = KingPDF()
        assert king.angular_cutoff == np.pi

    @pytest.mark.parametrize("cutoff", [np.pi / 4, np.pi / 2, np.pi])
    def test_custom_cutoff(self, cutoff):
        king = KingPDF(angular_cutoff=cutoff)
        assert king.angular_cutoff == cutoff


# ---------------------------------------------------------------------------
# PDF properties
# ---------------------------------------------------------------------------


class TestKingPDFEval:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_scalar_positive(self, alpha, beta):
        king = KingPDF()
        assert king.pdf(0.0, alpha, beta) > 0

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_array_positive(self, alpha, beta):
        king = KingPDF()
        theta = np.linspace(0, np.radians(5), 50)
        vals = king.pdf(theta, alpha, beta)
        assert np.all(vals >= 0)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_all_finite(self, alpha, beta):
        king = KingPDF()
        theta = np.linspace(0, np.radians(5), 50)
        assert np.all(np.isfinite(king.pdf(theta, alpha, beta)))

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_maximum_at_origin(self, alpha, beta):
        """PDF should be highest at theta=0 (or very close to it)."""
        king = KingPDF()
        theta = np.linspace(0, np.radians(10), 200)
        vals = king.pdf(theta, alpha, beta)
        assert np.argmax(vals) == 0

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_monotone_decreasing(self, alpha, beta):
        """PDF is strictly decreasing from 0 to the core region."""
        king = KingPDF()
        theta = np.linspace(0, min(5 * alpha, king.angular_cutoff), 100)
        vals = king.pdf(theta, alpha, beta)
        assert np.all(np.diff(vals) <= 0)

    @pytest.mark.parametrize("cutoff", CUTOFF_CASES)
    def test_zero_beyond_cutoff(self, cutoff):
        king = KingPDF(angular_cutoff=cutoff)
        alpha, beta = np.radians(1.0), 2.0
        beyond = np.array([cutoff + 0.01, cutoff + 0.1, cutoff + 0.5])
        beyond = beyond[(beyond > cutoff) & (beyond <= np.pi)]
        if len(beyond) == 0:
            pytest.skip("no points strictly beyond cutoff within the sphere")
        assert np.all(king.pdf(beyond, alpha, beta) == 0)

    @pytest.mark.parametrize("cutoff", CUTOFF_CASES)
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_normalizes_to_one(self, cutoff, alpha, beta):
        """2pi * integral of PDF * sin(theta) dtheta should equal 1."""
        if alpha >= cutoff:
            pytest.skip("alpha >= cutoff: most of the PDF is truncated")
        king = KingPDF(angular_cutoff=cutoff)
        integral = _sphere_integral(king, alpha, beta)
        assert_allclose(integral, 1.0, rtol=1e-3)

    def test_scalar_input_returns_finite(self):
        king = KingPDF()
        val = king.pdf(np.radians(1.0), np.radians(1.0), 2.0)
        assert np.isfinite(val)

    def test_broadcasting_alpha_array(self):
        """Evaluate PDF at one angle with multiple alpha values."""
        king = KingPDF()
        x = np.radians(1.0)
        alphas = np.radians([0.5, 1.0, 2.0])
        vals = king.pdf(x, alphas, 2.0)
        assert vals.shape == alphas.shape
        assert np.all(vals >= 0)

    def test_broadcasting_x_and_alpha(self):
        """x and alpha broadcast against each other."""
        king = KingPDF()
        x = np.radians(np.linspace(0.1, 3.0, 10))
        alpha = np.radians([0.5, 1.0, 2.0])
        vals = king.pdf(x[:, None], alpha[None, :], 2.0)
        assert vals.shape == (10, 3)


# ---------------------------------------------------------------------------
# CDF properties
# ---------------------------------------------------------------------------


class TestKingPDFCDF:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_at_zero_is_zero(self, alpha, beta):
        king = KingPDF()
        assert_allclose(king.cdf(0.0, alpha, beta), 0.0, atol=1e-6)

    @pytest.mark.parametrize("cutoff", CUTOFF_CASES)
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_at_cutoff_is_one(self, cutoff, alpha, beta):
        if alpha >= cutoff:
            pytest.skip("alpha >= cutoff: most of the PDF is truncated")
        king = KingPDF(angular_cutoff=cutoff)
        assert_allclose(king.cdf(cutoff, alpha, beta), 1.0, rtol=1e-3)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_monotone(self, alpha, beta):
        king = KingPDF()
        theta = np.linspace(0, np.radians(10), 100)
        vals = king.cdf(theta, alpha, beta)
        assert np.all(np.diff(vals) >= -1e-12)

    @pytest.mark.parametrize("cutoff", CUTOFF_CASES)
    def test_cdf_one_beyond_cutoff(self, cutoff):
        king = KingPDF(angular_cutoff=cutoff)
        alpha, beta = np.radians(0.5), 2.0
        if alpha >= cutoff:
            pytest.skip("alpha >= cutoff")
        beyond = min(cutoff + 0.1, np.pi)
        assert_allclose(king.cdf(beyond, alpha, beta), 1.0, rtol=1e-3)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_bounded_zero_to_one(self, alpha, beta):
        king = KingPDF()
        theta = np.linspace(0, np.pi, 200)
        vals = king.cdf(theta, alpha, beta)
        assert np.all(vals >= -1e-10)
        assert np.all(vals <= 1.0 + 1e-10)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_consistent_with_pdf(self, alpha, beta):
        """CDF(b) - CDF(a) should equal the spherical integral of PDF from a to b."""
        king = KingPDF()
        a, b = np.radians(0.5), np.radians(3.0)
        theta = np.linspace(a, b, 2000)
        pdf_integral = 2 * np.pi * np.trapezoid(king.pdf(theta, alpha, beta) * np.sin(theta), theta)
        cdf_diff = king.cdf(b, alpha, beta) - king.cdf(a, alpha, beta)
        assert_allclose(cdf_diff, pdf_integral, rtol=1e-3)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


class TestKingPDFNorm:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_norm_positive(self, alpha, beta):
        king = KingPDF()
        assert king.norm(alpha, beta) > 0

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_norm_finite(self, alpha, beta):
        king = KingPDF()
        assert np.isfinite(king.norm(alpha, beta))

    def test_norm_decreases_with_alpha(self):
        """Wider PSF → lower peak → smaller norm constant."""
        king = KingPDF()
        n1 = king.norm(np.radians(0.5), 2.0)
        n2 = king.norm(np.radians(2.0), 2.0)
        assert n1 > n2

    def test_norm_array_input(self):
        king = KingPDF()
        alphas = np.radians([0.5, 1.0, 2.0])
        norms = king.norm(alphas, 2.0)
        assert norms.shape == alphas.shape
        assert np.all(norms > 0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestKingPDFSample:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_sample_length(self, n):
        king = KingPDF()
        samples = king.sample(n, np.radians(1.0), 2.0)
        assert len(samples) == n

    @pytest.mark.parametrize("cutoff", [np.pi / 4, np.pi / 2, np.pi])
    def test_samples_within_cutoff(self, cutoff):
        king = KingPDF(angular_cutoff=cutoff)
        samples = king.sample(500, np.radians(1.0), 2.0)
        assert np.all(samples >= 0)
        assert np.all(samples <= cutoff + 1e-9)

    def test_sample_reproducible_with_rng(self):
        king = KingPDF()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = king.sample(100, np.radians(1.0), 2.0, rng=rng1)
        s2 = king.sample(100, np.radians(1.0), 2.0, rng=rng2)
        assert_allclose(s1, s2)

    def test_sample_different_seeds_differ(self):
        king = KingPDF()
        s1 = king.sample(100, np.radians(1.0), 2.0, rng=np.random.default_rng(0))
        s2 = king.sample(100, np.radians(1.0), 2.0, rng=np.random.default_rng(1))
        assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# KingPDF.evaluate
# ---------------------------------------------------------------------------


class TestKingPDFEvaluate:
    @pytest.fixture
    def king(self):
        return KingPDF(angular_cutoff=np.radians(10.0))

    def test_output_shape(self, king):
        src_ras = np.radians([0.0, 90.0])
        src_decs = np.radians([0.0, 30.0])
        ev_ras = np.radians(np.linspace(0, 10, 5))
        ev_decs = np.radians(np.linspace(-5, 5, 5))
        alpha = np.full(5, np.radians(1.0))
        beta = np.full(5, 2.0)
        result = king.evaluate(src_ras, src_decs, ev_ras, ev_decs, alpha, beta)
        assert result.shape == (5, 2)

    def test_returns_sparse_array(self, king):
        from scipy.sparse import csr_array

        result = king.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            np.array([2.0]),
        )
        assert isinstance(result, csr_array)

    def test_nonnegative(self, king):
        rng = np.random.default_rng(0)
        src_ras = np.radians([0.0])
        src_decs = np.radians([0.0])
        ev_ras = rng.uniform(0, 2 * np.pi, 50)
        ev_decs = np.arcsin(rng.uniform(-1, 1, 50))
        alpha = np.full(50, np.radians(1.0))
        beta = np.full(50, 2.0)
        result = king.evaluate(src_ras, src_decs, ev_ras, ev_decs, alpha, beta)
        assert np.all(result.toarray() >= 0)

    def test_zero_beyond_cutoff(self, king):
        src_ras = np.array([0.0])
        src_decs = np.array([0.0])
        ev_ras = np.array([0.0, 0.0])
        ev_decs = np.radians([0.0, 50.0])
        alpha = np.full(2, np.radians(1.0))
        beta = np.full(2, 2.0)
        result = king.evaluate(src_ras, src_decs, ev_ras, ev_decs, alpha, beta).toarray()
        assert result[0, 0] > 0
        assert result[1, 0] == 0

    def test_mask_gives_same_result(self, king):
        rng = np.random.default_rng(1)
        src_ras = np.radians([0.0, 45.0])
        src_decs = np.radians([0.0, 10.0])
        ev_ras = rng.uniform(0, 2 * np.pi, 30)
        ev_decs = np.arcsin(rng.uniform(-1, 1, 30))
        alpha = np.full(30, np.radians(1.0))
        beta = np.full(30, 2.0)
        first = king.evaluate(src_ras, src_decs, ev_ras, ev_decs, alpha, beta)
        second = king.evaluate(src_ras, src_decs, ev_ras, ev_decs, alpha, beta, mask=first)
        assert_allclose(first.toarray(), second.toarray(), rtol=1e-12)


# ---------------------------------------------------------------------------
# MarginalizedKingPDF
# ---------------------------------------------------------------------------


class TestMarginalizedKingPDFInit:
    def test_builds_cache_with_defaults(self):
        mkpdf = MarginalizedKingPDF(
            source_declination=np.radians([-30.0, 0.0, 30.0]),
            angular_cutoff=np.radians(10.0),
        )
        assert mkpdf._grid.shape[0] == 3
        assert np.all(np.isfinite(mkpdf._grid))

    def test_king_instance_stored(self):
        mkpdf = MarginalizedKingPDF(
            source_declination=np.radians([0.0]),
            angular_cutoff=np.radians(10.0),
        )
        assert isinstance(mkpdf.king, KingPDF)
        assert mkpdf.king.angular_cutoff == mkpdf.angular_cutoff

    def test_source_declination_stored_sorted(self):
        mkpdf = MarginalizedKingPDF(
            source_declination=np.radians([30.0, -30.0, 0.0]),
            angular_cutoff=np.radians(10.0),
        )
        assert np.all(np.diff(mkpdf.source_declination) >= 0)

    def test_custom_grid_sizes_respected(self):
        mkpdf = MarginalizedKingPDF(
            source_declination=np.radians([0.0]),
            angular_cutoff=np.radians(10.0),
            n_signed_delta_dec=40,
            n_ra_bins=20,
        )
        assert mkpdf._grid.shape == (1, 30, 20, 40)

    def test_invalid_points_alpha_raises(self):
        with pytest.raises(ValueError):
            MarginalizedKingPDF(
                source_declination=np.radians([0.0]),
                points_alpha=np.array([-0.1, 0.5, 1.0]),
            )

    def test_invalid_points_beta_raises(self):
        with pytest.raises(ValueError):
            MarginalizedKingPDF(
                source_declination=np.radians([0.0]),
                points_beta=np.array([0.5, 1.0, 2.0]),
            )


class TestMarginalizedKingPDFPdf:
    @pytest.fixture
    def mkpdf(self):
        return MarginalizedKingPDF(
            source_declination=np.radians(np.linspace(-80, 80, 21)),
            angular_cutoff=np.radians(10.0),
        )

    def test_returns_dense_array(self, mkpdf):
        result = mkpdf.pdf(np.radians([0.0, 1.0]), np.radians(1.0), 2.0, np.radians(0.0))
        assert isinstance(result, np.ndarray)

    def test_nonnegative(self, mkpdf):
        x = np.radians(np.linspace(-5.0, 5.0, 20))
        alpha = np.full(20, np.radians(1.0))
        beta = np.full(20, 2.0)
        result = mkpdf.pdf(x, alpha, beta, np.radians(0.0))
        assert np.all(result >= 0)

    def test_zero_beyond_cutoff(self, mkpdf):
        result = mkpdf.pdf(np.radians([0.0, 50.0]), np.radians(1.0), 2.0, np.radians(0.0))
        assert result[0] > 0
        assert result[1] == 0

    def test_peaks_near_source_declination(self, mkpdf):
        source_dec = np.radians(20.0)
        x = np.radians(np.linspace(15.0, 25.0, 21))
        alpha = np.full(len(x), np.radians(1.0))
        beta = np.full(len(x), 2.0)
        result = mkpdf.pdf(x, alpha, beta, source_dec)
        assert np.argmax(result) == np.argmin(np.abs(x - source_dec))

    def test_scalar_alpha_beta_broadcast(self, mkpdf):
        x = np.radians(np.linspace(-3.0, 3.0, 10))
        result = mkpdf.pdf(x, np.radians(1.0), 2.0, np.radians(0.0))
        assert result.shape == (10,)


class TestMarginalizedKingPDFEvaluate:
    @pytest.fixture
    def mkpdf(self):
        return MarginalizedKingPDF(
            source_declination=np.radians(np.linspace(-80, 80, 21)),
            angular_cutoff=np.radians(10.0),
        )

    def test_output_shape(self, mkpdf):
        source_decs = np.radians([-10.0, 0.0, 10.0])
        event_decs = np.radians(np.linspace(-15, 15, 5))
        alpha = np.full(5, np.radians(1.0))
        beta = np.full(5, 2.0)
        result = mkpdf.evaluate(source_decs, event_decs, alpha, beta)
        assert result.shape == (5, 3)

    def test_nonnegative(self, mkpdf):
        source_decs = np.radians([-10.0, 0.0, 10.0])
        event_decs = np.radians(np.linspace(-15, 15, 5))
        alpha = np.full(5, np.radians(1.0))
        beta = np.full(5, 2.0)
        result = mkpdf.evaluate(source_decs, event_decs, alpha, beta)
        assert np.all(result.toarray() >= 0)

    def test_zero_beyond_cutoff(self, mkpdf):
        source_decs = np.radians([0.0])
        event_decs = np.radians([0.0, 50.0])
        alpha = np.full(2, np.radians(1.0))
        beta = np.full(2, 2.0)
        result = mkpdf.evaluate(source_decs, event_decs, alpha, beta).toarray()
        assert result[0, 0] > 0
        assert result[1, 0] == 0

    def test_returns_sparse_array(self, mkpdf):
        from scipy.sparse import csr_array

        result = mkpdf.evaluate(
            np.radians([0.0]),
            np.radians([0.0]),
            np.array([np.radians(1.0)]),
            np.array([2.0]),
        )
        assert isinstance(result, csr_array)

    def test_mask_gives_same_result(self, mkpdf):
        source_decs = np.radians([-10.0, 0.0, 10.0])
        event_decs = np.radians(np.linspace(-15, 15, 30))
        alpha = np.full(30, np.radians(1.0))
        beta = np.full(30, 2.0)
        first = mkpdf.evaluate(source_decs, event_decs, alpha, beta)
        second = mkpdf.evaluate(source_decs, event_decs, alpha, beta, mask=first)
        assert_allclose(first.toarray(), second.toarray(), rtol=1e-12)
