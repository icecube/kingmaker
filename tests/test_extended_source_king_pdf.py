"""
Unit tests for ExtendedSourceKingPDF.

Covers initialization (composition, not inheritance), pdf() correctness
(normalization, boundary behaviour, shape), and evaluate() correctness
(sparse output, geometry screening, mask reuse).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_array

from kingmaker.pdf import ExtendedSourceKingPDF, KingPDF


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


# Minimal grid shared by most pdf() tests; built once per module.
@pytest.fixture(scope="module")
def ext_pdf():
    return ExtendedSourceKingPDF(
        points_alpha=np.radians(np.logspace(-1, 1, 10)),
        points_beta=np.logspace(np.log10(1.01), 1, 8),
        points_extension=np.radians(np.logspace(-1.5, np.log10(4.9), 8)),
        points_psi=np.concatenate([[0.0], np.logspace(-4, np.log10(np.pi), 200)]),
        n_quad=16,
    )


# Fixture for evaluate() tests: small angular_cutoff so far events are screened.
@pytest.fixture(scope="module")
def ext_eval():
    return ExtendedSourceKingPDF(
        angular_cutoff=np.radians(10.0),
        points_alpha=np.radians(np.logspace(-1, 1, 10)),
        points_beta=np.logspace(np.log10(1.01), 1, 8),
        points_extension=np.radians(np.logspace(-1.5, np.log10(4.9), 8)),
        points_psi=np.concatenate([[0.0], np.logspace(-4, np.log10(np.pi), 200)]),
        n_quad=16,
    )


PARAM_CASES = [
    pytest.param(np.radians(0.5), 2.0, np.radians(0.5), id="narrow-moderate-small-ext"),
    pytest.param(np.radians(1.0), 2.5, np.radians(1.0), id="moderate-moderate-med-ext"),
    pytest.param(np.radians(2.0), 4.0, np.radians(1.5), id="wide-heavy-med-ext"),
]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestExtendedSourceKingPDFInit:
    def test_not_instance_of_king_pdf(self, ext_pdf):
        assert not isinstance(ext_pdf, KingPDF)

    def test_default_angular_cutoff(self, ext_pdf):
        assert ext_pdf.angular_cutoff == pytest.approx(np.pi)

    def test_custom_angular_cutoff(self):
        cutoff = np.radians(5.0)
        ext = ExtendedSourceKingPDF(
            angular_cutoff=cutoff,
            points_alpha=np.radians([0.5, 1.0]),
            points_beta=np.array([1.5, 3.0]),
            points_extension=np.radians([0.5, 1.0]),
            n_quad=4,
        )
        assert ext.angular_cutoff == pytest.approx(cutoff)

    def test_default_maximum_sigma(self, ext_pdf):
        assert ext_pdf.maximum_sigma == pytest.approx(3.0)

    def test_custom_maximum_sigma(self):
        ext = ExtendedSourceKingPDF(
            maximum_sigma=5.0,
            points_alpha=np.radians([0.5, 1.0]),
            points_beta=np.array([1.5, 3.0]),
            points_extension=np.radians([0.5, 1.0]),
            n_quad=4,
        )
        assert ext.maximum_sigma == pytest.approx(5.0)

    def test_table_shape(self, ext_pdf):
        expected = (
            len(ext_pdf._log10_points_alpha),
            len(ext_pdf._log10_points_beta),
            len(ext_pdf._log10_points_extension),
            len(ext_pdf._points_psi),
        )
        assert ext_pdf._table.shape == expected

    def test_table_finite(self, ext_pdf):
        assert np.all(np.isfinite(ext_pdf._table))

    def test_table_nonneg(self, ext_pdf):
        assert np.all(ext_pdf._table >= 0.0)


# ---------------------------------------------------------------------------
# pdf()
# ---------------------------------------------------------------------------


class TestExtendedSourceKingPDFPdf:
    @pytest.mark.parametrize("alpha, beta, extension", PARAM_CASES)
    def test_pdf_valid(self, ext_pdf, alpha, beta, extension):
        psi = np.linspace(0, np.radians(5), 50)
        vals = ext_pdf.pdf(psi, np.full_like(psi, alpha), np.full_like(psi, beta), extension)
        assert np.all(vals >= 0)
        assert np.all(np.isfinite(vals))

    def test_zero_at_psi_zero(self, ext_pdf):
        val = ext_pdf.pdf(0.0, np.radians(1.0), 2.5, np.radians(1.0))
        assert val == 0.0

    def test_zero_beyond_psi_max(self):
        """Angles above _points_psi[-1] (but still ≤ π) must return 0.

        Use angular_cutoff=10° without a custom points_psi so the default
        upper bound is max_sigma*max_ext + angular_cutoff ≈ 16° < π, leaving
        room for test points on the sphere that are out-of-table.
        """
        small = ExtendedSourceKingPDF(
            angular_cutoff=np.radians(10.0),
            points_alpha=np.radians([0.5, 1.0, 2.0]),
            points_beta=np.array([1.5, 2.5, 5.0]),
            points_extension=np.radians([0.5, 1.0, 2.0]),
            n_quad=4,
        )
        psi_max = small._points_psi[-1]
        assert psi_max < np.pi, "fixture must end before π for this test to be meaningful"
        beyond = np.array([psi_max + np.radians(5.0), psi_max + np.radians(20.0)])
        beyond = beyond[beyond <= np.pi]
        alpha = np.full(len(beyond), np.radians(1.0))
        beta = np.full(len(beyond), 2.5)
        ext = np.full(len(beyond), np.radians(1.0))
        assert np.all(small.pdf(beyond, alpha, beta, ext) == 0.0)

    def test_output_shape_array(self, ext_pdf):
        psi = np.linspace(0.01, np.radians(5), 20)
        vals = ext_pdf.pdf(psi, np.radians(1.0), 2.5, np.radians(1.0))
        assert vals.shape == psi.shape

    def test_scalar_input_finite(self, ext_pdf):
        val = ext_pdf.pdf(np.radians(1.0), np.radians(1.0), 2.5, np.radians(1.0))
        assert np.isfinite(val)

    def test_oob_alpha_raises(self, ext_pdf):
        with pytest.raises(ValueError):
            ext_pdf.pdf(np.radians(1.0), np.radians(0.001), 2.5, np.radians(1.0))

    def test_oob_extension_raises(self, ext_pdf):
        with pytest.raises(ValueError):
            ext_pdf.pdf(np.radians(1.0), np.radians(1.0), 2.5, np.radians(10.0))

    @pytest.mark.parametrize("alpha, beta, extension", PARAM_CASES)
    def test_normalization(self, ext_pdf, alpha, beta, extension):
        """∫ pdf(ψ) 2π ψ dψ ≈ 1 (flat-sky)."""
        psi = np.linspace(1e-4, ext_pdf._points_psi[-1], 30_000)
        dpsi = psi[1] - psi[0]
        vals = ext_pdf.pdf(
            psi,
            np.full_like(psi, alpha),
            np.full_like(psi, beta),
            np.full_like(psi, extension),
        )
        integral = np.sum(vals * 2.0 * np.pi * psi) * dpsi
        assert_allclose(integral, 1.0, rtol=0.02)

    @pytest.mark.parametrize("alpha, beta, extension", PARAM_CASES)
    def test_small_extension_approaches_king(self, ext_pdf, alpha, beta, extension):
        """Convolved PDF with the smallest grid extension should be close to flat-sky King."""
        tiny_ext = ext_pdf._points_extension[0]
        psi = np.radians([0.5, 1.0, 2.0])
        psi = psi[psi < alpha * 3]  # stay in the PSF core where flat-sky is accurate
        if len(psi) == 0:
            pytest.skip("no test angles within PSF core for this alpha")

        flat_norm = (beta - 1.0) / (2.0 * np.pi * beta * alpha**2)
        flat_king = flat_norm * (1.0 + psi**2 / (2.0 * beta * alpha**2)) ** (-beta)
        conv = ext_pdf.pdf(
            psi,
            np.full_like(psi, alpha),
            np.full_like(psi, beta),
            np.full_like(psi, tiny_ext),
        )
        assert_allclose(conv, flat_king, rtol=0.15)

    def test_larger_extension_broader(self, ext_pdf):
        """Larger extension shifts probability outward, reducing the PDF near psi=0."""
        alpha = np.radians(1.0)
        beta = 2.5
        psi_near = np.radians(0.1)
        val_small = ext_pdf.pdf(psi_near, alpha, beta, ext_pdf._points_extension[0])
        val_large = ext_pdf.pdf(psi_near, alpha, beta, ext_pdf._points_extension[-1])
        assert val_small > val_large


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestExtendedSourceKingPDFEvaluate:
    def test_returns_csr_array(self, ext_eval):
        result = ext_eval.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            np.array([2.5]),
        )
        assert isinstance(result, csr_array)

    def test_output_shape(self, ext_eval):
        src_ras = np.radians([0.0, 45.0])
        src_decs = np.radians([0.0, 10.0])
        src_exts = np.radians([1.0, 1.0])
        ev_ras = np.radians(np.linspace(0, 5, 8))
        ev_decs = np.zeros(8)
        alpha = np.full(8, np.radians(1.0))
        beta = np.full(8, 2.5)
        result = ext_eval.evaluate(src_ras, src_decs, src_exts, ev_ras, ev_decs, alpha, beta)
        assert result.shape == (8, 2)

    def test_nonneg(self, ext_eval):
        rng = np.random.default_rng(0)
        ev_ras = rng.uniform(0, 2 * np.pi, 30)
        ev_decs = np.arcsin(rng.uniform(-1, 1, 30))
        alpha = np.full(30, np.radians(1.0))
        beta = np.full(30, 2.5)
        result = ext_eval.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            ev_ras,
            ev_decs,
            alpha,
            beta,
        )
        assert np.all(result.toarray() >= 0)

    def test_near_source_positive(self, ext_eval):
        """Events close to a source should get a positive PDF value."""
        result = ext_eval.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            np.array([np.radians(0.1)]),
            np.array([0.0]),
            np.array([np.radians(1.0)]),
            np.array([2.5]),
        )
        assert result.toarray()[0, 0] > 0

    def test_zero_beyond_search_radius(self, ext_eval):
        """Events beyond maximum_sigma * ext + angular_cutoff should be zero."""
        src_ext = np.radians(1.0)
        radius = ext_eval.maximum_sigma * src_ext + ext_eval.angular_cutoff
        # Place one event just inside and one well outside
        psi_far = min(radius + np.radians(5.0), np.pi)
        result = ext_eval.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([src_ext]),
            np.array([np.radians(0.5), psi_far]),
            np.array([0.0, 0.0]),
            np.array([np.radians(1.0), np.radians(1.0)]),
            np.array([2.5, 2.5]),
        ).toarray()
        assert result[0, 0] > 0
        assert result[1, 0] == 0.0

    def test_mask_gives_same_result(self, ext_eval):
        rng = np.random.default_rng(42)
        src_ras = np.radians([0.0, 45.0])
        src_decs = np.radians([0.0, 10.0])
        src_exts = np.radians([1.0, 2.0])
        ev_ras = rng.uniform(0, 2 * np.pi, 30)
        ev_decs = np.arcsin(rng.uniform(-1, 1, 30))
        alpha = np.full(30, np.radians(1.0))
        beta = np.full(30, 2.5)
        first = ext_eval.evaluate(src_ras, src_decs, src_exts, ev_ras, ev_decs, alpha, beta)
        second = ext_eval.evaluate(
            src_ras, src_decs, src_exts, ev_ras, ev_decs, alpha, beta, mask=first
        )
        assert_allclose(first.toarray(), second.toarray(), rtol=1e-12)

    def test_two_sources_prefer_nearest(self, ext_eval):
        """An event near source 0 should get a higher PDF for source 0 than source 1."""
        src_ras = np.radians([0.0, 90.0])
        src_decs = np.radians([0.0, 0.0])
        src_exts = np.radians([1.0, 1.0])
        ev_ras = np.radians([1.0])
        ev_decs = np.radians([0.0])
        alpha = np.array([np.radians(1.0)])
        beta = np.array([2.5])
        result = ext_eval.evaluate(
            src_ras, src_decs, src_exts, ev_ras, ev_decs, alpha, beta
        ).toarray()
        assert result[0, 0] > result[0, 1]
