"""
Unit tests for TemplateSmearedKingPDF.

Uses nside=32 (lmax=95, npix=12288) and small alpha/beta grids throughout
so that precompute_bl_grid is fast. Module-scoped fixtures avoid re-running
the expensive initialization for every test.
"""

import healpy as hp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from kingmaker.pdf import TemplateSmearedKingPDF

# ---------------------------------------------------------------------------
# Constants and shared parameters
# ---------------------------------------------------------------------------

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
LMAX = 3 * NSIDE - 1  # 95

# Small alpha/beta grids to keep precompute_bl_grid fast in tests.
# Start alpha at 1e-2 rad (~0.57 deg): values below ~theta_grid[1]=1e-4 cause the
# trapezoid rule to miss the PSF core in the first bin, so b_0 deviates from 1.
POINTS_ALPHA = np.logspace(-2, np.log10(np.pi) + 1e-2, 20)
POINTS_BETA = np.logspace(0, 1, 20) + 1e-5

# A broad PSF that is well-resolved at nside=32 (~1.8 deg/pixel)
ALPHA = np.radians(5.0)
BETA = 2.0


# ---------------------------------------------------------------------------
# Map fixtures (module scope: built once, shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uniform_map():
    return np.ones(NPIX, dtype=np.float64)


@pytest.fixture(scope="module")
def point_source_map():
    """Single bright pixel near the equator at RA=0."""
    m = np.zeros(NPIX, dtype=np.float64)
    m[hp.ang2pix(NSIDE, np.pi / 2, 0.0)] = 1.0
    return m


@pytest.fixture(scope="module")
def pdf_uniform(uniform_map):
    return TemplateSmearedKingPDF(
        uniform_map,
        points_alpha=POINTS_ALPHA,
        points_beta=POINTS_BETA,
    )


@pytest.fixture(scope="module")
def pdf_point(point_source_map):
    return TemplateSmearedKingPDF(
        point_source_map,
        points_alpha=POINTS_ALPHA,
        points_beta=POINTS_BETA,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFInit:
    def test_nside(self, pdf_uniform):
        assert pdf_uniform.nside == NSIDE

    def test_lmax_default(self, pdf_uniform):
        assert pdf_uniform.lmax == LMAX

    def test_custom_lmax(self, uniform_map):
        pdf = TemplateSmearedKingPDF(
            uniform_map, lmax=10, points_alpha=POINTS_ALPHA, points_beta=POINTS_BETA
        )
        assert pdf.lmax == 10

    def test_skymap_stored(self, pdf_uniform, uniform_map):
        assert_allclose(pdf_uniform.skymap, uniform_map)

    def test_skymap_alm_length(self, pdf_uniform):
        expected = hp.Alm.getsize(LMAX)
        assert len(pdf_uniform.skymap_alm) == expected

    def test_interpolation_method_stored(self, pdf_uniform):
        assert pdf_uniform.interpolation_method == "nearest"

    def test_invalid_interpolation_method_raises(self, uniform_map):
        with pytest.raises(ValueError, match="interpolation_method"):
            TemplateSmearedKingPDF(
                uniform_map,
                interpolation_method="cubic",
                points_alpha=POINTS_ALPHA,
                points_beta=POINTS_BETA,
            )

    def test_memory_limit_stored(self, uniform_map):
        pdf = TemplateSmearedKingPDF(
            uniform_map,
            memory_limit_gb=0.5,
            points_alpha=POINTS_ALPHA,
            points_beta=POINTS_BETA,
        )
        assert pdf.memory_limit_bytes == int(0.5e9)

    def test_bl_grid_shape(self, pdf_uniform):
        assert pdf_uniform.bl_grid.shape == (len(POINTS_ALPHA), len(POINTS_BETA), LMAX + 1)

    def test_bl_grid_finite(self, pdf_uniform):
        assert np.all(np.isfinite(pdf_uniform.bl_grid))

    def test_eval_coords_empty_by_default(self, pdf_uniform):
        assert pdf_uniform.eval_decs.size == 0
        assert pdf_uniform.eval_ras.size == 0

    def test_coords_set_at_init(self, uniform_map):
        decs = np.array([0.0, 0.1])
        ras = np.array([0.0, 0.2])
        pdf = TemplateSmearedKingPDF(
            uniform_map,
            eval_decs=decs,
            eval_ras=ras,
            points_alpha=POINTS_ALPHA,
            points_beta=POINTS_BETA,
        )
        assert_allclose(pdf.eval_decs, decs)
        assert_allclose(pdf.eval_ras, ras)


# ---------------------------------------------------------------------------
# get_king_b_l
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFBl:
    def test_bl_length(self, pdf_uniform):
        b_l = pdf_uniform.get_king_b_l(ALPHA, BETA)
        assert len(b_l) == LMAX + 1

    def test_bl_finite(self, pdf_uniform):
        b_l = pdf_uniform.get_king_b_l(ALPHA, BETA)
        assert np.all(np.isfinite(b_l))

    def test_bl0_is_one(self, pdf_uniform):
        """b_0 = 2π ∫ K(θ) P_0(cosθ) sinθ dθ = 1 for any normalized PSF."""
        b_l = pdf_uniform.get_king_b_l(ALPHA, BETA)
        assert_allclose(b_l[0], 1.0, rtol=1e-3)

    def test_bl_decreasing_for_broad_psf(self, pdf_uniform):
        """A broad PSF suppresses high-l modes: b_l should decrease with l."""
        b_l = pdf_uniform.get_king_b_l(ALPHA, BETA)
        # b_l values shouldn't increase over the bulk of the spectrum
        assert b_l[-1] < b_l[0]

    def test_bl_nonnegative(self, pdf_uniform):
        b_l = pdf_uniform.get_king_b_l(ALPHA, BETA)
        assert np.all(b_l >= -1e-6)

    def test_nearest_and_linear_agree_near_grid_center(self, uniform_map):
        """At a grid point both methods should return the same b_l."""
        pdf_n = TemplateSmearedKingPDF(
            uniform_map,
            interpolation_method="nearest",
            points_alpha=POINTS_ALPHA,
            points_beta=POINTS_BETA,
        )
        pdf_l = TemplateSmearedKingPDF(
            uniform_map,
            interpolation_method="linear",
            points_alpha=POINTS_ALPHA,
            points_beta=POINTS_BETA,
        )
        alpha_center = POINTS_ALPHA[len(POINTS_ALPHA) // 2]
        beta_center = POINTS_BETA[len(POINTS_BETA) // 2]
        assert_allclose(
            pdf_n.get_king_b_l(alpha_center, beta_center),
            pdf_l.get_king_b_l(alpha_center, beta_center),
            rtol=1e-2,
        )


# ---------------------------------------------------------------------------
# precompute_bl_grid
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFBlGrid:
    def test_shape(self, pdf_uniform):
        n_a, n_b = len(POINTS_ALPHA), len(POINTS_BETA)
        assert pdf_uniform.bl_grid.shape == (n_a, n_b, LMAX + 1)

    def test_monopole_column_is_one(self, pdf_uniform):
        """Every (alpha, beta) grid point should have b_0 ≈ 1."""
        assert_allclose(pdf_uniform.bl_grid[:, :, 0], 1.0, rtol=1e-3)

    def test_consistent_with_get_king_b_l(self, pdf_uniform):
        """bl_grid lookup should match direct get_king_b_l call."""
        alpha_center = POINTS_ALPHA[len(POINTS_ALPHA) // 2]
        beta_center = POINTS_BETA[len(POINTS_BETA) // 2]
        direct = pdf_uniform.get_king_b_l(alpha_center, beta_center)
        i = len(POINTS_ALPHA) // 2
        j = len(POINTS_BETA) // 2
        assert_allclose(direct, pdf_uniform.bl_grid[i, j], rtol=1e-10)


# ---------------------------------------------------------------------------
# set_coordinates
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFSetCoordinates:
    def test_cl_shape(self, pdf_uniform):
        decs = np.radians(np.array([0.0, 10.0, -10.0]))
        ras = np.radians(np.array([0.0, 30.0, 60.0]))
        pdf_uniform.set_coordinates(decs, ras)
        assert pdf_uniform._c_l.shape == (LMAX + 1, 3)

    def test_mismatched_shapes_raise(self, pdf_uniform):
        with pytest.raises(RuntimeError, match="declination"):
            pdf_uniform.set_coordinates(
                np.array([0.0, 0.1]),
                np.array([0.0]),
            )

    def test_idempotent_second_call(self, pdf_uniform):
        decs = np.radians(np.array([5.0, -5.0]))
        ras = np.radians(np.array([10.0, 20.0]))
        pdf_uniform.set_coordinates(decs, ras)
        c_l_first = pdf_uniform._c_l.copy()
        pdf_uniform.set_coordinates(decs, ras)  # should early-return
        assert_allclose(pdf_uniform._c_l, c_l_first)


# ---------------------------------------------------------------------------
# convolve_map
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFConvolveMap:
    def test_output_length(self, pdf_uniform):
        result = pdf_uniform.convolve_map(ALPHA, BETA)
        assert len(result) == NPIX

    def test_output_finite(self, pdf_uniform):
        result = pdf_uniform.convolve_map(ALPHA, BETA)
        assert np.all(np.isfinite(result))

    def test_uniform_map_stays_uniform(self, pdf_uniform):
        """A uniform map has only the l=0 mode; convolution leaves it unchanged.
        Tolerance is 0.3% to account for polar ringing from iter=1 in map2alm."""
        result = pdf_uniform.convolve_map(ALPHA, BETA)
        assert_allclose(result, result.mean(), rtol=3e-3)

    def test_total_flux_preserved(self, pdf_point):
        """Convolution preserves total solid-angle-weighted flux (b_0 = 1).
        The skymap is normalised to integrate to 1 sr^-1, so sum(pixels)*pixel_area = 1."""
        pixel_area = hp.nside2pixarea(NSIDE)
        convolved_flux = pdf_point.convolve_map(ALPHA, BETA).sum() * pixel_area
        assert_allclose(convolved_flux, 1.0, rtol=1e-2)

    def test_point_source_broadens(self, pdf_point):
        """Convolving a point source with a PSF should increase the number of nonzero pixels."""
        n_nonzero_before = np.sum(pdf_point.skymap > 0)
        n_nonzero_after = np.sum(pdf_point.convolve_map(ALPHA, BETA) > 0)
        assert n_nonzero_after > n_nonzero_before


# ---------------------------------------------------------------------------
# convolve_at_grid_point
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFConvolveAtGridPoint:
    def test_output_shape(self, pdf_uniform):
        decs = np.radians(np.array([0.0, 10.0, 20.0]))
        ras = np.radians(np.array([0.0, 30.0, 60.0]))
        pdf_uniform.set_coordinates(decs, ras)
        result = pdf_uniform.convolve_at_grid_point(ALPHA, BETA)
        assert result.shape == (3,)

    def test_matches_convolve_map_at_pixel_centers(self, pdf_point):
        """convolve_at_grid_point should agree with convolve_map at the same locations."""
        # Pick a handful of pixel centers to evaluate at
        test_pixels = np.array([0, 100, 500, 1000, 5000])
        colatitudes, longitudes = hp.pix2ang(NSIDE, test_pixels)
        decs = np.pi / 2 - colatitudes
        ras = longitudes

        pdf_point.set_coordinates(decs, ras)
        at_points = pdf_point.convolve_at_grid_point(ALPHA, BETA)
        full_map = pdf_point.convolve_map(ALPHA, BETA)

        assert_allclose(at_points, full_map[test_pixels], rtol=1e-4)

    def test_scalar_coords_work(self, pdf_uniform):
        pdf_uniform.set_coordinates(0.0, 0.0)
        result = pdf_uniform.convolve_at_grid_point(ALPHA, BETA)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


class TestTemplateSmearedKingPDFSample:
    def test_returns_two_arrays(self, pdf_uniform):
        ra, dec = pdf_uniform.sample(10, ALPHA, BETA)
        assert ra is not None and dec is not None

    @pytest.mark.parametrize("n", [1, 50, 200])
    def test_output_length(self, pdf_uniform, n):
        ra, dec = pdf_uniform.sample(n, ALPHA, BETA)
        assert len(ra) == n
        assert len(dec) == n

    def test_ra_in_range(self, pdf_uniform):
        ra, _ = pdf_uniform.sample(200, ALPHA, BETA)
        assert np.all(ra >= 0) and np.all(ra <= 2 * np.pi)

    def test_dec_in_range(self, pdf_uniform):
        _, dec = pdf_uniform.sample(200, ALPHA, BETA)
        assert np.all(dec >= -np.pi / 2) and np.all(dec <= np.pi / 2)

    def test_reproducible_with_rng(self, pdf_uniform):
        ra1, dec1 = pdf_uniform.sample(50, ALPHA, BETA, rng=np.random.default_rng(0))
        ra2, dec2 = pdf_uniform.sample(50, ALPHA, BETA, rng=np.random.default_rng(0))
        assert_allclose(ra1, ra2)
        assert_allclose(dec1, dec2)
