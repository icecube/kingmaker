from typing import Optional, Tuple, Union, cast
import numpy as np
import numpy.typing as npt
import healpy as hp
from scipy.interpolate import interpn
from scipy.special import legendre_p_all, sph_harm_y_all

from .distribution import _log10pi
from .distribution import _norm, _unnormalized_pdf, _unnormalized_cdf
from .utils import angular_distance, meshgrid2d


class KingPDF:
    """
    Calculate the probability density function (PDF) and cumulative distribution
    function (CDF) for the King spatial distribution.

    This class manages PDF and CDF evaluations with support for angular cutoffs
    and proper normalization over the sphere.

    Parameters
    ----------
    angular_cutoff : float, optional
        Maximum angular separation in radians. Default is pi (full sphere).
    """

    def __init__(self, *, angular_cutoff: float = np.pi) -> None:
        self.angular_cutoff = angular_cutoff

    def norm(
        self,
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        """
        Compute the normalization constant for given King parameters.

        Parameters
        ----------
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        float or ndarray
            Normalization constant(s) such that PDF integrates to 1.
        """
        return _norm(alpha, beta, self.angular_cutoff)

    def pdf_from_norm(
        self,
        x: Union[float, npt.NDArray[np.floating]],
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
        norm: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        """
        Evaluate the King kernel given a precomputed normalization constant.

        Equivalent to ``norm * unnormalized_pdf(x, alpha, beta)``, skipping
        the alpha/beta validation, angular-cutoff masking, and norm
        computation that :meth:`pdf` performs. Intended for callers that
        have already validated alpha/beta (e.g. against a previously-fit
        parameter grid) and already filtered ``x`` to be within
        ``self.angular_cutoff``, and that can compute norm once over a
        smaller deduplicated (alpha, beta) grid than ``len(x)`` -- e.g.
        :meth:`kingmaker.wrapper.KingSpatialLikelihood.set_events`.

        Parameters
        ----------
        x : float or ndarray
            Angular separation(s) in radians. Must already be
            ``<= self.angular_cutoff``; this is NOT checked.
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians. Must
            already be > 0; this is NOT checked.
        beta : float or ndarray
            King distribution beta parameter (tail weight). Must already
            be > 1; this is NOT checked.
        norm : float or ndarray
            Precomputed normalization constant(s), e.g. from :meth:`norm`.

        Returns
        -------
        float or ndarray
            ``norm * unnormalized_pdf(x, alpha, beta)``, broadcast over
            inputs.
        """
        return norm * _unnormalized_pdf(x, alpha, beta)

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.floating]],
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        """
        Evaluate the normalized King PDF at given angular separation(s).

        Returns zero for points beyond the angular cutoff. Handles broadcasting
        of input arrays and masks invalid regions.

        Parameters
        ----------
        x : float or ndarray
            Angular separation(s) from the source in radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Normalized PDF value(s) with units of probability/steradian.
        """
        # Scalar-like: check if we can shortcut using the angular cutoff.
        if np.isscalar(x) and (x > self.angular_cutoff):  # type: ignore[operator]
            return 0
        elif isinstance(x, np.ndarray) and x.size == 1:
            if float(x.flat[0]) > self.angular_cutoff:
                return 0

        if np.any(alpha <= 0):
            raise ValueError("Received alpha <= 0. The King distribution is not defined here.")
        if np.any(beta <= 1):
            raise ValueError("Received beta <= 1. The King distribution is not defined here.")

        # Broadcast
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

        # And mask
        normalized_pdf = np.zeros_like(x)
        mask = x <= self.angular_cutoff
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_pdf = _unnormalized_pdf(x, alpha, beta)
        normalized_pdf[mask] = norm * unnormalized_pdf

        return normalized_pdf

    def cdf(
        self,
        x: Union[float, npt.NDArray[np.floating]],
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        """
        Evaluate the normalized King CDF at given angular separation(s).

        Returns 1 for points beyond the angular cutoff. Handles broadcasting
        of input arrays and masks invalid regions.

        Parameters
        ----------
        x : float or ndarray
            Angular separation(s) from the source in radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Normalized CDF value(s) (cumulative probability).
        """
        # Scalar-like: check if we can shortcut using the angular cutoff.
        if np.isscalar(x):
            if x > self.angular_cutoff:  # type: ignore[operator]
                return 1
        elif isinstance(x, np.ndarray) and x.size == 1:
            if float(x.flat[0]) > self.angular_cutoff:
                return 1

        if np.any(alpha <= 0):
            raise ValueError(
                "Received alpha <= 0. The King distribution is onlydefined for alpha > 0."
            )
        if np.any(beta <= 1):
            raise ValueError(
                "Received beta <= 1. The King distribution is onlydefined for beta > 1."
            )

        # Broadcast
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

        # And mask
        normalized_cdf = np.ones_like(x)
        mask = x <= self.angular_cutoff
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_cdf = _unnormalized_cdf(x, alpha, beta)
        normalized_cdf[mask] = norm * unnormalized_cdf

        return normalized_cdf

    def marginalize(
        self,
        dec: float,
        alpha: float,
        beta: float,
        threshold: float = 1e-6,
        nbins: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Marginalize the King PDF over right ascension to obtain declination profile.

        Integrates the 2D King PDF over RA using adaptive binning to produce a
        1D profile in sin(declination). Only bins with PDF above threshold are computed.

        Parameters
        ----------
        dec : float
            Source declination in radians.
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        threshold : float, optional
            Minimum relative PDF value for bin inclusion. Default is 1e-6.
        nbins : int, optional
            Number of bins in sin(declination). If None, uses adaptive binning.

        Returns
        -------
        sindec_bins : ndarray
            Bin edges in sin(declination).
        marginalized : ndarray
            Marginalized PDF values at each bin.
        """
        if np.any(alpha <= 0):
            raise ValueError("Received alpha <= 0. The PDF is not defined here.")
        if np.any(beta <= 1):
            raise ValueError("Received beta <= 1. The PDF is not defined here.")

        # Generate the sindec binning
        sindec = np.sin(dec)
        if nbins is None:
            delta = min(0.025, (np.sin(dec + alpha) - np.sin(dec - alpha)) / 2)
            sindec_bins = np.concatenate(
                [np.arange(-1, sindec, delta), np.arange(sindec, 1, delta)]
            )
        else:
            sindec_bins = np.concatenate([[sindec, 1], np.linspace(-1, 1, nbins)])
            sindec_bins = np.unique(sindec_bins)

        # Mask unnecessary declination bins
        pdf_check = cast(
            npt.NDArray[np.floating], self.pdf(np.abs(dec - np.arcsin(sindec_bins)), alpha, beta)
        )
        decmask = pdf_check / pdf_check.max() > threshold

        # Generate a binning in RA from 0 to 2pi.
        nbins = int(2 * np.pi / alpha * 5)
        ra_bins = np.linspace(0, 2 * np.pi, nbins)

        # Mask unnecessary RA bins
        pdf_check = cast(npt.NDArray[np.floating], self.pdf(ra_bins, alpha, beta))
        ramask = pdf_check / pdf_check.max() > threshold
        ra_bins = ra_bins[ramask]

        # Calculate the PDF at each grid point
        ra_points, dec_points = meshgrid2d(ra_bins, np.arcsin(sindec_bins[decmask]))
        distance = angular_distance(0, dec, ra_points, dec_points)
        pdf_vals = cast(npt.NDArray[np.floating], self.pdf(distance, alpha, beta))

        # Integrate over RA using a simple trapezoid rule integration.
        # This should probably be checked for correct units. For example,
        # should this include a np.diff(sindec_bins) in the denominator?
        # I don't think it should, but someone should help check this.
        marginalized = np.zeros_like(sindec_bins)
        marginalized[decmask] = (
            (pdf_vals[:, :-1] + pdf_vals[:, 1:]) / 2 * np.diff(ra_bins)[None, :]
        ).sum(axis=1)
        return sindec_bins, marginalized

    def sample(
        self,
        n: int,
        alpha: float,
        beta: float,
        rng: Optional[np.random.Generator] = None,
        n_grid: int = 10000,
    ) -> npt.NDArray[np.floating]:
        """
        Sample angular separations from the King distribution via inverse CDF.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng().
        n_grid : int, optional
            Number of points in the CDF lookup grid. Higher values give more
            accurate sampling at the cost of memory and setup time. Default
            is 10000, which gives ~arcminute accuracy.

        Returns
        -------
        ndarray
            Angular separations in radians, shape (n,).
        """

        if np.any(alpha <= 0):
            raise ValueError("Received alpha <= 0. The PDF is not defined here.")
        if np.any(beta <= 1):
            raise ValueError("Received beta <= 1. The PDF is not defined here.")

        if rng is None:
            rng = np.random.default_rng()
        psi_grid = np.linspace(1e-6, self.angular_cutoff, n_grid)
        cdf_grid = cast(npt.NDArray[np.floating], self.cdf(psi_grid, alpha, beta))
        return np.interp(rng.uniform(0, cdf_grid[-1], n), cdf_grid, psi_grid)


class TemplateSmearedKingPDF(KingPDF):
    """
    King PDF convolved with a HEALPix template map using spherical harmonics.

    Uses spherical harmonic decomposition to efficiently convolve a King PSF
    with a template skymap. Pre-computes template harmonics for fast evaluation.

    Parameters
    ----------
    skymap : ndarray
        HEALPix map to convolve with King PDF. Will be normalized to integrate to 1.
    eval_decs : float or ndarray
        Declination(s) in radians where PDF will be evaluated.
    eval_ras : float or ndarray
        Right ascension(s) in radians where PDF will be evaluated.
    angular_cutoff : float, optional
        Maximum angular separation in radians. Default is pi.
    points_alpha : ndarray, optional
        Grid of alpha values for normalization interpolation.
    points_beta : ndarray, optional
        Grid of beta values for normalization interpolation.
    lmax : int, optional
        Maximum spherical harmonic degree. Default is 3*nside-1.
    interpolation_method : {"nearest", "linear"}, optional
        Method used in get_king_b_l to look up b_l coefficients from the
        precomputed grid. "nearest" (default) snaps to the closest grid point,
        returning one unique b_l vector per distinct grid cell limiting the
        number of maps generated and improving efficiency. The "linear" option
        bilinearly interpolates in log(alpha), log(beta) space.
    memory_limit_gb : float, optional
        Memory budget in GB for the sph_harm_y_all array allocated in
        set_coordinates. Points are processed in batches sized so that each
        batch stays within this limit. Default is 1.0 GB. At nside=256
        (lmax=767) each point costs ~9.4 MB, so the default allows ~100 points
        per batch; at nside=512 (lmax=1535) each point costs ~37.7 MB, so the
        default allows ~26 points per batch.
    """

    skymap: npt.NDArray[np.floating]
    bl_grid: npt.NDArray[np.floating]
    interpolation_method: str
    eval_decs: npt.NDArray[np.floating] = np.array([], dtype=np.float32)
    eval_ras: npt.NDArray[np.floating] = np.array([], dtype=np.float32)

    def __init__(
        self,
        skymap: npt.NDArray[np.floating],
        *,
        eval_decs: Optional[Union[float, npt.NDArray[np.floating]]] = None,
        eval_ras: Optional[Union[float, npt.NDArray[np.floating]]] = None,
        angular_cutoff: float = np.pi,
        points_alpha: npt.NDArray[np.floating] = np.logspace(-4, _log10pi + 1e-2, 100),
        points_beta: npt.NDArray[np.floating] = np.nextafter(np.logspace(0, 1, 100), np.inf),
        lmax: Optional[int] = None,
        interpolation_method: str = "nearest",
        memory_limit_gb: float = 1.0,
    ) -> None:
        if interpolation_method not in ("nearest", "linear"):
            raise ValueError(
                f"interpolation_method must be 'nearest' or 'linear', got {interpolation_method!r}"
            )
        self.interpolation_method = interpolation_method
        self.memory_limit_bytes = int(memory_limit_gb * 1e9)

        super().__init__(angular_cutoff=angular_cutoff)

        if np.any(points_alpha <= 0):
            raise ValueError(
                "Received points_alpha containing at least one point <= 0. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 0.0 when passing them into TemplateSmearedKingPDF."
            )
        if np.any(points_beta <= 1):
            raise ValueError(
                "Received points_beta containing at least one point <= 1. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 1.0 when passing them into TemplateSmearedKingPDF."
            )
        self.points_alpha = points_alpha
        self.points_beta = points_beta
        self.log10_points_alpha = np.log10(self.points_alpha)
        self.log10_points_beta = np.log10(self.points_beta)

        self.nside = hp.npix2nside(len(skymap))
        self.lmax = (3 * self.nside - 1) if lmax is None else lmax
        self.mmax = self.lmax

        # While we're here, normalize the skymap so it integrates to 1.
        # Then calculate the alm coefficients needed for convolution.
        self.skymap = skymap
        normalized_skymap = skymap / skymap.sum() / hp.nside2pixarea(self.nside)
        self.skymap_alm = hp.map2alm(normalized_skymap, lmax=self.lmax, mmax=self.mmax, iter=1)

        # Precompute the spherical harmonic indices needed for convolution.
        # hp.Alm.getlm builds (ls, ms) in C, avoiding a Python loop over lmax+1.
        self.ls, self.ms = hp.Alm.getlm(self.lmax)
        self._alm_indices = hp.Alm.getidx(self.lmax, self.ls, self.ms)

        # Precompute weighted alm = factors * a_lm once at init so set_coordinates
        # doesn't recompute them per call. Sort by l so np.add.reduceat can sum
        # contributions per degree without Python-level scatter (np.add.at).
        a_lm_flat = self.skymap_alm[self._alm_indices]
        factors = np.where(self.ms == 0, 1.0, 2.0)
        weighted_alm = factors * a_lm_flat
        sort_idx = np.argsort(self.ls, kind="stable")
        self.ls_sorted = self.ls[sort_idx]
        self.ms_sorted = self.ms[sort_idx]
        self.weighted_alm_sorted = weighted_alm[sort_idx]
        self.l_starts = np.searchsorted(self.ls_sorted, np.arange(self.lmax + 1))

        # Set up the evaluation coordinates
        if eval_decs is not None and eval_ras is not None:
            self.set_coordinates(eval_decs, eval_ras)

        # Pre-generate the Legendre polynomials needed for b_l calculations.
        # The theta grid is log-spaced so it resolves the PSF core accurately
        # down to ~0.0057 degrees (1e-4 rad), well below IceCube's resolution.
        self.theta_grid = np.concatenate(
            [[0.0], np.logspace(-4, np.log10(self.angular_cutoff), 1000)]
        )
        P_all = legendre_p_all(self.lmax, np.cos(self.theta_grid))[0]
        self.P_all = P_all[: self.lmax + 1]

        self.bl_grid = self.precompute_bl_grid()

    def set_coordinates(
        self,
        eval_decs: Union[float, npt.NDArray[np.floating]],
        eval_ras: Union[float, npt.NDArray[np.floating]],
    ) -> None:
        """
        Set evaluation coordinates and pre-compute spherical harmonics.

        Parameters
        ----------
        eval_decs : float or ndarray
            Declination(s) in radians.
        eval_ras : float or ndarray
            Right ascension(s) in radians.
        """
        eval_decs = np.atleast_1d(eval_decs)
        eval_ras = np.atleast_1d(eval_ras)

        # If there's a different number of declination and RA points,
        # there's something wrong.
        if np.atleast_1d(eval_decs).shape != eval_ras.shape:
            raise RuntimeError(
                "TemplateSmearedKingPDF::set_coordinates received different numbers"
                f" of declination values ({np.atleast_1d(eval_decs).shape}) and"
                f" right ascension values ({np.atleast_1d(eval_ras).shape})."
                " These need to match since each is assumed to be one source."
            )

        # If the coordinates match what we already have, do nothing.
        if (eval_decs.size == 0) or (
            np.all(np.equal(self.eval_decs.shape, eval_decs.shape))
            and np.all(np.equal(self.eval_decs, eval_decs))
            and np.all(np.equal(self.eval_ras, eval_ras))
        ):
            return

        self.eval_decs = np.atleast_1d(eval_decs)
        self.eval_ras = np.atleast_1d(eval_ras)
        assert self.eval_decs.shape == self.eval_ras.shape

        npts = len(self.eval_decs)

        # sph_harm_y_all returns shape (lmax+1, 2*mmax+1, batch) in complex128.
        # Batch over points so the raw array stays within memory_limit_bytes.
        bytes_per_point = np.complex128().nbytes * (self.lmax + 1) * (2 * self.mmax + 1)
        batch_size = max(1, self.memory_limit_bytes // bytes_per_point)

        self._c_l = np.zeros((self.lmax + 1, npts), dtype=np.float64)
        for start in range(0, npts, batch_size):
            end = min(start + batch_size, npts)
            raw = sph_harm_y_all(
                self.lmax,
                self.mmax,
                np.pi / 2 - self.eval_decs[start:end],
                self.eval_ras[start:end],
            )
            # Use l-sorted alm order so reduceat can sum contributions per degree
            # with a single contiguous-segment reduction instead of Python scatter.
            Y_lm_sorted = raw[self.ls_sorted, self.ms_sorted, :]  # (nalm, batch)
            contribs = np.real(self.weighted_alm_sorted[:, None] * Y_lm_sorted)  # (nalm, batch)
            self._c_l[:, start:end] = np.add.reduceat(contribs, self.l_starts, axis=0)
        return

    def skymap_to_alm(self) -> npt.NDArray[np.complexfloating]:
        """
        Convert the HEALPix skymap to spherical harmonic coefficients.

        Returns
        -------
        ndarray
            Complex spherical harmonic coefficients a_lm.
        """
        return hp.map2alm(self.skymap, lmax=self.lmax, mmax=self.mmax)

    def precompute_bl_grid(self) -> npt.NDArray[np.floating]:
        """
        Precompute b_l coefficients for all (alpha, beta) grid points via matmul.

        Evaluates the King PDF over the full (n_alpha, n_beta, n_theta) parameter
        grid, then computes all b_l integrals in a single matrix multiply rather
        than calling get_king_b_l once per grid point. Peak memory scales as
        O(n_alpha * n_beta * n_theta), replacing the naive approach that would
        require O(lmax * n_alpha * n_beta * n_theta).

        Returns
        -------
        ndarray, shape (n_alpha, n_beta, lmax + 1)
            Spherical harmonic b_l coefficients for each (alpha, beta) grid point,
            stored in interpn-ready axis order.
        """
        n_alpha = len(self.points_alpha)
        n_beta = len(self.points_beta)
        theta = self.theta_grid
        n_theta = len(theta)

        # Evaluate the King PDF over the full (n_alpha, n_beta, n_theta) grid.
        # Compute normalisation constants once per (alpha, beta) pair and broadcast
        # over theta, avoiding n_theta redundant norm lookups per grid point.
        alpha_grid, beta_grid = np.meshgrid(self.points_alpha, self.points_beta, indexing="ij")
        norms = self.norm(alpha_grid, beta_grid)
        unnorm = _unnormalized_pdf(
            theta[None, None, :], alpha_grid[:, :, None], beta_grid[:, :, None]
        )  # (n_alpha, n_beta, n_theta)
        pdf_vals = norms[:, :, None] * unnorm

        # Trapezoid quadrature weights for the non-uniform log-spaced theta grid.
        w = np.empty(n_theta)
        w[1:-1] = (theta[2:] - theta[:-2]) / 2
        w[0] = (theta[1] - theta[0]) / 2
        w[-1] = (theta[-1] - theta[-2]) / 2

        # Absorb 2pi * sin(theta) * w into P_all once to form P_weighted, then
        # use a single BLAS matmul instead of lmax+1 separate trapezoid calls.
        P_weighted = self.P_all * (2 * np.pi * np.sin(theta) * w)  # (lmax+1, n_theta)
        pdf_flat = pdf_vals.reshape(n_alpha * n_beta, n_theta)
        bl_flat = P_weighted @ pdf_flat.T  # (lmax+1, n_alpha * n_beta)

        # Store as (n_alpha, n_beta, lmax+1) so interpn can use it directly
        # without a transpose on every get_king_b_l call.
        return bl_flat.reshape(self.lmax + 1, n_alpha, n_beta).transpose(1, 2, 0)

    def get_king_b_l(self, alpha: float, beta: float) -> npt.NDArray[np.floating]:
        """
        Return spherical harmonic expansion coefficients b_l for the King PDF.

        Looks up b_l values from the precomputed grid (see precompute_bl_grid)
        using the interpolation method selected at initialization.

        "nearest" snaps to the closest (alpha, beta) grid point in log space,
        so events that fall in the same grid cell reuse the same b_l vector
        without triggering a new map convolution.  "linear" bilinearly
        interpolates in log(alpha), log(beta) space for smoother variation.

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Spherical harmonic coefficients b_l for degrees 0 to lmax.
        """
        if np.any(alpha <= 0):
            raise ValueError(
                "Received alpha containing at least one point <= 0. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 0.0 when passing them into TemplateSmearedKingPDF."
            )
        if np.any(beta <= 1):
            raise ValueError(
                "Received beta containing at least one point <= 1. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 1.0 when passing them into TemplateSmearedKingPDF."
            )
        log_a = np.log10(alpha)
        log_b = np.log10(beta)
        if self.interpolation_method == "nearest":
            i = np.searchsorted(self.log10_points_alpha, log_a)
            j = np.searchsorted(self.log10_points_beta, log_b)
            i = np.clip(i, 1, len(self.log10_points_alpha) - 1)
            j = np.clip(j, 1, len(self.log10_points_beta) - 1)

            # Adjust i and j to snap to the nearest grid point in log space
            # since searchsorted gives index of the right edge of the bin.
            if log_a - self.log10_points_alpha[i - 1] < self.log10_points_alpha[i] - log_a:
                i -= 1
            if log_b - self.log10_points_beta[j - 1] < self.log10_points_beta[j] - log_b:
                j -= 1
            return self.bl_grid[i, j]

        # "linear"
        result = interpn(
            (self.log10_points_alpha, self.log10_points_beta),
            self.bl_grid,  # (n_alpha, n_beta, lmax+1)
            np.array([[log_a, log_b]]),
            method="linear",
            bounds_error=True,
        )
        return result[0]

    def convolve_map(self, alpha: float, beta: float) -> npt.NDArray[np.floating]:
        """
        Convolve the template skymap with a King PSF and return full HEALPix map.

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Convolved HEALPix skymap at the same resolution as input.
        """
        if np.any(alpha <= 0):
            raise ValueError(
                "Received alpha containing at least one point <= 0. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 0.0 when passing them into TemplateSmearedKingPDF."
            )
        if np.any(beta <= 1):
            raise ValueError(
                "Received beta containing at least one point <= 1. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 1.0 when passing them into TemplateSmearedKingPDF."
            )
        b_l = self.get_king_b_l(alpha, beta)
        harmonic_convolution = hp.almxfl(alm=self.skymap_alm, fl=b_l, mmax=self.mmax, inplace=False)
        return hp.alm2map(harmonic_convolution, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def convolve_at_grid_point(
        self, alpha: float, beta: float
    ) -> Union[float, npt.NDArray[np.floating]]:
        """
        Evaluate convolved PDF only at pre-set grid points (eval_decs, eval_ras).

        More efficient than convolve_map() when only specific points are needed.
        Uses pre-computed spherical harmonics from set_coordinates().

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).

        Returns
        -------
        float or ndarray
            Convolved PDF value(s) at evaluation coordinates.
        """
        if np.any(alpha <= 0):
            raise ValueError(
                "Received alpha containing at least one point <= 0. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 0.0 when passing them into TemplateSmearedKingPDF."
            )
        if np.any(beta <= 1):
            raise ValueError(
                "Received beta containing at least one point <= 1. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 1.0 when passing them into TemplateSmearedKingPDF."
            )
        b_l = self.get_king_b_l(alpha, beta)
        return b_l @ self._c_l

    def marginalize(
        self,
        dec: float,
        alpha: float,
        beta: float,
        threshold: float = 1e-6,
        nbins: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        raise NotImplementedError("Signal subtraction for templates is not implemented.")

    def sample(
        self,
        n: int,
        alpha: float,
        beta: float,
        rng: Optional[np.random.Generator] = None,
        n_grid: int = 10000,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Sample reconstructed positions from the PSF-convolved template skymap.

        Convolves the template with the King PSF in harmonic space, then draws
        pixel indices weighted by the convolved map values. This is equivalent
        to drawing a true position from the template and applying a King PSF
        offset, but is more efficient because the convolution is done once for
        the whole map rather than per-event.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng().
        n_grid : int, optional
            Number of points in the CDF lookup grid. Higher values give more
            accurate sampling at the cost of memory and setup time. Default
            is 10000, which gives ~arcminute accuracy. Note that this parameter
            is ignored for this method since the sampling is done directly from
            the convolved map rather than via inverse CDF.

        Returns
        -------
        reco_ra : ndarray
            Reconstructed right ascension values in radians, shape (n,).
        reco_dec : ndarray
            Reconstructed declination values in radians, shape (n,).

        Notes
        -----
        Samples land at HEALPix pixel centres. The positional resolution is
        therefore limited by the skymap pixelisation (~`hp.nside2resol(nside)`).
        """
        if np.any(alpha <= 0):
            raise ValueError(
                "Received alpha containing at least one point <= 0. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 0.0 when passing them into TemplateSmearedKingPDF."
            )
        if np.any(beta <= 1):
            raise ValueError(
                "Received beta containing at least one point <= 1. The"
                " KingPDF isn't defined in this region. Ensure your points are"
                " all above 1.0 when passing them into TemplateSmearedKingPDF."
            )

        if rng is None:
            rng = np.random.default_rng()

        convolved = self.convolve_map(alpha, beta)
        weights = np.maximum(convolved, 0.0)
        pixel_indices = rng.choice(len(convolved), size=n, p=weights / weights.sum())

        colatitude, longitude = hp.pix2ang(self.nside, pixel_indices)
        return longitude, np.pi / 2 - colatitude  # reco_ra, reco_dec
