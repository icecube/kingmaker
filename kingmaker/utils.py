from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .distribution import _unnormalized_pdf


@njit(cache=True)
def _interp1d(x: float, xlow: float, xhigh: float, ylow: float, yhigh: float) -> float:
    """
    Perform 1D linear interpolation.

    Parameters
    ----------
    x : float
        Point at which to interpolate.
    xlow : float
        Lower x-coordinate of the interval.
    xhigh : float
        Upper x-coordinate of the interval.
    ylow : float
        Function value at xlow.
    yhigh : float
        Function value at xhigh.

    Returns
    -------
    float
        Linearly interpolated value at x.
    """
    return ylow + (yhigh - ylow) / (xhigh - xlow) * (x - xlow)


@njit(cache=True)
def angular_distance(
    src_ra: Union[float, npt.NDArray[np.floating]],
    src_dec: Union[float, npt.NDArray[np.floating]],
    ra: Union[float, npt.NDArray[np.floating]],
    dec: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Calculate angular distance on the sphere using the haversine formula.

    Computes the great-circle distance between celestial coordinates using
    spherical trigonometry.

    Parameters
    ----------
    src_ra : float or ndarray
        Source right ascension in radians.
    src_dec : float or ndarray
        Source declination in radians.
    ra : float or ndarray
        Target right ascension(s) in radians.
    dec : float or ndarray
        Target declination(s) in radians.

    Returns
    -------
    float or ndarray
        Angular separation(s) in radians.
    """
    cosDist = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(src_dec) * np.sin(dec)
    return np.arccos(np.minimum(np.maximum(cosDist, -1.0), 1.0))  # type: ignore[no-any-return]


@njit(cache=True)
def _pre_mask_and_distance(
    ra: npt.NDArray[np.floating],
    dec: npt.NDArray[np.floating],
    src_ra: float,
    src_dec: float,
    cutoff: float,
    ra_span: float,
) -> npt.NDArray[np.floating]:
    """Single-pass rectangular pre-filter and haversine for the single-source case.

    Combines the dec/RA bounding-box rejection and the exact angular distance
    into one loop over events, reading each record's ra/dec once from the same
    cache line. The haversine is evaluated only for events that survive both
    pre-checks (~0.6% at 10°, ~0.15% at 5°).

    Returns an array of length len(ra) where element i holds the angular
    distance to the source when event i is within `cutoff`, and -1.0 otherwise.
    """
    n = len(ra)
    dists = np.full(n, -1.0)
    cos_src_dec = np.cos(src_dec)
    sin_src_dec = np.sin(src_dec)
    for i in prange(n):
        if abs(dec[i] - src_dec) >= cutoff:
            continue
        ra_diff = abs(ra[i] - src_ra)
        if ra_diff > np.pi:
            ra_diff = 2 * np.pi - ra_diff
        if ra_diff >= ra_span:
            continue
        cos_dist = np.cos(ra[i] - src_ra) * cos_src_dec * np.cos(dec[i]) + sin_src_dec * np.sin(
            dec[i]
        )
        d = np.arccos(min(max(cos_dist, -1), 1))
        if d < cutoff:
            dists[i] = d
    return dists


@njit(cache=True)
def _marginalize_ra(
    dec_true: float,
    alpha: float,
    beta: float,
    norm: float,
    angular_cutoff: float,
    signed_delta_dec_grid: npt.NDArray[np.float64],
    ra_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Integrate the King PDF over RA in [0, pi] for each signed declination offset.

    For each entry in signed_delta_dec_grid, computes:

        M(delta_dec) = 2 * integral_0^pi King(psi(dRA, dec_true + delta_dec, dec_true),
                                              alpha, beta) d(dRA)

    where the factor of 2 exploits the RA symmetry of the King distribution.
    Trapezoid quadrature is applied over the ra_grid nodes. Entries where
    dec_true + delta_dec falls outside [-pi/2, pi/2] return 0.

    Parameters
    ----------
    dec_true : float
        True source declination in radians.
    alpha : float
        King distribution alpha parameter (scale) in radians.
    beta : float
        King distribution beta parameter (tail weight, > 1).
    norm : float
        Precomputed normalization constant for this (alpha, beta, angular_cutoff).
    angular_cutoff : float
        Maximum angular separation in radians; King PDF is zero beyond this.
    signed_delta_dec_grid : ndarray
        Grid of dec_reco - dec_true offsets in radians.
    ra_grid : ndarray
        Right ascension integration nodes in [0, pi] in radians.

    Returns
    -------
    ndarray
        RA-marginalized PDF values, one per signed_delta_dec_grid entry.
    """
    n_delta_dec = len(signed_delta_dec_grid)
    result = np.zeros(n_delta_dec)

    for i in range(n_delta_dec):
        dec_reco = dec_true + signed_delta_dec_grid[i]
        if dec_reco > np.pi / 2.0 or dec_reco < -np.pi / 2.0:
            continue

        # Compute angular distances for all RA grid points at once; source at RA=0.
        psi = angular_distance(0.0, dec_true, ra_grid, dec_reco)
        pdf = np.zeros_like(psi)
        for j in range(len(ra_grid)):
            if psi[j] <= angular_cutoff:
                pdf[j] = norm * _unnormalized_pdf(psi[j], alpha, beta)

        # Double the [0, pi] integral to account for [pi, 2*pi] by RA symmetry.
        result[i] = 2.0 * np.trapezoid(pdf, ra_grid)

    return result


@njit(parallel=True, cache=True)
def _build_marginalized_grid(
    dec_true_grid: npt.NDArray[np.float64],
    alpha_grid: npt.NDArray[np.float64],
    beta_grid: npt.NDArray[np.float64],
    norm_grid: npt.NDArray[np.float64],
    angular_cutoff: float,
    signed_delta_dec_grid: npt.NDArray[np.float64],
    ra_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Build the full 4D RA-marginalized King PDF grid.

    Iterates over (dec_true, alpha, beta), calling _marginalize_ra for each
    triple and storing results in the output array. The dec_true axis is
    parallelized with prange.

    Parameters
    ----------
    dec_true_grid : ndarray, shape (n_dec,)
        Source declination grid points in radians.
    alpha_grid : ndarray, shape (n_alpha,)
        King alpha grid points in radians.
    beta_grid : ndarray, shape (n_beta,)
        King beta grid points.
    norm_grid : ndarray, shape (n_alpha, n_beta)
        Precomputed normalization constants for each (alpha, beta) pair.
    angular_cutoff : float
        Maximum angular separation in radians.
    signed_delta_dec_grid : ndarray, shape (n_delta_dec,)
        Grid of dec_reco - dec_true offsets in radians.
    ra_grid : ndarray, shape (n_ra,)
        RA integration nodes in [0, pi] in radians.

    Returns
    -------
    ndarray, shape (n_dec, n_alpha, n_beta, n_delta_dec)
        Marginalized PDF values on the full parameter grid.
    """
    n_dec = len(dec_true_grid)
    n_alpha = len(alpha_grid)
    n_beta = len(beta_grid)
    n_delta_dec = len(signed_delta_dec_grid)

    grid = np.zeros((n_dec, n_alpha, n_beta, n_delta_dec))
    for i in prange(n_dec):
        for j in range(n_alpha):
            for k in range(n_beta):
                grid[i, j, k, :] = _marginalize_ra(
                    dec_true_grid[i],
                    alpha_grid[j],
                    beta_grid[k],
                    norm_grid[j, k],
                    angular_cutoff,
                    signed_delta_dec_grid,
                    ra_grid,
                )
    return grid


@njit(cache=True)
def meshgrid2d(
    a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Create a 2D meshgrid from 1D coordinate arrays, compatible with numba JIT compilation.

    Returns transposed grids in matrix indexing ('ij') convention.

    Parameters
    ----------
    a : ndarray
        1D array of coordinates for first dimension.
    b : ndarray
        1D array of coordinates for second dimension.

    Returns
    -------
    grid_a : ndarray
        2D grid of 'a' values with shape (len(b), len(a)).
    grid_b : ndarray
        2D grid of 'b' values with shape (len(b), len(a)).
    """
    output_a = np.empty((len(a), len(b)), dtype=a.dtype)
    output_b = np.empty((len(a), len(b)), dtype=b.dtype)

    for i in range(len(a)):
        output_a[i, :] = a[i]
    for j in range(len(b)):
        output_b[:, j] = b[j]
    return output_a.T, output_b.T
