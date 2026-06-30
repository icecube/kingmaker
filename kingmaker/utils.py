from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import njit, prange


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
