from typing import Union
import numpy as np
import numpy.typing as npt
from numba import vectorize, float32, float64

_log10pi: float = np.log10(np.pi)


@vectorize(
    [float32(float32, float32, float32), float64(float64, float64, float64)],
    target="cpu",
    cache=True,
)
def _unnormalized_pdf(
    x: Union[float, npt.NDArray[np.floating]],
    alpha: Union[float, npt.NDArray[np.floating]],
    beta: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Evaluate the unnormalized spherical King function (without solid angle Jacobian).

    Uses the exact spherical distance (1 - cos θ) in place of the flat-sky θ²/2,
    so this form is accurate for all angular scales:
        f(x) = [1 + (1 - cos x) / (alpha² * beta)]^(-beta)

    For small x, (1 - cos x) ≈ x²/2, recovering the flat-sky King function.

    Parameters
    ----------
    x : float or ndarray
        Angular separation from the source, in radians.
    alpha : float or ndarray
        King distribution alpha parameter (scale).
    beta : float or ndarray
        King distribution beta parameter (tail weight).

    Returns
    -------
    ndarray
        Unnormalized King function values with units of probability/sterradian.
    """
    return (1 + (1 - np.cos(x)) / (alpha**2 * beta)) ** -beta


@vectorize(
    [float32(float32, float32, float32), float64(float64, float64, float64)],
    target="cpu",
    cache=True,
)
def _unnormalized_cdf(
    x: Union[float, npt.NDArray[np.floating]], alpha: float, beta: float
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Evaluate the CDF of the radial King function (without solid angle Jacobian).

    Uses the exact spherical form via the substitution t = 1 - cos θ,
    dt = sin θ dθ, which reduces the solid-angle integral to a power law
    with a closed-form antiderivative. No flat-sky approximation.

    Parameters
    ----------
    x : scalar
        The location at which to calculate the CDF in radians.
    alpha : float
        King distribution alpha parameter (scale) in radians.
    beta : float
        King distribution beta parameter (tail weight). Must be > 1.

    Returns
    -------
    float
        Unnormalized partial integral ∫₀ˣ f(θ) · 2π · sin θ dθ.
    """
    alpha2beta = alpha**2 * beta
    return (2 * np.pi * alpha2beta / (beta - 1)) * (
        1 - (1 + (1 - np.cos(x)) / alpha2beta) ** (1 - beta)
    )


@vectorize(
    [float32(float32, float32, float32), float64(float64, float64, float64)],
    target="cpu",
    cache=True,
)
def _norm(
    alpha: Union[float, npt.NDArray[np.floating]],
    beta: Union[float, npt.NDArray[np.floating]],
    maximum: float,
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Compute the normalization constant for the King PDF over the sphere.

    The integral includes the sin(theta) for spherical coordinates and
    normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

    Parameters
    ----------
    alpha : float or ndarray
        King distribution alpha parameter (scale).
    beta : float or ndarray
        King distribution beta parameter (tail weight).
    maximum : float
        The maximum angular value for the distribution. This would normally
        just be pi, but can be smaller if the user is truncating the distribution.

    Returns
    -------
    ndarray
        Normalization constants such that PDF integrates to 1 over the sphere.
    """
    alpha2beta = alpha**2 * beta
    return (beta - 1) / (
        2 * np.pi * alpha2beta * (1 - (1 + (1 - np.cos(maximum)) / alpha2beta) ** (1 - beta))
    )  # type: ignore[no-any-return]
