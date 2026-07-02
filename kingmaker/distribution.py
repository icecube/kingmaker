from typing import Union
import numpy as np
import numpy.typing as npt
from numba import njit, vectorize, float32, float64

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
    Evaluate the unnormalized spherical King function (without solid angle Jacobian):
        f(x) = [1 + (1 - cos x) / (alpha² * beta)]^(-beta)

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
    with a closed-form antiderivative.

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


@njit(cache=True)
def _cdf_and_gradient(
    x: npt.NDArray[np.float64],
    alpha: float,
    beta: float,
    angular_cutoff: float,
) -> tuple:
    """
    Compute the normalized King CDF and its partial derivatives w.r.t. alpha and beta.

    Returns the same CDF values as _norm(alpha, beta, angular_cutoff) * _unnormalized_cdf(x, alpha, beta),
    plus the exact analytical partial derivatives at each point in x. Computing both in a single
    call avoids the two extra CDF evaluations that finite-difference Jacobians require per
    optimizer step.

    The CDF has the closed form:
        CDF(x) = [1 - power_base(x)^(1-beta)] / [1 - power_base(angular_cutoff)^(1-beta)]
    where power_base(x) = 1 + (1 - cos x) / (alpha^2 * beta).

    Note that we need to include the form of the normalization factor here, which makes things
    a little more complicated than simply differentiating the unnormalized CDF and multiplying
    by a normalization constant.T

    The alpha derivative follows from the chain rule through power_base^(1-beta) and the
    quotient rule on the normalization denominator. The beta derivative requires an
    additional log(power_base(x)) term from differentiating power_base^(1-beta) w.r.t. beta.

    Parameters
    ----------
    x : ndarray
        Angular separations in radians.
    alpha : float
        King alpha parameter (angular scale). Must be > 0.
    beta : float
        King beta parameter (tail index). Must be > 1.
    angular_cutoff : float
        Maximum angular separation used for normalization; CDF(angular_cutoff) = 1.

    Returns
    -------
    cdf : ndarray
        Normalized CDF values at each x, identical to _norm * _unnormalized_cdf.
    grad_alpha : ndarray
        Partial derivative of the CDF w.r.t. alpha at each x.
    grad_beta : ndarray
        Partial derivative of the CDF w.r.t. beta at each x.
    """
    # CDF values via existing normalized functions (no logic duplication)
    alpha2beta = alpha * alpha * beta
    norm_val = _norm(alpha, beta, angular_cutoff)
    cdf = norm_val * _unnormalized_cdf(x, alpha, beta)

    # norm_val = (beta-1) / (2*pi*alpha2beta*norm_denom), so recover norm_denom
    # and power_term_cutoff = 1 - norm_denom without repeating the power computation
    norm_denom = (beta - 1.0) / (2.0 * np.pi * alpha2beta * norm_val)
    power_term_cutoff = 1.0 - norm_denom

    # power_base_cutoff is needed for the log and ratio terms in the derivatives;
    # it cannot be recovered from norm_val alone
    one_minus_cos_cutoff = 1.0 - np.cos(angular_cutoff)
    power_base_cutoff = 1.0 + one_minus_cos_cutoff / alpha2beta

    d_norm_denom_dalpha = (
        2.0
        * (1.0 - beta)
        * one_minus_cos_cutoff
        * power_term_cutoff
        / (power_base_cutoff * alpha * alpha2beta)
    )
    d_norm_denom_dbeta = power_term_cutoff * (
        np.log(power_base_cutoff)
        - (beta - 1.0) * one_minus_cos_cutoff / (beta * alpha2beta * power_base_cutoff)
    )

    n = len(x)
    grad_alpha = np.empty(n)
    grad_beta = np.empty(n)

    for i in range(n):
        one_minus_cos = 1.0 - np.cos(x[i])
        power_base = 1.0 + one_minus_cos / alpha2beta
        power_term = power_base ** (1.0 - beta)

        dN_dalpha = (
            2.0 * (1.0 - beta) * one_minus_cos * power_term / (power_base * alpha * alpha2beta)
        )
        dN_dbeta = power_term * (
            np.log(power_base) - (beta - 1.0) * one_minus_cos / (beta * alpha2beta * power_base)
        )

        # Quotient rule: d(N/norm_denom)/dtheta = (dN - cdf * d_norm_denom) / norm_denom
        grad_alpha[i] = (dN_dalpha - cdf[i] * d_norm_denom_dalpha) / norm_denom
        grad_beta[i] = (dN_dbeta - cdf[i] * d_norm_denom_dbeta) / norm_denom

    return cdf, grad_alpha, grad_beta
