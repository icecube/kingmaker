from typing import Any, Dict, List, Optional, Tuple, Union, cast
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .distribution import _cdf_and_gradient
from .pdf import KingPDF
from .utils import angular_distance


class KingPSFFitter:
    """
    Fit King PSF parameters to simulated signal events in arbitrary dimensions.

    This class bins signal Monte Carlo events along user-specified observables
    and fits King distribution parameters (alpha, beta) to the angular error
    distribution in each bin.

    Parameters
    ----------
    signal_events : structured array
        Numpy structured array containing signal MC events. Must include:
        - 'ra', 'dec': reconstructed coordinates (radians)
        - 'true_ra', 'true_dec': true coordinates (radians)
        Additional fields can be used for parameterization binning.
    parametrization_bins : dict
        Dictionary mapping observable names to bin edges or number of bins.
        Keys must correspond to fields in signal_events.
        Values can be:
        - int: number of equal-probability bins
        - array-like: explicit bin edges
    dpsi_nbins : int, optional
        Number of bins in angular error (dpsi) for fitting. Default is 101.
    minimum_counts : int, optional
        Minimum number of events required in a bin for fitting. Default is 100.
    remove_weight_outliers : bool
        If True, remove events with weights beyond the bounds of weight_outlier_percentiles
        (calculated per parametrization bin) from the fit. Used to stabilize fits by removing
        weighting outliers. Default is True.
    weight_outlier_percentiles : list, optional
        The percentiles (ranging from 0-100) defining the range of weights to accept
        for the per-parametrization bin histogramming and fitting. Note that these are
        applied based on the weights based on sorted index value and not cumulative
        weight value like np.percentile. Default is [0, 95],
    weight_field : str, optional
        Field name for oneweight. If None, equal weights are used.
    true_ra_name : str
        Field name for the true value of the signal events' right ascension.
    true_dec_name : str
        Field name for the true value of the signal events' declination.
    true_energy_name : str
        Field name for the true value of the signal events' energy.
    spectral_indices : array-like, optional
        Spectral indices (gamma) for reweighting. Default is [2.0].
    angular_cutoff : float, optional
        Maximum angular separation for King PDF. Default is pi.

    Attributes
    ----------
    fit_alpha : ndarray
        Fitted alpha parameters for each bin.
    fit_beta : ndarray
        Fitted beta parameters for each bin.
    histograms : ndarray
        Histogram values for each bin.
    uncertainties : ndarray
        Histogram uncertainties for each bin.
    dpsi_bins : ndarray
        Angular error bin edges for each bin.
    fit_quality : ndarray
        Chi-square values indicating fit quality.
    """

    def __init__(
        self,
        signal_events: npt.NDArray[Any],
        parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]],
        dpsi_nbins: int = 101,
        minimum_counts: int = 100,
        remove_weight_outliers=True,
        weight_outlier_percentiles=[0, 95],
        weight_field: Optional[str] = "ow",
        true_ra_name: str = "trueRa",
        true_dec_name: str = "trueDec",
        true_energy_name: str = "trueE",
        spectral_indices: Optional[Union[List[float], npt.NDArray[np.floating]]] = None,
        angular_cutoff: float = np.pi,
    ) -> None:
        """Initialize the KingPSFFitter."""
        self.signal_events = signal_events
        self.remove_weight_outliers = remove_weight_outliers
        self.weight_outlier_percentiles = weight_outlier_percentiles

        self.weight_field = weight_field
        self.true_ra_name = true_ra_name
        self.true_dec_name = true_dec_name
        self.true_energy_name = true_energy_name
        self.dpsi_nbins = dpsi_nbins
        self.minimum_counts = minimum_counts
        self.spectral_indices = (
            np.atleast_1d(spectral_indices) if spectral_indices is not None else np.array([2.0])
        )
        self.angular_cutoff = angular_cutoff

        # Initialize King PDF
        self.king_pdf = KingPDF(angular_cutoff=angular_cutoff)

        # Validate and setup binning
        self._validate_fields(parametrization_bins)
        self.parametrization_bins = self._setup_bins(parametrization_bins)
        self.bin_names = list(self.parametrization_bins.keys())
        self.parametrization_shape = [len(bins) - 1 for bins in self.parametrization_bins.values()]

        # Calculate angular distances
        self.dpsi = angular_distance(
            self.signal_events["ra"],
            self.signal_events["dec"],
            self.signal_events[self.true_ra_name],
            self.signal_events[self.true_dec_name],
        )

        # Bin events
        self.event_indices = self._bin_events()

        # Pre-group events by flat bin index for O(1) per-bin lookup in fit_all_bins.
        # Replaces the per-iteration boolean-mask construction over all events.
        _shape = tuple(self.parametrization_shape)
        _idx_arrays = [
            np.clip(self.event_indices[k], 1, s) - 1  # 0-based, clipped in-range
            for k, s in zip(self.bin_names, _shape)
        ]
        _flat = np.ravel_multi_index(_idx_arrays, _shape)
        self._event_sort_order = np.argsort(_flat, kind="stable")
        _sorted_flat = _flat[self._event_sort_order]
        _n_bins = int(np.prod(_shape))
        self._bin_boundaries = np.searchsorted(_sorted_flat, np.arange(_n_bins + 1))

        # Initialize storage arrays
        self._initialize_storage()

    def _validate_fields(
        self, parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]]
    ) -> None:
        """
        Validate that required and parameterization fields exist in signal events.

        Parameters
        ----------
        parametrization_bins : dict
            Dictionary of binning specifications.

        Raises
        ------
        ValueError
            If required fields are missing.
        """
        required_fields = ["ra", "dec", self.true_ra_name, self.true_dec_name]
        names = self.signal_events.dtype.names or ()
        missing_required = [f for f in required_fields if f not in names]
        if missing_required:
            raise ValueError(f"Signal events missing required fields: {missing_required}")

        missing_params = [key for key in parametrization_bins.keys() if key not in names]
        if missing_params:
            raise ValueError(
                f"Parametrization fields {missing_params} not found in signal events. "
                f"Available fields: {names}"
            )

        if self.weight_field is not None and (self.weight_field not in names):
            raise ValueError(f"Weight field '{self.weight_field}' not found in signal events.")

    def _setup_bins(
        self, parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]]
    ) -> Dict[str, npt.NDArray[np.floating]]:
        """
        Convert binning specifications to explicit bin edges.

        Parameters
        ----------
        parametrization_bins : dict
            Dictionary mapping field names to bin specs (int or array).

        Returns
        -------
        dict
            Dictionary mapping field names to bin edge arrays.
        """
        bins_dict = {}
        for key, val in parametrization_bins.items():
            if isinstance(val, int):
                # Create equal-probability bins
                bins_dict[key] = self._get_percentile_bins(val, self.signal_events[key])
            elif isinstance(val, (tuple, list, np.ndarray)):
                bins_dict[key] = np.asarray(val)
            else:
                raise ValueError(
                    f"Unknown binning specification for '{key}': {val}. "
                    "Use int for number of bins or array-like for bin edges."
                )
        return bins_dict

    def _get_percentile_bins(
        self,
        nbins: int,
        values: npt.NDArray[np.floating],
        weights: Optional[npt.NDArray[np.floating]] = None,
    ) -> npt.NDArray[np.floating]:
        """
        Create bins with approximately equal number of (weighted) events.

        Parameters
        ----------
        nbins : int
            Number of bins to create.
        values : ndarray
            Values to bin.
        weights : ndarray, optional
            Event weights. If None, equal weights used.

        Returns
        -------
        ndarray
            Bin edges.
        """
        if weights is None:
            weights = np.ones(len(values))

        # Sort and create cumulative distribution
        sorted_idx = np.argsort(values)
        cumulative = np.cumsum(weights[sorted_idx]) / weights.sum()

        # Find bin edges at equal probability intervals
        percentiles = np.linspace(0, 1, nbins + 1)
        positions = np.searchsorted(cumulative, percentiles)
        positions = np.clip(positions, 0, len(values) - 1)

        # Handle duplicates by using unique values
        positions = np.unique(positions)
        bin_edges = values[sorted_idx][positions]

        # Ensure we have at least 2 edges (1 bin)
        if len(bin_edges) < 2:
            return np.array([values.min(), values.max()])

        return bin_edges

    def _bin_events(self) -> Dict[str, npt.NDArray[np.integer]]:
        """
        Assign each event to a bin index for each parameterization dimension.

        Returns
        -------
        dict
            Dictionary mapping field names to bin indices for each event.
        """
        event_indices = {}
        for key, bins in self.parametrization_bins.items():
            event_indices[key] = np.digitize(self.signal_events[key], bins)
        return event_indices

    def _initialize_storage(self) -> None:
        """Initialize arrays to store fit results and diagnostics."""
        shape_with_gamma = [len(self.spectral_indices)] + self.parametrization_shape

        # Fit parameters
        self.fit_alpha = np.full(shape_with_gamma, np.median(self.dpsi))
        self.fit_beta = np.full(shape_with_gamma, 2.25)

        # Diagnostics
        self.histograms = np.zeros(shape_with_gamma + [self.dpsi_nbins], dtype=float)
        self.uncertainties = np.zeros(shape_with_gamma + [self.dpsi_nbins], dtype=float)
        self.dpsi_bins = np.zeros(shape_with_gamma + [self.dpsi_nbins + 1], dtype=float)
        self.fit_quality = np.zeros(shape_with_gamma, dtype=float)
        self.event_counts = np.zeros(shape_with_gamma, dtype=int)

    def fit_all_bins(self, verbose: bool = True) -> Dict[str, npt.NDArray]:
        """
        Fit King PSF parameters in all bins.

        Iterates over all bins defined by parametrization_bins and spectral_indices,
        fitting King distribution parameters to the angular error distribution.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information. Default is True.

        Returns
        -------
        dict
            Dictionary containing fit results with keys:
            - 'alpha': fit_alpha array
            - 'beta': fit_beta array
            - 'histograms': histogram values
            - 'uncertainties': histogram uncertainties
            - 'dpsi_bins': angular error bin edges
            - 'fit_quality': chi-square values
            - 'event_counts': number of events per bin
        """
        if verbose:
            print(f"Fitting King PSF in {np.prod(self.parametrization_shape)} bins...")
            print(f"  Spectral indices: {self.spectral_indices}")
            print(f"  Binning dimensions: {self.bin_names}")

        # Iterate over spectral indices
        for g_idx, gamma in enumerate(self.spectral_indices):
            if verbose:
                print(f"\n  Spectral index γ = {gamma:.2f}")

            # Calculate event weights
            if self.weight_field is not None:
                weights = self.signal_events[self.weight_field] * self.signal_events[
                    self.true_energy_name
                ] ** (-gamma)
            else:
                weights = np.ones(len(self.signal_events))

            # Iterate over all bin combinations
            n_fitted = 0
            n_skipped = 0

            total_bins = np.prod(self.parametrization_shape)
            for bin_indices in tqdm(np.ndindex(*self.parametrization_shape), total=total_bins):
                flat_idx = int(np.ravel_multi_index(bin_indices, tuple(self.parametrization_shape)))
                event_idx = self._event_sort_order[
                    self._bin_boundaries[flat_idx] : self._bin_boundaries[flat_idx + 1]
                ]

                if self.remove_weight_outliers and len(event_idx) > 0:
                    bin_weights = weights[event_idx]
                    idx_range = [
                        int(len(bin_weights) * self.weight_outlier_percentiles[0] / 100),
                        int(len(bin_weights) * self.weight_outlier_percentiles[1] / 100),
                    ]
                    idx = np.digitize(bin_weights, np.unique(bin_weights))
                    event_idx = event_idx[(idx_range[0] <= idx) & (idx <= idx_range[1])]

                n_events = len(event_idx)
                param_idx = tuple([g_idx] + list(bin_indices))
                self.event_counts[param_idx] = n_events

                # Skip if insufficient events
                if n_events < self.minimum_counts:
                    n_skipped += 1
                    continue

                # Fit this bin
                success = self._fit_single_bin(event_idx, weights, param_idx)
                if success:
                    n_fitted += 1
                else:
                    n_skipped += 1

            if verbose:
                print(f"    Fitted {n_fitted} bins, skipped {n_skipped} bins")

        if verbose:
            print("\nFitting complete!")

        return {
            "alpha": self.fit_alpha,
            "beta": self.fit_beta,
            "histograms": self.histograms,
            "uncertainties": self.uncertainties,
            "dpsi_bins": self.dpsi_bins,
            "fit_quality": self.fit_quality,
            "event_counts": self.event_counts,
            "parametrization_bins": self.parametrization_bins,  # type: ignore[dict-item]
        }

    def _cdf_chi2(self, cdf_hist, cdf_variance, bins, alpha, beta):
        try:
            cdf, grad_alpha, grad_beta = _cdf_and_gradient(
                bins[1:], alpha, beta, self.angular_cutoff
            )
        except ZeroDivisionError:
            return 100000.0, np.zeros(2)

        scale = cdf_hist[-1] / cdf[-1]
        expected = scale * cdf
        residuals = cdf_hist - expected
        inv_variance = 1.0 / np.nextafter(cdf_variance, np.inf)

        n_bins = len(bins)
        val = np.sum(residuals**2 * inv_variance) / n_bins
        if not np.isfinite(val):
            return 100000.0, np.zeros(2)

        # d(chi2)/dtheta = (-2*scale/n_bins) * sum(r/var * (dCDF - (cdf/cdf_last)*dCDF_last))
        weighted_residuals = residuals * inv_variance
        cdf_ratio = cdf / cdf[-1]
        grad = (-2.0 * scale / n_bins) * np.array(
            [
                np.sum(weighted_residuals * (grad_alpha - cdf_ratio * grad_alpha[-1])),
                np.sum(weighted_residuals * (grad_beta - cdf_ratio * grad_beta[-1])),
            ]
        )
        return val, grad

    def _fit_single_bin(
        self,
        event_idx: npt.NDArray[np.intp],
        weights: npt.NDArray[np.floating],
        param_idx: Tuple[int, ...],
    ) -> bool:
        """
        Fit King parameters for a single bin.

        Parameters
        ----------
        event_idx : ndarray
            Integer indices of events in this bin.
        weights : ndarray
            Event weights.
        param_idx : tuple
            Index tuple for storing results.

        Returns
        -------
        bool
            True if fit succeeded, False otherwise.
        """
        # Extract events in this bin
        masked_dpsi = self.dpsi[event_idx]
        masked_weights = weights[event_idx]
        masked_weights /= masked_weights.sum()  # Normalize

        # Create bins for this subset. Also calculate the
        # phase space parameter while we're here. We'll need
        # it in order to store the histograms as PDFs later.
        dpsi_bins = self._get_percentile_bins(self.dpsi_nbins, masked_dpsi, masked_weights)
        dpsi_bins = np.unique(dpsi_bins)
        delta = -2 * np.pi * np.diff(np.cos(dpsi_bins))
        self.dpsi_bins[param_idx][: len(dpsi_bins)] = dpsi_bins

        # If we don't have enough bins, skip
        if len(dpsi_bins) < 3:
            return False

        bin_centers = (dpsi_bins[:-1] + dpsi_bins[1:]) / 2

        # Create weighted histogram
        hist, _ = np.histogram(masked_dpsi, bins=dpsi_bins, weights=masked_weights)
        hist2, _ = np.histogram(masked_dpsi, bins=dpsi_bins, weights=masked_weights**2)

        cdf_hist = np.cumsum(hist)
        cdf_variance = np.cumsum(hist2) / np.sum(hist) ** 2

        # Get initial guess from peak location.
        alpha_guess = bin_centers[np.searchsorted(cdf_hist, 0.5)]
        best_params = None
        best_chi2 = np.inf
        for beta in [1.25, 1.75, 2, 2.5, 4, 7, 9]:
            result = minimize(
                lambda params: self._cdf_chi2(cdf_hist, cdf_variance, dpsi_bins, *params),
                [alpha_guess, beta],
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (np.nextafter(1e-4, np.pi), np.nextafter(self.angular_cutoff, 0)),
                    (np.nextafter(1, 2), np.nextafter(1000, 1)),
                ],
            )

            if result.success and (best_chi2 > result.fun):
                best_params, best_chi2 = result.x, result.fun

        # Store histogram data (pad/truncate to match storage size).
        # Make sure to rescale by the phase space to get densities.
        n_store = min(len(hist), self.dpsi_nbins)
        self.histograms[param_idx][:n_store] = hist[:n_store] / delta
        self.uncertainties[param_idx][:n_store] = np.sqrt(hist2[:n_store]) / delta
        self.dpsi_bins[param_idx][: len(dpsi_bins)] = dpsi_bins

        # Store results if we found a solution
        if best_params is not None:
            self.fit_alpha[param_idx] = best_params[0]
            self.fit_beta[param_idx] = best_params[1]
            self.fit_quality[param_idx] = best_chi2
            return True

        return False

    def get_interpolator(self, gamma_index: int = 0) -> Tuple[Any, Any]:
        """
        Get an interpolator for fitted parameters at a given spectral index.

        Parameters
        ----------
        gamma_index : int, optional
            Index of the spectral index to use. Default is 0.

        Returns
        -------
        tuple
            (alpha_interpolator, beta_interpolator) functions that interpolate
            fitted parameters based on parameterization bin values.

        Notes
        -----
        Requires scipy.interpolate.RegularGridInterpolator (not imported by default
        to avoid dependency). This method will raise ImportError if scipy is not available.
        """
        from scipy.interpolate import RegularGridInterpolator

        # Get bin centers for each dimension
        bin_centers = []
        for key in self.bin_names:
            bins = self.parametrization_bins[key]
            centers = (bins[:-1] + bins[1:]) / 2
            bin_centers.append(centers)

        # Create interpolators
        alpha_interp = RegularGridInterpolator(
            tuple(bin_centers),
            self.fit_alpha[gamma_index],
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )

        beta_interp = RegularGridInterpolator(
            tuple(bin_centers),
            self.fit_beta[gamma_index],
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )

        return alpha_interp, beta_interp

    def plot_fit(
        self,
        bin_indices: Union[Tuple[int, ...], Dict[str, int]],
        gamma_index: int = 0,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot the fitted King PDF for a specific bin.

        Parameters
        ----------
        bin_indices : tuple or dict
            Indices of the bin to plot. Can be tuple of integers or dict
            mapping bin names to indices.
        gamma_index : int, optional
            Index of spectral index. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Notes
        -----
        Requires matplotlib (not imported by default). This method will raise
        ImportError if matplotlib is not available.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Convert dict to tuple if needed
        if isinstance(bin_indices, dict):
            bin_indices = tuple(bin_indices[key] for key in self.bin_names)

        param_idx = tuple([gamma_index] + list(bin_indices))

        # Get histogram data
        hist = self.histograms[param_idx]
        uncertainty = self.uncertainties[param_idx]
        bins = self.dpsi_bins[param_idx]

        # Only plot non-zero bins
        mask = hist > 0
        bin_centers = (bins[:-1] + bins[1:])[mask] / 2

        # Plot histogram
        ax.errorbar(
            np.degrees(bin_centers),
            hist[mask],
            yerr=uncertainty[mask],
            fmt="o",
            label="MC Events",
            color="black",
            markersize=4,
        )

        # Plot fitted King PDF
        alpha = self.fit_alpha[param_idx]
        beta = self.fit_beta[param_idx]
        dpsi_fine = np.linspace(0, min(8 * alpha, np.pi), 1000)
        pdf_fit = cast(npt.NDArray[np.floating], self.king_pdf.pdf(dpsi_fine, alpha, beta))
        pdf_fit *= hist[mask].max() / pdf_fit.max()  # Normalize for visualization

        ax.plot(np.degrees(dpsi_fine), pdf_fit, "-", linewidth=2, label="King Fit", color="blue")

        # Add labels
        ax.set_xlabel("Angular Error (degrees)")
        ax.set_ylabel("Normalized Density")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Add fit parameters to plot
        title = f"γ={self.spectral_indices[gamma_index]:.2f}, "
        title += f"α={np.degrees(alpha):.3f}°, β={beta:.2f}\n"
        for i, key in enumerate(self.bin_names):
            bin_idx = bin_indices[i]
            bins = self.parametrization_bins[key]
            title += f"{key}=[{bins[bin_idx]:.2e}, {bins[bin_idx + 1]:.2e}] "
        ax.set_title(title, fontsize=10)

        return ax
