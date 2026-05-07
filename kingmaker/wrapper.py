from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt

from os.path import exists
import logging
import numpy as np
import healpy as hp

from .pdf import KingPDF, TemplateSmearedKingPDF
from .fitting import KingPSFFitter
from .utils import angular_distance, _interp1d


class KingSpatialLikelihood:
    """Wrapper class to encapsulate King distribution functionality, including PDF evaluation and parameter fitting.
    This class provides a unified interface for working with King distributions, allowing users to easily fit simulation
    and evaluate the PDF over events for likelihood calculations.

    Users create an instance of KingSpatialLikelihood by passing in parameters, binning, and simulated events. The
    class then fits King distribution parameters using the requested parameter binning. From this point onward, users
    only need to call either the "PDF" or "template" evaluation methods with their events to obtain likelihood values
    for their analyses. These methods interpolate the fitted King distribution parameters per-event using the event's
    observable parameters and the provided binning, and then evaluate the PDF or template-smoothed PDF at the event's
    reconstructed equatorial position.
    """

    # Configuration parameters
    parametrization_bins: Dict[str, npt.NDArray[np.floating]]
    spectral_indices: npt.NDArray[np.floating]
    angular_cutoff: float
    cache_parameters: bool = True
    cache_name: str = "king_parameters_cache.npz"

    # Store an instance of the PDF class to use for evaluations. This will be
    # either a KingPDF (point source) or a TemplateSmearedKingPDF
    # (extended source with skymap).
    king_pdf: KingPDF
    template_pdf: TemplateSmearedKingPDF
    nside: int

    # Have some place to cache the per-event information so we don't need to
    # recalculate it every time we evaluate the PDF.
    events: Optional[Any] = None
    event_alpha: Dict[float, npt.NDArray[np.floating]]
    event_beta: Dict[float, npt.NDArray[np.floating]]
    event_distances: Union[npt.NDArray[np.floating], List[float]]
    map_index: Union[npt.NDArray[np.integer], List[int]]
    event_pvalue: Dict[float, Union[List, npt.NDArray[np.floating]]]

    # General warning flags
    multiple_source_warning_logged: bool = False

    def __init__(
        self,
        signal_events: npt.NDArray[Any],
        parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]],
        dpsi_nbins: int = 101,
        minimum_counts: int = 100,
        spectral_indices: Union[List[float], npt.NDArray[np.floating]] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        angular_cutoff: float = np.pi,
        skymap: Union[npt.NDArray[np.floating], None] = None,
        cache_parameters: bool = True,
        cache_name: str = "./king_parameters_cache.npz",
        remove_weight_outliers=True,
        weight_outlier_percentiles=[0, 95],
        weight_field: str = "ow",
        true_ra_name: str = "trueRa",
        true_dec_name: str = "trueDec",
        true_energy_name: str = "trueE",
    ):
        # Store some of the configuration parameters for this instance.
        # Note that we don't need to store the signal events, dpsi_nbins,
        # or minimum counts since they're only necessary for fitting during
        # initialization andnot for later evaluation. We'll also be storing
        # the parametrization bins later, since the user may have simply
        # passed in a number of bins instead of actual bin edges.
        self.spectral_indices = np.atleast_1d(spectral_indices)
        self.skymap = skymap

        # Set some default values for the event-level parameters.
        self.event_alpha, self.event_beta = {}, {}
        self.event_distances, self.map_index = [], []
        self.event_pvalue = {}

        # Obtain the King distribution parameters for all bins. If we're caching parameters
        # and a cache file exists, load from the cache instead of fitting. Otherwise,
        # run the fitter and potentially cache the results. Note that if we run the fitter,
        # we explicitly set the angular cutoff to pi: this is to ensure that we allow the
        # full histogram to be fit for each bin without artificially setting the PDF to 0
        # for some bins.
        fitted_parameters: Dict[str, npt.NDArray[np.floating]] = {}
        if cache_parameters and (cache_name is not None) and exists(cache_name):
            fitted_parameters_npz = np.load(cache_name, allow_pickle=True)
            for key in fitted_parameters_npz.files:
                fitted_parameters[key] = fitted_parameters_npz[key]
        else:
            fitter = KingPSFFitter(
                signal_events=signal_events,
                parametrization_bins=parametrization_bins,
                dpsi_nbins=dpsi_nbins,
                minimum_counts=minimum_counts,
                spectral_indices=spectral_indices,
                angular_cutoff=np.pi,
                remove_weight_outliers=remove_weight_outliers,
                weight_outlier_percentiles=weight_outlier_percentiles,
                weight_field=weight_field,
                true_ra_name=true_ra_name,
                true_dec_name=true_dec_name,
                true_energy_name=true_energy_name,
            )
            fitted_parameters = fitter.fit_all_bins(verbose=True)
            if cache_parameters and (cache_name is not None):
                np.savez(cache_name, **fitted_parameters)  # type: ignore[arg-type]

        # Store the fitted parameters and bins for later interpolation during PDF evaluation.
        self.parametrization_bins = fitted_parameters["parametrization_bins"]  # type: ignore[assignment]
        try:
            self.parametrization_bins.items()
        except AttributeError:
            self.parametrization_bins = self.parametrization_bins.item()

        # Extract the bin centers and keys for each event. The stored bins are
        # edges, but interpn requires coordinates matching the values shape.
        self.keys, self.bin_centers = [], []
        for key, edges in self.parametrization_bins.items():
            self.keys.append(key)
            self.bin_centers.append((edges[:-1] + edges[1:]) / 2)

        # And grab the fitted alpha/beta arrays
        self.alpha_values = fitted_parameters["alpha"]
        self.beta_values = fitted_parameters["beta"]

        # Instantiate the PDF object. If we have a template, use the template-smoothed
        # PDF; otherwise, use the standard PDF.
        if self.skymap is not None:
            self.nside = hp.npix2nside(len(self.skymap))
            self.template_pdf = TemplateSmearedKingPDF(
                skymap=self.skymap, angular_cutoff=angular_cutoff
            )
        else:
            self.king_pdf = KingPDF(angular_cutoff=angular_cutoff)

        return

    def events_match(self, events: npt.NDArray[Any]):
        return str(self.events) == str(events)

    def set_events(
        self,
        events: npt.NDArray[Any],
        source_ras: Optional[npt.NDArray[np.floating]] = None,
        source_decs: Optional[npt.NDArray[np.floating]] = None,
    ) -> None:
        """Calculate the King distribution parameters (alpha and beta) for a given set of events by interpolating
        the fitted parameters based on the event parameters and the provided binning. Then calculate the pvalues
        for each spectral index for each event and store them so we can interpolate them at runtime.
        """
        if self.events_match(events):
            return

        self.events = events
        self.source_ras = source_ras
        self.source_decs = source_decs

        # The source_ras and source_decs must be specified unless we're using the
        # template PDF. Ensure that this is the case and raise an error if not.
        if self.skymap is None and (source_ras is None or source_decs is None):
            raise ValueError(
                "This instance of the KingSpatialLikelihood was not configured with"
                " a skymap, suggesting that this should be a point source analysis."
                " However, no source locations were provided. Please ensure that"
                " source_ras and source_decs are provided when calling set_events."
            )
        if self.skymap:
            raise NotImplementedError(
                "This instance of KingSpatialLikelihood does not yet support skymaps."
            )

        # Make sure we have a matching number of source_ras and source_decs if we're given multiple sources.
        assert source_ras is not None
        if (source_ras is not None and source_decs is not None) and (
            len(source_ras) != len(source_decs)
        ):
            raise ValueError(
                "The number of source_ras and source_decs must match. Please ensure "
                "that these arrays have the same length when passing into set_events."
            )

        # Calculate the angular distance between events and sources
        self.event_distances = angular_distance(
            events["ra"], events["dec"], source_ras, source_decs
        )
        self.event_mask = self.event_distances < self.king_pdf.angular_cutoff
        self.event_distances = self.event_distances[self.event_mask]
        events = events[self.event_mask]

        # Interpolate to get the alpha and beta values for each spectral index for
        # each event in the given sample.
        def index(centers, values):
            i = np.searchsorted(centers, values).clip(1, len(centers) - 1)
            return np.where(values - centers[i - 1] < centers[i] - values, i - 1, i)

        event_indices = [index(self.bin_centers[i], events[key]) for i, key in enumerate(self.keys)]
        for i, gamma in enumerate(self.spectral_indices):
            self.event_alpha[gamma] = self.alpha_values[i][*event_indices]
            self.event_beta[gamma] = self.beta_values[i][*event_indices]

        # Start calculating the pvalues.
        # TODO: By assuming keys "ra" and "dec" exist and are usable here, we're
        # implicitly building in an IceCube-centric assumption. Ideally, we want
        # to make this code more generic. This will require us to accept the names
        # of these as configurable parameters.

        if (not self.multiple_source_warning_logged) and (len(source_ras) > 1):
            logging.warning(
                "Multiple source positions provided. This has not been tested and"
                " may not work as expected. Please check the results carefully!"
            )
            self.multiple_source_warning_logged = True

        for gamma in self.spectral_indices:
            self.event_pvalue[gamma] = self.king_pdf.pdf(
                self.event_distances, self.event_alpha[gamma], self.event_beta[gamma]
            )  # type: ignore[assignment, arg-type]
        return

    def evaluate_pdf(self, events: npt.NDArray[Any], gamma: float = 2) -> npt.NDArray[np.floating]:
        # If we haven't already calculated the per-event alpha and beta parameters, do so now.
        if not self.events_match(events):
            raise RuntimeError(
                "The events provided to evaluate_pdf do not match the events that were used to calculate the per-event parameters."
                " Please ensure that you call set_events with the same events that you later pass into evaluate_pdf."
            )

        # Interpolate over gamma to get the final result for each event
        idx = np.clip(
            np.searchsorted(self.spectral_indices, gamma) - 1, 0, len(self.spectral_indices) - 2
        )

        gamma_low, gamma_high = self.spectral_indices[idx], self.spectral_indices[idx + 1]
        result = np.zeros(len(self.events))
        result[self.event_mask] = _interp1d(
            gamma,
            gamma_low,
            gamma_high,
            self.event_pvalue[gamma_low],
            self.event_pvalue[gamma_high],
        )
        return result
