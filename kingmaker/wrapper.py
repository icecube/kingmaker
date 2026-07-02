from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt

from os.path import exists
import logging
import numpy as np

from scipy.sparse import csr_array

from .pdf import KingPDF, MarginalizedKingPDF
from .fitting import KingPSFFitter
from .utils import angular_distance, _pre_mask_and_distance, _interp1d


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

    king_pdf: KingPDF

    # Marginalized PDF (optional — enabled by passing marg_source_decs to __init__)
    mkpdf: Optional[MarginalizedKingPDF]
    _marg_source_decs: Optional[npt.NDArray[np.floating]]
    _marg_matrices: List[csr_array]

    # Source-level information
    source_ras: Optional[npt.NDArray[Any]] = None
    source_decs: Optional[npt.NDArray[Any]] = None

    # Have some place to cache the per-event information so we don't need to
    # recalculate it every time we evaluate the PDF.
    events: Optional[Any] = None
    event_distances: Union[npt.NDArray[np.floating], List[float]]
    map_index: Union[npt.NDArray[np.integer], List[int]]
    _pdf_matrices: List[csr_array]

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
        cache_parameters: bool = True,
        cache_name: str = "./king_parameters_cache.npz",
        remove_weight_outliers=True,
        weight_outlier_percentiles=(0, 95),
        weight_field: str = "ow",
        true_ra_name: str = "trueRa",
        true_dec_name: str = "trueDec",
        true_energy_name: str = "trueE",
        enable_marginalization: bool = False,
        marginalization_source_decs: Optional[npt.NDArray[np.floating]] = None,
        marginalization_angular_cutoff: Optional[float] = None,
        marginalization_points_alpha: Optional[npt.NDArray[np.floating]] = None,
        marginalization_points_beta: Optional[npt.NDArray[np.floating]] = None,
        marginalization_n_signed_delta_dec: int = 200,
        marginalization_n_ra_bins: int = 100,
    ):
        # Store some of the configuration parameters for this instance.
        # Note that we don't need to store the signal events, dpsi_nbins,
        # or minimum counts since they're only necessary for fitting during
        # initialization andnot for later evaluation. We'll also be storing
        # the parametrization bins later, since the user may have simply
        # passed in a number of bins instead of actual bin edges.
        self.spectral_indices = np.atleast_1d(spectral_indices)

        # Set some default values for the event-level parameters.
        self.event_distances, self.map_index = [], []
        self._pdf_matrices: List[csr_array] = []

        # Obtain the King distribution parameters for all bins. If we're caching parameters
        # and a cache file exists, load from the cache instead of fitting. Otherwise,
        # run the fitter and potentially cache the results. When running the fitter,
        # angular_cutoff is set to pi so every bin's full angular error distribution is fit.
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

        # Instantiate the PDF object.
        self.king_pdf = KingPDF(angular_cutoff=angular_cutoff)

        # Precompute the normalization constant for every (gamma, bin) combination
        # on the fitted grid once at init. set_events looks up per-event norms from
        # this array using the same nearest-bin indices as alpha/beta.
        self.norm_values = self.king_pdf.norm(self.alpha_values, self.beta_values)

        # Optionally build the RA-marginalized PDF for signal-subtraction likelihoods.
        self.mkpdf = None
        self._marg_source_decs = None
        self._marg_matrices = []
        if enable_marginalization:
            if marginalization_source_decs is None:
                raise ValueError(
                    "enable_marginalization=True requires marginalization_source_decs."
                )
            self._marg_source_decs = np.asarray(marginalization_source_decs, dtype=np.float64)
            cutoff = (
                marginalization_angular_cutoff
                if marginalization_angular_cutoff is not None
                else angular_cutoff
            )
            kwargs: Dict[str, Any] = {}
            if marginalization_points_alpha is not None:
                kwargs["points_alpha"] = marginalization_points_alpha
            if marginalization_points_beta is not None:
                kwargs["points_beta"] = marginalization_points_beta
            self.mkpdf = MarginalizedKingPDF(
                source_declination=self._marg_source_decs,
                angular_cutoff=cutoff,
                n_signed_delta_dec=marginalization_n_signed_delta_dec,
                n_ra_bins=marginalization_n_ra_bins,
                **kwargs,
            )
        return

    def _events_match(self, events: npt.NDArray[Any]) -> bool:
        if self.events is None:
            return False
        if events is None:
            return True
        if len(self.events) != len(events):
            return False
        result = np.array_equal(self.events["ra"][::10], events["ra"][::10])
        result &= np.array_equal(self.events["dec"][::10], events["dec"][::10])
        return result

    def _sources_match(self, source_ras: npt.NDArray[Any], source_decs: npt.NDArray[Any]) -> bool:
        if self.source_ras is None:
            return False
        if self.source_decs is None:
            return False
        if source_ras is None:
            return True
        if source_decs is None:
            return True
        if len(self.source_ras) != len(source_ras):
            return False
        if len(self.source_decs) != len(source_decs):
            return False
        return np.array_equal(self.source_ras, source_ras) and np.array_equal(
            self.source_decs, source_decs
        )

    def set_events(
        self,
        events: npt.NDArray[Any],
        source_ras: Optional[npt.NDArray[np.floating]],
        source_decs: Optional[npt.NDArray[np.floating]],
    ) -> None:
        """
        Cache per-event King PDF values for each spectral index ahead of a call
        to :meth:`evaluate_pdf`.

        For each event, the nearest parametrization bin is looked up to obtain
        per-event alpha/beta parameters, the angular distance to the source is
        computed, and the King PDF is evaluated and cached for every spectral
        index in ``spectral_indices``. This must be called before
        :meth:`evaluate_pdf`. Calling it again with the same ``events``,
        ``source_ras``, and ``source_decs`` as the previous call is a cheap
        no-op, so it is safe to call once per trial without checking first.

        Parameters
        ----------
        events : structured array
            Data events to evaluate. Must contain ``ra`` and ``dec`` fields
            (reconstructed equatorial coordinates, in radians) plus any fields
            referenced by ``parametrization_bins``.
        source_ras : ndarray
            Source right ascension(s) in radians.
        source_decs : ndarray
            Source declination(s) in radians. Must have the same length as
            ``source_ras``.

        Raises
        ------
        ValueError
            If ``source_ras``/``source_decs`` are not provided, or their
            lengths do not match.

        Notes
        -----
        Support for multiple simultaneous sources is experimental and logs a
        one-time warning; results should be checked carefully in that case.
        """
        if self._events_match(events) and self._sources_match(source_ras, source_decs):
            return

        self.events = events
        self.source_ras = source_ras
        self.source_decs = source_decs

        # Make sure we have a matching number of source_ras and source_decs if we're given multiple sources.
        if (source_ras is None) and (source_decs is None):
            raise ValueError(
                "No source_ras and source_decs were provided to the set_eventsfunction."
            )
        if (source_ras is None or source_decs is None) or (len(source_ras) != len(source_decs)):
            raise ValueError(
                "The number of source_ras and source_decs must match. Please ensure "
                "that these arrays have the same length when passing into set_events."
            )

        if (not self.multiple_source_warning_logged) and (len(source_ras) > 1):
            logging.warning(
                "Multiple source positions provided. This has not been tested and"
                " may not work as expected. Please check the results carefully!"
            )
            self.multiple_source_warning_logged = True

        # Calculate angular distances and build event_mask. For the common
        # single-source case with a sub-pi cutoff, a single compiled numba pass
        # does the rectangular (dec, RA) pre-filter and the haversine together,
        # reading each event's ra/dec only once.
        cutoff = self.king_pdf.angular_cutoff
        if len(source_ras) == 1 and cutoff < np.pi:
            src_ra = float(source_ras[0])
            src_dec = float(source_decs[0])
            ra_span = min(cutoff / max(abs(np.cos(src_dec)), np.sin(cutoff)), np.pi)
            dists_all = _pre_mask_and_distance(
                events["ra"], events["dec"], src_ra, src_dec, cutoff, ra_span
            )
            self.event_mask = dists_all >= 0
            self.event_distances = dists_all[self.event_mask]
        else:
            all_dists = angular_distance(events["ra"], events["dec"], source_ras, source_decs)
            self.event_mask = all_dists < cutoff
            self.event_distances = all_dists[self.event_mask]

        event_rows = np.where(self.event_mask)[0]
        all_alpha, all_beta, all_norm = self._lookup_event_grid(events)
        self._pdf_matrices = []
        for i in range(len(self.spectral_indices)):
            # Evaluate the King PDF directly using the precomputed per-bin norm.
            # alpha/beta come from the fitted grid (validated at fit time) and
            # event_distances are already within angular_cutoff by construction.
            values = self.king_pdf.pdf_from_norm(
                self.event_distances, all_alpha[i], all_beta[i], all_norm[i]
            )
            self._pdf_matrices.append(
                csr_array(
                    (values, (event_rows, np.zeros(len(event_rows), dtype=np.intp))),
                    shape=(len(events), 1),
                    dtype=np.float64,
                )
            )

        # Marginalized path: precompute one sparse (n_events, n_sources) matrix per
        # spectral index. The first call establishes the sparsity structure; all
        # subsequent calls pass that result as mask= so interpn operates on the same
        # fixed (event, source) pairs. The mask path in MarginalizedKingPDF.evaluate()
        # does not filter values > 0, so every matrix in _marg_matrices is guaranteed to
        # share identical .indices and .indptr — a requirement for the scalar lerp
        # on .data in evaluate_marginalized_pdf.
        if self.mkpdf is not None:
            all_alpha_full, all_beta_full = self._lookup_all_events_grid(events)
            self._marg_matrices = []
            mask_marg: Optional[csr_array] = None
            for i in range(len(self.spectral_indices)):
                mat = self.mkpdf.evaluate(
                    self._marg_source_decs,
                    events["dec"],
                    all_alpha_full[i],
                    all_beta_full[i],
                    mask=mask_marg,
                )
                if mask_marg is None:
                    mask_marg = mat
                self._marg_matrices.append(mat)
        return

    def _lookup_event_grid(self, events):
        """
        Nearest-bin lookup of alpha, beta, and norm for each (unmasked) event.

        Shared by :meth:`get_alpha_beta` and :meth:`set_events` so the
        nearest-bin index computation is only ever done once per call.
        """

        # Nearest-bin lookup. Extracts each field individually after masking.
        def index(centers, values):
            i = np.searchsorted(centers, values).clip(1, len(centers) - 1)
            return np.where(values - centers[i - 1] < centers[i] - values, i - 1, i)

        event_indices = tuple(
            index(self.bin_centers[i], events[key][self.event_mask])
            for i, key in enumerate(self.keys)
        )

        idx = (slice(None), *event_indices)
        return self.alpha_values[idx], self.beta_values[idx], self.norm_values[idx]

    def _lookup_all_events_grid(self, events):
        """
        Nearest-bin lookup of alpha and beta for every event, without applying
        ``event_mask``.

        Used by :meth:`set_events` to supply per-event PSF parameters for
        :meth:`MarginalizedKingPDF.evaluate`, which performs its own angular
        masking internally and therefore needs parameters for the full event set.

        Returns
        -------
        alpha : ndarray, shape (n_gamma, n_events)
        beta  : ndarray, shape (n_gamma, n_events)
        """

        def index(centers, values):
            i = np.searchsorted(centers, values).clip(1, len(centers) - 1)
            return np.where(values - centers[i - 1] < centers[i] - values, i - 1, i)

        event_indices = tuple(
            index(self.bin_centers[i], events[key]) for i, key in enumerate(self.keys)
        )
        idx = (slice(None), *event_indices)
        return self.alpha_values[idx], self.beta_values[idx]

    def get_alpha_beta(self, events):
        """
        Look up fitted alpha/beta parameters for each event via nearest-bin lookup.

        Used internally by :meth:`set_events`. Useful for inspecting the fitted
        King parameters assigned to specific events.

        Parameters
        ----------
        events : structured array
            Events to look up. Must contain the fields referenced by
            ``parametrization_bins``. Only events selected by the mask set in
            the most recent :meth:`set_events` call are returned.
        Returns
        -------
        alpha : ndarray, shape (n_gamma, n_masked_events)
            Fitted alpha values for each spectral index and event.
        beta : ndarray, shape (n_gamma, n_masked_events)
            Fitted beta values for each spectral index and event.
        """
        alpha, beta, _ = self._lookup_event_grid(events)
        return alpha, beta

    def get_alpha_beta_gamma(self, gamma, events=None, alpha=None, beta=None):
        """
        Get alpha/beta at a given spectral index, interpolating if necessary.

        If ``gamma`` matches one of the spectral indices the parameters were
        fit at, the stored values are returned directly. Otherwise, alpha and
        beta are linearly interpolated between the two bracketing spectral
        indices.

        Parameters
        ----------
        gamma : float
            Spectral index at which to evaluate alpha/beta. Can be any value;
            values outside the range of ``spectral_indices`` are extrapolated
            from the nearest bracketing pair.
        events : structured array, optional
            Events to look up alpha/beta for via :meth:`get_alpha_beta`. Only
            used if ``alpha``/``beta`` are not already provided.
        alpha : ndarray, optional
            Pre-computed alpha values for all spectral indices, e.g. from
            :meth:`get_alpha_beta`. If provided, ``events`` is ignored.
        beta : ndarray, optional
            Pre-computed beta values for all spectral indices, matching
            ``alpha``.

        Returns
        -------
        alpha_gamma : ndarray or float
            Alpha value(s) at the requested spectral index.
        beta_gamma : ndarray or float
            Beta value(s) at the requested spectral index.
        """
        if alpha is None:
            assert events is not None
            alpha, beta = self.get_alpha_beta(events)
        assert len(alpha) == len(beta)

        # If we have this gamma, just return it. Make sure to use copy()
        # so the caller doesn't get a reference into our array.
        # searchsorted with side="left" returns the exact index of an exact match.
        if gamma in self.spectral_indices:
            gamma_idx = np.searchsorted(self.spectral_indices, gamma)
            return alpha[gamma_idx].copy(), beta[gamma_idx].copy()

        # Clamp so the bracketing pair is always within the stored range.
        idx = np.clip(
            np.searchsorted(self.spectral_indices, gamma) - 1, 0, len(self.spectral_indices) - 2
        )
        gamma_low, gamma_high = self.spectral_indices[idx], self.spectral_indices[idx + 1]

        return (
            _interp1d(gamma, gamma_low, gamma_high, alpha[idx], alpha[idx + 1]),
            _interp1d(gamma, gamma_low, gamma_high, beta[idx], beta[idx + 1]),
        )

    def evaluate_pdf(self, events: npt.NDArray[Any], gamma: float = 2) -> csr_array:
        """
        Evaluate the King PDF at each event's position for a given spectral index.

        Uses the sparse PDF matrices cached by the most recent call to
        :meth:`set_events`, interpolating over spectral index as needed via a
        scalar lerp on the underlying ``.data`` arrays — O(nnz) and cache-warm.

        Parameters
        ----------
        events : structured array
            Data events to evaluate. Must be the same events most recently
            passed to :meth:`set_events`.
        gamma : float, optional
            Spectral index at which to evaluate the PDF. Default is 2.0.
            Interpolated between the bracketing values in ``spectral_indices``
            if not an exact match.

        Returns
        -------
        csr_array, shape (n_events, 1)
            Sparse PDF matrix. Nonzero entries hold the King PDF value
            (probability/steradian) for each event within the angular cutoff;
            events outside the cutoff are implicitly zero.

        Raises
        ------
        RuntimeError
            If ``events`` does not match the events passed to the most recent
            call to :meth:`set_events`.
        """
        if not self._events_match(events):
            raise RuntimeError(
                "The events provided to evaluate_pdf do not match the events that were used to calculate the per-event parameters."
                " Please ensure that you call set_events with the same events that you later pass into evaluate_pdf."
            )

        idx = np.clip(
            np.searchsorted(self.spectral_indices, gamma) - 1,
            0,
            len(self.spectral_indices) - 2,
        )

        gamma_low = self.spectral_indices[idx]
        gamma_high = self.spectral_indices[idx + 1]
        t = float(gamma - gamma_low) / float(gamma_high - gamma_low)
        lo = self._pdf_matrices[idx]
        hi = self._pdf_matrices[idx + 1]
        result = lo.copy()
        result.data[:] = (1.0 - t) * lo.data + t * hi.data
        return result.tocsr()

    def evaluate_marginalized_pdf(self, events: npt.NDArray[Any], gamma: float = 2) -> csr_array:
        """
        Evaluate the RA-marginalized King PDF for all (event, source) pairs.

        Returns a sparse ``(n_events, n_sources)`` matrix whose nonzero entries
        are the signal-subtraction PDF values at the requested spectral index.
        The result is obtained by linearly interpolating the precomputed sparse
        matrices from :meth:`set_events` directly on their underlying ``.data``
        arrays — both bracketing matrices share the same sparsity structure, so
        element-wise interpolation is valid and O(nnz).

        Must be called after :meth:`set_events` with the same ``events`` array.
        Requires ``enable_marginalization=True`` at construction.

        Parameters
        ----------
        events : structured array
            Data events. Must match the events passed to the most recent
            :meth:`set_events` call.
        gamma : float, optional
            Spectral index at which to evaluate. Interpolated between the two
            nearest values in ``spectral_indices``. Default is 2.0.

        Returns
        -------
        csr_array, shape (n_events, n_sources)
            Sparse marginalized PDF matrix. Zero entries correspond to
            (event, source) pairs separated by more than the marginalization
            angular cutoff.

        Raises
        ------
        RuntimeError
            If marginalization was not enabled at construction, if
            :meth:`set_events` has not been called, or if ``events`` does not
            match the cached event set.
        """
        if self.mkpdf is None:
            raise RuntimeError(
                "Marginalized PDF evaluation requires enable_marginalization=True "
                "and marginalization_source_decs to be set at construction."
            )
        if not self._events_match(events):
            raise RuntimeError(
                "The events provided to evaluate_marginalized_pdf do not match "
                "the events passed to the most recent set_events call."
            )
        if not self._marg_matrices:
            raise RuntimeError("No marginalized PDF matrices are cached. Call set_events first.")

        idx = np.clip(
            np.searchsorted(self.spectral_indices, gamma) - 1,
            0,
            len(self.spectral_indices) - 2,
        )
        gamma_low = self.spectral_indices[idx]
        gamma_high = self.spectral_indices[idx + 1]
        t = float(gamma - gamma_low) / float(gamma_high - gamma_low)

        lo = self._marg_matrices[idx]
        hi = self._marg_matrices[idx + 1]
        result = lo.copy()
        result.data[:] = (1.0 - t) * lo.data + t * hi.data
        return result.tocsr()


# class KingTemplateLikelihood:
#     # Configuration parameters
#     parametrization_bins: Dict[str, npt.NDArray[np.floating]]
#     spectral_indices: npt.NDArray[np.floating]
#     angular_cutoff: float
#     cache_parameters: bool = True
#     cache_name: str = "king_parameters_cache.npz"

#     # Store an instance of the PDF class to use for evaluations. This will be
#     # either a KingPDF for standard point source searches
#     king_pdf: KingTemplatePDF

#     # Source-level information
#     skymap: npt.NDArray[np.floating]
#     convolution_dtype: Any = np.float32
#     convolution_alphas: npt.NDArray[np.floating]
#     convolution_alphas: npt.NDArray[np.floating]
#     convolved_skymaps: npt.NDArray[np.floating]

#     # Have some place to cache the per-event information so we don't need to
#     # recalculate it every time we evaluate the PDF.
#     events: Optional[Any] = None
#     event_convolution_indices = np.NDArray[np.integer]

#     def __init__(
#         self,
#         skymap: npt.NDArray[np.floating],
#         signal_events: npt.NDArray[Any],
#         parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]],
#         convolution_nside : int = 128,
#         convolution_alphas : Union[npt.NDArray, int] = 10,
#         convolution_betas : Union[npt.NDArray, int] = 10,
#         convolution_dtype : Any = np.float32,
#         dpsi_nbins: int = 101,
#         minimum_counts: int = 100,
#         spectral_indices: Union[List[float], npt.NDArray[np.floating]] = [
#             1.0,
#             2.0,
#             3.0,
#             4.0,
#         ],
#         angular_cutoff: float = np.pi,
#         cache_parameters: bool = True,
#         cache_name: str = "./king_parameters_cache.npz",
#         remove_weight_outliers=True,
#         weight_outlier_percentiles=(0, 95),
#         weight_field: str = "ow",
#         true_ra_name: str = "trueRa",
#         true_dec_name: str = "trueDec",
#         true_energy_name: str = "trueE",
#     ):
#         # Store some of the configuration parameters for this instance.
#         # Note that we don't need to store the signal events, dpsi_nbins,
#         # or minimum counts since they're only necessary for fitting during
#         # initialization andnot for later evaluation. We'll also be storing
#         # the parametrization bins later, since the user may have simply
#         # passed in a number of bins instead of actual bin edges.
#         self.spectral_indices = np.atleast_1d(spectral_indices)

#         # Set some default values for the event-level parameters.
#         self.event_pvalue = {}

#         # Obtain the King distribution parameters for all bins. If we're caching parameters
#         # and a cache file exists, load from the cache instead of fitting. Otherwise,
#         # run the fitter and potentially cache the results. Note that if we run the fitter,
#         # we explicitly set the angular cutoff to pi: this is to ensure that we allow the
#         # full histogram to be fit for each bin without artificially setting the PDF to 0
#         # for some bins.
#         fitted_parameters: Dict[str, npt.NDArray[np.floating]] = {}
#         if cache_parameters and (cache_name is not None) and exists(cache_name):
#             fitted_parameters_npz = np.load(cache_name, allow_pickle=True)
#             for key in fitted_parameters_npz.files:
#                 fitted_parameters[key] = fitted_parameters_npz[key]
#         else:
#             fitter = KingPSFFitter(
#                 signal_events=signal_events,
#                 parametrization_bins=parametrization_bins,
#                 dpsi_nbins=dpsi_nbins,
#                 minimum_counts=minimum_counts,
#                 spectral_indices=spectral_indices,
#                 angular_cutoff=np.pi,
#                 remove_weight_outliers=remove_weight_outliers,
#                 weight_outlier_percentiles=weight_outlier_percentiles,
#                 weight_field=weight_field,
#                 true_ra_name=true_ra_name,
#                 true_dec_name=true_dec_name,
#                 true_energy_name=true_energy_name,
#             )
#             fitted_parameters = fitter.fit_all_bins(verbose=True)
#             if cache_parameters and (cache_name is not None):
#                 np.savez(cache_name, **fitted_parameters)  # type: ignore[arg-type]

#         # Store the fitted parameters and bins for later interpolation during PDF evaluation.
#         self.parametrization_bins = fitted_parameters["parametrization_bins"]  # type: ignore[assignment]
#         try:
#             self.parametrization_bins.items()
#         except AttributeError:
#             self.parametrization_bins = self.parametrization_bins.item()

#         # Extract the bin centers and keys for each event. The stored bins are
#         # edges, but interpn requires coordinates matching the values shape.
#         self.keys, self.bin_centers = [], []
#         for key, edges in self.parametrization_bins.items():
#             self.keys.append(key)
#             self.bin_centers.append((edges[:-1] + edges[1:]) / 2)

#         # And grab the fitted alpha/beta arrays
#         self.alpha_values = fitted_parameters["alpha"]
#         self.beta_values = fitted_parameters["beta"]

#         # Instantiate the PDF object.
#         self.king_pdf = KingTemplatePDF(angular_cutoff=angular_cutoff)
#         self.convolution_nside = convolution_nside
#         self.convolution_alphas = np.sort(convolution_alphas)
#         self.convolution_betas = np.sort(convolution_betas)
#         self.convolution_dtype = convolution_dtype

#         # We can now take the alpha/beta bins from the fitter and directly convert them to
#         # the correct indicies in convolution_alphas and convolution_betas.
#         # TODO: Should these be nearest-neighbors instead of flooring?
#         self.alpha_values_idx = np.searchsorted(self.convolution_alphas,
#                                                 self.alpha_values) - 1
#         self.beta_values_idx = np.searchsorted(self.convolution_betas,
#                                                self.beta_values) - 1

#         # We can also just do the convolutions now since we know the grid.
#         shape = (len(self.alpha_values), len(self.beta_values), hp.nside2npix(self.convolution_nsize))

#         # Warn the user if this is more than 1 GB...
#         expected_size = np.prod(shape) * self.convolution_dtype().nbytes
#         if expected_size / 1024**3 > 1:
#             print(f"WARNING: Requested shape (gamma, alpha, beta, skymap) = {shape}"
#                   f" with dtype {self.convolution_dtype}. This will give a total array"
#                   f" size of {expected_size} GB.")
#         if expected_size / 1024**3 > 4:
#             raise MemoryError(f"Requested shape (gamma, alpha, beta, skymap) = {shape}"
#                               f" with dtype {self.convlution_dtype} will have a total"
#                               f" size of {expected_size}. This seems unreasonable, so"
#                               " I'm kicking this back to you to reconsider.")

#         self.convolved_skymaps = np.empty(shape, dtype=self.convolution_dtype)

#         for index in np.nditer(shape[:-1]):
#             i, j = index
#             self.convolved_skymaps[i,j] = self.king_pdf.convolve_map(
#                 self.alpha_values[i], self.beta_values[j])
#         return

#     def _events_match(self, events: npt.NDArray[Any]) -> bool:
#         if self.events is None:
#             return False
#         if events is None:
#             return True
#         if len(self.events) != len(events):
#             return False
#         result = np.array_equal(self.events["ra"][::10], events["ra"][::10])
#         result &= np.array_equal(self.events["dec"][::10], events["dec"][::10])
#         return result

#     def set_events(
#         self,
#         events: npt.NDArray[Any],
#     ) -> None:
#         """Calculate per-event pvalues for each spectral index by interpolating
#         the King-convolved templates at the nearest parametrization bin for each event.
#         """
#         if self._events_match(events):
#             return

#         self.events = events

#         # Make sure we have a matching number of source_ras and source_decs if we're given multiple sources.
#         if (source_ras is None) and (source_decs is None):
#             raise ValueError(
#                 "No source_ras and source_decs were provided to the set_eventsfunction."
#             )
#         if (source_ras is None or source_decs is None) or (len(source_ras) != len(source_decs)):
#             raise ValueError(
#                 "The number of source_ras and source_decs must match. Please ensure "
#                 "that these arrays have the same length when passing into set_events."
#             )

#         # Nearest-bin lookup. These map the events from their parametrization bins to
#         # the correct healpix bins. We'll then do the gamma lookup later.
#         def index(centers, values):
#             i = np.searchsorted(centers, values).clip(1, len(centers) - 1)
#             return np.where(values - centers[i - 1] < centers[i] - values, i - 1, i)

#         event_indices = tuple(
#             index(self.bin_centers[i], events[key][self.event_mask])
#             for i, key in enumerate(self.keys)
#         )

#         self.event_convolution_indices = np.empty((len(self.spectral_indices),
#                                                    len(self.events)), dtype=int)
#         for i in range(len(self.spectral_indices)):
#             .......................

#         return


#     def evaluate_pdf(self, events: npt.NDArray[Any], gamma: float = 2) -> npt.NDArray[np.floating]:
#         # If we haven't already calculated the per-event alpha and beta parameters, do so now.
#         if not self._events_match(events):
#             raise RuntimeError(
#                 "The events provided to evaluate_pdf do not match the events that were used to calculate the per-event parameters."
#                 " Please ensure that you call set_events with the same events that you later pass into evaluate_pdf."
#             )

#         # Interpolate over gamma to get the final result for each event
#         idx = np.clip(
#             np.searchsorted(self.spectral_indices, gamma) - 1, 0, len(self.spectral_indices) - 2
#         )

#         gamma_low, gamma_high = self.spectral_indices[idx], self.spectral_indices[idx + 1]
#         result = np.zeros(len(self.events))
#         result[self.event_mask] = _interp1d(
#             gamma,
#             gamma_low,
#             gamma_high,
#             self.event_pvalue[gamma_low],
#             self.event_pvalue[gamma_high],
#         )
#         return result
