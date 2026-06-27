Examples
========

This page walks through the main workflows the package supports, with
short, copy-pasteable code for each. See :doc:`quickstart` first if you
haven't evaluated a King PDF yet. Each section below links to the full
Jupyter notebook in ``examples/`` for additional plots and benchmarks.

King PDF basics
----------------

:class:`~kingmaker.pdf.KingPDF` evaluates the PDF/CDF (see :doc:`quickstart`),
draws samples, and marginalizes over right ascension for signal-subtraction
likelihoods.

**Sampling angular offsets**

.. code-block:: python

   import numpy as np
   from kingmaker.pdf import KingPDF

   king = KingPDF(angular_cutoff=np.pi)
   alpha, beta = np.radians(1.0), 2.0

   # Angular separations (radians) drawn from the King distribution via
   # inverse CDF. These are offsets from the source, not full positions --
   # combine with a uniformly-random position angle to get a reconstructed
   # (ra, dec) for a given true source position.
   psi = king.sample(10_000, alpha, beta, n_grid=10_000)

**Marginalizing over right ascension**

.. code-block:: python

   source_dec = np.radians(30)
   sindec_bins, pdf_marginalized = king.marginalize(
       source_dec, alpha, beta, threshold=1e-6, nbins=100,
   )
   # pdf_marginalized is the expected signal contribution as a function of
   # sin(declination), for use in a signal-subtraction likelihood term.

*Common options:*

- ``angular_cutoff`` (constructor): truncates the PDF's support and
  renormalizes. Use this to restrict evaluation to a search window instead
  of the full sphere.
- ``n_grid`` (``sample``): size of the CDF lookup grid used for inverse
  transform sampling. Higher values trade memory/setup time for accuracy;
  the default of 10000 gives roughly arcminute accuracy.
- ``threshold`` / ``nbins`` (``marginalize``): ``threshold`` discards
  declination/RA bins whose relative PDF value is negligible (controls
  sparsity); ``nbins=None`` (default) uses adaptive binning sized to
  ``alpha`` instead of a fixed grid.

`basic_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/basic_demo.ipynb>`_
    Parameter effects, normalization checks, and sampling/evaluation speed
    benchmarks.

.. _fitting-psf-parameters:

Fitting PSF parameters from Monte Carlo
----------------------------------------

:class:`~kingmaker.fitting.KingPSFFitter` bins signal Monte Carlo along
arbitrary observables (energy, declination, angular error estimate, ...)
and fits King ``alpha``/``beta`` to the angular-error distribution in each
bin.

.. code-block:: python

   from kingmaker.fitting import KingPSFFitter
   from kingmaker.pdf import KingPDF
   import numpy as np

   # Build a small synthetic signal MC sample standing in for your own
   # simulation output -- a real dataset just needs the same field names.
   # Structured array with at least 'ra', 'dec' (reconstructed) and
   # 'trueRa', 'trueDec' (true) fields, in radians.
   rng = np.random.default_rng(0)
   n = 100_000
   true_logE = rng.uniform(2, 6, n)
   true_dec = np.arcsin(rng.uniform(-1, 1, n))
   true_ra = rng.uniform(0, 2 * np.pi, n)

   psi = KingPDF().sample(n, np.radians(1.0), 2.5, rng=rng)
   phi = rng.uniform(0, 2 * np.pi, n)
   reco_dec = np.arcsin(np.clip(
       np.sin(true_dec) * np.cos(psi) + np.cos(true_dec) * np.sin(psi) * np.cos(phi),
       -1, 1,
   ))
   reco_ra = true_ra + np.arctan2(
       np.sin(phi) * np.sin(psi),
       np.cos(true_dec) * np.cos(psi) - np.sin(true_dec) * np.sin(psi) * np.cos(phi),
   )

   signal_events = np.empty(n, dtype=[
       ("ra", float), ("dec", float),
       ("trueRa", float), ("trueDec", float),
       ("logE", float), ("ow", float), ("trueE", float),
   ])
   signal_events["ra"], signal_events["dec"] = reco_ra, reco_dec
   signal_events["trueRa"], signal_events["trueDec"] = true_ra, true_dec
   signal_events["logE"] = true_logE
   signal_events["ow"] = 1.0
   signal_events["trueE"] = 10**true_logE

   parametrization_bins = {"logE": 5, "dec": 4}  # equal-probability bins

   fitter = KingPSFFitter(
       signal_events=signal_events,
       parametrization_bins=parametrization_bins,
       dpsi_nbins=51,
       minimum_counts=100,
       weight_field="ow",
       spectral_indices=[2.0, 2.5, 3.0],
   )
   results = fitter.fit_all_bins(verbose=True)

   alpha_fit = results["alpha"]  # shape (n_gamma, n_logE, n_dec)
   beta_fit = results["beta"]

   # Continuous evaluation between bin centers:
   alpha_interp, beta_interp = fitter.get_interpolator(gamma_index=0)
   point = np.array([[3.5, np.arcsin(0.0)]])  # [logE, dec]
   alpha_value = alpha_interp(point)

   # Inspect a single bin's fit against its histogram:
   ax = fitter.plot_fit(bin_indices=(2, 2), gamma_index=0)

*Common options:*

- ``parametrization_bins``: each value is either an ``int`` (equal-probability
  bins computed from the MC) or an explicit array of bin edges.
- ``dpsi_nbins``: resolution of the angular-error histogram used in the fit.
- ``minimum_counts``: bins with fewer events than this are skipped entirely
  (left at the default initial guess rather than fit).
- ``remove_weight_outliers`` / ``weight_outlier_percentiles``: drop events
  with extreme weights (by sorted-index percentile, default ``[0, 95]``)
  before fitting, to keep a few outsized weights from destabilizing the fit.
- ``weight_field``: name of the per-event weight field (e.g. ``"ow"``); pass
  ``None`` to use equal weights.
- ``spectral_indices``: gamma values to fit independently. Each gets its own
  fitted alpha/beta grid, used later for interpolation over spectral index
  (see :class:`~kingmaker.wrapper.KingSpatialLikelihood` below).

`fitting_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/fitting_demo.ipynb>`_
    Fitting as a function of energy and declination, with diagnostic plots.

.. _point-source-likelihood:

End-to-end point-source likelihood
-----------------------------------

:class:`~kingmaker.wrapper.KingSpatialLikelihood` wraps
:class:`~kingmaker.fitting.KingPSFFitter` and
:class:`~kingmaker.pdf.KingPDF` behind a single interface: fit (or load
cached fit results) once, then evaluate the PDF per-event many times across
trials.

Continuing with the synthetic ``signal_events`` from the fitting example
above:

.. code-block:: python

   from kingmaker.wrapper import KingSpatialLikelihood
   import numpy as np

   wrapper = KingSpatialLikelihood(
       signal_events=signal_events,
       parametrization_bins=parametrization_bins,
       spectral_indices=[1.0, 2.0, 3.0, 4.0],
       cache_parameters=False,
   )

   # Stand-in "data" events and a source position for one trial.
   data_events = signal_events[:1000]
   source_ra, source_dec = 0.5, 0.2

   # Per trial: cache per-event parameters once, then evaluate as needed.
   wrapper.set_events(
       data_events,
       source_ras=np.array([source_ra]),
       source_decs=np.array([source_dec]),
   )
   pdf_values = wrapper.evaluate_pdf(data_events, gamma=2.0)
   pdf_values_steeper = wrapper.evaluate_pdf(data_events, gamma=2.5)  # interpolated

*Common options:*

- ``cache_parameters`` / ``cache_name``: when ``True`` and ``cache_name``
  exists on disk, fitting is skipped entirely and parameters are loaded from
  the cache; otherwise the fitter runs and (if ``cache_parameters``) saves
  its results there. Use this to avoid refitting across repeated runs/trials.
- ``spectral_indices``: the gamma grid that gets fit up front;
  ``evaluate_pdf(events, gamma=...)`` interpolates between the two bracketing
  values, so pick a range that covers the spectral indices you plan to test.
- ``parametrization_bins``: same ``int``-or-edges rules as
  :class:`~kingmaker.fitting.KingPSFFitter`.
- **Gotcha:** ``evaluate_pdf`` requires ``set_events`` to have been called
  first with the *same* ``events`` array, and raises ``RuntimeError``
  otherwise. Calling ``set_events`` repeatedly with identical events/sources
  is a cheap no-op, so it's safe to call once per trial unconditionally.

`likelihood_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/likelihood_demo.ipynb>`_
    Full walkthrough including event setup and spectral-index interpolation.

.. _template-smearing:

Template smearing for diffuse/extended sources
------------------------------------------------

:class:`~kingmaker.pdf.TemplateSmearedKingPDF` convolves a HEALPix template
map (e.g. Galactic diffuse emission) with the King PSF using a
spherical-harmonic expansion, avoiding a per-event real-space convolution.

.. code-block:: python

   from kingmaker.pdf import TemplateSmearedKingPDF
   import numpy as np
   import healpy as hp

   # A small synthetic HEALPix map standing in for a real diffuse template
   # (e.g. Fermi-LAT diffuse emission) -- concentrated near the equator like
   # a toy Galactic plane. Normalized to integrate to 1 internally.
   nside = 32
   colat, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
   skymap = np.exp(-((colat - np.pi / 2) ** 2) / (2 * np.radians(10) ** 2))

   tskp = TemplateSmearedKingPDF(
       skymap=skymap,
       interpolation_method="nearest",
       memory_limit_gb=1.0,
   )

   alpha, beta = np.radians(5.0), 2.0

   # Full convolved map (e.g. for plotting):
   convolved_map = tskp.convolve_map(alpha, beta)

   # Fast evaluation at a fixed set of source positions instead:
   eval_decs = np.radians([0.0, 30.0, -15.0])
   eval_ras = np.radians([0.0, 45.0, 90.0])
   tskp.set_coordinates(eval_decs, eval_ras)
   pdf_at_sources = tskp.convolve_at_grid_point(alpha, beta)

*Common options:*

- ``interpolation_method``: ``"nearest"`` (default) snaps each
  ``(alpha, beta)`` to the closest precomputed grid point -- cheaper, and
  events landing in the same grid cell reuse the same convolution.
  ``"linear"`` bilinearly interpolates in log(alpha)/log(beta) space for
  smoother variation at extra cost.
- ``lmax``: maximum spherical harmonic degree, defaulting to ``3 * nside - 1``
  of the input map. Lower it to reduce memory/compute at the cost of
  angular detail in the convolution.
- ``memory_limit_gb``: caps the batch size used when precomputing spherical
  harmonics for many ``set_coordinates`` points at once.
- ``points_alpha`` / ``points_beta``: the grid of King parameters over which
  the convolution is precomputed (100 log-spaced points each, by default).
  Increase density here if using ``"linear"`` interpolation and seeing
  visible discretization.

`template_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/template_demo.ipynb>`_
    Convolution of a Fermi-LAT diffuse template with the King PSF, including
    performance benchmarks against healpy's Gaussian smoothing.
