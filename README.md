# kingmaker

![Tests](https://github.com/USERNAME/kingmaker/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/USERNAME/kingmaker/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/kingmaker)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

A Python library for working with King/Moffat distributions for modeling point spread functions (PSFs) in high-energy astrophysics.

## Overview

The **King distribution** (also known as the **Moffat distribution**, see [wikipedia](https://en.wikipedia.org/wiki/Moffat_distribution).) is a two-parameter probability distribution commonly used to describe the point spread function (PSF) of astronomical observations. This package uses the exact spherical form:

```
f(theta | alpha, beta) ∝ [1 + (1 - cos(theta)) / (alpha^2 * beta)]^(-beta)
```

where:
- **theta** is the angular separation from the source
- **alpha** is the scale parameter controlling the width of the distribution
- **beta** is the shape parameter controlling the tail behavior (must be > 1)

The substitution `t = 1 - cos(theta)`, `dt = sin(theta) dtheta` yields an exact closed-form CDF, making both normalization and cumulative probabilities analytically tractable at all angular scales. For small angles, `1 - cos(theta) ~ theta^2/2`, recovering the familiar flat-sky form. In the limit beta -> inf the distribution converges to a Gaussian.

The distribution's flexibility in modeling both the core and tail regions makes it particularly well-suited for modeling extended tails to point spread functions. These are common in many instruments in astronomy. The King distribution offers significant advantages over simpler models (e.g., Gaussian or Rayleigh) by providing independent control over the core width (alpha) and tail weight (beta), allowing more accurate modeling of real detector responses.

This package implements the King/Moffat distribution on a sphere. The current implementation includes support for

- KingPDF evaluations with exact analytic normalizations and CDFs
- Numerically calculated "signal subtraction" PDFs
- HEALPix map convolutions and template support using spherical harmonics

## Features

- **Exact normalization**: Closed-form analytic normalization and CDF via spherical geometry, valid at all angular scales
- **Vectorized operations**: NumPy broadcasting for simultaneous evaluation at multiple points
- **JIT-compiled kernels**: All inner-loop functions compiled and cached with numba for near-native performance
- **Multi-dimensional fitting**: Parameterize PSF as function of energy, declination, angular error, etc.
- **Signal subtraction**: Built-in RA marginalization for likelihood-based analyses
- **Template smearing**: Incorporate diffuse backgrounds and Galactic plane effects via spherical harmonics
- **Flexible binning**: Support for both equal-probability and explicit bin edges

## Installation

### Basic Installation

Install directly from the repository:

```bash
git clone git@github.com/mjlarson/kingmaker.git
pip install ./kingmaker
```

### Dependencies

Core dependencies:
- `numpy` - Array operations
- `scipy` - Optimization and interpolation
- `numba` - JIT compilation for performance
- `matplotlib` - Plotting and visualization used in example scripts
- `healpy` - Spherical harmonic operations

## Usage

### Basic PDF Evaluation

```python
import numpy as np
from kingmaker.pdf import KingPDF

# Create a King PDF with full-sphere coverage
king = KingPDF(angular_cutoff=np.pi)

# Define parameters
alpha = np.radians(1.0)  # 1 degree scale
beta = 2.0               # Moderate tail weight

# Evaluate PDF at various angular separations
angles = np.linspace(0, np.radians(10), 100)
pdf_values = king.pdf(angles, alpha, beta)

# Calculate containment radius (68% probability)
from scipy.optimize import brentq
containment_68 = brentq(
    lambda x: king.cdf(x, alpha, beta) - 0.68,
    0, np.pi
)
print(f"68% containment: {np.degrees(containment_68):.2f} degrees")
```

### Fitting PSF Parameters to Monte Carlo

Fit King distribution parameters to simulated signal events as a function of an arbitrary set of observables:

```python
from kingmaker.fitting import KingPSFFitter
import numpy as np

# Load your signal MC events (structured array)
# Must contain: 'ra', 'dec', 'trueRa', 'trueDec'
signal_events = np.load('signal_mc.npy')

# Define parameterization binning
parametrization_bins = {
    'logE': np.linspace(2, 6, 11),            # Energy bins
    'dec': np.arcsin(np.linspace(-1, 1, 11)), # Declination bins
    'angErr': 10                              # Equal-probability bins in angular error
}

# Initialize fitter
fitter = KingPSFFitter(
    signal_events=signal_events,
    parametrization_bins=parametrization_bins,
    dpsi_nbins=51,              # Angular error bins for fitting
    minimum_counts=100,         # Min events per bin
    spectral_indices=[2.0],     # Spectral indices for power-law weighting
    angular_cutoff=np.pi        # Maximum angular error to accept
)

# Run fitting
results = fitter.fit_all_bins(verbose=True)

# Access fitted parameters
alpha_fit = results['alpha']  # Shape: (n_gamma, n_energy, n_dec, n_sigma)
beta_fit = results['beta']

# Get interpolators for continuous evaluation
alpha_interp, beta_interp = fitter.get_interpolator(gamma_index=0)

# Evaluate at arbitrary points
test_point = np.array([[3.5, np.arcsin(0.0), np.radians(1.0)]])  # [logE, dec, sigma]
alpha_value = alpha_interp(test_point)
beta_value = beta_interp(test_point)
```

### Signal-Subtracted Likelihood

`MarginalizedKingPDF` pre-computes the King PDF integrated over right ascension
on a grid and exposes two methods for different use cases:

- **`pdf(x, alpha, beta, source_dec)`** — evaluates the marginalized PDF for a
  *single* source at a given set of event declinations. Returns a dense NumPy
  array. Intended for plotting, diagnostics, and demonstrations.

- **`evaluate(source_decs, event_decs, alpha, beta)`** — evaluates the
  marginalized PDF for *all* (event, source) pairs at once and returns a
  `scipy.sparse.csr_array` of shape `(n_events, n_sources)`. The sparse format
  is efficient because events far from a source (beyond `angular_cutoff`)
  contribute zero and are omitted. This is the method to use in analyses.

```python
import numpy as np
from kingmaker.pdf import MarginalizedKingPDF

# Define source and event parameters.
source_decs = np.radians(np.linspace(-60, 60, 13))  # 13 sources
alpha = np.radians(1.0)
beta = 2.0

mkpdf = MarginalizedKingPDF(
    source_declination=source_decs,
    angular_cutoff=np.radians(10.0),
)

# Single-source dense evaluation — useful for plots and quick checks.
dec_reco = np.radians(np.linspace(-10, 10, 200))
alpha_arr = np.full(200, alpha)
beta_arr = np.full(200, beta)
profile = mkpdf.pdf(dec_reco, alpha_arr, beta_arr, source_dec=np.radians(0.0))

# Batch sparse evaluation over all events × sources — use this in analyses.
n_events = 10_000
rng = np.random.default_rng(42)
ev_decs = np.arcsin(rng.uniform(-1, 1, n_events))
alpha_evt = np.full(n_events, alpha)
beta_evt = np.full(n_events, beta)

pdf_matrix = mkpdf.evaluate(source_decs, ev_decs, alpha_evt, beta_evt)
# pdf_matrix has shape (n_events, n_sources); most entries are zero.
print(f"Nonzero fraction: {pdf_matrix.nnz / pdf_matrix.shape[0] / pdf_matrix.shape[1]:.3%}")

# On repeat calls with the same geometry, pass the previous result as a mask
# to skip the angular-distance masking loop entirely.
pdf_matrix_2 = mkpdf.evaluate(source_decs, ev_decs, alpha_evt, beta_evt, mask=pdf_matrix)
```

### Template Smearing

Apply a HEALPix template (e.g. Galactic diffuse emission) to the PSF via spherical harmonic convolution:

```python
from kingmaker.pdf import TemplateSmearedKingPDF
import healpy as hp

# Load a diffuse template as a HEALPix map
skymap = hp.read_map('galactic_template.fits')

# Create smeared PSF (pre-computes spherical harmonic expansion at init)
king_smeared = TemplateSmearedKingPDF(skymap, angular_cutoff=np.pi)

# Set evaluation coordinates, then convolve
eval_ras  = np.radians([0.0, 45.0, 90.0])
eval_decs = np.radians([0.0, 30.0, -15.0])
king_smeared.set_coordinates(eval_decs, eval_ras)
pdf_vals = king_smeared.convolve_at_grid_point(alpha, beta)
```


## Examples

The `examples/` directory contains Jupyter notebooks demonstrating:

- `basic_demo.ipynb` - King PDF basics, parameter effects, and speed comparisons
- `fitting_demo.ipynb` - Fitting PSF parameters to Monte Carlo simulations
- `template_demo.ipynb` - Working with diffuse templates and template smearing

## References

1. King, I. (1962). " The structure of star clusters. I. an empirical density law" *The Astronomical Journal*, 67, 471.
2. Moffat, A. F. J. (1969). "A theoretical investigation of focal stellar images in the photographic emulsion and application to photographic photometry." *Astronomy and Astrophysics*, 3, 455.
3. Fermi-LAT Collaboration. "LAT Point Spread Function." Fermi Science Support Center. https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms specified in the repository.

## Author

Michael Larson (mlarson@icecube.wisc.edu)

## Citation

If you use `kingmaker` in your research, please cite the appropriate astronomical references above and acknowledge the software in your publications.
