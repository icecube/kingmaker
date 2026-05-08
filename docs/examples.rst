Examples
========

The ``examples/`` directory contains Jupyter notebooks demonstrating typical workflows.

`basic_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/basic_demo.ipynb>`_
    King PDF basics, parameter effects, normalization, sampling, and speed benchmarks
    for :class:`~kingmaker.pdf.KingPDF`.

`fitting_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/fitting_demo.ipynb>`_
    Fitting King PSF parameters to Monte Carlo simulations using
    :class:`~kingmaker.fitting.KingPSFFitter` as a function of energy and declination.

`template_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/template_demo.ipynb>`_
    Spherical-harmonic convolution of a HEALPix template with the King PSF using
    :class:`~kingmaker.pdf.TemplateSmearedKingPDF`, including performance benchmarks.

`likelihood_demo.ipynb <https://github.com/mjlarson/kingmaker/blob/main/examples/likelihood_demo.ipynb>`_
    End-to-end likelihood analysis using the :class:`~kingmaker.wrapper.KingSpatialLikelihood`
    wrapper, covering event setup, spectral index interpolation, and PDF evaluation.
