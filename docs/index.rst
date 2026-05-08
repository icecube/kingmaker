kingmaker
=========

A Python library for working with King/Moffat distributions for modeling
point spread functions (PSFs) in high-energy astrophysics.

The **King distribution** is a two-parameter family commonly used to describe
PSFs with independent control over the core width (α) and tail weight (β):

.. math::

   f(\theta \mid \alpha, \beta) \propto \left[1 + \frac{1 - \cos\theta}{\alpha^2 \beta}\right]^{-\beta}

The substitution :math:`t = 1 - \cos\theta` yields an exact closed-form CDF
via a standard power-law integral, valid at all angular scales. For small
angles, :math:`1 - \cos\theta \approx \theta^2/2`, recovering the flat-sky form.

This package implements the King distribution on a sphere, including exact
analytic normalization, parameter fitting from simulation, and
spherical-harmonic convolution with HEALPix template maps.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
