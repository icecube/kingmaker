Quickstart
==========

This page gets you from a fresh install to a working King PDF evaluation in
a few lines. See :doc:`examples` for the deeper workflows (fitting PSF
parameters from Monte Carlo, the end-to-end likelihood wrapper, and template
smearing).

Evaluate a King PDF
--------------------

:class:`~kingmaker.pdf.KingPDF` is the core class. It is constructed with an
``angular_cutoff`` (in radians, defaulting to :math:`\pi`, i.e. the full
sphere) and exposes ``pdf``/``cdf`` methods that accept the King distribution
parameters ``alpha`` (core scale, radians) and ``beta`` (tail weight, must be
greater than 1):

.. code-block:: python

   import numpy as np
   from kingmaker.pdf import KingPDF

   # Full-sphere coverage; restrict via angular_cutoff to limit the
   # PDF's support (e.g. to a search window) and renormalize accordingly.
   king = KingPDF(angular_cutoff=np.pi)

   alpha = np.radians(1.0)  # 1 degree core scale
   beta = 2.0               # moderate tail weight

   angles = np.linspace(0, np.radians(10), 100)
   pdf_values = king.pdf(angles, alpha, beta)
   cdf_values = king.cdf(angles, alpha, beta)

Compute a containment radius
-----------------------------

Because the CDF is monotonic, standard root-finding gives containment radii
directly:

.. code-block:: python

   from scipy.optimize import brentq

   containment_68 = brentq(
       lambda x: king.cdf(x, alpha, beta) - 0.68,
       0, np.pi,
   )
   print(f"68% containment radius: {np.degrees(containment_68):.2f} degrees")

Next steps
----------

- For fitting ``alpha``/``beta`` to your own Monte Carlo, see
  :ref:`fitting-psf-parameters` in :doc:`examples`.
- For a ready-to-use point-source likelihood interface, see
  :ref:`point-source-likelihood` in :doc:`examples`.
- For diffuse/extended sources smeared with this PSF, see
  :ref:`template-smearing` in :doc:`examples`.
