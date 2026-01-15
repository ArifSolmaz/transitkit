TransitKit Documentation
========================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

**Professional Exoplanet Transit Light Curve Analysis Toolkit**

TransitKit provides publication-quality tools for transit detection, 
parameter estimation, validation, and visualization.

Installation
------------

.. code-block:: bash

   pip install transitkit

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from transitkit.core import (
       generate_transit_signal_mandel_agol,
       find_transits_bls_advanced,
       add_noise
   )

   # Generate synthetic transit
   time = np.linspace(0, 30, 2000)
   flux = generate_transit_signal_mandel_agol(time, period=5.0, depth=0.01)
   flux = add_noise(flux, 0.001)

   # Detect transit
   result = find_transits_bls_advanced(time, flux)
   print(f"Period: {result['period']:.4f} days")

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   tutorials/index
   api/index
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
