TransitKit Documentation
=======================

.. image:: _static/logo.png
   :alt: TransitKit Logo
   :width: 300px
   :align: center

**A comprehensive toolkit for exoplanet transit light curve analysis.**

.. raw:: html

   <div class="sd-container-fluid">
   <div class="sd-row">
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>🚀 Quick Start</h4>
   </div>
   <div class="sd-card-body">
   <p>Get started with TransitKit in minutes.</p>
   <a class="btn btn-primary" href="getting_started.html">Get Started</a>
   </div>
   </div>
   </div>
   
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>📚 User Guide</h4>
   </div>
   <div class="sd-card-body">
   <p>Complete guide to all features.</p>
   <a class="btn btn-primary" href="user_guide.html">Read Guide</a>
   </div>
   </div>
   </div>
   
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>🔧 API Reference</h4>
   </div>
   <div class="sd-card-body">
   <p>Detailed API documentation.</p>
   <a class="btn btn-primary" href="api_reference.html">View API</a>
   </div>
   </div>
   </div>
   
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>📊 Examples</h4>
   </div>
   <div class="sd-card-body">
   <p>Interactive tutorials and examples.</p>
   <a class="btn btn-primary" href="examples/index.html">View Examples</a>
   </div>
   </div>
   </div>
   
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>🛠️ Development</h4>
   </div>
   <div class="sd-card-body">
   <p>Contribute to TransitKit.</p>
   <a class="btn btn-primary" href="development/index.html">Developer Guide</a>
   </div>
   </div>
   </div>
   
   <div class="sd-col-12 sd-col-md-4">
   <div class="sd-card sd-shadow-sm">
   <div class="sd-card-header">
   <h4>❓ Support</h4>
   </div>
   <div class="sd-card-body">
   <p>Get help and report issues.</p>
   <a class="btn btn-primary" href="https://github.com/arifsolmaz/transitkit/issues">GitHub Issues</a>
   </div>
   </div>
   </div>
   </div>
   </div>

Introduction
------------

TransitKit is a comprehensive Python toolkit for analyzing exoplanet transit light curves.
It provides tools for data reduction, transit modeling, parameter fitting, and visualization
across multiple space missions and ground-based surveys.

Features
--------

* **Multi-Mission Support**: TESS, Kepler, K2, CHEOPS, PLATO, and ground-based surveys
* **Advanced Fitting Methods**: MCMC, nested sampling, Bayesian optimization, gradient-based methods
* **GPU Acceleration**: JAX-powered models for fast computation on GPUs
* **Interactive Visualization**: Real-time parameter adjustment with Bokeh and Plotly
* **Publication-Ready Outputs**: Automatic generation of LaTeX tables and high-resolution figures
* **Extensible Architecture**: Plugin system for custom transit models and noise models
* **Command Line Interface**: Full functionality available via CLI for scripting and automation
* **Web & Desktop Interfaces**: Streamlit web app and PyQt desktop application

Installation
------------

Basic installation:

.. code-block:: bash

   pip install transitkit

For all features:

.. code-block:: bash

   pip install "transitkit[full]"

For GPU support (Linux only):

.. code-block:: bash

   pip install "transitkit[gpu]"

For development:

.. code-block:: bash

   git clone https://github.com/arifsolmaz/transitkit
   cd transitkit
   pip install -e ".[dev]"

Quick Example
-------------

.. code-block:: python

   import transitkit as tk
   
   # Load TESS data for a target
   lc = tk.load_target("TIC 123456", mission="TESS", sector=1)
   
   # Remove outliers and detrend
   lc_clean = lc.remove_outliers().detrend()
   
   # Fit transit with MCMC
   result = tk.fit_transit(
       lc_clean, 
       method="mcmc",
       n_walkers=50,
       n_steps=2000
   )
   
   # Visualize results
   result.plot()
   
   # Print best-fit parameters
   print(result.summary())
   
   # Save results
   result.save("transit_fit.h5")

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:
   
   getting_started
   installation
   tutorials/index

.. toctree::
   :maxdepth: 3
   :caption: User Guide
   :hidden:
   
   user_guide/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:
   
   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:
   
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:
   
   development/index
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:
   
   license
   code_of_conduct

Citation
--------

If you use TransitKit in your research, please cite:

.. code-block:: bibtex

   @software{transitkit2024,
     author = {Arif Solmaz and Contributors},
     title = {TransitKit: Exoplanet Transit Light Curve Analysis Toolkit},
     year = {2024},
     publisher = {GitHub},
     url = {https://github.com/arifsolmaz/transitkit},
     version = {1.0.0}
   }

Support
-------

* `GitHub Issues <https://github.com/arifsolmaz/transitkit/issues>`_ - Report bugs and request features
* `GitHub Discussions <https://github.com/arifsolmaz/transitkit/discussions>`_ - Community forum
* `Documentation <https://transitkit.readthedocs.io>`_ - Complete documentation
* `Email <arif.solmaz@gmail.com>`_ - Direct contact

License
-------

TransitKit is released under the MIT License. See the :doc:`license` page for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`