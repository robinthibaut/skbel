.. -*- mode: rst -*-

|Travis|_  |Doc|_ |Black|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Downloads|_

.. |Travis| image:: https://travis-ci.com/robinthibaut/skbel.svg?branch=master
.. _Travis: https://travis-ci.com/robinthibaut/skbel

.. |Doc| image:: https://readthedocs.org/projects/skbel/badge/?version=latest
.. _Doc: https://skbel.readthedocs.io/en/latest/?badge=latest

.. |CodeCov| image:: https://codecov.io/gh/robinthibaut/skbel/branch/main/graph/badge.svg?token=S0T9NW3VK6
.. _CodeCov: https://codecov.io/gh/robinthibaut/skbel

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/skbel
.. _PythonVersion: https://img.shields.io/pypi/pyversions/skbel

.. |PyPi| image:: https://badge.fury.io/py/skbel.svg
.. _PyPi: https://badge.fury.io/py/skbel

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/369214956.svg
.. _DOI: https://zenodo.org/badge/latestdoi/369214956

.. |Downloads| image:: https://pepy.tech/badge/skbel
.. _Downloads: https://pepy.tech/project/skbel

.. |PythonMinVersion| replace:: 3.7
.. |NumPyMinVersion| replace:: 1.14.6
.. |SciPyMinVersion| replace:: 1.1.0
.. |JoblibMinVersion| replace:: 0.11
.. |MatplotlibMinVersion| replace:: 2.2.2
.. |Scikit-ImageMinVersion| replace:: 0.24.1
.. |PandasMinVersion| replace:: 0.25.0
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 5.0.1

.. image:: https://raw.githubusercontent.com/robinthibaut/skbel/master/docs/img/illu-01.png

**skbel** is a Python module for implementing the Bayesian Evidential Learning framework built on top of
scikit-learn and is distributed under the 3-Clause BSD license.

For more information, read the `documentation <https://skbel.readthedocs.io/en/latest/>`_ and run the example `notebook <https://www.kaggle.com/dsv/2648718>`_.

Installation
------------

Dependencies
~~~~~~~~~~~~

skbel requires:

- Python (>= |PythonMinVersion|)
- Scikit-Learn (>= |Scikit-ImageMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)

=======

Skbel plotting capabilities require Matplotlib (>= |MatplotlibMinVersion|).

User installation
~~~~~~~~~~~~~~~~~

The easiest way to install skbel is using ``pip``   ::

    pip install skbel


Development
-----------

We welcome new contributors of all experience levels.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/robinthibaut/skbel/
- Download releases: https://pypi.org/project/skbel/
- Issue tracker: https://github.com/robinthibaut/skbel/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/robinthibaut/skbel.git

Contributing
~~~~~~~~~~~~

Contributors and feedback from users are welcome. Don't hesitate to submit an issue or a PR, or request a new feature.


Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest skbel


Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (latest release): https://skbel.readthedocs.io/en/latest/

Communication
~~~~~~~~~~~~~

- Github Discussions: https://github.com/robinthibaut/skbel/discussions

How to cite
----------------

Thibaut, Robin, & Maximilian Ramgraber. (2021). SKBEL - Bayesian Evidential Learning framework built on top of scikit-learn (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.6205242

BibTeX::

    @software{thibaut_skbel_2021,
    author       = {Thibaut, Robin and Maximilian Ramgraber},
    title        = {{SKBEL} - Bayesian Evidential Learning framework built on top of scikit-learn},
    month        = {9},
    year         = 2021,
    publisher    = {Zenodo},
    version      = {v2.0.0},
    doi          = {10.5281/zenodo.6205242},
    url          = {https://doi.org/10.5281/zenodo.6205242},
    }

Notebooks and tutorials
------------------------

Nolwenn Lesparre, Nicolas Compaire, Thomas Hermans and Robin Thibaut. (2022). 4D Temperature Monitoring with BEL. [Dataset]. Kaggle. doi: 10.34740/kaggle/ds/2275519. url: https://doi.org/10.34740/kaggle/ds/2275519

Thibaut, Robin (2021). WHPA Prediction. [Dataset]. Kaggle. doi:10.34740/kaggle/dsv/2648718. url: https://www.kaggle.com/dsv/2648718

Peer-reviewed publications using SKBEL
--------------------------------------

Thibaut, Robin, Nicolas Compaire, Nolwenn Lesparre, Maximilian Ramgraber, Eric Laloy, and Thomas Hermans (Nov. 2022). “Comparing Well and Geophysical Data for Temperature Monitoring Within a Bayesian Experimental Design Framework”. In: Water Resources Research 58 (11). issn: 0043-1397. doi: 10.1029/2022WR033045. url: https://onlinelibrary.wiley.com/doi/10.1029/2022WR033045.

Thibaut, Robin, Eric Laloy, and Thomas Hermans (Dec. 2021). “A new framework for experimental design using Bayesian Evidential Learning: The case of wellhead protection area”. In: Journal of Hydrology 603, p. 126903. issn: 00221694. doi: 10.1016/j.jhydrol.2021.126903. url: https://linkinghub.elsevier.com/retrieve/pii/S0022169421009537.

Research project
----------------

Logs and results of the research project are available on the `project page <https://www.researchgate.net/project/A-new-framework-for-Experimental-Design-in-Earth-Sciences-using-Bayesian-Evidential-Learning-BEL4ED>`_.


