.. -*- mode: rst -*-

|Travis|_  |Doc|_ |Black|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Downloads|_

.. |Travis| image:: https://travis-ci.com/robinthibaut/skbel.svg?branch=master
.. _Travis: https://travis-ci.com/robinthibaut/skbel

.. |Doc| image:: https://readthedocs.org/projects/skbel/badge/?version=latest
.. _Doc: https://skbel.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue

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

    pip install -U skbel


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

Thibaut, R., 2021. SKBEL – Bayesian Evidential Learning framework built on top of Scikit-learn. Zenodo. https://doi.org/10.5281/zenodo.5526609
