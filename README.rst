.. -*- mode: rst -*-

|CI|_ |Doc|_ |Ruff|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Downloads|_

.. |CI| image:: https://github.com/robinthibaut/skbel/actions/workflows/ci.yml/badge.svg?branch=main
.. _CI: https://github.com/robinthibaut/skbel/actions/workflows/ci.yml

.. |Doc| image:: https://readthedocs.org/projects/skbel/badge/?version=latest
.. _Doc: https://skbel.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/skbel
.. _PythonVersion: https://img.shields.io/pypi/pyversions/skbel

.. |PyPi| image:: https://badge.fury.io/py/skbel.svg
.. _PyPi: https://badge.fury.io/py/skbel

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
.. _Ruff: https://github.com/astral-sh/ruff

.. |DOI| image:: https://zenodo.org/badge/369214956.svg
.. _DOI: https://zenodo.org/badge/latestdoi/369214956

.. |Downloads| image:: https://pepy.tech/badge/skbel
.. _Downloads: https://pepy.tech/project/skbel

.. |PythonMinVersion| replace:: 3.10

.. image:: https://raw.githubusercontent.com/robinthibaut/skbel/main/docs/img/illu-01.png

**skbel** is a Python module for implementing the Bayesian Evidential Learning framework built on top of
scikit-learn and is distributed under the 3-Clause BSD license.

For more information, read the `documentation <https://skbel.readthedocs.io/en/latest/>`_
(``latest`` follows the ``main`` branch) and run the example
`notebook <https://www.kaggle.com/dsv/2648718>`_.

Features
--------

- **BEL** (``skbel.BEL``): predictor and target reduction, canonical correlation
  analysis and posterior inference in ``mvn``, ``kde`` or ``tm`` (transport map)
  mode, with seeded, per-row reproducible posterior sampling.
- **Calibration and scores** (``skbel.metrics``): simulation-based calibration,
  PIT and interval coverage, plus CRPS, energy and Brier scores for posterior
  samples.
- **Ranking and acquisition** (``skbel.metrics``): expected action losses,
  prospective measurement ranking and exact finite acquisition planners.
- **Design and evaluation** (``skbel.design``, ``skbel.evaluation``): compare
  measurement designs on one shared simulation bank, then score and turn
  posteriors into event probabilities and decision risks.
- **Optional neural posterior** (``skbel.neural``): a Gaussian-mixture network
  ensemble with quantile recalibration (``neural`` extra).
- **Experimental prediction capsules** (``skbel.learning.portable``,
  ``skbel.learning.portable_kde``): export a fitted ``BEL`` to a data-only JSON
  document. Capsules and the exact acquisition planners are experimental and may
  change.

None of these tools establishes that a posterior is calibrated or valid for a
particular application.

Installation
------------

Requires Python ≥ |PythonMinVersion|. Core dependencies (``numpy``, ``scipy``,
``scikit-learn``, ``scikit-image``, ``pandas``, ``matplotlib``, ``seaborn``,
``joblib``, ``loguru``) are installed automatically.

User installation
~~~~~~~~~~~~~~~~~

With ``pip``::

    pip install skbel

With ``uv``::

    uv pip install skbel

Version 2.3 is not yet available on PyPI: ``pip install skbel`` currently
installs the previous release (2.1.14), which does not include the 2.3
features. To use them, install from the repository tag or from ``main``::

    pip install "git+https://github.com/robinthibaut/skbel.git@v2.3.0"

Optional extras. For 2.3, request them from the same tag source rather than from
PyPI:

- ``neural`` adds PyTorch for ``skbel.neural``::

      pip install "skbel[neural] @ git+https://github.com/robinthibaut/skbel.git@v2.3.0"

- ``bnn`` adds TensorFlow and TensorFlow Probability for ``skbel.bnn``::

      pip install "skbel[bnn] @ git+https://github.com/robinthibaut/skbel.git@v2.3.0"

Quickstart
~~~~~~~~~~

A small, deterministic fit and posterior sampling run on synthetic data::

    import numpy as np
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA

    from skbel import BEL

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(200, 6))
    X = Y @ rng.normal(size=(6, 10)) + 0.1 * rng.normal(size=(200, 10))

    bel = BEL(
        mode="mvn",
        X_pre_processing=PCA(n_components=4),
        Y_pre_processing=PCA(n_components=3),
        regression_model=CCA(n_components=2),
        random_state=0,
    )
    bel.fit(X[:180], Y[:180])
    samples = bel.predict(X[180:181], n_posts=50)
    print(samples.shape)  # (1, 50, 6): observations, posterior draws, target size

Repeated calls with the same ``random_state`` return the same samples.


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

Clone the repo and run the test suite with ``uv``::

    git clone https://github.com/robinthibaut/skbel.git
    cd skbel
    uv sync --extra dev
    uv run pytest

To run a single test::

    uv run pytest skbel/testing/test_basic.py::test_mvn

The reference arrays under ``skbel/testing/`` are deterministic outputs of the
fixed BEL pipeline. If a future scikit-learn or scipy release shifts the
canonical-correlation sign convention or numerical kernels enough to break the
regression checks, regenerate them with::

    uv run python scripts/regenerate_test_references.py


Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (follows the ``main`` branch): https://skbel.readthedocs.io/en/latest/
- HTML documentation (``stable`` build, may lag ``main``): https://skbel.readthedocs.io/en/stable/
- Changelog: https://github.com/robinthibaut/skbel/blob/main/CHANGELOG.md

Communication
~~~~~~~~~~~~~

- Github Discussions: https://github.com/robinthibaut/skbel/discussions

How to cite
----------------

Thibaut, Robin, & Maximilian Ramgraber. (2021). SKBEL - Bayesian Evidential Learning framework built on top of scikit-learn (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.6205242

BibTeX::

    @software{thibaut_skbel,
    author       = {Thibaut, Robin and Maximilian Ramgraber},
    title        = {{SKBEL} - Bayesian Evidential Learning framework built on top of scikit-learn},
    month        = {4},
    year         = 2026,
    publisher    = {Zenodo},
    version      = {v2.2.0},
    doi          = {10.5281/zenodo.6205242},
    url          = {https://doi.org/10.5281/zenodo.6205242},
    }

The DOI above is the *concept* DOI that always resolves to the latest Zenodo
release. For per-version DOIs, see https://zenodo.org/record/6205242 .

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


