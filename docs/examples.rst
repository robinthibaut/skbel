.. _examples:

Examples
=========

This section contains examples of how to use the various functions in the
package.

.. toctree::
   :maxdepth: 1

   ./examples/demo.ipynb

Download the examples as notebooks
----------------------------------

* :download:`Demo <./examples/demo.ipynb>`

Prospective measurement risk/utility ranking
---------------------------------------------

:download:`prospective_risk_ranking.py <../examples/prospective_risk_ranking.py>`
is a small, directly runnable, finite example composing
``expected_action_losses``, ``bayes_action_set``, and
``rank_prospective_measurements`` (see :doc:`paper_metrics`). The loss table's
values and units in that example are entirely caller-defined; it performs no
calibrated inference, has no hydrological or other field-performance meaning,
demonstrates no methodological novelty, and neither this package nor PR10
itself performs posterior updating or experimental design -- the example
enumerates outcomes and posteriors explicitly, by hand, over a fixed finite
set.