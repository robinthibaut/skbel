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
``rank_prospective_measurements`` (see :doc:`paper_metrics`). The values and
units of its loss table are defined by the caller. The example enumerates
outcomes and posteriors explicitly over a fixed finite set; it does not fit a
posterior or perform experimental design.