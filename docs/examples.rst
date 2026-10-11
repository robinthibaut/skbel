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

Other runnable scripts
----------------------

* :download:`neural_posterior.py <../examples/neural_posterior.py>` (requires the ``neural`` extra)
* :download:`bel_inference_to_decision.py <../examples/bel_inference_to_decision.py>`
* :download:`bel_kde_prediction_transfer.py <../examples/bel_kde_prediction_transfer.py>`
* :download:`finite_model_robustness.py <../examples/finite_model_robustness.py>`
