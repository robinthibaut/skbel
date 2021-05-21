SKBEL
==========

Bayesian Evidential Learning - A Prediction-Focused Approach
-----------------------------------------------------------------------------------------
### Introduction

<p align="center">
<img src="/docs/img/evidential.png" height="350">
</p>
<p align="center">
  Figure 1: The concept of BEL. d = predictor (observed data), h = target (parameter of interest), m = model.
<p align="center">

- The idea of BEL is to find a direct relationship between `d` (predictor) and `h` (target) in a reduced dimensional space with machine learning.
- Both `d` and `h` are generated by forward modelling from the same set of prior models `m`.
- Given a new measured predictor `d*`, this relationship is used to infer the posterior probability distribution of the target, without the need for a computationally expensive inversion. 
- The posterior distribution of the target is then sampled and backtransformed from the reduced dimensional space to the original space to predict posterior realizations of `h` given `d*`.
  
### Workflow

#### Forward modeling
- Examples of both `d` and `h` are generated through forward modeling from the same model `m`. Target and predictor are real, multi-dimensional random variables.
#### Pre-processing
- Specific pre-processing is applied to the data if necessary (such as scaling).
#### Dimensionality reduction
- Principal Component Analysis (PCA) is applied to both target and predictor to aggregate the correlated variables into a few independent Principal Components (PC’s).
#### Learning
- Canonical Correlation Analysis (CCA) transforms the two sets into pairs of Canonical Variates (CV’s) independent of each another.
#### Post-processing
- Specific post-processing is applied to the CV's if necessary (such as CV normalization).
#### Posterior distribution inference
- The mean `μ` and covariance `Σ` of the posterior distribution of an unknown target given an observed `d*` can be directly estimated from the CV's distribution.
#### Sampling and back-transformation to the original space
- The posterior distribution is sampled to obtain realizations of `h` in canonical space, successively back-transformed to the original space.

<p align="center">
<img src="/docs/img/img.png" height="500">
</p>
<p align="center">
  Figure 2: Typical BEL worflow.
<p align="center">
 
Example
-----------------------------------------------------------------------------------------
- All the details about the example can be found in [arXiv:2105.05539](https://arxiv.org/abs/2105.05539), and the code in `skbel/examples/demo.py`.
- It concerns a hydrogeological experiment consisting of predicting the wellhead protection area (WHPA) around a pumping well from measured breakthrough curves at said pumping well. 
- Predictor and target are generated through forward modeling from a set of hydrogeological model with different hydraulic conductivity fields (not shown).
- The predictor is the set of breakthrough curves coming from 6 different injection wells around the pumping well (Figure 3).
- The target is the WHPA (Figure 4).
  
For this example, the data is already pre-processed. We are working with 400 exmples of both `d` and `h` and consider one extra pair to be predicted. See details in the reference.
  
<p align="center">
<img src="/docs/img/data/curves.png" height="500" background-color: white>
</p>
<p align="center">
  Figure 3: Predictor set. Prior in the background and test data in thick lines.
<p align="center">

  <p align="center">
<img src="/docs/img/whpa.png" height="500" background-color: white>
</p>
<p align="center">
  Figure 4: Target set. Prior in the background (blue) and test data to predict in red.
<p align="center">
  
#### Building the BEL model
In this package, a BEL model consists of a succession of Pipelines (imported from scikit-learn).
  
```python
import os
from os.path import join as jp

import joblib
import pandas as pd
from loguru import logger
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

import demo_visualization as myvis
from demo_config import Setup
from skbel import utils
from skbel.learning.bel import BEL

  ```
We can then define a function that returns our desired BEL model :

```python
def init_bel():
    """
    Set all BEL pipelines
    """
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )

    # Canonical Correlation Analysis
    # Number of CCA components is chosen as the min number of PC
    n_pc_pred, n_pc_targ = 50, 30

    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    # Set PC cut
    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model
  ```
  
- The ```X_pre_processing``` and ```Y_pre_processing``` objects are pipelines which will first scale the data for predictor and target, then apply the dimension reduction through PCA.

- An arbitrary choice has to be made on the number of PC to keep for the predictor and the target. In this case, they are set to 50 and 30, respectively.

- The CCA operator `cca` is set to keep the maximum number of CV possible (30).

- The ```X_post_processing``` and ```Y_post_processing``` objects are pipelines which will normalize predictor and target CV's.
  
- Finally, the BEL model is constructed by passing as arguments all these pipelines in the `BEL` object.
  
#### Training the BEL model
A simple function can be defined to train our model.
  ```python
def bel_training(bel_,
                 *,
                 X_train_: pd.DataFrame,
                 x_test_: pd.DataFrame,
                 y_train_: pd.DataFrame,
                 y_test_: pd.DataFrame = None,
                 directory: str = None):
    """
    :param bel_: BEL model
    :param X_train_: Predictor set for training
    :param x_test_: Predictor "test"
    :param y_train_: Target set for training
    :param y_test_: "True" target (optional)
    :param directory: Path to the directory in which to unload the results
    :return:
    """
    #%% Directory in which to load forecasts
    if directory is None:
        sub_dir = os.getcwd()
    else:
        sub_dir = directory

    # Folders
    obj_dir = jp(sub_dir, "obj")  # Location to save the BEL model
    fig_data_dir = jp(sub_dir, "data")  # Location to save the raw data figures
    fig_pca_dir = jp(sub_dir, "pca")  # Location to save the PCA figures
    fig_cca_dir = jp(sub_dir, "cca")  # Location to save the CCA figures
    fig_pred_dir = jp(sub_dir, "uq")  # Location to save the prediction figures

    # Creates directories
    [
        utils.dirmaker(f, erase=True)
        for f in [
            obj_dir,
            fig_data_dir,
            fig_pca_dir,
            fig_cca_dir,
            fig_pred_dir,
        ]
    ]

    # %% Fit BEL model
    bel_.Y_obs = y_test_
    bel_.fit(X=X_train_, Y=y_train_)

    # %% Sample for the observation
    # Extract n random sample (target pc's).
    # The posterior distribution is computed within the method below.
    bel_.predict(x_test_)

    # Save the fitted BEL model
    joblib.dump(bel_, jp(obj_dir, "bel.pkl"))
    msg = f"model trained and saved in {obj_dir}"
    logger.info(msg)
  ```

#### Load the dataset and run everything
  
  ```python
if __name__ == "__main__":
    # Initiate BEL model
    model = init_bel()

    # Set directories
    data_dir = jp(os.getcwd(), "dataset")
    output_dir = jp(os.getcwd(), "results")

    # Load dataset
    X_train = pd.read_pickle(jp(data_dir, "X_train.pkl"))
    X_test = pd.read_pickle(jp(data_dir, "X_test.pkl"))
    y_train = pd.read_pickle(jp(data_dir, "y_train.pkl"))
    y_test = pd.read_pickle(jp(data_dir, "y_test.pkl"))

    # Train model
    bel_training(
        bel_=model,
        X_train_=X_train,
        x_test_=X_test,
        y_train_=y_train,
        y_test_=y_test,
        directory=output_dir,
    )

    # Plot the results
    bel = joblib.load(jp(output_dir, "obj", "bel.pkl"))
    bel.n_posts = Setup.HyperParameters.n_posts

    # Plot raw data
    myvis.plot_results(
        bel,
        base_dir=output_dir
    )

    # Plot PCA
    myvis.pca_vision(
        bel,
        base_dir=output_dir,
    )

    # Plot CCA
    myvis.cca_vision(bel=bel, base_dir=output_dir)
  ```
