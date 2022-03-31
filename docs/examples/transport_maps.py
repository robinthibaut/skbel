# Author: Maximilian Ramgraber, Massachusetts Institute of Technology, USA

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.gridspec import GridSpec

from skbel.tmaps import TransportMap  # My transport map toolbox

# =============================================================================
# Create samples from some complex target distribution
# =============================================================================

# How many particles do we want to draw?
N = 1000

# Let's draw samples from a banana-shaped distribution!
X = scipy.stats.norm.rvs(size=(N, 2), random_state=123456)
b = 1  # Twist factor
X[:, 1] += b * X[:, 0] ** 2

# Plot the target distribution samples; in your example, X would just be the
# samples you would fit your KDEs to, an N-by-2 matrix
gs = GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])
plt.subplot(gs[0])
plt.scatter(X[:, 0], X[:, 1])

# =============================================================================
# Build a transport map
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Define the map component functions
# -----------------------------------------------------------------------------

# Let's build a transport map! The way my code works, we define the function
# through a list of lists, which tells the code how to assemble the map's
# component functions. Empty list entries add a constant, [i] adds a linear
# term for the i-th dimension, [i,i] adds a quadratic term for the i-th
# dimension, and so forth. String modifiers such as [i,i,'HF'] denote a
# quadratic function with a hermite function weighting.

# Due to the way transport methods work, if you only want to condition a
# two-dimensional distribution, then you only need to construct, optimize, and
# use the map component for the second dimension. Each map component is split
# up into two parts: the monotone part, which only includes terms with the i-th
# dimension, and the nonmonotone part, which includes all terms with dimensions
# j smaller than i. In this case, assuming you want to condition on the first
# dimension/column of X, these are different maps you can try:

# (comment out the maps you are not using)

# # A linear transport map would be:
# nonmonotone = [
#     [[],[0]] ]

# monotone    = [
#     [[1]] ]

# # A moderately nonlinear transport map would be:
# # Note: the 'iRBF 1' is another special term, an integrated radial basis
# # function applied to the first dimension. Using it several times re-scales
# # and spreads out each terms
# nonmonotone = [
#     [[],[0],[0,0,'HF'],[0,0,0,'HF']] ]

# monotone    = [
#     [[1],'iRBF 1','iRBF 1','iRBF 1'] ]


# A highly nonlinear transport map would be:
nonmonotone = [
    [
        [],
        [0],
        [0, 0, "HF"],
        [0, 0, 0, "HF"],
        [0, 0, 0, 0, "HF"],
        [0, 0, 0, 0, 0, "HF"],
        [0, 0, 0, 0, 0, 0, "HF"],
        [0, 0, 0, 0, 0, 0, 0, "HF"],
    ]
]

monotone = [
    [
        [1],
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
        "iRBF 1",
    ]
]

# # If you are curious about what a full highly-nonlinear map could look like,
# # here you go:
# nonmonotone = [
#     [[]],
#     [[],[0],[0,0,'HF'],[0,0,0,'HF'],[0,0,0,0,'HF'],[0,0,0,0,0,'HF'],[0,0,0,0,0,0,'HF'],[0,0,0,0,0,0,0,'HF']] ]

# monotone    = [
#     [[0],'iRBF 0','iRBF 0','iRBF 0','iRBF 0','iRBF 0','iRBF 0','iRBF 0','iRBF 0'],
#     [[1],'iRBF 1','iRBF 1','iRBF 1','iRBF 1','iRBF 1','iRBF 1','iRBF 1','iRBF 1'] ]


# For some of the more complex operations, like the spiral distribution in my
# poster or most things involving multimodal distributions, it is necessary to
# consider cross-terms (i.e., the monotone function also requires the zero-th
# dimension in some form, not just the first). In this case, you will have to
# replace the monotonicity variable below with "integrated rectifier" and the
# requirement for monotone terms in the monotone variable is lifted, but the
# transport map becomes quite a bit more computationally expensive. I have
# stuck with the simpler case in this example.

# -------------------------------------------------------------------------
# Step 2: Construct and optimize the map
# -------------------------------------------------------------------------

# Initialize a transport map object
tm = TransportMap(
    monotone=monotone,
    nonmonotone=nonmonotone,
    X=X,
    polynomial_type="probabilist's hermite",
    monotonicity="separable monotonicity",
    standardize_samples=True,
    workers=1,
)  # Number of workers for the parallel optimization; 1 means no parallelization

# Optimize the transport map
start = time.time()
tm.optimize()
end = time.time()
print("Map optimization took " + str(end - start) + " seconds.")

# =============================================================================
# Condition the target distribution
# =============================================================================

# To extract posterior samples, we first map the samples X from the (unknown)
# target distribution to a standard Gaussian (norm_samples), then conditionally
# invert these samples given the values you observed to transform them into
# posterior samples.

# Map the target samples to the reference distribution; we only get a N-by-1
# vector because we only defined the map for the second dimension/column of X
# norm_samples = tm.map(X)
norm_samples = scipy.stats.norm.rvs(size=(500, 1), random_state=42)
# Now define the value we wish to condition on
x1_obs = 2.32  # our 'observed' value

# In the inversion, we pretend that we have already inverted the first dimension
# of X and obtained x1_obs, so we create fake, pre-calculated values for it
X_precalc = np.ones((500, 1)) * x1_obs

# Now invert the map conditionally. X_star are the posterior samples.
X_star = tm.inverse_map(
    X_precalc=X_precalc, Y=norm_samples
)  # Only necessary when heuristic is deactivated

# =============================================================================
# Plot the results
# =============================================================================

# Plot the observed value
xlims = plt.gca().get_xlim()
ylims = plt.gca().get_ylim()
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)
plt.plot(np.ones(2) * x1_obs, ylims, color="r")
plt.xlabel("dimension $X_1$")
plt.ylabel("dimension $X_2$")

# Plot the histogram of prior vs posterior
plt.subplot(gs[1])

# Create bins for the second dimension
nbins = 20
bins = [
    [x, x + np.diff(np.linspace(ylims[0], ylims[-1], nbins + 1)[:2])[0]]
    for x in np.linspace(ylims[0], ylims[-1], nbins + 1)
]

# Count entries in each bin for the prior and posterior
bincounts_prior = [
    np.sum(np.logical_and(X[:, 1] > bn[0], X[:, 1] <= bn[1])) for bn in bins
]
bincounts_posterior = [
    np.sum(np.logical_and(X_star > bn[0], X_star <= bn[1])) for bn in bins
]

# Plot prior and posterior in a horizontal bar plot
plt.barh(
    [np.mean(bn) for bn in bins],
    bincounts_prior,
    height=(bins[0][1] - bins[0][0]) * 0.8,
    alpha=0.5,
    label="prior",
)

plt.barh(
    [np.mean(bn) for bn in bins],
    bincounts_posterior,
    height=(bins[0][1] - bins[0][0]) * 0.8,
    alpha=0.5,
    label="posterior",
)

# Finish up the figure
plt.legend()
plt.gca().set_yticklabels([])
plt.gca().set_ylim(ylims)
plt.xlabel("$X_2$ particle count")

plt.show()
