# Inference contract fixes

This release documents two focused software correctness fixes. They define
library behavior only; they do not by themselves establish calibration or
validation for a particular application.

## BEL MVN noise

`BEL.predict(..., mode="mvn", noise=value)` now validates and uses an explicit
finite, non-negative scalar `value` for that call. `noise=None` retains and
resets the historical default multiplier of `0.01`. The value multiplies the
identity covariance in the PCA-score space before its projection through the
CCA rotations. It is therefore a projected-covariance multiplier in this
implementation, not a universal physical-space temperature standard deviation.

## CompositePCA scaling

For `CompositePCA(scale=True)`, a separate `StandardScaler` is now fitted for
each block's PCA scores during `fit`. `transform` uses those retained training
scalers and never fits on an evaluation batch; `inverse_transform` first undoes
the matching score scaling and then applies each PCA inverse transform.

`inverse_transform` accepts a two-dimensional score matrix shaped
`(n_samples, sum(fitted PCA component widths))`, or a one-dimensional score
vector of that length, which is treated as one sample. The fitted widths support
integer, `None`, and variance-fraction PCA component configuration. It returns
one two-dimensional reconstructed array per configured block. Block-count and
transformed-width mismatches raise `ValueError`. The valid `scale=False` path
continues to use PCA scores directly.
