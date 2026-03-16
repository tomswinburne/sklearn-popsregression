.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

.. _pops_regression:

POPS Regression
---------------

:class:`~popsregression.POPSRegression` is a Bayesian regression method
designed for low-noise data where standard Bayesian approaches underestimate
uncertainty due to model misspecification.

Background
~~~~~~~~~~

In many scientific applications, a surrogate model (e.g. a polynomial) is fit
to near-deterministic data from simulations. When the model class cannot
perfectly represent the target function, standard Bayesian regression methods
like :class:`~sklearn.linear_model.BayesianRidge` only capture *epistemic*
uncertainty (which decays with more data) and *aleatoric* uncertainty (noise,
which is negligible). They miss the dominant source of error:
**misspecification uncertainty** — the systematic error from using the wrong
model class.

POPS (Pointwise Optimal Parameter Sets) corrects this by:

1. Fitting a BayesianRidge model to obtain the posterior mean weights.
2. Computing pointwise corrections: for each training point, finding the
   parameter perturbation that would fit that point exactly.
3. Constructing a posterior over these corrections (via a hypercube in PCA
   space or as a raw ensemble).
4. Using this posterior to estimate misspecification uncertainty in predictions.

Basic Usage
~~~~~~~~~~~

::

    >>> import numpy as np
    >>> from popsregression import POPSRegression
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> rng = np.random.RandomState(42)
    >>> x = np.sort(rng.uniform(-1, 1, 30)) * 10
    >>> y = np.sin(x) * x + 0.01 * rng.randn(30)
    >>> poly = PolynomialFeatures(degree=4, include_bias=True)
    >>> X = poly.fit_transform(x.reshape(-1, 1))
    >>> model = POPSRegression()
    >>> model.fit(X, y)  # doctest: +ELLIPSIS
    POPSRegression()
    >>> y_pred, y_std = model.predict(X, return_std=True)

The returned ``y_std`` combines both epistemic and misspecification uncertainty.

Posterior Types
~~~~~~~~~~~~~~~

POPSRegression supports two posterior forms:

- ``'hypercube'`` (default): Fits a PCA-aligned hypercube to the pointwise
  corrections. This tends to give conservative uncertainty bounds and is
  recommended for most use cases.

- ``'ensemble'``: Uses the raw pointwise corrections directly as posterior
  samples. This can be useful when the number of features is small relative
  to the number of training points.

::

    >>> model = POPSRegression(posterior='ensemble')
    >>> model.fit(X, y)  # doctest: +ELLIPSIS
    POPSRegression(posterior='ensemble')

Prediction Options
~~~~~~~~~~~~~~~~~~

The ``predict`` method supports several return options::

    >>> model = POPSRegression().fit(X, y)
    >>> y_pred = model.predict(X)  # mean only
    >>> y_pred, y_std = model.predict(X, return_std=True)  # + combined std
    >>> y_pred, y_max, y_min = model.predict(X, return_bounds=True)  # + bounds
    >>> y_pred, y_ep_std = model.predict(X, return_epistemic_std=True)  # + epistemic only

All options can be combined::

    >>> result = model.predict(
    ...     X, return_std=True, return_bounds=True, return_epistemic_std=True
    ... )
    >>> y_pred, y_std, y_max, y_min, y_ep_std = result

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

POPSRegression is fully compatible with scikit-learn pipelines::

    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(PolynomialFeatures(degree=4), POPSRegression())
    >>> pipe.fit(x.reshape(-1, 1), y)  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> pipe.predict(x.reshape(-1, 1))  # doctest: +ELLIPSIS
    array([...])

References
~~~~~~~~~~

.. [1] Swinburne, T.D. and Perez, D. (2025).
       "Parameter uncertainties for imperfect surrogate models in the
       low-noise regime."
       Machine Learning: Science and Technology, 6, 015008.
