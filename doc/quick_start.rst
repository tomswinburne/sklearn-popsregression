.. _quick_start:

###############
Getting started
###############

Installation
============

Install from source::

    pip install .

Or for development::

    pip install -e .

Dependencies: ``scikit-learn>=1.6.1``, ``scipy>=1.6.0``, ``numpy>=1.20.0``.

Quick Example
=============

::

    import numpy as np
    from popsregression import POPSRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Generate low-noise data
    rng = np.random.RandomState(42)
    x = np.sort(rng.uniform(-1, 1, 50)) * 10
    y = np.sin(x) * x + 0.001 * rng.randn(50)

    # Create polynomial features
    poly = PolynomialFeatures(degree=4, include_bias=True)
    X = poly.fit_transform(x.reshape(-1, 1))

    # Fit POPS Regression
    model = POPSRegression(resampling_method='sobol')
    model.fit(X, y)

    # Predict with uncertainty
    y_pred, y_std = model.predict(X, return_std=True)

Running Tests
=============

.. prompt:: bash $

  pytest -vsl popsregression

Building Documentation
======================

.. prompt:: bash $

  cd doc && make html

Using pixi
==========

If you have `pixi <https://pixi.sh/>`_ installed, you can use the
pre-configured tasks::

    pixi run test    # run tests
    pixi run lint    # check code style
    pixi run build-doc  # build documentation
