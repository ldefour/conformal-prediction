# Sources:
# https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_cqr_tutorial.html
# https://mapie.readthedocs.io/en/latest/examples_regression/1-quickstart/plot_prefit.html#sphx-glr-examples-regression-1-quickstart-plot-prefit-py
# https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_ResidualNormalisedScore_tutorial.html#sphx-glr-examples-regression-4-tutorials-plot-residualnormalisedscore-tutorial-py

# Import dependencies
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from mapie.metrics import regression_coverage_score, regression_mean_width_score
from mapie.conformity_scores import ResidualNormalisedScore
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample

from utils.regression import (
    sort_y_values,
    plot_prediction_intervals,
    get_coverages_widths_by_bins,
)

warnings.filterwarnings("ignore")

# Global config
random_state = 27
rng = np.random.default_rng(random_state)
alpha = 0.1

# Load the data
data = fetch_california_housing(as_frame=True)
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data=data.target) * 100
df = pd.concat([X, y], axis=1)

# Split the data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y["MedHouseVal"], random_state=random_state, test_size=0.1
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, random_state=random_state, test_size=0.2
)

X_calib_prefit, X_res, y_calib_prefit, y_res = train_test_split(
    X_calib, y_calib, random_state=random_state, test_size=0.5
)

# Fit base and residual estimators
# Use Median CatBoostRegressor for MapieRegressor
base_estimator = CatBoostRegressor(
    loss_function=f"Quantile:alpha=0.5",
    random_state=random_state,
    verbose=False,
)
base_estimator.fit(X_train, y_train)

res_estimator = CatBoostRegressor(
    loss_function=f"Quantile:alpha=0.5",
    random_state=random_state,
    verbose=False,
)
res_estimator = res_estimator.fit(
    X_res, np.abs(np.subtract(y_res, base_estimator.predict(X_res)))
)

# Train CatBoostRegressor models for MapieQuantileRegressor
list_estimators_cqr = []
for alpha_ in [alpha / 2, (1 - (alpha / 2)), 0.5]:
    estimator_ = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha_}",
        random_state=random_state,
        verbose=False,
    )
    estimator_.fit(X_train, y_train)
    list_estimators_cqr.append(estimator_)

# Build mutiple conformal predictors
STRATEGIES = {
    "naive": {"method": "naive"},
    "cv_plus": {"method": "plus", "cv": 10},
    # "jackknife_plus_ab": {
    #     "method": "plus",
    #     "cv": Subsample(n_resamplings=50, random_state=random_state),
    # },
    "cqr": {},
    "split_with_res": {
        "cv": "prefit",
        "estimator": base_estimator,
        "conformity_score": ResidualNormalisedScore(
            residual_estimator=res_estimator, random_state=random_state, prefit=True
        ),
    },
}
y_pred, y_pis = {}, {}
y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
coverage, width = {}, {}
for strategy, params in STRATEGIES.items():
    if strategy == "cqr":
        mapie = MapieQuantileRegressor(list_estimators_cqr, cv="prefit")
        mapie.fit(X_calib, y_calib)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test)  # no need for alpha
    elif strategy == "split_with_res":
        mapie = MapieRegressor(**params, random_state=random_state)
        mapie.fit(X_calib_prefit, y_calib_prefit)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)
    else:
        mapie = MapieRegressor(base_estimator, **params, random_state=random_state)
        mapie.fit(X_train_full, y_train_full)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)

    (
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy],
    ) = sort_y_values(y_test, y_pred[strategy], y_pis[strategy])
    coverage[strategy] = regression_coverage_score(
        y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
    )
    width[strategy] = regression_mean_width_score(
        y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
    )

# Show sampled prediction intervals
perc_obs_plot = 0.02
num_plots = rng.choice(len(y_test), int(perc_obs_plot * len(y_test)), replace=False)
fig, axs = plt.subplots(2, 2, figsize=(15, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
for strategy, coord in zip(STRATEGIES.keys(), coords):
    plot_prediction_intervals(
        strategy,
        coord,
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy],
        coverage[strategy],
        width[strategy],
        num_plots,
    )
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
plt.legend(
    lines[:4],
    labels[:4],
    loc="upper center",
    bbox_to_anchor=(0, -0.15),
    fancybox=True,
    shadow=True,
    ncol=2,
)
plt.show()

# Show conditional coverage by target bins
bins = list(np.arange(0, 1, 0.1))
binned_data = get_coverages_widths_by_bins(
    "coverage", y_test_sorted, y_pred_sorted, lower_bound, upper_bound, STRATEGIES, bins
)
binned_data.T.plot.bar(figsize=(12, 4))
plt.axhline(1 - alpha, ls="--", color="k")
plt.ylabel("Conditional coverage")
plt.xlabel("Binned house prices")
plt.xticks(rotation=345)
plt.ylim(0.3, 1.0)
plt.legend(loc=[1, 0])
plt.show()

# Show width by target bins
binned_data = get_coverages_widths_by_bins(
    "width", y_test_sorted, y_pred_sorted, lower_bound, upper_bound, STRATEGIES, bins
)
binned_data.T.plot.bar(figsize=(12, 4))
plt.ylabel("Interval width")
plt.xlabel("Binned house prices")
plt.xticks(rotation=350)
plt.legend(loc=[1, 0])
plt.show()
