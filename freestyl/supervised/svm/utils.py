import sklearn.exceptions
import warnings
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


def support(ytrue, ypred):
    return ytrue.count(True)


class ScoreLogger:
    def __call__(self, y_true, y_probs):
        return y_probs

    def __name__(self):
        return f"ClassSpecificScorer({self._metric.__name__}, {self._classname})"


class ClassSpecificScorer:
    def __init__(self, metric, classname: str):
        self._metric = metric
        self._classname = classname

    def __call__(self, y_true, y_pred):
        y_true = [True if cls == self._classname else False for cls in y_true]
        y_pred = [True if cls == self._classname else False for cls in y_pred]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
            return self._metric(
                y_true,
                y_pred
            )

    def __name__(self):
        return f"ClassSpecificScorer({self._metric.__name__}, {self._classname})"


def plot_coefficients(
        coefs,
        feature_names,
        current_class,
        top_features=10,
        fig_kwargs: Optional[Dict] = None,
        warm_color: str = "red",
        cold_color: str = "blue"
):
    # Following function from Aneesha Bakharia
    # https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
    fig_kwargs = fig_kwargs or dict(figsize=(15, 15))
    top_positive_coefficients = np.argsort(coefs)[-top_features:]
    top_negative_coefficients = np.argsort(coefs)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    fig, ax = plt.subplots(**fig_kwargs)
    colors = [
        warm_color if c < 0 else cold_color
        for c in coefs[top_coefficients]
    ]
    ax.bar(np.arange(2 * top_features), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    ax.set_xticks(np.arange(0, 2 * top_features), minor=False)
    ax.set_xticklabels(feature_names[top_coefficients], rotation=60, ha='right', minor=False)
    ax.set_title("Coefficients for "+current_class)
    return fig
