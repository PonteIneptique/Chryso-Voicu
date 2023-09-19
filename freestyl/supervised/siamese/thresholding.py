from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from matplotlib import pyplot as plt
from freestyl.supervised.siamese.utils import PairsDataframe
from freestyl.utils import TypedDataframe


SpecificityDataframe = TypedDataframe[["Threshold", "Sensitivity", "Specificity", "F1", "AuRoc", "K"]]


def confusion_threshold_compute(df: PairsDataframe, threshold: float):
    """ Retrieve True Positive, False Positive, True negative and False negative counts for a threshold

    :param df: Dataframe with a `distance` column and whether it's a valid or invalid pair (correct=1, authors are
    the same)
    :param threshold: Distance at which we differentiate pairing (<threshold = Good Pair)
    """
    cnts = df[df.Distance <= threshold].IsAPair.value_counts().to_dict()
    tp = cnts.get(True, 0)
    fp = cnts.get(False, 0)
    cnts = df[df.Distance > threshold].IsAPair.value_counts().to_dict()
    tn = cnts.get(False, 0)
    fn = cnts.get(True, 0)
    return tp, fp, tn, fn


def sensivity_specificity_compute(df: PairsDataframe, threshold: float):
    """ Compute Sensitivity and Specificity

    :param df: Dataframe with a `Distance` column and whether it's a valid or invalid pair (IsAPair=True, authors are
    the same)
    :param threshold: Distance at which we differentiate pairing (<threshold = Good Pair)
    """
    tp, fp, tn, fn = confusion_threshold_compute(df, threshold)
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def binarized_f1(df: pd.DataFrame, threshold: float) -> float:
    return f1_score(df.IsAPair.to_numpy(), (df.Distance < threshold).to_numpy())


def binarized_auc(df: pd.DataFrame, threshold: float) -> float:
    return roc_auc_score(df.IsAPair.to_numpy(), (df.Distance < threshold).to_numpy())


def sensivity_specificity_curve(df: PairsDataframe, step: float = 0.005, k: int = 0) -> SpecificityDataframe:
    """ Produces a df for each threshold using step.

    :param df: Dataframe with a `distance` column and whether it's a valid or invalid pair (correct=1, authors are
    the same)
    :param step: Distance between each computer threshold
    """
    curve = []
    for step in np.arange(0+step, df.Distance.max(), step):
        curve.append(
            (
                step,
                *sensivity_specificity_compute(df, step),
                binarized_f1(df, threshold=step),
                binarized_auc(df, threshold=step),
                k
            )
        )
    return SpecificityDataframe(curve, columns=["Threshold", "Sensitivity", "Specificity", "F1", "AuRoc", "K"])


def optimize_sentitivity_and_specificity(df: SpecificityDataframe,
                                         sensitivity_weight: float = 1.0,
                                         specificity_weight: float = 1.0):
    return df.loc[
            (df.Sensitivity * sensitivity_weight + df.Specificity * specificity_weight)
            .sort_values(ascending=False)
            .index
    ].iloc[0]


def get_strong_threshold(df: PairsDataframe, percentiles: Tuple[float, ...] = (.25, .5, .75, .9)):
    """ Produces a df for each threshold using step.

    :param df: Dataframe with a `distance` column and whether it's a valid or invalid pair (correct=1, authors are
    the same)
    :param percentiles: Percentile at which we want to get the binarizing threshold
    """
    percents = df.groupby("IsAPair")["Distance"].describe(percentiles=percentiles)
    return [
        percents.loc[True, f"{int(per*100)}%"]
        for per in percentiles
    ]


def plot_sentitivity_and_specificity(df: PairsDataframe, ax: Optional[plt.Axes] = None, **fig_kwargs):
    curve = sensivity_specificity_curve(df)
    best = optimize_sentitivity_and_specificity(curve, 1.0, 2.0)
    if ax is None:
        fig = plt.figure(**fig_kwargs)
        ax = fig.gca()
    else:
        plt.sca(ax)

    q1, med, q3, d9 = get_strong_threshold(df)
    curve.plot.line(x="Threshold", y=["Sensitivity", "Specificity"], ax=ax)
    plt.axline((q1, 0), (q1, 1.0), color="grey", linestyle=":")
    plt.axline((q3, 0), (q3, 1.0), color="grey", linestyle=":")
    plt.axline((d9, 0), (d9, 1.0), color="grey", linestyle=":")
    plt.axline((best.Threshold, 0), (best.Threshold, 1.0), color="grey", linestyle="--")

    title = ax.set_title(
        f"Maximized couple with threshold T={best.Threshold:.3f}\n"
        f"Maximized sensitivity: {best.Sensitivity:.3f} & Maximized specificity: {best.Specificity:.3f}\n"
        f"Threshold at TP Median: {q1:.3f}\n"
        f"Threshold at TP 75%: {q3:.3f}\n"
        f"Threshold at TP 90%: {d9:.3f}"
    )
    return ax


