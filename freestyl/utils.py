import numpy as np
from pandas import DataFrame
from typing import Iterable, List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

__all__ = [
    "check_dataframe",
    "get_confusion_matrix_from_predictions",
    "get_colors",
    "TypedDataframe",
    "plot_aucroc_curve"
]

GenericAlias = type(List[str])


def plot_aucroc_curve(pairs: np.array, probability: np.array, nth: int=10, is_dist: bool = True):
    if is_dist:
        fpr, tpr, thresholds = metrics.roc_curve(pairs, -probability)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(pairs, probability)

    f, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.plot(fpr, tpr, 'o-', label="ROC curve")
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), label="diagonal")
    # Annotate the text every whatever
    last_y = None
    for x, y, txt in zip(fpr[::nth], tpr[::nth], thresholds[::nth]):
        if y == last_y:
            continue
        last_y = y
        if is_dist:
            txt = -txt
        plt.annotate(np.round(txt, 2), (x, y-0.04))
        if y == 1.0:
            break
    plt.legend(loc="upper left")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    return f, ax


class TypedDataframe(DataFrame):
    __class_getitem__ = classmethod(GenericAlias)


def check_dataframe(df: DataFrame, has_target: bool = False):
    """ Checks if a df has been built correctly for our functions

    :param df: Dataframe for supervised or unsupervised training
    :param has_target: Whether the df should have a target column
    """
    return True


def get_confusion_matrix_from_predictions(predictions, classes: Iterable[str]) -> DataFrame:
    unique_labels = sorted(list(set(classes)))
    return DataFrame(
        confusion_matrix(classes, predictions, labels=unique_labels),
        index=['true:{:}'.format(x) for x in unique_labels],
        columns=['pred:{:}'.format(x) for x in unique_labels]
    )


CMAP = plt.get_cmap('jet')


def get_colors(k: int) -> np.array:
    colors = CMAP(np.linspace(0, 1.0, k))
    return colors


def get_kfolds_train_dev_test(indexes):
    """ This function gets non-repeating N-Folds of train, dev, test based on a list of N List of indexes

    >>> list(get_kfolds_train_dev_test([[0], [1], [2], [3]])) == [
    ...   ([2, 3], [0], [1]),  ([0, 3], [1], [2]),  ([0, 1], [2], [3]), ([1, 2], [3], [0])]
    """
    for i in range(len(indexes)):
        ranges = list(range(len(indexes)))
        dev = ranges.pop(i)
        test = ranges.pop(i) if i < len(ranges) else ranges.pop(len(ranges) - i)
        train = ranges
        yield [index for train_idx in train for index in indexes[train_idx]], indexes[dev], indexes[test]


def get_kfolds_splits(iterable, n_folds):
    k, m = divmod(len(iterable), n_folds)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_folds))

