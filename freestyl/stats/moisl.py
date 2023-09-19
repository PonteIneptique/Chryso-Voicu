import numpy as np
import pandas as pd
from logging import getLogger
from freestyl.stats.frequencies import relative_frequencies
from freestyl.dataset.dataframe_wrapper import DataframeWrapper

__all__ = ["moisl", "apply_moisl"]


logger = getLogger(__name__)


def moisl(data: pd.DataFrame, z: float = 1.96) -> pd.DataFrame:
    """ Compute MOISL for each feature for a Dataframe of Texts (rows) per Ngrams (Columns)

    Index should be names of the texts
    Columns should be the feature value

    >>> moisl(pd.DataFrame(
    ...    [{"a": 1, "b": 25, "c": 3}, {"b": 23, "c": 5}, {"b": 30, "c": 20}, {"b": 15, "c": 1}],
    ...    index=["Mac1", "Mac2", "Mac3", "Mac4"]))
       freq  mean_prob     score   keep
    a   1.0   0.008621  47.89995  False
    b  93.0   0.805249  8.731698   True
    c  29.0   0.186130  7.856381   True
    """
    data = data.fillna(0)
    array = data.to_numpy()

    # Compute probabilities
    probs = relative_frequencies(array)

    # Transpose for the rest of what's to come
    probs = probs.transpose()
    array = array.transpose()
    data = data.transpose()

    assert array.shape == probs.shape

    # Output columns
    cols = ['freq', 'mean_prob', 'score', 'keep']
    feature_names = data.index.tolist()
    nb_features, _ = probs.shape

    # data for output
    df = pd.DataFrame(index=feature_names, columns=cols)

    # The axis parameter indicates which axis gets aggregated for the operation
    #  If we aggregate on axis 0 (rows), we sum columns ([[0 1], [2 3]] -> [sum([0,2]), sum([1, 3]) -> [2 4])
    #  If we aggregate on axis 0 (cols), we sum rows ([[0 1], [2 3]] -> [sum([0,1]), sum([2, 3]) -> [1 5])

    df.freq = array.sum(axis=1)
    df.mean_prob = probs.mean(axis=1)

    for row_index in range(nb_features):
        row = probs[row_index]
        mirror = (max(row) + min(row)) - row
        row = np.concatenate([row, mirror])
        e = 2 * np.std(row, ddof=1)
        m = row.mean()
        df.loc[feature_names[row_index], "score"] = (m * (1 - m)) * ((z / e)**2)

    df.keep = df.score <= min(array.sum(axis=0))

    return df


def apply_moisl(data: DataframeWrapper, z: float = 1.91, inplace: bool = True):
    features = moisl(data.xs, z=z).keep
    logger.info(f"Keeping only {features.tolist().count(True)} and updating df")
    data.update_features(
        # Very not optimized but I am lazy right now
        pd.Series(data.features)[features.tolist()].tolist()
    )

    logger.info(f"Changing the df with a relative measure for features")
    data.normalized.make_relative(inplace=inplace)
    data.normalized.normalize()

