import numpy as np
import pandas as pd

__all__ = ["df_relative_frequencies", "relative_frequencies", "normalizations", "z_transform"]


def df_relative_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    """ Compute features relatives frequencies to each text for a Dataframe of Texts (rows) per Ngrams (Columns)

    >>> df_relative_frequencies(pd.DataFrame(
    ...    [{"a": 1, "b": 25, "c": 3}, {"b": 23, "c": 5}, {"b": 30, "c": 20}, {"b": 15, "c": 1}],
    ...    index=["Mac1", "Mac2", "Mac3", "Mac4"]))
                 a         b         c
    Mac1  0.034483  0.862069  0.103448
    Mac2  0.000000  0.821429  0.178571
    Mac3  0.000000  0.600000  0.400000
    Mac4  0.000000  0.937500  0.062500
    """
    array = data.fillna(0).to_numpy(na_value=0)
    # Compute probabilities
    probs = relative_frequencies(array)
    data.loc[:, :] = probs
    return data


def relative_frequencies(data: np.array) -> np.array:
    """ Compute features relatives frequencies to each text for an array of Texts (rows) per Ngrams (Cols)

    >>> relative_frequencies(np.array([[1, 0, 3], [0, 3, 5]]))
    array([[0.25 , 0.   , 0.75 ],
           [0.   , 0.375, 0.625]])

    """
    all_frequencies = data.sum(axis=1, keepdims=True)
    return np.nan_to_num(data / all_frequencies)


def normalizations(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """ Normalize features for a Dataframe of Texts (rows) per Ngrams (columns)

    >>> rel_freqs = df_relative_frequencies(pd.DataFrame(
    ...    [{"a": 1, "b": 25, "c": 3}, {"b": 23, "c": 5}, {"b": 30, "c": 20}, {"b": 15, "c": 1}],
    ...    index=["Mac1", "Mac2", "Mac3", "Mac4"]))
    >>> normalizations(rel_freqs, inplace=True)
                 a         b         c
    Mac1  0.911953  0.238176 -0.334086
    Mac2 -0.971346  0.216710 -0.097590
    Mac3 -0.241881 -0.684594  0.687622
    Mac4 -0.377257  0.687994 -0.619953
    """
    df = data
    if not inplace:
        df = data.copy()

    df = df.transpose()

    array = df.to_numpy()
    # Apply Z-Transform
    array = z_transform(array)

    array = array / np.sqrt(np.square(array).sum(axis=0, keepdims=True))

    df.loc[:, :] = array
    return df.transpose()


def z_transform(x: np.array) -> np.array:
    """ Compute the Z-Transform of an array

    >>> rel_freqs = df_relative_frequencies(pd.DataFrame(
    ...    [{"a": 1, "b": 25, "c": 3}, {"b": 23, "c": 5}, {"b": 30, "c": 20}, {"b": 15, "c": 1}],
    ...    index=["Mac1", "Mac2", "Mac3", "Mac4"])).transpose()
    >>> z_transform(rel_freqs.to_numpy())
    array([[ 1.5       , -0.5       , -0.5       , -0.5       ],
           [ 0.39175744,  0.11155163, -1.41514547,  0.9118364 ],
           [-0.54951155, -0.05023463,  1.42140459, -0.82165841]])
    """

    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True, ddof=1)
