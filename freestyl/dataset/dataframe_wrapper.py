import warnings
import random
from typing import Optional, List, Iterable, Tuple, Union, Dict, Any
from logging import getLogger

import numpy as np
from numpy import round

from pandas import DataFrame, Series

from freestyl.errors import NoTargetsError, BadParameter
from freestyl.stats.frequencies import df_relative_frequencies, normalizations
from freestyl.utils import get_kfolds_splits, get_kfolds_train_dev_test

__all__ = ["DataframeWrapper"]


logger = getLogger(__name__)


def reduce(elements: List[int], ratio: float):
    size = len(elements)
    milestone = int(round(size * (1-ratio), 0))
    random.shuffle(elements)
    return elements[:milestone], elements[milestone:]


def flatten(l):
    return [item for sublist in l for item in sublist]


class DataframeWrapper:
    def __init__(
        self,
        df: DataFrame,
        is_normalized: bool = False,
        target: Optional[str] = None,
        label: Optional[Iterable[str]] = None,
        status_col: Optional[str] = None,
        x_ignore: Optional[Iterable[str]] = None,
        multi_label_joins: str = " - "
    ):
        """

        :param df: Dataframe containing classification data
        :param target: Column containing the target for supervised algorithms. If not filled, act as a non supervised
        df
        :param label: Metadata columns
        :param x_ignore: Field that should be ignored for producing features
        """
        self._df = df
        self._normalized: DataframeWrapper._NormalizedWrapper = DataframeWrapper._NormalizedWrapper(self)
        self._target: str = target
        self._label: Optional[Tuple[str, ...]] = (label, ) if isinstance(label, str) else tuple(label) if label else ()
        self._x_ignore: Optional[Tuple[str, ...]] = tuple(x_ignore) if x_ignore else ()
        self._multi_label_join = multi_label_joins
        self.status: Optional[str] = status_col

        self._processing: List[str] = []

        ignored = (*self._label, *self._x_ignore)
        if self.status:
            ignored = (*ignored, status_col)
            self.check_status(status_col)
        if target:
            ignored = (*ignored, target)

        self._features: Tuple[str, ...] = tuple([
            col for col in self._df.columns.tolist()
            if col not in ignored
        ])

    def ranked_features(self, n=500):
        df = DataFrame(self.xs, columns=self.features)
        return df.sum(axis=0).sort_values()[-n:].index.tolist()[::-1]

    def hasFullText(self) -> bool:
        return "full-text" in self.dataframe.columns

    @property
    def fulltext(self) -> List[str]:
        return self.dataframe["full-text"].tolist()

    def merge(self, other: "DataframeWrapper") -> "DataframeWrapper":
        self._df = self.dataframe.append(other.dataframe)
        if self.normalized.available():
            if not other.normalized.available():
                other.align_to(self)
            self.normalized._dataframe = self.normalized.dataframe.append(other.normalized.dataframe)
        return self

    def check_status(self, column: str):
        data = self._df[column].unique().tolist()
        if set(data) == {"test"}:
            logger.warning("Only test values found in the current dataframe")
            return True
        if "impostor" not in data:
            raise ValueError(f"Column {column} for status does not contain an `impostor` value.")
        if "candidate" not in data:
            raise ValueError(f"Column {column} for status does not contain a `candidate` value.")
        if "test" not in data:
            logger.warning(f"Column {column} for status does not contain a `test` value. It is recommended to use"
                           f" a single dataframe with training and test values.")

    def get_subset(
            self,
            status: str,
            sample: Optional[float] = None,
            group: bool = False,
            ys_is_label: bool = False
    ) -> Union[Tuple[np.array, np.array], List[Tuple[np.array, np.array]]]:
        df = self._df[self._df[self.status] == status]
        if sample:
            df = df.sample(axis=0, frac=sample)
        df = df.sort_values(by=self._target)
        if group:
            targets = df[self._target].unique()
            return [
                (
                    df[df[self._target] == cur_target].loc[:, self.features].to_numpy(na_value=0),
                    df[df[self._target] == cur_target].loc[:, self._target].to_numpy(na_value=0)
                )
                for cur_target in targets
            ]
        xs = df.loc[:, self.features]
        ys = df[self._target]
        return xs.to_numpy(na_value=0), (
            ys.to_numpy(na_value=0)
            if not ys_is_label else
            self.get_labels(subset=df, as_list=False).to_numpy()
        )

    def get_labels(self, as_list: bool = True, subset: Optional[DataFrame] = None) -> Union[List, Series]:
        df = self.dataframe if subset is None else subset
        if self._label is None or len(self._label) == 0:
            # Use indexes:
            if as_list:
                return df.index.tolist()
            else:
                return df.index.to_series()
        elif len(self._label) == 1:
            if as_list:
                return df[self._label[0]].tolist()
            else:
                return df[self._label[0]]
        else:
            if as_list:
                return [
                    self._multi_label_join.join(row.tolist())
                    for _, row in df[list(self._label)].iterrows()  # Will bug in Dendogram
                ]
            else:
                return df[list(self._label)]

    def align(self, other: "DataframeWrapper"):
        done = []
        for action in self._processing:
            if action in done:
                continue
            if action == "relative":
                other.normalized.make_relative(inplace=True)
            elif action == "normalization":
                other.normalized.normalize()
            elif action == "update-feature":
                missing_cols = list(set(self.features).difference(other.features))
                other.dataframe.loc[:, missing_cols] = 0.0
                other.update_features(self.features)
            else:
                raise BadParameter(f"Unknown processing step {action}")
            done.append(action)

    def align_to(self, model: "DataframeWrapper"):
        return model.align(self)

    @property
    def target(self) -> str:
        return self._target

    @property
    def features(self) -> Tuple[str, ...]:
        return self._features

    @property
    def dataframe(self) -> DataFrame:
        return self._df

    def __len__(self):
        return len(self._df)

    @property
    def xs(self) -> DataFrame:
        return self._df.loc[:, self.features]

    @property
    def ys(self) -> Series:
        if not self.target:
            raise NoTargetsError("This df has no target column registered. "
                                 "Use `DataframeWrapper(df, target='colname')`")
        return self._df[self.target]

    @property
    def normalized(self):
        return self._normalized

    @property
    def is_normalized(self):
        return self.normalized.available()

    def update_features(self, features: Iterable[str]):
        new_features = list(set(features).difference(self.features))
        if new_features:
            print(f"{len(features)} new features found, setting them as 0.")
            self._df = self.dataframe.reindex(self.dataframe.columns.tolist() + new_features, axis=1)
            self._df.fillna(0, axis=1, inplace=True)
        self._features = tuple(features)
        self._processing.append("update-feature")

    def dropna(self):
        self.update_features(self.xs.dropna(axis=1).columns.tolist())
        self.dataframe.dropna(axis=1, inplace=True)
        self.normalized.dropna()

    def drop_low(
        self,
        documents_min: Union[int, float] = 1,
        frequency_min: int = 10
    ):
        if frequency_min > 1:
            large_features = (self.xs.sum() > frequency_min)
            large_features = large_features[large_features]
            self.update_features(self.xs[large_features.index].columns.tolist())

        if isinstance(documents_min, float):
            documents_min = round(documents_min * self.xs.shape[0])
        if documents_min >= 1:
            large_features = self.xs[self.xs.columns].apply(
                lambda col: len(col.unique()) > documents_min
            )
            large_features = large_features[large_features]
            self.update_features(self.xs[large_features.index].columns.tolist())
            self._processing.append("update-feature")

    def split(self, ratio: float = .1, target_based: bool = True) -> Tuple["DataframeWrapper", "DataframeWrapper"]:
        if not target_based:
            dev = self.dataframe.sample(frac=ratio).index.tolist()
            keep = set(self.dataframe.index.tolist()).difference(set(dev))
        else:
            targets: Dict[str, Tuple[List[int], List[int]]] = {
                target: reduce(self.dataframe[self.dataframe[self.target] == target].index.tolist(), ratio)
                for target in self.ys.unique().tolist()
            }

            keep, dev = zip(*targets.values())
            keep = flatten(keep)
            dev = flatten(dev)

        return self.copy_indexes(keep, drop=False), self.copy_indexes(dev, drop=False)

    def copy_indexes(self, indexes, drop: bool = False, copy: bool = True) -> "DataframeWrapper":
        dfw = DataframeWrapper(
            df=self.dataframe.loc[indexes].copy(deep=copy),
            is_normalized=self.is_normalized,
            target=self.target,
            x_ignore=self._x_ignore,
            multi_label_joins=self._multi_label_join,
            label=self._label
        )
        dfw._features = self.features

        if self.normalized:
            if self.dataframe is self.normalized.dataframe:
                dev_normalized = dfw.dataframe
            else:
                dev_normalized = self.normalized.dataframe.loc[indexes].copy(deep=copy)
                if drop:
                    self.normalized.dataframe.drop(index=indexes, inplace=True)
            dfw.normalized._dataframe = dev_normalized
        if drop:
            self.dataframe.drop(index=indexes, inplace=True)
        dfw._processing = self._processing
        return dfw

    def k_folds(self, n=5, target_based=True) -> List[
        Tuple["DataframeWrapper", "DataframeWrapper", "DataframeWrapper"]]:
        """ Produces `n` splits of train, dev, test using targets in an equal way.

        """
        if not target_based:
            raise NotImplementedError()

        targets = {
            target: self.dataframe[self.dataframe[self.target] == target].index.tolist()
            for target in self.ys.unique().tolist()
        }

        for key, values in targets.items():
            if len(values) < n * 2:
                warnings.warn(f"Target {key} has less than N-Folds*2 values (Total: {len(values)}). "
                              f"To evaluate good pairing, we need at least two pairs in dev and test. "
                              f"It leaves only negative pairing out for {key}")

        # {target: [Tuple of train, dev, test indexes]}
        targets: Dict[Any, List[Tuple[List[int], List[int], List[int]]]] = {
            target: list(get_kfolds_train_dev_test(list(get_kfolds_splits(indexes, n))))
            for target, indexes in targets.items()
        }

        out = []

        for i in range(n):
            train_indexes, dev_indexes, test_indexes = [], [], []
            for target in targets:
                tr, de, te = targets[target][i]
                train_indexes.extend(tr), dev_indexes.extend(de), test_indexes.extend(te)
            random.shuffle(train_indexes)
            random.shuffle(dev_indexes)
            random.shuffle(test_indexes)
            out.append(
                (
                    self.copy_indexes(train_indexes, copy=False),
                    self.copy_indexes(dev_indexes, copy=False),
                    self.copy_indexes(test_indexes, copy=False)
                )
            )

        return out

    class _NormalizedWrapper:
        def __init__(self, top_object: "DataframeWrapper"):
            self._top = top_object
            # If self._dataframe is True, it refers to the top df
            self._dataframe: Optional[Union[DataFrame, bool]] = None

        def available(self):
            return self._dataframe is not None

        @property
        def dataframe(self) -> DataFrame:
            if self._dataframe is True:
                return self._top.dataframe
            elif self._dataframe is not None:
                return self._dataframe
            else:
                raise BadParameter("Current df had no normalization applied. Call `.normalize()`")

        def make_relative(self, inplace: bool = False):
            """ Transform the inner features as relative frequencies """
            if not inplace:
                self._dataframe = self._top.dataframe.copy(deep=True)
            else:
                self._dataframe = True
            self.dataframe.loc[:, self._top.features] = df_relative_frequencies(self.xs)
            self._top._processing.append("relative")

        def normalize(self):
            if self.dataframe.shape[0] > 1:
                self.dataframe.loc[:, self._top.features] = normalizations(self.xs, inplace=False)
                self._top._processing.append("normalization")
            else:
                logger.warning("Can't normalize a Dataframe with a single row")

        @property
        def xs(self):
            return self.dataframe.loc[:, self._top.features]

        @property
        def ys(self):
            return self._top.dataframe.ys

        def dropna(self):
            if self.available():
                self.dataframe.dropna(axis=1, inplace=True)

