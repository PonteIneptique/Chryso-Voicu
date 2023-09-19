import logging
from typing import Tuple, Optional, List
from math import ceil
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.cluster.hierarchy import dendrogram, fcluster, set_link_color_palette
from pyshs import catdes


__all__ = ["ManhattanClassifier"]

from freestyl.stats.manhattan import manhattan_ward
from freestyl.stats.moisl import moisl
from freestyl.utils import get_colors
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.errors import BadParameter

logger = logging.getLogger(__name__)


class ManhattanClassifier:
    def __init__(self, dataframe: DataframeWrapper = None):
        self._df: Optional[DataframeWrapper] = dataframe
        self._cah: Optional[np.array] = None
        self._clusters = None

    @property
    def features(self) -> Tuple[str, ...]:
        return self._df.features

    @property
    def cah(self) -> np.array:
        if self._cah is None:
            raise BadParameter("ManhattanClassifier.build() not called")
        return self._cah

    @property
    def data(self):
        return self._df

    @property
    def clusters(self):
        if self._clusters is None:
            raise BadParameter("Call .get_clusters(k) before asking for the .clusters property")
        return self._clusters

    def build(
        self,
        data: Optional[DataframeWrapper] = None,
        apply_moisl: Optional[float] = 1.91,
        normalize: bool = True
    ) -> DataframeWrapper:
        if data is None and self._df is None:
            raise BadParameter("The classifier does not use any df. Feed one at class initialization or"
                               " at building time.")
        elif data is not None:
            logger.info("Setting-up the df")
            self._df = data

        if apply_moisl:
            if self.data.is_normalized:
                raise BadParameter("Not yet implemented to deal with already normalized df")
            logger.info(f"Applying moisl with z={apply_moisl}")
            features = moisl(self.data.xs, z=apply_moisl).keep
            logger.info(f"Keeping only {features.tolist().count(True)} and updating df")
            self.data.update_features(
                # Very not optimized but I am lazy right now
                pd.Series(self.data.features)[features.tolist()].tolist()
            )

        logger.info(f"Changing the df with a relative measure for features")
        self.data.normalized.make_relative(inplace=False)

        if normalize:
            self.data.normalized.normalize()

        logger.info("Applying Manhattan Ward Linkage")
        self._cah = manhattan_ward(self.data.normalized.xs, optimal_ordering=True)
        return self.data

    def get_clusters(self, k: int = 9) -> pd.DataFrame:
        classes = fcluster(self.cah, k, criterion="maxclust").astype(str)

        labels = self.data.get_labels(as_list=False)
        if isinstance(labels, pd.DataFrame):
            df = labels.copy(deep=True)
            df["Class"] = classes
        elif isinstance(labels, pd.Series):
            serie = labels.copy(deep=True)
            df = pd.DataFrame([
                pd.Series(classes, name="Class"),
                serie
            ])
        else:
            df = pd.DataFrame(
                pd.Series(classes, name="Class"),
                index=labels
            )
        self._clusters = df
        return self._clusters

    def plot_dendogram(
        self,
        ax: Optional[plt.Axes] = None,
        k: int = 9
    ) -> Tuple[plt.Axes, dict]:
        """ Plot a dendogram based on the pre-computed linkage

        :param ax: If set, plots a dendogram on the given axes
        :param k: Number of clusters to cut at

        :returns:
            The axes given
            The dendogram scipy dictionary
                (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html )
            A df where individuals are the labels of the original y Dataframe and values are the cluster
                identifier

        ToDo: Fix a bug where k yields less 8 K instead of 9 and C yelds 10 classes (10 colors ?) instead of 9
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        _colors = get_colors(k + 1).tolist()
        set_link_color_palette([colors.to_hex(c, keep_alpha=True) for c in _colors])
        d = dendrogram(
            self.cah,
            ax=ax,
            labels=self.data.get_labels(as_list=True),
            color_threshold=self.cah[-(k-1), 2]
        )

        if ax is not None:
            for coll in ax.collections[:-1]:  # the last collection is the ungrouped level
                xmin, xmax = np.inf, -np.inf
                ymax = -np.inf
                for p in coll.get_paths():
                    (x0, _), (x1, y1) = p.get_extents().get_points()
                    xmin = min(xmin, x0)
                    xmax = max(xmax, x1)
                    ymax = max(ymax, y1)
                rec = plt.Rectangle((xmin - 4, 0), xmax - xmin + 8, ymax * 1.05,
                                    facecolor="none", alpha=0.8, edgecolor=coll.get_color()[0], linestyle="--")
                ax.add_patch(rec)

        # Retrieve colors to align leaves title with colors
        set_link_color_palette(None)
        return ax, d

    def get_features_eta2(self, top: Optional[int] = None) -> pd.DataFrame:
        normalized = self.data.normalized.xs.copy(deep=True)
        normalized["Class"] = self.clusters.Class
        _, x = catdes(normalized, vardep="Class")
        x.sort_values("Eta 2", ascending=False, inplace=True)
        if top:
            x = x.iloc[:top, :]
        return x

    def eta2_to_classes(self, eta2: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        List[str]
    ]:
        cols = eta2.index.tolist()
        eta2 = self.data.normalized.dataframe[cols].copy(deep=True)
        eta2["Class"] = self.clusters.Class
        return eta2, cols

    def plot_eta_2(
            self,
            eta2: pd.DataFrame,
            features: List[str],
            columns: int = 2,
            fig_kwargs: Optional[dict] = None
    ) -> Tuple[
        plt.Figure, Tuple[plt.Axes]
    ]:
        palette = get_colors(len(self.clusters.Class.unique()) + 1)
        nrows: int = ceil(len(features) / columns)
        fig, axes = plt.subplots(nrows=nrows, ncols=columns, sharex=True, **(fig_kwargs or {}))

        for ax, feature in zip([a for row in axes for a in row], features):
            sb.violinplot(data=eta2, x="Class", y=feature, ax=ax,
                          order=sorted(eta2["Class"].unique()),
                          palette=palette)

        return fig, axes
