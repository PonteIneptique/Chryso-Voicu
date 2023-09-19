from typing import Optional, Tuple, Generator, List, Any, Dict
from logging import getLogger
from itertools import combinations
from multiprocessing import Pool


import tqdm
from pandas import DataFrame
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.base import BaseSupervisedPipeline
from freestyl.supervised.compressed.models import Model
from sklearn.linear_model import LogisticRegression


__all__ = ["CompressedModel"]


logger = getLogger(__name__)


class CompressedModel(BaseSupervisedPipeline):
    def __init__(self, workers: int = 16, *args, **kwarg):
        self.vocab_size: int = 0
        self.ppm_order: int = 5
        self.normalization: Optional[None] = None
        self.logistic = LogisticRegression()
        self.verbose: bool = True
        self.workers: int = workers
        self.cache: Dict[str, Model] = {}

    def tqdm(self, iterator):
        if self.verbose:
            return tqdm.tqdm(iterator)
        else:
            return iterator

    def build(self, ppm_order: int = 5, normalization: Optional[str] = None):
        self.ppm_order = 5
        return

    def fit(self, data: DataframeWrapper, *args, **kwargs):
        if not data.hasFullText():
            raise ValueError("The DataFrame requires a `full-text` column.")
        self.vocab_size = len(set("".join(data.fulltext)))

        # Build dataset receiver
        truthes: List[int] = []
        distances: List[Tuple[float, float]] = []

        with Pool(processes=self.workers) as pool:
            for dist, truth in self.tqdm(pool.imap(self._compute_without_label, self.get_all_pairs(data))):
                distances.append(dist)
                truthes.append(truth)

        self.logistic.fit(distances, truthes)

    def _compute_without_label(self, args):
        text1, text2, is_a_pair = args
        return self._get_cross_entropy(text1, text2), is_a_pair

    def _compute_with_label_predict(self, args):
        text1, text2, is_a_pair, y1, y2 = args
        dist = self._get_cross_entropy(text1, text2)
        return self.logistic.predict_proba([dist]), is_a_pair, y1, y2

    def get_all_pairs(self, data: DataframeWrapper) -> Generator[Tuple[str, str, bool], None, None]:
        for text1, text2, truth, *_ in self.get_all_pairs_with_labels(data):
            yield text1, text2, truth

    def get_all_pairs_with_labels(self, data: DataframeWrapper) -> Generator[
        Tuple[str, str, bool, Any, Any], None, None
    ]:
        texts = data.fulltext
        ys = data.ys.tolist()
        labels = data.get_labels(as_list=True)
        for id1, id2 in combinations(range(len(texts)), 2):
            yield texts[id1], texts[id2], int(ys[id1] == ys[id2]), labels[id1], labels[id2]

    def predict(self, data: DataframeWrapper, radius: float = .01, *args, **kwargs):
        answers = []

        with Pool(processes=self.workers) as pool:
            for prediction, truth, y1, y2 in self.tqdm(
                    pool.imap(self._compute_with_label_predict, self.get_all_pairs_with_labels(data))
            ):
                # All values around (0.5 +/- radius) are transformed to 0.5
                if 0.5 - radius <= prediction[0, 1] <= 0.5 + radius:
                    prediction[0, 1] = 0.5

                answers.append({
                    'y1': y1,
                    'y2': y2,
                    'pair': bool(truth),
                    'distance': round(prediction[0, 1], 3)
                })

        return DataFrame(answers)

    def _get_cross_entropy(self, text1: str, text2: str,) -> Tuple[float, float]:
        """ Shortcut function to calculate the cross-entropy of text2 using the negated of text1 and vice-versa

        Returns the two cross-entropies
        """
        mod1 = Model(self.ppm_order, self.vocab_size)
        mod1.read(text1)
        d1 = mod1.get_cross_entropy(text2)
        mod2 = Model(self.ppm_order, self.vocab_size)
        mod2.read(text2)
        d2 = mod2.get_cross_entropy(text1)
        return round(d1, 4), round(d2, 4)
