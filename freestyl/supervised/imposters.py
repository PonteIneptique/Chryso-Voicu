import random
from typing import Literal, Optional, Union, Tuple, Dict, List, Callable
from logging import getLogger
from collections import defaultdict, Counter
from operator import itemgetter

import pandas as pd
import sklearn.preprocessing as preproc
import sklearn.pipeline as skp
from sklearn.metrics import DistanceMetric

import numpy as np
import matplotlib.pyplot as plt
import imblearn.under_sampling as under
import imblearn.over_sampling as over
import imblearn.combine as comb
import tqdm
import imblearn.pipeline as imbp

from pandas import DataFrame
from sklearn import preprocessing
from freestyl.errors import BadParameter, PipelineNotBuilt
from freestyl.stats.frequencies import relative_frequencies
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.base import BaseSupervisedPipeline
from freestyl.stats.moisl import moisl


__all__ = ["ImposterPipeline"]


logger = getLogger(__name__)
Pipeline = Union[skp.Pipeline, imbp.Pipeline]


class DistancePredict:
    def __init__(self, metric: str = "manhattan"):
        self.metric = DistanceMetric.get_metric(metric)

    def predict(self, X, test_index=-1):
        pairs = self.metric.pairwise(X)
        score = pairs[:test_index, test_index:]
        closest = np.argmin(score, axis=0)
        score = score[closest]
        return closest, score


class ImposterPipeline(BaseSupervisedPipeline):
    def __init__(self):
        self._pipeline = None
        self._hparams = {}
        self._features: Optional[List[str]] = None
        self._le: preprocessing.LabelEncoder = preprocessing.LabelEncoder()

    @property
    def classes(self) -> List[str]:
        return self.label_encoder.classes_.tolist()

    @property
    def label_encoder(self) -> preprocessing.LabelEncoder:
        return self._le

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise PipelineNotBuilt("`ImposterPipeline.build()` or `ImposterPipeline.from_hparams()` has not been used: "
                                   "No pipeline to fit has been found.")
        return self._pipeline

    def build(
            self,
            imposter_ratio: float = .5,
            feature_ratio: float = .5,
            normalize: bool = True,
            balance_candidate: bool = False,
            sampling: Optional[Literal["down", "Tomek", "up", "SMOTE", "SMOTETomek"]] = None,
            #adjust_class_weights: bool = False,
            #kernel: Literal["LinearSVC", "SVC"] = "LinearSVC",
            metric: str = "manhattan",
            seed: int = 42
    ) -> "ImposterPipeline":
        """ Build the pipeline

        :param imposter_ratio: Percentage of imposters that are required to be used
        :param feature_ratio: Percentage of feature to keep per run
        :param normalize: perform normalisations, i.e. z-scores and L2 (default True)
        :param sampling: up/downsampling strategy to use in imbalanced datasets
        :param adjust_class_weights: adjust class weights to balance imbalanced datasets, with weights
         inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
        :param kernel: kernel for SVM
        :param seed: Random Seed
        """
        if balance_candidate not in {None, "min", "max"}:
            raise ValueError(f"Cannot use `{balance_candidate}` as a value for balance_candidate, only None, `min`"
                             f" and `max` are allowed.")

        self._hparams["pipeline"] = {
            "imposter_ratio": imposter_ratio,
            "feature_ratio": feature_ratio,
            "normalize": normalize,
            "sampling": sampling,
            "metric": metric,
            "seed": seed,
            "balance_candidate": balance_candidate
        }
        return self

    def _get_pipeline(self) -> Dict[str, Callable]:
        logger.info("Creating pipeline according to user choices")
        estimators = []

        if self._hparams["pipeline"]["normalize"]:
            # Z-scores
            logger.info("Using normalization")
            estimators.append(('scaler', preproc.StandardScaler()))
            estimators.append(('normalizer', preproc.Normalizer()))

        if self._hparams["pipeline"]["sampling"] is not None:
            # cf. machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification
            # https://github.com/scikit-learn-contrib/imbalanced-learn
            # Tons of option, look up the best ones
            sampling = self._hparams["pipeline"]["sampling"]
            seed = self._hparams["pipeline"]["seed"]
            logger.info("Implementing strategy to solve imbalance in data using sampling")

            if sampling == 'down':
                estimators.append(('sampling', under.RandomUnderSampler(random_state=seed, replacement=False)))
            elif sampling == 'Tomek':
                estimators.append(('sampling', under.TomekLinks()))
            elif sampling == 'up':
                estimators.append(('sampling', over.RandomOverSampler(random_state=seed)))
            elif sampling == 'SMOTE':
                estimators.append(('sampling', over.SMOTE(random_state=seed)))
            elif sampling == 'SMOTETomek':
                estimators.append(('sampling', comb.SMOTETomek(random_state=seed)))
            else:
                raise BadParameter(f"Unknown sampling method `{sampling}`")

        logger.info("Adding the distance metric estimator")
        estimators.append(("predictor", DistancePredict(self._hparams["pipeline"]["metric"])))
        return dict(estimators)

    def apply_threshold(self, probs, threshold: float):
        preds_max_value = probs.max(axis=1)
        new_class_mask = (preds_max_value < threshold).astype(int) * np.full((probs.shape[0]), self._OoS)
        keep_class_mask = (preds_max_value >= threshold).astype(int)
        # max_class * keep_class makes "bad prediction" go 0
        # new_class_mask adds the value
        return probs.argmax(axis=1) * keep_class_mask + new_class_mask

    def cross_validate(self, *args, **kwargs):
        raise NotImplementedError("The ImposterPipeline class does not support the `cross_validate` method as it only "
                                  "uses distances. Use predict")

    def fit(self,
            data: DataframeWrapper,
            apply_moisl: Optional[float] = 1.91,
            *args, **kwargs
            ) -> "ImposterPipeline":
        if apply_moisl:
            if data.is_normalized:
                raise BadParameter("Not yet implemented to deal with already normalized df")
            logger.info(f"Applying moisl with z={apply_moisl}")
            features = moisl(data.xs, z=apply_moisl).keep
            logger.info(f"Keeping only {features.tolist().count(True)} and updating df")
            data.update_features(
                # Very not optimized but I am lazy right now
                pd.Series(data.features)[features.tolist()].tolist()
            )
        return self

    def predict(
            self,
            data: DataframeWrapper,
            test: Optional[DataframeWrapper] = None,
            use_normalized: bool = True,
            iterations: int = 100,
            verbose: bool = False,
            **kwargs) -> DataFrame:
        self.label_encoder.fit_transform(data.ys.tolist())

        if test is not None:
            test.align_to(data)

        def wrapper(iterable, **kwargs):
            if verbose:
                return tqdm.tqdm(iterable, **kwargs)
            return iterable

        score = defaultdict(lambda: defaultdict(Counter))
        for all_candidate_xs, all_candidate_ys in wrapper(data.get_subset("candidate", group=True)):
            for _ in wrapper(range(iterations), leave=False):
                impostors_xs, impostors_ys = data.get_subset(
                    "impostor",
                    sample=self._hparams["pipeline"]["imposter_ratio"]
                )
                # ToDo: If all are candidates, get other candidates
                candidate_xs = all_candidate_xs.copy()
                if balancing := self._hparams["pipeline"]["balance_candidate"]:
                    c = Counter(impostors_ys)
                    if balancing == "min":
                        _, max_candidate = min(c.items(), key=itemgetter(1))
                        logger.info(f"Reducing to the minimum amount of impostors {max_candidate}")
                    else:
                        _, max_candidate = c.most_common(1)[0]
                        logger.info(f"Reducing to the maximum amount of impostors {max_candidate}")
                    np.random.shuffle(candidate_xs)
                    candidate_xs = candidate_xs[:max_candidate, :]
                if test is not None:
                    test_xs = test.xs.to_numpy(na_value=0)
                    test_ys = test.get_labels(as_list=False)
                else:
                    test_xs, test_ys = data.get_subset("test", ys_is_label=True)
                    if test_xs.shape[0] == 0:
                        raise ValueError("You forgot to provide a test")

                assert test_xs.shape[1] == all_candidate_xs.shape[1]

                # Only use a fraction of the features
                features = list(range(all_candidate_xs.shape[-1]))
                random.shuffle(features)
                max_features = int(round(all_candidate_xs.shape[-1] * self._hparams["pipeline"]["feature_ratio"], 0))
                features = features[:max_features]
                candidate_xs = np.nan_to_num(relative_frequencies(candidate_xs[:, features]))
                impostors_xs = np.nan_to_num(relative_frequencies(impostors_xs[:, features]))
                test_xs = np.nan_to_num(relative_frequencies(test_xs[:, features]))

                pipeline = self._get_pipeline()
                if "scaler" in pipeline:
                    all_xs = pipeline["scaler"].fit_transform(
                        np.concatenate((candidate_xs, impostors_xs, test_xs), axis=0)
                    )
                    pipeline["normalizer"].fit(all_xs)
                    candidate_xs = all_xs[:candidate_xs.shape[0]]
                    impostors_xs = all_xs[candidate_xs.shape[0]:candidate_xs.shape[0]+impostors_xs.shape[0]]
                    test_xs = all_xs[candidate_xs.shape[0]+impostors_xs.shape[0]:]

                if "sampler" in pipeline:
                    impostors_xs, impostors_ys = pipeline["sample"].fit_resample(impostors_xs, impostors_ys)

                # Shape text_xs.shape[0]
                closest_idx, _ = pipeline["predictor"].predict(
                    np.concatenate(
                        (candidate_xs, impostors_xs, test_xs),
                        axis=0
                    ),
                    test_index=-test_xs.shape[0]
                )
                assert closest_idx.shape[0] == test_xs.shape[0]

                all_ys = np.concatenate((all_candidate_ys, impostors_ys), axis=0)
                proposed_candidate = all_ys[closest_idx]
                current_candidate = all_candidate_ys[0]
                for text, closest_class in zip(test_ys.tolist(), proposed_candidate.tolist()):
                    score[text][current_candidate][closest_class] += 1
                # ToDo: Implement a thing to check for the test
                # Note that tests can be one or multiple texts !

        return DataFrame([
            [
                text, candidate, proposal, val, iterations, val/iterations
            ]
            for text, scores in score.items()
            for candidate, proposal_dicts in scores.items()
            for proposal, val in proposal_dicts.items()
        ], columns=["Text", "Candidate", "Proposal", "Proposed", "Total-Iterations", "Percent"])


if __name__ == "__main__":
    import pandas as pd
    from loc_script import read_feature_frame
    from freestyl.dataset.dataframe_wrapper import DataframeWrapper

    data = read_feature_frame("/home/thibault/dev/chrysostylom/data/02-ngrams/affixes-3grams.csv", min_words=2000)
    data.reset_index(inplace=True)
    data.set_index("Titre", inplace=True)
    data.head(2)

    # data = data.sample(1.) # Shuffle the data !
    data["ImpostorMode"] = data.Auteur.apply(lambda x: "candidate" if x == "Chrysostome" else "impostor")
    # %%
    #keep = pd.read_csv("/home/tclerice/dev/chrysostylom/03-GT.csv", index_col="Titre").index.tolist()
    #test = data.loc[~data.index.isin(keep), :].reset_index()
    #print(f"Before filtering: {data.shape[0]} texts, after {data.loc[data.index.isin(keep), :].shape[0]}")
    #data = data.loc[data.index.isin(keep), :]
    #data.reset_index(inplace=True)
    data = DataframeWrapper(data, target="Auteur", label="Titre", status_col="ImpostorMode")
    data.xs.head(2)
    data.drop_low(documents_min=.05, frequency_min=1000)

    qcsd_df = pd.read_csv("/home/thibault/dev/chrysostylom/data/02-ngrams/QCSD-affixes-3grams.csv")
    QCSD = DataframeWrapper(qcsd_df, target="Auteur", label="Titre")

    pipeline = ImposterPipeline()
    pipeline.build(sampling="Tomek", seed=42, feature_ratio=.5, balance_candidate="max")
    pipeline.fit(data, apply_moisl=1.71)
    print(pipeline.predict(data, QCSD, iterations=300))
