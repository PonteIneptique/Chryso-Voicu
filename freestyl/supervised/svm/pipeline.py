import warnings
from typing import Literal, Optional, Union, Tuple, Dict, List
from logging import getLogger

import pandas as pd
import sklearn.svm as sk
import sklearn.decomposition as decomp
import sklearn.preprocessing as preproc
import sklearn.pipeline as skp
import sklearn.model_selection as skmodel
import sklearn.metrics as metrics
import sklearn.exceptions

import numpy as np
import matplotlib.pyplot as plt
import imblearn.under_sampling as under
import imblearn.over_sampling as over
import imblearn.combine as comb
import imblearn.pipeline as imbp

from pandas import DataFrame
from sklearn import preprocessing
from freestyl.errors import BadParameter, PipelineNotBuilt
from freestyl.utils import get_confusion_matrix_from_predictions
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.base import BaseSupervisedPipeline
from freestyl.supervised.svm.utils import plot_coefficients, ScoreLogger, ClassSpecificScorer

__all__ = ["SvmPipeline", "plot_coefficients"]


logger = getLogger(__name__)
Pipeline = Union[skp.Pipeline, imbp.Pipeline]


class LocalEncoder:
    def __init__(self, out_of_vocab: str = "Unknown"):
        self._classes = {}
        self._oov = out_of_vocab

    @property
    def classList(self):
        return list(self._classes.keys())

    def token_to_id(self, key: str, fit=False) -> int:
        if key in self._classes:
            return self._classes[key]
        if fit:
            self._classes[key] = len(self._classes)
            return self._classes[key]
        raise ValueError("Unknown Class")

    def fit_transform(self, sequence):
        return [self.token_to_id(key, fit=True) for key in sequence]

    def transform(self, sequence):
        return [self.token_to_id(key) for key in sequence]

    def inverse_transform(self, sequence):
        itt = {value:key for key, value in self._classes.items()}
        return [itt[val] for val in sequence]


class SvmPipeline(BaseSupervisedPipeline):
    def __init__(self):
        self._pipeline = None
        self._hparams = {}
        self._features: Optional[List[str]] = None
        self._le: LocalEncoder = LocalEncoder()

    @property
    def classes(self) -> List[str]:
        return self.label_encoder.classList

    @property
    def label_encoder(self) -> preprocessing.LabelEncoder:
        return self._le

    @property
    def features(self):
        if self._features is None:
            raise PipelineNotBuilt(".fit() was never called, hence features are unknown")
        return self._features

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise PipelineNotBuilt("`SvmPipeline.build()` or `SvmPipeline.from_hparams()` has not been used: "
                                   "No pipeline to fit has been found.")
        return self._pipeline

    def build(
            self,
            reduce_dimensions: Optional[Literal["pca", "som"]] = None,
            normalize: bool = True,
            sampling: Optional[Literal["down", "Tomek", "up", "SMOTE", "SMOTETomek"]] = None,
            adjust_class_weights: bool = False,
            kernel: Literal["LinearSVC", "SVC"] = "LinearSVC",
            seed: int = 42
    ) -> Pipeline:
        """ Build the pipeline

        :param reduce_dimensions: dimensionality reduction of input data. Implemented values are pca and som.
        :param normalize: perform normalisations, i.e. z-scores and L2 (default True)
        :param sampling: up/downsampling strategy to use in imbalanced datasets
        :param adjust_class_weights: adjust class weights to balance imbalanced datasets, with weights
         inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
        :param kernel: kernel for SVM
        :param seed: Random Seed
        """
        self._hparams["pipeline"] = {
            "reduce_dimensions": reduce_dimensions,
            "normalize": normalize,
            "sampling": sampling,
            "adjust_class_weights": adjust_class_weights,
            "kernel": kernel,
            "seed": seed
        }

        cw = None
        if adjust_class_weights:
            cw = "balanced"

        logger.info("Creating pipeline according to user choices")
        estimators = []

        if reduce_dimensions == 'pca':
            logger.info("Dimension reduction: Using PCA")
            estimators.append(('reduce_dimensions', decomp.PCA()))

        if normalize:
            # Z-scores
            logger.info("Using normalization")
            estimators.append(('scaler', preproc.StandardScaler()))
            estimators.append(('normalizer', preproc.Normalizer()))

        if sampling is not None:
            # cf. machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification
            # https://github.com/scikit-learn-contrib/imbalanced-learn
            # Tons of option, look up the best ones

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

        logger.info("Choosing SVM")
        if kernel == "LinearSVC":
            logger.info("Using Linear SVC")
            estimators.append(('model', sk.LinearSVC(class_weight=cw)))
        else:
            logger.info("Using SVC")
            estimators.append(('model', sk.SVC(kernel=kernel, class_weight=cw)))

        logger.info("Creating the pipeline")

        if 'sampling' in [k[0] for k in estimators]:
            pipe = imbp.Pipeline(estimators)
        else:
            pipe = skp.Pipeline(estimators)

        self._pipeline = pipe

        return pipe

    def cross_validate(
            self,
            data: DataframeWrapper,
            method: Literal["leave-one-out", "k-fold"],
            k: int = 10,
            use_normalized: bool = True,
            threshold: Optional[float] = 0,
            oos_label: str = "Unattributed"
    ) -> Tuple[Optional[DataFrame], DataFrame, DataFrame, DataFrame]:

        logger.info("Formatting data")
        # Save the classes
        classes = data.ys.tolist()

        # if test is not None:
        #     classes_test = list(test.loc[:, 'author'])
        #     test = test.drop(['author', 'lang'], axis=1)
        #     preds_index = list(test.index)

        logger.info("Using cross-validation")
        if method == 'leave-one-out':
            logger.info(f"-> Cross-validation is using {method}")
            cross_validate = skmodel.LeaveOneOut()
        elif method == 'k-fold':
            logger.info(f"-> Cross-validation is using {method}")
            cross_validate = skmodel.KFold(n_splits=k)
        else:
            raise BadParameter(f"Cross-validation `{method}` is unknown")

        logger.info(f"Using {cross_validate.get_n_splits(data.normalized.xs if use_normalized else data.xs)} samples")

        additional_score: Dict = {
        }
        scores = None
        if method == "k-fold":
            additional_score = {
                **{
                    f"${classname}(Precision)": metrics.make_scorer(
                        ClassSpecificScorer(metrics.precision_score, classname),
                        needs_proba=False,
                        needs_threshold=False
                    )
                    for classname in data.ys.unique().tolist()
                },
                **{
                    f"${classname}(Recall)": metrics.make_scorer(
                        ClassSpecificScorer(metrics.recall_score, classname),
                        needs_proba=False,
                        needs_threshold=False
                    )
                    for classname in data.ys.unique().tolist()
                },
                **{
                    f"${classname}(F1)": metrics.make_scorer(
                        ClassSpecificScorer(metrics.recall_score, classname),
                        needs_proba=False,
                        needs_threshold=False
                    )
                    for classname in data.ys.unique().tolist()
                },
                **{
                    f"${classname}(RelSupport)": metrics.make_scorer(
                        ClassSpecificScorer(support, classname),
                        needs_proba=False,
                        needs_threshold=False
                    )
                    for classname in data.ys.unique().tolist()
                }
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
                scores = pd.DataFrame(skmodel.cross_validate(
                    self.pipeline,
                    data.normalized.xs if use_normalized else data.xs,
                    classes,
                    cv=cross_validate,
                    verbose=1,
                    n_jobs=-1,
                    scoring={
                        "acc": "accuracy",
                        "pre": "precision_macro",
                        "rec": "recall_macro",
                        "f1": "f1_macro",
                        **additional_score
                    }
                ))
            scores.sort_index(axis=1, inplace=True)

        if threshold is not None:
            *transformed, self._OoS = self.label_encoder.fit_transform(classes + [oos_label])
        else:
            transformed = self.label_encoder.fit_transform(classes)

        preds = skmodel.cross_val_predict(
            self.pipeline,
            data.normalized.xs if use_normalized else data.xs,
            transformed,
            cv=cross_validate,
            verbose=1,
            n_jobs=-1,
            method="decision_function"
        )

        if threshold is not None:
            preds_max_class = self.apply_threshold(preds, threshold=threshold)
        else:
            preds_max_class = preds.argmax(axis=1)

        nbclasses = preds.shape[-1]

        probas = DataFrame(
            preds,
            columns=self.label_encoder.classList[:nbclasses],
            index=data.get_labels()
        )
        probas["Truth"] = classes
        preds = self.label_encoder.inverse_transform(preds_max_class)
        probas["Prediction"] = preds
        probas["Correct"] = probas.Truth == probas.Prediction

        # and now, leave one out evaluation (very small redundancy here, one line that could be stored elsewhere)
        confusion_matrix = get_confusion_matrix_from_predictions(
            preds,
            classes
        )

        mis_attributions = DataFrame(
            [
                (index, truth, pred)
                for (index, truth, pred) in zip(
                    data.get_labels(),
                    list(classes),
                    list(preds)
                )
                if truth != pred
            ],
            columns=["id", "True", "Pred"]
        ).set_index('id')

        return scores, probas, confusion_matrix, mis_attributions

    def apply_threshold(self, probs, threshold: float):
        preds_max_value = probs.max(axis=1)
        new_class_mask = (preds_max_value < threshold).astype(int) * np.full((probs.shape[0]), self._OoS)
        keep_class_mask = (preds_max_value >= threshold).astype(int)
        # max_class * keep_class makes "bad prediction" go 0
        # new_class_mask adds the value
        return probs.argmax(axis=1) * keep_class_mask + new_class_mask

    def fit(
        self,
        data: DataframeWrapper,
        use_normalized: bool = True,
        *args,
        **kwargs
    ) -> Pipeline:
        """ Function to fit the SVM

        :param data: Training data
        """
        self._features = data.xs.columns.tolist()
        return self.pipeline.fit(data.normalized.xs if use_normalized else data.xs, data.ys.tolist())

    def predict(
            self,
            data: DataframeWrapper,
            use_normalized: bool = True,
            threshold: Optional[float] = 0,
            *args,
            **kwargs
    ) -> DataFrame:
        logger.info("Getting final predictions")
        # Get the decision function too
        myclasses = self.pipeline.classes_

        decs = self.pipeline.decision_function(data.normalized.xs if use_normalized else data.xs)
        dists = {}
        for myclass in enumerate(myclasses):
            dists[myclass[1]] = [d[myclass[0]] for d in decs]

        return DataFrame(
            data={
                'prediction': self.label_encoder.inverse_transform(
                    self.apply_threshold(decs, threshold) if threshold is not None else decs
                ),
                **dists
            },
            index=data.get_labels(as_list=True)
        )

    def itest(self, train: DataframeWrapper, test: DataframeWrapper) -> DataFrame:
        preds = self.pipeline.predict(test.xs)
        return get_confusion_matrix_from_predictions(preds, (*train.ys.tolist(), *test.ys.tolist()))

    @property
    def coefs(self):
        # From SK Learn: coef_ndarray of shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
        if self.pipeline.named_steps['model'].coef_.shape[0] == 1:
            return np.concatenate([
                self.pipeline.named_steps['model'].coef_,
                self.pipeline.named_steps['model'].coef_,
            ], axis=0)
        return self.pipeline.named_steps['model'].coef_

    def get_weights(self) -> DataFrame:
        assert self.pipeline.named_steps['model'].coef_.shape[1] == len(self.features), \
            "Pipeline features and known features are of different shape. This happens when dimension_reduction " \
            "was used"
        return DataFrame(
            self.coefs,
            index=self.pipeline.classes_,
            columns=self.features
        )

    def plot_weights(
            self,
            data: DataframeWrapper,
            top_features: int = 10,
            *args, **kwargs
    ) -> Dict[str, plt.Figure]:
        figures = {}
        for i in range(len(self.pipeline.classes_)):
            figures[self.pipeline.classes_[i]] = plot_coefficients(
                self.coefs[i],
                data.features,
                self.pipeline.classes_[i],
                top_features=top_features,
                *args,
                **kwargs
            )
        return figures


if __name__ == "__main__":
    import pandas as pd
    from loc_script import read_feature_frame
    from freestyl.dataset.dataframe_wrapper import DataframeWrapper
    from freestyl.stats.moisl import apply_moisl

    data = read_feature_frame("/home/tclerice/dev/chrysostylom/data/02-ngrams/fwords.csv", min_words=2000)
    data.reset_index(inplace=True)
    data.set_index("Titre", inplace=True)
    data.head(2)

    data["Target"] = data.Auteur  # data.Auteur.apply(lambda x: "Non-Chrysostome" if x != "Chrysostome" else x)
    data.head(2)
    # data = data.sample(1.) # Shuffle the data !
    data["Target"] = data.Target.apply(lambda x: x if "Pseudo" not in x else "Spuria")
    data["Target"] = data.Target.apply(lambda x: x if "Pseudo" not in x else "Spuria")
    # %%
    keep = pd.read_csv("/home/tclerice/dev/chrysostylom/03-GT.csv", index_col="Titre").index.tolist()
    test = data.loc[~data.index.isin(keep), :].reset_index()
    print(f"Before filtering: {data.shape[0]} texts, after {data.loc[data.index.isin(keep), :].shape[0]}")
    data = data.loc[data.index.isin(keep), :]
    data.reset_index(inplace=True)
    data = DataframeWrapper(data, target="Target", label="Titre", x_ignore=["Auteur"])
    data.xs.head(2)
    data.drop_low(documents_min=.05, frequency_min=1000)
    print(len(data.features))
    data.normalized.make_relative(inplace=True)

    pipeline = SvmPipeline()
    pipeline.build(sampling="Tomek", seed=42)
    pipeline.fit(data)