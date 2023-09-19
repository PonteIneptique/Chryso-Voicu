from typing import Optional, List, Tuple, Literal, TYPE_CHECKING
import collections

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_metric_learning.distances import SNRDistance

import torch
from sklearn.preprocessing import LabelEncoder

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.model import BaseSiameseModule
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader
from freestyl.supervised.siamese.sequential.model import SiameseSequentialModule
from freestyl.supervised.siamese.sequential.data import make_dataloader as SequentialDataLoader, DocumentEncoder
from freestyl.utils import TypedDataframe

BestMatchDataframe = pd.DataFrame
PairsDataframe = TypedDataframe[["ComparedClass", "ComparedLabel", "ComparatorClass", "ComparatorLabel",
                                 "Distance", "Attribution", "IsAPair"]]
ScoreDataframe = TypedDataframe[["Accuracy", "Recall", "Precision", "K"]]


class History_dict(LightningLoggerBase):
    # https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list) # copy not necessary here
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "Logger_custom_plot"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        return

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else: # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (not len(self.history['epoch']) or    # len == 0:
                    not self.history['epoch'][-1] == metric_value) : # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass


def train_dataframewrappers(
        train: DataframeWrapper,
        dev: DataframeWrapper,
        test: Optional[DataframeWrapper] = None,
        accelerator: Optional[str] = "gpu",
        sample: Optional[int] = False,
        batch_size: int = 16,
        min_delta: float = 1e-4,
        patience: int = 30,
        sequential_text_key: str = "modified_text",
        callbacks=None,
        gpus: Optional[int] = 0,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        sequential_min_token_freq: int = 5,
        mode: Literal["sequential", "features"] = "features",
        **hyperparams
) -> SiameseFeatureModule:
    """ Train a modul using three different datasets, hyperparams are provided directly to the model initializer
    """
    label_encoder = LabelEncoder()
    classes = set(train.ys.tolist() + dev.ys.tolist())

    if test is not None:
        classes.update(set(test.ys.tolist()))

    label_encoder.fit(sorted(list(classes)) + ["[OOD]"])

    if mode == "sequential":
        make_dataloader = SequentialDataLoader
        doc_encoder = DocumentEncoder(
            bilevel=(hyperparams.get("sequential_model", SiameseSequentialModule.DEFAULT) == "AttentionalGRU"),
        )
        doc_encoder.from_series(train.dataframe[sequential_text_key], min_frequency=sequential_min_token_freq)

        model = SiameseSequentialModule(
            document_encoder=doc_encoder,
            label_encoder=label_encoder,
            **hyperparams
        )
    else:
        make_dataloader = FeatureDataLoader
        model = SiameseFeatureModule(
            train.features,
            label_encoder=label_encoder,
            **hyperparams
        )

    dl_train = make_dataloader(
        dataframe_wrapper=train,
        model=model,
        batch_size=batch_size,
        shuffle=True,
        sample=sample
    )
    dl_dev = make_dataloader(
        dataframe_wrapper=dev,
        model=model,
        label_encoder=label_encoder,
        shuffle=False
    )
    if test is not None:
        dl_test = make_dataloader(
            dataframe_wrapper=test,
            model=model,
            label_encoder=label_encoder,
            shuffle=False
        )

    trainer = pl.Trainer(
        devices=gpus,
        accelerator=accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="dev_loss", mode="min",
                min_delta=min_delta, patience=patience, verbose=False,
            ),
            *(callbacks or [])
        ],
        #logger=History_dict(),
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        detect_anomaly=True
    )
    trainer.fit(model, dl_train, dl_dev)
    model.eval()
    if test is not None:
        results = trainer.test(model, dl_test)
    return model.cpu(), trainer


def train_k_folds(
        dataframe: DataframeWrapper,
        ks: int = 5,
        sample: bool = False,
        accelerator: Optional[str] = "cuda",
        **hyperparams
) -> List[Tuple[SiameseFeatureModule, DataframeWrapper, DataframeWrapper]]:
    """ Train SiameseModule over {ks}-fold using df

    Returns models with train and test dataloaders (might remove the later)
    """
    models = []
    for train, dev, test in dataframe.k_folds(n=ks):
        models.append((
            train_dataframewrappers(
                train, dev, test,
                accelerator=accelerator,
                sample=sample,
                **hyperparams
            ),
            train,
            test
        ))
    return models


def _get_prob_pair_dataframe(trainer, model, compared, threshold):
    label_encoder = model.hparams["label_encoder"]

    if isinstance(model, SiameseFeatureModule):
        make_dataloader = FeatureDataLoader
    else:
        make_dataloader = SequentialDataLoader

    with model:
        vectors, classes, prob_matrix = trainer.predict(
            model=model, dataloaders=make_dataloader(compared, model=model, batch_size=len(compared.ys))
        )[0]

    prob_matrix = prob_matrix.cpu()
    vectors = vectors.cpu()
    classes = classes.cpu()

    classes_idx = torch.arange(classes.shape[0])
    pairs = torch.combinations(classes_idx, 2)
    prob_matrix = prob_matrix[pairs[:, 0], pairs[:, 1]].tolist()
    compared_classes = label_encoder.inverse_transform(classes.numpy())
    compared_labels = compared.get_labels(as_list=False)
    outputs = []

    # ToDo: Finish
    right_pairs = vectors[pairs[:, 1], :]
    right_classes = compared_classes[pairs[:, 1]]
    right_labels = compared_labels.iloc[pairs[:, 1]]
    left_pairs = vectors[pairs[:, 0], :]
    left_classes = compared_classes[pairs[:, 0]]
    left_labels = compared_labels.iloc[pairs[:, 0]]

    if isinstance(left_labels, pd.DataFrame):
        left_labels = [
            " - ".join(
                row.tolist()
            )
            for _, row in left_labels.iterrows()
        ]
    if isinstance(right_labels, pd.DataFrame):
        right_labels = [
            " - ".join(
                row.tolist()
            )
            for _, row in right_labels.iterrows()
        ]

    computed_distances = model.distance.pairwise_distance(left_pairs, right_pairs)

    # for i in range(len(prob_matrix)):
    for (compared_class, compared_label, comparator_class, comparator_label, comparison_distance, prob) in zip(
            left_classes.tolist(),
            left_labels,
            right_classes.tolist(),
            right_labels,
            computed_distances.tolist(),
            prob_matrix
    ):
        outputs.append(
            {
                "ComparedClass": compared_class,
                "ComparedLabel": compared_label,
                "ComparatorClass": comparator_class,
                "ComparatorLabel": comparator_label,
                "Distance": comparison_distance,
                "Probability": prob,
                "Attribution": prob > threshold,
                "IsAPair": (
                    None if "[OOD]" in (comparator_class, compared_class)
                    else compared_class == comparator_class
                ),
                "K": 0
            }
        )
    # df = DataFrame([
    #     (
    #         prob_matrix[i],
    #         classes[idx[i][0]],
    #         classes[idx[i][1]],
    #         #idx[i][0],
    #         #idx[i][1]
    #     )
    #
    # ], columns=["Probability", "Left", "Right"]#, "ID Left", "ID Right"]
    # )
    return PairsDataframe(outputs)


def get_df_prediction(
    trainer: pl.Trainer,
    model: BaseSiameseModule,
    compared: DataframeWrapper,
    comparator: Optional[DataframeWrapper] = None,
    threshold: float = .5,
    k: int = 0
) -> PairsDataframe:
    """ Gets predictions for a df using another df as comparison

    """
    label_encoder: LabelEncoder = model.hparams.label_encoder

    if isinstance(model, SiameseFeatureModule):
        make_dataloader = FeatureDataLoader
    else:
        make_dataloader = SequentialDataLoader

    model.eval()

    if model.linear is not None:
        return _get_prob_pair_dataframe(trainer, model, compared, threshold=threshold)
    else:
        # We first get the wonderful classes of compared
        compared_vectors, compared_classes = zip(
            *trainer.predict(model, make_dataloader(compared, model=model, batch_size=8))
        )
    if trainer.accelerator != "cpu":
        compared_vectors = torch.cat(compared_vectors).cpu()
        compared_classes = torch.cat(compared_classes).cpu()
    compared_classes = label_encoder.inverse_transform(compared_classes.numpy())
    compared_labels = compared.get_labels(as_list=False)
    compared_idx = torch.arange(compared_vectors.shape[0])

    # If we have a comparator set of vectors, we set up comparator and then we build pairs
    if comparator is not None:
        comparator_vectors, comparator_classes = zip(
            *trainer.predict(model, make_dataloader(comparator, model=model, batch_size=8))
        )
        if trainer.accelerator != "cpu":
            comparator_vectors = torch.cat(comparator_vectors).cpu()
            comparator_classes = torch.cat(comparator_classes).cpu()
        comparator_classes = label_encoder.inverse_transform(comparator_classes.numpy())
        comparator_labels = comparator.get_labels(as_list=True)
        comparator_idx = torch.arange(compared_vectors.shape[0])

        # We build up pairs
        pairs = torch.cartesian_prod(compared_idx, comparator_idx)
        right_pairs = comparator_vectors[pairs[:, 1], :]
        right_classes = comparator_classes[pairs[:, 1], :]
        right_labels = comparator_labels.iloc[pairs[:, 1]]
    elif isinstance(model.distance, SNRDistance):
        pairs = torch.combinations(compared_idx, 2)
        pairs = torch.cat(
            [
                pairs,
                torch.cat(
                    [
                        pairs[:, 1].unsqueeze(1),
                        pairs[:, 0].unsqueeze(1)
                    ],
                    dim=1
                )
            ],
            dim=0
        )
        right_pairs = compared_vectors[pairs[:, 1], :]
        right_classes = compared_classes[pairs[:, 1]]
        right_labels = compared_labels.iloc[pairs[:, 1]]
    else:
        pairs = torch.combinations(compared_idx, 2)
        right_pairs = compared_vectors[pairs[:, 1], :]
        right_classes = compared_classes[pairs[:, 1]]
        right_labels = compared_labels.iloc[pairs[:, 1]]

    left_pairs = compared_vectors[pairs[:, 0], :]
    left_classes = compared_classes[pairs[:, 0]]
    left_labels = compared_labels.iloc[pairs[:, 0]]

    computed_distances = model.distance.pairwise_distance(left_pairs, right_pairs)

    if isinstance(left_labels, pd.DataFrame):
        left_labels = [
            " - ".join(
                row.tolist()
            )
            for _, row in left_labels.iterrows()
        ]
    if isinstance(right_labels, pd.DataFrame):
        right_labels = [
            " - ".join(
                row.tolist()
            )
            for _, row in right_labels.iterrows()
        ]

    # closest = BestMatchDataframe(
    #     comparator_classes[computed_distances.argsort()][:, :top_n],
    #     columns=["Closest"] if top_n == 1 else [f"Closest{i}" for i in range(top_n)],
    #     index=compared_labels
    # )
    # closest["Author"] = compared_classes
    outputs = []
    for (compared_class, compared_label, comparator_class, comparator_label, comparison_distance) in zip(
        left_classes.tolist(),
        left_labels,
        right_classes.tolist(),
        right_labels,
        computed_distances.tolist()
    ):
        outputs.append(
            {
                "ComparedClass": compared_class,
                "ComparedLabel": compared_label,
                "ComparatorClass": comparator_class,
                "ComparatorLabel": comparator_label,
                "Distance": comparison_distance,
                "Probability": 1-comparison_distance,
                "Attribution": comparison_distance < threshold,
                "IsAPair": (
                    None if "[OOD]" in (comparator_class, compared_class)
                    else compared_class == comparator_class
                ),
                "K": k
            }
        )

    return PairsDataframe(outputs)


def score_from_preds(pairs_dataframe: PairsDataframe, k=0) -> ScoreDataframe:
    pairs_dataframe["Correct"] = pairs_dataframe.apply(
        lambda row: row.IsAPair == row.Attribution if row.IsAPair != None else None, axis=1)

    def compute(dataframe, filter: Optional[pd.Series] = None) -> pd.Series:
        if filter is not None:
            dataframe = dataframe[filter]
        pivot = dataframe.groupby(
            ["ComparedClass", "ComparatorClass", "Correct"]
        )["ComparedLabel"].count().reset_index().pivot_table(
            columns="Correct", index="ComparedClass", aggfunc="sum"
        ).fillna(0)
        p = pivot.get(('ComparedLabel', True), 0)
        n = pivot.get(('ComparedLabel', False), 0)
        return p / (p + n)

    accuracy = compute(pairs_dataframe)
    precision = compute(pairs_dataframe, pairs_dataframe.ComparedClass == pairs_dataframe.ComparatorClass)
    recall = compute(pairs_dataframe, pairs_dataframe.ComparedClass != pairs_dataframe.ComparatorClass)

    df = pd.merge(
        right_index=True, left_index=True,
        right=accuracy.rename("Accuracy"), left=precision.rename('Precision')
    ).merge(recall.rename("Recall"), right_index=True, left_index=True)
    df.index.rename("Class", inplace=True)
    df["K"] = k
    return df


def find_index_of_first_change(lst):
    lst = [x.split("$")[:2] for x in lst]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            return i
    return -1  # Return -1 if no change is found


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score


    # Source https://github.com/pan-webis-de/pan-code/blob/master/clef22/authorship-verification/pan22_verif_evaluator.py
    """

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)
