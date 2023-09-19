from typing import List, Optional, Literal, Tuple, Union

from sklearn.preprocessing import LabelEncoder

import pytorch_lightning as pl
import torch
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import LpDistance, SNRDistance
from torchmetrics import AUROC


class BaseSiameseModule(pl.LightningModule):
    def __init__(
            self,
            label_encoder: LabelEncoder,
            margin: float = 1.0,
            learning_rate: float = 1e-4,
            loss: Optional[str] = "contrastive",
            out_size: Optional[int] = None,
            optim: Literal["Adam", "SGD"] = "Adam",
            miner_for_dev: bool = False,
            pos_strategy: Literal["easy", "semihard", "hard"] = "easy",
            neg_strategy: Literal["easy", "semihard", "hard"] = "semihard",
            batch_size_linear: int = 16
    ):
        """ Model for dimension

        :param features: All properties such as n-grams, pos-grams, etc.
        :param label_encoder: Target classes label encoder
        :param dimension: Linear encoder output dimension
        :param margin: Margin use for loss, lr is the lear
        :param learning_rate: Learning rate
        :param loss: Loss used


        """
        super(BaseSiameseModule, self).__init__()
        self.hparams["margin"]: float = margin
        self.hparams["optim"]: float = optim
        self.hparams["learning_rate"]: float = learning_rate
        self.hparams["label_encoder"]: LabelEncoder = label_encoder
        self.hparams["miner_for_dev"]: bool = miner_for_dev
        self.hparams["pos_strategy"]: bool = pos_strategy
        self.hparams["neg_strategy"]: str = neg_strategy
        self.hparams["batch_size_linear"]: int = batch_size_linear

        # self.miner = miners.BatchEasyHardMiner()

        # Remember: AUC ROC should be maximized to ONE
        self.aucroc: AUROC = AUROC(num_classes=None)
        self.linear: Optional[torch.nn.Linear] = None

        if loss.lower().endswith("manhattan"):
            self.distance = LpDistance(power=1)
        elif "contrastive" in loss.lower():
            self.distance = SNRDistance()
        else:
            self.distance = LpDistance(power=2)

        if loss.startswith("contrastive"):
            self.loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=margin, distance=self.distance)
            self.loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=margin, distance=self.distance)
        elif loss == "stn_contrastive":
            self.loss_func = losses.SignalToNoiseRatioContrastiveLoss(neg_margin=margin)
            self.distance = SNRDistance()
        elif loss.startswith("triplet"):
            self.loss_func = losses.TripletMarginLoss(margin=margin, distance=self.distance)
        elif loss == "batch":
            self.loss_func = losses.CrossBatchMemory(
                losses.SignalToNoiseRatioContrastiveLoss(pos_margin=0, neg_margin=margin),
                embedding_size=out_size,
                memory_size=512
            )
            self.distance = SNRDistance()
        elif loss.startswith("linear"):
            self.linear = torch.nn.Linear(out_size, 1)
            self._subloss = torch.nn.BCEWithLogitsLoss()
            self._linear_miner = miners.BatchEasyHardMiner("all", "all")
            self.loss_func = self._linear_loss
        else:
            raise ValueError()

        self.miner = miners.BatchEasyHardMiner(
            distance=self.distance,
            pos_strategy=pos_strategy,
            neg_strategy=neg_strategy
        )
        self.predict_mode = "vector"
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.hparams["optim"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])
        return optimizer

    def _linear_loss(self, Xs, Ys, pairs: Optional[torch.Tensor] = None) -> torch.Tensor:
        distances, is_pair, _ = self.get_linear_probabilities(Xs, Ys, pairs)
        return self._subloss(distances.squeeze(), is_pair)

    def _linear_auroc(self, Xs: torch.Tensor, Ys: torch.Tensor, pairs: Optional[Tuple] = None):
        distances, is_pair, _ = self.get_linear_probabilities(Xs, Ys, pairs)
        distances = distances.squeeze()
        return self.aucroc(distances, is_pair)

    def linear_compute_metrics(
            self,
            Vs: torch.Tensor,
            Ys: torch.Tensor,
            pairs: Optional[Tuple] = None,
            get_loss: bool = True,
            get_auroc: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        logits, is_pair, pair_ids = self.get_linear_probabilities(Vs, Ys, pairs=pairs)
        out: List[torch.Tensor] = []
        if get_loss:
            out.append(self._subloss(logits, is_pair.float()))
        if get_auroc:
            out.append(self.aucroc(torch.sigmoid(logits), is_pair))
        return tuple(out)

    def get_linear_probabilities(self, Xs: torch.Tensor, Ys: torch.Tensor, pairs: Optional[Tuple] = None):
        if not pairs:
            (p_anchors, positives, n_anchors, negatives) = self._linear_miner(Xs, Ys)
        else:
            (p_anchors, positives, n_anchors, negatives) = pairs

        pos = torch.abs(Xs[p_anchors, :] - Xs[positives, :])
        neg = torch.abs(Xs[n_anchors, :] - Xs[negatives, :])
        distances = torch.cat([pos, neg], dim=0)

        if distances.shape[0] > 3*self.hparams["batch_size_linear"]:
            out: List[torch.Tensor] = []
            for minibatch in torch.split(distances, distances.shape[0]//self.hparams["batch_size_linear"], dim=0):
                out.append(self.linear(minibatch))
            distances = torch.cat(out, dim=0)
        else:
            distances = self.linear(distances)

        is_pair = torch.zeros(distances.shape, device=distances.device, dtype=torch.long)
        is_pair[:pos.size(0)] = 1
        # Pretty sure I am doing something wrong here
        # print(distances.squeeze(0).shape)
        # print(is_pair.shape)
        return (
            distances.squeeze(1),
            is_pair.squeeze(1),
            torch.cat(
                [
                    torch.cat([p_anchors.unsqueeze(1), positives.unsqueeze(1)], dim=1),
                    torch.cat([n_anchors.unsqueeze(1), negatives.unsqueeze(1)], dim=1)
                ],
                dim=0
            )
        )

    def training_step(self, batch, batch_idx):
        Xs, Ys = batch
        Vs = self.forward(Xs)
        pairs = self.miner(Vs, Ys)
        if self.linear is not None:
            loss, = self.linear_compute_metrics(Vs, Ys, pairs, get_loss=True)
            return loss
        else:
            return self.loss_func(Vs, Ys, pairs)

    def validation_step(self, batch, batch_idx):
        Xs, Ys = batch
        Vs = self.forward(Xs)
        pairs = None
        if self.hparams["miner_for_dev"]:
            pairs = self.miner(Vs, Ys)

        if self.linear is not None:
            loss, auroc = self.linear_compute_metrics(Vs, Ys, pairs, get_loss=True, get_auroc=True)
        else:
            loss = self.loss_func(Vs, Ys, pairs)
            auroc = self._comp_auroc_roc(Vs, Ys)

        self.log("dev_loss", loss, prog_bar = True)
        self.log("dev_auroc", auroc, prog_bar=True)
        return Vs, Ys

    def test_step(self, batch, batch_idx):
        Xs, Ys = batch
        if isinstance(Xs, torch.Tensor) and Xs.shape[0] == 1:
            return
        Vs = self.forward(Xs)

        if self.get_prob or self.linear is not None:
            auroc, loss = self.linear_compute_metrics(Vs, Ys, get_loss=True, get_auroc=True)
        else:
            auroc = self._comp_auroc_roc(Vs, Ys)
            loss = self.loss_func(Vs, Ys)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_auroc", auroc, prog_bar=True)

        return Vs, Ys

    @property
    def get_prob(self):
        return self.predict_mode == "linear"

    def _comp_auroc_roc(self, vectors, classes):
        """ There is definitely a bug here, given that it gives me a AUROC of 1 (or my negated is perfect).

        """
        distances, combinations = self.pairwise_probability(vectors)

        truthes = classes[combinations[:, 0]] == classes[combinations[:, 1]]

        # True = 1 but distance are minimized, so 1-distance == Proba of True positive
        return self.aucroc(distances, truthes)

    def predict_step(self, batch, batch_idx):
        Xs, Ys = batch
        Vs = self.forward(Xs)

        if self.get_prob:
            probs, classes, pairs = self.get_linear_probabilities(Vs, Ys)
            matrix = torch.eye(
                Vs.shape[0],
                device=Vs.device
            )
            matrix[pairs[:, 0], pairs[:, 1]] = probs
            return Vs, Ys, torch.sigmoid(matrix)
        return Vs, Ys

    def pairwise_probability(self, Vs):
        ids = torch.arange(0, Vs.shape[0], dtype=torch.long)
        combinations = torch.combinations(ids)

        return (
            1 - self.distance.pairwise_distance(
                Vs[combinations[:, 0], :],
                Vs[combinations[:, 1], :]
            ),
            combinations
        )

    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object
        if self.linear is None:
            raise ValueError("Can't use probability without a linear layer")
        self.predict_mode = "linear"
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.predict_mode = "vector"
