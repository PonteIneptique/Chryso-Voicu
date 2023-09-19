from typing import List, Optional, Literal, Union, Tuple

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

from freestyl.supervised.siamese.model import BaseSiameseModule


class SiameseFeatureModule(BaseSiameseModule):
    def __init__(
            self,
            features: List[str], label_encoder: LabelEncoder,
            dimension: Union[int, str, Tuple[Union[int, str]]] = 128,
            margin: float = 1.0,
            learning_rate: float = 1e-4,
            loss: Optional[str] = "contrastive",
            dropout=.3,
            optim: Literal["Adam", "SGD"] = "Adam",
            split_dim: Optional[int] = int,
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
        :param split_dim: Applies a similar approach

        """
        super(SiameseFeatureModule, self).__init__(
            label_encoder=label_encoder,
            margin=margin,
            learning_rate=learning_rate,
            loss=loss,
            out_size=self._decide_out_dim(dimension, len(tuple(features))),
            optim=optim,
            miner_for_dev=miner_for_dev,
            pos_strategy=pos_strategy,
            neg_strategy=neg_strategy,
            batch_size_linear=batch_size_linear
        )

        self.hparams["features"] = tuple(features)
        self.hparams["dimension"] = dimension
        self.hparams["dropout"] = dropout
        self.split_dim = self.hparams["split_dim"] = split_dim

        if dimension == 0:
            self.encoder = lambda x: x
        elif split_dim:
            if not isinstance(dimension, tuple) or len(dimension) != 3:
                raise ValueError("Dimension needs to be a tuple in case you're using a split feature vector of size "
                                 "three")
            self.left_encoder, left_out = self.build_linear(split_dim, dimension[0])
            self.right_encoder, right_out = self.build_linear(len(self.hparams["features"])-split_dim, dimension[1])
            self.concat_encoder, _ = self.build_linear(left_out+right_out, dimension[2])
        else:
            self.encoder, _ = self.build_linear(len(self.hparams["features"]), dimension)

        self.save_hyperparameters()

    def _decide_out_dim(self, dimension, feature_len: int) -> int:
        if dimension == 0:
            return feature_len
        if isinstance(dimension, str):
            return int(dimension.split()[-1])
        elif isinstance(dimension, tuple):
            return self._decide_out_dim(dimension[-1], feature_len)
        return dimension

    def build_linear(self, input_dim, dimension) -> Tuple[nn.Module, int]:
        norm = [
            nn.BatchNorm1d(input_dim)
        ]
        if isinstance(dimension, str):
            seq = [*norm, nn.Dropout(self.hparams["dropout"])]
            sequence = [input_dim] + [int(x) for x in dimension.split()]
            for inp, out in zip(sequence, sequence[1:]):
                seq.append(nn.Linear(inp, out))
            encoder = nn.Sequential(*norm, *seq)
        else:
            encoder = nn.Sequential(
                *norm,
                nn.Linear(input_dim, dimension)
            )
            out = dimension
        return encoder, out

    def forward(self, batch):
        if self.hparams["split_dim"]:
            left, right = batch[:, :self.split_dim], batch[:, self.split_dim:]
            batch = torch.cat([self.left_encoder(left), self.right_encoder(right)], dim=1)
            return self.concat_encoder(batch)
        else:
            return self.encoder(batch)
