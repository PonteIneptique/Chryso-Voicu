from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule


class DataFrameDataset(Dataset):
    def __init__(self, df: DataframeWrapper, encoder: LabelEncoder):
        self.df: DataframeWrapper = df
        self.encoder: LabelEncoder = encoder

        try:
            self.ys = self.encoder.transform(self.df.ys.tolist())
        except ValueError:
            self.ys = self.encoder.transform([
                y if y in self.encoder.classes_ else "[OOD]" for y in self.df.ys.tolist()
            ])

    def __len__(self):
        return len(self.df.normalized.xs)

    def __getitem__(self, idx):
        return self.df.normalized.xs.iloc[idx].fillna(0).to_numpy().astype(np.float32), self.ys[idx]


def _batchify(datas):
    """ Collation function for DataFrameDataset's Dataloaders"""
    xs, ys = zip(*datas)
    return torch.tensor(np.array(xs)), torch.tensor(ys)


def make_dataloader(
        dataframe_wrapper: DataframeWrapper,
        model: Optional[SiameseFeatureModule] = None,
        label_encoder: Optional[LabelEncoder] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        sample: Optional[int] = False,
) -> DataLoader:
    """ Produces a dataloader for a DataFrameWrapper and a given model
    """
    label_encoder = model.hparams.label_encoder if model is not None else label_encoder
    if not label_encoder:
        raise ValueError("A LabelEncoder or Model instance should be passed to this function")

    dfd = DataFrameDataset(
        dataframe_wrapper,
        encoder=label_encoder
    )

    sampler = None
    if sample:
        sampler = MPerClassSampler(labels=dfd.ys, m=sample, length_before_new_iter=len(dfd.ys))

    return DataLoader(
        dfd,
        batch_size=len(dataframe_wrapper.ys) if batch_size is None else batch_size,
        collate_fn=_batchify,
        shuffle=shuffle if shuffle and not sample else False,
        sampler=sampler if sample else None
    )
