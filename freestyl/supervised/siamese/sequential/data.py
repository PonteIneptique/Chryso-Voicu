import logging
from collections import Counter
from itertools import count
from typing import Iterable, Optional, Tuple, List, Literal

import regex
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchtext.vocab import vocab, Vocab
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from pandas import Series
from freestyl.dataset.dataframe_wrapper import DataframeWrapper


def filter_empty(sequences: List) -> List:
    return [
        el for el in sequences if el
    ]


def _length(sequence, size: int):
    return [sequence[i:i+size] for i in range(len(sequence) // size + int((len(sequence) % size) > 0))]


def punct_document_splitter(string: str):
    """ Split a document into a list of sentences in which we split tokens.

    >>> punct_document_splitter(".Hello... I am James Bond; Junior.")
    [['Hello'], ['I', 'am', 'James', 'Bond'], ['Junior']]
    """
    return filter_empty([
        [tok.strip() for tok in substring.strip().split() if tok.strip()]
        for substring in regex.split(r"[\.;·?!]+", string.replace(",", ""))
        if len(substring)
    ])


def length_document_splitter(string: str, size=25):
    """ Split a document into a list of sentences in which we split tokens.

    >>> punct_document_splitter(".Hello... I am James Bond; Junior.")
    [['Hello'], ['I', 'am', 'James', 'Bond'], ['Junior']]
    """
    return _length(filter_empty([
        tok.strip()
        for substring in regex.split(r"[\.;·?!]+", string.replace(",", ""))
        for tok in substring.strip().split()
    ]), size=size)


class DocumentEncoder(Module):
    """ Encodes the input

    """
    def __init__(
            self,
            voc: Optional[Vocab] = None,
            splitter: Literal["sentence", "size"] = "size",
            bilevel: bool = True
    ):
        super(DocumentEncoder, self).__init__()
        self._vocab: Vocab = voc or vocab({}, specials=["<UNK>", "<PAD>"])
        self._vocab.set_default_index(self._vocab["<UNK>"])
        self.bilevel: bool = bilevel
        if splitter == "size":
            self.splitter = length_document_splitter
        else:
            self.splitter = punct_document_splitter

    def __len__(self):
        return len(self._vocab)

    @property
    def pad_idx(self):
        return self._vocab["<PAD>"]

    def forward(self, document: str) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if self.bilevel:
            sentences, sizes = zip(*[
                (torch.tensor([self._vocab[tok] for tok in sentence]), len(sentence))
                for sentence in self.splitter(document)
            ])
            sentences = pad_sequence(sentences, batch_first=True, padding_value=self.pad_idx)
            return sentences, torch.tensor(sizes)
        else:
            sentences = torch.tensor([
                self._vocab[tok]
                for sentence in self.splitter(document)
                for tok in sentence
            ])
            return sentences, sentences.shape[0]

    def from_iterator(self, iterable: Iterable[str], min_frequency: int = 5):
        count = Counter()
        for document in iterable:
            for sentence in self.splitter(document):
                count += Counter(sentence)
        for token, freq in count.most_common(len(count)):
            if freq >= min_frequency:
                self._vocab.append_token(token)
        logging.info(f"Learnt {len(self._vocab)-2} type of tokens")

    def from_series(self, series: Series, min_frequency: int = 5):
        self.from_iterator(series.to_list(), min_frequency=min_frequency)


class DataFrameSequenceDataset(Dataset):
    def __init__(self,
                 dfw: DataframeWrapper,
                 document_encoder: DocumentEncoder,
                 label_encoder: LabelEncoder,
                 text_key: str = "modified_text",
                 sublevel: bool = True
                 ):
        self.df: DataframeWrapper = dfw
        self.key = text_key
        self.document_encoder: DocumentEncoder = document_encoder
        self.class_encoder: LabelEncoder = label_encoder

        try:
            self.ys = self.class_encoder.transform(self.df.ys.tolist())
        except ValueError:
            self.ys = self.class_encoder.transform([
                y if y in self.class_encoder.classes_ else "[OOD]" for y in self.df.ys.tolist()
            ])

    def __len__(self):
        return len(self.df.dataframe[self.key])

    def __getitem__(self, idx) -> Tuple[
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor],
        torch.LongTensor
    ]:
        """ Returns (Padded Sentences, Size of sentences, Number of Sentences), Class IDX

        """
        text, size = self.document_encoder(self.df.dataframe[self.key].iloc[idx])
        if self.document_encoder.bilevel:
            return (text, size, torch.tensor(text.shape[0])), torch.tensor(self.ys[idx])
        else:
            return (text, size), torch.tensor(self.ys[idx])


class DocumentBatcher(object):
    def __init__(self, pad_value: int = 1, bilevel: bool = True):
        self.pad_value = pad_value

    def __call__(self, documents: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Documents are List[(Padded Sentences, Size of sentences), Number of Sentences, Class IDX]

        Matrix of sentences,
        """

        Xs, Ys = tuple(zip(*documents))

        return Xs, torch.tensor(Ys)


def make_dataloader(
        dataframe_wrapper: DataframeWrapper,
        text_key: Optional[str] = "modified_text",
        model: Optional[Module] = None,
        document_encoder: Optional[DocumentEncoder] = None,
        label_encoder: Optional[LabelEncoder] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        sample: bool = False,
) -> DataLoader:
    """ Produces a dataloader for a DataFrameWrapper and a given model
    """
    label_encoder = model.hparams.label_encoder if model is not None else label_encoder
    if not label_encoder:
        raise ValueError("A LabelEncoder or Model instance should be passed to this function")

    document_encoder = model.document_encoder if model is not None else document_encoder
    if not document_encoder:
        raise ValueError("A LabelEncoder or Model instance should be passed to this function")

    sampler = None  # ToDo: Revise sampling strategies
    batcher = DocumentBatcher(document_encoder.pad_idx, bilevel=document_encoder.bilevel)

    return DataLoader(
        DataFrameSequenceDataset(
            dataframe_wrapper,
            document_encoder=document_encoder,
            label_encoder=label_encoder,
            text_key=text_key
        ),
        batch_size=len(dataframe_wrapper.ys) if batch_size is None else batch_size,
        collate_fn=batcher,
        shuffle=shuffle if shuffle and not sample else False,
        sampler=sampler if sample else None
    )


if __name__ == "__main__":
    from pandas import read_csv

    docs = read_csv("~/dev/Chryso-Voicu/tlg-test.csv")
    enc = DocumentEncoder(bilevel=False)
    enc.from_iterator(docs["modified_text"].to_list(), 2)
    dfw = DataframeWrapper(docs, target="author", label=["author", "title"])
    labenc = LabelEncoder()
    labenc.fit_transform(dfw.ys.tolist())
    dataset = DataFrameSequenceDataset(
        dfw,
        document_encoder=enc,
        label_encoder=labenc,
        text_key="modified_text"
    )

    dl = make_dataloader(
        dfw,
        document_encoder=enc,
        label_encoder=labenc,
        text_key="modified_text",
        batch_size=4
    )
    print(next(iter(dl)))
