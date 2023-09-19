"""
@author (For HAN related stuff): Viet Nguyen <nhviet1009@gmail.com>
"""
from typing import Optional, List, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.preprocessing import LabelEncoder

from freestyl.supervised.siamese.model import BaseSiameseModule
from freestyl.supervised.siamese.sequential.data import DocumentEncoder
from freestyl.supervised.siamese.sequential.utils import weight_init


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A batch first Pytorch tensor.
    sequence_lengths : `torch.LongTensor`, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    # Returns

    sorted_tensor : `torch.FloatTensor`
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : `torch.LongTensor`
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : `torch.LongTensor`
        Indices into the sorted_tensor such that
        `sorted_tensor.index_select(0, restoration_indices) == original_tensor`
    permutation_index : `torch.LongTensor`
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.

    License: AllenNLP
    """

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def run_rnn_on_unsorted(module: nn.GRU, batch: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    batch, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(batch, sizes.to(batch.device))
    batch = pack_padded_sequence(batch, lengths=sorted_sequence_lengths.cpu(), batch_first=True)
    batch, _ = module(batch)  # First is packing data, last is CT
    batch, _ = pad_packed_sequence(batch, batch_first=True, padding_value=0.0)
    return batch[restoration_indices]


def get_mask(sizes: torch.Tensor) -> torch.BoolTensor:
    max_len = int(sizes.max().cpu())
    return (
            torch.arange(max_len, device=sizes.device).expand(len(sizes), max_len) < sizes.unsqueeze(1)
    ).bool().unsqueeze(dim=-1)


class HANDocumentLevelModule(nn.Module):
    def __init__(self, input_size: int = 50, hidden_size=50, dropout: float = .15):
        super(HANDocumentLevelModule, self).__init__()

        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(2*hidden_size, 2*hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.context_weight.data.normal_(mean, std)

    def forward(self, sentences, size):
        # feature output and hidden state output
        sentences = self.dropout(sentences)

        word_output = run_rnn_on_unsorted(self.gru, sentences, size)
        word_attention = torch.tanh(self.dense(word_output))

        weights = torch.matmul(word_attention, self.context_weight)
        weights = F.softmax(weights, dim=1)

        mask = get_mask(size).to(weights.device)

        # weights : (batch_size, sentence_len, 1)
        weights = torch.where(
            mask != 0,
            weights,
            torch.full_like(mask, 0, dtype=torch.float, device=weights.device)
        )

        # weights : (batch_size, sentence_len, 1)
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        output = torch.sum((weights * word_output), dim=1)
        return output


class HANSentenceLevelModule(HANDocumentLevelModule):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size=50, dropout: float = .15):
        super(HANSentenceLevelModule, self).__init__(
            input_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

    def forward(self, sentences, size):
        # feature output and hidden state output
        return HANDocumentLevelModule.forward(self, self.emb(sentences), size)


class HierAttNet(nn.Module):
    # ToDo: This does not converge, there must be an implementation issue ?
    def __init__(
        self,
        # Words
        embedding_size: int,
        vocab_size: int,
        # Sentence
        sentence_hidden_size: int,
        # Documents
        document_hidden_size: int,
        # Dropout
        dropout: float = .15
    ):
        super(HierAttNet, self).__init__()
        self.sentence_hidden_size = sentence_hidden_size
        self.document_hidden_size = document_hidden_size

        self.word_att_net = HANSentenceLevelModule(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=sentence_hidden_size,
            dropout=dropout
        )
        self.sent_att_net = HANDocumentLevelModule(
            input_size=sentence_hidden_size*2,
            hidden_size=document_hidden_size,
            dropout=dropout
        )
        self.dense = nn.Linear(2*document_hidden_size, 2*document_hidden_size)

    def forward(self, batch):
        """ (Sentence, WordId)

        """
        _, _, document_sizes = zip(*batch)

        sentences = []
        for batched_sentences, sizes, _ in batch:
            sentences.append(
                self.word_att_net(batched_sentences, sizes)
            )

        document_batch = pad_sequence(sentences, batch_first=True)
        output = self.sent_att_net(document_batch, torch.tensor(document_sizes))
        return self.dense(output)


class TextCNN(nn.Module):
    def __init__(
        self,
        # Words
        embedding_size: int,
        vocab_size: int,
        pad_idx: int = 1,
        kernel_num: int = 100,
        kernel_sizes: List[int] = None,
        out_size: int = None,
        # Dropout
        dropout: float = .15
    ):
        super(TextCNN, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (ks, embedding_size)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None
        if out_size:
            self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, out_size)

    def forward(self, x):
        docs, docsizes = zip(*x)
        x = pad_sequence(docs, batch_first=True, padding_value=self.pad_idx)
        x = self.embed(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_dim)
        # #  input size (N,Cin,H,W)  output size (N,Cout,Hout,1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes)*kernel_num)
        x = self.dropout(x)
        if self.fc1 is not None:
            x = self.fc1(x)
        return x


class TextGRU(nn.Module):
    def __init__(
        self,
        # Words
        embedding_size: int,
        vocab_size: int,
        pad_idx: int = 1,
        document_hidden_size: int = 256,
        # Dropout
        dropout: float = .15
    ):
        super(TextGRU, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=self.pad_idx)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size=document_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.hidden_size = document_hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None

        weight_init(self)

    def forward(self, x):
        docs, docsizes = zip(*x)
        batch = pad_sequence(docs, batch_first=True, padding_value=self.pad_idx)
        docsizes = torch.tensor(docsizes, device=batch.device)
        # sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index
        sorted_tensor, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(batch, docsizes)
        # Pass through embedding
        sorted_tensor = self.embed(sorted_tensor)
        sorted_tensor = self.dropout(sorted_tensor)
        sorted_tensor = pack_padded_sequence(
            sorted_tensor,
            lengths=sorted_sequence_lengths.cpu(),
            batch_first=True,
        )
        _, final = self.gru(sorted_tensor)  # First is packing data, last is CT
        final = final.view(batch.size(0), self.hidden_size*2)
        #batch = pad_sequence(unpack_sequence(batch), batch_first=True)
        return final[restoration_indices]


class SiameseSequentialModule(BaseSiameseModule):
    DEFAULT = "AttentionalGRU"

    def __init__(
        self,
        label_encoder: LabelEncoder,
        document_encoder: DocumentEncoder,
        sequential_model: Literal["AttentionalGRU", "TextCNN"] = "AttentionalGRU",
        # Words
        embedding_size: int = 50,
        # Sentence
        sentence_hidden_size: int = 64,
        # Document
        document_hidden_size: int = 256,
        margin: float = 1.0,
        learning_rate: float = 1e-4,
        # CNN
        kernel_num: int = 100,
        kernel_sizes: List[int] = None,
        loss: Optional[str] = "contrastive",
        dropout=.15,
        optim: Literal["Adam", "SGD"] = "Adam"
    ):
        """ Model for dimension

        :param features: All properties such as n-grams, pos-grams, etc.
        :param label_encoder: Target classes label encoder
        :param dimension: Linear encoder output dimension
        :param margin: Margin use for loss, lr is the lear
        :param learning_rate: Learning rate
        :param loss: Loss used


        """
        super(SiameseSequentialModule, self).__init__(
            label_encoder=label_encoder,
            margin=margin,
            learning_rate=learning_rate,
            loss=loss,
            out_size=document_hidden_size*2,
            optim=optim
        )

        kernel_sizes = kernel_sizes or [2, 3, 4, 5]

        self.hparams["embedding_size"] = embedding_size
        self.hparams["sentence_hidden_size"] = sentence_hidden_size
        self.hparams["document_hidden_size"] = document_hidden_size
        self.hparams["kernel_sizes"] = kernel_sizes
        self.hparams["kernel_num"] = kernel_num
        self.hparams["dropout"] = dropout
        self.hparams["sequential_model"] = sequential_model

        self.document_encoder = document_encoder

        if sequential_model == "TextCNN":
            self.net = TextCNN(
                pad_idx=self.document_encoder.pad_idx,
                embedding_size=embedding_size,
                vocab_size=len(document_encoder),
                kernel_sizes=kernel_sizes,
                kernel_num=kernel_num,
                out_size=document_hidden_size*2,
                dropout=dropout
            )
        elif sequential_model == "TextGRU":
            self.net = TextGRU(
                pad_idx=self.document_encoder.pad_idx,
                embedding_size=embedding_size,
                vocab_size=len(document_encoder),
                document_hidden_size=document_hidden_size,
                dropout=dropout
            )
        else:
            self.net = HierAttNet(
                embedding_size=embedding_size,
                vocab_size=len(document_encoder),
                sentence_hidden_size=sentence_hidden_size,
                document_hidden_size=document_hidden_size,
                dropout=dropout
            )

        self.save_hyperparameters()

    def forward(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]):
        n = self.net(batch)
        return n


if __name__ == "__main__":
    abc = HANDocumentLevelModule()
