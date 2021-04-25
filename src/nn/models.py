from typing import Union, Optional, List
from contextlib import contextmanager

from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np

from src.nn.layers import LstmEncoder, LstmEncoderPacked, LstmDecoder, LstmDecoderPacked, get_pad_mask, SpatialDropout, \
    CnnEncoder


class LstmTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super(LstmTagger, self).__init__()

        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.directions = 2 if bidirectional else 1

        self.encoder = LstmEncoderPacked(vocab_size=char_vocab_size,
                                         emb_dim=emb_dim,
                                         hidden_size=hidden_size,
                                         lstm_layers=lstm_layers,
                                         layer_dropout=layer_dropout,
                                         spatial_dropout=spatial_dropout,
                                         bidirectional=bidirectional,
                                         padding_index=padding_index)

        self.fc_1 = nn.Linear(in_features=hidden_size * self.directions,
                              out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.layer_norm = nn.LayerNorm(hidden_size * self.directions)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction='none')

    def compute_outputs(self, sequences, true_lengths):
        encoder_seq, memory = self.encoder(sequences, true_lengths)
        encoder_seq = self.spatial_dropout(encoder_seq)
        encoder_seq = self.layer_norm(encoder_seq)
        encoder_out = self.fc_1(encoder_seq)
        encoder_out = torch.relu(encoder_out)
        encoder_out = self.fc_2(encoder_out)
        encoder_out = torch.relu(encoder_out)

        return encoder_out

    def forward(self, sequences, labels, true_lengths):
        scores = self.compute_outputs(sequences, true_lengths)

        scores = scores.view(-1, self.tag_vocab_size)
        labels = labels.view(-1)

        pad_mask = (sequences != self.padding_index).float()

        loss = self.loss(scores, labels)
        loss = loss.view(sequences.size(0), sequences.size(1))
        loss *= pad_mask
        loss = torch.sum(loss, axis=1)
        loss /= true_lengths
        loss = torch.mean(loss)

        return loss

    def predict(self, sequences: torch.tensor, true_lengths: Optional[torch.tensor] = None):
        if true_lengths is None:
            print("True lengths of sequences not specified!")
            true_lengths = [sequences.size(1)] * sequences.size(0)
            true_lengths = torch.tensor(true_lengths).long()

        scores = self.compute_outputs(sequences, true_lengths)

        predicted = scores.argmax(dim=2)

        return predicted.cpu().numpy()


class LstmCrfTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super(LstmCrfTagger, self).__init__()

        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.directions = 2 if bidirectional else 1

        self.encoder = LstmEncoderPacked(vocab_size=char_vocab_size,
                                         emb_dim=emb_dim,
                                         hidden_size=hidden_size,
                                         lstm_layers=lstm_layers,
                                         layer_dropout=layer_dropout,
                                         spatial_dropout=spatial_dropout,
                                         bidirectional=bidirectional,
                                         padding_index=padding_index)

        self.fc = nn.Linear(in_features=hidden_size * self.directions,
                            out_features=tag_vocab_size)

        self.crf = CRF(self.tag_vocab_size, batch_first=True)

    def compute_outputs(self, sequences):
        encoder_seq, memory = self.encoder(sequences)
        encoder_out = self.fc(encoder_seq)

        pad_mask = (sequences == self.padding_index).float()

        encoder_out[:, :, self.padding_index] += pad_mask * 10000

        return encoder_out

    def forward(self, sequences, labels):
        scores = self.compute_outputs(sequences)

        return -self.crf(scores, labels, reduction="sum")

    def predict(self, sentences):
        scores = self.compute_outputs(sentences)
        predicted = np.array(self.crf.decode(scores))

        return predicted


class CnnTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 emb_dim: int,
                 num_filters: int,
                 kernel_size: int,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0):
        super(CnnTagger, self).__init__()
        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.encoder = CnnEncoder(vocab_size=char_vocab_size,
                                  emb_dim=emb_dim,
                                  dropout=0.3,
                                  num_filters=num_filters,
                                  kernel_size=kernel_size,
                                  out_dim=100,
                                  padding_index=padding_index)

        self.fc = nn.Linear(in_features=num_filters,
                            out_features=tag_vocab_size)

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction='sum')

    def compute_outputs(self, sequences):
        encoder_seq = self.encoder(sequences)
        encoder_out = self.fc(encoder_seq)

        pad_mask = (sequences == self.padding_index).float()

        encoder_out[:, :, self.padding_index] += pad_mask * 10000

        return encoder_out

    def forward(self, sequences, labels):
        scores = self.compute_outputs(sequences)

        scores = scores.view(-1, self.tag_vocab_size)
        labels = labels.view(-1)

        return self.loss(scores, labels)

    def predict(self, sentences):
        scores = self.compute_outputs(sentences)

        predicted = scores.argmax(dim=2)

        return predicted.cpu().numpy()


class RandomTagger(nn.Module):

    def __init__(self,
                 labels: List[int],
                 seed: Optional[int] = None):

        self.labels = labels
        self.seed = seed

    def compute_outputs(self, sequences: Tensor):
        pass

    def forward(self, sequences: Tensor, labels: Tensor):
        pass

    def predict(self, sequences: Tensor) -> np.ndarray:
        size = tuple(sequences.size())
        with self.temp_seed(self.seed):
            predicted = np.random.choice(a=self.labels, size=size)

        return predicted

    @staticmethod
    @contextmanager
    def temp_seed(seed):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
