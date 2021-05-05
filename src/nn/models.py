from typing import Optional, List
from contextlib import contextmanager

from torch import nn
import torch
from torch import Tensor
from torchcrf import CRF
import numpy as np

from src.nn.layers import LstmEncoderPacked, LstmDecoder, LstmDecoderPacked, SpatialDropout, \
    CnnEncoder, TransformerEncoder


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


class BaselineTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 padding_index: int):
        super(BaselineTagger, self).__init__()
        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.embedding = nn.Embedding(num_embeddings=char_vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.fc_1 = nn.Linear(in_features=hidden_size,
                              out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction='none')

    def compute_outputs(self, sequences, true_lengths):
        sequences = self.embedding(sequences)
        sequences = self.fc_1(sequences)
        sequences = torch.relu(sequences)
        sequences = self.fc_2(sequences)
        sequences = torch.relu(sequences)
        return sequences

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


class BaselineCrfTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 padding_index: int):
        super(BaselineCrfTagger, self).__init__()
        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.embedding = nn.Embedding(num_embeddings=char_vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.fc_1 = nn.Linear(in_features=hidden_size,
                              out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.crf = CRF(self.tag_vocab_size, batch_first=True)

    def compute_outputs(self, sequences, true_lengths):
        pad_mask = (sequences == self.padding_index).float()

        sequences = self.embedding(sequences)
        sequences = self.fc_1(sequences)
        sequences = torch.relu(sequences)
        sequences = self.fc_2(sequences)
        sequences = torch.relu(sequences)

        sequences[:, :, self.padding_index] += pad_mask * 10000

        return sequences

    def forward(self, sequences, labels, true_lengths):
        scores = self.compute_outputs(sequences, true_lengths)

        loss = -self.crf(scores, labels, reduction="mean")

        return loss

    def predict(self, sequences: torch.tensor, true_lengths: Optional[torch.tensor] = None):
        if true_lengths is None:
            print("True lengths of sequences not specified!")
            true_lengths = [sequences.size(1)] * sequences.size(0)
            true_lengths = torch.tensor(true_lengths).long()

        scores = self.compute_outputs(sequences, true_lengths)
        predicted = np.array(self.crf.decode(scores))

        return predicted


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

        self.fc_1 = nn.Linear(in_features=hidden_size * self.directions,
                              out_features=hidden_size)

        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.layer_norm = nn.LayerNorm(hidden_size * self.directions)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.crf = CRF(self.tag_vocab_size, batch_first=True)

    def compute_outputs(self, sequences, true_lengths):
        encoder_seq, memory = self.encoder(sequences, true_lengths)
        encoder_seq = self.spatial_dropout(encoder_seq)
        encoder_seq = self.layer_norm(encoder_seq)
        encoder_out = self.fc_1(encoder_seq)
        encoder_out = torch.relu(encoder_out)
        encoder_out = self.fc_2(encoder_out)
        encoder_out = torch.relu(encoder_out)

        pad_mask = (sequences == self.padding_index).float()

        encoder_out[:, :, self.padding_index] += pad_mask * 10000

        return encoder_out

    def forward(self, sequences, labels, true_lengths):
        scores = self.compute_outputs(sequences, true_lengths)

        loss = -self.crf(scores, labels, reduction="mean")

        return loss

    def predict(self, sequences: torch.tensor, true_lengths: Optional[torch.tensor] = None):
        if true_lengths is None:
            print("True lengths of sequences not specified!")
            true_lengths = [sequences.size(1)] * sequences.size(0)
            true_lengths = torch.tensor(true_lengths).long()

        scores = self.compute_outputs(sequences, true_lengths)
        predicted = np.array(self.crf.decode(scores))

        return predicted


class CnnTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 num_filters: int,
                 emb_dim: int,
                 cnn_out_dim: int,
                 hidden_size: int,
                 convolution_layers: int,
                 kernel_sizes: int,
                 use_one_hot: bool,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0):
        super(CnnTagger, self).__init__()
        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.encoder = CnnEncoder(vocab_size=char_vocab_size,
                                  emb_dim=emb_dim,
                                  num_filters=num_filters,
                                  out_dim=cnn_out_dim,
                                  convolution_layers=convolution_layers,
                                  kernel_sizes=kernel_sizes,
                                  padding_index=padding_index,
                                  use_one_hot=use_one_hot)

        self.fc_1 = nn.Linear(in_features=cnn_out_dim,
                              out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.layer_norm = nn.LayerNorm(cnn_out_dim)
        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction='none')

    def compute_outputs(self, sequences, true_lengths):
        sequences = self.encoder(sequences)
        sequences = self.spatial_dropout(sequences)
        sequences = self.layer_norm(sequences)
        encoder_out = self.fc_1(sequences)
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
        scores = self.compute_outputs(sequences, true_lengths)

        predicted = scores.argmax(dim=2)

        return predicted.cpu().numpy()


class CnnCrfTagger(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int,
                 num_filters: int,
                 emb_dim: int,
                 cnn_out_dim: int,
                 hidden_size: int,
                 convolution_layers: int,
                 kernel_sizes: int,
                 use_one_hot: bool,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0):
        super(CnnCrfTagger, self).__init__()
        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.encoder = CnnEncoder(vocab_size=char_vocab_size,
                                  emb_dim=emb_dim,
                                  num_filters=num_filters,
                                  out_dim=cnn_out_dim,
                                  convolution_layers=convolution_layers,
                                  kernel_sizes=kernel_sizes,
                                  padding_index=padding_index,
                                  use_one_hot=use_one_hot)

        self.fc_1 = nn.Linear(in_features=cnn_out_dim,
                              out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size,
                              out_features=tag_vocab_size)

        self.layer_norm = nn.LayerNorm(cnn_out_dim)
        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.crf = CRF(self.tag_vocab_size, batch_first=True)

    def compute_outputs(self, sequences, true_lengths):

        pad_mask = (sequences == self.padding_index).float()

        sequences = self.encoder(sequences)
        sequences = self.spatial_dropout(sequences)
        sequences = self.layer_norm(sequences)
        encoder_out = self.fc_1(sequences)
        encoder_out = torch.relu(encoder_out)
        encoder_out = self.fc_2(encoder_out)
        encoder_out = torch.relu(encoder_out)

        encoder_out[:, :, self.padding_index] += pad_mask * 10000

        return encoder_out

    def forward(self, sequences, labels, true_lengths):
        scores = self.compute_outputs(sequences, true_lengths)

        loss = -self.crf(scores, labels, reduction="mean")

        return loss

    def predict(self, sequences: torch.tensor, true_lengths: Optional[torch.tensor] = None):
        scores = self.compute_outputs(sequences, true_lengths)

        predicted = scores.argmax(dim=2)

        return predicted.cpu().numpy()


class TransformerTagger(nn.Module):

    def __init__(self,
                 char_vocab_size,
                 tag_vocab_size,
                 emb_dim,
                 n_heads,
                 hidden_size,
                 dropout,
                 padding_index,
                 max_len,
                 num_layers):
        super(TransformerTagger, self).__init__()

        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.transformer_encoder = TransformerEncoder(vocab_size=char_vocab_size,
                                                      emb_dim=emb_dim,
                                                      n_heads=n_heads,
                                                      hidden_size=hidden_size,
                                                      dropout=dropout,
                                                      padding_index=self.padding_index,
                                                      max_len=max_len,
                                                      num_layers=num_layers)

        self.fc = nn.Linear(in_features=emb_dim,
                            out_features=tag_vocab_size)

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_index,
            reduction='none')

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.transformer_encoder.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_outputs(self, sequences, true_lengths):
        mask = sequences.ne(self.padding_index)
        sequences = self.transformer_encoder(sequences, mask)
        sequences = self.fc(sequences)

        return sequences

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


class TransformerCrfTagger(nn.Module):

    def __init__(self,
                 char_vocab_size,
                 tag_vocab_size,
                 emb_dim,
                 n_heads,
                 hidden_size,
                 dropout,
                 padding_index,
                 max_len,
                 num_layers):
        super(TransformerCrfTagger, self).__init__()

        self.tag_vocab_size = tag_vocab_size
        self.padding_index = padding_index

        self.transformer_encoder = TransformerEncoder(vocab_size=char_vocab_size,
                                                      emb_dim=emb_dim,
                                                      n_heads=n_heads,
                                                      hidden_size=hidden_size,
                                                      dropout=dropout,
                                                      padding_index=self.padding_index,
                                                      max_len=max_len,
                                                      num_layers=num_layers)

        self.fc = nn.Linear(in_features=emb_dim,
                            out_features=tag_vocab_size)

        self.crf = CRF(self.tag_vocab_size, batch_first=True)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.transformer_encoder.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_outputs(self, sequences, true_lengths):
        mask = sequences.ne(self.padding_index)
        sequences = self.transformer_encoder(sequences, mask)
        sequences = self.fc(sequences)
        return sequences

    def forward(self, sequences, labels, true_lengths):
        scores = self.compute_outputs(sequences, true_lengths)

        loss = -self.crf(scores, labels, reduction="mean")

        return loss

    def predict(self, sequences: torch.tensor, true_lengths: Optional[torch.tensor] = None):
        if true_lengths is None:
            print("True lengths of sequences not specified!")
            true_lengths = [sequences.size(1)] * sequences.size(0)
            true_lengths = torch.tensor(true_lengths).long()

        scores = self.compute_outputs(sequences, true_lengths)
        predicted = np.array(self.crf.decode(scores))

        return predicted
