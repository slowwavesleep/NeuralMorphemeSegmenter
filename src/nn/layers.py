import math
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import transformer_encoder
from transformer_encoder.utils import PositionalEncoding


class SpatialDropout(torch.nn.Dropout2d):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class LstmEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super(LstmEncoder, self).__init__()

        if lstm_layers < 2 and layer_dropout != 0:
            layer_dropout = 0

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=layer_dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, encoder_seq):
        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        output, memory = self.lstm(encoder_seq)

        return output, memory


class LstmEncoderPacked(LstmEncoder):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super().__init__(vocab_size,
                         emb_dim,
                         hidden_size,
                         lstm_layers,
                         layer_dropout,
                         spatial_dropout,
                         bidirectional,
                         padding_index)

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.bidirectional = bidirectional

    def forward(self, encoder_seq: torch.tensor, true_lengths: torch.tensor):
        initial_len = encoder_seq.size(-1)

        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        encoder_seq = self.layer_norm(encoder_seq)

        encoder_seq = pack_padded_sequence(input=encoder_seq,
                                           lengths=true_lengths,
                                           batch_first=True,
                                           enforce_sorted=False)

        encoder_seq, memory = self.lstm(encoder_seq)

        encoder_seq = pad_packed_sequence(sequence=encoder_seq,
                                          batch_first=True,
                                          total_length=initial_len)[0]

        if self.bidirectional:
            hidden, cell = memory
            hidden = hidden.permute(1, 0, 2)
            hidden = hidden.reshape(hidden.size(0), 1, -1).permute(1, 0, 2)
            cell = cell.permute(1, 0, 2)
            cell = cell.reshape(cell.size(0), 1, -1).permute(1, 0, 2)
            memory = hidden, cell

        return encoder_seq, memory


class LstmDecoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0,
                 head: bool = True):
        super(LstmDecoder, self).__init__()

        self.head = head

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=vocab_size)

    def forward(self, decoder_seq, memory):
        decoder_seq = self.embedding(decoder_seq)

        decoder_seq = self.spatial_dropout(decoder_seq)

        output, _ = self.lstm(decoder_seq, memory)

        if self.head:
            output = self.fc(output)

        return output


class LstmDecoderPacked(LstmDecoder):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0,
                 head: bool = True):
        super().__init__(vocab_size,
                         emb_dim,
                         hidden_size,
                         lstm_layers,
                         spatial_dropout,
                         padding_index,
                         head)

        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, decoder_seq, memory):
        initial_len = decoder_seq.size(-1)

        decoder_lens = get_non_pad_lens(decoder_seq)

        decoder_seq = self.embedding(decoder_seq)

        decoder_seq = self.spatial_dropout(decoder_seq)

        decoder_seq = self.layer_norm(decoder_seq)

        decoder_seq = pack_padded_sequence(input=decoder_seq,
                                           lengths=decoder_lens,
                                           batch_first=True,
                                           enforce_sorted=False)

        decoder_seq, _ = self.lstm(decoder_seq, memory)

        decoder_seq = pad_packed_sequence(sequence=decoder_seq,
                                          batch_first=True,
                                          total_length=initial_len)[0]

        if self.head:
            decoder_seq = self.fc(decoder_seq)

        return decoder_seq


class CnnEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 out_dim: int,
                 convolution_layers: int,
                 kernel_sizes: Union[List[int], int],
                 padding_index: int,
                 emb_dim: Optional[int],
                 use_one_hot: bool = True,
                 num_filters: int = 128,
                 scale: Optional[float] = None):

        super(CnnEncoder, self).__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        for kernel_size in kernel_sizes:
            assert kernel_size % 2

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.convolution_layers = convolution_layers
        self.kernel_sizes = kernel_sizes
        self.hidden_size = out_dim
        self.padding_index = padding_index
        self.use_one_hot = use_one_hot

        if scale is not None:
            self.scale = scale
        else:
            self.scale = np.sqrt(0.5)

        if self.use_one_hot or not self.emb_dim:
            self.representation_dim = self.vocab_size
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.emb_dim,
                                          padding_idx=self.padding_index)
            self.representation_dim = self.emb_dim

        self.convolutions_stacks = nn.ModuleList([self.build_convolution_stack(in_channels=self.representation_dim,
                                                                               out_channels=self.num_filters,
                                                                               kernel_size=kernel_size,
                                                                               num_layers=self.convolution_layers)
                                                  for kernel_size in self.kernel_sizes])

        self.residual_resize = nn.Linear(in_features=self.representation_dim,
                                         out_features=self.num_filters)

        self.final_residual_resize = nn.Linear(in_features=self.representation_dim,
                                               out_features=self.hidden_size)

        self.final_resize = nn.Linear(in_features=self.convolution_layers * len(self.kernel_sizes) * self.num_filters,
                                      out_features=self.hidden_size)

        self.layer_norm = nn.LayerNorm(self.num_filters)

    def apply_layer_norm(self, sequence):
        sequence = sequence.permute(0, 2, 1)
        sequence = self.layer_norm(sequence)
        sequence = sequence.permute(0, 2, 1)
        return sequence

    @staticmethod
    def build_convolution_stack(in_channels: int,
                                out_channels: int,
                                kernel_size: int,
                                num_layers: int):

        convolution_stack = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     padding=(kernel_size - 1) // 2)
                                           for _ in range(num_layers)])

        return convolution_stack

    def forward(self, sequence):
        if self.use_one_hot:
            sequence = F.one_hot(sequence, num_classes=self.vocab_size).float()
        else:
            sequence = self.embedding(sequence)
        # (batch_size, num_filters, seq_len)
        residual_sequence = self.residual_resize(sequence).permute(0, 2, 1)
        # (batch_size, seq_len, hidden_size)
        final_residual_sequence = self.final_residual_resize(sequence)
        # (batch_size, vocab_size, seq_len)
        sequence = sequence.permute(0, 2, 1)

        convolutions_out = []
        convolution_stack: nn.ModuleList
        for convolution_stack in self.convolutions_stacks:
            sequence_stack = []
            for convolution_layer in convolution_stack:
                convolved_sequence = convolution_layer(sequence)
                convolved_sequence = self.apply_layer_norm(convolved_sequence)
                convolved_sequence = torch.relu(convolved_sequence)
                # residual connection
                convolved_sequence = (convolved_sequence + residual_sequence) * self.scale
                sequence_stack.append(convolved_sequence)
            sequence_stack = torch.cat(sequence_stack, dim=1)
            convolutions_out.append(sequence_stack)
        # (batch_size, len(kernel_sizes) * num_layers * num_filters, seq_len)
        convolutions_out = torch.cat(convolutions_out, dim=1)
        sequence = convolutions_out.permute(0, 2, 1)

        sequence = self.final_resize(sequence)
        sequence += final_residual_sequence

        return sequence


class TransformerEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 n_heads,
                 hidden_size,
                 dropout,
                 padding_index,
                 max_len,
                 num_layers):
        super(TransformerEncoder, self).__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.pos_encoder = PositionalEncoding(d_model=emb_dim,
                                              dropout=dropout,
                                              max_len=max_len)

        self.transformer_encoder = transformer_encoder.TransformerEncoder(d_model=emb_dim,
                                                                          d_ff=hidden_size,
                                                                          n_heads=n_heads,
                                                                          n_layers=num_layers,
                                                                          dropout=dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.vocab_size)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)
        return x


def get_non_pad_lens(seq):
    lens = seq.size(-1) - (seq == 0).sum(-1)
    return lens
