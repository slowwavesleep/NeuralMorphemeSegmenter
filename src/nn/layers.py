import math
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
                 emb_dim: int,
                 spatial_dropout: float,
                 convolution_layers: int,
                 kernel_size: int,
                 padding_index: int,
                 scale: Optional[float] = None):
        super(CnnEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.convolution_layers = convolution_layers
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = np.sqrt(0.5)

        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels=self.emb_dim,
                                                     out_channels=2 * self.emb_dim,
                                                     kernel_size=kernel_size,
                                                     padding=(kernel_size - 1) // 2)
                                           for _ in range(self.convolution_layers)])

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, sequence):
        sequence = self.embedding(sequence)
        sequence = self.spatial_dropout(sequence)
        sequence = self.layer_norm(sequence)
        sequence_input = sequence.permute(0, 2, 1)

        for i, conv in enumerate(self.convolutions):
            convolved = conv(self.spatial_dropout(sequence_input))
            convolved = F.glu(convolved, dim=1)
            convolved = (convolved + sequence_input) * self.scale
            sequence_input = convolved

        convolved = convolved.permute(0, 2, 1)
        combined = (convolved + sequence) * self.scale

        return convolved, combined


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
