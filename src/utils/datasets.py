from typing import List, Tuple, Optional
import math

from torch.utils.data import Dataset
import torch

from src.utils.tokenizers import SymTokenizer


def sort_data(indices: List[int],
              sequences: List[str],
              labels: List[str]) -> Tuple[List[int], List[str], List[str]]:
    indices, sequences, labels = zip(*sorted(zip(indices, sequences, labels), key=lambda x: len(x[1])))
    return indices, sequences, labels


def make_batches(indices: List[int],
                 sequences: List[str],
                 labels: List[str],
                 batch_size: int) -> Tuple[List[List[int]], List[List[str]], List[List[str]]]:

    assert len(sequences) == len(labels) == len(indices)

    indices, sequences, labels = sort_data(indices, sequences, labels)

    identifier_batches = []
    text_batches = []
    label_batches = []

    for i_batch in range(math.ceil(len(sequences) / batch_size)):
        identifier_batches.append(indices[i_batch * batch_size:(i_batch + 1) * batch_size])
        text_batches.append(sequences[i_batch * batch_size:(i_batch + 1) * batch_size])
        label_batches.append(labels[i_batch * batch_size:(i_batch + 1) * batch_size])

    return identifier_batches, text_batches, label_batches


class BmesSegmentationDataset(Dataset):

    def __init__(self,
                 *,
                 indices: List[int],
                 original: List[str],
                 segmented: List[str],
                 original_tokenizer: SymTokenizer,
                 bmes_tokenizer: SymTokenizer,
                 batch_size: int,
                 pad_index: int,
                 unk_index: int,
                 max_len: int):
        assert len(original) == len(segmented)

        self.batch_size = batch_size

        self.index_batches, self.original_batches, self.segmented_batches = make_batches(indices=indices,
                                                                                         sequences=original,
                                                                                         labels=segmented,
                                                                                         batch_size=self.batch_size)

        self.unk_index = unk_index
        self.pad_index = pad_index

        self.max_len = max_len

        self.original_tokenizer = original_tokenizer
        self.bmes_tokenizer = bmes_tokenizer

    def __len__(self) -> int:
        return len(self.index_batches)

    def prepare_sample(self,
                       sequence: str,
                       max_len: int,
                       bmes: bool) -> Tuple[List[int], int]:
        if not bmes:
            sequence = self.original_tokenizer.encode(sequence)
        else:
            sequence = self.bmes_tokenizer.encode(sequence)

        sequence = sequence[:max_len]
        true_len = len(sequence)
        pads = [self.pad_index] * (max_len - len(sequence))
        sequence += pads

        return sequence, true_len

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        index_batch = self.index_batches[index]
        original_batch = self.original_batches[index]
        segmented_batch = self.segmented_batches[index]

        max_len = min([self.max_len, max([len(sample) for sample in original_batch])])

        batch_indices = []
        batch_x = []
        batch_y = []
        batch_lengths = []

        for index, sample in enumerate(original_batch):
            identifier = index_batch[index]
            x, true_len = self.prepare_sample(sample, max_len, bmes=False)
            y, _ = self.prepare_sample(segmented_batch[index], max_len, bmes=True)
            batch_indices.append(identifier)
            batch_x.append(x)
            batch_y.append(y)
            batch_lengths.append(true_len)

        batch_indices = torch.tensor(batch_indices).long()
        batch_x = torch.tensor(batch_x).long()
        batch_y = torch.tensor(batch_y).long()
        batch_lengths = torch.tensor(batch_lengths).long()

        return batch_indices, batch_x, batch_y, batch_lengths


# class BmesSegmentationDataset(Dataset):
#
#     def __init__(self,
#                  *,
#                  indices: List[int],
#                  original: List[str],
#                  segmented: List[str],
#                  original_tokenizer: SymTokenizer,
#                  bmes_tokenizer: SymTokenizer,
#                  pad_index: int,
#                  unk_index: int,
#                  max_len: int):
#         self.indices = indices
#         self.original = original
#         self.segmented = segmented
#
#         assert len(original) == len(segmented)
#
#         self.unk_index = unk_index
#         self.pad_index = pad_index
#
#         self.max_len = max_len
#
#         self.index2char = None
#         self.char2index = None
#
#         self.original_tokenizer = original_tokenizer
#         self.bmes_tokenizer = bmes_tokenizer
#
#     def __len__(self) -> int:
#         return len(self.original)
#
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int, int]:
#         encoder_seq = self.original_tokenizer.encode(self.original[index])
#         target_seq = self.bmes_tokenizer.encode(self.segmented[index])
#
#         true_length = len(encoder_seq)
#         item_index = self.indices[index]
#
#         encoder_seq = self.original_tokenizer.pad_or_clip(encoder_seq,
#                                                           max_len=self.max_len)
#         target_seq = self.bmes_tokenizer.pad_or_clip(target_seq,
#                                                      max_len=self.max_len)
#
#         encoder_seq = torch.tensor(encoder_seq).long()
#         target_seq = torch.tensor(target_seq).long()
#
#         return encoder_seq, target_seq, true_length, item_index