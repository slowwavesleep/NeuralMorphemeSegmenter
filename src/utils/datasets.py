from typing import List, Tuple, Optional

from torch.utils.data import Dataset
from torch import Tensor


class BmesSegmentationDataset(Dataset):

    def __init__(self,
                 *,
                 indices: List[int],
                 original: List[str],
                 segmented: List[str],
                 sym_tokenizer,
                 pad_index: int,
                 unk_index: int,
                 max_len: Optional[int] = None,
                 convert_to_bmes: Optional[bool] = True):
        self.indices = indices
        self.original = original
        self.segmented = segmented

        assert len(original) == len(segmented)

        self.unk_index = unk_index
        self.pad_index = pad_index

        self.max_len = max_len

        self.index2char = None
        self.char2index = None

        self.original_tokenizer = sym_tokenizer(pad_index=self.pad_index,
                                                unk_index=self.unk_index).build_vocab(original)

        self.bmes_tokenizer = sym_tokenizer(pad_index=self.pad_index,
                                            unk_index=self.unk_index,
                                            convert_to_bmes=convert_to_bmes).build_vocab(segmented)

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int, int]:
        encoder_seq = self.original_tokenizer.encode(self.original[index])
        target_seq = self.bmes_tokenizer.encode(self.segmented[index])

        true_length = len(encoder_seq)
        item_index = self.indices[index]

        encoder_seq = self.original_tokenizer.pad_or_clip(encoder_seq,
                                                          max_len=self.max_len)
        target_seq = self.bmes_tokenizer.pad_or_clip(target_seq,
                                                     max_len=self.max_len)

        encoder_seq = Tensor(encoder_seq).long()
        target_seq = Tensor(target_seq).long()

        return encoder_seq, target_seq, true_length, item_index
