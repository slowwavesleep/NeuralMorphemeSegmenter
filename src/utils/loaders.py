from torch.utils.data import Dataset
from typing import List


class SegmentationDataset(Dataset):

    def __init__(self,
                 original: List[str],
                 segmented: List[str],
                 pad_index: int,
                 unk_index: int,
                 max_len: int):

        self.original = original
        self.segmented = segmented

        self.unk_index = unk_index
        self.pad_index = pad_index

        self.max_len = max_len

        self.index2char = None
        self.char2index = None







