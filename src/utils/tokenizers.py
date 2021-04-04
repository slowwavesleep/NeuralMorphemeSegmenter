from typing import List, Tuple, Dict


def _sequence2bmes(segmented_sequence: str, *, sep: str = "|") -> str:
    segments = segmented_sequence.split(sep)
    result = []
    for segment in segments:
        result.append(_segment2bmes(segment))
    return "".join(result)


def _segment2bmes(segment: str) -> str:
    segment_len = len(segment)
    if segment_len == 1:
        return "S"
    elif segment_len == 2:
        return "BE"
    elif segment_len > 2:
        return f"B{'M' * (segment_len - 2)}E"
    else:
        return ""


def _build_sym_index(sequences: List[str],
                     pad_index: int,
                     unk_index: int) -> Tuple[Dict[int, str], Dict[str, int]]:

    unique_chars = set("".join(sequences))
    index2sym = {index: ch for index, ch in enumerate(unique_chars)}

    if pad_index in index2sym:
        index2sym[len(index2sym)] = index2sym[pad_index]
        index2sym[pad_index] = "<PAD>"
    else:
        index2sym[pad_index] = "<PAD>"

    if unk_index in index2sym:
        index2sym[len(index2sym)] = index2sym[unk_index]
        index2sym[unk_index] = "<UNK>"
    else:
        index2sym[unk_index] = "<UNK>"

    sym2index = {value: key for key, value in index2sym.items()}

    return index2sym, sym2index


class SymTokenizer:

    def __init__(self,
                 *,
                 pad_index: int,
                 unk_index: int,
                 convert_to_bmes: bool = False):

        self.pad_index = pad_index
        self.unk_index = unk_index

        self.convert_to_bmes = convert_to_bmes

        self._index2sym = None
        self._sym2index = None

        self._vocab_flag = False

    def build_vocab(self,
                    sequences: List[str]):

        if self.convert_to_bmes:
            sequences = [_sequence2bmes(sequence) for sequence in sequences]

        self._index2sym, self._sym2index = _build_sym_index(sequences, self.pad_index, self.unk_index)

        self._vocab_flag = True

        return self

    @property
    def vocab_size(self):
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        return len(self._index2sym)

    def encode(self, sequence: str):
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        if self.convert_to_bmes:
            sequence = _sequence2bmes(sequence)
        return [self._sym2index.get(element, self.unk_index) for element in sequence]

    def decode(self, sequence: List[int]):
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        return "".join([self._index2sym.get(element, "<UNK>") for element in sequence])

    def pad_or_clip(self, sequence: List[int], max_len: int, *, pre_pad: bool = False):
        sequence = sequence[:max_len]
        num_pads = max(0, max_len - len(sequence))
        pads = [self.pad_index] * num_pads

        if pads:
            if pre_pad:
                sequence = pads + sequence
            else:
                sequence = sequence + pads
            return sequence
        else:
            return sequence
