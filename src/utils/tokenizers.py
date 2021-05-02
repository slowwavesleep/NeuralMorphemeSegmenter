import json
from typing import List, Tuple, Dict


def bmes2sequence(original_sequence: str, bmes_tags: str, *, sep: str = "|") -> str:
    assert len(original_sequence) == len(bmes_tags)

    output = []
    for ch, tag in zip(original_sequence, bmes_tags):
        output.append(ch)
        if tag in ("S", "E"):
            output.append(sep)
    if output[-1] == sep:
        output = output[:-1]
    return "".join(output)


def sequence2bmes(segmented_sequence: str, *, sep: str = "|") -> str:
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
                     unk_index: int) -> Tuple[Dict[int, str], Dict[str, int], List[int]]:
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

    labels = [key for key in index2sym.keys() if key not in (pad_index, unk_index)]

    return index2sym, sym2index, labels


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

        self.meaningful_label_indices = None

        self._vocab_flag = False

    def build_vocab(self,
                    sequences: List[str]):

        if self.convert_to_bmes:
            sequences = [sequence2bmes(sequence) for sequence in sequences]

        self._index2sym, self._sym2index, self.meaningful_label_indices = _build_sym_index(sequences,
                                                                                           self.pad_index,
                                                                                           self.unk_index)

        self._vocab_flag = True

        return self

    def vocab_from_file(self,
                        path: str):
        with open(path) as file:
            data = json.loads(file.read())

        self._index2sym = data["index2sym"]
        self._sym2index = data["sym2index"]
        self.meaningful_label_indices = data["meaningful_label_indices"]

        if self._sym2index["<PAD>"] != self.pad_index or self._sym2index["<UNK>"] != self.unk_index:
            raise ValueError("Token index mismatch")

        self._vocab_flag = True

        return self

    def vocab_to_file(self,
                      path: str):
        with open(path, "w") as file:
            data = {"index2sym": self._index2sym,
                    "sym2index": self._sym2index,
                    "meaningful_label_indices": self.meaningful_label_indices}
            file.write(json.dumps(data,
                                  ensure_ascii=False,
                                  indent=4))

    @property
    def labels(self) -> dict:
        labels = self._sym2index.copy()
        return labels

    @property
    def vocab_size(self) -> int:
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        return len(self._index2sym)

    def encode(self, sequence: str) -> List[int]:
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        if self.convert_to_bmes:
            sequence = sequence2bmes(sequence)
        return [self._sym2index.get(element, self.unk_index) for element in sequence]

    def decode(self, sequence: List[int]) -> str:
        if not self._vocab_flag:
            raise RuntimeError("Tokenizer vocabulary has not been initialized!")
        return "".join([self._index2sym.get(element, "<UNK>") for element in sequence])

    def pad_or_clip(self,
                    sequence: List[int],
                    max_len: int,
                    *,
                    pre_pad: bool = False) -> List[int]:

        sequence = sequence[:max_len]
        num_pads = max(0, max_len - len(sequence))
        pads = [self.pad_index] * num_pads

        if pads:
            if pre_pad:
                sequence = pads + sequence
            else:
                sequence = sequence + pads

        return sequence
