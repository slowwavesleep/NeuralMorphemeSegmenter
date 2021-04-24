import itertools
import json
from typing import Iterable, List, Tuple


def remove_pads(sequences: Iterable[Iterable[int]],
                true_lengths: Iterable[int],
                *,
                pre_pad: bool = False) -> List[List[int]]:

    assert len(sequences) == len(true_lengths)

    output = []

    for element, true_length in zip(sequences, true_lengths):
        if pre_pad:
            element = element[max(0, len(element) - true_length):]
        else:
            element = element[:true_length]
        output.append(list(element))

    return output


def flatten_list(list_to_flatten: List[list]) -> list:
    return list(itertools.chain(*list_to_flatten))


def read_converted_data(path: str) -> Tuple[List[int], List[str], List[str]]:
    indices = []
    original = []
    segmented = []
    with open(path) as file:
        for line in file:
            data = json.loads(line)
            indices.append(data["index"])
            original.append(data["original"])
            segmented.append(data["segmented"])

    return indices, original, segmented
