import itertools
from typing import Iterable, List


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