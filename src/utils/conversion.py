import json
import re
from sklearn.utils import shuffle
from typing import List, Tuple, Dict

from constants import SEP_TOKEN
from src.utils.tokenizers import sequence2bmes


def write_examples(path: str, data: List[Tuple[str, str]], seed: int, remove_duplicates: bool = True):
    if remove_duplicates:
        data = list(set(data))
    data = shuffle(data, random_state=seed)
    index = 1
    with open(path, "w") as out_file:
        for original, segmented in data:
            data = {"index": index,
                    "original": original,
                    "segmented": segmented,
                    "bmes": sequence2bmes(segmented)}
            index += 1
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")


def remove_labels(labeled_segmented_example: str) -> str:
    segments = re.findall(r"[^:A-Zaz\/]+", labeled_segmented_example.strip("\n"))
    return SEP_TOKEN.join(segments)


def remove_sep(segmented_form: str, *, sep: str = SEP_TOKEN) -> str:
    return segmented_form.replace(sep, "")


def process_forms(data: Dict[str, str], *, sep: str = SEP_TOKEN) -> List[Tuple[str, str]]:
    result = []
    for value in data.values():
        if value:
            original = remove_sep(value, sep=sep).lower()
            segmented = value.lower()
            result.append((original, segmented))
    return result
