import re
import json
from typing import List, Tuple
from pathlib import Path

import numpy as np

from constants import SEP_TOKEN, ORIGINAL_LEMMAS_PATHS


def remove_labels(labeled_segmented_example: str) -> str:
    segments = re.findall(r"[^:A-Zaz\/]+", labeled_segmented_example.strip("\n"))
    return SEP_TOKEN.join(segments)


def write_examples(path: str, data: List[Tuple[str, str]]):
    index = 1
    with open(path, "w") as out_file:
        for original, segmented in data:
            data = {"index": index,
                    "original": original,
                    "segmented": remove_labels(segmented)}
            index += 1
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    Path("data/converted_lemmas").mkdir(parents=True, exist_ok=True)
    for key, value in ORIGINAL_LEMMAS_PATHS.items():
        if key == "train":
            with open(f"data/converted_lemmas/lemmas_{key}.jsonl", "w") as out_file:
                with open(value) as in_file:
                    index = 1
                    for line in in_file:
                        original, segmented = line.split("\t")
                        data = {"index": index,
                                "original": original,
                                "segmented": remove_labels(segmented)}
                        index += 1
                        out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            examples = []
            with open(value) as in_file:
                for line in in_file:
                    original, segmented = line.split("\t")
                    examples.append((original, segmented))
            split = int(np.floor(len(examples) / 2))
            valid = examples[:split]
            test = examples[split:]

            write_examples(f"data/converted_lemmas/lemmas_valid.jsonl", valid)
            write_examples(f"data/converted_lemmas/lemmas_test.jsonl", test)


if __name__ == "__main__":
    main()


