import re
import json
from typing import List, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split

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
    examples = []
    has_splits = []
    for key, value in ORIGINAL_LEMMAS_PATHS.items():
        with open(value) as in_file:
            for line in in_file:
                original, segmented = line.split("\t")
                segmented = remove_labels(segmented)
                examples.append((original, segmented))
                has_splits.append(SEP_TOKEN in segmented)

    examples_train, examples_test, has_splits_train, has_splits_test = train_test_split(examples,
                                                                                        has_splits,
                                                                                        test_size=0.3,
                                                                                        random_state=42,
                                                                                        shuffle=True,
                                                                                        stratify=has_splits)

    examples_valid, examples_test, _, _ = train_test_split(examples_test,
                                                           has_splits_test,
                                                           test_size=0.5,
                                                           random_state=42,
                                                           shuffle=True,
                                                           stratify=has_splits_test)

    write_examples(data=examples_train, path="data/converted_lemmas/lemmas_train.jsonl")
    write_examples(data=examples_valid, path="data/converted_lemmas/lemmas_valid.jsonl")
    write_examples(data=examples_test, path="data/converted_lemmas/lemmas_test.jsonl")


if __name__ == "__main__":
    main()


