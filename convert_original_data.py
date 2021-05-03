import math
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils.conversion import write_examples, remove_labels
from constants import SEP_TOKEN, ORIGINAL_LEMMAS_PATHS, DATA_PATHS, SHUFFLE_SEED, LOW_RESOURCE_SIZE


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

    n_low_resource_examples = math.floor(len(examples_train) * LOW_RESOURCE_SIZE)

    write_examples(data=examples_train, path=DATA_PATHS["lemmas"]["train"], seed=SHUFFLE_SEED)
    write_examples(data=examples_train[:n_low_resource_examples],
                   path=DATA_PATHS["lemmas_low_resource"]["train"],
                   seed=SHUFFLE_SEED)
    write_examples(data=examples_valid, path=DATA_PATHS["lemmas"]["valid"], seed=SHUFFLE_SEED)
    write_examples(data=examples_test, path=DATA_PATHS["lemmas"]["test"], seed=SHUFFLE_SEED)


if __name__ == "__main__":
    main()


