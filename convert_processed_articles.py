import json
import math
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from constants import DATA_PATHS, PROCESSED_ARTICLES_PATH, LOW_RESOURCE_SIZE, SHUFFLE_SEED
from src.utils.conversion import process_forms, write_examples


def main():
    Path("data/converted_forms").mkdir(parents=True, exist_ok=True)

    examples = []
    with open(PROCESSED_ARTICLES_PATH) as file:
        for line in file:
            example = json.loads(line)
            examples.append(example["generated_forms"])

    examples_train, examples_test = train_test_split(examples,
                                                     test_size=0.3,
                                                     random_state=SHUFFLE_SEED,
                                                     shuffle=True)

    examples_valid, examples_test = train_test_split(examples_test,
                                                     test_size=0.5,
                                                     random_state=SHUFFLE_SEED,
                                                     shuffle=True)

    # forms of the same word stay in the same split
    train = []
    for example in examples_train:
        train.extend(process_forms(example))
    write_examples(DATA_PATHS["forms"]["train"], train, seed=SHUFFLE_SEED)

    # simulate low resource language for training
    n_low_resource_examples = math.floor(len(train) * LOW_RESOURCE_SIZE)
    write_examples(DATA_PATHS["forms_low_resource"]["train"],
                   train[:n_low_resource_examples],
                   seed=SHUFFLE_SEED)

    valid = []
    for example in examples_valid:
        valid.extend(process_forms(example))
    write_examples(DATA_PATHS["forms"]["valid"], valid, seed=SHUFFLE_SEED)

    test = []
    for example in examples_test:
        test.extend(process_forms(example))
    write_examples(DATA_PATHS["forms"]["test"], test, seed=SHUFFLE_SEED)

    # forms of the same word are spread between different splits

    shuffled = shuffle(train + test + valid, random_state=SHUFFLE_SEED)

    shuffled_train, shuffled_test = train_test_split(shuffled,
                                                     test_size=0.3,
                                                     random_state=SHUFFLE_SEED,
                                                     shuffle=True)

    shuffled_valid, shuffled_test = train_test_split(shuffled_test,
                                                     test_size=0.5,
                                                     random_state=SHUFFLE_SEED,
                                                     shuffle=True)

    write_examples(DATA_PATHS["forms_shuffled"]["train"], shuffled_train, seed=SHUFFLE_SEED)
    write_examples(DATA_PATHS["forms_shuffled_low_resource"]["train"],
                   shuffled_train[:n_low_resource_examples],
                   seed=SHUFFLE_SEED)

    write_examples(DATA_PATHS["forms_shuffled"]["valid"], shuffled_valid, seed=SHUFFLE_SEED)
    write_examples(DATA_PATHS["forms_shuffled"]["test"], shuffled_test, seed=SHUFFLE_SEED)


if __name__ == "__main__":
    main()
