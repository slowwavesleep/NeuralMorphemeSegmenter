import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from constants import CONVERTED_FORMS_PATHS, PROCESSED_ARTICLES_PATH
from src.utils.conversion import process_forms, write_examples


# TODO add different splitting strategies
def main():
    Path("data/converted_forms").mkdir(parents=True, exist_ok=True)

    examples = []
    with open(PROCESSED_ARTICLES_PATH) as file:
        for line in file:
            example = json.loads(line)
            examples.append(example["generated_forms"])

    examples_train, examples_test = train_test_split(examples,
                                                     test_size=0.3,
                                                     random_state=42,
                                                     shuffle=True)

    examples_valid, examples_test = train_test_split(examples_test,
                                                     test_size=0.5,
                                                     random_state=42,
                                                     shuffle=True)

    train = []
    for example in examples_train:
        train.extend(process_forms(example))
    write_examples(CONVERTED_FORMS_PATHS["train"], train)

    valid = []
    for example in examples_valid:
        valid.extend(process_forms(example))
    write_examples(CONVERTED_FORMS_PATHS["valid"], valid)

    test = []
    for example in examples_test:
        test.extend(process_forms(example))
    write_examples(CONVERTED_FORMS_PATHS["test"], test)


if __name__ == "__main__":
    main()
