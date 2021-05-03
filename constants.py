SEP_TOKEN = "|"

PROCESSED_ARTICLES_PATH = "data/processed_articles/processed_articles.jsonl"

ORIGINAL_LEMMAS_PATHS = {"train": "data/original/train_Tikhonov_reformat.txt",
                         "test": "data/original/test_Tikhonov_reformat.txt"}

DATA_PATHS = {
    "lemmas": {"train": "data/converted_lemmas/lemmas_train.jsonl",
               "valid": "data/converted_lemmas/lemmas_valid.jsonl",
               "test": "data/converted_lemmas/lemmas_test.jsonl"},

    "lemmas_low_resource": {"train": "data/converted_lemmas/lemmas_low_resource_train.jsonl",
                            "valid": "data/converted_lemmas/lemmas_valid.jsonl",
                            "test": "data/converted_lemmas/lemmas_test.jsonl"},

    "forms": {"train": "data/converted_forms/forms_train.jsonl",
              "valid": "data/converted_forms/forms_valid.jsonl",
              "test": "data/converted_forms/forms_test.jsonl"},

    "forms_low_resource": {"train": "data/converted_forms/forms_low_resource_train.jsonl",
                           "valid": "data/converted_forms/forms_valid.jsonl",
                           "test": "data/converted_forms/forms_test.jsonl"},

    "forms_shuffled": {"train": "data/converted_forms/forms_shuffled_train.jsonl",
                       "valid": "data/converted_forms/forms_shuffled_valid.jsonl",
                       "test": "data/converted_forms/forms_shuffled_test.jsonl"},

    "forms_shuffled_low_resource": {"train": "data/converted_forms/forms_shuffled_low_resource_train.jsonl",
                                    "valid": "data/converted_forms/forms_shuffled_valid.jsonl",
                                    "test": "data/converted_forms/forms_shuffled_test.jsonl"}
}

TOKENIZERS_DIR = "./tokenizers"

SHUFFLE_SEED = 42
LOW_RESOURCE_SIZE = 0.05
PAD_INDEX = 0
UNK_INDEX = 1
MAX_LEN = 36
