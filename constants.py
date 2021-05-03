SEP_TOKEN = "|"

PROCESSED_ARTICLES_PATH = "data/processed_articles/processed_articles.jsonl"

ORIGINAL_LEMMAS_PATHS = {"train": "data/original/train_Tikhonov_reformat.txt",
                         "test": "data/original/test_Tikhonov_reformat.txt"}

DATA_PATHS = {"lemmas": {"train": "data/converted_lemmas/lemmas_train.jsonl",
                         "valid": "data/converted_lemmas/lemmas_valid.jsonl",
                         "test": "data/converted_lemmas/lemmas_test.jsonl"},

              "forms": {"train": "data/converted_forms/forms_train.jsonl",
                        "valid": "data/converted_forms/forms_valid.jsonl",
                        "test": "data/converted_forms/forms_test.jsonl"}}

TOKENIZERS_DIR = "./tokenizers"

PAD_INDEX = 0
UNK_INDEX = 1
MAX_LEN = 36
