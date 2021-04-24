SEP_TOKEN = "|"

PROCESSED_ARTICLES_PATH = "data/processed_articles/processed_articles.jsonl"

ORIGINAL_LEMMAS_PATHS = {"train": "data/original/train_Tikhonov_reformat.txt",
                         "test": "data/original/test_Tikhonov_reformat.txt"}

CONVERTED_LEMMAS_PATHS = {"train": "data/converted_lemmas/lemmas_train.jsonl",
                          "valid": "data/converted_lemmas/lemmas_valid.jsonl",
                          "test": "data/converted_lemmas/lemmas_test.jsonl"}

CONVERTED_FORMS_PATHS = {"train": "data/converted_forms/forms_train.jsonl",
                         "valid": "data/converted_forms/forms_valid.jsonl",
                         "test": "data/converted_forms/forms_test.jsonl"}

PAD_INDEX = 0
UNK_INDEX = 1
MAX_LEN = 36

MODELS_MAP = {"lstm": None,
              "lstm_crf": None,
              "cnn": None,
              "cnn_crf": None,
              "transformer": None,
              "transformer_crf": None}
