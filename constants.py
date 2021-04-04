SEP_TOKEN = "|"

ORIGINAL_LEMMAS_PATHS = {"train": "data/original/train_Tikhonov_reformat.txt",
                         "test": "data/original/test_Tikhonov_reformat.txt"}

CONVERTED_LEMMAS_PATHS = {"train": "data/lemmas_train.jsonl",
                          "test": "data/lemmas_test.jsonl"}

PAD_INDEX = 0
UNK_INDEX = 1
MAX_LEN = 36