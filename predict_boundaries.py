import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import PAD_INDEX, UNK_INDEX, TOKENIZERS_DIR, DATA_PATHS, MAX_LEN
from src.utils.datasets import BmesSegmentationDataset
from src.utils.etc import read_experiment_data
from src.utils.tokenizers import SymTokenizer
from src.nn.models import LstmCrfTagger, LstmTagger
from src.utils.segmenters import NeuralSegmenter

model_name = "LstmTagger"
experiment_id = "cfb7f313753d42ee84c26c8708f51ebd"
model_path = f"models/{model_name}/{experiment_id}/best_model_state_dict.pth"
model_config_log = f"logs/{model_name}/{experiment_id}/config.json"
batch_size = 1024
train_type = "lemmas"

with open(model_config_log) as file:
    config = json.loads(file.read())

model_params = config["model_params"]

original_tokenizer_path = f"{TOKENIZERS_DIR}/original.json"
bmes_tokenizer_path = f"{TOKENIZERS_DIR}/bmes.json"

original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX).vocab_from_file(original_tokenizer_path)

bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                              unk_index=UNK_INDEX,
                              convert_to_bmes=True).vocab_from_file(bmes_tokenizer_path)

device = torch.device("cuda")

model = LstmTagger(char_vocab_size=original_tokenizer.vocab_size,
                   tag_vocab_size=bmes_tokenizer.vocab_size,
                   padding_index=PAD_INDEX,
                   **model_params)

model.load_state_dict(torch.load(model_path))

model.to(device)

segmenter = NeuralSegmenter(original_tokenizer=original_tokenizer,
                            bmes_tokenizer=bmes_tokenizer,
                            model=model,
                            device=device)

test_indices, test_original, test_segmented = read_experiment_data(DATA_PATHS[train_type.lower()]["test"])

test_ds = BmesSegmentationDataset(indices=test_indices,
                                  original=test_original,
                                  segmented=test_segmented,
                                  original_tokenizer=original_tokenizer,
                                  bmes_tokenizer=bmes_tokenizer,
                                  pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  max_len=MAX_LEN,
                                  batch_size=batch_size)

test_loader = DataLoader(test_ds, batch_size=1)

# print(segmenter.segment_batch(test_original[:1024]))
# results = []
# for i in tqdm(range(0, len(test_original))):
#     batch_slice = slice(i, i + batch_size)
#     batch_indices = test_indices[batch_slice]
#     batch_original = test_original[batch_slice]
#     batch_segmented = test_segmented[batch_slice]
#     predictions = segmenter.segment_batch(batch_original)
#     batch_results = []
#     for index, original, segmented, prediction in zip(batch_indices, batch_original, batch_segmented, predictions):
#         batch_results.append({"index": index,
#                               "original": original,
#                               "segmented": segmented,
#                               "prediction": prediction,
#                               "match": segmented == prediction})
#     results.extend(batch_results)
