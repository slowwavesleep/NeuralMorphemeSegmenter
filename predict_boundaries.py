import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import PAD_INDEX, UNK_INDEX, TOKENIZERS_DIR, DATA_PATHS, MAX_LEN
from src.utils.datasets import BmesSegmentationDataset
from src.utils.etc import read_experiment_data
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.nn.models import LstmCrfTagger, LstmTagger
from src.utils.segmenters import NeuralSegmenter

# TODO parametrize all of this
model_name = "LstmTagger"
experiment_id = "cfb7f313753d42ee84c26c8708f51ebd"
model_path = f"models/{model_name}/{experiment_id}/best_model_state_dict.pth"
model_config_log = f"logs/{model_name}/{experiment_id}/config.json"
batch_size = 1024
train_type = "lemmas"
write_dir = f"data/predictions/{train_type}/{model_name}/{experiment_id}"
write_path = f"{write_dir}/predictions.jsonl"

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

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
verbose = True
model.eval()
progress_bar = tqdm(total=len(test_loader), disable=not verbose, desc="Evaluate")

indices = []
original_words = []
segmented_words = []
predictions = []
true_lengths = []

for index_seq, encoder_seq, target_seq, true_lens in test_loader:
    index_seq = index_seq.squeeze(0)
    encoder_seq = encoder_seq.to(device).squeeze(0)
    target_seq = target_seq.to(device).squeeze(0)
    true_lens = true_lens.to(device).squeeze(0)
    with torch.no_grad():
        batch_predictions = model.predict(encoder_seq, true_lens)
    indices.extend(index_seq.cpu().numpy())
    original_words.extend(encoder_seq.cpu().numpy())
    predictions.extend(batch_predictions)
    segmented_words.extend(target_seq.cpu().numpy())
    true_lengths.extend(true_lens.cpu().numpy())
    progress_bar.update()

progress_bar.close()

with open(write_path, "w") as file:
    for index, original_word, target_segmentation, prediction, true_length in zip(indices,
                                                                                  original_words,
                                                                                  segmented_words,
                                                                                  predictions,
                                                                                  true_lengths):
        original_word = original_tokenizer.decode(original_word)[:true_length]
        target_bmes = bmes_tokenizer.decode(target_segmentation)[:true_length]
        predicted_bmes = bmes_tokenizer.decode(prediction)[:true_length]

        target_segmentation_word = bmes2sequence(original_sequence=original_word,
                                                 bmes_tags=target_bmes)

        predicted_segmentation_word = bmes2sequence(original_sequence=original_word,
                                                    bmes_tags=predicted_bmes)

        data = {"index": int(index),
                "original": original_word,
                "segmented": target_segmentation_word,
                "bmes": target_bmes,
                "predicted_segmented": predicted_segmentation_word,
                "predicted_bmes": predicted_bmes,
                "match": target_segmentation_word == predicted_segmentation_word}

        file.write(json.dumps(data, ensure_ascii=False) + "\n")

