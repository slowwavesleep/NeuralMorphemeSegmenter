import json
import os
import argparse
from operator import itemgetter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import PAD_INDEX, UNK_INDEX, TOKENIZERS_DIR, DATA_PATHS, MAX_LEN
from src.utils.datasets import BmesSegmentationDataset
from src.utils.etc import read_experiment_data
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.utils.segmenters import NeuralSegmenter

parser = argparse.ArgumentParser(description="Generate boundary predictions on test examples")
parser.add_argument("--model_name", help="Model to use for generation", type=str)
parser.add_argument("--train_type", help="Specify the dataset the model was trained on", type=str)
parser.add_argument("--experiment_id", help="Specify a particular successful experiment ID", type=str, default=None)
parser.add_argument("--batch_size", help="Batch size for predictions", type=int, default=1024)
args = parser.parse_args()

experiments = []
with open("./logs/successful_experiments.jsonl") as file:
    for line in file:
        experiments.append(json.loads(line))

if not experiments:
    raise ValueError("No experiments available")

experiments = [experiment for experiment in experiments
               if experiment["model_name"] == args.model_name and experiment["train_type"] == args.train_type]

if args.experiment_id is not None:
    experiments = [experiment for experiment in experiments if experiment["experiment_id"] == args.experiment_id]
    if not experiments:
        raise ValueError("Specified experiment ID not found")

if not experiments:
    raise ValueError("No suitable experiment found")
elif len(experiments) == 1:
    experiment_info = experiments[0]
else:
    experiments = sorted(experiments, key=itemgetter("test_accuracy"), reverse=True)
    experiment_info = experiments[0]

model_name = args.model_name
experiment_id = experiment_info["experiment_id"]

model_path = f"models/{model_name}/{experiment_id}/best_model_state_dict.pth"
model_config_log = f"logs/{model_name}/{experiment_id}/config.json"
batch_size = args.batch_size
train_type = experiment_info["train_type"]

write_dir = f"data/predictions/{train_type}/{model_name}/{experiment_id}"
write_path = f"{write_dir}/predictions.jsonl"

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

with open(model_config_log) as file:
    config = json.loads(file.read())

if model_name != "RandomTagger":
    model_params = config["model_params"]
else:
    model_params = None

original_tokenizer_path = f"{TOKENIZERS_DIR}/original.json"
bmes_tokenizer_path = f"{TOKENIZERS_DIR}/bmes.json"

original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX).vocab_from_file(original_tokenizer_path)

bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                              unk_index=UNK_INDEX,
                              convert_to_bmes=True).vocab_from_file(bmes_tokenizer_path)

device = torch.device("cuda")

if model_name == "BaselineTagger":
    from src.nn.models import BaselineTagger

    model = BaselineTagger(char_vocab_size=original_tokenizer.vocab_size,
                           tag_vocab_size=bmes_tokenizer.vocab_size,
                           padding_index=PAD_INDEX,
                           **model_params)

elif model_name == "BaselineCrfTagger":
    from src.nn.models import BaselineCrfTagger

    model = BaselineCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                              tag_vocab_size=bmes_tokenizer.vocab_size,
                              padding_index=PAD_INDEX,
                              **model_params)


elif model_name == "LstmTagger":
    from src.nn.models import LstmTagger

    model = LstmTagger(char_vocab_size=original_tokenizer.vocab_size,
                       tag_vocab_size=bmes_tokenizer.vocab_size,
                       padding_index=PAD_INDEX,
                       **model_params)

elif model_name == "LstmCrfTagger":
    from src.nn.models import LstmCrfTagger

    model = LstmCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                          tag_vocab_size=bmes_tokenizer.vocab_size,
                          padding_index=PAD_INDEX,
                          **model_params)

elif model_name == "CnnTagger":
    from src.nn.models import CnnTagger

    model = CnnTagger(char_vocab_size=original_tokenizer.vocab_size,
                      tag_vocab_size=bmes_tokenizer.vocab_size,
                      padding_index=PAD_INDEX,
                      **model_params)

elif model_name == "CnnCrfTagger":
    from src.nn.models import CnnCrfTagger

    model = CnnCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                         tag_vocab_size=bmes_tokenizer.vocab_size,
                         padding_index=PAD_INDEX,
                         **model_params)


elif model_name == "TransformerTagger":
    from src.nn.models import TransformerTagger

    model = TransformerTagger(char_vocab_size=original_tokenizer.vocab_size,
                              tag_vocab_size=bmes_tokenizer.vocab_size,
                              padding_index=PAD_INDEX,
                              max_len=MAX_LEN,
                              **model_params)

elif model_name == "TransformerCrfTagger":
    from src.nn.models import TransformerCrfTagger

    model = TransformerCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                                 tag_vocab_size=bmes_tokenizer.vocab_size,
                                 padding_index=PAD_INDEX,
                                 max_len=MAX_LEN,
                                 **model_params)

else:
    raise NotImplementedError


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

print(f"Generating predictions by {model_name} (experiment ID: {experiment_id}) on {train_type}")
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
        # TODO fix this properly
        original_word = original_tokenizer.decode(original_word)[:true_length].replace("<UNK>", "ё")
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

print(f"The results are located in {write_path}")
