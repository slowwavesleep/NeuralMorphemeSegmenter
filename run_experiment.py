import json
from functools import partial
import argparse
import os
import uuid
from datetime import datetime

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from yaml import safe_load

from constants import UNK_INDEX, PAD_INDEX, MAX_LEN, TOKENIZERS_DIR, DATA_PATHS
from src.utils.etc import read_experiment_data
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle
from src.nn.testing_process import testing_cycle
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter
from src.utils.tokenizers import SymTokenizer

parser = argparse.ArgumentParser(description='Run model with specified settings')
parser.add_argument("-p", "--path", help="Path to a YAML configuration file", type=str)
args = parser.parse_args()

if args.path is None:
    config_path = "./configs/default.yml"
else:
    config_path = args.path

with open(config_path) as file:
    config = safe_load(file)

experiment_id: str = uuid.uuid4().hex

flow_control: dict = config["flow_control"]
train_params: dict = config["train_params"]
write_log: bool = flow_control.get("write_log", False)
# allows reinitialization of tokenizers' vocabs from scratch
initialize_tokenizers: bool = flow_control.get("initialize_tokenizers", False)

# train parameters
batch_size = train_params["batch_size"]
model_name = train_params["model_name"]
n_epochs = train_params["n_epochs"]
train_type = train_params["train_type"]
n_without_improvements = train_params["n_without_improvements"]
grad_clip = train_params["grad_clip"]
lr = float(train_params["lr"])
early_stopping = train_params["early_stopping"]
save_best = train_params["save_best"]
save_last = train_params["save_last"]
seed = train_params.get("seed", None)

if model_name != "RandomTagger":
    # specific to models
    model_params = config["model_params"]
else:
    model_params = None

if model_name == "RandomTagger":
    # random tagger can't be trained
    flow_control["train_model"] = False

if train_type.lower() not in DATA_PATHS.keys():
    raise NotImplementedError

# load the relevant data
results_path = f"data/results/{train_type.lower()}/{model_name}/{experiment_id}"
if not os.path.exists(results_path):
    os.makedirs(results_path)

train_indices, train_original, train_segmented = read_experiment_data(DATA_PATHS[train_type.lower()]["train"])
valid_indices, valid_original, valid_segmented = read_experiment_data(DATA_PATHS[train_type.lower()]["valid"])
test_indices, test_original, test_segmented = read_experiment_data(DATA_PATHS[train_type.lower()]["test"])

# prepare the tokenizers
original_tokenizer_path = f"{TOKENIZERS_DIR}/original.json"
bmes_tokenizer_path = f"{TOKENIZERS_DIR}/bmes.json"

# initialize from scratch
if initialize_tokenizers or not (os.path.exists(original_tokenizer_path) and os.path.exists(bmes_tokenizer_path)):

    if not os.path.exists(TOKENIZERS_DIR):
        os.makedirs(TOKENIZERS_DIR)

    original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                      unk_index=UNK_INDEX).build_vocab(train_original)

    original_tokenizer.vocab_to_file(original_tokenizer_path)

    bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  convert_to_bmes=True).build_vocab(train_segmented)

    bmes_tokenizer.vocab_to_file(bmes_tokenizer_path)

# or load existing vocab
else:
    original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                      unk_index=UNK_INDEX).vocab_from_file(original_tokenizer_path)

    bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  convert_to_bmes=True).vocab_from_file(bmes_tokenizer_path)

# initialize sequence bucketing datasets
train_ds = BmesSegmentationDataset(indices=train_indices,
                                   original=train_original,
                                   segmented=train_segmented,
                                   original_tokenizer=original_tokenizer,
                                   bmes_tokenizer=bmes_tokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN,
                                   batch_size=batch_size)

valid_ds = BmesSegmentationDataset(indices=valid_indices,
                                   original=valid_original,
                                   segmented=valid_segmented,
                                   original_tokenizer=original_tokenizer,
                                   bmes_tokenizer=bmes_tokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN,
                                   batch_size=batch_size)

# initialize data loaders
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1)

# TODO don't hard-code this?
# tokenwise metrics to evaluate (ratio of correct examples (`example accuracy`) is always evaluated)
metrics = {"f1_score": partial(f1_score, average="weighted", zero_division=0),
           "accuracy": accuracy_score,
           "precision": partial(precision_score, average="weighted", zero_division=0),
           "recall": partial(recall_score, average="weighted", zero_division=0)}

if model_params:
    print(f"Initializing {model_name} with the following parameters:")
    for key, value in model_params.items():
        print(f"    {key}: {value}")

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

elif model_name == "RandomTagger":
    from src.nn.models import RandomTagger

    model = RandomTagger(labels=bmes_tokenizer.meaningful_label_indices,
                         seed=seed)

else:
    raise NotImplementedError

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise NotImplementedError

if write_log:
    log_save_dir = f"./logs/{model_name}/{experiment_id}"
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
else:
    log_save_dir = None

if flow_control["train_model"]:
    print(f"Starting the training of {model_name} on {train_type} for {n_epochs} epochs...")

    if seed is not None:
        import random
        import numpy as np

        print(f"Setting seed: {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    model.to(device)

    best_valid_accuracy = training_cycle(experiment_id=experiment_id,
                                         model=model,
                                         train_loader=train_loader,
                                         validation_loader=valid_loader,
                                         optimizer=optimizer,
                                         device=device,
                                         clip=grad_clip,
                                         metrics=metrics,
                                         epochs=n_epochs,
                                         early_stopping=early_stopping,
                                         n_without_improvements=n_without_improvements,
                                         save_best=save_best,
                                         save_last=save_last,
                                         write_log=write_log,
                                         log_save_dir=log_save_dir)

    if write_log:
        with open(f"{log_save_dir}/config.json", "w") as file:
            info = {"experiment_id": experiment_id,
                    "train_params": train_params,
                    "model_params": model_params}
            file.write(json.dumps(info, indent=4))
else:
    best_valid_accuracy = None

if model_name == "RandomTagger":
    segmenter = RandomSegmenter(original_tokenizer=bmes_tokenizer,
                                bmes_tokenizer=bmes_tokenizer,
                                model=model)
else:
    segmenter = NeuralSegmenter(original_tokenizer=original_tokenizer,
                                bmes_tokenizer=bmes_tokenizer,
                                model=model,
                                device=device)

if flow_control["test_model"]:
    print(f"\nTesting {model_name}...")
    test_accuracy = testing_cycle(experiment_id=experiment_id,
                                  segmenter=segmenter,
                                  indices=test_indices,
                                  original=test_original,
                                  segmented=test_segmented,
                                  original_tokenizer=original_tokenizer,
                                  bmes_tokenizer=bmes_tokenizer,
                                  metrics=metrics,
                                  device=device,
                                  pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  max_len=MAX_LEN,
                                  batch_size=batch_size,
                                  write_log=write_log,
                                  log_save_dir=log_save_dir)
else:
    test_accuracy = None

print(f"Experiment on {model_name} successfully carried out")
print(f"Experiment ID: {experiment_id}")

# store timestamps and ids of successful experiments along with scores
if write_log:
    with open("./logs/successful_experiments.jsonl", "a") as file:
        info = {"experiment_id": experiment_id,
                "model_name": model_name,
                "train_type": train_type.lower(),
                "finished": datetime.now().isoformat(),
                "best_valid_accuracy": best_valid_accuracy,
                "test_accuracy": test_accuracy}
        file.write(json.dumps(info) + "\n")
