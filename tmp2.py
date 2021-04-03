from utils.data import IntentDataset
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np
from utils.data import batch_probs
from models import IntentBertModel
from typing import Union, Callable, Dict
import json
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
from functools import partial

VOCAB_FILE_NAME = "vocab.txt"
MAX_SEQ_LEN = 100
BATCH_SIZE = 18
EPOCHS = 10


def train(*,
          model,
          data_loader,
          criterion,
          optimizer,
          device,
          clip: float = 3.,
          last_n_losses: int = 500,
          verbose: bool = True):
    losses = []

    progress_bar = tqdm(total=len(data_loader), disable=not verbose, desc='Train')

    model.train()

    for batch in data_loader:
        x = batch["input_ids"].to(device)
        y = batch["intent_label"].to(device)

        pred = model(x, batch["attention_mask"].to(device), batch["token_type_ids"].to(device))

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 perplexity=np.exp(np.mean(losses[-last_n_losses:])))

        progress_bar.update()

    progress_bar.close()

    return losses


def evaluate(*,
             model: Module,
             data_loader: DataLoader,
             criterion: Module,
             device: object,
             stats: Union[dict, None] = None,
             metrics_fns: Union[None, Dict[str, Callable]] = None,
             last_n_losses: int = 500,
             verbose: bool = True):
    losses = []
    if not stats:
        stats = {
            "example_stats": defaultdict(lambda: []),
            "predictions": defaultdict(lambda: [])
        }

    if metrics_fns and not stats.get("metrics"):
        stats["metrics"] = defaultdict(lambda: [])

    progress_bar = tqdm(total=len(data_loader), disable=not verbose, desc='Evaluate')

    model.eval()

    true_labels_dict = dict()

    for batch in data_loader:

        identifiers = batch["ind"].to(device)
        texts = batch["input_ids"].to(device)
        true_labels = batch["intent_label"].to(device)

        with torch.no_grad():
            pred = model(texts, batch["attention_mask"].to(device), batch["token_type_ids"].to(device))

        loss = criterion(pred, true_labels)

        losses.append(loss.item())

        # true label probabilities
        true_probs = batch_probs(true_labels, pred)

        # labels with the highest probability
        predicted_labels = torch.argmax(pred, dim=1)

        for identifier, true_label_probability, highest_prob_label, true_label in zip(identifiers,
                                                                                      true_probs,
                                                                                      predicted_labels,
                                                                                      true_labels):
            stats["example_stats"][identifier.item()].append(true_label_probability.item())
            stats["predictions"][identifier.item()].append(highest_prob_label.item())
            true_labels_dict[identifier.item()] = true_label.item()

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 perplexity=np.exp(np.mean(losses[-last_n_losses:])))

        progress_bar.update()

    progress_bar.close()

    stats["true_labels"] = true_labels_dict
    if metrics_fns:
        model_preds = []
        ground_truth_labels = []
        for key in stats["true_labels"].keys() & stats["predictions"].keys():
            model_preds.append(stats["predictions"][key][-1])
            ground_truth_labels.append(stats["true_labels"][key])

        for name, fn in metrics_fns.items():
            score = fn(ground_truth_labels, model_preds)
            stats["metrics"][name].append(score)

        for key, value in stats["metrics"].items():
            print("\n")
            print(f"{key}: {value[-1]}")
            print("\n")

    return losses, stats


tokenizer = BertWordPieceTokenizer(VOCAB_FILE_NAME,
                                   lowercase=True)

tokenizer.enable_padding(max_length=MAX_SEQ_LEN)

train_ds = IntentDataset(data_path="data/banking/train_10.csv",
                         tokenizer=tokenizer,
                         max_seq_length=MAX_SEQ_LEN,
                         vocab_file_name=VOCAB_FILE_NAME)

val_ds = IntentDataset(data_path="data/banking/val.csv",
                       tokenizer=tokenizer,
                       max_seq_length=MAX_SEQ_LEN,
                       vocab_file_name=VOCAB_FILE_NAME)

test_ds = IntentDataset(data_path="data/banking/test.csv",
                        tokenizer=tokenizer,
                        max_seq_length=MAX_SEQ_LEN,
                        vocab_file_name=VOCAB_FILE_NAME)

train_loader = DataLoader(dataset=train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(dataset=val_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

test_loader = DataLoader(dataset=test_ds,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

device = torch.device('cuda')

model = IntentBertModel("bert-base-uncased",
                        0.3,
                        len(train_ds.intent_label_to_idx))

model.to(device)

optimizer = AdamW(params=model.parameters(), lr=5e-5, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss()


def training_cycle(*,
                   model,
                   train_loader,
                   val_loader,
                   test_loader,
                   criterion,
                   optimizer,
                   device,
                   base_path,
                   epochs):

    train_stats = None
    val_stats = None
    test_stats = None

    for n_epoch in range(1, epochs + 1):
        train(model=model,
              data_loader=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              device=device)

        _, train_stats = evaluate(model=model,
                                  data_loader=train_loader,
                                  criterion=criterion,
                                  device=device,
                                  stats=train_stats,
                                  metrics_fns={"accuracy": accuracy_score,
                                               "weighted_f1": partial(f1_score, average="weighted")})

        _, val_stats = evaluate(model=model,
                                data_loader=val_loader,
                                criterion=criterion,
                                device=device,
                                stats=val_stats,
                                metrics_fns={"accuracy": accuracy_score,
                                             "weighted_f1": partial(f1_score, average="weighted")})

        _, test_stats = evaluate(model=model,
                                 data_loader=test_loader,
                                 criterion=criterion,
                                 device=device,
                                 stats=test_stats,
                                 metrics_fns={"accuracy": accuracy_score,
                                              "weighted_f1": partial(f1_score, average="weighted")})

    with open(base_path + "train_stats.json", "w") as file:
        file.write(json.dumps(train_stats))

    with open(base_path + "val_stats.json", "w") as file:
        file.write(json.dumps(val_stats))

    with open(base_path + "test_stats.json", "w") as file:
        file.write(json.dumps(test_stats))


training_cycle(model=model,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               criterion=criterion,
               optimizer=optimizer,
               device=device,
               base_path="evaluation/banking/few_shot/",
               epochs=EPOCHS)