import json
from functools import partial

import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from constants import UNK_INDEX, PAD_INDEX, CONVERTED_LEMMAS_PATHS, MAX_LEN
from src.utils.metrics import evaluate_tokenwise_metric, evaluate_examplewise_accuracy
from src.utils.tokenizers import SymTokenizer
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle
from src.nn.models import LstmTagger, RandomTagger


def read_converted_lemmas(path: str):
    original = []
    segmented = []
    with open(path) as file:
        for line in file:
            data = json.loads(line)
            original.append(data["original"])
            segmented.append(data["segmented"])

    return original, segmented


train_original, train_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["train"])

test_original, test_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["test"])

# word_tokenizer = SymTokenizer(pad_index=PAD_INDEX, unk_index=UNK_INDEX).build_vocab(original)
# bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX, unk_index=UNK_INDEX, convert_to_bmes=True).build_vocab(segmented)
#
# print(word_tokenizer.pad_or_clip(word_tokenizer.encode(original[1]), 15))
# print(segmented[1])
# print(bmes_tokenizer.pad_or_clip(bmes_tokenizer.encode(segmented[1]), 15))

train_ds = BmesSegmentationDataset(original=train_original,
                                   segmented=train_segmented,
                                   sym_tokenizer=SymTokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN)

test_ds = BmesSegmentationDataset(original=test_original,
                                  segmented=test_segmented,
                                  sym_tokenizer=SymTokenizer,
                                  pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  max_len=MAX_LEN)

enc = LstmTagger(char_vocab_size=train_ds.original_tokenizer.vocab_size,
                 tag_vocab_size=train_ds.bmes_tokenizer.vocab_size,
                 emb_dim=256,
                 hidden_size=256,
                 bidirectional=True,
                 padding_index=PAD_INDEX)

# enc = LstmCrfTagger(char_vocab_size=train_ds.original_tokenizer.vocab_size,
#                     tag_vocab_size=train_ds.bmes_tokenizer.vocab_size,
#                     emb_dim=256,
#                     hidden_size=256,
#                     bidirectional=True,
#                     padding_index=PAD_INDEX)

# print(train_ds[0][0])
# print(train_ds[0][0].size())
#
# print(enc.forward(train_ds[0][0].unsqueeze(0), train_ds[0][1].unsqueeze(0)))

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
valid_loader = DataLoader(test_ds, batch_size=1024)

optimizer = torch.optim.Adam(params=enc.parameters())

device = torch.device('cuda')

enc.to(device)

# print(train_ds.bmes_tokenizer._index2sym)
#

if True:
    training_cycle(model=enc,
                   train_loader=train_loader,
                   validation_loader=valid_loader,
                   optimizer=optimizer,
                   device=device,
                   clip=3.,
                   metrics={"f1_score": partial(f1_score, average="weighted"),
                            "accuracy": accuracy_score,
                            "precision": partial(precision_score, average="weighted"),
                            "recall": partial(recall_score, average="weighted")},
                   epochs=3)

scores = []

ex_scores = []

for x, y, true_lens in valid_loader:
    preds = enc.predict(x.to(device))
    y = y.cpu().numpy()

    scores.append(evaluate_tokenwise_metric(y, preds, true_lens, partial(f1_score, average="weighted")))

print(np.mean(scores))

# random_tagger = RandomTagger(seed=100,
#                              labels=train_ds.bmes_tokenizer.meaningful_label_indices)
#
# for x, y, true_lens in valid_loader:
#     preds = random_tagger.predict(x)
#     y = y.cpu().numpy()
#     scores.append(evaluate_tokenwise_metric(y, preds, true_lens, partial(f1_score, average="weighted")))
#     ex_scores.append(evaluate_examplewise_accuracy(y, preds, true_lens))
#
# print(np.mean(scores))
# print(np.mean(ex_scores))