import json
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import f1_score

from constants import UNK_INDEX, PAD_INDEX, CONVERTED_LEMMAS_PATHS, MAX_LEN
from src.utils.tokenizers import SymTokenizer
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import train, evaluate, training_cycle
from src.nn.layers import LstmEncoder
from src.nn.models import LstmTagger, LstmCrfTagger


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

labels = list(set(train_ds.bmes_tokenizer._index2sym.keys()) - {PAD_INDEX})

print(labels)

training_cycle(enc, train_loader, valid_loader, optimizer, device, 10., 2)

scores = []

for x, y in valid_loader:
    preds = enc.predict(x.to(device))
    preds = preds.flatten()
    y = y.cpu().numpy().flatten()
    scores.append(f1_score(y, preds, average="micro", labels=labels))

print(np.mean(scores))

# print(test_ds.original_tokenizer.decode(x[121, :].cpu().numpy()))
# print(test_ds.bmes_tokenizer.decode(preds[121, :]))
# print(test_ds.bmes_tokenizer.decode(y.cpu().numpy()[121, :]))
