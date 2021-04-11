from functools import partial

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from constants import UNK_INDEX, PAD_INDEX, CONVERTED_LEMMAS_PATHS, MAX_LEN
from src.utils.etc import read_converted_lemmas
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle, testing_cycle
from src.nn.models import LstmCrfTagger, LstmTagger

train_indices, train_original, train_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["train"])
valid_indices, valid_original, valid_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["valid"])
test_indices, test_original, test_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["test"])

train_ds = BmesSegmentationDataset(indices=train_indices,
                                   original=train_original,
                                   segmented=train_segmented,
                                   sym_tokenizer=SymTokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN)

valid_ds = BmesSegmentationDataset(indices=valid_indices,
                                   original=valid_original,
                                   segmented=valid_segmented,
                                   sym_tokenizer=SymTokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN)

test_ds = BmesSegmentationDataset(indices=test_indices,
                                  original=test_original,
                                  segmented=test_segmented,
                                  sym_tokenizer=SymTokenizer,
                                  pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX,
                                  max_len=MAX_LEN)

HIDDEN_SIZE = 256
EMB_DIM = 32
SPATIAL_DROPOUT = 0.1
EPOCHS = 10
CLIP = 40.

# enc = LstmTagger(char_vocab_size=train_ds.original_tokenizer.vocab_size,
#                  tag_vocab_size=train_ds.bmes_tokenizer.vocab_size,
#                  emb_dim=EMB_DIM,
#                  hidden_size=HIDDEN_SIZE,
#                  spatial_dropout=SPATIAL_DROPOUT,
#                  bidirectional=True,
#                  padding_index=PAD_INDEX)

enc = LstmCrfTagger(char_vocab_size=train_ds.original_tokenizer.vocab_size,
                    tag_vocab_size=train_ds.bmes_tokenizer.vocab_size,
                    emb_dim=EMB_DIM,
                    hidden_size=HIDDEN_SIZE,
                    spatial_dropout=SPATIAL_DROPOUT,
                    bidirectional=True,
                    padding_index=PAD_INDEX)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)

optimizer = torch.optim.Adam(params=enc.parameters())
# optimizer = torch.optim.SGD(params=enc.parameters(), lr=1e-5)
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
                   clip=CLIP,
                   metrics={"f1_score": partial(f1_score, average="weighted"),
                            "accuracy": accuracy_score,
                            "precision": partial(precision_score, average="weighted"),
                            "recall": partial(recall_score, average="weighted")},
                   epochs=EPOCHS)

scores = []

ex_scores = []


for x, y, true_lens, _ in valid_loader:
    preds = enc.predict(x.to(device))
    x = x.cpu().numpy()
    for ex, pred, tr in zip(x, preds, true_lens):
        # pass
        print(bmes2sequence(valid_ds.original_tokenizer.decode(ex[:tr]), valid_ds.bmes_tokenizer.decode(pred[:tr])))
        # print(valid_ds.original_tokenizer.decode(ex))
        # print(valid_ds.bmes_tokenizer.decode(pred))
    break

    # y = y.cpu().numpy()

    # scores.append(evaluate_tokenwise_metric(y, preds, true_lens, partial(f1_score, average="weighted")))
    # ex_scores.append(evaluate_examplewise_accuracy(y, preds, true_lens))

# print(np.mean(scores))
# print(np.mean(ex_scores))

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
