from functools import partial

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from constants import UNK_INDEX, PAD_INDEX, CONVERTED_LEMMAS_PATHS, MAX_LEN
from src.utils.etc import read_converted_lemmas
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle
from src.nn.models import LstmCrfTagger, LstmTagger, CnnTagger, RandomTagger
from src.nn.layers import CnnEncoder
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter

train_indices, train_original, train_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["train"])
valid_indices, valid_original, valid_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["valid"])
test_indices, test_original, test_segmented = read_converted_lemmas(CONVERTED_LEMMAS_PATHS["test"])

# TODO refactor tokenizers
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
EPOCHS = 2
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

# enc = CnnTagger(char_vocab_size=train_ds.original_tokenizer.vocab_size,
#                 tag_vocab_size=train_ds.bmes_tokenizer.vocab_size,
#                 emb_dim=EMB_DIM,
#                 num_filters=300,
#                 kernel_size=3,
#                 padding_index=PAD_INDEX)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)

optimizer = torch.optim.Adam(params=enc.parameters())
device = torch.device('cuda')

enc.to(device)

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

# segmenter = RandomSegmenter(original_tokenizer=train_ds.bmes_tokenizer,
#                             bmes_tokenizer=train_ds.bmes_tokenizer,
#                             labels=train_ds.bmes_tokenizer.meaningful_label_indices)

segmenter = NeuralSegmenter(original_tokenizer=train_ds.bmes_tokenizer,
                            bmes_tokenizer=train_ds.bmes_tokenizer,
                            model=enc,
                            device=device,
                            seed=1)

