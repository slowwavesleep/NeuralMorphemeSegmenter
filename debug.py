from functools import partial

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from constants import UNK_INDEX, PAD_INDEX, CONVERTED_LEMMAS_PATHS, MAX_LEN, CONVERTED_FORMS_PATHS
from src.utils.etc import read_converted_data
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle
from src.nn.testing_process import testing_cycle
from src.nn.models import LstmCrfTagger, LstmTagger, CnnTagger, RandomTagger
from src.nn.layers import CnnEncoder
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter
from src.utils.tokenizers import SymTokenizer

# TODO 1) fix metrics 2) save models

TRAIN_MODEL = True
TEST_MODEL = True

BATCH_SIZE = 256
HIDDEN_SIZE = 1024
EMB_DIM = 32
SPATIAL_DROPOUT = 0.1
EPOCHS = 5
CLIP = 5.

# RESULTS_PATH = "data/results/lemmas/"
# train_indices, train_original, train_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["train"])
# valid_indices, valid_original, valid_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["valid"])
# test_indices, test_original, test_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["test"])

RESULTS_PATH = "data/results/forms/"
train_indices, train_original, train_segmented = read_converted_data(CONVERTED_FORMS_PATHS["train"])
valid_indices, valid_original, valid_segmented = read_converted_data(CONVERTED_FORMS_PATHS["valid"])
test_indices, test_original, test_segmented = read_converted_data(CONVERTED_FORMS_PATHS["test"])


original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX).build_vocab(train_original)

bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                              unk_index=UNK_INDEX,
                              convert_to_bmes=True).build_vocab(train_segmented)

train_ds = BmesSegmentationDataset(indices=train_indices,
                                   original=train_original,
                                   segmented=train_segmented,
                                   original_tokenizer=original_tokenizer,
                                   bmes_tokenizer=bmes_tokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN,
                                   batch_size=BATCH_SIZE)

valid_ds = BmesSegmentationDataset(indices=valid_indices,
                                   original=valid_original,
                                   segmented=valid_segmented,
                                   original_tokenizer=original_tokenizer,
                                   bmes_tokenizer=bmes_tokenizer,
                                   pad_index=PAD_INDEX,
                                   unk_index=UNK_INDEX,
                                   max_len=MAX_LEN,
                                   batch_size=BATCH_SIZE)

enc = LstmTagger(char_vocab_size=original_tokenizer.vocab_size,
                 tag_vocab_size=bmes_tokenizer.vocab_size,
                 emb_dim=EMB_DIM,
                 hidden_size=HIDDEN_SIZE,
                 spatial_dropout=SPATIAL_DROPOUT,
                 bidirectional=True,
                 padding_index=PAD_INDEX)

# enc = LstmCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
#                     tag_vocab_size=bmes_tokenizer.vocab_size,
#                     emb_dim=EMB_DIM,
#                     hidden_size=HIDDEN_SIZE,
#                     spatial_dropout=SPATIAL_DROPOUT,
#                     bidirectional=True,
#                     padding_index=PAD_INDEX)

# enc = CnnTagger(char_vocab_size=original_tokenizer.vocab_size,
#                 tag_vocab_size=bmes_tokenizer.vocab_size,
#                 emb_dim=EMB_DIM,
#                 num_filters=300,
#                 kernel_size=3,
#                 padding_index=PAD_INDEX)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1)


optimizer = torch.optim.Adam(params=enc.parameters())
device = torch.device('cuda')

enc.to(device)

metrics = {"f1_score": partial(f1_score, average="weighted"),
           "accuracy": accuracy_score,
           "precision": partial(precision_score, average="weighted"),
           "recall": partial(recall_score, average="weighted")}

if TRAIN_MODEL:
    training_cycle(model=enc,
                   train_loader=train_loader,
                   validation_loader=valid_loader,
                   optimizer=optimizer,
                   device=device,
                   clip=CLIP,
                   metrics=metrics,
                   epochs=EPOCHS)

# segmenter = RandomSegmenter(original_tokenizer=bmes_tokenizer,
#                             bmes_tokenizer=bmes_tokenizer,
#                             labels=bmes_tokenizer.meaningful_label_indices)

segmenter = NeuralSegmenter(original_tokenizer=original_tokenizer,
                            bmes_tokenizer=bmes_tokenizer,
                            model=enc,
                            device=device,
                            seed=1)

# for indices, x, y, lens in valid_loader:
#     print(indices)
#     print(x)
#     print(y)
#     print(lens)
#     print(train_ds.original_tokenizer.decode(x.squeeze(0)[-1].detach().numpy()))
#     print(valid_ds.original_tokenizer.decode(x.squeeze(0)[-1].detach().numpy()))
#     print(train_ds.bmes_tokenizer.decode(y.squeeze(0)[-1].detach().numpy()))
#     break



if TEST_MODEL:
    testing_cycle(segmenter=segmenter,
                  indices=test_indices,
                  original=test_original,
                  segmented=test_segmented,
                  original_tokenizer=original_tokenizer,
                  bmes_tokenizer=bmes_tokenizer,
                  write_path=RESULTS_PATH,
                  metrics=metrics,
                  device=device,
                  pad_index=PAD_INDEX,
                  unk_index=UNK_INDEX,
                  max_len=MAX_LEN,
                  batch_size=BATCH_SIZE)

