from functools import partial
import argparse

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from yaml import safe_load

from constants import UNK_INDEX, PAD_INDEX, MAX_LEN
from src.utils.etc import read_converted_data
from src.utils.tokenizers import SymTokenizer, bmes2sequence
from src.utils.datasets import BmesSegmentationDataset
from src.nn.training_process import training_cycle
from src.nn.testing_process import testing_cycle
from src.nn.models import CnnTagger, RandomTagger
from src.nn.layers import CnnEncoder
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter
from src.utils.tokenizers import SymTokenizer


# parser = argparse.ArgumentParser(description='Run model with specified settings.')
# parser.add_argument(dest='config', type=str, help='Path to config file.')
# args = parser.parse_args()


TRAIN_MODEL = True
TEST_MODEL = False
# TODO individual file names for different model types
WRITE_RESULTS = False
N_WITHOUT_IMPROVEMENTS = 4
TRAIN_TYPE = "lemmas"
# MODEL_NAME = "RandomTagger"
# MODEL_NAME = "LstmTagger"
# MODEL_NAME = "TransformerTagger"
MODEL_NAME = "LstmCrfTagger"

BATCH_SIZE = 128
HIDDEN_SIZE = 512
EMB_DIM = 512
SPATIAL_DROPOUT = 0.3
EPOCHS = 1
CLIP = 3.
LSTM_LAYERS = 3
LAYER_DROPOUT = 0.3
BIDIRECTIONAL = True

NUM_HEADS = 4
NUM_LAYERS = 3

LR = 1e-05

if MODEL_NAME == "RandomTagger":
    TRAIN_MODEL = False

if TRAIN_TYPE.lower() == "lemmas":
    from constants import CONVERTED_LEMMAS_PATHS

    RESULTS_PATH = "data/results/lemmas/"
    train_indices, train_original, train_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["train"])
    valid_indices, valid_original, valid_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["valid"])
    test_indices, test_original, test_segmented = read_converted_data(CONVERTED_LEMMAS_PATHS["test"])

elif TRAIN_TYPE.lower() == "forms":
    from constants import CONVERTED_FORMS_PATHS

    RESULTS_PATH = "data/results/forms/"
    train_indices, train_original, train_segmented = read_converted_data(CONVERTED_FORMS_PATHS["train"])
    valid_indices, valid_original, valid_segmented = read_converted_data(CONVERTED_FORMS_PATHS["valid"])
    test_indices, test_original, test_segmented = read_converted_data(CONVERTED_FORMS_PATHS["test"])
else:
    # TODO specify exception
    raise Exception

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

if MODEL_NAME == "LstmTagger":
    from src.nn.models import LstmTagger
    enc = LstmTagger(char_vocab_size=original_tokenizer.vocab_size,
                     tag_vocab_size=bmes_tokenizer.vocab_size,
                     emb_dim=EMB_DIM,
                     hidden_size=HIDDEN_SIZE,
                     spatial_dropout=SPATIAL_DROPOUT,
                     bidirectional=BIDIRECTIONAL,
                     padding_index=PAD_INDEX,
                     lstm_layers=LSTM_LAYERS,
                     layer_dropout=LAYER_DROPOUT)

elif MODEL_NAME == "TransformerTagger":
    from src.nn.models import TransformerTagger
    enc = TransformerTagger(char_vocab_size=original_tokenizer.vocab_size,
                            tag_vocab_size=bmes_tokenizer.vocab_size,
                            emb_dim=EMB_DIM,
                            n_heads=NUM_HEADS,
                            hidden_size=HIDDEN_SIZE,
                            dropout=SPATIAL_DROPOUT,
                            padding_index=PAD_INDEX,
                            max_len=MAX_LEN,
                            num_layers=NUM_LAYERS)

elif MODEL_NAME == "LstmCrfTagger":
    from src.nn.models import LstmCrfTagger
    enc = LstmCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                        tag_vocab_size=bmes_tokenizer.vocab_size,
                        emb_dim=EMB_DIM,
                        hidden_size=HIDDEN_SIZE,
                        spatial_dropout=SPATIAL_DROPOUT,
                        bidirectional=True,
                        padding_index=PAD_INDEX,
                        lstm_layers=LSTM_LAYERS,
                        layer_dropout=LAYER_DROPOUT)

elif MODEL_NAME == "RandomTagger":
    enc = None

else:
    raise Exception



# enc = CnnTagger(char_vocab_size=original_tokenizer.vocab_size,
#                 tag_vocab_size=bmes_tokenizer.vocab_size,
#                 emb_dim=EMB_DIM,
#                 num_filters=300,
#                 kernel_size=3,
#                 padding_index=PAD_INDEX)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1)

# for a, b, c, d in train_loader:
#     print(b.size())
#     out = enc(b.squeeze(0), c.squeeze(0))
#     print(out)
#     break


metrics = {"f1_score": partial(f1_score, average="weighted", zero_division=0),
           "accuracy": accuracy_score,
           "precision": partial(precision_score, average="weighted", zero_division=0),
           "recall": partial(recall_score, average="weighted", zero_division=0)}

device = torch.device('cuda')

if TRAIN_MODEL:

    optimizer = torch.optim.Adam(params=enc.parameters())
    enc.to(device)
    training_cycle(model=enc,
                   train_loader=train_loader,
                   validation_loader=valid_loader,
                   optimizer=optimizer,
                   device=device,
                   clip=CLIP,
                   metrics=metrics,
                   epochs=EPOCHS,
                   n_without_improvements=N_WITHOUT_IMPROVEMENTS)


if MODEL_NAME == "RandomTagger":
    segmenter = RandomSegmenter(original_tokenizer=bmes_tokenizer,
                                bmes_tokenizer=bmes_tokenizer,
                                labels=bmes_tokenizer.meaningful_label_indices)
else:
    segmenter = NeuralSegmenter(original_tokenizer=original_tokenizer,
                                bmes_tokenizer=bmes_tokenizer,
                                model=enc,
                                device=device,
                                seed=1)

if TEST_MODEL:
    testing_cycle(segmenter=segmenter,
                  indices=test_indices,
                  original=test_original,
                  segmented=test_segmented,
                  original_tokenizer=original_tokenizer,
                  bmes_tokenizer=bmes_tokenizer,
                  write_predictions=WRITE_RESULTS,
                  write_path=RESULTS_PATH,
                  metrics=metrics,
                  device=device,
                  pad_index=PAD_INDEX,
                  unk_index=UNK_INDEX,
                  max_len=MAX_LEN,
                  batch_size=BATCH_SIZE)
