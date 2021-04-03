from typing import List, Dict, Tuple, Union
import math
import zipfile
import csv
import json
import pickle
import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tokenize import wordpunct_tokenize
# from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_intent_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            reader = csv.reader(open(data_path))
            next(reader, None)
            for utt, intent in tqdm(reader):
                encoded = tokenizer.encode(utt)
                # print("*" * 25)
                # print(utt)
                # print(encoded.ids)
                # print(encoded.tokens)
                # print("*" * 25)

                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def batch_probs(true_indices, predictions):
    predictions = torch.softmax(predictions, 1)
    true_probs = torch.gather(predictions, 1, true_indices.unsqueeze(-1)).squeeze(-1)
    return true_probs


def basic_tokenize(text: str, vocab: Dict[str, int], *, unk_index: int = 1, lower: bool = True) -> List[int]:
    if lower:
        text = text.lower()
    tokenized = wordpunct_tokenize(text)
    return [vocab.get(token, unk_index) for token in tokenized]


def vectorize(tokenized_text: List[int], embeddings: np.ndarray) -> List[np.ndarray]:
    return [embeddings[token] for token in tokenized_text]


def index_categories(labels: List[str]) -> Dict[str, int]:
    cat_dict = dict()
    cat_dict["UNK"] = len(cat_dict)
    labels = set(labels)
    for index, label in enumerate(labels):
        cat_dict[label] = index
    return cat_dict


def load_embeddings(zip_path: str,
                    filename: str,
                    *,
                    max_words: int,
                    pad_token: str = "<PAD>",
                    unk_token: str = "<UNK>",
                    vocab_size: Union[int, None] = None,
                    embedding_dim: Union[int, None] = None) -> Tuple[Dict[str, int], np.ndarray]:
    vocab = dict()
    embeddings = list()

    with zipfile.ZipFile(zip_path) as zipped_file:
        with zipped_file.open(filename) as file_object:

            if not vocab_size and not embedding_dim:
                vocab_size, embedding_dim = file_object.readline().decode("utf-8").strip().split()

                vocab_size = int(vocab_size)
                embedding_dim = int(embedding_dim)

            if max_words:
                max_words = vocab_size if max_words <= 0 else max_words

            vocab[pad_token] = len(vocab)
            vocab[unk_token] = len(vocab)
            embeddings.append(np.zeros(embedding_dim))

            for line in file_object:
                parts = line.decode('utf-8').strip().split()

                token = ' '.join(parts[:-embedding_dim]).lower()

                if token in vocab:
                    continue

                word_vector = np.array(list(map(float, parts[-embedding_dim:])))

                vocab[token] = len(vocab)
                embeddings.append(word_vector)

                if len(vocab) == max_words:
                    break

    embeddings = np.stack(embeddings)

    return vocab, embeddings


def sort_data(identifiers: List, texts: List, labels: List) -> Tuple[List, List, List]:
    identifiers, texts, labels = zip(*sorted(zip(identifiers, texts, labels), key=lambda x: len(x[1])))
    return identifiers, texts, labels


def make_batches(identifiers: List[int],
                 tokenized_texts: List[List[int]],
                 labels: List[int],
                 batch_size: int) -> Tuple[list, List[List[List[int]]], List[List[int]]]:

    assert len(tokenized_texts) == len(labels) == len(identifiers)

    identifier_batches = []
    text_batches = []
    label_batches = []

    for i_batch in range(math.ceil(len(tokenized_texts) / batch_size)):
        identifier_batches.append(identifiers[i_batch * batch_size:(i_batch + 1) * batch_size])
        text_batches.append(tokenized_texts[i_batch * batch_size:(i_batch + 1) * batch_size])
        label_batches.append(labels[i_batch * batch_size:(i_batch + 1) * batch_size])

    return identifier_batches, text_batches, label_batches


class SequenceBucketingData(Dataset):

    def __init__(self,
                 identifiers: List[int],
                 texts: List[List[int]],
                 labels: List[int],
                 max_len: int,
                 batch_size: int,
                 pad_index: int):

        self.batch_size = batch_size

        identifiers, texts, labels = sort_data(identifiers, texts, labels)

        self.identifiers, self.texts, self.labels = make_batches(identifiers, texts, labels, self.batch_size)

        self.max_len = max_len

        self.pad_index = pad_index

    def __len__(self):
        return len(self.texts)

    def prepare_sample(self, sequence: List[int], max_len: int):
        sequence = sequence[:max_len]

        pads = [self.pad_index] * (max_len - len(sequence))

        sequence += pads

        return sequence

    def __getitem__(self, index: int):
        identifier_batch = self.identifiers[index]
        text_batch = self.texts[index]
        label_batch = self.labels[index]

        max_len = min([self.max_len, max([len(sample) for sample in text_batch])])

        batch_id = []
        batch_x = []
        batch_y = []

        for index, sample in enumerate(text_batch):
            identifier = identifier_batch[index]
            x = self.prepare_sample(sample, max_len)
            y = label_batch[index]
            batch_id.append(identifier)
            batch_x.append(x)
            batch_y.append(y)

        batch_id = torch.tensor(batch_id).long()
        batch_x = torch.tensor(batch_x).long()
        batch_y = torch.tensor(batch_y).long()

        return batch_id, batch_x, batch_y