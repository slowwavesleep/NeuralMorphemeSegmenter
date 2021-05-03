from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn
from torch import Tensor

from src.nn.models import RandomTagger
from src.utils.tokenizers import SymTokenizer, bmes2sequence


class AbstractSegmenter(ABC):

    @abstractmethod
    def tag_example(self, example: str):
        ...

    @abstractmethod
    def segment_example(self, example: str):
        ...

    @abstractmethod
    def tag_batch(self, example_batch: List[str]):
        ...

    @abstractmethod
    def segment_batch(self, example_batch: List[str]):
        ...


class RandomSegmenter(AbstractSegmenter):

    def __init__(self,
                 original_tokenizer: SymTokenizer,
                 bmes_tokenizer: SymTokenizer,
                 model: RandomTagger,
                 *,
                 sep: str = "|"):

        self.sep = sep
        self.original_tokenizer = original_tokenizer
        self.bmes_tokenizer = bmes_tokenizer
        self.tagger = model

    def tag_example(self, example: str):
        encoded = self.original_tokenizer.encode(example)
        encoded = Tensor(encoded).long().unsqueeze(0)

        prediction = self.tagger.predict(encoded).squeeze(0).tolist()
        prediction = self.bmes_tokenizer.decode(prediction)
        return prediction

    def segment_example(self, example: str):
        tags: str = self.tag_example(example)
        segmented = bmes2sequence(example, tags, sep=self.sep)
        return segmented

    def tag_batch(self, example_batch: List[str]):
        true_lengths = [len(example) for example in example_batch]
        max_length = max(true_lengths)

        processed_examples = []

        for example in example_batch:
            encoded = self.original_tokenizer.encode(example)
            padded = self.original_tokenizer.pad_or_clip(encoded, max_len=max_length)
            processed_examples.append(padded)

        processed_examples = Tensor(processed_examples).long()
        predictions = self.tagger.predict(processed_examples).tolist()

        return predictions, true_lengths

    def segment_batch(self, example_batch: List[str]):
        tags_batch: List[str]
        true_lengths: List[str]
        tags_batch, true_lengths = self.tag_batch(example_batch)

        segmented_batch = []

        for example, tags, true_length in zip(example_batch, tags_batch, true_lengths):
            decoded_tags = self.bmes_tokenizer.decode(tags)[:true_length]
            segmented = bmes2sequence(example, decoded_tags, sep=self.sep)
            segmented_batch.append(segmented)

        return segmented_batch


class NeuralSegmenter(AbstractSegmenter):

    def __init__(self,
                 original_tokenizer: SymTokenizer,
                 bmes_tokenizer: SymTokenizer,
                 model: nn.Module,
                 device: object,
                 seed: Optional[int] = None,
                 *,
                 sep: str = "|"):

        self.seed = seed
        self.device = device
        self.original_tokenizer = original_tokenizer
        self.bmes_tokenizer = bmes_tokenizer
        self.tagger = model
        self.sep = sep

    def tag_example(self, example):

        true_length = torch.tensor(len(example)).long().to(self.device).unsqueeze(0)

        encoded = self.original_tokenizer.encode(example)
        encoded = torch.tensor(encoded).long().unsqueeze(0).to(self.device)
        prediction = self.tagger.predict(encoded, true_length).squeeze(0).tolist()
        prediction = self.bmes_tokenizer.decode(prediction)
        return prediction

    def segment_example(self, example: str):
        tags: str = self.tag_example(example)
        segmented = bmes2sequence(example, tags, sep=self.sep)
        return segmented

    def tag_batch(self, example_batch: List[str]):
        true_lengths = [len(example) for example in example_batch]
        max_length = max(true_lengths)
        true_lengths = torch.tensor(true_lengths).long().to(self.device)

        processed_examples = []

        for example in example_batch:
            encoded = self.original_tokenizer.encode(example)
            padded = self.original_tokenizer.pad_or_clip(encoded, max_len=max_length)
            processed_examples.append(padded)

        processed_examples = torch.tensor(processed_examples).long().to(self.device)
        predictions = self.tagger.predict(processed_examples, true_lengths).tolist()

        return predictions, true_lengths

    def segment_batch(self, example_batch: List[str]):
        tags_batch: List[str]
        true_lengths: List[str]
        tags_batch, true_lengths = self.tag_batch(example_batch)

        segmented_batch = []

        for example, tags, true_length in zip(example_batch, tags_batch, true_lengths):
            decoded_tags = self.bmes_tokenizer.decode(tags)[:true_length]
            segmented = bmes2sequence(example, decoded_tags, sep=self.sep)
            segmented_batch.append(segmented)

        return segmented_batch
