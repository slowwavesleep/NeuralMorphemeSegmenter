from abc import ABC, abstractmethod


class AbstractTagger(ABC):

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def tag_example(self, example):
        ...

    @abstractmethod
    def tag_batch(self, example_batch):
        ...
