from typing import Callable, Iterable

from src.utils.etc import flatten_list, remove_pads


def evaluate_tokenwise_metric(y_true, y_pred, true_lengths, scoring_fn: Callable) -> float:
    y_true = flatten_list(remove_pads(y_true, true_lengths))
    y_pred = flatten_list(remove_pads(y_pred, true_lengths))
    return scoring_fn(y_true, y_pred)


def evaluate_examplewise_accuracy(y_true, y_pred, true_lengths) -> float:
    y_true = remove_pads(y_true, true_lengths)
    y_pred = remove_pads(y_pred, true_lengths)

    total = 0
    correct = 0

    for label, prediction in zip(y_true, y_pred):
        if label == prediction:
            correct += 1
        total += 1

    if total:
        return correct/total
    else:
        return 0.
