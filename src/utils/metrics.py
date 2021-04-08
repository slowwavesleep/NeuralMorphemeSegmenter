from sklearn.metrics import f1_score

from src.utils.etc import flatten_list, remove_pads


def evaluate_metric_padded(y_true, y_pred, true_lengths):
    y_true = flatten_list(remove_pads(y_true, true_lengths))
    y_pred = flatten_list(remove_pads(y_pred, true_lengths))
    return f1_score(y_true, y_pred, average="weighted")