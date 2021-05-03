import os
import json
from collections import defaultdict
from typing import Union, List, Optional

from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np

from src.utils.datasets import BmesSegmentationDataset
from src.utils.metrics import evaluate_batch
from src.nn.training_process import evaluate
from src.utils.tokenizers import SymTokenizer


def testing_cycle(experiment_id: str,
                  model: Module,
                  indices: List[int],
                  original: List[str],
                  segmented: List[str],
                  original_tokenizer: SymTokenizer,
                  bmes_tokenizer: SymTokenizer,
                  metrics: dict,
                  device: object,
                  pad_index: int,
                  unk_index: int,
                  batch_size: int,
                  max_len: Optional[int] = None,
                  write_log: bool = True,
                  log_save_dir: Optional[str] = None) -> float:

    model_name = model.__class__.__name__

    if write_log and not log_save_dir:
        log_save_dir = f"./logs/{model_name}/{experiment_id}"
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

    if not max_len:
        max_len = max([len(example)] for example in original)

    data = BmesSegmentationDataset(indices=indices,
                                   original=original,
                                   segmented=segmented,
                                   original_tokenizer=original_tokenizer,
                                   bmes_tokenizer=bmes_tokenizer,
                                   pad_index=pad_index,
                                   unk_index=unk_index,
                                   max_len=max_len,
                                   batch_size=batch_size)

    data_loader = DataLoader(data, batch_size=1)

    if model_name == "RandomTagger":

        overall_scores = evaluate_random_baseline(data_loader, device, metrics, model)

    else:
        _, overall_scores, _ = evaluate(model=model,
                                        loader=data_loader,
                                        device=device,
                                        metrics=metrics)

    message = "\n" + "*" * 50 + "\n"

    message += "Statistics on test data:\n"
    message += f"    example_accuracy: {overall_scores['example_accuracy']:.6f}\n"

    for name, score in overall_scores.items():
        if name != "example_accuracy":
            message += f"    {name}: {score:.6f}\n"

    message += "*" * 50

    print(message)

    if write_log:
        with open(f"{log_save_dir}/test_log.json", "w") as file:
            info = overall_scores
            file.write(json.dumps(info, indent=4))

    test_accuracy = overall_scores["example_accuracy"]

    return test_accuracy


def write_predictions_to_file():
    # TODO move write predictions here
    pass
    # TODO refactor to use batches
    # predicted_segmentation = segmenter.tag_batch(original)
    # print(predicted_segmentation)
    # if write_predictions and write_path:
    #     print("Writing results to file...")
    #     with open(file_path, "w") as file:
    #     #     for index, example, target, prediction in zip(indices, original, segmented, predicted_segmentation):
    #         for index, example, target in zip(indices, original, segmented):
    #             output = {"index": index,
    #                       "original": example,
    #                       "segmented": sequence2bmes(target),
    #                       "predicted": segmenter.tag_example(example),
    #                       "correct": sequence2bmes(target) == segmenter.tag_example(example)}
    #
    #             file.write(json.dumps(output, ensure_ascii=False) + "\n")
    # elif write_predictions and not write_path:
    #     print("No write path was specified!")
    # else:
    #     pass


def evaluate_random_baseline(data_loader: DataLoader,
                             device: object,
                             metrics: dict,
                             model: Module) -> Optional[dict]:

    batch_scores = defaultdict(lambda: [])
    for index_seq, encoder_seq, target_seq, true_lens in data_loader:
        index_seq = index_seq.squeeze(0)
        encoder_seq = encoder_seq.to(device).squeeze(0)
        target_seq = target_seq.to(device).squeeze(0)
        true_lens = true_lens.to(device).squeeze(0)

        preds = model.predict(encoder_seq)

        batch_scores = evaluate_batch(y_true=target_seq,
                                      y_pred=preds,
                                      metrics=metrics,
                                      batch_scores=batch_scores,
                                      true_lengths=true_lens)
    if batch_scores:
        overall_scores = {name: np.mean(values) for name, values in batch_scores.items()}
        return overall_scores



