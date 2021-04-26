import json
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional

from torch.utils.data import DataLoader
import numpy as np

from src.utils.datasets import BmesSegmentationDataset
from src.utils.metrics import evaluate_batch
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter
from src.nn.training_process import evaluate
from src.utils.tokenizers import sequence2bmes


def testing_cycle(segmenter: Union[RandomSegmenter, NeuralSegmenter],
                  indices: List[int],
                  original: List[str],
                  segmented: List[str],
                  original_tokenizer,
                  bmes_tokenizer,
                  metrics: dict,
                  device: object,
                  pad_index: int,
                  unk_index: int,
                  batch_size: int,
                  write_predictions: bool = False,
                  write_path: Optional[str] = None,
                  max_len: Optional[int] = None):
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

    if segmenter.__class__.__name__ == "RandomSegmenter":

        overall_scores = evaluate_random_baseline(data_loader, device, metrics, segmenter)

    else:
        _, overall_scores, _ = evaluate(model=segmenter.tagger,
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

    file_path = Path(write_path)
    file_path.mkdir(parents=True, exist_ok=True)
    file_path = file_path / "test.jsonl"

    # TODO refactor to use batches
    # predicted_segmentation = segmenter.tag_batch(original)
    if write_predictions and write_path:
        with open(file_path, "w") as file:
        #     for index, example, target, prediction in zip(indices, original, segmented, predicted_segmentation):
            for index, example, target in zip(indices, original, segmented):
                output = {"index": index,
                          "original": example,
                          "segmented": sequence2bmes(target),
                          "predicted": segmenter.tag_example(example),
                          "correct": sequence2bmes(target) == segmenter.tag_example(example)}

                file.write(json.dumps(output, ensure_ascii=False) + "\n")
    elif write_predictions and not write_path:
        print("No write path was specified!")
    else:
        pass


def evaluate_random_baseline(data_loader: DataLoader,
                             device: object,
                             metrics: dict,
                             segmenter: RandomSegmenter) -> Optional[dict]:

    batch_scores = defaultdict(lambda: [])
    for index_seq, encoder_seq, target_seq, true_lens in data_loader:
        index_seq = index_seq.squeeze(0)
        encoder_seq = encoder_seq.to(device).squeeze(0)
        target_seq = target_seq.to(device).squeeze(0)
        true_lens = true_lens.to(device).squeeze(0)

        preds = segmenter.tagger.predict(encoder_seq)

        batch_scores = evaluate_batch(y_true=target_seq,
                                      y_pred=preds,
                                      metrics=metrics,
                                      batch_scores=batch_scores,
                                      true_lengths=true_lens)
    if batch_scores:
        overall_scores = {name: np.mean(values) for name, values in batch_scores.items()}
        return overall_scores



