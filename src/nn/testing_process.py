import json
from pathlib import Path
from typing import Union, List, Optional

from torch.utils.data import DataLoader

from src.utils.datasets import BmesSegmentationDataset
from src.utils.segmenters import RandomSegmenter, NeuralSegmenter
from src.nn.training_process import evaluate
from src.utils.tokenizers import sequence2bmes


def testing_cycle(segmenter: Union[RandomSegmenter, NeuralSegmenter],
                  indices: List[int],
                  original: List[str],
                  segmented: List[str],
                  original_tokenizer,
                  bmes_tokenizer,
                  write_path: str,
                  metrics: dict,
                  device: object,
                  pad_index: int,
                  unk_index: int,
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
                                   max_len=max_len)

    data_loader = DataLoader(data, batch_size=512)

    _, overall_scores, _ = evaluate(model=segmenter.tagger,
                                    loader=data_loader,
                                    device=device,
                                    metrics=metrics)

    message = "\n" + "*" * 50 + "\n"

    message += "Statistics on test data:\n"
    message += f"    examplewise_accuracy: {overall_scores['example-wise_accuracy']}\n"

    for name, score in overall_scores.items():
        if name != "example-wise_accuracy":
            message += f"    {name}: {score}\n"

    message += "*" * 50

    print(message)

    file_path = Path(write_path)
    file_path.mkdir(parents=True, exist_ok=True)
    file_path = file_path / "test.jsonl"

    # predicted_segmentation = segmenter.tag_batch(original)
    with open(file_path, "w") as file:
    #     for index, example, target, prediction in zip(indices, original, segmented, predicted_segmentation):
        count = 0
        total = 0
        for index, example, target in zip(indices, original, segmented):

            count += sequence2bmes(target) == segmenter.tag_example(example)
            total += 1
            output = {"index": index,
                      "original": example,
                      "segmented": sequence2bmes(target),
                      "predicted": segmenter.tag_example(example)}

            file.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(count / total)




