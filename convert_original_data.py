import re
import json

from constants import SEP_TOKEN, ORIGINAL_LEMMAS_PATHS


def remove_labels(labeled_segmented_example: str):
    segments = re.findall(r"[^:A-Zaz\/]+", labeled_segmented_example.strip("\n"))
    return SEP_TOKEN.join(segments)


def main():
    for key, value in ORIGINAL_LEMMAS_PATHS.items():
        with open(f"data/lemmas_{key}.jsonl", "w") as out_file:
            with open(value) as in_file:
                for line in in_file:
                    original, segmented = line.split("\t")
                    data = {"original": original,
                            "segmented": remove_labels(segmented)}
                    out_file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


