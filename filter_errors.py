import json
from pathlib import Path
import tarfile

PREDICTIONS_DIR = "./data/predictions"

folder = Path(PREDICTIONS_DIR)

pred_paths = folder.glob("**/*.jsonl")
compiled_errors_path = "error_analysis/errors.jsonl"
zipped_path = "error_analysis/errors.tar.gz"

with open(compiled_errors_path, "w") as file:
    pass

for path in pred_paths:
    experiment_id = path.parent.parts[-1]
    model_name = path.parent.parent.parts[-1]
    train_type = path.parent.parent.parent.parts[-1]

    with open(compiled_errors_path, "a") as w_file:
        with open(path) as file:
            for line in file:
                data = json.loads(line)
                if not data["match"] and not set("<UNK>").intersection(data["original"]):
                    error = {"experiment_id": experiment_id, "model_name": model_name, "train_type": train_type, **data}
                    w_file.write(json.dumps(error, ensure_ascii=False) + "\n")

with tarfile.open(zipped_path, "w:gz") as file:
    file.add(compiled_errors_path)
