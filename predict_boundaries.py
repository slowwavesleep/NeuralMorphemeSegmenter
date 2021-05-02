# load the best model
# predict tags in batches
# write to file
from yaml import safe_load
import torch

from constants import PAD_INDEX, UNK_INDEX, TOKENIZERS_DIR
from src.utils.tokenizers import SymTokenizer
from src.nn.models import LstmCrfTagger
from src.utils.segmenters import NeuralSegmenter

experiment_id = "d24ee762d18640d0adc2cf7042b668c1"
model_path = f"models/LstmCrfTagger/{experiment_id}/best_model_state_dict.pth"
# replace with json config from log
model_config_path = "configs/lstmcrf_lemmas.yml"

with open(model_config_path) as file:
    config = safe_load(file)

model_params = config["model_params"]

original_tokenizer_path = f"{TOKENIZERS_DIR}/original.json"
bmes_tokenizer_path = f"{TOKENIZERS_DIR}/bmes.json"

original_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                                  unk_index=UNK_INDEX).vocab_from_file(original_tokenizer_path)

bmes_tokenizer = SymTokenizer(pad_index=PAD_INDEX,
                              unk_index=UNK_INDEX,
                              convert_to_bmes=True).vocab_from_file(bmes_tokenizer_path)


device = torch.device("cuda")

model = LstmCrfTagger(char_vocab_size=original_tokenizer.vocab_size,
                      tag_vocab_size=bmes_tokenizer.vocab_size,
                      padding_index=PAD_INDEX,
                      **model_params)

model.load_state_dict(torch.load(model_path))

model.to(device)

segmenter = NeuralSegmenter(original_tokenizer=original_tokenizer,
                            bmes_tokenizer=bmes_tokenizer,
                            model=model,
                            device=device,
                            seed=None)


print((segmenter.segment_example("примерять")))
