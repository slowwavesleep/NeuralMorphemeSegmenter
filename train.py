import json
from src.utils.tokenizers import SymTokenizer

original = []

with open("data/lemmas_train.jsonl") as file:
    for line in file:
        data = json.loads(line)
        original.append(data["original"])


tokenizer = SymTokenizer(1, 0, length=20).build_vocab(original)

print(tokenizer.decode([tokenizer.encode(el) for el in original][5]))
# print(tokenizer._index2sym)