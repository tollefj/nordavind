import jsonlines
from datasets import Dataset, DatasetDict
from prompts import create_conversation

paths = {
    "train": "../data/train_prompted.jsonl",
    "test": "../data/test_prompted.jsonl",
}


def get_data(split="train"):
    data = None
    with jsonlines.open(paths[split], "r") as reader:
        data = list(reader)

    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        create_conversation,
        remove_columns=dataset.features,
        batched=False,
    )
