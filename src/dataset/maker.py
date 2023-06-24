import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.examples = []
        for file_path in os.listdir(data_dir):
            if file_path.endswith(".json"):
                with open(os.path.join(data_dir, file_path), "r", encoding="utf8") as f:
                    data = json.load(f)
                    text = data["text"]
                    label = data["label"]
                    encoded = tokenizer(
                        text, padding="max_length", truncation=True, max_length=512
                    )
                    self.examples.append((encoded, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
