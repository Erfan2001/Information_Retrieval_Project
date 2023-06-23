import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for filename in os.listdir(root_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(root_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    text = data["text"]
                    label = data["label"]
                    self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, label = self.samples[index]
        # Convert text to a tensor using your preferred text encoding method
        # For example, you can use a tokenizer from the `transformers` library
        input_ids = ...
        attention_mask = ...
        token_type_ids = ...
        text_tensor = torch.tensor([input_ids, attention_mask, token_type_ids])
        # Convert label to a tensor
        label_tensor = torch.tensor(label)
        return text_tensor, label_tensor
