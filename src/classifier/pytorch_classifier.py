import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class Pytorch_Classifier(nn.Module):
    def __init__(self, num_classes, model_type,configs):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(configs.drop_out)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        dropout_output = self.dropout(out[1])
        linear_output = self.linear(dropout_output)
        probas = self.softmax(linear_output)
        return probas
