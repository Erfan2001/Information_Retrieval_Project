from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import random_split


class MyClassifier:
    def __int__(self, configs):
        ...

    def train(self):
        ...

    def inference(
        self, text_list: List[str], get_prob: bool = True
    ) -> List[Dict[str, Any]]:
        ...


def get_model(config):
    model_cfg = config.get("model")
    model = MyClassifier(config=model_cfg)
    return model


def load_model(path):
    ...
