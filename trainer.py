from fire import Fire
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

# ------
from src.classifier.classifier import get_model
from src.dataset.dataset import prepare_dataset
from src.utils import span_time, load_config_file
from src.dataset.maker import MyDataset


# @span_time()
def trainer():
    config = load_config_file()

    # Adjust seed values
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Prepare Dataset
    dataset = prepare_dataset(config)

    # Model
    model = get_model(config)
    model.train()
    model.save_model_results()


if __name__ == "__main__":
    Fire(trainer)
