# import timespan
from datetime import datetime
import numpy as np
from huggingface_hub import login
#-----------
from src.tools.config import pars_args


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def span_time():
    business_hours = [
        "9:00-17:00|mon-fri|*|*",  # is between 9 a.m. to 5 p.m. on Mon to Fri
        "!*|*|1|jan",  # not new years
        "!*|*|25|dec",  # not christmas
        "!*|thu|22-28|nov",  # not thanksgiving
    ]
    if timespan.match(business_hours, datetime.now()):
        print("The model can be trained!")
    else:
        print("The model cannot be trained! (Off Day)")


def load_config_file():
    args = pars_args()
    return args


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def access_HF(token:str):
    login(token = token)
