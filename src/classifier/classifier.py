from typing import List, Dict, Any
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.dataset.maker import MyDataset

class MyClassifier:
    def __init__(self, configs):
        self.hps = configs

    def train(self):
        # Load the pre-trained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=self.hps.m
        )
        # Load your own dataset
        train_dataset = MyDataset(os.path.join("src\data", "train"), tokenizer)
        val_dataset = MyDataset(os.path.join("src\data", "valid"), tokenizer)
        # Set up the training arguments
        training_args = TrainingArguments(
            output_dir=self.hps.save_root,
            num_train_epochs=self.hps.n_epochs,
            per_device_train_batch_size=self.hps.batch_size,
            per_device_eval_batch_size=self.hps.batch_size,
            weight_decay=0.01,
            evaluate_during_training=True,
            logging_dir=self.hps.log_root,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.hps.lr,
            load_best_model_at_end=True,
            no_cuda=false,
        )
        # Set up the trainer
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=val_dataset,
        # )
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=val_dataset,
        #     tokenizer=tokenizer,
        #     # data_collator=data_collator,
        #     # compute_metrics=compute_metrics,
        # )

        # Train the model
        trainer.train()

    def inference(
        self, text_list: List[str], get_prob: bool = True
    ) -> List[Dict[str, Any]]:
        ...


def get_model(config):
    model_cfg = config.model
    match model_cfg:
        case "Transformer":
            model = MyClassifier(config)
        case _:
            model = MyClassifier(config)
    return model


def load_model(path):
    ...
