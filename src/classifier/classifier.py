from typing import List, Dict, Any
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from torch import nn
import evaluate
#--------
from src.dataset.maker import MyDataset
from src.dataset.dataset import load_dataset_HF,label2id,id2label
from utils import preprocess_function,access_HF
from tools.evaluation import compute_metrics
from src.dataset.pytorch_dataset.index import flat_files,pytorch_dataset
from src.classifier.pytorch_classifier import TextClassificationDataset,Pytorch_Classifier

class MyClassifier:
    def __init__(self, configs):
        self.hps = configs

    def train(self):
        """
            Train model by recognizing a specific model
            We have four different models:
            1) HF_Transformers => distilbert-base-uncased
            2) HF_Transformers => bert-base-multilingual-cased
            3) Pure Pytorch => distilbert-base-uncased + Other Layers
            4) Pure Pytorch => bert-base-multilingual-cased + Other Layers
        """
        # id2label = {0: "Beinolmelal", 1: "Eghtesadi", 2: "Ejtemaee", 3: "ElmiVaDaneshghai", 4: "FarhangHonarVaResane", 5: "Siasi", 6: "Varzeshi"}
        # label2id = {"Beinolmelal": 0, "Eghtesadi": 1, "Ejtemaee":2, "ElmiVaDaneshghai":3, "FarhangHonarVaResane":4, "Siasi":5, "Varzeshi":6}
        match self.hps.model:
            case "bert":
                loaded_dataset = load_dataset_HF(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                tokenized_dataset = x.map(preprocess_function, batched=True)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                id2label_value=id2label(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                label2id_value=label2id(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                            num_labels=7, id2label=id2label, label2id=label2id)
                if(self.hps.token):
                    access_HF(self.hps.token)
                training_args = TrainingArguments(
                                output_dir=self.hps.save_root,
                                learning_rate= self.hps.lr or 2e-5,
                                per_device_train_batch_size=self.hps.batch_size or 16,
                                per_device_eval_batch_size= self.hps.batch_size or 16,
                                num_train_epochs=self.hps.n_epochs or 2,
                                evaluation_strategy="epoch",
                                save_strategy="epoch",
                                load_best_model_at_end=True,
                                push_to_hub=True,
                                )
                trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                )
                trainer.train()

            case "multilingual":
                loaded_dataset = load_dataset_HF(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                tokenized_dataset = x.map(preprocess_function, batched=True)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                id2label_value=id2label(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                label2id_value=label2id(os.path.join(self.hps.cache_dir,'tokenized_dataset'))
                model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                            num_labels=7, id2label=id2label, label2id=label2id)
                if(self.hps.token):
                    access_HF(self.hps.token)
                training_args = TrainingArguments(
                                output_dir=self.hps.save_root,
                                learning_rate= self.hps.lr or 2e-5,
                                per_device_train_batch_size=self.hps.batch_size or 16,
                                per_device_eval_batch_size= self.hps.batch_size or 16,
                                num_train_epochs=self.hps.n_epochs or 2,
                                evaluation_strategy="epoch",
                                save_strategy="epoch",
                                load_best_model_at_end=True,
                                push_to_hub=True,
                                )
                trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                )
                trainer.train()
                
            case "pytorch_bert":
                label2id_value=label2id(os.path.join(self.hps.cache_dir,'normalized_dataset'))
                all_pytorch_data = pytorch_dataset(label2id_value,os.path.join(self.hps.cache_dir,'normalized_dataset'))
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                dataset = TextClassificationDataset(total_list, tokenizer, max_length=128)
                dataloader = DataLoader(dataset, batch_size=self.hps.batch_size or 16)
                if self.hps.cuda:
                    device = torch.device("cuda:0")
                    logger.info("[INFO] Use cuda")

                else:
                    device = torch.device("cpu")
                    logger.info("[INFO] Use CPU")
                model = Pytorch_Classifier(self.hps.m,'bert-base-uncased',self.hps).to(device)
                loss_function = nn.NLLLoss()
                optimizer = AdamW(model.parameters(), lr=self.hps.lr or 2e-5)
                accuracy = evaluate.load("accuracy")
                for epoch in range(self.hps,n_epochs):
                    total_loss = 0
                    total_correct = 0
                    total_samples = 0
                    model.train()
                    for batch in tqdm(dataloader, desc=f"Training Epoch: {epoch}", unit="batch"):
                        # Unpack the batch
                        input_ids, attention_mask, labels = batch
                        input_ids = batch[input_ids].to(device)
                        attention_mask = batch[attention_mask].to(device)
                        labels = batch[labels].to(device)
                        # Forward pass
                        outputs = model(input_ids, attention_mask)
                        loss = loss_function(outputs, labels)
                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # Calculate Accuracy
                        total_loss += loss.item()
                        predictions = np.argmax(outputs.detach().numpy(), axis=1)
                        total_correct += accuracy.compute(predictions=predictions, references=labels.detach().numpy()).get('accuracy')
                        total_samples += labels.size(0)
                    total_acc = 100 * total_correct / total_samples
                    print(f"Epoch {epoch} loss: {total_loss / len(dataloader)} Acc: {total_acc}")

                torch.save(model.state_dict(), self.hps.save_root)
            case "pytorch_multilingual":
                label2id_value=label2id(os.path.join(self.hps.cache_dir,'normalized_dataset'))
                all_pytorch_data = pytorch_dataset(label2id_value,os.path.join(self.hps.cache_dir,'normalized_dataset'))
                tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
                dataset = TextClassificationDataset(total_list, tokenizer, max_length=128)
                dataloader = DataLoader(dataset, batch_size=self.hps.batch_size or 16)
                if self.hps.cuda:
                    device = torch.device("cuda:0")
                    logger.info("[INFO] Use cuda")

                else:
                    device = torch.device("cpu")
                    logger.info("[INFO] Use CPU")
                model = Pytorch_Classifier(self.hps.m,'bert-base-multilingual-cased',self.hps).to(device)
                loss_function = nn.NLLLoss()
                optimizer = AdamW(model.parameters(), lr=self.hps.lr or 2e-5)
                accuracy = evaluate.load("accuracy")
                for epoch in range(self.hps,n_epochs):
                    total_loss = 0
                    total_correct = 0
                    total_samples = 0
                    model.train()
                    for batch in tqdm(dataloader, desc=f"Training Epoch: {epoch}", unit="batch"):
                        # Unpack the batch
                        input_ids, attention_mask, labels = batch
                        input_ids = batch[input_ids].to(device)
                        attention_mask = batch[attention_mask].to(device)
                        labels = batch[labels].to(device)
                        # Forward pass
                        outputs = model(input_ids, attention_mask)
                        loss = loss_function(outputs, labels)
                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # Calculate Accuracy
                        total_loss += loss.item()
                        predictions = np.argmax(outputs.detach().numpy(), axis=1)
                        total_correct += accuracy.compute(predictions=predictions, references=labels.detach().numpy()).get('accuracy')
                        total_samples += labels.size(0)
                    total_acc = 100 * total_correct / total_samples
                    print(f"Epoch {epoch} loss: {total_loss / len(dataloader)} Acc: {total_acc}")

                torch.save(model.state_dict(), self.hps.save_root)

    def inference(self,API_URL,token, input_path, norm_output_path, token_output_path):
        """
            Test our model:
            In this case, after publishing our model to HuggingFace, we evaluate our model by using api.
        """
        # Preprocess
        normalizer(input_path, norm_output_path)
        tokenizer(norm_output_path, token_output_path)
        # Test model
        headers = {"Authorization": f"Bearer {token}"}
        id2label = {0: "Beinolmelal", 1: "Eghtesadi", 2: "Ejtemaee", 3: "ElmiVaDaneshghai", 4: "FarhangHonarVaResane", 5: "Siasi", 6: "Varzeshi"}
        # Access to HF model
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        # Calculating scores for all documents
        for root, dirs, files in os.walk(token_output_path):
            for file in files:
                with open(f'{token_output_path}/{file}') as f:
                    text = json.load(f)
                    text = text['text'][:600:]
                    output = query({
                    "inputs": f'{text}',
                    })
                    max = output[0][0]['score']
                    maxLabel = output[0][0]['label']
                    for label in output[0]:
                        if label['score'] > max:
                            max = label['score']
                            maxLabel = label['label']
                    print(f"FileName: ${file} => Label: {maxLabel}")


def get_model(config):
    model_cfg = config.model
    model = MyClassifier(config)
    return model

