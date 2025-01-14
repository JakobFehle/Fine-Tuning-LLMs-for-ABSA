import pandas as pd
import numpy as np
import torch
import os
import transformers
import argparse

from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scipy.special import expit
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding
from utils.preprocessing import loadDataset
from utils.evaluation import createResults, convertLabels, extractAspects
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")

@dataclass
class DataArgs:
    task: str = field(
        default="acd",
        metadata={"help": 'ABSA-Task'}
    )
    split: int = field(
        default=0,
        metadata={"help": 'Split for Eval'}
    )
    lr_setting: int = field(
        default=0,
        metadata={"help": 'Low Resource Setting'}
    )
    dataset: str = field(
        default='GERestaurant',
        metadata={"help": 'Dataset'}
    )
    data_path: str = field(
        default='data/'
    )
    
@dataclass
class TrainingArgs:
    output_path: Optional[str] = field(
        default="../results/bert-clf"
    )
    model_name: str = field()
    learning_rate: Optional[float] = field(
        default=5e-5
    )
    batch_size: Optional[int] = field(
        default=8
    )
    epochs: Optional[int] = field(
        default=5
    )
    steps: Optional[int] = field(
        default=0
    )

class CustomDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["label"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class MultiLabelABSA:
    def __init__(self, args):
        self.task = args.task
        self.split = args.split
        self.lr_setting = args.lr_setting
        train, evaluation, self.label_space = loadDataset(args.dataset, args.lr_setting, args.task, args.split, args.original_split)
        self.dataset_name = args.dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.train, self.eval = self.preprocessData(train, self.task, self.tokenizer, True), self.preprocessData(evaluation, self.task, self.tokenizer)
        print("Device count: ", torch.cuda.device_count())
        self.gpu_count = torch.cuda.device_count()
        self.model = self.createModel(self.model_name, len(self.train[0]['label']))
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def preprocessData(self, data, task, tokenizer, train = False):
        texts = data["text"].tolist()
        
        labels = data['labels']
        if task == 'acd':
            labels = [[tuple.split(', ')[0][1:] for tuple in sample] for sample in data['labels']] # Extract aspects only

        if train == True:
            self.mlb = MultiLabelBinarizer()
            labels = self.mlb.fit_transform(labels)
        else:
            labels = self.mlb.transform(labels)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        return CustomDataset(encodings, labels)

    def createModel(self, model_id, num_labels):
        print('Creating Model: ', model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        return model

    def computeMetrics(self, eval_pred):
        predictions, gold = eval_pred
        
        predictions = (expit(predictions) > 0.5)

        pr = (expit(predictions) > 0.5)
        la = [l==1 for l in gold]
        
        predictions = self.mlb.inverse_transform(predictions)
        gold = self.mlb.inverse_transform(gold)

        predictions = [list(sample) for sample in predictions]
        gold = [list(sample) for sample in gold]

        gold_extracted = []
        pred_extracted = []
        
        # Extract Labels from outputs
        for gt in gold:
            gold_extracted.append(extractAspects(f"[{', '.join(gt)}]", self.task, False, False))
    
        for pred in predictions:
            pred_extracted.append(extractAspects(f"[{', '.join(pred)}]", self.task, False, True))
        
        gold_labels, _ = convertLabels(gold_extracted, self.task, self.label_space)
        pred_labels, false_predictions = convertLabels(pred_extracted, self.task, self.label_space)
        
        self.results = createResults(pred_labels, gold_labels, self.label_space, self.task)
        self.predictions = predictions

        if self.task == 'acd':
            return self.results[0]
        else:
            return self.results[1]

    def trainModel(self, lr, epochs, batch_size):

        training_args = TrainingArguments(
            output_dir="outputs",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir="logs",
            logging_steps=100,
            logging_strategy="epoch",
            max_steps = args.max_steps,
            bf16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.eval,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics,
            # class_weights = self.class_weights
        )
        
        print("Using the following hyperparameters: lr=" + str(lr) + " - epochs=" + str(epochs) + " - batch=" + str(batch_size*self.gpu_count))
        # print("Model: ", MODEL_ID_EN if self.dataset_name == 'rest-16' else MODEL_ID_GER)
        
        trainer.train()

        return trainer


    def evalModel(self, trainer, results_path, test = True):

        # Save results as tsv
        if not os.path.exists(results_path):
            os.makedirs(output_path, exist_ok=True)
            
        if self.task == 'acd':
            results, predictions = trainer.evaluate()
            results_asp, _, _, _, _ = self.results
            
            pd.DataFrame.from_dict(self.results_asp).transpose().to_csv(results_path + 'metrics_asp.tsv', sep = "\t")
                
        else:
            results, predictions = trainer.evaluate()
            results_asp, results_asp_pol, results_pairs, results_pol, _ = self.results
            
            pd.DataFrame.from_dict(sesults_asp).transpose().to_csv(results_path + 'metrics_asp.tsv', sep = "\t")
            pd.DataFrame.from_dict(results_asp_pol).transpose().to_csv(results_path + 'metrics_asp_pol.tsv', sep = "\t")
            pd.DataFrame.from_dict(results_pairs).transpose().to_csv(results_path + 'metrics_pairs.tsv', sep = "\t")
            pd.DataFrame.from_dict(results_pol).transpose().to_csv(results_path + 'metrics_pol.tsv', sep = "\t")

        # Save outputs to file
        with open(results_path + 'predictions.txt', 'w') as f:
            for line in self.predictions:
                f.write(f"{str(line).encode('utf-8')}\n")

        with open(results_path + 'config.txt', 'w') as f:
            for k,v in vars(trainer.args).items():
                f.write(f"{k}: {v}\n")
                
    def train_eval(self, args):

        trainer = self.trainModel(args.learning_rate, args.epochs, int(args.batch_size/self.gpu_count))

        results_path = f'{self.output_path}{self.task}_{self.dataset_name}_{self.lr_setting}_6_{round(args.batch_size,9)}_{args.batch_size}_{args.epochs}/' 

        
        self.evalModel(trainer, results_path)

if __name__ == "__main__":

    hfparser = transformers.HfArgumentParser([DataArgs, TrainingArgs])

    data_config, training_config, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(data_config),
        **vars(training_config)
    )
        
    absa = MultiLabelABSA(args)  
    absa.train_eval(args)

