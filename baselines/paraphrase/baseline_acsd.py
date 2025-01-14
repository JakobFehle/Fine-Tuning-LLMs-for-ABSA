import pandas as pd
import numpy as np
import torch
import os
import transformers
import argparse
import re

from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq
from utils.preprocessing import loadDataset
from utils.evaluation import createResults, convertLabels, extractAspects
from datetime import datetime, timedelta
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
warnings.filterwarnings("ignore", category=UserWarning)

CAT_TO_TERM_GERestaurant = {
    "SERVICE": "Service",
    "FOOD": "Essen",
    "PRICE": "Preis",
    "AMBIENCE": "Ambiente",
    "GENERAL-IMPRESSION": "Allgemeiner Eindruck"
}

CAT_TO_TERM_Rest16 = {
    "AMBIENCE#GENERAL": "Ambience general",
    "DRINKS#PRICES": "Drinks prices",
    "DRINKS#QUALITY": "Drinks quality",
    "DRINKS#STYLE_OPTIONS": "Drinks style options",
    "FOOD#PRICES": "Food prices",
    "FOOD#QUALITY": "Food quality",
    "FOOD#STYLE_OPTIONS": "Food style options",
    "LOCATION#GENERAL": "Location general",
    "RESTAURANT#GENERAL": "Restaurant general",
    "RESTAURANT#MISCELLANEOUS": "Restaurant miscellaneous",
    "RESTAURANT#PRICES": "Restaurant prices",
    "SERVICE#GENERAL": "Service general"
}

POL_TO_TERM_GERestaurant = {
    "NEGATIVE": "schlecht",
    "NEUTRAL": "ok",
    "POSITIVE": "gut"
}

POL_TO_TERM_Rest16 = {
    "NEGATIVE": "bad",
    "NEUTRAL": "ok",
    "POSITIVE": "great"
}

TERM_TO_CAT_GERestaurant = {
    "Service": "SERVICE",
    "Essen": "FOOD",
    "Preis": "PRICE",
    "Ambiente": "AMBIENCE",
    "Allgemeiner Eindruck": "GENERAL-IMPRESSION"
}

TERM_TO_CAT_Rest16 = {
    "Ambience general": "AMBIENCE#GENERAL",
    "Drinks prices": "DRINKS#PRICES",
    "Drinks quality": "DRINKS#QUALITY",
    "Drinks style options": "DRINKS#STYLE_OPTIONS",
    "Food prices": "FOOD#PRICES",
    "Food quality": "FOOD#QUALITY",
    "Food style options": "FOOD#STYLE_OPTIONS",
    "Location general": "LOCATION#GENERAL",
    "Restaurant general": "RESTAURANT#GENERAL",
    "Restaurant miscellaneous": "RESTAURANT#MISCELLANEOUS",
    "Restaurant prices": "RESTAURANT#PRICES",
    "Service general": "SERVICE#GENERAL"
}

TERM_TO_POL_GERestaurant = {
    "schlecht": "NEGATIVE",
    "ok": "NEUTRAL",
    "gut": "POSITIVE"
}

TERM_TO_POL_Rest16 = {
    "bad": "NEGATIVE",
    "ok": "NEUTRAL",
    "great": "POSITIVE"
}

TEXT_TEMPLATE_DE = "{ac_text} ist {polarity_text}, weil {aspect_term_text} {polarity_text} ist."
TEXT_TEMPLATE_EN = "{ac_text} is {polarity_text}, because {aspect_term_text} is {polarity_text}."

TEXT_PATTERN_DE = r"(.*) ist (.*), weil (.*) (.*) ist."
TEXT_PATTERN_EN = r"(.*) is (.*), because (.*) is (.*)."

IT_TOKEN_DE = 'es'
IT_TOKEN_EN = 'it'

@dataclass
class DataArgs:
    task: str = field(
        default="tasd",
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
        default="../results_pp/para/"
    )
    model_name: Optional[str] = field(
        default="t5-large"
    )
    learning_rate: Optional[float] = field(
        default=3e-4
    )
    batch_size: Optional[int] = field(
        default=8
    )
    epochs: Optional[int] = field(
        default=20
    )
    steps: Optional[int] = field(
        default=0
    )
    gradient_steps: Optional[int] = field(
        default=1
    )

class CustomDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class ParaphraseABSA:
    def __init__(self, args):
        self.task = args.task
        self.split = args.split
        self.lr_setting = args.lr_setting
        self.dataset_name = args.dataset
        self.model_name = args.model_name
        self.output_path = args.output_path
        train, eval, self.label_space = loadDataset(args.dataset, args.lr_setting, args.task, args.split, args.original_split)
        self.cat_to_term_dict, self.term_to_cat_dict, self.pol_to_term_dict, self.term_to_pol_dict, self.text_template, self.text_pattern, self.it_token = self.loadPhraseDicts(args.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Device count: ", torch.cuda.device_count())
        self.gpu_count = torch.cuda.device_count()
        self.train, self.eval = self.preprocessData(train, self.tokenizer), self.preprocessData(eval, self.tokenizer)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)

    def loadPhraseDicts(self, dataset):
        if dataset == 'GERestaurant':
            return CAT_TO_TERM_GERestaurant, TERM_TO_CAT_GERestaurant, POL_TO_TERM_GERestaurant, TERM_TO_POL_GERestaurant, TEXT_TEMPLATE_DE, TEXT_PATTERN_DE, IT_TOKEN_DE
        elif dataset == 'rest-16':
            return CAT_TO_TERM_Rest16, TERM_TO_CAT_Rest16, POL_TO_TERM_Rest16, TERM_TO_POL_Rest16, TEXT_TEMPLATE_EN, TEXT_PATTERN_EN, IT_TOKEN_EN
        else:
            print('Dataset name not valid.')
            
    def preprocessData(self, data, tokenizer):
        def labelToText(sample):
            if sample[2] != "NULL":
                aspect_term_text = sample[2]
            else:
                aspect_term_text = self.it_token
            ac_text = self.cat_to_term_dict[sample[0]]
            polarity_text = self.pol_to_term_dict[sample[1]]
            return self.text_template.format(ac_text=ac_text, polarity_text=polarity_text, aspect_term_text=aspect_term_text)

        def createOutput(samples):
            output_text = ''
            for sample in samples:
                output_text += labelToText(sample) + ' [SSEP] '
            return output_text[:-1]

        input_texts = data["text"].tolist()
        
        output_phrases = []

        for sample in list(data['labels_phrases']):
            aspects = extractAspects(', '.join(sample), 'tasd', False, False)
            output_phrases.append(createOutput(aspects))
        
        input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        output_encodings = tokenizer(output_phrases, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        return CustomDataset(input_encodings, output_encodings['input_ids'])

    def createModel(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def computeMetrics(self, eval_pred):
        predictions, ground_truth = eval_pred

        predictions = np.where(predictions != -100,
                           predictions, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)

        ground_truth = np.where(ground_truth != -100, ground_truth, self.tokenizer.pad_token_id)
        decoded_ground_truth = self.tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

        # print(decoded_preds)
        predictions_tuples = []
        for pred in decoded_preds:
            labels_extracted = []
            if pred != '':
                sentences = pred.split('[SSEP]')
                if len(sentences) > 0:
                    for label_strings in sentences:
                        match = re.match(self.text_pattern, label_strings.strip())

                        if match:
                            labels_extracted.append([self.term_to_cat_dict[match.group(1)] if match.group(1) in self.term_to_cat_dict else match.group(1), self.term_to_pol_dict[match.group(2)] if match.group(2) in self.term_to_pol_dict else match.group(2), 'NULL' if match.group(3) == 'es' or match.group(3) == 'it' else match.group(3)])
            predictions_tuples.append(labels_extracted)

        ground_truth_tuples = []
        for gold in decoded_ground_truth:
            labels_extracted = []
            if gold != '':
                sentences = gold.split('[SSEP]')
                if len(sentences) > 0:
                    for label_strings in sentences:
                        match = re.match(self.text_pattern, label_strings.strip())

                        if match:
                            labels_extracted.append([self.term_to_cat_dict[match.group(1)] if match.group(1) in self.term_to_cat_dict else match.group(1), self.term_to_pol_dict[match.group(2)] if match.group(2) in self.term_to_pol_dict else match.group(2), 'NULL' if match.group(3) == 'es' or match.group(3) == 'it' else match.group(3)])
            ground_truth_tuples.append(labels_extracted)

        self.predictions, self.false_predictions = convertLabels(predictions_tuples, self.task, self.label_space)
        ground_truths, _ = convertLabels(ground_truth_tuples, self.task, self.label_space)

        self.results = createResults(self.predictions, ground_truths, self.label_space, self.task)
        return self.results[4]

    def trainModel(self, lr, epochs, batch_size, args):

        training_args = Seq2SeqTrainingArguments(
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
            report_to="none",
            predict_with_generate=True,
            generation_max_length=256,
            weight_decay=0.01,
            gradient_accumulation_steps = args.gradient_steps
        )

        trainer = Seq2SeqTrainer(
            model_init=self.createModel,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.eval,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics
        )
        
        print("Using the following hyperparameters: lr=" + str(lr) + " - epochs=" + str(epochs) + " - batch=" + str(batch_size*args.gradient_steps*self.gpu_count))

        trainer.train()

        return trainer

    def evalModel(self, trainer, output_path, test = True):

        # Save results as tsv
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        _ = trainer.evaluate()
        
        results_asp, results_asp_pol, results_pairs, results_pol, results_phrases = self.results
        pd.DataFrame.from_dict(results_asp).transpose().to_csv(output_path + 'metrics_asp.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_asp_pol).transpose().to_csv(output_path + 'metrics_asp_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_pairs).transpose().to_csv(output_path + 'metrics_pairs.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_pol).transpose().to_csv(output_path + 'metrics_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_phrases).transpose().to_csv(output_path + 'metrics_phrases.tsv', sep = "\t")

        with open(output_path + 'config.txt', 'w') as f:
            for k,v in vars(trainer.args).items():
                f.write(f"{k}: {v}\n")
        
        # Save outputs to file
        with open(output_path + 'predictions.txt', 'w') as f:
            for line in self.predictions:
                f.write(f"{str(line).encode('utf-8')}\n")

        # Save false output labels to file
        if(len(self.false_predictions) > 0):
            with open(output_path + 'false_predictions.txt', 'w') as f:
                for line in self.false_predictions:
                    f.write(f"{str(line).encode('utf-8')}\n")
                
    def train_eval(self, args):
        results_path = f'{self.output_path}{self.task}_{self.dataset_name}_{self.lr_setting}_{self.split}_{round(args.learning_rate,9)}_{args.batch_size}_{args.epochs}/' 
        
        trainer = self.trainModel(args.learning_rate, args.epochs, int(args.batch_size/(self.gpu_count * args.gradient_steps)), args)

        self.evalModel(trainer, results_path)

if __name__ == "__main__":

    hfparser = transformers.HfArgumentParser([DataArgs, TrainingArgs])

    data_config, training_config, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(data_config),
        **vars(training_config)
    )
        
    absa = ParaphraseABSA(args)
    absa.train_eval(args)

