# Code overall dirty but functional

import os
import gc
import torch
import json
import time
import glob
import pandas as pd
import numpy as np
import sys

from preprocessing import loadDataset, createPrompts

utils = os.path.abspath('../src/utils/')
sys.path.append(utils)

from tqdm import tqdm
from datetime import datetime
from transformers import set_seed
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from accelerate.utils import release_memory

from config import Config
from evaluation import (
    StoppingCriteriaSub, createResults, extractAspects, convertLabels, sortCheckpoints
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def createSamplingParams(config):
    STOP_WORDS = ['### Input:', '\n\n', 'Sentence:']
    STOP_WORDS_COT = ['### Input:', '\n\n', '<|eot_id|>']
    
    return SamplingParams(
        temperature = config.temperature, 
        stop = STOP_WORDS if config.prompt_style != 'cot' else STOP_WORDS_COT,
        max_tokens = config.max_new_tokens,
        top_k = config.top_k,
        top_p = config.top_p,
        skip_special_tokens = True
    )

def evaluate(model, config, results_path, prompts_test, ground_truth_labels, label_space):

    sampling_params = createSamplingParams(config)
    
    time_start = time.time()
    
    model_outputs = model.generate(prompts_test, sampling_params)
    
    time_end = time.time()
    eval_time = time_end - time_start 
    
    #Save Metrics
    os.makedirs(results_path, exist_ok=True)
    # Save generated Outputs
    with open(results_path + 'predictions.txt', 'w') as f:
        for out in model_outputs:
            f.write(f"{out.outputs[0].text.encode('utf-8')}\n")
    
    tokens_prompt, tokens_generated = 0, 0
    ground_truth, predictions = [], []

    # Extract ABSA Elements from generated Output
    for gt in ground_truth_labels:
        ground_truth.append(extractAspects(gt, config.task, config.prompt_style == 'cot'))

    for out in model_outputs:
        predictions.append(extractAspects(out.outputs[0].text, config.task, config.prompt_style == 'cot', True))
        tokens_prompt += len(out.prompt_token_ids)
        tokens_generated += len(out.outputs[0].token_ids)       

    # Combine ABSA Elements to String
    gold_labels, _ = convertLabels(ground_truth, config.task, label_space)
    pred_labels, false_predictions = convertLabels(predictions, config.task, label_space)

    results_asp, results_asp_pol, results_pairs, results_pol, results_phrases = createResults(pred_labels, gold_labels, label_space, config.task)

    #Save Metrics
    os.makedirs(results_path, exist_ok=True)

    if results_asp:
        pd.DataFrame.from_dict(results_asp).transpose().to_csv(results_path + 'metrics_asp.tsv', sep = "\t")
    if results_asp_pol:
        pd.DataFrame.from_dict(results_asp_pol).transpose().to_csv(results_path + 'metrics_asp_pol.tsv', sep = "\t")
    if results_pairs:
        pd.DataFrame.from_dict(results_pairs).transpose().to_csv(results_path + 'metrics_pairs.tsv', sep = "\t")
    if results_phrases:
        pd.DataFrame.from_dict(results_phrases).transpose().to_csv(results_path + 'metrics_phrases.tsv', sep = "\t")
    if results_pol:
        pd.DataFrame.from_dict(results_pol).transpose().to_csv(results_path + 'metrics_pol.tsv', sep = "\t")

    # Save generated Outputs
    with open(results_path + 'predictions.txt', 'w') as f:
        for out in model_outputs:
            f.write(f"{out.outputs[0].text.encode('utf-8')}\n")

    # Save Prompt Example
    with open(results_path + 'prompt.txt', 'w') as f:
        f.write(f"{prompts_test[0].encode('utf-8')}")

    # Save falsely generated Labels
    if false_predictions:
        with open(results_path + 'false_predictions.txt', 'w') as f:
            for line in false_predictions:
                f.write(f"{line.encode('utf-8')}\n")
        
def main():

    config = Config()
    
    # Load model with vllm
    model = LLM(
        model=config.model_name_or_path, 
        tokenizer=config.model_name_or_path,
        dtype='bfloat16',
        max_model_len=config.max_seq_length,
        tensor_parallel_size=2,
        seed = config.seed,
        gpu_memory_utilization = config.gpu_memory_utilization,
    )
    
    if config.split == 0: #HT
        if config.task == 'all':
            TASKS = ['acd', 'acsa', 'e2e', 'tasd']
        else:
            TASKS = [config.task]
    
        if config.low_resource_setting == -1:
            LOW_RESOURCE_SETTINGS = [0, 1000, 500]
        else:
            LOW_RESOURCE_SETTINGS = [config.low_resource_setting]
    
        if config.few_shots == -1:
            FEW_SHOTS = [5, 10, 25]
        else:
            FEW_SHOTS = [config.few_shots]
    
        if config.prompt_style == 'all':
            PROMPT_STYLES = ['basic', 'context', 'cot']
        else:
            PROMPT_STYLES = [config.prompt_style]
            
        for TASK in TASKS:
            for FEW_SHOT in FEW_SHOTS:
                for LOW_RESOURCE_SETTING in LOW_RESOURCE_SETTINGS:
                    for PROMPT_STYLE in PROMPT_STYLES:
                        if TASK == 'acd' and PROMPT_STYLE == 'cot':
                            continue
                        if config.prompt_style == 'all':
                            PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
                        else:
                            PROMPT_STYLES = [config.prompt_style]
                        config.prompt_style = PROMPT_STYLE
                        config.low_resource_setting = LOW_RESOURCE_SETTING
                        config.task = TASK
                        
                        if PROMPT_STYLE == 'cot':
                            config.max_new_tokens = 1000
                        else:
                            config.max_new_tokens = 200
                        
                        df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split, config.original_split)
                        prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, FEW_SHOT, config)
                        
                        set_seed(config.seed)
                        config.low_resource_setting = 'full' if config.low_resource_setting == 0 else config.low_resource_setting
                        
                        config.model_config = '_'.join([config.task, config.dataset, config.prompt_style, str(config.split), str(config.low_resource_setting) if config.original_split != True else 'orig', str(FEW_SHOT)])    
                            
                        if config.run_tag != '':
                            results_path = f'{config.output_dir}/{config.run_tag}/{config.model_config}/'
                        else:
                            results_path = f'{config.output_dir}/{config.model_config}/'
                        print(results_path)
                        evaluate(model, config, results_path, prompts_test, ground_truth_labels, label_space)

    else:
        SPLITS = [1,2,3,4,5] if config.original_split == False else [0]
        for SPLIT in SPLITS:
            config.split = SPLIT
            df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split, config.original_split)
            prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, config.few_shots, config)
            
            set_seed(config.seed)
            config.low_resource_setting = 'full' if config.low_resource_setting == 0 else config.low_resource_setting
        
            config.model_config = '_'.join([config.task, config.dataset, config.prompt_style, str(config.split), str(config.low_resource_setting) if config.original_split != True else 'orig', str(config.few_shots)])    
                
            if config.run_tag != '':
                results_path = f'{config.output_dir}/{config.run_tag}/{config.model_config}/'
            else:
                results_path = f'{config.output_dir}/{config.model_config}/'
    
    
    
            evaluate(model, config, results_path, prompts_test, ground_truth_labels, label_space)

if __name__ == "__main__":
    main()