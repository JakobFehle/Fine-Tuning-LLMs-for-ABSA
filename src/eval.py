# Code derived from https://github.com/artidoro/qlora

import os
import gc
import torch
import json
import time
import glob
import pandas as pd
import numpy as np
import sys

utils = os.path.abspath('../../src/utils/')
sys.path.append(utils)

from tqdm import tqdm
from datetime import datetime
from transformers import set_seed
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from accelerate.utils import release_memory
from preprocessing import loadDataset, createPrompts
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

def evaluate(model, config, results_path, lora_checkpoints, prompts_test, ground_truth_labels, label_space):

    sampling_params = createSamplingParams(config)
        
    for index, lora_path in enumerate(lora_checkpoints):

        # Skip checkpoints till desired epoch if CV or Final Evaluation
        if config.split != 0 or config.original_split:
            if index + 1 != config.epoch:
                continue
        
        time_start = time.time()
        
        model_outputs = model.generate(prompts_test, sampling_params, lora_request=LoRARequest("adapter", index+1, lora_path))
        
        time_end = time.time()
        eval_time = time_end - time_start 
        
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
        results_path_epoch = results_path + f"{index + 1}/" if not config.original_split else f"{results_path}{config.epoch}/"
        os.makedirs(results_path_epoch, exist_ok=True)

        if results_asp:
            pd.DataFrame.from_dict(results_asp).transpose().to_csv(results_path_epoch + 'metrics_asp.tsv', sep = "\t")
        if results_asp_pol:
            pd.DataFrame.from_dict(results_asp_pol).transpose().to_csv(results_path_epoch + 'metrics_asp_pol.tsv', sep = "\t")
        if results_pairs:
            pd.DataFrame.from_dict(results_pairs).transpose().to_csv(results_path_epoch + 'metrics_pairs.tsv', sep = "\t")
        if results_phrases:
            pd.DataFrame.from_dict(results_phrases).transpose().to_csv(results_path_epoch + 'metrics_phrases.tsv', sep = "\t")
        if results_pol:
            pd.DataFrame.from_dict(results_pol).transpose().to_csv(results_path_epoch + 'metrics_pol.tsv', sep = "\t")

        # Save generated Outputs
        with open(results_path_epoch + 'predictions.txt', 'w') as f:
            for out in model_outputs:
                f.write(f"{out.outputs[0].text.encode('utf-8')}\n")

        # Save Prompt Example
        with open(results_path_epoch + 'prompt.txt', 'w') as f:
            f.write(f"{prompts_test[0].encode('utf-8')}")

        # Save falsely generated Labels
        if false_predictions:
            with open(results_path_epoch + 'false_predictions.txt', 'w') as f:
                for line in false_predictions:
                    f.write(f"{line.encode('utf-8')}\n")

        # Save Evaluation Config
        with open(results_path_epoch + 'eval_config.json', 'w') as f:
            config = vars(config)
            config['inference_tokens'] = tokens_generated / eval_time
            config['inference_tokens_w_prompt'] = (tokens_generated + tokens_prompt) / eval_time
            json.dump(config, f)
        
def main():

    config = Config()
    print(config)      
    
    df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split, config.original_split)
    prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, config)

    set_seed(config.seed)
    config.low_resource_setting = 'full' if config.low_resource_setting == 0 else config.low_resource_setting

    config.model_config = '_'.join([config.task, config.dataset, config.prompt_style, config.learning_rate, config.lora_r, config.lora_alpha, config.lora_dropout, str(config.split), str(config.low_resource_setting) if config.original_split != True else 'orig'])    
    model_folders = glob.glob(f'../models/{config.model_config + config.run_tag}/checkpoint-*')
    
    if not model_folders:
        print(f"No checkpoint files found with config: {config.model_config + config.run_tag}")
        
    else:
        lora_checkpoints = sortCheckpoints(model_folders)
        
        if config.run_tag != '':
            results_path = f'{config.output_dir}/{config.run_tag}/{config.model_config}_'
        else:
            results_path = f'{config.output_dir}/{config.model_config}_'
        
        # Load model with vllm
        model = LLM(
            model=config.model_name_or_path, 
            tokenizer=config.model_name_or_path,
            dtype='bfloat16',
            max_model_len=config.max_seq_length,
            tensor_parallel_size=torch.cuda.device_count(),
            seed = config.seed,
            gpu_memory_utilization = config.gpu_memory_utilization,
            enable_lora = True, 
            max_lora_rank=32
        )
        
        evaluate(model, config, results_path, lora_checkpoints, prompts_test, ground_truth_labels, label_space)

if __name__ == "__main__":
    main()