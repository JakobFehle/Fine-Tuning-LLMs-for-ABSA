import subprocess
import sys
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# CONSTANTS

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
LORA_DROPOUT = 0.05
QUANT = 4
BATCH_SIZE = 8
EPOCHS = 10
MAX_NEW_TOKENS = 500
LANG = "en"
ORIGINAL_SPLIT = False

RESULTS_PATH = '../results/'
RUN_TAG = ''

###
# Final Evaluation Phase
###

col_names = ['task', 'dataset', 'prompt', 'learning_rate', 'lora_r', 'lora_alpha', 'lora_dropout', 'split', 'lr_setting', 'epoch', 'f1-micro', 'f1-macro', 'accuracy']

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']
runs = []

for folder_name in folder_names:
    try:
        cond_name = folder_name.split('/')[-1]
        cond_parameters = cond_name.split('_')
        
        filename = ''
        
        if cond_parameters[4] == 'acd':
            filename = 'metrics_asp.tsv'
        elif cond_parameters[4] == 'acsa':
            filename = 'metrics_asp_pol.tsv'
        elif cond_parameters[4] == 'e2e':
            filename = 'metrics_pol.tsv'
        elif cond_parameters[4] == 'tasd':
            filename = 'metrics_phrases.tsv'
            
        df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, filename), sep = '\t')
        df = df.set_index(df.columns[0])
        
        cond_parameters.append(df.loc['Micro-AVG', 'f1'])
        cond_parameters.append(df.loc['Macro-AVG', 'f1'])
        cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
        runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)


for DATASET in ['rest-16', 'GERestaurant']: 
    for TASK in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
        for PROMPT_STYLE in PROMPT_STYLES:

            LOW_RESOURCE_SETTING = 0
            SPLIT = 0
            
            results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == str('full' if LOW_RESOURCE_SETTING == 0 else LOW_RESOURCE_SETTING), results_all['dataset'] == DATASET, results_all['task'] == TASK, results_all['split'] == '0', results_all['prompt'] == MODEL_PROMPT_STYLE])].sort_values(by = ['f1-micro'], ascending = False)
            results_sub = results_sub[['dataset', 'task', 'prompt', 'learning_rate', 'lr_setting', 'lora_r', 'lora_alpha', 'epoch', 'f1-micro', 'f1-macro']]
            results_sub = results_sub.reset_index()
                        
            LEARNING_RATE = results_sub.at[0, 'learning_rate']
            LORA_R = int(results_sub.at[0, 'lora_r'])
            LORA_ALPHA = int(results_sub.at[0, 'lora_alpha'])
            EPOCHS = int(results_sub.at[0, 'epoch'])

            # Clear System RAM for vllm
            subprocess.call(['sh', './utils/freeRam.sh'])

            command = f"python3 eval.py \
            --model_name_or_path {MODEL_NAME} \
            --lang {LANG} \
            --task {TASK} \
            --prompt_style {MODEL_PROMPT_STYLE} \
            --low_resource_setting {LOW_RESOURCE_SETTING} \
            --split {SPLIT} \
            --dataset {DATASET} \
            --max_new_tokens {MAX_NEW_TOKENS} \
            --epoch {EPOCHS} \
            --original_split {ORIGINAL_SPLIT} \
            --run_tag {RUN_TAG}"
            process = subprocess.Popen(command, shell=True)
            process.wait()