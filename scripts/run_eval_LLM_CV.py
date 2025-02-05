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

RESULTS_PATH = '../results/ft_llm'
RUN_TAG = ''

###
# Cross Evaluation Phase
###

col_names = ['task', 'dataset', 'prompt', 'learning_rate', 'lora_r', 'lora_alpha', 'lora_dropout', 'split', 'lr_setting', 'epoch', 'f1-micro', 'f1-macro', 'accuracy']

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']
runs = []

for folder_name in folder_names:
    try:
        cond_name = folder_name.split('/')[-1]
        cond_parameters = cond_name.split('_')
        
        filename = ''
        
        if cond_parameters[0] == 'acd':
            filename = 'metrics_asp.tsv'
        elif cond_parameters[0] == 'acsa':
            filename = 'metrics_asp_pol.tsv'
        elif cond_parameters[0] == 'e2e':
            filename = 'metrics_pol.tsv'
        elif cond_parameters[0] == 'tasd':
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


for DATASET in ['GERestaurant', 'rest-16']: 
    for TASK in ['acd', 'acsa', 'e2e','e2e-e', 'tasd']:
        for LOW_RESOURCE_SETTING in [0, 1000, 500]:
            PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
            for PROMPT_STYLE in PROMPT_STYLES:
                
                results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == str('full' if LOW_RESOURCE_SETTING == 0 else LOW_RESOURCE_SETTING), results_all['dataset'] == DATASET, results_all['task'] == TASK, results_all['split'] == '0', results_all['prompt'] == PROMPT_STYLE])].sort_values(by = ['f1-micro'], ascending = False)
                results_sub = results_sub[['dataset', 'task', 'prompt', 'learning_rate', 'lr_setting', 'lora_r', 'lora_alpha', 'epoch', 'f1-micro', 'f1-macro']]
                results_sub = results_sub.reset_index()
                            
                LEARNING_RATE = results_sub.at[0, 'learning_rate']
                LORA_R = int(results_sub.at[0, 'lora_r'])
                LORA_ALPHA = int(results_sub.at[0, 'lora_alpha'])
                EPOCHS = int(results_sub.at[0, 'epoch'])

                for SPLIT in [1,2,3,4,5]:
    
                    # Clear System RAM for vllm
                    subprocess.call(['sh', '../src/utils/freeRam.sh'])
    
                    command = f"python3 ../src/eval.py \
                    --model_name_or_path {MODEL_NAME} \
                    --lang {LANG} \
                    --task {TASK} \
                    --learning_rate {LEARNING_RATE} \
                    --prompt_style {PROMPT_STYLE} \
                    --low_resource_setting {LOW_RESOURCE_SETTING} \
                    --split {SPLIT} \
                    --lora_r {LORA_R} \
                    --lora_alpha {LORA_ALPHA} \
                    --dataset {DATASET} \
                    --max_new_tokens {MAX_NEW_TOKENS} \
                    --epoch {EPOCHS} \
                    --output_dir {RESULTS_PATH} \
                    --bf16 \
                    --flash_attention"

                    command += f" --run_tag {RUN_TAG}" if RUN_TAG != '' else ''
                    command += f" --original_split" if ORIGINAL_SPLIT == True else ''
                    
                    process = subprocess.Popen(command, shell=True)
                    process.wait()