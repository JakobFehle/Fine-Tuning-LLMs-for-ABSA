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
QLORA_QUANT = 4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER = 'constant'
LANG = 'en'
ORIGINAL_SPLIT = True

start = time.time()
counter = 0

RUN_TAG = ''
RESULTS_PATH = '../results/'

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

DATASET = ['rest-16', 'GERestaurant'][int(sys.argv[1])]

for TASK in ['acd', 'acsa', 'e2e','e2e-e', 'tasd']:
    PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
    for PROMPT_STYLE in PROMPT_STYLES:
        SPLIT = 0
        LOW_RESOURCE_SETTING = 0

        results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == str('full' if LOW_RESOURCE_SETTING == 0 else LOW_RESOURCE_SETTING), results_all['dataset'] == DATASET, results_all['task'] == TASK, results_all['split'] == '0', results_all['prompt'] == PROMPT_STYLE])].sort_values(by = ['f1-micro'], ascending = False)
        results_sub = results_sub[['dataset', 'task', 'prompt', 'learning_rate', 'lr_setting', 'lora_r', 'lora_alpha', 'epoch', 'f1-micro', 'f1-macro']]
        results_sub = results_sub.reset_index()
    
        print(results_sub.head(3))
        
        LEARNING_RATE = results_sub.at[0, 'learning_rate']
        LORA_R = int(results_sub.at[0, 'lora_r'])
        LORA_ALPHA = int(results_sub.at[0, 'lora_alpha'])
        EPOCHS = int(results_sub.at[0, 'epoch'])
                        
        counter += 1
        command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 train.py \
        --model_name_or_path {MODEL_NAME} \
        --lora_r {LORA_R} \
        --lora_alpha {LORA_ALPHA} \
        --lora_dropout {LORA_DROPOUT} \
        --quant {QLORA_QUANT} \
        --learning_rate {LEARNING_RATE} \
        --per_device_train_batch_size {BATCH_SIZE} \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
        --lang {LANG} \
        --prompt_style {PROMPT_STYLE} \
        --num_train_epochs {EPOCHS} \
        --low_resource_setting {LOW_RESOURCE_SETTING} \
        --dataset {DATASET} \
        --task {TASK} \
        --split {SPLIT} \
        --lr_scheduler {LR_SCHEDULER}  \
        --original_split {ORIGINAL_SPLIT} \
        --run_tag {RUN_TAG}"
        process = subprocess.Popen(command, shell=True)
        process.wait()


end = time.time()

training_time = end - start
training_time = str(timedelta(seconds=training_time))
print('################')
print('##')
print(f'##  Total Training Time of {counter} runs: {training_time}')
print('##')
print('################')


