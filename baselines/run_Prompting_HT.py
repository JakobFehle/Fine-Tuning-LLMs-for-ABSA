import subprocess
import sys
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# CONSTANTS

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # Instruct?

MAX_NEW_TOKENS = 500
LANG = "en"
CONTEXT = 8192

OUTPUT_DIR = '../results/prompting/'
RUN_TAG = ''
TEMPERATURE = 0
SPLIT = 0

TASK = 'all'
LOW_RESOURCE_SETTING = -1
PROMPT_STYLE = 'all'
FEW_SHOTS = -1

# for DATASET in ['GERestaurant', 'rest-16']:
      
#     # Clear System RAM for vllm
#     subprocess.call(['sh', '../src/utils/freeRam.sh'])

#     command = f"python3 prompting/eval.py \
#                 --model_name_or_path {MODEL_NAME} \
#                 --lang {LANG} \
#                 --task {TASK} \
#                 --output_dir {OUTPUT_DIR} \
#                 --prompt_style {PROMPT_STYLE} \
#                 --low_resource_setting {LOW_RESOURCE_SETTING} \
#                 --split {SPLIT} \
#                 --dataset {DATASET} \
#                 --max_new_tokens {MAX_NEW_TOKENS} \
#                 --few_shots {FEW_SHOTS} \
#                 --gpu_memory_utilization 0.85 \
#                 --temperature {TEMPERATURE} \
#                 --max_seq_length {CONTEXT}"
#     command += f" --run_tag {RUN_TAG}" if RUN_TAG != '' else ''

#     process = subprocess.Popen(command, shell=True)
#     process.wait()


# RESULTS_PATH = '../results/prompting/'

# col_names = ['task', 'dataset', 'prompt', 'split', 'lr_setting', 'few_shots', 'f1-micro', 'f1-macro', 'accuracy']

# folder_names = [folder for folder in os.listdir(os.path.join(OUTPUT_DIR)) if os.path.isdir(os.path.join(OUTPUT_DIR, folder)) and folder != '.ipynb_checkpoints']
# runs = []

# for folder_name in folder_names:
#     try:
#         cond_name = folder_name.split('/')[-1]
#         cond_parameters = cond_name.split('_')
        
#         filename = ''
        
#         if cond_parameters[0] == 'acd':
#             filename = 'metrics_asp.tsv'
#         elif cond_parameters[0] == 'acsa':
#             filename = 'metrics_asp_pol.tsv'
#         elif cond_parameters[0] == 'e2e':
#             filename = 'metrics_pol.tsv'
#         elif cond_parameters[0] == 'tasd':
#             filename = 'metrics_phrases.tsv'
            
#         df = pd.read_csv(os.path.join(OUTPUT_DIR, folder_name, filename), sep = '\t')
#         df = df.set_index(df.columns[0])
        
#         cond_parameters.append(df.loc['Micro-AVG', 'f1'])
#         cond_parameters.append(df.loc['Macro-AVG', 'f1'])
#         cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
#         runs.append(cond_parameters)
#     except:
#         pass

# results_all = pd.DataFrame(runs, columns = col_names)

# for DATASET in ['rest-16', 'GERestaurant']:
#     for TASK in ['acd', 'acsa', 'e2e', 'tasd']:
#         for LOW_RESOURCE_SETTING in [0, 1000, 500]:
                
#             results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == str('full' if LOW_RESOURCE_SETTING == 0 else LOW_RESOURCE_SETTING), results_all['dataset'] == DATASET, results_all['task'] == TASK, results_all['split'] == '0'])].sort_values(by = ['f1-micro'], ascending = False)
#             results_sub = results_sub.reset_index()

#             print(results_sub.head(3))
#             FEW_SHOTS = int(results_sub.at[0, 'few_shots'])
#             PROMPT_STYLE = results_sub.at[0, 'prompt']

#             for SPLIT in [1]:
#                 if PROMPT_STYLE == 'cot':
#                     MAX_NEW_TOKENS = 1000
#                 else:
#                     MAX_NEW_TOKENS = 200
                            
#                 # Clear System RAM for vllm
#                 subprocess.call(['sh', '../src/utils/freeRam.sh'])

#                 command = f"python3 prompting/eval.py \
#                 --model_name_or_path {MODEL_NAME} \
#                 --lang {LANG} \
#                 --task {TASK} \
#                 --output_dir {OUTPUT_DIR} \
#                 --prompt_style {PROMPT_STYLE} \
#                 --low_resource_setting {LOW_RESOURCE_SETTING} \
#                 --split {SPLIT} \
#                 --dataset {DATASET} \
#                 --max_new_tokens {MAX_NEW_TOKENS} \
#                 --few_shots {FEW_SHOTS} \
#                 --gpu_memory_utilization 0.85 \
#                 --temperature {TEMPERATURE} \
#                 --max_seq_length {CONTEXT}"
#                 command += f" --run_tag {RUN_TAG}" if RUN_TAG != '' else ''

#                 process = subprocess.Popen(command, shell=True)
#                 process.wait()


# Original Dataset

SPLIT = 0
ORIGINAL_SPLIT = True
FEW_SHOTS = 25
PROMPT_STYLE = 'context'
LOW_RESOURCE_SETTING = 0
MAX_NEW_TOKENS = 200

for DATASET in ['rest-16', 'GERestaurant']:
    for TASK in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        # Clear System RAM for vllm
        subprocess.call(['sh', '../src/utils/freeRam.sh'])

        command = f"python3 prompting/eval.py \
        --model_name_or_path {MODEL_NAME} \
        --lang {LANG} \
        --task {TASK} \
        --output_dir {OUTPUT_DIR} \
        --prompt_style {PROMPT_STYLE} \
        --low_resource_setting {LOW_RESOURCE_SETTING} \
        --split {SPLIT} \
        --dataset {DATASET} \
        --max_new_tokens {MAX_NEW_TOKENS} \
        --few_shots {FEW_SHOTS} \
        --gpu_memory_utilization 0.85 \
        --temperature {TEMPERATURE} \
        {'--original_split' if ORIGINAL_SPLIT else ''} \
        --max_seq_length {CONTEXT}"
        command += f" --run_tag {RUN_TAG}" if RUN_TAG != '' else ''

        process = subprocess.Popen(command, shell=True)
        process.wait()
