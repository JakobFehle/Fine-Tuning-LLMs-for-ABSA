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

LORA_R_ALPHA_COMB = [[8,8],[8,16],[32,32],[32,64]]

RUN_TAG = ''

###
# Hyperparameter Valdiation Phase
###

runs = []

for DATASET in ['GERestaurant', 'rest-16']: 
    for TASK in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        for LOW_RESOURCE_SETTING in [0, 1000, 500]:
            PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
            for PROMPT_STYLE in PROMPT_STYLES:
                for LEARNING_RATE in [0.0003, 0.00003]:
                    for LORA_R, LORA_ALPHA in LORA_R_ALPHA_COMB:
                        SPLIT = 0
        
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