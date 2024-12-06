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

LORA_R_ALPHA_COMB = [[8,8],[8,16],[32,32],[32,64]]

RUN_TAG = ''

###
# Hyperparameter Valdiation Phase
###

runs = []

for DATASET in ['rest-16', 'GERestaurant']: 
    for TASK in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        for LOW_RESOURCE_SETTING in [0, 1000, 500]:
            PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
            for PROMPT_STYLE in PROMPT_STYLES:
                for LORA_R, LORA_ALPHA in LORA_R_ALPHA_COMB:
                    SPLIT = 0
    
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