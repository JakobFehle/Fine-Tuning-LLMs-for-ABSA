import subprocess
import sys
import os
import pandas as pd
import numpy as np

# CONSTANTS

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
LORA_DROPOUT = 0.05
QLORA_QUANT = 4
BATCH_SIZE = 8
EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER = 'constant'
LANG = 'en'
ORIGINAL_SPLIT = False

LORA_R_ALPHA_COMB = [[8,8],[8,16],[32,32],[32,64]]

RUN_TAG = ''
RESULTS_PATH = '../results/'

###
# Hyperparameter Validation Phase
###

DATASET = ['GERestaurant', 'rest-16'][int(sys.argv[1])]

for TASK in [['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']]:
    for LOW_RESOURCE_SETTING in [0, 500, 1000]:
        PROMPT_STYLES = ['basic', 'context'] if TASK == 'acd' else ['basic', 'context', 'cot']
        for PROMPT_STYLE in PROMPT_STYLES:
            for LEARNING_RATE in [0.0003, 0.00003]:
                for LORA_R, LORA_ALPHA in LORA_R_ALPHA_COMB:
                    SPLIT = 0
                        
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
                    --lr_scheduler {LR_SCHEDULER} \
                    --original_split {ORIGINAL_SPLIT} \
                    --run_tag {RUN_TAG}"
                    process = subprocess.Popen(command, shell=True)
                    process.wait()