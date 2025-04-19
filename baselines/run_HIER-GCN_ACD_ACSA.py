import subprocess
import sys
import os
import pandas as pd
import numpy as np

DATA_DIR = 'hier_gcn/data'
MODEL = 'GCN'
OUTPUT_PATH = '../results/hier_gcn/'
BASE_EPOCHS = 20
BATCH_SIZE = 8

###
# HT
###

TASK_NAME = ["GERestaurantACSA", "Rest16ACSA"][int(sys.argv[1])]
TASK = 'acsa'

for LR_SETTING in [0, 1000, 500]:
    for SPLIT in [0]:  

        # Calculate amount of Training Steps
        DO_STEPS = [False] if LR_SETTING == 0 else [True, False]
        for STEPS in DO_STEPS:
            if STEPS == True:
                # Train split lenghts for hyperparameter tuning phase (5/6 of original train sizes)
                if TASK_NAME == 'GERestaurantACSA':
                    dataset_base_len = 1795                            
                elif TASK_NAME == 'Rest16ACSA':
                    dataset_base_len = 1423

                low_resource_steps_per_e =  (dataset_base_len if LR_SETTING == 0 else (LR_SETTING * 5/6)) / BATCH_SIZE

                MAX_STEPS = int((dataset_base_len * BASE_EPOCHS) / BATCH_SIZE)

                # Transform to #Epochs for Output-Path
                EPOCHS = round(MAX_STEPS / low_resource_steps_per_e)

                print(TASK_NAME, LR_SETTING, EPOCHS)
            else:
                EPOCHS = 20
                
            command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 hier_gcn/run_classifier_gcn.py \
                  --task_name {TASK_NAME} \
                  --do_train \
                  --do_eval \
                  --model_type {MODEL}\
                  --do_lower_case \
                  --data_dir {DATA_DIR} \
                  --low_resource_setting {LR_SETTING} \
                  --split {SPLIT} \
                  --max_seq_length 128 \
                  --train_batch_size {BATCH_SIZE} \
                  --learning_rate 5e-5 \
                  --num_train_epochs {EPOCHS} \
                  --output_dir {OUTPUT_PATH} \
                  --task {TASK}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

###
# Cross Evaluation Phase
###

col_names = ['task', 'dataset', 'lr_setting', 'split', 'learning_rate', 'batch_size', 'epochs', 'f1-micro']
folder_names = [folder for folder in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, folder)) and folder != '.ipynb_checkpoints']

runs = []

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')
        
        if cond_parameters[3] == '0':
            cond_params = cond_parameters.copy()
            
            with open(os.path.join(OUTPUT_PATH, folder_name, 'cate_eval_results.txt'), 'r') as f:
                f1 = f.readlines()[3].split(' = ')[1]
            
            cond_params.append(round(float(f1), 2))
            cond_params[0] = 'acd'
            runs.append(cond_params)

            cond_params = cond_parameters.copy()
            with open(os.path.join(OUTPUT_PATH, folder_name, 'eval_results.txt'), 'r') as f:
                f1 = f.readlines()[3].split(' = ')[1]
            
            cond_params.append(round(float(f1), 2))
            cond_params[0] = 'acsa'
            runs.append(cond_params)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)

for TASK_NAME in ["GERestaurantACSA", "Rest16ACSA"]:
    for TASK in ['acsa']:
        for LR_SETTING in [0, 1000, 500]:
            DATASET = 'GERestaurant' if TASK_NAME == 'GERestaurantACSA' else 'rest-16'
            results_sub = results_all[np.logical_and.reduce([results_all['task'] == TASK, results_all['lr_setting'] == str(LR_SETTING), results_all['dataset'] == DATASET, results_all['split'] == '0'])].sort_values(by = ['f1-micro'], ascending = False)
            results_sub = results_sub.reset_index()
            
            EPOCHS = int(eval(results_sub.at[0, 'epochs']))
            
            for SPLIT in  [1,2,3,4,5]:
                
                command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 hier_gcn/run_classifier_gcn.py \
                      --task_name {TASK_NAME} \
                      --do_train \
                      --do_eval \
                      --model_type {MODEL}\
                      --do_lower_case \
                      --data_dir {DATA_DIR} \
                      --low_resource_setting {LR_SETTING} \
                      --split {SPLIT} \
                      --max_seq_length 128 \
                      --train_batch_size 8 \
                      --learning_rate 5e-5 \
                      --num_train_epochs {EPOCHS} \
                      --output_dir {OUTPUT_PATH} \
                      --task {TASK}"
                process = subprocess.Popen(command, shell=True)
                process.wait()

        EPOCHS = 20
        LR_SETTING = 50
        for SPLIT in  [1,2,3,4,5]:
            
            command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 hier_gcn/run_classifier_gcn.py \
                  --task_name {TASK_NAME} \
                  --do_train \
                  --do_eval \
                  --model_type {MODEL}\
                  --do_lower_case \
                  --data_dir {DATA_DIR} \
                  --low_resource_setting {LR_SETTING} \
                  --split {SPLIT} \
                  --max_seq_length 128 \
                  --train_batch_size 8 \
                  --learning_rate 5e-5 \
                  --num_train_epochs {EPOCHS} \
                  --output_dir {OUTPUT_PATH} \
                  --task {TASK}"
            process = subprocess.Popen(command, shell=True)
            process.wait()
