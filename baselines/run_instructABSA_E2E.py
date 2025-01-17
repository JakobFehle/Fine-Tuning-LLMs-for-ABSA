import subprocess
import sys
import os
import pandas as pd
import numpy as np

# Enable HT
ORIGINAL_SPLIT = False 
BATCH_SIZE = 8
BASE_EPOCHS = 4
MAX_STEPS = 0

###
# HT
###
LEARNING_RATE = 5e-5

DATASET, MODEL = [['GERestaurant', 't5-base'], ['rest-16', 'allenai/tk-instruct-base-def-pos']][int(sys.argv[1])]
SPLIT = 0

for DATA_PATH, OUTPUT_PATH in [['data', '../results/instructABSA']]:
    for LR_SETTING in ['1000', '500', 'full']:
        # Calculate amount of Training Steps
        DO_STEPS = [False] if LR_SETTING == 0 else [True, False]
        for STEPS in DO_STEPS:
            if STEPS == True:
                # Train split lenghts for hyperparameter tuning phase (5/6 of original train sizes)
                if DATASET == 'GERestaurant':
                    dataset_base_len = 1795                            
                elif DATASET == 'rest-16':
                    dataset_base_len = 1423

                low_resource_steps_per_e =  (dataset_base_len if LR_SETTING == 'full' else (int(LR_SETTING) * 5/6)) / BATCH_SIZE

                MAX_STEPS = int((dataset_base_len * BASE_EPOCHS) / BATCH_SIZE)

                # Transform to #Epochs for Output-Path
                EPOCHS = round(MAX_STEPS / low_resource_steps_per_e)
                
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 instructABSA/run_model.py \
                        -mode train -model_checkpoint {MODEL} \
                        -task joint \
                        -output_dir instructABSA/Models \
                        -inst_type 2 \
                        -id_tr_data_path instructABSA/{DATA_PATH}/{DATASET}/train_{LR_SETTING}.csv \
                        -id_te_data_path instructABSA/{DATA_PATH}/{DATASET}/val_{LR_SETTING}.csv \
                        -evaluation_strategy no \
                        -learning_rate {LEARNING_RATE} \
                        -per_device_train_batch_size {BATCH_SIZE} \
                        -num_train_epochs {BASE_EPOCHS if STEPS == False else EPOCHS} \
                        -steps {MAX_STEPS} \
                        -lr_setting {LR_SETTING} \
                        -dataset {DATASET} \
                        -split {SPLIT}"
            process = subprocess.Popen(command, shell=True)
            process.wait()
        
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 instructABSA/run_model.py \
                        -mode eval  \
                        -model_checkpoint {MODEL} \
                        -task joint \
                        -output_dir instructABSA/Models \
                        -inst_type 2 \
                        -id_tr_data_path instructABSA/{DATA_PATH}/{DATASET}/train_{LR_SETTING}.csv \
                        -id_te_data_path instructABSA/{DATA_PATH}/{DATASET}/val_{LR_SETTING}.csv \
                        -evaluation_strategy no \
                        -learning_rate {LEARNING_RATE} \
                        -per_device_train_batch_size {BATCH_SIZE} \
                        -num_train_epochs {BASE_EPOCHS if STEPS == False else EPOCHS} \
                        -steps {MAX_STEPS} \
                        -lr_setting {LR_SETTING} \
                        -dataset {DATASET} \
                        -split {SPLIT} \
                        -output_path {OUTPUT_PATH}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

###
# Cross Evaluation Phase
###

OUTPUT_PATH = '../results/instructABSA'

col_names = ['dataset', 'lr_setting', 'split', 'learning_rate', 'epochs', 'f1-micro']
filenames = [file for file in os.listdir(os.path.join(OUTPUT_PATH)) if file != '.ipynb_checkpoints']

runs = []

for file in filenames:
    try:
        print(file)
        cond_name = file.split('.tsv')[0]
        print(cond_name)
        cond_parameters = cond_name.split('_')
        print(cond_parameters)
        
        with open(os.path.join(OUTPUT_PATH, file), 'r') as f:
            f1 = f.readlines()[-1].split('\t')[1]
        
        cond_parameters.append(f1)
        runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)
results_all['epochs'] = results_all['epochs'].astype(float).astype(int)


# CV with Test Set
for DATA_PATH, OUTPUT_PATH in [['data', '../results/instructABSA']]:
    for LR_SETTING in ['1000', '500', 'full']:

        results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == LR_SETTING, results_all['dataset'] == DATASET, results_all['split'] == '0'])].sort_values(by = ['f1-micro'], ascending = False)
        results_sub = results_sub.reset_index()
    
        print(results_sub.head(3))
        
        EPOCHS = results_sub.at[0, 'epochs']
        STEPS = True if EPOCHS != BASE_EPOCHS else False
        
        if STEPS == True:
            # Train split lenghts for cross evaluation (4/6 of original train sizes)
            if DATASET == 'GERestaurant':
                dataset_base_len = 1436                            
            elif DATASET == 'rest-16':
                dataset_base_len = 1138

            low_resource_steps_per_e =  (dataset_base_len if LR_SETTING == 0 else (LR_SETTING * 4/6)) / BATCH_SIZE

            MAX_STEPS = int((dataset_base_len * BASE_EPOCHS) / BATCH_SIZE)

            # Transform to #Epochs for Output-Path
            EPOCHS = round(MAX_STEPS / low_resource_steps_per_e)
        
        for SPLIT in [1,2,3,4,5]:
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 instructABSA/run_model.py \
                        -mode train \
                        -model_checkpoint {MODEL} \
                        -task joint \
                        -output_dir instructABSA/Models \
                        -inst_type 2 \
                        -id_tr_data_path instructABSA/{DATA_PATH}/{DATASET}/split_{SPLIT}/train_{LR_SETTING}.csv \
                        -id_te_data_path instructABSA/{DATA_PATH}/{DATASET}/split_{SPLIT}/test_{LR_SETTING}.csv \
                        -evaluation_strategy no \
                        -learning_rate {LEARNING_RATE} \
                        -per_device_train_batch_size {BATCH_SIZE} \
                        -num_train_epochs {BASE_EPOCHS if STEPS == False else EPOCHS} \
                        -steps {MAX_STEPS} \
                        -lr_setting {LR_SETTING} \
                        -dataset {DATASET} \
                        -split {SPLIT}"
            process = subprocess.Popen(command, shell=True)
            process.wait()
        
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 instructABSA/run_model.py \
                        -mode eval  \
                        -model_checkpoint {MODEL} \
                        -task joint \
                        -output_dir instructABSA/Models \
                        -inst_type 2 \
                        -id_tr_data_path instructABSA/{DATA_PATH}/{DATASET}/split_{SPLIT}/train_{LR_SETTING}.csv \
                        -id_te_data_path instructABSA/{DATA_PATH}/{DATASET}/split_{SPLIT}/test_full.csv \
                        -evaluation_strategy no \
                        -learning_rate {LEARNING_RATE} \
                        -per_device_train_batch_size {BATCH_SIZE} \
                        -num_train_epochs {BASE_EPOCHS if STEPS == False else EPOCHS} \
                        -steps {MAX_STEPS} \
                        -lr_setting {LR_SETTING} \
                        -dataset {DATASET} \
                        -split {SPLIT} \
                        -output_path {OUTPUT_PATH}"
            process = subprocess.Popen(command, shell=True)
            process.wait()