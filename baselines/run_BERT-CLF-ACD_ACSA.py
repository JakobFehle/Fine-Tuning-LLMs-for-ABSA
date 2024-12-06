import subprocess
import sys
import os
import pandas as pd
import numpy as np

# Enable HT
BATCH_SIZE = 16
BASE_EPOCHS = 3
MAX_STEPS = 0
RESULTS_PATH = '../results/bert-clf/'

###
# Hyperparameter Validation Phase
###

DATASET, MODEL_NAME = [['GERestaurant', 'deepset/gbert-large'], ['rest-16', 'google-bert/bert-large-uncased']][int(sys.argv[1])]

for LR_SETTING in [0, 1000, 500]:
    for TASK in ['acd', 'acsa']:
        HT_LEARNING_RATES = [2e-5] if DATASET == 'GERestaurant' else [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]
        for LEARNING_RATE in HT_LEARNING_RATES:
            for SPLIT in [0]: 
                # Calculate amount of Training Steps
                DO_STEPS = [False] if LR_SETTING == 0 else [True, False]
                for STEPS in DO_STEPS:
                    if STEPS == True:
                        # Train split lenghts for hyperparameter tuning phase (5/6 of original train sizes)
                        if DATASET == 'GERestaurant':
                            dataset_base_len = 1795                            
                        elif DATASET == 'rest-16':
                            dataset_base_len = 1423
    
                        low_resource_steps_per_e =  (dataset_base_len if LR_SETTING == 0 else (LR_SETTING * 5/6)) / BATCH_SIZE
    
                        MAX_STEPS = int((dataset_base_len * BASE_EPOCHS) / BATCH_SIZE)
    
                        # Transform to #Epochs for Output-Path
                        EPOCHS = round(MAX_STEPS / low_resource_steps_per_e)
                    command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 bert_clf/baseline_acd_acsa.py --task {TASK} --lr_setting {LR_SETTING} --split {SPLIT} --dataset {DATASET} --model_name {MODEL_NAME} --learning_rate {LEARNING_RATE} --batch_size {BATCH_SIZE} --epochs {BASE_EPOCHS if STEPS == False else EPOCHS} --steps {MAX_STEPS} --output_path {RESULTS_PATH}"
                    process = subprocess.Popen(command, shell=True)
                    process.wait()


###
# Cross-Evaluation Phase
###

METHOD = 'mlcf'
RESULTS_PATH = '../results'

col_names = ['task', 'dataset', 'lr-setting', 'split', 'learning-rate', 'batch_size', 'epochs', 'f1-micro', 'f1-macro', 'accuracy']
folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH, METHOD)) if os.path.isdir(os.path.join(RESULTS_PATH, METHOD, folder)) and folder != '.ipynb_checkpoints']

runs = []

for folder_name in folder_names:
    try:
        
        cond_name = folder_name.split('/')[-1]
        cond_parameters = cond_name.split('_')
        

        if cond_parameters[0] == 'acd':
            filename = 'metrics_asp.tsv'
        elif cond_parameters[0] == 'acsa':
            filename = 'metrics_asp_pol.tsv'
        elif cond_parameters[0] == 'e2e':
            filename = 'metrics_pol.tsv'
        elif cond_parameters[0] == 'acsd':
            filename = 'metrics_phrases.tsv'

        df = pd.read_csv(os.path.join(RESULTS_PATH, METHOD, folder_name,filename), sep = '\t')
        df = df.set_index(df.columns[0])
        
        cond_parameters.append(df.loc['Micro-AVG', 'f1'])
        cond_parameters.append(df.loc['Macro-AVG', 'f1'])
        cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
        runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)
results_all['learning-rate'] = results_all['learning-rate'].astype(float)

for LR_SETTING in [0, 1000, 500]:
    for TASK in ['acd', 'acsa']:
        for SPLIT in [1,2,3,4,5]:
            results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == str(LOW_RESOURCE_SETTING), results_all['dataset'] == DATASET, results_all['task'] == TASK, results_all['split'] == '0'])].sort_values(by = ['f1-micro'], ascending = False)
            results_sub = results_sub.reset_index()
        
            print(results_sub.head(3))
            
            LEARNING_RATE = results_sub.at[0, 'learning_rate']
            EPOCHS = int(results_sub.at[0, 'epoch'])

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
            
            command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 bert_clf/baseline_acd_acsa.py --task {TASK} --lr_setting {LR_SETTING} --split {SPLIT} --dataset {DATASET} --model_name {MODEL_NAME} --learning_rate {LEARNING_RATE} --batch_size {BATCH_SIZE} --epochs {BASE_EPOCHS if STEPS == False else EPOCHS} --steps {MAX_STEPS} --output_path {RESULTS_PATH}"
                    process = subprocess.Popen(command, shell=True)
                    process.wait()