import subprocess
import sys
import os
import pandas as pd
import numpy as np
import glob

###
# HT
###
LEARNING_RATE = 2e-5
EPOCHS = 30
BATCH_SIZE = 24

DATASET, MODEL_NAME = [['GERestaurant', 'gbert-base'], ['rest-16', 'uncased_L-12_H-768_A-12']][int(sys.argv[1])]

for DATA_PATH, OUT_DIR in [['tas_bert/data', '../results/tas_bert/']]:
    for LR_SETTING in ['500', '1000', 'full']:
        for SPLIT in [0]:
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 tas_bert/TAS_BERT_joint.py \
                        --data_dir {DATA_PATH} \
                        --output_dir {OUT_DIR} \
                        --vocab_file tas_bert/{MODEL_NAME}/vocab.txt \
                        --bert_config_file tas_bert/{MODEL_NAME}/bert_config.json \
                        --init_checkpoint tas_bert/{MODEL_NAME}/pytorch_model.bin \
                        --tokenize_method word_split \
                        --use_crf \
                        --eval_test \
                        --do_lower_case \
                        --max_seq_length 128 \
                        --train_batch_size 24 \
                        --eval_batch_size 8 \
                        --learning_rate {LEARNING_RATE} \
                        --num_train_epochs {EPOCHS} \
                        --dataset {DATASET} \
                        --split {SPLIT} \
                        --lr_setting {LR_SETTING}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 tas_bert/evaluation_for_TSD_ASD_TASD.py \
                        --output_dir {OUT_DIR}{DATASET}/three_joint/BIO/{DATASET}_{LR_SETTING}_{SPLIT}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.0 \
                        --num_epochs {EPOCHS} \
                        --tag_schema BIO"
            process = subprocess.Popen(command, shell=True)
            process.wait()


###
# Cross Evaluation Phase
###

col_names = ['dataset', 'lr_setting', 'split', 'learning_rate', 'batch_size', 'epochs', 'f1-micro']
folder_names = [folder for folder in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, folder)) and folder != '.ipynb_checkpoints']

result_files = glob.glob(f"{OUT_DIR}/*/three_joint/BIO/*/results.txt")

runs = []
for results_file in result_files:
    print(results_file)
    try:
        cond_parameters = results_file.split('/')[-2]
        print(cond_parameters)
        cond_parameters = cond_parameters.split('_')[:5]
        print(cond_parameters)
        
        if cond_parameters[2] == '0':
            df = pd.read_csv(results_file, sep = '\t')
            df = df.set_index(df.columns[0])
    
            max_epoch = df['f1'].idxmax()
            
            cond_parameters.extend([max_epoch, df.loc[max_epoch, 'f1']])
            runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)
print(results_all)

# CV with Test Set
for DATA_PATH, OUTPUT_PATH in [['tas_bert/data', '../results/tas_bert/']]:
    for LR_SETTING in ['500', '1000', 'full']:

        results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == LR_SETTING, results_all['dataset'] == DATASET])].sort_values(by = ['f1-micro'], ascending = False)
        results_sub = results_sub.reset_index()

        EPOCHS = int(results_sub.at[0, 'epochs'])
        
        for SPLIT in [1,2,3,4,5]:
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 tas_bert/TAS_BERT_joint.py \
            --data_dir {DATA_PATH} \
            --output_dir {OUTPUT_PATH} \
            --vocab_file tas_bert/{MODEL_NAME}/vocab.txt \
            --bert_config_file tas_bert/{MODEL_NAME}/bert_config.json \
            --init_checkpoint tas_bert/{MODEL_NAME}/pytorch_model.bin \
            --tokenize_method word_split \
            --use_crf \
            --eval_test \
            --do_lower_case \
            --max_seq_length 128 \
            --train_batch_size 24 \
            --eval_batch_size 8 \
            --learning_rate {LEARNING_RATE} \
            --num_train_epochs {EPOCHS} \
            --dataset {DATASET} \
            --split {SPLIT} \
            --lr_setting {LR_SETTING}"
            process = subprocess.Popen(command, shell=True)
            process.wait()
    
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 tas_bert/evaluation_for_TSD_ASD_TASD.py \
            --output_dir {OUT_DIR}{DATASET}/three_joint/BIO/{DATASET}_{LR_SETTING}_{SPLIT}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.0 \
            --num_epochs {EPOCHS} \
            --tag_schema BIO"
            process = subprocess.Popen(command, shell=True)
            process.wait()

        