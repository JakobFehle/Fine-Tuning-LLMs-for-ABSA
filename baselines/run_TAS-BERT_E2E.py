import subprocess
import sys
import os

# Enable HT
ORIGINAL_SPLIT = False 
BATCH_SIZE = 8
EPOCHS = 4

###
# HT
###
LEARNING_RATE = 2e-5
EPOCHS = 30
BATCH_SIZE = 24

DATASET, MODEL_NAME = [['GERestaurant', 'gbert-base'], ['rest-16', 'uncased_L-12_H-768_A-12']][int(sys.argv[1])]

for DATA_PATH, OUT_DIR in [['data', '../results/tas_bert/']]:
    for LR_SETTING in ['500', '1000', 'full']:
        for SPLIT in [0]:
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 TAS_BERT_joint.py \
                        --data_dir {DATA_PATH} \
                        --output_dir {OUT_DIR} \
                        --vocab_file {MODEL_NAME}/vocab.txt \
                        --bert_config_file {MODEL_NAME}/bert_config.json \
                        --init_checkpoint {MODEL_NAME}/pytorch_model.bin \
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

            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 evaluation_for_TSD_ASD_TASD.py \
                        --output_dir {OUT_DIR}/{DATASET}/three_joint/BIO/{DATASET}_{LR_SETTING}_{SPLIT}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.0 \
                        --num_epochs {EPOCHS} \
                        --tag_schema BIO"
            process = subprocess.Popen(command, shell=True)
            process.wait()


###
# Cross Evaluation Phase
###

METHOD = 'tas_bert'
RESULTS_PATH = '../results'

col_names = ['dataset', 'lr-setting', 'split', 'learning-rate', 'batch_size', 'epochs', 'f1-micro']
folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH, METHOD)) if os.path.isdir(os.path.join(RESULTS_PATH, METHOD, folder)) and folder != '.ipynb_checkpoints']

runs = []

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')[:4]
        
        if cond_parameters[2] == '0':
            df = pd.read_csv(os.path.join(RESULTS_PATH, METHOD, folder_name, 'results.txt'), sep = '\t')
            df = df.set_index(df.columns[0])
    
            max_epoch = df['f1'].idxmax()
            
            cond_parameters.extend([max_epoch, df.loc[max_epoch, 'f1']])
            runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)

# CV with Test Set
for DATA_PATH, OUTPUT_PATH in [['data', '../results/tas_bert/']]:
    for LR_SETTING in ['full', '1000', '500']:

        results_sub = results_all[np.logical_and.reduce([results_all['lr_setting'] == LR_SETTING, results_all['dataset'] == DATASET])].sort_values(by = ['f1-micro'], ascending = False)
        results_sub = results_sub.reset_index()

        EPOCHS = int(results_sub.at[0, 'epochs'])
        
        for SPLIT in [1,2,3,4,5]:
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 TAS_BERT_joint.py \
            --data_dir {DATA_PATH} \
            --output_dir {OUTPUT_PATH} \
            --vocab_file {MODEL_NAME}/vocab.txt \
            --bert_config_file {MODEL_NAME}/bert_config.json \
            --init_checkpoint {MODEL_NAME}/pytorch_model.bin \
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
    
            command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 evaluation_for_TSD_ASD_TASD.py \
            --output_dir {OUT_DIR}/{DATASET}/three_joint/BIO/{DATASET}_{LR_SETTING}_{SPLIT}_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}.0 \
            --num_epochs {EPOCHS} \
            --tag_schema BIO"
            process = subprocess.Popen(command, shell=True)
            process.wait()

        