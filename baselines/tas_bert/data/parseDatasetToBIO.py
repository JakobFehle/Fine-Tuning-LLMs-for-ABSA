import subprocess
import sys
import os

for DATA_PATH in ['']:
    for DATASET in ['rest-16', 'GERestaurant']:
        for LR_SETTING in ['500', '1000', 'full']:
            for SPLIT in [0,1,2,3,4,5]:
                command = f"python3 data_preprocessing_for_TAS.py --dataset {DATASET} --lr_setting {LR_SETTING} --split {SPLIT}"
                process = subprocess.Popen(command, shell=True)
                process.wait()