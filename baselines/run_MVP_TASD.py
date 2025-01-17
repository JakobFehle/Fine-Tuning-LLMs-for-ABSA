import subprocess
import sys
import os


###
# CV
###

LEARNING_RATE = 1e-4
DATA_PATH = 'mvp/data/'
TASK = 'tasd'
MODEL = 't5-base'
SEED = 42

output_path = "../results/mvp/"
OUTPUT_PATH = os.path.abspath(output_path)


CV_SETTINGS = [ # LR_SETTING, BATCH_SIZE, EPOCHS
    ['500', 8, 50],
    ['1000', 8, 30],
    ['full', 16, 20],
]

DATASET = ['rest-16', 'GERestaurant'][int(sys.argv[1])]

for SPLIT in [1,2,3,4,5]:
    for LR_SETTING, BATCH_SIZE, EPOCHS in CV_SETTINGS:
        OUT_DIR = f"{OUTPUT_PATH}/{DATASET}_{TASK}_{SPLIT}_{LR_SETTING}_{EPOCHS}"
        command = [
            sys.executable,  # The Python interpreter
            "mvp/src/classifier.py",  # The script to run
            "--data_path", DATA_PATH,
            "--dataset", DATASET,
            "--split", str(SPLIT),
            "--lr_setting", LR_SETTING,
            "--model_name_or_path", MODEL,
            "--output_dir", OUT_DIR,
            "--num_train_epochs", str(EPOCHS),
            "--save_top_k", "0",
            "--task", TASK,
            "--top_k", "5",
            "--ctrl_token", "post",
            "--multi_path",
            "--num_path", "5",
            "--seed", str(SEED),
            "--train_batch_size", str(BATCH_SIZE),
            "--gradient_accumulation_steps", "1",
            "--learning_rate", str(LEARNING_RATE),
            "--lowercase" if DATASET == 'rest-16' else '', # Disable Lowercasing for German
            "--sort_label",
            "--data_ratio", "1.0",
            "--check_val_every_n_epoch", str(EPOCHS),
            "--agg_strategy", "vote",
            "--eval_batch_size", "32",
            "--constrained_decode",
            "--do_train"
        ]
        
        # Add the environment variable as a prefix
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
        
        # Run the subprocess
        process = subprocess.Popen(command, env=env)
        process.wait()
        