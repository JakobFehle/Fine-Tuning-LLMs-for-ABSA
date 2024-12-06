# Code derived from https://github.com/artidoro/qlora

import pandas as pd
import random
import bitsandbytes as bnb
import torch
import transformers
import numpy as np
import os
import wandb
import time

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import set_seed
from utils.config import Config
from utils.preprocessing import loadDataset, createPrompts
from unsloth import FastLlamaModel

HF_TOKEN = ""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
       
def find_all_linear_names(model, bits):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
    
def loadModelForTraining(config):
    if config.bf16:
        compute_dtype = torch.bfloat16
    elif config.quant == 8 or config.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    print(f'loading model {config.model_name_or_path}...')
    model, tokenizer = FastLlamaModel.from_pretrained(
        config.model_name_or_path, 
        max_seq_length=config.max_seq_length,
        dtype=compute_dtype,
        load_in_4bit = True if config.quant == 4 else False
    )

    tokenizer.add_special_tokens({'pad_token': tokenizer.convert_ids_to_tokens(model.config.eos_token_id)})

    model = FastLlamaModel.get_peft_model(model, 
                                            target_modules=find_all_linear_names(model, config.quant),
                                            lora_alpha=config.lora_alpha,
                                            r=config.lora_r,
                                            bias="none",
                                            use_gradient_checkpointing = True,
                                            random_state=config.seed)

    return model, tokenizer
    
def main():
    
    config = Config()
    print(config)

    set_seed(config.seed)
    
    # Create Training Config based on Parameters
    train_config = '_'.join([config.task, config.dataset, config.prompt_style,  str(config.learning_rate), str(config.lora_r), str(config.lora_alpha), str(config.lora_dropout), str(config.split), str(config.low_resource_setting) if config.low_resource_setting != 0 else ('orig' if config.original_split == True else 'full')])

    train_config = f'{train_config}-{str(config.run_tag)}' if str(config.run_tag) != '' else train_config

    if config.wandb == True:
        wandb_project = "absa-prompt-finetune"
        os.environ['WANDB_PROJECT'] = wandb_project

    model, tokenizer = loadModelForTraining(config)
    
    df_train, df_test, label_space = loadDataset(config.data_path, config.dataset, config.low_resource_setting, config.task, config.split,  config.original_split)
    prompts_train, prompts_test, ground_truth_labels = createPrompts(df_train, df_test, config, eos_token = tokenizer.eos_token)
    
    prompts_train = pd.DataFrame(prompts_train, columns = ['text'])
    prompts_test = pd.DataFrame(prompts_test, columns = ['text'])
    
    ds_train = Dataset.from_pandas(prompts_train)
    ds_test = Dataset.from_pandas(prompts_test)
    
    ds_train = ds_train.map(lambda x: {'length': len(x['text']) + len(x['text'])})

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        packing=False,
        dataset_text_field="text",
        neftune_noise_alpha=config.neftune_noise_alpha,
        max_seq_length=config.max_seq_length,
        args=TrainingArguments(
            bf16=True if config.bf16 else False,
            fp16=False if config.bf16 else True,
            ddp_find_unused_parameters=False,
            output_dir = '../models/' + train_config,
            seed = config.seed,
            report_to = "wandb" if config.wandb==True else "none",
            run_name = f"{train_config}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            disable_tqdm = False,
            max_grad_norm = 0.3,
            per_device_train_batch_size = config.per_device_train_batch_size,
            per_device_eval_batch_size = config.per_device_eval_batch_size,
            gradient_accumulation_steps = config.gradient_accumulation_steps,
            learning_rate = config.learning_rate,
            lr_scheduler_type = config.lr_scheduler_type,
            group_by_length = config.group_by_length,
            num_train_epochs = config.num_train_epochs,
            logging_steps = config.logging_steps,
            save_strategy = config.save_strategy,
            evaluation_strategy = config.evaluation_strategy,
            optim = config.optim
        ),
    )
    
    # Start training and track computation time
    time_1 = time.time()
    train_result = trainer.train(resume_from_checkpoint=config.from_checkpoint)
    time_2 = time.time()

    # Save model configuration and training time
    with open('../models/' + train_config + '/train_config.txt', 'w') as f:
        for k,v in vars(config).items():
            f.write(f"{k}: {v}\n")
        training_time = time_2 - time_1
        training_time = str(timedelta(seconds=training_time))
        f.write(f"Training time: {training_time}\n")

if __name__ == "__main__":
    main()