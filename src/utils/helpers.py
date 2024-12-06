from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field

FEW_SHOT_CONFIGS = ['25shot_rand', '10shot_rand', '5shot_rand', '25shot_asp', '10shot_asp', '5shot_asp']

STOP_WORDS_LLM = ["neutral)]\n", "positiv)]\n", "negativ)]\n"]
STOP_WORDS_OPENAI = ['Satz:', '\n\nSatz:', '\n']

CHECKPOINT_DICT_FULL = {
    'checkpoint-426': '1',
    'checkpoint-852': '2',
    'checkpoint-1278': '3',
    'checkpoint-1704': '4',
    'checkpoint-2130': '5',
    'checkpoint-2556': '6',
    'checkpoint-2982': '7',
    'checkpoint-3408': '8',
    'checkpoint-3834': '9',
    'checkpoint-4260': '10',
    'checkpoint-4686': '11',
    'checkpoint-5112': '12',
    'checkpoint-5538': '13',
    'checkpoint-5964': '14',
    'checkpoint-6390': '15',
    'checkpoint-6816': '16',
    'checkpoint-7242': '17',
    'checkpoint-7668': '18',
    'checkpoint-8094': '19',
    'checkpoint-8520': '20',
    'checkpoint-8946': '21',
    'checkpoint-9372': '22',
    'checkpoint-9798': '23',
    'checkpoint-10224': '24',
    'checkpoint-10650': '25',
    'checkpoint-11076': '26',
    'checkpoint-11502': '27',
    'checkpoint-11928': '28',
    'checkpoint-12354': '29',
    'checkpoint-12780': '30'
}

CHECKPOINT_DICT_500 = {
    'checkpoint-63': '1',
    'checkpoint-126': '2',
    'checkpoint-189': '3',
    'checkpoint-252': '4',
    'checkpoint-315': '5',
    'checkpoint-378': '6',
    'checkpoint-441': '7',
    'checkpoint-504': '8',
    'checkpoint-567': '9',
    'checkpoint-630': '10'
}

CHECKPOINT_DICT_1000 = {
    'checkpoint-125': '1',
    'checkpoint-250': '2',
    'checkpoint-375': '3',
    'checkpoint-500': '4',
    'checkpoint-625': '5',
    'checkpoint-750': '6',
    'checkpoint-875': '7',
    'checkpoint-1000': '8',
    'checkpoint-1125': '9',
    'checkpoint-1250': '10'
}

CHECKPOINT_DICT_2000 = {
    'checkpoint-250': '1',
    'checkpoint-500': '2',
    'checkpoint-750': '3',
    'checkpoint-1000': '4',
    'checkpoint-1250': '5',
    'checkpoint-1500': '6',
    'checkpoint-1750': '7',
    'checkpoint-2000': '8',
    'checkpoint-2250': '9',
    'checkpoint-2500': '10'
}
@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-13b-hf",
        metadata={"help": 'Base model name for training or inference'}
    )
    seed: int = field(
        default=42,
        metadata={"help": 'Seed to ensure reproducability.'}
    )
    model_task: Optional[str] = field(
        default="acsa",
        metadata={"help": "Which ABSA Task the model was trained on. ['acd', 'acsa', 'acsd']"}
    )
    model_shots: Optional[str] = field(
        default="",
        metadata={"help": 'Amount and style of few shot examples the model is trained on.'}
    )
    model_prompt_style: Optional[str] = field(
        default="short",
        metadata={"help": 'Style of the prompt the model is trained on.'}
    )
    model_lang: Optional[str] = field(
        default="ger",
        metadata={"help": 'Language of the prompt.'}
    )
    model_quant: int = field(
        default=4,
        metadata={"help": "In which bit precision the model was trained on."}
    )
    model_lr: float = field(
        default=0.0005, 
        metadata={"help": 'The learning rate the model was trained on.'}
    )
    model_lora_alpha: int = field(
        default=16, 
        metadata={"help": 'The lora alpha value the model was trained on.'}
    )
    model_lora_r: int = field(
        default=16, 
        metadata={"help": 'The lora r value the model was trained on.'}
    )
    model_lora_dropout: float = field(
        default=0.05, 
        metadata={"help": 'The lora dropout the model was trained on.'}
    )
    quant: int = field(
        default=16,
        metadata={"help": "How many bits to use for quantization."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Compute dtype of the model."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Compute dtype of the model."}
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": 'If to enable flash attention.'}
    )
    run_tag: Optional[str] = field(
        default='',
        metadata={"help": 'Additional run-specific tag for the output-folder'}
    ) 
    epoch: Optional[int] = field(
        default=1,
        metadata={"help": 'Epoch checkpoint of the model'}
    )
    
@dataclass
class DataArgs:
    dataset: str = field(
        default='hotel',
        metadata={"help": "Which dataset to use: ['hotel', 'rest' or 'germeval']"}
    )
    lang: str = field(
        default="ger",
        metadata={"help": 'Language of the prompt.'}
    )
    shots: str = field(
        default="",
        metadata={"help": 'Amount and style of few shot examples for evaluation.'}
    )
    prompt_style: str = field(
        default="short",
        metadata={"help": 'Style of the prompt for evaluation.'}
    )
    low_resource_setting: int = field(
        default=0,
        metadata={"help": 'Amount of samples to train on. (0 -> full dataset; 500 samples; 1000 samples; 2000 samples)'}
    )
    split: int = field(
        default=0,
        metadata={"help": 'Which split of the dataset to use.'}
    )
    wandb: bool = field(
        default=True,
        metadata={"help": 'If to report to wandb'}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum context length during training and inference."},
    )
    task: Optional[str] = field(
        default="acsa",
        metadata={"help": "Which ABSA Task the model was trained on. ['acd', 'acsa', 'acsd']"}
    )
    test: Optional[bool] = field(
        default=False
    )
    api: Optional[bool] = field(
        default=False,
        metadata={"help": 'If prompt format has to be in OpenAI-API format.'}
    )
    json: Optional[bool] = field(
        default=False
    )
    original_split: Optional[bool] = field(
        default=False
    )

@dataclass
class SFTTrainingArgs:
    per_device_train_batch_size: int = field(
        default=4, 
        metadata={"help": 'The training batch size per GPU.'}
    )
    per_device_eval_batch_size: int = field(
        default=4, 
        metadata={"help": 'The evaluation batch size per GPU.'}
    )
    gradient_accumulation_steps: int = field(
        default=1, 
        metadata={"help": 'Amount of gradients to accumulate before performing an optimizer step'}
    )
    learning_rate: float = field(
        default=0.0005, 
        metadata={"help": 'The learning rate.'}
    )
    lr_scheduler_type: str = field(
        default='constant', 
        metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'}
    )
    group_by_length: bool = field(
        default=True, 
        metadata={"help": 'Group sequences into batches with same length.'}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": 'Amount of epochs to train.'}
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": 'The frequency of update steps after which to log the loss.'}
    )
    save_strategy: str = field(
        default='epoch', 
        metadata={"help": 'When to save checkpoints.'}
    )
    evaluation_strategy: str = field(
        default='epoch', 
        metadata={"help": 'When to compute eval loss on eval dataset.'}
    )
    optim: str = field(
        default='paged_adamw_32bit', 
        metadata={"help": 'The optimizer to be used.'}
    )


@dataclass
class TrainingConfig:
    lora_r: int = field(
        default=16,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=24000,
        metadata={"help": "Free memory per gpu."}
    )
    from_checkpoint: Optional[bool] = field(
        default= False
    )
    
@dataclass
class GenerationArgs:
    max_new_tokens: Optional[int] = field(
        default=100,
        metadata={"help": "Maximum sequence length for new tokens during inference."},
    )
    do_sample: Optional[bool] = field(
        default=False
    )
    use_cache: Optional[bool] = field(
        default=True
    )
    top_k: Optional[float] = field(
        default=-1
    )
    top_p: Optional[float] = field(
        default=1
    )
    temperature: Optional[float] = field(
        default=0
    )