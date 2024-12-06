# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
import pdb
from manager import *
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold

gm = GPUManager()
device = gm.auto_choice(mode=0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

from bert_utils.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modeling import GCNclassification
from bert_utils.tokenization import BertTokenizer
from bert_utils.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import *
from eval_metrics import *
import gc

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input source data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                     "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                     "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="model to choose.")
    parser.add_argument("--split",
                        default=None,
                        type=int,
                        required=True,
                        help="Dataset split.")
    parser.add_argument("--low_resource_setting",
                        default=None,
                        type=int,
                        required=True,
                        help="Low resource setting.")
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True)
    
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    
    args = parser.parse_args()

    if args.task_name == 'Rest16ACSA':
        args.domain_type = 'restaurant'
        args.bert_model = 'bert-base-uncased'
    elif args.task_name == 'GERestaurantACSA':
        args.domain_type = 'restaurant'
        args.bert_model = 'gbert-base'
        
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list[0])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model_dict = {
        'GCN': GCNclassification,
    }

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:

        # Prepare data loader
        train_examples = processor.get_train_examples(args.data_dir, args.split, args.low_resource_setting)
        print(label_list)
        train_category_map, train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode,task_name)
        # plt.matshow(train_category_map)
        # plt.show()
        # pdb.set_trace()
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            category_label_ids = torch.tensor([f.category_label_id for f in train_features], dtype=torch.long)
            sentiment_label_ids = torch.tensor([f.sentiment_label_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(input_ids, input_mask, segment_ids, category_label_ids, sentiment_label_ids)
        
        eval_examples = processor.get_dev_examples(args.data_dir, args.split, args.low_resource_setting)

        eval_category_map, eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode,task_name)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_category_label_ids = torch.tensor([f.category_label_id for f in eval_features], dtype=torch.long)
            all_sentiment_label_ids = torch.tensor([f.sentiment_label_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_category_label_ids, all_sentiment_label_ids)
        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Prepare optimizer

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        model = model_dict[args.model_type].from_pretrained(args.bert_model, num_labels=num_labels)
        if args.local_rank == 0:
            torch.distributed.barrier()

        if args.fp16:
            model.half()

        model.to(device)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        train_category_map_gpu = [torch.tensor(train_category_map[i], dtype=torch.float).to(device) for i in range(len(train_category_map))]

        for _e in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)
                _input_ids, _input_mask, _segment_ids, _category_label_ids, _sentiment_label_ids = batch

                # define a new function to compute loss values for both output_modes
                if args.model_type != 'Baseline_1' and args.model_type != 'AddOD':
                    loss, c_loss, s_loss, category_logits, sentiment_logits = model(_e, train_category_map_gpu, _input_ids, token_type_ids=_segment_ids, attention_mask=_input_mask,
                        cate_labels=_category_label_ids, senti_labels=_sentiment_label_ids)
                else:
                    logits, loss = model(_input_ids, token_type_ids=_segment_ids, attention_mask=_input_mask, senti_labels=_sentiment_label_ids)
                    c_loss = None
                    s_loss = None

                if step % 30 == 0:
                    print('Loss is {} .'.format(loss))
                    print('cate_loss is {} .'.format(c_loss))
                    print('senti_loss is {} .\n'.format(s_loss))
                step += 1
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                else:
                    loss = loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += _input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        model.eval()

        result, fix_result = hier_pred_eval(args, 1, train_category_map_gpu, logger, model, eval_dataloader, device, task_name, eval_type='test')

        torch.cuda.empty_cache()
        gc.collect()

        if args.task_name == 'Rest16ACSA':
            path_suffix = f'{args.task}_rest-16'
        elif args.task_name == 'GERestaurantACSA':
            path_suffix = f'{args.task}_GERestaurant'

        output_dir = f'{args.output_dir}/{path_suffix}_{args.low_resource_setting}_{args.split}_{args.learning_rate}_{args.train_batch_size}_{args.num_train_epochs}'
        
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)
        
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        cate_output_eval_file = os.path.join(output_dir, "cate_eval_results.txt")

        with open(cate_output_eval_file, "w") as writer:
            logger.info("***** Category Eval results *****")
            for key in sorted(fix_result.keys()):
                logger.info("  %s = %s", key, str(fix_result[key]))
                writer.write("%s = %s\n" % (key, str(fix_result[key])))

if __name__ == "__main__":
    main()
