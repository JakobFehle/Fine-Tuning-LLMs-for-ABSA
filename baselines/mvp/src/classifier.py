import argparse
import os
import sys
import gc
import logging
import pickle
import random
import json
import time
import re
from itertools import permutations
from functools import partial
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, LearningRateMonitor

from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.file_utils import ModelOutput
from transformers.models.t5.modeling_t5 import *

from const import *

from tqdm import tqdm



def extract_spans_para(seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it' or at.lower() == 'es':
                at = 'null'
        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))

    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    if verbose:
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, verbose=True, task="asqp"):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(gold_seqs[i], 'gold')
        pred_list = extract_spans_para(pred_seqs[i], 'pred')
        if (task == "tasd"):
          gold_list = [tup[:-1] for tup in gold_list]
          pred_list = [tup[:-1] for tup in pred_list]

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)
    scores["all_preds"] = all_preds
    scores["all_labels"] = all_labels

    return scores, all_labels, all_preds


def get_element_tokens(task):
    dic = {
        "aste":
            ["[A]", "[O]", "[S]"],
        "tasd":
            ["[A]", "[C]", "[S]"],
        "aocs":
        ["[A]", "[O]", "[C]", "[S]"],
        "asqp":
            ["[A]", "[O]", "[C]", "[S]"],
    }
    return dic[task]

optim_orders_all = None

def get_orders(task, data, data_type, args, sents, labels):
    # Empfehlung: Optimale Reihenfolge der Elemente vorab 1x berechnen. 
    # uncomment to calculate orders from scratch
    
    if torch.cuda.is_available():
          device = torch.device('cuda')
    else:
          device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = MyT5ForConditionalGenerationScore.from_pretrained(
            "t5-base").to(device)

    # if args.dataset in optim_orders_all_predefined[task].keys():
    #     if args.
    
    global optim_orders_all

    if optim_orders_all == None:
        optim_orders_all = choose_best_order_global(sents, labels, model,
                                             tokenizer, device,
                                             args.task)

    #     print(optim_orders_all)

    if args.single_view_type == 'rank':
           orders = optim_orders_all#[task]["rest16"] # delete [task][data] falls selber berechnet werden
    elif args.single_view_type == 'rand':
           orders = [random.Random(args.seed).choice(
               optim_orders_all[task][data])]
    elif args.single_view_type == "heuristic":
           orders = heuristic_orders[task]

    return orders




def cal_entropy(inputs, preds, model_path, tokenizer, device=torch.device('cuda')):
    all_entropy = []
    model = MyT5ForConditionalGenerationScore.from_pretrained(model_path).to(
        device)
    batch_size = 8
    _inputs = [' '.join(s) for s in inputs]
    _preds = [' '.join(s) for s in preds]
    for id in range(0, len(inputs), batch_size):
        in_batch = _inputs[id: min(id + batch_size, len(inputs))]
        pred_batch = _preds[id: min(id + batch_size, len(inputs))]
        assert len(in_batch) == len(pred_batch)
        tokenized_input = tokenizer.batch_encode_plus(in_batch,
                                                      max_length=200,
                                                      padding="max_length",
                                                      truncation=True,
                                                      return_tensors="pt")
        tokenized_target = tokenizer.batch_encode_plus(pred_batch,
                                                       max_length=200,
                                                       padding="max_length",
                                                       truncation=True,
                                                       return_tensors="pt")

        target_ids = tokenized_target["input_ids"].to(device)

        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        outputs = model(
            input_ids=tokenized_input["input_ids"].to(device),
            attention_mask=tokenized_input["attention_mask"].to(device),
            labels=target_ids,
            decoder_attention_mask=tokenized_target["attention_mask"].to(device))

        loss, entropy = outputs[0]
        all_entropy.extend(entropy)
    return all_entropy


def order_scores_function(quad_list, cur_sent, model, tokenizer, device, task):
    q = get_element_tokens(task)

    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)

    for each_order in all_orders:
        cur_order = "  ".join(each_order) + " "
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(all_inputs,
                                                  max_length=200,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt")
    tokenized_target = tokenizer.batch_encode_plus(all_targets,
                                                   max_length=200,
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    target_ids = tokenized_target["input_ids"].to(device)

    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device))

    loss, entropy = outputs[0]
    results = {}
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}

    return results


def choose_best_order_global(sents, labels, model, tokenizer, device, task):
    q = get_element_tokens(task)
    all_orders = permutations(q)
    all_orders_list = []
    scores = []

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        scores.append(0)

    for i in range(len(sents)):
        label = labels[i]
        sent = sents[i]

        quad_list = []
        for _tuple in label:
            # parse ASTE tuple
            if task == "aste":
                _tuple = parse_aste_tuple(_tuple, sent)

            at, ac, sp, ot = get_task_tuple(_tuple, task, 'en' if args.dataset == 'rest-16' else 'ger')

            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            element_list = []
            for key in q:
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)

            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:4])
                    content.append(e[4:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        order_scores = order_scores_function(quad_list, sent, model, tokenizer,
                                             device, task)

        for e in order_scores:
            e = e[:-1].replace("  ", " ")
            index = all_orders_list.index(e)
            scores[index] += order_scores[e.replace(" ", "  ")+" "]['entropy']

    indexes = np.argsort(np.array(scores))  # [::-1]
    returned_orders = []
    for i in indexes:
        returned_orders.append(all_orders_list[i])

    return returned_orders


def parse_aste_tuple(_tuple, sent):
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # parse at
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # parse ot
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res


def get_task_tuple(_tuple, task, lang):
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp:
        if True:
        # if lang == 'en':
            sp = sentword2opinion[sp.lower()] if sp in sentword2opinion \
                else senttag2opinion[sp.lower()]  # 'POS' -> 'good'
        # else:
        #     sp = sentword2opinion_GER[sp.lower()] if sp in sentword2opinion \
        #         else senttag2opinion_GER[sp.lower()]  # 'POS' -> 'gut'
    if at and at.lower() == 'null':  # for implicit aspect term
        if True:
        # if lang == 'en':
            at = 'it'
        # else:
        #     at = 'es'

    return at, ac, sp, ot


def add_prompt(sent, orders, task, data_name, args):

    # add ctrl_token
    if args.ctrl_token == "none":
        pass
    elif args.ctrl_token == "post":
        sent = sent + orders
    elif args.ctrl_token == "pre":
        sent = orders + sent
    else:
        raise NotImplementedError
    return sent


def get_para_targets(sents, labels, data_name, data_type, top_k, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []
    if task in ['aste', 'tasd']:
        # at most 5 orders for triple tasks
        top_k = min(5, top_k)

    optim_orders = get_orders(task, data_name, data_type, args, sents, labels)[:top_k]

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]
        cur_sent_str = " ".join(cur_sent)

        # ASTE: parse at & ot
        if task == 'aste':
            assert len(label[0]) == 3
            parsed_label = []
            for _tuple in label:
                parsed_tuple = parse_aste_tuple(_tuple, sents[i])
                parsed_label.append(parsed_tuple)
            label = parsed_label

        # sort label by order of appearance
        # at, ac, sp, ot
        if args.sort_label and len(label) > 1:
            label_pos = {}
            for _tuple in label:
                at, ac, sp, ot = get_task_tuple(_tuple, task, 'en' if args.dataset == 'rest-16' else 'ger')

                # get last at / ot position
                at_pos = cur_sent_str.find(at) if at else -1
                ot_pos = cur_sent_str.find(ot) if ot else -1
                last_pos = max(at_pos, ot_pos)
                last_pos = 1e4 if last_pos < 0 else last_pos
                label_pos[tuple(_tuple)] = last_pos
            new_label = [
                list(k)
                for k, _ in sorted(label_pos.items(), key=lambda x: x[1])
            ]
            label = new_label

        quad_list = []
        for _tuple in label:
            at, ac, sp, ot = get_task_tuple(_tuple, task, 'en' if args.dataset == 'rest-16' else 'ger')
            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            token_end = 3

            element_list = []
            for key in optim_orders[0].split(" "):
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        for o in optim_orders:
            tar = []
            for each_q in quad_list:
                tar.append(each_q[o][1])

            targets.append(" [SSEP] ".join(tar))
            # add prompt
            new_sent = add_prompt(cur_sent, o.split(), task, data_name, args)
            new_sents.append(new_sent)

    return new_sents, targets


def get_para_targets_dev(sents, labels, data_name, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    new_sents = []
    targets = []
    optim_orders = get_orders(task, data_name, "test", args, sents=None, labels=None)
    top_order = optim_orders[0].split(" ")
    for sent, label in zip(sents, labels):
        all_quad_sentences = []
        for _tuple in label:
            # parse ASTE tuple
            if task == "aste":
                _tuple = parse_aste_tuple(_tuple, sent)

            at, ac, sp, ot = get_task_tuple(_tuple, task, 'en' if args.dataset == 'rest-16' else 'ger')

            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            element_list = []
            for key in top_order:
                element_list.append("{} {}".format(key, element_dict[key]))

            one_quad_sentence = " ".join(element_list)
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)

        # add prompt
        sent = add_prompt(sent, top_order, task, data_name, args)

        new_sents.append(sent)
    return new_sents, targets

def read_line_examples_from_file(data_path,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if "unified" in task_name:
                _task, _data, line = line.split("\t")
                tasks.append(_task)
                datas.append(_data)
            else:
                tasks.append(task_name)
                datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels

def get_transformed_io(data_path, data_name, data_type, top_k, args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, args.task, args.dataset, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # low resource
    if data_type == 'train' and args.data_ratio != 1.0:
        num_sample = int(len(inputs) * args.data_ratio)
        sample_indices = random.sample(list(range(0, len(inputs))), num_sample)
        sample_inputs = [inputs[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        inputs, labels = sample_inputs, sample_labels
        print(
            f"Low resource: {args.data_ratio}, total train examples = {num_sample}")
        if num_sample <= 20:
            print("Labels:", sample_labels)

    if data_type == "train" or args.eval_data_split == "dev" or data_type == "test":
        new_inputs, targets = get_para_targets(inputs, labels, data_name,
                                               data_type, top_k, args.task,
                                               args)
    else:
        new_inputs, targets = get_para_targets_dev(inputs, labels, data_name,
                                                   args.task, args)

    print(len(inputs), len(new_inputs), len(targets))
    return new_inputs, targets


class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 task_name,
                 split,
                 lr_setting,
                 data_name,
                 data_type,
                 top_k,
                 args,
                 max_len=128):
        if data_type == 'train':
            print('Loading Train Dataset')
            self.data_path = f'{args.data_path}/{task_name}/{data_name}/split_{split}/{data_type}_{lr_setting}.txt'
        elif data_type == 'val':
            print('Loading Validation Dataset')
            self.data_path = f'{args.data_path}/{task_name}/{data_name}/split_{split}/test_500.txt'
        else:
            print('Loading Test Dataset')
            self.data_path = f'{args.data_path}/{task_name}/{data_name}/split_{split}/{data_type}_full.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.data_name = data_name
        self.data_type = data_type
        self.args = args

        self.top_k = top_k

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):

        if self.args.multi_task:
            inputs, targets = get_transformed_io_unified(
                self.data_path, self.task_name, self.data_name, self.data_type,
                self.top_k, self.args)
        else:
            inputs, targets = get_transformed_io(self.data_path,
                                                 self.data_name,
                                                 self.data_type, self.top_k,
                                                 self.args)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference
            target_max_length = 1024 if self.data_type == "test" else self.max_len

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=target_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


_CONFIG_FOR_DOC = "T5Config"

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum()
    return entropy

# add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class MyT5ForConditionalGenerationScore(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="sum")
            loss = []
            entropy = []
            for i in range(lm_logits.size()[0]):
                loss_i = loss_fct(lm_logits[i], labels[i])
                ent = calc_entropy(lm_logits[i, 0: decoder_attention_mask[i].sum().item()])
                loss.append(loss_i.item())
                entropy.append(ent.item())
            loss = [loss, entropy]
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

# add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class MyT5ForConditionalGeneration(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            #lm_logits_max, lm_logits_max_index = torch.max(lm_logits, dim=-1)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer, args):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.args = args
        
        self.precompute_tokens()

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False, task=self.args.task)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        print("load training data.")
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    split=args.split,
                                    lr_setting=args.lr_setting,
                                    data_type="train",
                                    top_k=self.config.top_k,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True
            if args.data_ratio > 0.3 else False, # don't drop on few-shot
            shuffle=True,
            num_workers=4)

        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  split=args.split,
                                  lr_setting=args.lr_setting,
                                  data_type="val",
                                  top_k=self.config.num_path,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=4)


    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def precompute_tokens(self):
        dic = {"cate_tokens":{}, "all_tokens":{}, "sentiment_tokens":{}, 'special_tokens':[]}
        for task in force_words.keys():
            dic["all_tokens"][task] = {}
            for dataset in force_words[task].keys():
                cur_list = force_words[task][dataset]
                tokenize_res = []
                for w in cur_list:
                    tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
                dic["all_tokens"][task][dataset] = tokenize_res
        for k,v in cate_list.items():
            tokenize_res = []
            for w in v:
                tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            dic["cate_tokens"][str(k)] = tokenize_res
        sp_tokenize_res = []
        for sp in ['great', 'ok', 'bad', 'gut', 'ok', 'schlecht']:
            sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
        for task in force_words.keys():
            dic['sentiment_tokens'][str(task)] = sp_tokenize_res
        #dic['sentiment_tokens'] = sp_tokenize_res
        special_tokens_tokenize_res = []
        for w in ['[O','[A','[S','[C','[SS']:
            special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
        special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
        dic['special_tokens'] = special_tokens_tokenize_res
        
        self.force_tokens = dic 
    
    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):
        """
        Constrained Decoding
        # ids = self.tokenizer("text", return_tensors='pt')['input_ids'].tolist()[0]
        """
            
        force_tokens = self.force_tokens

        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
            'it': [34],
            # 'es': [3, 15, 7],
            'null': [206,195]
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['EP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 
       
        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1]

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens['sentiment_tokens'][str(task)]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            if task != 'aste': 
                if True:
                # if data_name == 'rest-16':
                    force_list.extend(to_id['it'] + [1])  
                # elif data_name == 'GERestaurant':
                #     force_list.extend(to_id['es'] + [1])  
            ret = force_list  
        elif cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens['cate_tokens'][str(data_name)]
        elif cur_term in to_id['OT']:  # OT
            force_list = source_ids[batch_id].tolist()
            if task == "acos":
                force_list.extend(to_id['null'])  # null
            ret = force_list
        else:
            raise ValueError(cur_term)    

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        return ret


def evaluate(model, task, split, lr_setting, dataset, args, data_type):
    """
    Compute scores given the predictions and gold labels
    """

    outputs, targets, probs = [], [], []
    num_path = args.num_path
    
    if task in ['aste', 'tasd']:
        num_path = min(5, num_path)
        
    dataset = ABSADataset(model.tokenizer,
                          task_name=task,
                          split=split,
                          lr_setting=lr_setting,
                          data_type=data_type,
                          top_k=num_path,
                          args=args,
                          data_name=dataset,
                          max_len=args.max_seq_length)
    
    data_loader = DataLoader(dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=2)
    
    device = torch.device('cuda:0')
    model.model.to(device)
    model.model.eval()

    for batch in tqdm(data_loader):
        # beam search

        outs = model.model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            max_length=args.max_seq_length,
            num_beams=args.beam_size,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=partial(
               model.prefix_allowed_tokens_fn, task, args.dataset,
               batch['source_ids']) if args.constrained_decode else None,
        )
        dec = [
            model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        outputs.extend(dec)
        targets.extend(target)

    if args.multi_path:
        targets = targets[::num_path]

        # get outputs
        _outputs = outputs # backup
        outputs = [] # new outputs
        if args.agg_strategy == 'post_rank':
            inputs = [ele for ele in sents for _ in range(num_path)]
            assert len(_outputs) == len(inputs), (len(_outputs), len(inputs))
            preds = [[o] for o in _outputs] 
            model_path = os.path.join(args.output_dir, "final")
            scores = cal_entropy(inputs, preds, model_path, model.tokenizer)

        for i in range(0, len(targets)):
            o_idx = i * num_path
            multi_outputs = _outputs[o_idx:o_idx + num_path]

            if args.agg_strategy == 'post_rank':
                multi_probs = scores[o_idx:o_idx + args.num_path]
                assert len(multi_outputs) == len(multi_probs)

                sorted_outputs = [i for _,i in sorted(zip(multi_probs,multi_outputs))]
                outputs.append(sorted_outputs[0])
                continue
            elif args.agg_strategy == "pre_rank":
                outputs.append(multi_outputs[0])
                continue
            elif args.agg_strategy == 'rand':
                outputs.append(random.choice(multi_outputs))
                continue
            elif args.agg_strategy == 'vote':
                all_quads = []
                for s in multi_outputs:
                    all_quads.extend(
                        extract_spans_para(seq=s, seq_type='pred'))

                output_quads = []
                counter = dict(Counter(all_quads))
                for quad, count in counter.items():
                    # keep freq >= num_path / 2
                    if count >= len(multi_outputs) / 2:
                        output_quads.append(quad)

                # recover output
                output = []
                for q in output_quads:
                    ac, at, sp, ot = q
                    if task == "aste":
                        if 'null' not in [at, ot, sp]:  # aste has no 'null', for zero-shot only
                            output.append(f'[A] {at} [O] {ot} [S] {sp}')

                    elif task  == "tasd":
                        output.append(f"[A] {at} [S] {sp} [C] {ac}")

                    elif task in ["asqp", "acos"]:
                        output.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

                    else:
                        raise NotImplementedError

                target_quads = extract_spans_para(seq=targets[i],
                                                seq_type='gold')

                # if no output, use the first path
                output_str = " [SSEP] ".join(
                    output) if output else multi_outputs[0]

                outputs.append(output_str)

    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])

    scores, all_labels, all_preds = compute_scores(outputs,
                                                   targets,
                                                   verbose=True, task=args.task)
    return scores


import signal
import sys

def signal_handler(signal, frame):
    print("Training interrupted by user!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train_function_mvp(args):

    set_seed(args.seed)

    # training process
    if args.do_train:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
        
        print("\n****** Conduct Training ******")

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
        model = T5FineTuner(args, tfm_model, tokenizer, args)

        # load data
        train_loader = model.train_dataloader()

        # config optimizer
        t_total = ((len(train_loader.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(args.num_train_epochs))

        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total
        }

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=20,
                                            verbose=True,
                                            mode="max")
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback, early_stop_callback,
                TQDMProgressBar(refresh_rate=10), lr_monitor
            ],
        )

        trainer = pl.Trainer(**train_params)

        try:
          trainer.fit(model)
        except KeyboardInterrupt:
          print("Training has been stopped manually.")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        # save the final model
        #model.model.save_pretrained(os.path.join(args.output_dir, "final"))
        #tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print("Finish training and saving the model!")

        scores = evaluate(model, args.task, args.split, args.lr_setting, args.dataset, args,
                                data_type=args.eval_data_split)


    return scores
    
def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument(
        "--task",
        default='asqp',
        choices=["asqp", "acos", "aste", "tasd", "unified", "unified3"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--dataset",
        default='rest-16',
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_data_split",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument(
        "--split",
        default=1,
        choices=[1,2,3,4,5],
        type=int,
    )
    parser.add_argument(
        "--lr_setting",
        default="full",
        choices=["full", "1000", "500"],
        type=str,
    )
    
    parser.add_argument("--model_name_or_path",
                        default='t5-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir",
                        default='outputs/temp',
                        type=str,
                        help="Output directory")
    parser.add_argument("--load_ckpt_name",
                        default=None,
                        type=str,
                        help="load ckpt path")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument(
        "--do_inference",
        default=False,
        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--multi_path", action='store_true')
    parser.add_argument("--num_path", default=1, type=int)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--single_view_type",
                    default="rank",
                    choices=["rank", "rand", "heuristic"],
                    type=str)
    parser.add_argument("--ctrl_token",
                        default="post",
                        choices=["post", "pre", "none"],
                        type=str)
    parser.add_argument("--sort_label",
                        action='store_true',
                        help="sort tuple by order of appearance")
    parser.add_argument("--load_path_cache",
                        action='store_true',
                        help="load decoded path from cache")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--multi_task", action='store_true')
    parser.add_argument("--constrained_decode",
                        action="store_true",
                        help='constrained decoding when evaluating')
    parser.add_argument('--agg_strategy', type=str, default='vote', choices=['vote', 'rand', 'heuristic', 'pre_rank', 'post_rank'])
    parser.add_argument("--data_ratio",
                        default=1.0,
                        type=float,
                        help="low resource data ratio")

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    os.makedirs(args.output_dir, exist_ok=True)

    return args
    
if __name__ == '__main__':
    args = init_args()
    set_seed(args.seed)
    f1_res = train_function_mvp(args)

    f1_str = "F1: {}".format(f1_res['f1'])
    log_file_path = os.path.join(args.output_dir, "result.txt")
    with open(log_file_path, "a+") as f:
        f.write(f1_str)


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

