import pandas as pd
import logging
import os
from os import truncate
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import json
from dataclasses import dataclass, asdict
from multiprocessing import Pool
import multiprocessing
import math
from random import sample
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList
)


logger = logging.getLogger(__name__)

def read_data(file_name, percent, random_seed):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    inputs = []
    targets = []
    for index, d in enumerate(data):
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        inputs.append(d['input'])
        targets.append(d['target'])
    dict_ = {'input': inputs, 'output': targets}
    df_data = pd.DataFrame(dict_)
    df_data.dropna(axis=0, how='any')

    # randomly extract *percent of the data
    num_samples = int(len(df_data)*percent)
    print(f'the number of num_samples is {len(df_data)}')
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    print(f'the number of num_samples is {len(df_data)}')

    return df_data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    
def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


class Seq2SeqDataset(Dataset):
    def __init__(self, args, data, mode):
        if not args.emotion_prediction:
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o] for i, o in zip(inputs, outputs)]
        elif mode == 'dev':
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o] for i, o in zip(inputs, outputs)]
        else:
            inputs = list(data["input"])
            inputs = [i.split('***') for i in inputs]
            # counter = 0
            # for i in inputs:
            #     if len(i) != 2:
            #         counter += 1
            # print(counter)
            outputs = list(data['output'])
            self.examples = [[i[0], i[1], o] for i, o in zip(inputs, outputs)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode

    def __call__(self, batch):
        if self.mode == "dev":
            batch = [d[0] for d in batch]
            inputs = self.tokenizer(batch, max_length=self.args.max_length, truncation=True, padding=True, return_tensors='pt')
        else:
            inputs = preprocess_data_batch(batch, self.tokenizer, self.args)

        return inputs


def preprocess_data_batch(data, tokenizer, args):
    
    inputs = [d[0] for d in data]
    inputs_pred = None
    if args.emotion_prediction:
        inputs_pred = [d[1] for d in data]
    targets = [d[-1] for d in data]

    if args.model_type == "decoder":
        if args.mode == "pretrain":
            inputs = tokenizer(
                inputs,
                max_length=args.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            labels = inputs['input_ids'].clone().contiguous()
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            type_token_ids = inputs['attention_mask'].long()
            inputs['labels'] = labels
            inputs["type_token_ids"] = type_token_ids
            return inputs
            
        # decoder-only model
        inputs = tokenizer(
            inputs,
            max_length=args.max_length - 1,
            truncation=True
        )

        targets = tokenizer(
            targets,
            add_special_tokens=False,
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        concat_input = [input_ids[i] + target_ids[i] for i in range(len(input_ids))]
        concat_input = [c_[: args.max_length] for c_ in concat_input]
        if not args.open_ended:
            concat_input = [c_ids + [tokenizer.eos_token_id] for c_ids in concat_input]

        type_token_ids = [[0] * len(input_ids[i]) + [1] * (len(concat_input[i]) - len(input_ids[i])) for i in range(len(input_ids))]
        attention_mask = [[1] * len(concat_input[i]) for i in range(len(input_ids))]
        
        max_batch_length = 0
        for i in range(len(input_ids)):
            max_batch_length = max(max_batch_length, len(type_token_ids[i]))
        


        if  args.emotion_prediction:
            inputs_pred = tokenizer(
                inputs_pred,
                max_length=args.max_length - 1,
                truncation=True
            )
            # 其中，max_length 参数指定了输入文本的最大长度，但是由于 Transformers 模型需要在输入文本的末尾添加一个特殊的结束符 [SEP]，所以实际上输入文本的最大长度应该是 max_length-1。
            input_pred_ids = inputs_pred['input_ids'] #  从tokenizer结果中取出inputs_ids
            concate_pred_input = [input_pred_ids[i] + target_ids[i] for i in range(len(input_pred_ids))] # 逐个样本拼接target_ids
            concate_pred_input = [c_[: args.max_length] for c_ in concate_pred_input] # 按照最大长度进行截断
            if not args.open_ended:
                concate_pred_input = [c_ids + [tokenizer.eos_token_id] for c_ids in concate_pred_input]

            pred_type_token_ids = [[0] * len(input_pred_ids[i]) + [1] * (len(concate_pred_input[i]) - len(input_pred_ids[i])) for i in range(len(input_pred_ids))]
            pred_attention_mask = [[1] * len(concate_pred_input[i]) for i in range(len(input_pred_ids))]

            # max_batch_length = 0
            for i in range(len(input_pred_ids)):
                max_batch_length = max(max_batch_length, len(pred_type_token_ids[i]))

        type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in type_token_ids]
        attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in attention_mask]
        concat_input = [[tokenizer.pad_token_id] * (max_batch_length - len(ids)) + ids for ids in concat_input]
        type_token_ids = torch.Tensor(type_token_ids).long()
        attention_mask = torch.Tensor(attention_mask).long()
        concat_input = torch.Tensor(concat_input).long()
        labels = concat_input.clone().contiguous()
        labels[type_token_ids[:, :] == 0] = -100

        if args.emotion_prediction:
            pred_type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in pred_type_token_ids]
            pred_attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in pred_attention_mask]
            pred_concat_input = [[tokenizer.pad_token_id] * (max_batch_length -len(ids)) + ids for ids in concate_pred_input]
            pred_type_token_ids = torch.Tensor(pred_type_token_ids).long()
            pred_attention_mask = torch.Tensor(pred_attention_mask).long()
            pred_concat_input = torch.Tensor(pred_concat_input).long()
            pred_labels = pred_concat_input.clone().contiguous()
            pred_labels[pred_type_token_ids[:, :] == 0] = -100

            concat_input = torch.concat([concat_input, pred_concat_input], dim=0)
            attention_mask = torch.concat([attention_mask, pred_attention_mask], dim=0)
            type_token_ids = torch.concat([type_token_ids, pred_type_token_ids], dim=0)
            labels = torch.concat([labels, pred_labels], dim=0)                    


        if "chatglm" in args.model_name_or_path and not "chatglm2" in args.model_name_or_path:
            attention_mask = attention_mask.bool()
        return {
            "input_ids": concat_input,
            "attention_mask": attention_mask,
            "type_token_ids": type_token_ids,
            "labels": labels,
            "tradoff": args.beta
        }
    else:
        ## encoder-decoder model
        inputs = tokenizer(
            inputs,
            max_length=args.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        targets = tokenizer(
            targets,
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids[:, :] == tokenizer.pad_token_id] = 0
        type_token_ids = torch.ones_like(target_ids)
        type_token_ids[target_ids[:, :] == tokenizer.pad_token_id] = 0
        labels = target_ids.clone().contiguous()
        labels[target_ids[:, :] == tokenizer.pad_token_id] = -100
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
            "type_token_ids": torch.LongTensor(type_token_ids)
        }

# class SHanglinModel(Pretrained):
#     def __init__(self):
#         self.model = LlamaModel
#         self.lm_head = nn.Linear

@dataclass
class ModelArgs:
    model_type: str = "decoder"
    model_name_or_path: str = "YOUR_MODEL_PATH"
    checkpoint_dir: str = None
    output_dir: str = "YOUR_OUTPUT_DIR_PATH"
    data_dir: str = "DATASET_PATH"
    deepspeed_config = "./deepspeed_config.json"
    do_train: bool = True
    do_eval: bool = False
    num_train_epochs = 10
    warmup_ratio: float = 0.1
    warmup_steps: int = None
    save_steps: int = 500
    weight_decay: float = 0.0
    max_seq_length: int = 96
    max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = None
    top_p: float = None
    learning_rate: float = 3e-5
    preprocess_inputs: bool = True
    clip_norm: float = 1.0
    open_ended: bool = False
    batch_size: int = 32
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora: bool = True
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(asdict(self), indent=5))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))