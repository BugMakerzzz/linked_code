import os
import re
import sys
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
import sys
import json
# import mlflow
# from sklearn import metrics
from tqdm.auto import tqdm
from datasets import load_dataset
from prompt import Prompter
from process import *
from torch.cuda.amp import autocast as autocast, GradScaler

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
OUTPUT_DIR = "./result/llama"
MODEL_PATH = "/mnt/publiccache/huggingface/Llama-2-7b-chat-hf"
# LORA_PATH = './result/Llama/lr_0.0001_e5_s17_b64_vs500_ws200_t20230912_Tue_21:47:56'


# MICRO_BATCH_SIZE = 1 # this could actually be 5 but i like powers of 2
train_batch_size = 16
dev_batch_size = 4
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
gradient_accumulation_steps = 4 
epochs = 2  # we don't always need 3 tbh
learning_rate = 1e-4  # the Karpathy constant
cutoff_len = 512  # 256 accounts for about 96% of the data
val_set_size = 500
warm_up_steps = 300

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}


local_rank = int(os.environ.get("LOCAL_RANK"))
#%% dpp initialize
is_main_process = (not ddp or local_rank == 0) 
if ddp:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print("launch process",local_rank)
    world_size = torch.distributed.get_world_size()
#%% reproducibility
seed = 17
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
device = 'cuda'

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    # load_in_8bit=True,
    device_map=device_map,
).bfloat16()
# model = prepare_model_for_int8_training(model)
tokenizer = LlamaTokenizer.from_pretrained(
    MODEL_PATH,
    padding_side = 'left',
)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    inference_mode=False, 
    task_type=TaskType.CAUSAL_LM,
)
# model = PeftModel.from_pretrained(
#             model,
#             LORA_PATH,
#             device_map=device_map,
#             is_trainable=True
#         )

model = get_peft_model(model, config)
# model.config.inference_mode=False

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


data_length = 5000
dataset = 'hella'
input_parser = InputParser(dataset, data_length, split='train', shuffle=False)
js_data = []
for tup in input_parser:
    question = tup[1]
    options = tup[2]
    label = tup[3]
    answer = f'({label}) {options[eval(label)-1]}'
    js_data.append({'question': question, 'answer':answer})

random.shuffle(js_data)
train_data = js_data[val_set_size:]
dev_data = js_data[:val_set_size]

def transfer_answer_to_label(answer):
    match = re.findall(r'[1-5]\)',answer)
    if match:
        pred = match[-1][:-1]
        return eval(pred) - 1
    else:
        return -1
    
def collate_fn_for_critic(batch):
    new_batch = {'input_ids':[], 'labels':[]}
    for data in batch:
        question = data['question']
        answer = data['answer']

        user_prompt =  """Below is an question, use your commonsense knowledge to answer the correct option.
        Question: {}
        Answer: """
        
        full_input = user_prompt.format(question)
        len_input_tokens = (
            len(
                tokenizer(
                    full_input,
                    truncation=True,
                    max_length=cutoff_len,
                )["input_ids"]
            )
        )  # no eos token

        full_token = tokenizer(
            full_input + answer,
            padding="max_length",
            truncation=True,
            max_length=cutoff_len,
        )['input_ids']
        
        ids = full_token
        labels = [-100] * len_input_tokens + full_token[len_input_tokens:]
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = -100
        new_batch['input_ids'].append(ids)
        new_batch['labels'].append(labels)
    new_batch['input_ids'] = torch.tensor(np.array(new_batch['input_ids']))
    new_batch['labels'] = torch.tensor(np.array(new_batch['labels']))

    return new_batch


collate_fn = collate_fn_for_critic
train_sampler = None

shuffle = True
if shuffle:
    train_sampler = torch.utils.data.RandomSampler(train_data)
if ddp:
    node_batch_size = train_batch_size // world_size
    train_sampler = torch.utils.data.DistributedSampler(train_data,shuffle=shuffle)
train_dataloader = torch.utils.data.DataLoader(train_data,
    batch_size=node_batch_size,sampler=train_sampler,
    collate_fn=collate_fn)
if is_main_process:
    dev_dataloader = torch.utils.data.DataLoader(dev_data,
        batch_size=dev_batch_size,shuffle=False,
        collate_fn=collate_fn)


model_name = f"{dataset}_lr_{learning_rate}_e{epochs}_s{seed}_b{train_batch_size}_vs{val_set_size}_ws{warm_up_steps}_t{time.strftime('%Y%m%d_%a_%H:%M:%S')}"
iterations = epochs * ((len(train_dataloader.dataset)-1)// train_batch_size + 1)
validation_cycle = int(((len(train_dataloader.dataset)-1)// train_batch_size + 1) // 5)
# if is_main_process:
#     mlflow.set_experiment("fintune_llama")
#     mlflow.start_run()
#     mlflow.set_tags({
#         "dataset_path":'wino_train_l_1w',
#         "model":model_name,
#         "output_path":OUTPUT_DIR,
#     })
# #freeze model
#
model = model.to(device=device)
# model_ = model #for generation
best_dev_acc = 0
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
# warmup_scheduler = get_constant_schedule_with_warmup(optimizer,warm_up_steps)
warmup_scheduler = get_cosine_schedule_with_warmup(optimizer,warm_up_steps,iterations)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.1)

step = 0
train_iter = iter(train_dataloader)
if is_main_process:
    pbar = tqdm(total=iterations,dynamic_ncols=True)
while step <= iterations:
    if is_main_process and (step % validation_cycle == 0 and step > 0): #validation
        with torch.no_grad():
            model.eval()
            pbar.set_description('validating...')
            dev_loss = 0.
            dev_sample_count = 0
            dev_labels = []
            dev_preds = []
            correct_count = 0
            for batch in tqdm(dev_dataloader,desc=f'validating ...',leave=False):
                if val_set_size > 0 and dev_sample_count>=val_set_size:
                    break
                batch = {k:v.to(device=device) for k,v in batch.items()}
                result = model(**batch)
                loss = result['loss']
                dev_sample_count += result['logits'].size(0)
                dev_loss += loss.item() * result['logits'].size(0)
                full_tokens = tokenizer.batch_decode(batch['input_ids'])
                labels = []
                inputs = []
                for x in full_tokens:
                    input, label = x.split('Answer: ')
                    input += 'Answer: '
                    input = tokenizer(input, 
                                      padding="max_length",
                                      truncation=True,
                                      max_length=cutoff_len,
                                    )['input_ids']
                    inputs.append(input)
                    labels.append(label)
                inputs = torch.tensor(np.array(inputs)).to(device)
                generation_output = model.module.generate(
                    input_ids=inputs,
                    max_new_tokens=16,
                )
                preds = tokenizer.batch_decode(generation_output)
                dev_preds.extend(preds)
                dev_labels.extend(labels)
                del inputs
                del preds
                del labels
                del result
                del loss
            dev_loss = dev_loss/dev_sample_count
            for i in range(dev_sample_count):
                pred = dev_preds[i].split('Answer: ')[-1]
                label = dev_labels[i]
                print(pred)
                print(label)
                if transfer_answer_to_label(pred) == transfer_answer_to_label(label):
                    correct_count += 1
            dev_accuracy = correct_count / dev_sample_count
            dev_metrics = {'dev/loss':dev_loss, 'dev/acc':dev_accuracy}
            print(str(dev_metrics))
            # mlflow.log_metrics(dev_metrics,step)
            if dev_accuracy > best_dev_acc:
                best_dev_acc = dev_accuracy
            # if best_check_comp(dev_metrics[best_check_metric],best_dev_metric) or not save_best:
            #     best_dev_metric = dev_metrics[best_check_metric]
                save_path = os.path.join(OUTPUT_DIR, model_name+f'_acc{dev_accuracy}')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                model.module.save_pretrained(save_path)
        plateau_scheduler.step(dev_loss)
    model.train()
    optimizer.zero_grad()
    
    batch_loss = torch.tensor(0.).to(device)
    for tiny_step in range(gradient_accumulation_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        batch = {k:v.to(device=device) for k,v in batch.items()}
        # with torch.cuda.amp.autocast():
        result = model(**batch)
        loss = result['loss'] / gradient_accumulation_steps
        # scaler.scale(loss).backward()
        loss.backward()
        batch_loss += loss.item()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    lr_dic = {'learning_rate':lr}
    # mlflow.log_metrics(lr_dic,step)    
    optimizer.step()
    
    # scaler.step(optimizer)
    # scaler.update()
    warmup_scheduler.step()
    step+=1
    if ddp:
        # loss = loss.detach()
        losses = [torch.zeros_like(batch_loss) for i in range(world_size)]
        torch.distributed.all_gather(tensor_list=losses,tensor=batch_loss)
        batch_loss = torch.stack(losses).mean()
    if is_main_process:
        pbar.set_description('training...')
        pbar.update()
        # mlflow.log_metric('train/loss',batch_loss.item(),step)
    del result
    del loss
if is_main_process:
    pbar.close()
print('Done!!!')



