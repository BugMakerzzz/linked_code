import os
import re
import sys
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, RobertaForMultipleChoice, get_cosine_schedule_with_warmup, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
import sys
import json
# from sklearn import metrics
from tqdm.auto import tqdm
from process import *
from datasets import load_metric

OUTPUT_DIR = "./result/roberta/"
MODEL_PATH = "/mnt/publiccache/huggingface/roberta-large"
# MODEL_PATH = "./result/roberta/lr_0.0001_e5_s17_b16_vs500_ws200_gs8_t20240605_Wed_13:35:06_acc0.716"
# MICRO_BATCH_SIZE = 1 # this could actually be 5 but i like powers of 2
train_batch_size = 16
dev_batch_size = 4
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
gradient_accumulation_steps = 8
epochs = 10  # we don't always need 3 tbh
learning_rate = 1e-4  # the Karpathy constant
cutoff_len = 512 # 256 accounts for about 96% of the data
val_set_size = 500
warm_up_steps = 500

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

dataset = 'siqa'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = RobertaForMultipleChoice.from_pretrained(MODEL_PATH, device_map=device_map)

data_length = 5000
input_parser = InputParser(dataset, data_length, split='train', shuffle=False)
js_data = []
for tup in input_parser:
    stem = tup[0]
    question = tup[1]
    options = tup[2]
    label = eval(tup[3]) - 1
    choice0 = options[0]
    choice1 = options[1]
    if dataset == 'hella':
        choice2 = options[2]
        choice3 = options[3]
        js_data.append({'question':stem, 
                        'choice0':choice0, 
                        'choice1':choice1, 
                        'choice2':choice2, 
                        'choice3':choice3, 
                        'label':label})
    elif dataset == 'piqa':
        js_data.append({'question':stem, 
                        'choice0':choice0, 
                        'choice1':choice1,  
                        'label':label})
    elif dataset == 'siqa':
        choice2 = options[2]
        js_data.append({'question':stem, 
                        'choice0':choice0, 
                        'choice1':choice1, 
                        'choice2':choice2,  
                        'label':label})
    else:
        sentence1 = stem.replace('_', choice0)
        sentence2 = stem.replace('_', choice1)
        if label == 0:
            label1 = 0
            label2 = 1
        else:
            label1 = 1
            label2 = 0
        js_data.append({'sentence': sentence1, 'label':label1})
        js_data.append({'sentence': sentence2, 'label':label2})

# if dataset == 'wino':
#     val_set_size *= 2   
#     cutoff_len = 128     
random.shuffle(js_data)
train_data = js_data[val_set_size:]
dev_data = js_data[:val_set_size]

    
def collate_fn_for_critic(batch):
    new_batch = {'input_ids':[], 'labels':[], 'attention_mask':[]}
    for data in batch:
        if dataset == 'hella':
            question = data['question']
            label = data['label']
            choice0 = data['choice0']
            choice1 = data['choice1']
            choice2 = data['choice2']
            choice3 = data['choice3']

            encoding = tokenizer([question, question, question, question], [choice0, choice1, choice2, choice3], truncation=True, padding="max_length",
                        max_length=cutoff_len, return_tensors="pt")
        
        elif dataset == 'siqa':
            question = data['question']
            label = data['label']
            choice0 = data['choice0']
            choice1 = data['choice1']
            choice2 = data['choice2']

            encoding = tokenizer([question, question, question], [choice0, choice1, choice2], truncation=True, padding="max_length",
                        max_length=cutoff_len, return_tensors="pt")
        elif dataset == 'piqa':
            question = data['question']
            label = data['label']
            choice0 = data['choice0']
            choice1 = data['choice1']

            encoding = tokenizer([question, question], [choice0, choice1], truncation=True, padding="max_length",
                        max_length=cutoff_len, return_tensors="pt")
        else:
            sentence = data['sentence']
            label = data['label']
            encoding = tokenizer(sentence, truncation=True, padding="max_length",
                        max_length=cutoff_len, return_tensors="pt")
        
        input_ids = torch.squeeze(encoding.input_ids)
        attention_mask = torch.squeeze(encoding.attention_mask)
        new_batch['input_ids'].append(input_ids.tolist())
        new_batch['labels'].append(label)
        new_batch['attention_mask'].append(attention_mask.tolist())
    new_batch['input_ids'] = torch.tensor(np.array(new_batch['input_ids']))
    new_batch['labels'] = torch.tensor(np.array(new_batch['labels']))
    new_batch['attention_mask'] = torch.tensor(np.array(new_batch['attention_mask']))
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


model_name = f"lr_{learning_rate}_e{epochs}_s{seed}_b{train_batch_size}_vs{val_set_size}_ws{warm_up_steps}_gs{gradient_accumulation_steps}_t{time.strftime('%Y%m%d_%a_%H:%M:%S')}"
iterations = epochs * ((len(train_dataloader.dataset)-1)// train_batch_size + 1)
validation_cycle = int(((len(train_dataloader.dataset)-1)// train_batch_size + 1) // 10)
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
    accuracy = load_metric("./accuracy.py")
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
                pred = torch.argmax(result['logits'], axis=1)
                loss = result['loss']
                dev_sample_count += result['logits'].size(0)
                dev_loss += loss.item() * result['logits'].size(0)
                dev_preds.extend(pred)
                dev_labels.extend(batch['labels'])
                del result
                del loss
                del batch
            dev_loss = dev_loss/dev_sample_count
            dev_acc = accuracy.compute(predictions=dev_preds, references=dev_labels)['accuracy']
            dev_metrics = {'dev_loss':dev_loss, 'dev_acc': dev_acc}
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print(f'best_dev_acc:{best_dev_acc}')
                save_path = os.path.join(OUTPUT_DIR, model_name+f'_acc{dev_acc}')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
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
    del result
    del loss
if is_main_process:
    pbar.close()
print('Done!!!')



