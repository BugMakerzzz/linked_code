import os
import re
import sys
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel,
    TaskType,
    set_peft_model_state_dict
)
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
import sys
import json
import mlflow
# from sklearn import metrics
from tqdm.auto import tqdm
from datasets import load_dataset

from torch.cuda.amp import autocast as autocast, GradScaler

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
# DATA_PATH = './data/reader_train/wino_train_data_l.json'
OUTPUT_DIR = "./result/Llama_reader/"
MODEL_PATH = "./model/Llama-2-7b-chat-hf"
# LORA_PATH = './result/Llama/lr_0.0001_e5_s17_b64_vs500_ws200_t20230912_Tue_21:47:56'
RAW_DATA_PATH = './data/reader_train/wino_train_data_2k.json'
DATA_PATH = './data/reader_train/wino_train_data_1w.json'
# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )


# MICRO_BATCH_SIZE = 1 # this could actually be 5 but i like powers of 2
train_batch_size = 16
dev_batch_size = 2
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
gradient_accumulation_steps = 4 
epochs = 5  # we don't always need 3 tbh
learning_rate = 3e-4  # the Karpathy constant
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
    load_in_8bit=True,
    device_map=device_map,
)

model = prepare_model_for_int8_training(model)
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


# js_data = []
# with open(RAW_DATA_PATH, 'r') as f:
#     js_data = json.load(f)
#     f.close()
# data = []
# for i in range(0, len(js_data), 3):
#     question = js_data[i]['question']
#     if question != js_data[i+1]['question'] or question != js_data[i+2]['question']:
#         print("WRONG!")
#     knowledge = "Knowledge:"
#     for j in range(3):
#         knowledge += js_data[i+j]['knowledge'].strip('Knowledge:')
#     answer = js_data[i]['answer']
#     data.append({'question':question, 'knowledge':knowledge, 'answer':answer})
# with open(DATA_PATH, 'w') as f:
#     json.dump(data, f, indent=4)
#     f.close()
    
data = load_dataset("json", data_files=DATA_PATH)
train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=17
    )
train_data = train_val["train"].shuffle()
dev_data = train_val["test"].shuffle()

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
        knowledge = data['knowledge']
        label = data['label']
        answer = f'Label:{label}'
        
        user_prompt =  f"""Below is a question paired with related knowledge. Choose a correct option that appropriately fill in the blank of the context.
        {question}
        {knowledge}
        Response: """
        # user_prompt =  f"""Below is an question, paired with related knowledge. 
        # Please classify whether the knowledge is useful (label 0) or harmful(label 1) for answering the question correctly. 
        # Question: {question}
        # {knowledge}
        # Your response should be in this form:
        # Label: """
        
        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=cutoff_len,
                )["input_ids"]
            )
        )  # no eos token

        full_token = tokenizer(
            user_prompt + answer,
            padding="max_length",
            truncation=True,
            max_length=cutoff_len,
        )['input_ids']
        
        ids = full_token
        labels = [-100] * len_user_prompt_tokens + full_token[len_user_prompt_tokens:]
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


model_name = f"lr_{learning_rate}_e{epochs}_s{seed}_b{train_batch_size}_vs{val_set_size}_ws{warm_up_steps}_t{time.strftime('%Y%m%d_%a_%H:%M:%S')}"
iterations = epochs * ((len(train_dataloader.dataset)-1)// train_batch_size + 1)
validation_cycle = int(((len(train_dataloader.dataset)-1)// train_batch_size + 1) // 4)
if is_main_process:
    mlflow.set_experiment("train_Llama_reader")
    mlflow.start_run()
    mlflow.set_tags({
        "dataset_path":DATA_PATH,
        "model":model_name,
        "output_path":OUTPUT_DIR,
    })
#freeze model
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
                    input, label = x.split('Response: ')
                    input += 'Response: '
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
                pred = dev_preds[i].split('Response:')[-1]
                label = dev_labels[i]
                if transfer_answer_to_label(pred) == transfer_answer_to_label(label):
                    correct_count += 1
            dev_accuracy = correct_count / dev_sample_count
            # dev_accuracy = metrics.accuracy_score(dev_labels,dev_preds)
            # dev_precision = metrics.precision_score(dev_labels,dev_preds)
            # dev_recall = metrics.recall_score(dev_labels,dev_preds)
            # dev_precisions, dev_recalls, dev_thresholds = metrics.precision_recall_curve(dev_labels,dev_probs)
            # tn, fp, fn, tp = metrics.confusion_matrix(dev_labels, dev_preds,normalize="all").ravel()
            dev_metrics = {'dev/loss':dev_loss, 'dev/acc':dev_accuracy}
            print(str(dev_metrics))
            # for r in [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]:
            #     idx = 0
            #     while dev_recalls[idx] > r:
            #         idx += 1
            #     dev_metrics[f"dev/precision-{r*100:.0f}"] = dev_precisions[idx]
            # dev_ap = metrics.average_precision_score(dev_labels,dev_probs)
            # dev_metrics['dev/ap']=dev_ap
            mlflow.log_metrics(dev_metrics,step)
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
    mlflow.log_metrics(lr_dic,step)    
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
        mlflow.log_metric('train/loss',batch_loss.item(),step)
    del result
    del loss
if is_main_process:
    pbar.close()
print('Done!!!')



