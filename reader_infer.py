import os
import torch
import torch.nn as nn
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
from process import *


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


dev_batch_size = 4
cutoff_len = 256  # 256 accounts for about 96% of the data


device = 'cuda'

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
).half().cuda()
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
    inference_mode=True, 
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


InputParser('wino')



    
def collate_fn_for_critic(batch):
    new_batch = {'input_ids':[], 'labels':[]}
    for data in batch:
        question = data['question']
        knowledge = data['knowledge']
        label = data['label']

        
        user_prompt =  f"""Below is a question paired with related knowledge. Choose a correct option that appropriately fill in the blank of the context.
        {question}
        {knowledge}
        Response: """
        
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

dev_dataloader = torch.utils.data.DataLoader(dev_data,
    batch_size=dev_batch_size,shuffle=False,
    collate_fn=collate_fn)

model = model.to(device=device)


with torch.no_grad():
    for batch in tqdm(dev_dataloader,leave=False):
        batch = {k:v.to(device=device) for k,v in batch.items()}
        result = model(**batch)
        loss = result['loss']
            



