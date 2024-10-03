import re
import json
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import sys
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from typing import Any, Dict, List, Tuple
from config import *
import torch
import torch.nn.functional as F
import sys
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaForSequenceClassification, RobertaForSequenceClassification, RobertaForMultipleChoice, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import faiss
import random 
from transformers import AutoTokenizer, RobertaForMultipleChoice
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel,
    TaskType,
    set_peft_model_state_dict
)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
random.seed(17)

async def process_api_requests(
    request_ls: list,
    results: list,
    request_url: str,
    api_key: str,
    model: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name = 'cl100k_base',
    temperature = 1.0,
    sample_cnt = 1,
    max_attempts = 100,
    logging_level = 20,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = 'chat/completions'
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    list_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
        # `requests` will provide requests one at a time
    requests = request_ls.__iter__()

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif list_not_finished:
                try:
                    # get new request
                    request_json = {"model":model, "messages":next(requests), "temperature":temperature, "n":sample_cnt}
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        results=results,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        results: list
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        # logging.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json,
                    # proxy='http://Sept:20001228@127.0.0.1:14396'
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            results.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

# functions

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1 


def get_data_path(dataset, split='dev'):
    if dataset == 'wino':
        if split == 'dev':
            return wino_dev_data_path
        elif split == 'train':
            return wino_train_data_path
    elif dataset == 'csqa2':
        if split == 'train':
            return csqa2_train_data_path
        else:
            return csqa2_dev_data_path
    elif dataset == 'hella':
        if split =='train':
            return hella_train_data_path
        else:
            return hella_dev_data_path
    elif dataset == 'piqa':
        if split == 'train':
            return piqa_train_data_path
        else:
            return piqa_dev_data_path
    elif dataset == 'siqa':
        if split == 'train':
            return siqa_train_data_path
        else:
            return siqa_dev_data_path
    return 

def get_task_name(task):
    return TASK_NAME_DICT[task]


def set_log():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)        
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    fh = logging.FileHandler(log_path,encoding="utf-8")
    fh.setLevel(logging.INFO)
    formator = logging.Formatter(fmt= "%(message)s")
    fh.setFormatter(formator)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def delete_answer(answer):
    delete_idx = len(answer) - 5
    iter = re.finditer(r"\(1-5\)", answer)
    for i in iter:
        delete_idx = i.start()
    answer = answer[:delete_idx]
    return answer

def get_index_ignore_case(pred, options):
    pred = pred.lower()
    for i in range(len(options)):
        option = options[i].lower()
        if pred == option:
            return i
    return -1



def llm_generate(requests, task, model, temperature=1.0, sample_cnt=1, dataset='wino'):
    results = []
    if task == CRITIC_RATE_COT:
        model_path = f'./result/{model}'
        model = SentenceTransformer(model_path).cuda()
        for message in requests:
            query, text = message.split('[START]')
            query_embedding = model.encode(query).reshape(1,-1)
            text_embedding = model.encode(text).reshape(1,-1)
            rate = cosine_similarity(query_embedding, text_embedding).item()
            results.append([message, rate])
    elif task == RERANK_KNOWLEDGE:
        model_path = f'./result/{model}'
        model = CrossEncoder(model_path, max_length=512)
        # model = SentenceTransformer(model_path).cuda()
        for message in requests:
            query, text = message.split('[SEP]')
            rate = model.predict([query, text])
            # query_embedding = model.encode(query).reshape(1,-1)
            # text_embedding = model.encode(text).reshape(1,-1)
            # rate = cosine_similarity(query_embedding, text_embedding).item()
            results.append([message, rate])
    elif task == CLASSIFY_KNOWLEDGE:
        device = 'cuda:0'        
        llama_path = "./model/Llama-2-7b-chat-hf"
        lora_path = f"./result/{model}"
        tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        model = LlamaForSequenceClassification.from_pretrained(
            llama_path, num_labels=4
        ).half().to(device)
        tokenizer = LlamaTokenizer.from_pretrained(
            # model_path,
            llama_path,
            padding_side = 'left',
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            # torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()
        model.eval()
        for message in tqdm(requests):
            inputs = tokenizer(message, return_tensors="pt").to(device)
            result = model(**inputs)
            pred = torch.argmax(result['logits'], axis=1).item()
            results.append([message, pred])
            del result
            del pred            
    elif task == READER_KNOWLEDGE_ANSWER:
        # llama_path = "/mnt/publiccache/huggingface/Llama-2-7b-chat-hf"
        # tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        # device = 'cuda'
        # model = LlamaForCausalLM.from_pretrained(
        #     llama_path,
        #     # load_in_8bit=False,
        #     # torch_dtype=torch.float16,
        #     # device_map="auto",
        # ).bfloat16().to(device)
        # config = LoraConfig(
        #     r=LORA_R,
        #     lora_alpha=LORA_ALPHA,
        #     target_modules=TARGET_MODULES,
        #     lora_dropout=LORA_DROPOUT,
        #     bias="none",
        #     inference_mode=False, 
        #     task_type=TaskType.CAUSAL_LM,
        # )

        # model = get_peft_model(model, config)
        llama_path = "/mnt/publiccache/huggingface/Llama-2-7b-chat-hf"
        LORA_PATH = f"./result/{model}"
        tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        device = 'cuda'
        model = LlamaForCausalLM.from_pretrained(
            llama_path,
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            # device_map="auto",
        ).bfloat16().to(device)
        model = PeftModel.from_pretrained(
            model,
            LORA_PATH,
        )
        model.eval()
        for message in tqdm(requests):
            # sentence1, sentence2, _ = message.split('[SEP2]')
            response = reader_generate(message, device, model, tokenizer)
            # response = compare_loss(sentence1, sentence2, device, model, tokenizer)
            results.append([message, response])
    elif task == READER_ANSWER:
        llama_path = "/mnt/publiccache/huggingface/Llama-2-7b-chat-hf"
        LORA_PATH = f"./result/{model}"
        tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        device = 'cuda'
        model = LlamaForCausalLM.from_pretrained(
            llama_path,
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            # device_map="auto",
        ).half().to(device)
        model = PeftModel.from_pretrained(
            model,
            LORA_PATH,
        )
        model.eval()
        for message in tqdm(requests):
            response = reader_generate(message, device, model, tokenizer)
            results.append([message, response])
    elif task == UNIFIED_ANSWER:
        tokenizer = AutoTokenizer.from_pretrained('/mnt/publiccache/huggingface/unifiedqa-t5-3b')
        model = AutoModelForSeq2SeqLM.from_pretrained('/mnt/publiccache/huggingface/unifiedqa-t5-3b', device_map='auto')
        
        model.eval()
        results = []
        for message in tqdm(requests):
            input_ids = tokenizer(message, return_tensors="pt")['input_ids'].to(model.device)
            result = model.generate(input_ids=input_ids)
            response = tokenizer.decode(result[0], skip_special_tokens=True)
            # pred = torch.argmax(result['logits'], axis=1).item()
            results.append([message, response])
    elif task == ROBERTA_ANSWER:
        device = 'cuda:0'
        if dataset == 'wino':
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = RobertaForSequenceClassification.from_pretrained(model).to(device)
            with torch.no_grad():
                for message in requests:
                    sentence1, sentence2 = message.split('[SEP]')
                    encoding1 = tokenizer(sentence1, return_tensors="pt").to(device)
                    result1 = model(**encoding1)
                    pred1 = torch.argmax(result1['logits'], axis=1).item()
                    encoding2 = tokenizer(sentence2, return_tensors="pt").to(device)
                    result2 = model(**encoding2)
                    pred2 = torch.argmax(result2['logits'], axis=1).item()
                    if pred1 == 0 and pred2 == 1:
                        results.append([message, 'Answer: (1)'])
                    elif pred2 == 0 and pred1 == 1:
                        results.append([message, 'Answer: (2)'])
                    else:
                        results.append([message, 'Answer: None'])
        else:
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = RobertaForMultipleChoice.from_pretrained(model).to(device)
            for message in requests:
                sentence_ls = message.split('[CLS]')
                questions = []
                choices = []
                for sent in sentence_ls:
                    question, choice = sent.split('[SEP]')
                    questions.append(question)
                    choices.append(choice)
                encoding = tokenizer(questions, choices, padding=True, return_tensors="pt").to(device)
                result = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
                pred = torch.argmax(result['logits'], axis=1).item()
                answer = f'Answer: ({str(pred + 1)})'
                results.append([message, answer])
    else:
        asyncio.run(
            process_api_requests(
                request_ls = requests,
                request_url = request_url,
                api_key=OPENAI_API_KEY,
                model=model,
                max_requests_per_minute=float(max_requests_per_minute),
                max_tokens_per_minute=float(max_tokens_per_minute),
                results=results,
                temperature=temperature,
                sample_cnt=sample_cnt
            )
           
        )
                
    return results

def reader_generate(input, device, model, tokenizer, max_new_tokens=16):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("Answer: ")[-1].strip()


def compare_loss(sentence1, sentence2, device, model, tokenizer):
    
    def cal_loss(inputs, device=device, model=model, tokenizer=tokenizer):
        question, answer = inputs.split('[SEP1]')
        with torch.no_grad():
            len_question_tokens = (
                len(
                    tokenizer(
                        question,
                        truncation=True,
                        max_length=256,
                    )["input_ids"]
                )
            ) 
            full_token = tokenizer(
                question + answer,
                truncation=True,
                max_length=256,
            )['input_ids']
            
            labels = [-100] * len_question_tokens + full_token[len_question_tokens:]
            labels = torch.tensor(np.array(labels))
            full_token = torch.tensor(np.array(full_token))
            full_token.to(device)
            labels.to(device)
            loss = model(full_token, labels=labels)['loss'] 
            return loss
    
    if cal_loss(sentence1) < cal_loss(sentence2):
        return "(1)"
    else:
        return "(2)"   


def embedding_examples():
    model_path = './model/all-mpnet-base-v2'
    file_path = './data/classifier_train/wino_train_data_2_class.json'
    result_path = './result/example_pool/wino_improve_knowledge.json'
    model = SentenceTransformer(model_path).cuda()
  
    q_embedding_dic = {}
    q_knowledge_dic = {}
    with open(file_path, 'r') as f:
        js_data = json.load(f)
        for data in tqdm(js_data):
            question = data['question']
            knowledge = data['knowledge']
            label = data['label']
            if question not in q_embedding_dic.keys():
                embedding = model.encode(question).tolist()
                q_embedding_dic[question] = embedding
                q_knowledge_dic[question] = {label:[knowledge]}
            else:
                if label not in q_knowledge_dic[question].keys():
                    q_knowledge_dic[question][label] = [knowledge]
                else:
                    q_knowledge_dic[question][label].append(knowledge)
        f.close()
    
    results = []
    for question in q_embedding_dic.keys():
        embedding = q_embedding_dic[question]
        knowledge_ls = q_knowledge_dic[question]
        message = {'question':question, 'embedding':embedding, 'knowledge':knowledge_ls}
        results.append(message)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def data_recall(faiss_index, model, query, top_k):
    query_embedding = model.encode([query])
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index


def create_index(datas_embedding):
    index = faiss.IndexFlatL2(datas_embedding.shape[1])  
    index.add(datas_embedding) 
    return index


def create_faiss():
    
    file_path = './result/example_pool/wino_improve_knowledge.json'
    result_path = './result/example_pool/wino_improve_knowledge.index'

    with open(file_path, 'r') as f:
        js_data = json.load(f)
        f.close()
        
    data_embeddings = []
    for data in js_data:
        embedding = np.array(data['embedding'])
        data_embeddings.append(embedding)
    data_embeddings = np.array(data_embeddings)
    faiss_index = create_index(data_embeddings)
    faiss.write_index(faiss_index, result_path)
    return faiss_index


def get_related_example(question, model, data, index, top_k=3):
    results = []
    sim_data_Index = data_recall(index, model, question, top_k)
    for idx in sim_data_Index[0]:
        tup = data[idx]
        q = tup['question']
        k = random.choice(tup['knowledge'])
        results.append([q,k])
    return results


def get_tf_example(question, model, data, index, top_k=3):
    results = []
    sim_data_Index = data_recall(index, model, question, top_k)
    for idx in sim_data_Index[0]:
        tup = data[idx]
        q = tup['question']
        k = random.choice(tup['knowledge'])
        results.append([q,k])
    return results

def get_related_knowledge(question, model, data, index, top_k=3):
    results = []
    sim_data_Index = data_recall(index, model, question, top_k)
    for idx in sim_data_Index[0]:
        tup = data[idx]
        k = tup['knowledge']
        results.append(k)
    return results

def reader_infer(requests, temperature):
    results = []
    llama_path = "./model/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    # device = 'cuda:3'
    model = LlamaForCausalLM.from_pretrained(
        llama_path,
        # load_in_8bit=False,
        # torch_dtype=torch.float16,
        device_map="auto",
    ).half()
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        inference_mode=False, 
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    model.eval()
        
    def evaluate(
        input,
        temperature=1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=8,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            # num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output
        
    for message in tqdm(requests):
        response = evaluate(message, temperature=temperature)
        results.append([message, response])
    return 

# def ft_reader_infer(question):
#     llama_path = "./model/Llama-2-7b-chat-hf"
#         lora_path = f"./result/{model}"
#         tokenizer = LlamaTokenizer.from_pretrained(llama_path)
#         device = 'cuda:3'
#         model = LlamaForCausalLM.from_pretrained(
#             llama_path,
#             # load_in_8bit=False,
#             # torch_dtype=torch.float16,
#             # device_map="auto",
#         ).half().to(device)
#         model = PeftModel.from_pretrained(
#             model,
#             lora_path,
#             # torch_dtype=torch.float16,
#         )
#         model = model.merge_and_unload()
#         model.eval()
        
#         def evaluate(
#             input,
#             temperature=1,
#             top_p=0.75,
#             top_k=40,
#             num_beams=4,
#             max_new_tokens=16,
#             **kwargs,
#         ):
#             inputs = tokenizer(input, return_tensors="pt")
#             input_ids = inputs["input_ids"].to(device)
#             generation_config = GenerationConfig(
#                 temperature=temperature,
#                 # top_p=top_p,
#                 # top_k=top_k,
#                 # num_beams=num_beams,
#                 **kwargs,
#             )
#             with torch.no_grad():
#                 generation_output = model.generate(
#                     input_ids=input_ids,
#                     generation_config=generation_config,
#                     return_dict_in_generate=True,
#                     output_scores=True,
#                     max_new_tokens=max_new_tokens,
#                 )
#             s = generation_output.sequences[0]
#             output = tokenizer.decode(s)
#             return output.split("Response:")[1].strip()
        
#         for message in tqdm(requests):
#             response = evaluate(message, temperature=temperature)
#             results.append([message, response])