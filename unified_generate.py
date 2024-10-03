import faulthandler
faulthandler.enable()
import json
from config import *
from process import InputParser, Recorder, OutputParser, Probe
from prompt import Prompter
import random 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
random.seed(17)   
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('/mnt/publiccache/huggingface/unifiedqa-t5-3b')
model = AutoModelForSeq2SeqLM.from_pretrained('/mnt/publiccache/huggingface/unifiedqa-t5-3b', device_map='auto')
data_length = 500
dataset = 'wino'
if dataset == 'siqa':
    shuffle = True
else:
    shuffle = False
input_parser = InputParser(dataset, data_length, split='dev', shuffle=shuffle)
recorder = Recorder()
prompter = Prompter(dataset)
output_parser = OutputParser()

requests = []
for data in input_parser:
    stem = data[0]
    question = data[1]
    recorder.log_message(data[0], data[1], data[2], data[3])
    query = question
    recorder.log_query_question_map(query, question)
    requests.append(query)

model.eval()
results = []
for message in tqdm(requests):
    input_ids = tokenizer(message, return_tensors="pt")['input_ids'].to(model.device)
    result = model.generate(input_ids=input_ids)
    response = tokenizer.decode(result[0], skip_special_tokens=True)
    # pred = torch.argmax(result['logits'], axis=1).item()
    results.append([message, response])



probe_targets = [   
                        GET_MATCH,   
                        GET_QUERY, 
                        GET_LABEL,
                        GET_PRED]
probe = Probe(probe_targets)
rouge = Rouge()
    
for data in results:
    query = data[0]
    answer = data[1]
    question = recorder.get_question(query)
    label = recorder.get_label(question)
    options = recorder.get_options(question)
    preds = [-1]
    for i in range(len(options)):
        rouge_score = rouge.get_scores(hyps=answer, refs=options[i], avg=False)
        preds.append(rouge_score[0]['rouge-l']['f'])
    pred = np.argmax(np.array(preds))
    pred = str(pred)
    message = {'question':question, 'query':query, 'answer':answer, 'option':options, 'pred':pred, 'label':label, 'score':None}
    probe.get_info(**message)

probe.cal_info()
knowledge_classify_js = [] 
for message in probe:
    match = message[GET_MATCH]
    question = message[GET_QUERY][0]
    label = message[GET_LABEL]
    pred = message[GET_PRED]
    message = {'question':question, 'match':match, 'label':label, 'pred':pred}
    knowledge_classify_js.append(message)


with open(f'./result/{dataset}/unifiqa.json', 'w', encoding='utf-8') as f:
    json.dump(knowledge_classify_js, f, indent=4)
