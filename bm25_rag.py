from rank_bm25 import BM25Okapi
import json
import json
from config import *
from process import InputParser, Recorder, OutputParser, Probe
from prompt import Prompter
from utils import llm_generate
from tqdm import tqdm 

data_path = '/mnt/userdata/ljc/code/linked_code/data/atomic2020/knowledge_sentence.json'

with open(data_path, 'r') as f:
    data = json.load(f)

documents = [x['knowledge'] for x  in data]
tokenized_documents = [doc.split() for doc in documents]

# 初始化BM25
bm25 = BM25Okapi(tokenized_documents)


data_length = 500
dataset = 'wino'
if dataset in ['siqa']:
    shuffle = True 
else:
    shuffle = False
input_parser = InputParser(dataset, data_length, split='dev', shuffle=shuffle)
recorder = Recorder()
prompter = Prompter(dataset)
output_parser = OutputParser()
requests = []
for data in tqdm(input_parser):
    question = data[1]
    recorder.log_message(data[0], data[1], data[2], data[3])
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_indices = bm25_scores.argsort()[::-1][:3]  # 取前3个相关文档
    retrieved_docs = [documents[i] for i in top_n_indices]
    knowledge = (' ').join(retrieved_docs)
    content = 'Knowledge: ' + knowledge + '\nQuestion: ' + question
    messages = prompter.wrap_input(content, task=KNOWLEDGE_ANSWER, icl_cnt=3) 
    query = messages[-1]['content']
    recorder.log_query_question_map(query, question)
    requests.append(messages)
results = llm_generate(requests, task=KNOWLEDGE_ANSWER, model='gpt-3.5-turbo-0613', sample_cnt=1)
output_parser.parse(results, KNOWLEDGE_ANSWER)
temp_file = f'./result/{dataset}/bm25_knowledge.json'
output_parser.dump(temp_file)
probe_targets = [   GET_MATCH,   
                    GET_QUERY, 
                    GET_LABEL,
                    GET_PRED]
probe = Probe(probe_targets)

for data in output_parser:
    query = data[0]
    answer = data[1]
    pred = data[2]
    question = recorder.get_question(query)
    label = recorder.get_label(question)
    options = recorder.get_options(question)
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


with open(f'./result/{dataset}/bm25rag.json', 'w', encoding='utf-8') as f:
    json.dump(knowledge_classify_js, f, indent=4)
