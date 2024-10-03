import json
from openai_chat import chat_generate

method = 'direct'
dataset = 'gsm8k'

data_path = f'./data/{dataset}/test.jsonl'
example_path = f'./data/{dataset}/{method}.json'
data = []
with open(data_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        data.append(item)
with open(example_path, 'r') as f:
    examples = json.load(f) 
instruction = "Please act as a math teacher to answer the given math question"
prompts = []
prompts.append({'role':'system', 'content':instruction})
for example in examples:
    prompts.append({'role':'user', 'content':'Question: ' + example['question']})
    prompts.append({'role':'assistant', 'content':example['answer']})

input_labels = {}
inputs = []
for item in data[:2]:
    question = 'Question: ' + item['question']
    if dataset == 'gsm8k':
        label = item['answer'].split('####')[-1].strip()
    else:
        label = '-1'
    input_labels[question] = label
    input = prompts + [{'role':'user', 'content':question}]
    print(input)
    inputs.append(input)
    
model = 'gpt-3.5-turbo'
responses = chat_generate(inputs, model)
results = []
correct = 0
for result in responses:
    query = result[0]['messages'][-1]['content']
    result = result[-1]['choices'][0]['message']['content']
    label = input_labels[query]
    pred = result.split(':')[-1].strip()
    if label in pred:
        correct += 1
        match = True
    else:
        match = False 
    msg = {'question':query, 'result':result, 'label':label, 'pred':pred, 'match':match}
    results.append(msg)    
    
result_path = f"./result/{dataset}/{method}.json"
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)

print(correct)