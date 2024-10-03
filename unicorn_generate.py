import random
import json
from process import InputParser
random.seed(17)

dataset = 'hella'
if dataset in ['siqa']:
    shuffle = True
else:
    shuffle = False

result_path = f"./result/{dataset}/unicorn.txt"

preds = []
with open(result_path, 'r') as f:
    for line in f:
        if dataset in ['wino', 'siqa']:
            preds.append(line.strip())
        else:
            pred = str(eval(line.strip()) + 1)
            preds.append(pred)

    
input_parser = InputParser(dataset, 2000, split='dev', shuffle=False)

idx = 0
results = []
for data in input_parser:
    question = data[1]
    label = data[3]
    pred = preds[idx]
    match = (pred == label)
    message = {'question':question, 'match':match, 'label':label, 'pred':pred}
    idx += 1
    results.append(message)
if shuffle:
    random.shuffle(results)
results = results[:500]

with open(f'./result/{dataset}/unicorn.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)