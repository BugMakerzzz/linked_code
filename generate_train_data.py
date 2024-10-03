import faulthandler
faulthandler.enable()
import json
from config import *
from process import InputParser, Recorder, OutputParser, Probe
from prompt import Prompter
from utils import llm_generate
import random 
random.seed(17)
# def generate_reader_train_data():
#     data_length = 10000
#     input_parser = InputParser('wino', data_length, split='train')
#     recorder = Recorder()
#     prompter = Prompter('wino')
#     output_parser = OutputParser()

#     requests = []
#     for data in input_parser:
#         stem = data[0]
#         question = data[1]
#         options = data[2]
#         label = data[3]
#         recorder.log_message(data[0], data[1], data[2], data[3])
#         content = "Question: " + question + f'\n Hint: Answer: ({label}) ' + str(options[eval(label) - 1])
#         messages = prompter.wrap_input(content, task=HINT_GENERATE_KNOWLEDGE, icl_cnt=5) 
#         query = messages[-1]['content']
#         recorder.log_query_question_map(query, question)
#         requests.append(messages)

#     results = llm_generate(requests, model='gpt-3.5-turbo')
#     output_parser.parse(results, HINT_GENERATE_KNOWLEDGE)
#     knowldege_path = './data/reader_train/wino_train_data.json'
#     with open(knowldege_path, 'w', encoding='utf-8') as f:
#         messages = []
#         for data in output_parser:
#             query = data[0]
#             answer = data[1]
#             pred = data[2]
#             question = recorder.get_question(query)
#             label = recorder.get_label(question)
#             # print(label)
#             options = recorder.get_options(question)
#             knowledge = answer
#             answer = f"Answer: ({label}) {options[eval(label)-1]}"
#             message = {'Question':question, 'Knowledge':knowledge, 'Answer':answer}
#             messages.append(message)
#         json.dump(messages, f, indent=4)
#         f.close()
        
        
def record_wrong_cot_train_data():
    data_length = 1000
    input_parser = InputParser('wino', data_length, split='train', shuffle=True)
    recorder = Recorder()
    prompter = Prompter('wino')
    output_parser = OutputParser()

    requests = []
    for data in input_parser:
        stem = data[0]
        question = data[1]
        options = data[2]
        label = data[3]
        recorder.log_message(data[0], data[1], data[2], data[3])
        content = question
        messages = prompter.wrap_input(content, task=BASE_COT_ANSWER, icl_cnt=5) 
        query = messages[-1]['content']
        recorder.log_query_question_map(query, question)
        requests.append(messages)

    results = llm_generate(requests, model='gpt-3.5-turbo-0613', sample_cnt=5, temperature=1.3)
    output_parser.parse(results, BASE_COT_ANSWER)
    mid_file_path = './data/critic_train/wrong_question_log_m.json'
    
    probe_targets = [   GET_QUERY, 
                        GET_ANSWER, 
                        GET_PRED, 
                        GET_LABEL,
                        GET_SC_CNT]
    
    probe = Probe(probe_targets)
    for data in output_parser:
        query = data[0]
        answer = data[1]
        pred = data[2]
        question = recorder.get_question(query)
        label = recorder.get_label(question)
        options = recorder.get_options(question)
        message = {'question':question, 'query':query, 'pred':pred, 'answer':answer, 'option':options, 'label':label}
        probe.get_info(**message)
    
    probe.cal_info()
    correct = 0
    with open(mid_file_path, 'w', encoding='utf-8') as f:
        messages = []
        for message in probe:
            query = tuple(message[GET_QUERY])[0]
            pred = message[GET_PRED]
            label = message[GET_LABEL]
            answer = message[GET_ANSWER]
            sc_cnt = message[GET_SC_CNT]
            if len(sc_cnt.keys()) == 1 and pred[0] == label:
                correct += 1
                continue
            log = {'question':query, 'pred':pred, 'label':label, 'answer':answer, 'sc_cnt':sc_cnt}
            messages.append(log)
        json.dump(messages, f, indent=4)
        f.close()
    print(f'Acc: {correct / data_length}')



def generate_cot_train_data():
    data_length = 5002
    input_parser = InputParser('wino', data_length, split='train', shuffle=True)
    recorder = Recorder()
    prompter = Prompter('wino')
    output_parser = OutputParser()

    
    for data in input_parser:
        question = data[1]
        options = data[2]
        label = data[3]
        recorder.log_message(data[0], data[1], data[2], data[3])
   
   
    mid_file_path = './data/critic_train/wrong_question_log_m.json'
    with open(mid_file_path, 'r', encoding='utf-8') as f:
        dic = json.load(f)

    requests = []
    for tup in dic:
        question = tup['question']
        label = recorder.get_label(question)
        options = recorder.get_options(question)
        answer = f"Answer: ({label}) {options[eval(label)-1]}"
        content = "Question: " + question + f'\n Hint: {answer}'
        messages = prompter.wrap_input(content, task=HINT_GENERATE_COT, icl_cnt=5) 
        query = messages[-1]['content']
        recorder.log_query_question_map(query, question)
        requests.append(messages)

    results = llm_generate(requests, model='gpt-3.5-turbo-0613', sample_cnt=5, temperature=1.3)
    output_parser.parse(results, HINT_GENERATE_COT)
    
    
    cot_js = []

    for data in output_parser:
        query = data[0]
        answer = data[1]
        question = recorder.get_question(query)
        message = {'question':question, 'cot':answer, 'label': True}
        cot_js.append(message)
    
    requests = []
    for tup in dic:
        question = tup['question']
        label = recorder.get_label(question)
        wrong_label = '1' if label == '2' else '2'
        options = recorder.get_options(question)
        wrong_answer = f"Answer: ({wrong_label}) {options[eval(wrong_label)-1]}"
        content = "Question: " + question + f'\n Hint: {wrong_answer}'
        messages = prompter.wrap_input(content, task=HINT_GENERATE_WRONG_COT, icl_cnt=5) 
        query = messages[-1]['content']
        recorder.log_query_question_map(query, question)
        requests.append(messages)

    results = llm_generate(requests, model='gpt-3.5-turbo-0613', sample_cnt=5, temperature=1.3)
    output_parser.parse(results, HINT_GENERATE_WRONG_COT)

    for data in output_parser:
        query = data[0]
        answer = data[1]
        question = recorder.get_question(query)
        message = {'question':question, 'cot':answer, 'label': False}
        cot_js.append(message)
        
    cot_file_path = './data/critic_train/cot.json'
    with open(cot_file_path, 'w', encoding='utf-8') as f: 
        json.dump(cot_js, f, indent=4)
    
    
def test():
    wino_questions = [
    """Question: John fell down while walking on the pavement and decided to take the lawn instead. The _ is dry.\n(1) pavement (2) lawn \n Hint: Answer: (2) lawn",
    """
    ]
    
    wino_answers = [
    """Answer: (1) Jeffrey
    """
    ]
    
    
    prompter = Prompter('wino')
    requests = []
    for question in wino_questions:
        idx = wino_questions.index(question)
        content = 'Question: ' + question + "Hint: " + wino_answers[idx]
        # content += "Let's think step by step: "
        message = prompter.wrap_input(content, HINT_GENERATE_WRONG_COT, 4)
        requests.append(message)
    results = llm_generate(requests, model='gpt-3.5-turbo-0613')
    output_parser = OutputParser()
    output_parser.parse(results, HINT_GENERATE_WRONG_COT)
    for data in output_parser:
        print(data[0])
        print(data[1])


def generate_classifier_train_question():
    data_length = 2000
    # data_length = 1
    dataset = 'siqa'
    input_parser = InputParser(dataset, data_length, split='train', shuffle=True)
    recorder = Recorder()
    prompter = Prompter(dataset)
    output_parser = OutputParser()
    requests = []
    for data in input_parser:
        question = data[1]
        recorder.log_message(data[0], data[1], data[2], data[3])
        content = question
        messages = prompter.wrap_input(content, task=DIERCT_ANSWER, icl_cnt=3) 
        query = messages[-1]['content']
        recorder.log_query_question_map(query, question)
        requests.append(messages)

    results = llm_generate(requests, task=DIERCT_ANSWER, model='gpt-3.5-turbo-0613', sample_cnt=5)
    output_parser.parse(results, DIERCT_ANSWER)
    
    probe_targets = [   
                        GET_MATCH,   
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
        message = {'question':question, 'query':query, 'pred':pred, 'answer':answer, 'option':options, 'label':label, 'score':None}
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
    
   
    with open(f'./data/classifier_train/siqa_question_label.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_classify_js, f, indent=4)
    


def generate_classifier_train_knowledge():
    data_length = 2000
    input_parser = InputParser('siqa', data_length, split='train', shuffle=True)
    recorder = Recorder()
    prompter = Prompter('siqa')
    output_parser = OutputParser()
    
    with open('./data/classifier_train/siqa_question_label.json', 'r') as f:
       question_match_js = json.load(f)
    
    question_match_dic = {}
    for js_data in question_match_js:
        question_match_dic[js_data['question']] = js_data['match']
    
    recorder = Recorder()
    
    results = []
    requests = []
    for data in input_parser:
        stem = data[0]
        question = data[1]
        recorder.log_message(data[0], data[1], data[2], data[3])
        content = question
        messages = prompter.wrap_input(content, task=GENERATE_KNOWLEDGE, icl_cnt=3) 
        query = messages[-1]['content']
        recorder.log_query_question_map(query, question)
        requests.append(messages)
    results = llm_generate(requests, model='gpt-3.5-turbo-0613', task=GENERATE_KNOWLEDGE, temperature=1.3, sample_cnt=5)

    output_parser.parse(results, GENERATE_KNOWLEDGE)
    
    temp_file = './data/reward_train/' + f'siqa_knowledge.json'
    output_parser.dump(temp_file)
    # output_parser.load(temp_file)
    
    # max_cnt = 3000
    # cnt = 0
    requests = []
    for data in output_parser:
        query = data[0]
        answer = data[1]
        question = recorder.get_question(query)
        content = answer + '\nQuestion: ' + question
        messages = prompter.wrap_input(content, task=KNOWLEDGE_ANSWER, icl_cnt=5)
        new_query = messages[-1]['content']
        recorder.update_query_question_map(query, new_query)
        requests.append(messages)
        # cnt += 1
        
    results = llm_generate(requests, model='gpt-3.5-turbo-0613', task=KNOWLEDGE_ANSWER, sample_cnt=3)
    output_parser.parse(results, KNOWLEDGE_ANSWER)
    
    query_preds_dic = {}
    for data in output_parser:
        query = data[0]
        answer = data[1]
        pred = data[2]
        if query not in query_preds_dic.keys():
            query_preds_dic[query] = [pred]
        else:
            query_preds_dic[query].append(pred)
            
            
    knowledge_classify_js = []        
    for query, preds in query_preds_dic.items():
        question = recorder.get_question(query)
        label = recorder.get_label(question)
        knowledge = query.split('\nQuestion: ')[0]
        cnt = 0
        for pred in preds:
            if pred == label:
                cnt += 1
        if question_match_dic[question]:
            q_label = 1
        else:
            q_label = 0
        if cnt > len(preds) // 2:
            if q_label:
                label = 1
            else:
                label = 0
            message = {'question':question, 'knowledge':knowledge, 'q_label': q_label, 'k_label':0, 'label':label}
        else:
            if q_label:
                label = 3
            else:
                label = 2
            message = {'question':question, 'knowledge':knowledge, 'q_label': q_label, 'k_label':1, 'label':label}
        knowledge_classify_js.append(message)
    
    knowledge_classify_path = './data/reward_train/siqa_knowldege_label.json'
    with open(knowledge_classify_path, 'w', encoding='utf-8') as f: 
        json.dump(knowledge_classify_js, f, indent=4)


def generate_classifier_train_data():
    # max_count_dic = {1:4000, 3:4000}
    # count_dic = { 1:0, 3:0}
  
    with open('./data/reward_train/siqa_knowldege_label.json', 'r') as f:
        js_data = json.load(f)
        f.close()
    # with open('./data/reward_train/hella_knowldege_label_5k_2.json', 'r') as f:
    #     js_data_2 = json.load(f)
    #     f.close()
    
    knowledge_pool = {}
    for data in js_data:
        question = data['question']
        knowledge = data['knowledge']
        label = data['k_label']
        if question not in knowledge_pool.keys():
            knowledge_pool[question] = {label:[knowledge]}
        else:
            if label in knowledge_pool[question].keys():
                knowledge_pool[question][label].append(knowledge)
            else:
                knowledge_pool[question][label] = [knowledge]
    
    result_js = []    
    for k, v in knowledge_pool.items():
        question = k
        knowledge_label = v
        if len(knowledge_label.keys()) <= 1:
            continue
        for label, knowledge_ls in knowledge_label.items():
            for knowledge in knowledge_ls:
                message = {'question':question, 'knowledge':knowledge, 'label':label}
                result_js.append(message)
    with open('./data/reward_train/siqa_train_data.json', 'w') as f:
        json.dump(result_js, f, indent=4)
generate_classifier_train_data()