from process import *
from config import *
from prompt import Prompter
from utils import *
import random
import argparse
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
# parser = argparse.ArgumentParser()
# parser.add_argument('-t1', '--task1', type=int, help='Radius of cylinder')

random.seed(17)

def run(dataset, load_knowledge=False):
    input_parser = InputParser(dataset, data_length, shuffle=shuffle)
    recorder = Recorder()
    prompter = Prompter(dataset)
    logger = set_log()
    logger.info('-'*40 + 'Start logging' + '-'*40)
    logger.info(f'Debug Mode: {debug_mode}')
    logger.info(f'Dataset: {dataset}')
    logger.info(f'Data length: {data_length}')
    logger.info(f'Shuffle Data: {shuffle}')


    inputs = [] 
    cnt = 0
    tokens = 0
    for data in input_parser:
        recorder.log_message(data[0], data[1], data[2], data[3])
        recorder.log_query_question_map(cnt, question=data[1])
        inputs.append([cnt, "", ""])
        cnt += 1
        # stem = data[0]
        # question = data[1]
        # if task_steps[0] == REWRITE_QUESTION:
        #     content = question
        # else:
        #     content = stem  
        # messages = prompter.wrap_input(content, task=task_steps[0], icl_cnt=icl_steps[0]) 
        # query = messages[-1]['content']
        # recorder.log_query_question_map(query, question)
        # inputs.append([query, "", ""])
    
    
    def run_step(inputs, step):
        icl_cnt = icl_steps[step]
        sample_cnt = sample_steps[step]
        temperature = temperature_steps[step]   
        task = task_steps[step]
        model = openai_model
        requests = []
        question_set = set()
        for data in inputs: 
            task = task_steps[step]
            query = data[0]
            answer = data[1]
            value = data[2]
            question = recorder.get_question(query)
            stem = recorder.get_question_stem(question)
            label = recorder.get_label(question)
            options = recorder.get_options(question)

            if task in [GENERATE_ONE_KNOWLEDGE, GENERATE_KNOWLEDGE, GENERATE_APP_KNOWLEDGE]:
                if step != 0:
                    content = answer
                else:
                    if dataset == 'wino':
                        content = stem
                    else:
                        content = question
            elif task == EVALUATE_KNOWLEDGE:
                knowledge = answer
                content = knowledge
            elif task == RERANK_KNOWLEDGE:
                model = rerank_checkpoint
                knowledge = answer
                # content =  knowledge + '\nQuestion: ' + question
                content = question + '[SEP]' + knowledge
            elif task == RETRIEVE_KNOWLEDGE:
                knowledge = prompter.get_related_atomic_knowledge(question)
                content =  'Knowledge: ' + knowledge + '\nQuestion: ' + question
            elif task == TRANSFORM_KNOWLEDGE:   
                knowledge = answer
                content = 'Question: ' + stem + '\n' + knowledge
            elif task == ROBERTA_ANSWER:
                model = roberta_checkpoint
                knowledge = answer
                if dataset == 'wino':
                    sentence1 = knowledge + 'Question: ' + stem.replace('_', options[0])
                    sentence2 = knowledge + 'Question: ' + stem.replace('_', options[1])
                    content = sentence1 + "[SEP]" + sentence2
                elif dataset == 'siqa':
                    sentences = []
                    for i in range(3):
                        sentences.append(stem + '[SEP]' + options[i])
                    content = ('[CLS]').join(sentences)  
                elif dataset == 'piqa':
                    sentences = []
                    for i in range(2):
                        sentences.append(stem + '[SEP]' + options[i])
                    content = ('[CLS]').join(sentences)  
                else:
                    sentences = []
                    for i in range(4):
                        sentences.append(stem + '[SEP]' + options[i])
                    content = ('[CLS]').join(sentences)   
            elif task == INFER_ANSWER:
                infer = answer 
                content = 'Question: ' + question + '\n' + infer + ' So the answer is: ' 
            elif task == ASSESS_QUESTION_KNOWLEDGE:
                knowledge = answer
                content = question + '[SEP]' + knowledge
            elif task in [KNOWLEDGE_ANSWER, KNOWLEDGE_BASE_COT_ANSWER]:
                knowledge = answer
                if knowledge == "Knowledge:":
                    content = question
                    task = DIERCT_ANSWER
                else:
                    content = knowledge + '\nQuestion: ' + question 
            elif task == READER_ANSWER:
                model = reader_checkpoint
                content = question

            elif task == GENERATE_PYKE:
                knowledge = answer
                content = knowledge + '\nQuestion: ' + question
            elif task == READER_KNOWLEDGE_ANSWER:
                model = reader_checkpoint
                knowledge = answer 
                content = question + '[SEP]' + knowledge
            elif task == HINT_GENERATE_KNOWLEDGE:
                statement = answer
                content = statement
            elif task == GENERATE_BASE_COT:
                content = 'Question: ' + question
            elif task == BASE_COT_ANSWER:
                content = question
            elif task == CRITIC_RATE_COT:
                model = 'critic_model'
                content = question + '[START]' + answer
            # elif task == CLASSIFY_KNOWLEDGE:
            #     knowledge = answer
            #     model = classifier_checkpoint
            #     content = question + '[SEP]' + knowledge
            elif task in [DIERCT_ANSWER, GENERATE_SUP_QUESTION, REWRITE_QUESTION]:
                if question in question_set:
                    continue
                question_set.add(question)
                content = question
            elif task == GENERATE_QUERY:
                content = question
            elif task == QUERY_ANSWER:
                content = answer
            elif task == REFLECT_GENERATE_KNOWLEDGE:
                knowledge = answer
                content = 'Question: ' + question + '\nWrong Knowledge:' + knowledge.strip('Knowledge:')
                if value >= top_k - 1:
                    messages = prompter.wrap_input(content, task=task, icl_cnt=icl_cnt) 
                    new_query = messages[-1]['content']
                    recorder.update_query_question_map(query, new_query)
                    continue
            
            
            messages = prompter.wrap_input(content, task=task, icl_cnt=icl_cnt) 
            
            if task in [READER_KNOWLEDGE_ANSWER, CRITIC_RATE_COT, CLASSIFY_KNOWLEDGE, RERANK_KNOWLEDGE, READER_ANSWER, ROBERTA_ANSWER]: 
                new_query = messages
            else:
                new_query = messages[-1]['content']
            recorder.update_query_question_map(query, new_query)
            
            if load_knowledge:
                continue
            requests.append(messages)
            
            # if extra_content:
            #     messages = prompter.wrap_input(extra_content, task=task, icl_cnt=icl_cnt) 
            #     new_query = messages[-1]['content']
            #     recorder.update_query_question_map(query, new_query)
            #     requests.append(messages)
            
        logger.info(f'Task: {TASK_NAME_DICT[task]}')
        logger.info(f'Model name: {model}')
        logger.info(f'Sample count: {str(sample_cnt)}') 
        logger.info(f'Example count: {str(icl_cnt)}')
        logger.info(f'Temperature: {str(temperature)}')
        results = llm_generate(requests, task, model, temperature, sample_cnt, dataset)
        return results
    
    
    def add_question_knowledge_rate(inputs):
        # value_map = {0:1, 1:-1, 2:0, }
        value_map = {0:1, 1:0.5, 2:-0.5, 3:-1}
        for data in inputs:
            query = data[0]
            knowledge = data[0].split('\n')[-1]
            if not knowledge:
                continue
            rate = value_map[data[2]]
            question = recorder.get_question(query)
            recorder.add_knowledge(question, knowledge, rate)
    
        return 
    
    
    def sort_knowledge():
        results = []
        for question, knowledge_pool in recorder.traverse_knowledge_pools().items():
            if knowledge_pool:
                knowledge_pool = sorted(knowledge_pool.items(), key=lambda x:x[1])
            for i in range(len(knowledge_pool)):
                query = recorder.get_query(question)
                knowledge = knowledge_pool[i][0]
                message = [query, knowledge, i]
                results.append(message)
        return results
    
    
    def prepare_rerank_knowledge():
        results = []
        for question, knowledge_pool in recorder.traverse_knowledge_pools().items():
            for knowledge, value in knowledge_pool.items():
                if value > 0:
                    continue
                query = recorder.get_query(question)
                message = [query, knowledge, value]
                results.append(message)
        return results
    
    
    def collect_top_knowledge():
        logger.info(f'Top k: {top_k}')
        logger.info(f'Collect Method {collect_method}')
        # logger.info(f'Collect Order Asending')
        top_results = []
        
        for question, knowledge_pool in recorder.traverse_knowledge_pools().items():
            
            
            if knowledge_pool:
                knowledge_pool = sorted(knowledge_pool.items(), key=lambda x:x[1], reverse=True)
                if collect_method == 'none':
                    random.shuffle(knowledge_pool)
                
                if top_k > len(knowledge_pool):
                    max_idx = len(knowledge_pool)
                else:
                    max_idx = top_k
                
                # knowledge_ls = []
                # knowledge = "Knowledge:"
                for i in range(max_idx-1, -1, -1):
                # for i in range(max_idx):
                    knowledge = "Knowledge:"
                    k, r = knowledge_pool[i]
                    k_ls = k.split("Knowledge:")
                    if len(k_ls) == 2:
                        knowledge += k_ls[1]
                    else:
                        knowledge += k
                    recorder.add_out_knowledge_score(question, r)
                    # if r >= 0 or collect_method == 'none':
                    # knowledge_ls.append([k, r])
                
                
                    query = recorder.get_query(question)
            # # append_knowledge = knowledge_ls[-1][0]
            # vote_num = 0
            
            # while vote_num < len(knowledge_ls):
            #     k = knowledge_ls[vote_num][0]
            #     for i in range(vote_num+1):
            #         top_results.append([query, k, ""])
            #     vote_num += 1
            # for i in range(vote_num + 1):
            # for tup in knowledge_ls:
            #     top_results.append([query, tup[0], ""])
            # raw_knowledge = knowledge
            # top_results.append([query, raw_knowledge, ""])
            # for i in range(2):
            #     knowledge += raw_knowledge.split('Knowledge')[1]
                top_results.append([query, knowledge, ""])
            # top_results.append([query, append_knowledge, ""])            
        return top_results
    
    
   
    def add_rerank_knowledge(inputs):
        results = []
        for data in inputs:
            query = data[0]
            answer = data[1]
            rate = data[2]
            # knowledge = query.split('\n')[0]
            knowledge = query.split('[SEP]')[-1]
            question = recorder.get_question(query)
            recorder.add_knowledge(question, knowledge, rate)
            results.append([query, answer, rate])
        return results
    
    
    def add_generate_knowledge(inputs):
        results = []
        for data in inputs:
            query = data[0]
            knowledge = data[1]
            question = recorder.get_question(query)
            recorder.add_knowledge(question, knowledge)
            results.append([query, knowledge, ''])
        return results
    
    
    def filter_cot(inputs):
        results = []
        question_best_rate = {}
        question_best_cot = {}
        question_query = {}
        for data in inputs:
            query = data[0]
            cot = query.split('[START]')[-1].split('[END]')[0]
            question = recorder.get_question(query)
            rate = data[1]
            if question not in question_best_rate.keys() or rate > question_best_rate[question]:
                question_best_rate[question] = rate
                question_best_cot[question] = cot
                question_query[question] = query
        for question, cot in question_best_cot.items():
            results.append([question_query[question], cot, question_best_rate[question]])
        return results
    
    
    def cons_answer(inputs):
        results = []
        for data in inputs:
            query = data[0]
            answer = data[1]
            value = data[2]
            statement = query.split('Statement 2: ')[-1]
            pred = recorder.transform_pred(statement)
            if value > 0.5 or pred == 'None':
                results.append([query, answer, pred])
            else:
                if pred == '1':
                    pred = '2'
                else:
                    pred = '1'
                results.append([query, answer, pred])    
        return results    
            
        
    def prepare_hint(inputs):
        results = []
        for data in inputs:
            query = data[0]
            question = recorder.get_question(query)
            stem = recorder.get_question_stem(question)
            options = recorder.get_options(question)
            for option in options:
                if dataset == 'wino':
                    statement = stem.replace('_', option)
                elif dataset == 'hella':
                    statement = stem + " " + option
                message = [query, statement, ""]
                results.append(message) 
        return results 
    
    
    def filter_knowledge(inputs):
        for data in inputs:
            query = data[0]
            knowledge = query
            question = recorder.get_question(query)
            value = data[2]
            if value < 0:
                recorder.del_knowledge(question, knowledge)
        return 
    

    
    
    parser = OutputParser()
    for i in range(len(task_steps)):
        task = task_steps[i]
        
        if task == RERANK_KNOWLEDGE:
            inputs = prepare_rerank_knowledge()
        elif task == REFLECT_GENERATE_KNOWLEDGE:
            inputs = sort_knowledge()
        elif task in [KNOWLEDGE_ANSWER, KNOWLEDGE_BASE_COT_ANSWER, READER_KNOWLEDGE_ANSWER, READER_ANSWER, ROBERTA_ANSWER] and i != 0:
            inputs = collect_top_knowledge()
        elif task in [GENERATE_APP_KNOWLEDGE]:
            prompter.init_faiss(task)
        elif task == HINT_GENERATE_KNOWLEDGE:
            inputs = prepare_hint(inputs)
        
        
        results = run_step(inputs, i)

        if load_knowledge:
            logger.info(f'Load Mode: {load_knowledge}')
            load_knowledge = False
            logger.info(f'Load Path: {knowledge_path}')
            parser.load(knowledge_path)
        else:
            parser.parse(results, task)
        
        if debug_mode:
            print(results)
            for result in results:
                tokens += result[-1]['usage']['total_tokens']
        if task in [GENERATE_KNOWLEDGE, GENERATE_APP_KNOWLEDGE, REFLECT_GENERATE_KNOWLEDGE, HINT_GENERATE_KNOWLEDGE, DIERCT_ANSWER]:
            temp_file_name = f'_step{i+1}_{TASK_NAME_DICT[task]}.json'
            parser.dump(result_path+temp_file_name)
            
        inputs = parser
        if task == RERANK_KNOWLEDGE:
            inputs = add_rerank_knowledge(inputs)
        # elif task == REWRITE_QUESTION:
        #     inputs = update_question(inputs)
        elif task == EVALUATE_KNOWLEDGE:
            inputs = filter_knowledge(inputs)
        elif task == CONS_CHECK:
            inputs = cons_answer(inputs)
        elif task == CRITIC_RATE_COT:
            inputs = filter_cot(inputs)
        elif task in [CLASSIFY_KNOWLEDGE, ASSESS_QUESTION_KNOWLEDGE]:
            inputs = add_question_knowledge_rate(inputs)
        elif task in [GENERATE_KNOWLEDGE, GENERATE_ONE_KNOWLEDGE, HINT_GENERATE_KNOWLEDGE, QUERY_ANSWER, GENERATE_APP_KNOWLEDGE, REFLECT_GENERATE_KNOWLEDGE]:  
            inputs = add_generate_knowledge(inputs)
        # elif task == BASE_COT_ANSWER:
        #     inputs = pre_vote(inputs)
            
        
    probe = Probe(probe_targets)
    for data in inputs:
        query = data[0]
        answer = data[1]
        pred = data[2]
        question = recorder.get_question(query)
        label = recorder.get_label(question)
        options = recorder.get_options(question)
        knowledge_score = recorder.get_out_knowledge_score(question)
        message = {'question':question, 'query':query, 'pred':pred, 'answer':answer, 'option':options, 'label':label, 'score':knowledge_score}
        probe.get_info(**message)
        
    cnt = 0
    correct = 0
    fail_case_messages = []
    true_case_messages = []

    probe.cal_info()
    for message in probe:
        match = message[GET_MATCH]
        if match:
            correct += 1
            true_case_messages.append(message)
        else:
            fail_case_messages.append(message)
        cnt += 1
    acc = correct / cnt
    print(acc)
    logger.info(f'Result path: {result_path}')
    logger.info(f'Accuracy: {acc}')   


    def log(f, messages):
        for message in messages:
            if GET_QUERY in message.keys():
                if isinstance(message[GET_QUERY], list):
                    for question in message[GET_QUERY]:
                        f.write(question + '\n')
                else:
                    f.write(message[GET_QUERY] + '\n')
            if GET_ANSWER in message.keys():
                if isinstance(message[GET_ANSWER], list):
                    for answer in message[GET_ANSWER]:
                        f.write(answer + '\n')
                else:
                    f.write(message[GET_ANSWER] + '\n')
            if GET_PRED in message.keys():
                f.write('Pred: ' + str(message[GET_PRED]) + '\n')
            if GET_LABEL in message.keys():
                f.write('Label: ' + message[GET_LABEL] + '\n')
            if GET_SC_CNT in message.keys():
                f.write('Option self-consistency counts: ' + str(message[GET_SC_CNT]) + '\n')
            if GET_KNOWLEDGE_SCORE in message.keys():
                f.write('Knowledge scores: ' + str(message[GET_KNOWLEDGE_SCORE]) + '\n')
            f.write('\n')
    

    with open(result_path + '_fail.txt', 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc}' + '\n')
        log(f, fail_case_messages)
    with open(result_path + '_true.txt', 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc}' + '\n') 
        log(f, true_case_messages) 
        
    logger.info('-'*40 + 'End logging' + '-'*40)
    logger.info('\n')
    print(tokens / data_length)
    return 


if __name__ == '__main__':
    probe_targets = [GET_MATCH, 
                        GET_QUERY, 
                        GET_ANSWER, 
                        GET_PRED, 
                        GET_LABEL,
                        GET_SC_CNT,
                        GET_KNOWLEDGE_SCORE]

    # openai_model = 'gpt-3.5-turbo-0613'
    openai_model = 'gpt-3.5-turbo-0613'

    
    task_steps = [ 
                    GENERATE_KNOWLEDGE,
                    # RERANK_KNOWLEDGE,
                    # READER_ANSWER,
                    # READER_KNOWLEDGE_ANSWER
                    # BASE_COT_ANSWER
                    KNOWLEDGE_ANSWER
                    # ROBERTA_ANSWER
                    # DIERCT_ANSWER
                  ]
   
    icl_steps_dic = {
        GENERATE_KNOWLEDGE:5,
        KNOWLEDGE_ANSWER:5,
        RERANK_KNOWLEDGE:0,
        CLASSIFY_KNOWLEDGE:0,
        READER_KNOWLEDGE_ANSWER:0,
        DIERCT_ANSWER:3,
        REFLECT_GENERATE_KNOWLEDGE:3,
        BASE_COT_ANSWER:5,
        KNOWLEDGE_BASE_COT_ANSWER:3,
        REWRITE_QUESTION:3,
        GENERATE_ONE_KNOWLEDGE:5,
        GENERATE_SUP_QUESTION:3,
        ASSESS_QUESTION_KNOWLEDGE:4,
        EVALUATE_KNOWLEDGE:5,
        GENERATE_QUERY:5,
        QUERY_ANSWER:5,
        GENERATE_APP_KNOWLEDGE:5,
        RETRIEVE_KNOWLEDGE:5,
        HINT_GENERATE_KNOWLEDGE:5,
        READER_ANSWER:0,
        ROBERTA_ANSWER:0
    }
    
    sample_dic = {
        GENERATE_KNOWLEDGE:5,
        KNOWLEDGE_ANSWER:3,
        RERANK_KNOWLEDGE:1,
        CLASSIFY_KNOWLEDGE:1,
        READER_KNOWLEDGE_ANSWER:1,
        DIERCT_ANSWER:1,
        REFLECT_GENERATE_KNOWLEDGE:3,
        BASE_COT_ANSWER:3,
        KNOWLEDGE_BASE_COT_ANSWER:3,
        REWRITE_QUESTION:1,
        GENERATE_ONE_KNOWLEDGE:5,
        GENERATE_SUP_QUESTION:3,
        ASSESS_QUESTION_KNOWLEDGE:1,
        EVALUATE_KNOWLEDGE:1,
        GENERATE_QUERY:3,
        QUERY_ANSWER:3,
        GENERATE_APP_KNOWLEDGE:5,
        RETRIEVE_KNOWLEDGE:1,
        HINT_GENERATE_KNOWLEDGE:3,
        READER_ANSWER:1,
        ROBERTA_ANSWER:1
    }
    
    
    temperature_dic = {
        GENERATE_KNOWLEDGE:1.3,
        KNOWLEDGE_ANSWER:0.7,
        RERANK_KNOWLEDGE:0.7,
        CLASSIFY_KNOWLEDGE:1.0,
        READER_KNOWLEDGE_ANSWER:1.0,
        DIERCT_ANSWER:0.7,
        REFLECT_GENERATE_KNOWLEDGE:1.0,
        BASE_COT_ANSWER:1.3,
        KNOWLEDGE_BASE_COT_ANSWER:0.7,
        REWRITE_QUESTION:0.7,
        GENERATE_ONE_KNOWLEDGE:1.3,
        GENERATE_SUP_QUESTION:1.0,
        ASSESS_QUESTION_KNOWLEDGE:0.7,
        EVALUATE_KNOWLEDGE:0.7,
        GENERATE_QUERY:1.3,
        QUERY_ANSWER:1.0,
        GENERATE_APP_KNOWLEDGE:1.0,
        RETRIEVE_KNOWLEDGE:0.7,
        HINT_GENERATE_KNOWLEDGE:1.3,
        READER_ANSWER:1.0,
        ROBERTA_ANSWER:1.0
    }
    
    
    icl_steps = [icl_steps_dic[task_steps[i]]  for i in range(len(task_steps))]
    
    sample_steps = [sample_dic[task_steps[i]] for i in range(len(task_steps))]
    
    temperature_steps = [temperature_dic[task_steps[i]] for i in range(len(task_steps))]
    
    
    top_k = 1
    
    debug_mode = True
    # collect_method = 'rerank'
    collect_method = 'none'
    
    if debug_mode: 
        data_length = 10
    else:
        data_length = 500
        
   
    
    dataset = 'piqa'
    # dataset = 'piqa'
    # dataset = 'hella'
    # dataset = 'csqa2'
    if dataset == 'wino':
        case_num = 'case_token'
        shuffle = False
        rerank_checkpoint = 'reward_model/wino'
        result_path = wino_result_dir + case_num
        verify_model_path = './result/verifier/wino/lr_1e-05_e2_s17_b28_vs1500_ws500_gs8_t20231020_Fri_19:31:51_acc0.8253333333333334'
        knowledge_path = './result/wino/case11_step1_GENERATE_KNOWLEDGE_temp.json'
        reader_checkpoint = 'Llama_reader/lr_0.001_e8_s17_b64_vs500_ws300_t20230913_Wed_00:28:31'
        roberta_checkpoint = './result/roberta/lr_5e-05_e2_s17_b64_vs1000_ws300_gs16_t20231018_Wed_00:01:56_acc0.931'
            # knowledge_path = './result/winogrande/case10_step1_HINT_GENERATE_KNOWLEDGE_temp.json'
    elif dataset == 'piqa':
        case_num = 'case_token'
        shuffle = False
        knowledge_path = './result/piqa/case6_step1_GENERATE_KNOWLEDGE.json'
        rerank_checkpoint = 'reward_model/piqa'
        roberta_checkpoint = './result/roberta/lr_0.0001_e10_s17_b16_vs500_ws500_gs8_t20240613_Thu_22:10:37_acc0.804'
        result_path = piqa_result_dir + case_num
    elif dataset == 'siqa':
        shuffle = True
        case_num = 'case_token'
        rerank_checkpoint = 'reward_model/siqa'
        knowledge_path = './result/siqa/case6_step1_GENERATE_KNOWLEDGE.json'
        roberta_checkpoint = './result/roberta/lr_3e-05_e2_s17_b8_vs500_ws500_gs8_t20240604_Tue_18:50:16_acc0.826'
        reader_checkpoint = 'llama/siqa_lr_0.0001_e2_s17_b16_vs500_ws300_t20240608_Sat_11:50:26_acc0.566'
        result_path = siqa_result_dir + case_num
    else:
    # case_num = 'case18'
        case_num = 'case_token'
        shuffle = False
        verify_model_path = './result/verifier/hella/lr_1e-05_e2_s17_b32_vs1500_ws500_gs8_t20231020_Fri_10:26:33_acc0.9253333333333333'
        rerank_checkpoint = 'reward_model/hella'
        result_path = hella_result_dir + case_num
        reader_checkpoint = 'llama/hella_lr_0.0001_e2_s17_b16_vs500_ws300_t20240608_Sat_15:50:35_acc0.444'
        knowledge_path = './result/hella/case14_step1_GENERATE_KNOWLEDGE.json'
        roberta_checkpoint = './result/roberta/lr_5e-05_e2_s17_b16_vs500_ws500_gs8_t20231017_Tue_17:34:48_acc0.722'
    
    run(load_knowledge=False, dataset=dataset)
    
    
    
