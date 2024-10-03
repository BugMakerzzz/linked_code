from config import *
from utils import get_data_path, delete_answer, get_index_ignore_case
import json
import re
import random
random.seed(17)

class InputParser():
    def __init__(self, dataset, data_length, split='dev', shuffle=True) -> None:
        self.__question_stem_ls = []
        self.__label_ls = []
        self.__option_ls = []
        self.__idx = 0
        self.__len = data_length
        self.__load_data(dataset, split=split)
        if shuffle:
            self.__shuffle_data()
    
    def __load_labels(self, label_path, add_one):
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                if add_one:
                    labels.append(str(eval(line[:-1]) + 1))
                else:
                    labels.append(line[:-1])
        return labels
    
    
    def __load_data(self, dataset, split):
        data_path = get_data_path(dataset, split)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label = ""
                if dataset == 'csqa':
                    question = data['question']['stem']
                    label = data['answerKey']
                    options = []
                    for i in range(len(data['question']['choices'])):
                        tup = data['question']['choices'][i]
                        options.append(tup['text'])
                elif dataset == 'csqa2':
                    question = data['question']
                    label = data['answer']
                    if label == 'yes':
                        label = '1'
                    else:
                        label = '2' 
                    options = ['yes', 'no']
                elif dataset == 'wino':
                    question = data['sentence']
                    label = data['answer']
                    options = [data['option1'], data['option2']]
                elif dataset == 'hella':
                    question = data['ctx']
                    label = str(data['label'] + 1)
                    options = []
                    for i in range(len(data['endings'])):
                        options.append(data['endings'][i])
                elif dataset =='siqa':
                    context = data['context']
                    question = data['question']
                    question = context + " " + question
                    options = [ data['answerA'], data['answerB'],data['answerC']]
                elif dataset == 'piqa':
                    question = data['goal'] + '_'
                    options = [data['sol1'], data['sol2']]
                self.__question_stem_ls.append(question)
                self.__option_ls.append(options)
                if label:
                    self.__label_ls.append(label)
            if dataset == 'siqa':
                if split == 'train':
                    self.__label_ls = self.__load_labels('./data/SocialIQA/train-labels.lst', add_one=False)
                else:
                    self.__label_ls = self.__load_labels('./data/SocialIQA/dev-labels.lst', add_one=False)
            elif dataset == 'piqa':
                if split == 'train':
                    self.__label_ls = self.__load_labels('./data/PIQA/train-labels.lst', add_one=True)
                else:
                    self.__label_ls = self.__load_labels('./data/PIQA/valid-labels.lst', add_one=True)
        if len(self.__label_ls) < self.__len:
            self.__len = len(self.__label_ls)


    def __shuffle_data(self):
        tuples = list(zip(self.__question_stem_ls, self.__option_ls, self.__label_ls))
        random.shuffle(tuples)
        self.__question_stem_ls, self.__option_ls, self.__label_ls = zip(*tuples)
        return 
    
    
    def __get_next_question_stem(self):
        return self.__question_stem_ls[self.__idx]
      
        
    def __get_next_options(self):
        return self.__option_ls[self.__idx]
    
    
    def __get_next_option_string(self):
        options = self.__get_next_options()
        option_string = ""
        for i in range(len(options)):
            option = options[i]
            option_string += f"({i+1}) " + option + " "
        return option_string
    
    
    def __get_next_question(self):
        question_stem = self.__get_next_question_stem()
        question = question_stem + '\n' + self.__get_next_option_string()
        return question 
        
    
    def __get_next_label(self):
        return self.__label_ls[self.__idx]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__idx < self.__len:
            stem = self.__get_next_question_stem()
            question = self.__get_next_question()
            option = self.__get_next_options()
            label = self.__get_next_label()
            self.__idx += 1
            return stem, question, option, label
        else:
            raise StopIteration()
    
    def __len__(self):
        return self.__len
    

class OutputParser():
    def __init__(self) -> None:
        self.__querys = []
        self.__answers = []
        self.__preds = []
        self.__idx = 0
        self.__len = 0
        self.__no_parse_flag = False
        return
    
    
    def __reset(self):
        self.__querys = []
        self.__answers = []
        self.__preds = []
        self.__idx = 0
        self.__len = 0
    
    def reset_idx(self):
        self.__idx = 0
        
    
    def parse(self, results, task):
        self.__reset()
        for result in results:
            if isinstance(result[0], str):
                self.__no_parse_flag = True
                query = result[0]
                answers = result[1]
            else:
                self.__no_parse_flag = False
                query = result[0]['messages'][-1]['content']
                answers = result[1]['choices']   
            if task == EVALUATE_KNOWLEDGE:
                self.__querys.append(query)
                self.__parse_eval_rate(answers)
            elif task == CONS_CHECK:
                self.__querys.append(query)
                self.__parse_cons_rate(answers)
            elif task == ASSESS_QUESTION_KNOWLEDGE:
                self.__querys.append(query)
                self.__parse_rerank_rate(answers)
            elif task in [READER_KNOWLEDGE_ANSWER, READER_ANSWER, ROBERTA_ANSWER]:
                self.__querys.append(query)
                self.__answers.append(answers)
                self.__preds.append(self.__extract_pred(answers))
            elif task in [CLASSIFY_KNOWLEDGE, CRITIC_RATE_COT, RERANK_KNOWLEDGE]:
                self.__querys.append(query)
                self.__answers.append(answers)
                self.__preds.append(answers)
            else:
                for item in answers:
                    answer = item['message']['content']
                    self.__querys.append(query)
                    self.__answers.append(answer)
                    self.__preds.append(self.__extract_pred(answer))
        self.__len = len(self.__querys)
        return 
    
    
    
    def __extract_pred(self, answer):
        match = re.findall(r'[1-5]\)',answer)
        if match:
            pred = match[-1][:-1]
            return pred
        else:
            return 'None'
    

    
    def __parse_eval_rate(self, answers):
        rate = 0
        for item in answers:
            answer = item['message']['content'].lower()
            if 'good' in answer:
                rate += 1
            elif 'bad' in answer:
                rate -= 1
        self.__answers.append(str(rate))
        self.__preds.append(rate)
        return
    
    def __parse_cons_rate(self, answers):
        if self.__no_parse_flag:
            self.__answers.append(answers)
            self.__preds.append(eval(answers))
            return 
        rate = 0
        for item in answers:
            answer = item['message']['content'].lower()
            if 'yes' in answer:
                rate += 1
            elif 'no' in answer:
                rate -= 1
        self.__answers.append(str(rate))
        self.__preds.append(rate)
        return
    
    def __parse_rerank_rate(self, answers):
        rate = 0
        for item in answers:
            answer = item['message']['content']
            match = re.findall(r'-?\d+\.\d*',answer)
            if match:
                rate += eval(match[0])
        rate /= len(answers)
        self.__answers.append(str(rate))
        self.__preds.append(rate)
        return
    
    
    def __get_next_query(self):
        return self.__querys[self.__idx]
    
    
    def __get_next_answer(self):
        # if self.__sc and self.__type not in NEED_ANSWER_TYPES:
        #     return self.__get_filter_answer()
        # else:
        return self.__answers[self.__idx]
    
    
    def __get_next_pred(self):
        return self.__preds[self.__idx]
       
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__idx < self.__len:
            query = self.__get_next_query()
            answer = self.__get_next_answer()
            pred = self.__get_next_pred()
            self.__idx += 1
            return query, answer, pred
        else:
            raise StopIteration()
    
    def __len__(self):
        return self.__len
    
    def __getitem__(self, idx):
        query = self.__querys[idx]
        answer = self.__answers[idx]
        pred = self.__preds[idx]
        return query, answer, pred
    
    
    def dump(self, file_path):
        js_data = []
        for idx in range(self.__len):
            query = self.__querys[idx]
            answer = self.__answers[idx]
            pred = self.__preds[idx]
            message = {'query':query, 'answer':answer, 'pred':pred}
            js_data.append(message)
        with open(file_path, 'w', encoding='utf-8') as f:   
            json.dump(js_data, f, indent=4)
        return 
    
    
    def load(self, file_path):
        self.__reset()
        with open(file_path, 'r') as f:
            js_data = json.load(f)
        for data in js_data:
            self.__querys.append(data['query'])
            self.__answers.append(data['answer'])
            self.__preds.append(data['pred'])
        self.__len = len(self.__querys)
        return 
    
    
class Probe():
    def __init__(self, probe_targets) -> None:
        self.__targets = probe_targets
        self.__result_dic = {}
        for key in self.__targets:
            self.__result_dic[key] = []
        self.__preds = {}
        self.__labels = {}
        self.__options = {}
        self.__answers = {}
        self.__querys = {}
        self.__scores = {}
        self.__idx = 0
        self.__len = 0
        return 
    
     
    def __get_match(self, preds, label):
        if len(preds) > 1:
            cnt = len(preds)
            count_dic = self.__get_pred_counts(preds)
            if label not in count_dic.keys():
                return False
            elif count_dic[label] > cnt // 2:
                return True
            else:
                return False
        else:
            pred = preds[0]
            if pred == label:
               return True
            else:
                return False
    
    
    def __get_pred_counts(self, preds):
        count_dic = {}
        for pred in preds:
            if pred in count_dic.keys():
                count_dic[pred] = count_dic[pred] + 1
            else:
                count_dic[pred] = 1
        return count_dic
    
    
    def __get_counts_match(self, answer, options, pred):
        if isinstance(answer, list):
            cnt = len(answer)
            match_cnt = 0
            for i in range(cnt):
                match_cnt += self.__get_one_count_match(answer[i], options, pred[i])
            return match_cnt / cnt
        else:
            return self.__get_one_count_match(answer, options, pred)
    
    
    def __get_one_count_match(self, answer, options, pred):
        max_option = ''
        max_cnt = 0
        answer = delete_answer(answer)
        for i in range(len(options)):
            option = options[i]
            match = re.findall(pattern=option, string=answer, flags=re.I)
            if len(match) > max_cnt:
                max_cnt = len(match)
                max_option = str(i+1)
        if isinstance(pred, list):
                pred = max(pred, key=pred.count)
        if max_option == pred:
            return 1
        else:
            return 0
       
        
    def __get_last_match(self, answer, options, pred):  
        pattern = "("
        for option in options:
            pattern += option + "|"
        pattern =  pattern[:-1] + ")"
        pattern = re.compile(pattern, flags=re.I)
        if isinstance(answer, list):
            match_cnt = 0
            for i in range(len(answer)):
                match = re.findall(pattern=pattern, string=delete_answer(answer[i]))
                if not match:
                    continue
                else:
                    match_option = match[-1]
                    option_num = str(get_index_ignore_case(match_option, options) + 1)  
                    if pred[i] == option_num:
                        match_cnt += 1
            return match_cnt / len(answer)
        else:
            answer = delete_answer(answer)
            match = re.findall(pattern=pattern, string=answer)
            if not match:
                return 0
            else:
                match_option = match[-1]
                option_num = str(get_index_ignore_case(match_option, options) + 1)    
                if pred == option_num:
                    return 1
                else:
                    return 0
                

    def cal_info(self):
        for question in self.__preds.keys():
            preds = self.__preds[question]
            label = self.__labels[question]
            query = self.__querys[question]
            answer = self.__answers[question]
            options = self.__options[question]
            scores = self.__scores[question]
            if GET_MATCH in self.__result_dic.keys():
                self.__result_dic[GET_MATCH].append(self.__get_match(preds, label))
            if GET_QUERY in self.__result_dic.keys():
                self.__result_dic[GET_QUERY].append(query)
            if GET_ANSWER in self.__result_dic.keys():
                self.__result_dic[GET_ANSWER].append(answer)
            if GET_PRED in self.__result_dic.keys():
                self.__result_dic[GET_PRED].append(preds)
            if GET_LABEL in self.__result_dic.keys():
                self.__result_dic[GET_LABEL].append(label)
            if GET_SC_CNT in self.__result_dic.keys():
                self.__result_dic[GET_SC_CNT].append(self.__get_pred_counts(preds))
            if GET_LAST_MATCH in self.__result_dic.keys():
                self.__result_dic[GET_LAST_MATCH].append(self.__get_last_match(answer, options, preds))
            if GET_COUNT_MATCH in self.__result_dic.keys():
                self.__result_dic[GET_COUNT_MATCH].append(self.__get_counts_match(answer, options, preds))
            if GET_KNOWLEDGE_SCORE in self.__result_dic.keys():
                self.__result_dic[GET_KNOWLEDGE_SCORE].append(scores)
            self.__len += 1
    
    def get_info(self, **kwargs):
        question = kwargs['question']
        query = kwargs['query']
        if question in self.__querys.keys():
            self.__querys[question].append(query)
        else:
            self.__querys[question] = [query]
        pred = kwargs['pred']
        if question in self.__preds.keys():
            self.__preds[question].append(pred)
        else:
            self.__preds[question] = [pred]
        answer = kwargs['answer']
        if question in self.__answers.keys():
            self.__answers[question].append(answer)
        else:
            self.__answers[question] = [answer]
        options = kwargs['option']
        self.__options[question] = options
        label = kwargs['label']
        self.__labels[question] = label
        score = kwargs['score']
        self.__scores[question] = score
        return
    
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__idx < self.__len:
            message = {}
            for key, values in self.__result_dic.items():
                message[key] = values[self.__idx]
            self.__idx += 1
            return message
        else:
            raise StopIteration()
        
        

class Recorder():
    def __init__(self) -> None:
        self.__question_knowledge_pool = {}
        self.__question_out_knowledge_score = {}
        self.__question_label_map = {}
        self.__question_options_map = {}
        self.__question_stem_map = {}
        self.__query_question_map = {}
        self.__question_query_map = {}
        self.__question_map_switch = False 
        self.__question_map = {}
        self.__statement_pred_map = {}
        return
    

    def log_message(self, *args):
        stem, question, options, label = args
        self.__question_label_map[question] = label
        self.__question_stem_map[question] = stem
        self.__question_options_map[question] = options
        # self.__question_graph_map[question] = self.__idx
        # self.__idx += 1
    

    def get_label(self, question):
        if self.__question_map_switch:
            question = self.__question_map[question]
        if question in self.__question_label_map.keys():
            return self.__question_label_map[question]
        else:
            print('Question Not In Keys!')
            return None
    

    def log_query_question_map(self, query, question):
        self.__query_question_map[query] = question
        self.__question_query_map[question] = query
    
   
    def update_query_question_map(self, old_query, new_query):
        if old_query not in self.__query_question_map.keys():
            print('Query Not In Keys!')
        else:
            question = self.__query_question_map[old_query]
            self.__query_question_map[new_query] = question
            self.__question_query_map[question] = new_query
            
    
    def update_question(self, question, new_question):
        new_question = new_question + '\n' + self.get_option_string(question)
        self.__question_map[question] = new_question
        self.__question_map[new_question] = question
        self.__question_map_switch = True
    
    
    def get_question(self, query):
        if self.__question_map_switch:
            return self.__question_map[self.__query_question_map[query]]
        else:
            return self.__query_question_map[query]
    
    def get_query(self, question):
        return self.__question_query_map[question]
    
    
    def get_question_stem(self, question):
        if self.__question_map_switch:
            question = self.__question_map[question]
        if question in self.__question_stem_map.keys():
            return self.__question_stem_map[question]
        else:
            print('Question Not In Keys!')
            return None
    
    
    def get_options(self, question):
        if self.__question_map_switch:
            question = self.__question_map[question]
        if question in self.__question_options_map.keys():
            return self.__question_options_map[question]
        else:
            print('Question Not In Keys!')
            return None
    

    def get_knowledge_pool(self, question):
        if question in self.__question_knowledge_pool.keys():
            return self.__question_knowledge_pool[question]
        else:
            return None
    
    def traverse_knowledge_pools(self):
        return self.__question_knowledge_pool
    
    
    def get_out_knowledge_score(self, question):
        if question in self.__question_out_knowledge_score.keys():
            return self.__question_out_knowledge_score[question]
        else:
            return None
    
    def get_knowledge_score(self, question, knowledge):
        knowledge_pool = self.get_knowledge_pool(question)
        knowledge = knowledge.strip()
        if knowledge in knowledge_pool.keys():
            return knowledge_pool[knowledge]
        return None
               
        
    def get_option_string(self, question):
        options = self.get_options(question)
        option_string = ""
        for i in range(len(options)):
            option = options[i]
            option_string += f"({i+1}) " + option + " "
        return option_string
    
    
    def add_knowledge(self, question, knowledge, rate=0):
        if question in self.__question_knowledge_pool.keys():
            self.__question_knowledge_pool[question][knowledge.strip()] = rate
        else:
            self.__question_knowledge_pool[question] = {knowledge.strip():rate}
    
    
    def del_knowledge(self, question, knowledge):
        knowledge_pool = self.__question_knowledge_pool[question]
        if knowledge.strip() in knowledge_pool.keys(): 
            del self.__question_knowledge_pool[question][knowledge.strip()]
        return 
    
        
    
    def add_out_knowledge_score(self, question, rate):
        if question in self.__question_out_knowledge_score.keys():
            self.__question_out_knowledge_score[question].append(rate)
        else:
            self.__question_out_knowledge_score[question] = [rate]
        
    
    def fill_question(self, question, pred):
        stem = self.__question_stem_map[question]
        options = self.__question_options_map[question]
        if pred.isdigit():
            idx = eval(pred) - 1
            option = options[idx]
            statement = stem.replace('_', option)
        else:
            statement = "None"
        self.__statement_pred_map[statement] = pred
        return statement
    
    
    
    def transform_pred(self, statement):
        return self.__statement_pred_map[statement]
    # def get_graph_idx(self, question):
    #     if question in self.__question_graph_map.keys():
    #         return self.__question_graph_map[question]
    #     else:
    #         print('Question Not In Keys!')
    #         return None


# class PykeInterpreter:
#     def __init__(self, save_path) -> None:
#         self.save_path = save_path

#     def interpret_facts(self, facts):
#         with open(os.path.join(self.sample_dir, 'facts.kfb'), 'w') as f:
#             for fact in facts:
#                 # check for invalid facts
#                 if not fact.find('$x') >= 0:
#                     f.write(fact + '\n')

#     # example rule: Vumpuses($x, True) >>> Tumpuses($x, True)
#     # def parse_forward_rule(self, f_index, rule):
#     #     premise, conclusion = rule.split('>>>')
#     #     premise = premise.strip()
#     #     conclusion = conclusion.strip()
#     #     pyke_rule = f'''fact{f_index}\n\tforeach\n\t\tfacts.{premise}\n\tassert\n\t\tfacts.{conclusion}'''
#     #     return pyke_rule
    
#     # example rule: Furry($x, True) && Quite($x, True) >>> White($x, True)
#     def parse_forward_rule(self, f_index, rule):
#         premise, conclusion = rule.split('>>>')
#         premise = premise.strip()
#         # split the premise into multiple facts if needed
#         premise = premise.split('&&')
#         premise_list = [p.strip() for p in premise]

#         conclusion = conclusion.strip()
#         # split the conclusion into multiple facts if needed
#         conclusion = conclusion.split('&&')
#         conclusion_list = [c.strip() for c in conclusion]

#         # create the Pyke rule
#         pyke_rule = f'''fact{f_index}\n\tforeach'''
#         for p in premise_list:
#             pyke_rule += f'''\n\t\tfacts.{p}'''
#         pyke_rule += f'''\n\tassert'''
#         for c in conclusion_list:
#             pyke_rule += f'''\n\t\tfacts.{c}'''
#         return pyke_rule

#     def interpret_rules(self, rules):
#         pyke_rules = []
#         for idx, rule in enumerate(rules):
#             pyke_rules.append(self.parse_forward_rule(idx + 1, rule))

#         with open(os.path.join(self.sample_dir, 'rules.krb'), 'w') as f:
#             f.write('\n\n'.join(pyke_rules))

#     def interpret_program(self, logic_program : LogicProgram):
#         # create the folder to save the Pyke program
#         sample_dir = os.path.join(self.save_path, logic_program.sample_id)
#         if not os.path.exists(sample_dir):
#             os.makedirs(sample_dir)
#         self.sample_dir = sample_dir
        
#         try:
#             self.interpret_facts(logic_program.Facts)
#             self.interpret_rules(logic_program.Rules)
#         except:
#             return False
#         return True
