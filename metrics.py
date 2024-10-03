from config import *
from process import InputParser
import json

def compare_results(file1_path, file2_path):
    file1_question_set = set()
    file2_question_set = set()
    with open(file1_path, 'r') as f:
        for line in f:
            if line[:3] == "(1)":
                last_line = last_line.strip('Question:').strip()
                file1_question_set.add(last_line)
            last_line = line
    with open(file2_path, 'r') as f:
        for line in f:
            if "(1)" in line and "(2)" in line:
                last_line = last_line.strip('Question:').strip()
                file2_question_set.add(last_line)     
            last_line = line 
    with open(file1_path, 'r') as f:
        for line in f:
            if line[:8] == 'Question':
                line = line.strip("Question: ")
                file1_question_set.add(line)
    with open(file2_path, 'r') as f:
        for line in f:
            if line[:8] == 'Question':
                line = line.strip("Question: ")
                file2_question_set.add(line)
    set1 = file1_question_set - file2_question_set
    print('More good:')
    print(len(set1))
    for tup in set1:
        print(tup)
    set2 = file2_question_set - file1_question_set
    print('More bad:')
    print(len(set2))
    for tup in set2:
        print(tup)     


# def count_labels(file):
#     label1_cnt = 0
#     label2_cnt = 0
#     with open(file, 'r') as f:
#         for line in f:
#             if 'Label: 2' in line:
#                 label2_cnt += 1
#             elif 'Label: 1' in line:
#                 label1_cnt += 1
#     print(f'Label 1 wrong: {label1_cnt}')
#     print(f'Label 2 wrong: {label2_cnt}')

def cal_ep_score(base_case, cal_case):
    e_score = 0
    p_score = 0
    base_true_question_set = set()
    base_false_question_set = set()
    question_set = set()
    with open(base_case+'_true.txt', 'r') as f:
        last_line = ""
        for line in f:
            if '(1)' in line and '(2)' in line:
                question = last_line.strip('\n').split('Question:')[-1].rstrip('_').strip()
                base_true_question_set.add(question) 
                question_set.add(question)
            last_line = line
    with open(base_case+'_fail.txt', 'r') as f:
        last_line = ""
        for line in f:
            if '(1)' in line and '(2)' in line:
                question = last_line.strip('\n').split('Question:')[-1].rstrip('_').strip()
                base_false_question_set.add(question) 
                question_set.add(question)
            last_line = line
    
    
    cal_true_question_set = set()
    cal_false_question_set = set()    

    # with open(cal_case+'.json', 'r') as f:
    #     data = json.load(f)
    # for item in data:
    #     if 'question' not in item.keys():
    #         continue
    #     question = item['question']
    #     if 'dpr' in cal_case:
    #         for q in question_set:
    #             if q in question:
    #                 question = q
    #                 break
    #     elif 'Knowledge' in question:
    #         question = question.split('\n')[1].split('Question:')[-1].strip()
    #     else:
    #         question = question.split('\n')[0].split('Question:')[-1].strip() 
    #     # print(question) 
    #     if item['match']:
    #         cal_true_question_set.add(question)
    #     else:
    #         cal_false_question_set.add(question) 
    with open(cal_case+'_true.txt', 'r') as f:
        last_line = ""
        for line in f:
            if '(1)' in line and '(2)' in line:
                question = last_line.strip('\n').split('Question:')[-1].rstrip('_').strip()
                cal_true_question_set.add(question)
            last_line = line
    with open(cal_case+'_fail.txt', 'r') as f:
        last_line = ""
        for line in f:
            if '(1)' in line and '(2)' in line:
                question = last_line.strip('\n').split('Question:')[-1].rstrip('_').strip()
                cal_false_question_set.add(question)
            last_line = line
    e_score = len(base_false_question_set & cal_true_question_set) / len(base_false_question_set)
    p_score = 1 - len(base_true_question_set & cal_false_question_set) / len(base_true_question_set)
    ep_score = 2 * (e_score * p_score) / (e_score + p_score)
    print (f'e-score:{e_score}, p-score:{p_score}, ep-score:{ep_score}')
    
if __name__ == '__main__':
    base_file = './result/hella/case40'
    # cal_file ='./result/hella/unifiqa' 
    cal_file = './result/hella/bm25rag'
    # base_file ='./result/wino/case14'
    # base_file = './result/siqa/case3'
    # cal_file = '../toxic_cot/result/piqa/ChatGPT_dpr_500'
    # base_file = './result/piqa/case1'
    # cal_file = './result/piqa/case10'
    cal_ep_score(base_file, cal_file)
    # file1 = csqa2_short_result_dir + 'case3_fail.txt'
    # file2 = csqa2_short_result_dir + 'case4_fail.txt'
    # file1 = wino_result_dir + 'case11_fail.txt'
    # file2 = wino_result_dir + 'case34_fail.txt'
    # file1 = hella_result_dir + 'case3_fail.txt'
    # file2 = hella_result_dir + 'case5_fail.txt'
    # compare_results(file1, file2)
    # # count_labels(file1)