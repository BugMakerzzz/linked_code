from config import *
from utils import get_related_example, get_related_knowledge
from sentence_transformers import SentenceTransformer
# import faiss
import json

blank_prompt = "Given the question, '{}' please fill the blank with the correct answer from the following options:{} You should return the serial number of the correct option and give the reason."
cot_prompt = "Given the question, '{}', let's think it step by step:"
cot_option_prompt = "Please fill the blank in the question with the correct answer from the following options:{} You should return the serial number of the correct option."
context_question_prompt = "According to the context: '{}', given the question: '{}', please select the correct answer from the following options:{} You should return the serial number of the correct option and give the reason."
question_prompt = "Given the question: '{}', please select the correct answer from the following options:{} You should return the serial number of the correct option and give the reason."
knowledge_cot_prompt_wo_k = "Question: Given the statement:{}, please fill the blank with the correct answer from the following options:{}"
knowledge_cot_prompt = "Question: Given the statement:{}, please generate the related knowledge beginning with 'Knowledge:' and fill the blank with the correct answer from the following options:{}, which begins with 'Answer:'."
qa_knowledge_cot_prompt = "Question: Given the question:{}, please generate the related knowledge beginning with 'Knowledge:', and select the correct answer from the following options:{}, which begins with 'Answer:'."
### 以上prompt想暂时废弃，可以试着将form限制和问题本身分离
cot_answer_assistant = "You are a helpful assistant that break down the question step by step and choose the correct answer to the question."
cot_assistant = "You are a helpful assistant that break down the question step by step to solve the question"
generate_knowledge_assistant = "You are a helpful assistant that generate knowledge according to the question."
direct_choice_assistant = "You are a helpful assistant that use your own knowledge to choose the correct answer to the question."
generate_sup_question_assistant = "You are a helpful assistant that generate a complementary question according to the target question as a mid step to solve it."
knowledge_choice_assistant = "You are a helpful assistant that choose the correct answer to the question based on the given knowledge"
generate_knowledge_answers_assistant = "You are a helpful assistant that generate knowledge and choose the correct answer according to the question."
exclude_cot_assistant = "You are a helpful assistant that break down the question, exclude the wrong options and choose the correct answer to the question."
knowledge_base_cot_assistant = "You are a helpful assistant that break down the question step by step and choose the correct answer to the question according to the provided question and knowledge."
consistency_assistant = "You are a helpful assistant that judge whether the first statement imply the second statement."
knowledge_infer_assistant = "You are a helpful assistant that generate generate an inference about the question stem based on the given knowledge."
infer_choice_assistant = "You are a helpful assistant that choose the correct answer to the question based on the given inference."
generate_sub_question_assistant = "You are a helpful assistant that split the problem into related sub-problems."
sub_question_answer_assistant = "You are a helpful assistant that answer the questions sequentially."
goal_extraction_assistant = "You are a helpful assistant that extract the goal of the answer."
res_cot_answer_assistant = "You are a helpful assistant that use the provided thought to infer the result of the question."
evaluate_knowledge_assistant = "You are a helpful assistant that use your own commonsense knowledge to rate the truthfulness of the given knowledge."
retrieve_knowledge_assistant = "You are a helpful assistant that use your own commonsense knowledge to rate the correlation of the given knowledge to the question."
transform_knowledge_assistant = "You are a helpful assistant that transform the general commonsense knowledge to the specific inference according to the context of the question."
infer_answer_assistant = "You are a helpful assistant that uses the provided inference as your thoughts to reason the correct answer of the question."
rewrite_question_assistant = "You are a helpful assistant that rewrite the provided questions in your own way."
generate_graph_assistant = "You are a helpful assistant that generate a knowledge graph according to the given context."
generate_pyke_assistant = "You are a helpful assistant that generate logical program according to the given knowledge and question."
hint_generate_knowledge_assistant = "You are a helpful assistant that generate the correct knowledge to the statements."
hint_generate_cot_assistant = "You are a helpful assistant that generate the correct reasoning from questions to answers according to the question and answer."
reflect_generate_knowledge_assistant = "You are a helpful assistant that generate more helpful knowledge for solving the question correctly compared to the provided useless knowledge."
assess_question_knowledge_assistant = "You are a helpful assitant that assess the impact of knowledge on answering questions correctly."
llama_sys_prompt = """<s>[INST] <<SYS>>
{}
<</SYS>>

"""


assess_question_knowledge_instru_prompt = """You are given a question, paired with related knowledge. Please classify whether the knowledge is useful(label 0) or harmless(label 1) or useless(label 2) or harmful(label 3) for answering the question correctly. Your response should be in this form:
'Label:{label}'
You can only choose the label from 0,1,2,3.
Now here are some examples:
"""

reflect_generate_knowledge_instru_prompt = """You are given a question and a related useless knowledge, please generate new useful knowledge by reflecting on the deficiencies in providing knowledge. Your response should be in this form:
'Knowledge:{new_knowledge}'
Remember you cannot directly answer the question as your knowledge and the knowledge should be helpful for yourself to answer the question.
Now here are some examples:
"""

rewrite_question_instru_prompt = """Please rewrite the given question in your own way in the following form:
'{question}
{Options}'
Now here are some examples:
"""



generate_question_instru_prompt = """Use your commonsense knowledge to generate a knowledge graph according to the given context. Your response should be in this form:
Knowledge 1:({head}, {relation}, {tail})
Knowledge 2:({head}, {relation}, {tail})
......
Remember you should list all of the knowledge triples in the graph and use as few entities as possible.
Here are some examples:
"""

hint_generate_knowledge_instru_prompt = """Use your commonsense knowledge to generate knowledge according to some statements. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot simply restate or rewrite the question as your knowledge and the knowledge should be as general as possible.
Now the questions and answers are as follows:
"""

hint_generate_cot_instru_prompt = """Use your commonsense knowledge to generate the reasoning process for some questions. Your response should be in this form:
'Step 1: {content}
Step 2: {content}
......
[END]'
You should use no more than 3 steps and output '[END]' as the end of the reasoning.
You are given some questions and the corresponding answers as hints. Remember you should make sure the reasoning process can generate the answer provided from the question.
Now the questions and answers are as follows:
"""


cot_answer_instru_prompt = """Use your commonsense knowledge to choice correct answer for some questions and give the reasoning process. Your response should be in this form:
'{Reasoning_content}
So the answer is: ({option}) {answer}'
Now answer the following questions:
"""

cot_instru_prompt = """Use your commonsense knowledge to reply the reasoning process of the given question. Your response should be in this form:
'Step 1: {content}
Step 2: {content}
......
[END]'
Now answer the following questions:
"""

goal_extraction_instru_prompt = """Please extract the goal of the question to rewrite it. Your response should be in this form:
'Question: {content}'
Now here is the question:
"""

res_cot_answer_instru_prompt = """Use the provided thought to infer the correct answer for some questions. Your response should be in this form:
'Answer: (1) {answer}' (If you think the first option in the question stem is correct)
or 'Answer: (2) {answer}' (If you think the second option in the question stem is correct)
If there is not proper option, you can give 'Answer: None'.
Now answer the following questions according to these thoughts:
"""

evaluate_knowledge_instru_prompt = """Use your commonsense knowledge to rate the given knowledge. Your response should be in this form:
'Rate: good/bad/not sure'
You can only reply 'good' (If the knowledge is a piece of commonsense that is true in most cases)
or 'bad' (If the knowledge is a piece of commonsense that is wrong in most cases)
or 'not sure' (If you are not sure whether the knowledge is true or false).
Now here is the knowledge:
"""

retrieve_knowledge_instru_prompt = """Please measure the correlation of the given knowledge to the question. Your response should be in this form:
'Rate: {value}'
You should give a value between -1 and 1 to evaluate the relevance of the given knowledge to the question. 
Value '-1' means this piece of knowledge misleads the correct answer to the question or completely contradicts the intent of the question.
Value '0' means this piece of knowledge is no related with the question or does not help in answering the question directly.
Value '1' means this piece of knowledge is highly related with the question and can help you to answer the question better.
To sum up, the higher the rating value is, the more relevant the knowledge to the problem.
Now here are the examples:
"""

transform_knowledge_instru_prompt = """Please transform each piece of the given knowledge to the specific inference in the context of the given question.
Now here are the examples:
"""

sup_question_instru_prompt = """Use your commonsense knowledge to generate a key supplementary question, in order to answer the given question correctly. Your response should be in this form:
'{question}
(1) {option1} (2) {option2}'
Remember that the question must be in the form of two-choice questions.
Now here is the knowledge and question:
"""

knowledge_and_answer_instru_prompt = """Use your commonsense knowledge to choice correct answer for some questions. Your response should be in this form:
'Knowledge: {knowledge}
Answer: ({option}) {answer}'
If there is not proper option, you can give 'Answer: None'. Remember you cannot simply restate or rewrite the question as your knowledge.
Now answer the following questions:
"""

answer_instru_prompt = """Use your commonsense knowledge to choose correct answer for some questions. Your response should be in this form:
'Answer: ({option}) {answer}'
If there is not proper option, you can give 'Answer: None'.
Now answer the following questions:
"""

generate_knowledge_instru_prompt = """Use your commonsense knowledge to generate knowledge for some questions. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot directly answer the question as your knowledge.
Now the question is as follows:
"""

one_knowledge_instru_prompt = """Use your commonsense knowledge to generate only one piece of knowledge for some questions. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot simply restate or rewrite the question as your knowledge.
Now the question is as follows:
"""

extra_knowledge_instru_prompt = """Given a question, use your commonsense knowledge to generate another piece of knowledge that is different from the existing knowledge. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot simply restate or rewrite the question as your knowledge, the generated knowledge should differ from the given knowledge and you can only generate one piece of knowledge.
Now the given knowledge and question is as follows:
"""

knowledge_answer_instru_prompt = """Use the provided knowledge and your own commonsense knowledge to choice correct answer for some questions. Your response should be in this form:
'Answer: ({option}) {answer}'
If there is not proper option, you can give 'Answer: None'.
Now answer the following questions:
"""

# knowledge_answer_instru_prompt = """Use the provided knowledge and your own commonsense knowledge to choose correct answer to fill in the blank in some questions in order to complete a true statement. Your response should be in this form:
# 'Answer: ({option}) {answer}'
# If there is not proper option, you can give 'Answer: None'.
# Now answer the following questions:
# """



knowledge_infer_instru_prompt = """Use the provided knowledge to generate an inference about some questions. Your response should be in this form:
'Inference: {Inference}'
If there is not proper inference, you can give 'Inference: None'.
Now the knowledge and question is as follows:
"""

infer_answer_instru_prompt = """Use the provided inference as your reasoning thoughts to choose correct answer for some questions. Your response should be in this form:
'({option}) {answer}'
If there is not proper option, you can give 'None'.
Now answer the following questions:
"""

# knowledge_base_cot_instru_prompt = """Copy the knowledge provided as your chain of reasoning to derive the correct answer to the question. Your response should be in this form:
# '{knowledge} 
# So the answer is: ({option}) {answer}'
# Now given the knowledge, answer the following question:
# """

knowledge_base_cot_instru_prompt = """Generate your chain of reasoning based on the provided knowledge to derive the correct answer to the question. Your response should be in this form:
'{reasoning} 
So the answer is: ({option}) {answer}'
Now given the knowledge, answer the following question:
"""

redun_cot_instru_prompt = """Use your commonsense knowledge to choice correct answer for some questions and give the reasoning process. Your response should be in this form:
'{reason} So the answer is: ({option}) {answer}'
Remember you should repeat each sentence in the reasoning from both sides. Now answer the following question:
"""

oppo_cot_instru_prompt = """Use your commonsense knowledge to choice correct answer for some questions and give the reasoning process. Your response should be in this form:
"Knowledge: {Knowledge}
From {answer1}'s perspective, {reason1} So '{answer1}' is the correct/wrong answer.
From {answer2}'s perspective, {reason2} So '{answer2}' is the correct/wrong answer.
So the answer is: ({option}) {answer}"
Now answer the following questions:
"""

consistency_instru_prompt = """Given two statements, please judge whether the first sentence imply the second sentence. Your response should be in this form:
Judgement: {Yes/No}
Now here are the statements:
"""

syllogism_knowledge_instru_prompt = """Use your commonsense knowledge to generate knowledge which is used for reasoning on some questions. Your response should be in this form:
"Rule: {knowledge} (You should generate some general knowledge here)
Case: {knowledge} (You should generate some specific knowledge about the problem stem here)"
Remember you cannot simply restate or rewrite the question as your knowledge.
Now the question is as follows:
"""

syllogism_answer_instru_prompt = """Given the question and following true statements, which option behind the question can be logically concluded to fill in the blank? Provide your answer using syllogistic principles. You will get input in this form:
"Question: {question}
Rule: {rules}
Case: {cases}
You should give your response in the following form restrictly:
Answer: ({option}) {answer}
If there is not proper option, you can give 'Answer: None'.
Now answer the following questions:
"""

sub_question_instru_prompt = """Please split the provided question into different sub questions in order to solve it. Your response should be in this form:
'Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: {content}
Question 4: {content}
......
Question i: What the answer of the question?'
Remember that the question should not repeats each other.
You can not fill in the blank with the option identified as the correct answer, and perform reverse reasoning based on it.
Now here is the provided question:
"""

sub_question_answer_instru_prompt = """Please answer the questions provided in order, you can use the information in the former answers and your own commonsense knowledge for a better reply. Your response should be in this form:
'Answer 1: {answer}
Answer 2: {answer}
......
Answer i: {answer}. So the answer is ({option}) {answer}'.
Remember that you should answer the main question in the last answer. If there is no proper answer, you can give 'Answer: None'.
Now answer the following questions:
"""

# final_answer_extract_prompt = """With provided question and answer, please extract the correct option of the question from the answer. Your response should be in this form:
# Answer: ({option}) {answer}.
# If there is not proper option, you can give 'Answer: None'.
# Now here is the question and the related answer:
# """

wino_sub_questions = [
"""Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: What might be the reason why the test was hard for Samuel?
Question 4: What the meaning of the test was a breeze for Randy?
Question 5: What might be the reason why the test was a breeze for Randy?
Question 6: What the answer of the question?
""",
"""Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: What the meaning of manipulating people?
Question 4: What the meaning of worming their way into the life of others?
Question 5: What might be the reason why Kyle slowly wormed their way into the life of Derrick,?
Question 6: What the answer of the question?
""",
"""Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: What the meaning of being very capricious all the time?
Question 4: What Donald might do next as Donald was very grounded?
Question 5: What Michael might do next as Michael often got lost in their daydreams?
Question 6: What the answer of the question?
""", 
"""Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: What might be the reason why Betty was able to help Rachel treat their asthma?
Question 4: What the answer of the question?
""",
"""Question 1: What the goal of the question?
Question 2: What the relation bettween the goal and the given information in the stem?
Question 3: What the meaning of being sick with the stomach flu?
Question 4: What Jeffrey might do next as Jeffrey was sick with the stomach flu and not hungry?
Question 5: What Christopher might do next as Christopher was starving?
Question 6: What the answer of the question?
"""
]

wino_sub_question_answers = [
"""Answer 1: The question asks whether it is Samuel or Randy who had failed to study for the test.
Answer 2: Based on the key conjuction 'since' behind '_', the goal is the reason that leads to the former information.
Answer 3: Since we know that Samuel feel the test very hard, she may not prepare or study for it.
Answer 4: The test was a breeze for Randy indicates that the test is easy for Randy.
Answer 5: From Answer 4 we can know that the test was a breeze for Randy indicates Randy feel the test easy. Then, if she feels the test easy, that is may because she study hard for the test.
Answer 6: From Answer 3 and Answer 5, we can get that Samuel may not study for the test, but Randy does. From Answer 1, we can know the answer asks who fails to study for the test. So the answer is: (1) Samuel.
""",
"""Answer 1: The question asks whether it is Samuel or Randy who is seen as good and manipulating people.
Answer 2: Based on the key conjuction 'because' behind '_', the goal is the reason that leads to the former information.
Answer 3: A person is seen as manipulating people, that means he like to interact with others and others like him.
Answer 4: A person wormes his way into other's life, that means he cares about others.
Answer 5: Kyle slowly wormed their way into the life of Derrick, that's may because Kyle is friendly and cares about others.
Answer 6: From Answer 5, we can get Kyle is friendly and cares about others, from Answer 3, we can get that the person who is seen as manipulating people likes to interact with others. Thus, Kyle is likely to be seen as manipulating people. From Answer 1, we can get that the question asks who is the man that is seen as manipulating people. So the answer is: (1) Kyle.  
""",
"""Answer 1: The question asks whether it is Donald or Michael who was very capricious all the time.
Answer 2: As there is no explicit key conjuction behind '_', the goal is the event that happens after the former information.
Answer 3: A person is capricious all the time means he does everything only according to his own ideas.
Answer 4: If Donald was very grounded, he tends to be reasonable and does not like to fantasize.
Answer 5: If Michael often got lost in their daydreams, he is seen as unrealistic and egocentric.
Answer 6: From Answer 5, we can get that Michael is egocentric. From Answer 3, we can get A person is capricious all the time means he does everything only according to his own ideas. So Michael tends to be capricious. From Answer 4, we can get that Donald is reasonable and does not like to fantasize, so he is not capricious. From Answer 1, we can get the question asks who was very capricious all the time. So the answer is: (2) Michael.
""", 
"""Answer 1: The question asks whether it is Betty or Rachel who has asthma too.
Answer 2: Based on the key conjuction 'because' behind '_', the goal is the reason that leads to the former information.
Answer 3: If Betty was able to help Rachel treat their asthma, she may have the experience to treat it.
Answer 4: From Answer 3, we can get that Betty has the experience to treat asthma, so we can infer that Betty may have asthma as Rachel does. From Answer 1, we can get the question asks who has asthma too. Besides, since Rachel is clearly being treated for asthma, it is not possible to use 'too' here to ask if he has asthma. So the answer is: (1) Betty.
""",
"""Answer 1: The question asks whether it is Jeffrey or Christopher who ordered food.
Answer 2: As there is no explicit key conjuction behind '_', the goal is the event that happens after the former information.
Answer 3: Stomach flu is a viral infection in the digestive system, which make it hard for people to eat food.
Answer 4: If Jeffrey was sick with the stomach flu and not hungry, he may not want to eat anything.
Answer 5: If Christopher was starving, then he tends to eat food.
Answer 6: From Answer 3 and Answer 4, we can get that Jeffrey may not want to eat anything, then he will not order food. From Answer 5, we can get that Christopher tends to eat food, then he wiil order food. From Answer 1, we can get that the question asks who ordered food. So the answer is: (2) Christopher.
"""
]

wino_question_goals = [
"""Question: Whether it is Samuel or Randy who had failed to study for the test?
""",
"""Question: Whether it is Samuel or Randy who is seen as good and manipulating people?
""",
"""Question: Whether it is Donald or Michael who was very capricious all the time?
""", 
"""Question: Whether it is Betty or Rachel who has asthma too?
""",
"""Question: Whether it is Jeffrey or Christopher who ordered food?
"""
    
]

wino_syllogism_knowledge = [
"""Rule: If someone feel the test very hard, then he failed to study for a test. If someone want to pass the test, then he need to study for it. If a person feels the test easy, then he have studied hard for it.
Case: If the test is a breeze for Randy, then Randy feel the test easy.
""",
"""Rule: If someone cares about others, then he is good. If a person cares about other people's lives, then he is manipulating people.
Case: If Kyle slowly wormes his way into Derrick's life, then Kyle cares about others. 
""",
"""Rule: If someone is reasonable, then he is not seen as changeable and unaccountable. If someone is changeable and unaccountable, then he is very capricious all the time. If someone is egocentric, then he tends to be very capricious all the time.
Case: If Donald is grounded, then Donald is reasonable. If Michael often got lost in the daydreams, then Michael is egocentric.
""",
"""Rule: If someone have the experience to treat asthma, then he may has asthma.
Case: If Betty is able to help Rachel treat the asthma, then she have the experience to treat it.
""",
"""Rule: If someone feel hard to eat food, then he does not want to eat food. If someone is not hungry, then he does not want to eat food. If someone does not want to eat food, then he does not order food. If someone want to eat food, then he order food.
Case: If Jeffrey is sick with the stomach flu, then it may be hard for Jeffrey to eat food. If Christopher was starving, then he want to eat food. 
"""
]


wino_consistency_examples = [
"""Statement 1: If someone feel the test very hard, then he failed to study for a test.
Statement 2: The test was hard for Samuel but a breeze for Randy, since Randy had failed to study for it.
""",
"""Statement 1: A person wormes his way into other's life, because he cares about others.
Statement 2: Kyle slowly wormed their way into the life of Derrick, because Kyle was good and manipulating people.
""",
"""Statement 1: A person often gets lost in his daydreams, he is seen as egocentric. 
Statement 2: Donald was very grounded but Michael often got lost in their daydreams. Donald was very capricious all the time.
""",
"""Statement 1: A person who has a mild disease may knows how to treat it. 
Statement 2: After stopping when running, Betty was able to help Rachel treat their asthma because Betty has it too.
""",
"""Statement 1: A person is starve so that he need to eat something next.
Statement 2: Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. Christopher ordered food.
"""
]


wino_consistency_judge = [
"""Judgement: No
""",
"""Judgement: Yes
""",
"""Judgement: No
""",
"""Judgement: Yes
""",
"""Judgement: Yes
"""
]


# gpt4_v1_prompt_head = """I will provide you with a few scenarios and questions related to commonsense reasoning. Please observe the examples and answer the question that follows.

# """

# gpt4_v1_prompt_body = """Example {}:
# Scenario: {}
# Question: {}

# """

# gpt4_v1_prompt_tail = """Now, please apply your commonsense reasoning to the following scenario:

# Scenario: {}
# Question: {}
# Options:
# A) {}
# B) {}

# Choose the most reasonable answer and explain your choice using sound logic and commonsense understanding.
# """


wino_choices = [
"""A) Samuel B) Randy
""",
"""A) Kyle B) Derrick
""",
"""A) Donald B) Michael
""",
"""A) Betty B) Rachel
""",
"""A) Jeffrey B) Christopher
"""
]


wino_cot_answers_v1 = [
"""Answer: To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. Since we know that Samuel feel the test very hard, she may fail to study for it. So the answer is: A) Samuel.
""",
"""Answer: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him. Since Kyle slowly wormed their way into the life of Derric, Kyle will be seen as good and manipulating people. So the answer is: A) Kyle.
""",
"""Answer: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas. Since Michael often gets lost in their daydreams but Donald does not, Michael is seen as very capricious all the time. So the answer is: B) Michael.
""",
"""Answer: A person usually need to stop running to do other things. A person who has a mild disease may knows how to treat it. Asthma usually causes difficulty in breathing and is easily triggered by exercise. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors. Since Since Betty was able to help Rachel treat their asthma, Betty may has it, too. So the answer is: A) Betty.
""",
"""Answer: Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. Ordering food means to buy food online. A person may eat food after ordering it. Since Christopher was starving and Jeffery was sick and not hungry, Christopher tended to order food. So the answer is: B) Christopher.
"""
]


wino_one_knowledge_infer = [
"""Inference: The test was hard for Samuel, so Samuel failed to study for a test.
""",
"""Inference: Kyle slowly wormed their way into the life of Derrick, because Kyle cares about others.
""",
"""Inference: Michael often got lost in their daydreams, Michael is seen as unrealistic and egocentric. 
""",
"""Inference: Since Betty was able to help Rachel treat their asthma, Betty has asthma, too.
""",
"""Inference: Christopher was starving, so that Christopher needs to eat something next.
"""
]


wino_question_stems = [
"""The test was hard for Samuel but a breeze for Randy , since _ had failed to study for it.
""",
"""Kyle slowly wormed their way into the life of Derrick, because _ was good and manipulating people.
""",
"""Donald was very grounded but Michael often got lost in their daydreams. _ was very capricious all the time.
""",
"""After stopping when running, Betty was able to help Rachel treat their asthma because _ has it too.
""",
"""Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. _ ordered food.
"""
]


wino_questions = [
"""The test was hard for Samuel but a breeze for Randy , since _ had failed to study for it.
(1) Samuel (2) Randy
""",
"""Kyle slowly wormed their way into the life of Derrick, because _ was good and manipulating people.
(1) Kyle (2) Derrick
""",
"""Donald was very grounded but Michael often got lost in their daydreams. _ was very capricious all the time.
(1) Donald (2) Michael
""",
"""After stopping when running, Betty was able to help Rachel treat their asthma because _ has it too.
(1) Betty (2) Rachel
""",
"""Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. _ ordered food.
(1) Jeffrey (2) Christopher
"""
]

wino_reflect_knowledge = [
"""Knowledge: If a person fail to study for a test, he may not like learning or be busy with doing other things.
""",
"""Knowledge: Manipulating people refers to the act of trying to influence or control others' thoughts, feelings, or actions, often in a deceptive or dishonest manner, which is seen as a very bad performance.
""",
"""Knowledge: If a person is very grounded, he tends to work hard and become successful.
"""
]




# wino_knowledge_v2 = [
# """Knowledge: To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it.
# """,
# """Knowledge: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
# """,
# """Knowledge: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas.
# """,
# """Knowledge: A person who has a mild disease may knows how to treat it. A person is able to help treat the asthma because he has the experience to treat it.
# """,
# """Knowledge: A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. A person may eat food after ordering it. 
# """
# ]


wino_rewrite_questions = [
"""Given that the test was hard for Samuel but a breeze for Randy, in that case, which of them probably had failed to study for it?
(1) Samuel (2) Randy
""",
"""Since Kyle slowly wormed their way into the life of Derrick, who is seen as good and manipulating people?
(1) Kyle (2) Derrick
""",
"""As Donald was very grounded while Michael often got lost in their daydreams. If any of them is very capricious all the time, who might that person be?
(1) Donald (2) Michael
""",
"""Given that after stopping when running, Betty was able to help Rachel treat their asthma, because who has it too?
(1) Betty (2) Rachel
""",
"""Given that Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. Who ordered food?
(1) Jeffrey (2) Christopher
"""    
]

wino_graphs = [
"""
Knowledge 1: (The test, is hard for, Randy)
Knowledge 2: ()
The test was hard for Samuel but a breeze for Randy , since _ had failed to study for it.
(1) Samuel (2) Randy
""",
"""Kyle slowly wormed their way into the life of Derrick, because _ was good and manipulating people.
(1) Kyle (2) Derrick
""",
"""Donald was very grounded but Michael often got lost in their daydreams. _ was very capricious all the time.
(1) Donald (2) Michael
""",
"""After stopping when running, Betty was able to help Rachel treat their asthma because _ has it too.
(1) Betty (2) Rachel
""",
"""Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. _ ordered food.
(1) Jeffrey (2) Christopher
"""
]


wino_sup_questions = [
"""If someone fails to study for a test, then the test is probably _ for him.
(1) easy (2) hard
""",
"""If a person worms his way into other's lift, he is seen as _.
(1) good (2) bad
""",
"""Be capricious means that someone often _.
(1) works hard (2) daydreams
""",
"""If someone has asthma, he may _ to treat it.
(1) have experience (2) unable
""",
"""People usually order food because they are _.
(1) starving (2) full
"""
]

wino_sup2_questions = [
"""Question: If someone fails to study for a test, then the test is probably easy for him.
(1) yes (2) no
""",
"""Question: If a person worms his way into other's lift, he is seen as good.
(1) yes (2) no
""",
"""Question: Be capricious means that someone often daydreams.
(1) yes (2) no
""",
"""Question: If someone has asthma, he may have experience to treat it.
(1) yes (2) no
""",
"""Question: People usually order food because they are full.
(1) yes (2) no
"""
]

wino_sup2_answers = [
"""Answer: (2) no
""",
"""Answer: (1) yes
""",
"""Answer: (1) yes
""",
"""Answer: (1) yes
""",
"""Answer: (2) no
"""    
]


wino_sup_answers = [
"""Answer: (2) hard
""",
"""Answer: (1) good
""",
"""Answer: (2) daydreams
""",
"""Answer: (1) have experience
""",
"""Answer: (1) starving
"""    
]


wino_knowledge_answers = [
""""Knowledge: To pass a test, a person need to study for it. If a perso n feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it.
Answer: (1) Samuel
""",
"""Knowledge: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
Answer: (1) Kyle
""",
"""Knowledge: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas.
Answer: (2) Michael
""",
"""Knowledge: A person usually need to stop running to do other things. A person who has a mild disease may knows how to treat it. Asthma usually causes difficulty in breathing and is easily triggered by exercise. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors.
Answer: (1) Betty
""",
"""Knowledge: Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. Ordering food means to buy food online. A person may eat food after ordering it.
Answer: (2) Christopher
"""
]


wino_knowledge = [
"""Knowledge: To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it.
""",
"""Knowledge: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
""",
"""Knowledge: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas.
""",
"""Knowledge: A person who has a mild disease may knows how to treat it. A person is able to help treat the asthma because he has the experience to treat it.
""",
"""Knowledge: A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. A person may eat food after ordering it. 
"""
]



wino_knowledge_infer = [ 
"""Knowledge: If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. If someone feel the test very hard, then he failed to study for a test.
Randy feels the test like a breeze, it means the test is easy for Randy. Randy feels the test easy, because Randy studies hard for it. Samuel feels the test very hard, then Samuel failed to study for a test.""",
"""Knowledge: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
Kyle wormes his way into Derrick's life, because Kyle is friendly and approachable. Kyle is considered good. Whether Kyle or Derrick is seen as manipulating people, that depends on which of them like to interact with others and others like him.""",
"""Knowledge: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas.
Donald is grounded means Donald works hard and does not like to fantasize. Michael often gets lost in his daydreams, Michael is seen as unrealistic and egocentric. Whether Donald or Michael is capricious all the time depends on which of them does everything only according to his own ideas.""",
"""Knowledge: A person is able to help treat the asthma because he has the experience to treat it. 
Betty is able to help treat the asthma because Betty has the experience to treat it. """,
"""Knowledge: A person is not hungry means he does not want to eat something. A person who need to eat food tends to order food.  
Christopher is starve so that Christopher needs to eat something next. Christopher needs to eat food, so Christopher tends to order food."""
]

wino_one_knowledge = [
"""Knowledge: If someone feel the test very hard, then he failed to study for a test.
""",
"""Knowledge: A person wormes his way into other's life, because he cares about others.
""",
"""Knowledge: A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. 
""",
"""Knowledge: A person who has a mild disease may knows how to treat it. 
""",
"""Knowledge: A person is starve so that he need to eat something next.  
"""
]

wino_retrieve_knowledge = [
"""Knowledge: A person feels the test like a breeze, it means the test is easy for him.
""",
"""Knowledge: Since Kyle slowly wormed their way into the life of Derric, Derric may be a good friend for others.
""",
"""Knowledge: A person often gets lost in his daydreams, he is seen as unrealistic.
""",
"""Knowledge: Asthma usually causes difficulty in breathing and is easily triggered by exercise.
""",
"""Knowledge: If someone is sick, he may need to eat something to get strength.
"""      
]

wino_retrieve_rates = [
"""Rate: 0.6
""",
"""Rate: -0.5
""",
"""Rate: 0.1
""",
"""Rate: 0.0
""",
"""Rate: -0.9
"""
]

wino_extra_knowledge = [
"""Knowledge: If someone feel the test like a breeze, he may have studied hard for it.
""",
"""Knowledge: A person cares about other's life, he is seen as good.
""",
"""Knowledge: If a person is grounded, he is not seen as changeable and unaccountable. 
""",
"""Knowledge: A person is able to help treat the asthma because he has the experience to treat it. 
""",
"""Knowledge: A person who need to eat food tends to order food.  
"""
]


wino_wrong_answers = [
"""Answer: (2) Randy
""",
"""Answer: (2) Derrick
""",
"""Answer: (1) Donald
""",
"""Answer: (2) Rachel
""",
"""Answer: (1) Jeffrey
"""
]

wino_hint_statements = [
"""The test was hard for Samuel but a breeze for Randy , since Samuel had failed to study for it.
""",
"""Kyle slowly wormed their way into the life of Derrick, because Kyle was good and manipulating people.
""",
"""Donald was very grounded but Michael often got lost in their daydreams. Michael was very capricious all the time.
""",
"""After stopping when running, Betty was able to help Rachel treat their asthma because Betty has it too.
""",
"""Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. Christopher ordered food.
"""
]


wino_answers = [
"""Answer: (1) Samuel
""",
"""Answer: (1) Kyle
""",
"""Answer: (2) Michael
""",
"""Answer: (1) Betty
""",
"""Answer: (2) Christopher
"""
]

wino_answer_options = [
"""(1) Samuel
""",
"""(1) Kyle
""",
"""(2) Michael
""",
"""(1) Betty
""",
"""(2) Christopher
"""
]

wino_cot = [
"""Reasoning: To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. Since we know that Samuel feel the test very hard, she may fail to study for it.
""",
"""Reasoning: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him. Since Kyle slowly wormed their way into the life of Derric, Kyle will be seen as good and manipulating people.
""",
"""Reasoning: A person is grounded tends to work hard and do not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas. Since Michael often gets lost in their daydreams but Donald does not, Michael is seen as very capricious all the time and Donald isn't.
""",
"""Reasoning: A person who has had a mild disease may knows how to treat it. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors. Since Betty was able to help Rachel treat their asthma, Betty may have the experience of get it. Thus, Betty may have asthma, too. 
""",
"""Reasoning: Stomach flu is a viral infection in the digestive system, which makes people hard to eat. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. A person may eat food after ordering it. Since Christopher was starving and Jeffery was sick and not hungry, Christopher tended to order food in order to eat.
"""    
]

wino_cot_answers = [
"""To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. Since we know that Samuel feel the test very hard, she may fail to study for it. 
So the answer is: (1) Samuel.
""",
"""A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him. Since Kyle slowly wormed their way into the life of Derric, Kyle will be seen as good and manipulating people. 
So the answer is: (1) Kyle.
""",
"""A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas. Since Michael often gets lost in their daydreams but Donald does not, Michael is seen as very capricious all the time. 
So the answer is: (2) Michael.
""",
"""A person usually need to stop running to do other things. A person who has a mild disease may knows how to treat it. Asthma usually causes difficulty in breathing and is easily triggered by exercise. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors. Since Since Betty was able to help Rachel treat their asthma, Betty may has it, too. 
So the answer is: (1) Betty.
""",
"""Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. Ordering food means to buy food online. A person may eat food after ordering it. Since Christopher was starving and Jeffery was sick and not hungry, Christopher tended to order food. 
So the answer is: (2) Christopher.
"""
]


wino_copy_knowledge_answers = [
"""To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it.
So the answer is: (1) Samuel.
""",
"""A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
So the answer is: (1) Kyle.
""",
"""A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas.
So the answer is: (2) Michael.
""",
"""A person who has a mild disease may knows how to treat it. A person is able to help treat the asthma because he has the experience to treat it.
So the answer is: (1) Betty.
""",
"""A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. A person may eat food after ordering it. 
So the answer is: (2) Christopher.
"""
]
    


wino_exclude_cot_answers = [
"""The test is a breeze for Randy means it is easy for her to pass the exam. Therefore, he probably had studied hard for the exam. The question asks who had failed to studied for the exam. So the answer is: (1) Samuel.
""",
"""Since others worms his way to Derrick's life, Derrick is seen as passive. Manipulating people means manipulating other's mental and emotional sides, it's an extremely active action, which contradicts with Derrick's personality. Besides, we cannot judge whether Derrick is a good person from the question. So the answer is: (1) Kyle.
""",
"""Donald was very grounded means he is sensible and reasonable, and that he understands the importance of ordinary things in life. Therefore, Donald tends to maintain a stable mood. However, being capricious all the time means one's mood is unaccountable and changeable, which conflicts with Donald's personality. So the answer is: (2) Michael.
""",
"""Since Rachel is the one who is treated by others, it is not appropriate to use 'too' to describe his sick behavior here. Besides, the one who was able to help Rachel treat his asthma may has it too, since asthma is not a very serious disease and may not needs a doctor to treat. So the answer is: (1) Betty.
""",
"""Stomach flu is a viral infection in the digestive system, since Jeffrey was sick with the stomach flu, it may be hard for him to eat something. Besides, Jeffrey is not hungry, so he does not need food. As the question asks who ordered food, the answer is probably the one who is starving and need to eat food, rather than Jeffrey. So the answer is: (2) Christopher.
"""
]


wino_redun_cot_answers = [
"""To pass a test, a person need to study for it. Not to pass a test, a person does not need to study for it. If a person feel the test like a breeze, it means the test is easy for him. If a person does not feel the test like a breeze, it means the test is not easy for him. A person feels the test easy, because he studies hard for it. A person feels the test not easy or hard, because he fails to study hard for it. Since we know that Samuel feel the test very hard, she may fail to study for it. Since we know that Randy does not feel the test very hard, she probabily does not fail to study for it. So the answer is: (1) Samuel.
""",
"""A person wormes his way into other's life, because he is friendly and approachable. A person does not worm his way into other's life, because he may not be friendly or approachable. A friendly person is considered good. A unfriendly person is not considered good. A person is seen as manipulating people, that means he like to interact with others and others like him. A person is not seen as manipulating people, that means he may do not like to interact with others and others like him. Since Kyle slowly wormed their way into the life of Derric, Kyle will be seen as good and manipulating people. Since  Derric does not worm their way into the life of others, Derric will not be seen as good and manipulating people. So the answer is: (1) Kyle.
""",
"""A person is grounded means he works hard and does not like to fantasize. A person is not grounded means he does not work hard and likes to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person never get lost in his daydreams, he is not seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas. A person is not capricious means he does not do everything only according to his own ideas. Since Michael often gets lost in their daydreams but Donald does not, Michael is seen as very capricious all the time. Since Donald does not often get lost in their daydreams, but Michael does, Donald is not seen as very capricious all the time. So the answer is: (2) Michael.
""",
"""A person who has a mild disease may know how to treat it. A person who has a serious disease may do not know how to treat it. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors. Since Betty was able to help Rachel treat their asthma, Betty may has it, too. Since Rachel was not able to help treat their asthma, Rachel may first has it and does not have it, too. So the answer is: (1) Betty.
""",
"""Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not starve so that he does not need to eat something next. A person is not hungry means he does not want to eat something. A person is hungry means he wants to eat something. Ordering food means to buy food online. A person may eat food after ordering it. A person does not eat food if he does not order it. Since Christopher was starving and Jeffery was sick and not hungry, Christopher tended to order food. Since Jeffery was not starving and Christopher was not sick and hungry, Jeffery did not tend to order food. So the answer is: (2) Christopher.
"""
]

wino_oppo_cot_answers = [
"""Knowledge: To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. 
From Samuel's perspective, since we know that Samuel feel the test very hard, she may fail to study for it. So 'Samuel' is the correct option.
From Randy's perspective, the test is a breeze for Randy means it is easy for her to pass the exam. Therefore, he probably had studied hard for the exam. So 'Randy' is the wrong option.
So the answer is: (1) Samuel.
""",
"""Knowledge: A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him.
From Kyle's perspective, according to the knowledge, since Kyle slowly wormed their way into the life of others, Kyle will be seen as good and manipulating people. So 'Kyle' is the correct option.
From Derrick's perspective, since others worms his way to Derrick's life, Derrick is seen as passive. Manipulating people means manipulating other's mental and emotional sides, it's an extremely active action, which contradicts with Derrick's personality. Besides, we cannot judge whether Derrick is a good person from the question. So 'Derrick' is the wrong option.
So the answer is: (1) Kyle.
""",
"""Knowledge: A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. Being capricious all the time means one's mood is unaccountable and changeable.
From Donald's perspective, Donald was very grounded means he is sensible and reasonable, and that he understands the importance of ordinary things in life. Therefore, Donald tends to maintain a stable mood. However, being capricious all the time means one's mood is changeable, which conflicts with Donald's personality. So 'Donald' is the wrong option.
From Michael's perspective, since Michael often gets lost in their daydreams, Michael is seen as changeable and unaccountable. Hence, being very capricious all the time is a good description for Michael. So 'Michael' is the correct option.
So the answer is: (2) Michael.
""",
"""Knowledge: A person who has experience of getting a mild disease may knows how to treat it. Asthma usually causes difficulty in breathing and is easily triggered by exercise. Asthma is ususally not a serious disease, which means it does not have to be treated by doctors.
From Betty's perspective, since Betty was able to help others treat their asthma, she probably have the experience of getting the Asthma. Betty may has it, too. So 'Betty' is the correct option.
From Rachel's perspective, since Rachel is the one who is treated by others, it is not appropriate to use 'too' to describe his sick behavior here. So 'Rachel' is the correct option.
So the answer is: (1) Betty.
""",
"""Knowledge: Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. Ordering food means to buy food online. A person may eat food after ordering it.
From Jeffrey's perspective, since Jeffrey was sick with the stomach flu, it may be hard for him to eat something. Besides, Jeffrey is not hungry, so he does not need food. So 'Jeffrey' is the wrong option.
From Christopher's perspective, as the question asks who ordered food, the answer is probably the one who is starving and need to eat food. Since Christopher was starving , Christopher tended to order food. So Christopher is the correct option. 
So the answer is: (2) Christopher.
"""
]


csqa_knowledge_cot_exmaple_questions = [
"""She was giggling and flirting, his mustache elicited what for her?
(A) joyfulness (B) smile (C) getting excited (D) enthusiasm (E) sexual attraction
""",   
"""To deal with sadness many people will do what on their iPhone?
(A) cry (B) listen to music (C) tragic film (D) take phone call (E) get drunk
""",
"""Where is a stool likely to be found?
(A) internet cafe (B) blacksmith's shop (C) office (D) bar (E) building
"""
]

csqa_knowledge_cot_exmaple_answers = [
"""Knowledge: Flirting is a sexual behavior involving body language, or spoken or written communication. Joyfullness and smile both reflect that a person is happy, but they are not necessarily connected with sex. Getting excited and enthusiasm both describe a person is excited, ususally used in sports. Since a person is sexually attracted, he may be flirting and giggling.
Answer: (E) sexual attraction
""",   
"""Knowledge: iPhone is a series of smartphones combining mobile telephone, digital camera, music player, and personal computing technologies. iPhone can not be used to cry or get drunk. When a person feel sad, he may listen to music to relieve the sadness. Tragic film is the film based on human suffering, which may cause bad mood. Taking phone call is a way for people's communication.
Answer: (B) listen to music
""",
"""Knowledge: A stool is a seat without a back or arms. An Internet cafe is a type of cafe, which provides Internet access to the public. Blacksmith's shop is a workshop equipped with a forge, and can be used for smelting or forging weapons, metal tools, and armors. An office is generally a room or other area where administrative work is done. Bar is a place that serves alcohol. A building is an enclosed structure, particularly one constructed of solid materials, with some kind of roof and walls.
Answer: (D) bar
"""    
]


wino_evaluate_knowledge_examples = [
"""Knowledge: If someone feel the test very hard, then he failed to study for a test.
""",
"""Knowledge: A person wormes his way into other's life, because he hates others.
""",
"""Knowledge: A person who has a disease may knows how to treat it. 
""",
"""Knowledge: A person is not hungry so that he tends to eat something next.  
""",
"""Knowledge: A person often gets lost in his daydreams, he is seen as egocentric. 
"""    
]

wino_knowledge_evaluation_examples = [
"""Rate: good
""",
"""Rate: bad.
""",
"""Rate: not sure. 
""",
"""Rate: bad. 
""",
"""Rate: good.  
"""        
]



knowledge_answer_examples = [
"""Knowledge: Batteries can produce steady electrical current. 
Answer: (B) a battery
""",
"""Knowledge: Rubbing hands produces heat because of friction.
Answer: (A) dry palms
""",
"""Knowledge: Electronic maps are the modern version of paper atlas.
Answer: (D) atlas
""",
"""Knowledge: One should sit still when getting a haircut.
Answer: (C) sits on the chair next to the sink.
""",
"""Knowledge: Human has two arms and two legs.
Answer: (E) four
""",
"""Knowledge: Natural light provides energy for photosynthesis.
Answer: (D) plants sprouting, blooming and wilting
""",
"""Knowledge: A hand saw is used for making cuts. A hand drill is used for making holes.
Answer: (A) Use a hand saw to cut the handles.
""",
"""Knowledge: The stomach is part of the digestive system.
Answer: (C) breaks food into nutrients
""",
"""Knowledge: Lower pH means more acid.
Answer: (B) decreases
""",
"""Knowledge: In autumn, the days get short and the nights get long.
Answer: (E) leaves
""",
"""Knowledge: A messy room likely contains dirty clothes.
Answer: (C) Pick up the dirty clothes
""",
"""Knowledge: There are more trees in the forests than in the fields.
Answer: (B) forests
"""
]

csqa2_question_stems = [
"""For a woman the process of putting on her garments would be done in this order: underwear, slip, then dress.
""",
"""Artists will sometimes let a client order custom paintings.
""",
"""In many states, a person is capable of rolling a marijuana joint without breaking the law.
""",
"""Relatives is a term that is used for strangers.
""",
"""After a horse wins a competition, he is considered slow.
"""
]

csqa2_rerank_knowledge = [
"""Knowledge: underwear is a item of clothing worn beneath outer clothes, usually in direct contact with the skin. 
""",
"""Knowledge: Artists creates paintings completely based on their own ideas
""",
"""Knowledge: Marijuana is a psychoactive drug from the cannabis plant. 
""",
"""Knowledge: Usually people have more strangers than relatives.
""",
"""Knowledge: A horse competition tests the overall abilities of rider. 
""" 
]

csqa2_rerank_rates = [
"""Rate: 0.7
""",
"""Rate: -0.9
""",
"""Rate: 0.3
""",
"""Rate: 0.0
""",
"""Rate: -0.3
"""
]


csqa2_questions = [
"""For a woman the process of putting on her garments would be done in this order: underwear, slip, then dress.
(1) yes (2) no
""",
"""Artists will sometimes let a client order custom paintings.
(1) yes (2) no
""",
"""In many states, a person is capable of rolling a marijuana joint without breaking the law.
(1) yes (2) no
""",
"""Relatives is a term that is used for strangers.
(1) yes (2) no
""",
"""After a horse wins a competition, he is considered slow.
(1) yes (2) no
"""
]

csqa2_knowledge = [
"""Knowledge: People usually dress from the inside out. 
""",
"""Knowledge: The artists usually create the paintings following their own ideas, but some artists may draw them according to the customer's thoughts.
""",
"""Knowledge: In the USA, many states allow people to roll a marijuana for entertainment.  
""",
"""Knowledge: If a person is your relative, then you are likely to be familiar.
""",
"""Knowledge: If a horse wins the competition, it is seen as good and fast. 
"""
]

csqa2_one_knowledge = [
"""Knowledge: underwear is a item of clothing worn beneath outer clothes, usually in direct contact with the skin. A slip dress is a woman's dress that closely resembles an underslip or petticoat, usually inside the dress. A dress is a garment traditionally worn by women or girls consisting of a skirt with an attached bodice. People usually dress from the inside out. 
""",
"""Knowledge: Artists can create paintings and sometimes sell them to make money. A client can buy paintings from the artist. The artists usually create the paintings following their own ideas, but some artists may draw them according to the customer's thoughts.
""",
"""Knowledge: Marijuana is a psychoactive drug from the cannabis plant. It is illegal to use, sell and possess marijuana in most countries, except for a few countries including the United States. In the USA, many states allow people to roll a marijuana for entertainment.  
""",
"""Knowledge: The relatives is a person connected by blood or marriage. The strangers is a person whom one does not know or with whom one is not familiar. If a person is your relative, then you are likely to be familiar.
""",
"""Knowledge: A horse competition tests the overall abilities of horse and rider in competition at dressage, cross-country and endurance riding, and stadium show jumping. If a horse wins the competition, it is seen as good and fast. 
"""
]

csqa2_sup_questions = [
"""Question: Do people usually put on underwear before dress?
(1) yes (2) no
""",
"""Question: Do artists always paint according to whatever they want?
(1) yes (2) no
""",
"""Question: Do the law in all of the states prohibit marijuana?
(1) yes (2) no
""",
"""Question: For most people, do relatives as familiar as strangers?
(1) yes (2) no
""",
"""Question: Does a horse need to be fast to win a competition?
(1) yes (2) no
"""
]

csqa2_sup_answers = [
"""Answer: (1) yes
""",
"""Answer: (2) no
""",
"""Answer: (2) no
""",
"""Answer: (2) no
""",
"""Answer: (1) yes
"""    
]

csqa2_answers = [
"""Answer: (1) yes
""",
"""Answer: (1) yes
""",
"""Answer: (1) yes
""",
"""Answer: (2) no
""",
"""Answer: (2) no
"""
]

csqa2_cot_answers = [
"""Underwear is a item of clothing worn beneath outer clothes, usually in direct contact with the skin. A slip dress is a woman's dress that closely resembles an underslip or petticoat, usually inside the dress. A dress is a garment traditionally worn by women or girls consisting of a skirt with an attached bodice. People usually dress from the inside out. So the answer is: (1) yes.
""",
"""Artists can create paintings and sometimes sell them to make money. A client can buy paintings from the artist. The artists usually create the paintings following their own ideas, but some artists may draw them according to the customer's thoughts. In this way, artists sometimes may let a client order custom paintings. So the answer is: (1) yes.
""",
"""Marijuana is a psychoactive drug from the cannabis plant. It is illegal to use, sell and possess marijuana in most countries, except for a few countries including the United States. In the USA, many states allow people to roll a marijuana for entertainment, which means a person is capable of rolling a marijuana joint without breaking the law in these states. So the answer is: (1) yes.
""",
"""The relatives is a person connected by blood or marriage. The strangers is a person whom one does not know or with whom one is not familiar. If a person is your relative, then you are likely to be familiar. In that case, the strangers can not be seen as the relatives in most cases. So the answer is: (2) no.
""",
"""A horse competition tests the overall abilities of horse and rider in competition at dressage, cross-country and endurance riding, and stadium show jumping. If a horse wins the competition, it is seen as good and fast. Since that, after a horse wins a competition, he cannot be considered slow. So the answer is: (2) no.
"""
]

csqa2_knowledge_answers = [
"""Knowledge: underwear is a item of clothing worn beneath outer clothes, usually in direct contact with the skin. A slip dress is a woman's dress that closely resembles an underslip or petticoat, usually inside the dress. A dress is a garment traditionally worn by women or girls consisting of a skirt with an attached bodice. People usually dress from the inside out. 
Answer: (1) yes
""",
"""Knowledge: Artists can create paintings and sometimes sell them to make money. A client can buy paintings from the artist. The artists usually create the paintings following their own ideas, but some artists may draw them according to the customer's thoughts.
Answer: (1) yes
""",
"""Knowledge: Marijuana is a psychoactive drug from the cannabis plant. It is illegal to use, sell and possess marijuana in most countries, except for a few countries including the United States. In the USA, many states allow people to roll a marijuana for entertainment.  
Answer: (1) yes
""",
"""Knowledge: The relatives is a person connected by blood or marriage. The strangers is a person whom one does not know or with whom one is not familiar. If a person is your relative, then you are likely to be familiar. 
Answer: (2) no
""",
"""Knowledge: A horse competition tests the overall abilities of horse and rider in competition at dressage, cross-country and endurance riding, and stadium show jumping. If a horse wins the competition, it is seen as good and fast. 
Answer: (2) no
"""    
]

social_questions = [
"""Taylor led Riley far away, so they can enjoy the forest all by themselves. How would Taylor feel afterwards?
(1) happy (2) In love with Riley (3) bored
""",
"""
Riley was going camping with some friends. they bought a new tent.What will Riley want to do next?
(1) have a tent (2) get a book (3) get firewood
""",
"""
Addison found the bottle in the trash after looking for it around the house. What will Addison want to do next?
(1) ask around (2) drink the bottle (3) inspect the bottle
""",
"""
Austin built Quinn's house last week as a surprise for Quinn's 50th birthday party. Why did Austin do this?
(1) do something nice for a friend (2) give themselves a birthday present (3) please them
""",
"""
Alex found a house in a wonderful neighborhood with great schools. How would Alex feel afterwards?
(1) upset that the floors were carpeted (2) was mad that the trees were cut down (3) was excited to decorate the new house
"""
]

social_answers = [
"""Answer: (1) happy.
""",
"""Answer: (3) get firewood.
""",
"""Answer: (3) inspect the bottle
""",
"""Answer: (1) do something nice for a friend.
""",
"""Answer: (3) was excited to decorate the new house
"""
]


social_cot_answers = [
"""If someone enjoys something, it means he can get much fun from it. As they enjoy the forest, they tend to feel happy afterwards. Playing in the forest is not typically related with falling in love. Besides, two people probably have fallen in love before enjoy something all by themselves, not afterwards. If a person enjoys something, he can't feel bored afterwards.
So the answer is: (1) happy.
""",
"""Camping is a form of outdoor recreation or outdoor education involving overnight stays with a basic temporary shelter such as a tent. When camping, people usually need to pitch a tent, get firewood, make fire, etc. Since they bought a tent, Riley has already had a tent, she does not need to do it next. Getting a book is not necessarily after buying a tent, and people don't need to do it when camping. Because firewood is used for making fire, and making fire is necessary for camping, Riley tend to get firewood next compared with other options.
So the answer is: (3) get firewood.
""",
"""Since Addison looks for the bottle around the house, it is important for him. The bottle has no relation with asking others, so asking aroung cannot happen next. Since the bootle is found in the trash, the water inside is probably gone, so Addison can not drink it anymore. On the other hand, we usually drink the water rather than drink the bottle. As the bottle is important for Addison, there may be something in it, so Addison probably wants to inspect the bottle next.
So the answer is: (3) inspect the bottle
""",
"""People often send presents at one's birthday party. If a person sents a present to his friend, he is seen as a nice friend. Since Austin built a house for Quinn for his birthday, the house is a nice birthday present for Quinn. According to this, we can conclude that Austin wants to give Quinn a nice present for his birthday. Since the house is built for Quinn, it is not a present for themselves. Sending present is a custom for people to celebrate their birthdays, people usually don't do it in order to please others.
So the answer is: (1) do something nice for a friend.
""",
"""After people find a new home, it often needs to be cleaned and renovated. The carpet is usually upon the floor, but people usually don't feel upset because of it. The trees were cut down may let Alex feel mad, but considering that he think the neighborhood is wonderful, he's more likely to be in a good mood. People often feel excited if they move into a new house in a wonderful place.
So the answer is: (3) was excited to decorate the new house
"""
]

social_knowledge = [
"""Knowledge: If someone enjoys something, it means he can get much fun from it. As they enjoy the forest, they tend to feel happy afterwards. Playing in the forest is not typically related with falling in love. Besides, two people probably have fallen in love before enjoy something all by themselves, not afterwards. If a person enjoys something, he can't feel bored afterwards.
""",
"""Knowledge: Camping is a form of outdoor recreation or outdoor education involving overnight stays with a basic temporary shelter such as a tent. When camping, people usually need to pitch a tent, get firewood, make fire, etc. Since they bought a tent, Riley has already had a tent, she does not need to do it next. Getting a book is not necessarily after buying a tent, and people don't need to do it when camping.
""",
"""Knowledge: Since Addison looks for the bottle around the house, it is important for him. The bottle has no relation with asking others, so asking aroung cannot happen next. Since the bootle is found in the trash, the water inside is probably gone, so Addison can not drink it anymore. On the other hand, we usually drink the water rather than drink the bottle. 
""",
"""Knowledge: People often send presents at one's birthday party. If a person sents a present to his friend, he is seen as a nice friend. Since Austin built a house for Quinn for his birthday, the house is a nice birthday present for Quinn. According to this, we can conclude that Austin wants to give Quinn a nice present for his birthday. Since the house is built for Quinn, it is not a present for themselves. Sending present is a custom for people to celebrate their birthdays, people usually don't do it in order to please others.
""",
"""Knowledge: After people find a new home, it often needs to be cleaned and renovated. The carpet is usually upon the floor, but people usually don't feel upset because of it. The trees were cut down may let Alex feel mad, but considering that he think the neighborhood is wonderful, he's more likely to be in a good mood. People often feel excited if they move into a new house in a wonderful place.
"""
]

generate_pyke_instru_prompt = """Task Description: You are given some knowledge and a question. The task is to: 
1) define all the predicates in the knowledge and question
2) parse the knowledge and question into logic rules based on the defined predicates
3) write all the facts mentioned in the knowledge and question
4) parse the question into the logic form
Your response should be in this form:
Predicates:
{content} (Define all the predicates here)
Facts:
{content} (Write all the facts here)
Rules:
{content} (parse all the if-then rules here)
Query:
{content} (parse the question into the logic form here)
Here is an example:
"""

wino_pyke_questions = [
"""Knowledge:  If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. 
Question: The test was hard for Samuel but a breeze for Randy , since _ had failed to study for it.
(1) Samuel (2) Randy
"""
]

wino_pyke_answers= [
"""Predicates:
StudyForTest($x, bool) ::: Does x study for test?
FailStudyForTest($x, bool) ::: Does x fail to study for test?
TestEasy($x, bool) ::: Does x feel test easy?
TestBreeze($x, bool) ::: Does x feel test like a breeze?
TestHard($x, bool) ::: Does x feel test hard?
Facts:
TestBreeze(Randy, True) ::: Randy feels test like a breeze.
TestHard(Samuel, True) ::: Samuel feels test hard.
Rules:
StudyForTest($x, True) >>> FailStudyForTest($x, False) ::: If someone study for test then he does not fail to study for test.
StudyForTest($x, False) >>> FailStudyForTest($x, True) ::: If someone does not study for test then he fails to study for test.
TestBreeze($x, True) >>> TestEasy($x, True) ::: If someone feels test like a breeze then he feels test easy.
TestEasy($x, True) >>> StudyForTest($x, True) ::: If someone feels test easy then he studys for test. 
TestHard($x, True) >>> StudyForTest($x, False) ::: If someone feels test hard then he does not study for test.
Query:
FailStudyForTest(Samuel, True) ::: Samuel fails to study for test.
FailStudyForTest(Randy, True) ::: Randy fails to study for test.
"""
]

reader_ft_prompt =  """Below is an question, use your commonsense knowledge to answer the correct option.
Question: {}
Answer: """
        


reader_knowledge_prompt = """Below is an question, paired with some related knowledge, use the knowledge to answer the correct option.
Question: {}
{}
Answer: """
        

# classifier_prompt = """Below is an question, paired with related knowledge. 
# Please classify whether the knowledge is useful (label 0) or harmful (label 1) for answering the question correctly.
# Question: {}
# {}"""

classifier_prompt = """Below is an question, paired with related knowledge. 
Please classify whether the knowledge is useful (label 0) or harmful (label 1) or useless (label 2) for answering the question correctly.
Question: {}
{}"""

classifier_prompt = """Below is an question, paired with related knowledge. 
Please classify whether the knowledge is useful (label 0) or harmless (label 1) or useless (label 2) or harmful(label 3) for answering the question correctly.
Question: {}
{}"""

wino_cot_auto = [
"""Step 1: We know that the test was hard for Samuel. This means that Samuel had difficulty with the test. 
Step 2: We know that the test was a breeze for Randy. This means that Randy found the test easy. 
Step 3: We are looking for the person who failed to study for the test. Based on the information given, it is logical to assume that Samuel, who had difficulty with the test, could be the one who failed to study. 
[END]
""",
"""Step 1: First, let's determine who is the subject of the sentence. The subject is "Kyle slowly wormed their way into the life of Derrick." 
Step 2: Now, let's consider the reason why Kyle wormed their way into Derrick's life. The reason given is that Kyle was good at manipulating people. 
[END]
""",
"""Step 1: First, we know that Donald is grounded and Michael often gets lost in daydreams. The word "capricious" means unpredictable or impulsive, so we need to determine who is more likely to exhibit these traits.
Step 2: Based on the information given, it is more likely that Michael is the one who is capricious. This is because daydreaming often involves being lost in one's thoughts and not being grounded in reality.
[END]
""",
"""Step 1: Since Betty was able to help treat Rachel's asthma, it means Betty must have the knowledge and experience to do so. 
[END]
""",
"""Step 1: Since Jeffrey was sick with the stomach flu and not hungry, it would not make sense for him to order food. On the other hand, Christopher was starving, indicating that he would likely be the one to order food.
[END]
"""
]

wino_cot_answers_auto = [
"""We know that the test was hard for Samuel. This means that Samuel had difficulty with the test. We know that the test was a breeze for Randy. This means that Randy found the test easy. We are looking for the person who failed to study for the test. Based on the information given, it is logical to assume that Samuel, who had difficulty with the test, could be the one who failed to study. 
So the answer is: (1) Samuel.
""",
"""First, let's determine who is the subject of the sentence. The subject is "Kyle slowly wormed their way into the life of Derrick." Now, let's consider the reason why Kyle wormed their way into Derrick's life. The reason given is that Kyle was good at manipulating people. 
So the answer is: (1) Kyle.
""",
"""First, we know that Donald is grounded and Michael often gets lost in daydreams. The word "capricious" means unpredictable or impulsive, so we need to determine who is more likely to exhibit these traits. Based on the information given, it is more likely that Michael is the one who is capricious. This is because daydreaming often involves being lost in one's thoughts and not being grounded in reality.
So the answer is: (2) Michael.
""",
"""Since Betty was able to help treat Rachel's asthma, it means Betty must have the knowledge and experience to do so. 
So the answer is: (1) Betty.
""",
"""Since Jeffrey was sick with the stomach flu and not hungry, it would not make sense for him to order food. On the other hand, Christopher was starving, indicating that he would likely be the one to order food.
So the answer is: (2) Christopher.
"""
]


wino_cot_wrong_auto = [
"""Step 1: The question states that the test was hard for Samuel but a breeze for Randy, implying that one of them failed to study for it.
Step 2: Since the test was hard for Samuel, it can be inferred that he studied for it. Therefore, the one who failed to study for the test must be Randy.
[END]
""",
"""Step 1: We know that Kyle slowly wormed their way into the life of Derrick. This means that Kyle gradually became a part of Derrick's life. 
Step 2: We know that Kyle was good at manipulating people. This means that Kyle had the ability to influence or control others. 
Step 3: We are looking for the individual who was manipulated by Kyle. Based on the information given, it is logical to assume that Derrick, who had Kyle worm their way into his life, could be the person who was manipulated. 
[END]
""",
"""Step 1: Based on the statement "Donald was very grounded but Michael often got lost in their daydreams," we can infer that Donald is the one who is described as being grounded.
Step 2: The statement in the question "___ was very capricious all the time" implies that the person being referred to is always capricious. Since it is already established that Donald is grounded, the only possible answer is (1) Donald.
[END]
""",
"""Step 1: Since Betty was able to help Rachel treat their asthma, it means Rachel has asthma and Betty was able to provide assistance.
Step 2: Therefore, the answer must be Rachel.
[END]
""",
"""Step 1: The question states that Jeffrey was sick with the stomach flu and not hungry, while Christopher was starving.
Step 2: Since Christopher was starving, it can be inferred that he ordered food.
Step 3: Therefore, the answer must be Jeffrey, as he was not hungry and did not order food.
[END]
"""
]

hella_question_stems = [
"""Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then""",
"""The man in the center is demonstrating a hairstyle on the person wearing the blue shirt. the man in the blue shirt""",
"""The roof is done and a view of the entire house is shown to show off the finished roof. the woman""",
"""People practice ballet in a studio alone and in couples. then""",
"""A person is seen standing on a tennis court bouncing a ball. another man"""   
]

hella_questions = [
"""Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then
(1) the man adds wax to the windshield and cuts it. (2) a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled. (3) the man puts on a christmas coat, knitted with netting. (4) the man continues removing the snow on his car.""",
"""The man in the center is demonstrating a hairstyle on the person wearing the blue shirt. the man in the blue shirt
(1) is standing on the sponge cutting the hair of the person wearing the blue shirt. (2) is doing the hairstyle with his hand and the hairspray. (3) sits on the chair next to the sink. (4) is being shown eye to eye.""",
"""The roof is done and a view of the entire house is shown to show off the finished roof. the woman
(1) is standing in front of the home, smiling while talking. (2) interviews the man again and leaves the room. (3) shows the soil with two lay-ups of shingle and applies a layer onto the top shingle. (4) stacks the bags on the side and begins putting stencils on the top.""",
"""People practice ballet in a studio alone and in couples. then
(1) a man begins dancing and top dancing standing on the ground. (2) a boy and a girl dance ballet, then a man enter and dance with the girl. (3) the band performs ballet in the studio and in the open gathered. (4) people dances and dances together, dancing.""",
"""A person is seen standing on a tennis court bouncing a ball. another man
(1) takes his turn speaking to the camera. (2) walks up hitting a birdie. (3) is seen standing ready on the other side in front of a large audience. (4) is holding the racket next to him."""
]

hella_answers = [
"""Answer: (4) the man continues removing the snow on his car.""",
"""Answer: (3) sits on the chair next to the sink.""",
"""Answer: (1) is standing in front of the home, smiling while talking.""",
"""Answer: (2) a boy and a girl dance ballet, then a man enter and dance with the girl.""",
"""Answer: (3) is seen standing ready on the other side in front of a large audience."""
]

hella_knowledge = [
"""Knowledge: Snow must be removed from a car before one can drive it.""",
"""Knowledge: One should sit still when getting a haircut. People often sits down when they are get a haircut.""",
"""Knowledge: One usually feels pleased after finishing a home improvement project. If a person feels pleased, he tends to smile.""",
"""Knowledge: Ballet is a kind of dance. If people pratice ballet, they often dance with their partener or dance alone.""",
"""Knowledge: The player who bounces the ball is the one who serves. In the tennis race, two players are located on opposite sides of the court, one is serving and the other is preparing to receive the ball."""
]

hella_cot_answers = [
"""Snow must be removed from a car before one can drive it. Since there is some snow covering the car, so the man tends to remove it next.
So the answer is: (4) the man continues removing the snow on his car.
""",
"""One should sit still when getting a haircut. People often sits down when they are get a haircut. From the context, we can get that the man in the blue shirt is getting a haircut, so he should sit down and wait.
So the answer is: (3) sits on the chair next to the sink.""",
"""One usually feels pleased after finishing a home improvement project. If a person feels pleased, he tends to smile. Since the woman has finished all the work, she may be proud of it and feel pleased. Then she may smile.
So the answer is (1) is standing in front of the home, smiling while talking.""",
"""Ballet is a kind of dance. If people pratice ballet, they often dance with their partener or dance alone, which matches the description of option (2)
So the answer is (2) a boy and a girl dance ballet, then a man enter and dance with the girl.
""",
"""Knowledge: The player who bounces the ball is the one who serves. In the tennis race, two players are located on opposite sides of the court, one is serving and the other is preparing to receive the ball. Since there is a person who serves, another one should stands ready.
So the answer is (3) is seen standing ready on the other side in front of a large audience."""   
]

hella_copy_knowledge_answers = [
"""Snow must be removed from a car before one can drive it.
So the answer is: (4) the man continues removing the snow on his car.""",
"""Knowledge: One should sit still when getting a haircut. People often sits down when they are get a haircut.
So the answer is: (3) sits on the chair next to the sink.""",
"""Knowledge: One usually feels pleased after finishing a home improvement project. If a person feels pleased, he tends to smile.
So the answer is (1) is standing in front of the home, smiling while talking.""",
"""Knowledge: Ballet is a kind of dance. If people pratice ballet, they often dance with their partener or dance alone.
So the answer is (2) a boy and a girl dance ballet, then a man enter and dance with the girl.""",
"""Knowledge: The player who bounces the ball is the one who serves. In the tennis race, two players are located on opposite sides of the court, one is serving and the other is preparing to receive the ball.
So the answer is (3) is seen standing ready on the other side in front of a large audience."""

]


hella_rerank_knowledge = [
"""Knowledge: Snow must be removed from a car before one can drive it.""",
"""Knowledge: People often sits down when they are cutting the hair of others.""",
"""Knowledge: If someone finish some hard task, he won't feel pleased. """,
"""Knowledge: Ballet is a kind of dance that need practicing.""",
"""Knowledge: The player who bounces the ball is the one who serves. 
""" 
]




hella_rerank_rates = [
"""Rate: 0.8
""",
"""Rate: -0.3
""",
"""Rate: -0.9
""",
"""Rate: 0.1
""",
"""Rate: 0.3
"""
]

# wino_reflect_questions = [
# """Kyle loves chocolate and candy and Steven loves pickles and lemons. _ loves sour foods.\n(1) Kyle (2) Steven""",
# """Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n(1) Ian (2) Dennis """,
# """Avery was debating on taking up German instead of Latin, because the _ was newer.\n(1) Latin (2) German """]

wino_reflect_wrong_knowledge = [
"""Knowledge: A person who loves sour foods enjoys taste sensations that are tangy and acidic. Sour foods typically include items such as lemons, sour candy, and pickles. Therefore, both Kyle and Steven could potentially be the one who loves sour foods.""",
"""Knowledge: Ian volunteered to eat Dennis's menudo despite already having a bowl because he dislikes eating intestines specifically. Eating intestine may not be preferred by Ian due to personal taste or dietary preferences.""",
"""Knowledge: Avery was debating on taking up German instead of Latin, because German is newer."""
]

# wino_reflect_true_knowledge = [
# """Knowledge: A person who loves sour foods has a preference for flavors that are tangy or acidic.""",
# """Knowledge: Dennis dislikes eating intestine because he finds it disgusting or distasteful. Therefore, Ian volunteered to eat Dennis's menudo because Ian is willing to eat intestine and does not despise it like Dennis does.""",
# """Knowledge: Avery is considering taking up German instead of Latin because German is a newer language compared to Latin. The word \"newer\" suggests that German is a more modern or recently developed language compared to Latin, which is a classical language that dates back to ancient times."""   
# ]


wino_questions_v2 = [
"Feeling a draft, William asked Neil to please close the front door because _ was farther away.\n(1) William (2) Neil ",
"Lawrence did not like the smell of Joel's new soap but _ liked musky smells.\n(1) Lawrence (2) Joel ",
"Dennis gave his hammer to Robert so he could hammer the nails. _ had no hammers.\n(1) Dennis (2) Robert ",
]

wino_knowledge_v2 = [
"Knowledge: In this scenario, William asked Neil to close the front door because Neil was closer to it. It implies that William wanted the front door to be closed, and since Neil was nearer to the door, it made logical sense for William to ask Neil to do it. The reason for asking Neil to close the front door is simply that Neil was in a more convenient position to do so.",
"Knowledge: A person's scent preferences can vary. Not everyone likes the same smells. Lawrence did not like the smell of Joel's new soap indicating that he has a different preference and does not enjoy that particular scent. In contrast, someone, whose identity is unknown in the original question, liked musky smells in general.",
"Knowledge: In this scenario, Dennis gave his hammer to Robert because Robert did not have any hammers of his own. This implies that Robert did not have a tool specifically designed for hammering nails.",
]

query_generate_knowledge_assistant = """You are a helpful assistant that generate related querys according to the given question. Remeber your query can not include specific person's name.
You should generate querys that help you to retrieve related knowledge to the question. Your response should be in this form:
'Query:{query}'
"""

query_answer_assistant = """You are a helpful assistant that use your commonsense knowledge to answer the given query.
Your response should be in this form:
'Knowledge:{answer}'
"""


wino_querys = [
"""Query: What may happens after a person fail to study for a test?
""",
"""Query: What character does a person have if he wormes his way into other's life?
""",
"""Query: What character does a person have if a person is very capricious all the time?
""",
"""Query: What may the reason that a person can help other to treat their asthma?
""",
"""Query: What may happen next if a person is starving?
"""
]

wino_query_answers = [
"""Knowledge: If person fail to study for a test, he may fails to pass the test next.
""",
"""Knowledge: If a person wormes his way into other's life, he is seen as good and kind.
""",
"""Knowledge: If a person is very capricious all the time, he is seen as living in his own world.
""",
"""Knowledge: If a person can help other to treat their asthma, he may has it too.
""",
"""Knowledge: If a person is starving, he tends to eat something.
""" 
]

generate_app_knowledge_assistant = """You are a helpful asasistant that generate applicable knowledge according to the question. You should generate knowledge that help you correctly answer the question. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot directly answer the question as your knowledge.
"""


piqa_questions = [
"""How do you flood a room?
(1) fill it with objects. (2) fill it with water.
""",
"""How can I get oil stains out of my driveway?
(1) Douse each stain with a couple cans of beer. (2) Douse each stain with a couple cans of soda.
""",
"""Soothe a painful sunburn.
(1) Wait until brewed tea bag is cool, then apply on burn. (2) Wait until brewed tea bag is hot, then apply on burn.
""",
"""What can I use for fuel in an alcohol stove?
(1) Use acetone. (2) Use vinegar.
""",
"""How can I cut the handles of metal cutlery?
(1) Use a hand saw to cut the handles. (2) Use a hand drill to cut the handles.
"""]

piqa_answers = [
"""Answer: (2) fill it with water.
""",
"""Answer: (2) Douse each stain with a couple cans of soda.
""",
"""Answer: (1) Wait until brewed tea bag is cool, then apply on burn.
""",
"""Answer: (1) Use acetone.
""",
"""Answer: (1) Use a hand saw to cut the handles.
"""]

piqa_cot_answers = [
"""Too much water can cause flooding. Thus, if we want to flood a room, we should use water. 
So the answer is: (2) fill it with water.
""",
"""Sodium carbonate solution can wash away oil stains. The soda is a kind of sodium carbonate solution. Thus, you can use cans of soda to get oil stains out of your driveway. 
So the answer is: (2) Douse each stain with a couple cans of soda.
""",
"""Sunburn can be alleviated by applying cold material. Thus, you should apply cool tea rather than hot tea bag to soothe your sunburn. 
So the answer is: (1) Wait until brewed tea bag is cool, then apply on burn.
""",
"""Acetone is flammable, while vinegar is not. If you want to use something for fuel, the thing you use should be flammable. Thus, you should use acetone for fuel in an alcohol stove. 
So the answer is: (1) Use acetone.
""",
"""A hand saw is used for making cuts and a hand drill is used for making holes. If you want to cut something, you should use a hand saw rather than hand drill. 
So the answer is: (1) Use a hand saw to cut the handles.
"""   
]

piqa_knowledge = [
"""Knowledge: Too much water can cause flooding.
""",
"""Knowledge: Sodium carbonate solution can wash away oil stains. The soda is a kind of sodium carbonate solution.
""",
"""Knowledge: Sunburn can be alleviated by applying cold material.
""",
"""Knowledge: Acetone is flammable, while vinegar is not. If you want to use something for fuel, the thing you use should be flammable.
""",
"""Knowledge: A hand saw is used for making cuts and a hand drill is used for making holes. If you want to cut something, you should use a hand saw rather than hand drill. 
"""
]

retrieve_generate_assistant = """You are a helpful assistant that generate knowledge according to the question.
If basic knowledge is provided, you should generate more knowledge according to it. Else, use your commonsense knowledge to generate knowledge. Your response should be in this form:
'Knowledge: {knowledge}'
Remember you cannot directly answer the question as your knowledge.
"""



class Prompter():
    def __init__(self, dataset) -> None:
        self.__dataset = dataset
        self.__example_data = None
        self.__example_index = None
        
        return 
    
    def __get_instruction(self, type) -> str:
        if type == DIERCT_ANSWER:
            return answer_instru_prompt
        elif type == BASE_COT_ANSWER:
            return cot_answer_instru_prompt
        elif type == GENERATE_KNOWLEDGE:
            return generate_knowledge_instru_prompt
        elif type == KNOWLEDGE_ANSWER:
            return knowledge_answer_instru_prompt
        elif type == KNOWLEDGE_COT_ANSWER:
            return knowledge_and_answer_instru_prompt
        elif type == GENERATE_SUP_QUESTION:
            return sup_question_instru_prompt
        elif type == SUP_COT_ANSWER:
            return answer_instru_prompt
        elif type == EXCLUDE_COT_ANSWER:
            return cot_answer_instru_prompt
        elif type == KNOWLEDGE_BASE_COT_ANSWER:
            return knowledge_base_cot_instru_prompt
        elif type == REDUN_COT_ANSWER:
            return redun_cot_instru_prompt
        elif type == OPPO_COT_ANSWER:
            return oppo_cot_instru_prompt
        elif type == CONS_CHECK:
            return consistency_instru_prompt
        elif type == GENERATE_SYLLOGISM_KNOWLEDGE:
            return syllogism_knowledge_instru_prompt
        elif type == SYLLOGISM_KNOWLEDGE_ANSWER:
            return syllogism_answer_instru_prompt
        elif type == GENERATE_ONE_KNOWLEDGE:
            return one_knowledge_instru_prompt
        elif type == GENERATE_EXTRA_KNOWLEDGE:
            return extra_knowledge_instru_prompt
        elif type == CONTINUOUS_KNOWLEDGE_ANSWER:
            return knowledge_answer_instru_prompt
        elif type == KNOWLEDGE_ONE_INFER:
            return knowledge_infer_instru_prompt
        elif type == INFER_ANSWER:
            return infer_answer_instru_prompt
        elif type == GENERATE_SUB_QUESTION:
            return sub_question_instru_prompt
        elif type == SUB_QUESTION_ANSWER:
            return sub_question_answer_instru_prompt
        elif type == GENERATE_QUESTION_GOAL:
            return goal_extraction_instru_prompt
        elif type == GENERATE_BASE_COT:
            return cot_instru_prompt
        elif type == RES_COT_ANSWER:
            return res_cot_answer_instru_prompt
        elif type == EVALUATE_KNOWLEDGE:
            return evaluate_knowledge_instru_prompt
        elif type == EVALUATE_KNOWLEDGE_ANSWER:
            return knowledge_answer_instru_prompt
        elif type == RETRIEVE_KNOWLEDGE:
            return retrieve_knowledge_instru_prompt
        elif type == TRANSFORM_KNOWLEDGE:
            return transform_knowledge_instru_prompt
        elif type == REWRITE_QUESTION:
            return rewrite_question_instru_prompt
        elif type == GENERATE_PYKE:
            return generate_pyke_instru_prompt
        elif type == HINT_GENERATE_KNOWLEDGE:
            return hint_generate_knowledge_instru_prompt
        elif type == HINT_GENERATE_COT:
            return hint_generate_cot_instru_prompt
        elif type == HINT_GENERATE_WRONG_COT:
            return hint_generate_cot_instru_prompt
        elif type == REFLECT_GENERATE_KNOWLEDGE:
            return reflect_generate_knowledge_instru_prompt
        elif type == READER_KNOWLEDGE_ANSWER:
            return reader_knowledge_prompt
        elif type == CLASSIFY_KNOWLEDGE:
            return classifier_prompt
        elif type == ASSESS_QUESTION_KNOWLEDGE:
            return assess_question_knowledge_instru_prompt
        elif type == READER_ANSWER:
            return reader_ft_prompt
        return ""
    
    def __get_assistant(self, type) -> str:
        if type == DIERCT_ANSWER:
            return direct_choice_assistant
        elif type == BASE_COT_ANSWER:
            return cot_answer_assistant
        elif type == GENERATE_KNOWLEDGE:
            return generate_knowledge_assistant
        elif type == KNOWLEDGE_ANSWER:
            return knowledge_choice_assistant
        elif type == KNOWLEDGE_COT_ANSWER:
            return generate_knowledge_answers_assistant
        elif type == GENERATE_SUP_QUESTION:
            return generate_sup_question_assistant
        elif type == SUP_COT_ANSWER:
            return knowledge_choice_assistant
        elif type == EXCLUDE_COT_ANSWER:
            return exclude_cot_assistant
        elif type == KNOWLEDGE_BASE_COT_ANSWER:
            return knowledge_base_cot_assistant
        elif type == REDUN_COT_ANSWER:
            return cot_answer_assistant
        elif type == OPPO_COT_ANSWER:
            return cot_answer_assistant
        elif type == CONS_CHECK:
            return consistency_assistant
        elif type == GENERATE_SYLLOGISM_KNOWLEDGE:
            return generate_knowledge_assistant
        elif type == SYLLOGISM_KNOWLEDGE_ANSWER:
            return knowledge_choice_assistant
        elif type == GENERATE_ONE_KNOWLEDGE:
            return generate_knowledge_assistant
        elif type == GENERATE_EXTRA_KNOWLEDGE:
            return generate_knowledge_assistant
        elif type == CONTINUOUS_KNOWLEDGE_ANSWER:
            return knowledge_choice_assistant
        elif type == KNOWLEDGE_ONE_INFER:
            return knowledge_infer_assistant
        elif type == INFER_ANSWER:
            return infer_answer_assistant
        elif type == GENERATE_SUB_QUESTION:
            return generate_sub_question_assistant
        elif type == SUB_QUESTION_ANSWER:
            return sub_question_answer_assistant
        elif type == GENERATE_QUESTION_GOAL:
            return goal_extraction_assistant
        elif type == GENERATE_BASE_COT:
            return cot_assistant
        elif type == RES_COT_ANSWER:
            return res_cot_answer_assistant
        elif type == EVALUATE_KNOWLEDGE:
            return evaluate_knowledge_assistant
        elif type == EVALUATE_KNOWLEDGE_ANSWER:
            return knowledge_choice_assistant
        elif type == RETRIEVE_KNOWLEDGE:
            return retrieve_knowledge_assistant
        elif type == TRANSFORM_KNOWLEDGE:
            return transform_knowledge_assistant
        elif type == REWRITE_QUESTION:
            return rewrite_question_assistant
        elif type == GENERATE_PYKE:
            return generate_pyke_assistant
        elif type == HINT_GENERATE_KNOWLEDGE:
            return hint_generate_knowledge_assistant
        elif type == HINT_GENERATE_COT:
            return hint_generate_cot_assistant
        elif type == HINT_GENERATE_WRONG_COT:
            return hint_generate_cot_assistant
        elif type == REFLECT_GENERATE_KNOWLEDGE:
            return reflect_generate_knowledge_assistant
        elif type == ASSESS_QUESTION_KNOWLEDGE:
            return assess_question_knowledge_assistant
        elif type == GENERATE_QUERY:
            return query_generate_knowledge_assistant
        elif type == QUERY_ANSWER:
            return query_answer_assistant
        elif type == GENERATE_APP_KNOWLEDGE:
            return generate_app_knowledge_assistant
        return None
    
    def __get_example(self, type, cnt, auto_example=True):
        question_ls = []
        answer_ls = []
        if type == DIERCT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_answers
            elif self.__dataset == 'csqa2':
                question_ls = csqa2_questions
                answer_ls = csqa2_answers
            elif self.__dataset == 'hella':
                question_ls = hella_questions
                answer_ls = hella_answers
            elif self.__dataset == 'piqa':
                question_ls = piqa_questions
                answer_ls = piqa_answers
            elif self.__dataset == 'siqa':
                question_ls = social_questions
                answer_ls = social_answers
        elif type == BASE_COT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_cot_answers
            elif self.__dataset == 'csqa2':
                question_ls = csqa2_questions
                answer_ls = csqa2_cot_answers
            elif self.__dataset == 'hella':
                question_ls = hella_questions
                answer_ls = hella_cot_answers
            elif self.__dataset == 'piqa':
                question_ls = piqa_questions
                answer_ls = piqa_cot_answers
            elif self.__dataset == 'siqa':
                question_ls = social_questions
                answer_ls = social_cot_answers
        elif type == GENERATE_KNOWLEDGE:
            if self.__dataset == 'wino':
                question_ls = wino_question_stems
                answer_ls = wino_knowledge
            elif self.__dataset == 'csqa2':
                question_ls = csqa2_questions
                answer_ls = csqa2_knowledge
            elif self.__dataset == 'hella':
                question_ls = hella_questions
                answer_ls = hella_knowledge
            elif self.__dataset == 'piqa':
                question_ls = piqa_questions
                answer_ls = piqa_knowledge
            elif self.__dataset == 'siqa':
                question_ls = social_questions
                answer_ls = social_knowledge
        elif type == KNOWLEDGE_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_knowledge[i] + "Question: " + wino_questions[i])
                answer_ls = wino_answers
            elif self.__dataset == 'csqa2':
                for i in range(cnt):
                    question_ls.append(csqa2_knowledge[i] + "Question: " + csqa2_questions[i])
                answer_ls = csqa2_answers
            elif self.__dataset == 'hella':
                for i in range(cnt):
                    question_ls.append(hella_knowledge[i] + "\nQuestion: " + hella_questions[i])
                answer_ls = hella_answers
            elif self.__dataset == 'piqa':
                for i in range(cnt):
                    question_ls.append(piqa_knowledge[i] + "\nQuestion: " + piqa_questions[i])
                answer_ls = piqa_answers
            elif self.__dataset == 'siqa':
                for i in range(cnt):
                    question_ls.append(piqa_knowledge[i] + "\nQuestion: " + piqa_questions[i])
                answer_ls = piqa_answers
        elif type == KNOWLEDGE_COT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_knowledge_answers
            elif self.__dataset == 'csqa2':
                question_ls = csqa2_questions
                answer_ls = csqa2_knowledge_answers
        elif type == GENERATE_SUP_QUESTION:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_sup_questions
            elif self.__dataset == 'csqa2':
                for i in range(cnt):
                    question_ls.append(csqa2_knowledge[i] + "Question: " + csqa2_question_stems[i])
                answer_ls = csqa2_sup_questions
        elif type == SUP_COT_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_knowledge[i] + wino_sup_questions[i] + "Question: " + wino_questions[i])
                for i in range(cnt):
                    answer_ls.append(wino_sup_answers[i] + wino_answers[i])
            elif self.__dataset == 'csqa2':
                for i in range(cnt):
                    question_ls.append(csqa2_knowledge[i] + csqa2_sup_questions[i] + "Question: " + csqa2_questions[i])
                for i in range(cnt):
                    answer_ls.append(csqa2_sup_answers[i] + csqa2_answers[i])
        elif type == EXCLUDE_COT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_exclude_cot_answers
            # elif self.type == 'csqa2':
            #     question_ls = csqa2_questions
            #     answer_ls = csqa2_cot_answers
        elif type == KNOWLEDGE_BASE_COT_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_knowledge[i] + "Question: " + wino_questions[i])
                answer_ls = wino_copy_knowledge_answers
            elif self.__dataset == 'csqa2':
                for i in range(cnt):
                    question_ls.append(csqa2_knowledge[i] + "Question: " + csqa2_questions[i])
                answer_ls = csqa2_cot_answers
            elif self.__dataset == 'hella':
                for i in range(cnt):
                    question_ls.append(hella_knowledge[i] + "\nQuestion: " +  hella_questions[i])
                answer_ls = hella_copy_knowledge_answers
        elif type == REDUN_COT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_redun_cot_answers
            # elif self.dataset == 'csqa2':
            #     question_ls = csqa2_questions
            #     answer_ls = csqa2_cot_answers
        elif type == OPPO_COT_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_oppo_cot_answers
            # elif self.dataset == 'csqa2':
            #     question_ls = csqa2_questions
            #     answer_ls = csqa2_cot_answers
        elif type == CONS_CHECK:
            if self.__dataset == 'wino':
                question_ls = wino_consistency_examples
                answer_ls = wino_consistency_judge
        elif type == GENERATE_SYLLOGISM_KNOWLEDGE:
            if self.__dataset == 'wino':
                question_ls = wino_question_stems
                answer_ls = wino_syllogism_knowledge
        elif type == SYLLOGISM_KNOWLEDGE_ANSWER:
             if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Question: " + wino_question_stems[i] + wino_syllogism_knowledge[i])
                answer_ls = wino_answers    
        elif type == GENERATE_ONE_KNOWLEDGE:
            if self.__dataset == 'wino':
                question_ls = wino_question_stems
                answer_ls = wino_one_knowledge
            elif self.__dataset == 'csqa2':
                question_ls = csqa2_question_stems
                answer_ls = csqa2_one_knowledge
        elif type == GENERATE_EXTRA_KNOWLEDGE:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_one_knowledge[i] + "Question: " + wino_question_stems[i])
                answer_ls = wino_extra_knowledge
        elif type == CONTINUOUS_KNOWLEDGE_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_one_knowledge[i] + wino_extra_knowledge[i] + "Question: " + wino_questions[i])
                answer_ls = wino_answers
        elif type == KNOWLEDGE_ONE_INFER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_one_knowledge[i] + "Question: " + wino_question_stems[i])
                answer_ls = wino_one_knowledge_infer
        elif type == INFER_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    infer = wino_knowledge_infer[i].split('\n')[1]
                    question_ls.append("Question: " + wino_questions[i] + infer + ' So the answer is: ')
                answer_ls = wino_answer_options
        elif type == GENERATE_SUB_QUESTION:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_sub_questions
        elif type == SUB_QUESTION_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Main Question: " + wino_questions[i] + wino_sub_questions[i])
                answer_ls = wino_sub_question_answers
        elif type == GENERATE_BASE_COT:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Question: " + wino_questions[i])
                answer_ls = wino_cot_auto
        elif type == GENERATE_QUESTION_GOAL:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_question_goals
        elif type == RES_COT_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_cot[i] + wino_question_goals[i])
                answer_ls = wino_answers
        elif type == EVALUATE_KNOWLEDGE:
            if self.__dataset == 'wino':
                question_ls = wino_evaluate_knowledge_examples
                answer_ls = wino_knowledge_evaluation_examples
        elif type == EVALUATE_KNOWLEDGE_ANSWER:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_one_knowledge[i] + "Question: " + wino_questions[i])
                answer_ls = wino_answers
        elif type == RETRIEVE_KNOWLEDGE:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append(wino_retrieve_knowledge[i] + "Question: " + wino_questions[i])
                answer_ls = wino_retrieve_rates
            elif self.__dataset == 'csqa2':
                for i in range(cnt):
                    question_ls.append(csqa2_rerank_knowledge[i] + "Question: " + csqa2_questions[i])
                answer_ls = csqa2_rerank_rates
            elif self.__dataset == 'hella':
                for i in range(cnt):
                    question_ls.append(hella_rerank_knowledge[i] + "\nQuestion: " + hella_questions[i])
                answer_ls = hella_rerank_rates
        elif type == TRANSFORM_KNOWLEDGE:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    knowledge = wino_knowledge_infer[i].split('\n')[0]
                    infer = wino_knowledge_infer[i].split('\n')[1]
                    question_ls.append("Question: " + wino_question_stems[i] + knowledge)
                    answer_ls.append(infer)      
        elif type == GENERATE_PYKE:
            if self.__dataset == 'wino':
                question_ls = wino_pyke_questions
                answer_ls = wino_pyke_answers
        elif type == HINT_GENERATE_KNOWLEDGE:
            if self.__dataset == 'wino':
                question_ls = wino_hint_statements
                answer_ls = wino_knowledge
        elif type == HINT_GENERATE_COT:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Question: " + wino_questions[i] + "Hint: " + wino_answers[i])
                answer_ls = wino_cot_auto
        elif type == HINT_GENERATE_WRONG_COT:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Question: " + wino_questions[i] + "Hint: " + wino_wrong_answers[i])
                answer_ls = wino_cot_wrong_auto
        elif type == REFLECT_GENERATE_KNOWLEDGE:
            if self.__dataset == 'wino':
                for i in range(cnt):
                    question_ls.append("Question: " + wino_questions[i] + "Wrong Knowledge: " + wino_reflect_knowledge[i].strip('Knowledge:'))
                answer_ls = wino_knowledge
        elif type == REWRITE_QUESTION:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_rewrite_questions
        elif type == GENERATE_QUERY:
            if self.__dataset == 'wino':
                question_ls = wino_questions
                answer_ls = wino_querys
        elif type == QUERY_ANSWER:
            if self.__dataset == 'wino':
                question_ls = wino_querys
                answer_ls = wino_query_answers
        return question_ls[:cnt], answer_ls[:cnt]
    
    
    def init_faiss(self, task):

        if task == GENERATE_APP_KNOWLEDGE:
            data_path = './result/example_pool/wino_true_knowledge.json'
            index_path = './result/example_pool/wino_true_knowledge.index'
        else:
            data_path = './result/example_pool/wino_improve_knowledge.json'
            index_path = './result/example_pool/wino_improve_knowledge.index'
        index = faiss.read_index(index_path)
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.__example_data = data
        self.__example_index = index
        return
             
    def get_related_atomic_knowledge(self, question):
        embed_model = SentenceTransformer('./model/all-mpnet-base-v2')
        with open('./data/atomic2020/knowledge_sentence.json', 'r') as f:
                atomic_data = json.load(f)
        atomic_index = faiss.read_index('./data/atomic2020/knowledge_sentence.index')
        return get_related_knowledge(question, embed_model, atomic_data, atomic_index, top_k=1)[0]

    
    # def get_related_tf_example(self, question):
        
    
    def wrap_input(self, x, task, icl_cnt, options=None):
        assistant = self.__get_assistant(task)
        instruction = self.__get_instruction(task)
        # logger.info(f'Assistant Prompt: {assistant}')
        # logger.info(f'Instruction Prompt: {instruction}')
        messages=[{"role":"system", "content":assistant}]
        if icl_cnt:
            examples = [] 
            if task in [GENERATE_APP_KNOWLEDGE]:
                example_questions, example_answers = zip(*get_related_example(x,self.__embed_model,self.__example_data, self.__example_index, icl_cnt)) 
            else:  
                example_questions, example_answers = self.__get_example(task, icl_cnt)
            examples.append({"role":"user", "content":instruction + example_questions[0]})
            examples.append({"role":"assistant", "content":example_answers[0]}) 
            for i in range(icl_cnt - 1):
                examples.append({"role":"user", "content": example_questions[i+1]})
                examples.append({"role":"assistant", "content":example_answers[i+1]})
            messages.extend(examples)
            messages.append({"role": "user", "content":x}) 
        else:
            if task == READER_ANSWER:
                messages = instruction.format(x)
            elif task == READER_KNOWLEDGE_ANSWER:
                question, knowledge = x.split('[SEP]')
                messages = instruction.format(question, knowledge)
            elif task in [CRITIC_RATE_COT, RERANK_KNOWLEDGE, READER_KNOWLEDGE_ANSWER, ROBERTA_ANSWER]:
                messages = x
            elif task == CLASSIFY_KNOWLEDGE:
                question, knowledge = x.split('[SEP]')
                messages = instruction.format(question, knowledge)
            else:
                x = instruction + x
                messages.append({"role": "user", "content":x}) 
        return messages
    