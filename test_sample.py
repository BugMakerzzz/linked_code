import openai
import os
import spacy
from prompt import *
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

prompter = Prompter('wino')

question = """Question: Sarah was a much better surgeon than Maria so _ always got the easier cases.
(1) Sarah (2) Maria"""

knowledge = """"""

content = question + '\n' + knowledge

messages = prompter.wrap_input(content, KNOWLEDGE_BASE_COT_ANSWER, 5)

completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        n=5
        )  

answers = completion['choices']
for item in answers:
    answer = item['message']['content']
    print(answer)