
import openai
# from embeddings.openai import OpenAIEmbeddings
# import os
# import pandas as pd
# import time
# import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

def extract_entities(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    # Regular expressions to find libraries, modules, classes, functions and variables
    import_pattern = re.compile(r'^\s*import (\w+)', re.MULTILINE)
    from_import_pattern = re.compile(r'^\s*from (\w+) import', re.MULTILINE)
    module_pattern = re.compile(r'^\s*module (\w+):*\<*', re.MULTILINE)
    class_pattern = re.compile(r'^\s*class (\w+):*', re.MULTILINE)
    function_pattern = re.compile(r'^\s*def (\w+)\((.*?)\):*', re.MULTILINE)
    variable_pattern = re.compile(r'^\s*(\w+)\s*=', re.MULTILINE)

    imports = import_pattern.findall(code)
    from_imports = from_import_pattern.findall(code)
    modules = module_pattern.findall(code)
    classes = class_pattern.findall(code)
    functions = function_pattern.findall(code)
    variables = variable_pattern.findall(code)

    functions_with_parameters = {}
    for func_match in functions:
        func_name, param_str = func_match
        param_list = [param.strip() for param in param_str.split(',') if param.strip()]
        functions_with_parameters[func_name] = param_list

    return {'imports': imports, 'from_imports': from_imports, 'modules': modules, 'classes': classes, 'functions': functions, 'functions_with_parameters': functions_with_parameters, 'variables': variables}

def second_ques_creator(module):
    question = ['Generate a module named '+ module[0] + '. ', 'Create a module named '+ module[0]+'.']
    start = random.choice(question)

    return f"{start}"

def third_ques_creator(classes):
    question = ['Generate a class named '+ classes[0] + '. ', 'Create a class named '+ classes[0]+'.']
    start = random.choice(question)

    return f"{start}"

def forth_ques_creator( function):
    question = 'This class will have '+ ''.join(str(len(function))) + ' functions named'

    for func in result['functions_with_parameters']:
        question += " ' "+ func + " ', "
    return f"{question}"

def fifth_ques_creator(func_param):
    question = "Write the fuctions with mentioned parameters: "
    for func, params in result['functions_with_parameters'].items():
        question += str(func) + ":" +str(params)
    return f"{question}"

def sixth_ques_creator(variable):
    question = "Inside the funtion use '"+ variable[0] + "' variable."

    return f"{question}"


def get_completion(prompt, model="gpt-4-1106-preview"):
# def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages)

    return response.choices[0].message["content"]

def create_response(prompt):
    print()
    print("Response from ChatGPT:")
    print(response)
    print("____________________________________________________________________________________________________________________________________")

def remove_comments(code):
    # Regular expression to match both single-line and multi-line comments
    pattern = re.compile(r"(#.*?$|/\*.*?\*/|//.*?$)", re.MULTILINE | re.DOTALL)

    # Remove comments from the code
    code_without_comments = re.sub(pattern, "", code)

    return code_without_comments

def post_processing(response):

    # Find code snippets enclosed within triple backticks
    code_blocks = re.findall(r'```([\s\S]*?)```', response)

    # Concatenate the code snippets
    extracted_code = '\n'.join(code_blocks)

    main_pattern = re.compile(r"if\s+__name__\s*==\s*\"__main__\":\s*([\s\S]*?)(?=\n\S|\Z)")

    # Remove the if __name__ == "__main__" block
    if bool(main_pattern.search(extracted_code)):
      code_without_main_block = re.sub(main_pattern, "", extracted_code)
      post_processed_code = remove_comments(code_without_main_block)

    else:
      post_processed_code = remove_comments(extracted_code)

    return post_processed_code


def similarity_checking(github_f, chatgpt_f):

    # github_file = github_f

    # with open(github_file, "r+") as file:
    #         github_data = file.read().replace('\n', '')
    #         file.write(github_data)
    # #print(github_data)

    github_data = github_f.replace('\n', '')
    chatgpt_data = chatgpt_f.replace('\n', '')

    G = github_data
    C = chatgpt_data

    # tokenization
    G_list = word_tokenize(G)
    C_list = word_tokenize(C)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 =[]
    l2 =[]

    # remove stop words from the string
    G_set = {w for w in G_list if not w in sw}
    C_set = {w for w in C_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = G_set.union(C_set)
    for w in rvector:
      if w in G_set: l1.append(1) # create a vector
      else: l1.append(0)
      if w in C_set: l2.append(1)
      else: l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    cos_percentage = float(cosine * 100)

    return cos_percentage

if __name__ == "__main__":

    print()
    print("Write the topic or project name.")
    user_answer = input("Enter your answer: ")
    topic_name = user_answer
    openai.api_key = "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"  # Sir has provided this key
    # openai.api_key = "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"  # Created by own

    #data = input("Please enter a github text file:")

    if topic_name == 'AdmonitionBlock' or topic_name == 'admonitionBlock' or topic_name == 'admonitionblock' or topic_name == 'Admonitionblock':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\admonitionBlock_github.txt"

    elif topic_name == 'Marlin' or topic_name == 'Test version' or topic_name == 'Test_version':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\test_version_github.txt"

    elif topic_name == 'Marlin' or topic_name == 'marlin_py' or topic_name == 'marlinPy':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\marlin_py_github.txt"

    elif topic_name == 'GanShare' or topic_name == 'ganShare' or topic_name == 'promptClass':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\GanShare_prompt_class_init_method_github.txt"

    elif topic_name == 'icorpus' or topic_name == 'get_lcs_inner' or topic_name == 'Icorpus':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\def_get_lcs_inner_github.txt"

    elif topic_name == 'python from numpy' or topic_name == 'vec2' or topic_name == 'Vec2':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\vec2_github.txt"

    elif topic_name == 'bestMobabot' or topic_name == 'bestmobabot' or topic_name == 'Test database':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\Test_database_github.txt"

    elif topic_name == 'lady blackbird' or topic_name == 'Tools_names' or topic_name == 'tools_names':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\Tools_names_github.txt"

    elif topic_name == 'MachineRay2' or topic_name == 'machineRay2' or topic_name == 'Machine Ray 2':
        github_f = "C:\\Users\\User\\PycharmProjects\\ChatGPT_project_rulebased_approach\\machineRay2_github.txt"


    with open(github_f, "r+") as file:
          github_data = file.read()
          file.write(github_data)
    #print(github_data)



    file_path = github_f
    result = extract_entities(file_path)

    first_prompt = " Write a code for a project named  " + topic_name + " in python."
    print("Prompt to chatgpt:")
    print()
    print(first_prompt)

    response = get_completion(first_prompt)
    create_response(response)

    github_f_r = remove_comments(github_data)
    chatgpt_f = post_processing(response)

    similarity = similarity_checking(github_f_r, chatgpt_f)
    print("similarity: ", similarity, "%")
    print(
        "____________________________________________________________________________________________________________________________________")

    if len(result['modules']) > 0:

        second_prompt = second_ques_creator(result['modules'])
        print("Prompt to chatgpt:")
        print()
        print(second_prompt)
        response = get_completion(second_prompt)
        create_response(response)

        github_f_r = remove_comments(github_data)
        chatgpt_f = post_processing(response)

        similarity = similarity_checking(github_f_r, chatgpt_f)
        print("similarity: ", similarity, "%")
        print(
          "____________________________________________________________________________________________________________________________________")

    if len(result['classes']) > 0:

        third_prompt = third_ques_creator(result['classes'])
        print("Prompt to chatgpt:")
        print()
        print(third_prompt)
        response = get_completion(third_prompt)
        create_response(response)
        github_f_r = remove_comments(github_data)
        chatgpt_f = post_processing(response)

        similarity = similarity_checking(github_f_r, chatgpt_f)
        print("similarity: ", similarity, "%")
        print(
          "____________________________________________________________________________________________________________________________________")

    if len(result['functions']) > 0:
        forth_prompt = forth_ques_creator(result['functions_with_parameters'])
        print("Prompt to chatgpt:")
        print()
        print(forth_prompt)
        response = get_completion(forth_prompt)
        create_response(response)

        github_f_r = remove_comments(github_data)
        chatgpt_f = post_processing(response)

        similarity = similarity_checking(github_f_r, chatgpt_f)
        print("similarity: ", similarity, "%")
        print(
          "____________________________________________________________________________________________________________________________________")

    if len(result['functions']) > 0:
        fifth_prompt = fifth_ques_creator(result['functions_with_parameters'])
        print("Prompt to chatgpt:")
        print()
        print(fifth_prompt)
        response = get_completion(fifth_prompt)
        create_response(response)

        github_f_r = remove_comments(github_data)
        chatgpt_f = post_processing(response)

        similarity = similarity_checking(github_f_r, chatgpt_f)
        print("similarity: ", similarity, "%")
        print(
            "____________________________________________________________________________________________________________________________________")

    if len(result['variables']) > 0:
        sixth_prompt = sixth_ques_creator(result['variables'])
        print("Prompt to chatgpt:")
        print()
        print(sixth_prompt)
        response = get_completion(sixth_prompt)
        create_response(response)

        github_f_r = remove_comments(github_data)
        chatgpt_f = post_processing(response)

        similarity = similarity_checking(github_f_r, chatgpt_f)
        print("similarity: ", similarity, "%")
        print(
            "____________________________________________________________________________________________________________________________________")

