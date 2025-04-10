#!/usr/bin/env python
# coding: utf-8


from huggingface_hub import login
import os
from tqdm import tqdm
import sys
import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.36.2"])
import torch
import pandas as pd

from datasets import load_dataset, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging, LlamaForCausalLM, LlamaTokenizer
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

import numpy as np
from operator import itemgetter
import pickle
from clingo.control import Control
from clingo.symbol import parse_term
import random
import networkx as nx
from itertools import combinations
from itertools import permutations
import re
import string
import bz2
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# decommentare e prendere il token
hugging_token = "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"
login(hugging_token)

torch.cuda.is_available()

torch.cuda.device_count()

torch.manual_seed(56)


class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret


def incipit():
    return "Write an ASP program for the following problem."


def copilot_label_assignments(labels, predicate_name, test):

    labels_set = ''.join([f'"{x}",' for x in labels[:-1]]) + f'"{labels[-1]}"'

    if not test:
        s1 = f'''Develop an ASP program that assigns exactly one label from the specified set {labels_set} to a collection of elements defined by the predicate "{predicate_name}".'''

        s2 = f'''Create an ASP application that maps one label belonging to {labels_set} to a set of elements based on the predicate "{predicate_name}".'''

        s3 = f'''Write an ASP script that associates exactly one label from the set {labels_set} with a group of elements determined by the predicate "{predicate_name}".'''

        s4 = f'''Design an ASP solution that links a single label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s5 = f'''Craft an ASP program that assigns one label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s6 = f'''Implement an ASP code snippet that tags elements with one label from the set {labels_set} according to the predicate "{predicate_name}".'''

        s7 = f'''Build an ASP application that links exactly one label from {labels_set} to a set of elements identified by the predicate "{predicate_name}".'''

        s8 = f'''Write an ASP script that connects a single label from {labels_set} to each element defined by the predicate "{predicate_name}".'''

        s9 = f'''Develop an ASP solution to map one label from {labels_set} to elements as per the predicate "{predicate_name}".'''

        s10 = f'''Create an ASP program that assigns just one label from {labels_set} to a collection of elements determined by the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP solution that links a single label from {labels_set} with elements identified by the predicate "{predicate_name}".'''

        s12 = f'''Compose an ASP script to link only one label from {labels_set} to a group of elements according to the predicate "{predicate_name}".'''

        s13 = f'''Draft an ASP program that maps a single label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP code that attaches one label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s15 = f'''Develop an ASP application that links a single label from {labels_set} with elements as indicated by the predicate "{predicate_name}".'''

        s16 = f'''Create an ASP script that connects one label from {labels_set} to elements based on the predicate "{predicate_name}".'''

        s17 = f'''Build an ASP program that maps a single label from {labels_set} to elements guided by the predicate "{predicate_name}".'''

        s18 = f'''Write an ASP solution that tags elements with a single label from {labels_set} according to the predicate "{predicate_name}".'''

        s19 = f'''Design an ASP script to link exactly one label from {labels_set} to elements under the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP application that assigns exactly one label from {labels_set} to elements determined by the predicate "{predicate_name}".'''
    
        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:

        s1 = f'''Develop an ASP script that ensures each element, as specified by the predicate "{predicate_name}", receives exactly one label from the set {labels_set}.'''

        s2 = f'''Create an ASP solution to assign one specific label from {labels_set} to a group of elements as defined by the predicate "{predicate_name}".'''

        s3 = f'''Write an ASP application that maps a single label from {labels_set} to every element identified by the predicate "{predicate_name}".'''

        s4 = f'''Design an ASP script to connect each element, as determined by the predicate "{predicate_name}", with one label from {labels_set}.'''

        s5 = f'''Craft an ASP solution that associates precisely one label from {labels_set} with elements specified by the predicate "{predicate_name}".'''

        s6 = f'''Implement an ASP application to tag elements, defined by the predicate "{predicate_name}", with one label from the set {labels_set}.'''

        s7 = f'''Build an ASP program that links each element identified by the predicate "{predicate_name}" to a single label from {labels_set}.'''

        s8 = f'''Write an ASP code snippet to connect a single label from {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s9 = f'''Develop an ASP solution to map one specific label from {labels_set} to each element defined by the predicate "{predicate_name}".'''

        s10 = f'''Create an ASP script that assigns a single label from {labels_set} to a group of elements as indicated by the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP program that links each element, as identified by the predicate "{predicate_name}", with one label from {labels_set}.'''

        s12 = f'''Compose an ASP application that assigns one label from {labels_set} to every element defined by the predicate "{predicate_name}".'''

        s13 = f'''Draft an ASP code that connects a single label from the set {labels_set} to elements specified by the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP solution that links one label from {labels_set} with each element identified by the predicate "{predicate_name}".'''

        s15 = f'''Develop an ASP application to assign one label from {labels_set} to elements defined by the predicate "{predicate_name}".'''

        s16 = f'''Create an ASP script that maps a single label from {labels_set} to a collection of elements specified by the predicate "{predicate_name}".'''

        s17 = f'''Build an ASP code snippet to link one label from {labels_set} to elements identified by the predicate "{predicate_name}".'''

        s18 = f'''Write an ASP solution to connect each element defined by the predicate "{predicate_name}" with a single label from {labels_set}.'''

        s19 = f'''Design an ASP application to assign one label from {labels_set} to every element specified by the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP program that maps a single label from the set {labels_set} to elements determined by the predicate "{predicate_name}".'''

        s21 = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {labels_set}.'''


        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def label_assignment(labels, predicate_name, prompt_invariance, test):

    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}.'''

    if(prompt_invariance):
        rewritten_questions = copilot_label_assignments(labels_to_assign, predicate_name, test)

    answer = ""
    for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).'''

    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f.append(f"{predicate_name}(1..5).")  
    
    return questions, answers, f


def copilot_prevent_value(predicate_name, value, label, test):

    if not test:
        s1 = f'''Create an ASP program that ensures the predicate "{predicate_name}" with a value of {value} is not assigned to the label "{label}".'''

        s2 = f'''Write an ASP solution that prohibits the assignment of the predicate "{predicate_name}" with a value of {value} to the label "{label}".'''

        s3 = f'''Develop an ASP program that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s4 = f'''Craft an ASP script that prevents the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s5 = f'''Implement an ASP application that avoids assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s6 = f'''Design an ASP solution that excludes the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s7 = f'''Build an ASP program that ensures the predicate "{predicate_name}" having value {value} cannot be assigned to the label "{label}".'''

        s8 = f'''Create an ASP code snippet that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s9 = f'''Write an ASP program that prevents the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s10 = f'''Develop an ASP solution that avoids assigning the predicate "{predicate_name}" having value {value} to the label "{label}".'''

        s11 = f'''Formulate an ASP program that ensures the predicate "{predicate_name}" with a value of {value} is not linked to the label "{label}".'''

        s12 = f'''Compose an ASP script that prohibits the predicate "{predicate_name}" with a value of {value} from being assigned to the label "{label}".'''

        s13 = f'''Draft an ASP solution that disallows linking the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s14 = f'''Generate an ASP application that prevents the predicate "{predicate_name}" having value {value} from being mapped to the label "{label}".'''

        s15 = f'''Develop an ASP script that avoids assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s16 = f'''Create an ASP solution that excludes the predicate "{predicate_name}" with a value of {value} from the label "{label}".'''

        s17 = f'''Build an ASP program that disallows associating the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s18 = f'''Write an ASP application that ensures the predicate "{predicate_name}" with value {value} is not assigned to the label "{label}".'''

        s19 = f'''Design an ASP script that prohibits the assignment of the predicate "{predicate_name}" with value {value} to the label "{label}".'''
        
        s20 = f'''Create an ASP program that prevents the predicate "{predicate_name}" with a value of {value} from being linked to the label "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Develop an ASP application that avoids the predicate "{predicate_name}" with a value of {value} being linked to the label "{label}".'''

        s2 = f'''Compose an ASP solution to ensure the predicate "{predicate_name}" with value {value} is not associated with the label "{label}".'''

        s3 = f'''Create an ASP script that excludes the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s4 = f'''Generate an ASP application to prevent linking the predicate "{predicate_name}" with a value of {value} to the label "{label}".'''

        s5 = f'''Draft an ASP program to disallow assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s6 = f'''Formulate an ASP code that ensures the predicate "{predicate_name}" having value {value} is not connected to the label "{label}".'''

        s7 = f'''Produce an ASP program that prevents associating the predicate "{predicate_name}" with value {value} with the label "{label}".'''

        s8 = f'''Build an ASP solution that disallows the predicate "{predicate_name}" having value {value} from being assigned to the label "{label}".'''

        s9 = f'''Craft an ASP application to avoid mapping the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s10 = f'''Create an ASP code snippet to ensure the predicate "{predicate_name}" with a value of {value} is not linked to the label "{label}".'''

        s11 = f'''Write an ASP script that prevents the predicate "{predicate_name}" with value {value} from being assigned to the label "{label}".'''

        s12 = f'''Develop an ASP application to disallow connecting the predicate "{predicate_name}" having value {value} with the label "{label}".'''

        s13 = f'''Compose an ASP solution that avoids the predicate "{predicate_name}" with value {value} being mapped to the label "{label}".'''

        s14 = f'''Generate an ASP code to exclude linking the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s15 = f'''Formulate an ASP script to ensure the predicate "{predicate_name}" having value {value} is not associated with the label "{label}".'''

        s16 = f'''Design an ASP application that prohibits assigning the predicate "{predicate_name}" with value {value} to the label "{label}".'''

        s17 = f'''Produce an ASP solution that disallows the predicate "{predicate_name}" with value {value} from being mapped to the label "{label}".'''

        s18 = f'''Create an ASP script to avoid associating the predicate "{predicate_name}" having value {value} with the "{label}" label.'''

        s19 = f'''Draft an ASP program to prevent the predicate "{predicate_name}" with value {value} from being linked to the label "{label}".'''

        s20 = f'''Write an ASP application that excludes the predicate "{predicate_name}" with value {value} from being assigned to the label "{label}".'''

        s21 = f'''{incipit()} Prevent the predicate "{predicate_name}" with value "{value}" from having label "{label}".'''
    
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def prevent_value(labels, predicate_name, prompt_invariance, test):
    
    f = []
    fact = ''

    n_values = 20

    questions, answers = [], []
    rewritten_questions = []

    value = np.random.randint(1, n_values)

    label = labels[np.random.randint(0, len(labels))]
    question = f'''{incipit()} Prevent the predicate "{predicate_name}" with value "{value}" from having label "{label}".'''
    
    if(prompt_invariance):
        rewritten_questions = copilot_prevent_value(predicate_name, value, label, test)

    answer = f''':-assign({value},{label}).'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    fact += f'''{predicate_name}(1..{n_values}).'''
    for label in labels[:-1]:
        fact += f'''assign(X,"{label}")|'''

    fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''

    f.append(fact)    

    return questions, answers, f


def copilot_generate_combinations(predicate_name_1, predicate_name_2, test):

    if not test:
        s1 = f'''Develop an ASP program that computes all possible combinations of elements from two sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s2 = f'''Write an ASP solution that generates the cross-product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s3 = f'''Create an ASP program that produces all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s4 = f'''Design an ASP script that calculates the Cartesian product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s5 = f'''Implement an ASP application that finds all combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s6 = f'''Craft an ASP solution that enumerates every possible pairing of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s7 = f'''Build an ASP program that lists all valid combinations of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s8 = f'''Create an ASP code snippet that computes the cross-product of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s9 = f'''Write an ASP program that generates all valid pairings of elements expressed by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s10 = f'''Develop an ASP solution that calculates the Cartesian product of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s11 = f'''Compose an ASP program that determines all possible combinations of elements from two sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s12 = f'''Generate an ASP solution that produces the cross-product of elements between the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s13 = f'''Create an ASP script that forms all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s14 = f'''Design an ASP program that computes the Cartesian product of elements from the sets represented by "{predicate_name_1}" and "{predicate_name_2}".'''

        s15 = f'''Implement an ASP solution that finds combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s16 = f'''Craft an ASP script that enumerates all possible pairings of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s17 = f'''Develop an ASP code that lists valid combinations of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s18 = f'''Write an ASP snippet that computes the cross-product of elements in the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s19 = f'''Generate an ASP application that creates all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s20 = f'''Compose an ASP solution to calculate the Cartesian product of elements in the sets represented by "{predicate_name_1}" and "{predicate_name_2}".'''
    
    
        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Design an ASP solution to compute all possible pairings of elements from two sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s2 = f'''Craft an ASP program to generate the cross-product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s3 = f'''Develop an ASP code snippet to produce all valid combinations of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s4 = f'''Compose an ASP script to calculate the Cartesian product of elements represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s5 = f'''Write an ASP application that finds all pairings of elements from the sets defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s6 = f'''Formulate an ASP program that enumerates every possible combination of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s7 = f'''Create an ASP solution to list all valid pairings of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s8 = f'''Generate an ASP code to compute the cross-product of elements in the sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s9 = f'''Develop an ASP script to produce all valid pairings of elements as defined by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s10 = f'''Craft an ASP application that calculates the Cartesian product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s11 = f'''Write an ASP program that determines all possible combinations of elements from sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s12 = f'''Compose an ASP script that generates the cross-product of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s13 = f'''Formulate an ASP code snippet to form all valid pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s14 = f'''Create an ASP program to calculate the Cartesian product of elements from sets represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s15 = f'''Develop an ASP solution that finds all pairings of elements from the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s16 = f'''Generate an ASP script to enumerate all possible pairings of elements from the sets "{predicate_name_1}" and "{predicate_name_2}".'''

        s17 = f'''Craft an ASP application to list valid combinations of elements between the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s18 = f'''Write an ASP program that computes the cross-product of elements in the sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s19 = f'''Produce an ASP script to generate all valid pairings of elements as represented by the predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s20 = f'''Create an ASP solution to calculate the Cartesian product of elements from sets defined by "{predicate_name_1}" and "{predicate_name_2}".'''

        s21 = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def generate_combinations(predicate_name_1, predicate_name_2, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}".'''
    
    if(prompt_invariance):
        rewritten_questions = copilot_generate_combinations(predicate_name_1, predicate_name_2, test)
    
    answer = f"combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y)."
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)
    
    f = f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).'''

    return questions, answers, f


def copilot_select_value(predicate_name, label, test):

    
    if not test:
        s1 = f'''Create an ASP program that retrieves all values associated with the predicate "{predicate_name}" labeled as "{label}".'''

        s2 = f'''Write an ASP program to extract values linked to the predicate "{predicate_name}" with the label "{label}".'''

        s3 = f'''Develop an ASP solution that identifies all values related to the label "{label}" within the predicate "{predicate_name}".'''

        s4 = f'''Craft an ASP program that collects data associated with the label "{label}" for the predicate "{predicate_name}".'''

        s5 = f'''Construct an ASP script to fetch values corresponding to the label "{label}" within the predicate "{predicate_name}".'''

        s6 = f'''Generate an ASP code snippet that retrieves all relevant values for the label "{label}" in the context of the predicate "{predicate_name}".'''

        s7 = f'''Produce an ASP implementation that selects values tied to the label "{label}" under the predicate "{predicate_name}".'''

        s8 = f'''Write an ASP script to obtain all values labeled as "{label}" within the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP program to capture values associated with the label "{label}" in the context of the predicate "{predicate_name}".'''

        s10 = f'''Compose an ASP solution that identifies and retrieves values labeled "{label}" under the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP script to gather values related to the label "{label}" within the predicate "{predicate_name}".'''

        s12 = f'''Create an ASP program that extracts values linked to the "{predicate_name}" predicate and labeled as "{label}".'''

        s13 = f'''Develop an ASP script to collect data associated with the "{predicate_name}" predicate and the label "{label}".'''

        s14 = f'''Design an ASP application that retrieves values associated with the label "{label}" within the predicate "{predicate_name}".'''

        s15 = f'''Write an ASP code snippet to fetch values linked to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s16 = f'''Generate an ASP program that identifies all values associated with the label "{label}" within the pred icate "{predicate_name}".'''        
        
        s17 = f'''Craft an ASP application to gather all values tied to the label "{label}" under the predicate "{predicate_name}".'''

        s18 = f'''Produce an ASP script that extracts values related to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s19 = f'''Develop an ASP program that retrieves data associated with the label "{label}" within the predicate "{predicate_name}".'''

        s20 = f'''Create an ASP solution to capture values labeled as "{label}" within the predicate "{predicate_name}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Formulate an ASP application to fetch all values tied to the predicate "{predicate_name}" and labeled as "{label}".'''

        s2 = f'''Draft an ASP code to retrieve values associated with the predicate "{predicate_name}" and the label "{label}".'''

        s3 = f'''Generate an ASP script that identifies all values within the predicate "{predicate_name}" that are linked to the label "{label}".'''

        s4 = f'''Compose an ASP solution to gather data from the predicate "{predicate_name}" associated with the label "{label}".'''

        s5 = f'''Develop an ASP program to select values tied to the label "{label}" within the predicate "{predicate_name}".'''

        s6 = f'''Craft an ASP code snippet to capture all relevant values for the label "{label}" within the predicate "{predicate_name}".'''

        s7 = f'''Write an ASP script to collect values associated with the label "{label}" from the predicate "{predicate_name}".'''

        s8 = f'''Create an ASP solution that retrieves all values labeled "{label}" within the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP application to fetch values tied to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s10 = f'''Produce an ASP program to gather and retrieve values linked to the label "{label}" in the predicate "{predicate_name}".'''

        s11 = f'''Formulate an ASP script that extracts values related to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s12 = f'''Write an ASP application to collect values linked to the predicate "{predicate_name}" and labeled as "{label}".'''

        s13 = f'''Develop an ASP solution that gathers data associated with the label "{label}" within the predicate "{predicate_name}".'''

        s14 = f'''Generate an ASP code snippet to capture values related to the label "{label}" in the predicate "{predicate_name}".'''

        s15 = f'''Compose an ASP program to identify values labeled as "{label}" within the predicate "{predicate_name}".'''

        s16 = f'''Craft an ASP application to fetch all values linked to the label "{label}" in the context of the predicate "{predicate_name}".'''

        s17 = f'''Design an ASP program to gather values tied to the label "{label}" within the context of the predicate "{predicate_name}".'''

        s18 = f'''Create an ASP code to retrieve values associated with the label "{label}" within the predicate "{predicate_name}".'''

        s19 = f'''Develop an ASP script to capture all values linked to the label "{label}" within the predicate "{predicate_name}".'''

        s20 = f'''Write an ASP solution to collect values tied to the predicate "{predicate_name}" and labeled as "{label}".'''

        s21 = f'''{incipit()} Select all values associated to the predicate "{predicate_name}" with label "{label}".'''

        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def select_value(predicate_name, label, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Select all values associated to the predicate "{predicate_name}" with label "{label}".'''

    if(prompt_invariance):
        rewritten_questions = copilot_select_value(predicate_name, label, test)

    answer = f'''select(X):-{predicate_name}(X,"{label}").'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)
    
    f = f'''{predicate_name}(1..5, "{label}").'''

    return questions, answers, f


def copilot_execute_join(predicate_name_1, predicate_name_2, a, b, random_attribute, test):

    
    if not test:
        s1 = f'''Write an ASP program for the following problem. Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each {predicate_name_1} the {random_attribute} of {predicate_name_2}.'''

        s2 = f'''Develop an ASP program for the problem described. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s3 = f'''Create an ASP program for the following task. The predicate "{predicate_name_1}" has fields {a}, and the predicate "{predicate_name_2}" has fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s4 = f'''Construct an ASP program to solve the given problem. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s5 = f'''Draft an ASP program for the problem at hand. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that matches each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s6 = f'''Formulate an ASP program for the following scenario. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s7 = f'''Generate an ASP program for this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s8 = f'''Compose an ASP program to address the given issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s9 = f'''Write an ASP program for this challenge. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s10 = f'''Devise an ASP program for the described problem. Use the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s11 = f'''Create an ASP program to solve this issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s12 = f'''Form an ASP program for the specified problem. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s13 = f'''Write an ASP program to tackle the problem. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s14 = f'''Design an ASP program for this task. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s15 = f'''Develop an ASP program to address the issue. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s16 = f'''Compose an ASP program for this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s17 = f'''Write an ASP program to solve this problem. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s18 = f'''Draft an ASP program to address the given challenge. Consider the predicate "{predicate_name_1}" with fields {a} and the predicate "{predicate_name_2}" with fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}.'''

        s19 = f'''Generate an ASP program for the described task. The predicate "{predicate_name_1}" includes fields {a}, and the predicate "{predicate_name_2}" contains fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s20 = f'''Formulate an ASP program to address this problem. Consider the predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Create an ASP script to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given that "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s2 = f'''Write an ASP application to address the problem where the predicate "{predicate_name_1}" has fields {a}, and the predicate "{predicate_name_2}" has fields {b}. Define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}.'''

        s3 = f'''Develop an ASP solution that defines the predicate "{predicate_name_1}_{predicate_name_2}" to link each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s4 = f'''Compose an ASP code snippet to define the predicate "{predicate_name_1}_{predicate_name_2}" linking each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s5 = f'''Craft an ASP solution that addresses the problem of defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given that "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s6 = f'''Generate an ASP program to create the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s7 = f'''Design an ASP application to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s8 = f'''Formulate an ASP program that defines the predicate "{predicate_name_1}_{predicate_name_2}" to associate each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and {b} of "{predicate_name_2}".'''

        s9 = f'''Compose an ASP script that addresses the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}", with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s10 = f'''Create an ASP solution to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s11 = f'''Write an ASP program to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which associates each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s12 = f'''Develop an ASP solution to create the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s13 = f'''Draft an ASP script to define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s14 = f'''Generate an ASP program to address the problem of defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s15 = f'''Craft an ASP solution to define the predicate "{predicate_name_1}_{predicate_name_2}" that associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using the fields {a} of "{predicate_name_1}" and the fields {b} of "{predicate_name_2}".'''

        s16 = f'''Formulate an ASP program to create the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s17 = f'''Design an ASP application to solve the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, given "{predicate_name_1}" has fields {a} and "{predicate_name_2}" has fields {b}.'''

        s18 = f'''Create an ASP program to define the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, using fields {a} for "{predicate_name_1}" and fields {b} for "{predicate_name_2}".'''

        s19 = f'''Compose an ASP script to address the problem by defining the predicate "{predicate_name_1}_{predicate_name_2}" which associates each {predicate_name_1} with the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''

        s20 = f'''Develop an ASP program to solve the problem by creating the predicate "{predicate_name_1}_{predicate_name_2}" that links each {predicate_name_1} to the {random_attribute} of {predicate_name_2}, with "{predicate_name_1}" having fields {a} and "{predicate_name_2}" having fields {b}.'''
        
        s21 = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each "{predicate_name_1}" the "{random_attribute}" of "{predicate_name_2}".'''
 
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])
        
    return s

def execute_join(predicate_name_1, predicate_name_2, attributes, prompt_invariance, test):
    questions, answers = [], []
    rewritten_questions = []

    f = []

    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):

            fact = ''

            n_attributes = attributes_1
            attributes = np.array(attributes, dtype='U18')
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            chosen_attributes[0] = f"ID"
            chosen_attributes[random_pos] = f"{predicate_name_2}ID"

            string_chosen_attributes = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes += f'"{chosen_attributes[-1]}"'
            fact += f'''{predicate_name_1}({string_chosen_attributes}).'''

            a = ''
            for attr in chosen_attributes[:-1]:
                a += f'"{attr}",'
            a += f'"{chosen_attributes[-1]}"'

            p = f"{predicate_name_1}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    p += "X"
                elif i == random_pos:
                    p += "Y"
                else:
                    p += "_"

                p += ","

            if random_pos == len(chosen_attributes) - 1:
                p += "Y)"
            else:
                p += "_)"

            n_attributes = attributes_2
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            chosen_attributes[0] = "ID"

            string_chosen_attributes_2 = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes_2 += f'"{chosen_attributes[-1]}"'
            fact += f'''{predicate_name_2}({string_chosen_attributes_2}).'''

            random_attribute_index = np.random.randint(1, n_attributes)
            random_attribute = chosen_attributes[random_attribute_index]

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each "{predicate_name_1}" the "{random_attribute}" of "{predicate_name_2}".'''
            
            if(prompt_invariance):
                rewritten_questions = copilot_execute_join(predicate_name_1, predicate_name_2, a, b, random_attribute, test)
        
            q = f"{predicate_name_2}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    q += "Y"
                elif i == random_attribute_index:
                    q += "Z"
                else:
                    q += "_"

                q += ","

            if random_attribute_index == len(chosen_attributes) - 1:
                q += "Z)"
            else:
                q += "_)"

            answer = f'''{predicate_name_1}_{predicate_name_2}(X,Z):-{p},{q}.'''
            rewritten_answers = np.repeat(answer, len(rewritten_questions))

            rewritten_facts = np.repeat(fact, len(rewritten_questions))

            if(len(rewritten_questions)>0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
                f.extend(rewritten_facts)
            else:
                questions.append(question)
                answers.append(answer)
                f.append(fact)

    return questions, answers, f


def copilot_transitive_closure(closure_name, predicate_name, test):

    
    if not test:
        s1 = f'''Create an ASP program that establishes the transitive closure of the predicate "{predicate_name}", defined as predicate "{closure_name}".'''

        s2 = f'''Write an ASP program that computes the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s3 = f'''Develop an ASP solution that derives the predicate "{closure_name}" by extending the transitive closure of the predicate "{predicate_name}".'''

        s4 = f'''Craft an ASP program that constructs the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s5 = f'''Construct an ASP script that defines the predicate "{closure_name}" as the transitive closure of the given predicate "{predicate_name}".'''

        s6 = f'''Generate an ASP code snippet that establishes the transitive closure "{closure_name}" of the predicate "{predicate_name}".'''

        s7 = f'''Produce an ASP implementation that infers the predicate "{closure_name}" using the transitive closure of the predicate"{predicate_name}".'''

        s8 = f'''Write an ASP rule that computes the predicate "{closure_name}" by extending the reachability of the predicate "{predicate_name}".'''

        s9 = f'''Design an ASP program that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''

        s10 = f'''Compose an ASP solution that defines the predicate "{closure_name}" based on the transitive connections inferred from the predicate "{predicate_name}".'''

        s11 = f'''Create an ASP program that calculates the transitive closure of the predicate "{predicate_name}", defining it as "{closure_name}".'''

        s12 = f'''Write an ASP solution to compute the transitive closure of the predicate "{predicate_name}", resulting in the definition of the predicate "{closure_name}".'''

        s13 = f'''Develop an ASP code that extends the transitive closure of the predicate "{predicate_name}" to form the predicate "{closure_name}".'''

        s14 = f'''Craft an ASP program to construct the predicate "{closure_name}" based on the transitive closure derived from the predicate "{predicate_name}".'''

        s15 = f'''Implement an ASP script that establishes the transitive closure of the predicate "{predicate_name}" and defines it as "{closure_name}".'''

        s16 = f'''Generate an ASP code snippet that links the predicate "{predicate_name}" to its transitive closure, defined as "{closure_name}".'''

        s17 = f'''Produce an ASP program that computes the transitive closure of the predicate "{predicate_name}" and infers the predicate "{closure_name}".'''

        s18 = f'''Write an ASP rule to compute the predicate "{closure_name}" by determining the transitive closure of the predicate "{predicate_name}".'''

        s19 = f'''Design an ASP program to establish the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s20 = f'''Compose an ASP solution that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''


        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Write an ASP application to compute the transitive closure of the predicate "{predicate_name}", resulting in the definition of the predicate "{closure_name}".'''

        s2 = f'''Compose an ASP solution that calculates the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s3 = f'''Develop an ASP script that derives the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s4 = f'''Generate an ASP program to construct the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s5 = f'''Create an ASP solution that establishes the transitive closure of the predicate "{predicate_name}", defined as "{closure_name}".'''

        s6 = f'''Formulate an ASP code snippet to establish the predicate "{closure_name}" by computing the transitive closure of the predicate "{predicate_name}".'''

        s7 = f'''Design an ASP program that infers the predicate "{closure_name}" using the transitive closure of the predicate "{predicate_name}".'''

        s8 = f'''Craft an ASP solution to compute the predicate "{closure_name}" by extending the transitive closure of the predicate "{predicate_name}".'''

        s9 = f'''Produce an ASP script that links the predicate "{closure_name}" to the transitive closure of the predicate "{predicate_name}".'''

        s10 = f'''Write an ASP application that defines the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s11 = f'''Create an ASP code snippet to determine the transitive closure of the predicate "{predicate_name}", resulting in the predicate "{closure_name}".'''

        s12 = f'''Generate an ASP solution that computes the transitive closure of the predicate "{predicate_name}", defining the predicate "{closure_name}".'''        
        
        s13 = f'''Compose an ASP script to extend the transitive closure of the predicate "{predicate_name}" and form the "{closure_name}".'''

        s14 = f'''Develop an ASP application that constructs the predicate "{closure_name}" based on the transitive closure of the predicate "{predicate_name}".'''

        s15 = f'''Formulate an ASP solution to establish the transitive closure of the predicate "{predicate_name}", defined as "{closure_name}".'''

        s16 = f'''Design an ASP code to link the predicate "{predicate_name}" to its transitive closure, defined as "{closure_name}".'''

        s17 = f'''Craft an ASP script that infers the predicate "{closure_name}" by computing the transitive closure of the predicate "{predicate_name}".'''

        s18 = f'''Produce an ASP program to compute the transitive closure of the predicate "{predicate_name}" and define it as "{closure_name}".'''

        s19 = f'''Create an ASP solution that establishes the predicate "{closure_name}" through the transitive closure of the predicate "{predicate_name}".'''

        s20 = f'''Develop an ASP script to link the predicate "{predicate_name}" to its transitive closure, resulting in the predicate "{closure_name}".'''

        s21 = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}".'''
        
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def transitive_closure(closure_name, predicate_name, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    question = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}".'''

    if(prompt_invariance):
        rewritten_questions = copilot_transitive_closure(closure_name, predicate_name, test)
    
    answer = f'''{closure_name}(X,Y):-{predicate_name}(X,Y).\n{closure_name}(X,Y):-{predicate_name}(X,Z),{closure_name}(Z,Y).'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)
    
    f = f'''{predicate_name}(1..3, 1..4).'''

    return questions, answers, f


def copilot_preferences(predicate_name, label, value, cost_value, cost_level, test):

    if not test:
        s1 = f'''Create an ASP program to ensure that the predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s2 = f'''Write an ASP program to avoid linking the predicate "{predicate_name}" with value "{value}" to "{label}". If such a link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s3 = f'''Develop an ASP solution to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s4 = f'''Craft an ASP program to ensure the predicate "{predicate_name}" with value "{value}" does not associate with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s5 = f'''Formulate an ASP script to make sure the predicate "{predicate_name}" with value "{value}" is not linked with "{label}". If such a connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s6 = f'''Generate an ASP code to ensure that the predicate "{predicate_name}" with value "{value}" remains unlinked to "{label}". Any occurrence of this link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s7 = f'''Create an ASP implementation to make sure the predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If it is, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s8 = f'''Write an ASP rule to avoid the predicate "{predicate_name}" with value "{value}" being linked to "{label}". If such an association exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s9 = f'''Design an ASP program to keep the predicate "{predicate_name}" with value "{value}" unlinked from "{label}". If they are associated, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s10 = f'''Compose an ASP solution to ensure the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". If this link is established, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s11 = f'''Formulate an ASP program to ensure the predicate "{predicate_name}" with value "{value}" does not match with "{label}". If this match occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s12 = f'''Create an ASP script to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}". If such a connection is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s13 = f'''Develop an ASP solution to ensure the predicate "{predicate_name}" with value "{value}" is not connected with "{label}". If this connection exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s14 = f'''Craft an ASP program to ensure the predicate "{predicate_name}" with value "{value}" does not associate with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s15 = f'''Generate an ASP code snippet to ensure the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". If this association is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s16 = f'''Compose an ASP solution to avoid the predicate "{predicate_name}" with value "{value}" being linked to "{label}". If such an association exists, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s17 = f'''Build an ASP program to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}". If they are connected, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s18 = f'''Develop an ASP script to ensure the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". If this link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s19 = f'''Create an ASP program to ensure the predicate "{predicate_name}" with value "{value}" remains unlinked to "{label}". If this link occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s20 = f'''Formulate an ASP solution to ensure the predicate "{predicate_name}" with value "{value}" is not connected to "{label}". If this connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Design an ASP solution to prevent the predicate "{predicate_name}" with value "{value}" from being linked to "{label}". If this occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s2 = f'''Craft an ASP program to ensure that the predicate "{predicate_name}" with value "{value}" is not associated with "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if it does.'''

        s3 = f'''Develop an ASP code snippet to avoid linking the predicate "{predicate_name}" with value "{value}" to "{label}". If such a link is found, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s4 = f'''Write an ASP program that disallows the association between "{predicate_name}" with value "{value}" and "{label}", with a cost of "{cost_value}" at level "{cost_level}" if this association occurs.'''

        s5 = f'''Generate an ASP application to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if associated.'''

        s6 = f'''Compose an ASP script to ensure the predicate "{predicate_name}" with value "{value}" does not link to "{label}". If this connection happens, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s7 = f'''Formulate an ASP solution to prevent the association of the predicate "{predicate_name}" with value "{value}" with "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s8 = f'''Create an ASP program that keeps the predicate "{predicate_name}" with value "{value}" unlinked from "{label}". If linked, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s9 = f'''Develop an ASP application to avoid the predicate "{predicate_name}" with value "{value}" being associated with "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if found.'''

        s10 = f'''Generate an ASP script to ensure the predicate "{predicate_name}" with value "{value}" is not linked to "{label}". Any occurrence incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s11 = f'''Draft an ASP solution to make sure the predicate "{predicate_name}" with value "{value}" is not connected to "{label}". If connected, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s12 = f'''Write an ASP application that avoids the predicate "{predicate_name}" with value "{value}" from being linked to "{label}", incurring a cost of "{cost_value}" at level "{cost_level}" if linked.'''

        s13 = f'''Compose an ASP program to keep the predicate "{predicate_name}" with value "{value}" separate from "{label}". If this association occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s14 = f'''Craft an ASP solution to prevent the linking of the predicate "{predicate_name}" with value "{value}" to "{label}". Any link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s15 = f'''Create an ASP code to ensure that the predicate "{predicate_name}" with value "{value}" does not associate with "{label}". If it does, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s16 = f'''Formulate an ASP application to avoid the predicate "{predicate_name}" with value "{value}" being linked to "{label}". If linked, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s17 = f'''Generate an ASP program to disallow the association of the predicate "{predicate_name}" with value "{value}" with "{label}". If associated, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s18 = f'''Develop an ASP script to keep the predicate "{predicate_name}" with value "{value}" unlinked from "{label}". Any occurrence incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s19 = f'''Compose an ASP solution to prevent the linking of the predicate "{predicate_name}" with value "{value}" to "{label}". Any link incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s20 = f'''Craft an ASP application to avoid the predicate "{predicate_name}" with value "{value}" from being associated with "{label}". If this occurs, it incurs a cost of "{cost_value}" at level "{cost_level}".'''

        s21 = f'''{incipit()} I would prefer that predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this occurs, it costs "{cost_value}" at level "{cost_level}".'''
            
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def preferences(predicate_name, labels, prompt_invariance, test):
    questions, answers, f = [], [], []
    rewritten_questions = []
    n_values = 20

    for cost_value in range(1, 3):
        for cost_level in range(1, 3):
            value = np.random.randint(1, n_values)

            label = labels[np.random.randint(0, len(labels))]
            question = f'''{incipit()} I would prefer that predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this occurs, it costs "{cost_value}" at level "{cost_level}".'''
            
            if(prompt_invariance):
                rewritten_questions = copilot_preferences(predicate_name, label, value, cost_value, cost_level, test)
            
            answer = f''':~assign({value},"{label}").[{cost_value}@{cost_level}]'''
            rewritten_answers = np.repeat(answer, len(rewritten_questions))

            fact = f'''{predicate_name}(1..{n_values}).'''

            for label in labels[:-1]:
                fact += f'''assign(X,"{label}")|'''
            fact += f'''assign(X,"{labels[-1]}"):-{predicate_name}(X).'''



            if(len(rewritten_questions)>0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
                f.extend(fact)
            else:
                questions.append(question)
                answers.append(answer)
                f.append(fact)

    return questions, answers, f


def copilot_minimizing(predicate_name, label, test):

    if not test:
        s1 = f'''Write an ASP program to minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

        s2 = f'''Create an ASP program to reduce the number of predicate "{predicate_name}" associated with label "{label}".'''

        s3 = f'''Compose an ASP program to limit the quantity of predicate "{predicate_name}" labeled with "{label}".'''

        s4 = f'''Draft an ASP program to decrease the instances of predicate "{predicate_name}" tagged as "{label}".'''

        s5 = f'''Generate an ASP program to cut down on the number of predicate "{predicate_name}" marked with label "{label}".'''

        s6 = f'''Produce an ASP program to lower the count of predicate "{predicate_name}" connected to label "{label}".'''

        s7 = f'''Write an ASP program to lessen the amount of predicate "{predicate_name}" designated with label "{label}".'''

        s8 = f'''Create an ASP program to diminish the frequency of predicate "{predicate_name}" tagged with "{label}".'''

        s9 = f'''Design an ASP program to shrink the number of predicate "{predicate_name}" that has the label "{label}".'''

        s10 = f'''Formulate an ASP program to reduce the occurrence of predicate "{predicate_name}" associated with label "{label}".'''

        s11 = f'''Develop an ASP script to minimize the number of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to reduce the frequency of "{predicate_name}" predicates tagged with "{label}".'''

        s13 = f'''Write an ASP code snippet to limit the instances of "{predicate_name}" predicates associated with "{label}".'''

        s14 = f'''Compose an ASP application to cut down the quantity of "{predicate_name}" predicates labeled "{label}".'''

        s15 = f'''Draft an ASP solution to decrease the number of "{predicate_name}" predicates marked with label "{label}".'''

        s16 = f'''Generate an ASP script to lower the count of "{predicate_name}" predicates connected to label "{label}".'''

        s17 = f'''Produce an ASP application to lessen the amount of "{predicate_name}" predicates designated with "{label}".'''

        s18 = f'''Write an ASP program to diminish the frequency of "{predicate_name}" predicates tagged as "{label}".'''

        s19 = f'''Create an ASP script to shrink the number of "{predicate_name}" predicates that have the label "{label}".'''

        s20 = f'''Formulate an ASP solution to reduce the occurrence of "{predicate_name}" predicates associated with label "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Write an ASP program to minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

        s2 = f'''Create an ASP program to reduce the number of predicate "{predicate_name}" associated with label "{label}".'''

        s3 = f'''Compose an ASP program to limit the quantity of predicate "{predicate_name}" labeled with "{label}".'''

        s4 = f'''Draft an ASP program to decrease the instances of predicate "{predicate_name}" tagged as "{label}".'''

        s5 = f'''Generate an ASP program to cut down on the number of predicate "{predicate_name}" marked with label "{label}".'''

        s6 = f'''Produce an ASP program to lower the count of predicate "{predicate_name}" connected to label "{label}".'''

        s7 = f'''Write an ASP program to lessen the amount of predicate "{predicate_name}" designated with label "{label}".'''

        s8 = f'''Create an ASP program to diminish the frequency of predicate "{predicate_name}" tagged with "{label}".'''

        s9 = f'''Design an ASP program to shrink the number of predicate "{predicate_name}" that has the label "{label}".'''

        s10 = f'''Formulate an ASP program to reduce the occurrence of predicate "{predicate_name}" associated with label "{label}".'''

        s11 = f'''Develop an ASP script to minimize the number of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to reduce the frequency of "{predicate_name}" predicates tagged with "{label}".'''

        s13 = f'''Write an ASP code snippet to limit the instances of "{predicate_name}" predicates associated with "{label}".'''

        s14 = f'''Compose an ASP application to cut down the quantity of "{predicate_name}" predicates labeled "{label}".'''

        s15 = f'''Draft an ASP solution to decrease the number of "{predicate_name}" predicates marked with label "{label}".'''

        s16 = f'''Generate an ASP script to lower the count of "{predicate_name}" predicates connected to label "{label}".'''

        s17 = f'''Produce an ASP application to lessen the amount of "{predicate_name}" predicates designated with "{label}".'''

        s18 = f'''Write an ASP program to diminish the frequency of "{predicate_name}" predicates tagged as "{label}".'''

        s19 = f'''Create an ASP script to shrink the number of "{predicate_name}" predicates that have the label "{label}".'''

        s20 = f'''Formulate an ASP solution to reduce the occurrence of "{predicate_name}" predicates associated with label "{label}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])

    return s

def minimizing(predicate_name, labels, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Minimize the number of predicate "{predicate_name}" tagged with label "{label}".'''

    if(prompt_invariance):
        rewritten_questions = copilot_minimizing(predicate_name, label, test)

    answer = f''':~{predicate_name}(X),assign(X,"{label}").[1@1,X]'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)
    
    # USELESS FOR NOW  
    f = []

    return questions, answers, f


def copilot_maximizing(predicate_name, label, test):

    if not test:
        s1 = f'''Write an ASP program to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s2 = f'''Create an ASP program to increase the frequency of "{predicate_name}" predicates with the label "{label}".'''

        s3 = f'''Compose an ASP program to boost the number of "{predicate_name}" predicates tagged with "{label}".'''

        s4 = f'''Draft an ASP program to enhance the occurrence of "{predicate_name}" predicates identified with the label "{label}".'''

        s5 = f'''Generate an ASP program to amplify the instances of "{predicate_name}" predicates marked as "{label}".'''

        s6 = f'''Produce an ASP program to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s7 = f'''Write an ASP program to heighten the occurrence of "{predicate_name}" predicates labeled "{label}".'''

        s8 = f'''Create an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s9 = f'''Design an ASP program to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s10 = f'''Formulate an ASP program to maximize the frequency of "{predicate_name}" predicates identified with "{label}".'''

        s11 = f'''Develop an ASP script to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to increase the number of "{predicate_name}" predicates with the label "{label}".'''

        s13 = f'''Write an ASP code snippet to boost the instances of "{predicate_name}" predicates tagged with "{label}".'''

        s14 = f'''Compose an ASP application to enhance the number of "{predicate_name}" predicates identified with the label "{label}".'''

        s15 = f'''Draft an ASP solution to amplify the frequency of "{predicate_name}" predicates marked as "{label}".'''

        s16 = f'''Generate an ASP script to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s17 = f'''Produce an ASP application to heighten the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s18 = f'''Write an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s19 = f'''Create an ASP script to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s20 = f'''Formulate an ASP solution to maximize the instances of "{predicate_name}" predicates identified with "{label}".'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Write an ASP program to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s2 = f'''Create an ASP program to increase the frequency of "{predicate_name}" predicates with the label "{label}".'''

        s3 = f'''Compose an ASP program to boost the number of "{predicate_name}" predicates tagged with "{label}".'''

        s4 = f'''Draft an ASP program to enhance the occurrence of "{predicate_name}" predicates identified with the label "{label}".'''

        s5 = f'''Generate an ASP program to amplify the instances of "{predicate_name}" predicates marked as "{label}".'''

        s6 = f'''Produce an ASP program to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s7 = f'''Write an ASP program to heighten the occurrence of "{predicate_name}" predicates labeled "{label}".'''

        s8 = f'''Create an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s9 = f'''Design an ASP program to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s10 = f'''Formulate an ASP program to maximize the frequency of "{predicate_name}" predicates identified with "{label}".'''

        s11 = f'''Develop an ASP script to maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s12 = f'''Create an ASP solution to increase the number of "{predicate_name}" predicates with the label "{label}".'''

        s13 = f'''Write an ASP code snippet to boost the instances of "{predicate_name}" predicates tagged with "{label}".'''

        s14 = f'''Compose an ASP application to enhance the number of "{predicate_name}" predicates identified with the label "{label}".'''

        s15 = f'''Draft an ASP solution to amplify the frequency of "{predicate_name}" predicates marked as "{label}".'''

        s16 = f'''Generate an ASP script to elevate the count of "{predicate_name}" predicates associated with "{label}".'''

        s17 = f'''Produce an ASP application to heighten the occurrence of "{predicate_name}" predicates labeled as "{label}".'''

        s18 = f'''Write an ASP program to raise the number of "{predicate_name}" predicates tagged with "{label}".'''

        s19 = f'''Create an ASP script to increase the presence of "{predicate_name}" predicates with the label "{label}".'''

        s20 = f'''Formulate an ASP solution to maximize the instances of "{predicate_name}" predicates identified with "{label}".'''

        s = []

        for i in range(1, 21):
            s.append(vars()['s' + str(i)])

    return s

def maximizing(predicate_name, labels, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]

    question = f'''{incipit()} Maximize the occurrence of "{predicate_name}" predicates labeled as "{label}".'''
    
    if(prompt_invariance):
        rewritten_questions = copilot_maximizing(predicate_name, label, test)

    answer = f''':~{predicate_name}(X),not assign(X,"{label}").[1@1,X]'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f = []

    return questions, answers, f


def copilot_select_by_negative_condition(predicate_name, not_predicate_name, label, test):

    if not test:
        s1 = f'''Write an ASP program to select all values associated with the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and label "{label}".'''

        s2 = f'''Create an ASP program to fetch all values linked to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s3 = f'''Compose an ASP program to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and the label "{label}".'''

        s4 = f'''Draft an ASP program to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and having the label "{label}".'''

        s5 = f'''Generate an ASP program to retrieve all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s6 = f'''Produce an ASP program to gather all values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and the label "{label}".'''

        s7 = f'''Write an ASP script to select all values linked to the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s8 = f'''Create an ASP program to choose all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and the label "{label}".'''

        s9 = f'''Design an ASP program to find all values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and carrying the label "{label}".'''

        s10 = f'''Formulate an ASP program to gather all values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s11 = f'''Develop an ASP script to collect all values related to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and the label "{label}".'''

        s12 = f'''Create an ASP solution to identify all values linked to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s13 = f'''Compose an ASP code to find all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and carrying the label "{label}".'''

        s14 = f'''Draft an ASP application to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s15 = f'''Generate an ASP solution to retrieve all values associated with the predicate "{predicate_name}" but not with the predicate "{not_predicate_name}" and having the label "{label}".'''

        s16 = f'''Write an ASP program to fetch all values linked to the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s17 = f'''Create an ASP script to gather all values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and the label "{label}".'''

        s18 = f'''Design an ASP program to capture all values related to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s19 = f'''Compose an ASP code snippet to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s20 = f'''Generate an ASP script to select all values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and having the label "{label}".'''


        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Write an ASP script to select all values tied to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled as "{label}".'''

        s2 = f'''Create an ASP application to fetch values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s3 = f'''Compose an ASP solution to identify all values connected to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s4 = f'''Draft an ASP program to retrieve values tied to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s5 = f'''Generate an ASP script to gather values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s6 = f'''Produce an ASP code snippet to collect values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s7 = f'''Write an ASP application to select values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s8 = f'''Create an ASP solution to fetch values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s9 = f'''Design an ASP program to identify values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s10 = f'''Formulate an ASP code to gather values associated with the predicate "{predicate_name}" but not connected to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s11 = f'''Develop an ASP script to collect values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s12 = f'''Create an ASP program to capture values associated with the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s13 = f'''Compose an ASP application to find values connected to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s14 = f'''Draft an ASP solution to identify values associated with the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s15 = f'''Generate an ASP code snippet to retrieve values linked to the predicate "{predicate_name}" but not to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s16 = f'''Produce an ASP program to gather values associated with the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s17 = f'''Write an ASP script to select values connected to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s18 = f'''Create an ASP application to collect values tied to the predicate "{predicate_name}" but not linked to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s19 = f'''Design an ASP solution to capture values associated with the predicate "{predicate_name}" but not tied to the predicate "{not_predicate_name}" and labeled "{label}".'''

        s20 = f'''Formulate an ASP code to select values linked to the predicate "{predicate_name}" but not associated with the predicate "{not_predicate_name}" and labeled "{label}".'''

        s21 = f'''{incipit()} Select all values associated with predicate "{predicate_name}" but not associated with predicate "{not_predicate_name}" and label "{label}".''' 
    
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def select_by_negative_condition(predicate_name, not_predicate_name, labels, prompt_invariance, test):

    questions, answers = [], []
    rewritten_questions = []

    label = labels[np.random.randint(0, len(labels))]
    
    question = f'''{incipit()} Select all values associated with predicate "{predicate_name}" but not associated with predicate "{not_predicate_name}" and label "{label}".''' 
    
    if(prompt_invariance):
        rewritten_questions = copilot_select_by_negative_condition(predicate_name, not_predicate_name, label, test)

    answer = f'''select(X):-{predicate_name}(X),not {not_predicate_name}(X,"{label}").'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    chosen_labels = list(set(list(np.random.choice(labels, size=4, replace=False))).union({label}))
    combinations = list(zip(range(1, 4), chosen_labels))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    fact = f'''{predicate_name}(1..3).'''

    for i, l in combinations:
        fact += f'''{not_predicate_name}({i},"{l}").'''
    
    f = fact

    return questions, answers, f


def copilot_select_by_numeric_condition(predicate_name, condition, condition_value, test):

    if not test:
        s1 = f'''Write an ASP program to select all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s2 = f'''Create an ASP program to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s3 = f'''Compose an ASP program to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s4 = f'''Draft an ASP program to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s5 = f'''Generate an ASP program to retrieve all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s6 = f'''Produce an ASP program to gather all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s7 = f'''Write an ASP script to select all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s8 = f'''Create an ASP program to choose all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s9 = f'''Design an ASP program to find all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s10 = f'''Formulate an ASP program to gather all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s11 = f'''Develop an ASP script to select all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s12 = f'''Create an ASP solution to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s13 = f'''Compose an ASP code to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s14 = f'''Draft an ASP application to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s15 = f'''Generate an ASP solution to retrieve all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s16 = f'''Write an ASP program to fetch all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s17 = f'''Create an ASP script to gather all values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s18 = f'''Design an ASP program to capture all values related to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s19 = f'''Compose an ASP code snippet to identify all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s20 = f'''Generate an ASP script to select all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s = []
        for i in range(1, 21):
            s.append(vars()['s' + str(i)])
    else:
        
        s1 = f'''Create an ASP application to fetch all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s2 = f'''Write an ASP solution to select values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s3 = f'''Develop an ASP program to gather all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s4 = f'''Formulate an ASP script to identify values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s5 = f'''Craft an ASP code to retrieve values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s6 = f'''Generate an ASP application to select all values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s7 = f'''Compose an ASP program to fetch values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s8 = f'''Design an ASP solution to capture all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s9 = f'''Draft an ASP code snippet to identify values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s10 = f'''Produce an ASP script to retrieve values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s11 = f'''Create an ASP application to select values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s12 = f'''Formulate an ASP solution to gather all values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s13 = f'''Craft an ASP program to fetch values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s14 = f'''Generate an ASP code to capture values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s15 = f'''Develop an ASP application to retrieve all values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s16 = f'''Compose an ASP script to select values linked to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s17 = f'''Write an ASP solution to identify values tied to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s18 = f'''Design an ASP program to gather values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s19 = f'''Formulate an ASP application to fetch values connected to the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s20 = f'''Craft an ASP code snippet to select values associated with the predicate "{predicate_name}" with a value {condition} than {condition_value}.'''

        s21 = f'''{incipit()} Select all values associated with predicate "{predicate_name}" with a value {condition} than {condition_value}.''' 
        
        s = []

        for i in range(1, 22):
            s.append(vars()['s' + str(i)])

    return s

def select_by_numeric_condition(predicate_name, prompt_invariance, test):
    # condition \in [!=, <, >, <=, >=]

    n_values = 100

    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}

    questions, answers = [], []
    rewritten_questions = []

    for condition, condition_symbol in condition_dict.items():
        condition_value = np.random.randint(1, n_values)
    
        question = f'''{incipit()} Select all values associated with predicate "{predicate_name}" with a value {condition} than {condition_value}.''' 
        
        if(prompt_invariance):
            rewritten_questions = copilot_select_by_numeric_condition(predicate_name, condition, condition_value, test)
        
        answer = f'''select(X):-{predicate_name}(X,C),C{condition_symbol}{condition_value}.'''
        rewritten_answers = np.repeat(answer, len(rewritten_questions))

        if(len(rewritten_questions)>0):
            questions.extend(rewritten_questions)
            answers.extend(rewritten_answers)
        else:
            questions.append(question)
            answers.append(answer)
        

    f = f'''{predicate_name}(1..3, 1..{n_values}).'''
    
    return questions, answers, f



#####           COMPLEX         #####

def join_numeric_filtering(predicate_name_1, predicate_name_2, attributes):       
    
    n_values = 100

    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}
    
    questions, answers, f = [], [], []    
    
    #####       parte join
    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):
            for condition, condition_symbol in condition_dict.items():

                condition_value = np.random.randint(1, n_values)
                fact = ''

                n_attributes = attributes_1
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                chosen_attributes[0] = f"ID"
                chosen_attributes[random_pos] = f"{predicate_name_2}ID"

                string_chosen_attributes = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
                string_chosen_attributes += f'"{chosen_attributes[-1]}"'
                
                chosen_labels = np.random.choice(attributes, size=n_attributes, replace=False)

                if(random_pos==1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos-1)]}"'
                    fact += f'''{predicate_name_1}(0..3, 0..2,{string_chosen_labels}).'''
                    
                elif(random_pos>1):
                    if(random_pos<n_attributes-1):
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos-1]])}'''
                        string_chosen_labels += f'"{chosen_labels[-(random_pos-1)]}"'
                        fact += f'''{predicate_name_1}(0..3,{string_chosen_labels},'''
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos+1:-1]])}'''
                        string_chosen_labels += f'"{chosen_labels[-1]}"'
                        fact += f'''0..2, {string_chosen_labels}).'''
                    elif(random_pos==n_attributes-1):
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-1]])}'''
                        fact += f'''{predicate_name_1}(0..3,{string_chosen_labels}0..2).'''
                

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                p = f"{predicate_name_1}("
                for i in range(len(chosen_attributes) - 1):
                    if i == 0:
                        p += "X"
                    elif i == random_pos:
                        p += "Y"
                    else:
                        p += "_"

                    p += ","

                if random_pos == len(chosen_attributes) - 1:
                    p += "Y)"
                else:
                    p += "_)"

                n_attributes = attributes_2
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                random_attribute_index = np.random.randint(1, n_attributes)
                random_attribute = chosen_attributes[random_attribute_index]
                
                temp = chosen_attributes[random_attribute_index]
                chosen_attributes[random_attribute_index] = f'''1..{n_values}'''

                string_chosen_attributes_2 = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
                string_chosen_attributes_2 += f'"{chosen_attributes[-1]}"'
                
                chosen_attributes[random_attribute_index] = temp

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each "{predicate_name_1}" the "{random_attribute}" of "{predicate_name_2}". In addition, select all values associated to the predicate {predicate_name_1}_{predicate_name_2} where "{random_attribute}" is {condition} than {condition_value}.'''
            
                q = f"{predicate_name_2}("
                for i in range(len(chosen_attributes) - 1):
                    if i == 0:
                        q += "Y"
                    elif i == random_attribute_index:
                        q += "Z"
                    else:
                        q += "_"

                    q += ","

                if random_attribute_index == len(chosen_attributes) - 1:
                    q += "Z)"
                else:
                    q += "_)"

                answer = f'''{predicate_name_1}_{predicate_name_2}(X,Z):-{p},{q}.\nselect(X):-{predicate_name_1}_{predicate_name_2}(X,Z),Z{condition_symbol}{condition_value}.'''

                chosen_labels = np.random.choice(attributes, size=n_attributes, replace=False)


                if(random_attribute_index==1):
                    if(n_attributes==2):
                        fact += f'''{predicate_name_2}(0..2,1..100).'''
                    else:
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[2:-1]])}'''
                        string_chosen_labels += f'"{chosen_labels[-1]}"'
                        fact += f'''{predicate_name_2}(0..2,1..100,{string_chosen_labels}).'''
                    
                elif(random_attribute_index>1):
                    if(random_attribute_index<n_attributes-1):
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_attribute_index-1]])}'''
                        string_chosen_labels += f'"{chosen_labels[-(random_attribute_index-1)]}"'
                        fact += f'''{predicate_name_2}(0..2,{string_chosen_labels},1..100,'''
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_attribute_index+1:-1]])}'''
                        string_chosen_labels += f'"{chosen_labels[-1]}"'
                        fact += f'''{string_chosen_labels}).'''
                    elif(random_attribute_index==n_attributes-1):
                        string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                        string_chosen_labels += f'"{chosen_labels[-2]}"'
                        fact += f'''{predicate_name_2}(0..2,{string_chosen_labels},1..100).'''
    

                questions.append(question)
                answers.append(answer)
                f.append(fact)
    
    return questions, answers, f

def join_filtering(predicate_name_1, predicate_name_2, attributes, predicates):       #### verificata con Erica
    questions, answers = [], []
    rewritten_questions = []

    f = []

    for attributes_1 in range(3, 6):
        for attributes_2 in range(2, 5):

            fact = ''

            n_attributes = attributes_1
            attributes = np.array(attributes, dtype='U18')
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            chosen_attributes[0] = f"ID"
            chosen_attributes[random_pos] = f"{predicate_name_2}ID"

            string_chosen_attributes = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes += f'"{chosen_attributes[-1]}"'

            chosen_labels = np.random.choice(predicates, size=n_attributes, replace=False)

            if(random_pos==1):
                string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos-1]])}'''
                string_chosen_labels += f'"{chosen_labels[-(random_pos-1)]}"'
                fact += f'''{predicate_name_1}(0..3, 0..4,{string_chosen_labels}).'''
                
            elif(random_pos>1):
                if(random_pos<n_attributes-1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos-1)]}"'
                    fact += f'''{predicate_name_1}(0..3,{string_chosen_labels},'''
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos+1:-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-1]}"'
                    fact += f'''0..4, {string_chosen_labels}).'''
                elif(random_pos==n_attributes-1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-1]])}'''
                    fact += f'''{predicate_name_1}(0..3,{string_chosen_labels}0..4).'''
            
            
            a = ''
            for attr in chosen_attributes[:-1]:
                a += f'"{attr}",'
            a += f'"{chosen_attributes[-1]}"'

            p = f"{predicate_name_1}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    p += "X"
                elif i == random_pos:
                    p += "Y"
                else:
                    p += "_"

                p += ","

            if random_pos == len(chosen_attributes) - 1:
                p += "Y)"
            else:
                p += "_)"


            n_attributes = attributes_2
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos_2 = np.random.randint(1, attributes_2)
            chosen_attributes[0] = "ID"

            string_chosen_attributes_2 = f'''{''.join([f'"{x}",' for x in chosen_attributes[:-1]])}'''
            string_chosen_attributes_2 += f'"{chosen_attributes[-1]}"'
            
            chosen_labels = np.random.choice(predicates, size=n_attributes+1, replace=False)
            not_label = chosen_labels[-1]
            
            if(random_pos_2==1):
                if(n_attributes==2):
                    fact += f'''{predicate_name_2}(0..2,"{chosen_labels[1]}").'''
                    fact += f'''{predicate_name_2}(2..4,"{not_label}").'''
                else:
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(0..2,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''
                
            elif(random_pos_2>1):
                if(random_pos_2<n_attributes-1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos_2-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos_2-1)]}"'
                    fact += f'''{predicate_name_2}(0..3,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:random_pos_2-1]])}'''
                    string_chosen_labels += f'"{chosen_labels[-(random_pos_2-1)]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos_2+1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''0..4, {string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[random_pos_2+1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''
                elif(random_pos_2==n_attributes-1):
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(0..2,{string_chosen_labels}).'''
                    chosen_labels[random_pos_2] = not_label
                    string_chosen_labels = f'''{''.join([f'"{x}",' for x in chosen_labels[1:-2]])}'''
                    string_chosen_labels += f'"{chosen_labels[-2]}"'
                    fact += f'''{predicate_name_2}(2..4,{string_chosen_labels}).'''

            random_attribute_index = random_pos_2
            random_attribute = chosen_attributes[random_attribute_index]

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            question = f'''{incipit()} Consider predicate "{predicate_name_1}" having fields {a}, and the predicate "{predicate_name_2}" having fields {b}. Define a predicate "{predicate_name_1}_{predicate_name_2}" that associates to each {predicate_name_1} the attribute {random_attribute} of {predicate_name_2}. In addition, select all values associated to the predicate "{predicate_name_1}_{predicate_name_2}" with label "{not_label}".'''
            
            q = f"{predicate_name_2}("
            for i in range(len(chosen_attributes) - 1):
                if i == 0:
                    q += "Y"
                elif i == random_attribute_index:
                    q += "Z"
                else:
                    q += "_"

                q += ","

            if random_attribute_index == len(chosen_attributes) - 1:
                q += "Z)"
            else:
                q += "_)"

            answer = f'''{predicate_name_1}_{predicate_name_2}(X,Z):-{p},{q}.\nselect(X):-{predicate_name_1}_{predicate_name_2}(X,"{not_label}").'''
            rewritten_answers = np.repeat(answer, len(rewritten_questions))

            rewritten_facts = np.repeat(fact, len(rewritten_questions))

            if(len(rewritten_questions)>0):
                questions.extend(rewritten_questions)
                answers.extend(rewritten_answers)
                f.extend(rewritten_facts)
            else:
                questions.append(question)
                answers.append(answer)
                f.append(fact)

    return questions, answers, f

def closure_guessing(labels, predicate_name, closure_name):
    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"

    question = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}". Then, assign exactly one label from the set {','.join([f"{x}" for x in labels_to_assign])} to each element in "{closure_name}".'''

    answer = f'''{closure_name}(X,Y):-{predicate_name}(X,Y).\n{closure_name}(X,Y):-{predicate_name}(X,Z),{closure_name}(Z,Y).\n'''
    for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{closure_name}(X,_).'''

    
    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
    else:
        questions.append(question)
        answers.append(answer)
    
    f = [f'''{predicate_name}(1..3, 1..4).''']

    return questions, answers, f

def closure_negative_filtering(labels, predicate_name, closure_name):
    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)

    question = f'''{incipit()} Define predicate "{closure_name}" as the transitive closure of predicate "{predicate_name}". Then, assign exactly one label from the set {','.join([f"{x}" for x in labels_to_assign])} to each element in "{closure_name}".'''

    answer = f'''{closure_name}(X,Y):-{predicate_name}(X,Y).\n{closure_name}(X,Y):-{predicate_name}(X,Z),{closure_name}(Z,Y).\n'''
    for label in labels_to_assign[:-1]:
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{closure_name}(X,_).'''

    
    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
    else:
        questions.append(question)
        answers.append(answer)
    
    f = [f'''{predicate_name}(1..3, 1..4).''']

    return questions, answers, f

def guessing_constraint(labels, predicate_name):       #### verificata con Erica
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_values=20

    value = np.random.randint(1, n_values)

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    notlabel = np.random.choice(labels_to_assign)
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}. Then prevent the predicate "{predicate_name}" with value "{value}" from having label "{notlabel}".'''

    answer = ""
    for label in labels_to_assign[:-1]:
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).\n:-assign({value}, "{notlabel}").'''

    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f.append(f"{predicate_name}(1..{n_values}).")  
    
    return questions, answers, f

def guessing_preference(labels, predicate_name):
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10
    n_values = 20

    for cost_value in range(1, 3):
        for cost_level in range(1, 3):
            value = np.random.randint(1, n_values)

        n_labels = np.random.randint(2, n_max)
        labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
        label = np.random.choice(labels_to_assign)
        
        question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])} but I would prefer that predicate "{predicate_name}" with value "{value}" is not associated with "{label}". If this occurs, it costs "{cost_value}" at level "{cost_level}".'''

        answer = ""
        for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
            answer += f'''assign(X,"{label}")|'''
        answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).\n:~assign({value}, {label}).[{cost_value}@{cost_level}])'''

        rewritten_answers = np.repeat(answer, len(rewritten_questions))

        if(len(rewritten_questions)>0):
            questions.extend(rewritten_questions)
            answers.extend(rewritten_answers)
        else:
            questions.append(question)
            answers.append(answer)

        f.append(f"{predicate_name}(1..{n_values}).")  
    
    return questions, answers, f

def guessing_negative_filtering(labels, predicate_name):
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    labels_to_avoid = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    notlabel = np.random.choice(labels_to_avoid)
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}. Consider only the predicate {predicate_name} not associated with label "{notlabel}".'''

    answer = ""
    for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X, _), not {predicate_name}(X, {notlabel}).'''

    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f.append(f"{predicate_name}(1..5, 1..5).")  
    
    return questions, answers, f

def guessing_numeric_filtering(labels, predicate_name, attribute1, attribute2):
    f = []

    n_values = 100

    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    for condition, condition_symbol in condition_dict.items():

        condition_value = np.random.randint(1, n_values)
             
        n_labels = np.random.randint(2, n_max)
        labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
        
        question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements expressed by predicate {predicate_name} and labels {attribute1}, {attribute2}, having label {attribute2} {condition} than {condition_value}. The labels are {','.join([f"{x}" for x in labels_to_assign])}.'''

        answer = ""
        for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
            answer += f'''assign(X,"{label}")|'''
        answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X, Y), Y{condition_symbol}{condition_value} .'''

        rewritten_answers = np.repeat(answer, len(rewritten_questions))

        if(len(rewritten_questions)>0):
            questions.extend(rewritten_questions)
            answers.extend(rewritten_answers)
        else:
            questions.append(question)
            answers.append(answer)

    f.append(f"{predicate_name}(1..5, {n_values}).")  
        
    return questions, answers, f

def guessing_filtering(labels, predicate_name):
    f = []

    questions, answers = [], []
    rewritten_questions = []

    n_max = 10

    n_labels = np.random.randint(2, n_max)
    labels_to_assign = np.random.choice(labels, size=n_labels, replace=False)       # dalla raccolta di etichette ne sceglie "size" e senza rimpiazzo   Crea quindi un array di dimensione size scegliendo casualmente gli elementi da "labels"
    othlabel = np.random.choice(labels_to_assign)
    question = f'''{incipit()} Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate {predicate_name}. The labels are {','.join([f"{x}" for x in labels_to_assign])}.  Then, filter and return only the elements assigned to label {othlabel}.".'''

    answer = ""
    for label in labels_to_assign[:-1]:     #si ferma al penultimo elemento perché l'ultimo verrà messo manualmente sotto
        answer += f'''assign(X,"{label}")|'''
    answer += f'''assign(X,"{labels_to_assign[-1]}"):-{predicate_name}(X).\nselect(X):-assign(X, "{othlabel}").'''

    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
    else:
        questions.append(question)
        answers.append(answer)

    f.append(f"{predicate_name}(1..5).")  
    
    return questions, answers, f

def combination_negative_filtering(labels, predicate_name_1, predicate_name_2, predicate_name_3):       #### verificata con Erica
    
    f = []
    questions, answers = [], []
    rewritten_questions = []

    some_labels = np.random.choice(labels, size=3, replace=False)
    label = some_labels[-1]

    question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}". In addition, select all values associated with predicate combination but not associated with predicate "{predicate_name_3}" and label "{label}".'''
    
    answer = f'''combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y).\nselect(X):-combination(X,_), not {predicate_name_3}(X, "{label}").'''
    rewritten_answers = np.repeat(answer, len(rewritten_questions))

    fact = f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).{predicate_name_3}(0..1,"{label}").'''
    for l in some_labels[:-1]:
        fact += f'''{predicate_name_3}(2..3,"{l}").'''

    if(len(rewritten_questions)>0):
        questions.extend(rewritten_questions)
        answers.extend(rewritten_answers)
        f.extend(fact)
    else:
        questions.append(question)
        answers.append(answer)
        f.append(fact)
    
    

    return questions, answers, f

def combination_numeric_filtering(labels, predicate_name_1, predicate_name_2):       
    
    condition_dict = {"different": "!=", "greater": ">", "lower": "<", "greater or equal": ">=", "lower or equal": "<="}

    questions, answers = [], []
    rewritten_questions = []

    some_labels = np.random.choice(labels, size=3, replace=False)
    label = some_labels[-1]

    n_values = 100

    for condition, condition_symbol in condition_dict.items():

        condition_value = np.random.randint(1, n_values)
    
        question = f'''{incipit()} Generate all the combinations of elements from two sets. The two sets are represented by predicates "{predicate_name_1}" and "{predicate_name_2}". In addition, select all values associated with predicate combination with a value {condition} than {condition_value}.'''
        
        answer = f'''combination(X,Y):-{predicate_name_1}(X),{predicate_name_2}(Y).\nselect(X):-combination(X,Y), Y{condition_symbol}{condition_value}.'''
        rewritten_answers = np.repeat(answer, len(rewritten_questions))

        if(len(rewritten_questions)>0):
            questions.extend(rewritten_questions)
            answers.extend(rewritten_answers)
        else:
            questions.append(question)
            answers.append(answer)
        
        f = f'''{predicate_name_1}(1..4).{predicate_name_2}(1..5).'''

    return questions, answers, f

######   potrei aggiungere guessing e preference ovvero assign... [1@2]


def generate_subproblems(size, train_size, validation, print_proportions=True):

    colors = ["red", "green", "blue", "yellow", "brown", "orange", "purple", "gray", "cyan"]
    cities = ["rome", "paris", "venice", "new york", "london", "amsterdam", "dubai", "tokyo", "shangai", "florence"]
    labels = ["color", "person", "tree", "car", "moto", "bike", "table", "food", "element", "street", "object"]
    attributes = ["price", "name", "city", "age", "author", "creator", "shape", "height", "description"]

    predicates = colors + cities + labels + attributes
    closures = ["path", "flights", "ancestors", "destinations", "arrivals"]
    
    questions = []
    answers = []
    facts = []

    # CHANGE 1. -> EXCLUDING MINIMAZION AND MAXIMIZATION, AND SET 5 (INSTEAD OF 10) IN RANGE OF PREVENT

    for i in tqdm(range(size), total=size):
        if not validation:
            np.random.seed(i)
        else:
            np.random.seed(train_size + i)

        match turn:
            case "core":
                for _ in range(10):
                    question_assignments, answer_assignments, f = label_assignment(predicates, np.random.choice(predicates), False, False)
                    questions.extend(question_assignments)
                    answers.extend(answer_assignments)
                    facts.extend(f)

                n_questions_assignment = len(questions)
                
                for _ in range(5):
                    question_prevents, answer_prevents, f = prevent_value(predicates, np.random.choice(predicates), False, False)
                    questions.extend(question_prevents)
                    answers.extend(answer_prevents)
                    facts.extend(f)

                n_questions_prevent = len(questions) - n_questions_assignment
                
                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, f = generate_combinations(p_1, p_2, False, False)

                questions.extend(questions_combinations)
                answers.extend(answers_combinations)
                facts.extend(f)

                questions_select, answers_select, f = select_value(np.random.choice(predicates), np.random.choice(predicates), False, False)

                questions.extend(questions_select)
                answers.extend(answers_select)
                facts.extend(f)

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, f = execute_join(p_1, p_2, attributes, False, False)

                questions.extend(questions_join)
                answers.extend(answers_join)
                facts.extend(f)

                questions_closure, answers_closure, f = transitive_closure(np.random.choice(closures), np.random.choice(predicates), False, False)

                questions.extend(questions_closure)
                answers.extend(answers_closure)
                facts.extend(f)

                questions_preferences, answers_preferences, f = preferences(np.random.choice(predicates), predicates, False, False)

                questions.extend(questions_preferences)
                answers.extend(answers_preferences)
                facts.extend(f)

                # questions_minimizing, answers_minimizing, f = minimizing(np.random.choice(predicates), predicates, False, False)

                # questions.extend(questions_minimizing)
                # answers.extend(answers_minimizing)
                # facts.extend(f)

                # questions_maximizing, answers_maximizing, f = maximizing(np.random.choice(predicates), predicates, False, False)

                # questions.extend(questions_maximizing)
                # answers.extend(answers_maximizing)
                # facts.extend(f)

                questions_negative, answers_negative, f = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, False, False)

                questions.extend(questions_negative)
                answers.extend(answers_negative)
                facts.extend(f)

                questions_numeric_condition, answers_numeric_condition, f = select_by_numeric_condition(np.random.choice(predicates), False, False)

                questions.extend(questions_numeric_condition)
                answers.extend(answers_numeric_condition)
                facts.extend(f)

                if print_proportions:
                    print("ass",n_questions_assignment, n_questions_assignment*size, n_questions_assignment*size/len_questions*100)
                    print("prev",n_questions_prevent, n_questions_prevent*size, n_questions_prevent*size/len_questions*100)
                    print("comb",n_questions_combination, n_questions_combination*size, n_questions_combination*size/len_questions*100)
                    print("join",n_questions_join, n_questions_join*size, n_questions_join*size/len_questions*100)
                    print("clos",n_questions_closure, n_questions_closure*size, n_questions_closure*size/len_questions*100)
                    print("pref",n_questions_preferences, n_questions_preferences*size, n_questions_preferences*size/len_questions*100)
                    print("filt",n_questions_select, n_questions_select*size, n_questions_select*size/len_questions*100)
                    print("neg filt",n_questions_negative, n_questions_negative*size, n_questions_negative*size/len_questions*100)
                    print("num filt",n_questions_numeric, n_questions_numeric*size, n_questions_numeric*size/len_questions*100)
                    break

            case "core-invariance":
                prompt_invariance=True
                n_questions_assignment = n_questions_prevent = n_questions_combination = 0
                n_questions_join = n_questions_closure = n_questions_preferences = n_questions_select = 0
                n_questions_negative = n_questions_numeric = 0
                
                for _ in range(80): #assignment
                    question_assignments, answer_assignments, f = label_assignment(predicates, np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(question_assignments)
                    answers.extend(answer_assignments)
                    facts.extend(f)
                    n_questions_assignment+=len(question_assignments)

                for _ in range(40): #constraint
                    question_prevents, answer_prevents, f = prevent_value(predicates, np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(question_prevents)
                    answers.extend(answer_prevents)
                    facts.extend(f)
                    n_questions_prevent += len(question_prevents)

                for _ in range(35): #combination
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_combinations, answers_combinations, f = generate_combinations(p_1, p_2, prompt_invariance, False)
                    questions.extend(questions_combinations)
                    answers.extend(answers_combinations)
                    facts.extend(f)
                    n_questions_combination += len(questions_combinations)

                for _ in range(20): #join
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_join, answers_join, f = execute_join(p_1, p_2, attributes, prompt_invariance, False)
                    questions.extend(questions_join)
                    answers.extend(answers_join)
                    facts.extend(f)
                    n_questions_join += len(questions_join)

                for _ in range(40): #closure
                    questions_closure, answers_closure, f = transitive_closure(np.random.choice(closures), np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(questions_closure)
                    answers.extend(answers_closure)
                    facts.extend(f)
                    n_questions_closure += len(questions_closure)

                for _ in range(5):  #preference
                    questions_preferences, answers_preferences, f = preferences(np.random.choice(predicates), predicates, prompt_invariance, False)
                    questions.extend(questions_preferences)
                    answers.extend(answers_preferences)
                    facts.extend(f)
                    n_questions_preferences += len(questions_preferences)
                
                for _ in range(60): #filtering
                    questions_select, answers_select, f = select_value(np.random.choice(predicates), np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(questions_select)
                    answers.extend(answers_select)
                    facts.extend(f)
                    n_questions_select += len(questions_select)

                for _ in range(20):  #negative filtering
                    questions_negative, answers_negative, f = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, False)
                    questions.extend(questions_negative)
                    answers.extend(answers_negative)
                    facts.extend(f)
                    n_questions_negative += len(questions_negative)

                for _ in range(20): #numeric filtering
                    questions_numeric_condition, answers_numeric_condition, f = select_by_numeric_condition(np.random.choice(predicates), prompt_invariance, False)
                    questions.extend(questions_numeric_condition)
                    answers.extend(answers_numeric_condition)
                    facts.extend(f)
                    n_questions_numeric += len(questions_numeric_condition)

                if print_proportions:
                    tot_questions = n_questions_assignment + n_questions_prevent + n_questions_combination + n_questions_join + n_questions_closure + n_questions_preferences + n_questions_select + n_questions_negative + n_questions_numeric
                    tot_questions_size = tot_questions * size
                    print("size = ",size)
                    print("tot_questions = ", tot_questions, "tot_questions_size = ", tot_questions_size)
                    print("ass",n_questions_assignment, n_questions_assignment*size, n_questions_assignment*size/tot_questions_size*100)
                    print("prev",n_questions_prevent, n_questions_prevent*size, n_questions_prevent*size/tot_questions_size*100)
                    print("comb",n_questions_combination, n_questions_combination*size, n_questions_combination*size/tot_questions_size*100)
                    print("join",n_questions_join, n_questions_join*size, n_questions_join*size/tot_questions_size*100)
                    print("clos",n_questions_closure, n_questions_closure*size, n_questions_closure*size/tot_questions_size*100)
                    print("pref",n_questions_preferences, n_questions_preferences*size, n_questions_preferences*size/tot_questions_size*100)
                    print("filt",n_questions_select, n_questions_select*size, n_questions_select*size/tot_questions_size*100)
                    print("neg filt",n_questions_negative, n_questions_negative*size, n_questions_negative*size/tot_questions_size*100)
                    print("num filt",n_questions_numeric, n_questions_numeric*size, n_questions_numeric*size/tot_questions_size*100)
                    sys.exit(1)

            case "core-invariance-complex":
                n_questions_jnf = n_questions_jf = n_questions_cg = n_questions_clnef = n_questions_gc = n_questions_gp = n_questions_gnef = n_questions_gnuf = n_questions_gf = n_questions_cnef = n_questions_cnuf = 0
                
                # for _ in range(5):  # join numeric filtering
                #     p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                #     questions_jnf, answers_jnf, f = join_numeric_filtering(p_1, p_2, attributes)

                #     questions.extend(questions_jnf)
                #     answers.extend(answers_jnf)
                #     facts.extend(f)

                #     n_questions_jnf += len(questions_jnf)

                  

                #     # 1. Join + Numeric Filtering
                #     # "Write an ASP program for the following problem. 
                #     # Consider predicate "{predicate_name_1}" having fields {a}, 
                #     # and the predicate "{predicate_name_2}" having fields {b}. 
                #     # Define a predicate "{predicate_name_1}_{predicate_name_2}" that 
                #     # associates to each "{predicate_name_1}" the "{random_attribute}" of 
                #     # "{predicate_name_2}"only where "{random_attribute}" is {condition} than {condition_value}."

                #     # ➡️ Motivazione: Il modello deve contemporaneamente effettuare un join e poi applicare un filtro numerico. MAGARI SI FA INSIEME FILTRO E JOIN   

                #     #########################################################################################################

                for _ in range(5):  # join filtering
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_jf, answers_jf, f = join_filtering(p_1, p_2, attributes, predicates)

                    questions.extend(questions_jf)
                    answers.extend(answers_jf)
                    facts.extend(f)

                    n_questions_jf += len(questions_jf)

                    # 2. Join + Negative Filtering
                    # "Write an ASP program that creates a new predicate {predicate_name_1}_{predicate_name_2} by joining {predicate_name_1} and {predicate_name_2}. However, exclude from the result any instance where {predicate_name_1} is associated with {not_label}."

                    # 🔹 Esempio ASP:

                    # join(X, Y) :- predicate1(X, A), predicate2(Y, B), not predicate1(X, not_label).
                    # ➡️ Obiettivo: Creare un join, escludendo elementi con un’etichetta specifica.

                # for _ in range(5):  # closure guessing
                #     questions_cg, answers_cg, f = closure_guessing(attributes, np.random.choice(predicates), np.random.choice(closures))

                #     questions.extend(questions_cg)
                #     answers.extend(answers_cg)
                #     facts.extend(f)

                #     n_questions_cg += len(questions_cg)

                #         # 2. Closure + Assignment
                #         # "Write an ASP program that defines predicate "{closure_name}" as the transitive closure of predicate 
                #         # "{predicate_name}". 
                #         # Then, assign exactly one label from the set {','.join([f"{x}" for x in labels_to_assign])} 
                #         # to each element in "{closure_name}"."

                #         # ➡️ Obiettivo: Il modello calcola la chiusura e poi assegna un'etichetta ai risultati.

                #         ########################################################################################################

                # for _ in range(5):  # closure negative filtering
                #     questions_clnef, answers_clnef, f = closure_negative_filtering(attributes, np.random.choice(predicates), np.random.choice(closures))

                #     questions.extend(questions_clnef)
                #     answers.extend(answers_clnef)
                #     facts.extend(f)

                #     n_questions_clnef += len(questions_clnef)

                for _ in range(30):  # guessing constraint
                    questions_gc, answers_gc, f = guessing_constraint(labels, np.random.choice(predicates))

                    questions.extend(questions_gc)
                    answers.extend(answers_gc)
                    facts.extend(f)

                    n_questions_gc += len(questions_gc)
                
                # for _ in range(5):  # guessing preference
                #     questions_gp, answers_gp, f = guessing_preference(labels, np.random.choice(predicates))

                #     questions.extend(questions_gp)
                #     answers.extend(answers_gp)
                #     facts.extend(f)

                #     n_questions_gp += len(questions_gp)

                # for _ in range(5):  # guessing negative filtering
                #     questions_gnef, answers_gnef, f = guessing_negative_filtering(attributes, np.random.choice(predicates))

                #     questions.extend(questions_gnef)
                #     answers.extend(answers_gnef)
                #     facts.extend(f)

                #     n_questions_gnef += len(questions_gnef)

                #         # 3. Guessing + Negative Filtering
                #         # "Write an ASP program that assigns exactly one label from a given set to each element in predicate "{predicate_name}". 
                #         # The assignment should be guessed between "{label1}" and "{label2}", but an element cannot be assigned a label 
                #         # if it is already associated with "{label3}" in "{predicate_name}".

                #         # ➡️ Obiettivo: Il modello deve indovinare (guessing) un'assegnazione tra due etichette ma evitando elementi che soddisfano una condizione di negative filtering.

                #         ########################################################################################################

                # for _ in range(5):  # guessing numeric filtering
                #     questions_gnuf, answers_gnuf, f = guessing_numeric_filtering(attributes, np.random.choice(predicates), np.random.choice(attributes), np.random.choice(attributes))

                #     questions.extend(questions_gnuf)
                #     answers.extend(answers_gnuf)
                #     facts.extend(f)

                #     n_questions_gnuf += len(questions_gnuf)

                #         # 4. Guessing + Numeric Filtering
                #         # "Write an ASP program that guesses a label {label1} or {label2} for each element in {predicate_name}. 
                #         # Then, filter only the elements where the assigned value is {condition} than {condition_value}."

                #         # 🔹 Esempio ASP:

                #         # assign(X, label1) | assign(X, label2) :- predicate1(X, Value), Value > condition_value.

                #         # ➡️ Obiettivo: Assegnare etichette con guessing e poi applicare un filtro numerico.

                #         ########################################################################################################

                # for _ in range(5):  # guessing filtering
                #     questions_gf, answers_gf, f = guessing_filtering(attributes, np.random.choice(predicates))

                #     questions.extend(questions_gf)
                #     answers.extend(answers_gf)
                #     facts.extend(f)

                #     n_questions_gf += len(questions_gf)

                #         # 4. Assignment + Filtering
                #         # "Write an ASP program that assigns exactly one label from a given set to each element in predicate {predicate_name}. Then, filter and return only the elements assigned to label {label}."

                #         # 🔹 Esempio ASP:
                #         # assign(X, label1) | assign(X, label2) :- predicate1(X).
                #         # select(X) :- assign(X, label1).
                    
                #         # ➡️ Obiettivo: Il modello deve fare guessing per l'assegnazione e poi filtrare solo certi risultati.

                #         ########################################################################################################

                for _ in range(15):  # combination negative filtering
                    questions_cnef, answers_cnef, f = combination_negative_filtering(labels, np.random.choice(predicates), np.random.choice(predicates), np.random.choice(predicates))

                    questions.extend(questions_cnef)
                    answers.extend(answers_cnef)
                    facts.extend(f)

                    n_questions_cnef += len(questions_cnef)

                # for _ in range(5):  # combination numeric filtering
                #     questions_cnuf, answers_cnuf, f = combination_numeric_filtering(labels, np.random.choice(predicates), np.random.choice(predicates))

                #     questions.extend(questions_cnuf)
                #     answers.extend(answers_cnuf)
                #     facts.extend(f)

                #     n_questions_cnuf += len(questions_cnuf)

                       
                if print_proportions:
                    # print("jnumf",n_questions_jnf, n_questions_jnf*size, n_questions_jnf*size/len_questions*100)
                    print("jf",n_questions_jf, n_questions_jf*size, n_questions_jf*size/len_questions*100)
                    # print("clg",n_questions_cg, n_questions_cg*size, n_questions_cg*size/len_questions*100)
                    # print("clnef",n_questions_clnef, n_questions_clnef*size, n_questions_clnef*size/len_questions*100)
                    print("gc",n_questions_gc, n_questions_gc*size, n_questions_gc*size/len_questions*100)
                    # print("gp",n_questions_gp, n_questions_gp*size, n_questions_gp*size/len_questions*100)
                    # print("gnef",n_questions_gnef, n_questions_gnef*size, n_questions_gnef*size/len_questions*100)
                    # print("gnuf",n_questions_gnuf, n_questions_gnuf*size, n_questions_gnuf*size/len_questions*100)
                    # print("gf",n_questions_gf, n_questions_gf*size, n_questions_gf*size/len_questions*100)
                    print("cnef",n_questions_cnef, n_questions_cnef*size, n_questions_cnef*size/len_questions*100)
                    # print("cnuf",n_questions_cnuf, n_questions_cnuf*size, n_questions_cnuf*size/len_questions*100)
                    sum = n_questions_jf + n_questions_gc + n_questions_cnef
                    print("tot = ", sum, " ", sum*size)
                    sys.exit(1) 

                    print("jnumf",n_questions_jnf, n_questions_jnf*size, n_questions_jnf*size/len_questions*100)
                    print("jf",n_questions_jf, n_questions_jf*size, n_questions_jf*size/len_questions*100)
                    print("clg",n_questions_cg, n_questions_cg*size, n_questions_cg*size/len_questions*100)
                    print("clnef",n_questions_clnef, n_questions_clnef*size, n_questions_clnef*size/len_questions*100)
                    print("gc",n_questions_gc, n_questions_gc*size, n_questions_gc*size/len_questions*100)
                    print("gp",n_questions_gp, n_questions_gp*size, n_questions_gp*size/len_questions*100)
                    print("gnef",n_questions_gnef, n_questions_gnef*size, n_questions_gnef*size/len_questions*100)
                    print("gnuf",n_questions_gnuf, n_questions_gnuf*size, n_questions_gnuf*size/len_questions*100)
                    print("gf",n_questions_gf, n_questions_gf*size, n_questions_gf*size/len_questions*100)
                    print("cnef",n_questions_cnef, n_questions_cnef*size, n_questions_cnef*size/len_questions*100)
                    print("cnuf",n_questions_cnuf, n_questions_cnuf*size, n_questions_cnuf*size/len_questions*100)
                    sum = n_questions_jn + n_questions_jnf + n_questions_cg + n_questions_gn + n_questions_gnf + n_questions_gf + n_questions_cc
                    print("tot = ", sum, " ", sum*size)
                    break                

            case "base":
                n_questions_jnf = n_questions_jf = n_questions_cg = n_questions_clnef = n_questions_gc = n_questions_gp = n_questions_gnef = n_questions_gnuf = n_questions_gf = n_questions_cnef = n_questions_cnuf = 0
                
                # for _ in range(5):  # join numeric filtering
                #     p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                #     questions_jnf, answers_jnf, f = join_numeric_filtering(p_1, p_2, attributes)

                #     questions.extend(questions_jnf)
                #     answers.extend(answers_jnf)
                #     facts.extend(f)

                #     n_questions_jnf += len(questions_jnf)

                  

                #     # 1. Join + Numeric Filtering
                #     # "Write an ASP program for the following problem. 
                #     # Consider predicate "{predicate_name_1}" having fields {a}, 
                #     # and the predicate "{predicate_name_2}" having fields {b}. 
                #     # Define a predicate "{predicate_name_1}_{predicate_name_2}" that 
                #     # associates to each "{predicate_name_1}" the "{random_attribute}" of 
                #     # "{predicate_name_2}"only where "{random_attribute}" is {condition} than {condition_value}."

                #     # ➡️ Motivazione: Il modello deve contemporaneamente effettuare un join e poi applicare un filtro numerico. MAGARI SI FA INSIEME FILTRO E JOIN   

                #     #########################################################################################################

                for _ in range(20):  # join filtering
                    p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                    questions_jf, answers_jf, f = join_filtering(p_1, p_2, attributes, predicates)

                    questions.extend(questions_jf)
                    answers.extend(answers_jf)
                    facts.extend(f)

                    n_questions_jf += len(questions_jf)

                    # 2. Join + Negative Filtering
                    # "Write an ASP program that creates a new predicate {predicate_name_1}_{predicate_name_2} by joining {predicate_name_1} and {predicate_name_2}. However, exclude from the result any instance where {predicate_name_1} is associated with {not_label}."

                    # 🔹 Esempio ASP:

                    # join(X, Y) :- predicate1(X, A), predicate2(Y, B), not predicate1(X, not_label).
                    # ➡️ Obiettivo: Creare un join, escludendo elementi con un’etichetta specifica.

                # for _ in range(5):  # closure guessing
                #     questions_cg, answers_cg, f = closure_guessing(attributes, np.random.choice(predicates), np.random.choice(closures))

                #     questions.extend(questions_cg)
                #     answers.extend(answers_cg)
                #     facts.extend(f)

                #     n_questions_cg += len(questions_cg)

                #         # 2. Closure + Assignment
                #         # "Write an ASP program that defines predicate "{closure_name}" as the transitive closure of predicate 
                #         # "{predicate_name}". 
                #         # Then, assign exactly one label from the set {','.join([f"{x}" for x in labels_to_assign])} 
                #         # to each element in "{closure_name}"."

                #         # ➡️ Obiettivo: Il modello calcola la chiusura e poi assegna un'etichetta ai risultati.

                #         ########################################################################################################

                # for _ in range(5):  # closure negative filtering
                #     questions_clnef, answers_clnef, f = closure_negative_filtering(attributes, np.random.choice(predicates), np.random.choice(closures))

                #     questions.extend(questions_clnef)
                #     answers.extend(answers_clnef)
                #     facts.extend(f)

                #     n_questions_clnef += len(questions_clnef)

                for _ in range(135):  # guessing constraint
                    questions_gc, answers_gc, f = guessing_constraint(labels, np.random.choice(predicates))

                    questions.extend(questions_gc)
                    answers.extend(answers_gc)
                    facts.extend(f)

                    n_questions_gc += len(questions_gc)
                
                # for _ in range(5):  # guessing preference
                #     questions_gp, answers_gp, f = guessing_preference(labels, np.random.choice(predicates))

                #     questions.extend(questions_gp)
                #     answers.extend(answers_gp)
                #     facts.extend(f)

                #     n_questions_gp += len(questions_gp)

                # for _ in range(5):  # guessing negative filtering
                #     questions_gnef, answers_gnef, f = guessing_negative_filtering(attributes, np.random.choice(predicates))

                #     questions.extend(questions_gnef)
                #     answers.extend(answers_gnef)
                #     facts.extend(f)

                #     n_questions_gnef += len(questions_gnef)

                #         # 3. Guessing + Negative Filtering
                #         # "Write an ASP program that assigns exactly one label from a given set to each element in predicate "{predicate_name}". 
                #         # The assignment should be guessed between "{label1}" and "{label2}", but an element cannot be assigned a label 
                #         # if it is already associated with "{label3}" in "{predicate_name}".

                #         # ➡️ Obiettivo: Il modello deve indovinare (guessing) un'assegnazione tra due etichette ma evitando elementi che soddisfano una condizione di negative filtering.

                #         ########################################################################################################

                # for _ in range(5):  # guessing numeric filtering
                #     questions_gnuf, answers_gnuf, f = guessing_numeric_filtering(attributes, np.random.choice(predicates), np.random.choice(attributes), np.random.choice(attributes))

                #     questions.extend(questions_gnuf)
                #     answers.extend(answers_gnuf)
                #     facts.extend(f)

                #     n_questions_gnuf += len(questions_gnuf)

                #         # 4. Guessing + Numeric Filtering
                #         # "Write an ASP program that guesses a label {label1} or {label2} for each element in {predicate_name}. 
                #         # Then, filter only the elements where the assigned value is {condition} than {condition_value}."

                #         # 🔹 Esempio ASP:

                #         # assign(X, label1) | assign(X, label2) :- predicate1(X, Value), Value > condition_value.

                #         # ➡️ Obiettivo: Assegnare etichette con guessing e poi applicare un filtro numerico.

                #         ########################################################################################################

                # for _ in range(5):  # guessing filtering
                #     questions_gf, answers_gf, f = guessing_filtering(attributes, np.random.choice(predicates))

                #     questions.extend(questions_gf)
                #     answers.extend(answers_gf)
                #     facts.extend(f)

                #     n_questions_gf += len(questions_gf)

                #         # 4. Assignment + Filtering
                #         # "Write an ASP program that assigns exactly one label from a given set to each element in predicate {predicate_name}. Then, filter and return only the elements assigned to label {label}."

                #         # 🔹 Esempio ASP:
                #         # assign(X, label1) | assign(X, label2) :- predicate1(X).
                #         # select(X) :- assign(X, label1).
                    
                #         # ➡️ Obiettivo: Il modello deve fare guessing per l'assegnazione e poi filtrare solo certi risultati.

                #         ########################################################################################################

                for _ in range(117):  # combination negative filtering
                    questions_cnef, answers_cnef, f = combination_negative_filtering(labels, np.random.choice(predicates), np.random.choice(predicates), np.random.choice(predicates))

                    questions.extend(questions_cnef)
                    answers.extend(answers_cnef)
                    facts.extend(f)

                    n_questions_cnef += len(questions_cnef)

                # for _ in range(5):  # combination numeric filtering
                #     questions_cnuf, answers_cnuf, f = combination_numeric_filtering(labels, np.random.choice(predicates), np.random.choice(predicates))

                #     questions.extend(questions_cnuf)
                #     answers.extend(answers_cnuf)
                #     facts.extend(f)

                #     n_questions_cnuf += len(questions_cnuf)

                       
                if print_proportions:
                    # print("jnumf",n_questions_jnf, n_questions_jnf*size, n_questions_jnf*size/len_questions*100)
                    print("jf",n_questions_jf, n_questions_jf*size, n_questions_jf*size/len_questions*100)
                    # print("clg",n_questions_cg, n_questions_cg*size, n_questions_cg*size/len_questions*100)
                    # print("clnef",n_questions_clnef, n_questions_clnef*size, n_questions_clnef*size/len_questions*100)
                    print("gc",n_questions_gc, n_questions_gc*size, n_questions_gc*size/len_questions*100)
                    # print("gp",n_questions_gp, n_questions_gp*size, n_questions_gp*size/len_questions*100)
                    # print("gnef",n_questions_gnef, n_questions_gnef*size, n_questions_gnef*size/len_questions*100)
                    # print("gnuf",n_questions_gnuf, n_questions_gnuf*size, n_questions_gnuf*size/len_questions*100)
                    # print("gf",n_questions_gf, n_questions_gf*size, n_questions_gf*size/len_questions*100)
                    print("cnef",n_questions_cnef, n_questions_cnef*size, n_questions_cnef*size/len_questions*100)
                    # print("cnuf",n_questions_cnuf, n_questions_cnuf*size, n_questions_cnuf*size/len_questions*100)
                    sum = n_questions_jf + n_questions_gc + n_questions_cnef
                    print("tot = ", sum, " ", sum*size)
                    sys.exit(1) 


    random.seed(42)
    temp = list(zip(questions, answers))


    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    questions, answers = list(res1), list(res2)

    return questions, answers


def compress_csv(file_path):
    # Creiamo il nome del file .bz2 di output

    output_filename = file_path.replace(".csv", ".csv.bz2")
    
    # Verifica se il file esiste prima di comprimere

    if not os.path.isfile(file_path):
        print(f"Errore: Il file {file_path} non esiste.")
        return
    
    try:
        # Apriamo il file originale in modalità lettura e il file .bz2 in modalità scrittura
        with open(file_path, 'rb') as f_in, bz2.BZ2File(output_filename, 'wb') as f_out:
            # Copia il contenuto del file originale nel file compresso
            f_out.writelines(f_in)
            print(f"File {file_path} compresso")
    
    except Exception as e:
        print(f"Errore durante la compressione del file {file_path}: {e}")


##############   riscrive la domanda nell'output per questa funzione qua
def formatting_func(example):
            text = f"Question: {example['question'][0]}\nAnswer: {example['answer'][0]}"
            return [text]


def generate_response(question):
    try:
        inputs_device = model.device
        inputs = tokenizer(question, return_tensors="pt").to(inputs_device)
        
        outputs = model.generate(**inputs, max_new_tokens=150)                                                  #restituisce un tensore
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        print(f"Errore durante la generazione della risposta: {e}")
        return "Errore durante la generazione della risposta."

"""
Args:
    program (str): a string of ASP program
    opt (bool): if true, only optimal answer sets are returned   
    leave it to False when there is no weak constraint
"""

def gen_answer_set(program, opt=False):
    clingo_control = Control(['1', '--warn=none', '--opt-mode=optN', '-t', '4'])
    models = []
    try:
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])], context=Context())
    except Exception as e:
        return ["error"]
    if opt:
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
    else:
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
    models = [[str(atom) for atom in model] for model in models]
    
    return models

def check_semantics(correct_models, generated_models):

    set_correct_models = set(map(frozenset, correct_models))
    set_generated_models = set(map(frozenset, generated_models))

    jaccard = len(set_correct_models.intersection(set_generated_models))/len(set_correct_models.union(set_generated_models))
    return jaccard


def generate_test_cases():      ##  genera 10 prompt, con dati diversi rispetto al training, da sottoporre al modello per test leggero

    with open('output.txt', 'w') as f:
        f.write("\n")

    colors = ["pink", "white", "black", "dark magenta", "light blue"] 
    cities = ["cosenza", "delhi", "cairo", "mumbai", "moscow", "singapore", "chicago", "toronto", "barcelona"]
    labels = ["wall", "chair", "roof", "flower", "butterfly", "laptop", "desk", "cloud", "storm"]
    attributes = ["surname", "owner", "lake", "hair", "weight", "strength", "quality"]

    predicates = colors + cities + labels + attributes 
    closures = ["loops", "family", "trains", "journey"]

    np.random.seed()

    match turn:   ####        a seconda del turno genero i test-cases
        case "core":
            print("ASSIGNMENT")
            question_assignments, _, _ = label_assignment(predicates, np.random.choice(predicates), False, False)
            answer = generate_response(question_assignments[0])
            with open('output.txt', 'a') as f:
                f.write("non inv\n")
                f.write(question_assignments[0])
                f.write(answer)
                f.write("\n\n")

            
            print("PREVENT")
            question_prevents, _, _ = prevent_value(predicates, np.random.choice(predicates), False, False)
            answer = generate_response(question_prevents[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(question_prevents[0])
                f.write(answer)
                f.write("\n\n")

            
            print("COMBINATIONS")
            p_1, p_2 = np.random.choice(predicates, 2, replace=False)
            questions, _, _ = generate_combinations(p_1, p_2, False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            
            print("SELECT VALUE")
            questions, _, _ = select_value(np.random.choice(predicates), np.random.choice(predicates), False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("non inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            
            print("JOIN")
            p_1, p_2 = np.random.choice(predicates, 2, replace=False)

            n_attributes = np.random.randint(3, 6)
            attributes = np.array(attributes, dtype='U18')
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            
            chosen_attributes[0] = f"ID"

            chosen_attributes[random_pos] = f"{p_2}ID"

            a = ''
            for attr in chosen_attributes[:-1]:
                a += f'"{attr}",'
            a += f'"{chosen_attributes[-1]}"'

            n_attributes = np.random.randint(2, 5)
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            chosen_attributes[0] = "ID"

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            
            questions = []
            questions, _, _ = execute_join(p_1, p_2, attributes, False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            print("TRANSITIVE CLOSURE")
            questions, _, _ = transitive_closure(np.random.choice(closures), np.random.choice(predicates), False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

        
            print("PREFERENCES")
            questions, _, _ = preferences(np.random.choice(predicates), predicates, False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            
            # print("MINIMIZATION")           #####           FINETUNARE
            # questions, _, _ = minimizing(np.random.choice(predicates), predicates, False)
            # answer = generate_response(questions[0])
            # print(answer)
            # print("*************")
            # with open('output.txt', 'a') as f:
            #     f.write("noninv\n")
            #     f.write(questions[0])
            #     f.write(answer)
            #     f.write("\n\n")

            
            # print("MAXIMIZATION")                     #####       OK 
            # questions, _, _ = maximizing(np.random.choice(predicates), predicates, False)
            # answer = generate_response(questions[0])
            # print(answer)
            # print("*************")
            # with open('output.txt', 'a') as f:
            #     f.write("noninv\n")
            #     f.write(questions[0])
            #     f.write(answer)
            #     f.write("\n\n")

            
            print("NEGATIVE CONDITION")
            questions, _, _ = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            
            print("NUMERIC CONDITION")
            conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]

            questions, _, _ = select_by_numeric_condition(np.random.choice(predicates), False, False)
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("noninv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

        case "core-invariance":
            question_assignments = []
            question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
            answer = generate_response(question_assignments[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(question_assignments[0])
                f.write("\n\n")
                f.write(answer)
                f.write("\n\n")

            question_prevents = []
            question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
            answer = generate_response(question_prevents[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(question_prevents[0])
                f.write(answer)
                f.write("\n\n")

            p_1, p_2 = np.random.choice(predicates, 2, replace=False)
            questions = []
            questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            questions = []
            questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            n_attributes = np.random.randint(3, 6)
            attributes = np.array(attributes, dtype='U18')
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            random_pos = np.random.randint(1, n_attributes)
            
            chosen_attributes[0] = f"ID"

            chosen_attributes[random_pos] = f"{p_2}ID"

            a = ''
            for attr in chosen_attributes[:-1]:
                a += f'"{attr}",'
            a += f'"{chosen_attributes[-1]}"'

            n_attributes = np.random.randint(2, 5)
            chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
            chosen_attributes[0] = "ID"

            b = ''
            for attr in chosen_attributes[:-1]:
                b += f'"{attr}",'
            b += f'"{chosen_attributes[-1]}"'

            
            questions = []
            questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            questions = []
            questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            questions = []
            questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

            # questions = []                  #####       FINETUNARE
            # questions.append(f'''Construct an ASP program to minimize the presence of the "{np.random.choice(predicates)}" predicate tagged with the label "{np.random.choice(predicates)}".''')
            # answer = generate_response(questions[0])
            # print(answer)
            # print("*************")
            # with open('output.txt', 'a') as f:
            #     f.write("inv\n")
            #     f.write(questions[0])
            #     f.write(answer)
            #     f.write("\n\n")

            # questions = []                            #########       OK
            # questions.append(f'''Construct an ASP program to boost the count of "{np.random.choice(predicates)}" predicates that are labeled as "{np.random.choice(predicates)}".''')
            # answer = generate_response(questions[0])
            # print(answer)
            # print("*************")
            # with open('output.txt', 'a') as f:
            #     f.write("inv\n")
            #     f.write(questions[0])
            #     f.write(answer)
            #     f.write("\n\n")

            questions = []
            questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")


            conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
            questions = []
            questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
            answer = generate_response(questions[0])
            with open('output.txt', 'a') as f:
                f.write("inv\n")
                f.write(questions[0])
                f.write(answer)
                f.write("\n\n")

        case "core-invariance-complex":
            questions = []
            for _ in range(1):
                p_1, p_2 = np.random.choice(predicates, 2, replace=False)  
                question, answer, f = join_numeric_filtering(p_1, p_2, attributes)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("join_num\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("join_neg\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                question, answer, f = closure_guessing(predicates, np.random.choice(predicates), np.random.choice(closures))
                questions.append(question)
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("clo_guess\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                question, answer, f = closure_negative_filtering(predicates, np.random.choice(predicates), np.random.choice(closures))
                questions.append(question)
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("clo_neg_filt\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("guess_constr\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")
                    
                question, answer, f = guessing_preference(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("guess_pref\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")


                question, answer, f = guessing_negative_filtering(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("guess_neg\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")
                    
                question, answer, f = guessing_numeric_filtering(predicates, np.random.choice(predicates), np.random.choice(attributes),np.random.choice(attributes))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("guess_num\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                question, answer, f = guessing_filtering(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("guess_filter\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("comb_constr\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = combination_numeric_filtering(labels, p_1, p_2)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as f:
                    f.write("comb_num\n")
                    f.write(question[0])
                    f.write("\ngenerated: ")
                    f.write(answerg)
                    f.write("\nDesired: ")
                    f.write(answer[0])
                    f.write("\n\n")
        
        case "base":
            questions = []
            for _ in range(1):
                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)  
                # question, answer, f = join_numeric_filtering(p_1, p_2, attributes)
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("join_num\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                # question, answer, f = closure_guessing(predicates, np.random.choice(predicates), np.random.choice(closures))
                # questions.append(question)
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("clo_guess\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")

                # question, answer, f = closure_negative_filtering(predicates, np.random.choice(predicates), np.random.choice(closures))
                # questions.append(question)
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("clo_neg_filt\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")
                    
                # question, answer, f = guessing_preference(predicates, np.random.choice(predicates))
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("guess_pref\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")


                # question, answer, f = guessing_negative_filtering(predicates, np.random.choice(predicates))
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("guess_neg\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")
                    
                # question, answer, f = guessing_numeric_filtering(predicates, np.random.choice(predicates), np.random.choice(attributes),np.random.choice(attributes))
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("guess_num\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")

                # question, answer, f = guessing_filtering(predicates, np.random.choice(predicates))
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("guess_filter\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # question, answer, f = combination_numeric_filtering(labels, p_1, p_2)
                # questions.append(question[0])
                # answerg = generate_response(question[0])
                # with open('output.txt', 'a') as f:
                #     f.write("comb_num\n")
                #     f.write(question[0])
                #     f.write("\ngenerated: ")
                #     f.write(answerg)
                #     f.write("\nDesired: ")
                #     f.write(answer[0])
                #     f.write("\n\n")


                #######     prompt dal training set

                # question, answer = f'''Write an ASP program for the following problem. Consider predicate ""price"" having fields ""ID"",""romeID"",""city"", and the predicate ""rome"" having fields ""ID"",""name"",""age"",""shape"". Define a predicate ""price_rome"" that associates to each price the attribute age of rome. In addition, select all values associated to the predicate ""price_rome"" with label ""brown"".''',f'''price_rome(X,Z):-price(X,Y,_),rome(Y,_,Z,_).select(X):-price_rome(X,""brown"").'''
                # questions.append(question)
                # answerg = generate_response(question)
                # with open('output.txt', 'a') as o:
                #     o.write("join filter\n")
                #     o.write(question)
                #     o.write("\ngenerated: \n")
                #     o.write(answerg)
                #     o.write("\nDesired: \n")
                #     o.write(answer)
                #     o.write("\n\n\n")
                
                # question, answer = f'''Write an ASP program for the following problem. Assign exactly a label among a given set of labels to a set of elements. The set of elements is expressed by predicate price. The labels are moto,table,food,bike,tree,color,car. Then prevent the predicate ""price"" with value ""8"" from having label ""tree"".''',f'''assign(X,""moto"")|assign(X,""table"")|assign(X,""food"")|assign(X,""bike"")|assign(X,""tree"")|assign(X,""color"")|assign(X,""car""):-price(X).:-assign(8, ""tree"").'''
                # questions.append(question)
                # answerg = generate_response(question)
                # with open('output.txt', 'a') as o:
                #     o.write("guess constr\n")
                #     o.write(question)
                #     o.write("\ngenerated: \n")
                #     o.write(answerg)
                #     o.write("\nDesired: \n")
                #     o.write(answer)
                #     o.write("\n\n\n")
                
                # question, answer = f'''Write an ASP program for the following problem. Generate all the combinations of elements from two sets. The two sets are represented by predicates ""blue"" and ""bike"". In addition, select all values associated with predicate combination but not associated with predicate ""cyan"" and label ""color"".''',f'''combination(X,Y):-blue(X),bike(Y).select(X):-combination(X,_), not cyan(X, ""color"").'''
                # questions.append(question)
                # answerg = generate_response(question)
                # with open('output.txt', 'a') as o:
                #     o.write("comb_neg\n")
                #     o.write(question)
                #     o.write("\ngenerated: \n")
                #     o.write(answerg)
                #     o.write("\nDesired: \n")
                #     o.write(answer)
                #     o.write("\n\n")

        case "averaged_1":
            questions = []
            for _ in range(1):

                question_assignments = []
                question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(question_assignments[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_assignments[0])
                    f.write("\n\n")
                    f.write(answer)
                    f.write("\n\n")

                question_prevents = []
                question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
                answer = generate_response(question_prevents[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_prevents[0])
                    f.write(answer)
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions = []
                questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                n_attributes = np.random.randint(3, 6)
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                
                chosen_attributes[0] = f"ID"

                chosen_attributes[random_pos] = f"{p_2}ID"

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                n_attributes = np.random.randint(2, 5)
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                
                questions = []
                questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
                questions = []
                questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                ### complexxx

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

        case "averaged_3":
            questions = []
            for _ in range(1):

                question_assignments = []
                question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(question_assignments[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_assignments[0])
                    f.write("\n\n")
                    f.write(answer)
                    f.write("\n\n")

                question_prevents = []
                question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
                answer = generate_response(question_prevents[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_prevents[0])
                    f.write(answer)
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions = []
                questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                n_attributes = np.random.randint(3, 6)
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                
                chosen_attributes[0] = f"ID"

                chosen_attributes[random_pos] = f"{p_2}ID"

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                n_attributes = np.random.randint(2, 5)
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                
                questions = []
                questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
                questions = []
                questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                ### complexxx

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

        case "averaged_5":
            questions = []
            for _ in range(1):

                question_assignments = []
                question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(question_assignments[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_assignments[0])
                    f.write("\n\n")
                    f.write(answer)
                    f.write("\n\n")

                question_prevents = []
                question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
                answer = generate_response(question_prevents[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_prevents[0])
                    f.write(answer)
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions = []
                questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                n_attributes = np.random.randint(3, 6)
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                
                chosen_attributes[0] = f"ID"

                chosen_attributes[random_pos] = f"{p_2}ID"

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                n_attributes = np.random.randint(2, 5)
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                
                questions = []
                questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
                questions = []
                questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                ### complexxx

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

        case "averaged_7":
            questions = []
            for _ in range(1):

                question_assignments = []
                question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(question_assignments[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_assignments[0])
                    f.write("\n\n")
                    f.write(answer)
                    f.write("\n\n")

                question_prevents = []
                question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
                answer = generate_response(question_prevents[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_prevents[0])
                    f.write(answer)
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions = []
                questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                n_attributes = np.random.randint(3, 6)
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                
                chosen_attributes[0] = f"ID"

                chosen_attributes[random_pos] = f"{p_2}ID"

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                n_attributes = np.random.randint(2, 5)
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                
                questions = []
                questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
                questions = []
                questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                ### complexxx

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

        case "averaged_9":
            questions = []
            for _ in range(1):

                question_assignments = []
                question_assignments.append(f'''Design an ASP program that assign only a label from the specified set {', '.join([f"{x}" for x in np.random.choice(labels, size=np.random.randint(2, 5), replace=False)])} with a collection of items defined by the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(question_assignments[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_assignments[0])
                    f.write("\n\n")
                    f.write(answer)
                    f.write("\n\n")

                question_prevents = []
                question_prevents.append(f'''Formulate an ASP program that restricts the predicate "{np.random.choice(predicates)}" with value {np.random.randint(1, 20)} from being associated with the label "{np.random.randint(0, len(labels))}" .''')
                answer = generate_response(question_prevents[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(question_prevents[0])
                    f.write(answer)
                    f.write("\n\n")

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions = []
                questions.append(f'''Formulate an ASP solution that computes the intersections of elements between the sets represented by "{p_1}" and "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP solution to extract all values associated with the label "{np.random.choice(predicates)}" within the context of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                n_attributes = np.random.randint(3, 6)
                attributes = np.array(attributes, dtype='U18')
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                random_pos = np.random.randint(1, n_attributes)
                
                chosen_attributes[0] = f"ID"

                chosen_attributes[random_pos] = f"{p_2}ID"

                a = ''
                for attr in chosen_attributes[:-1]:
                    a += f'"{attr}",'
                a += f'"{chosen_attributes[-1]}"'

                n_attributes = np.random.randint(2, 5)
                chosen_attributes = np.random.choice(attributes, size=n_attributes, replace=False)
                chosen_attributes[0] = "ID"

                b = ''
                for attr in chosen_attributes[:-1]:
                    b += f'"{attr}",'
                b += f'"{chosen_attributes[-1]}"'

                
                questions = []
                questions.append(f'''Develop an ASP program to solve the following issue. Defined the predicate "{p_1}" with fields {a} and the predicate "{p_2}" with fields {b}, create a predicate "{p_1}_{p_2}" that links each "{p_1}" with the "{chosen_attributes[np.random.randint(1, len(chosen_attributes))]}" of "{p_2}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program that calculates the predicate "{np.random.choice(closures)}" as the transitive closure of the predicate "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Construct an ASP program to avoid the predicate "{np.random.choice(predicates)}" with value "{np.random.randint(1, 20)}" from being associated with label "{np.random.choice(predicates)}". If this association occurs, impose a penalty of "{np.random.randint(1, 3)}" at level "{np.random.randint(1, 3)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")

                questions = []
                questions.append(f'''Formulate an ASP program to identify all values linked to the predicate "{np.random.choice(predicates)}" but not associated with the predicate "{np.random.choice(predicates)}" and labeled as "{np.random.choice(predicates)}".''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                conditions = ["different", "greater", "lower", "greater or equal", "lower or equal"]
                questions = []
                questions.append(f'''Formulate an ASP solution to select all values linked to the predicate "{np.random.choice(predicates)}" with a value {np.random.choice(conditions)} than {np.random.randint(1, 100)}.''')
                answer = generate_response(questions[0])
                with open('output.txt', 'a') as f:
                    f.write("inv\n")
                    f.write(questions[0])
                    f.write(answer)
                    f.write("\n\n")


                ### complexxx

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                question, answer, f = join_filtering(p_1, p_2, attributes, predicates)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("join_filter\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                question, answer, f = guessing_constraint(predicates, np.random.choice(predicates))
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("guess_constr\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                question, answer, f = combination_negative_filtering(labels, p_1, p_2, p_3)
                questions.append(question[0])
                answerg = generate_response(question[0])
                with open('output.txt', 'a') as o:
                    o.write("comb_neg\n")
                    o.write(question[0])
                    o.write("\ngenerated: \n")
                    o.write(answerg)
                    o.write("\nDesired: \n")
                    o.write(answer[0])
                    o.write("\nFacts: \n")
                    o.write(f[0])
                    o.write("\n\n\n")

                

def build_test_set():   #   costruisce domande, risposte e fatti per come dovrebbero essere

    seed = 721345631

    colors = ["pink", "white", "black", "darkmagenta", "lightblue"]
    cities = ["cosenza", "delhi", "cairo", "mumbai", "moscow", "singapore", "chicago", "toronto", "barcelona"]
    labels = ["wall", "chair", "roof", "flower", "butterfly", "laptop", "desk", "cloud", "storm"]
    attributes = ["surname", "owner", "lake", "hair", "weight", "strength", "quality"]

    predicates = colors + cities + labels + attributes
    closures = ["loops", "family", "trains", "journey"]

    test_tuples = []

    match turn:
        case "core":
            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)
                
                chosen = 0

                questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates, np.random.choice(predicates), False, False)

                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[chosen]])

                questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates, np.random.choice(predicates), False, False)

                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2, False, False)

                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes, False, False)

                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures), np.random.choice(predicates), False, False)

                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[chosen]])

                questions_preferences, answers_preferences, facts_preferences = preferences(np.random.choice(predicates),predicates, False, False)

                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[chosen]])

                questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates), np.random.choice(predicates), False, False)

                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[chosen]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, False, False)

                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[chosen]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(np.random.choice(predicates), False, False)

                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[chosen]])
            
        case "core-invariance":
            prompt_invariance=True

            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)

                questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates, np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[0]])

                questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates, np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21*9)             ###########     21*9 perché ogni chiamata a join genera 9 tipi di join e 20 prompt per ogni tipo
                
                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures), np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[0]])

                questions_preferences, answers_preferences, facts_preferences = preferences(np.random.choice(predicates),predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[0]])

                questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates), np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[0]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[0]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[0]])
    
        case "core-invariance-complex":
            prompt_invariance = True

            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)
                
                ## start invariance

                questions_assignments, answers_assignments, facts_assignments = label_assignment(predicates, np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments])

                questions_prevents, answers_prevents, facts_prevents = prevent_value(predicates, np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = generate_combinations(p_1, p_2, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = execute_join(p_1, p_2, attributes, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21*9)             ###########     21*9 perché ogni chiamata a join genera 9 tipi di join e 20 prompt per ogni tipo
                
                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = transitive_closure(np.random.choice(closures), np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure])

                questions_preferences, answers_preferences, facts_preferences = preferences(np.random.choice(predicates),predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[0]])

                questions_filtering, answers_filtering, facts_filtering = select_value(np.random.choice(predicates), np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = select_by_numeric_condition(np.random.choice(predicates), prompt_invariance, T21ST)

                chosen = np.random.randint(0, 21)
                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering])
    
                ## start complex 
                       
                chosen = 0    
                
                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_jnf, answers_jnf, facts_jnf = join_numeric_filtering(p_1, p_2, attributes)

                # test_tuples.append([questions_jnf[chosen], answers_jnf[chosen], facts_jnf[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_jneg, answers_jneg, facts_jneg = join_filtering(p_1, p_2, attributes, predicates)

                test_tuples.append([questions_jneg[chosen], answers_jneg[chosen], facts_jneg[chosen]])

                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_cg, answers_cg, facts_cg = closure_guessing(labels, p_1, p_2)

                # test_tuples.append([questions_cg[chosen], answers_cg[chosen], facts_cg[chosen]])

                # questions_cnef, answers_cnef, facts_cnef = closure_negative_filtering(labels, np.random.choice(predicates), np.random.choice(predicates))

                # test_tuples.append([questions_cnef[chosen], answers_cnef[chosen], facts_cnef[chosen]])

                questions_gc, answers_gc, facts_gc = guessing_constraint(labels, np.random.choice(predicates))

                test_tuples.append([questions_gc[chosen], answers_gc[chosen], facts_gc[chosen]])
                # questions_gp, answers_gp, facts_gp = guessing_preference(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gp[chosen], answers_gp[chosen], facts_gp[chosen]])
                # questions_gnef, answers_gnef, facts_gnef = guessing_negative_filtering(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gnef[chosen], answers_gnef[chosen], facts_gnef[chosen]])

                # questions_gnf, answers_gnf, facts_gnf = guessing_numeric_filtering(predicates, np.random.choice(predicates), np.random.choice(attributes), np.random.choice(attributes))

                # test_tuples.append([questions_gnf[chosen], answers_gnf[chosen], facts_gnf[chosen]])

                # questions_gf, answers_gf, facts_gf = guessing_filtering(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gf[chosen], answers_gf[chosen], facts_gf[chosen]])

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                questions_cnf, answers_cnf, facts_cnf = combination_negative_filtering(labels, p_1, p_2, p_3)

                test_tuples.append([questions_cnf[chosen], answers_cnf[chosen], facts_cnf[chosen]])

                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_cnuf, answers_cnuf, facts_cnuf = combination_numeric_filtering(labels, p_1, p_2)

                # test_tuples.append([questions_cnuf[chosen], answers_cnuf[chosen], facts_cnuf[chosen]])

        case "base":
            for i in range(test_size):
                np.random.seed(seed % (i + 1) * 19)
                
                ## start complex 
                       
                chosen = 0    
                
                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_jnf, answers_jnf, facts_jnf = join_numeric_filtering(p_1, p_2, attributes)

                # test_tuples.append([questions_jnf[chosen], answers_jnf[chosen], facts_jnf[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_jneg, answers_jneg, facts_jneg = join_filtering(p_1, p_2, attributes, predicates)

                test_tuples.append([questions_jneg[chosen], answers_jneg[chosen], facts_jneg[chosen]])

                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_cg, answers_cg, facts_cg = closure_guessing(labels, p_1, p_2)

                # test_tuples.append([questions_cg[chosen], answers_cg[chosen], facts_cg[chosen]])

                # questions_cnef, answers_cnef, facts_cnef = closure_negative_filtering(labels, np.random.choice(predicates), np.random.choice(predicates))

                # test_tuples.append([questions_cnef[chosen], answers_cnef[chosen], facts_cnef[chosen]])

                questions_gc, answers_gc, facts_gc = guessing_constraint(labels, np.random.choice(predicates))

                test_tuples.append([questions_gc[chosen], answers_gc[chosen], facts_gc[chosen]])
                
                # questions_gp, answers_gp, facts_gp = guessing_preference(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gp[chosen], answers_gp[chosen], facts_gp[chosen]])
                
                # questions_gnef, answers_gnef, facts_gnef = guessing_negative_filtering(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gnef[chosen], answers_gnef[chosen], facts_gnef[chosen]])

                # questions_gnf, answers_gnf, facts_gnf = guessing_numeric_filtering(predicates, np.random.choice(predicates), np.random.choice(attributes), np.random.choice(attributes))

                # test_tuples.append([questions_gnf[chosen], answers_gnf[chosen], facts_gnf[chosen]])

                # questions_gf, answers_gf, facts_gf = guessing_filtering(labels, np.random.choice(predicates))

                # test_tuples.append([questions_gf[chosen], answers_gf[chosen], facts_gf[chosen]])

                p_1, p_2, p_3 = np.random.choice(predicates, 3, replace=False)
                questions_cnf, answers_cnf, facts_cnf = combination_negative_filtering(labels, p_1, p_2, p_3)

                test_tuples.append([questions_cnf[chosen], answers_cnf[chosen], facts_cnf[chosen]])

                # p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                # questions_cnuf, answers_cnuf, facts_cnuf = combination_numeric_filtering(labels, p_1, p_2)

                # test_tuples.append([questions_cnuf[chosen], answers_cnuf[chosen], facts_cnuf[chosen]])

    return test_tuples


def save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict):
    
    with open(syntactic_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_dict, f)
    
    with open(semantic_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_dict, f)

    with open(syntactic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_proportion_dict, f)

    with open(semantic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_proportion_dict, f)


base_model = "google/gemma-2b-it"

core_model = "gemma-2b-it-core"

invariance_model = "gemma-2b-it-core-invariance"

comples_model = "gemma-2b-it-core-invariance-complex"

output_dir = "toto/"

core_model_path = output_dir + core_model

invariance_model_path = output_dir + invariance_model

complex_model_path = output_dir + comples_model

exhaustive_folder = "exhaustive/"

data_folder = "data/"


compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)



##  CHOOSE BETWEEN [ ' core ' , ' core-invariance ' , ' core-invariance-complex ' ]
##  OR AVERAGED [...]

turn = "averaged_1"               ## "base" per testare i complex

MODEL_TO_USE = "gemma"

DATASET_GENERATION = True
TRAIN = False 
LOAD = True                            # (not TRAIN)       Load for testing
TEST = True                            # if you want to test the model, also on a limited number of prompts
TEST_DATASET_GENERATION = False         # if you need to create a new test set
T21ST = True                           # if you want the test tuple for the core-invariance model to be different from the 20 that the model was trained on
EXHAUSTIVE = False                      # if you want the exhaustive test done directly after the fine-tuning
SHOW_RESULTS = False                    # if you want the results to be shown


match turn:
    case "core":
        model_to_train = base_model
        model_saving_path = core_model_path
        model_to_test = core_model_path

        token = hugging_token

        train_file_name = "data/train_core.csv"
        val_file_name = "data/val_core.csv"

        test_set_file_name  = "data/test_core.csv"
        
        target_modules="all-linear"
        
        tot_size = 100000
        
        len_questions = 2960000
        
        test_size = 1000
        
        results_path = "Core/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "core_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "core_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "core_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "core_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedCore.txt"
        errors_file_name  = exhaustive_folder + "errorsCore.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Core.txt"
        
    case "core-invariance":
        model_to_train = core_model_path
        model_saving_path = invariance_model_path
        model_to_test = invariance_model_path

        token = hugging_token
        
        train_file_name = "data/train_invariance.csv"
        val_file_name = "data/val_invariance.csv"

        test_set_file_name  = "data/test_core_invariance.csv"
        if T21ST:
            test_set_file_name = "data/test_core_invariance_21.csv"
        
        target_modules=None
        
        tot_size = 100
        
        len_questions = 920000
        
        test_size = 1000

        results_path = "Core-Invariance/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "core_invariance_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "core_invariance_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "core_invariance_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "core_invariance_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedCorInv.txt"
        errors_file_name  = exhaustive_folder + "errorsCorInv.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0CorInv.txt"
        
    case "core-invariance-complex":
        model_to_train = invariance_model_path                               
        model_saving_path = complex_model_path
        model_to_test = complex_model_path

        token = hugging_token
        
        train_file_name = "data/train_complex.csv"
        val_file_name = "data/val_complex.csv"

        test_set_file_name  = "data/test_core_invariance_complexNuovo.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3928000
        
        test_size = 500

        results_path = "Complex/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "complex_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "complex_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "complex_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "complex_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedComplex.txt"
        errors_file_name  = exhaustive_folder + "errorsComplex.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Complex.txt"

    case "base":
        model_to_train = base_model                               
        model_saving_path = "toto/base->complex"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplex.csv"
        val_file_name = "data/val_basecomplex.csv"

        test_set_file_name  = "data/test_basecomplex.csv"
        
        target_modules="all-linear"
        
        tot_size = 10000
        
        len_questions = 3456000
        
        test_size = 1000

        results_path = "BaseComplex/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "complex_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "complex_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "complex_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "complex_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedComplex.txt"
        errors_file_name  = exhaustive_folder + "errorsComplex.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Complex.txt"

    case "averaged_1":
        model_to_train = base_model                               
        model_saving_path = "toto/gemma-2b-it-core-invariance-complex-averaged_0.1"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplexxxx.csv"
        val_file_name = "data/val_basecomplexxxx.csv"

        test_set_file_name  = "data/test_basecomplexxxx.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3312000
        
        test_size = 1000

        results_path = "averaged_1/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "averaged_1_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "averaged_1_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "averaged_1_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "averaged_1_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedaveraged_1.txt"
        errors_file_name  = exhaustive_folder + "errorsaveraged_1.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0averaged_1.txt"

    case "averaged_3":
        model_to_train = base_model                               
        model_saving_path = "toto/gemma-2b-it-core-invariance-complex-averaged_0.3"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplexxxx.csv"
        val_file_name = "data/val_basecomplexxxx.csv"

        test_set_file_name  = "data/test_basecomplexxxx.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3312000
        
        test_size = 1000

        results_path = "averaged_1/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "averaged_1_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "averaged_1_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "averaged_1_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "averaged_1_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedaveraged_1.txt"
        errors_file_name  = exhaustive_folder + "errorsaveraged_1.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0averaged_1.txt"

    case "averaged_5":
        model_to_train = base_model                               
        model_saving_path = "toto/gemma-2b-it-core-invariance-complex-averaged_0.5"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplexxxx.csv"
        val_file_name = "data/val_basecomplexxxx.csv"

        test_set_file_name  = "data/test_basecomplexxxx.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3312000
        
        test_size = 1000

        results_path = "averaged_1/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "averaged_1_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "averaged_1_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "averaged_1_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "averaged_1_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedaveraged_1.txt"
        errors_file_name  = exhaustive_folder + "errorsaveraged_1.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0averaged_1.txt"

    case "averaged_7":
        model_to_train = base_model                               
        model_saving_path = "toto/gemma-2b-it-core-invariance-complex-averaged_0.7"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplexxxx.csv"
        val_file_name = "data/val_basecomplexxxx.csv"

        test_set_file_name  = "data/test_basecomplexxxx.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3312000
        
        test_size = 1000

        results_path = "averaged_1/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "averaged_1_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "averaged_1_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "averaged_1_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "averaged_1_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedaveraged_1.txt"
        errors_file_name  = exhaustive_folder + "errorsaveraged_1.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0averaged_1.txt"

    case "averaged_9":
        model_to_train = base_model                               
        model_saving_path = "toto/gemma-2b-it-core-invariance-complex-averaged_0.9"
        model_to_test = model_saving_path

        token = hugging_token
        
        train_file_name = "data/train_basecomplexxxx.csv"
        val_file_name = "data/val_basecomplexxxx.csv"

        test_set_file_name  = "data/test_basecomplexxxx.csv"
        
        target_modules=None
        
        tot_size = 10000
        
        len_questions = 3312000
        
        test_size = 1000

        results_path = "averaged_1/"
        exhaustive_folder += results_path

        syntactic_dict_fn = exhaustive_folder + "averaged_1_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "averaged_1_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "averaged_1_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "averaged_1_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedaveraged_1.txt"
        errors_file_name  = exhaustive_folder + "errorsaveraged_1.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0averaged_1.txt"

    case _:
        print("NON DISPONIBILE !")
        sys.exit(1) 


if DATASET_GENERATION:
    print("generating training set..!")

    size = int(0.8 * tot_size)
    val_size = int(0.2 * tot_size)

    questions, answers = generate_subproblems(size, size, validation=False)
    val_questions, val_answers = generate_subproblems(val_size, size, validation=True)

    print("len questions = ", len(questions))
    print("len answers = ", len(answers))

    len_questions = len(questions)

    d = {"question": questions, "answer": answers}
    val_d = {"question": val_questions, "answer": val_answers}

    train_df = pd.DataFrame(d)
    val_df = pd.DataFrame(val_d)

    train_df.to_csv(train_file_name, index=False)
    val_df.to_csv(val_file_name, index=False)

    # compress_csv(train_file_name)
    # compress_csv(val_file_name)

if TRAIN:
    torch.cuda.empty_cache()

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    print("model to train = ", model_to_train)

    train_df = pd.read_csv(train_file_name)
    val_df = pd.read_csv(val_file_name)

    #   I dataset vengono convertiti nel formato accettato da transformers
    train_dataset = Dataset.from_dict(train_df)
    val_dataset = Dataset.from_dict(val_df)

    print("Training set lenght", train_dataset.num_rows)           ##########################################
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,                                             ################# carico il modello di google e poi ci aggiungerò i pesi lora
        quantization_config=quant_config,
        device_map="auto",      ## prova assegnando GPU
        token=token
        # max_memory={0: "30GB", 1: "30GB", 2: "30GB", 3: "30GB"}
    )

    model.config.use_cache = True
    model.config.pretraining_tp = 1

    if turn != "core" and turn != "base":
        model = PeftModel.from_pretrained(model, model_to_train, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,          #########           GIà 8 è BASSO
        gradient_accumulation_steps=1,
        max_steps=200,                          #########           non farà mai un'epoca intera se imposto max steps
        logging_steps=5,
        eval_steps=5,
        do_train=True,
        do_eval=True,
        save_strategy='no',
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    if MODEL_TO_USE == "llama":

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_params,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=tokenizer,
            args=training_params,
            packing=False,
        )

    elif MODEL_TO_USE == "gemma":

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_params,
            max_seq_length=None,
            tokenizer=tokenizer,
            args=training_params,
            packing=False,
            formatting_func=formatting_func
        )

    trainer.train()

    # merged_model = model.merge_and_unload()               # USATO PER SALVARE IL MODELLO COMPLETO E NON SOLO I PESI LORA
    # mergerd_model.save_pretrained()

    trainer.model.save_pretrained(model_saving_path)
    trainer.tokenizer.save_pretrained(model_saving_path)
 
    print("model tuned in -> ", model_saving_path)

if LOAD:
    print("model to test = ", model_to_test)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",      
        token=token
    )
    
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    
    model = PeftModel.from_pretrained(model, model_to_test, is_trainable=True)      ###  NECESSARIO is_trainable PER RIADDESTRARE UN MODELLO PEFT ALTRIMENTI I PESI SAREBBERO FREEZED
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

if TEST:
    print("model to test = ", model_to_test)

    if TEST_DATASET_GENERATION:
        print("generating test set...")

        test_tuples = build_test_set()

        print("TEST SET SIZE: ", len(test_tuples))

        test_df = pd.DataFrame(test_tuples, columns=["prompt", "answer", "fact"])

        test_df.to_csv(test_set_file_name , index=False)

        compress_csv(test_set_file_name)

    if EXHAUSTIVE:
        if not TEST_DATASET_GENERATION:     # leggo da file solo se non ho appena generato
            test_df = pd.read_csv(test_set_file_name)
            test_tuples = test_df.to_numpy()

        print("file_name for the test: ", test_set_file_name )
        print("n_domande ->", len(test_tuples))
        
        # definizione tipi di problemi
        problems = ["assignment", "constraint", "combination", "join", "closure", "preference", "filtering", "negative_filtering", "numeric_filtering"]
        if turn =="core-invariance-complex":
            # problems.extend(["join_numeric_filtering", "join_filtering", "closure_guessing", "closure_negative_filtering", "guessing_constraint", "guessing_preference", "guessing_negative_filtering", "guessing_numeric_filtering", "guessing_filtering", "combination_negative_filtering", "combination_numeric_filtering"])
            problems.extend(["join_filtering", "guessing_constraint", "combination_negative_filtering"])
        if turn == "base":
            problems = ["join_filtering", "guessing_constraint", "combination_negative_filtering"]
        

        problems_index_dict = dict(zip(range(0, len(problems)), problems))
        problems_count_dict = dict.fromkeys(range(0, len(problems)), 0)
        problems_syntactic_dict = dict.fromkeys(range(0, len(problems)), 0)
        problems_semantic_dict = dict.fromkeys(range(0, len(problems)), 0)
        problems_syntactic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)
        problems_semantic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)


        # inizializzazione files debugging
        with open(errors_file_name, 'w') as p:
                p.write("\n")

        with open(parsed_file_name, 'w') as r:
                r.write("\n")

        with open(jaccard0_file_name, 'w') as j:
                j.write("\n")

        for i, (q, a, f) in enumerate(tqdm(test_tuples, total=len(test_tuples))):

            generated_a = generate_response(q)

            # Rimozione di "Answer" dalla risposta
            if "Answer" in generated_a:
                parsed_generated_a = generated_a[generated_a.index("Answer"):].replace("Answer: ", "").replace("`", "")
            else:
                parsed_generated_a=generated_a

            index = i % len(problems)
        
            problems_count_dict[index] += 1

            # Lista per mantenere l'ordine
            unique_rules = []
            seen = set()
            
            for line in parsed_generated_a.splitlines():
                line = line.strip()
                if (":-" in line or ":~" in line) and line not in seen:
                    unique_rules.append(line)  # Aggiungi la riga alla lista
                    seen.add(line)  # Segna la riga come già vista

            # Converti la lista in una stringa con newline
            parsed_generated_a = "\n".join(unique_rules)
            
            if problems_index_dict[index] == "closure":
                parsed_generated_a = '\n'.join(parsed_generated_a.split("\n")[:2])
            elif problems_index_dict[index] == "preference":
                parsed_generated_a = parsed_generated_a.split("\n")[0]
            elif problems_index_dict[index] == "join_numeric_filtering":
                parsed_generated_a = "".join(parsed_generated_a.split("\n")[:1])
            elif problems_index_dict[index] == "join_filtering":
                parsed_generated_a = "".join(parsed_generated_a.split("\n")[:2])
            elif problems_index_dict[index] == "closure_guessing":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:3])
            elif problems_index_dict[index] == "guessing_constraint":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
            elif problems_index_dict[index] == "guessing_negative_filtering":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
            elif problems_index_dict[index] == "guessing_numeric_filtering":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
            elif problems_index_dict[index] == "guessing_filtering":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
            elif problems_index_dict[index] == "combination_negative_filtering":
                parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
            else:
                parsed_generated_a = parsed_generated_a.split("\n")[0]  


            with open(parsed_file_name, 'a') as r:
                r.write(str(i))
                r.write("\n")
                r.write(str(problems_index_dict[index]))
                r.write("\n\nquestion: \n")
                r.write(q)
                r.write("\n\nanswer from file: \n")
                r.write(a)
                r.write("\n\nparsed from model: \n")
                r.write(parsed_generated_a)
                r.write("\n\nfacts: \n")
                r.write(f)
                r.write("\n\ngenerated: \n")
                r.write(generated_a)
                r.write("\n\nunique_rules: \n")
                r.write(str(unique_rules))


            answer_set = gen_answer_set(a + f)
            generated_answer_set = gen_answer_set(parsed_generated_a + f)

            if len(generated_answer_set) == 0:
                problems_syntactic_dict[index] += 1
                problems_syntactic_proportion_dict[index] = problems_syntactic_dict[index] / problems_count_dict[index]
            else:
                if generated_answer_set[0] != "error":  # syntactic check did not fail
                    problems_syntactic_dict[index] += 1
                    problems_syntactic_proportion_dict[index] = problems_syntactic_dict[index] / problems_count_dict[index]
                else:
                    with open(errors_file_name, 'a') as p:
                        p.write("i: ")
                        p.write(str(i))
                        p.write("\n\nindex: ")
                        p.write(str(index))
                        p.write("\n\n")
                        p.write(str(problems_index_dict[index]))
                        p.write("\n\nquestion: ")
                        p.write(q)
                        p.write("\n\nanswer from file: ")
                        p.write(a)
                        p.write("\n\nfacts: \n")
                        p.write(f)
                        p.write("\n\ngenerated_answer: ")
                        p.write(generated_a)
                        p.write("\n\nparsed answer: ")
                        p.write(parsed_generated_a)
                        p.write("\n\nanswerset from file: ")
                        p.write(str(answer_set))
                        p.write("\n\nanswerset from parsed: ")
                        p.write(str(generated_answer_set))
                        p.write("\n\n")

                jaccard = check_semantics(answer_set, generated_answer_set)
                if jaccard == 1.:
                    problems_semantic_dict[index] += 1
                    problems_semantic_proportion_dict[index] = problems_semantic_dict[index] / problems_count_dict[index]
                else :
                    with open(jaccard0_file_name, 'a') as r:
                        r.write("i: ")
                        r.write(str(i))
                        r.write("\n\nindex: ")
                        r.write(str(index))
                        r.write("\n\n")
                        r.write(str(problems_index_dict[index]))
                        r.write("\n\nquestion: ")
                        r.write(q)
                        r.write("\n\nanswer from file: ")
                        r.write(a)
                        r.write("\n\nfacts: \n")
                        r.write(f)
                        r.write("\n\ngenerated: \n")
                        r.write(generated_a)
                        # r.write("\n\nunique_rules: \n")
                        # r.write(str(unique_rules))
                        r.write("\n\nparsed: \n")
                        r.write(parsed_generated_a)
                        r.write("\n\nwanted answer_Set: ")
                        r.write(str(answer_set))
                        r.write("\n\ngenerated answer_Set: ")
                        r.write(str(generated_answer_set))
                        r.write("\n\njaccard: ")
                        r.write(str(jaccard))
                        r.write("\n\n\n")    
                
                with open(parsed_file_name, 'a') as r:
                    r.write("\n\njaccard: ")
                    r.write(str(jaccard))
                    r.write("\n\nAS desired:\t")
                    r.write(str(answer_set))
                    r.write("\nAS obtained:\t")
                    r.write(str(generated_answer_set))
                    r.write("\n\n\n")

            
        print("Final saving")
        save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict)
    
    else:
        generate_test_cases()

if SHOW_RESULTS:
    files = [syntactic_dict_fn, syntactic_prop_dict_fn, semantic_dict_fn, semantic_prop_dict_fn]

    for f in files:
        print(f)
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
            print(data)
            print("\n")
        except FileNotFoundError:
            print(f"File not found: {f}")
        except pickle.UnpicklingError:
            print("Error: The file content is not a valid pickle format.")
        except EOFError:
            print("Error: The file is incomplete or corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    ended = datetime.now()

    print("test ended -> ")
    print(ended.strftime("%d-%m-%Y %H:%M:%S"))


