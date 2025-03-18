import os
from tqdm import tqdm

import sys
import subprocess

import torch
import pandas as pd

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)

from clingo.control import Control
from clingo.symbol import parse_term

import pickle

from peft import LoraConfig, PeftModel, get_peft_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  ################################################################### Set to use GPUs 0 and 1


hugging_token = "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"

from huggingface_hub import login

login(hugging_token)

torch.manual_seed(56)

torch.cuda.empty_cache()

print("device_count: ", torch.cuda.device_count())

class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret


def gen_answer_set(program, opt=False):
    """
        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
    clingo_control = Control(['1', '--warn=none', '--opt-mode=optN', '-t', '4'])
    models = []
    try:
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])], context=Context())
    except Exception as e:
        # breakpoint()
        return ["error"]
    if opt:
        clingo_control.solve(
            on_model=lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
    else:
        clingo_control.solve(on_model=lambda model: models.append(model.symbols(atoms=True)))
    models = [[str(atom) for atom in model] for model in models]
    if(len(models)==0):
        print("OOOOOOOOOOOOOOPS!")
    return models

base_model = "google/gemma-2b-it"

core_model = "gemma-2b-it-core"

invariance_model = "gemma-2b-it-invariance"

comples_model = "gemma-2b-it-core-invariance-complex"

output_dir = "toto/"

core_model_path = output_dir + core_model

invariance_model_path = output_dir + invariance_model

complex_model_path = output_dir + comples_model

exhaustive_folder = "exhaustive/"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

##############      FINETUNANDO DIVERSI LIVELLI DI MODELLO CON PEFT è NECESSARIO CARICARE IL MODELLO BASE PERCHè PEFT NON SALVA L'INTERO MODELLO MA SOLO I PESI AGGIORNATI
    ##############      QUINDI CARICO IL MODELLO BASE CON AUTOMODEL... E POI CARICO IL NUOVO CON PEFTMODEL
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

##  CHOOSE BETWEEN [ ' core ' , ' core-invariance ' , ' comples ' ]

turn = "core"

match turn:
    case "core":
        model_name = "gemma-2b-it-core"
    
        file_name = "data/test_core.csv"
        
        model_to_test = core_model_path
        
        exhaustive_folder += "Core/"

        syntactic_dict_fn = exhaustive_folder + "core_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "core_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "core_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "core_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedCore.txt"
        errors_file_name  = exhaustive_folder + "errorsCore.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Core.txt"
    case "invariance":
        model_name = "gemma-2b-it-invariance"
    
        file_name = "data/test_invariance.csv"
        
        model_to_test = invariance_model_path

        exhaustive_folder += "InvarianceOnly/"
        
        syntactic_dict_fn = exhaustive_folder + "invariance_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "invariance_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "invariance_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "invariance_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedInv.txt"
        errors_file_name  = exhaustive_folder + "errorsInv.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Inv.txt"
    case "core-invariance":
        model_name = "gemma-2b-it-invariance"
        
        file_name = "data/test_core_invariance.csv"
        
        model_to_test = invariance_model_path
        # model_to_test = "toto/gemma-2b-it-invariance-2"
        
        exhaustive_folder += "Core-Invariance/"
        
        syntactic_dict_fn = exhaustive_folder + "core_invariance_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "core_invariance_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "core_invariance_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "core_invariance_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedCorInv.txt"
        errors_file_name  = exhaustive_folder + "errorsCorInv.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0CorInv.txt"
    case "comples":
        model_name = "gemma-2b-it-core-invariance-complex"
        
        file_name = "data/test_comples.csv"
        
        model_to_test = complex_model_path
        
        exhaustive_folder += "Complex/"
        
        syntactic_dict_fn = exhaustive_folder + "complex_syntactic_test_scores_dict.pkl"
        semantic_dict_fn = exhaustive_folder + "complex_semantic_test_scores_dict.pkl"
        syntactic_prop_dict_fn = exhaustive_folder + "complex_syntactic_prop_test_scores_dict.pkl"
        semantic_prop_dict_fn = exhaustive_folder + "complex_semantic_prop_test_scores_dict.pkl"

        parsed_file_name = exhaustive_folder + "parsedComplex.txt"
        errors_file_name  = exhaustive_folder + "errorsComplex.txt"
        jaccard0_file_name = exhaustive_folder + "jaccard0Complex.txt"

model = PeftModel.from_pretrained(model, model_to_test, is_trainable=True)      ###  NECESSARIO is_trainable PER RIADDESTRARE UN MODELLO PEFT ALTRIMENTI I PESI SAREBBERO FREEZED

tokenizer = AutoTokenizer.from_pretrained(model_to_test)

model.config.use_cache = True
model.config.pretraining_tp = 1
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

test_df = pd.read_csv(file_name)
test_tuples = test_df.to_numpy()

print("file_name for the test: ", file_name)
print("n_domande ->", len(test_tuples))


def generate_response(question):
    inputs_device = model.device
    inputs = tokenizer(question, return_tensors="pt").to(inputs_device)         #       DA STRINGA A TENSORE DI PYTORCH PER ESSERE COMPATIBILE COL MODELLO, CON .to() SI PASSA AL DISPOSITIVO

    outputs = model.generate(**inputs, max_new_tokens=150)                      #       GENERA LA RISPOSTA DEL MODELLO, CON LUNGHEZZA LIMITATA
    return tokenizer.decode(outputs[0], skip_special_tokens=True)               #       DECODIFICA IL TENSORE IN STRINGA, RIMUOVENDO TOKEN SPECIALI


def check_semantics(correct_models, generated_models):
    set_correct_models = set(map(frozenset, correct_models))
    set_generated_models = set(map(frozenset, generated_models))

    jaccard = len(set_correct_models.intersection(set_generated_models)) / len(set_correct_models.union(set_generated_models))
    return jaccard


def save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict):
    with open(syntactic_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_dict, f)

    with open(semantic_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_dict, f)

    with open(syntactic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_proportion_dict, f)

    with open(semantic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_proportion_dict, f)


# definizione tipi di problemi
if turn != "comples":
    problems = ["assignment", "constraint", "combination", "join", "closure", "preference", "filtering", "negative_filtering", "numeric_filtering"]
else:
    problems = ["join_numeric_filtering", "join_filtering", "closure_guessing", "guessing_constraint", "guessing_numeric_filtering", "guessing_filtering", "combination_constraint"]


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

    # Regole di parsing in base al modello
    if turn != "comples":
        if problems_index_dict[index] == "closure":
            parsed_generated_a = '.'.join(parsed_generated_a.split(".")[:2]) + "."
        elif problems_index_dict[index] == "preference":
            parsed_generated_a = parsed_generated_a.split("]")[0] + "]"
        else:
            parsed_generated_a = parsed_generated_a.split(".")[0] + "."
    else:
        # Lista per mantenere l'ordine
        unique_rules = []
        seen = set()  

        for line in parsed_generated_a.splitlines():
            line = line.strip()
            if (":-" in line or ":~" in line) and line not in seen:
                line = line.rstrip(".")+"."
                unique_rules.append(line)  # Aggiungi la riga alla lista
                seen.add(line)  # Segna la riga come già vista

        # Converti la lista in una stringa con newline
        parsed_generated_a = "\n".join(unique_rules)
        
        if problems_index_dict[index] == "join_numeric_filtering":
            parsed_generated_a = "".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "join_filtering":
            parsed_generated_a = "".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "closure_guessing":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:3])
        elif problems_index_dict[index] == "guessing_constraint":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "guessing_numeric_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "guessing_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
        elif problems_index_dict[index] == "combination_constraint":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])

        with open(parsed_file_name, 'a') as r:
            r.write(str(i))
            r.write("\n")
            r.write(str(problems_index_dict[index]))
            r.write("\n\nquestion: \n")
            r.write(q)
            r.write("\n\nanswer from file: \n")
            r.write(a)
            r.write("\n\ngenerated: \n")
            r.write(generated_a)
            r.write("\n\nunique_rules: \n")
            r.write(str(unique_rules))
            r.write("\n\nparsed: \n")
            r.write(parsed_generated_a)
            r.write("\n\n\n")


   
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
            # print("ERROR on the following tuples")
            # print(generated_a)
            # print(parsed_generated_a)
            print("******")
            with open(errors_file_name, 'a') as p:
                p.write("i: ")
                p.write(str(i))
                p.write("\n\nindex: ")
                p.write(str(index))
                r.write("\n\n")
                r.write(str(problems_index_dict[index]))
                p.write("\n\nquestion: ")
                p.write(q)
                p.write("\n\nanswer from file: ")
                p.write(a)
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
                p.write("\n\nindex: ")
                p.write(str(index))
                r.write("\n\n")
                r.write(str(problems_index_dict[index]))
                p.write("\n\nquestion: ")
                p.write(q)
                p.write("\n\nanswer from file: ")
                p.write(a)
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
            r.write("facts: \n")
            r.write(f)
            r.write("\n\nwanted answer_Set: \n")
            r.write(str(answer_set))
            r.write("\n\ngenerated answer_Set: \n")
            r.write(str(generated_answer_set))
            r.write("\n\njaccard: ")
            r.write(str(jaccard))
            r.write("\n\n\n")

    if i > 0 and not i % 100:
        print(f"Test Dictionaries saved at step {i}")
        save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict)

print("Final saving")
save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict)




########################                APERTURA FILE DI TEST                   ###########################################

match turn:
    case "core":
        file_path1 = 'core_semantic_prop_test_scores_dict.pkl'
        file_path2 = 'core_semantic_test_scores_dict.pkl'
        file_path3 = 'core_syntactic_prop_test_scores_dict.pkl'
        file_path4 = 'core_syntactic_test_scores_dict.pkl'
        path = "Core/"
    case "invariance":
        file_path1 = 'invariance_semantic_prop_test_scores_dict.pkl'
        file_path2 = 'invariance_semantic_test_scores_dict.pkl'
        file_path3 = 'invariance_syntactic_prop_test_scores_dict.pkl'
        file_path4 = 'invariance_syntactic_test_scores_dict.pkl'
        path = "InvarianceOnly/"
    case "core-invariance":
        file_path1 = "core_invariance_semantic_prop_test_scores_dict.pkl"
        file_path2 = "core_invariance_semantic_test_scores_dict.pkl"
        file_path3 = "core_invariance_syntactic_prop_test_scores_dict.pkl"
        file_path4 = "core_invariance_syntactic_test_scores_dict.pkl"
        path = "Core-Invariance/"
    case "comples":
        file_path1 = "complex_semantic_prop_test_scores_dict.pkl"
        file_path2 = "complex_semantic_test_scores_dict.pkl"
        file_path3 = "complex_syntactic_prop_test_scores_dict.pkl"
        file_path4 = "complex_syntactic_test_scores_dict.pkl"
        path = "Complex/"

path = "exhaustive/" + path

files = [path + file_path1, path + file_path2, path + file_path3, path + file_path4]

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