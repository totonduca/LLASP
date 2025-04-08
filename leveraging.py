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
from collections import OrderedDict


model_core_invariance_path = "toto/gemma-2b-it-core-invariance"
model_complex_path = "toto/base->complex"

data_folder = "toto/"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


model_core_invariance = AutoModelForCausalLM.from_pretrained(
    model_core_invariance_path,
    quantization_config=quant_config,
    device_map="auto"
)

model_complex = AutoModelForCausalLM.from_pretrained(
    model_complex_path,
    quantization_config=quant_config,
    device_map="auto"
)

model_core_invariance_complex = AutoModelForCausalLM.from_pretrained(
    model_complex_path,
    quantization_config=quant_config,
    device_map="auto"
)

assert model_core_invariance.state_dict().keys() == model_complex.state_dict().keys(), "Model architecture should be the same"

Alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

## per ogni alpha cosstruisco il dizionario dei pesi (quindi medie pesate in diverso modo)

for alpha in Alphas:
	saving_path = data_folder + f"gemma-2b-it-core-invariance-complex-averaged_{alpha}"
	model_core_invariance_complex_weight = {}
	for key in model_core_invariance.state_dict():
		model_core_invariance_complex_weight[key] = alpha * model_core_invariance.state_dict()[key] + (1 - alpha) * model_complex.state_dict()[key] ## state_dict prende l'intero stato del modello, quindi i pesi di ogni layer dell'architettura
	model_core_invariance_complex.load_state_dict(model_core_invariance_complex_weight)     # lo carico sul modello
	model_core_invariance_complex.save_pretrained(saving_path)
