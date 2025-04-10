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
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, inject_adapter_in_model
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



base_model = "google/gemma-2b-it"
model_core_invariance_complex_path = "toto/gemma-2b-it-core-invariance-complex"
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

target_modules=None

loraConfig = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

token = "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"
login(token)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    token=token
)


model_core_invariance = PeftModel.from_pretrained(model, model_core_invariance_path, is_trainable=False)
model_core_invariance = inject_adapter_in_model(loraConfig, model_core_invariance)

model_complex = PeftModel.from_pretrained(model, model_complex_path, is_trainable=False)
model_complex = inject_adapter_in_model(loraConfig, model_complex)


Alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

## per ogni alpha cosstruisco il dizionario dei pesi (quindi medie pesate in diverso modo)

sd1 = get_peft_model_state_dict(model_core_invariance)
# sd1 = model_core_invariance.state_dict()
# sd2 = model_complex.state_dict()
sd2 = get_peft_model_state_dict(model_complex)

model_dict = model.state_dict()

assert sd1.keys() == sd2.keys(), "Model architecture should be the same"

for alpha in Alphas:
	saving_path = data_folder + f"gemma-2b-it-core-invariance-complex-averaged_{alpha}"
	model_core_invariance_complex_weight = OrderedDict()
	for key in sd1:
		model_core_invariance_complex_weight[key] = alpha * sd1[key] + (1 - alpha) * sd2[key] ## state_dict prende l'intero stato del modello, quindi i pesi di ogni layer dell'architettura
	model_core_invariance_complex = PeftModel.from_pretrained(model, model_core_invariance_path, is_trainable=False)
	model_core_invariance_complex = inject_adapter_in_model(loraConfig, model_core_invariance_complex)
	missing_keys, unexpected_keys = set_peft_model_state_dict(model_core_invariance_complex, model_core_invariance_complex_weight)     # lo carico sul modello
	for key in missing_keys:
		model_core_invariance_complex_weight[key]=model_dict[key.replace("base_model.model.", "")]
	missing_keys, unexpected_keys = set_peft_model_state_dict(model_core_invariance_complex, model_core_invariance_complex_weight)     # lo carico sul modello
	assert len(missing_keys) == len(unexpected_keys) == 0
	model_core_invariance_complex.save_pretrained(saving_path)
