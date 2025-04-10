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

model_to_test = "toto/gemma-2b-it-core-invariance-complex-averaged_0.1"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    token=token
)

model.config.use_cache = True
model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(model, model_to_test, is_trainable=False)

tokenizer = AutoTokenizer.from_pretrained(base_model)
    
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"