from AgentRec import AgentRec
from util.conf import ModelConf
import time
import os
import random
import numpy as np
import torch

def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    seedSet("2024")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    models = {
        'Graph-Based Baseline Models': ['RuleAgent', 'MF']
    }

    print('=' * 80)
    print('   AgentRec: A library for recommendation.   ')
    print_models("Available Models", models)

    # model = input('Please enter the model you want to run:')
    model = 'RuleAgent'

    s = time.time()
    all_models = sum(models.values(), [])
    if model in all_models:
        conf = ModelConf(f'./conf/{model}.yaml')
        rec = AgentRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
