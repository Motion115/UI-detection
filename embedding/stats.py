import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
out_path = './enrico_corpus/'

# load the pkl file
with open(os.path.join(out_path, 'enrico_expanded_embedding.pkl'), 'rb') as f:
    fullset = pickle.load(f)

print(fullset[0])
