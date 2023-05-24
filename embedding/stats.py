import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
out_path = './enrico_corpus/'

# load the pkl file
with open(os.path.join(out_path, 'enrico_embedding_corpus.pkl'), 'rb') as f:
    corpus = pickle.load(f)

# calculate the label_id count
label_id_count = corpus['label_id'].value_counts()
print(label_id_count)
