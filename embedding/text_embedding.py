import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
base_path = './enrico_corpus/texts/'
out_path = './enrico_corpus/'


# load the pkl file
with open(os.path.join(out_path, 'vectors.pkl'), 'rb') as f:
    out_data = pickle.load(f)

count = 0
# check if the vectors are zero vectors
for data in out_data:
    vector = data['vector']
    if np.sum(vector) == 0:
        count += 1

print(count)