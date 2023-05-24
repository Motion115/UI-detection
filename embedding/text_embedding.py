import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
base_path = './enrico_corpus/texts/'
out_path = './enrico_corpus/'

# load the pkl file
with open(os.path.join(out_path, 'nlp_embedding.pkl'), 'rb') as f:
    nlp_embedding = pickle.load(f)

# load the pkl file
with open("./enrico_corpus/vision_embedding.pkl", 'rb') as f:
    cv_embedding = pickle.load(f)


# merge the two dataframes
df = pd.merge(nlp_embedding, cv_embedding, on='id')
# rename the columns
df = df.rename(columns={"vector": "nlp_embedding", "embedding": "cv_embedding"})
# move lable to the last column
df = df[['id', 'nlp_embedding', 'cv_embedding', 'label_id', 'label']]
# save the dataframe as pkl file
df.to_pickle(os.path.join(out_path, "enrico_embedding_corpus.pkl"))
print(df)
