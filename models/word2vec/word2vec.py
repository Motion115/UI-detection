import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
base_path = './enrico_corpus/texts_curated/'
out_path = './enrico_corpus/'

import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use

out_data = []
# get all the files in base_path
files = os.listdir(base_path)
for file in tqdm(files):
    df = pd.read_csv(os.path.join(base_path, file))
    # read the column text into a list
    texts = df['text'].tolist()
    # for each text, transform to str
    texts = [str(text) for text in texts]
    # open up the lists in texts and merge into a single list
    texts = [word for text in texts for word in text]
    # deduplicate the list
    texts = list(set(texts))
    # create a 100 by 1 zero vector
    vector = np.zeros((1, 100))
    total_texts = len(texts)
    if total_texts == 0:
        out_data.append({'id': file.split(".")[0], 'nlp_embedding': vector})
    else:
        # for each text, get the vector
        for text in texts:
            try:
                embedding = model[text]
                # add the vector to the zero vector
                vector += embedding.reshape((1, 100))
            except:
                total_texts -= 1
        
        # get the average vector
        if total_texts != 0:
            # calculate the mean vector
            vector /= total_texts
        out_data.append({'id': file.split(".")[0], 'nlp_embedding': vector})

# save the out_data as pkl file
with open(os.path.join(out_path, "nlp_embedding.pkl"), 'wb') as f:
    pickle.dump(out_data, f)
