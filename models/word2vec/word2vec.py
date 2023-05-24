import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
base_path = './enrico_corpus/texts/'
out_path = './enrico_corpus/'

import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use
# traverse through the model
for i, word in enumerate(model.key_to_index):
    print(word)
    if i == 10:
        break
'''
out_data = []

count = 0
# get all the files in base_path
files = os.listdir(base_path)
for file in tqdm(files):
    df = pd.read_csv(os.path.join(base_path, file))
    # read the column text into a list
    texts = df['text'].tolist()
    # for each text, transform to str
    texts = [str(text) for text in texts]
    # for each text, split it into words, and transform all to lowercases
    texts = [text.lower().split() for text in texts]
    # open up the lists in texts and merge into a single list
    texts = [word for text in texts for word in text]
    # deduplicate the list
    texts = list(set(texts))
    # strip all the punctuations
    texts = [text.strip('.,?!;-:\'\"[]()_') for text in texts]
    # create a 100 by 1 zero vector
    vector = np.zeros((100, 1))
    total_texts = len(texts)
    if total_texts == 0:
        out_data.append({'id': file.strip(".")[0], 'vector': vector})
    else:
        # for each text, get the vector
        for text in texts:
            try:
                model[text]
            except:
                total_texts -= 1
                count += 1
                print('word not found: ' + text)
        # get the average vector
        if total_texts != 0:
            vector = vector / total_texts
        out_data.append({'id': file.strip(".")[0], 'vector': vector})

print(count)
# outdata to pkl file

with open(os.path.join(out_path, 'vectors.pkl'), 'wb') as f:
    pickle.dump(out_data, f)

'''