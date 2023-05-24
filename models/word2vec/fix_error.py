import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import difflib
base_path = './enrico_corpus/texts/'
out_path = './enrico_corpus/texts_curated/'

# read existing_words.pkl
with open(os.path.join("./models/word2vec/", 'existing_words.pkl'), 'rb') as f:
    existing_words = pickle.load(f)

def calculate(): 
    # read csv files in base_path
    files = os.listdir(base_path)
    # sort files for error checking and mid-run recovery
    files.sort()
    for file in tqdm(files):
        df = pd.read_csv(os.path.join(base_path, file))
        # traverse through the column text
        for i, texts in enumerate(df['text']):
            texts = str(texts)
            # first, split the text into words
            texts = texts.split()
            # second, transform all to lowercases
            texts = [text.lower() for text in texts]
            # third, strip all the punctuations
            texts = [text.strip('.,?!;-:\'\"[]()_') for text in texts]
            # fourth, remove empty strings
            texts = [text for text in texts if text != '']

            # judge if the word is in existing_words
            for j in range(len(texts)):
                word = texts[j]
                if word not in existing_words:
                    # find the most similar word
                    most_similar_word = difflib.get_close_matches(word, existing_words, n=1)
                    if len(most_similar_word) == 0:
                        # if no similar word is found, replace with ''
                        most_similar_word = ''
                    else:
                        texts[j] = most_similar_word[0]

            # fifth, join the words back into a single string
            text = ' '.join(texts)
            # sixth, replace the text with the new text
            df.loc[i, 'text'] = text

        df.to_csv(os.path.join(out_path, file), index=False)

calculate()