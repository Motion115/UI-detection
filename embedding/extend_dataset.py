import pickle
import os
import numpy as np
from tqdm import tqdm

pkl_file = os.path.join("./enrico_corpus/enrico_embedding.pkl")
#read the pkl file
with open(pkl_file, 'rb') as f:
    corpus = pickle.load(f)
        
# split the corpus according to the "label_id" value in the corpus
# create a dictionary to store the split corpus
corpus_dict = {}
for element in corpus:
    label_id = element['label_id']
    if label_id not in corpus_dict:
        corpus_dict[label_id] = []
    corpus_dict[label_id].append(element)

expanded_corpus = []
for key, value in tqdm(corpus_dict.items()):
    # iteratively go through the val in value list, trace back when the total count is less than 1000
    count = 0
    indexer, total_count = 0, len(value)
    while True:
        if count < 1000:
            # get the element by indexer
            element = value[indexer]
            anchor_embedding = element['embedding']
            # get positive embedding, randaomly choose an index from the list not itself
            positive_index = np.random.choice([i for i in range(total_count) if i != indexer])
            positive_embedding = value[positive_index]['embedding']
            # get negative embedding, randomly choose a key from the corpus_dict
            # first, randomly choose a key not the current key
            negative_key = np.random.choice([k for k in corpus_dict.keys() if k != key])
            # second, randomly choose an index from the list
            negative_index = np.random.choice([i for i in range(len(corpus_dict[negative_key]))])
            negative_embedding = corpus_dict[negative_key][negative_index]['embedding']
            choice_str = str(key) + "_" + str(indexer) + "_" + str(positive_index) + "_" + str(negative_key) + "_" + str(negative_index)
            # append to the list
            expanded_corpus.append({"class": str(key) + "-" + str(count), "choice":choice_str, "anchor_embedding": anchor_embedding, "positive_embedding": positive_embedding, "negative_embedding": negative_embedding})
            if indexer == total_count - 1:
                indexer = 0
            else:
                indexer += 1
            count += 1
        else:
            break

# save expanded_corpus as pkl file
with open(os.path.join("./enrico_corpus/enrico_expanded_embedding.pkl"), 'wb') as f:
    pickle.dump(expanded_corpus, f)

