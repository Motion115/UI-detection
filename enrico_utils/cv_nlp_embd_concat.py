import os
import numpy as np
import pickle
import torch
from tqdm import tqdm

source_path = "./enrico_corpus/"
out_path = './enrico_corpus/868-embds/'
config = {
    'in_data': "nlp_only"
}
# modes include all, cv_only, nlp_only

# load the pkl file
with open(os.path.join(source_path, 'nlp_embedding.pkl'), 'rb') as f:
    nlp_embedding = pickle.load(f)

# load the pkl file
with open(os.path.join(source_path, 'vision_embedding.pkl'), 'rb') as f:
    cv_embedding = pickle.load(f)

fullset = []
# cv_embedding is a list of dict, nlp_embdding is also a list of dict
# since cv_embedding is smaller, we loop through cv_embedding
for i in tqdm(range(len(cv_embedding))):
    # get the id
    id = cv_embedding[i]['id']
    # get the label_id
    label_id = cv_embedding[i]['label_id']
    # get the label
    label = cv_embedding[i]['label']
    # get the embedding
    embedding = cv_embedding[i]['embedding']
    # normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    if config['in_data'] == 'nlp_only':
        # clear all content in embedding to make it a zero vector
        embedding = np.zeros((1, embedding.shape[1]))
    # get the nlp_embedding
    if config['in_data'] == 'cv_only':
        nlp = np.zeros((1, 100))
    else:
        # find in nlp_embedding dict with the same id
        for item in nlp_embedding:
            if item['id'] == id:
                nlp = item['nlp_embedding']
                # judge if nlp is a zero vector
                if np.linalg.norm(nlp) == 0:
                    nlp = nlp
                else:
                    # normalize the nlp embedding
                    nlp = nlp / np.linalg.norm(nlp)
                    #nlp = nlp
                break
    # merge the two embeddings
    embedding = np.concatenate((embedding, nlp), axis=1)
    # embedding to torch tensor
    embedding = torch.from_numpy(embedding)
    # set embedding require grad
    embedding.requires_grad = True
    # set embedding to float
    embedding = embedding.float()
    # append to fullset
    fullset.append({"id": id, "label_id": label_id, "label": label, "embedding": embedding})


if config['in_data'] == 'cv_only':
    # save fullset as pkl file
    with open(os.path.join(out_path, 'enrico_embedding_cv.pkl'), 'wb') as f:
        pickle.dump(fullset, f)
elif config['in_data'] == 'nlp_only':
    # save fullset as pkl file
    with open(os.path.join(out_path, 'enrico_embedding_nlp.pkl'), 'wb') as f:
        pickle.dump(fullset, f)
else:
    # save fullset as pkl file
    with open(os.path.join(out_path, 'enrico_embedding.pkl'), 'wb') as f:
        pickle.dump(fullset, f)