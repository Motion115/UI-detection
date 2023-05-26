from enrico_utils.get_embedding_data import EnricoInferDataset
from models.modalfusion.fuse import Fusion
import torch
import pickle
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = './weights/fusemodel/fuse_epoch_111.ckpt'
checkpoint = torch.load(best_model)
net = Fusion()
net.to(device)
net.load_state_dict(checkpoint['net'])

torch.no_grad()
# torch set to eval mode
net.eval()

# load data
pkl_file = os.path.join("./enrico_corpus/enrico_embedding.pkl")
with open(pkl_file, 'rb') as f:
    corpus = pickle.load(f)

embed_150_list = []
 
for i in tqdm(range(len(corpus))):
    id = corpus[i]['id']
    label_id = corpus[i]['label_id']
    label = corpus[i]['label']
    embedding = corpus[i]['embedding']
    embedding = embedding.to(device)
    embedding_150 = net(embedding)
    embedding_numpy = embedding_150.cpu().detach().numpy()
    embed_150_list.append({"id":id, "label_id": label_id, "label": label, "numpy_embedding": embedding_numpy})

# save embed_150_list as pkl file
with open(os.path.join("./enrico_corpus/enrico_embedding_150.pkl"), 'wb') as f:
    pickle.dump(embed_150_list, f)


