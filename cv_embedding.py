from enrico_utils.get_data import EnricoDataset
from models.cv.vgg import VGG
import torch
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os

dataset = EnricoDataset("enrico_corpus", mode='all', img_dim_x=256, img_dim_y=256)

ids = dataset.keys
source_name = dataset.example_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = './weights/vgg/enrico_epoch_101.ckpt'
checkpoint = torch.load(best_model)
net = VGG(num_classes=20)
net.to(device)
net.load_state_dict(checkpoint['net'])

torch.no_grad()
# torch set to eval mode
net.eval()

pic_embed_dict = []

# use tqdm to monitor
for i in tqdm(range(len(ids))):
    id = ids[i]
    actual_id = source_name[id]['screen_id']
    actual_class = source_name[id]['topic']
    pic, wireframe, label = dataset.__getitem__(i)
    pic = pic.to(device)
    # reshape pic to 4D tensor
    pic = pic.reshape(1, 3, 256, 256)
    embedding = net.get_embedding(pic)
    # embedding to numpy array
    embedding = embedding.cpu().detach().numpy()
    pic_embed_dict.append({"id":actual_id, "label_id": label, "label": actual_class, "embedding": embedding})

# save pic_embed_dict as pkl file
with open(os.path.join("./enrico_corpus/vision_embedding.pkl"), 'wb') as f:
    pickle.dump(pic_embed_dict, f)