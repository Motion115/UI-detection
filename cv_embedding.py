import torch
from tqdm import tqdm
from enrico_utils.get_data import get_dataloader


(train_loader, val_loader, test_loader), weights = get_dataloader("enrico_corpus", 
    img_dim_x=224, img_dim_y=224,batch_size=8)

'''
# create a shuffle of a list of numbers, then output it into a csv
import random
import csv

l = list(range(0, 1458))
random.shuffle(l)
with open('shuffle.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(l)

# read the shuffle.csv
with open('shuffle.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(data[0])
'''

