import easyocr
reader = easyocr.Reader(['en'], gpu = True)

from tqdm import tqdm
import pandas as pd
import os
from PIL import Image

base_path = './enrico_corpus/screenshot_std/'
target_path = './enrico_corpus/texts/'

# list files in base_path
files = os.listdir(base_path)

for f in tqdm(files):
    result = reader.readtext(os.path.join(base_path, f))
    # create a dataframe, with columns as pos_x, pos_y and text
    records = []
    for res in result:
        if res[2] > 0.4:
            records.append({'pos_x': res[0][0][0], 'pos_y': res[0][0][1], 'text': res[1]})
    if len(records) == 0:
        records.append({'pos_x': None, 'pos_y': None, 'text': None})
    df = pd.DataFrame(records)
    f = f.split('.')[0]
    df.to_csv(os.path.join(target_path, f + '.csv'), index=False)

out_files = os.listdir(target_path)
for f in tqdm(out_files):
    df = pd.read_csv(os.path.join(target_path, f))
    # filter out the rows with pos_y < 20
    df = df[df['pos_y'] > 20]
    df.to_csv(os.path.join(target_path, f), index=False)
