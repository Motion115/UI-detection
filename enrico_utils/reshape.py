from tqdm import tqdm
import os
from PIL import Image

base_path = './enrico_corpus/screenshots/'
target_path = './enrico_corpus/screenshot_std/'

files = os.listdir(base_path)
for f in tqdm(files):
    screenImg = Image.open(os.path.join(base_path, f)).convert("RGB")
    screenImg = screenImg.resize((400, 800))
    screenImg.save(os.path.join(target_path, f))