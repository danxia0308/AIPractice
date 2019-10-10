from tqdm import tqdm
import numpy as np

for i in tqdm(np.arange(100)):
    print(i)

names = np.array(['how','are','you','baby'])
for name in tqdm(names, total=4, desc="desc-"):
    print(name)