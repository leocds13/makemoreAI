import requests
import torch
import os
# import matplotlib.pyplot as plt

if not os.path.isfile("names.txt"):
    res = requests.get(
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", 
        timeout=60
    )
    with open("names.txt", 'w+', encoding='utf-8') as f:
        f.write(res.text)

words = []
with open("names.txt", 'r', encoding='utf-8') as f:
    words = f.read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Create training set of bigrams (x,y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch1]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

