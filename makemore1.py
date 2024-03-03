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

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        bigram = (ch1, ch2)
        N[ix1, ix2] += 1

# plt.figure(figsize=(8,8))
# plt.imshow(N, cmap='Blues', aspect='auto', extent=(-0.5, 26.5, 26.5, -0.5))
# plt.rcParams.update({'font.size': 8})
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
# plt.axis('off')
# plt.show(block=True)

g = torch.Generator().manual_seed(2147483647)

P = (N+1).float()
P /= P.sum(1, keepdim=True)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    # print(''.join(out))

log_likelihood = 0.0
n=0
for w in words:
# for w in ["leonardojq"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {prob:.4f} - {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')