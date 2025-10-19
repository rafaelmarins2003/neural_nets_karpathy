import torch, numpy as np, matplotlib.pyplot as plt
words = open('makemore/names.txt', 'r').read().splitlines()

N = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

itos = {i:s for s,i in stoi.items()}
V = len(stoi)
plt.figure(figsize=(min(18, 0.6*V), min(18, 0.6*V)))
plt.imshow(N.numpy(), cmap='Blues')

for i in range(V):
    for j in range(V):
        chstr = f"{itos[i]}{itos[j]}"
        val = int(N[i, j])
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=8)
        plt.text(j, i, str(val), ha='center', va='top', color='gray', fontsize=8)

plt.xticks(range(V), [itos[j] for j in range(V)], rotation=90)
plt.yticks(range(V), [itos[i] for i in range(V)])
plt.tight_layout()
plt.axis('off')
plt.show()
