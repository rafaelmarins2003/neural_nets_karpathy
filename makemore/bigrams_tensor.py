import torch, numpy as np, matplotlib.pyplot as plt
words = open('makemore/names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
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

# p = N[0].float()
# p = p / p.sum()
# print(p)
#
# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p = p / p.sum()
# print(p)
#
# torch.multinomial(p, num_samples=20, replacement=True, generator=g)

P = N.float()
# P = P / P.sum()
# Precisamos fazer a divisão não pela soma total da matriz, mas pela soma de cada linha da matriz, ou seja, o sum deve retornar uma lista com 27 somas de linha, não uma soma total de toda a matriz.
P /= P.sum(1, keepdims=True)


g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # p = N[ix].float()
        # p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
