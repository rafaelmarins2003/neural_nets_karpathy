import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from backpropagation.backprop import Value
from backpropagation.viz import draw_dot

# plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid();

# inputs x1,x2 (variaveis)
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights w1, w2 (Peso que vai ser responsavel por otimizar o modelo)
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias of the neuron
b = Value(6.881373580195432, label='b')

# x1*w1 + x2w2 ... + b
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'
# Mudança na forma de chamada do tanh
# ------------------------------------------------------------------------------
e = (2*n).exp()
o = (e - 1) / (e + 1)
# ------------------------------------------------------------------------------
o.label = 'o'
o.backward()

n = draw_dot(o)

n.render('grafo', view=True)