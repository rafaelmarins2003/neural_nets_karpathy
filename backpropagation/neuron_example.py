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

o = n.tanh(); o.label = 'o'

n_before = draw_dot(o)

n_before.render('grafo', view=True)


# Iniciando a backpropagation (sair derivando pra trás...)
# ----------------------------------------------------------------------------
o.grad = 1.0

# o = tanh(n)
# Derivada padrão do tanh
# do/dn = 1 - o**2 = aprox. 0.5
n.grad = 0.5

# Seguindo a lógica do exemplo anterior, sempre que a soma dos 2 é n, então o grad deles é igual ao grad de n
x1w1x2w2.grad = 0.5
b.grad = 0.5

# A mesma ideia se aplica a x1w1 e x2ww2, pois eles somados dão x1w1x2w2, portanto o grad deles é igual ao grad de x1w1x2w2
x1w1.grad = 0.5
x2w2.grad = 0.5

# Agora indo para a lógica de quando o operador que gerou n é multiplicação. Ex: a * b = n,
# para achar o grad de a, basta fazer b.data * n.grad, e vice-versa.
w1.grad = x1.data * x1w1.grad
x1.grad = w1.data * x1w1.grad

w2.grad = x2.data * x2w2.grad
x2.grad = w2.data * x2w2.grad


n_after = draw_dot(o)

n_after.render('grafo', view=True)
