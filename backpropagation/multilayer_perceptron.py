import random

from backpropagation.viz import draw_dot
from backpropagation.backprop import Value

class Neuron:
    def __init__(self, nin):
        self.w =[Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# # Teste Neuron
# x = [2.0]
# n = Neuron(2)
# print(n(x))
#
# # Teste layer
# x = [2.0, 3.0]
# n = Layer(2, 3)
# print(n(x))

# Teste MLP
# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4, 4, 1])
# print(n(x))

# Preciso corrigir algumas coisas na função atual para usar o draw_dot
# draw_dot(n(x))
#
# xs = [
#     [2.0, 3.0, -1.0],
#     [3.0, -1.0, 0.5],
#     [0.5, 1.0, 1.0],
#     [1.0, 1.0, -1.0]
# ]
# ys = [1.0, -1.0, -1.0, 1.0]
# n = MLP(3, [4, 4, 1])
# ypred = [n(x) for x in xs]
