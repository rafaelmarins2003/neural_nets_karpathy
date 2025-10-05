from backpropagation.multilayer_perceptron import MLP
from backpropagation.backprop import Value

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])
ypred = [n(x) for x in xs]
print(ypred)

# Loss function soma o comparativo entre o output esperado e o real de cada saida, dado que foram colocados 4 exemplos de e entradas distintas.
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
print(f"Soma das diferenças de resultado obtido e esperado (loss function): {loss.data}")

# Lembre-se que o MLP é uma lista de layers que possui uma lista de neuronios que possui uma lista de weights que vão operar em cima do valor de entrada.
print(f"Valor exemplo de um weight dentro do nosso sistema: {n.layers[0].neurons[0].w[0].data}")
print(f"Valor do grad desse weight antes da backward(): {n.layers[0].neurons[0].w[0].grad}")

# Se fizermos o backward prop, teremos um valor de grad para os nós internos e assim eventualmente vamos poder começar a modificar seus valores e minimzar o erro.
loss.backward()
print("---------Aplicado a backward em cima da do resultado da loss function.----------------")
print(f"Novo grad calculado para o weight: {n.layers[0].neurons[0].w[0].grad}")

# Parametros w e b dentro da nn
print(f"Num de parametros w e b dentro da nn: {len(n.parameters())}")

# Aqui inicia-se o processo de otimização com gradiente descendente dos pesos para que a nossa loss diminua
# for p in n.parameters():
#     p.data += -0.01 * p.grad

# Processo completo da aplicação do gradiente descendente aplicado a rede em um número de iterações dinamico.

nn = MLP(3, [4,4,1])
xxs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

print(f"{"-"*30}Iterações loss function{"-"*30}")
for k in range(20):
    # forward pass
    ypred = [n(x) for x in xxs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.15 * p.grad

    print(f"Iteração: {k+1}, Loss: {loss.data}")
print(f"Y previsto após as iterações: {ypred}")
