from graphviz import Digraph
from backpropagation.backprop import Value
import os
# ajuste para o local onde instalou o Graphviz
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# Visualização do gráfico
# --------------------------------------------------------------------------------
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # Cria o nó de valor (com data e grad)
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')

        # Se houver operação, cria um nó intermediário com o nome da operação
        if n._op:
            op_id = uid + n._op  # identificador único
            dot.node(name=op_id, label=n._op, shape='circle')
            dot.edge(op_id, uid)  # operação → resultado

    for n1, n2 in edges:
        uid1 = str(id(n1))
        uid2 = str(id(n2))
        if n2._op:
            op_id = uid2 + n2._op
            dot.edge(uid1, op_id)  # filho → operação
        else:
            dot.edge(uid1, uid2)  # fallback direto (caso raro)

    return dot

def lol():

    h = 0.0001

    a = Value(2.0 + h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f= Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data

    print((L2 - L1)/h)
    return L
# ------------------------------------------------------------------------------

# --- Exemplo mínimo de uso ---
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f= Value(-2.0, label='f')
L = d * f; L.label = 'L'
L1 = L.data
L.grad = 1
f.grad = 4.0
d.grad = -2.0

root = L

# dot_before_grad = draw_dot(root)
#
# dot_before_grad.render('grafo', view=True)


# Entendendo como funciona o backpropagation e as derivadas.
# ------------------------------------------------------------------------------
"""
dd / dc = ?
dd / de = ?

# Sabemos que
d = c + e

# Definição da derivada para o nosso caso:
(f(x+h) - f(x))/h

# Portanto,
f(x+h) = (c+h + e) 
f(x) = (c+e)

# Isso resulta em h/h, pois (c+h + e) - (c+e) = h
# Resultado dd/dc = 1.0 = h/h
# E dd/de também será 1.0, seguindo a exata mesma lógica.

dd / dc = 1.0
dd / de = 1.0

# Agora, para encontramos uma derivada de dois valores que não se relacionam diretamente como foi o caso acima.
# Será necessário utilizar a regra de cadeia.

# Queremos:
 - dL / dc

# Sabemos:
 - dL / dd
    L = d * f
    (((d+h) * f) - (d*f))/h
    dL / dd = f
# E f = -2.0 (foi definido diretamente) 

 - dd / dc = 1.0

# Usando regra da cadeia podemos chegar a essa conclusão:
dL / dc = (dL / dd) * (dd / dc)

# Ou seja, dL / dc = -2.0 * 1.0 = -2.0
"""

# dL / dd
# L = d * f
# (((d + h) * f) - (f * d)) / h
# = f = -2.0
d.grad = -2.0


# dL / de
# L = d * f
# (((f + h) * d) - (f * d)) / h
# = d = 4.0
f.grad = 4.0


# dL / dc = -2.0
c.grad = -2.0


# Processo lógico é igual ao de c.
# dl / de = -2.0
e.grad = -2.0


# dL / db

# de / db
# e = a * b
# (((b + h) * a) - (b*a))/h
# = a = 2.0

# dL / de = -2.0

# dL / db = (dL / de) * (de / da) = -4.0
b.grad = -4.0


# dL / da

# de / da
# e = a * b
# (((a + h) * b) - (b*a))/h
# = b = -3.0

# dL / de = -2.0

# dL / db = (dL / de) * (de / da) = 6.0
a.grad = 6.0

# dot_after_grad = draw_dot(root)
#
# dot_after_grad.render('grafo', view=True)

# ------------------------------------------------------------------------------

# Processo de otimização
# ------------------------------------------------------------------------------
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f
print(L.data)

