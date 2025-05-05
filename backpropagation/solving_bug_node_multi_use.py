from backpropagation.backprop import Value
from backpropagation.viz import draw_dot

# Casos em que un nó em usado mais de uma vez nas operações seguintes, faz com que da forma que a gente faz a backprop atual quebrar
# Para isso, vamos mudar a class Value para aonde tem _backward, ao invés de atribuir diretamente a variavel, vai ser um acumuo: = vai para +=
# Isso já o suficiente para corrigir o problema. Antes o grad de a era 1, mas deveria ser 2, depois da modificação, ficou 2.
a = Value(3.0, label='a')
b = a + a ; b.label = 'b'
b.backward()

n = draw_dot(b)

n.render('grafo', view=True)