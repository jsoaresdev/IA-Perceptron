# IA‑Perceptron 🧠

Implementação didática em Python do algoritmo **Perceptron**, o modelo de rede neural mais simples — um único neurônio com função de ativação.

---

## Índice

- [O que é um Perceptron](#o-que-é-um-perceptron)  
- [Como funciona](#como-funciona)  
- [Estrutura do repositório](#estrutura-do-repositório)  
- [Pré‑requisitos](#pré‑requisitos)  
- [Como usar](#como-usar)  
- [Exemplo de uso](#exemplo-de-uso)  
- [Visualizações](#visualizações)  
- [Licença](#licença)  
- [Referências](#referências)

---

## O que é um Perceptron

O perceptron é o classificador binário mais básico, criado inicialmente por Warren McCulloch e Walter Pitts em 1943, e com a primeira implementação em hardware por Frank Rosenblatt em 1957. Ele mapeia um vetor de entradas `x` para uma saída binária `f(x)` por meio de um produto escalar com o vetor de pesos `w` e um viés `b`, seguido por uma função de ativação:  

```
f(x) = 1 if w·x + b ≥ 0 else 0
```

---

## Como funciona

1. Inicia pesos `w` (aleatórios ou definidos) e viés `b`.  
2. Para cada amostra de treinamento `(x, y)`, calcula `u = w·x + b`.  
3. Aplica a função de ativação (degrau) produzindo saída `ŷ`.  
4. Calcula o erro `e = y – ŷ`.  
5. Atualiza os pesos e o viés:  
   ```
   w = w + η * e * x
   b = b + η * e
   ```
6. Repete para um número definido de épocas, até convergência.

---


## Pré‑requisitos

- Python 3.6+  
- (Opcional, mas recomendado):  
  - `numpy` – operações eficientes com vetores  
  - `matplotlib` e `seaborn` – visualização de dados e fronteiras de decisão  

Instale com:

```bash
pip install numpy matplotlib seaborn
```

---

## Como usar

Importe e use em seu próprio script:

```python
from perceptron import Perceptron

# Conjunto de treinamento (XOR, OR, AND, etc.)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 1]  # OR

p = Perceptron(learning_rate=0.1, n_iter=20)
p.fit(X, Y)

print("Pesos:", p.weights_, "Viés:", p.bias_)
```

Ou usando um exemplo pronto:

```bash
python examples/train_or.py
```

---

## Exemplo de uso

Para treinar um perceptron para a função lógica OR e visualizar sua fronteira de decisão:

```python
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0,1,1,1])

p = Perceptron(learning_rate=0.5, n_iter=30)
p.fit(X, Y)

# Plot dos pontos e linha de decisão
xx = np.linspace(-0.5, 1.5, 100)
yy = -(p.weights_[0]*xx + p.bias_) / p.weights_[1]
plt.scatter(X[:,0], X[:,1], c=Y, cmap='bwr')
plt.plot(xx, yy, 'k--')
plt.title("Perceptron treinado para OR")
plt.show()
```

---

## Visualizações

O diretório `examples/` contém scripts que:
- Plotam os dados com cores por classe (usando seaborn)
- Mostram a linha/função de decisão
- Exibem a evolução do erro ao longo das épocas

---

## Licença

MIT © [jsoaresdev](https://github.com/jsoaresdev)

---

## Referências

- Implementação simples em Python – como referência para OR, AND e visualização  
- Artigo da Wikipédia sobre Perceptron

---

## Contribuições

Contribuições são bem‑vindas!  
Faça fork, crie uma branch com melhorias, testes ou novos exemplos, e abra um pull request 😊

---

## Contato

Para dúvidas, sugestões ou colaborações, abra uma *issue* ou entre em contato diretamente via GitHub.
