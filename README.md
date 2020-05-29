# Sign language analysis

Breve introduzione all'utilizzo della rete ANN con e senza la libreria keras

---

## Introduzione

Questa analisi si divide in due parti:
- l'implementazione di una Artificial Neural Network senza l'utilizzo della libreria `keras`
- l'implementazione della medesima rete con l'utilizzo della libreria `keras`

I dati analizzati provengono dal [sign-language-digits-dataset](https://www.kaggle.com/ardamavi/sign-language-digits-dataset/kernels) di Kaggle e contengono 2062 immagini rappresentanti i numeri da 0 a 9.

Io utilizzerò solamente i numeri zero ed uno, 410 immagini in totale.

![](Images/image1.png)

---

## Costruzione ANN senza `keras`

Il primo passo è l'inizializzazione di:
- errori (random)
- pesi (inizialmente zero)
- dimensione degli strati

```
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,  
                  "bias1": np.zeros((3,1)),                             
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1, 
                  "bias2": np.zeros((y_train.shape[0],1))}              
    return parameters
```

Poi dobbiamo creare la funzione per la *forward propagation*. Questa è l'insieme di tutti i passaggi dai pixels alla funzione di costo.\
1. Consideriamo `z = (w.T)x + b` \
dove: 
- `x` è un *pixel array*
- `w.T` sono i pesi trasposti (*weights*)
- `b` gli errori (*bias*)
Calcoliamo la probabilità `y_head` applicando la *funzione sigmoide* a `z`.

```
def forward_propagation_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    # utilizziamo la funzione tanh
    A1 = np.tanh(Z1) 
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
```

3. Poi calcoliamo la *funzione di perdita* come somma di tutti gli errori, come funzione di perdita utilizziamo la *Cross-Entropy*

![](https://image.ibb.co/nyR9LU/as.jpg)

```
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost
```

Il passo successivo è implementare la *Backward Propagation*
```
def backward_propagation_NN(parameters, cache, X, Y):
    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads
```
