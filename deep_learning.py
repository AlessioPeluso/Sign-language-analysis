
# --------------------------
# ----- DEEP LEARNING ------
# --------------------------

# Questo script è diviso in due parti:
# 1. la prima parte descrive una Rete Neurale ANN implementata "a mano" 
# 2. la seconda parte descrive la rete ANN utilizzando keras e sklearn

# --------------------------------------------------------------------------------------------------

# librerie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output


# --------------------------------------------------------------------------------------------------

# ----- i dati sono presi da --> Sign-language-digits-dataset di Kaggle

# sono presenti 2062 immagini di sign language digits che vanno da 0 a 9, lavoro solo con sign 0 e 1 

# carico i dati
x_l = np.load('/Deep_Learning/data/X.npy')
Y_l = np.load('/Deep_Learning/data/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[810].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[830].reshape(img_size, img_size))
plt.axis('off')


# per creare un image array concateno gli array di segno zero ed uno
# creo poi un array label 0 per i sign image 0 e 1 per i sign image 1

# unisco la sequenza di array sull'asse delle righe (axis = 0)
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) 
zero  = np.zeros(205)
one   = np.ones(205)
Y = np.concatenate((zero, one), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# la dimensione di X è (410, 64, 64)  dove 410 indica che abbiamo 410 immagini tra 0 e 1 
# 64 indica che le immagini sono 64x64 pixels
# la dimensione di Y è (410,1) quindi abbiamo 410 labels (0,1)

# creiamo gli arrays x_train, y_train, x_test, y_test 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=16)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

# bisogna trasformare gli X da tridimensionale a bidimensionale per utilizzarlo come input per il modello 

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

# possiamo vedere che nel train set abbiamo 348 immagini da 4096 pixels

# prendiamo il trasposto
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# --------------------------------------------------------------------------------------------------

# Artificial Neural Network (ANN)

# --- inizializziamo i pesi e gli errori
# i pesi vengono inizializzati random
# il bias viene posto inizialmente a 0
# inizializziamo inoltre i parametri e la dimensione del layer
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1, # random 
                  "bias1": np.zeros((3,1)),                             # zero
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1, # random
                  "bias2": np.zeros((y_train.shape[0],1))}              # zero
    return parameters

# --- La forward propagation
# tutti i passaggi dal pixel al costo sono chiamati forward propagation e si possono riassumere in:
# trovare z = w.T*x+b
# y_head = sigmoid(z)
# loss(error) = loss(y,y_head)
# cost = sum(loss)

def forward_propagation_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1) # utilizziamo la funzione tanh
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# --- funzione di perdita e funzione di costo
# Cross entropy 
# calcolo il costo
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost

# --- Backward propagation
# nella Backward Propagation utilizziamo yhat trovato nella forward
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

# --- Update dei parametri
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    return parameters

# Previsioni con i parametri i pesi e gli errori "imparati"
def predict_NN(parameters,x_test):
    # x_test è input per la forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # se z è maggiore di 0.5, la nostra previsione è sign_one (y_head=1),
    # se z è minore di 0.5,la nostra previsione è sign_zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# --- creazione del modello
# 2-Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    # inizializzazione parametri e layer size
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)

        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    # previsioni
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)

# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


# --------------------------------------------------------------------------------------------------


# --- implementazione con la libreria keras
# diamo uno sguardo ai parametri della libreria keras
# - units: dimensione del nodo output
# - kernel_initializer: per inizializzare i pesi
# - activation: activation function, usiamo relu
# - input_dim: input dimension cioè il numero di pixels nella nostra immagine (4096 px)
# - optimizer: usiamo adam optimizer
#     Adam è l'algoritmo di ottimizzazione più efficiente per l'allenamento delle reti neurali.
#     alcuni dei vantaggi si Adam sono:
#     - scarso utilizzo della memoria (relativamente)
#     - buoni risultati anche con poco tuning degli iper-parametri
# - loss: usiamo la cross-entropy
# - metrics: usiamo l'accuracy.
# - cross_val_score: usiamo la cross validation
# - epochs: numero di iterazioni

# ANN
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential 
from keras.layers import Dense 
def build_classifier():
    classifier = Sequential() # inizializzazione della rete
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
