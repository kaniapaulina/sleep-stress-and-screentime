import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# === POBIERANIE DANYCH i ich przerobienie
data = pd.read_csv("digital_diet_mental_health.csv")
data = data.sample(frac=1).reset_index(drop=True)

# Przerabianie danych gender i locational_type na (0,1) (False/True) w osobnych kolumnach by model je nie odbierał rangowo i żeby nie wpływało to na wagi
data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'location_type'])
data = data.astype(float)

# TARGET
y = data['sleep_duration_hours'].values.reshape(2000, 1)
# FEATURES
X = data.drop(columns='sleep_duration_hours').values

# Feature Scaling, by każda kolumna byla z skali (0-1)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Zamiana Data Frame na matryce
data = np.array(data)

rows, cols = X.shape

# input layers: (28-1) columns
# learning data: 2000
# hidden layers: 484
# output layers: hours of sleep predicted - 1 layer

# === NEURAL NETWORK The Prologue
class Sleep_Prediction:
    def __init__(self):
        self.input = cols
        self.output = 1
        self.hidden_units = 128

        np.random.seed(1)
        self.w1 = np.random.randn(self.input, self.hidden_units)*0.01
        self.w2 = np.random.randn(self.hidden_units, 64)*0.01
        self.w3 = np.random.randn(64, self.output) * 0.01

    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T)
        self.a2 = self.ReLU(self.z2)
        self.z3 = np.dot(self.w2.T, self.a2)
        self.a3 = self.ReLU(self.z3)
        self.z4 = np.dot(self.w3.T, self.a3)
        self.a4 = self.z4
        return self.a4

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def _loss(self, predict, y):
        m = y.shape[0]
        #logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        #loss = -1*np.sum(logprobs) / m
        loss = 1/m*np.sum((predict - y.T)**2)
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        m = X.shape[0]
        #dz3 = np.multiply(delta3, self.ReLU_prime(self.z3))
        dz4 = predict - y.T # Shape: (1, m)
        self.dw3 = (1 / m) * np.dot(self.a3, dz4.T) # Shape: (64, 1)
        #self.dw2 = (1 / m) * np.sum(np.multiply(self.a2, dz3), axis=1).reshape(self.w2.shape)
        delta3 = np.dot(self.w3, dz4)
        dz3 = delta3 * self.ReLU_prime(self.z3)
        self.dw2 = (1 / m) * np.dot(self.a2, dz3.T)

        delta2 = np.dot(self.w2, dz3)
        dz2 = delta2 * self.ReLU_prime(self.z2)
        self.dw1 = (1 / m) * np.dot(X.T, dz2.T)

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def ReLU_prime(self, z):
        return (z>0).astype(float)

    def _update(self, learning_rate=0.01):
        self.w1 = self.w1 - learning_rate * self.dw1
        self.w2 = self.w2 - learning_rate * self.dw2
        self.w3 = self.w3 - learning_rate * self.dw3

    def train(self, X, y, iteration=5000):
        for i in range(iteration):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            if i % 10 == 0:
                print("loss: ", loss)

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        #y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat.T)

    def score(self, predict, y):
        #cnt = np.sum(predict == y)
        #return (cnt / len(y)) * 100
        return np.mean(np.abs(predict - y))

def train():
    X_train = X[:1500]
    X_test = X[1500:]

    y_train = y[:1500]
    y_test = y[1500:]

    clr = Sleep_Prediction()  # initialize the model

    clr.train(X_train, y_train)  # train model
    pre_y = clr.predict(X_test)  # predict
    score = clr.score(pre_y, y_test)  # get the accuracy score

    #print('=== PREDICT: ', pre_y)
    #print('=== ANSWER:', y_test)
    print('=== SCORE: ', score)

train()