import pandas as pd
import numpy as np

# ustawienie widoku kolumn dla lepszego podglądu
pd.set_option('display.max_columns', None)

# === POBIERANIE DANYCH ===
#data = pd.read_csv("digital_diet_mental_health.csv")
data = pd.read_csv("sleep_mobile_stress_dataset_15000.csv")
data = data.sample(frac=1).reset_index(drop=True)

# Przerabianie danych gender i locational_type na (0,1) (False/True) w osobnych kolumnach by model je nie odbierał rangowo i żeby nie wpływało to na wagi
data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'occupation'])
data = data.astype(float)

# TARGET
y = data['sleep_duration_hours'].values.reshape(15000, 1)
# FEATURES
X = data.drop(columns='sleep_duration_hours').values

# Feature Scaling, by każda kolumna byla z skali (0-1)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Zamiana Data Frame na matryce
data = np.array(data)

rows, cols = X.shape

# input layers: (28+1-1) columns
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
        self.w1 = np.random.randn(self.input, self.hidden_units)* np.sqrt(2. / self.input)
        self.w2 = np.random.randn(self.hidden_units, 64)* np.sqrt(2. / self.hidden_units)
        self.w3 = np.random.randn(64, self.output)* np.sqrt(2. / 64 )

        # Velocity
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.v3 = np.zeros_like(self.w3)

        # Biases - initialized to 0
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((self.output, 1))

    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.a2 = self.leaky_ReLU(self.z2)

        self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.a3 = self.leaky_ReLU(self.z3)

        self.z4 = np.dot(self.w3.T, self.a3) + self.b3
        self.a4 = self.z4

        return self.a4

    def leaky_ReLU(self, Z):
        #return np.maximum(Z, 0)
        return np.where(Z > 0, Z, Z * 0.01)

    def _loss(self, predict, y):
        m = y.shape[0]
        #logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        #loss = -1*np.sum(logprobs) / m
        loss = 1/m*np.sum((predict - y.T)**2)
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        rows = X.shape[0]
        lambda_param = 0

        dz4 = predict - y.T # Shape: (1, rows)

        self.dw3 = (1 / rows) * np.dot(self.a3, dz4.T) + (lambda_param * self.w3) # Shape: (64, 1)
        delta3 = np.dot(self.w3, dz4)
        self.db3 = (1/rows) * np.sum(dz4, axis=1, keepdims=True)
        dz3 = delta3 * self.ReLU_prime(self.z3)

        self.dw2 = (1 / rows) * np.dot(self.a2, dz3.T) + (lambda_param * self.w2)
        delta2 = np.dot(self.w2, dz3)
        self.db2 = (1 / rows) * np.sum(dz3, axis=1, keepdims=True)
        dz2 = delta2 * self.ReLU_prime(self.z2)

        self.dw1 = (1 / rows) * np.dot(X.T, dz2.T) + (lambda_param * self.w1)
        delta1 = np.dot(self.w1, dz2)
        self.db1 = (1 / rows) * np.sum(dz2, axis=1, keepdims=True)

    def ReLU_prime(self, z):
        #return (z>0).astype(float)
        dz = np.ones_like(z)
        dz[z <= 0] = 0.01
        return dz

    def _update(self, learning_rate=0.01):
        beta = 0.9
        self.v1 = beta * self.v1 + (1-beta) * self.dw1
        self.w1 = self.w1 - learning_rate * self.v1
        self.b1 = self.b1 - learning_rate * self.db1

        self.v2 = beta * self.v2 + (1 - beta) * self.dw2
        self.w2 = self.w2 - learning_rate * self.v2
        self.b2 = self.b2 - learning_rate * self.db2

        self.v3 = beta * self.v3 + (1 - beta) * self.dw3
        self.w3 = self.w3 - learning_rate * self.v3
        self.b3 = self.b3 - learning_rate * self.db3

    def train(self, X, y, X_train, y_train, X_test, y_test, iteration=1000):
        learning_rate = 0.01

        batch_size = 32
        rows = X.shape[0]

        for i in range(iteration):
            indices = np.random.permutation(rows)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start in range(0, rows, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self._backward_propagation(X_batch, y_batch)
                self._update(learning_rate)

            if i % 100 == 0:
                full_y_hat = self._forward_propagation(X_train)
                train_mae = np.mean(np.abs(full_y_hat.T - y_train)) * 10

                test_y_hat = self._forward_propagation(X_test)
                test_mae = np.mean(np.abs(test_y_hat.T - y_test)) * 10
                print(f"Iter {i} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")

            if i % 50 == 0:
                learning_rate *= 0.9

    def predict(self, X):
        y_hat_scaled = self._forward_propagation(X)
        return np.array(y_hat_scaled.T)*10

    def score(self, predict, y):
        return np.mean(np.abs(predict - y))

def train():
    X_train = X[:12000]
    X_test = X[12000:]

    y_train = y[:12000]
    y_test = y[12000:]

    clr = Sleep_Prediction()  # initialize the model

    clr.train(X, y/10, X_train, y_train / 10, X_test, y_test / 10)  # train model
    pre_y = clr.predict(X_test)  # predict
    score = clr.score(pre_y, y_test)  # get the accuracy score

    print('=== SCORE: ', score)

train()