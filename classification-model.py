"""
Model Klasyfikacyjny przepowiadający czy ktoś jest zdrowy czy nie
"""
import pandas as pd
import numpy as np

# ustawienie widoku kolumn dla lepszego podglądu
pd.set_option('display.max_columns', None)

# =======================================================
# ===== POBIERANIE DANYCH
data = pd.read_csv("digital_diet_mental_health.csv")
data = data.sample(frac=1).reset_index(drop=True)

# Przerabianie danych gender i locational_type na (0,1) (False/True) w osobnych kolumnach by model je nie odbierał rangowo i żeby nie wpływało to na wagi
data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'location_type'])
data = data.astype(float)

# Scaling, by każda kolumna byla z skali (0-1)
data = (data - data.mean()) / data.std()

# Deciding who is and isnt healthy
data['is_depressed'] = np.where(
    (data['mental_health_score'] < 0.4) |
    (data['stress_level'] > 0.7) |
    (data['weekly_anxiety_score'] > 0.8),
    1.0, 0.0)

# TARGET
y = data['is_depressed'].values.reshape(2000, 1)
# FEATURES
X = data.drop('is_depressed', axis=1).values

rows, cols = X.shape

# input layers: (28-1) = 27 columns
# learning data: 2000 rows
# hidden layers: 128
# output layers: is depressed (0-1) - 1 layer

# =======================================================
# ===== NEURAL NETWORK The Prologue
class Mentally_Unwell_Prediction:
    # Initialize the model
    def __init__(self):
        self.input = cols
        self.output = 1
        self.hidden_units = 128

        # Initialize matrix of weights
        self.w1 = np.random.randn(self.input, self.hidden_units)* np.sqrt(2. / self.input)  #27*128 matrix
        self.w2 = np.random.randn(self.hidden_units, 64)* np.sqrt(2. / self.hidden_units)   #128*64 matrix
        self.w3 = np.random.randn(64, self.output)* np.sqrt(2. / 64 )                       #64*1 matrix

        # Velocity
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.v3 = np.zeros_like(self.w3)

        # Biases - initialized to 0
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((self.output, 1))

    # Foward move from input layer through hidden layers, multiplying neuron by weight
    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.a2 = self.ReLU(self.z2)

        self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.a3 = self.ReLU(self.z3)

        self.z4 = np.dot(self.w3.T, self.a3) + self.b3
        self.a4 = self._sigmoid(self.z4)

        return self.a4

    # Rectified Linear Unit
    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, predict, y):
        m = y.shape[0]
        loss = -1 / m * np.sum(y.T * np.log(predict + 1e-15) + (1 - y.T) * np.log(1 - predict + 1e-15))
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        rows = X.shape[0]

        dz4 = predict - y.T # Shape: (1, rows)

        self.dw3 = (1 / rows) * np.dot(self.a3, dz4.T) # Shape: (64, 1)
        delta3 = np.dot(self.w3, dz4)
        self.db3 = (1/rows) * np.sum(dz4, axis=1, keepdims=True)
        dz3 = delta3 * self.ReLU_prime(self.z3)

        self.dw2 = (1 / rows) * np.dot(self.a2, dz3.T)
        delta2 = np.dot(self.w2, dz3)
        self.db2 = (1 / rows) * np.sum(dz3, axis=1, keepdims=True)
        dz2 = delta2 * self.ReLU_prime(self.z2)

        self.dw1 = (1 / rows) * np.dot(X.T, dz2.T)
        delta1 = np.dot(self.w1, dz2)
        self.db1 = (1 / rows) * np.sum(dz2, axis=1, keepdims=True)

    def ReLU_prime(self, z):
        return (z>0).astype(float)

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

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

    def train(self, X, y, iteration=3000):
        learning_rate = 0.005

        batch_size = 32
        rows = X.shape[0]

        for i in range(iteration):
            self._backward_propagation(X, y)
            self._update(learning_rate)

            if i % 100 == 0:
                full_y_hat = self._forward_propagation(X)
                predictions = (full_y_hat.T > 0.5).astype(float)
                accuracy = np.mean(predictions == y)

                print(f"Iter {i} | Loss: {self._loss(full_y_hat, y):.4f} | Accuracy: {accuracy * 100:.2f}%")

            if i % 200 == 0:
                learning_rate *= 0.95

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)

    def score(self, predict, y):
        predict = predict.flatten()
        y = y.flatten()
        cnt = np.sum(predict == y)
        return (cnt / len(y)) * 100

def train():
    X_train = X[:1600]
    X_test = X[1600:]

    y_train = y[:1600]
    y_test = y[1600:]

    clr = Mentally_Unwell_Prediction()  # initialize the model

    clr.train(X_train, y_train)  # train model
    pre_y = clr.predict(X_test)  # predict
    score = clr.score(pre_y, y_test)  # get the accuracy score

    print(f'=== SCORE: {score:.2f}%')

    return clr

clr = train()

# TESTOWANIE MODELU
import io

csv_data = """user_id,age,gender,daily_screen_time_hours,phone_usage_hours,laptop_usage_hours,tablet_usage_hours,tv_usage_hours,social_media_hours,work_related_hours,entertainment_hours,gaming_hours,sleep_duration_hours,sleep_quality,mood_rating,stress_level,physical_activity_hours_per_week,location_type,mental_health_score,uses_wellness_apps,eats_healthy,caffeine_intake_mg_per_day,weekly_anxiety_score,weekly_depression_score,mindfulness_minutes_per_day
user_1,51,Female,4.8,3.4,1.3,1.6,1.6,4.1,2.0,1.0,1.7,6.6,6,6,10,0.7,Urban,32,1,1,125.2,13,15,4.0
user_2,64,Male,3.9,3.5,1.8,0.9,2.0,2.7,3.1,1.0,1.5,4.5,7,5,6,4.3,Suburban,75,0,1,150.4,19,18,6.5
user_3,41,Other,10.5,2.1,2.6,0.7,2.2,3.0,2.8,4.1,1.7,7.1,9,5,5,3.1,Suburban,22,0,0,187.9,7,3,6.9
user_extreme_work,28,Male,14.0,6.0,7.0,0.5,0.5,2.0,10.0,1.0,1.0,4.5,3,3,10,0.5,Urban,15,0,0,400.0,25,20,0.0
user_gamer,22,Other,12.0,3.0,2.0,0.0,7.0,2.0,1.0,9.0,8.0,5.5,5,7,4,2.0,Suburban,60,0,0,150.0,10,8,15.0
user_healthy_senior,72,Female,2.5,1.5,1.0,0.0,0.0,1.0,0.5,1.0,0.0,8.5,9,9,2,10.5,Rural,90,1,1,0.0,2,1,45.0
user_anxious_student,20,Female,9.5,7.0,2.0,0.5,0.0,6.5,2.0,1.0,0.0,5.0,4,4,8,1.5,Urban,40,1,0,250.0,18,12,5.0
user_balanced,35,Male,5.0,2.5,2.0,0.5,0.0,1.5,3.0,0.5,0.0,7.5,8,7,3,5.0,Suburban,78,1,1,50.0,5,3,20.0"""

new_samples = pd.read_csv(io.StringIO(csv_data))

def predict_new_users(model, new_data, original_df):
    if 'user_id' in new_data.columns:
        new_data = new_data.drop('user_id', axis=1)

    new_data = pd.get_dummies(new_data)

    model_columns = original_df.drop('is_depressed', axis=1).columns
    new_data = new_data.reindex(columns=model_columns, fill_value=0)

    orig_features = original_df.drop('is_depressed', axis=1)
    new_data_scaled = (new_data - orig_features.min()) / (orig_features.max() - orig_features.min())

    X_custom = new_data_scaled.values
    raw_probs = model._forward_propagation(X_custom)  # Prawdopodobieństwa

    for i, prob in enumerate(raw_probs.T):
        status = "Unwell" if prob >= 0.5 else "Healty"
        print(f"Test {i + 1}: Chance for depression: {prob[0] * 100:.2f}% -> Diagnose: {status}")

predict_new_users(clr, new_samples, data)