"""
Regression model predicting hours of sleep based on factors such as: stress levels, screentime, age etc.
"""

# === LIBRARIES
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# === PREPARING DATA
data = pd.read_csv("digital_diet_mental_health.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'location_type'])
data = data.astype(float)

data['stress_phone_interaction'] = data['stress_level'] * data['phone_usage_hours']
data['total_digital_load'] = data['phone_usage_hours'] + data['laptop_usage_hours'] + data['gaming_hours']

real_data = data.copy()

data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# TARGET
y = data['sleep_duration_hours'].values.reshape(2000, 1)
# FEATURES
X = data.drop(columns='sleep_duration_hours').values

rows, cols = X.shape

# === NEURAL NETWORK
class Sleep_Prediction:
    """
    A primitive neural network written from scratch
    Parameters:
        input layers: 28 (all) - 1 (what we want to predict) = 27 columns
        learning data: 2000 rows
        hidden layers: 128
        output layers: hours of sleep predicted - 1 layer
    """

    def __init__(self):
        self.input = cols
        self.output = 1
        self.hidden_units = 128

        # Weights
        self.w1 = np.random.randn(self.input, self.hidden_units)* np.sqrt(2. / self.input)
        self.w2 = np.random.randn(self.hidden_units, 64)* np.sqrt(2. / self.hidden_units)
        self.w3 = np.random.randn(64, self.output)* np.sqrt(2. / 64 )

        # Velocity
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.v3 = np.zeros_like(self.w3)

        # Biases
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((self.output, 1))

    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.a2 = self.ReLU(self.z2)

        self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.a3 = self.ReLU(self.z3)

        self.z4 = np.dot(self.w3.T, self.a3) + self.b3
        self.a4 = self.z4

        return self.a4

    def ReLU(self, Z):
        return np.where(Z > 0, Z, Z * 0.01)  # Leaky ReLU

    def _loss(self, predict, y):
        m = y.shape[0]
        loss = 1/m * np.sum((predict - y.T)**2)
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        rows = X.shape[0]
        lambda_param = 0.001

        dz4 = predict - y.T

        self.dw3 = (1 / rows) * np.dot(self.a3, dz4.T) + (lambda_param * self.w3)
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
        return np.where(z > 0, 1, 0.01) # Leaky ReLu

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

    def train(self, X_train, y_train, X_test, y_test, iteration=1000):
        learning_rate = 0.008
        batch_size = 32
        rows = X_train.shape[0]

        for i in range(iteration):
            idx = np.random.permutation(rows)

            X_s, y_s = X_train[idx], y_train[idx]

            for s in range(0, rows, batch_size):
                X_b, y_b = X_s[s:s + batch_size], y_s[s:s + batch_size]

                self._backward_propagation(X_b, y_b)
                self._update(learning_rate)

            if i % 100 == 0:
                full_y_hat = self._forward_propagation(X_train)
                train_mae = np.mean(np.abs(full_y_hat.T -  y_train))*10

                full_y_test = self._forward_propagation(X_test)
                test_mae = np.mean(np.abs(full_y_test.T - y_test))*10

                print(f"Iter {i} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")

            if i % 100 == 0:
                learning_rate *= 0.8

    def predict(self, X):
        y_hat_scaled = self._forward_propagation(X)
        return np.array(y_hat_scaled.T)*10

    def score(self, predict, y):
        return np.mean(np.abs(predict - y))

# === TRAINING THE MODEL
def train():
    X_train = X[:1600]
    X_test = X[1600:]

    y_train = y[:1600]
    y_test = y[1600:]

    clr = Sleep_Prediction()

    clr.train(X_train, y_train/10, X_test, y_test/10)
    pre_y = clr.predict(X_test)
    score = clr.score(pre_y, y_test)

    print('=== SCORE: ', score)

    def show_comparison(model, X_test, y_test):
        predictions = model.predict(X_test)
        comparison = pd.DataFrame({
            'Actual Hours': y_test.flatten().round(2),
            'Predicted Hours': predictions.flatten().round(2)
        })
        comparison['Error (Minutes)'] = (np.abs(comparison['Actual Hours'] - comparison['Predicted Hours']) * 60).round(
            0)

        print("\n=== ACTUAL VS PREDICTED ===")
        print(comparison.head(10)*10)

        print(f"\nAverage Error: {comparison['Error (Minutes)'].mean():.1f} minutes")

    show_comparison(clr, X_test, y_test)

    return clr

clr = train()

# TESTING THE MODEL
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
    new_data = pd.get_dummies(new_data, columns=['gender', 'location_type'], dtype=float)

    new_data['stress_phone_interaction'] = new_data['stress_level'] * new_data['phone_usage_hours']
    new_data['total_digital_load'] = new_data['phone_usage_hours'] + new_data['laptop_usage_hours'] + new_data[
        'gaming_hours']

    train_cols = [c for c in original_df.columns if c != 'sleep_duration_hours']

    new_data = new_data.reindex(columns=train_cols, fill_value=0)

    t_min = original_df[train_cols].min()
    t_max = original_df[train_cols].max()

    X_custom_scaled = (new_data - t_min) / (t_max - t_min)

    predictions = model.predict(X_custom_scaled.values)

    print("\n=== FINAL CORRECTED PREDICTIONS ===")
    for i, hours in enumerate(predictions):
        print(f"User {i + 1}: Predicted {hours[0]*10:.2f} hours of sleep")

predict_new_users(clr, new_samples, real_data)
