"""
=======================================================================
  MODEL KLASYFIKACYJNY - Przewidywanie stanu zdrowia psychicznego
=======================================================================
  Cel: Na podstawie czasu przed ekranem, snu i innych czynników klasyfikator decyduje, czy dana osoba jest psychicznie "zdrowa" czy "niezdrowa" (is_depressed: 0 lub 1).

  Architektura:
    Warstwa wejściowa: 27 cech
    Warstwa ukryta 1: 128 neuronów (ReLU)
    Warstwa ukryta 2: 64 neurony (ReLU)
    Warstwa wyjściowa: 1 neuron (Sigmoid → prawdopodobieństwo 0-1)

  Optymalizacja: SGD z mini-batchami + Momentum (β = 0.9)
  Funkcja straty: Binary Cross-Entropy (Log Loss)
  # --------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------
# READING & PREPARING DATA
# --------------------------------------------------------------------
data = pd.read_csv("../data/digital_diet_mental_health.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'location_type'])
data = data.astype(float)

data = (data - data.mean()) / data.std()

data['is_depressed'] = np.where(
    (data['mental_health_score'] < 0.4) |
    (data['stress_level'] > 0.7) |
    (data['weekly_anxiety_score'] > 0.8),
    1.0, 0.0)

real_data = data.copy()

# TARGET (y) and FEATURES (X)
y = data['is_depressed'].values.reshape(2000, 1)
X = data.drop('is_depressed', axis=1).values

rows, cols = X.shape

# --------------------------------------------------------------------
# NEURAL NETWORK: A masterclass
# --------------------------------------------------------------------
class Mentally_Unwell_Prediction:
    """
    A multilayered neural network that binary classifies someones mental health status
    """
    # === Initialize the model
    def __init__(self, hidden_units = 128):
        self.input = cols
        self.output = 1
        self.hidden_units = hidden_units

        # Weights
        self.w1 = np.random.randn(self.input, self.hidden_units) * np.sqrt(2. / self.input)
        self.w2 = np.random.randn(self.hidden_units, 64) * np.sqrt(2. / self.hidden_units)
        self.w3 = np.random.randn(64, self.output) * np.sqrt(2. / 64)

        # Velocity
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.v3 = np.zeros_like(self.w3)

        # Biases - initialized to 0
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((self.output, 1))

    # === Foward Propagation
    # Foward move from input layer through hidden layers, multiplying neuron by weight
    def _forward_propagation(self, X):
        # Layer 1 (from input to hidden layer 1)
        self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.a2 = self._ReLU(self.z2)

        # Layer 2 (from hidden layer 1 to hidden layer 2)
        self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.a3 = self._ReLU(self.z3)

        # Output layer (Sigmoid for binary classification)
        self.z4 = np.dot(self.w3.T, self.a3) + self.b3
        self.a4 = self._sigmoid(self.z4)

        return self.a4

    # === Activation Function
    # Rectified Linear Unit (hidden layers)
    def _ReLU(self, Z): return np.maximum(Z, 0)

    # Sigmoid (output layer)
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))

    # === Loss Function
    # Binary Cross Entropy (Log Loss)
    def _loss(self, predict, y):
        """
        BCE = -(1/m) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
        The lower the result, the better the model
        """
        m = y.shape[0]
        logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        loss =- np.sum(logprobs) / m
        return loss

    # === Backwards Propagation
    # Calculating the gradient moving backwards
    def _backward_propagation(self, X, y):
        """
        Calculating the derivative of the loss function over every weight
        Chain: ∂L/∂w = ∂L/∂z · ∂z/∂w
        """
        predict = self._forward_propagation(X)
        rows = X.shape[0]

        # Output Layer (Sigmoid + BCE)
        dz4 = predict - y.T

        self.dw3 = (1 / rows) * np.dot(self.a3, dz4.T)
        self.db3 = (1 / rows) * np.sum(dz4, axis=1, keepdims=True)

        # Hidden Layer 2 (ReLU)
        dz3 = np.dot(self.w3, dz4) * self.ReLU_prime(self.z3)
        self.dw2 = (1 / rows) * np.dot(self.a2, dz3.T)
        self.db2 = (1 / rows) * np.sum(dz3, axis=1, keepdims=True)

        # Hidden Layer 1 (ReLU)
        dz2 = np.dot(self.w2, dz3) * self.ReLU_prime(self.z2)
        self.dw1 = (1 / rows) * np.dot(X.T, dz2.T)
        self.db1 = (1 / rows) * np.sum(dz2, axis=1, keepdims=True)

    def ReLU_prime(self, z): return (z>0).astype(float)
    def sigmoid_prime(self, z): return self._sigmoid(z) * (1 - self._sigmoid(z))

    # === Update Parameters
    # SGD Momentum
    def _update(self, learning_rate=0.01):
        """
        Momentum smoothes out the gradient rise
        v = β·v + (1-β)·grad
        w = w - lr·v
        """
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

    # === TRAINING
    def train(self, X_train, y_train, iteration=1000, learning_rate=0.001, batch_size=16):
        rows = X_train.shape[0]

        for i in range(iteration):
            idx = np.random.permutation(rows)

            X_s, y_s = X_train[idx], y_train[idx]

            for s in range(0, rows, batch_size):
                X_b, y_b = X_s[s:s + batch_size], y_s[s:s + batch_size]

                self._backward_propagation(X_b, y_b)
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

# --------------------------------------------------------------------
# TRAINING THE MODEL
# --------------------------------------------------------------------
def train(seperator=1600):
    X_train = X[:seperator]
    X_test = X[seperator:]

    y_train = y[:seperator]
    y_test = y[seperator:]

    print("=" * 60)
    print("CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    model = Mentally_Unwell_Prediction()

    model.train(X_train, y_train)
    pre_y = model.predict(X_test)
    score = model.score(pre_y, y_test)

    print(f'=== SCORE: {score:.2f}%')

    # === Comparison of last 15 data rows
    def show_comparison(model, X_test, y_test):
        probs = model.predict(X_test)
        predictions = (probs.T > 0.5).astype(int)

        comparison = pd.DataFrame({
            'Actual Healthstatus': y_test.flatten(),
            'Predicted Health status': predictions.flatten()
        })

        print("\n=== ACTUAL VS PREDICTED ===")
        print(comparison.tail(15).to_string(index=False))
        acc = (comparison['Actual Healthstatus']==comparison['Predicted Health status']).mean()
        print(f"\nAverage Error: {acc*100:.1f}")

    show_comparison(model, X_test, y_test)

    return model

# --------------------------------------------------------------------
# TESTING THE MODEL on different data
# --------------------------------------------------------------------
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

    train_features = original_df.drop(columns=['is_depressed'])
    model_columns = train_features.columns

    new_data = new_data.reindex(columns=model_columns, fill_value=0)

    new_data_scaled = (new_data - train_features.mean()) / train_features.std()

    X_custom = new_data_scaled.values
    raw_probs = model._forward_propagation(X_custom)

    print("\n=== FINAL TEST RESULTS ===")
    for i, prob in enumerate(raw_probs.T):
        risk = prob[0]
        status = "Unwell" if risk >= 0.5 else "Healthy"
        print(f"Test {i + 1}: Health -> Diagnose: {status}")

# --------------------------------------------------------------------
# PARAMETRIC TESTS - Grid Search
# --------------------------------------------------------------------
def test_classification_params():
    learning_rates = [0.01, 0.005, 0.003, 0.001]
    hidden_units_list = [64, 128, 256, 512]
    batch_sizes = [16, 32, 64, 128]

    X_train = X[:1600]
    X_test = X[1600:]

    y_train = y[:1600]
    y_test = y[1600:]

    results = []

    for lr in learning_rates:
        for hu in hidden_units_list:
            for bs in batch_sizes:

                print(f"\nTEST: lr={lr}, hidden={hu}, batch={bs}")

                model = Mentally_Unwell_Prediction(hidden_units=hu)

                model.train(
                    X_train,
                    y_train,
                    iteration=1000,
                    learning_rate=lr,
                    batch_size=bs
                )

                pred = model.predict(X_test)
                score = model.score(pred, y_test)

                results.append({
                    "learning rate":     lr,
                    "hidden units":      hu,
                    "batch size":        bs,
                    "train accuracy":    score
                })


    df = pd.DataFrame(results)

    print("\n=== RESULTS ===")
    print(df.sort_values("accuracy", ascending=False))

    best = df.loc[df['train accuracy'].idxmax()]
    print(best[['learning rate', 'hidden units', 'batch size', 'train accuracy']].to_string(index=False))

    df.to_csv("../test-results/classification/classification_param_tests.csv", index=False)
    return df

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
def main_func():
    clr = train()
    predict_new_users(clr, new_samples, real_data)

    df_class_results = test_classification_params()
    print(df_class_results.sort_values("accuracy", ascending=False).head(10))

if __name__ == "__main__":
    main_func()