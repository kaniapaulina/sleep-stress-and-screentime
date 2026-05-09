import pandas as pd
import numpy as np

def regression_model_test():
    data = pd.read_csv(r"../data/digital_diet_mental_health.csv")
    #data = (data.sample(frac=1)
    data = data.reset_index(drop=True)

    data = data.drop('user_id', axis=1)
    data = pd.get_dummies(data, columns=['gender', 'location_type'])
    data = data.astype(float)

    data['stress_phone_interaction'] = data['stress_level'] * data['phone_usage_hours']
    data['total_digital_load'] = data['phone_usage_hours'] + data['laptop_usage_hours'] + data['gaming_hours']

    real_data = data.copy()

    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    y = data['sleep_duration_hours'].values.reshape(2000, 1)
    X = data.drop(columns='sleep_duration_hours').values

    class Sleep_Prediction:
        def __init__(self, layers, activation='relu', keep_prob=0.9):
            self.weights = []
            self.velocity = []
            self.biases = []
            self.masks = []

            self.act = activation
            self.keep_prob = keep_prob
            self.a = []
            self.z = []

            for i in range(1, len(layers)):
                weight = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2. / layers[i-1])
                velocity = np.zeros_like(weight)
                biases = np.zeros((layers[i], 1))

                self.weights.append(weight)
                self.velocity.append(velocity)
                self.biases.append(biases)

        def activation(self, Z, type):
            if type == 'relu': return np.maximum(0, Z)
            if type == 'sigmoid': return 1 / (1 + np.exp(-Z))
            if type == 'tanh': return np.tanh(Z)
            if type == 'leaky_relu': return np.where(Z > 0, Z, Z * 0.01)
            return Z

        def activation_prime(self, Z, type):
            if type == 'relu': return (Z > 0).astype(float)
            if type == 'sigmoid':
                s = 1 / (1 + np.exp(-Z))
                return s * (1 - s)
            if type == 'tanh': return 1 - np.tanh(Z) ** 2
            if type == 'leaky_relu': return np.where(Z > 0, 1, 0.01)
            return 1

        def _forward_propagation(self, X, training=True):
            self.a = [X.T]
            self.z = []
            self.masks = []
            curr_a = X.T

            for i in range(len(self.weights) - 1):
                curr_z = np.dot(self.weights[i].T, curr_a) + self.biases[i]
                curr_a = self.activation(curr_z, self.act)

                if training:
                    mask = (np.random.rand(*curr_a.shape) < self.keep_prob)
                    curr_a = (curr_a * mask) / self.keep_prob
                    self.masks.append(mask)
                else:
                    self.masks.append(None)

                self.z.append(curr_z)
                self.a.append(curr_a)

            last_z = np.dot(self.weights[-1].T, curr_a) + self.biases[-1]
            self.z.append(last_z)
            self.a.append(last_z)
            return self.a[-1]


        def _loss(self, predict, y):
            m = y.shape[0]
            loss = 1 / m * np.sum((predict - y.T) ** 2)
            return loss

        def _backward_propagation(self, X, y):
            rows = X.shape[0]
            lambda_param = 0.001
            self.grads_w = [None] * len(self.weights)
            self.grads_b = [None] * len(self.biases)

            dz = self.a[-1] - y.T

            for i in reversed(range(len(self.weights))):
                self.grads_w[i] = (1 / rows) * np.dot(self.a[i], dz.T) + (lambda_param * self.weights[i])
                self.grads_b[i] = (1 / rows) * np.sum(dz, axis=1, keepdims=True)

                if i > 0:
                    dz = np.dot(self.weights[i], dz) * self.activation_prime(self.z[i - 1], self.act)
                    if self.masks[i - 1] is not None:
                        dz = (dz * self.masks[i - 1]) / self.keep_prob

        def _update(self, lr, beta=0.9):
            for i in range(len(self.weights)):
                self.velocity[i] = beta * self.velocity[i] + (1 - beta) * self.grads_w[i]
                self.weights[i] -= lr * self.velocity[i]
                self.biases[i] -= lr * self.grads_b[i]

        def train(self, X_train, y_train, iteration=1000, lr=0.005, batch_size=32):
            for _ in range(iteration):
                idx = np.random.permutation(X_train.shape[0])
                X_s, y_s = X_train[idx], y_train[idx]
                for s in range(0, X_train.shape[0], batch_size):
                    self._forward_propagation(X_s[s:s + batch_size])
                    self._backward_propagation(X_s[s:s + batch_size], y_s[s:s + batch_size])
                    self._update(lr)

        def predict(self, X):
            y_hat_scaled = self._forward_propagation(X, training=False)
            return np.maximum(0, np.array(y_hat_scaled.T))

        def score(self, predict, y):
            return np.mean(np.abs(predict - y))


    def run_full_analysis():
        results = []

        base_arch = [29, 64, 32, 1]
        base_act = 'relu'
        base_lr = 0.005
        base_bs = 32
        base_sep = 1600
        base_iter = 400
        base_prob = 0.9

        params_to_test = {
            "architecture": [
                [29, 1], [29, 32, 1], [29, 64, 1], [29, 64, 32, 1], [29, 128, 64, 1],[29, 128, 64, 32, 1]
            ],
            "activation_function": ['relu', 'tanh', 'sigmoid', 'leaky_relu'],
            "learning_rate": [0.01, 0.005, 0.001, 0.0005],
            "batch_size": [16, 32, 64, 128],
            "seperator": [1000, 1200, 1500, 1600, 1800],
            "iteration": [200, 500, 1000, 2000],
            "dropout_probability": [1, 0.9, 0.8, 0.5]
        }

        for param_name, values in params_to_test.items():

            for val in values:
                print(f"Testing {param_name} : {val}")
                arch = val if param_name == "architecture" else base_arch
                act = val if param_name == "activation_function" else base_act
                lr = val if param_name == "learning_rate" else base_lr
                bs = val if param_name == "batch_size" else base_bs
                sep = val if param_name == "seperator" else base_sep
                it = val if param_name == "iteration" else base_iter
                prob = val if param_name == "dropout_probability" else base_prob

                repeat_train_mae = []
                repeat_test_mae = []

                for i in range(10):
                    y_raw = real_data['sleep_duration_hours'].values.reshape(-1, 1)
                    X_raw = real_data.drop(columns='sleep_duration_hours').values

                    sep = sep
                    X_train_raw, X_test_raw = X_raw[:sep], X_raw[sep:]
                    y_train_raw, y_test_raw = y_raw[:sep], y_raw[sep:]

                    x_min, x_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
                    y_min, y_max = y_train_raw.min(), y_train_raw.max()

                    x_range = np.where((x_max - x_min) == 0, 1, x_max - x_min)
                    y_range = (y_max - y_min) if (y_max - y_min) != 0 else 1

                    X_train = (X_train_raw - x_min) / x_range
                    X_test = (X_test_raw - x_min) / x_range

                    y_train = (y_train_raw - y_min) / y_range
                    y_test = (y_test_raw - y_min) / y_range

                    model = Sleep_Prediction(layers=arch, activation=act, keep_prob=prob)
                    model.train(X_train, y_train, iteration=it, lr=lr, batch_size=bs)

                    train_mae = model.score(model.predict(X_train), y_train)
                    test_mae = model.score(model.predict(X_test), y_test)

                    repeat_train_mae.append(train_mae)
                    repeat_test_mae.append(test_mae)

                    y_range = y_max - y_min

                print(np.mean(repeat_train_mae))
                print(round(np.mean(repeat_train_mae)*y_range, 2))

                results.append({
                    "Tested Param": param_name,
                    "Value": str(val),
                    "Train MAE": np.mean(repeat_train_mae),
                    "Test MAE": np.mean(repeat_test_mae),
                    "Train Error": round(np.mean(repeat_train_mae)*y_range, 2),
                    "Test Error":  round(np.mean(repeat_test_mae)*y_range, 2)
                })

        return pd.DataFrame(results)

    df = run_full_analysis()
    df.to_csv(r"../test-results/regression/param_tests_results_new.csv", index=False)

regression_model_test()