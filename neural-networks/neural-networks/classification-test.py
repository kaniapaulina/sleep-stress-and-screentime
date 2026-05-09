import pandas as pd
import numpy as np

def classification_model_test():
    data = pd.read_csv(r"../data/digital_diet_mental_health.csv")
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

    y = data['is_depressed'].values.reshape(2000, 1)
    X = data.drop('is_depressed', axis=1).values

    class Mentally_Unwell_Prediction:
        def __init__(self, layers, activation='relu'):
            self.weights = []
            self.velocity = []
            self.biases = []

            self.act = activation
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

        def _forward_propagation(self, X):
            self.a = [X.T]
            self.z = []
            curr_a = X.T

            for i in range(len(self.weights) - 1):
                curr_z = np.dot(self.weights[i].T, curr_a) + self.biases[i]
                curr_a = self.activation(curr_z, self.act)
                self.z.append(curr_z)
                self.a.append(curr_a)

            last_z = np.dot(self.weights[-1].T, curr_a) + self.biases[-1]
            self.z.append(last_z)
            self.a.append(self.activation(last_z, 'sigmoid'))
            return self.a[-1]


        def _loss(self, predict, y):
            m = y.shape[0]
            logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
            loss =- np.sum(logprobs) / m
            return loss

        def _backward_propagation(self, X_batch, y_batch):
            rows = X_batch.shape[0]
            lambda_param = 0.001
            self.grads_w = [None] * len(self.weights)
            self.grads_b = [None] * len(self.biases)

            dz = self.a[-1] - y_batch.T

            for i in reversed(range(len(self.weights))):
                self.grads_w[i] = (1 / rows) * np.dot(self.a[i], dz.T) + (lambda_param * self.weights[i])
                self.grads_b[i] = (1 / rows) * np.sum(dz, axis=1, keepdims=True)

                if i > 0:
                    dz = np.dot(self.weights[i], dz) * self.activation_prime(self.z[i - 1], self.act)

        def _update(self, lr, beta=0.9):
            for i in range(len(self.weights)):
                self.velocity[i] = beta * self.velocity[i] + (1 - beta) * self.grads_w[i]
                self.weights[i] -= lr * self.velocity[i]
                self.biases[i] -= lr * self.grads_b[i]

        def train(self, X_train, y_train, iteration=1000, lr=0.005, batch_size=32):
            for _ in range(iteration):
                idx = np .random.permutation(X_train.shape[0])
                X_s, y_s = X_train[idx], y_train[idx]
                for s in range(0, X_train.shape[0], batch_size):
                    self._forward_propagation(X_s[s:s + batch_size])
                    self._backward_propagation(X_s[s:s + batch_size], y_s[s:s + batch_size])
                    self._update(lr)

        def predict(self, X):
            y_hat = self._forward_propagation(X)
            y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
            return np.array(y_hat)

        def score(self, predict, y):
            predict = predict.flatten()
            y = y.flatten()
            cnt = np.sum(predict == y)
            return (cnt / len(y)) * 100

    def train(sep, it, lr, bs):
        X_train = X[:sep]
        X_test = X[sep:]

        y_train = y[:sep]
        y_test = y[sep:]

        clr = Mentally_Unwell_Prediction()

        clr.train(X_train, y_train, it, lr, bs)
        pre_y = clr.predict(X_test)
        score = clr.score(pre_y, y_test)

        return score

    def run_full_analysis():
        results = []

        base_arch = [28, 32, 1]
        base_act = 'relu'
        base_lr = 0.005
        base_bs = 32
        base_sep = 1600
        base_iter = 500

        params_to_test = {
            "architecture": [
                [28, 1], [28, 32, 1], [28, 64, 1], [28, 64, 32, 1], [28, 128, 64, 1],
            ],
            "activation_function": ['relu', 'tanh', 'sigmoid', 'leaky_relu'],
            "learning_rate": [0.01, 0.005, 0.001, 0.0005],
            "batch_size": [16, 32, 64, 128],
            "seperator": [1000, 1200, 1500, 1600, 1800],
            "iteration": [200, 500, 1000, 2000]
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

                repeat_train_acc = []
                repeat_test_acc = []

                for i in range(10):
                    X_train, X_test = X[:sep], X[sep:]
                    y_train, y_test = y[:sep], y[sep:]

                    model = Mentally_Unwell_Prediction(layers=arch, activation=act)
                    model.train(X_train, y_train, iteration=it, lr=lr, batch_size=bs)

                    train_mae = model.score(model.predict(X_train), y_train)
                    test_mae = model.score(model.predict(X_test), y_test)

                    repeat_train_acc.append(train_mae)
                    repeat_test_acc.append(test_mae)

                results.append({
                    "Tested Param": param_name,
                    "Value": str(val),
                    "Train Accuracy": np.mean(repeat_train_acc),
                    "Test Accuracy": np.mean(repeat_test_acc),
                })

        return pd.DataFrame(results)

    df = run_full_analysis()
    df.to_csv(r"../test-results/classification/param_tests_results.csv", index=False)

classification_model_test()