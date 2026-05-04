"""
  PORÓWNANIE MODELI REGRESYJNYCH

  Modele:
    1. Regresja Liniowa         (parametry: model, alpha (regularyzacja Ridge), alpha (regularyzacja Lasso))
    2. k-Najbliższych Sąsiadów  (parametry: liczba sasiadow, weights, metric)
    3. Las Losowy               (parametry: ilosc drzew, glebokosc drzew, ilosc lisci)
    4. MLP Regressor (NN)       (parametry: warstwy ukryte, tempo uczenia, funkcja aktywacji, rozmiar batcha)
    5. Decision Tree           (parametry: głębokość, minimalna liczba próbek do podziału, minimalna liczba próbek w liściu, kryterium podziału)
    6. SVM                     (parametry: C, jądro, gamma)
"""

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("test-results/regression", exist_ok=True)

def prepare_data():
    data = pd.read_csv("data/digital_diet_mental_health.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = data.drop('user_id', axis=1)

    data = pd.get_dummies(data, columns=['gender', 'location_type'])

    data['stress_phone_interaction'] = data['stress_level'] * data['phone_usage_hours']
    data['total_digital_load'] = (
        data['phone_usage_hours'] +
        data['laptop_usage_hours'] +
        data['gaming_hours']
    )

    y = data['sleep_duration_hours'].values
    X = data.drop(columns='sleep_duration_hours').values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Dane] Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def find_best_parameter(X_train, X_test, y_train, y_test):

    pipelines = {
        "Linear Regression": Pipeline([('model', LinearRegression())]),
        "K-Nearest Neighbors": Pipeline([('model', KNeighborsRegressor())]),
        "Random Forest": Pipeline([('model', RandomForestRegressor(random_state=42))]),
        "MLP Regressor": Pipeline([('model', MLPRegressor(random_state=42, early_stopping=True, max_iter=2000))]),
        "Decision Tree": Pipeline([('model', DecisionTreeRegressor(random_state=42))]),
        "SVM": Pipeline([('model', SVR())])
    }

    param_grids = {
        "Linear Regression": {
        },
        "K-Nearest Neighbors": {
            'model__n_neighbors': [1, 3, 5, 10, 20],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        },
        "Random Forest": {
            'model__n_estimators': [10, 50, 100, 200],
            'model__max_depth': [None, 3, 5, 10],
            'model__min_samples_leaf': [1, 2, 5, 10]
        },
        "MLP Regressor": {
            'model__hidden_layer_sizes': [(64,), (128, 64), (64, 32)],
            'model__learning_rate_init': [0.01, 0.001],
            'model__activation': ['relu', 'tanh'],
            'model__batch_size': [16, 32, 64]
        },
        "Decision Tree": {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10, 50],
            'model__criterion': ['squared_error', 'friedman_mse'],
            'model__min_samples_leaf': [1, 2, 4]
        },
        "SVM": {
            'model__C': [0.1, 1, 10],
            'model__kernel': ["linear", "poly", "rbf", "sigmoid"],
            'model__gamma': ["scale", "auto", 0.01, 0.1],

        }
    }

    best_results = {}
    all_results = []

    for name, pipe in pipelines.items():
        print(f"Running GridSearchCV for {name}...")

        grid = GridSearchCV(
            pipe,
            param_grids[name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_results[name] = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        }

        df = pd.DataFrame(grid.cv_results_)
        df['model_name'] = name
        cols = ['model_name', 'mean_test_score', 'std_test_score', 'rank_test_score'] + [c for c in df.columns if 'param_' in c]
        all_results.append(df[cols])

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("test-results/regression/grid_search_full_results.csv", index=False)

    print("=" * 60)
    for name, result in best_results.items():
        print(f"\n{name} Best Params: {result['best_params']}")
    print("=" * 60)


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),

        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),

        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),

        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }

    return results

def add_result(results, model_name, parameter_name, parameter_value, metrics):
    results.append({
        "model": model_name,
        "parameter_name": parameter_name,
        "parameter_value": str(parameter_value),
        **metrics
    })

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(
            n_neighbors=20
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "MLP Regressor (NN)":   MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.01,
            batch_size=32,
            activation='relu',
            solver='adam',
            max_iter=2000,
            momentum=0.9,
            early_stopping=True,
            random_state=42
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=4,
            min_samples_split=40,
            criterion="squared_error",
            random_state=42
        ),
        "SVM": SVR(
            C=0.01,
            kernel="rbf",
            gamma=0.01
        ),
    }

    results = []
    print("=" * 65)
    print("  MODELE Z DOMYŚLNYMI PARAMETRAMI")
    print("=" * 65)

    for name, model in models.items():
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)

        print(f"  {name:<30} | Train MAE: {values['train_mae']:.4f} h | Test MAE: {values['test_mae']:.4f} h")
        add_result(results, name, "default", "default",  values)


def test_linear_regression(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - typ modelu: [OLS, Ridge z różnym alpha, Lasso z różnym alpha]
        - alpha (regularyzacja Ridge): [0.01, 0.1, 1.0, 10.0]
        - alpha (regularyzacja Lasso): [0.001, 0.01, 0.1, 1.0]
    """
    results = []

    print("\n[Linear Regression] Testowanie wariantów regularyzacji")

    model = LinearRegression()
    values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
    add_result(results, "Linear Regression", "regularization", "None (OLS)", values)

    # Ridge
    print("[Linear Regression] Testowanie parametru alpha (Ridge)")
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Linear Regression", "Ridge", alpha, values)

    # Lasso
    print("[Linear Regression] Testowanie parametru alpha (Lasso)")
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        model = Lasso(alpha=alpha, max_iter=5000)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Linear Regression", "Lasso", alpha, values)

    return results

def test_knn(X_train, X_test, y_train, y_test):
    '''
    Testowane parametry:
        - Liczba sąsiadów: [1, 3, 5, 10, 20]
        - Weights: ['uniform', 'distance']
        - Metric: ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    '''
    results = []

    print("\n[KNN] Testowanie parametru n_neighbors")
    for k in [1, 3, 5, 10, 20]:
        model = KNeighborsRegressor(n_neighbors=k)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "n_neighbors", k, values)

    print("[KNN] Testowanie parametru weights")
    for w in ['uniform', 'distance']:
        model = KNeighborsRegressor(n_neighbors=5, weights=w)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "weights", w, values)

    print("[KNN] Testowanie parametru metric")
    for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        model = KNeighborsRegressor(n_neighbors=5, metric=metric)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "metric", metric, values)

    return results

def test_random_forest(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - Ilość drzew: [10, 50, 100, 200]
        - Głębokość drzewa: [None, 3, 5, 10]
        - Ilość liści: [1, 2, 5, 10]
    """
    results = []

    print("\n[Random Forest] Testowanie parametru n_estimators")
    for n in [10, 50, 100, 200]:
        model = RandomForestRegressor(n_estimators=n, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "n_estimators", n, values)

    print("[Random Forest] Testowanie parametru max_depth")
    for d in [None, 3, 5, 10]:
        model = RandomForestRegressor(n_estimators=100, max_depth=d, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "max_depth", d, values)

    print("[Random Forest] Testowanie parametru min_samples_leaf")
    for msl in [1, 2, 5, 10]:
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=msl, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "min_samples_leaf", msl, values)

    return results

def test_mlp(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - Warstwy ukryte: [(64,), (128, 64), (128, 64, 32), (256, 128, 64)]
        - Tempo uczenia: [0.01, 0.005, 0.001, 0.0005]
        - Funkcja aktywacji: ['relu', 'tanh', 'logistic', 'identity']
        - Rozmiar batcha: [16, 32, 64, 128]
    """
    results = []

    print("\n[MLP] Testowanie parametru hidden_layer_sizes")
    for hl in [(64,), (128, 64), (128, 64, 32), (256, 128, 64)]:
        model = MLPRegressor(hidden_layer_sizes=hl, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "hidden_layer_sizes", hl, values)

    print("[MLP] Testowanie parametru learning_rate_init")
    for lr in [0.01, 0.005, 0.001, 0.0005]:
        model = MLPRegressor(learning_rate_init=lr, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "learning_rate_init", lr, values)

    print("[MLP] Testowanie parametru activation")
    for act in ['relu', 'tanh', 'logistic', 'identity']:
        model = MLPRegressor(activation=act, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "activation", act, values)

    print("[MLP] Testowanie parametru batch_size")
    for bs in [16, 32, 64, 128]:
        model = MLPRegressor(batch_size=bs, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "batch_size", bs, values)

    return results

def test_decision_tree(X_train, X_test, y_train, y_test):
    """
    Parametry testowane:
        - max_depth: [3, 5, 10, None]
        - min_samples_split: [2, 5, 10, 50]
        - criterion: ['absolute_error', 'squared_error', 'friedman_mse']
        - min_samples_leaf: [1, 2, 4]
    """
    results = []

    print("\n[Decision Tree] Testowanie parametru max_depth")
    for hl in [3, 5, 10, None]:
        model = DecisionTreeRegressor(max_depth=hl, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "max_depth", hl, values)

    print("\n[Decision Tree] Testowanie parametru min_samples_split")
    for ms in [2, 5, 10, 50]:
        model = DecisionTreeRegressor(min_samples_leaf=ms, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "min_samples_split", ms, values)

    print("\n[Decision Tree] Testowanie parametru criterion")
    for criterion in ['absolute_error', 'squared_error', 'friedman_mse']:
        model = DecisionTreeRegressor(criterion=criterion, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "criterion", criterion, values)

    print("\n[Decision Tree] Testowanie parametru min_samples_leaf")
    for ms in [1, 2, 4]:
        model = DecisionTreeRegressor(min_samples_leaf=ms, random_state=42)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "min_samples_leaf", ms, values)

    return results

def test_svm(X_train, X_test, y_train, y_test):
    """
    Parametry testowane:
        - C: [0.1, 1, 10]
        - kernel: ['linear', 'rbf', 'poly', 'sigmoid']
        - gamma: [0.1, 1, 'auto', 'scale']
    """
    results = []

    print("\n[SVM] Testowanie parametru C")
    for C in [0.1, 1, 10]:
        model = SVR(kernel='linear', C=C)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "C", C, values)

    print("\n[SVM] Testowanie parametru kernel")
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        model = SVR(kernel=kernel)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "kernel", kernel, values)

    print("\n[SVM] Testowanie parametru gamma")
    for gamma in [0.1, 1, 'auto', 'scale']:
        model = SVR(gamma=gamma)
        values = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "gamma", gamma, values)

    return results


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    find_best_parameter(X_train, X_test, y_train, y_test)
    compare_models(X_train, X_test, y_train, y_test)

    all_results = []
    all_results += test_linear_regression(X_train, X_test, y_train, y_test)
    all_results += test_knn(X_train, X_test, y_train, y_test)
    all_results += test_random_forest(X_train, X_test, y_train, y_test)
    all_results += test_mlp(X_train, X_test, y_train, y_test)

    df = pd.DataFrame(all_results)

    out_path = "test-results/regression/regression_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  NAJLEPSZA KONFIGURACJA DLA KAŻDEGO MODELU")
    print("=" * 65)
    best = df.loc[df.groupby('model')['test_mae'].idxmin()]
    print(best[['model', 'parameter_name', 'parameter_value', 'test_mae']].to_string(index=False))

    return df


if __name__ == "__main__":
    main()
