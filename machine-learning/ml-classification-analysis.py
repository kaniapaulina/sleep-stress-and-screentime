"""
  PORÓWNANIE MODELI KLASYFIKACYJNYCH

  Modele:
    1. Regresja Logistyczna    (parametry: C (regularyzacja), solver, penalty)
    2. k-Najbliższych Sąsiadów (parametry: liczba sasiadow, weights, metric)
    3. Las Losowy              (parametry: ilosc drzew, glebokosc drzew, ilosc lisci)
    4. MLP Classifier (NN)     (parametry: warstwy ukryte, tempo uczenia, funkcja aktywacji, rozmiar batcha)
    5. Decision Tree           (parametry: głębokość, minimalna liczba próbek do podziału, minimalna liczba próbek w liściu, kryterium podziału)
    6. SVM                     (parametry: C, jądro, gamma)
"""

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("test-results/classification", exist_ok=True)

def prepare_data():
    data = pd.read_csv("data/digital_diet_mental_health.csv")
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)

    data = data.drop('user_id', axis=1)
    data = pd.get_dummies(data, columns=['gender', 'location_type'])
    data = data.astype(float)
    data = (data - data.mean()) / data.std()

    data['is_depressed'] = np.where(
        (data['mental_health_score'] < 0.4) |
        (data['stress_level'] > 0.7) |
        (data['weekly_anxiety_score'] > 0.8),
        1.0, 0.0
    )

    y = data['is_depressed'].values
    X = data.drop('is_depressed', axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Dane] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def find_best_parameter(X_train, X_test, y_train, y_test):
    sampler = SMOTE(random_state=42)

    pipelines = {
        "Logistic Regression": Pipeline([('sampler', sampler), ('model', LogisticRegression(random_state=42))]),
        "K-Nearest Neighbors": Pipeline([('sampler', sampler), ('model', KNeighborsClassifier())]),
        "Random Forest": Pipeline([('sampler', sampler), ('model', RandomForestClassifier(random_state=42))]),
        "MLP Classifier": Pipeline([('sampler', sampler), ('model', MLPClassifier(random_state=42, early_stopping=True, max_iter=2000))]),
        "Decision Tree": Pipeline([('sampler', sampler), ('model', DecisionTreeClassifier(random_state=42))]),
        "SVM": Pipeline([('sampler', sampler), ('model', SVC(random_state=42))])
    }

    param_grids = {
        "Logistic Regression": {
            'model__C': [0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg']
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
        "MLP Classifier": {
            'model__hidden_layer_sizes': [(64,), (128, 64), (64, 32)],
            'model__learning_rate_init': [0.01, 0.001],
            'model__activation': ['relu', 'tanh'],
            'model__batch_size': [16, 32, 64]
        },
        "Decision Tree": {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10, 50],
            'model__criterion': ['gini', 'entropy'],
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
            scoring='f1_macro',
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
    results_df.to_csv("test-results/classification/grid_search_full_results.csv", index=False)

    print("=" * 60)
    for name, result in best_results.items():
        print(f"\n{name} Best Params: {result['best_params']}")
    print("=" * 60)

def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
    }

    return results

def add_result(results, model_name, parameter_name, parameter_value, values):
    results.append({
        "model": model_name,
        "parameter_name": parameter_name,
        "parameter_value": str(parameter_value),
        **values
    })

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "MLP Classifier (NN)": MLPClassifier(
            hidden_layer_sizes= (128, 64),
            learning_rate_init= 0.01,
            batch_size=32,
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=2,
            criterion="gini",
            random_state=42
        ),
        "SVM": SVC(
            C=10,
            kernel="rbf",
            gamma=0.1,
            random_state=42
        ),
    }

    results = []
    print("=" * 65)
    print("  MODELE Z DOMYŚLNYMI PARAMETRAMI")
    print("=" * 65)

    for name, model in models.items():
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)

        print(f"  {name:<30} | train: {values['train_accuracy']*100:.2f}%  test: {values['test_accuracy']*100:.2f}%")
        add_result(results, name, "default", "default", values=values)


def test_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - C (regularyzacja): [0.01, 0.1, 1, 10, 100]
        - solver: ['lbfgs', 'liblinear', 'saga', 'newton-cg']
        - penalty: ['l1', 'l2', 'elasticnet', None]
    """
    results = []

    print("\n[Logistic Regression] Testowanie parametru C (regularyzacja)")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Logistic Regression", "C", C, values=values)

    print("[Logistic Regression] Testowanie parametru solver")
    for solver in ['lbfgs', 'liblinear', 'saga', 'newton-cg']:
        try:
            model = LogisticRegression(solver=solver, max_iter=1000, random_state=42)
            values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
            add_result(results, "Logistic Regression", "solver", solver, values=values)
        except Exception as e:
            print(f"  Pominięto solver={solver}: {e}")

    print("[Logistic Regression] Testowanie parametru penalty")
    for penalty, solver in [('l1','liblinear'), ('l2','lbfgs'), ('elasticnet','saga'), (None,'lbfgs')]:
        try:
            kwargs = {"l1_ratio": 0.5} if penalty == 'elasticnet' else {}
            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000, random_state=42, **kwargs)
            values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
            add_result(results, "Logistic Regression", "penalty", penalty, values=values)
        except Exception as e:
            print(f"  Pominięto penalty={penalty}: {e}")

    return results

def test_knn(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - n_neighbors: [1, 3, 5, 10, 20]
        - weights: ['uniform', 'distance']
        - metric: ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    """
    results = []

    print("\n[KNN] Testowanie parametru n_neighbors")
    for k in [1, 3, 5, 10, 20]:
        model = KNeighborsClassifier(n_neighbors=k)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "n_neighbors", k, values=values)

    print("[KNN] Testowanie parametru weights")
    for w in ['uniform', 'distance']:
        model = KNeighborsClassifier(n_neighbors=5, weights=w)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "weights", w, values=values)

    print("[KNN] Testowanie parametru metric")
    for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        model = KNeighborsClassifier(n_neighbors=5, metric=metric)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "KNN", "metric", metric, values=values)

    return results

def test_random_forest(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - n_estimators: [10, 50, 100, 200]
        - max_depth: [None, 3, 5, 10]
        - min_samples_leaf: [1, 2, 5, 10]
    """
    results = []

    print("\n[Random Forest] Testowanie parametru n_estimators")
    for n in [10, 50, 100, 200]:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "n_estimators", n, values=values)

    print("[Random Forest] Testowanie parametru max_depth")
    for d in [None, 3, 5, 10]:
        model = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "max_depth", d, values=values)

    print("[Random Forest] Testowanie parametru min_samples_leaf")
    for msl in [1, 2, 5, 10]:
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=msl, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Random Forest", "min_samples_leaf", msl, values=values)

    return results

def test_mlp(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
        - hidden_layer_sizes: [(64,), (128,64), (128,64,32), (256,128,64)]
        - learning_rate_init: [0.01, 0.005, 0.001, 0.0005]
        - activation: ['relu', 'tanh', 'logistic', 'identity']
        - batch_size: [16, 32, 64, 128]
    """
    results = []

    print("\n[MLP] Testowanie parametru hidden_layer_sizes")
    for hl in [(64,), (128, 64), (128, 64, 32), (256, 128, 64)]:
        model = MLPClassifier(hidden_layer_sizes=hl, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "hidden_layer_sizes", hl, values=values)

    print("[MLP] Testowanie parametru learning_rate_init")
    for lr in [0.01, 0.005, 0.001, 0.0005]:
        model = MLPClassifier(learning_rate_init=lr, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "learning_rate_init", lr, values=values)

    print("[MLP] Testowanie parametru activation")
    for act in ['relu', 'tanh', 'logistic', 'identity']:
        model = MLPClassifier(activation=act, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "activation", act, values=values)

    print("[MLP] Testowanie parametru batch_size")
    for bs in [16, 32, 64, 128]:
        model = MLPClassifier(batch_size=bs, max_iter=1000, early_stopping=True, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "MLP", "batch_size", bs, values=values)

    return results

def test_decision_tree(X_train, X_test, y_train, y_test):
    """
    Parametry testowane:
        - max_depth: [3, 5, 10, None]
        - min_samples_split: [2, 5, 10, 50]
        - criterion: ['gini', 'entropy']
        - min_samples_leaf: [1, 2, 4]
    """
    results = []

    print("\n[Decision Tree] Testowanie parametru max_depth")
    for hl in [3, 5, 10, None]:
        model = DecisionTreeClassifier(max_depth=hl, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "max_depth", hl, values=values)

    print("\n[Decision Tree] Testowanie parametru min_samples_split")
    for ms in [2, 5, 10, 50]:
        model = DecisionTreeClassifier(min_samples_leaf=ms, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "min_samples_split", ms, values=values)

    print("\n[Decision Tree] Testowanie parametru criterion")
    for criterion in ['gini', 'entropy']:
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "criterion", criterion, values=values)

    print("\n[Decision Tree] Testowanie parametru min_samples_leaf")
    for ms in [1, 2, 4]:
        model = DecisionTreeClassifier(min_samples_leaf=ms, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "Decision Tree", "min_samples_leaf", ms, values=values)

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
        model = SVC(kernel='linear', C=C, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "C", C, values=values)

    print("\n[SVM] Testowanie parametru kernel")
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        model = SVC(kernel=kernel, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "kernel", kernel, values=values)

    print("\n[SVM] Testowanie parametru gamma")
    for gamma in [0.1, 1, 'auto', 'scale']:
        model = SVC(gamma=gamma, random_state=42)
        values = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
        add_result(results, "SVM", "gamma", gamma, values=values)

    return results

def main():
    X_train, X_test, y_train, y_test = prepare_data()
    find_best_parameter(X_train, X_test, y_train, y_test)
    compare_models(X_train, X_test, y_train, y_test)

    all_results = []
    all_results += test_logistic_regression(X_train, X_test, y_train, y_test)
    all_results += test_knn(X_train, X_test, y_train, y_test)
    all_results += test_random_forest(X_train, X_test, y_train, y_test)
    all_results += test_mlp(X_train, X_test, y_train, y_test)
    all_results += test_decision_tree(X_train, X_test, y_train, y_test)
    all_results += test_svm(X_train, X_test, y_train, y_test)

    df = pd.DataFrame(all_results)

    out_path = "test-results/classification/classification_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  NAJLEPSZA KONFIGURACJA DLA KAŻDEGO MODELU")
    print("=" * 65)
    best = df.loc[df.groupby('model')['test_accuracy'].idxmax()]
    print(best[['model', 'parameter_name', 'parameter_value', 'train_accuracy', 'test_accuracy']].to_string(index=False))

if __name__ == "__main__":
    main()