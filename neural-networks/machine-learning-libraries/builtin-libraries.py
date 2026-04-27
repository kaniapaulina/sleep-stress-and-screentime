import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

# ==================================================================
# CLASSIFICATION
# ==================================================================

# -------------------------------------------------------------------
#  PRZYGOTOWYWANIE DANYCH
# -------------------------------------------------------------------
def class_prepare_data():
    data = pd.read_csv("../data/digital_diet_mental_health.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

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
    X_test = scaler.transform(X_test)

    print(f"[Dane] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[Dane] Klasa 1 (niezdrowi): {y.mean() * 100:.1f}%\n")

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
#  WBUDOWANE METODY SKIKITLEARN
# -------------------------------------------------------------------
def class_compare_models(X_train, X_test, y_train, y_test):
    """
    Trenuje i porównuje 4 modele z domyślnymi (sensownymi) parametrami.
    Odpowiada parametrom użytym w autorskim modelu NN.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP Classifier (NN)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            learning_rate_init=0.005,
            batch_size=32,
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    }

    results = []
    print(" = MODELE Z DOMYŚLNYMI PARAMETRAMI =")

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_f1 = f1_score(y_test, model.predict(X_test))

        print(f"  {name:<30} | train: {train_acc * 100:.2f}%  test: {test_acc * 100:.2f}%  F1: {test_f1:.3f}")
        results.append({
            "model": name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "f1_score": round(test_f1, 4)
        })

    return results




# ==================================================================
# REGRESSION
# ==================================================================

# -------------------------------------------------------------------
#  PRZYGOTOWYWANIE DANYCH
# -------------------------------------------------------------------
def reg_prepare_data():
    data = pd.read_csv("../data/digital_diet_mental_health.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = data.drop('user_id', axis=1)

    data = pd.get_dummies(data, columns=['gender', 'location_type'])

    # Feature engineering (jak w modelu własnym)
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
    X_test = scaler.transform(X_test)

    print(f"[Dane] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[Dane] Zakres snu: {y.min():.1f} – {y.max():.1f} h  |  Śr.: {y.mean():.2f} h\n")

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
#  WBUDOWANE METODY SKIKITLEARN
# -------------------------------------------------------------------
def reg_compare_default_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "MLP Regressor (NN)": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            learning_rate_init=0.005,
            batch_size=32,
            activation='relu',
            solver='adam',
            max_iter=2000,
            momentum=0.9,
            early_stopping=True,
            random_state=42
        )
    }

    results = []
    print(" = MODELE Z DOMYŚLNYMI PARAMETRAMI")

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        train_mae = mean_absolute_error(y_train, model.predict(X_train))

        print(f"  {name:<30} | train R²: {train_r2:.4f}  test R²: {test_r2:.4f}  MAE: {test_mae:.4f} h")
        results.append({
            "model": name,
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 4),
            "test_mae": round(test_mae, 4),
        })

    return results





# ==================================================================
# MAIN FUNCTION
# ==================================================================

def main():
    # CLASSIFICATION
    X_train, X_test, y_train, y_test = class_prepare_data()
    all_results = class_compare_models(X_train, X_test, y_train, y_test)
    df = pd.DataFrame(all_results)

    out_path = "classification_builtin_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  NAJLEPSZA KONFIGURACJA - KLASYFIKACJA")
    print("=" * 65)
    best = df.loc[df.groupby('model')['test_accuracy'].idxmax()]
    print(best[['model', 'train_accuracy', 'test_accuracy', 'f1_score']].to_string(
        index=False))

    # REGRESSION
    X_train, X_test, y_train, y_test = reg_prepare_data()
    all_results = reg_compare_default_models(X_train, X_test, y_train, y_test)
    df = pd.DataFrame(all_results)

    out_path = "regression_builtin_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  NAJLEPSZA KONFIGURACJA - REGRESJA")
    print("=" * 65)
    best = df.loc[df.groupby('model')['test_r2'].idxmax()]
    print(best[['model', 'train_r2', 'test_r2', 'test_mae']].to_string(index=False))



if __name__ == "__main__":
    main()