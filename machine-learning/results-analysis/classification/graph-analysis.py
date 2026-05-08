import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)

results_df = pd.read_csv("../../test-results/classification/classification_comparison.csv")

def create_graph(df, model, param):
    data = df[
        (df["model"] == model) &
        (df["parameter_name"] == param)
        ].copy()

    data["parameter_value"] = data["parameter_value"].fillna("None")

    data["Value_Numeric"] = pd.to_numeric(data["parameter_value"], errors='coerce')

    if data["Value_Numeric"].notna().all():
        data = data.sort_values("Value_Numeric")
    else:
        data = data.sort_values("parameter_value")

    x = data["parameter_value"]
    train = data["train_accuracy"]
    test = data["test_accuracy"]

    x_labels = [str(val) for val in x]
    x_indices = list(range(len(x_labels)))

    plt.figure()
    plt.plot(x, train, marker="o", label="Train Accuracy", color="#4f7bb8")
    plt.plot(x, test, marker="o", label="Test Accuracy", color="#4fb87e")

    plt.title(f"{model}")
    plt.xlabel(f"{str(param)}")
    plt.ylabel("Accuracy")
    plt.xticks(x_indices, x_labels)
    plt.legend()
    plt.grid()

    path = f"{model}/{param}.png"
    plt.savefig(path)

models = ["Logistic Regression", "KNN", "Random Forest", "MLP", "Decision Tree", "SVM"]
for model in models:
    os.makedirs(f"{model}", exist_ok=True)
    if model == "Logistic Regression":
        for param in ['C', 'solver', 'penalty']:
            create_graph(results_df, model, param)
    if model == "KNN":
        for param in ['n_neighbors', 'weights', 'metric']:
            create_graph(results_df, model, param)
    if model == "Random Forest":
        for param in ['n_estimators', 'max_depth', 'min_samples_leaf']:
            create_graph(results_df, model, param)
    if model == "MLP":
        for param in ['hidden_layer_sizes', 'activation', 'batch_size', 'learning_rate_init']:
            create_graph(results_df, model, param)
    if model == "Decision Tree":
        for param in ['max_depth', 'min_samples_leaf', 'min_samples_split', 'criterion']:
            create_graph(results_df, model, param)
    if model == "SVM":
        for param in ['C', 'kernel', 'gamma']:
            create_graph(results_df, model, param)
