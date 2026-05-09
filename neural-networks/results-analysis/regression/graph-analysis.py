import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)

results_df = pd.read_csv("../../test-results/regression/param_tests_results.csv")

def create_graph(df, param):
    data = df[
        (df["Tested Param"] == param)
        ].copy()

    data["Tested Param"] = data["Tested Param"].fillna("None")

    data["Value_Numeric"] = pd.to_numeric(data["Value"], errors='coerce')

    if data["Value_Numeric"].notna().all():
        data = data.sort_values("Value_Numeric")
    else:
        data = data.sort_values("Value")

    x = data["Value"].astype(str)
    train = data["Train MAE (Avg)"]
    test = data["Test MAE (Avg)"]

    plt.figure()
    plt.plot(x, train, marker="o", label="Train MAE", color="#d14d73")
    plt.plot(x, test, marker="o", label="Test MAE", color="#fcd560")

    plt.title(f"{param}")
    plt.xlabel(f"{str(param)}")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()

    path = f"{param}.png"
    plt.savefig(path)

params = ["architecture", "activation_function", "learning_rate", "batch_size", "seperator", "iteration"]
for param in params:
    if param == "architecture":
        create_graph(results_df, param)
    if param == "activation_function":
        create_graph(results_df, param)
    if param == "learning_rate":
        create_graph(results_df, param)
    if param == "batch_size":
        create_graph(results_df, param)
    if param == "seperator":
        create_graph(results_df, param)
    if param == "iteration":
        create_graph(results_df, param)
