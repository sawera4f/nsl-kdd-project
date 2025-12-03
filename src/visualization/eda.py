import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def basic_info(df):
    print("\n  Aperçu du Dataset  ")
    print(df.head())
    print("\n  Types des colonnes  ")
    print(df.dtypes)
    print("\n  Nombre de lignes et colonnes  ")
    print(df.shape)

def value_counts_info(df):
    print("\n  Répartition protocol_type  ")
    print(df["protocol_type"].value_counts())

    print("\n  Répartition service  ")
    print(df["service"].value_counts())

    print("\n  Répartition flag  ")
    print(df["flag"].value_counts())

def plot_pie_target(df):
    counts = df["Target"].value_counts()
    labels = ["Normal", "Attack"]

    plt.figure(figsize=(7,7))
    plt.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e"],
        startangle=90
    )
    plt.title("The percentage of Normal and Attack Requests in dataset")

    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/pie_distribution.png", dpi=300)
    plt.close()
    print(" Graphique pie sauvegardé : reports/figures/pie_distribution.png")
