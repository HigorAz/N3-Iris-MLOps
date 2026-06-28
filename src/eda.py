"""Analise multivariada da base Iris. Gera graficos em reports/."""
import os
import matplotlib
matplotlib.use("Agg")  # backend sem janela (roda no docker/CI)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

from src.data import FEATURES


def gera_relatorios(cfg):
    df = pd.read_csv(cfg["dados"]["caminho"])
    alvo = cfg["dados"]["alvo"]
    out = cfg["saidas"]["relatorios"]
    os.makedirs(out, exist_ok=True)

    # 1. Pairplot - relacao entre todas as variaveis por especie
    g = sns.pairplot(df, hue=alvo, vars=FEATURES)
    g.figure.suptitle("Relacao entre variaveis por especie", y=1.02)
    g.savefig(os.path.join(out, "pairplot.png"), bbox_inches="tight")
    plt.close("all")

    # 2. Correlacao
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Matriz de correlacao")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "correlacao.png"))
    plt.close()

    # 3. Boxplots por especie
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x=alvo, y=col, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "boxplots.png"))
    plt.close()

    # 4. PCA em 2D - visao geral da separacao das classes
    comp = PCA(n_components=2).fit_transform(df[FEATURES])
    df_pca = pd.DataFrame(comp, columns=["PC1", "PC2"])
    df_pca[alvo] = df[alvo]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue=alvo)
    plt.title("PCA (2 componentes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pca.png"))
    plt.close()

    print(f"Graficos da analise multivariada salvos em {out}/")
