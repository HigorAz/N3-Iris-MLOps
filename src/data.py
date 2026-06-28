"""Importa a base e prepara os dados para treino."""
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ordem das features esperada pelo modelo (tambem usada pela API)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def carrega_config(caminho="config/config.yaml"):
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carrega_dados(cfg):
    """Le o CSV e devolve X e y."""
    df = pd.read_csv(cfg["dados"]["caminho"])
    y = df[cfg["dados"]["alvo"]]
    X = df[FEATURES]
    return X, y


def prepara(cfg):
    """Split estratificado + padronizacao + encoding do alvo.

    Retorna os conjuntos ja transformados e os transformadores
    (precisamos deles depois para servir o modelo na API).
    """
    X, y = carrega_dados(cfg)
    X = X.to_numpy()  # evita warning de feature names ao prever via array

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=cfg["dados"]["test_size"],
        random_state=cfg["dados"]["random_state"],
        stratify=y_enc,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, y_tr, y_te, scaler, encoder
