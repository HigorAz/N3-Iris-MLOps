"""Constroi os 3 modelos a partir do models.yaml."""
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def carrega_lista(caminho="config/models.yaml"):
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)["modelos"]


def _keras_mlp(params, n_features, n_classes):
    # import dentro da funcao: so carrega o tensorflow se realmente for usar
    import tensorflow as tf

    tf.random.set_seed(42)
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Input(shape=(n_features,)))
    for n in params["camadas"]:
        modelo.add(tf.keras.layers.Dense(n, activation="relu"))
    modelo.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(params.get("learning_rate", 0.01)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


class WrapperKeras:
    """Deixa o modelo Keras com a mesma interface dos modelos sklearn."""

    def __init__(self, modelo, params):
        self.modelo = modelo
        self.params = params

    def fit(self, X, y):
        self.modelo.fit(
            X, y,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            verbose=0,
        )
        return self

    def predict(self, X):
        return np.argmax(self.modelo.predict(X, verbose=0), axis=1)

    def predict_proba(self, X):
        return self.modelo.predict(X, verbose=0)


def constroi(tipo, params, n_features, n_classes):
    if tipo == "sklearn_rf":
        return RandomForestClassifier(random_state=42, **params)
    if tipo == "sklearn_svm":
        return SVC(probability=True, random_state=42, **params)
    if tipo == "keras_mlp":
        return WrapperKeras(_keras_mlp(params, n_features, n_classes), params)
    raise ValueError(f"Tipo de modelo desconhecido: {tipo}")
