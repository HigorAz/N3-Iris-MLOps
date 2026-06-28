import os
import pytest
from fastapi.testclient import TestClient

# so roda os testes da API se ja existe um modelo treinado
pytestmark = pytest.mark.skipif(
    not os.path.exists("models/metadata.json"),
    reason="modelo nao treinado ainda (rode: python -m src.train)",
)

from api.main import app


def test_health():
    # context manager dispara o lifespan, que carrega o modelo
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_setosa():
    corpo = {"sepal_length": 5.1, "sepal_width": 3.5,
             "petal_length": 1.4, "petal_width": 0.2}
    with TestClient(app) as client:
        r = client.post("/predict", json=corpo)
    assert r.status_code == 200
    d = r.json()
    assert d["especie"] in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    assert abs(sum(d["probabilidades"].values()) - 1.0) < 0.05
