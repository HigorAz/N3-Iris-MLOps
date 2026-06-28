"""Servico FastAPI: serve o modelo treinado e a pagina web."""
import os
import json
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import EntradaFlor, Predicao
from src.data import FEATURES

PASTA = "models"
STATIC = os.path.join(os.path.dirname(__file__), "static")

# guarda o modelo carregado em memoria
_estado = {"modelo": None, "scaler": None, "encoder": None, "meta": None}


def _carrega_keras(caminho):
    import tensorflow as tf
    return tf.keras.models.load_model(caminho)


def carrega_modelo():
    """Le o modelo salvo e os pre-processadores. Chamado no startup e apos retreino."""
    meta_path = os.path.join(PASTA, "metadata.json")
    if not os.path.exists(meta_path):
        return False

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    arq = os.path.join(PASTA, meta["arquivo"])
    if meta["tipo"] == "keras_mlp":
        modelo = _carrega_keras(arq)
        _estado["keras"] = True
    else:
        modelo = joblib.load(arq)
        _estado["keras"] = False

    _estado["modelo"] = modelo
    _estado["scaler"] = joblib.load(os.path.join(PASTA, "scaler.joblib"))
    _estado["encoder"] = joblib.load(os.path.join(PASTA, "encoder.joblib"))
    _estado["meta"] = meta
    return True


@asynccontextmanager
async def lifespan(app):
    carrega_modelo()
    yield


app = FastAPI(title="Iris MLOps", description="Classificador de flores Iris",
              lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": _estado["modelo"] is not None}


@app.get("/model-info")
def model_info():
    if _estado["meta"] is None:
        raise HTTPException(503, "Nenhum modelo treinado. Rode o treino primeiro.")
    m = _estado["meta"]
    return {
        "modelo": m["modelo"],
        "tipo": m["tipo"],
        "versao": m["versao"],
        "metrica_selecao": m["metrica_selecao"],
        "metricas": m["metricas"],
        "classes": m["classes"],
    }


@app.post("/predict", response_model=Predicao)
def predict(entrada: EntradaFlor):
    if _estado["modelo"] is None:
        raise HTTPException(503, "Nenhum modelo treinado. Rode o treino primeiro.")

    enc, scaler, meta = _estado["encoder"], _estado["scaler"], _estado["meta"]
    x = np.array([[getattr(entrada, c) for c in FEATURES]])
    x = scaler.transform(x)

    if _estado["keras"]:
        proba = _estado["modelo"].predict(x, verbose=0)[0]
    else:
        proba = _estado["modelo"].predict_proba(x)[0]

    idx = int(np.argmax(proba))
    especie = enc.inverse_transform([idx])[0]
    probs = {enc.inverse_transform([i])[0]: round(float(p), 4) for i, p in enumerate(proba)}

    return Predicao(especie=especie, probabilidades=probs,
                    modelo=meta["modelo"], versao=meta["versao"])


@app.post("/retrain")
def retrain():
    """Dispara o retreino e recarrega o modelo (util para demo ao vivo)."""
    from src.train import treina
    meta = treina()
    carrega_modelo()
    return {"status": "retreinado", "modelo": meta["modelo"],
            "versao": meta["versao"], "metricas": meta["metricas"]}


# pagina web (formulario). Deve ficar por ultimo para nao capturar as rotas acima.
@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
