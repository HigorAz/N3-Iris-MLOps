"""Pipeline de treino: importa base -> EDA -> treina 3 modelos ->
avalia -> escolhe o melhor pela metrica -> salva -> registra no MLflow."""
import os
import json
import warnings

# silencia aviso de deprecacao do SVC(probability=True) no sklearn 1.9+
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")  # silencia aviso de git no mlflow

import joblib
import mlflow

from src import data, models, evaluate, eda


def _salva_modelo(obj, tipo, pasta):
    """Salva sklearn (joblib) ou keras (formato nativo) e devolve o caminho."""
    if tipo == "keras_mlp":
        caminho = os.path.join(pasta, "modelo.keras")
        obj.modelo.save(caminho)
    else:
        caminho = os.path.join(pasta, "modelo.joblib")
        joblib.dump(obj, caminho)
    return caminho


def treina():
    cfg = data.carrega_config()
    metrica = cfg["selecao"]["metrica"]
    pasta_modelos = cfg["saidas"]["modelos"]
    os.makedirs(pasta_modelos, exist_ok=True)

    # MLflow: usa env var (docker) ou o que estiver no config
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    # 1. analise multivariada
    eda.gera_relatorios(cfg)

    # 2. dados prontos
    X_tr, X_te, y_tr, y_te, scaler, encoder = data.prepara(cfg)
    n_features = X_tr.shape[1]
    n_classes = len(encoder.classes_)

    # 3-5. treina cada modelo, avalia e guarda o melhor
    melhor = None
    resultados = []

    for spec in models.carrega_lista():
        nome, tipo, params = spec["nome"], spec["tipo"], spec["params"]
        print(f"\n>> Treinando {nome} ({tipo})")

        with mlflow.start_run(run_name=nome):
            modelo = models.constroi(tipo, params, n_features, n_classes)
            modelo.fit(X_tr, y_tr)

            y_pred = modelo.predict(X_te)
            m = evaluate.metricas(y_te, y_pred)
            print("   ", {k: round(v, 4) for k, v in m.items()})

            mlflow.log_params(params)
            mlflow.log_metrics(m)

            resultados.append({"nome": nome, "tipo": tipo, "metricas": m})
            if melhor is None or m[metrica] > melhor["metricas"][metrica]:
                melhor = {"nome": nome, "tipo": tipo, "metricas": m, "modelo": modelo}

    # 6. salva o melhor + pre-processadores + metadados
    print(f"\n== Melhor modelo: {melhor['nome']} "
          f"({metrica}={melhor['metricas'][metrica]:.4f})")

    caminho_modelo = _salva_modelo(melhor["modelo"], melhor["tipo"], pasta_modelos)
    joblib.dump(scaler, os.path.join(pasta_modelos, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(pasta_modelos, "encoder.joblib"))

    versao = _proxima_versao(pasta_modelos)
    meta = {
        "modelo": melhor["nome"],
        "tipo": melhor["tipo"],
        "arquivo": os.path.basename(caminho_modelo),
        "metrica_selecao": metrica,
        "metricas": melhor["metricas"],
        "matriz_confusao": evaluate.matriz_confusao(y_te, melhor["modelo"].predict(X_te)),
        "features": data.FEATURES,
        "classes": encoder.classes_.tolist(),
        "versao": versao,
        "comparacao": resultados,
    }
    with open(os.path.join(pasta_modelos, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Modelo v{versao} salvo em {pasta_modelos}/")
    return meta


def _proxima_versao(pasta):
    """Incrementa a versao com base no metadata anterior (controle simples)."""
    caminho = os.path.join(pasta, "metadata.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            return json.load(f).get("versao", 0) + 1
    return 1


if __name__ == "__main__":
    treina()
