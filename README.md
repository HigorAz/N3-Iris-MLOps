# N3 — Produtização de ML (MLOps) com a base Iris

Projeto da disciplina de Ciência de Dados (Engenharia de Software). Implementa um ciclo
completo de MLOps: treino de modelos de classificação, seleção automática do melhor,
serviço de predição (API + página web), rastreamento de experimentos com MLflow,
empacotamento em Docker e ciclo de retreino automatizado via GitHub Actions.

#### Alunos: Higor Azevedo, Nathan Cielusinski e Nicolas Gustavo Conte.

Base de dados: **Iris** (fonte: https://www.kaggle.com/datasets/arshid/iris-flower-dataset).

## O que o projeto faz

1. **Importa a base** e faz a **análise multivariada** (gráficos em `reports/`).
2. **Treina 3 modelos**: Random Forest, SVM (scikit-learn) e uma rede neural (TensorFlow/Keras).
3. **Avalia** todos por accuracy, precision, recall e F1.
4. **Escolhe o melhor automaticamente** pela métrica definida em `config/config.yaml`.
5. **Salva** o modelo vencedor, os pré-processadores e os metadados em `models/`.
6. **Serve** o modelo numa **API FastAPI** com uma **página web** para enviar novos dados.
7. **Rastreia** parâmetros e métricas no **MLflow**.
8. **Retreina automaticamente** via **GitHub Actions** (e endpoint `/retrain`).

## Estrutura

```
config/      configuracoes em YAML (pipeline e definicao dos 3 modelos)
data/        iris.csv
src/         data, eda, models, evaluate, train
api/         FastAPI (main, schemas) + static/index.html
models/      modelo salvo + metadata.json (gerado)
reports/     graficos da analise multivariada (gerado)
tests/       testes pytest
.github/     workflow de CI/CD (retreino automatico)
```

## Como rodar com Docker (recomendado)

Pré-requisito: Docker + Docker Compose.

```bash
# 1. treina e gera o modelo (loga no mlflow)
docker compose run --rm train

# 2. sobe a API e o MLflow
docker compose up api mlflow
```

Depois acesse:
- Aplicação web / API: http://localhost:8000
- Documentação interativa da API: http://localhost:8000/docs
- MLflow (experimentos): http://localhost:5000

## Como rodar localmente (sem Docker)

```bash
pip install -r requirements.txt
python -m src.train          # treina, escolhe e salva o melhor modelo
uvicorn api.main:app --reload  # sobe a API em http://localhost:8000
pytest -q                    # testes
```

(Atalhos equivalentes no `Makefile`: `make train`, `make serve`, `make test`, `make up`.)

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Página web com formulário |
| GET | `/health` | Status do serviço |
| GET | `/model-info` | Modelo ativo, versão e métricas |
| POST | `/predict` | Recebe as 4 medidas e retorna a espécie + probabilidades |
| POST | `/retrain` | Dispara o retreino e recarrega o modelo |

Exemplo de predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Configuração

Todo o comportamento é controlado por arquivos YAML, sem mexer no código:
- `config/config.yaml` — base, divisão treino/teste, métrica de seleção, saídas.
- `config/models.yaml` — os 3 modelos e seus hiperparâmetros.

## Ciclo automatizado (MLOps)

O workflow `.github/workflows/mlops.yml` roda a cada push, semanalmente (cron) ou sob demanda:
testa, retreina, valida o novo modelo e, se houver mudança, **commita o modelo atualizado**
de volta ao repositório — simulando a atualização contínua do modelo em produção.
