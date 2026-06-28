.PHONY: install train serve test up down

install:
	pip install -r requirements.txt

train:
	python -m src.train

serve:
	uvicorn api.main:app --reload

test:
	pytest -q

# sobe tudo no docker: treina e depois api + mlflow
up:
	docker compose run --rm train
	docker compose up api mlflow

down:
	docker compose down
