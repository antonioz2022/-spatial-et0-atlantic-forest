# Atalhos do projeto ET0. Uso:
#   make build
#   make demo
#   make run BBOX="-41 -34 -11 -4" DATA=data/2023
#   make ui
#
# Variáveis sobreponíveis na linha de comando:
BBOX ?= -41 -34 -11 -4
DATA ?= data/2023
REGIAO ?=

.PHONY: build demo run ui dashboard test clean install help

help:
	@echo "Alvos: build | demo | run BBOX=\"...\" DATA=... | ui | dashboard | test | clean | install"

## ---- Docker (caminho principal / critérios de aceitação) ----
build:            ## constrói a imagem
	docker compose build

demo:             ## roda o pipeline em dados sintéticos (rápido)
	docker compose run --rm app --demo --fast

run:              ## roda com dados reais do INMET (passe BBOX e DATA)
	docker compose run --rm app --inmet $(DATA) --bbox $(BBOX) $(if $(REGIAO),--regiao $(REGIAO),)

ui:               ## sobe a UI do MLflow em http://localhost:5000
	docker compose up mlflow

dashboard:        ## sobe o dashboard Streamlit em http://localhost:8501
	docker compose up dashboard

clean:            ## remove saídas e runs locais (NÃO apaga ./data)
	rm -rf outputs/* mlruns/*

## ---- Local (sem Docker) ----
install:          ## instala deps fixadas + o pacote (editável)
	pip install -r requirements.txt
	pip install --no-deps -e .

test:             ## instala deps de teste e roda o smoke test (pytest)
	pip install -r requirements-dev.txt
	pip install --no-deps -e .
	pytest
