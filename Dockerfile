FROM python:3.11-slim

WORKDIR /app

# Dependências fixadas primeiro (camada cacheada — só muda se requirements.txt mudar).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código do pacote + instalação editável sem re-resolver dependências.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Store local do MLflow (sobrescrito pelo docker-compose).
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Entrada = CLI do pacote. `docker compose run --rm app --demo --fast` etc.
ENTRYPOINT ["python", "-m", "et0spatial"]
CMD ["--help"]
