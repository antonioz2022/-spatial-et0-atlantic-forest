# ET0 spatial — reprodução + contribuição (Baratto et al., 2022)

> ⚠️ **Reprodução de _metodologia_, não dos números exatos.** Este projeto
> reproduz a **metodologia** de Baratto et al. (2022) em **outra região e período**
> (Espírito Santo, 2023, estações automáticas do INMET) — é uma **adaptação
> metodológica deliberada**, não uma replicação dos valores do paper. O que se
> reproduz é o **padrão** (os métodos empatam e o IDW simples compete com o RF),
> e não os números. A lista completa de desvios, com a justificativa de cada um,
> está em [Desvios em relação ao paper](#desvios-em-relação-ao-paper).

Reproduz a comparação de interpoladores de **evapotranspiração de referência
(ET0) diária** do paper de Baratto et al. (2022) — IDW (potências 1–5), ADW e
Random Forest — por validação cruzada *leave-one-out* (LOO), nas métricas **d de
Willmott, RMSE, BIAS e MAE**; e adiciona uma **contribuição** que o paper não
trouxe:

1. **OK** — krigagem ordinária (benchmark geoestatístico ausente no paper);
2. **RFRK** — *Random Forest Regression Kriging* (RF nas covariáveis + krigagem
   dos resíduos), que ataca a fraqueza do RF (perde nas métricas e gera mapas
   "em blocos");
3. **Teste de significância** (Wilcoxon pareado dia a dia);
4. **Importância de variáveis do RF** (o paper afirma a vantagem e nunca a mostra).

> Paper: Baratto et al. (2022), *"Random forest for spatialization of daily
> evapotranspiration (ET0) in watersheds in the Atlantic Forest"*,
> *Environ Monit Assess* 194:449.

A execução é empacotada em **CLI + MLflow + dashboard Streamlit + Docker**.

---

## Estrutura

```
src/et0spatial/
  core.py       # ET0 (Hargreaves), interpoladores IDW/ADW/RF/OK/RFRK,
                #   LOO-CV, métricas (d, RMSE, BIAS, MAE), Wilcoxon, importância
  ingest.py     # leitura dos CSVs do INMET
  figures.py    # as 4 figuras do relatório
  pipeline.py   # orquestração ponta a ponta + geração das saídas + dados sintéticos (--demo)
  tracking.py   # rastreamento MLflow (params, metrics, artifacts, tags)
  dashboard.py  # dashboard Streamlit (lê os runs do MLflow e os apresenta)
  cli.py        # interface de linha de comando (python -m et0spatial)
requirements.txt  pyproject.toml  Dockerfile  docker-compose.yml  Makefile
tests/test_smoke.py
figuras/         # figuras do relatório (versionadas)
data/            # CSVs do INMET (não versionado)
outputs/         # tabelas + figuras geradas (runtime)
mlruns/          # store local do MLflow (runtime)
```

**Saídas** (em `outputs/`, e anexadas como *artifacts* no MLflow):
`resultados.md`, `tabela3.csv`, `pred_long.parquet`, `fig_metricas.png`,
`fig_scatter.png`, `fig_mapas.png`, `fig_importancia.png`.

---

## Como rodar (Docker)

Pré-requisitos: Docker + Docker Compose.

```bash
# 1) construir a imagem
docker compose build

# 2) rodar o pipeline em DADOS SINTÉTICOS (não precisa baixar nada)
docker compose run --rm app --demo --fast      # ~1–3 min
#   (sem --fast: usa todos os dias e 200 árvores; mais lento)

# 3) abrir o dashboard (Streamlit) em http://localhost:8501
docker compose up dashboard                      # Ctrl+C para sair

# 4) abrir a UI do MLflow em http://localhost:5000
docker compose up mlflow                         # Ctrl+C para sair
```

> **macOS:** o *AirPlay Receiver* (ControlCenter) costuma ocupar a porta 5000.
> Se a UI não subir (`address already in use`), desligue-o em *Ajustes do Sistema
> → Geral → AirDrop e Handoff → Receptor AirPlay*, **ou** use outra porta:
> `MLFLOW_PORT=5001 docker compose up mlflow` → http://localhost:5001 .

Tudo aparece em `./outputs/` (tabelas + 4 figuras) e em `./mlruns/` (um run do
MLflow com params, métricas e artefatos).

### Com dados reais do INMET

```bash
docker compose run --rm app --inmet data/2023 --bbox -42.0 -39.5 -21.5 -17.8 --regiao espirito-santo
#   --bbox = LON_MIN LON_MAX LAT_MIN LAT_MAX   (ajuste para a sua região)
#   --regiao  rótulo do run no MLflow (opcional)
```

> O contêiner **não acessa a rede** em runtime: você baixa e descompacta os CSVs
> em `./data` (ver abaixo); os volumes `./data`, `./outputs`, `./mlruns` são
> montados no contêiner.

### Atalhos (Makefile)

```bash
make build                                       # docker compose build
make demo                                        # --demo --fast
make run BBOX="-42 -39.5 -21.5 -17.8" DATA=data/2023   # dados reais
make dashboard                                   # dashboard Streamlit (porta 8501)
make ui                                          # UI do MLflow (porta 5000)
make clean                                       # apaga outputs/ e mlruns/ (não toca em data/)
```

---

## Dados do INMET (estações automáticas)

1. Acesse **https://portal.inmet.gov.br/dadoshistoricos** e baixe o ZIP de
   **1 ano** de estações automáticas.
2. Descompacte numa pasta sob `./data`, por exemplo `./data/2023/` (vários
   `.CSV`, um por estação).
3. **Na primeira vez, abra um `.CSV`** e confira os nomes das colunas: o layout
   do INMET muda um pouco por ano. O leitor (`ingest.py`) é defensivo (acha
   colunas por nome aproximado, trata vírgula decimal, `-9999`, `latin-1`), mas
   se a temperatura/data tiver nome muito diferente ele avisa qual estação pulou.
4. Escolha uma **região compacta** via `--bbox`. O cabeçalho de cada CSV traz
   LATITUDE/LONGITUDE/ALTITUDE — as covariáveis usadas são **lon, lat, altitude**
   (sem GIS).

> O recorte usado no relatório é o **Espírito Santo** (`--bbox -42.0 -39.5 -21.5
> -17.8`), que tem 12 estações automáticas — próximo das 11 estações do paper.

---

## Uso local (sem Docker)

Requer Python 3.11+.

```bash
make install            # = pip install -r requirements.txt && pip install --no-deps -e .
python -m et0spatial --demo --fast
python -m et0spatial --inmet data/2023 --bbox -42.0 -39.5 -21.5 -17.8

# dashboard Streamlit (lê o ./mlruns local) → http://localhost:8501
streamlit run src/et0spatial/dashboard.py

# UI do MLflow (lê o ./mlruns local)
mlflow ui --backend-store-uri ./mlruns

# smoke test (instala deps de teste e roda pytest)
make test
```

### Flags da CLI

| flag | descrição |
|---|---|
| `--demo` | dados sintéticos (não precisa de dados) |
| `--inmet PASTA` | pasta com os CSVs do INMET |
| `--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX` | recorte da região (graus) |
| `--fast` | iteração rápida: 1 dia a cada 2 e 80 árvores (em vez de 200) |
| `--out PASTA` | pasta de saída (default `outputs`) |
| `--regiao RÓTULO` | nome da região p/ o run do MLflow |
| `--max-days N` | limita o nº de dias do LOO |
| `--no-mlflow` | roda sem MLflow |

---

## MLflow — o que é registrado

Experimento **`et0-spatial-interpolation`**, um run por execução
(`run_name = {regiao}-{ano}`), store local por arquivo
(`MLFLOW_TRACKING_URI=file:///app/mlruns` no Docker; `./mlruns` localmente).

- **Params:** `bbox`, `ano`, `n_estacoes`, `n_dias_usados`,
  `et0_method=hargreaves_samani`, `covariaveis=lon,lat,alt`, `rf_trees`,
  `cdd_km`, `metodos`, `fast`, `regiao`, `dataset`.
- **Metrics:** `{metodo}__{metrica}` para cada um dos 9 métodos e
  `metrica ∈ {d, RMSE, BIAS, MAE}` (ex.: `RFRK__d`, `IDW5__RMSE`); e os
  p-valores `wilcoxon_p__RFRK_vs_{metodo}`.
- **Tags:** `foundation`, `dataset` (`INMET-automaticas`/`sintetico`) e
  `git_commit` (se houver repositório git).
- **Artifacts:** a pasta `outputs/` inteira (4 figuras + `resultados.md` +
  `tabela3.csv` + `pred_long.parquet`).

---

## Dashboard (Streamlit)

`docker compose up dashboard` → **http://localhost:8501** (ou
`streamlit run src/et0spatial/dashboard.py` localmente). O painel lê os runs do
MLflow (`mlruns/`) e mostra:

- uma **visão geral** comparando todas as execuções (dataset, ano, nº de
  estações/dias e métricas-chave);
- por execução escolhida: os **parâmetros**, a **Tabela 3** (d/RMSE/BIAS/MAE por
  método) com gráficos de barras, a **significância** (Wilcoxon, RFRK vs. cada
  método) e as **4 figuras**.

Se a porta 8501 estiver ocupada: `DASHBOARD_PORT=8502 docker compose up dashboard`.

---

## Desvios em relação ao paper

A reprodução é de **metodologia**, em outra região/período — não dos números exatos:

- Estações **automáticas** do INMET (não as 11 convencionais do paper) — são as de
  download livre e já trazem Tmax/Tmin; o recorte do ES rende 12 estações (≈ as 11
  do paper), na mesma Mata Atlântica. *Por quê:* viabiliza a reprodução com dados
  abertos, sem perder a escala/região do estudo original.
- ET0 por **Hargreaves-Samani** (não Penman-Monteith): é o fallback FAO-56 para
  escassez de dados e o método do banco BRAUM citado no paper. (Para PM, há o
  pacote `refet`.)
- Covariáveis reduzidas a **lon, lat, altitude** (sem declividade/aspecto/
  distância ao mar) — todas saem do **cabeçalho do próprio CSV**. *Por quê:* elimina
  a dependência de GIS/DEM e mantém a execução 100% reprodutível e offline.
- **ADW**: forma canônica de New et al. (WE = w·(1+a)); o paper escreve (1−a). *Por
  quê:* é só o **sinal do termo angular** — a forma canônica é a que de fato
  penaliza vizinhos agrupados; trocar o sinal é trivial (1 linha) e muda pouco. Não
  é erro: é a equação original de New et al. (2000).
- **OK**: variograma exponencial com sill = variância amostral e range = 1/3 da
  distância máxima (estável para poucas estações).
