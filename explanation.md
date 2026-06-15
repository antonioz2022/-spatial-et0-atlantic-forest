# Como o projeto foi feito — da coleta de dados ao fim

Caminho completo do projeto, com as fórmulas e parâmetros usados no código:
**coleta → limpeza → cálculo da ET0 → interpolação → validação → análises →
saídas → infraestrutura**.

---

## Visão geral do pipeline

```
CSVs do INMET ─▶ leitura/limpeza ─▶ Tmax/Tmin diárias ─▶ ET0 (Hargreaves-Samani)
      │                                                          │
      └────────── coords (lon, lat, alt) ◀───────────────────────┘
                              │
                              ▼
          CDD (1×, série toda)        tabela ET0 (datas × estações)
                              │                 │
                              └──────┬──────────┘
                                     ▼
              LOO-CV dia a dia, 9 métodos  (IDW1–5, ADW, RF | OK, RFRK)
                                     │
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                          ▼
     métricas por método      Wilcoxon pareado        importância de variáveis
     (d, RMSE, BIAS, MAE)     (RFRK vs cada um)        (lon, lat, alt, doy)
            │                        │                          │
            └────────────┬───────────┴──────────────────────────┘
                         ▼
        saídas: resultados.md, tabela3.csv, pred_long.parquet, 4 figuras
                         ▼
        MLflow (params, métricas, artefatos) + Docker (reprodutível)
```

Cada caixa corresponde a uma função real dos módulos `core.py` (motor),
`ingest.py` (leitura do INMET), `figures.py` (figuras) e `pipeline.py`
(orquestração).

---

## Etapa 1 — Coleta de dados (INMET)

Os dados vêm das **estações meteorológicas automáticas do INMET**, baixadas do
portal de dados históricos (`portal.inmet.gov.br/dadoshistoricos`). O fluxo é:

1. Baixar o ZIP de **1 ano** de estações automáticas e descompactar em
   `./data/<ano>/` — vários arquivos `.CSV`, **um por estação**.
2. Cada CSV traz, no cabeçalho, os metadados da estação:
   `REGIÃO; UF; ESTAÇÃO; CÓDIGO; LATITUDE; LONGITUDE; ALTITUDE`; e, no corpo, as
   medições **horárias** (temperatura do ar, etc.).
3. Escolher uma **região compacta** com um *bounding box* (`--bbox lon_min
   lon_max lat_min lat_max`), de modo a pegar ~15–30 estações próximas.

> O contêiner **não acessa a internet em runtime** — a coleta é feita à mão pelo
> usuário, que coloca os CSVs em `./data`. Isso mantém a execução offline e
> reprodutível.
>
> Para testar o pipeline **sem dados**, há o modo `--demo`, que gera um conjunto
> sintético (20 estações, 1 ano) com um sinal espacial + sazonal realista.

---

## Etapa 2 — Leitura e limpeza dos CSVs (`ingest.py`)

O leitor do INMET é **defensivo de propósito**, porque o layout dos CSVs muda um
pouco a cada ano. A função `parse_inmet_csv` faz, por arquivo:

- Lê em **`latin-1`** (acentuação do INMET) e localiza a **linha de cabeçalho dos
  dados** procurando uma linha que contenha "Data" e vários `;`.
- Extrai do cabeçalho `lat`, `lon`, `alt`, `código`, etc. (normalizando texto:
  remove acentos, caixa alta).
- **Acha as colunas por nome aproximado** (ex.: "TEMPERATURA DO AR - BULBO SECO",
  ou variações), tolerando nomes diferentes entre anos.
- Trata **vírgula decimal** (`"23,4"` → `23.4`), valores faltantes do INMET
  (`-9999`, `NULL`, vazio) → `NaN`, e datas em vários formatos (`YYYY-MM-DD`,
  `YYYY/MM/DD`, `DD/MM/YYYY`).
- **Agrega o horário em diário:** `Tmax` = máximo e `Tmin` = mínimo da temperatura
  ao longo do dia (com *fallback* para as colunas horárias de máx/mín do INMET se
  o bulbo seco faltar). Mantém só dias com **amplitude válida** (`Tmax > Tmin`).

A função `load_inmet_dir` percorre a pasta inteira, **filtra pelo `bbox`**,
**descarta estações com poucos dias válidos** (`min_days=200`) e devolve duas
tabelas: uma "longa" (`date, station, tmax, tmin`) e a de metadados
(`station → lon, lat, alt`). Estações com qualquer problema são **puladas com
aviso**, não derrubam a execução.

---

## Etapa 3 — Cálculo da ET0 (Hargreaves-Samani)

Com `Tmax`/`Tmin` diárias, a `build_et0_table` calcula a **evapotranspiração de
referência (ET0)** por **Hargreaves-Samani** (`et0_hargreaves` em `core.py`):

1. **Radiação no topo da atmosfera** `Ra` (MJ m⁻² dia⁻¹), em função da latitude e
   do dia do ano (fórmulas FAO-56: distância Terra-Sol `dr`, declinação solar e
   ângulo horário do pôr do sol).
2. Conversão para mm: `Ra_mm = 0,408 · Ra`.
3. **ET0** = `0,0023 · (Tmean + 17,8) · √(Tmax − Tmin) · Ra_mm`,
   com `Tmean = (Tmax + Tmin)/2`.

Por que Hargreaves e não Penman-Monteith? Porque ele precisa **só de
temperatura** — é o *fallback* recomendado pela FAO-56 quando faltam dados de
vento/umidade/radiação, e é o método do banco BRAUM citado no paper. Isso elimina
a dependência de GIS e de variáveis extras. (Ver "Desvios" no fim.)

O resultado é pivotado numa tabela **`et0_df`** (linhas = datas, colunas =
estações) e numa **`coords_df`** (`lon, lat, alt` por estação). Essas duas
tabelas alimentam todo o resto.

---

## Etapa 4 — Preparação para a interpolação

Antes da validação, estima-se **uma única vez** a **CDD** (*correlation decay
distance*) com `estimate_cdd`: para cada par de estações, calcula a correlação
das séries de ET0 e a distância (Haversine) entre elas, e ajusta
`r = exp(−x / CDD)`; o `CDD` é o comprimento de decaimento. Ela é estimada da
**série inteira** (não dá para estimá-la com um dia só) e injetada no método ADW.

---

## Etapa 5 — Os interpoladores (reprodução + contribuição)

Todos têm a **mesma interface** `m.fit(coords, values, cov).predict(coords_new,
cov_new)`, com `coords` em (lon, lat) graus e `cov` = covariáveis (altitude).

**Reprodução (do paper):**

- **IDW1…IDW5** — *Inverse Distance Weighting*: peso `w = d^(−p)`, com `p` de 1 a
  5. Quanto maior `p`, mais "local" a interpolação.
- **ADW** — *Angular Distance Weighting* (New et al., 2000), com 8 vizinhos e
  `m = 4`: peso de distância `w = (exp(−x/CDD))^m` **multiplicado** por um termo
  angular `(1 + a)` que **favorece vizinhos espalhados em direções diferentes**
  (penaliza vizinhos amontoados do mesmo lado).
- **RF** — Random Forest (`RFInterp`) tendo como atributos **lon, lat e
  altitude**, exatamente como no paper.

**Contribuição (não estava no paper):**

- **OK** — Krigagem Ordinária: variograma **exponencial** com *sill* = variância
  amostral e *range* = 1/3 da distância máxima (escolha estável para poucas
  estações); resolve o sistema de krigagem por pseudo-inversa. É o **benchmark
  geoestatístico** que faltava.
- **RFRK** — *Random Forest Regression Kriging* (a contribuição-estrela):
  `fit` treina o RF nas covariáveis, calcula os **resíduos** `y − RF(x)`, e
  **kriga os resíduos** com OK; `predict` devolve `RF(x_novo) + OK_resíduos(x_novo)`.
  Ou seja: **o detalhe de relevo do RF + a autocorrelação espacial da krigagem**.
  Ataca diretamente as duas fraquezas do RF — perder nas métricas e produzir a
  superfície "em blocos".

---

## Etapa 6 — Validação cruzada *leave-one-out* (`loo_by_day`)

O protocolo é o mesmo do paper, **dia a dia**:

```
para cada DIA:
    pega as estações com ET0 válida nesse dia (pula o dia se houver < 5)
    para cada MÉTODO (os 9):
        para cada ESTAÇÃO h (retirada, "held-out"):
            treina o método nas OUTRAS estações daquele dia
            prevê o valor em h
            guarda (observado h, previsto h)
```

É uma **validação espacial**: dentro de cada dia, esconde-se uma estação e tenta-
se prevê-la com as vizinhas. Acumulando sobre todos os dias e estações, obtêm-se
as métricas globais por método. A função devolve:

- **`overall`** — métricas (`d, RMSE, BIAS, MAE`) por método → a **Tabela 3**;
- **`pred_long`** — uma linha por `(data, estação, método)` com `obs` e `pred`,
  que alimenta as figuras, o teste de Wilcoxon e re-análises.

No modo `--fast`, subamostra-se 1 dia a cada 2 e usa-se 80 árvores (em vez de
200) para iterar mais rápido — sem mudar o algoritmo.

---

## Etapa 7 — Métricas

Calculadas em `core.py`, no mesmo sentido do paper:

- **d de Willmott** (índice de concordância, 0–1, maior é melhor):
  `d = 1 − Σ(E−O)² / Σ(|E−Ō| + |O−Ō|)²`.
- **RMSE** — erro quadrático médio (mm/dia).
- **BIAS** = `média(O − E)` — **negativo ⇒ o método superestima** (convenção do
  paper).
- **MAE** — erro absoluto médio (acrescentado por completude).

(`O` = observado, `E` = estimado, `Ō` = média dos observados.)

---

## Etapa 8 — Análises da contribuição

**Teste de significância (Wilcoxon pareado).** `wilcoxon_vs` calcula o **RMSE de
cada dia** por método (`perday_metric`) e aplica o **teste de Wilcoxon pareado**
do RFRK contra cada outro método, dia a dia. Isso responde a pergunta que o paper
deixou no ar — *"a diferença entre métodos é pequena/relevante?"* — com um
**p-valor** (p < 0,05 ⇒ diferença significativa).

**Importância de variáveis do RF.** `rf_feature_importance` treina **um** Random
Forest **empilhando todos os dias**, com as variáveis **lon, lat, altitude e
`doy` (dia do ano)**, e devolve a importância média de cada uma.

> ⚠️ Detalhe importante: este RF da *importância* é **separado** do RF usado no
> LOO. No LOO, o RF vê só um dia por vez (não tem `doy`). Aqui, ao empilhar os
> dias e **incluir `doy`**, medimos **quanto da capacidade preditiva do RF vem do
> ciclo sazonal** versus da posição/relevo — exatamente a promessa que o paper
> fez e nunca mostrou.

---

## Etapa 9 — Saídas (`pipeline.py` + `figures.py`)

A orquestração grava em `outputs/`:

- **`resultados.md`** — Tabela 3 + tabela de Wilcoxon + importância (em Markdown,
  pronto pra colar no relatório);
- **`tabela3.csv`** — a Tabela 3 em CSV;
- **`pred_long.parquet`** — todos os pares observado×previsto (para re-análises);
- **4 figuras:**
  - `fig_metricas.png` — barras de `d` e `RMSE` por método;
  - `fig_scatter.png` — dispersão observado×estimado, **verão vs inverno**, para
    RF e RFRK;
  - `fig_mapas.png` — ET0 interpolada (IDW2 × RF × RFRK) num dia de verão e num de
    inverno, mostrando os **"blocos" do RF** contra a **suavidade do RFRK**
    (a altitude na grade é aproximada por IDW, pois não há DEM);
  - `fig_importancia.png` — importância de variáveis do RF.

---

## Etapa 10 — Infraestrutura (reprodutibilidade)

O que torna o trabalho **reprodutível e rastreável**:

- **CLI** (`python -m et0spatial`): orquestra tudo via flags
  (`--inmet/--bbox/--demo/--fast/--out/--regiao/…`).
- **MLflow** (`tracking.py`): cada execução vira um **run** no experimento
  `et0-spatial-interpolation`, registrando **parâmetros** (bbox, ano, nº de
  estações/dias, método de ET0, covariáveis, nº de árvores, CDD, …),
  **métricas** (`{método}__{métrica}` e `wilcoxon_p__RFRK_vs_*`), **tags** e os
  **artefatos** (a pasta `outputs/` inteira). Store local por arquivo (`mlruns/`),
  com UI navegável (porta 5000).
- **Dashboard Streamlit** (`dashboard.py`): painel que **lê os runs do MLflow** e
  apresenta, de forma amigável, a visão geral comparando execuções e, por
  execução, os parâmetros, a Tabela 3 (com gráficos), o Wilcoxon e as 4 figuras
  (porta 8501).
- **Docker** (`Dockerfile` + `docker-compose.yml`): imagem `python:3.11-slim`
  com dependências **fixadas** (`requirements.txt`), três serviços (`app` roda o
  pipeline; `mlflow` sobe a UI técnica; `dashboard` sobe o painel Streamlit).
  Garante o mesmo ambiente em qualquer máquina.
- **Smoke test** (`tests/test_smoke.py`): roda o pipeline em modo demo e verifica
  que as 7 saídas e o run do MLflow foram criados.

---

## O que os resultados mostram (Espírito Santo, 2023)

1. **A tensão do paper se confirma.** Os métodos ficam muito próximos: o RF
   lidera *d*/RMSE/MAE por margem mínima (d = 0,948; RMSE = 0,564), com IDW1/IDW2
   praticamente empatados (d = 0,946–0,947). O IDW simples é competitivo com o RF.
2. **O RFRK não vence nas métricas, mas tem valor.** Pelo Wilcoxon (RMSE diário),
   o RFRK é significativamente melhor que a krigagem pura (OK) e que os IDW de
   expoente alto, empata com ADW/IDW3 e é significativamente pior que o RF — por
   apenas 0,008 mm/dia (significância ≠ relevância). Em compensação, suaviza os
   "blocos" do RF (ver `fig_mapas.png`) e tem menor viés.
3. **A habilidade do RF é sazonal, não espacial.** A importância de variáveis dá
   `doy ≈ 0,87`, com lon+lat+altitude somando ~0,13: o RF acerta sobretudo por
   capturar o ciclo sazonal, e pouco pela posição/relevo. Isso também explica por
   que o RFRK não melhora as métricas — sobra pouca estrutura espacial nos
   resíduos para a krigagem explorar.

A discussão completa, com tabelas e figuras, está no `RELATORIO.md`.

---

## Desvios assumidos (reprodução de *metodologia*, não dos números)

São escolhas deliberadas — reproduzimos o **método**, em outra região/período:

- Estações **automáticas** do INMET (não as 11 convencionais do paper).
- ET0 por **Hargreaves-Samani** (não Penman-Monteith) — *fallback* FAO-56.
- Covariáveis reduzidas a **lon, lat, altitude** (sem declividade/aspecto/
  distância ao mar) → trabalho futuro.
- **ADW** na forma canônica de New et al. (`WE = w·(1+a)`); o paper escreve
  `(1−a)`.
- **OK** com variograma exponencial fixo (sill = variância, range = 1/3 da
  distância máxima); ajuste de variograma por *fold* = trabalho futuro.

---

## Como reproduzir (resumo)

```bash
docker compose build
docker compose run --rm app --demo --fast                  # teste sintético
docker compose run --rm app --inmet data/2023 --bbox -42.0 -39.5 -21.5 -17.8   # dados reais (ES)
docker compose up dashboard                                 # painel  → :8501
MLFLOW_PORT=5001 docker compose up mlflow                   # UI completa do MLflow
```

Detalhes completos de instalação, dados e flags estão no **`README.md`**.
