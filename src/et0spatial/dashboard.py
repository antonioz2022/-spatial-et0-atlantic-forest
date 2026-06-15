"""
dashboard.py
============
Painel Streamlit que apresenta os resultados lendo o store de experimentos do
MLflow.

Para cada execução registrada mostra:
  - parâmetros do run;
  - Tabela 3 (métricas d/RMSE/BIAS/MAE por método) + gráficos comparativos;
  - significância (Wilcoxon pareado, RFRK vs. cada método);
  - as 4 figuras (lidas dos artifacts do run).
E uma visão geral comparando todas as execuções.

Rodar:
    streamlit run src/et0spatial/dashboard.py            # local (lê ./mlruns)
    docker compose up dashboard                           # http://localhost:8501
"""
import os

import pandas as pd
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "et0-spatial-interpolation"
METHODS_ORDER = ["IDW1", "IDW2", "IDW3", "IDW4", "IDW5", "ADW", "RF", "OK", "RFRK"]
METRICS = ["d", "RMSE", "BIAS", "MAE"]
FIGURES = [
    ("fig_metricas.png", "Métricas por método (d e RMSE)"),
    ("fig_scatter.png", "Observado × estimado — verão vs. inverno"),
    ("fig_mapas.png", "Mapas interpolados — blocos do RF × suavidade do RFRK"),
    ("fig_importancia.png", "Importância de variáveis do RF"),
]

st.set_page_config(page_title="ET0 spatial — painel", layout="wide")

# O MLflow lê MLFLOW_TRACKING_URI do ambiente; no Docker é file:///app/mlruns,
# localmente cai no ./mlruns relativo ao diretório de execução.
_uri = os.environ.get("MLFLOW_TRACKING_URI")
if _uri:
    mlflow.set_tracking_uri(_uri)
client = MlflowClient()

st.title("ET0 spatial — painel de experimentos")
st.caption("Reprodução + contribuição (Baratto et al., 2022) · IDW/ADW/RF · OK/RFRK · "
           "dados lidos do MLflow")


def _runs():
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return None, []
    runs = client.search_runs(
        [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=500)
    return exp, runs


exp, runs = _runs()

if exp is None or not runs:
    st.warning(
        f"Nenhuma execução encontrada no experimento **{EXPERIMENT}**.\n\n"
        "Rode o pipeline primeiro, por exemplo:\n\n"
        "```\ndocker compose run --rm app --demo --fast\n```")
    st.stop()


def run_name(r):
    return r.data.tags.get("mlflow.runName", r.info.run_id[:8])


def run_label(r):
    ts = pd.to_datetime(r.info.start_time, unit="ms")
    return f"{run_name(r)} · {ts:%Y-%m-%d %H:%M} · {r.info.run_id[:8]}"


def metrics_table(metrics):
    idx = [m for m in METHODS_ORDER if any(f"{m}__{k}" in metrics for k in METRICS)]
    tab = pd.DataFrame(index=idx, columns=METRICS, dtype=float)
    for m in idx:
        for k in METRICS:
            tab.loc[m, k] = metrics.get(f"{m}__{k}")
    return tab


# ===================================================================
# Visão geral — comparar todas as execuções
# ===================================================================
st.subheader(f"Execuções registradas ({len(runs)})")
rows = []
for r in runs:
    m, p = r.data.metrics, r.data.params
    rows.append({
        "run": run_name(r),
        "dataset": p.get("dataset", ""),
        "ano": p.get("ano", ""),
        "n_estações": p.get("n_estacoes", ""),
        "n_dias": p.get("n_dias_usados", ""),
        "RFRK d": m.get("RFRK__d"),
        "RFRK RMSE": m.get("RFRK__RMSE"),
        "RF RMSE": m.get("RF__RMSE"),
        "IDW5 d": m.get("IDW5__d"),
        "iniciado": pd.to_datetime(r.info.start_time, unit="ms"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ===================================================================
# Detalhe de uma execução
# ===================================================================
st.divider()
labels = {run_label(r): r for r in runs}
sel = st.selectbox("Detalhar execução:", list(labels))
run = labels[sel]
rid = run.info.run_id
metrics, params = run.data.metrics, run.data.params

with st.expander("Parâmetros do run", expanded=False):
    st.dataframe(pd.Series(params, name="valor").rename_axis("parâmetro").to_frame(),
                 use_container_width=True)

# ---- Tabela 3 ----
st.subheader("Tabela 3 — métricas por método (LOO-CV)")
tab = metrics_table(metrics)
if tab.empty:
    st.info("Este run não tem métricas por método registradas.")
else:
    st.dataframe(tab.style.format("{:.3f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("d de Willmott — maior é melhor")
        st.bar_chart(tab["d"], color="#3b7dd8")
    with c2:
        st.caption("RMSE (mm/dia) — menor é melhor")
        st.bar_chart(tab["RMSE"], color="#d8743b")

# ---- Wilcoxon ----
st.subheader("Significância — RFRK vs. cada método (Wilcoxon pareado, RMSE diário)")
wil = []
for k, v in metrics.items():
    if k.startswith("wilcoxon_p__RFRK_vs_"):
        wil.append({"método": k.replace("wilcoxon_p__RFRK_vs_", ""),
                    "p-valor": v, "significativo (p<0,05)": bool(v < 0.05)})
if wil:
    wil_df = pd.DataFrame(wil).sort_values("método").set_index("método")
    st.dataframe(wil_df.style.format({"p-valor": "{:.2e}"}), use_container_width=True)
    st.caption("p < 0,05 ⇒ a diferença de RMSE diário entre o RFRK e o outro método "
               "é estatisticamente significativa.")
else:
    st.info("Sem p-valores de Wilcoxon neste run.")

# ---- Figuras (dos artifacts do run) ----
st.subheader("Figuras")
for fname, title in FIGURES:
    try:
        path = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=fname)
        st.markdown(f"**{title}**")
        st.image(path, use_container_width=True)
    except Exception:
        st.info(f"`{fname}` não disponível neste run.")

st.divider()
st.caption("Para a UI completa do MLflow: `docker compose up mlflow` (porta 5000).")
