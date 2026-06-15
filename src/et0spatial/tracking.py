"""
tracking.py
===========
Envolve a execução do pipeline com MLflow.

Um run por execução, experimento `et0-spatial-interpolation`, store local por
arquivo (`MLFLOW_TRACKING_URI=file:///app/mlruns` no Docker; default `./mlruns`
fora dele). Registra:

  Params : bbox, ano, n_estacoes, n_dias_usados, et0_method, covariaveis,
           rf_trees, cdd_km, metodos, fast, regiao
  Metrics: {metodo}__{metrica}  para metrica in {d, RMSE, BIAS, MAE}
           wilcoxon_p__RFRK_vs_{metodo}
  Tags   : git_commit (se houver repo), foundation, dataset
  Artifacts: a pasta outputs/ inteira (4 figuras + resultados.md + tabela3.csv
             + pred_long.parquet)
"""
from __future__ import annotations
import os
import math
import subprocess

# Silencia o aviso do GitPython quando não há `git` no PATH (imagem slim).
# O MLflow tenta detectar o commit-fonte; sem git, isso é só ruído.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from . import pipeline
from .pipeline import PipelineConfig, COVARIAVEIS_STR, ET0_METHOD

EXPERIMENT = "et0-spatial-interpolation"


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _safe_log_metric(mlflow, name, value):
    """Loga uma métrica tolerando NaN/None e nomes/valores problemáticos."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if math.isnan(v) or math.isinf(v):
        return  # MLflow recente aceita, mas evitamos depender disso
    try:
        mlflow.log_metric(name, v)
    except Exception as e:  # pragma: no cover - não derruba o run por 1 métrica
        print(f"  [mlflow] não consegui logar {name}: {e}")


def run(cfg: PipelineConfig) -> pipeline.Result:
    import mlflow

    # carrega os dados antes para nomear o run como {regiao}-{ano}
    et0_df, coords_df, dataset_kind = pipeline.load_data(cfg)
    ano = pipeline.dominant_year(et0_df)
    regiao = cfg.regiao or ("sintetico" if dataset_kind == "sintetico" else "inmet")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{regiao}-{ano}"):
        mlflow.log_params({
            "bbox": ",".join(str(x) for x in cfg.bbox) if cfg.bbox else "NA",
            "ano": ano,
            "n_estacoes": int(coords_df.shape[0]),
            "et0_method": ET0_METHOD,
            "covariaveis": COVARIAVEIS_STR,
            "rf_trees": cfg.rf_trees,
            "fast": cfg.fast,
            "regiao": regiao,
            "dataset": dataset_kind,
        })

        result = pipeline.compute(cfg, et0_df, coords_df, dataset_kind)

        # params conhecidos só depois do compute
        mlflow.log_params({
            "n_dias_usados": result.n_dias_usados,
            "cdd_km": round(result.cdd_km, 3),
            "metodos": ",".join(result.metodos),
        })

        # ---- métricas: {metodo}__{metrica} ----
        for m in result.overall.index:
            for k in ["d", "RMSE", "BIAS", "MAE"]:
                _safe_log_metric(mlflow, f"{m}__{k}", result.overall.loc[m, k])

        # ---- p-valores do Wilcoxon: wilcoxon_p__RFRK_vs_{metodo} ----
        for m in result.wilcoxon.index:
            _safe_log_metric(mlflow, f"wilcoxon_p__RFRK_vs_{m}",
                             result.wilcoxon.loc[m, "p_value"])

        # ---- tags ----
        tags = {"foundation": "et0spatial", "dataset": dataset_kind}
        gc = _git_commit()
        if gc:
            tags["git_commit"] = gc
        mlflow.set_tags(tags)

        # ---- artifacts: a pasta outputs/ inteira ----
        mlflow.log_artifacts(result.out_dir)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLflow] experimento '{EXPERIMENT}' | run '{regiao}-{ano}' | id={run_id}")
    return result
