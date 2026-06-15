"""
pipeline.py
===========
Orquestra o experimento de ponta a ponta:

    dados -> ET0 (Hargreaves) -> CDD -> LOO-CV (9 métodos) -> métricas
          -> Wilcoxon -> importância de variáveis -> 4 figuras + tabelas

A reprodução (IDW1-5, ADW, RF) e a contribuição (OK, RFRK, Wilcoxon,
importância) saem todas do mesmo `core.loo_by_day`.

Saídas gravadas em `cfg.out` (default `outputs/`):
    resultados.md  tabela3.csv  pred_long.parquet
    fig_metricas.png  fig_scatter.png  fig_mapas.png  fig_importancia.png
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import core as es
from . import figures

COV_COLS = ["alt"]            # covariáveis usadas por RF/RFRK (além de lon/lat)
COVARIAVEIS_STR = "lon,lat,alt"
ET0_METHOD = "hargreaves_samani"


# --------------------------------------------------------------------
# Configuração de uma execução
# --------------------------------------------------------------------
@dataclass
class PipelineConfig:
    out: str = "outputs"
    inmet: str | None = None
    bbox: tuple | None = None
    demo: bool = False
    fast: bool = False
    regiao: str | None = None
    max_days: int | None = None   # corta o nº de dias após a subamostragem (testes/smoke)

    @property
    def rf_trees(self) -> int:
        # 80 árvores no modo rápido, 200 no normal
        return 80 if self.fast else 200


@dataclass
class Result:
    overall: pd.DataFrame
    pred_long: pd.DataFrame
    wilcoxon: pd.DataFrame
    importance: pd.Series
    cdd_km: float
    n_estacoes: int
    n_dias_usados: int
    ano: int
    rf_trees: int
    metodos: list
    dataset: str
    out_dir: str
    artifacts: list = field(default_factory=list)


# --------------------------------------------------------------------
# Dados sintéticos do modo --demo
# --------------------------------------------------------------------
def synthetic():
    rng = np.random.default_rng(42)
    n = 20
    lon = rng.uniform(-40.5, -35.0, n); lat = rng.uniform(-10.0, -5.0, n)
    alt = rng.uniform(5, 900, n)
    st = [f"E{i:02d}" for i in range(n)]
    coords_df = pd.DataFrame({"lon": lon, "lat": lat, "alt": alt}, index=st)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    doy = dates.dayofyear.values
    season = 1.0 * np.sin(2 * np.pi * (doy - 80) / 365)
    spatial = 0.06 * (lat + 8) - 0.0011 * alt
    M = np.empty((len(dates), n))
    for j in range(n):
        M[:, j] = 4.2 + spatial[j] + season + rng.normal(0, 0.25, len(dates))
    et0_df = pd.DataFrame(np.clip(M, 0.5, None), index=dates, columns=st)
    et0_df = et0_df.mask(rng.random(et0_df.shape) < 0.03)
    return et0_df, coords_df


# --------------------------------------------------------------------
# Carregamento dos dados (sintético ou INMET)
# --------------------------------------------------------------------
def load_data(cfg: PipelineConfig):
    """Retorna (et0_df, coords_df, dataset_kind)."""
    if cfg.inmet:
        from . import ingest as ing
        long_df, meta = ing.load_inmet_dir(cfg.inmet, bbox=cfg.bbox)
        et0_df, coords_df = ing.build_et0_table(long_df, meta)
        return et0_df, coords_df, "INMET-automaticas"
    print(">> modo sintético (--demo). Para dados reais use --inmet PASTA --bbox ...")
    et0_df, coords_df = synthetic()
    return et0_df, coords_df, "sintetico"


def dominant_year(et0_df) -> int:
    years = pd.DatetimeIndex(et0_df.index).year
    return int(pd.Series(years).mode().iloc[0])


# --------------------------------------------------------------------
# Execução do pipeline científico + gravação das saídas
# --------------------------------------------------------------------
def compute(cfg: PipelineConfig, et0_df, coords_df, dataset_kind: str) -> Result:
    os.makedirs(cfg.out, exist_ok=True)
    print(f"Estações: {coords_df.shape[0]} | dias: {et0_df.shape[0]}")

    trees = cfg.rf_trees
    run_df = et0_df.iloc[::2] if cfg.fast else et0_df
    if cfg.max_days is not None:
        run_df = run_df.iloc[: cfg.max_days]

    cdd = es.estimate_cdd(coords_df[["lon", "lat"]].values, et0_df.values.T)
    print(f"CDD: {cdd:.1f} km | LOO rodando ({len(run_df)} dias, {trees} árvores)...")

    overall, pred_long = es.loo_by_day(
        run_df, coords_df, cov_cols=COV_COLS,
        model_factories=es.default_factories(rf_trees=trees), cdd_km=cdd, verbose=True)

    wil = es.wilcoxon_vs(pred_long, ref="RFRK", metric="RMSE")
    imp = es.rf_feature_importance(run_df, coords_df, cov_cols=COV_COLS,
                                   use_coords=True, add_doy=True, n_estimators=200)

    # ---- tabelas em markdown (cole no relatório) ----
    res_md = os.path.join(cfg.out, "resultados.md")
    with open(res_md, "w") as f:
        f.write("# Resultados\n\n## Tabela 3 — reprodução + contribuição (LOO-CV)\n\n")
        f.write(overall.to_markdown() + "\n\n")
        f.write("## Significância — RFRK vs. cada método (Wilcoxon pareado, RMSE diário)\n\n")
        f.write("p < 0,05 = diferença significativa.\n\n")
        f.write(wil.round(4).to_markdown() + "\n\n")
        f.write("## Importância de variáveis (RF)\n\n")
        f.write(imp.round(3).to_frame("importância").to_markdown() + "\n")
    tab_csv = os.path.join(cfg.out, "tabela3.csv")
    overall.to_csv(tab_csv)
    # dados brutos previsto×observado (alimenta MLflow e re-análises)
    pred_parquet = os.path.join(cfg.out, "pred_long.parquet")
    pred_long.to_parquet(pred_parquet, index=False)

    # ---- figuras ----
    f_met = os.path.join(cfg.out, "fig_metricas.png")
    f_sca = os.path.join(cfg.out, "fig_scatter.png")
    f_map = os.path.join(cfg.out, "fig_mapas.png")
    f_imp = os.path.join(cfg.out, "fig_importancia.png")
    figures.fig_metricas(overall, f_met)
    figures.fig_scatter(pred_long, f_sca)
    figures.fig_mapas(et0_df, coords_df, f_map)
    figures.fig_importancia(imp, f_imp)

    print("\n=== TABELA 3 ===")
    print(overall.to_string())
    print("\n=== Wilcoxon (RFRK vs.) ===")
    print(wil.round(4).to_string())
    print("\n=== Importância ===")
    print(imp.round(3).to_string())
    print(f"\nTudo salvo em ./{cfg.out}/  (resultados.md + tabela3.csv + pred_long.parquet + 4 figuras).")

    return Result(
        overall=overall, pred_long=pred_long, wilcoxon=wil, importance=imp,
        cdd_km=float(cdd), n_estacoes=int(coords_df.shape[0]), n_dias_usados=int(len(run_df)),
        ano=dominant_year(et0_df), rf_trees=trees, metodos=list(overall.index),
        dataset=dataset_kind, out_dir=cfg.out,
        artifacts=[res_md, tab_csv, pred_parquet, f_met, f_sca, f_map, f_imp],
    )


def run_local(cfg: PipelineConfig) -> Result:
    """Executa o pipeline sem MLflow (modo --no-mlflow / debug)."""
    et0_df, coords_df, dataset_kind = load_data(cfg)
    return compute(cfg, et0_df, coords_df, dataset_kind)
