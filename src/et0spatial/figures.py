"""
figures.py
==========
Gera as quatro figuras do relatório:

    fig_metricas      -> d de Willmott e RMSE por método
    fig_scatter       -> observado × previsto, verão vs inverno
    fig_mapas         -> IDW2 × RF × RFRK (artefato "em blocos" do RF)
    fig_importancia   -> importância de variáveis do RF (doy vs espaço)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import core as es


def season_of(dates):
    # Hemisfério Sul: verão = DJF, inverno = JJA
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    m = idx.month
    return np.where(np.isin(m, [12, 1, 2]), "verão",
            np.where(np.isin(m, [6, 7, 8]), "inverno", "outro"))


def fig_metricas(overall, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    overall["d"].plot.bar(ax=ax[0], color="#3b7dd8"); ax[0].set_title("Índice d de Willmott"); ax[0].set_ylim(0, 1)
    overall["RMSE"].plot.bar(ax=ax[1], color="#d8743b"); ax[1].set_title("RMSE (mm/dia)")
    for a in ax:
        a.set_xlabel(""); a.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_scatter(pred_long, path, methods=("RF", "RFRK")):
    pl = pred_long.copy(); pl["season"] = season_of(pl["date"])
    fig, axes = plt.subplots(len(methods), 2, figsize=(9, 4 * len(methods)))
    axes = np.atleast_2d(axes)
    for i, m in enumerate(methods):
        for j, s in enumerate(["verão", "inverno"]):
            sub = pl[(pl["method"] == m) & (pl["season"] == s)]
            ax = axes[i, j]
            if len(sub):
                ax.scatter(sub["obs"], sub["pred"], s=6, alpha=0.3, color="#2a6")
                lim = [min(sub["obs"].min(), sub["pred"].min()),
                       max(sub["obs"].max(), sub["pred"].max())]
                ax.plot(lim, lim, "k--", lw=1)
                d = es.willmott_d(sub["obs"].values, sub["pred"].values)
                r = es.rmse(sub["obs"].values, sub["pred"].values)
                ax.set_title(f"{m} — {s}  (d={d:.2f}, RMSE={r:.2f})", fontsize=10)
            ax.set_xlabel("ET0 observada"); ax.set_ylabel("ET0 estimada")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _grid(coords_df, nx=70, ny=70):
    lo, hi = coords_df["lon"].min(), coords_df["lon"].max()
    la, ha = coords_df["lat"].min(), coords_df["lat"].max()
    gx = np.linspace(lo, hi, nx); gy = np.linspace(la, ha, ny)
    XX, YY = np.meshgrid(gx, gy)
    G = np.column_stack([XX.ravel(), YY.ravel()])
    return G, XX, YY, (lo, hi, la, ha)


def fig_mapas(et0_df, coords_df, path):
    coords = coords_df[["lon", "lat"]].values
    alt = coords_df["alt"].values
    G, XX, YY, ext = _grid(coords_df)
    # altitude na grade via IDW (não temos DEM -> aproximação p/ a covariável)
    alt_grid = es.IDW(power=2).fit(coords, alt).predict(G)
    # escolhe um dia de verão e um de inverno com mais estações válidas
    s = season_of(et0_df.index)
    def pick(season):
        sub = et0_df[s == season]
        return sub.loc[sub.notna().sum(axis=1).idxmax()]
    rows = [("Verão", pick("verão")), ("Inverno", pick("inverno"))]
    methods = [("IDW2", es.IDW(power=2)), ("RF", es.RFInterp(n_estimators=200)),
               ("RFRK", es.RFRK(n_estimators=200))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ri, (lab, day) in enumerate(rows):
        y = day.values.astype(float); mask = np.isfinite(y)
        c = coords[mask]; yv = y[mask]; av = alt[mask]
        vmin, vmax = np.nanmin(yv), np.nanmax(yv)
        for ci, (mn, mdl) in enumerate(methods):
            cov_tr = av[:, None] if mn in ("RF", "RFRK") else None
            cov_gr = alt_grid[:, None] if mn in ("RF", "RFRK") else None
            mdl.fit(c, yv, cov_tr)
            Z = mdl.predict(G, cov_gr).reshape(XX.shape)
            ax = axes[ri, ci]
            im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto",
                           cmap="YlOrRd", vmin=vmin, vmax=vmax)
            ax.scatter(c[:, 0], c[:, 1], c="k", s=12)
            ax.set_title(f"{lab} — {mn}", fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("ET0 interpolada — repare nos 'blocos' do RF vs. a suavidade do RFRK", y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_importancia(imp, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp.iloc[::-1].plot.barh(ax=ax, color="#6a3bd8")
    ax.set_title("Importância de variáveis (RF)"); ax.set_xlabel("importância")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
