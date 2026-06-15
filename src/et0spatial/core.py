"""
etspatial.py
============
Reprodução + contribuição do paper:
  Baratto et al. (2022), "Random forest for spatialization of daily
  evapotranspiration (ET0) in watersheds in the Atlantic Forest".

Reproduz a comparação de interpoladores (IDW, ADW, RF) via validação cruzada
leave-one-out (LOO), dia a dia, com as métricas d de Willmott, RMSE e BIAS.

Contribuição (não estava no paper):
  - OK   : krigagem ordinária (benchmark geoestatístico ausente no paper)
  - RFRK : Random Forest Regression Kriging (RF nas covariáveis + krigagem
           dos resíduos) -> ataca a fraqueza do paper (RF perdeu no número e
           gera superfície "em blocos").
  - Teste de significância (Wilcoxon pareado dia a dia).
  - Importância de variáveis do RF (promessa não cumprida no paper).

Interface comum dos interpoladores:
    m = Modelo(...); m.fit(coords, values, cov=None); m.predict(coords_new, cov_new=None)
    coords: array (n,2) em (lon, lat) graus.
    cov   : array (n,k) de covariáveis (ex.: altitude). Usado só por RF/RFRK.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import wilcoxon

# =====================================================================
# Distâncias (Haversine, km)
# =====================================================================
_R_EARTH = 6371.0

def haversine(lon1, lat1, lon2, lat2):
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    dlat = (lat2 - lat1) * p
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2
    return 2 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def pairwise_dist(coords):
    coords = np.asarray(coords, float)
    return haversine(coords[:, 0][:, None], coords[:, 1][:, None],
                     coords[:, 0][None, :], coords[:, 1][None, :])

def cross_dist(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return haversine(a[:, 0][:, None], a[:, 1][:, None],
                     b[:, 0][None, :], b[:, 1][None, :])

# =====================================================================
# ET0 por Hargreaves-Samani (só Tmax/Tmin -> dados limpos, sem GIS)
#   Defensável: é o fallback FAO-56 p/ escassez de dados (Allen et al. 1998),
#   o mesmo método do banco BRAUM citado no paper.
# =====================================================================
def extraterrestrial_radiation(lat_deg, doy):
    """Radiação no topo da atmosfera Ra (MJ m-2 dia-1), FAO-56."""
    lat = np.deg2rad(np.asarray(lat_deg, float))
    doy = np.asarray(doy, float)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    gsc = 0.0820
    return (24 * 60 / np.pi) * gsc * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))

def et0_hargreaves(tmax, tmin, lat_deg, doy):
    """ET0 diária (mm/dia) por Hargreaves-Samani."""
    ra_mm = 0.408 * extraterrestrial_radiation(lat_deg, doy)  # MJ -> mm
    tmean = (np.asarray(tmax, float) + np.asarray(tmin, float)) / 2
    tr = np.clip(np.asarray(tmax, float) - np.asarray(tmin, float), 0, None)
    return 0.0023 * (tmean + 17.8) * np.sqrt(tr) * ra_mm

# =====================================================================
# CDD (correlation decay distance) — estimada UMA vez da série completa,
# como no paper (não dá p/ estimar com 1 dia só dentro do LOO).
# =====================================================================
def estimate_cdd(coords, Y, min_overlap=10):
    """Y: (n_stations, n_days). Ajusta r = exp(-x/CDD) -> CDD em km."""
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    D = pairwise_dist(coords)
    rs, ds = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = Y[i], Y[j]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < min_overlap:
                continue
            if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
                continue
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            if np.isfinite(r) and r > 0:
                rs.append(r); ds.append(D[i, j])
    if len(rs) < 3:
        return float(np.median(D[D > 0]))
    slope = np.polyfit(np.array(ds), np.log(np.array(rs)), 1)[0]
    if slope < 0:
        return float(-1.0 / slope)
    return float(np.median(D[D > 0]))

# =====================================================================
# Interpoladores
# =====================================================================
@dataclass
class IDW:
    """Inverse Distance Weighting. power = expoente p (paper varia 1..5)."""
    power: float = 2.0
    coords_: np.ndarray = field(default=None, repr=False)
    values_: np.ndarray = field(default=None, repr=False)

    def fit(self, coords, values, cov=None):
        self.coords_ = np.asarray(coords, float)
        self.values_ = np.asarray(values, float)
        return self

    def predict(self, coords_new, cov_new=None):
        d = cross_dist(coords_new, self.coords_)
        out = np.empty(d.shape[0])
        for i in range(d.shape[0]):
            di = d[i]
            zero = di <= 1e-9
            if zero.any():
                out[i] = self.values_[zero].mean()
            else:
                w = di ** (-self.power)
                out[i] = np.sum(w * self.values_) / np.sum(w)
        return out


@dataclass
class ADW:
    """Angular Distance Weighting (New et al. 2000). nj=8, m=4 no paper.
    Peso de distância w = (exp(-x/CDD))**m ; termo angular favorece estações
    angularmente espalhadas. Obs.: o paper escreve WE=w(1-a); a forma canônica
    de New et al. é WE=w(1+a) (usada aqui). Troque o sinal em `predict` se
    quiser bater exatamente com a equação 5 do paper."""
    n_neighbors: int = 8
    m: float = 4.0
    cdd_km: float | None = None  # injetada pelo harness (estimate_cdd)
    coords_: np.ndarray = field(default=None, repr=False)
    values_: np.ndarray = field(default=None, repr=False)

    def fit(self, coords, values, cov=None):
        self.coords_ = np.asarray(coords, float)
        self.values_ = np.asarray(values, float)
        if self.cdd_km is None:
            dd = pairwise_dist(self.coords_)
            self.cdd_run_ = float(np.median(dd[dd > 0]))
        else:
            self.cdd_run_ = float(self.cdd_km)
        return self

    def predict(self, coords_new, cov_new=None):
        cn = np.asarray(coords_new, float)
        d = cross_dist(cn, self.coords_)
        out = np.empty(cn.shape[0])
        k = min(self.n_neighbors, self.coords_.shape[0])
        for i in range(cn.shape[0]):
            di = d[i]
            zero = di <= 1e-9
            if zero.any():
                out[i] = self.values_[zero].mean(); continue
            idx = np.argsort(di)[:k]
            x = di[idx]
            w = (np.exp(-x / self.cdd_run_)) ** self.m
            vec = self.coords_[idx] - cn[i]              # direções ponto->vizinho
            norms = np.linalg.norm(vec, axis=1) + 1e-12
            ang = np.zeros(k)
            for li in range(k):
                num = den = 0.0
                for lj in range(k):
                    if lj == li:
                        continue
                    cos = np.dot(vec[li], vec[lj]) / (norms[li] * norms[lj])
                    num += w[lj] * (1 - cos); den += w[lj]
                ang[li] = num / den if den > 0 else 0.0
            we = w * (1 + ang)
            out[i] = np.sum(we * self.values_[idx]) / np.sum(we)
        return out


@dataclass
class RFInterp:
    """Random Forest baseado em covariáveis (como no paper). Inclui lon/lat
    como features se use_coords=True."""
    n_estimators: int = 200
    use_coords: bool = True
    random_state: int = 0
    rf_: RandomForestRegressor = field(default=None, repr=False)

    def _X(self, coords, cov):
        parts = []
        if self.use_coords:
            parts.append(np.asarray(coords, float))
        if cov is not None:
            parts.append(np.asarray(cov, float))
        return np.hstack(parts)

    def fit(self, coords, values, cov=None):
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        self.rf_.fit(self._X(coords, cov), np.asarray(values, float))
        return self

    def predict(self, coords_new, cov_new=None):
        return self.rf_.predict(self._X(coords_new, cov_new))


def _exp_variogram(h, nugget, sill, rng):
    return nugget + sill * (1 - np.exp(-h / np.maximum(rng, 1e-9)))


@dataclass
class OK:
    """Krigagem Ordinária com modelo exponencial. Por padrão usa sill = var
    amostral e range = 1/3 da distância máxima (estável p/ poucas estações).
    Passe range_km p/ fixar o alcance."""
    range_km: float | None = None
    nugget_frac: float = 0.0
    coords_: np.ndarray = field(default=None, repr=False)
    values_: np.ndarray = field(default=None, repr=False)

    def fit(self, coords, values, cov=None):
        self.coords_ = np.asarray(coords, float)
        self.values_ = np.asarray(values, float)
        n = len(self.values_)
        D = pairwise_dist(self.coords_)
        var = float(np.var(self.values_))
        self.sill_ = var if var > 0 else 1.0
        self.nugget_ = self.nugget_frac * self.sill_
        self.rng_ = float(self.range_km) if self.range_km is not None \
            else max(np.max(D) / 3.0, 1e-6)
        G = _exp_variogram(D, self.nugget_, self.sill_, self.rng_)
        C = (self.sill_ + self.nugget_) - G
        A = np.ones((n + 1, n + 1)); A[:n, :n] = C; A[n, n] = 0.0
        self.A_inv_ = np.linalg.pinv(A)
        return self

    def predict(self, coords_new, cov_new=None):
        cn = np.asarray(coords_new, float)
        d = cross_dist(cn, self.coords_)
        G = _exp_variogram(d, self.nugget_, self.sill_, self.rng_)
        c0 = (self.sill_ + self.nugget_) - G
        n = len(self.values_)
        out = np.empty(cn.shape[0])
        for i in range(cn.shape[0]):
            b = np.empty(n + 1); b[:n] = c0[i]; b[n] = 1.0
            w = self.A_inv_ @ b
            out[i] = float(np.sum(w[:n] * self.values_))
        return out


@dataclass
class RFRK:
    """Random Forest Regression Kriging: RF(covariáveis) + OK(resíduos).
    Esta é a contribuição-estrela: junta detalhe de relevo do RF com a
    autocorrelação espacial da krigagem."""
    n_estimators: int = 200
    range_km: float | None = None
    use_coords: bool = True
    rf_: RFInterp = field(default=None, repr=False)
    ok_: OK = field(default=None, repr=False)

    def fit(self, coords, values, cov=None):
        self.rf_ = RFInterp(self.n_estimators, self.use_coords).fit(coords, values, cov)
        resid = np.asarray(values, float) - self.rf_.predict(coords, cov)
        self.ok_ = OK(self.range_km).fit(coords, resid)
        return self

    def predict(self, coords_new, cov_new=None):
        return self.rf_.predict(coords_new, cov_new) + self.ok_.predict(coords_new)

# =====================================================================
# Métricas (BIAS no sentido do paper: O - E; negativo => superestima)
# =====================================================================
def willmott_d(O, E):
    O = np.asarray(O, float); E = np.asarray(E, float)
    ob = O.mean()
    num = np.sum((E - O) ** 2)
    den = np.sum((np.abs(E - ob) + np.abs(O - ob)) ** 2)
    return float(1 - num / den) if den > 0 else np.nan

def rmse(O, E):
    return float(np.sqrt(np.mean((np.asarray(E, float) - np.asarray(O, float)) ** 2)))

def bias(O, E):
    return float(np.mean(np.asarray(O, float) - np.asarray(E, float)))

def mae(O, E):
    return float(np.mean(np.abs(np.asarray(E, float) - np.asarray(O, float))))

_METRIC_FN = {"d": willmott_d, "RMSE": rmse, "BIAS": bias, "MAE": mae}

# =====================================================================
# Validação cruzada Leave-One-Out, dia a dia (igual ao paper)
# =====================================================================
def loo_by_day(et0_df, coords_df, cov_cols, model_factories,
               cdd_km=None, min_stations=5, verbose=True):
    """
    et0_df    : DataFrame (index=datas, colunas=id_estacao) de ET0 diária.
    coords_df : DataFrame (index=id_estacao) com colunas ['lon','lat'] + cov_cols.
    cov_cols  : lista de colunas usadas como covariáveis (ex.: ['alt']). Pode ser [].
    model_factories : dict nome -> callable() que devolve um interpolador NOVO.
    cdd_km    : CDD global p/ o ADW (use estimate_cdd antes).
    Retorna   : (overall_df, pred_long_df)
                overall_df: métricas globais por método (d, RMSE, BIAS, MAE).
                pred_long_df: colunas [date, station, method, obs, pred].
    """
    stations = list(et0_df.columns)
    coords = coords_df.loc[stations, ["lon", "lat"]].values.astype(float)
    cov = coords_df.loc[stations, cov_cols].values.astype(float) if cov_cols else None
    names = list(model_factories)

    O_all = {m: [] for m in names}
    E_all = {m: [] for m in names}
    rows = []
    dates = list(et0_df.index)
    for di, date in enumerate(dates):
        if verbose and di % 50 == 0:
            print(f"  dia {di+1}/{len(dates)}", flush=True)
        y = et0_df.loc[date].values.astype(float)
        mask = np.isfinite(y)
        sidx = np.where(mask)[0]
        if sidx.size < min_stations:
            continue
        for name, factory in model_factories.items():
            Od, Ed = [], []
            for h in sidx:                       # estação retirada (held-out)
                tr = sidx[sidx != h]
                mdl = factory()
                if cdd_km is not None and hasattr(mdl, "cdd_km"):
                    mdl.cdd_km = cdd_km
                covtr = cov[tr] if cov is not None else None
                covh = cov[h:h + 1] if cov is not None else None
                mdl.fit(coords[tr], y[tr], covtr)
                pred = float(mdl.predict(coords[h:h + 1], covh)[0])
                Od.append(y[h]); Ed.append(pred)
                rows.append((date, stations[h], name, y[h], pred))
            O_all[name].extend(Od); E_all[name].extend(Ed)

    overall = pd.DataFrame(
        {m: {k: fn(O_all[m], E_all[m]) for k, fn in _METRIC_FN.items()} for m in names}
    ).T[["d", "RMSE", "BIAS", "MAE"]].round(3)
    pred_long = pd.DataFrame(rows, columns=["date", "station", "method", "obs", "pred"])
    return overall, pred_long

# =====================================================================
# Análises da contribuição
# =====================================================================
def perday_metric(pred_long, metric="RMSE"):
    """DataFrame datas x métodos com a métrica calculada por dia."""
    fn = _METRIC_FN[metric]
    g = (pred_long.groupby(["date", "method"])
         .apply(lambda x: fn(x["obs"].values, x["pred"].values), include_groups=False))
    return g.unstack("method")

def wilcoxon_vs(pred_long, ref="RFRK", metric="RMSE"):
    """Wilcoxon pareado (dia a dia) do método `ref` contra cada outro.
    p<0.05 => diferença significativa. Responde o 'a diferença é pequena'
    que o paper deixou no ar."""
    M = perday_metric(pred_long, metric).dropna()
    res = {}
    for m in M.columns:
        if m == ref:
            continue
        try:
            _, p = wilcoxon(M[ref], M[m])
        except ValueError:
            p = np.nan
        res[m] = {f"med_{ref}": float(M[ref].median()),
                  "med_outro": float(M[m].median()),
                  "dif_mediana": float(M[ref].median() - M[m].median()),
                  "p_value": float(p),
                  "signif_5pct": bool(np.isfinite(p) and p < 0.05)}
    return pd.DataFrame(res).T

def rf_feature_importance(et0_df, coords_df, cov_cols, use_coords=True,
                          add_doy=True, n_estimators=300):
    """Treina UM RF empilhando todos os dias e devolve a importância média
    das variáveis. Inclui 'doy' (dia do ano) p/ medir quanto da skill é
    sazonal vs espacial — exatamente o que o paper afirmou mas não mostrou."""
    stations = list(et0_df.columns)
    coords = coords_df.loc[stations, ["lon", "lat"]].values.astype(float)
    cov = coords_df.loc[stations, cov_cols].values.astype(float) if cov_cols else None
    Xs, ys = [], []
    for date in et0_df.index:
        y = et0_df.loc[date].values.astype(float)
        mask = np.isfinite(y)
        feats = []
        if use_coords:
            feats.append(coords[mask])
        if cov is not None:
            feats.append(cov[mask])
        if add_doy:
            doy = np.full(mask.sum(), pd.Timestamp(date).dayofyear, float)
            feats.append(doy[:, None])
        Xs.append(np.hstack(feats)); ys.append(y[mask])
    X = np.vstack(Xs); y = np.concatenate(ys)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-1).fit(X, y)
    names = (["lon", "lat"] if use_coords else []) + list(cov_cols) + (["doy"] if add_doy else [])
    return pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)

def default_factories(rf_trees=200):
    """O conjunto do trabalho: 3 do paper (IDW1-5, ADW, RF) + 2 da contribuição (OK, RFRK)."""
    fac = {f"IDW{p}": (lambda p=p: IDW(power=p)) for p in range(1, 6)}
    fac["ADW"] = lambda: ADW(n_neighbors=8, m=4)
    fac["RF"] = lambda: RFInterp(n_estimators=rf_trees)
    fac["OK"] = lambda: OK()
    fac["RFRK"] = lambda: RFRK(n_estimators=rf_trees)
    return fac
