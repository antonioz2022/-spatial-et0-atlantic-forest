"""
ingest.py
=========
Lê os CSVs históricos de estações AUTOMÁTICAS do INMET
(https://portal.inmet.gov.br/dadoshistoricos — baixe o ZIP de 1 ano e descompacte).

De cada CSV extrai: REGIÃO/UF/ESTAÇÃO/CÓDIGO/LATITUDE/LONGITUDE/ALTITUDE no
cabeçalho, e a temperatura do ar horária -> Tmax/Tmin diárias.
Depois calcula ET0 diária por Hargreaves-Samani e monta as tabelas que o
pipeline consome.

ATENÇÃO: o layout dos CSVs do INMET muda um pouco por ano. Este leitor é
defensivo (busca colunas por nome aproximado, trata vírgula decimal, -9999,
latin-1), mas ABRA UM CSV e confira os nomes de coluna na primeira vez.

Uso:
    from et0spatial import ingest as ing
    long_df, meta = ing.load_inmet_dir("data/2023", bbox=(-42, -39.5, -21.5, -17.8))
    et0_df, coords_df = ing.build_et0_table(long_df, meta)
    # et0_df, coords_df vão direto pro core.loo_by_day(...)
"""
import os
import glob
import unicodedata
import numpy as np
import pandas as pd
from .core import et0_hargreaves


def _strip(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return " ".join(s.split())


def _num(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", ".")
    if s in ("", "-9999", "-9999.0", "NULL", "NAN"):
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return np.nan if v <= -9990 else v


def parse_inmet_csv(path):
    """Devolve (meta:dict, daily:DataFrame[date,tmax,tmin]) de um CSV do INMET."""
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # ---- cabeçalho de metadados (primeiras ~8 linhas "CHAVE:;valor") ----
    meta, header_idx = {}, None
    for i, ln in enumerate(lines[:15]):
        key = _strip(ln.split(";")[0])
        parts = ln.rstrip("\n").split(";")
        val = parts[1].strip() if len(parts) > 1 else ""
        if "LATITUDE" in key:
            meta["lat"] = _num(val)
        elif "LONGITUDE" in key:
            meta["lon"] = _num(val)
        elif "ALTITUDE" in key:
            meta["alt"] = _num(val)
        elif "CODIGO" in key or "CÓDIGO" in key:
            meta["code"] = val
        elif key.startswith("ESTACAO") or key.startswith("ESTAÇÃO"):
            meta["name"] = val
        elif key == "UF":
            meta["uf"] = val
        # a linha de cabeçalho dos dados costuma conter "Data" e ";"
        if header_idx is None and ("DATA" in _strip(ln) and ln.count(";") > 3):
            header_idx = i

    if header_idx is None:
        header_idx = 8  # fallback padrão do INMET

    df = pd.read_csv(path, sep=";", encoding="latin-1", skiprows=header_idx,
                     dtype=str, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # ---- localizar colunas por nome aproximado ----
    def find(*needles):
        for c in df.columns:
            cu = _strip(c)
            if all(n in cu for n in needles):
                return c
        return None

    col_date = find("DATA")
    col_t = (find("TEMPERATURA DO AR", "BULBO SECO")
             or find("TEMPERATURA", "AR", "HORARIA")
             or find("TEMPERATURA"))
    col_tmax = find("TEMPERATURA MAXIMA")
    col_tmin = find("TEMPERATURA MINIMA")
    if col_date is None or (col_t is None and col_tmax is None):
        raise ValueError(f"{os.path.basename(path)}: não achei colunas de data/temperatura. "
                         f"Colunas: {list(df.columns)[:8]}...")

    # data robusta (YYYY/MM/DD ou YYYY-MM-DD ou DD/MM/YYYY)
    d = (df[col_date].astype(str).str.replace("/", "-", regex=False))
    date = pd.to_datetime(d, errors="coerce", dayfirst=False)
    bad = date.isna()
    if bad.any():
        date.loc[bad] = pd.to_datetime(d[bad], errors="coerce", dayfirst=True)

    out = pd.DataFrame({"date": date.dt.normalize()})
    if col_t is not None:
        out["t"] = df[col_t].map(_num)
    if col_tmax is not None:
        out["tmax_h"] = df[col_tmax].map(_num)
    if col_tmin is not None:
        out["tmin_h"] = df[col_tmin].map(_num)
    out = out.dropna(subset=["date"])

    # Tmax/Tmin diárias: usa máx/mín do bulbo seco horário; se faltar, usa as
    # colunas de máx/mín horárias do INMET.
    agg = {}
    if "t" in out:
        agg["tmax"] = ("t", "max"); agg["tmin"] = ("t", "min")
    daily = out.groupby("date").agg(**agg) if agg else None
    if daily is None or daily["tmax"].isna().all():
        g = out.groupby("date")
        daily = pd.DataFrame({
            "tmax": g["tmax_h"].max() if "tmax_h" in out else np.nan,
            "tmin": g["tmin_h"].min() if "tmin_h" in out else np.nan,
        })
    daily = daily.reset_index()
    # exige amplitude válida
    daily = daily[(daily["tmax"] > daily["tmin"]) & daily["tmax"].notna() & daily["tmin"].notna()]
    return meta, daily


def load_inmet_dir(folder, bbox=None, min_days=200):
    """Lê todos os .CSV de uma pasta. bbox=(lon_min,lon_max,lat_min,lat_max)
    filtra a região. min_days descarta estações com poucos dias válidos.
    Retorna (long_df[date,station,tmax,tmin], meta_df[station,lon,lat,alt])."""
    paths = glob.glob(os.path.join(folder, "*.CSV")) + glob.glob(os.path.join(folder, "*.csv"))
    long_rows, metas = [], []
    for p in paths:
        try:
            meta, daily = parse_inmet_csv(p)
        except Exception as e:
            print(f"  [pulei] {os.path.basename(p)}: {e}")
            continue
        if not all(k in meta and np.isfinite(meta[k]) for k in ("lon", "lat", "alt")):
            continue
        if bbox is not None:
            lo, ho, la, ha = bbox
            if not (lo <= meta["lon"] <= ho and la <= meta["lat"] <= ha):
                continue
        if len(daily) < min_days:
            continue
        st = meta.get("code") or os.path.splitext(os.path.basename(p))[0]
        metas.append({"station": st, "lon": meta["lon"], "lat": meta["lat"], "alt": meta["alt"]})
        daily = daily.assign(station=st)
        long_rows.append(daily)

    if not long_rows:
        raise RuntimeError("Nenhuma estação válida. Cheque a pasta, o bbox e o layout dos CSVs.")
    long_df = pd.concat(long_rows, ignore_index=True)
    meta_df = pd.DataFrame(metas).drop_duplicates("station").set_index("station")
    print(f"Estações carregadas: {len(meta_df)} | linhas dia-estação: {len(long_df)}")
    return long_df, meta_df


def build_et0_table(long_df, meta_df):
    """Calcula ET0 (Hargreaves) e devolve (et0_df: datas x estações, coords_df)."""
    df = long_df.merge(meta_df["lat"], left_on="station", right_index=True)
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    df["et0"] = et0_hargreaves(df["tmax"].values, df["tmin"].values,
                               df["lat"].values, df["doy"].values)
    et0_df = df.pivot_table(index="date", columns="station", values="et0")
    coords_df = meta_df.loc[et0_df.columns, ["lon", "lat", "alt"]].copy()
    return et0_df.sort_index(), coords_df


if __name__ == "__main__":
    print(__doc__)
    print(">>> Baixe 1 ano em https://portal.inmet.gov.br/dadoshistoricos, "
          "descompacte numa pasta e chame load_inmet_dir(pasta, bbox=...).")
