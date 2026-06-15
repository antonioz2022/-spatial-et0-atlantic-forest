"""
Smoke test: roda a CLI em modo sintético (--demo --fast, poucos dias) e checa
que (1) as 7 saídas foram gravadas e (2) um run do MLflow foi criado com as
métricas {metodo}__{metrica} e os p-valores do Wilcoxon.

Rode com:  pytest        (após `make install` / `pip install -e .`)
"""
import os
import sys
import subprocess


def test_demo_pipeline_and_mlflow(tmp_path):
    out = tmp_path / "outputs"
    mlruns = tmp_path / "mlruns"

    env = dict(os.environ)
    env["MLFLOW_TRACKING_URI"] = mlruns.as_uri()  # file:///...

    r = subprocess.run(
        [sys.executable, "-m", "et0spatial", "--demo", "--fast",
         "--max-days", "14", "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, f"CLI falhou:\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}"

    # (1) saídas esperadas
    esperados = [
        "resultados.md", "tabela3.csv", "pred_long.parquet",
        "fig_metricas.png", "fig_scatter.png", "fig_mapas.png", "fig_importancia.png",
    ]
    for name in esperados:
        assert (out / name).exists(), f"saída ausente: {name}"

    # (2) run do MLflow com métricas e Wilcoxon
    import mlflow

    mlflow.set_tracking_uri(mlruns.as_uri())
    exp = mlflow.get_experiment_by_name("et0-spatial-interpolation")
    assert exp is not None, "experimento MLflow não criado"
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1, "nenhum run registrado no MLflow"

    cols = set(runs.columns)
    assert "metrics.RFRK__d" in cols, "métrica RFRK__d ausente no run"
    assert "metrics.IDW5__RMSE" in cols, "métrica IDW5__RMSE ausente no run"
    assert any(c.startswith("metrics.wilcoxon_p__RFRK_vs_") for c in cols), \
        "p-valores do Wilcoxon ausentes no run"
    assert "params.et0_method" in cols, "params do run ausentes"
