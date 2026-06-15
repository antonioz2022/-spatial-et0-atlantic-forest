"""
et0spatial
==========
Reprodução + contribuição do paper Baratto et al. (2022), "Random forest for
spatialization of daily evapotranspiration (ET0) in watersheds in the Atlantic
Forest" (Environ Monit Assess 194:449).

Submódulos:
    core      -> motor: ET0, interpoladores, LOO-CV, métricas, Wilcoxon,
                 importância de variáveis.
    ingest    -> leitura dos CSVs do INMET.
    figures   -> as 4 figuras do relatório.
    pipeline  -> orquestração ponta a ponta + geração das saídas.
    tracking  -> rastreamento MLflow.
    cli       -> interface de linha de comando (`python -m et0spatial`).
"""
__version__ = "0.1.0"
