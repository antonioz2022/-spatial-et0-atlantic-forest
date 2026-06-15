"""
cli.py
======
Interface de linha de comando do pacote (`python -m et0spatial ...`).

Exemplos:
    python -m et0spatial --demo --fast               # dados sintéticos (rápido)
    python -m et0spatial --inmet data/2023 --bbox -41 -34 -11 -4
    python -m et0spatial --inmet data/2023 --bbox -41 -34 -11 -4 --regiao espirito-santo

Orquestra o pipeline (core.py) e grava `outputs/` (4 figuras + resultados.md
+ tabela3.csv + pred_long.parquet), registrando tudo no MLflow.
"""
from __future__ import annotations
import argparse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="et0spatial",
        description="ET0 spatial — reprodução + contribuição (Baratto et al., 2022). "
                    "IDW1-5/ADW/RF (reprodução) + OK/RFRK/Wilcoxon/importância (contribuição), "
                    "via LOO-CV, com rastreamento MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--inmet", default=None, metavar="PASTA",
                    help="pasta com os CSVs descompactados do INMET (dados reais)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="recorte da região (graus). Ex.: -41 -34 -11 -4")
    ap.add_argument("--demo", action="store_true",
                    help="modo sintético (não precisa de dados)")
    ap.add_argument("--fast", action="store_true",
                    help="iteração rápida: subamostra dias (1 a cada 2) e usa 80 árvores")
    ap.add_argument("--out", default="outputs",
                    help="pasta de saída das tabelas e figuras")
    ap.add_argument("--regiao", default=None,
                    help="rótulo da região p/ nomear o run do MLflow (default: sintetico/inmet)")
    ap.add_argument("--max-days", type=int, default=None, dest="max_days",
                    help="limita o nº de dias usados no LOO (testes/smoke; default: todos)")
    ap.add_argument("--no-mlflow", action="store_true", dest="no_mlflow",
                    help="executa sem MLflow (depuração)")
    return ap


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.demo and not args.inmet:
        parser.error("especifique --demo (dados sintéticos) ou --inmet PASTA (dados reais do INMET)")

    # imports pesados só depois do parse (mantém --help instantâneo)
    from .pipeline import PipelineConfig

    cfg = PipelineConfig(
        out=args.out,
        inmet=args.inmet,
        bbox=tuple(args.bbox) if args.bbox else None,
        demo=args.demo,
        fast=args.fast,
        regiao=args.regiao,
        max_days=args.max_days,
    )

    if args.no_mlflow:
        from . import pipeline
        pipeline.run_local(cfg)
    else:
        from . import tracking
        tracking.run(cfg)


if __name__ == "__main__":
    main()
