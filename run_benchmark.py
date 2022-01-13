import argparse
from pathlib import Path

import yaml
from sklearn.model_selection import ParameterGrid

from scoring import encoder_scoring, pipe_scoring, stat_scoring
from src import MLBenchmark, StatBenchmark

BACKUP_DIR = Path("./src/data")
CONFIG_DIR = Path("./config")
DATASET_DIR = Path("./datasets")
RESULT_DIR = Path("./results")


def run_stat_benchmark(args):
    n_samplings = args.n_samplings
    seed = args.seed
    table_name = args.table

    benchmark_name = f"{n_samplings}-{seed}"

    backup_file = BACKUP_DIR / f"stat_{benchmark_name}.pkl"
    table_file = CONFIG_DIR / "stat-table" / f"{table_name}.yml"
    result_file = RESULT_DIR / "stat" / f"{benchmark_name}.csv"

    benchmark = StatBenchmark(n_samplings=n_samplings, seed=seed)

    if backup_file.is_file():
        benchmark.load(backup_file)

    with open(table_file, "r") as fp:
        table = list(yaml.safe_load_all(fp))

        sample_table = [row for row in table if "model" not in row]
        sample_table = [ParameterGrid(row) for row in sample_table]
        sample_table = [row for row_lst in sample_table for row in row_lst]

        model_table = [row for row in table if "model" in row]

    benchmark.run(model_table, sample_table)
    scores = benchmark.score(stat_scoring)

    benchmark.dump(backup_file)
    scores.to_csv(result_file, index=False)

    print(f"Results written to {result_file}")


def run_ml_benchmark(args):
    benchmark_name = args.benchmark
    table_name = args.table

    backup_file = BACKUP_DIR / f"ml_{benchmark_name}.pkl"
    benchmark_file = CONFIG_DIR / "ml-benchmark" / f"{benchmark_name}.yml"
    table_file = CONFIG_DIR / "ml-table" / f"{table_name}.yml"
    encoder_result_file = RESULT_DIR / "encoder" / f"{benchmark_name}.csv"
    pipe_result_file = RESULT_DIR / "pipe" / f"{benchmark_name}.csv"

    with open(benchmark_file, "r") as fp:
        benchmark_params = yaml.safe_load(fp)
        dataset = benchmark_params.pop("dataset")
        dataset_file = DATASET_DIR / f"{dataset}.csv"
        benchmark = MLBenchmark(dataset_file, **benchmark_params)

    if backup_file.is_file():
        benchmark.load(backup_file)

    with open(table_file, "r") as fp:
        table = list(yaml.safe_load_all(fp))
        encoder_table = [row for row in table if "encoder" in row]
        classifier_table = [row for row in table if "classifier" in row]

    benchmark.run(encoder_table, classifier_table)
    encoder_scores = benchmark.encoder_score(encoder_scoring)
    pipe_scores = benchmark.pipe_score(pipe_scoring)

    benchmark.dump(backup_file)
    encoder_scores.to_csv(encoder_result_file, index=False)
    pipe_scores.to_csv(pipe_result_file, index=False)

    print(f"Results written to {encoder_result_file} and {pipe_result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    stat_parser = subparsers.add_parser("stat")
    stat_parser.add_argument("--n-samplings", default=100, type=int)
    stat_parser.add_argument("--seed", default=20210902, type=int)
    stat_parser.add_argument("--table", default="default")
    stat_parser.set_defaults(func=run_stat_benchmark)

    ml_parser = subparsers.add_parser("ml")
    ml_parser.add_argument("benchmark")
    ml_parser.add_argument("--table", default="default")
    ml_parser.set_defaults(func=run_ml_benchmark)

    args = parser.parse_args()
    args.func(args)
