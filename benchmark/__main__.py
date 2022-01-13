import argparse
import logging

from ._ml import ml_task
from ._run import run_benchmark
from ._stat import inference_task

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="benchmark_name", required=True)
stat_parser = subparsers.add_parser("stat")
ml_parser = subparsers.add_parser("ml")
parser.add_argument("--n-runs", type=int)
stat_parser.add_argument("--beta-params", action="extend", type=tuple)
stat_parser.add_argument("--n-categories", action="extend", type=int)
stat_parser.add_argument("--n-obsevations", action="extend", type=int)
ml_parser.add_argument("--dataset", action="extend")
ml_parser.add_argument("--encoders", action="extend")
ml_parser.add_argument("--classifiers", action="extend")
stat_parser.set_defaults(task=inference_task)
ml_parser.set_defaults(task=ml_task)
args = parser.parse_args()
run_benchmark(**vars(args))
