import logging
import warnings

import numpy as np
import pandas as pd
from config import RESULT_DIR, SEED, VERBOSE
from sklearn.model_selection import ParameterGrid


def run_benchmark(*, benchmark_name, task, param_grid, n_runs, csv_file):
    csv_file = RESULT_DIR / f"{benchmark_name}.csv"
    grid = ParameterGrid(param_grid)
    results = pd.read_csv(csv_file) if csv_file.is_file() else pd.DataFrame()

    for run_id in range(n_runs):
        for param_id, params in enumerate(grid):
            head_msg = f"run #{run_id} ({param_id + 1}/{len(grid)})"
            body_msg = "; ".join(f"{key}={val}" for key, val in params)
            msg = " ".join([head_msg, "--", body_msg])

            params |= {"entropy": (SEED, run_id)}
            mask = [results[key] == val for key, val in params.items()]
            if np.prod(mask, axis=0).any():
                logging.info("CACHED " + msg)
                continue
            logging.info("START " + msg)
            with warnings.catch_warnings():
                if not VERBOSE:
                    warnings.simplefilter("ignore")
                task_results = params | task(**params)
            results.append(task_results, ignore_index=True)
            results.to_csv(csv_file, index=False)
            logging.info("END " + head_msg)
