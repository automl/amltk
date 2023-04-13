from byop.store import PathBucket

from research.ensembles_with_tabpfn.utils.config import ALGO_NAMES, METRICS, EXPERIMENT_RUNS_WO_ALGOS, C_MODEL, \
    DATASET_REF, FOLDS, SAMPLES
from itertools import product

import logging

LEVEL = logging.INFO
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(LEVEL)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json

from statistics import mean

# Global font size update
plt.rcParams['font.size'] = '16'


def _corr_matrix_plot(corr_matrix, complement_model_name):
    # re-order to have complement model last
    col_order = list(corr_matrix)
    col_order.remove(complement_model_name)
    corr_matrix = corr_matrix[col_order + [complement_model_name]]

    # Make it not squared such that bottom row and first column are gone.
    masked_corr = corr_matrix.loc[~np.all(corr_matrix == -1, axis=1), ~np.all(corr_matrix == -1, axis=0)]

    # -- Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(masked_corr, annot=True, mask=masked_corr == -1)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()

    # -- Make complement thing stand out somehow
    highlight_x_ticks = [x for x in ax.get_xticklabels() if x.get_text() == complement_model_name][0]
    x_cord = highlight_x_ticks._x
    # set the properties of the ticklabel
    highlight_x_ticks.set_weight("bold")
    highlight_x_ticks.set_size(20)
    highlight_x_ticks.set_color("green")

    for annot in ax.texts:
        if annot._x == x_cord:
            # set the properties of the heatmap annot
            annot.set_weight("bold")
            annot.set_color("green")
            annot.set_size(20)

    plt.show()


def _run(c_model, metric_name, dataset_ref, data_sample_names):
    path_to_analysis_data = f"./data_space/analysis_data/{metric_name}/{dataset_ref}"

    result_stats_list = []
    corr_df_list = []

    for data_sample_name in data_sample_names:
        result_bucket = PathBucket(path_to_analysis_data + f"/{data_sample_name}")
        result_stats_list.append(result_bucket["results_stats.json"].load())
        corr_df_list.append(result_bucket["correlation_matrix.csv"].load())

        # TODO: compute & save individual sample results here?

    overall_result_bucket = PathBucket(path_to_analysis_data)
    disp_res = overall_result_bucket["disparity_results.json"].load()

    # -- Compute results over splits
    avg_corr_df = None
    sanity = None
    for df in corr_df_list:
        if avg_corr_df is None:
            sanity = [list(df.index), list(df.columns)]
            avg_corr_df = df
        else:
            assert list(df.index) == sanity[0]
            assert list(df.columns) == sanity[1]
            avg_corr_df += df
    avg_corr_df /= len(corr_df_list)
    avg_result_stats = {k: mean([ele[k] for ele in result_stats_list]) for k in result_stats_list[0].keys()}

    # -- Analysis
    _corr_matrix_plot(avg_corr_df, c_model)
    print(json.dumps(avg_result_stats, sort_keys=True, indent=4))
    print(json.dumps(disp_res, sort_keys=True, indent=4))


def _run_wrapper():
    # TODO:
    #   - decide on how to compute this (SLURM OR LOCAL GIVEN DATA)
    logging.basicConfig(level=logging.INFO)

    for metric_name in METRICS:
        for dataset_ref in DATASET_REF:
            logger.info(f"Plot for {C_MODEL} analysis for {metric_name} on dataset {dataset_ref}")
            _run("LM", metric_name, dataset_ref,
                 [f"f{fold_i}_s{sample_i}" for fold_i, sample_i in product(FOLDS, SAMPLES)])


if __name__ == "__main__":  # MP safeguard
    _run_wrapper()
