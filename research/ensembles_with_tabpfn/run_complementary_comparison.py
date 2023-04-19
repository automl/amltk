from research.ensembles_with_tabpfn.complementary_code.data_handler import read_all_base_models
from research.ensembles_with_tabpfn.utils.config import ALGO_NAMES, METRIC_MAP, EXPERIMENT_RUNS_WO_ALGOS, C_ALGO, \
    EXPERIMENT_HIGH_LEVEL, init_metric_data
from research.ensembles_with_tabpfn.complementary_code.ensembling_performance_boost import \
    get_data_for_performance_increase_with_new_model
from research.ensembles_with_tabpfn.complementary_code.complementary_analysis.correlation_analysis import \
    correlation_analysis
from research.ensembles_with_tabpfn.complementary_code.complementary_analysis.ensemble_diversity_analysis import \
    ensemble_diversity_analysis, ensemble_disparity_analysis, compute_left_bergman_centroid

from byop.ensembling.ensemble_preprocessing import prune_base_models
from byop.store import PathBucket

import logging

LEVEL = logging.INFO
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(LEVEL)


def _performance_analysis(base_models, algo_names, complement_model_name, metric_data,
                          seed, X_train, X_test, y_train, y_test):
    # - Get the best model per algorithm (based on validation score)
    # TODO: take best or X best? what is more appropriate for the analysis?
    base_models = prune_base_models(base_models, max_number_base_models=len(algo_names), pruning_method="SiloTopN",
                                    maximize_validation_score=metric_data["maximize"])

    # - Split
    complement_model = [bm for bm in base_models if bm.config["algorithm"] == complement_model_name][0]
    base_models = [bm for bm in base_models if bm.config["algorithm"] != complement_model_name]

    # -- Performance analysis
    # As we are using cross-validation, X_val = X_train and y_val = y_train
    ens_bm_predictions, ens_bm_with_complement_predictions, \
        score_bm, score_bm_with_complement, val_score_bm, val_score_bm_with_complement = \
        get_data_for_performance_increase_with_new_model(base_models, complement_model,
                                                         metric_data["loss_function_proba_in"],
                                                         metric_data["function_proba_in"],
                                                         metric_data["loss_to_score_function"],
                                                         seed, X_train, X_test, y_train, y_test)
    logger.info(
        f"Score of Ensemble - base models: {score_bm} | Score of Ensemble - base models with complement: {score_bm_with_complement}")

    return ens_bm_predictions, ens_bm_with_complement_predictions, \
        score_bm, score_bm_with_complement, val_score_bm, val_score_bm_with_complement


def _run(algo_names, complement_algorithm_name, metric_name, data_sample_name, dataset_ref):
    path_to_base_model_data = f"./data_space/base_model_data/{metric_name}"
    path_to_analysis_data = f"./data_space/analysis_data/{metric_name}"
    metric_data = METRIC_MAP[metric_name]

    # TODO: fix random seed management
    seed = 41124

    # -- Get all base models
    base_models, X_train, X_test, y_train, y_test, meta_data = read_all_base_models(path_to_base_model_data,
                                                                                    dataset_ref, data_sample_name,
                                                                                    algo_names)

    metric_data = init_metric_data(metric_data, meta_data)

    ens_bm_predictions, ens_bm_with_complement_predictions, \
        score_bm, score_bm_with_complement, val_score_bm, val_score_bm_with_complement \
        = _performance_analysis(base_models, algo_names, complement_algorithm_name, metric_data, seed,
                                X_train, X_test, y_train, y_test)

    # -- Pruning for remaining analysis?
    # FIXME: use this if we want to analyze w.r.t. good performing models only
    # base_models = prune_base_models(base_models, max_number_base_models=5, pruning_method="SiloTopN",
    #                                 maximize_validation_score=metric_data["maximize"])

    ens_prediction_correlation, all_base_models_correlation_df, context_predictive_performance_df = \
        correlation_analysis(y_train, y_test, ens_bm_predictions, ens_bm_with_complement_predictions, base_models,
                             complement_algorithm_name, metric_data["function_proba_in"])

    bm_diversity, ens_bm_with_complement_diversity = \
        ensemble_diversity_analysis(y_train, y_test, base_models, complement_algorithm_name)

    # -- Save results for next step (analysis)
    result_bucket = PathBucket(path_to_analysis_data + f"/{dataset_ref}/{data_sample_name}")
    result_stats = {
        "test_score_standard_base_models": score_bm,
        "test_score_standard_base_models_with_complement": score_bm_with_complement,
        "val_score_standard_base_models": val_score_bm,
        "val_score_standard_base_models_with_complement": val_score_bm_with_complement,
        "correlation_ensemble_predictions": ens_prediction_correlation,
        "sample_of_diversity_base_models": bm_diversity,
        "sample_of_diversity_base_models_with_complement": ens_bm_with_complement_diversity
    }

    result_bucket.update(
        {
            "results_stats.json": result_stats,
            "correlation_matrix.csv": all_base_models_correlation_df,
            "context_predictive_performance.csv": context_predictive_performance_df
        }
    )

    # -- Store data for disparity
    for bm_id, bm in enumerate(base_models):
        algo_name = bm.config["algorithm"]
        pred_bucket = PathBucket(path_to_base_model_data + f"/{algo_name}/{dataset_ref}/base_models")
        pred_bucket.update({
            f"{data_sample_name}_pred_{bm_id}.pkl": bm
        })


def _run_disparity_analysis(algo_names, complement_model_name, metric_name, dataset_ref):
    path_to_base_model_data = f"./data_space/base_model_data/{metric_name}"
    path_to_analysis_data = f"./data_space/analysis_data/{metric_name}"

    # -- Get left bergman centroids for each algorithm
    lbc_list = []
    for algo_name in algo_names:
        pred_bucket = PathBucket(path_to_base_model_data + f"/{algo_name}/{dataset_ref}/base_models")
        algo_bms = [algo_bm.load() for algo_bm in pred_bucket.values()]

        left_bergman_centroid = compute_left_bergman_centroid([algo_bm.val_probabilities for algo_bm in algo_bms])
        lbc_list.append((algo_name, left_bergman_centroid))

    # -- split
    complement_model = [lbc for algo_name, lbc in lbc_list if algo_name == complement_model_name][0]
    base_models = [lbc for algo_name, lbc in lbc_list if algo_name != complement_model_name]

    bm_disp, bm_c_disp = ensemble_disparity_analysis(base_models, complement_model)

    result_bucket = PathBucket(path_to_analysis_data + f"/{dataset_ref}")
    result_stats = {
        "disparity_standard_base_models": bm_disp,
        "disparity_standard_base_models_with_complement": bm_c_disp,
    }
    result_bucket.update(
        {
            "disparity_results.json": result_stats,
        }
    )


def _run_wrapper():
    # TODO:
    #   - transform into a SLURM executable script that takes as cli arguments:
    #       - algorithm name; dataset/task id; ...
    logging.basicConfig(level=logging.INFO)

    for metric_name, fold_i, sample_i, dataset_ref in EXPERIMENT_RUNS_WO_ALGOS:
        logger.info(f"\n\nStart {C_ALGO} analysis for {metric_name} on dataset {dataset_ref} (f{fold_i}_s{sample_i})")
        _run(ALGO_NAMES, C_ALGO, metric_name, f"f{fold_i}_s{sample_i}", dataset_ref)

    for metric_name, dataset_ref in EXPERIMENT_HIGH_LEVEL:
        logger.info(f"\n\nStart disparty analysis for {metric_name} on dataset {dataset_ref}")
        _run_disparity_analysis(ALGO_NAMES, C_ALGO, metric_name, dataset_ref)


if __name__ == "__main__":  # MP safeguard
    _run_wrapper()
