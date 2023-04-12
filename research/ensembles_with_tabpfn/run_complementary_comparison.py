from research.ensembles_with_tabpfn.complementary_code.data_handler import read_all_base_models
from research.ensembles_with_tabpfn.utils.config import ALGO_NAMES, METRIC_MAP
from research.ensembles_with_tabpfn.complementary_code.ensembling_performance_boost import \
    get_data_for_performance_increase_with_new_model
from research.ensembles_with_tabpfn.complementary_code.complementary_analysis.correlation_analysis import \
    correlation_analysis
from research.ensembles_with_tabpfn.complementary_code.complementary_analysis.ensemble_diversity_analysis import \
    ensemble_diversity_analysis

from byop.ensembling.ensemble_preprocessing import prune_base_models
from byop.store import PathBucket

import logging

LEVEL = logging.INFO
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(LEVEL)


def _run():
    path_to_base_model_data = "./data_space/base_model_data"
    path_to_analysis_data = "./data_space/analysis_data"
    bm_bucket_name = "debug"
    metric_name = "balanced_accuracy"
    metric_data = METRIC_MAP[metric_name]
    seed = 41124

    # TODO: do this for all models to have a baseline / comparison or only do this for TabPFN?
    complement_model_name = "XT"  # which model to analysis as complementary

    # -- Get all base models
    base_models, X_train, X_test, y_train, y_test = read_all_base_models(path_to_base_model_data, bm_bucket_name,
                                                                         ALGO_NAMES)

    # - Get the best model per algorithm (based on validation score)
    base_models = prune_base_models(base_models, max_number_base_models=len(ALGO_NAMES), pruning_method="SiloTopN",
                                    maximize_validation_score=metric_data["maximize"])

    # - Split
    complement_model = [bm for bm in base_models if bm.config["algorithm"] == complement_model_name][0]
    base_models = [bm for bm in base_models if bm.config["algorithm"] != complement_model_name]

    # -- Diversity analysis
    bm_diversity, ens_bm_with_complement_diversity = ensemble_diversity_analysis(y_train, y_test, base_models,
                                                                                 complement_model)

    # -- Performance analysis
    # As we are using cross-validation, X_val = X_train and y_val = y_train
    ens_bm_predictions, ens_bm_with_complement_predictions, score_bm, score_bm_with_complement = \
        get_data_for_performance_increase_with_new_model(base_models, complement_model,
                                                         metric_data["loss_function_proba_in"],
                                                         metric_data["function_proba_in"],
                                                         seed, X_train, X_test, y_train, y_test)

    logger.info(
        f"Score of Ensemble - base models: {score_bm} | Score of Ensemble - base models with complement: {score_bm_with_complement}")

    # -- Correlation analysis
    ens_prediction_correlation, all_base_models_correlation_df = \
        correlation_analysis(y_train, y_test, ens_bm_predictions, ens_bm_with_complement_predictions, base_models,
                             complement_model)

    logger.info(f"Correlation of ensemble predictions: {ens_prediction_correlation}")
    logger.info(f"Correlation of all base models predictions: {all_base_models_correlation_df}")

    # -- Save results for next step (analysis)
    result_bucket = PathBucket(path_to_analysis_data + f"/{bm_bucket_name}")
    result_stats = {
        "test_score_standard_base_models": score_bm,
        "test_score_standard_base_models_with_complement": score_bm_with_complement,
        "correlation_ensemble_predictions": ens_prediction_correlation,
        "sample_of_diversity_base_models": bm_diversity,
        "sample_of_diversity_base_models_with_complement": ens_bm_with_complement_diversity
    }

    result_bucket.update(
        {
            "results_stats.json": result_stats,
            "correlation_matrix.csv": all_base_models_correlation_df

        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # TODO:
    #   - make agnostic to dataset
    _run()
