from __future__ import annotations

import logging

import numpy as np

from byop.optimization import Trial
from byop.pipeline import Pipeline
from byop.smac import SMACOptimizer
from byop.store import PathBucket

LEVEL = logging.INFO
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(LEVEL)

from research.ensembles_with_tabpfn.base_model_code.experiment_pipeline_builder import build_pipeline
from research.ensembles_with_tabpfn.base_model_code.data_handler import setup_data_bucket
from research.ensembles_with_tabpfn.base_model_code.validation_procedure import predict_fit_repeated_cross_validation
from research.ensembles_with_tabpfn.utils.config import ALL_EXPERIMENT_RUNS

from research.ensembles_with_tabpfn.utils.config import METRIC_MAP


def target_function(trial: Trial, /, bucket: PathBucket, pipeline: Pipeline, metric_data: dict) -> Trial.Report:
    X_train, X_test, y_train, y_test = (
        bucket["X_train.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_test.npy"].load(),
    )

    pipeline = pipeline.configure(trial.config)
    sklearn_pipeline = pipeline.build()

    # Begin the trial, the context block makes sure
    with trial.begin():
        train_score, val_score, test_score, val_probabilities, val_predictions, test_probabilities, test_predictions \
            = predict_fit_repeated_cross_validation(2, 5, sklearn_pipeline,
                                                    X_train, y_train,
                                                    X_test, y_test,
                                                    metric_data)

    if trial.exception:
        return trial.fail(cost=np.inf)

    # Save all of this to the file system
    scores = {
        f"train_{metric_data['name']}": train_score,
        f"validation_{metric_data['name']}": val_score,
        f"test_{metric_data['name']}": test_score
    }

    bucket.update(
        {
            f"trial_{trial.name}_config.json": dict(trial.config),
            f"trial_{trial.name}_scores.json": scores,
            # TODO, if we want to have a saved model, we would need to create a bagged model of sklearn_pipelines
            #   for n-repeated k-fold cross-validation. As we do not need it (IMO), we can skip it for now.
            # f"trial_{trial.name}.pkl": sklearn_pipeline,
            f"trial_{trial.name}_val_predictions.npy": val_predictions,
            f"trial_{trial.name}_val_probabilities.npy": val_probabilities,
            f"trial_{trial.name}_test_predictions.npy": test_predictions,
            f"trial_{trial.name}_test_probabilities.npy": test_probabilities
        }
    )
    val_accuracy = scores[f"validation_{metric_data['name']}"]
    return trial.success(cost=metric_data["to_loss_function"](val_accuracy))


def _run(algorithm_name, metric_name, data_sample_name, dataset_ref):
    # TODO: decide on correct random state management for experiments
    seed = 42
    # TODO: decide on parallelization levels (at algorithm level, at validation level, at optimizer level?)
    # n_workers = 4

    # -- Setup Data and Optimizer
    metric_data = METRIC_MAP[metric_name]
    data_bucket = setup_data_bucket(dataset_ref, data_sample_name, seed,
                                    f"./data_space/base_model_data/{algorithm_name}/{dataset_ref}/{data_sample_name}")
    pipeline = build_pipeline(algorithm_name)
    optimizer = SMACOptimizer.HPO(space=pipeline.space(), seed=seed)
    objective = Trial.Objective(target_function, bucket=data_bucket, pipeline=pipeline, metric_data=metric_data)

    # -- ASK/TELL USAGE FOR DEBUGGING AND DEPENDING ON PARALLELIZATION APPROACH FOREVER
    logger.info("Start optimization ...")
    for _ in range(10):
        # TODO: figure out if this produces duplicates and returns an error if everything has been already evaluated
        # TODO: define this based on time or number of samples (samples must be better right...?)
        trial = optimizer.ask()
        report = objective(trial)

        # raise and crash if error
        if isinstance(report, Trial.FailReport):
            raise report.exception

        optimizer.tell(report)
        logger.info(report)

    logger.info("...finished!")


def _run_wrapper():
    # TODO:
    #   - transform into a SLURM executable script that takes as cli arguments:
    #       - algorithm name; dataset/task id; ...
    logging.basicConfig(level=logging.INFO)

    for algorithm_name, metric_name, fold_i, sample_i, dataset_ref in ALL_EXPERIMENT_RUNS:
        logger.info(f"Start {algorithm_name} for {metric_name} on dataset {dataset_ref} (f{fold_i}_f{sample_i})")
        _run(algorithm_name, metric_name, f"f{fold_i}_s{sample_i}", dataset_ref)


if __name__ == "__main__":  # MP safeguard
    _run_wrapper()
