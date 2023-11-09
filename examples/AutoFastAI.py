"""Automated FastAI example for tabular datasets
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn] fastai`
## Imports
"""

from __future__ import annotations

from typing import Any
from asyncio import Future
from pathlib import Path
import shutil

import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from amltk.smac import SMACOptimizer
from amltk.optimization import History, Trial
from amltk.pipeline import Pipeline, choice , group, split, step
from amltk.scheduling import Scheduler, Task
from amltk.sklearn.data import split_data
from amltk.store import PathBucket


from fastai.data.block import CategoryBlock
from fastai.tabular.core import TabularPandas
from fastai.metrics import accuracy
from fastai.tabular.learner import tabular_learner
from fastai.tabular.model import tabular_config
import torch
import pandas as pd

"""
## Neural Network Classifier
Below is the FastAI implementation of tabular model class and a wrapper class to encapsulate the FastAI with the Sklearn style, 
we could also use a custom pipeline, see the [Pipeline](site:guides/pipeline.md#pipeline-building).
"""

class FastAI_Classifier:
    def __init__(self, 
        learning_rate = 	0.001,
        weight_decay = None,
        weight_decay_bn_bias = None,
        momentums = (0.95, 0.85, 0.95),
        dropout_prob = None,  
        embeding_layer_dropout_prob = 0.0, 
        use_batch_norm = True, 
        batch_norm_final = False , 
        batch_norm_cont = True, 
        activation_func = 'ReLU', 
        lin_first = True):
            
        self.learner = None
        self.lr = learning_rate
        self.wd = weight_decay
        self.wd_bn_bias =weight_decay_bn_bias
        self.moms = momentums

        act_cls = {'ReLU':torch.nn.ReLU(),'GELU':torch.nn.GELU()}
        self.config = tabular_config(
            ps=dropout_prob,
            embed_p=embeding_layer_dropout_prob,
            use_bn=use_batch_norm, bn_final = batch_norm_final,
            bn_cont=batch_norm_cont, 
            lin_first=lin_first,
            act_cls=act_cls[activation_func]) #, 
        
        self.metrics =accuracy
        self.epochs = 10

    def fit(self, X, y):
        df = pd.DataFrame(X,columns=[f'feat_{i}' for i in range(X.shape[1])])
        feature_list = list(df.columns)
        df['target'] = y
        dls =  TabularPandas(
                            df, y_names = 'target',
                            cont_names=feature_list,
                            y_block = CategoryBlock()).dataloaders()
        self.learner = tabular_learner(
            dls,
            metrics=self.metrics,
            config=self.config,
            lr = self.lr,
            wd = self.wd,
            wd_bn_bias = self.wd_bn_bias,
            moms = self.moms
        )
        with  self.learner.no_logging():
             self.learner.fit(self.epochs)       
        return self.learner
        
    def predict(self, X):
        y_preds = self.predict_proba(X)
        return np.argmax(y_preds, axis = 1)
    
    def predict_proba(self, X):
        df_test = pd.DataFrame(X,columns=[f'feat_{i}' for i in range(X.shape[1])])
        with  self.learner.no_logging():
            test_dl_l = self.learner.dls.test_dl(df_test)
            y_preds= self.learner.get_preds(dl=test_dl_l)
        y_preds = y_preds[0].numpy() # isolate predictions
        return y_preds

"""
## Dataset

Below is just a small function to help us get the dataset from OpenML and encode the
labels.
"""

def get_dataset(
    dataset_id: str | int,
    *,
    seed: int,
    splits: dict[str, float],
) -> dict[str, Any]:
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_features_meta_data=False,
        download_qualities=False,
    )

    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)

    return split_data(X, _y, splits=splits, seed=seed)  # type: ignore


"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
the `RandomForestClassifier`.

For more on definitions of pipelines, see the [Pipeline](site:guides/pipeline.md)
guide.
"""
categorical_imputer = step(
    "categoricals",
    SimpleImputer,
    config={
        "strategy": "constant",
        "fill_value": "missing",
    },
)
one_hot_encoding = step("ohe", OneHotEncoder, config={"drop": "first"})

numerical_imputer = step(
    "numerics",
    SimpleImputer,
    space={"strategy": ["mean", "median"]},
)

pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        group(  # <!> (3)!
            "categoricals",
            step(
                "category_imputer",
                SimpleImputer,
                space={
                    "strategy": ["most_frequent", "constant"],
                    "fill_value": ["missing"],
                },
            )
            | step(
                "ohe",
                OneHotEncoder,
                space={
                    "min_frequency": (0.01, 0.1),
                    "handle_unknown": ["ignore", "infrequent_if_exist"],
                },
                config={"drop": "first"},
            ),
        ),
        group(  # <!> (2)!
            "numerics",
            step(
                "numerical_imputer",
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            )
            | step(
                "variance_threshold",
                VarianceThreshold,
                space={"threshold": (0.0, 0.2)},
            )
            | choice(
                "scaler",
                step("standard", StandardScaler),
                step("minmax", MinMaxScaler),
                step("robust", RobustScaler),
                step("passthrough", FunctionTransformer),
            ),
        ),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    ),
    step(
         "classifier",
         FastAI_Classifier,
         config={"momentums": (0.95, 0.85, 0.95)},
         space={
            "learning_rate": (0.0001, 0.01),
            "weight_decay": (0.0, 0.1),
            "weight_decay_bn_bias": [True,False],
            "dropout_prob": (0.2,0.6),
            "embeding_layer_dropout_prob": (0.2,0.6),
            "use_batch_norm": [True,False],
            "batch_norm_final": [True,False],
            "batch_norm_cont" :[True,False],
            "activation_func" :['ReLU', 'GELU'],
            "lin_first":[True,False],
         })
)

print(pipeline)
print(pipeline.space())

"""
## Target Function
The function we will optimize must take in a `Trial` and return a `Trial.Report`.
We also pass in a [`PathBucket`][amltk.store.Bucket] which is a dict-like view of the
file system, where we have our dataset stored.

We also pass in our [`Pipeline`][amltk.pipeline.Pipeline] representation of our
pipeline, which we will use to build our sklearn pipeline with a specific
`trial.config` suggested by the [`Optimizer`][amltk.optimization.Optimizer].
"""

def target_function(
    trial: Trial,
    /,
    bucket: PathBucket,
    _pipeline: Pipeline,
) -> Trial.Report:
    # Load in data
    X_train, X_val, X_test, y_train, y_val, y_test = (
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
        bucket["y_test.npy"].load(),
    )

    # Configure the pipeline with the trial config before building it.
    configured_pipeline = _pipeline.configure(trial.config)
    sklearn_pipeline = configured_pipeline.build()

    # Fit the pipeline, indicating when you want to start the trial timing and error
    # catchnig.
    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

    # If an exception happened, we use `trial.fail` to indicate that the
    # trial failed
    if trial.exception:
        trial.store(
            {
                "exception.txt": f"{trial.exception}\n traceback: {trial.traceback}",
                "config.json": dict(trial.config),
            },
            where=bucket,
        )
        return trial.fail(cost=np.inf)

    # Make our predictions with the model
    train_predictions = sklearn_pipeline.predict(X_train)
    val_predictions = sklearn_pipeline.predict(X_val)
    test_predictions = sklearn_pipeline.predict(X_test)

    val_probabilites = sklearn_pipeline.predict_proba(X_val)

    # Save the scores to the summary of the trial
    val_accuracy = accuracy_score(val_predictions, y_val)
    trial.summary.update(
        {
            "train/acc": accuracy_score(train_predictions, y_train),
            "val/acc": val_accuracy,
            "test/acc": accuracy_score(test_predictions, y_test),
        },
    )

    # Save all of this to the file system
    trial.store(
        {
            "config.json": dict(trial.config),
            "scores.json": trial.summary,
            "model.pkl": sklearn_pipeline,
            "val_probabilities.npy": val_probabilites,
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
        where=bucket,
    )

    # Finally report the success
    return trial.success(cost=1 - val_accuracy)


"""
### Getting and storing data
We use a [`PathBucket`][amltk.store.PathBucket] to store the data. This is a dict-like
view of the file system.
"""
seed = 42
data = get_dataset(31, seed=seed, splits={"train": 0.6, "val": 0.2, "test": 0.2})

X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]

path = Path("result/AutoFastAI")
if path.exists():
    shutil.rmtree(path)

bucket = PathBucket(path, clean=True, create=True)
bucket.store(
    {
        "X_train.csv": X_train,
        "X_val.csv": X_val,
        "X_test.csv": X_test,
        "y_train.npy": y_train,
        "y_val.npy": y_val,
        "y_test.npy": y_test,
    },
)

"""
### Setting up the Scheduler, Task and Optimizer

Please check out the full [guides](site:guides/index.md) to learn more!

We then create an [`SMACOptimizer`][amltk.smac.SMACOptimizer] which will
optimize the pipeline. We pass in the space of the pipeline, which is the space of
the hyperparameters we want to optimize.
"""
scheduler = Scheduler.with_processes(4)
optimizer = SMACOptimizer.create(space=pipeline.space(), seed=seed)

"""
Next we create a [`Task`][amltk.Task], passing in the function we
want to run and the scheduler we will run it in.
"""
task = Task(target_function, scheduler)

print(task)
"""
We use the callback decorators of the [`Scheduler`][amltk.scheduling.Scheduler] and
the [`Task`][amltk.Task] to add callbacks that get called
during events that happen during the running of the scheduler. Using this, we can
control the flow of how things run.
Check out the [task guide](site:guides/tasks.md) for more.
"""

@scheduler.on_start
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task.submit(trial, bucket=bucket, _pipeline=pipeline)


@task.on_result
def tell_optimizer(future: Future, report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)
    

trial_history = History()


@task.on_result
def add_to_history(future: Future, report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)


@task.on_result
def launch_another_task(*_: Any) -> None:
    """When we get a report, evaluate another trial."""
    trial = optimizer.ask()
    task.submit(trial, bucket=bucket, _pipeline=pipeline)

"""
If something goes wrong, we likely want to stop the scheduler.
"""
@task.on_exception
def stop_scheduler_on_exception(*_: Any) -> None:
    scheduler.stop()

@task.on_cancelled
def stop_scheduler_on_cancelled(_: Any) -> None:
    scheduler.stop()


"""
## Running the Whole Thing

### Setting the system to run

Lastly we use [`Scheduler.run`][amltk.scheduling.Scheduler.run] to run the
scheduler. We pass in a timeout of 20 seconds.
"""
scheduler.run(timeout=20)

print("Trial history:")
print(trial_history)
history_df = trial_history.df()
print(history_df)
