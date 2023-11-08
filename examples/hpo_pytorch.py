"""Hyperparameter Optimization for the Pytorch framework
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn] torch`
## Imports
"""

from __future__ import annotations

from asyncio import Future
from pathlib import Path
from typing import Any

import numpy as np
import openml

from amltk.sklearn.data import split_data
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from amltk.optimization import History, Trial
from amltk.pipeline import Pipeline, split, step
from amltk.scheduling import Scheduler, Task

from amltk.smac import SMACOptimizer
from amltk.store import PathBucket

import torch
import os
import shutil


"""
## Neural Network Classifier
Below is the PyTorch implementation of a simple neural network (MLP) class and since we use the Sklearn pipeline,
the following is a wrapper class to encapsulate the neural network with the Sklearn style, 
we could also use a custom pipeline or the Pytorch pipeline, see the [Pipeline](site:guides/pipeline.md#pipeline-building).
"""

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_data: list, dropout_factor):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList()
        for size in layers_data:
            self.hidden_layers.append(torch.nn.Linear(input_size, size))
            self.hidden_layers.append(torch.nn.ReLU())
             # For the next layer
            input_size = size
            self.hidden_layers.append(torch.nn.Dropout(dropout_factor))
        self.output = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()        
    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.sigmoid(self.output(x))
        return x
        
class NN_Classifier:
    def __init__(self, neurons, depth, learning_rate, batch_size, dropout_factor, n_epochs ):
        self.layers_data = [neurons]*depth
        self.dropout_factor = dropout_factor
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCELoss()  # loss function = binary cross entropy
        self.n_epochs = n_epochs   # number of epochs to run
        self.batch_size = batch_size  # size of each batch
        self.model = None
        self.device = self.get_default_device()

    def get_default_device(self): # Pick freer GPU if available, else CPU
        if torch.cuda.is_available():
            res = os.popen('nvidia-smi  --query-gpu=memory.free --format=csv').read()
            memory_free = [int(s) for s in res.split() if s.isdigit()]
            free_gpu_id =  np.argmax(memory_free)
            return torch.device(type='cuda', index=free_gpu_id)
        else:
            return torch.device('cpu')

    def fit(self, X, y):
        input_size = X.shape[1]
        self.model = NeuralNetwork(input_size, self.layers_data, self.dropout_factor).to(self.device)
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate)
        batch_points = torch.arange(0, len(X), self.batch_size)
        for epoch in range(self.n_epochs):
            self.model.train()
            for point in batch_points:
                X_batch = torch.tensor(X[point:point+self.batch_size], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y[point:point+self.batch_size], dtype=torch.float32).reshape(-1, 1).to(self.device)
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        return self.model
        
    def predict(self, X):
        self.model.eval()
        y_pred = self.model(torch.tensor(X, dtype=torch.float32).to(self.device))
        return y_pred.cpu().round().detach().numpy().reshape(-1)

"""
## Dataset

Below is just a small function to help us get the dataset from OpenML and encode the labels.
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
the classifier.

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

feature_preprocessing = split(
    "feature_preprocessing",
    categorical_imputer | one_hot_encoding,
    numerical_imputer,
    item=ColumnTransformer,
    config={
        "categoricals": make_column_selector(dtype_include=object),
        "numerics": make_column_selector(dtype_include=np.number),
    },
)

classifier = step(
        "classifier",
        NN_Classifier,
        space={
            "neurons": 100,
            "depth": [2, 3],
            "batch_size": [32, 64, 128],
            "dropout_factor": (0.2,0.5),
            "learning_rate": (0.001,0.01),
            "n_epochs": 100,
        },
)

pipeline = Pipeline.create(
    feature_preprocessing,
    classifier
)

print(pipeline)
print(pipeline.space())

"""
## Target Function

Next we establish the actual target function we wish to evaluate, that is,
the function we wish to optimize. In this case, we are optimizing the
accuracy of the model on the validation set.

The target function takes a [`Trial`][amltk.optimization.Trial] object, which
has the configuration of the pipeline to evaluate and provides utility
to time, and return the results of the evaluation, whether it be a success
or failure.

We make use of a [`PathBucket`][amltk.store.PathBucket]
to store and load the data, and the `Pipeline` we defined above to
configure the pipeline with the hyperparameters we are optimizing over.

For more details, please check out the [Optimization](site:guides/optimization.md)
guide for more details.
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
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
        where=bucket,
    )

    # Finally report the success
    return trial.success(cost=1 - val_accuracy)



"""
## Running the Whole Thing

### Getting and storing data
We use a [`PathBucket`][amltk.store.PathBucket] to store the data. This is a dict-like
view of the file system.
"""
seed = 42

data = get_dataset(31, seed=seed, splits={"train": 0.6, "val": 0.2, "test": 0.2})

X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]

output_path = Path("result/hpo_pytorch")
if output_path.exists():
    shutil.rmtree(output_path)

bucket = PathBucket(output_path, clean=True, create=True)
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
Setting up the Scheduler, Task and Optimizer
Please check out the full [guides](site:guides/index.md) to learn more!
"""

scheduler = Scheduler.with_processes(32)
optimizer = SMACOptimizer.create(space=pipeline.space(), seed=seed)


"""
Next we create a [`Task`][amltk.Task], passing in the function we
want to run and the scheduler we will run it in.
Check out the [task guide](site:guides/tasks.md) for more.
"""

task = Task(target_function, scheduler)
print(task)

@scheduler.on_start
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task.submit(trial, bucket=bucket, _pipeline=pipeline)


@task.on_result
def tell_optimizer(future: Future, report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)
    

"""
We can use the [`History`][amltk.optimization.History] class to store the reports we get
from the [`Task`][amltk.Task]. We can then use this to analyze the results of the
optimization afterwords.
"""
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

@task.on_exception
def stop_scheduler_on_exception(*_: Any) -> None:
    scheduler.stop() 

@task.on_cancelled
def stop_scheduler_on_cancelled(_: Any) -> None:
    scheduler.stop()

"""
### Setting the system to run

Lastly we use [`Scheduler.run`][amltk.scheduling.Scheduler.run] to run the
scheduler. We pass in a timeout of 120 seconds.
"""
scheduler.run(timeout=120)

print("Trial history:")
history_df = trial_history.df()
print(history_df)
