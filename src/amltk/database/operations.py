"""Database Operations.

This module provides functionality for saving trial reports and associated metrics
in a database. It includes operations to connect to the database, save experiment
information, trial reports, and associated metrics. The module defines models for
the database tables used in the storage.

Usage Example:

```python
        from amltk.database.operations import save_report_in_database
        from amltk.optimization import Trial, Metric

        loss_metric = Metric("loss", minimize=True)
        trial = Trial(name="trial", config={"x": 1}, metrics=[loss_metric])

        with trial.begin():
            # Do some work
            report = trial.success(loss=1)

        print(report)

        save_report_in_database(report)
 ```
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from uuid import uuid4

from amltk import Trial
from amltk.database.models import (
    ExperimentModel,
    ReportMetricModel,
    TrialReportModel,
    db,
)


def save_report_in_database(
        report: Trial.Report | Iterable[Trial.Report],
        experiment_name: str | None = None,
) -> None:
    """Save trial reports and associated metrics in the database.

    This function connects to the database, saves the experiment information,
    trial reports, and associated metrics. It then closes the database connection.

    Args:
        report (Trial.Report | Iterable[Trial.Report]): A single trial report
            or an iterable of trial reports to be saved in the database.
        experiment_name (str, optional): Name of the experiment. If not provided,
            a unique identifier will be generated. Defaults to None.

    Returns:
        None
    """
    # Check if the database connection is not open
    if db.is_closed():
        db.connect()

    if experiment_name:
        experiment = ExperimentModel.get_or_create(name=experiment_name)[0]
    else:
        experiment_name = f"{uuid4()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment = ExperimentModel.create(name=experiment_name)

    if isinstance(report, Trial.Report):
        # Handle a single report
        save_single_report(report, experiment)
    elif isinstance(report, Iterable):
        # Handle an iterable of reports
        for single_report in report:
            save_single_report(single_report, experiment)

    # Close the database connection
    db.close()


def save_single_report(
        report: Trial.Report,
        experiment: ExperimentModel,
) -> None:
    """Save a single trial report in the database.

    Args:
        report (Trial.Report): The trial report to be saved.
        experiment (ExperimentModel): The associated experiment.

    Returns:
        None
    """
    # Create a TrialReportModel entry
    trial_report = TrialReportModel.create(
        name=report.name,
        status=str(report.status),
        trial_seed=report.trial.seed if report.trial.seed else None,
        exception=str(report.exception) if report.exception else None,
        traceback=str(report.traceback) if report.traceback else None,
        bucket_path=str(report.bucket.path),
        experiment=experiment,
        metric_accuracy=report.metrics.get("accuracy"),
    )

    # Create ReportMetricModel entries for each metric value
    for metric_value in report.metric_values:
        ReportMetricModel.create(
            report=trial_report,
            metric_name=metric_value.metric.name,
            value=metric_value.value,
        )
