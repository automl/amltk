"""Database Models Module.

This module defines Peewee models for the database tables used in the storage of trial
reports and associated metrics. The models include `ExperimentModel` for experiments,
`TrialReportModel` for trial reports, and `ReportMetricModel` for metric values.
"""

from __future__ import annotations

from peewee import (
    CharField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# Connect to the SQLite database
db = SqliteDatabase("experiment_results.db")


class ExperimentModel(Model):
    """Model representing an experiment.

    Attributes:
        name (CharField): Name of the experiment (unique).
    """

    name = CharField(unique=True)

    class Meta:
        """Meta class for setting database connection."""
        database = db


class TrialReportModel(Model):
    """Model representing a trial report.

    Attributes:
        name (CharField): Name of the trial report (unique).
        status (CharField): Status of the trial report.
        trial_seed (IntegerField): Seed used for the trial.
        exception (TextField): Exception information if any (nullable).
        traceback (TextField): Traceback information if any (nullable).
        bucket_path (TextField): Path to the bucket associated with the report.
        experiment (ForeignKeyField): Reference to the associated experiment.
    """

    name = CharField(unique=True)
    status = CharField()
    trial_seed = IntegerField()
    exception = TextField(null=True)
    traceback = TextField(null=True)
    bucket_path = TextField()
    experiment = ForeignKeyField(ExperimentModel, backref="reports")

    class Meta:
        """Meta class for setting database connection."""
        database = db


class ReportMetricModel(Model):
    """Model representing a metric associated with a trial report.

    Attributes:
        report (ForeignKeyField): Foreign key reference to the associated trial report.
        metric_name (CharField): Name of the metric.
        value (FloatField): Value of the metric.
    """

    report = ForeignKeyField(TrialReportModel, backref="metrics")
    metric_name = CharField()
    value = FloatField()

    class Meta:
        """Meta class for setting database connection."""
        database = db


# Connect to the database and create tables
db.connect()
db.create_tables([ExperimentModel, TrialReportModel, ReportMetricModel])
