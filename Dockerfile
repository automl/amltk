FROM python:3.10

RUN pip install -U pip && pip install -U setuptools wheel

COPY pyproject.toml /code/
COPY src /code/
WORKDIR /code/
RUN pip install -e ".[dev,doc,smac]"
RUN pip install openml