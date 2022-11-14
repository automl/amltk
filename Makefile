install:
	pip install -e ".[dev, smac]"
	pre-commit install
	pre-commit install --hook-type commit-msg

format:
	black byop
	black tests
	isort byop
	isort tests
	pycln byop
	pycln tests

check:
	pylint byop
	pylint tests
	mypy byop
	pydocstyle byop
