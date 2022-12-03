install:
	pip install -e ".[dev, smac]"
	pre-commit install
	pre-commit install --hook-type commit-msg

fix:
	black --quiet byop tests
	pycln --quiet byop tests
	ruff --silent --exit-zero --no-cache --fix byop tests

check:
	pre-commit run --all-files
