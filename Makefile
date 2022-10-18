install:
	pip install -e ".[dev, smac]"
	pre-commit install
	pre-commit install --hook-type commit-msg
