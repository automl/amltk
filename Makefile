.PHONY: install fix check docs bump

install:
	pip install -e ".[dev, smac]"
	pre-commit install
	pre-commit install --hook-type commit-msg

fix:
	black --quiet src tests
	ruff --silent --exit-zero --no-cache --fix src tests

check:
	pre-commit run --all-files

docs:
	mkdocs serve --watch-theme

bump:
	cz bump --dry-run || echo ""
	@echo ""
	@echo "Use 'cz bump' to activate these changes"
