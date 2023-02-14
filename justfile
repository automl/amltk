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
  cz bump || exit
  git push origin "v$(cz version --project)"
