install:
  pip install -e ".[dev, test, doc]"
  pre-commit install
  pre-commit install --hook-type commit-msg

fix:
  black --quiet src tests
  ruff --silent --exit-zero --no-cache --fix src tests

check:
  pre-commit run --all-files

docs:
  python -m webbrowser -t "http://127.0.0.1:8000/"
  while true; do mkdocs serve --watch-theme; done

bump:
  cz bump || exit
  git push
  git push origin "v$(cz version --project)"
