install:
  pip install -e ".[dev, test, doc, smac, optuna]"
  pre-commit install
  pre-commit install --hook-type commit-msg

fix:
  black --quiet src tests
  ruff --silent --exit-zero --no-cache --fix src tests

check:
  pre-commit run --all-files

check-types:
  mypy src

docs:
  python -m webbrowser -t "http://127.0.0.1:8000/"
  while true; do mkdocs serve --watch-theme; done

bump:
  cz bump || exit
  git push
  git push origin "v$(cz version --project)"

publish:
  echo "TODO"

pr-feat name:
  git pull origin main
  git checkout -b feat-{{name}} main
  git push --set-upstream origin feat-{{name}}

pr-doc name:
  git pull origin main
  git checkout -b doc-{{name}} main
  git push --set-upstream origin doc-{{name}}

pr-fix name:
  git pull origin main
  git checkout -b fix-{{name}} main
  git push --set-upstream origin fix-{{name}}

pr-other name:
  git pull origin main
  git checkout -b other-{{name}} main
  git push --set-upstream origin other-{{name}}
test:
  pytest -x --lf
