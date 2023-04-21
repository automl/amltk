# List all possible commands
help:
  just --list

# List all possible commands
list:
  just --list

# Install all required and development dependancies
install:
  pip install -e ".[dev, test, doc, smac, optuna]"
  pre-commit install
  pre-commit install --hook-type commit-msg

# Run formatters and linters to fix up code
fix:
  black --quiet src tests
  ruff --silent --exit-zero --no-cache --fix src tests

# Run pre-commit to check all files
check:
  pre-commit run --all-files

# Run mypy over the source code to find typing errors
check-types:
  mypy src

# Launch the docs server locally and open the webpage
docs exec_doc_code="true" example="None" offline="false":
  python -m webbrowser -t "http://127.0.0.1:8000/"
  AMLTK_DOC_RENDER_EXAMPLES={{example}} \
    AMLTK_DOCS_OFFLINNE={{offline}} \
    AMLTK_EXEC_DOCS={{exec_doc_code}} mkdocs serve --watch-theme
  # https://github.com/pawamoy/markdown-exec/issues/19

# Bump the version and generate the changelog based off commit messages
bump:
  cz bump || exit
  git push
  git push origin "v$(cz version --project)"

# Publish the repo to pypi
publish:
  echo "TODO"

# Create a `feat` PR with <name>
pr-feat name:
  git pull origin main
  git checkout -b feat-{{name}} main
  git push --set-upstream origin feat-{{name}}

# Create a `doc` PR with <name>
pr-doc name:
  git pull origin main
  git checkout -b doc-{{name}} main
  git push --set-upstream origin doc-{{name}}

# Create a `fix` PR with <name>
pr-fix name:
  git pull origin main
  git checkout -b fix-{{name}} main
  git push --set-upstream origin fix-{{name}}

# Create a `chore` PR with <name>
pr-chore name:
  git pull origin main
  git checkout -b chore-{{name}} main
  git push --set-upstream origin chore-{{name}}

# Create an `other` PR with <name>
pr-other name:
  git pull origin main
  git checkout -b other-{{name}} main
  git push --set-upstream origin other-{{name}}

# Run all tests but stop on the first failure
test:
  pytest -x -m "not example"

# Run all tests, stopping on the first failure and continuing from the last failure and skipping examples
test-iter:
  pytest -x --lf -m "not example"

test-examples:
  pytest "tests/test_examples.py" -x --lf
