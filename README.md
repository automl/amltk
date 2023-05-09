# AutoML Toolkit
A set of re-usable components for building complex
configurable pipelines, agnostic to:
* Search space implementation
* How to build your pipeline
* Where compute happens

... yet providing sensible defaults and options to plug in your own.

## Installation
```bash
git clone git@github.com:automl/byop.git

# If using `just`
just install-dev

# otherwise, for everything
pip install -e ".[dev, test, doc, doc-examples]"
```

## Docs
This library uses [`mkdocs`](https://squidfunk.github.io/mkdocs-material/getting-started/) for markdown style documentation.
```bash
just docs

# Otherwise
python -m webbrowser -t "http://127.0.0.1:8000/" \
    AMLTK_DOC_RENDER_EXAMPLES={{example}} \
    AMLTK_DOCS_OFFLINNE={{offline}} \
    AMLTK_EXEC_DOCS={{exec_doc_code}} \
    mkdocs serve --watch-theme
```
