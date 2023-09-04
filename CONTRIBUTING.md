# Contributing
Thanks for checking out the contribution page!
`amltk` is open to all open-source contributions, whether it be fixes,
features, integrations or even just simple doc fixes.

This guide will walk through some simple guidelines on the workflow
of `amltk` but also some design principles that are core to its
development. Some of these principles may be new for first time
contributors and examples will be given where necessary.

## Setting up
The core workflows of `amltk` are accessed through the [`justfile`](https://github.com/casey/just)
in the root of the working directory. It is recommended to have this
installed with their [simple one-liners on their repo](https://github.com/casey/just#installation).
All of these were developed with bash in mind and your usage with other platforms
may vary,
please use the `justfile` as reference if this is the case.

#### Forking
If you are contributing from outside the `automl` org and under your own
github profile, you'll have to create your own [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

If you use the [`hub`]() tool from github, you can do this locally with:
```bash
# Clones this repo
hub clone automl/amltk

# Forks the repo to your own user account and sets up tracking
# to match your repo, not the automl/amltk version.
hub fork
```

#### Installation
To install `amltk` for development, we rely on specific dependencies
that are not required for the actual library to run. These are listed
in the `pyproject.toml` under the `[project.optional-dependencies]` header.

You can install all of these by doing the following:
```bash
# Create a virtual environment in your preferred way.
python -m venv .my-virtual-env
source ./.my-virtual-env/bin/activate

# Install all required dependencies
just install
```

For future convenience, see [Easy Virtual Environments](#easy-virtual-environments)

#### Setting up code quality tools
When you ran `just install`, the tool [`pre-commit`](https://pre-commit.com/) was installed.

This is a framework that the repo has set up to run a set of code
quality tools upon each commit, fixing up easy to fix issues, run some
automatic formatting and run a static type checker on the code in the
repository. The configuration for `pre-commit` can be found in
`.pre-commit-config.yaml`.

To run these checks at any time, use the command `just fix`, followed
by `just check`. Any list of errors will be presented to you, and we
recommend fixing these before committing.

While these can certainly be skipped, these checks will be run using
github actions, a Continuous Integration (CI) service. If there are
problems you are not sure how to fix, please feel free to discuss them
in the Pull Request and we will help you solve them!

To see a list of tools used and their purposes,
please see the section on [Code Quality](#code-quality).

#### Creating a new branch
We follow a Pull Request into `main` workflow, which is essentially
that any contributions to `amltk` should be done in a branch with
a pull request to the `main` branch. We prefer a branch name that
describes the kind of pull request that it is. We have provided some
default options but please feel free to use your own if you are familiar
with these workflows:

```bash
# These utilities will pull the most recent `main` branch,
# create a new branch with your `branchname` and and push
# the new branch back to github
just pr-feat branchname  # Creates a branch feat-branchname
just pr-doc branchname   # Creates a branch doc-branchname
just pr-fix branchname   # Creates a branch fix-branchname
just pr-other branchname # Creates a branch other-branchname
```

#### Submitting a PR
If you are unfamiliar with creating a PR on github, please check
out this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

Please provide a short summary of your changes to help give context
to any reviewers who may be looking at your code. If submitting a
more complex PR that changes behaviours,
please consider more context when describing not only what you changed
but why and how.

We may ask you to break up your changes into smaller atomic units that
are easier to verify and review, but we will describe this process to
you if required.

#### Reviews
Once your PR is submitted, we will likely have comments and changes that
are required. Please be patient as we may not be able to respond
immediately. If there are only minor comments, we will simply annotate
your code where these changes are required and upon fixing them, we will
happily merge these into the `main` branch and thank you for your
open-source contributions!

Good practice is to actually review your own PR after submitting it.
You'll often find small issues such as out-of-sync doc strings or even small
logical issues. In general, if you can't understand your own PR, it's likely 
we won't either.

##### Granting access to your fork
If the PR requires larger structural changes or more discussion, there
will likely be a few back-and-forth discussion points which we will
actively respond to help get your contribution in.

If you do not wish to actively monitor the PR for whatever reason,
granting us access to modify your PR will substantially help
integration. To do so, please follow the instructions
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork#enabling-repository-maintainer-permissions-on-existing-pull-requests).

## Commits
This library uses [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
as a way to write commits. This makes commit messages simpler and easier
to read as well as allows for an easier time managing the repo, such as
changelogs and versioning. This is important enough that we even enforce
this through `pre-commit` to fail the commit if the message does not follow
the format. Please follow the link above to find out more but for reference,
here are some short examples:
```git
fix(scheduler): Use X instead of Y
feat(pipeline): Allow for Z
refactor(Optuna): Move integrations to seperate file
doc(Example): Integrating custom space parser
```

## Testing
Our testing for the library is done using [`pytest`](https://docs.pytest.org/),
with some additional utilities from [`pytest-coverage`](https://pytest-cov.readthedocs.io/en/latest/)
for code coverage and [`pytest-cases`](https://smarie.github.io/python-pytest-cases/)
for test structure.

In general, writing a test and running `just test` to test the whole library should be sufficient.
If you need more fine-grained control, such as only testing a particular test, please refer to [this
cheatsheet](https://gist.github.com/kwmiebach/3fd49612ef7a52b5ce3a).

```bash
pytest                              # Test whole library and examples
pytest "tests/path/to/testfile.py"  # Test a particular file
pytest -k "test_name_of_my_test"    # Test a particular test
```

> :warning: In general, you should prefer to run `just test` over `pytest` if new to testing.
This will run all test until it hits its first failure which allows for better incremental testing.
It will also avoid running the examples which are often longer and saved for CI.

### Testing examples
If testing any added examples, please use the `just test-examples` command, which is
a shortcut for `pytest "tests/test_examples.py" -x --lf`. There is unfortunately no way
to sub-select one.

If you are not sure how to test your contribution or need some pointers to get started, please
reach out in the PR comments and we will be happy to assist!

## Code Quality
To ensure a consistent code quality and to reduce noise in the PR,
there are a selection of code quality tools that run.

### Pre-commit - Quality Enforcer
These will be run automatically before a commit can be done with
[`pre-commit`](https://pre-commit.com/). The configuration for this can be found in
`.pre-commit-config.yaml`. All of these can be manually triggered using `just check`.

### Ruff - Code Linting
The primary linter we use is [`ruff`](https://github.com/charliermarsh/ruff), an amazingly fast
python code linter which subsumes many previously used linters like,
`pylint`, `flake8`, `pep8`, and even import sorters like `isort`.
This also includes automatic fixes for many of the smaller problems that occur.
The fixes can be done with `just fix`.

### Mypy - Static Type Checking
This codebase also relies heavily on pythons `typing` and [`mypy`](https://mypy.readthedocs.io/en/stable/)
to ensure correctness across modules. Running this standalone on all files can take some time,
so we don't require you to run this, our automated testers will. If you wish to do so manually,
then use `just check-types`.

If any of the typing concepts are confusing, now is a good chance to learn,
and we would be happy to assist in helping properly type your PR if things do not work. If all else
fails, please feel free to introduce a `# type: ignore` to tell `mypy` to shut up **along with a
description to why it is there**. This will help future contributors and maintainers understand the
reasons behind these ignores. You can find a cheatsheet for basic mypy type hinting [here](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#cheat-sheet-py3).

### Black - Code Formatiing
Lastly, we use [`black`](https://github.com/psf/black) which is a python code formatter. This
will not change any logical meaning of your python code but simply format it in a consistent
manner so that the code is consistent and follows the same standards. This can be run with
`just fix`.

## Git workflow
We follow a _PR into trunk_ development flow (whatever that's meant to be called),
in which all development is done in feature branches and the merged into `main`.
The reason for feature branches is to allow multiple maintainers to actively work on
`amltk` without interfering.

## Documentation
The documentation is handled by [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/)
with some additional plugins. This is a markdown based documentation generator which allows
custom documentation and renders API documentation written in the code itself.

Where possible prefer to describe most of your documentation in code, with more custom
documentation generally not required for PRs.

You can find a collection of features for custom documentation [here](https://squidfunk.github.io/mkdocs-material/reference/)
as well as code reference documentation [here](https://mkdocstrings.github.io/usage/)

### Viewing Documentation
You can live view documentation changes by running `just docs`,
which will open your webbrowser and run `mkdocs --serve` to watch all files
and live update any changes that occur.

### Examples
The [`./examples`](examples/index.md) folder is where you can find our runnable
examples for AutoML-ToolKit.

When generating the documentation locally, the `just docs` command will
not run any examples, only render their code. You can control the running
of examples with

```bash
 # Run no examples
just docs
just docs "None"

# Run all examples
just docs "all"

# Run a single example called "Example Title"
just docs "Example Title"

# Run two examples called "Example1" and "Example2"
just docs "Example1, Example2"
```

#### Example Syntax
An example is just a python file, using the triple quote `"""`
comments to switch between commentary and code blocks.

The first `"""` block is special, in that the first line,
in this case **My Example Name** is the name of the example,
with anything following it being simple commentary.

```python
"""My Example Name

Here's a short description.
"""
from x import a
from y import b


"""
This is a commentary section. To see what can go in here,
take a look at https://squidfunk.github.io/mkdocs-material/reference/

Below we set some variables to some values.

!!! note "Special note"

    We use the variables p, q for fun.
"""
p = 2  # (1)!
p = 3  # <!> (2)!

print(p) # (3)!

# 1. You can add annotations to lines, where the text to annotate goes at
    the bottom of the code block, before the next commentary section.
    https://squidfunk.github.io/mkdocs-material/reference/annotations/
# 2. You can use <!> to highlight specific lines
# 3. Anything printed out using `print` will be rendered
"""
This concludes this example, check out ./examples for examples on how
to create ... examples.
"""
```

# Maintainer Guide
This section serves as a guide for active maintainers of `amltk` to
keep the ship running smoothly and help foster a growing user-base.
All maintainers must be familiar with the rest of the `CONTRIBUTING.md`.

#### Ethos
We appreciate all open-source contributions, whether that be a question,
issue or PR. This also pertains to potentially first-time contributors
and people new to Python and open-source in general. This includes
objective non-personal criticisms. We will try to be as helpful
and communicative as possible with respect to our availability,
and encourage open discussion.

To foster growth and contribution, we will guide users through the
library as required and encourage any and all contributions. If more
work is required on a PR, please encourage users to grant access to their
fork such that we can actively contribute to their contribution and utilize
a collaborative approach. This will also help prevent staling contributions.

In the event of any individual who makes personal attacks or derogative
comments, please maintain decorum, respond nicely, and if issues persist,
then inform the user they will be blocked.

#### Merging
We use `squash-merge` from feature branches to keep the commit log
to the `convential-commits` standard. This helps automate systems.

In the case of staling PR's, these will likely need a forceful rebase
from the `main` branch. This often has a negative impact on the commit
history of a pull request but this will be removed by `squash-merge`.

#### Workflows
To keep things relatively uniform, we try support recommended workflows
through the `justfile`. If there is a workflow that you prefer and is
not covered, please add your own.

#### Automation
Maintaining repositories is time-consuming work,
whether that be benchmarking, experimenting, testing, versioning,
issues, pull requests, documentation and anything else tangential to
code features. Any and all automation to the repository is greatly
appreciated but should be documented in the `Maintainers` section.

#### Versioning
Using [`convential-commits`](https://www.conventionalcommits.org/en/v1.0.0/) and `commitizen`, the versioning of
`amltk` automatically keeps a `CHANGELOG.md` and bumps versions in
any respective files.

Whenever a version needs to be bumped, this workflow has been
automated with `just bump`, which will bump all version strings
and keeping to semvar versioning, using the commit history as a guide.
Try to avoid using the `!` flag with a commit to indicate a major version
bump unless consensus has been reached. Perhaps once we have released
several major versions and a stabilized API, we may utilize this more freely.

#### Dependencies
One of the hardest parts of maintenance for a mature library,
especially one that supports integrations from both mature and
research code is managing dependencies. Where possible,
**prefer not adding an explicit dependency**. This mainly holds for
the **required** dependencies which all users must install. For
developer dependencies, please feel free to add one with good
justification. When integrating some machine learning ecosystem like
`scikit-learn` or `pytorch`, please try to bundle these dependencies
as **optional** and reflect so accordingly in the code.

There is some utility to work with optional dependencies in `amltk.types`,
such as `safe_isinstance` and `safe_issubclass`, to not rely on the
library being installed for runtime type checking. For static
compile time type checking, please use mypy's `if TYPE_CHECKING:`
idiom. This will prevent runtime errors for users who do not have
these dependencies installed. For example:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

def draw_configspace(self, space: ConfigurationSpace) -> None:
  ...
```

The exception to this rule is any modules a user must explicitly import
for the integration. In this case, it is fine to assume the user has the
required dependencies and any error generated is considered user error and
if possible guide them to the `pip install "amltk[optional_dep]"` that
they require for the integration.

#### Dependency updates
We have `dependabot` enabled in the repository using
the `.github/dependabot.yml`. This bot will periodically
make pull requests to the repository that update dependencies. Do
not accept these blindly but rather wait for any CI to finish and
ensure all tests still pass.

#### Long Term Decisions
Whenever faced with a long impacting decision, e.g. do we always
use `"cost"` as the values to return in a `Trial`, please make
an issue with the header `[Decision] Title Description` and
append the label `(decision)` on github. This lets us
revisit decisions made as well as the reasoning behind them.
