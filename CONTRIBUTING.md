# Contributing

Thanks for checking out the contribution page!
`amltk` is open to all open-source contributions, whether it be fixes,
features, integrations or even just simple doc fixes.

This guide will walk through some simple guidelines on the workflow
of `amltk` but also some design principles that are core to its
development. Some of these principles may be new for first time
contributors and examples will be given where necessary.

### Clone the repo

Clone the repo manually or with the below `hub` cli from GitHub.

```bash
hub repo fork automl/amltk
```

## Quickstart

Below is a quickstart guide for those familiar with open-source development. You
do not need to use `just`, however we provide it as a convenient workflow tool.
Please refer to the `justfile` for the commands ran if you wish to use your own
workflow.

> **_NOTE:_** If you are using Windows, please go to the [Windows installation](#windows-installation) section.

```bash
# Install just, the Makefile-like tool for this repo
# https://github.com/casey/just#installation
sudo apt install just

# Make virtual env (however you like)
python -m venv .my-virtual-env
source ./.my-virtual-env/bin/activate

# Install the library with dev dependancies
just install

# ... make a new branch
just pr-feat my-new-feature

# ... make changes
# ... commit changes

# Run tests
just test

# Run the documentation, fix any warnings
just docs

# Run pre-commit checks
just check

# ... fix anything that needs fixing

# Push to your fork
git push

# Create a PR (opening the browser too)
hub pull-request --browse
```

Below we will go into more detail on each of these steps.

## Setting up

The core workflows of `amltk` are accessed through the [`justfile`](https://github.com/casey/just)
It is recommended to have this installed with
their [simple one-liners on their repo](https://github.com/casey/just#installation).
All of these were developed with bash in mind and your usage with other platforms may vary,
please use the `justfile` as reference if this is the case.

#### Forking

If you are contributing from outside the `automl` org and under your own
github profile, you'll have to create your own [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

If you use the [`hub`](https://hub.github.com/) tool from github, you can do this locally with:

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
out
this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

Please provide a short summary of your changes to help give context to any reviewers who
may be looking at your code. If submitting a more complex PR that changes behaviours,
please consider more context when describing not only what you changed but why and how.

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
> This will run all test until it hits its first failure which allows for better incremental testing.
> It will also avoid running the examples which are often longer and saved for CI.

### Testing examples

If testing any added examples, please use the `just test-examples` command, which is
a shortcut for `pytest "tests/test_examples.py" -x --lf`. There is unfortunately no way
to sub-select one.

If you are not sure how to test your contribution or need some pointers to get started, please
reach out in the PR comments and we will be happy to assist!

## Code Quality

To ensure a consistent code quality and to reduce noise in the PR,
there are a selection of code quality tools that run.

These will be run automatically before a commit can be done with
[`pre-commit`](https://pre-commit.com/). The configuration for this can be found in
`.pre-commit-config.yaml`. All of these can be manually triggered using `just check`.

```bash
pre-commit run --all-files
```

This will automatically run the below tools.

### Ruff - Code Linting and formatting

The primary linter we use is [`ruff`](https://github.com/charliermarsh/ruff). The fixes and formatting
can be done manually as so:

```bash
ruff --fix src
ruff format src

# Or this which does both
just fix
```

### Mypy - Static Type Checking

This codebase also relies heavily on pythons `typing` and [`mypy`](https://mypy.readthedocs.io/en/stable/)
to ensure correctness across modules. Running this standalone on all files can take some time,
so we don't require you to run this, our automated testers will. If you wish to do so manually,
then use `just check-types`.

If any of the typing concepts are confusing, now is a good chance to learn,
and we would be happy to assist in helping properly type your PR if things do not work. If all else
fails, please feel free to introduce a `# type: ignore` to tell `mypy` to shut up **along with a
description to why it is there**. This will help future contributors and maintainers understand the
reasons behind these ignores. You can find a cheatsheet for basic mypy type
hinting [here](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#cheat-sheet-py3).

## Git workflow

We follow a _PR into trunk_ development flow (whatever that's meant to be called),
in which all development is done in feature branches and the merged into `main`.
The reason for feature branches is to allow multiple maintainers to actively work on
`amltk` without interfering. The `main` branch is locked down, meaning commits can not be made directly to main,
and features actions to trigger releases.

## Documentation

The documentation is handled by [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/)
with some additional plugins. This is a markdown based documentation generator which allows
custom documentation and renders API documentation written in the code itself.

Document features where the code lives. For example, if you introduce a new function, prefer
to add documentation with an example in the docstring of the function itself. There is also
some module level documentation which is used as a reference for the API. Most likely
updating this is only required for larger changes.

You can find a collection of features for custom
documentation [here](https://squidfunk.github.io/mkdocs-material/reference/)
as well as code reference documentation [here](https://mkdocstrings.github.io/usage/)

### Viewing Documentation

You can live view documentation changes by running `just docs`,
which will open your webbrowser and run `mkdocs --serve` to watch all files
and live update any changes that occur.

Please do not ignore warnings. The CI will fail if there are any warnings
and we will not merge any PR's that have warnings.

### Example Syntax

If creating an example, there is a custom format used to render `.py` files
and convert them to markdown that we can host.

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

print(p)  # (3)!

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

Please familiarize yourself with conventional commits and ensure that
the PR is up-to-date with the `main` branch before merging.

There should be no manual versioning, as this will take place during
releases automatically.

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

## Windows Installation

If you are not using Windows, feel free to skip this section.

### Installing WSL (Windows Subsystem for Linux)

1. Install WSL and Ubuntu by following the steps outlined in
   the [official Ubuntu installation guide](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview).

### Setting up PyCharm with WSL

1. Open the cloned repo in PyCharm and navigate to "Add new Interpreter" -> "On WSL..."
2. Choose WSL and specify the directory of the virtual env.
3. Open a PyCharm terminal and click "New predefined session" and select Ubuntu.

### Installing Dependencies

Since WSL is a Linux environment, you need to install Python separately, even if you have it on your Windows machine.

In the terminal, run the following commands to set up the project dependencies:

```bash
sudo apt update && sudo apt upgrade
sudo apt install python3.10
sudo apt install python3-pip
pip install --upgrade pip setuptools
sudo apt install python3.10-venv
python3 -m venv .venv
```

Then create a virtual environment and activate it:

```bash
source .venv/bin/activate
just install
```