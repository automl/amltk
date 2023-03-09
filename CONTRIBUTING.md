# Contributing
Thanks for checking out the contribution page!
`amltk` is open to all open-source contributions, whether it be fixes,
features, integrations or even just simple doc fixes.

This guide will walkthrough some simple guidelines on the workflow
of `amltk` but also some of the design principles that are core to its
development. Some of these principles may be new for first time
contributors and examples will be given where necessary.

## Setting up
The core workflows of `amltk` are accessed through the [`justfile`](https://github.com/casey/just)
in the root of the working directory. It is recommeneded to have this
installed with their simple one liner on their repo. All of these were
developed with bash in mind and your usage with other platforms may vary,
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
To install `amltk` for development, we rely on specify dependancies
that are not required for the actual library to run. There are listed
in the `pyproject.toml` under the `[project.optional-dependencies]` header.

You can install all of these by doing the following:
```bash
# Create a virtual environment in your preffered way.
python -m venv .my-virtual-env
source ./.my-virtual-env/bin/activate

# Install all required dependancies
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
by `just check`. Any list of errors will be presented to you and will
recommend fixing these before commiting.

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
are easier to verify and review but we will describe this process to
you if required.

#### Reviews
One your PR is submitted, we will likely have comments and changes that
are required. Please be patient as we may not be able to respond
immediately. If there are only minor comments, we will simply annotate
your code where these changes are required and upon fixing them, we will
happily merge these into the `main` branch and thank you for your
open-source contributions!

##### Granting access to your fork
If the PR requires larger structural changes or more discussion, there
will likely be a few back-and-forth discussion points which we will
actively respond to help get your contribution in.

If you do not wish to actively monitor the PR for whatever reason,
granting us access to modify your PR will substantially help
integration. To do so, please follow the instructions
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork#enabling-repository-maintainer-permissions-on-existing-pull-requests).


## Commits
TODO Mention conventional commits, link to docs, some basic use cases
and finally motivation to do so

## Testing

TODO Mention we use pytest and some basic commands to do so


## Code Quality

TODO Link to all the tools and what they do


## Git workflow
TODO Mention `main` branch with prs directly into main and some motivation

## Documentation
TODO basic instructions on how to view it, set it up contiunously. Links
to mkdocs and provide basic references for some of its cool features

## Type Driven Development
If you are unfamiliar with `types` in `python` then please consider
spending some time with the following section. `amltk` relies heavily
on types to ensure code editors can be as smart as possible, helping
guide users to writing correct code while using `amltk`.

#### What is Type Driven Development?
TODO: Link to a basic introduction to typing.

It probably means a lot of things, but for the purposes of `amltk` it
means bundle objects that belong together, together. This also means
favour re-usable components that are decoupled from where they are
intended to be used. Where possible, prefer immutable objects, i.e.
ones that are not meant to be modified once constructed. Prefer
composition over inheritance, using generics to indicate what's
being composed.

Lets take a look at a few examples at these different concepts.

##### A Timer
TODO: Link to dataclasses

Consider we need some timing functionality, that is we need to start
some `Timer` and then eventually `stop` it and record a `start`, `end`
and `duration`. We will use pythons `@dataclass` to create a simple
implementation and slowly iterate until we reach a well designed timer.

```python
from dataclasses import dataclass
import time

@dataclass
class Timer:
  start: float | None = None
  end: float | None = None
  duration: float | None = None

  def begin(self) -> None:
    self.start = time.time()

  def finish(self) -> None:
    self.end = time.time()
    self.duration = self.start - self.end

timer = Timer()
timer.start()
# ... do stuff
timer.end()
print(f"{timer=}")
```

There are a few notable problems here if a user strays off the "happy path"
and does something incorrectly.
```python
timer = Timer()
timer.begin()
print(timer.duration)  # This is None
timer.end()  # User accidentally calls end instead of finish
milliseconds_since_begining = time.end - time.start  # Whoops, no time interval
```

The problems here stem from the fact that `Timer` is mutable.
Trying to cram more functionality into `Timer` is just going to make it
more complex and prone to error. We can solve this by, you guess it,
types and immutability.

First let's make it immutable by using a indicative `@classmethod` and
a dedicated return type to represent a time.

```python
@dataclass
class timer:
    start_time: float

    @classmethod
    def start(cls) -> timer:
        return timer(start_time=time.time())

    def stop(self) -> tuple[float, float, float]:
        end = time.time()
        duration = self.end - self.start_time
        return self.start_time, end, duration

timer = timer.start()
# do something ...
begin, end, duration = timer.stop()  # get the information we care about
# do something else ...
begin, end, duration = timer.stop()  # we can even get another timestamp

```

Great, by adding the restriction that our class can not be changed, we
actually simplify the interface and introduce new functionality by
turning our `Timer` into more of a `Stopwatch` where multiple durations
can be retrieved. Notice how it's now quite difficult to misuse the class
or have it be in some invalid state. The one minor pain point someone
might have is a dislike of a `@classmethod` for construction, but this
enables advanced users to actually use the `Timer` in more complex ways,
such as providing their own `start_time` if required.


One problem still exists though, we are returning a tuple of `3` floats,
where the return value could be misinterpreted. The solution here is to
convert the multiple **return values** into a single **return type**.

```python
@dataclass
class TimeInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        # We can compute this cheaply when required.
        return self.end - self.start

@dataclass
class Timer:
    start_time: float

    @classmethod
    def start(cls) -> timer:
        return timer(start_time=time.time())

    def stop(self) -> TimeInterval:
        return TimeInterval(start=self.start_time, end=time.time())

timer = timer.start()
# do something ...
interval = timer.stop()
print(f"{interval=}")  # Interval(start=..., end=...)
print(interval.duration)
```

By introducing a simple type instead of multiple values, editors
can now be smarter and help guide a user of `Timer` to what the timer
returns when stopped and what's contained. The benfit of wrapping these
means we can now also add more information to our `TimeInterval` without
complicating the return type.

```python
@dataclass
class TimeInterval:
    start: float
    end: float
    unit: str   # We can attach more info to the interval

    @property
    def duration(self) -> float:
        # We can compute this cheaply when required.
        return self.end - self.start

@dataclass
class Timer:
    start_time: float

    @classmethod
    def start(cls) -> timer:
        return timer(start_time=time.time())

    def stop(self) -> TimeInterval:
        return TimeInterval(start=self.start_time, end=time.time(), unit="s")
```


The last modification we will make is to utilize an `Enum` to ensure no
`str` typos occur. Importing Enum's are annoying for consumers of an API
so we attach some convenience for users to the class they care about,
namely the `Timer`. The same could be done for the `TimeInterval` too.
This has an added benefit we can nicely implement a time interval
conversion function, to switch between units. Admittedly `start`
and `end` lose some contextual meaning here but the same principles apply
elsewhere.

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

class TimeUnit(Enum):
    SECONDS = auto()
    MILLISECONDS = auto()
    ...

@dataclass
class TimeInterval:
    start: float
    end: float
    unit: TimeUnit

    @property
    def duration(self) -> float:
        # We can compute this cheaply when required.
        return self.end - self.start

    # NOTE: Notice how the function does not modify itself but
    # returns a modified copy. This is immutability in action.
    def to(self, unit: TimeUnit) -> TimeInterval:
        """Convert this time interval to a different set of units."""
        new_start = ...
        new_end = ...
        return TimeInterval(start=new_start, end=new_end, unit=units)


@dataclass
class Timer:
    start_time: float

    # NOTE: Notice how we use the `ClassVar` here to indicate
    # that `units` is a class variable.
    units: ClassVar[type[TimeUnit]] = TimeUnit

    @classmethod
    def start(cls) -> timer:
        return timer(start_time=time.time())

    def stop(self) -> TimeInterval:
        return TimeInterval(
            start=self.start_time,
            end=time.time(),
            unit=TimeUnit.SECONDS,
        )


# Usage:
timer = Timer.start()
interval = timer.stop()

# Either
ms_interval = timer.to(Timer.units.MILLISECONDS)
```

There are many more modifications that are possible but the take-away
point is that by using immutability and types, we have made it
much harder for the user to misuse the API of `Timer` but also left
the door open for much more functionality to be implemented. The
`amltk.timing.Timer` shows a variant of this example which also
allows users to specify the `kind` of timer they want, i.e.
`"cpu"`, `"wall"` or `"process"` time, while bundling all required
information together. While it is by no means complete, it is quite
extensible for the future without needing to break API.


#### What Type is it? Generics
The library relies extensively **composition** with limited places of
_inheritance_ driven design. By asking a user to _inherit_ from a class
as a means of implementing their desired behaviours, you complicate
both your own `class` design, but also the job of a user.

However, sometimes you really do need a user to implement a certain
contract, for this we will use python's version of interfaces, `Protocol`s.
By relying on **interfaces**, which do not implement any functionality, you
provide a clean interaction point between people who wish to integrate
code and the main libraries inner workings, without intertwining the two.

The first advanced type feature to introduce is the `TypeVar`. This is
akin to typed languages **generic**

If you've already used some basic `mypy` typing in Python, you've likely
already used this without knowing it.

```python
from typing import Any

x: list[int] = ...
x: dict[str, Any] = ...
```

In the above two examples, you've declared `int` as the specialized type
of the container `list`, and likewise for the `dict`, you've indicated
that the keys are `str` and that the values can be `Any`thing.

## Tips

#### Easy virtual environments

These two things go a long way for creating virtaul environments
and activating them. Place them in your `.bashrc` or equivalent to use them.
```bash
pyvenv () {
    # Setup a virtual environment in the current directory if one does
    # not exist
    if exists '.venv'
    then
        echo '.venv already exists'
        return 1
    fi

    if ! emptyvar $VIRTUAL_ENV
    then
        deactivate
    fi

    echo "Using $(python -V) located at $(which python)"
    python -m venv .venv
    source './.venv/bin/activate'

    pip install --upgrade pip
    pip install wheel
}

# Activate a virtual environment in the current directory
alias pyshell='source ./.venv/bin/activate'
```
#### Editor Integrations
This serves as a reference for properly setting up your editor to take
advantage of all the types and [`code quality tools`](#code-quality) that `amltk` relies on. Please take
time to do so to get a much happier coding experience.

##### VSCode
TODO
##### PyCharm
TODO
##### Neovim
TODO

---

# Maintainer Guide
This section serves as a guide for active maintainers of `amltk` to
keep the ship running smoothly and help foster a growing user-base.
All maintainers but be familiar with the rest of the `CONTRIBUTING.md`.

#### Ethos
We appreciate all open-source contributions, whether that be a question,
issue or PR. This also pertains to potentially first-time contributors
and people new to Python and open-source in general. This includes
objective non-personal criticisms. We will try to be as helpful
and communicative as possible with respect to our availability,
and encourage open-discussion.

To foster growth and contribution, we will guide users through the
library as required and encourage any and all contributions. If more
work is required on a PR, please encourage users to grant access to their
fork such that we can actively contribute to their contribution and utilize
a collaborative approach. This will also help prevent staling contributions.

In the event of any indivdual who makes personal attacks or derogative
comments, please maintain decorum, respond nicely, and if issues persist,
then inform the user they will be blocked.

#### Merging
TODO: Insert link for convential commits

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
Mainting repositories is time consuming work,
whether that be benchmarking, experimenting, testing, versioning,
issues, pull requests, documentation and anything else tangential to
code features. Any and all automation to the repository is greatly
appreciated but should be documented in the `Maintainers` section.

#### Versioning
TODO: Insert links

Using [`convential-commits`]() and `commitizen`, the versioning of
`amltk` automatically keeps a `CHANGELOG.md` and bumps versions in
any respective files.

Whenever a version needs to be bumped, this workflow has been
automated with `just bump`, which will bump all version strings
and keeping to semvar versioning, using the commit history as a guide.
Try to avoid using the `!` flag with a commit to indicate a major version
bump unless concessus has been reached. Perhaps once we have released
several major versions and stabalized API, we may utilize this more freely.


#### Dependancies
One of the hardest parts of maintenance for a mature library,
especially one that supports integrations from both mature and
research code is managing dependancies. Where possible,
**prefer not adding an explicit dependancy**. This mainly holds for
the **required** dependancies which all users must install. For
developer dependancies, please feel free to add one with good
justification. If integration some machine learning eco-system some
as `scikit-learn` or `pytorch`, please try to bundle these dependancies
as **optional** and reflect so accordingly in the code.

There is some utility to handle such in `amltk.types` such as
`safe_isinstance` and `safe_issubclass` to not rely on the
library being installed for runtime type checking. For static
compile time type checking, please use mypy's `if TYPE_CHECKING:`
idiom. This will prevent runtime errors for users who do not have
these dependancies installed. For example:

```python
from typing import TYPE_CHECKING:

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

def draw_configspace(self, space: ConfigurationSpace) -> None:
  ...
```

The exception to this rule is any modules a user must explicitly import
for the integration. In this case, it is fine to assume the user has the
required dependancies and any error generated is considered user error and
if possible guide them to the `pip install "amltk[optional_dep]"` that
they require for the integration.


#### Dependancy updates
We have `dependabot` enabled in the repository using
the `.github/dependabot.yml`. This bot will periodically
make pull requests to the repository that update dependancies. Do
not accept these blindly but rather wait for any CI to finish and
ensure all tests still pass.
