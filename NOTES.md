# Attrs and Mypy
---
* 2022-13-03

Unfortunatly Attrs, being the metaclass generating beast it is has some slight
issues with Mypy and keywords, not allowing `kw_only=True`.
```python
from attrs import frozen

@frozen(kw_only=True)
class A:
    name: str

@frozen(kw_only=True)
class B(A):
    pass

obj1 = A(name="hello world")  # Fine
obj1_name = obj1.name

obj2 = B(name="hello world") 
obj2_name = obj2.name  # Fine but triggers a mypy/pycharm error
```

Current decided work around is just to duplicate attributes in the base class, this is
required for good documentation of these kinds of classes anyways as inhertiance and
documentation does not go together very well.

```python
from attrs import frozen

@frozen(kw_only=True)
class A:
    name: str

@frozen(kw_only=True)
class B(A):
    name: str
```

# Pipeline / Step `asdict`
I thought it made sense to add an `asdict` method to the pipeline but from trying to 
implement it, it does not seem like it makes parsing the pipeline any easier. 

The main issue is representing splits, in which we want the key to be the `name` but
then the value has to be both the split object itself, as well as all the paths. This
ends up more or less being the definition of the pipeline itself.

There is also the slightly more subtle problem in that a dictionary is not indicating
order where as a pipeline is ordered. Hence we should use a list and we can already
do list like operations using iter.

```python
from byop.pipeline import Pipeline, step, choice
pipeline = Pipeline.create(step("A", 1), step("B", 2), step("C", 3))

# This would seem fine
pipeline.asdict()
# {
#     "A": step("A", 1),
#     "B": step("B", 2),
#     "C": step("C", 3),
# }


# When we use a choice, using as simple dict layout would leave no way to access the
# choice object itself.
pipeline = Pipeline.create(step("A", 1), choice("choice", step("B", 2), step("C", 3)))
pipeline.asdict()
# {
#   "A": step("A", 1),
#   "choice" : {                   # <- we have no access to the choice object here
#       "B": step("B", 2),
#       "C": step("C", 3),
#   }
```

We could add a more complex structure such that each value is a tuple but this ends
up making the structure more complicated.

# Reasoning for using a class other than Pipeline for consturcting
Mainly the type signatures within the Pipeline would become bloated as we have to
account for quite a few different object types that would be part of it. However,
if we were to forgo types then it would make a lot of sense to put it in there,
I guess we need to see how it feels and to have it external is easier to refactor
in the long run.

# Use of `frozen` and `kw_only` in attrs for defining most classes
* The `frozen` decorates the class as a dataclass that is also immutable (at least not without
some hackery). Immutable structures are good for code where complex objects hold references
to other complex objects, as we will not accidentally change an objects state without explicitly
calling a `copy` or `mutate` to create a new object from the old one. I should probably
leave a reference somewhere on thre benefits of immutability but hopefully this is a known thing.
* The use of `kw_only` is to help prevent API breaking changes. Yes it leads to more verbose code
but things are unlikely to break, simply becuase the order or paramters changed or inhertance
is used and the order of parameters is suddenly different.

# Possibly user `.pyi` files
This would definitely help with the `@overload` spam within the main code but this introduces
a point of duplicated effort. The type stub `.pyi` files would have to be kept up to date
with any changes within the actual python code. While this may be fine if we could have automated
checks, I'm not sure they really provide this kind of possiblity. It would have to be checked
explicitly within my neovim setup but also pycharm for Aron. Even perhaps VSCode. I somehow
doubt consistency across these different IDE's.

# Use of `Ok, Err, Result`
I very much like this paradigm popularized in functional languages, it's a much better form
of error propogation as calling classes don't need to wrap everthing in try/except and can
forward and explictly check for errors and act accordingly, even chaning based on results.

However, basic users of the library should not be exposed to this. As a result, all public facing
API that is automated for the user should raise explicit errors. If a user is hooking into the
framework, for example to implement their own Space, then it is okay to require them to use
the `Ok, Err, Result` types.

# `Self` type
It's often quite good practice to use the `Self` type in a class like this:
```python
from typing import Self

class A:

    def __init__(self, x: int):
        self.x = x

    def copy(self) -> Self:
        return self.__class__(x=x)

class B:
    pass

a = A()
copy_a = a.copy()  # Is an A type

b = B()
copy_b = b.copy()  # Is a B type
```

If you instead used the definiteion `copy(self) -> A`, then when `B` inheritis it and you call
`b.copy`, then the type returned would be `A`.

```python
from typing import Self

class A:

    def __init__(self, x: int):
        self.x = x

    def copy(self) -> A:  # Notice here
        return self.__class__(x=x)

class B:
    pass

a = A()
copy_a = a.copy()  # Is an A type

b = B()
copy_b = b.copy()  # Is a A type
```

However trying to do so in the `Pipeline` class raised two issues. One is simply `mypy` complained
that the variable `Self` can not be used as a type. This could be due to the fact `typing_extensions`
are used for now but this should be check back on.

The second issue is in `create` where we allow a default `name` argument to be `None`.
If we use `Self` here, it infers that the `name` argument should always be the same as whatever `Self` binds
to. However sometimes we would like a different Name and this would break `Self`'ness. This is fine
and overloaded without issue at the moment but anyone inheriting from this class would face issues.

```python
    @classmethod
    @overload
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
    ) -> Pipeline[Key, str]:  # We can't make this `Self[Key, str]` as the types `Key, Name` are already bound to `Self`
        ...

    @classmethod
    @overload
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
        name: Name,
    ) -> Self:
        ...

    @classmethod
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
        name: Name | None = None,
    ) -> Self | Pipeline[Key, str]:
        """Create a pipeline from a sequence of steps.

        Args:
            *steps: The steps to create the pipeline from
            name (optional): The name of the pipeline. Defaults to a uuid

        Returns:
            Pipeline
        """
        # Expand out any pipelines in the init
        expanded = [s.steps if isinstance(s, Pipeline) else s for s in steps]
        step_sequence = list(Step.chain(*expanded))

        # Cleanest seperation we can do for now, in which things are only correct if the subclass
        # if explicitly using the `name` argument. It's
        if name is not None:
            return cls(name=name, steps=step_sequence)

        return Pipeline(name=str(uuid4()), steps=step_sequence)
```
