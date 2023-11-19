"""A collection of functional programming tools.

This module contains a collection of tools for functional programming.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from functools import partial, reduce
from inspect import isclass, signature
from itertools import count
from typing import (
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    Union,
)

T = TypeVar("T")
V = TypeVar("V")
V2 = TypeVar("V2")
K = TypeVar("K", bound=Hashable)
VK = TypeVar("VK", bound=Hashable)
RecMapping: TypeAlias = Mapping[K, Union["RecMapping[K, V]", V]]


def prefix_keys(d: Mapping[str, V], prefix: str) -> dict[str, V]:
    """Prefix the keys of a mapping.

    ```python exec="true" source="material-block" result="python" title="prefix_keys"
    from amltk._functional import prefix_keys

    d = {"a": 1, "b": 2}
    print(prefix_keys(d, "c:"))
    ```
    """
    return {prefix + k: v for k, v in d.items()}


def dict_get_not_none(d: Mapping[K, V], key: K, default: V2) -> V | V2:
    """Get a value from a dictionary, or a default value if it is None.

    ```python exec="true" source="material-block" result="python" title="dict_get_not_none"
    from amltk._functional import dict_get_not_none

    d = {"a": None, "b": 2}
    print(dict_get_not_none(d, "a", 1))  # d.get("a", 1) would return None
    ```
    """  # noqa: E501
    if (item := d.get(key)) is not None:
        return item
    return default


def mapping_select(d: Mapping[str, V], prefix: str) -> dict[str, V]:
    """Select a subset of a mapping.

    ```python exec="true" source="material-block" result="python" title="mapping_select"
    from amltk._functional import mapping_select

    d = {"a:b:c": 1, "a:b:d": 2, "c:elephant": 3}
    print(mapping_select(d, "a:b:"))
    # {"c": 1, "d": 2}
    ```

    Args:
        d: The mapping to select from.
        prefix: The prefix to select.

    Returns:
        The selected subset of the mapping.
    """
    return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}


def flatten_dict(d: RecMapping[str, V], *, delim: str | None = None) -> dict[str, V]:
    """Flatten a recursive mapping.

    ```python exec="true" source="material-block" result="python" title="flatten_dict"
    from amltk._functional import flatten_dict

    d = {"a": 1, "b": {"c": 2, "d": 3}}
    print(flatten_dict(d))
    # {"a": 1, "b:c": 2, "b:d": 3}
    ```

    Args:
        d: The recursive mapping to flatten.
        delim: The delimiter to use between keys.

    Returns:
        The flattened mapping.
    """
    delim = delim or ":"

    def _flatten_dict(d: RecMapping[str, V], prefix: str) -> Iterator[tuple[str, V]]:
        for k, v in d.items():
            if isinstance(v, Mapping):
                yield from _flatten_dict(v, prefix + k + delim)
            else:
                yield prefix + k, v

    return dict(_flatten_dict(d, ""))


def reverse_enumerate(
    seq: Sequence[T],
    start: int | None = None,
) -> Iterator[tuple[int, T]]:
    """Reverse enumerate.

    This function is similar to enumerate, but it iterates over the
    sequence in reverse.

    ```python exec="true" source="material-block" result="python" title="reverse_enumerate"
    from amltk._functional import reverse_enumerate

    xs = ["a", "b", "c"]
    for i, x in reverse_enumerate(xs):
        print(i, x)
    ```

    Args:
        seq: The sequence to iterate over.
        start: The starting index.

    Returns:
        An iterator over the sequence.
    """  # noqa: E501
    if start is None:
        start = len(seq) - 1
    yield from zip(count(start, -1), reversed(seq))


def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """Recursive version of getattr.

    This function is similar to getattr, but it allows you to get
    attributes using '.' notation.

    ```python exec="true" source="material-block" result="python" title="rgetattr"
    from amltk._functional import rgetattr

    class A:
        x = 1

    class B:
        a = A()

    b = B()
    print(rgetattr(b, "a.x"))
    ```

    See Also:
        * https://stackoverflow.com/a/31174427/5332072

    Args:
        obj: The object to get the attribute from.
        attr: The attribute to get.
        *args: The default value to return if the attribute is not found.

    Returns:
        The attribute.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj, *attr.split(".")])


def funcname(
    func: Callable,
    default: str | None = None,
    *,
    with_paren: bool = False,
) -> str:
    """Get the name of a function.

    Args:
        func: The function to get the name of.
        default: The default value to return if the name cannot be
            determined automatically.
        with_paren: Whether to include the parentheses.

    Returns:
        The name of the function.
    """
    from amltk.scheduling.task import Task

    if isinstance(func, Task):
        fname = funcname(func.function)
    elif isinstance(func, partial):
        fname = funcname(func.func)
    elif hasattr(func, "__qualname__"):
        fname = func.__qualname__
    elif hasattr(func, "__class__"):
        fname = func.__class__.__name__
    elif default is not None:
        fname = default
    else:
        fname = str(func)

    return f"{fname}()" if with_paren else fname


def fullname(o: Any) -> str:
    """Get the full name of an object.

    Args:
        o: The object to get the name of.

    Returns:
        The full name of the object.
    """
    klass = o if isclass(o) else o.__class__
    if (module := str(klass.__module__)) != "builtins":
        return module + "." + str(klass.__qualname__)

    return str(klass.__qualname__)  # avoid outputs like 'builtins.str'


def classname(c: Any, default: str | None = None) -> str:
    """Get the classname of an object.

    Args:
        c: The item to get the classname of.
        default: The default value to return if the name cannot be
            determined automatically.

    Returns:
        The name of the class.
    """
    if isinstance(c, type):
        return c.__name__
    if hasattr(c, "__class__"):
        return str(c.__class__.__name__)
    if default is not None:
        return default
    return str(c)


def entity_name(
    thing: Any,
    default: str | None = None,
) -> str:
    """Get the name of a thing.

    Args:
        thing: The thing to get the name of.
        default: The default value to return if the name cannot be
            determined automatically.

    Returns:
        The name of the thing.
    """
    if isinstance(thing, str):
        return thing
    if isinstance(thing, type) or hasattr(thing, "__class__"):
        return classname(thing)
    if callable(thing):
        return funcname(thing)
    if hasattr(thing, "__name__"):
        return str(thing.__name__)
    if default is not None:
        return default
    return str(thing)


def callstring(f: Callable, *args: Any, **kwargs: Any) -> str:
    """Get a string representation of a function call.

    Args:
        f: The function to get the string representation of.
        *args: The positional arguments.
        **kwargs: The keyword arguments.

    Returns:
        The string representation of the function call.
    """
    # Iterate over all args, convert them to str, and join them
    args_str = ""
    if any(args):
        args_str += ", ".join(map(str, args))
    if any(kwargs):
        args_str += ", ".join(f"{k}={v}" for k, v in kwargs.items())

    return f"{funcname(f)}({args_str})"


def compare_accumulate(
    xs: Iterable[T],
    op: Callable[[T, T], bool],
    *,
    ffill: bool = False,
) -> Iterator[T]:
    """Compare and accumulate.

    The most recent item to return True is carried forward and yielded.
    Once another item in the iterator returns True, it is then the value to be
    carried forward.

    ```python exec="true" source="material-block" result="python" title="compare_accumulate"
    from amltk._functional import compare_accumulate

    xs = [5, 4, 6, 2, 1, 8]
    print(list(compare_accumulate(xs, lambda x, y: x > y)))
    # [5, 6, 8]

    print(list(compare_accumulate(xs, lambda x, y: x > y, ffill=True)))
    # [5, 5, 6, 6, 6, 8]
    ```

    Args:
        xs: The iterable to compare and accumulate.
        op: The comparison operator.
        ffill: Whether to forward fill. If this is `True`, any item that returns
            `False` will be replaced with the most recent item that returned `True`.
    """  # noqa: E501
    itr = iter(xs)
    current = next(itr, None)
    if current is None:
        return

    yield current
    for x in itr:
        if op(x, current):
            yield x
            current = x
        elif ffill:
            yield current


def transformations(
    x: T,
    transforms: Iterable[Callable[[T], T]],
) -> Iterator[T]:
    """Apply transformations to an object.

    !!! note "Typing"

        The typing specifies that the domain of the type of `x` stays constant, i.e.
        `x` is always of type `T`. However this is not a strict requirement and you
        can safely ignore the type warnings if so.

    ```python exec="true" source="material-block" result="python" title="transforms"
    from amltk._functional import transformations

    def f(x):
        return x + 1

    def g(x):
        return x * 2

    def h(x):
        return x - 1

    x = 1
    steps = list(transformations(x, [f, g, h]))
    print(steps)
    # [1, 2, 4, 3]
    ```

    Args:
        x: The object to transform.
        transforms: The transformations to apply.

    Returns:
        An iterator over the transformed object.
    """
    yield x

    itr = iter(transforms)
    t = next(itr, None)
    if t is not None:
        yield from transformations(t(x), itr)


def select_by_signature(
    f: Callable,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the kwargs that are in the signature of `f`.

    Args:
        f: The function to select kwargs for.
        kwargs: The kwargs to select from.

    Returns:
        The selected kwargs.
    """
    if not any(kwargs):
        return {}

    sig = signature(f)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class Flag(Generic[T]):
    """A flag.

    This class is used to store a value that can be reset to its
    initial value.

    ```python
    flag = Flag(1)
    flag.value  # 1

    flag.set(2)
    flag.value  # 2

    flag.reset()
    flag.value  # 1
    ```

    Args:
        initial: The initial value.

    Attributes:
        value: The current flag value.
        initial: The initial value.
    """

    def __init__(self, initial: T) -> None:
        """Initialize the flag."""
        super().__init__()
        self.value = initial
        self.initial = initial

    def reset(self) -> None:
        """Reset the flag to its initial value."""
        self.value = self.initial

    def set(self, value: T) -> None:
        """Set the flag value."""
        self.value = value

    def __bool__(self) -> bool:
        """Get the flag value."""
        return bool(self.value)
