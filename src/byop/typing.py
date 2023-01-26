"""Stores low-level types used through the library."""
from __future__ import annotations

from typing import Any, Hashable, Mapping, TypeVar

from typing_extensions import TypeAlias

# Generic for objects that are aware of a space but not the specific kind
Space = TypeVar("Space", covariant=True)

# Type alias for kinds of Seeded objects
Seed: TypeAlias = int

# The name of an individual step, requires being Hashable
Key = TypeVar("Key", bound=Hashable)

# A name of a pipeline
Name = TypeVar("Name", bound=Hashable)

# A built pipeline object
BuiltPipeline = TypeVar("BuiltPipeline", covariant=True)

# An object representing a configuration of a pipeline.
# Notable examples include a Configuration object from ConfigSpace
# of which Mapping[Key, Any] is a supertype
# TODO: For now we only support Mapping[str, Any] but this serves
# as an abstraction point if we need it.
Config: TypeAlias = Mapping[Key, Any]
