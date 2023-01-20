"""Stores low-level types used through the library."""
from __future__ import annotations

from typing import Hashable, TypeVar

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
