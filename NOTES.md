# Attrs and Mypy
---
* 2022-13-03

Unfortunatly Attrs, being the metaclass generating beast it is has some slight
issues with Mypy and keywords, not allowing`kw_only=True`.
```python
from attrs import frozen, define

@define
class A:
    name: str

@define
class B(A):
    pass

obj1 = B("hello world")  # Fine
obj2 = B(a="hello world")  # Fine but triggers a mypy/pycharm error
```

Current work-arounds are just rely on positional arguments or use `# type ignore`. As
inheritance and defaults can get nasty, we would rather stick to `kw_only=True` and
use `# type ignore` for now.