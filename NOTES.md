# Attrs and Mypy
---
* 2022-13-03

Unfortunatly Attrs, being the metaclass generating beast it is has some slight
issues with Mypy and keywords, not allowing`kw_only=True`.
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

### Pipeline / Step `asdict`
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