## 1.3.1 (2023-12-05)

### Fix

- **pyproject**: Change to recognized classifier for PyPI (#189)

## 1.3.0 (2023-12-05)

### Feat

- **History**: Default to normalizing time of history output

### Fix

- Provide more information if `built_item` fails (#187)
- **_doc**: Use `isinstance` on types
- **Optimizers**: Default to optimizer name #174

### Refactor

- **History**: History provides mutator methods
- Move `StoredValue` to own file in `.store`

## 1.2.4 (2023-11-25)

### Fix

- **docs**: Optimizer inline examples (#172)

## 1.2.3 (2023-11-24)

### Fix

- **Trial**: Add table to rich renderables (#170)
- **dask-jobqueue**: Default to `scale()` for predictable behaviour (#168)

## 1.2.2 (2023-11-23)

### Fix

- **test**: remove stray output from test/docs

## 1.2.1 (2023-11-22)

### Fix

- **pipeline**: `request()` correctly sets config key
- **Scheduler**: Make sure it's displayable in notebook

## 1.2.0 (2023-11-20)

### Feat

- **sklearn**: Special keywords for splits

### Fix

- **trials**: Always use a `PathBucket` with optimizers
- **Trial**: Trial should now always have a bucket

## 1.1.1 (2023-11-19)

### Fix

- **doc**: Add classifiers to pypi for bades (#155)

## 1.1.0 (2023-11-19)

### Feat

- **scheduler**: method `call_later` (#145)

### Refactor

- **optimization**: Add concept of `Metric` (#150)

## 1.0.1 (2023-11-15)

### Fix

- **CHANGELOG**: Fresh changelog
