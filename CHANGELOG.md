## v0.6.0 (2023-02-22)

### Feat

- **buckets**: Easy way to store and load data

## v0.5.0 (2023-02-20)

### Feat

- **scheduler**: Clean concept of task

## v0.4.0 (2023-02-17)

### Feat

- **scheduler**: Two way coms, nicer API

## v0.3.0 (2023-02-15)

### Feat

- **Task**: Communcation with workers, simpler api

## v0.2.1 (2023-02-14)

### Fix

- **dev**: Swtich to justfile
- **dev**: `make bump` now pushes generated tag

## v0.2.0 (2023-02-14)

### Feat

- **scheduler**: context `when` for callbacks

### Fix

- Change `Config` to be generic TypeVar

## v0.1.1 (2023-02-12)

## v0.1.0 (2023-02-12)

### Feat

- **scheduler**: Implement scheduler
- **pipeline**: Attach `build`
- **sklearn**: Build sklearn pipelines!
- **configuring**: Add `configure` to pipeline
- **parsing**: Add `parse` to `Pipeline`
- **configuration**: Add default configurators
- Add builder function
- **parsers**: Allow for simple configspace types
- Making `Split` act like a mapping
- **pipeline**: Add `Assembler`
- **pipeline**: Add `configure`
- **pipeline**: `select` for choosing choices in a pipeline
- **pipeline**: `replace` and `remove`
- configspace generator

### Fix

- typing
- **parsers**: ConfigSpace extra early delimiter
- Escape type annotaiton as str
- Makefile

### Refactor

- **parser**: Make overloads simpler
- **parsing**: Change to pure function
- **space_parsers**: Move to individual files
- Move to src layout
- **typing**: Reduce typing on Steps
- Fixup

## v0.0.3 (2022-10-18)

### Fix

- Testing if this fixed a version bump
- **versioning**: Sync the different version places

## v0.0.2 (2022-10-18)

### Fix

- **pyproject.toml**: Change version recognition to include prefix `v`
