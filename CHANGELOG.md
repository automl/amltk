## 0.10.0 (2023-11-13)

### Feat

- **Pynisher**: add  for introspection of capability

### Fix

- Add `__main__` gaurd to examples

## 0.9.0 (2023-11-13)

### Feat

- **rich**: Enable rich visual output (#138)
- **NEPS**: Add integration for NePS
- **xgboost**: Add simple component
- **PathBucket**: add `rmdir`
- **sklearn**: Estimator of fixed preds/probs
- **sklearn**: Create VotingX of prefitted estimators
- **Requeue**: Data structures for a queue which can requeue
- **Profiler**: Eanble with just __call__
- **Profiler**: Toggle switch
- **Pipeline**: Allow requesting of parameters
- **PathBucket**: Allow for pandas pickle with `.pdpickle`
- **Profiler**: Object to accumulate profiles
- **Task**: Simplify event types
- **Event**: Allow multiple event subscribers for events
- **Metafeatures**: Basic metafeature calculators (#137)
- **Metafeatures**: Basic metafeature calculators (#136)
- **Metafeatures**: Basic metafeature calculators (#135)
- **Scheduler**: Add `loky` backend as an option
- **Bucket**: `delete_from_storage`
- **Bucket**: `delete_from_storage`
- **scheduler**: Add loop exception handler
- Allow plugin to be added post init
- **CallLimiter**: Allow predication on other tasks
- **sklearn**: Allow custom pipeline type and builder_kwargs
- **Step**: `transform_context=` for `config_transform=`
- **Step**: `config_transform=`
- **data**: `flatten_if_1d`
- **Trial**: profiles to be added to summary
- **Trial**: `measure` for profiling sections of code
- **Profiling**: Combine Memory and Timing
- **Pynisher**: Enable passing `mp_context`
- **Report**: Add status for cancelled trial (#131)
- **Memory**: Add memory tracking (#130)
- **Batch**: `batch.cancel(fire_event=...)`
- Depend more on `deepcopy` than shallow
- **Events**: Allow subscribers to forward events
- **Timer**: Allow contextmanager style timing
- **Task**: Batched tasks (#121)
- **Timer**: Allow `prefix=` to `to_dict()`
- **threadpoolctl**: Plugin to help limit threads (#117)
- **SMAC**: Verify cost reported (#116)
- **data**: `to_numpy`
- **Task**: Parameter `stop_on=` (#114)
- **randomness**: as_randomstate (#113)
- **pipeline**: Allow `meta` field (#106)
- **History**: Save/load from disk (#96)
- **data**: Dtype reduction func for pandas/numpy (#95)
- **optimization**: Basic Mutli-fidelity for SMAC (#94)
- **pipeline**: `group()` to group steps together
- **Step**: Convenience methods with two-way traversal (#84)
- **scheduler**: `Scheduler.run(end_on_exception: bool = True, raises: bool = True)` (#80)
- History (#78)
- **RandomSearch**: Pass in custom sampler, duplicate handling, doc (#76)
- **Sampler**: Option for `duplicates: bool | list[Config]` (#70)
- **dask_jobqueue**: cluster executors (#61)
- **scheduling**: Sequential Executor (#60)
- **pipeline**: Consolidate parsing, sampling and configuring to be more uniform (#55)
- **pipeline**: `attach(modules=..., searchables=...)` (#32)
- **controller**: add default scheduler for AskAndTell (#20)
- **Optimizers**: Add support for optuna (#15)
- **Bucket**: Enable `/` operator for subdirs (#12)
- **Task**: Concurrent task limits (#13)
- **tasks**: Limit tasks with pynisher (#11)

### Fix

- **examples**: adding a continuous search space to the simple_hbo example (#139)
- **conversions**: `probabilities_to_classes` corrected
- **Scheduling**: Minor updates
- **Event**: Remove trailing comma from log messages
- **Pipeline**: Requests sets correct key
- **events**: Typing
- **Profile**: Remove extranuous `:`
- **Builder**: Pass error to super()
- **History**: More lenient about missing fields
- **Step**: Don't consider `old_parent` in eq
- **Sklearn**: Splitter now works with choices inside
- **Group**: Less freedy config picking
- **Sampling**: Typing
- **Scheduler**: Don't wait on terminate
- **measure**: Byte size correctly handles str
- **ensembling**: Remove uneeded check
- **Profiling**: Make safe to exceptions
- **Trial**: Serialization of reports
- **Scheduler**: Extract future exceptions
- **History**: Freeze reports before generating df
- Remove stray debugging
- **Loaders**: Allow Series and parquet
- **Scheduler**: More logging to `INFO`
- **Scheduler**: `stop` event set in loop
- **Trial**: Implement `on_cancelled`
- **History**: Fix defragmented warning from pandas
- **Timing**: decorator classmethod order
- **SMAC**: Float cost is accepted
- **example**: Simpe HPO uses `scheduler.on_empty`
- **scheduler**: Less strict on timer
- **scheduler**: Using `get_event_loop` for py3.8
- **data**: Export data.conversions
- **History**: Missing used of `reversed`
- **History**: `sortby(reverse=)`
- **import**: Make `group` accessible (#105)
- **pipeline**: Qualified name only parent splits (#104)
- **sklearn**: Check item is subclass ColumnTransformer (#103)
- **ConfigSpace**: Copy concrete `ConfiguraitonSpace` objects (#85)
- **Pipeline**: `traverse()` would duplicate items (#81)
- **sklearn**: `data_split(stratified=)` unpack items (#75)
- **Sampler**: Better error message, `max_attempts=10` (default) (#71)
- **ConfigSpace**: Don't try insert Constants from config (#67)
- Remove bad import
- **examples**: example runner (#62)
- Pass `seed` correctly to `default_rng` if `int` (#59)
- **Task**: Robust Scheduling w/ CommTasks | fix(typing): Make typing compatible with 3.8 (#52)
- **deps**: Add `pynisher` as required dependancy (#51)
- **bucket**: permit .yml files (#30)
- **smac**: pass seed to trial (#42)
- attrs dependancy (#34)
- **optuna**: Accept `int` or `float` for reported values (#29)
- **Optuna**: Now verifies trial values based on study direction (#25)
- **loader**: Use given path to determine which loader to use (#16)
- **readme**: correct github link

### Refactor

- Sorry (#143)
- **Scheduler**: Revamp events/scheduler and its docs (#141)
- **Events**: Remove bloat and move to composition as adviced use of `Emitter` (#140)
- **Ensembling**: simplify `weighted_ensemble_caruana`
- **Trial**: Improve profiling
- **Trail**: Use `BaseException`
- **TaskPlugin**: Remove parameter generics
- **Profiling**: Memory measures vms and rss
- **Trial**: Rename `measure` to `profile`
- **Scheduling**: More robustness
- **ensemble**: Rename parameter
- **Task**: Remove `Trail.Task`
- **Event**: More information for `INFO` (#126)
- **Task**: Use `UniqueRef` type (#125)
- **Task**: call `emit` on Subscriber
- **Trial**: Compose with Task, not inherit (#124)
- **Comm**: Make it a plugin, not a task (#93)
- **scheduler**: change _submit() to public (#53)
- **parsing,configuring,building**: Remove `result` dependancy (#27)
- **Pipeline**: Remove Key and Name typevars, use `str` (#24)
- **Trial**: A config is now always assumed to be a mapping (#23)
- **tasks**: Events and scheduling (#17)

## v0.8.0 (2023-03-03)

### Feat

- **optimizer**: SMAC intregration

### Fix

- Merge conflict

## v0.7.0 (2023-02-24)

### Feat

- **control**: AskAndTell controller

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
