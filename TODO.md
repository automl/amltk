# Pipeline
- [x] Pipeline `iter` deep and shallow
  * Done using `iter` for shallow and `traverse` for deep
- [x] Pipeline `walk` to iter like `traverse` the pipeline but get info about direct
parents and and splits 
- [x] ~~`as_dict` for dictionary representation of pipeline~~
  * See NOTES.md
- [x] `remove` and `replace` deeply
- [x] Extracting subpipelines
  - [x] `select` - Select a subpipeline
  - [x] `configure` - Configure a pipeline
    - [ ] Requires a test

# Assembler
- [] Write tests to check that `auto` works as intended.

### Space Parsers
- [ ] Test configspace parser
- [ ] Test None parser

### Builders
- [ ] Sklearn Pipeline

# ConfigSpace
- [ ] Allow for just a hyperparameter in the space of a `component` rather than requireing
  a configspace in the pipeline parser

# Spaces
- [ ] Grid
  - [ ] Impl
  - [ ] Tests
- [ ] ConfigSpace
  - [x] Impl
  - [ ] Tests


# Evalauations
- [ ] Results
- [ ] Evaluators
