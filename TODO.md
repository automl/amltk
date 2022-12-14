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

### Components
- [ ] Technically you can create a choice that also has an associated space with it at the moment.
  This is not exposed through the `choice` api function but it's possible. Nothing inherintely wrong
  with this and we shouldn't prevent it honestly, but the ConfigSpace parser currently raises an
  error when this happens. _(We could deal with this technically but for simplicity we don't)_

# Assembler
- [] Write tests to check that `auto` works as intended.

### Space Parsers
- [ ] Test configspace parser
- [ ] Test None parser
- [ ] Test configspace parser deals with parents being the choice well

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
