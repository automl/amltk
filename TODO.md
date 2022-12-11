# Pipeline
- [x] Pipeline `iter` deep and shallow
  * Done using `iter` for shallow and `traverse` for deep
- [x] Pipeline `walk` to iter like `traverse` the pipeline but get info about direct
parents and and splits 
- [x] ~~`as_dict` for dictionary representation of pipeline~~
  * See NOTES.md
- [x] `remove` and `replace` deeply
- [ ] Extracting subpipelines
  * Implemented `select` for it
  * Need to implement `configure` then
  - [ ] With a mapping object
- [ ] Typing with the individual steps is hard as we erase types to base class `Step` to accomodate
    all types for the flow, but we lose information on absolute types. This is relatively impossible
    anyways for all the complex operations. The question is the "same as what is the type of `["a", 1, None]`"?
    To solve this type erasure, the Pipeline methods will return Union types over all possible step types.
    This doesnt solve the issue but at least narrows it down for the user.. i.e. `list[str | int | None]`
    

# Spaces
- [ ] Grid
  - [ ] Impl
  - [ ] Tests
- [ ] ConfigSpace
  - [x] Impl
  - [ ] Tests

# Builders
- [ ] Sklearn Pipeline

# Evalauations
- [ ] Results
- [ ] Evaluators
