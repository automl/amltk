`amltk` is a package designed to help build and do research on AutoML Systems,
allowing you to spend more time on research and less time on engineering.

Some core components of `amltk`:

* Defining flexible [`Pipelines`][TODO], their search space and automatic
  ways to export useful objects such as an [`SklearnPipeline`][sklearn.]
  
  with utility to [`parse`][TODO] search space, [`configure`][TODO] them
  and finally [`build`] them into useable objects for your ML workflow.
* Setting up complex workflows with concepts with [`Tasks`][TODO], [`Events`][TODO] and
  a [`Scheduler`][TODO] to run your AutoML system in an
  event driven manner.

All of these were built with extensibility in mind!

* Perform [Hyperparameter Optimization (HPO)][TODO] sweeps on these pipelines using
  your favourite optimizer.
