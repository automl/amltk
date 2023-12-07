---
title: 'AMLTK: A Modular AutoML Toolkit in Python'
tags:
  - Python
  - Machine Learning
  - AutoML
  - Hyperparameter Optimization
  - Modular
  - Data Science
authors:
  - name: Edward Bergman
    orcid: 0009-0003-4390-7614
    corresponding: true
    equal-contrib: false
    affiliation: 1
  - name: Matthias Feurer
    orcid: 0000-0001-9611-8588
    corresponding: true
    equal-contrib: false
    affiliation: “2, 3”
  - name: Aron Bahram
    orcid: 0009-0002-8896-2863
    corresponding: false
    equal-contrib: false
    affiliation: 1
  - name: Amir Rezaei Balef
    orcid: 0000-0002-6882-0051
    corresponding: false
    equal-contrib: false
    affiliation: 4
  - name: Lennart Purucker
    orcid: 0009-0001-1181-0549
    corresponding: false
    equal-contrib: false
    affiliation: 1
  - name: Sarah Segel
    orcid: 0009-0005-2966-266X
    corresponding: false
    equal-contrib: false
    affiliation: 5
  - name: Marius Lindauer
    orcid: 0000-0002-9675-3175
    corresponding: true
    equal-contrib: false
    affiliation: “5,6”
  - name: Frank Hutter
    orcid: 0000-0002-2037-3694
    corresponding: true
    equal-contrib: false
    affiliation: 1
  - name: Katharina Eggensperger
    orcid: 0000-0002-0309-401X
    corresponding: true
    equal-contrib: false
    affiliation: 4
affiliations:
  - name: University of Freiburg, Germany
    index: 1
  - name: LMU Munich, Germany
    index: 2
  - name: Munich Center for Machine Learning
    index: 3
  - name: University of Tübingen, Germany
    index: 4
  - name: Leibniz University Hannover, Germany
    index: 5
  - name: L3S Research Center, Germany
    index: 6
date: 6 December 2023
bibliography: paper.bib
---

# Summary
Machine Learning is a core building block in novel data-driven applications.
Practitioners face many ambiguous design decisions while developing practical machine learning (ML) solutions.
Automated machine learning (AutoML) facilitates the development of machine learning applications by providing efficient methods for optimizing hyperparameters, searching for neural architectures, or constructing whole ML pipelines [@hutter-book19a].
Thereby, design decisions such as the choice of modelling, pre-processing, and training algorithm are crucial to obtaining well-performing solutions.
By automatically obtaining ML solutions, AutoML aims to lower the barrier to leveraging machine learning and reduce the time needed to develop or adapt ML solutions for new domains or data.

Highly performant software packages for automatically building ML pipelines given data, so-called AutoML systems, are available and can be used off-the-shelf.
Typically, AutoML systems evaluate ML models sequentially to return a well-performing single best model or multiple models combined into an ensemble.
Existing AutoML systems are typically highly engineered monolithic software developed for specific use cases to perform well and robustly under various conditions.

With the growing amount of data and design decisions for ML, there is also a growing need to improve our understanding of the design decisions of AutoML systems.
Current state-of-the-art systems vary in implemented paradigms (stacking [@erickson-arxiv20a] vs CASH [@thornton-kdd13a], optimizing a pre-defined pipeline structure [@thornton-kdd13a] vs evolving open-ended pipelines [@olson-gecco16a]) and also use different methods
within one paradigm (i.e. Bayesian optimization [@thornton-kdd13a; @feurer-nips15a] or Genetic Programming [@olson-gecco16a; @gijsbers-joss19a] as the optimization algorithm,
different search spaces for the same machine learning algorithm cf. [@olson-gecco16a; @gijsbers-joss19a; @thornton-kdd13a; @feurer-nips15a], different post-hoc ensemble methods or even no post-hoc ensembling at all cf. [@feurer-nips15a; @autoprognosis; @wang2021flaml]),
raising many research questions and opportunities to study improved algorithms and novel applications.

AMLTK (Automated Machine Learning ToolKit) is a collection of components that enable researchers and developers to easily implement AutoML systems without the need for common boilerplate code.
AMLTK addresses this with a modular perspective on AutoML systems, aiming to cover various existing AutoML system paradigms in principle.
It contributes to the field three-fold:
(a) Enabling systematic comparison of AutoML design decisions with a higher level of reproducibility,
(b) fast prototyping and evaluation of new AutoML methods,
and (c) easy adaptation of developed solutions to new tasks.

In addition, it also facilitates easy integration and swapping of components from various AutoML tools, for example,
an optimizer from [SMAC](https://github.com/automl/SMAC3) [@lindauer-jmlr22a] or [Optuna](https://github.com/optuna/optuna) [@akiba-kdd19a],
a search space from [ConfigSpace](https://github.com/automl/ConfigSpace) [@lindauer-arxiv19a] or Optuna,
as well as the integration with additional tools such as the visualization and analysis tool [DeepCAVE](https://github.com/automl/DeepCAVE) [@sass-realml22a].
These provided integrations are done without the need to modify AMLTK’s source code, enabling users to extend the framework as their needs require.
Overall, AMLTK lowers the barrier to engaging with AutoML research and, thus, opens up the opportunity to bundle research efforts towards flexible and effective AutoML systems.

AMLTK is designed for AutoML researchers to develop and study novel AutoML systems and domain experts to adapt these AutoML systems for novel use cases.
AMLTK is based on the experience of a subset of the authors in developing AutoML systems (Auto-sklearn [@feurer-nips15a; @feurer-jmlr22a] and Auto-PyTorch [@zimmer-tpami21a]) and their effort to unify their code bases.
Last but not least, we also believe that this toolkit will help educate students and support ML practitioners in engaging with AutoML systems.

# Statement of Need
Current AutoML systems are monolithic and provide little opportunity for customization.
As a result, researchers often build new AutoML systems to implement a new methodology.
This results in two issues:
(1) it creates a barrier to research on AutoML systems,
and (2) it hinders the fair comparison of new components in AutoML systems.
Recent examples of open source AutoML systems are AutoGluon [@erickson-arxiv20a], GAMA [@gijsbers-joss19a; @gijsbers-kdd21a], and Auto-Sklearn [@feurer-nips15a; @feurer-jmlr22a].

To give an example for Issue (1), a researcher working on new optimization methods for AutoML would need to develop all components of an AutoML system in order to evaluate their method because current systems do not allow for easy replacement of the optimization method, as pointed out by @mohr-ml23a.
Also, a researcher wanting to study a variation of an existing system would need to go through an extensive, potentially undocumented codebase to find the right place to apply their variation.
The tight integration of components allows for highly efficient systems but poses a high barrier to new research and novel, innovative AutoML systems.

Issue (2) is also a huge problem.
A recent benchmark study [@gijsbers-arxiv23a] extensively compared multiple AutoML systems on a common set of ML tasks.
While such benchmarking efforts are necessary to assess the current state of the art, we note that each system uses its own implementation of the search space, optimization, evaluation and ensembling, making a principled comparison and ablation study virtually impossible and leaving potential performance gains by combining solutions unnoticed.
Instead of comparing different methods, researchers are actually comparing the implementations.
By providing a unified toolkit for AutoML, researchers can focus on comparing the changes they have made while leaving all other parts of the AutoML system as they were.

# Acknowledgements
Edward Bergman was partially supported by TAILOR, a project funded by EU Horizon 2020 research and innovation programme under GA No 952215.
Katharina Eggensperger and Amir Balef acknowledge funding by the German Research Foundation under Germany's Excellence Strategy - ECX number 2064/1 - Project number 390727645.
Marius Lindauer acknowledges support by the Federal Ministry of Education and Research (BMBF) under the project AI service center KISSKI (grantno. 01IS22093C).
Lennart Purucker acknowledges funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) -- SFB 1597 -- 499552394.
Sarah Segel acknowledges funding by the European Union (ERC, "ixAutoML", grant no. 101041029).
Frank Hutter acknowledges funding by the European Union (ERC Consolidator Grant “DeepLearning 2.0”, grant no. 101045765)
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency.
Neither the European Union nor the granting authority can be held responsible for them.

# References
