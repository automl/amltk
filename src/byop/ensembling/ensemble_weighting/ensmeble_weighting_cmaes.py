# Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)
import cma
from ..abstract_numerical_solver_ensemble import AbstractNumericalSolverEnsemble
from typing import List
import warnings
import numpy as np


class EnsembleWeightingCMAES(AbstractNumericalSolverEnsemble):
    """Numerical Solver CMA-ES to find a weight vector for ensemble weighting.

    Super Parameters
    ----------
        see NumericalSolverBase for more details on args and kwargs

    --- Method Parameters
   batch_size: int or str, default="dynamic"
        The batch size of CMA-ES ("popsize" for CMAES).
    sigma0: float, default=0.2
        The sigma for CMA-ES.
    """

    def __init__(self, *args, batch_size="dynamic", sigma0=0.2, **kwargs) -> None:

        # Determine batch size
        if isinstance(batch_size, int):
            tmp_batch_size = batch_size
        elif batch_size == "dynamic":
            # Following CMA-ES default
            # base on the number of base models (args[0])
            tmp_batch_size = int(4 + 3 * np.log(len(args[0])))
        else:
            raise ValueError(f"Unknown batch size argument! Got: {batch_size}")

        super().__init__(*args, **kwargs, batch_size=tmp_batch_size)
        self.sigma0 = sigma0

    def _minimize(self, predictions: List[np.ndarray], labels: np.ndarray, _start_weight_vector: np.ndarray,
                  n_evaluations):
        internal_n_iterations, n_rest_evaluations = self._compute_internal_iterations(n_evaluations)
        es = self._setup_cma(_start_weight_vector)
        val_loss_over_iterations = []

        # Iterations
        for itr in range(1, internal_n_iterations + 1):
            # Ask/tell
            solutions = es.ask()
            es.tell(solutions, self._evaluate_batch_of_solutions(solutions, predictions, labels))
            # es.disp(modulo=1)  # modulo=1 to print every iteration

            # Iteration finalization
            val_loss_over_iterations.append(es.result.fbest)

        # -- ask/tell rest solutions
        if n_rest_evaluations > 0:
            solutions = es.ask(n_rest_evaluations)
            es.best.update(solutions,
                           arf=self._evaluate_batch_of_solutions(solutions, predictions, labels))

            warnings.warn("Evaluated {} rest solutions in a remainder iteration.".format(n_rest_evaluations))
            val_loss_over_iterations.append(es.result.fbest)

        return es.result.fbest, es.result.xbest, val_loss_over_iterations

    def _compute_internal_iterations(self, n_evals):
        # -- Determine iteration handling
        internal_n_iterations = n_evals // self.batch_size
        if n_evals % self.batch_size == 0:
            n_rest_evaluations = 0
        else:
            n_rest_evaluations = n_evals % self.batch_size

        return internal_n_iterations, n_rest_evaluations

    def _setup_cma(self, _start_weight_vector) -> cma.CMAEvolutionStrategy:

        # Setup CMA
        opts = cma.CMAOptions()
        opts.set("seed", self.random_state.randint(0, 1000000))
        opts.set("popsize", self.batch_size)
        # opts.set("maxfevals", self.remaining_evaluations_)  # Not used because we control by hand.

        es = cma.CMAEvolutionStrategy(_start_weight_vector, self.sigma0, inopts=opts)

        return es
