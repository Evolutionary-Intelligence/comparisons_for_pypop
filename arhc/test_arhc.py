import time
from scipy import ones, dot, power, array

from pybrain.optimization import StochasticHillClimber as Optimizer

from pybrain.rl.environments.functions.function import FunctionEnvironment


class ElliFunction(FunctionEnvironment):
    """ Ellipsoid. """
        
    a = 1000
    
    def __init__(self, *args, **kwargs):
        FunctionEnvironment.__init__(self, *args, **kwargs)
        self._as = array([power(self.a, 2*i/(self.xdim-1.)) for i in range(self.xdim)])
        
    def f(self, x):
        return dot(self._as*x, x)


print('Optimizer:', Optimizer.__name__)
ndim_problem = 200
ellipsoid = ElliFunction(ndim_problem)
print('fitness of starting search point:', ellipsoid(ones(ndim_problem)))
start_time = time.time()
optimizer = Optimizer(ElliFunction(ndim_problem), ones(ndim_problem), verbose=True)
optimizer.learn()
print('best-so-far solution:', optimizer.bestEvaluable.params)
print('len of best-so-far solution:', len(optimizer.bestEvaluable))
print('best-so-far fitness:', optimizer.bestEvaluation)
print('used number of fitness evaluations:', optimizer.numEvaluations)
print('runtime:', time.time() - start_time)
