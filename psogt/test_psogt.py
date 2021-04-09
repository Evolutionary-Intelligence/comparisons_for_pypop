import time
from scipy import ones, dot, power, array

from pybrain.optimization.populationbased.pso import ParticleSwarmOptimizer as Optimizer

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
ndim_problem = 2
ellipsoid = ElliFunction(ndim_problem)
print('fitness of starting search point:', ellipsoid(ones(ndim_problem)))
boundaries = [(0, 2) for _ in range(ndim_problem)]
start_time = time.time()
optimizer = Optimizer(ElliFunction(ndim_problem), verbose=True, boundaries=boundaries)
optimizer.learn()
print('best-so-far solution:', optimizer.bestEvaluable)
print('len of best-so-far solution:', len(optimizer.bestEvaluable))
print('best-so-far fitness:', optimizer.bestEvaluation)
print('used number of fitness evaluations:', optimizer.numEvaluations)
print('runtime:', time.time() - start_time)
print(optimizer.minimize)
for p in optimizer.particles:
    print(p.velocity)
print('-' * 27)
for p in optimizer.particles:
    print(p.position)
