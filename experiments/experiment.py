'''
Created on 21/06/2017

@author: mashallaryan
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
import time
import cocoex
from cocoex import Suite, Observer, log_level
import numpy as np
from bboptim.optimization import *
from mashall.functions import random_search

def default_observers(update={}):
    """return a map from suite names to default observer names"""
    # this is a function only to make the doc available and
    # because @property doesn't work on module level
    _default_observers.update(update)
    return _default_observers
_default_observers = {
    'bbob': 'bbob',
    'bbob-biobj': 'bbob-biobj',
    'bbob-biobj-ext': 'bbob-biobj',
    'bbob-constrained': 'bbob',
    'bbob-largescale': 'bbob',  # todo: needs to be confirmed
    }
class ObserverOptions(dict):
    """a `dict` with observer options which can be passed to
    the (C-based) `Observer` via the `as_string` property.

    See http://numbbo.github.io/coco-doc/C/#observer-parameters
    for details on the available (C-based) options.
    Details: When the `Observer` class in future accepts a dictionary
    also, this class becomes superfluous and could be replaced by a method
    `default_observer_options` similar to `default_observers`.
    """
    def __init__(self, options={}):
        """set default options from global variables and input ``options``.
        Default values are created "dynamically" based on the setting
        of module-wide variables `SOLVER`, `suite_name`, and `budget`.
        """
        dict.__init__(self, options)
    def update(self, *args, **kwargs):
        """add or update options"""
        dict.update(self, *args, **kwargs)
        return self
    def update_gracefully(self, options):
        """update from each entry of parameter ``options: dict`` but only
        if key is not already present
        """
        for key in options:
            if key not in self:
                self[key] = options[key]
        return self
    @property
    def as_string(self):
        """string representation which is accepted by `Observer` class,
        which calls the underlying C interface
        """
        s = str(self).replace(',', ' ')
        for c in ["u'", 'u"', "'", '"', "{", "}"]:
            s = s.replace(c, '')
        return s


class ShortInfo(object):
    """print minimal info during benchmarking.
    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.
    Example output:
        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done
        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done
        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done
    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0

    def print(self, problem, end="", **kwargs):

        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()

    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs

    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
        return s

    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
            self.f_current = f
        # print_flush(res)
        return res
    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")
    @staticmethod
    def short_time_stap():
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

# algorithm info
# algorithm_name
# function
# dimension
# instance
# budget
class Experiment():

    def __init__(self,suite_name,func_id, dim, instance, max_num_instances,  n_init_steps, n_steps , base_output_folder, **kwrgs ):
        """

        """
        self.n_init_steps, self.n_steps = n_init_steps, n_steps


        # initialize seed for the instance so that all instances in all optimization
        np.random.seed(123)
        initseeds = np.random.randint(1000,2000,size=max_num_instances)
        np.random.seed(initseeds[instance-1])
        # prepare suite
        # set dimension and function id
        self.problem_info = "f_{:3}, {:>3}d instance {:>3}\n".format( str(func_id), str(dim), str(instance))
        suite_options="dimensions: {} function_indices: {} ".format(dim, func_id)
        # set instance id  (number of runs)
        suite_instance = "instances:{}-{}".format(instance,instance)
        # realize suite
        self.suite = Suite(suite_name=suite_name, suite_instance=suite_instance  , suite_options=suite_options)
        self.problem = self.suite[0]
        alg_name = kwrgs.get("alg_name")  # should be obtained from optimizer
        alg_info = kwrgs.get("alg_info") # should be obtained from optimizer

        output_folder = base_output_folder+  "{}_on_f{}_inst_{}_dim{}_budget{}".format(  alg_name
                                                                                            , func_id
                                                                                            , instance
                                                                                            , dim
                                                                                            , self.n_steps )

        observer_options = ObserverOptions({  # is (inherited from) a dictionary
                         'algorithm_info': alg_info, # CHANGE/INCOMMENT THIS!
                         'algorithm_name': alg_name,  # default already provided from SOLVER name
                         'result_folder':  output_folder,  # default already provided from several global vars
                       })

        observer_name = default_observers()[suite_name]
        self.observer = Observer(observer_name, observer_options.as_string)
        self.optimizer = self.create_optimizer(**kwrgs)



    def create_optimizer(self, **kwrgs):
        """
        convenience method for initializing the optimizer
        """
        cma_popsize = kwrgs.get("cma_popsize", 100)
        cma_max_itr = kwrgs.get("cma_max_itr", 100)
        model_type  = kwrgs.get("model_type")
        assert model_type is not None
        lower_bound = self.problem.lower_bounds
        upper_bound = self.problem.upper_bounds
        box = [lower_bound , upper_bound]

        xt, yt = random_search(self.problem, lower_bound, upper_bound, self.n_init_steps)


        self.observer.observe(self.problem)

        return Optimizer( bounding_box=box, init_observatins=[xt,yt]
                            , evaluation_function = self.problem
                            , **kwrgs)


    def optimize(self):
        """
         main optimization loop
         parameters:
            `n_init_steps`: int
                             number of random initial samplessteps to create an initial set of observations over which the model is built
            `n_steps` : int
                        number of expensive optimization steps
        """
        short_info = ShortInfo()
        short_info.print(self.problem)
        i = 0
        while i < self.n_steps:
            self.optimizer.step()
            i+=1
            print ('{}: main optimization loop step {}'.format(self.problem_info, i))

        self.problem.free()


#
# def random_search(fun, lbounds, ubounds, budget):
#     """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
#
#     lbounds, ubounds = np.array(lbounds), np.array(ubounds)
#     dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
#     max_chunk_size = 1 + 4e4 / dim
#     while budget > 0:
#         chunk = int(min([budget, max_chunk_size]))
#         # about five times faster than "for k in range(budget):..."
#         X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
#         F = [fun(x) for x in X]
#         if fun.number_of_objectives == 1:
#             index = np.argmin(F)
#             if f_min is None or F[index] < f_min:
#                 x_min, f_min = X[index], F[index]
#         budget -= chunk
#     return x_min
#
#
#  print("start")
#
# exp = Experiment(suite_name ='bbob', func_id=2, dim=10, instance=1, max_num_instances=30
#                  ,  n_init_steps=20, n_steps=1000
#                  , base_output_folder =""
#                  , alg_name = "random"
#                  , alg_info = "just a test" )
# exp.test(budget=1000)
#
# print("done")
