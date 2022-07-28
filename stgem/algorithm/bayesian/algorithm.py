#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm

class BayesOptSampler(Algorithm):

    """Bayesian Optimization sampler : Defined only for continuous domains.
    For discrete inputs define another sampler"""
    def __init__(self):
        super().__init__()
        try:
            import GPyOpt
        except ModuleNotFoundError:
            raise RuntimeError(
                'BayesOptSampler requires GPyOpt to be installed')
        self.bounds = []

    def getVector(self,  active_outputs, test_repository):
        import GPyOpt # do this here to avoid slow import when unused
        X, _, Y = test_repository.get()
        X = np.asarray([x.inputs for x in X])
        minRHO = np.vstack([min([y[i] for i in active_outputs]) for y in Y])
        BO = GPyOpt.methods.BayesianOptimization(
                f=None, batch_size=1,
                domain=self.bounds, X=X, Y=minRHO, normalize_Y=False)
        sample = BO.suggest_next_locations()[0]
        return np.asarray(tuple(sample))

    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        self.rounds = 0
        for i in range(self.search_space.input_dimension):
            self.bounds.append({'name':'x_'+str(i), 'type': 'continuous',
                                'domain': (-1,1)})

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
        self.test = self.getVector(active_outputs, test_repository)
        self.rounds += 1
        return self.test