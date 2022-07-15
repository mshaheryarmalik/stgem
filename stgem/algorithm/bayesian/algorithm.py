#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm

class BayesOptSampler(Algorithm):

    """Bayesian Optimization sampler : Defined only for continuous domains.
    For discrete inputs define another sampler"""
    def __init__(self, domain, init_num): #BO_params
        super().__init__()
        try:
            import GPyOpt
        except ModuleNotFoundError:
            raise RuntimeError(
                'BayesOptSampler requires GPyOpt to be installed')

        #BoxSampler
        self.dimension = domain.standardizedDimension
        if not self.dimension >= 0:
            raise RuntimeError(f'{self.__class__.__name__} supports only'
                               ' continuous standardizable Domains')

        #DomainSampler
        self.domain = domain
        self.last_sample = None

        self.init_num = init_num #BO_params.init_num
        if self.init_num < 1:
            raise RuntimeError(
                'init_num for BayesOptSampler must be at least 1')
        self.bounds = []
        for i in range(self.dimension):
            self.bounds.append({'name':'x_'+str(i), 'type': 'continuous',
                                'domain': (0,1)})
        self.X = np.empty((0, self.dimension))
        self.Y = np.empty((0, 1))

    # DomainSample
    def getSample(self):
        sample, info = self.getVector()
        return np.array(self.domain.unstandardize(sample)), info # Np array to make sample reshape-able

    # BoxSample
    def getVector(self, feedback=None):
        import GPyOpt   # do this here to avoid slow import when unused

        if len(self.X) < self.init_num:
            # Do random sampling
            sample = np.random.uniform(0, 1, self.dimension)
        else:
            BO = GPyOpt.methods.BayesianOptimization(
                f=None, batch_size=1,
                domain=self.bounds, X=self.X, Y=self.Y, normalize_Y=True) # normalize_Y=False
            sample = BO.suggest_next_locations()[0]
        return tuple(sample), None

    # DomainSample
    def update(self, sample, info, rho):
        self.updateVector(self.domain.standardize(sample), info, rho)

    # BoxSample
    def updateVector(self, vector, info, rho):
        self.X = np.vstack((self.X, np.atleast_2d(vector)))
        self.Y = np.vstack((self.Y, np.atleast_2d(rho)))

    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        self.rounds = 0

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
        if self.rounds > 0: # Update algorithm with system results after first round
            _, _, y = test_repository.get(-1)
            rho = min([y[i] for i in active_outputs])
            #rho = y[-1] # Only update with robustness
            self.update(self.test, self.info, rho)
        self.test, self.info = self.getSample()
        self.rounds += 1

        return self.test