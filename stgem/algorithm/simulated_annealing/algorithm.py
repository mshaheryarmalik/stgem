#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm
import time

class simulated_annealing(Algorithm):

    default_parameters = {"radius": 0.1}
    
    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        self.prec = 0
        self.rounds = 0

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):

        #====Find the minimum between the variables of a vector (the objectives) that are in the active outputs====
        def find_min_vect(active_outputs,v):
            temp=[]
            for i in active_outputs:
                temp.append(v[i]) 
            minimum = min(temp)
            return minimum  

        #====Return the proba to accept a bad solution====
        def find_proba(delta,T):
            return np.exp(delta/ T)

        #====Finds a random neighbor of the point x====    
        def neighbor(x):
            neighbor = x + self.radius * np.random.uniform(-1, 1, size=self.search_space.input_dimension)
            np.clip(neighbor, -1, 1, out=neighbor)
            return neighbor
        
        #====Calculation of the temperature====
        def init_T():
            return 0.5*budget_remaining**4

        dim = self.search_space.input_dimension #input dimension
        invalid = 0 #number of invalid test
        T = init_T() #the temperature which is between 0 and 100 

        while True and (self.rounds >= 2):
            self.rounds += 1
            x1,z1,y1 =test_repository.get(-1) #values of the previous test (which is called the "next")
            x2,z2,y2 =test_repository.get(-2 + self.prec) #values of the test that we get before the previous one (wich is called the "prev")

            f_Next = find_min_vect(active_outputs, y1) #the minimum objective of the previous test
            f_Prev = find_min_vect(active_outputs, y2) #the minimum objective that is kept (before the previous one)

            #====If the solution is better we accept and if it is not we accept with a certain probability====
            if (f_Prev > f_Next):
                #====Calculate the next test from the new solution====
                new_test = neighbor(x1.inputs)
                #====The previous index is the new solution (so the index is -2)====
                self.prec = 0
            else:
                p1 = find_proba(f_Prev - f_Next,T)
                #====Accept less good solutions====
                if p1 > np.random.random():
                    #====Calculate the next test from the less good solution====
                    new_test = neighbor(x1.inputs)
                    #====The previous index is the less good solution (so the index is -2)====
                    self.prec=0
                    #====If we don't accept the solution====
                else :
                    #====Calculate the next test from the previous solution====
                    new_test = neighbor(x2.inputs)
                    #====The previous value remains the same====
                    self.prec-=1
            if self.search_space.is_valid(new_test) == 0:
                invalid += 1
                continue
            break

        #====Neighbor for the second iteration====    
        if (self.rounds==1):
            x,z,y =test_repository.get(-1)
            new_test = neighbor(x.inputs)
            self.rounds += 1
        
        #====Random for the first iteration====
        if (self.rounds==0):
            new_test = np.random.uniform(-1, 1, size=self.search_space.input_dimension)
            self.prec = 0
            self.rounds += 1
        

        return np.array(new_test)





