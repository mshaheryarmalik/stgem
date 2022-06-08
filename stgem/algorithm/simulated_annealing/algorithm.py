#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm

class simulated_annealing(Algorithm):
    
    prec = 0  #prec value index

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

        S = 0.01 #precision 
        dim = self.search_space.input_dimension #input dimension
        new_test=[] #intitialisation of the result
        rounds = 0 #number of loop
        invalid = 0 #number of invalid test
        T = budget_remaining #the temperature which is between 0 and 100 

        while True:
          rounds += 1
          x1,z1,y1 =test_repository.get(-1) #values of the previous test (which is called the "next")
          x2,z2,y2 =test_repository.get(-2 + simulated_annealing.prec) #values of the test that we get before the previous one (wich is called the "prev")

          f_Next = find_min_vect(active_outputs, y1) #the minimum objective of the previous test
          f_Prev = find_min_vect(active_outputs, y2) #the minimum objective that is keep (before the previous one)

          #====If the solution is better we accept and if it is not we accept with a certain probability====
          if (f_Prev > f_Next): 
            #====Calculate the next test from the new solution====
            for i in range(dim):
              bounds=self.search_space.sut.input_range[i]
              new_test.append(x1.inputs[i] + S * (np.random.uniform(-1,1)*(bounds[1]- bounds[0])))
            #====The previous index is the new solution (so the index is -2)====
            simulated_annealing.prec = 0
          else:
            p1 = find_proba(f_Prev - f_Next,T)
            #====Accept less good solutions====
            if p1 > np.random.random():
              #====Calculate the next test from the less good solution====
              for i in range(dim):
                bounds=self.search_space.sut.input_range[i]
                new_test.append(x1.inputs[i] + S * (np.random.uniform(-1,1)*(bounds[1]- bounds[0])))         
              #====The previous index is the less good solution (so the index is -2)====
              simulated_annealing.prec=0
            #====If we don't accept the solution====
            else :
              #====Calculate the next test from the previous solution====
              for i in range(dim):
                bounds=self.search_space.sut.input_range[i]
                new_test.append(x2.inputs[i] + S * (np.random.uniform(-1,1)*(bounds[1]- bounds[0])))
              #====The previous index remains the same====
              simulated_annealing.prec-=1
          if self.search_space.is_valid(new_test) == 0:
              invalid += 1
              continue

          break

        
        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------
        self.perf.save_history("N_tests_generated", rounds)
        self.perf.save_history("N_invalid_tests_generated", invalid)
        print("new test")
        print(new_test)
        print(simulated_annealing.prec)

        return np.array(new_test)





