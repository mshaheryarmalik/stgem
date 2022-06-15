#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm
import time
import math

class simulated_annealing(Algorithm):

    default_parameters = {"radius": 0.1}

    def append_new_line(file_name, text_to_append):
        with open(file_name, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            file_object.write(text_to_append)

    
    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        #self.radius = 0.01
        self.prec = 0
        self.rounds = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0
        self.t0 = time.time()

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):

        def append_new_line(file_name, text_to_append):
            with open(file_name, "a+") as file_object:
                file_object.seek(0)
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                file_object.write(text_to_append)
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

        def neighbor(x):
            """Finds a random neighbor of the point x. Currently this is just a
            random displacement of x."""
            neighbor = x + self.radius * np.random.uniform(-1, 1, size=self.search_space.input_dimension)
            np.clip(neighbor, -1, 1, out=neighbor)
            return neighbor
 
        dim = self.search_space.input_dimension #input dimension
        invalid = 0 #number of invalid test
        T = 0.5*budget_remaining**4 #the temperature which is between 0 and 100 
        #T = math.exp(math.log(101)*budget_remaining)-1

        while True and (self.rounds >= 2):
            self.rounds += 1
            x1,z1,y1 =test_repository.get(-1) #values of the previous test (which is called the "next")
            x2,z2,y2 =test_repository.get(-2 + self.prec) #values of the test that we get before the previous one (wich is called the "prev")

            f_Next = find_min_vect(active_outputs, y1) #the minimum objective of the previous test
            f_Prev = find_min_vect(active_outputs, y2) #the minimum objective that is keep (before the previous one)

            #append_new_line("result/temp.txt",str(f_Prev - f_Next))
            #====If the solution is better we accept and if it is not we accept with a certain probability====
            if (f_Prev > f_Next):
                self.r1 += 1
                #====Calculate the next test from the new solution====
                new_test = neighbor(x1.inputs)
                #====The previous index is the new solution (so the index is -2)====
                self.prec = 0
            else:
                p1 = find_proba(f_Prev - f_Next,T)
                #====Accept less good solutions====
                if p1 > np.random.random():
                    #append_new_line("result/temp.txt",str("1,")+str(time.time() - self.t0))
                    self.r2 += 1
                    #====Calculate the next test from the less good solution====
                    new_test = neighbor(x1.inputs)
                    #====The previous index is the less good solution (so the index is -2)====
                    self.prec=0
                    #====If we don't accept the solution====
                else :
                    #append_new_line("result/temp.txt",str("2,")+str(time.time() - self.t0))
                    self.r3 += 1
                    #====Calculate the next test from the previous solution====
                    new_test = neighbor(x2.inputs)
                    #====The previous index remains the same====
                    self.prec-=1
            if self.search_space.is_valid(new_test) == 0:
                invalid += 1
                continue
            break

        #====Random for the 2 first iterations====
        if (self.rounds<2):
            new_test = np.random.uniform(-1, 1, size=self.search_space.input_dimension)
            self.prec = 0
            self.rounds += 1
        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------
        self.perf.save_history("N_tests_generated", self.rounds)
        self.perf.save_history("N_invalid_tests_generated", invalid)
        print(self.r1)
        print(self.r2)
        print(self.r3)

        return np.array(new_test)





