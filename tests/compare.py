import importlib, os, sys


import click,time

from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.experiment import Experiment
from stgem.sut.python import PythonFunction
from stgem.sut.mo3d.sut import MO3D
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.simulated_annealing.algorithm import simulated_annealing

from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model

def get_seed_factory(init_seed=0):
    def seed_generator(init_seed):
        c = init_seed
        while True:
            yield c
            c += 1

    g = seed_generator(init_seed)
    return lambda: next(g)

def append_new_line(file_name, text_to_append):
	    with open(file_name, "a+") as file_object:
	        file_object.seek(0)
	        data = file_object.read(100)
	        if len(data) > 0:
	            file_object.write("\n")
	        file_object.write(text_to_append)

def get_generator_simulated_annealing():
	def generator_simulated_annealing():
		return STGEM(
	            description="mo3dSA",
	            sut=MO3D(),
	            budget=Budget(),
	            objectives=[Minimize(selected=[0], scale=True),
	                        Minimize(selected=[1], scale=True),
	                        Minimize(selected=[2], scale=True)
	                        ],
	            objective_selector=ObjectiveSelectorMAB(warm_up=50),
	            steps=[
	                    Search(budget_threshold={"executions": 5000},
	                        mode="stop_at_first_objective",
	                        algorithm=simulated_annealing()
	                            )]
	            )
	return generator_simulated_annealing	

def get_generator_random():
	def generator_random():
		return STGEM(
	            description="mo3dRandom",
	            sut=MO3D(),
	            budget=Budget(),
	            objectives=[Minimize(selected=[0], scale=True),
	                        Minimize(selected=[1], scale=True),
	                        Minimize(selected=[2], scale=True)
	                        ],
	            objective_selector=ObjectiveSelectorMAB(warm_up=50),
	            steps=[
	                    Search(budget_threshold={"executions": 5000},
		                    	mode="stop_at_first_objective",
		                    	algorithm=Random(model_factory=(lambda: Uniform())))
	                            ]
	            )
	return generator_random


N = 1000
init_seed = 5

def callback(idx, result, done):
	def append_new_line(file_name, text_to_append):
	    with open(file_name, "a+") as file_object:
	        file_object.seek(0)
	        data = file_object.read(100)
	        if len(data) > 0:
	            file_object.write("\n")
	        file_object.write(text_to_append)

	append_new_line("result/"+result.description+".txt",str(idx)+","+str(result.test_repository.minimum_objective))


experiment1 = Experiment(N, get_generator_simulated_annealing() , get_seed_factory(init_seed),result_callback=callback)

N_workers = 1
t0 = time.time() 
experiment1.run(N_workers=N_workers)
append_new_line("result/mo3dSA.txt", str(time.time() - t0)+ " seconds")


init_seed = 15


print("1")

experiment2 = Experiment(N, get_generator_random() , get_seed_factory(init_seed),result_callback=callback)

t0 = time.time() 
experiment2.run(N_workers=N_workers)
append_new_line("result/mo3dRandom.txt", str(time.time() - t0)+ " seconds")
print("2")
