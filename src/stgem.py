#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys, json

from job import Job

if __name__ == "__main__":
    jobs = []
    if len(sys.argv) > 1:
        # Load jobs from the json files specified on the command line.
        for file_name in sys.argv[1:]:
            if not os.path.exists(file_name):
                raise SystemExit("The file '{}' does not exist.".format(file_name))
            with open(file_name) as f:
                jobs.append(json.load(f))
    else:
        # Random
        job_desc = {"sut": "odroid.OdroidSUT",
               "sut_parameters": {},
               "objective_func": "ObjectiveMaxSelected",
               "objective_func_parameters": {"selected": [0]},
               "objective_selector": "ObjectiveSelectorMAB",
               "objective_selector_parameters": {"warm_up": 60},
               "algorithm": "random.Random",
               "algorithm_parameters": {"use_predefined_random_data": False,
                                        "predefined_random_data": {"test_inputs": None,
                                                                   "test_outputs": None}},
               "job_parameters": {"N_tests": 300}
               }

        print(json.dumps(job_desc))
        raise SystemExit
        jobs.append(job_desc)

    for job in jobs:
        ajob = Job(job)
        ajob.start()

