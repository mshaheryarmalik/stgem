#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, json

from stgem.job import Job
import dill as pickle
from multiprocessing import Pool

def split_path(s):
    rest, tail = os.path.split(s)
    if rest in ("", os.path.sep):
        return tail,
    return split_path(rest) + (tail,)


def restore_from_file(file_name):
    with open(file_name, "rb") as file:
        obj=pickle.load(file)
    return obj

def dump_to_file(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)


def run_one_job(description,resume):
    output_filename = description["job_parameters"]["output_file"]
    # we execute a job if we are not in resume mode
    # or if we are in resume mode but the output file does not exist
    if resume is None or not os.path.exists(output_filename):
        job = Job().setup_from_dict(description)
        jr = job.run()
        jr.dump_to_file(output_filename)

def start(files, n, seed, resume, multiprocess):

    descriptions = []

    # 1. prepare job descriptions
    if resume is None:
        # not in resume mode, we build the resume file
        resume_filename= "output/unfinished_jobs.pickle"

        for i, file_name in enumerate(files):
            if not os.path.exists(file_name):
                raise SystemExit("The job file '{}' does not exist.".format(file_name))

            N_executions = n[i] if i < len(n) else 1
            for j in range(N_executions):
                with open(file_name) as f:
                    description = json.load(f)

                # Add the directory for loading user-written modules.
                f = os.path.dirname(file_name)
                description["job_parameters"]["module_path"] = ".".join(x for x in split_path(os.path.dirname(file_name)))

                # Now we add 1 to the seed for each consecutive copy of the job.
                SEED = seed[i] + j if i < len(seed) else None
                description["job_parameters"]["seed"] = SEED
                description["job_parameters"]["output_file"] = "output/job_{}_{}.pickle".format(i, j)
                descriptions.append(description)

        # always create a resume file, even if there is a single job since it may not terminate
        dump_to_file(obj=descriptions, file_name=resume_filename)
    else:
        # resume argument given, it checks if an unfinished pickle by the same name exists
        # If exists that pickled data is restored to the descriptions
        # If not raise system exit with an error message file "The resume file ... does not exist."
        resume_filename = resume
        if not os.path.exists(resume_filename):
            raise SystemExit("The resume file '{}' does not exist.".format(resume_filename))
        else:
            descriptions = restore_from_file(file_name=resume_filename)

    # 2. execute job descriptions
    if multiprocess <= 1:
        # sequential execution of jobs
        for description in descriptions:
            run_one_job(description, resume)
    else:
        # parallel execution of jobs in different processes
        with Pool(multiprocess) as pool:
            pool.starmap(run_one_job, zip(descriptions, [resume] * len(descriptions)))
            pool.close()
            pool.join()

    # 3. clean up
    os.remove(resume_filename)


