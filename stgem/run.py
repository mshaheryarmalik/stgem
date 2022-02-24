#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, json

from stgem.job import Job, JobResult
import dill as pickle


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


def start(files, n, seed, resume):
    # Parse the descriptions from the command line arguments.

    descriptions = []
    # resume argument given, it checks if an unfinished pickle by the same name exists
    # If exists that pickled data is restored to the descriptions
    # If not raise system exit with an error message file "The resume file ... does not exist."
    if len(resume) > 0:
        resume_file = os.path.join('output',
                                   'unfinished_{}.pickle'.format(os.path.basename(resume[0].rsplit('.', 1)[0])))

        if not os.path.exists(resume_file):
            raise SystemExit("The resume file '{}' does not exist.".format(resume_file))

        else:
            restore = restore_from_file(file_name=resume_file)
            descriptions = restore

    else:
        for i, file_name in enumerate(files):
            if not os.path.exists(file_name):
                raise SystemExit("The file '{}' does not exist.".format(file_name))

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

    for description, des_index in zip(descriptions, range(len(descriptions))):
        job = Job().setup_from_dict(description)
        jr = job.start()
        jr.dump_to_file(job.description["job_parameters"]["output_file"])

        # If the job is not single test run it creates pickles
        # make pickle file link, name depending on a new run or a resume
        if len(descriptions) > 1:
            current_file = files[0] if len(resume) == 0 else resume[0]
            unfinished_pickle_file = 'output/unfinished_{}.pickle'.format(
                                     os.path.basename(current_file.rsplit('.', 1)[0]))

            if des_index < len(descriptions) - 1:
                dump_to_file(obj=descriptions[des_index + 1:], file_name=unfinished_pickle_file)
            else:
                os.remove(unfinished_pickle_file)
