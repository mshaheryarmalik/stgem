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
        print("usage: stgem.py job_description_file [... job_description_file_n]")
        raise SystemExit

    for job in jobs:
        ajob = Job(job)
        ajob.start()