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
            jobs.append(Job().setup_from_file(file_name))
    else:
        print("usage: stgem.py job_description_file [... job_description_file_n]")
        raise SystemExit

    for n, job in enumerate(jobs):
        jr = job.start()
        jr.dump_to_file("job_{}_output.pickle".format(n))

