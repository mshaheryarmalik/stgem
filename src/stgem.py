#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys, json

import click

from job import Job

@click.command()
@click.argument("files", type=str, nargs=-1)
@click.option("-n", type=int, multiple=True)
@click.option("--seed", "-s", type=int, multiple=True)
def start(files, n, seed):

    # Parse the descriptions from the command line arguments.
    descriptions = []
    for i, file_name in enumerate(files):
        if not os.path.exists(file_name):
            raise SystemExit("The file '{}' does not exist.".format(file_name))

        N_executions = n[i] if i < len(n) else 1
        for j in range(N_executions):
            with open(file_name) as f:
                description = json.load(f)

            # Now we add 1 to the seed for each consecutive copy of the job.
            SEED = seed[i] + j if i < len(seed) else None
            description["job_parameters"]["seed"] = SEED
            description["job_parameters"]["output_file"] = "job_{}_{}.pickle".format(i, j)

            descriptions.append(description)

    for description in descriptions:
        job = Job().setup_from_dict(description)
        jr = job.start()
        jr.dump_to_file(job.description["job_parameters"]["output_file"])

if __name__ == "__main__":
    start()

