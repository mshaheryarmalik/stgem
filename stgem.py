#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click

import stgem.run

@click.command()
@click.argument("files", type=str, nargs=-1)
@click.option("-n", type=int, multiple=True)
@click.option("--seed", "-s", type=int, multiple=True)
@click.option("--resume", "-r", type=click.Path(file_okay=True, dir_okay=True, readable=True,writable=True), multiple=False)
@click.option("--threads","-t", type=int, default=1)
def main(files, n, seed, resume,threads):
    stgem.run.start(files, n, seed, resume,threads)

if __name__== "__main__":
    main()

