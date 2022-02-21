#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click

import stgem.run

@click.command()
@click.argument("files", type=str, nargs=-1)
@click.option("-n", type=int, multiple=True)
@click.option("--seed", "-s", type=int, multiple=True)
def main(files, n, seed):
    stgem.run.start(files, n, seed)

if __name__== "__main__":
    main()

