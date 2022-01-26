#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import subprocess, sys
import pathlib

command = 'C:\\Users\\japeltom\\AppData\\Local\\Programs\\Python\\Python37\\python.exe main.py'


def run_on_powershell(scpt):
    p = subprocess.Popen(["powershell.exe",
                          scpt],

                         stdout=sys.stdout)
    p.communicate()


for i in range(30):
    run_on_powershell(command)
