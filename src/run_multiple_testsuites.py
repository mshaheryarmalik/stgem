import os
import subprocess, sys
import pathlib

command = 'python main.py'


def run_on_powershell(scpt):
    p = subprocess.Popen(["powershell.exe",
                          scpt],

                         stdout=sys.stdout)
    p.communicate()


for i in range(30):
    run_on_powershell(command)
