import os
import subprocess, sys
import pathlib

command = 'C:\\Users\\japeltom\\AppData\\Local\\Programs\\Python\\Python39\\python.exe main.py'


def run_on_powershell(scpt):
    p = subprocess.Popen(["powershell.exe",
                          scpt],

                         stdout=sys.stdout)
    p.communicate()


for i in range(5):
    run_on_powershell(command)
