import unittest
import glob

# Find job files.
files = glob.glob("test_jobs/*.json")
files.sort()

def func(file_name):
    from stgem.job import Job
    result = Job().setup_from_file(file_name).run()

# Create a class with a test method for each job file.
methods = {"test_" + file_name[:-5]:lambda self, file_name=file_name: func(file_name) for file_name in files}
C = type("TestJobs", (unittest.TestCase, ), methods)

if __name__ == "__main__":
    unittest.main()

