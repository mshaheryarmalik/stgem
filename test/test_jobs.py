import unittest
import glob


class JobTestCase(unittest.TestCase):
    def test_00_json_file(self):
        from stgem.job import Job
        fns=glob.glob("test_jobs/*.json")
        fns.sort()
        for fn in fns:
            print("# Test job fle",fn)
            result = Job().setup_from_file(fn).run()

    def test_mo3dk_pymodule(self):
        import mo3dk_python


if __name__ == '__main__':
    unittest.main()
