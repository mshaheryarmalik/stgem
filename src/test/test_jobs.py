import unittest


class JobTestCase(unittest.TestCase):
    def test_10_random_file(self):
        from job import Job
        result = Job().setup_from_file("test_jobs/random.json").start()

    def test_20_mo3dk_pymodule(self):
        import mo3dk_python

    def test_30_mo3dk_file(self):
        from job import Job
        result = Job().setup_from_file("test_jobs/mo3dk.json").start()
        assert result.falsified



if __name__ == '__main__':
    unittest.main()
