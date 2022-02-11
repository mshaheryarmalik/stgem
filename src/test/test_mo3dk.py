import unittest

class MO3DkTestCase(unittest.TestCase):
    def test_job_file(self):
        from job import Job
        result = Job().setup_from_file("test_jobs/mo3dk.json").start()
        assert result


if __name__ == '__main__':
    unittest.main()
