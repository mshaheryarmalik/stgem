import unittest


class MyTestCase(unittest.TestCase):
    def test_dump(self):
        from stgem.job import Job, JobResult
        result = Job().setup_from_file("test_jobs/random.json").run()
        result.dump_to_file("temp.pickle")
        result2= JobResult.restore_from_file("temp.pickle")
        for s in result2.step_results:
            print(s.success)


if __name__ == '__main__':
    unittest.main()
