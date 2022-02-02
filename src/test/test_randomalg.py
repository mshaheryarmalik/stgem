import unittest

class RandomAlgorithmTestCase(unittest.TestCase):
    def test_random_alg(self):
        from job import Job
        job_desc = {"sut": "odroid.OdroidSUT",
                    "sut_parameters": {},
                    "objective_func": "ObjectiveMaxSelected",
                    "objective_func_parameters": {"selected": [0]},
                    "objective_selector": "ObjectiveSelectorMAB",
                    "objective_selector_parameters": {"warm_up": 60},
                    "algorithm": "random.Random",
                    "algorithm_parameters": {"use_predefined_random_data": False,
                                             "predefined_random_data": {"test_inputs": None,
                                                                        "test_outputs": None}},
                    "job_parameters": {"N_tests": 20}
                    }

        ajob = Job(job_desc)
        ajob.start()


if __name__ == '__main__':
    unittest.main()
