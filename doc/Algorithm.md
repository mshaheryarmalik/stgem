# Algorithm
The purpose of an instance of the `Algorithm` class is to generate tests to be executed on the SUT. These tests together with the corresponding outputs form the test suite for the SUT.

## Algorithm Initialization
An algorithm, especially an algorithm utilizing machine learning algorithms, can make use of so-called models. Models correspond to instances of the `Model` class, and their usage is internal to the `Algorithm` class. Models are described in more details in the respective documentation. If models are used, there should be one model corresponding to each SUT output component.

The `__init__` method of an algorithm takes four arguments `model_factory`, `model`, `models`, and `parameters` which all default to `None`. The dictionary `parameters` is supposed to contain all static parameters the algorithm needs. If `parameters` is `None`, then the class variable `default_parameters` is used. Any dictionary values required by the algorithm but not provided by `parameters` are filled in from `default_parameters`. A value `self.parameters["x"]` can be directly accessed as `self.x`.

The other arguments relate to model creation. If the algorithm does not utilize models, these arguments can be left unspecified. At most one of `model_factory`, `model`, and `models` can equal not `None`. The value which is not `None` determines the model initialization. If `model_factory` is not `None`, it should be a function that returns instances of the `Model` class when called without arguments. If `model` is not `None`, then it should be an instance of the `Model` class, and all the required `Model` classes will be clones of this instance. If `models` is not `None`, then it should be a list of instances of the `Model` class to be used as models.

The `Algorithm` class has a `setup` method taking arguments `search_space`, `device`, and `logger`. This method is called when the corresponding STGEM step is being setup, right before the corresponding STGEM generator is being run. The `setup` method needs to be idempotent meaning that calling it several times results in the same outcome. It is mandatory to call the parent class `setup` method.

* `search_space`: Instance of `SearchSpace`. Provides access to the search space of the SUT.
* `device`: Pytorch device object providing access to a device where the machine learning models (if any) reside.
* `logger`: An instance of `Logger`. Provides logging capabilities via the method `self.log`.

Additionally a method `initialize` (without arguments) is provided which is called just before the `train` method (see below) is called for the first time. Any initialization that does not naturally go into the `setup` method should go here. This is a good place to put resource-heavy initialization tasks. In this way several `Algorithm` objects can be initialized and set up in advance without significant resource penalties.

## Algorithm Finalization
Right after the final call to the `generate_next_test` method (see below), the method `finalize` (without arguments) is called. This can be used for cleanup tasks as after this the `Algorithm` object is no longer used.

## Common Algorithm Attributes
Here are attributes common to all instances of `Algorithm`.

* `device`: Pytorch device object.
* `logger`: Logger object.
* `models`: A list of `Model` objects used by the algorithm.
* `N_models`: Number of model objects used by the algorithm. This equals the number of output components of the SUT.
* `perf`: An instance of `PerformanceData` tracking the algorithm performance.
* `search_space`: Search space of the SUT.

## Common Algorithm Methods
Here are methods common to all instances of `Algorithm`.

### Initialization
The method `initialize` is called right before the method `train` is called for the first time.

### Finalization
The method `finalize` is called right after the method `generate_next_test` is called for the last time. After this, the methods `train` or `generate_next_test` are no longer called.

### Training
The main loop of the STGEM `Step` object corresponding to the algorithm calls first the method `train` of the `Algorithm` object. The purpose of this method is to do any test generation tasks that broadly belong to the theme of training. This could be training of machine learning models or something similar. The actual implementation is in the method `do_train` which each inheriting class should implement; the method `train` is only a wrapper which tracks the execution time and saves in into `self.perf`. The method takes the arguments `active_outputs`, `test_repository`, and `budget_remaining.

* `active_outputs`: A list of indices indicating which outputs are active (this is determined by objective selectors). This information is typically used to train only the models corresponding to the active outputs in order to save resources, but this information can be safely ignored.
* `test_repository`: An instance of `TestRepository` which can be used to access all previously executed tests and their results.
* `budget_remaining`: A float in `[0, 1]` indicating how many percentage of the budget remain when the method is called.

### Test Generation
After calling the `train` method, the main STGEM `Step` loop calls the method `generate_next_test` of the `Algorithm` object. The aim of this method is to return a new test to be executed on the SUT (the actual execution is done externally). The outcome of the returned test is available during the next call via the `TestRepository` object. This method is only a wrapper to `do_generate_next_test` tracking generation time in `self.perf`. All inhereting classes should put the actual implementation in `do_generate_next_test`. This method has the same arguments as `train` above.

## Exceptions
TODO

