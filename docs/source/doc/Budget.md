# Budget
Budget controls for how long the steps of an STGEM object are executed. Each step has one or several budget thresholds and, when one of the budget thresholds is exhausted, the execution terminates and the execution of the next step (if any) begins. The budget threshold can be based on time or number of executions on the SUT, and the user can define more complex budgets. Notice that the execution of the step can be terminated before the budget is exhausted, for example, when a falsifying input is found.

The methods `do_train` and `do_generate_next_test` of an instance of the `Algorithm` class get an argument `budget_remaining` which is a float in the interval [0,1] indicating how much budget is left (1=full budget left, 0=no budget left, 0.5=half of the budget left, etc.). In the case of several budget thresholds, the number `budget_remaining` relates to the budget threshold which is closest to being exhausted. The number `budget_remaining` is always reported relative to the current step; see below.

Internally a `Budget` object keeps track of quantities and the actual budget values are computed as functions of these quantities. For each of the default budgets, there exists a corresponding quantity and the actual budget value is simply found by returning the corresponding quantity value. See below for further details.

## Default Quantities and Budgets
The default `Budget` object specifies the following quantities and budgets:
* `executions`: Total number of test executions on the SUT.
* `execution_time`: Total execution time (in seconds) on the SUT.
* `generation_time`: Total time (in seconds) on algorithm test generation phase.
* `training_time`: Total time (in seconds) on algorithm training phase.
* `wall_time`: Wall time (in seconds) since the budget thresholds were updated for the first time.

The user may extend the budget class to specify additional quantities and budgets.

## Budget Initialization for STGEM
Currently an instance of `Budget` is automatically created when an `STGEM` generator is created. A custom budget can be specified via the `budget` argument.

## Setting Budget Thresholds for a STGEM Step
When creating an instance of `Step`, budget thresholds need to be specified via the keyword argument `budget_threshold`. For example,

```
Search(mode="exhaust_budget",
       budget_threshold={"executions": 100, "wall_time": 60},
       algorithm=Random(model_factory=(lambda: Uniform()))
      )
```

specifies a step that performs uniform random search until 100 tests have been executed on the SUT or one minute of wall time has passed (whichever condition is fulfilled first).

Notice that budgets always refer to totals, so they are not additive. This means that if two search steps are defined with respective SUT execution budgets of 20 and 100, a total of 100 tests will be executed. The first step finishes when 20 tests have been executed, and the remaining step has 100 - 20 = 80 executions left in the budget. The same is true for time budgets.

However, the argument `budget_remaining` described above is always relative to the step. In the above example, the second search step initially has `budget_remaining = 1.0` as none of the 80 possible executions has been performed. When a total of 30 executions have been performed (10 for the second step), we have `budget_remaining = 0.875` since 1 - (30 - 20)/80 = 0.875.

## Reporting How Much Budget Is Left
The remaining budgets can be found with the `used` method. The result is a dictionary whose values determine how much is left of each budget (described as a number in [0,1] as above). The method `remaining` simply returns the minimum of these numbers.

## Updating the Budget Thresholds
Budget thresholds are updated using the method `update_theshold`. For example

```
budget.update_threshold({"executions": 100})
```

updates the execution threshold to 100.

## Consuming the Budget
Budget is consumed by calling the method `consume`. For example

```
budget.consume("training_time", 5)
```

consumes 5 seconds of training time. Another way is to write

```
budget.consume(output)
```

where `output` is an `SUTOutput` object. By default consumption based on such an object does nothing, but the user can change the behavior by reimplementing the `consume_on_output` method.

When running an `STGEM` object, budget consumption is done internally when steps are executed, and the user generally does not need to do it manually.

## More on Quantities
This section is mainly relevant if the user wants to define a custom budget by extending the `Budget` class. A `Budget` object has an attribute `self.quantities` which tracks consumable quantities. So, for example, the code `budget.consume("training_time", 5)` adds `5` to the dictionary value `self.quantities["training_time"]`. The actual amount of budget used is found by calling the function `self.budgets["training_time"]` with argument `self.quantities`. By default, this function simply returns the value corresponding to the key `"training_time"`. However, this way of computing consumed budget values from the quantities allows more complex budgets to be defined. For example, we may write

```
self.budgets["custom_budget"] = lambda quantities: quantities["generation_time"] + quantities["training_time"]
budget.update_threshold({"custom_budget": 3600})
```

to create a more complex budget `"custom_budget"` which tracks the total of generation and training time used and limits the total to one hour.

