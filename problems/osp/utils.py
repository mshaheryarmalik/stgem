import re
from farn.farn import execute_command_set, logger
from typing import Any, Dict, List
from numpy import dtype
from pandas import DataFrame, Series
from dictIO import DictReader
from dictIO.utils.path import relative_path
from farn.core import Cases
from farn.farn import create_param_dict_files
from farn.farn import create_case_folders
from pathlib import Path
from typing import Dict, List
from farn.core import Case, Parameter
from numpy.linalg import norm

command_sets: Dict[str, List[str]] = {
    "prepare": [
        "copy ..\\..\\gunnerus-dp\\template\\caseDict .",
        # copy a (case-agnostic) ospx config file from a template directory into the case folder
        "dictParser caseDict",  # parse the ospx config file. This will make it case-specific
        "ospCaseBuilder parsed.caseDict",  # build the (case-specific) OspSystemStructure.xml
    ],
    "run": [
        "cosim run OspSystemStructure.xml -b 0 -d 3600",  # run OSP cosim
    ],
    "post": [
        "watchCosim -d -p watchDict",
        # optional post-processing. watchCosim creates a sub-folder 'results' in the case folder
    ],
}


def execute_test_case(case_name, case_parameters):
    global command_sets

    cases: Cases = Cases()
    case_name = "test_" + str(case_name)
    case_folder: Path = Path("./cases") / case_name

    case_parameters: List[Parameter] = [
        Parameter("current_velocity0", case_parameters[0]),  # TODO: Confirm exact names
        Parameter("current_velocity1", case_parameters[1])
        # Parameter("Current_Velocity[2]", case_parameters[2]),
        # Parameter("Current_Velocity[3]", case_parameters[3]),
        # Parameter("Current_Velocity[4]", case_parameters[4]),
        # Parameter("Current_Velocity[5]", case_parameters[5])
    ]

    case: Case = Case(
        case=case_name,
        path=case_folder,
        is_leaf=True,
        parameters=case_parameters,
        command_sets=command_sets,
    )
    cases.append(case)
    execute_osp_commands(cases)
    result = post_processing(cases)
    return result


def execute_osp_commands(cases):
    global command_sets

    _ = create_case_folders(cases)
    _ = create_param_dict_files(cases)

    _ = execute_command_set(
        cases=cases,
        command_set="prepare",
    )

    _ = execute_command_set(
        cases=cases,
        command_set="run",
    )

    _ = execute_command_set(
        cases=cases,
        command_set="post",
    )


def post_processing(cases):
    mapping: Dict[str, Dict[str, Any]] = {
        "Desired_Position": {  # column name you want to map a variable to
            "key": "reference_model|Desired_Position[1]:stdev",  # variable in FMU
            "unit": 1,  # usually 1, unless you want to apply a scaling factor
        },
        "Vessel_Position": {  # column name you want to map a variable to
            "key": "vessel_model|Vessel_Position[1]:stdev",  # variable in FMU
            "unit": 1,  # usually 1, unless you want to apply a scaling factor
        }
    }

    # column names
    names: List[str] = [name for name in mapping if not re.search("(^_|COMMENT)", name)]

    series: Dict[str, Series] = {
        "path": Series(data=None, dtype=dtype(str), name="path"),
    }

    for index in range(len(cases)):

        case = cases[index]
        case_folder: Path = case.path
        result_folder: Path = case_folder / "results"
        result_dict_file: Path = result_folder / "watchDict-OspSystemStructure-resultDict"  # Output of watchCosim

        series["path"].loc[index] = str(relative_path(Path.cwd(), case_folder))

        result_dict = DictReader.read(result_dict_file, includes=False, comments=False)

        for name in names:
            value: Any = None
            value_eval_string = "result_dict['" + "']['".join(mapping[name]["key"].split(":")) + "']"
            try:
                value = eval(value_eval_string)
            except Exception:
                logger.warning(f'"{value_eval_string}" not in {result_dict_file}')
                continue

            if name not in series:
                series[name] = Series(data=None, dtype=dtype(type(value)), name=name)

            if value is not None:
                series[name].loc[index] = value

    df: DataFrame = DataFrame(data=series)
    # df.set_index("path", inplace=True)  # optional. Makes the 'path' column the DataFrame's index

    # TODO: return max distance b/w targeted position and actual position over the simulation using above columns
    # Calculate trigonometric distance or STL or max deviation as metrics
    # Current using stdev
    result = norm(df['Desired_Position'] - df['Vessel_Position'])
    return result
