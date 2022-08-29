## Setup environment

To setup your local dev environment, you will need the following tools.

1.  [git](https://github.com/) for code repository management.
2.  [python](https://www.python.org/) to build and code in Keras.

The following commands checks the tools above are successfully installed. Note
that stgem requires at least Python 3.7 to run.

```shell
git --version
python --version
```

A [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
(venv) is a powerful tool to create a self-contained environment that isolates
any change from the system level config. It is highly recommended to avoid any
unexpected dependency or version issue.

With the following commands, you create a new venv, named `venv_dir`.

```shell
mkdir venv_dir
python3 -m venv venv_dir
```

You can activate the venv with the following command. You should always run the
tests with the venv activated. You need to activate the venv every time you open
a new shell.

```shell
source venv_dir/bin/activate  # for linux or MacOS
venv_dir\Scripts\activate.bat  # for Windows
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies into the venv. 

```shell
git clone https://gitlab.abo.fi/stc.git
cd stgem
pip install -r requirements.txt
````
```

The environment setup is completed. 

### Install MATLAB

Installing MATLAB is only necessary if you want to use a SUT that uses MATLAB. If this is the case, you should install MATLAB in your computer and obtain a license for it. Then you should install MATLAB support for your Python interpreter following the instructions from:

https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

After that, you can check your setup with these Python commands:

```python
import matlab.engine
eng = matlab.engine.start_matlab()
```

### Run tests

You can use pytest to run all the tests automatically:

```shell
pip install pytest
cd test
pytest
```

## How to contribute code

Follow these steps to submit your code contribution.


### Step 1. Open an issue, Create a merge request

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

1. Select the gitlab issue that you plan to work on. If there is no issue, create one.
2. At the gitlab issue page, `assign yourself` the issue, if this is not yet the case.
3. At the gitlab issue page, `Create merge request`. You can do that even before you starting coding. This will create a git branch to work with the issue.


### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in section
"Setup environment".

4. Work on your issue. Commit often and write describe commit messages You should commit at least at the end of each working day, even if the code is not complete. This way you will get constant feedback on your task. 
5. If you have questions or comments it is better to write them in the gitlab issue page. This way you will get a detailed answer in written.
6. Always document your code and write automatic tests. Tests in the top-level `tests` folders are executed automatically by the continuous integration system. You can see the reports in gitlab, CI/CD, Pipelines page.
7. You should watch for changes in the main branch, specially if you work in the same branch for many days. Merge updates from main to your branch often. This way you simplify the  integration work in the future.


### Step 3. Mark the merge request as ready 

8. Once your task is ready, `Mark as ready` the merge request. 

A reviewer will review the pull request and provide comments.  There may be several rounds of comments and code changes before merge pull request gets
approved by the reviewer. 

### Step 4. Merge into Main

Once the merge request is approved, the reviewer will take care of the merging.

## How to report errors

If you want to report an error please make sure that your report contains the following information:

1. Where to find your code for example a link to a gitlab commit description page.
2. The command that you are running from your shell. If the error seems to be related to missing files or modules or module import errors, report the directory from where you are running that command and your `PYTHONPATH` environment variable.
3. The error stack trace and / or relevant program output. Use a format that is easy to read. In Markdown you can use triple quotes.
 
This information is really useful for us to reproduce the errors and try to solve it.



