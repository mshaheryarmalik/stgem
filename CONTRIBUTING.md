## How to contribute code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the main branch in [aidoartt/stgem](https://gitlab.abo.fi/aidoart/stgem).

### Step 4. Sign the Contributor Agreement

TBA

### Step 5. Code review

A reviewer will review the pull request and provide comments. 

There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.


### Step 6. Merging

Once the pull request is approved, a team member will take care of the merging.


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
git clone https://gitlab.abo.fi/aidoart/stgem.git
cd stgem
pip install -r requirements.txt
```

The environment setup is completed. 


## Run tests

You can use pytest to run all the tests automatically:

```shell
pip install pytest
cd test
pytest
```
