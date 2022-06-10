# TLTk for Robustness-Based Falsification
We use the library TLTk for computing values of robustness functions. This library is available at [1]. However, we needed to do some in-house modifications to the code, so the fork available at [2] should be installed. The installation is done as follows on Ubuntu.

* Install the Ubuntu package `build-essential` for compilation support.
* Install the Ubuntu package `python3-dev` for Python headers.
* Clone the repository `git clone https://gitlab.abo.fi/japeltom/tltk`.
* `cd tltk/robustness`
* Use `pip3 install -e .` to install TLTk locally.

[1] https://bitbucket.org/versyslab/tltk/
[2] https://gitlab.abo.fi/japeltom/tltk

