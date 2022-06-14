# MATLAB Installation for STGEM

* Go to <https://mathworks.com/products/get-matlab.html>, click download, and create a MathWorks account using your university email.
* Download MATLAB for Linux and extract it.
* Run `./install` as root. If the installer does not appear, run `xhost +SI:localuser:root` prior to `./install`. See [0].
* Follow the installer's instructions.
* Make sure to install all Simulink packages.
* Certain benchmarks need certain toolboxes:
    * Control System Toolbox: F16 (ARCH-COMP-2021)
    * Deep Learning Toolbox: SC (ARCH-COMP-2021)
* Make sure to select symbolic link creation in the installer.
* After the installation has completed, run `python3 setup.py install` in `/usr/local/MATLAB/R2021b/extern/engines/python` (this is the default installation location; note version differences).
* Make sure the package `libpython3` is installed in Ubuntu.
* If everything installed correctly, running `import matlab; import matlab.engine` in Python should execute without errors.
* Notice that MATLAB support for Linux is flaky. For example, running several MATLAB instances in parallel computation might lead to silent crashes.

[0] <https://mathworks.com/matlabcentral/answers/1464434-why-is-the-linux-matlab-install-script-not-opening-the-installer-window>

# TLTk for Robustness-Based Falsification
We use the library TLTk for computing values of robustness functions. This library is available at [0]. However, we needed to do some in-house modifications to the code, so the fork available at [1] should be installed. The installation is done as follows on Ubuntu.

* Install the Ubuntu package `build-essential` for compilation support.
* Install the Ubuntu package `python3-dev` for Python headers.
* Clone the repository `git clone https://gitlab.abo.fi/japeltom/tltk`.
* `cd tltk/robustness`
* Use `pip3 install -e .` to install TLTk locally.

[0] <https://bitbucket.org/versyslab/tltk/>
[1] <https://gitlab.abo.fi/japeltom/tltk>

