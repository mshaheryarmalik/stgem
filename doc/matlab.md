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
* If everything installed correctly, running `import matlab; import matlab.engine` in Python should execute without errors.

[0] <https://mathworks.com/matlabcentral/answers/1464434-why-is-the-linux-matlab-install-script-not-opening-the-installer-window>
      
