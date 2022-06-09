    • Go to https://se.mathworks.com/downloads/web_downloads/
      
    • Create a MathWorks account, using your university email	
      
    • Click Download for Linux
      
    • Unzip
     
    • In the unzipped folder, do:
     
      xhost +SI:localuser:root
      sudo ./install 
            
    • Login to your MathWorks account
      
    • Select the license you want to use
      
    • Follow the installation wizard's instructions
      
    • Check all Simulink packages
      
    • Create symbolic links to MATLAB scrips in /usr/local/bin 
      
    • Install with pip:
        ◦ matlab
        ◦ Pillow
        ◦ matplotlib
      
    • Install with synaptic
        ◦ distutils
        ◦ libpython
      
    • In MATLAB/extern/engines/python/, do: sudo python3.8 -m setup.py install
      
    • move MATLAB/extern/engines/python/build/lib/matlab into your project folder, and import matlab.engine from here
      
    • import matlab and import PATH.matlab.engine should now be working
