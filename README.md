### Code for BSc thesis - "Acceleration of CVA calculations using Futhark"
First use futhark pkg to install the required packages, by using

    futhark pkg sync
To compile the two versions and run a test dataset, use the command:

    Make clean all 	
Alternatively, make only a single version by using make with target:

    Make cva-map.exe
Or 

    Make cva-expand.exe

To benchmark the implementations, futhark-bench can be used as following:

    futhark bench --backend=opencl -e test cva-map.fut

