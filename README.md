# PubHunt
_Hunt for Bitcoin public keys._

## It searches for random compressed public keys for given hash160.

#
# The idea to do this

This is only useful for Bitcoin [puzzle transaction](https://www.blockchain.com/btc/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15).

For the puzzles ```64```, ```66```, ```67```, ```68```, ```69```, ```71``` or ```72``` and some more, there are no public keys are available, so if we can able to find public keys for those addresses, then [Pollard Kangaroo](https://github.com/JeanLucPons/Kangaroo) algorithm can be used to solve those puzzles.

That's it, cheers 🍺 

# Usage

This is GPU only, no CPU support. 

```
PubHunt.exe -h
PubHunt [-check] [-h] [-v]
        [-gi GPU ids: 0,1...] [-gx gridsize: g0x,g0y,g1x,g1y, ...]
        [-o outputfile] [inputFile]

 -v                       : Print version
 -gi gpuId1,gpuId2,...    : List of GPU(s) to use, default is 0
 -gx g1x,g1y,g2x,g2y, ... : Specify GPU(s) kernel gridsize, default is 8*(MP number),128
 -o outputfile            : Output results to the specified file
 -l                       : List cuda enabled devices
 -check                   : Check Int calculations
 inputFile                : List of the hash160, one per line in hex format (text mode)
```

For example:
```
PubHunt.exe -gi 0 -gx 8192,1024 hashP64.txt

PubHunt v1.00

DEVICE       : GPU
GPU IDS      : 0
GPU GRIDSIZE : 8192x1024
NUM HASH160  : 1
OUTPUT FILE  : Found.txt
GPU          : GPU #0 GeForce GTX 1650 (14x64 cores) Grid(8192x1024)

[00:01:43] [GPU: 533.23 MH/s] [T: 54,475,620,352 (36 bit)] [F: 0]
```

## Building
##### Windows
- Microsoft Visual Studio Community 2019 
- CUDA version 10.0
##### Linux
 - Edit the makefile and set up the appropriate CUDA SDK and compiler paths for nvcc. Or pass them as variables to `make` command.

    ```make
    CUDA       = /usr/local/cuda-11.0
    CXXCUDA    = /usr/bin/g++
    ```
 - To build with CUDA: pass CCAP value according to your GPU compute capability
    ```sh
    $ make CCAP=35 all
    ```

## License
PubHunt is licensed under GPLv3.

## Disclaimer
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.

