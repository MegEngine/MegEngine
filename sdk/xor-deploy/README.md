# Example to deploy a MegEngine model using C++

* Step 1: compile MegEngine for deployment following [README.md](../../README.md)

* Step 2: compile the example by

    ```
    $CXX -o xor_deploy -I$MGE_INSTALL_PATH/include  xor_deploy.cpp -L$MGE_INSTALL_PATH/lib64/ -lmegengine
    ```

    where `$CXX` is the C++ compiler and `$MGE_INSTALL_PATH` is the MegEngine install path.

* Step 3: run with dumped model

     The dumped model can be obtained by running [xornet.py](xornet.py)


    ```
    LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib64:$LD_LIBRARY_PATH ./xor_deploy xornet_deploy.mge 0.6 0.9
    ```

    Sample output:

    ```
    Predicted: 0.999988 1.2095e-05
    ```

