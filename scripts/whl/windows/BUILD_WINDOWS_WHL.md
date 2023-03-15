### Steps

+ Setup the 7-Zip (Install the program to C:/Program Files (x86) or other position. If you install it to other position, please copy the path and change the path in llvm_install.py and cuda_cudnn_install.py. The 7-zip can be downloaded at https://www.7-zip.org/).

+ Download the TensorRT from [home page](https://developer.nvidia.com/zh-cn/tensorrt) and extract it to `C:/tools`.

+ Clone the source from github.

  ```shell
  git clone https://github.com/MegEngine/MegEngine.git
  ```

+ Install the python by the script (Note: Please make sure the python tool:"3.6.8", "3.7.7", "3.8.3", "3.9.4" and "3.10.1" not installed in your PC.). You may get the error:`FullyQualifiedErrorId : UnauthorizedAccess.`, you can follow this [link](https://answers.microsoft.com/en-us/windows/forum/all/fullyqualifiederrorid-unauthorizedaccess/a73a564a-9870-42c7-bd5e-7072eb1a3136) to deal with it.

  ```powershell
  .\scripts\whl\windows\python_install.ps1
  ```

+ Install the Visual Studio Build Tool by the script.

  ```powershell
  .\scripts\whl\windows\vs_buildtool_install.ps1
  ```
  
+ Modify the TensorRT root path in build_whl.sh, or you can download the TensorRT 7.2.3.4 and extract it to `C:/tools` (PS: You can change the TRT_ROOT_DIR defined in build_whl.sh, so you can change the position of the TensorRT).

+ Build the MegEngine.

  ```shell
  ./scripts/whl/windows/build_whl.sh
  ```
*** Note:If you use the cu118 to build the whl, please install the cudnn manually. ***
