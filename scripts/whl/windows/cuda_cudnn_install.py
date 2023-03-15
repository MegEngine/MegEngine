#-*- coding:utf-8 -*-
"""
    Implementation based on the script of conda
    Reference:https://github.com/numba/conda-recipe-cudatoolkit/blob/master/scripts/build.py
    Nvidia Developer Site: https://developer.nvidia.com
"""
import os
import subprocess
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory as tempdir
from distutils.dir_util import copy_tree
os.environ["PATH"] = r"C:\Program Files\7-Zip;"+os.environ["PATH"]
#
config = {}
config["cu112"] = {
    "version": "11.2.0",
    "driver":"460.89",
    "cudnn_name":"cudnn-8.2.1.32-hae0fe6e_0.tar.bz2"
}

config["cu118"] = {
    "version":"11.8.0", 
    "driver": "522.06",
    "cudnn_name": ""
}

config["cu114"]={
    "version":"11.4.0",
    "driver":"471.11", 
    "cudnn_name": "cudnn-8.2.1.32-hae0fe6e_0.tar.bz2"
}

config["cu110"]={
    "version":"11.1.0",
    "driver":"456.43",
    "cudnn_name": "cudnn-8.0.5.39-h36d860d_1.tar.bz2"
}

config["cu101"]={
    "version":"10.1.105",
    "driver":"418.96",
    "cudnn_name": "cudnn-7.6.5.32-h36d860d_1.tar.bz2"
}

class BaseExtracter:
    def __init__(self, sdk_name, install_path) -> None:
        #you can change .com to .cn, if you cannot download it from nvidia.com
        cuda_base_url = "https://developer.download.nvidia.com/compute/cuda/{}/local_installers/cuda_{}_{}_win10.exe"
        cuda_base_name = "{}_{}_win10.exe"
        if sdk_name == "cu118":
            cuda_base_url="https://developer.download.nvidia.com/compute/cuda/{}/local_installers/cuda_{}_{}_windows.exe"
            cuda_base_name = "cuda_{}_{}_windows.exe"
        self.config = config[sdk_name]
        version = self.config["version"]
        driver = self.config["driver"]
        self.cuda_download_url = cuda_base_url.format(version, version, driver)
        self.install_path = install_path
        self.package_name = cuda_base_name.format(version, driver)
        #We use the mirror site provided by the CRA of SUSTech to download the cudnn, you can change it.
        cudnn_base_url = "https://mirrors.sustech.edu.cn/anaconda/cloud/conda-forge/win-64/"
        self.cudnn_download_url = cudnn_base_url+self.config["cudnn_name"]
        
    
    def extract(self):
        raise NotImplementedError

class CudaExtracter(BaseExtracter):
    def __init__(self, sdk_name, install_path="./cuda") -> None:
        super(CudaExtracter, self).__init__(sdk_name, install_path)
        
    
    def extract(self):
        download_cmd = ["curl.exe", "-SL", "-o", self.package_name, self.cuda_download_url]
        if not os.path.isfile(self.package_name):
            print("Try to download CUDA {} from {}".format(self.package_name, self.cuda_download_url))
            subprocess.run(download_cmd)
        else:
            print("Setup file {} is exists, skip downloading".format(self.package_name))
        with tempdir() as tmpdir:
            cmd = ["7z", "x", f"-o{tmpdir}", self.package_name]
            subprocess.run(cmd, env=os.environ.copy(), check=True)
            target_dir = self.install_path
            nvcc_dir = os.path.join(target_dir, "nvcc")
            toolkitpath = tmpdir
            # ignore=shutil.ignore_patterns('*.nvi') 
            for toolkitpathroot, subdirs, files in os.walk(toolkitpath):
                for file in files:
                    src_file = os.path.join(toolkitpathroot, file)
                    os.chmod(src_file, 0o777)
                for subdir in subdirs:
                    if subdir in ['CUDAVisualStudioIntegration'] and (subdir not in Path(toolkitpathroot).parts ):
                        src = os.path.join(toolkitpathroot, subdir)
                        dst = os.path.join(target_dir, subdir)
                        copy_tree(src, dst)
                    elif subdir in ['bin','include','lib','extras','libdevice','nvvm'] and (subdir not in Path(toolkitpathroot).parts ):
                        src = os.path.join(toolkitpathroot, subdir)
                        nvcc_dst = os.path.join(nvcc_dir, subdir)
                        copy_tree(src, nvcc_dst)
            os.remove(self.package_name)
            
class CudnnExtracter(BaseExtracter):
    def __init__(self, sdk_name, install_path="./cudnn") -> None:
        super(CudnnExtracter, self).__init__(sdk_name, install_path)
        
    
    def extract(self):
        if self.config["version"] == "11.8.0":
            print("The cudnn for cudatoolkit-11.8 is not be supported now, please download the cudnn-8.6"\
                  "to the install directory:{} manually".format(self.install_path))
            return
        output_name = self.cudnn_download_url.split("/")[-1]
        print(output_name)
        download_cmd = ["curl.exe", "-SL", "-o", output_name, self.cudnn_download_url]
        if not os.path.isfile(output_name):
            print("Try to download cudnn from {}".format(self.cudnn_download_url))
            subprocess.run(download_cmd)
        else:
            print("Cudnn file {} is exists, skip downloading".format(self.package_name))
        tmp_path = os.path.join(self.install_path, output_name[:-4])
        cmd = ["7z", "x", f"-o{self.install_path}", output_name]
        subprocess.run(cmd)
        cmd = ["7z", "x", f"-o{self.install_path}", f"{self.install_path}/{output_name[:-4]}"]
        subprocess.run(cmd)
        os.remove(tmp_path)

        
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("LLVM install procedure")
    parser.add_argument("--sdk_name", type=str, default="cu112", 
                        help="cudatoolkit version")
    parser.add_argument("--cuda_path", type=str, default="./cuda_tool")
    parser.add_argument("--cudnn_path", type=str, default="./cuda_tool")
    args=parser.parse_args()
    sdk_name = args.sdk_name
    e = CudaExtracter(sdk_name=sdk_name, install_path=args.cuda_path)
    e.extract()
    x = CudnnExtracter(sdk_name=sdk_name, install_path=args.cudnn_path)
    x.extract()
    #print("test")