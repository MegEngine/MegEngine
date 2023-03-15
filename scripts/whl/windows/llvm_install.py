#-*-coding:utf-8-*-
"""
    Install the llvm.
"""
import os
import subprocess
os.environ["PATH"] = r"C:\Program Files\7-Zip;"+os.environ["PATH"]

class LLVMInstaller:
    def __init__(self, install_path="./llvm") -> None:
        self.install_path = install_path
        self.download_url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/LLVM-12.0.1-win64.exe"
        self.pakage_name = "LLVM-12.0.1-win64.exe"
    
    def setup(self):
        download_url = ["curl.exe", "-SL", self.download_url, "--output", self.pakage_name]
        if not os.path.exists(self.pakage_name):
            subprocess.run(download_url)
        else:
            print("The cmake package {} is exists, skip download".format(self.pakage_name))
        
        setup_cmd = ["7z", "x", f"-o{self.install_path}", self.pakage_name]
        subprocess.run(setup_cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("LLVM install procedure")
    parser.add_argument("--install_path", type=str, default="./llvm_tool", 
                        help="the path to install the cmake")
    args=parser.parse_args()
    llvm = LLVMInstaller(args.install_path)
    llvm.setup()