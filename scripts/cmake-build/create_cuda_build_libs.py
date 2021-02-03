#!/usr/bin/env python3

import argparse
import os
import subprocess
import glob

def handle_cuda_libs(path):
    subprocess.check_call('rm -rf tmp && rm -rf tmp_sub', shell=True)
    print('\nhandle cuda file from.{}'.format(path))
    cmd = 'dpkg-deb -xv {} tmp'.format(path)
    subprocess.check_call(cmd, shell=True)
    sub_debs = glob.glob('tmp/**/*.deb', recursive=True)
    assert(len(sub_debs) > 0)
    for sub_deb in sub_debs:
        subprocess.check_call('rm -rf tmp_sub', shell=True)
        print('handle sub_deb: {}'.format(sub_deb))
        cmd = 'dpkg-deb -xv {} tmp_sub'.format(sub_deb)
        subprocess.check_call(cmd, shell=True)
        sub_sub_debs = glob.glob('tmp_sub/**/*.deb', recursive=True)
        assert(len(sub_sub_debs) == 0)
        if (os.path.isdir('tmp_sub/usr/share/')):
            subprocess.check_call('cp -v tmp_sub/usr/share/* output/ -rf', shell=True)
        if (os.path.isdir('tmp_sub/usr/local/')):
            subprocess.check_call('cp -v tmp_sub/usr/local/* output/ -rf', shell=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-s",
            "--sbsa_mode",
            action="store_true",
            help="create cuda sbsa libs, which means use to x86 cross build for aarch64 cuda libs",
            )

    parser.add_argument(
            "-t",
            "--target_aarch",
            type=str,
            choices=['x86-64', 'aarch64'],
            help="create libs build for, now support x86-64 and aarch64",
            dest="target_aarch",
            required=True,
            )

    parser.add_argument(
            "-d",
            "--cudnn_deb",
            help="cudnn deb package, download from: download from: https://developer.nvidia.com/cudnn-download-survey",
            dest="cudnn_deb",
            type=str,
            required=True,
            )

    parser.add_argument(
            "-r",
            "--trt_deb",
            help="trt deb package, download from: https://developer.nvidia.com/nvidia-tensorrt-download",
            dest="trt_deb",
            type=str,
            required=True,
            )

    parser.add_argument(
            "-c",
            "--cuda_deb",
            help="cuda deb package, download from: https://developer.nvidia.com/cuda-downloads",
            dest="cuda_deb",
            type=str,
            required=True,
            )

    parser.add_argument(
            "-a",
            "--cuda_aarch64_deb",
            help="cuda aarch64 libs package: download from: https://developer.nvidia.com/cuda-downloads",
            type=str,
            dest="cuda_aarch64_deb",
            )

    args = parser.parse_args()

    if (args.target_aarch == 'x86-64' and args.sbsa_mode):
        print('ERROR: sbsa_mode only support target_aarch = \'aarch64\' now')
        exit(-1)

    if (args.sbsa_mode and not args.cuda_aarch64_deb):
        print('ERROR: sbsa_mode need -a/--cuda_aarch64_deb to provide cuda aarch64 libs package')
        exit(-1)

    if (not os.path.isfile(args.cuda_deb)):
        print('ERROR: can not find file:{}'.format(args.cuda_deb))
        exit(-1)

    if (args.sbsa_mode and not os.path.isfile(args.cuda_aarch64_deb)):
        print('ERROR: can not find file:{}'.format(args.cuda_aarch64_deb))
        exit(-1)

    if (not os.path.isfile(args.cudnn_deb)):
        print('ERROR: can not find file:{}'.format(args.cudnn_deb))
        exit(-1)

    if (not os.path.isfile(args.trt_deb)):
        print('ERROR: can not find file:{}'.format(args.trt_deb))
        exit(-1)

    print("CONFIG SUMMARY: create cuda cmake build libs for {}, is for sbsa_mode: {}".format(args.target_aarch, args.sbsa_mode))

    cmd = 'rm -rf output && mkdir output'
    subprocess.check_call(cmd, shell=True)

    #handle cuda
    handle_cuda_libs(args.cuda_deb)

    #handle sbsa_mode
    if (args.sbsa_mode):
        handle_cuda_libs(args.cuda_aarch64_deb)

    # check cuda/sbsa_mode valid and handle link
    nvcc = glob.glob('./output/*/bin/nvcc', recursive=True)
    cuda_version = nvcc[0][9:-9]
    print('cuda version: {}'.format(cuda_version))
    assert(len(nvcc) == 1)
    if (args.sbsa_mode):
        subprocess.check_call('file {} | grep {}'.format(nvcc[0], 'x86-64'), shell=True)
        remove_x86_64_libs = ['targets/x86_64-linux', 'include', 'lib64']
        for remove_lib in remove_x86_64_libs:
            subprocess.check_call('rm -rf ./output/{}/{}'.format(cuda_version, remove_lib), shell=True)
        #create link for sbsa
        cwd = os.getcwd()
        os.chdir('output/{}'.format(cuda_version))
        cmd = 'ln -s targets/sbsa-linux/include/ include && ln -s targets/sbsa-linux/lib/ lib64'
        subprocess.check_call(cmd, shell=True)
        #handle libnvrtc.so
        readelf_nvrtc = os.popen('readelf -d lib64/stubs/libnvrtc.so | grep SONAME').read().split('\n')[0]
        loc = readelf_nvrtc.find('[')
        libnvrtc_with_version = readelf_nvrtc[loc+1:-1]
        print('libnvrtc_with_version: {}'.format(libnvrtc_with_version))
        cmd = 'cp lib64/stubs/libnvrtc.so lib64/{}'.format(libnvrtc_with_version)
        subprocess.check_call(cmd, shell=True)
        os.chdir(cwd)
    else:
        subprocess.check_call('file {} | grep {}'.format(nvcc[0], args.target_aarch), shell=True)

    # handle cudnn
    subprocess.check_call('rm -rf tmp && rm -rf tmp_sub && mkdir tmp', shell=True)
    print('\nhandle cuda file from.{}'.format(args.cudnn_deb))
    # FIXME: later release cudnn may dir not with cuda, nvidia may fix later!!
    cmd = 'tar -xvf {} -C tmp && mv tmp/cuda output/cudnn'.format(args.cudnn_deb)
    subprocess.check_call(cmd, shell=True)
    cudnn_libs = glob.glob('output/cudnn/lib64/libcudnn.so*')
    cudnn_real_libs = []
    for lib in cudnn_libs:
        if (not os.path.islink(lib)):
            cudnn_real_libs.append(lib)
    assert(len(cudnn_real_libs) > 0)
    for lib in cudnn_real_libs:
        subprocess.check_call('file {} | grep {}'.format(lib, args.target_aarch), shell=True)

    # handle trt
    print('\nhandle cuda file from.{}'.format(args.trt_deb))
    cmd = 'tar -xvf {} -C output'.format(args.trt_deb)
    subprocess.check_call(cmd, shell=True)
    trt_libs = glob.glob('output/TensorRT-*/lib/libnvinfer.so.*')
    trt_real_libs = []
    for lib in trt_libs:
        if (not os.path.islink(lib)):
            trt_real_libs.append(lib)
    assert(len(trt_real_libs) > 0)
    for lib in trt_real_libs:
        subprocess.check_call('file {} | grep {}'.format(lib, args.target_aarch), shell=True)

if __name__ == "__main__":
    main()
