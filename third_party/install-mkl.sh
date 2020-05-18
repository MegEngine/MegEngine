#!/bin/bash -e

cd $(dirname $0)

#FIXME: anaconda just upload serval lastest version, so this version may lose efficacy
echo "this script only for linux/macos/windows-unix-like-env(MSYS etc) prepare MKL env"
echo "if you build windows for native at cmd.exe, powershell env or Visual Studio GUI,"
echo  "u need download MKL package and untar manually"
echo "refs: https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download/windows.html"
MKL_VERSION=2019.5
MKL_PATCH=281
CONDA_BASE_URL=https://anaconda.org/intel
OS=$(uname -s)
FILE_PREFIX=null
TAR=tar

if [ $OS = "Darwin" ];then
    FILE_PREFIX=osx
elif [ $OS = "Linux" ];then
    FILE_PREFIX=linux
elif [[ $OS =~ "NT" ]]; then
    FILE_PREFIX=win
    # NT use /c/Windows/system32/tar will stuck for big file
    # so we back to GNU tar
    TAR=/usr/bin/tar
else
    echo "DO NOT SUPPORT OS NOW"
    exit -1
fi

echo "config FILE_PREFIX to: $FILE_PREFIX"

rm -rf mkl

for platform in 32 64
do
    if [ $OS = "Darwin" ]&&[ $platform = 32 ];then
        echo "strip 32 bit file for Darwin"
        continue
    fi
    mkdir -p mkl/x86_${platform}
    for package in "mkl-include" "mkl-static"
    do
        DOWNLOAD_FILE=${package}-${MKL_VERSION}-intel_${MKL_PATCH}.tar.bz2
        echo "Installing ${DOWNLOAD_FILE} for x86_${platform}..."
        URL=${CONDA_BASE_URL}/${package}/${MKL_VERSION}/download/$FILE_PREFIX-${platform}/${DOWNLOAD_FILE}
        echo "try download mkl package from: ${URL}"
        wget -q --show-progress "${URL}" -O mkl/x86_${platform}/${DOWNLOAD_FILE}
        $TAR xvj -C mkl/x86_${platform} -f mkl/x86_${platform}/${DOWNLOAD_FILE}
    done
done
