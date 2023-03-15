#!/bin/bash -e

cd $(dirname $0)

#FIXME: anaconda just upload serval lastest version, so this version may lose efficacy
echo "this script only for linux/macos/windows-unix-like-env(MSYS etc) prepare MKL env"
echo "if you build windows for native at cmd.exe, powershell env or Visual Studio GUI,"
echo  "u need download MKL package and untar manually"
echo "refs: https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download/windows.html"

OS=$(uname -s)
TAR=tar

if [[ -z ${MKL_VERSION} ]];then
    MKL_VERSION=2019.5
    MKL_PATCH=281
fi
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

if [[ -z ${FTP_URL_PREFIX} ]];then
    DOWNLOAD_FILE='${package}-${MKL_VERSION}-intel_${MKL_PATCH}.tar.bz2'
    URL='https://anaconda.org/intel/${package}/${MKL_VERSION}/download/$FILE_PREFIX-${platform}/'${DOWNLOAD_FILE}
    #if you can not download the file from anaconda.org, you can uncommit this URL to download it from the mirror supported by the CRA of SUStech
    #URL='https://mirrors.sustech.edu.cn/anaconda/cloud/intel/$FILE_PREFIX-${platform}/'${DOWNLOAD_FILE} 
else
    DOWNLOAD_FILE='${package}.tar.bz2'
    URL='$FTP_URL_PREFIX/$FILE_PREFIX-${platform}-'${DOWNLOAD_FILE}
fi

for platform in 32 64
do
    if [ $OS = "Darwin" ]&&[ $platform = 32 ];then
        echo "strip 32 bit file for Darwin"
        continue
    fi
    mkdir -p mkl/x86_${platform}
    for package in "mkl-include" "mkl-static"
    do
        echo "Installing $(eval echo $DOWNLOAD_FILE) for x86_${platform}..."
        echo "try download mkl package from: $(eval echo $URL)"
        if [ ${FILE_PREFIX} == "win" ]; then
            curl -SL "$(eval echo $URL)" --output mkl/x86_${platform}/"$(eval echo $DOWNLOAD_FILE)"
        else
            wget -q --show-progress "$(eval echo $URL)" -O mkl/x86_${platform}/"$(eval echo $DOWNLOAD_FILE)"
        fi
        $TAR xvj -C mkl/x86_${platform} -f mkl/x86_${platform}/"$(eval echo $DOWNLOAD_FILE)"
    done
done
