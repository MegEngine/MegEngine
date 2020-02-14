#!/bin/bash -e

cd $(dirname $0)

MKL_VERSION=2019.5
MKL_PATCH=281
CONDA_BASE_URL=https://anaconda.org/intel

rm -rf mkl

for platform in 32 64
do
    mkdir -p mkl/x86_${platform}
    for package in "mkl-include" "mkl-static"
    do
        echo "Installing ${package} for x86_${platform}..."
        URL=${CONDA_BASE_URL}/${package}/${MKL_VERSION}/download/linux-${platform}/${package}-${MKL_VERSION}-intel_${MKL_PATCH}.tar.bz2
        wget -q --show-progress "${URL}" -O - | tar xj -C mkl/x86_${platform}
    done
done
