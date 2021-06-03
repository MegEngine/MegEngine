#!/bin/bash -e

GET_PIP_URL='https://bootstrap.pypa.io/get-pip.py'
GET_PIP_URL_35='https://bootstrap.pypa.io/pip/3.5/get-pip.py'
SWIG_URL='https://codeload.github.com/swig/swig/tar.gz/refs/tags/rel-3.0.12'
LLVM_URL='https://github.com/llvm-mirror/llvm/archive/release_60.tar.gz' 
CLANG_URL='https://github.com/llvm-mirror/clang/archive/release_60.tar.gz'
NINJA_URL='https://codeload.github.com/ninja-build/ninja/tar.gz/refs/tags/v1.10.0'


ARCH=$1
echo "ARCH: ${ARCH}"
yum install -y pcre-devel devtoolset-9-libatomic-devel.${ARCH}
yum install -y devtoolset-8 devtoolset-8-libatomic-devel.${ARCH}
# install a default python3 for cmake PYTHON3_EXECUTABLE_WITHOUT_VERSION
yum install -y python3 python3-devel
python3 -m pip install cython
python3 -m pip install numpy

ALL_PYTHON="35m 36m 37m 38"
numpy_version="1.18.1"
if [ ${ARCH} = "aarch64" ];then
    # numpy do not have 1.18.1 on aarch64 linux, so we use another fix version
    numpy_version="1.19.5"
fi
for ver in ${ALL_PYTHON}
do
    python_ver=${ver:0:2}
    PIP_URL=${GET_PIP_URL}
    if [ ${ver} = "35m" ];then
        PIP_URL=${GET_PIP_URL_35}
    fi
    echo "use pip url: ${PIP_URL}"
    curl ${PIP_URL} | /opt/python/cp${python_ver}-cp${ver}/bin/python - \
	--no-cache-dir --only-binary :all:
    if [ ${ARCH} = "aarch64" ] && [ ${ver} = "35m" ];then
        # aarch64 linux python3.5 pip do not provide binary package
        /opt/python/cp${python_ver}-cp${ver}/bin/pip install --no-cache-dir numpy setuptools==46.1.3
    else
        /opt/python/cp${python_ver}-cp${ver}/bin/pip install \
        --no-cache-dir --only-binary :all: numpy==${numpy_version} setuptools==46.1.3
    fi
done

pushd /home >/dev/null
    echo "Install swig"
    curl -sSL ${SWIG_URL} | tar xz
    pushd swig-rel-3.0.12 >/dev/null
        ./autogen.sh
        mkdir build
       	pushd build >/dev/null
	    ../configure
	    make -j$(nproc)
	    make install
        popd >/dev/null
    popd >/dev/null
    rm -rf swig-3.0.12
    
    echo "Install llvm"
    curl -sSL ${LLVM_URL} | tar xz
    pushd llvm-release_60 >/dev/null
        mkdir build
       	pushd build >/dev/null
            cmake .. -DCMAKE_PREFIX_PATH=/opt/python/cp36-cp36m/ \
		-DCMAKE_BUILD_TYPE=Release
	    make -j$(nproc)
	    make install
	popd >/dev/null
    popd >/dev/null
    rm -rf llvm-release_60

    echo "Install clang"
    curl -sSL ${CLANG_URL} | tar xz
    pushd clang-release_60 >/dev/null
        mkdir build
       	pushd build >/dev/null
            cmake .. -DCMAKE_PREFIX_PATH=/opt/python/cp36-cp36m/ \
                -DCMAKE_BUILD_TYPE=Release
	    make -j$(nproc)
	    make install
	popd >/dev/null
    popd >/dev/null
    rm -rf clang-release_60 
    echo "Install ninja build"
    curl -sSL ${NINJA_URL} | tar xz
    pushd ninja-1.10.0 >/dev/null
        mkdir build
	pushd build >/dev/null
            cmake .. -DCMAKE_BUILD_TYPE=Release
	    make -j$(nproc)
	    cp ninja /usr/bin/
	popd >/dev/null
    popd >/dev/null
    rm -rf ninja-1.10.0
popd >/dev/null

pushd /tmp >/dev/null
    curl -sSL https://github.com/NixOS/patchelf/archive/0.12.tar.gz | tar xz
    pushd /tmp/patchelf-0.12 >/dev/null
        sed -i '331s/32/256/' ./src/patchelf.cc
        ./bootstrap.sh && ./configure && make install-strip
    popd
    rm -rf /tmp/patchelf-0.12
popd

yum clean all
