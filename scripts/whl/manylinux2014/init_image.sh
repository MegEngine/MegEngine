#!/bin/bash -e

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
python3 -m pip install "cython<3.0" -i https://mirrors.aliyun.com/pypi/simple
python3 -m pip install numpy -i https://mirrors.aliyun.com/pypi/simple

ALL_PYTHON="36m 37m 38 39 310"
for ver in ${ALL_PYTHON}
do
    python_ver=`echo $ver | tr -d m`
    numpy_version="1.19.5"
    if [ ${ver} = "310" ];then
        numpy_version="1.21.6"
    fi
    /opt/python/cp${python_ver}-cp${ver}/bin/pip install \
    --no-cache-dir --only-binary :all: numpy==${numpy_version} setuptools==46.1.3 \
    -i https://mirrors.aliyun.com/pypi/simple
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
