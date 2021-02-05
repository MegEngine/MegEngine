#!/bin/bash -e

GET_PIP_URL='https://bootstrap.pypa.io/get-pip.py'
SWIG_URL='https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz?use_mirror=autoselect'
LLVM_URL='https://github.com/llvm-mirror/llvm/archive/release_60.tar.gz' 
CLANG_URL='https://github.com/llvm-mirror/clang/archive/release_60.tar.gz'


yum install -y pcre-devel devtoolset-9-libatomic-devel.x86_64
yum install -y devtoolset-8 devtoolset-8-libatomic-devel.x86_64

for ver in 35m 36m 37m 38 
do
    python_ver=${ver:0:2}
    curl ${GET_PIP_URL} | /opt/python/cp${python_ver}-cp${ver}/bin/python - \
	--no-cache-dir --only-binary :all:
    /opt/python/cp${python_ver}-cp${ver}/bin/pip install \
	--no-cache-dir --only-binary :all: numpy==1.18.1 setuptools==46.1.3
done

pushd /home >/dev/null
    echo "Install swig"
    curl -sSL ${SWIG_URL} | tar xz
    pushd swig-3.0.12 >/dev/null
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
popd >/dev/null

pushd /tmp >/dev/null
    curl -sSL https://github.com/NixOS/patchelf/archive/0.12.tar.gz | tar xz
    pushd /tmp/patchelf-0.12 >/dev/null
        sed -i '331s/32/64/' ./src/patchelf.cc
        ./bootstrap.sh && ./configure && make install-strip
    popd
    rm -rf /tmp/patchelf-0.12
popd

yum clean all
