#!/bin/bash -e

ALL_PYTHON="3.6.10 3.7.7 3.8.3 3.9.4 3.10.1"
SWIG_URL='https://codeload.github.com/swig/swig/tar.gz/refs/tags/rel-3.0.12'

NINJA_URL='https://codeload.github.com/ninja-build/ninja/tar.gz/refs/tags/v1.10.0'
for ver in ${ALL_PYTHON}
do
    numpy_version="1.19.5"
    if [ ${ver} = "3.10.1" ];then
        numpy_version="1.21.6"
    fi
    echo "Install python ${ver}"
    env PYTHON_CONFIGURE_OPTS="--enable-shared" ~/.pyenv/bin/pyenv install ${ver}
    ~/.pyenv/versions/${ver}/bin/python3 -m pip install --upgrade pip
    ~/.pyenv/versions/${ver}/bin/python3 -m pip install numpy==${numpy_version} setuptools==46.1.3 wheel
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
echo "Install patchelf"
curl -sSL https://github.com/NixOS/patchelf/archive/0.12.tar.gz | tar xz
pushd /tmp/patchelf-0.12 >/dev/null
sed -i '331s/32/256/' ./src/patchelf.cc
./bootstrap.sh && ./configure && make install-strip
popd
rm -rf /tmp/patchelf-0.12
popd
