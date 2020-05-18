#!/usr/bin/env bash
set -e

MAKEFILE_TYPE="Unix"
OS=$(uname -s)

if [[ $OS =~ "NT" ]]; then
    echo "BUILD in NT ..."
    MAKEFILE_TYPE="Unix"
fi

function build_flatc() {
    BUILD_DIR=$1/build_dir/host_flatc/build
    INSTALL_DIR=$BUILD_DIR/../install
    if [ -e $BUILD_DIR ];then
        echo "clean old dir: $BUILD_DIR"
        rm -rf $BUILD_DIR
    fi
    if [ -e $INSTALL_DIR ];then
        echo "clean old dir: $INSTALL_DIR"
        rm -rf $INSTALL_DIR
    fi

    echo "create build dir"
    mkdir -p $BUILD_DIR
    mkdir -p $INSTALL_DIR
    cd $BUILD_DIR
    cmake -G "$MAKEFILE_TYPE Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_BUILD_FLATHASH=OFF \
        -DFLATBUFFERS_BUILD_FLATLIB=OFF \
        -DFLATBUFFERS_LIBCXX_WITH_CLANG=OFF \
        $SRC_DIR/third_party/flatbuffers

    make -j$(nproc)
    make install/strip
}
