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
    REMOVE_OLD_BUILD=$2

    if [ $REMOVE_OLD_BUILD = "true" ]; then
        echo "remove old build/install dir"
        rm -rf $INSTALL_DIR
        rm -rf $BUILD_DIR
    else
        echo "strip remove old build"
    fi

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

function try_remove_old_build() {
    REMOVE_OLD_BUILD=$1
    echo $REMOVE_OLD_BUILD
    BUILD_DIR=$2
    INSTALL_DIR=$3

    if [ $REMOVE_OLD_BUILD = "true" ]; then
        echo "remove old build/install dir"
        rm -rf ${BUILD_DIR}
        rm -rf ${INSTALL_DIR}
    else
        echo "strip remove old build"
    fi
}
