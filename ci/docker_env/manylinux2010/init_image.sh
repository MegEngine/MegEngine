#!/bin/bash -e

GET_PIP_URL='https://bootstrap.pypa.io/get-pip.py'
SWIG_URL='https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz?use_mirror=autoselect'
LLVM_URL='https://github.com/llvm-mirror/llvm/archive/release_60.tar.gz' 
CLANG_URL='https://github.com/llvm-mirror/clang/archive/release_60.tar.gz'

yum erase -y cmake cmake28
yum install -y python34-pip pcre-devel devtoolset-8-libatomic-devel.x86_64

pip3 install --no-cache-dir --only-binary :all: -U pip==19.1
pip3 install --no-cache-dir --only-binary :all: cmake==3.16.3

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
    curl -sSL https://github.com/NixOS/patchelf/archive/0.10.tar.gz | tar xz
    pushd /tmp/patchelf-0.10 >/dev/null
        patch -p1 <<'EOF'
diff --git a/src/patchelf.cc b/src/patchelf.cc
index 0b4965a..7aae7a4 100644
--- a/src/patchelf.cc
+++ b/src/patchelf.cc
@@ -1074,13 +1074,6 @@ void ElfFile<ElfFileParamNames>::modifySoname(sonameMode op, const std::string &
         return;
     }

-    /* Zero out the previous SONAME */
-    unsigned int sonameSize = 0;
-    if (soname) {
-        sonameSize = strlen(soname);
-        memset(soname, 'X', sonameSize);
-    }
-
     debug("new SONAME is '%s'\n", newSoname.c_str());

     /* Grow the .dynstr section to make room for the new SONAME. */
@@ -1264,7 +1257,6 @@ void ElfFile<ElfFileParamNames>::modifyRPath(RPathOp op,
     unsigned int rpathSize = 0;
     if (rpath) {
         rpathSize = strlen(rpath);
-        memset(rpath, 'X', rpathSize);
     }

     debug("new rpath is '%s'\n", newRPath.c_str());

EOF
        ./bootstrap.sh && ./configure && make install-strip
    popd
    rm -rf /tmp/patchelf-0.10
popd

yum clean all
