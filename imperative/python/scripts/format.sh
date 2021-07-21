#!/usr/bin/env bash
set -e
cd $(dirname $0)/..

ISORT_ARG=""
BLACK_ARG=""

while getopts 'd' OPT; do
    case $OPT in
        d)
            ISORT_ARG="--diff --check-only"
            BLACK_ARG="--diff --check"
            ;;
        ?)
            echo "Usage: `basename $0` [-d]"
    esac
done

directories=(megengine test)
if [[ -d examples ]]; then
    directories+=(examples)
fi
# do not isort megengine/__init__.py file, caused we must
# init library load path before load dependent lib in core
isort $ISORT_ARG -j $(nproc) -rc "${directories[@]}" -s megengine/__init__.py
black $BLACK_ARG --target-version=py35 -- "${directories[@]}"
