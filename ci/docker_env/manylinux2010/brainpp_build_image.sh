#!/bin/bash

cd $(dirname $0)

docker build -t env_manylinux2010:latest . \
    --build-arg platform="brainpp" \
    --build-arg http_proxy="http://proxy.i.brainpp.cn:3128" \
    --build-arg https_proxy="http://proxy.i.brainpp.cn:3128" \
    --build-arg no_proxy="brainpp.cn,.brainpp.ml,.megvii-inc.com,.megvii-op.org,127.0.0.1,localhost"
