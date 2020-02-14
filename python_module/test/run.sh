#!/bin/bash

cd $(dirname "${BASH_SOURCE[0]}")/..

pytest -m 'not internet' \
    --ignore test/pytorch_comparison \
    --ignore test/integration/manual \
    --ignore megengine/docs \
    megengine test
