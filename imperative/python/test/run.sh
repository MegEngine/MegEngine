#!/bin/bash -e

test_dirs="megengine test"

TEST_PLAT=$1

if [[ "$TEST_PLAT" == cpu ]]; then
    echo "only test cpu pytest"
elif [[ "$TEST_PLAT" == cuda ]]; then
    echo "test both cpu and gpu pytest"
else
    echo "Argument must cpu or cuda"
    exit 1
fi

pushd $(dirname "${BASH_SOURCE[0]}")/.. >/dev/null
    PYTHONPATH="." PY_IGNORE_IMPORTMISMATCH=1 python3 -m pytest $test_dirs -m 'not isolated_distributed'
    if [[ "$TEST_PLAT" == cuda ]]; then
        echo "test GPU pytest now"
        PYTHONPATH="." PY_IGNORE_IMPORTMISMATCH=1 python3 -m pytest $test_dirs -m 'isolated_distributed'
    fi
popd >/dev/null
