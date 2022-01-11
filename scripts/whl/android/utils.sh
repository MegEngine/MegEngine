#!/bin/bash -e

source ${SRC_DIR}/scripts/whl/utils/utils.sh

ALL_PYTHON=${ALL_PYTHON}
# FIXME: now imperative py code do not support 3.10
# but megenginelite and megbrain support it, so we
# config with 3.10.1 now
FULL_PYTHON_VER="3.8.3 3.9.9 3.10.1"
if [[ -z ${ALL_PYTHON} ]]
then
    ALL_PYTHON=${FULL_PYTHON_VER}
else
    check_python_version_is_valid "${ALL_PYTHON}" "${FULL_PYTHON_VER}"
fi

# FIXME python3.10+ self build have some issue, need config env
# _PYTHON_SYSCONFIGDATA_NAME remove this env after find the build issue
# do not care about this, apt install python3.10 do not have this issue
export _PYTHON_SYSCONFIGDATA_NAME="_sysconfigdata__linux_aarch64-linux-android"

function check_termux_env() {
    echo "check is in termux env or not"
    info=`command -v termux-info || true`
    if [[ "${info}" =~ "com.termux" ]]; then
        echo "find termux-info at: ${info}"
        echo "check env now"
        ENVS="PREFIX HOME"
        for check_env in ${ENVS}
        do
            echo "try check env: ${check_env}"
            if [[ "${!check_env}" =~ "termux" ]]; then
                echo "env ${check_env} is: ${!check_env}"
            else
                echo "invalid ${check_env} env, may broken termux env"
                exit -1
            fi
        done
    else
        echo "invalid env, only support build android whl at termux env, please refs to: scripts/whl/BUILD_PYTHON_WHL_README.md to init env"
        exit -1
    fi
}
