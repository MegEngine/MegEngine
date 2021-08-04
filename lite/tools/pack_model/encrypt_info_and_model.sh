#!/usr/bin/env bash
set -e

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-i info.json : input info.json file"
    echo "-m model: model name"
    echo "-e encryption mode: encryption mode rc4 encrypt_predefined_rc4 "
    echo "-o output name: output name"
    echo "-n input model name: input model name match with info.json"
    echo "-h : show usage"
    exit -1
}

while getopts "i:m:e:o:n:h" arg
do
    case $arg in
        i)
            INFO_NAME=$OPTARG
            ;;
        m)
            MODEL_NAME=$OPTARG
            ;;
        n)
            INPUT_MODEL_NAME=$OPTARG
            ;;
        e)
            ENCRYPT_MODE=$OPTARG
            ;;
        o)
            OUTPUT_NAME=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "show usage"
            usage
            ;;
    esac
done
echo "----------------------------------------------------"
echo "commad args summary:"
echo "INFO_NAME: $INFO_NAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "ENCRYPT_MODE: $ENCRYPT_MODE"
echo "OUTPUT_NAME: $OUTPUT_NAME"
echo "INPUT_MODEL_NAME: $INPUT_MODEL_NAME"
echo "----------------------------------------------------"

if [[ $INFO_NAME == '' ]]; then
    echo "INFO_NAME is NULL,exit now..."
    exit -1
fi
if [[ $MODEL_NAME == '' ]]; then
    echo "MODEL_NAME is NULL,exit now..."
    exit -1
fi
if [[ $INPUT_MODEL_NAME == '' ]]; then
    echo "INPUT_MODEL_NAME is NULL,exit now..."
    exit -1
fi
if [[ $OUTPUT_NAME == '' ]]; then
    echo "OUTPUT_NAME is NULL,exit now..."
    exit -1
fi
ENCRYPT_INFO_NAME=$INFO_NAME.pr_rc4.emod
ENCRYPT_MODEL_NAME=$MODEL_NAME.pr_rc4.emod
./rc4_encryptor $ENCRYPT_MODE $INFO_NAME $INFO_NAME.pr_rc4.emod
./rc4_encryptor $ENCRYPT_MODE $MODEL_NAME $MODEL_NAME.pr_rc4.emod


ENCRYPT_INFO_NAME=$INFO_NAME.pr_rc4.emod
python3 pack_model_and_info.py --input-model=$ENCRYPT_MODEL_NAME --model-name=$INPUT_MODEL_NAME --model-cryption="RC4_default" --info-cryption="RC4_default" --input-info=$ENCRYPT_INFO_NAME --info-parser="LITE_default" -o $OUTPUT_NAME
