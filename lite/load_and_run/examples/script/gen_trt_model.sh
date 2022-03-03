#!/bin/bash

CUR_DIR="$( cd "$(dirname $0)" >/dev/null 2>&1 ; pwd -P )"

# find correct trtexec version, only works for internal ci and brainpp env setups
CUDA_VERSION=$(nvcc --version | grep -o "[0-9].\.[0-9]*" | head -n 1)
SEARCH_PATH=$(echo `which nvcc | xargs dirname`/../../)
TRT_CANDIDATE=$(find `cd $SEARCH_PATH; pwd` -name "trtexec" | grep "bin/trtexec" | grep $CUDA_VERSION)
TRT_CANDIDATE=${TRT_CANDIDATE%$'\n'*}
TRT_LIB_PATH=$(readlink -f "`dirname $TRT_CANDIDATE`/../lib")
MODELS_PATH=$(readlink -f "${CUR_DIR}/../model_source")

# generate mge model
rm -rf $MODELS_PATH/conv_demo_f32_without_data.mge
python3 ${CUR_DIR}/conv_demo.py --dir $MODELS_PATH

# generate caffe model with mge convert
# INSTALL mgeconvert:
# python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git --user --install-option="--targets=caffe"
rm -rf $MODELS_PATH/conv_demo.prototxt $MODELS_PATH/conv_demo.caffemodel
convert mge_to_caffe -i $MODELS_PATH/conv_demo_f32_without_data.mge -c $MODELS_PATH/conv_demo.prototxt -b $MODELS_PATH/conv_demo.caffemodel

# generate trt model
rm -rf $MODELS_PATH/conv_demo.trt
echo "WARNING: config cudnn and cublas path before run trtexec"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB_PATH 
echo $LD_LIBRARY_PATH
$TRT_CANDIDATE --deploy="$MODELS_PATH/conv_demo.prototxt" --model="$MODELS_PATH/conv_demo.caffemodel" --output="conv1_Convolution" --batch=1 --saveEngine="$MODELS_PATH/conv_demo.trt"

# redump trt model into mgb model
rm -rf $MODELS_PATH/trt_conv_demo.pkl $MODELS_PATH/trt_conv_demo_with_data.mgb
python3 $CUR_DIR/dump_trt.py $MODELS_PATH/conv_demo.trt $MODELS_PATH/trt_conv_demo.pkl --isize="1,3,4,4"
$CUR_DIR/../../dump_with_testcase.py $MODELS_PATH/trt_conv_demo.pkl -o $MODELS_PATH/trt_conv_demo_with_data.mgb -d "#rand(0, 255)" --no-assert