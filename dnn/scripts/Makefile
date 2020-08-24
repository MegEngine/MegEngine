PARAM_DEFS := ../src/common/elemwise/each_mode.inl

ELEMWISE_IMPL := ../src/cuda/cond_take/kimpl \
	../src/cuda/elemwise/special_kimpl \
	../src/cuda/elemwise/kimpl \
	../src/rocm/elemwise/special_kimpl \
	../src/rocm/elemwise/kimpl \
	../src/naive/elemwise/kimpl \
	../src/cuda/elemwise_multi_type/kimpl

CUDA_CONV_IMPL := ../src/cuda/conv_bias/int8/kimpl ../src/cuda/conv_bias/int8_imma/kimpl ../src/cuda/batch_conv_bias/int8/kimpl

all: ${PARAM_DEFS} ${ELEMWISE_IMPL} ${CUDA_CONV_IMPL}

../src/common/elemwise/each_mode.inl: gen_elemwise_each_mode.py
	./$^ $@

../src/cuda/cond_take/kimpl: gen_cond_take_kern_impls.py
	./$^ --type cuda $@

../src/cuda/elemwise/special_kimpl: gen_elemwise_special_kern_impls.py
	./$^ --type cuda $@

../src/rocm/elemwise/special_kimpl: gen_elemwise_special_kern_impls.py
	./$^ --type hip $@

../src/cuda/elemwise/kimpl: gen_elemwise_kern_impls.py
	./$^ --type cuda $@

../src/rocm/elemwise/kimpl: gen_elemwise_kern_impls.py
	./$^ --type hip $@

../src/%/elemwise/kimpl: gen_elemwise_kern_impls.py
	./$^ $@

../src/cuda/elemwise_multi_type/kimpl: gen_elemwise_multi_type_kern_impls.py
	./$^ --type cuda $@

../src/cuda/conv_bias/int8/kimpl: gen_cuda_conv_bias_kern_impls.py gen_cutlass_conv_bias_kern_impls.py
	./gen_cuda_conv_bias_kern_impls.py --type dp4a $@
	./gen_cutlass_conv_bias_kern_impls.py --type dp4a $@

../src/cuda/conv_bias/int8_imma/kimpl: gen_cuda_conv_bias_kern_impls.py gen_cutlass_conv_bias_kern_impls.py
	./gen_cuda_conv_bias_kern_impls.py --type imma $@
	./gen_cutlass_conv_bias_kern_impls.py --type imma $@

../src/cuda/batch_conv_bias/int8/kimpl: gen_cuda_batch_conv_bias_kern_impls.py
	./$^ --type dp4a $@

.PHONY: all
