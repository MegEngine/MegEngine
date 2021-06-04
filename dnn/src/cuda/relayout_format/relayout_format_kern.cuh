/**
 * \file dnn/src/cuda/relayout_format/relayout_format_kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/memory_utils.cuh"
#include "src/cuda/relayout_format/translayout.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {
namespace internal {
using namespace memory;

struct LayoutType {
    static constexpr uint32_t NCHWx = 0;
    static constexpr uint32_t NHWC = 1;
};

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_ = LayoutType::NCHWx>
class TensorIteratorOverChannel;

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_>
class TensorIteratorOverChannel {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int lane_size_in_type =
            (width * pack_size * size_nbits) / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            (pack_size * size_nbits) >= (8 * sizeof(Type))
                    ? (pack_size * size_nbits / (8 * sizeof(Type)))
                    : (width * pack_size * size_nbits / (8 * sizeof(Type)));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST TensorIteratorOverChannel()
            : pointer{nullptr}, chan_stride_in_elements{0}, channel{0} {}
    MEGDNN_HOST TensorIteratorOverChannel(Type* pointer_,
                                          int chan_stride_in_elements_,
                                          int channel_, int, int)
            : pointer{pointer_},
              chan_stride_in_elements{chan_stride_in_elements_},
              channel{channel_} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += (c_idx / pack_size) * chan_stride_in_elements +
                   hw_idx * pack_size * size_nbits / (8 * sizeof(Type));
        channel -= c_idx;
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(
            size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                bool guard = i < channel;
                global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ +
                                                j * pack_size_in_type),
                        guard, zero_point);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                bool guard = i < channel;
                global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ +
                                                j * pack_size_in_type),
                        guard);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += (chan_blk / pack_size) * chan_stride_in_elements;
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int chan_stride_in_elements;
    int channel;
};

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
class TensorIteratorOverChannel<Type_, pack_size_, chan_blk_, width_,
                                size_nbits_, LayoutType::NHWC> {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            pack_size * size_nbits / (8 * sizeof(Type));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST TensorIteratorOverChannel()
            : pointer{nullptr}, hw_stride_in_elements{0}, channel{0} {}
    MEGDNN_HOST TensorIteratorOverChannel(Type* pointer_,
                                          int hw_stride_in_elements_,
                                          int channel_, int, int)
            : pointer{pointer_},
              hw_stride_in_elements{hw_stride_in_elements_},
              channel{channel_} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += c_idx * size_nbits / (8 * sizeof(Type)) +
                   hw_idx * hw_stride_in_elements;
        channel -= c_idx;
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(
            size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < width; ++i) {
#pragma unroll
            for (int j = 0; j < chan_blk; j += pack_size) {
                int frag_idx = i * (chan_blk / pack_size) + (j / pack_size);
                bool guard = j < channel;
                global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(
                                pointer_ + j * size_nbits / (8 * sizeof(Type))),
                        guard, zero_point);
            }
            pointer_ += hw_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < width; ++i) {
#pragma unroll
            for (int j = 0; j < chan_blk; j += pack_size) {
                int frag_idx = i * (chan_blk / pack_size) + (j / pack_size);
                bool guard = j < channel;
                global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(
                                pointer_ + j * size_nbits / (8 * sizeof(Type))),
                        guard);
            }
            pointer_ += hw_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += chan_blk * size_nbits / (8 * sizeof(Type));
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int hw_stride_in_elements;
    int channel;
};


template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_ = LayoutType::NCHWx>
class MaskedTensorIteratorOverChannel;

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_>
class MaskedTensorIteratorOverChannel {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int lane_size_in_type =
            (width * pack_size * size_nbits) / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            (pack_size * size_nbits) >= (8 * sizeof(Type))
                    ? (pack_size * size_nbits / (8 * sizeof(Type)))
                    : (width * pack_size * size_nbits / (8 * sizeof(Type)));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    static constexpr int accesses = elements_in_type / pack_size_in_type;
    static constexpr int mask_size = (accesses + 32 - 1) / 32;
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST MaskedTensorIteratorOverChannel()
            : pointer{nullptr}, chan_stride_in_elements{0}, channel{0} {}
    MEGDNN_HOST MaskedTensorIteratorOverChannel(Type* pointer_,
                                                int chan_stride_in_elements_,
                                                int channel_, int bound_,
                                                int div_)
            : pointer{pointer_},
              chan_stride_in_elements{chan_stride_in_elements_},
              channel{channel_},
              bound{bound_},
              div{uint32_t(div_)} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += (c_idx / pack_size) * chan_stride_in_elements;
        channel -= c_idx;
        int w[lane_size_in_type / pack_size_in_type];
#pragma unroll
        for (int i = 0; i < mask_size; ++i) {
            mask[i] = 0;
        }
#pragma unroll
        for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
            int offset = hw_idx + j;
            int h = (int)((uint32_t)(offset) / div);
            w[j] = (int)((uint32_t)(offset) % div);
            stride[j] = (h * bound + w[j]) * pack_size * size_nbits /
                        (8 * sizeof(Type));
        }
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                bool guard = (i < channel) && (w[j] < bound);
                int index = (i / pack_size) *
                                    (lane_size_in_type / pack_size_in_type) +
                            j;
                int mask_index = (index >> 5);
                int mask_shift = (index & 0x1f);
                mask[mask_index] |= (guard << mask_shift);
            }
        }
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(
            size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ + stride[j]), guard,
                        zero_point);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
        Type* pointer_ = pointer;
#pragma unroll
        for (int i = 0; i < chan_blk; i += pack_size) {
#pragma unroll
            for (int j = 0; j < lane_size_in_type / pack_size_in_type; j++) {
                int frag_idx = i / pack_size *
                                       (lane_size_in_type / pack_size_in_type) +
                               j;
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(pointer_ + stride[j]), guard);
            }
            pointer_ += chan_stride_in_elements;
        }
    }

    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += (chan_blk / pack_size) * chan_stride_in_elements;
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int chan_stride_in_elements;
    int channel;
    int bound;
    Uint32Fastdiv div;
    uint32_t mask[mask_size];
    size_t stride[lane_size_in_type / pack_size_in_type];
};

template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_>
class MaskedTensorIteratorOverChannel<Type_, pack_size_, chan_blk_, width_,
                                      size_nbits_, LayoutType::NHWC> {
public:
    using Type = Type_;
    static constexpr int pack_size = pack_size_;
    static constexpr int chan_blk = chan_blk_;
    static constexpr int width = width_;
    static constexpr int size_nbits = size_nbits_;
    static constexpr int elements_in_type =
            chan_blk * width * size_nbits / (8 * sizeof(Type));
    static constexpr int lane_size_in_type =
            (width * pack_size * size_nbits) / (8 * sizeof(Type));
    static constexpr int pack_size_in_type =
            pack_size * size_nbits / (8 * sizeof(Type));
    static constexpr int pack_size_in_byte = pack_size_in_type * sizeof(Type);
    static constexpr int accesses = elements_in_type / pack_size_in_type;
    static constexpr int mask_size = (accesses + 32 - 1) / 32;
    using AccessType = array_wrapper<Type, pack_size_in_type>;
    using Fragment = array_wrapper<Type, elements_in_type>;

    MEGDNN_HOST MaskedTensorIteratorOverChannel()
            : pointer{nullptr}, hw_stride_in_elements{0}, channel{0} {}
    MEGDNN_HOST MaskedTensorIteratorOverChannel(Type* pointer_,
                                                int hw_stride_in_elements_,
                                                int channel_, int bound_,
                                                int div_)
            : pointer{pointer_},
              hw_stride_in_elements{hw_stride_in_elements_},
              channel{channel_},
              bound{bound_},
              div{uint32_t(div_)} {}

    MEGDNN_DEVICE __forceinline__ void initialize(int c_idx, int hw_idx) {
        pointer += c_idx * size_nbits / (8 * sizeof(Type));
        channel -= c_idx;
#pragma unroll
        for (int i = 0; i < mask_size; ++i) {
            mask[i] = 0;
        }
#pragma unroll
        for (int i = 0; i < width; ++i) {
            int offset = hw_idx + i;
            int h = (int)((uint32_t)(offset) / div);
            int w = (int)((uint32_t)(offset) % div);
            stride[i] = (h * bound + w) * hw_stride_in_elements;
#pragma unroll
            for (int j = 0; j < chan_blk; j += pack_size) {
                bool guard = (j < channel) && (w < bound);
                int index = i * (chan_blk / pack_size) + (j / pack_size);
                int mask_index = (index >> 5);
                int mask_shift = (index & 0x1f);
                mask[mask_index] |= (guard << mask_shift);
            }
        }
    }

    MEGDNN_DEVICE __forceinline__ void add_pointer_offset(
            size_t offset_in_type) {
        pointer += offset_in_type;
    }

    MEGDNN_DEVICE __forceinline__ void load(Fragment& frag, int zero_point) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
#pragma unroll
        for (int i = 0; i < width; ++i) {
            Type* pointer_ = pointer + stride[i];
#pragma unroll
            for (int j = 0; j < chan_blk; j+= pack_size) {
                int frag_idx = i * (chan_blk / pack_size) + (j / pack_size);
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                global_load<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(
                                pointer_ + j * size_nbits / (8 * sizeof(Type))),
                        guard, zero_point);
            }
        }
    }

    MEGDNN_DEVICE __forceinline__ void store(const Fragment& frag) {
        const AccessType* frag_ptr = reinterpret_cast<const AccessType*>(&frag);
#pragma unroll
        for (int i = 0; i < width; ++i) {
            Type* pointer_ = pointer + stride[i];
#pragma unroll
            for (int j = 0; j < chan_blk; j+= pack_size) {
                int frag_idx = i * (chan_blk / pack_size) + (j / pack_size);
                int mask_index = (frag_idx >> 5);
                int mask_shift = (frag_idx & 0x1f);
                bool guard = (mask[mask_index] & (1 << mask_shift));
                global_store<AccessType, pack_size_in_byte>(
                        frag_ptr[frag_idx],
                        reinterpret_cast<void*>(
                                pointer_ + j * size_nbits / (8 * sizeof(Type))),
                        guard);
            }
        }
    }

    MEGDNN_DEVICE __forceinline__ void advance() {
        pointer += chan_blk * size_nbits / (8 * sizeof(Type));
        channel -= chan_blk;
    }

private:
    Type* pointer;
    int hw_stride_in_elements;
    int channel;
    int bound;
    Uint32Fastdiv div;
    uint32_t mask[mask_size];
    size_t stride[width];
};

template <bool padding_, typename Type_, int pack_size_, int chan_blk_,
          int width_, int size_nbits_,
          uint32_t layout_type_ = LayoutType::NCHWx>
struct TensorIteratorPolicy;
template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_>
struct TensorIteratorPolicy<true, Type_, pack_size_, chan_blk_, width_,
                            size_nbits_, layout_type_> {
    using TensorIterator =
            MaskedTensorIteratorOverChannel<Type_, pack_size_, chan_blk_,
                                            width_, size_nbits_, layout_type_>;
};
template <typename Type_, int pack_size_, int chan_blk_, int width_,
          int size_nbits_, uint32_t layout_type_>
struct TensorIteratorPolicy<false, Type_, pack_size_, chan_blk_, width_,
                            size_nbits_, layout_type_> {
    using TensorIterator =
            TensorIteratorOverChannel<Type_, pack_size_, chan_blk_, width_,
                                      size_nbits_, layout_type_>;
};

template <typename SrcIterator_, typename DstIterator_, typename Transpose_,
          typename CudaPostProcess_>
struct RelayoutProblem {
    using SrcIterator = SrcIterator_;
    using DstIterator = DstIterator_;
    using Transpose = Transpose_;
    using CudaPostProcess = CudaPostProcess_;
    MEGDNN_STATIC_ASSERT(SrcIterator::chan_blk == DstIterator::chan_blk,
                         "channel block mismatch");
    MEGDNN_STATIC_ASSERT(SrcIterator::width == DstIterator::width,
                         "width block mismatch");
    MEGDNN_STATIC_ASSERT(SrcIterator::size_nbits == DstIterator::size_nbits,
                         "size in bits of elements mismatch");
    static constexpr int pack_chan = SrcIterator::chan_blk;
    static constexpr int pack_width = SrcIterator::width;
    using DnnSrcType = typename CudaPostProcess::SrcType;
    using DnnDstType = typename CudaPostProcess::DstType;
    struct Param {
        SrcIterator src_iterator;
        DstIterator dst_iterator;
        CudaPostProcess post_process;
        int n_stride_src;
        int n_stride_dst;
        int batch_size;
        int channels;
        int hw;
        int zero_point;
        MEGDNN_HOST MEGDNN_DEVICE Param(SrcIterator src_iterator_,
                                        DstIterator dst_iterator_,
                                        CudaPostProcess post_process_,
                                        int n_stride_src_, int n_stride_dst_,
                                        int batch_size_, int channels_, int hw_,
                                        int zero_point_)
                : src_iterator{src_iterator_},
                  dst_iterator{dst_iterator_},
                  post_process{post_process_},
                  n_stride_src{n_stride_src_},
                  n_stride_dst{n_stride_dst_},
                  batch_size{batch_size_},
                  channels{channels_},
                  hw{hw_},
                  zero_point{zero_point_} {}
    };
};

template <typename RelayoutProblem_>
__global__ void relayout_kern(typename RelayoutProblem_::Param param) {
    using SrcIterator = typename RelayoutProblem_::SrcIterator;
    using DstIterator = typename RelayoutProblem_::DstIterator;
    static constexpr int pack_chan = RelayoutProblem_::pack_chan;
    static constexpr int pack_width = RelayoutProblem_::pack_width;
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_offset = thread_idx * pack_width;
    const int hw_idx = (thread_offset % param.hw);
    const int nc_blks = thread_offset / param.hw;
    const int c_blks = (param.channels + pack_chan - 1) / pack_chan;
    const int n_idx = nc_blks / c_blks;
    const int c_blk_idx = nc_blks % c_blks;
    const int c_idx = c_blk_idx * pack_chan;
    if (n_idx < param.batch_size) {
        const int src_offset = n_idx * param.n_stride_src;
        const int dst_offset = n_idx * param.n_stride_dst;
        param.src_iterator.add_pointer_offset(src_offset);
        param.dst_iterator.add_pointer_offset(dst_offset);
        param.src_iterator.initialize(c_idx, hw_idx);
        param.dst_iterator.initialize(c_idx, hw_idx);
        typename SrcIterator::Fragment src_frag;
        typename DstIterator::Fragment dst_frag;
        int zp = make_zero<SrcIterator::size_nbits>(param.zero_point);
        param.src_iterator.load(src_frag, zp);
        RelayoutProblem_::Transpose::trans(
                reinterpret_cast<typename SrcIterator::Fragment&>(dst_frag),
                src_frag, param.post_process);
        param.dst_iterator.store(dst_frag);
    }
}

}  // namespace internal
}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn
