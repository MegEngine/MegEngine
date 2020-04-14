/**
 * \file dnn/include/megdnn/oprs/general.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/internal/opr_header_prologue.h"
#include "megdnn/thin/small_vector.h"

namespace megdnn {

/*!
 * \brief standard element-wise operator
 *
 * Inputs must have same dtype, and their shapes must broadcastable into a final
 * shape. They can have arbitrary layouts, but non-contiguous and non-broadcast
 * layouts may harm performance seriously.
 *
 * Output dtype is the same as input dtype (note that even for compare oprs this
 * is true, e.g. float == float returns value of float). Output layout must be
 * contiguous.
 */
class ElemwiseForward: public OperatorBase {
    DEF_OPR_PARAM(Elemwise);
    DEF_OPR_IMPL(ElemwiseForward, OperatorBase, -1, 1);

    public:
        using Mode = Param::Mode;

        //! information about a mode
        struct ModeTrait {
            uint32_t arity;     //!< number of inputs needed
            bool commutable;    //!< whether arity == 2 and inputs commutable
            bool allow_int;     //!< whether int inputs allowed
            bool allow_float;   //!< whether float inputs allowed
            const char* name;   //!< name of the mode


            ModeTrait():
                arity(0), commutable(0), allow_int(0), allow_float(0),
                name(NULL)
            {}

            //! get trait from a mode; this function is thread safe
            static const ModeTrait& from_mode(Mode mode);
        };

        //! get trait of current mode
        const ModeTrait& mode_trait() const {
            return ModeTrait::from_mode(m_param.mode);
        }

        /**
         * \param[in] src input tensor
         * \param[out] dst output tensor
         *
         * src and dst should have the same shape;
         * layouts should be contiguous;
         * the underlying data pointer can point to the same memory region for
         * src and dst.
         */
        virtual void exec(_megdnn_in const TensorNDArray &src,
                _megdnn_tensor_out dst) = 0;

        //! deduce output shape (do not check whether arity matches)
        static void deduce_shape(
                const TensorShapeArray &src,
                TensorShape &dst);

        static void deduce_format(const TensorFormatArray& src,
                                  TensorFormat& dst);

        //! deduce output layout
        void deduce_layout(const TensorLayoutArray &src,
                TensorLayout &dst);

    protected:
        //! throw exception if incorrect layout; broadcast input shape to
        //! output shape
        void check_layout_and_broadcast(
                const TensorLayoutPtrArray &src, const TensorLayout &dst);

    private:
        void check_dtype(DType dtype);
};
using Elemwise = ElemwiseForward;

/*!
 * \brief compute ``x**a`` where ``a`` is a constant from the Param
 *
 * This opr is usually not directly accessible by the end user and it is created
 * by mgb optimizer, aiming to work around numerical stability issues with pow.
 * For example ``powf(x, 2.f)`` with ``x < 0`` in fast math mode may return NaN.
 *
 * Like elemwise, this opr supports arbitrary strides. But it should only be
 * used with monotone strides. Input and output should have the same
 * float-category dtype.
 */
class PowC : public OperatorBase {
    DEF_OPR_PARAM(PowC);
    DEF_OPR_IMPL(PowC, OperatorBase, 1, 1);

public:
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst);

    //! compatible API for mgb; workspace is not used
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace) {
        return exec(src, dst);
    }

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) {
        // the impls should require no workspace; this can be later changed to a
        // virtual function if this situation changes
        return 0;
    }

    void deduce_layout(const TensorLayout& src, TensorLayout& dst) {
        dst.dtype = src.dtype;
        dst.init_contiguous_stride(src);
    }

protected:
    /*!
     * Perform the computing where layouts have been verified.
     *
     * \p src can have arbitrary layout, and \p dst is contiguous. They have the
     * same shape and dtype.
     *
     * The implementation should not access param(). It should check \p exp_f
     * and \p exp_i for the exponent value. Exactly one of them would be
     * non-null.
     *
     * Note: \p exp_f and \p exp_i must be dereferenced before dispatching any
     * kernel. They are allocated on the caller's stack.
     */
    virtual void do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                         const float* exp_f, const int* exp_i) = 0;
};

/*!
 * \brief modify a tensor inplace by adding another tensor to it
 *
 * dst and delta can have arbitrary layout but must have the same shape.
 */
class AddUpdateForward: public OperatorBase {
    DEF_OPR_PARAM(AddUpdate);
    DEF_OPR_IMPL(AddUpdateForward, OperatorBase, -1, 1);

    public:
        virtual void exec(
                _megdnn_tensor_inout dst, _megdnn_tensor_in delta) = 0;

    protected:
        void check_exec(const TensorLayout &dst, const TensorLayout &delta);
};
using AddUpdate = AddUpdateForward;

class ReduceForward: public OperatorBase {
    DEF_OPR_PARAM(Reduce);
    DEF_OPR_IMPL(ReduceForward, OperatorBase, 1, 1);

    public:
        using Mode = Param::Mode;
        using DataType = Param::DataType;

        /**
         * \param[in] src input tensor
         * \param[out] dst output tensor
         *
         * src and dst should be contiguous.
         * src and dst should be of the same shape for all dimensions except
         * param().axis.
         * the param().axis-th dimension shape for dst should be one.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src, TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src, const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Reduce = ReduceForward;

class CumsumForward: public OperatorBase {
    DEF_OPR_PARAM(Cumsum);
    DEF_OPR_IMPL(CumsumForward, OperatorBase, 1, 1);

    public:
        /**
         * \param[in] src input tensor
         * \param[out] dst output tensor
         *
         * src and dst should be contiguous.
         * src and dst should have the same shape.
         *
         * The exclusive flag specifies whether the current element it taken
         * into account when calculating results.
         *
         * The reverse flag specifies whether cumsum is forward (
         * from 0 to n) or backward (from n downto 0).
         *
         * Example:
         *  exclusive && reverse:
         *   dst_i = src_{i+1} + src_{i+2} + ... + src_{n-1}
         *  exclusive && !reverse
         *   dst_i = src_0 + src_1 + ... + src_{i-1}
         *  !exclusive && reverse:
         *   dst_i = src_i + src_{i+1} + ... + src_{n-1}
         *  !exclusive && !reverse:
         *   dst_i = src_0 + src_1 + ... + src_i
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src, TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src, const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Cumsum = CumsumForward;

// mxx can be max or min
class ArgmxxBase: public OperatorBase {
    DEF_OPR_IMPL_CTOR(ArgmxxBase, OperatorBase);
    DEF_OPR_PARAM(Axis);

    protected:
        void check_layout_fwd(const TensorLayout &src,
                const TensorLayout &dst);
};

class ArgmaxForward: public ArgmxxBase {
    DEF_OPR_IMPL(ArgmaxForward, ArgmxxBase, 1, 1);
    public:
        /**
         * \param[in] src input tensor
         * \param[out] dst output tensor containing the argmax indices
         *
         * src and dst should be contiguous.
         * src and dst should be of the same shape for all dimensions except
         * param().axis.
         * the param().axis-th dimension shape for dst should be one.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Argmax = ArgmaxForward;

class ArgminForward: public ArgmxxBase {
    DEF_OPR_IMPL(ArgminForward, ArgmxxBase, 1, 1);
    public:
        /**
         * \param[in] src input tensor
         * \param[out] dst output tensor containing the argmax indices
         *
         * src and dst should be contiguous.
         * src and dst should be of the same shape for all dimensions except
         * param().axis.
         * the param().axis-th dimension shape for dst should be one.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Argmin = ArgminForward;

/*!
 * \brief take values from input according to given condition
 *
 * Output two tensors:
 *  1. values copied from *data*, with same dtype as *data*
 *  2. selected indices with dtype int32; note that it is 1-dimensional and
 *     based on the flatten input.
 *
 * Require data and mask to have the same shape and both be contiguous.
 */
class CondTake : public OperatorBase {
    DEF_OPR_IMPL(CondTake, OperatorBase, 2, 2);
    DEF_OPR_PARAM(CondTake);

public:
    using Output = std::array<TensorND, 2>;
    using OutputDType = std::array<DType, 2>;

    OutputDType infer_dtype(DType data, DType mask);

    virtual size_t get_workspace_in_bytes(const TensorLayout& data) = 0;

    virtual Output exec(_megdnn_tensor_in data, _megdnn_tensor_in mask,
                        _megdnn_workspace workspace,
                        DynOutMallocPolicyCall malloc_policy) = 0;

protected:
    //! check input layouts and get flattened size
    size_t check_exec_get_size(const TensorLayout& data,
                               const TensorLayout& mask,
                               size_t workspace_in_bytes);
};

class TransposeForward: public OperatorBase {
    DEF_OPR_IMPL(TransposeForward, OperatorBase, 1, 1);
    DEF_OPR_PARAM(Empty);
    public:
        /**
         * \param[in] src (m, n) stride[0] >= n && stride[1] == 1
         * \param[out] dst (n, m) stride[0] >= m && stride[1] == 1
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src, TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src, const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Transpose = TransposeForward;

/**
 * Change a tensor to another layout that has the same dtype and total number of
 * elements, and non-overlapping stride.
 *
 * ON CPU:
 * This operator is optimized for some cases(e.g. both dst and last dim of src
 * are contiguous)
 *
 * ON CUDA:
 * More contiguous the input/output layouts, higher performance. There is also
 * special optimization for broadcast case.
 */
class RelayoutForward: public OperatorBase {
    DEF_OPR_IMPL(RelayoutForward, OperatorBase, 1, 1);
    DEF_OPR_PARAM(Empty);
    public:
        /*!
         * \brief execute relayout opr
         *
         * This operator should be placed on the same computing device of *dst*.
         *
         * \param src_handle handle of input tensor; for CUDA d2d copy, the
         *      src handle can be on a different GPU for copy tensor with
         *      non-contig dims <= 2
         */
        virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                Handle *src_handle = nullptr) = 0;
    protected:
        //! check layout and collapse contiguous
        void check_layout_and_canonize(
                TensorLayout &src, TensorLayout &dst);
};
using Relayout = RelayoutForward;

/**
 * \brief Base class for Concat and Split operators
 */
class ConcatSplitBase: public OperatorBase {
    public:
        using Param = param::Axis;

        ConcatSplitBase(Handle *handle);
        const Param &param() const { return m_param; }
        Param &param() { return m_param; }
    protected:
        void check_layout_common(const TensorLayoutArray &srcs,
                const TensorLayout &dst);
        Param m_param;
        /**
         * \brief a helper function
         *
         * A = shape[0] * shape[1] * ... * shape[axis-1]
         * B = {srcs[0].shape[axis], srcs[1].shape[axis], ...}
         * C = shape[axis+1] * shape[axis+2] * ... * shape[ndim-1]
         */
        void get_ABC(const TensorShapeArray &srcs,
                size_t &A,
                size_t *B,
                size_t &C);
        thin_function<TensorLayout(const TensorND &tensor)> m_get_layout;
        thin_function<TensorShape(const TensorLayout &layout)> m_get_shape;
};

class ConcatForward: public ConcatSplitBase {
    DEF_OPR_IMPL(ConcatForward, ConcatSplitBase, 1, 1);
    public:
        /**
         * \param[in] srcs a vector containing all inputs to be concatenated
         * \param[out] dst the output tensor.
         *
         * All tensors in srcs and dst should be contiguous.
         * All tensors should have the same shape for all axes except
         * param().axis.
         * For the param().axis-th axis, the axis shape for dst should be the
         * sum of corresponding axis shapes for all srcs.
         */
        virtual void exec(_megdnn_in const TensorNDArray &srcs,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayoutArray &srcs,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(
                const TensorLayoutArray &srcs,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayoutArray &srcs,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Concat = ConcatForward;

class SplitForward: public ConcatSplitBase {
    DEF_OPR_IMPL(SplitForward, ConcatSplitBase, 1, 1);
    public:
        /**
         * \param[in] src input tensor
         * \param[out] dsts a vector containing all splitted result
         *
         * All tensors in src and dsts should be contiguous.
         * All tensors should have the same shape for all axes except
         * param().axis.
         * For the param().axis-th axis, the axis shape for src should be the
         * sum of corresponding axis shapes for all dsts.
         */
        virtual void exec(_megdnn_tensor_in src,
                const TensorNDArray &dsts,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayoutArray &dsts) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayoutArray &dsts,
                size_t workspace_in_bytes);
};
using Split = SplitForward;

/**
 * \brief Base class for ParamPackConcat and ParamPackSplit Operators.
 *
 * ParamPack oprs act like Concat and Split, but they also are optimized for a
 * large number of inputs and can handle alignment requirements. Axis is also
 * not supported.
 *
 * The offsets can be generated by gen_offsets().
 */
class ParamPackConcatSplitBase : public OperatorBase {
protected:
    void check_exec(const TensorLayout& concated, const TensorLayout& offsets,
                    const TensorLayout& parts);

public:
    using Param = megdnn::param::Empty;
    ParamPackConcatSplitBase(Handle* handle) : OperatorBase(handle) {}

    //! generate offsets to be used with ParamPackConcat and ParamPackSplit
    static std::vector<dt_int32> gen_offsets(const TensorShapeArray& shapes,
                                             size_t alignment,
                                             size_t dtype_size);
};

/**
 * \brief ParamPackConcat, used for calculating gradient of ParamPackSplit
 * Combine multiple gradient tensors into a single large tensor, use copy
 * strategy due to AddUpdate or other dynamic situation.
 */
class ParamPackConcat: public ParamPackConcatSplitBase {
    DEF_OPR_IMPL(ParamPackConcat, ParamPackConcatSplitBase, 2, 1);

public:
    /*
     * \param[in] srcs: TensorND on cpu. srcs[i] corresponding to the
     *                  address of i-th Tensor.
     * \param[in] offsets: with size `2 * srcs.shape[0]`.
     *                  offsets[i * 2] and offsets[i * 2 + 1] means
     *                  the begin and the end of srcs[i]'s offsets in dst
     * \param[out] dst: output TensorND, live on cpu or gpu
     */
    virtual void exec(_megdnn_tensor_in srcs, _megdnn_tensor_in offsets,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorShapeArray& srcs,
                                          const TensorShape& offsets,
                                          const TensorShape& dst) = 0;
};

/**
 * \brief base class for Tile and Repeat
 */
class TileRepeatBase: public OperatorBase {
    public:
        TileRepeatBase(Handle *handle):  OperatorBase(handle) {}
        struct Param {
            TensorShape times;
        };
        Param &param() { return m_param; }
        const Param &param() const { return m_param; }
    protected:
        void check_layout_fwd(const TensorLayout &src,
                const TensorLayout &dst);
        void deduce_layout_fwd(const TensorLayout &src,
                TensorLayout &dst);
        /**
         * Assuming src/dst/times are already simplified on entrance.
         */
        size_t get_workspace_in_bytes_fwd(const TensorShape &src,
                const TensorShape &dst,
                const TensorShape &times,
                DType dtype);
        Param m_param;
};

class TileBase: public TileRepeatBase {
    public:
        TileBase(Handle *handle): TileRepeatBase(handle) {}
    protected:
        void simplify_shape(const TensorShape &src,
                const TensorShape &dst,
                const TensorShape &times,
                TensorShape &src2,
                TensorShape &dst2,
                TensorShape &times2);
        /**
         * This is a helper function that would facilitate other backends'
         * implementation.
         */
        size_t get_workspace_in_bytes_fwd(const TensorLayout &src,
                const TensorLayout &dst);
};

class TileForward: public TileBase {
    DEF_OPR_IMPL(TileForward, TileBase, 1, 1);
    public:
        /**
         * \brief Tile src times to get dst.
         * \param[in] src input tensor
         * \param[out] dst output tensor
         * \param[out] workspace temporary workspace
         *
         * src and dst must be contiguous.
         * dst.shape should be {src.shape[0]*param().times[0],
         * src.shape[1]*param().times[1], ...}
         *
         * \see http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
         *
         * Difference between Tile and Repeat:
         *  Tiling `abc' twice yields `abcabc', whereas repeating `abc' twice
         *  yields `aabbcc'.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src, const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Tile = TileForward;

class TileBackward: public TileBase {
    DEF_OPR_IMPL(TileBackward, TileBase, 1, 1);
    public:
        /**
         * \param[in] diff the backpropagated gradient wrt. dst
         * \param[out] grad the backpropagated gradient wrt. src
         * \param[out] workspace temporary workspace
         */
        virtual void exec(_megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &diff,
                const TensorLayout &grad) = 0;
    protected:
        void check_exec(const TensorLayout &diff, const TensorLayout &grad,
                size_t workspace_in_bytes);
};

class RepeatBase: public TileRepeatBase {
    public:
        RepeatBase(Handle *handle): TileRepeatBase(handle) {}
    protected:
        void simplify_shape(const TensorShape &src,
                const TensorShape &dst,
                const TensorShape &times,
                TensorShape &src2,
                TensorShape &dst2,
                TensorShape &times2);
        /**
         * This is a helper function that would facilitate other backends'
         * implementation.
         */
        size_t get_workspace_in_bytes_fwd(const TensorLayout &src,
                const TensorLayout &dst);
};

class RepeatForward: public RepeatBase {
    DEF_OPR_IMPL(RepeatForward, RepeatBase, 1, 1);
    public:
        /**
         * \brief Repeat src times to get dst.
         * \param[in] src input tensor
         * \param[out] dst output tensor
         * \param[out] workspace temporary workspace
         *
         * src and dst must be contiguous.
         * dst.shape should be {src.shape[0]*param().times[0],
         * src.shape[1]*param().times[1], ...}
         *
         * \see http://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
         * \see TileForward
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using Repeat = RepeatForward;

class RepeatBackward: public RepeatBase {
    DEF_OPR_IMPL(RepeatBackward, RepeatBase, 1, 1);
    public:
        /**
         * \param[in] diff the backpropagated gradient wrt. dst
         * \param[out] grad the backpropagated gradient wrt. src
         * \param[out] workspace temporary workspace
         */
        virtual void exec(_megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &diff,
                const TensorLayout &grad) = 0;
    protected:
        void check_exec(const TensorLayout &diff,
                const TensorLayout &grad,
                size_t workspace_in_bytes);
};

class ArgsortForward: public OperatorBase {
    DEF_OPR_IMPL(ArgsortForward, OperatorBase, 1, 2);
    DEF_OPR_PARAM(Argsort);
    public:
        using Order = Param::Order;
        /**
         * \param[in] src (m, n)
         * \param[out] dst (m, n)
         * \param[out] indices (m, n)
         *
         * src, dst and indices should be contiguous.
         * Performing m independent sorting on m arrays of length n.
         * Sorting arrays and storing the resulting array in `dst',
         * and the corresponding indices in `indices'.
         *
         * Indices range from 0 to n-1.
         *
         * Note that indices is a TensorND of type int.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_tensor_out indices,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                TensorLayout &dst,
                TensorLayout &indices);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &dst,
                const TensorLayout &indices) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &dst,
                const TensorLayout &indices,
                size_t workspace_in_bytes);
};
using Argsort = ArgsortForward;

/*!
 * \brief backward opr for Argsort
 *
 * Note: the name is kept for backward compatibility. This opr is actually a
 * batched value setter. It is used for gradient computing of Argsort and TopK.
 */
class ArgsortBackward : public OperatorBase {
    DEF_OPR_IMPL(ArgsortBackward, OperatorBase, 2, 1);
    DEF_OPR_PARAM(Empty);

public:
    /**
     * \param[in] diff (m, k) the backpropagated gradient wrt. dst
     * \param[in] indices (m, k) the `indices' parameter in
     *                           ArgsortForward::exec
     * \param[out] grad (m, n) the backpropagated gradient wrt. src
     *
     * Constraint: n >= k. Untouched values would be initialized as zero.
     */
    virtual void exec(_megdnn_tensor_in diff, _megdnn_tensor_in indices,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& diff,
                                          const TensorLayout& indices,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& diff, const TensorLayout& indices,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class TopK : public OperatorBase {
    DEF_OPR_IMPL(TopK, OperatorBase, 1, 2);
    DEF_OPR_PARAM(TopK);

protected:
    //! impl exec; inputs have been validated
    virtual void do_exec(int k, _megdnn_tensor_in data,
                         _megdnn_tensor_out values, int32_t* indices,
                         _megdnn_workspace workspace) = 0;

public:
    /*!
     * \param[in] k if positive, compute the smallest top-k values; otherwise
     *      compute the largest top-k values
     * \param[in] data (m, n) input data, where top-k is computed on the
     *      second axis. The second dimension must be contiguous, and the first
     *      dimension can have arbitrary stride.
     * \param[out] values (m, ) or (m, k) output values; its shape depends
     *      on mode
     * \param[out] indices () or (m, ) or (m, k) output values; its shape
     *      depends on mode
     */
    void exec(int k, _megdnn_tensor_in data, _megdnn_tensor_out values,
              _megdnn_tensor_out indices, _megdnn_workspace workspace);
    virtual size_t get_workspace_in_bytes(int k, const TensorLayout& data,
                                          const TensorLayout& values,
                                          const TensorLayout& indices) = 0;

    void deduce_layout(int k, const TensorLayout& data, TensorLayout& values,
                       TensorLayout& indices);
};

/*!
 * \brief convert dtype of *src* to match dtype of *dst*; *src* may have
 *      arbitrary layout and *dst* must be contiguous.
 */
class TypeCvtForward: public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(TypeCvtForward, OperatorBase, 1, 1);
    public:
        virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) = 0;
    protected:
        void check_exec(const TensorLayout &src, const TensorLayout &dst);
};
using TypeCvt = TypeCvtForward;

class IndexingRemapBase: public OperatorBase {
    public:
        using Param = param::IndexingRemap;

        IndexingRemapBase(Handle *handle): OperatorBase(handle) {}
        Param &param() { return m_param; }
        const Param &param() const { return m_param; }
    protected:
        Param m_param;
        void check_layout_fwd(const TensorLayout &src,
                const TensorLayout &map,
                const TensorLayout &dst);
};

class IndexingRemapForward: public IndexingRemapBase {
    DEF_OPR_IMPL(IndexingRemapForward, IndexingRemapBase, 2, 1);
    public:
        /**
         * \param[in] src input tensor
         * \param[in] map input map
         * \param[out] dst output tensor
         *
         * Suppose:
         *  the shape of src is \f$(s_0, s_1, ..., s_{m-1}\f$;
         *  the shape of dst is \f$(d_0, d_1, ..., d_{n-1})\f$;
         * then:
         *  the shape of map must be \f$(d_0, d_1, ..., d_{n-1}, m)\f$.
         *
         * The last dimension of map indicates the src indices for the
         * corresponding dst entry.
         *
         * src and dst can be non-contiguous in a non-overlapping manner.
         */
        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in map,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        void deduce_layout(const TensorLayout &src,
                const TensorLayout &map,
                TensorLayout &dst);
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &map,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &map,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using IndexingRemap = IndexingRemapForward;
// The using directives preserve backward compatibility.
using TensorRemapForward = IndexingRemap;
using TensorRemap = TensorRemapForward;

class IndexingRemapBackward: public IndexingRemapBase {
    DEF_OPR_IMPL(IndexingRemapBackward, IndexingRemapBase, 2, 1);
    public:
        /**
         * \param[in] diff the backpropagated gradient wrt. dst
         * \param[in] map the `map' parameter in IndexingRemapForward::exec
         * \param[out] grad the backpropagated gradient wrt. src
         */
        virtual void exec(_megdnn_tensor_in diff,
                _megdnn_tensor_in map,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &diff,
                const TensorLayout &map,
                const TensorLayout &grad) = 0;
    protected:
        void check_exec(const TensorLayout &diff,
                const TensorLayout &map,
                const TensorLayout &grad,
                size_t workspace_in_bytes);
};
// The using directives preserve backward compatibility.
using TensorRemapBackward = IndexingRemapBackward;

class Linspace: public OperatorBase {
    DEF_OPR_IMPL(Linspace, OperatorBase, 0, 1);
    DEF_OPR_PARAM(LinspaceFull);
    public:
        /**
         * \param[out] dst must be 1d.
         *
         * \see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
         */
        virtual void exec(_megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &dst, size_t workspace_in_bytes);
};

class Eye: public OperatorBase {
    DEF_OPR_IMPL(Eye, OperatorBase, 0, 1);
    DEF_OPR_PARAM(Eye);
    public:
        /**
         * \see http://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
         */
        virtual void exec(_megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &dst, size_t workspace_in_bytes);
};

class IndexingOneHotBase: public OperatorBase {
    DEF_OPR_IMPL_CTOR(IndexingOneHotBase, OperatorBase);
    DEF_OPR_PARAM(Axis);

    protected:
        void deduce_layout_fwd(const TensorLayout &src,
                const TensorLayout &index,
                TensorLayout &dst);
        void check_layout_fwd(const TensorLayout &src,
                const TensorLayout &index,
                const TensorLayout &dst);
};

/*!
 * \brief Indexing for one-hot encoding
 *
 * Given src, axis and index,
 * for all valid (n-1)-dimensional subscript tuples i iterating through index:
 * dst[i[0], ..., i[axis-1], 0, i[axis], ..., i[n-2]] =
 * inp[i[0], ..., i[axis-1], index[i], i[axis], ..., i[n-2]]
 *
 * \param[in] src n-dimensional input data
 * \param[in] index (n-1)-dimensional index, must be int
 * \param[out] dst n-dimensional output data
 */
class IndexingOneHotForward: public IndexingOneHotBase {
    DEF_OPR_IMPL(IndexingOneHotForward, IndexingOneHotBase, 2, 1);

    public:
        void deduce_layout(const TensorLayout &src,
                const TensorLayout &index, TensorLayout &dst) {
            deduce_layout_fwd(src, index, dst);
        }

        virtual void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in index,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &index,
                const TensorLayout &dst) = 0;
    protected:
        void check_exec(const TensorLayout &src,
                const TensorLayout &index, const TensorLayout &dst,
                size_t workspace_in_bytes);
};
using IndexingOneHot = IndexingOneHotForward;

/*!
 * \brief set-subtensor corresponding to IndexingOneHotForward
 *
 * \param[in,out] data n-dimensional input and output data, whose sub part
 *      corresponding to *index* would be replaced by *sub*
 * \param[in] index (n-1)-dimensional index, must be int
 * \param[in] sub n-dimensional sub tensor to be filled in *data*
 */
class IndexingSetOneHotForward: public IndexingOneHotBase {
    DEF_OPR_IMPL(IndexingSetOneHotForward, IndexingOneHotBase, -1, 1);

    public:
        virtual void exec(_megdnn_tensor_inout data, _megdnn_tensor_in index,
                _megdnn_tensor_in sub, _megdnn_workspace workspace) = 0;
        virtual size_t get_workspace_in_bytes(const TensorLayout &data,
                const TensorLayout &index,
                const TensorLayout &sub) = 0;
    protected:
        void check_exec(const TensorLayout &data,
                const TensorLayout &index, const TensorLayout &sub,
                size_t workspace_in_bytes);
};
using IndexingSetOneHot = IndexingSetOneHotForward;

/*!
 * \brief base class for indexing on multiple axes using vector indices
 *
 * Note that the indexing axes are required to be sorted in ascending order
 */
class IndexingMultiAxisVecBase: public OperatorBase {
    DEF_OPR_IMPL_CTOR(IndexingMultiAxisVecBase, OperatorBase);
    DEF_OPR_PARAM(Empty);

    public:
        struct AxisIndexer {
            size_t axis;
            TensorND vec;
        };

        struct AxisIndexerLayoutOnly {
            size_t axis;
            TensorLayout layout;
        };

        using IndexDesc = std::vector<AxisIndexer>;
        using IndexDescLayoutOnly = std::vector<AxisIndexerLayoutOnly>;

        /*!
         * \brief convert IndexDesc to IndexDescLayoutOnly
         */
        static IndexDescLayoutOnly extract_index_layout(const IndexDesc &index);

        /*!
         * \brief get the axes on src that are not used in index
         * \param[out] out output buffer; suggested size is
         *      TensorLayout::MAX_NDIM
         * \return number of elements written to *out*
         */
        static size_t get_nonindex_axes(size_t src_ndim, const IndexDesc &index,
                size_t *out);

        /*!
         * \brief get contiguous-collapsed layout for indexing on value
         * \param idx_axis indexer axis on value (i.e. ExecInfo::idx_axis)
         * \return a tensor layout and an axis to iterate over *value* and also
         *      access *data*; stride of layout on that axis would be zero, and
         *      strides on other axes correspond to the strides in *data*
         */
        static std::pair<TensorLayout, size_t> get_value_iter_optimized_layout(
                const TensorLayout &data, const TensorLayout &value,
                const IndexDesc &index, size_t idx_axis);

        //! helper info for kernel implementation
        struct ExecInfo {
            //! axis in value used by indexer
            size_t idx_axis;
            ptrdiff_t value_stride;

            void* error_tracker;
            megcore::AsyncErrorInfo* error_info;
        };

    protected:
        /*!
         * \return axis on dst used by indexer (i.e. ExecInfo::idx_axis)
         */
        static size_t deduce_layout_fwd(
                const TensorLayout &data,
                const IndexDescLayoutOnly &index,
                TensorLayout &dst);

        static ExecInfo check_exec_noworkspace(
                const TensorLayout &data, const TensorLayout &value,
                const IndexDesc &index, IndexDescLayoutOnly &index_layout);
};

/*!
 * \brief compute indexing result, like numpy advanced indexing
 *
 * src can have arbitrary layout, but dst must be dim1-contig
 */
class IndexingMultiAxisVec: public IndexingMultiAxisVecBase {
    DEF_OPR_IMPL(IndexingMultiAxisVec, IndexingMultiAxisVecBase, 0, 1);

    public:
        virtual void exec(_megdnn_tensor_in src,
                const IndexDesc &index,
                _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;

        /*!
         * \brief get workspace size based on output shape and indexing axes
         */
        size_t get_workspace_in_bytes(
                const TensorShape &dst,
                const size_t *axes, size_t nr_axes);

        static void deduce_layout(
                const TensorLayout &data,
                const IndexDescLayoutOnly &index,
                TensorLayout &dst) {
            deduce_layout_fwd(data, index, dst);
        }
    protected:

        virtual size_t get_workspace_in_bytes(size_t dst_idx_size) = 0;

        ExecInfo check_exec(
                const TensorLayout &src,
                const IndexDesc &index,
                const TensorLayout &dst,
                size_t workspace_in_bytes);
};

/*!
 * \brief base class for modifying data by given index
 *
 * data can have arbitrary layout, but value must be dim1-contig
 */
class IndexingModifyMultiAxisVecBase: public IndexingMultiAxisVecBase {
    DEF_OPR_IMPL_CTOR(IndexingModifyMultiAxisVecBase, IndexingMultiAxisVecBase);

    public:
        virtual void exec(
                _megdnn_tensor_inout data, _megdnn_tensor_in value,
                const IndexDesc &index,
                _megdnn_workspace workspace) = 0;

        /*!
         * \brief get workspace size based on shape of value input and indexing
         *      axes
         */
        size_t get_workspace_in_bytes(
                const TensorShape &value,
                const size_t *axes, size_t nr_axes);

    protected:
        ExecInfo check_exec(
                const TensorLayout &data, const TensorLayout &value,
                const IndexDesc &index,
                size_t workspace_in_bytes);

        virtual size_t get_workspace_in_bytes(size_t value_idx_size) = 0;
};

//! set value to indexed locations; index values must be non-overlapping
class IndexingSetMultiAxisVec: public IndexingModifyMultiAxisVecBase {
    DEF_OPR_IMPL(IndexingSetMultiAxisVec,
            IndexingModifyMultiAxisVecBase, 0, 0);
};

//! add value to indexed locations; index values must be non-overlapping
class IndexingIncrMultiAxisVec: public IndexingModifyMultiAxisVecBase {
    DEF_OPR_IMPL(IndexingIncrMultiAxisVec,
            IndexingModifyMultiAxisVecBase, 0, 0);
};

class MeshBase : public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL_CTOR(MeshBase, OperatorBase);

public:
    using AxisIndexer = IndexingMultiAxisVecBase::AxisIndexer;
    using IndexDesc = IndexingMultiAxisVecBase::IndexDesc;
    using AxisIndexerLayoutOnly =
            IndexingMultiAxisVecBase::AxisIndexerLayoutOnly;
    using IndexDescLayoutOnly = IndexingMultiAxisVecBase::IndexDescLayoutOnly;

    size_t get_workspace_in_bytes(const TensorShape&, const size_t*, size_t) {
        return 0;
    }

protected:
    virtual void check_exec(const TensorLayout& origin,
                            const TensorLayout& indexed, const IndexDesc& desc);
};

class NormalMeshBase : public MeshBase {
    DEF_OPR_IMPL(NormalMeshBase, MeshBase, 0, 0);

protected:
    virtual void check_exec(const TensorLayout& origin,
                            const TensorLayout& indexed,
                            const IndexDesc& desc) override final;
};

class NormalMeshModifyBase : public NormalMeshBase {
    DEF_OPR_IMPL_CTOR(NormalMeshModifyBase, NormalMeshBase);

public:
    virtual void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
                      const IndexDesc& desc, _megdnn_workspace workspace) = 0;
};

class BatchedMeshBase : public MeshBase {
    DEF_OPR_IMPL_CTOR(BatchedMeshBase, MeshBase);

protected:
    virtual void check_exec(const TensorLayout& origin,
                            const TensorLayout& indexed,
                            const IndexDesc& desc) override final;
};

class BatchedMeshModifyBase : public BatchedMeshBase {
    DEF_OPR_IMPL_CTOR(BatchedMeshModifyBase, BatchedMeshBase);

public:
    virtual void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
                      const IndexDesc& desc, _megdnn_workspace workspace) = 0;
};

class MeshIndexing : public NormalMeshBase {
    DEF_OPR_IMPL(MeshIndexing, NormalMeshBase, 0, 0);

public:
    virtual void exec(_megdnn_tensor_in src, const IndexDesc& desc,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;

    static void deduce_layout(const TensorLayout& inp,
                              const IndexDescLayoutOnly& layouts,
                              TensorLayout& out_layout);
};

class IncrMeshIndexing : public NormalMeshModifyBase {
    DEF_OPR_IMPL(IncrMeshIndexing, NormalMeshModifyBase, 0, 0);
};

class SetMeshIndexing : public NormalMeshModifyBase {
    DEF_OPR_IMPL(SetMeshIndexing, NormalMeshModifyBase, 0, 0);
};

class BatchedMeshIndexing : public BatchedMeshBase {
    DEF_OPR_IMPL(BatchedMeshIndexing, BatchedMeshBase, 0, 0);

public:
    virtual void exec(_megdnn_tensor_in src, const IndexDesc& desc,
                      _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;

    static void deduce_layout(const TensorLayout& inp,
                              const IndexDescLayoutOnly& layouts,
                              TensorLayout& out_layout);
};

class BatchedIncrMeshIndexing : public BatchedMeshModifyBase {
    DEF_OPR_IMPL(BatchedIncrMeshIndexing, BatchedMeshModifyBase, 0, 0);
};

class BatchedSetMeshIndexing : public BatchedMeshModifyBase {
    DEF_OPR_IMPL(BatchedSetMeshIndexing, BatchedMeshModifyBase, 0, 0);
};

class RelayoutFormat : public OperatorBase {
    DEF_OPR_PARAM(RelayoutFormat);
    DEF_OPR_IMPL(RelayoutFormat, OperatorBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    void deduce_format(TensorFormat src, TensorFormat& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);

    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);

    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
    void deduce_exec_layout(const TensorLayout& src, const TensorLayout& dst,
                            TensorLayout& exec_src, TensorLayout& exec_dst);
};
}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
