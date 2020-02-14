/**
 * \file dnn/include/megdnn/basic_types.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/arch.h"
#include "megdnn/dtype.h"
#include "megdnn/internal/defs.h"

#if MEGDNN_CC_HOST
#include <string>
#include <type_traits>
#include <vector>
#include <cstdarg>
#include "megdnn/thin/small_vector.h"
#endif  // MEGDNN_CC_HOST

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {

class ErrorHandler {
#if MEGDNN_CC_HOST
    static ErrorHandler* sm_inst;
    static ErrorHandler* inst();

protected:
    MEGDNN_NORETURN virtual void do_on_megdnn_error(const std::string& msg) = 0;

    MEGDNN_NORETURN virtual void do_on_tensor_reshape_error(
            const std::string& msg) {
        on_megdnn_error(msg);
    }

    ~ErrorHandler() = default;

#endif
public:
    //! called on general megdnn error
    MEGDNN_NORETURN static void on_megdnn_error(const char* msg);

    //! called on tensor reshape error
    MEGDNN_NORETURN static void on_tensor_reshape_error(const char* msg);

#if MEGDNN_CC_HOST
    MEGDNN_NORETURN static void on_megdnn_error(const std::string& msg);
    MEGDNN_NORETURN static void on_tensor_reshape_error(const std::string& msg);

    /*!
     * \brief set the global error handler instance
     *
     * This method is not thread-safe. The caller is responsible to ensure the
     * ErrorHandler is a global object with enough life span.
     *
     * \return original error handler
     */
    static void set_handler(ErrorHandler* handler);

#endif  // MEGDNN_CC_HOST
};

#if MEGDNN_CC_HOST
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

typedef void (*LogHandler)(LogLevel level, const char* file, const char* func,
                           int line, const char* fmt, va_list ap);

/*!
 * \brief set the callback to receive all log messages
 *
 * Note: the log handler can be NULL (which is also the default value). In this
 * case, no log message would be recorded.
 *
 * \return original log handler
 */
LogHandler set_log_handler(LogHandler handler);
#endif

/**
 * \brief Describing the tensor shape.
 *
 * Uninitialized shape: ndim == 0; total_nr_elems() is also defined to be 0
 *
 * Empty shape: ndim > 0 && shape[i] == 0 for 0 <= i < ndim; it is always
 * considered non-contiguous.
 */
struct TensorShape {
    static MEGDNN_CONSTEXPR size_t MAX_NDIM = MEGDNN_MAX_NDIM;

#if MEGDNN_CC_HOST
    size_t shape[MAX_NDIM], ndim = 0;
#else
    size_t shape[MAX_NDIM], ndim;
#endif

#if MEGDNN_CC_HOST
    TensorShape() = default;
    TensorShape(const TensorShape& rhs) = default;
    TensorShape(const SmallVector<size_t>& init_shape);
    TensorShape(std::initializer_list<size_t> init_shape);
    std::string to_string() const;
#endif

    //! total number of elements
    size_t total_nr_elems() const;

    //! check whether two shapes are equal
    bool eq_shape(const TensorShape& rhs) const;

    //! check whether the shape can be treated as a scalar
    bool is_scalar() const { return ndim == 1 && shape[0] == 1; }

    //! check whether ndim != 0 and at least one shape is 0
    bool is_empty() const;

    //! access single element, without boundary check
    size_t& operator[](size_t i) { return shape[i]; }
    size_t operator[](size_t i) const { return shape[i]; }
};

class Handle;
/**
 * \brief Describing the tensor shape with its actual layout in memory and dtype
 *
 * x(i, j, ...) is stored at offset
 * stride[0]*i + stride[1]*j + ..., in number of elements; physical offset needs
 * to be multiplied by dtype size.
 */
struct TensorLayout : public TensorShape {
    /*!
     * \brief Describes min and max offsets of tensor elements with respect to
     *      its first element, so all tensor elements are guaranteed to be in
     *      the range [elem[0]+low, elem[0]+high).
     */
    struct Span {
        ptrdiff_t low_elem, low_byte;
        size_t high_elem, high_byte;

        Span(ptrdiff_t low_elem, ptrdiff_t low_byte, size_t high_elem,
             size_t high_byte)
                : low_elem(low_elem),
                  low_byte(low_byte),
                  high_elem(high_elem),
                  high_byte(high_byte) {}
        size_t dist_elem() const { return high_elem - low_elem; }

        size_t dist_byte() const { return high_byte - low_byte; }
    };

    /*!
     * \brief Describing the requirements for tensor layouts
     *
     * Some runtime (e.g. opencl) may have alignment requirements for special
     * memory types (e.g. image in texture memory). Format objects can be used
     * to impose such constraints on methods related to tensor strides.
     *
     * Note that ImplBase is defined in tensor_format.h
     */
    class Format {
    public:
        class ImplBase;

#if MEGDNN_CC_HOST
        Format();

        const ImplBase* impl() const { return m_impl; }

        enum class Type;

        //! get impl type; defined in tensor_format.h
        inline Type type() const;

        //! convert to the implementation class; exception would be raised if
        //! type mismatches
        template <class Impl>
        const Impl& as_impl() const {
            static_assert(std::is_base_of<ImplBase, Impl>::value, "bad type");
            if (type() != Impl::TYPE) {
                on_bad_cvt(Impl::TYPE);
            }
            return *static_cast<const Impl*>(m_impl);
        }

        //! get human-readable string description of this format
        std::string to_string() const;

        std::string serialize() const;
        static Format deserialize(const std::string& bin, const Handle* handle);

        //! whether this is the default tensor format
        bool is_default() const;

        bool operator==(Format rhs) const { return m_impl == rhs.m_impl; }
        bool operator!=(Format rhs) const { return m_impl != rhs.m_impl; }
#endif

    private:
        const ImplBase* m_impl;

#if MEGDNN_CC_HOST
        Format(ImplBase* impl) : m_impl{impl} {}
        MEGDNN_NORETURN void on_bad_cvt(Type dst_type) const;
#endif
    };

    ptrdiff_t stride[MAX_NDIM];
    DType dtype;
    Format format;

#if MEGDNN_CC_HOST
    TensorLayout();

    TensorLayout(const TensorLayout& layout) = default;

    //! create empty layout with given dtype
    explicit TensorLayout(DType dtype_);

    TensorLayout(DType dtype_, Format format);

    //! create layout with given shape and contiguous stride.
    TensorLayout(const TensorShape& shape, DType dtype);

    TensorLayout(const TensorShape& shape, DType dtype, Format format);

    //! creating layout with user-specified shape and stride.
    TensorLayout(const TensorShape& shape, const std::vector<ptrdiff_t>& stride,
                 DType dtype);

    TensorLayout(const TensorShape& shape, const std::vector<ptrdiff_t>& stride,
                 DType dtype, Format format);

    /* =================== inplace modifiers =================== */

    /*!
     * \brief init stride to be contiguous
     *
     * Use current shape and format
     *
     * \return total number of elements
     */
    size_t init_contiguous_stride();

    /*!
     * \brief init stride to be contiguous by first assigning shape
     *
     * Use current format.
     */
    size_t init_contiguous_stride(const TensorShape& shape);

    size_t init_contiguous_stride(const TensorShape& shape, Format format);

    /*!
     * \brief inplace version of remove_axis
     */
    void remove_axis_inplace(size_t idx);

    /*!
     * \brief add an axis before given *axis* with given shape and stride
     *
     * Other shapes and strides would not be changed.
     */
    void add_axis_inplace(size_t axis, size_t shape, ptrdiff_t stride);

    /*!
     * \brief add an axis before given *axis*, with shape 1 and contiguous
     *      stride
     */
    void add_axis_cont_inplace(size_t axis) {
        add_axis_inplace(axis, 1, stride[axis] * shape[axis]);
    }

    /* =================== generate new layout =================== */

    /**
     * \brief Returns the layout with permuted dimensions.
     *
     * example:
     *  (2, 0, 1) -> AxBxC to CxAxB
     */
    TensorLayout dimshuffle(const std::vector<size_t>& dims) const
            MEGDNN_WARN_UNUSED_RESULT;

    /**
     * \brief Remove an axis from the layout by moving later shape/stride
     *      elements earlier. No extra check is performed.
     */
    TensorLayout remove_axis(size_t idx) const MEGDNN_WARN_UNUSED_RESULT;

    /**
     * \brief Returns a different view.
     *
     * \throw TensorReshapeError if no stride exists for target shape.
     */
    TensorLayout reshape(const TensorShape& shape) const
            MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief try to reshape to another view; return whether these two shapes
     *      are compatible
     * \return true iff there exists target stride so this layout can be
     *      converted to target shape and the elements can match.
     */
    bool try_reshape(TensorLayout& output,
                     const TensorShape& shape) const MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief Broadcast on dims with shape == 1 to match target *shape*.
     * \throw TensorReshapeError if could not be satisfied
     */
    TensorLayout broadcast(const TensorShape& shape) const
            MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief Collapse consecutive axes with contiguous layout together
     *
     * This transforms the tensor into a canonized form. For empty tensors or
     * scalar, the result would always be a one-dimensional empty or scalar,
     * with stride being 1.
     */
    TensorLayout collapse_contiguous() const MEGDNN_WARN_UNUSED_RESULT;

    /* =================== properties =================== */

    std::string to_string() const;
#endif  // MEGDNN_CC_HOST

    /*!
     * \brief check whether the is contiguous under its format definition
     *
     * See is_contiguous_spec() in Format impl classes for more detail. When the
     * format is default, this is equivalent to is_physical_contiguous().
     *
     * Note that empty tensors (i.e. with 0 shapes) are not considered as
     * contiguous.
     */
    bool is_contiguous() const;

    //! check whether it is physically contiguous disregarding format
    bool is_physical_contiguous() const;

    /*!
     * \brief check whether the layout is monotonous
     *
     * A tensor is monotonous if abs(stride[i]) >= abs(stride[i+1])*shape[i+1]
     */
    bool is_abs_monotonous_allow_brdcst() const;

    /*!
     * \brief check whether the layout is contiguous, allowing broadcasting
     *
     * This checks whether the underlying storage is contiguous, where
     * broadcasting is also considered to be so.
     */
    bool is_contiguous_allow_brdcst() const;

    /*!
     * \brief if this function returns true, then no two elements can occupy the
     *      same memory slot
     *
     * Note that this test is a sufficient but not necessary condition for the
     * layout being non-overlapping: when this function returns false, it is
     * still possible that actually no two elements share the same memory
     * location.
     */
    bool is_non_overlapping_strong() const;

    bool eq_layout(const TensorLayout& rhs) const;

    //! get lowest and highest offset reachable from this layout
    Span span() const;
};

/**
 * \brief A simple encapsulation class for n-dimensional tensor.
 */
struct TensorND {
    void* raw_ptr;
    TensorLayout layout;

    TensorND() : raw_ptr(NULL) {}

    TensorND(void* raw_ptr_, const TensorLayout& layout_)
            : raw_ptr(raw_ptr_), layout(layout_) {}

    //! get typed pointer; type check is performed
    template <typename T>
    T* ptr() const {
        layout.dtype.assert_is_ctype<T>();
        return static_cast<T*>(raw_ptr);
    }

    //! get typed pointer of compatible type
    template <typename T>
    T* compatible_ptr() const {
        layout.dtype.assert_is_compatible_ctype<T>();
        return reinterpret_cast<T*>(raw_ptr);
    }
};

#if MEGDNN_CC_HOST
using TensorFormat = TensorLayout::Format;
using TensorShapeArray = SmallVector<TensorShape>;
using TensorNDArray = SmallVector<TensorND>;
using TensorLayoutArray = SmallVector<TensorLayout>;
using TensorLayoutPtrArray = SmallVector<TensorLayout*>;
using TensorFormatArray = SmallVector<TensorFormat>;
#endif

/**
 * \brief A struct representing workspace.
 *
 * It differs from TensorND in that workspace does not have a "layout" concept.
 */
struct Workspace {
    dt_byte* raw_ptr;
    size_t size;

    Workspace() : raw_ptr(NULL), size(0) {}

    Workspace(dt_byte* raw_ptr_, size_t size_)
            : raw_ptr(raw_ptr_), size(size_) {}

    template <typename T>
    T* ptr(size_t offset_in_bytes = 0) const {
        return static_cast<T*>(static_cast<void*>(raw_ptr + offset_in_bytes));
    }
};

#if MEGDNN_CC_HOST

/*!
 * \brief manage output and workspace memory for dynamic output oprs
 */
class DynOutMallocPolicy {
protected:
    ~DynOutMallocPolicy() = default;

public:
    /*!
     * \brief allocate an output var
     * \param id output index, starting from 0
     * \param dtype requested output data type
     * \param shape requested output shape
     * \param user_data extra user data passed in DynOutMallocPolicyCall
     */
    virtual TensorND alloc_output(size_t id, DType dtype,
                                  const TensorShape& shape,
                                  void* user_data) = 0;

    /*!
     * \brief allocate workspace memory
     * \param sz requested workspace in bytes
     */
    virtual void* alloc_workspace(size_t sz, void* user_data) = 0;

    /*!
     * \brief free workspace memory
     *
     * Every operator should guarantee that alloc_workspace() and
     * free_workspace() calls are matched
     */
    virtual void free_workspace(void* ptr, void* user_data) = 0;
};

/*!
 * \brief bind a DynOutMallocPolicy with arbitrary user data
 */
struct DynOutMallocPolicyCall {
    DynOutMallocPolicy* policy;
    void* user_data;

    DynOutMallocPolicyCall(DynOutMallocPolicy* p = nullptr, void* ud = nullptr)
            : policy{p}, user_data{ud} {}

    TensorND alloc_output(size_t id, DType dtype, const TensorShape& shape) {
        return policy->alloc_output(id, dtype, shape, user_data);
    }

    /*!
     * \brief allocate workspace with return type conversion
     * \tparam elem element type for size calculation
     * \param nr_elem number of elements; allocated size is sizeof(elem) *
     *      nr_elem
     */
    template <typename T = void, typename elem = T>
    T* alloc_workspace(size_t nr_elem) {
        using real_elem =
                typename std::conditional<std::is_same<elem, void>::value,
                                          uint8_t, elem>::type;
        return static_cast<T*>(policy->alloc_workspace(
                nr_elem * sizeof(real_elem), user_data));
    }

    void free_workspace(void* ptr) {
        return policy->free_workspace(ptr, user_data);
    }
};

#endif  // MEGDNN_CC_HOST

}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
