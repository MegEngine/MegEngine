#pragma once

#include "common_enum_c.h"
#include "macro.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace lite {

/**
 * @struct Layout
 *
 * @brief Description of the way of data organized in a tensor
 */
struct LITE_API Layout {
    static constexpr uint32_t MAXDIM = 7;               ///< max dims
    size_t shapes[MAXDIM];                              ///< shape of each dim
    size_t ndim = 0;                                    ///< actual number of dims
    LiteDataType data_type = LiteDataType::LITE_FLOAT;  ///< date type

    /**
     * @brief get the size of byte of an element of this layout
     *
     * @return size of byte of an element
     */
    size_t get_elem_size() const;

    /**
     * @brief compare equality of two layouts
     *
     * @param[in] other other layout
     *
     * @return result of comparation
     * - true this layout is equal to other
     * - flase this layout is not equal to other
     */
    bool operator==(const Layout& other) const;
};

/**
 * @brief warpper of the MegEngine Tensor
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * Some more things here.
 *
 * .. note::
 *
 *    * If the tensor memory is set through :cpp:func:`~reset()` interface, the memory
 *      is managed by the user, it will not be freed by the tensor;
 *    * If the ``device_type`` or ``layout`` is not set, when copy form other source
 *      tensor, its device and layout will be copy form the source tensor;
 *    * If ``is_pinned_host`` is set, the storage memory of the tensor is pinned memory,
 *      this is used to Optimize the H2D or D2H memory copy, if the device or layout
 *      is not set, when copy form other device(CUDA) tensor, this tensor
 *      will be automatically set to pinned tensor.
 *
 * .. warning::
 *
 *    The memory is not alloc directly, when call :cpp:func:`get_memory_ptr()` the
 *    memory will be allocated in tensor implement, which will be deleted automatically.
 *
 * \endverbatim
 */
class LITE_API Tensor {
    class TensorImpl;

public:
    class TensorImplBase;

    /*!
     * @name Constructor
     *
     * @param[in] device_type The desired device type of created Tensor.
     * - LITE_CPU CPU Tensor
     * - LITE_CUDA CUDA Tensor
     * - LITE_OPENCL OpenCL Tensor
     * - LITE_ATLAS Atlas Tensor
     * - LITE_NPU NPU Tensor
     * - LITE_CAMBRICON Cambricon Tensor
     * - LITE_AX AX Tensor
     * - LITE_DEVICE_DEFAULT Tensor on default device
     *
     * @param[in] device_id The desired device id of created Tensor.
     *
     * @param[in] stream_id The desired stream id of created Tensor on disired device
     *
     * @param[in] backend desired backend of created Tensor.
     * - LITE_DEFAULT backend is MegEngine
     * - LITE_RK_NPU backend is RKNN NPU
     *
     * @param[in] is_pinned_host Whether to use pinned memory.
     * - false use nornal memory
     * - true use pinned memory[main on CUDA]
     *
     * @param[in] layout The desired layout of created Tensor.
     *
     */
    //@{

    //! Default constructor
    Tensor();

    //! Constructor
    Tensor(LiteDeviceType device_type, bool is_pinned_host = false);

    //! Constructor
    Tensor(LiteDeviceType device_type, const Layout& layout,
           bool is_pinned_host = false);

    //! Constructor
    Tensor(int device_id, LiteDeviceType device_type, const Layout& layout = {},
           bool is_pinned_host = false);

    //! Constructor
    Tensor(int device_id, int stream_id, LiteDeviceType device_type,
           bool is_pinned_host = false);

    //! Constructor
    Tensor(LiteBackend backend, LiteDeviceType device_type = LiteDeviceType::LITE_CPU,
           int device_id = 0, const Layout& layout = {}, bool is_pinned_host = false);
    //@}

    //! Deconstructor
    ~Tensor();

    /**
     * @brief Get device type of this Tensor
     *
     * @return device type
     * - LITE_CPU CPU Tensor
     * - LITE_CUDA CUDA Tensor
     * - LITE_OPENCL OpenCL Tensor
     * - LITE_ATLAS Atlas Tensor
     * - LITE_NPU NPU Tensor
     * - LITE_CAMBRICON Cambricon Tensor
     * - LITE_AX AX Tensor
     * - LITE_DEVICE_DEFAULT Tensor on default device
     */
    LiteDeviceType get_device_type() const { return m_device_type; };

    //! Get device id of this Tensor
    int get_device_id() const { return m_device_id; };

    //! Get layout of this Tensor
    Layout get_layout() const { return m_layout; };

    /**
     * @brief whether Tensor is on pinned memory
     *
     * @return whether Tensor is on pinned memory
     * - false nornal memory
     * - true pinned memory
     */
    bool is_pinned_host() const { return m_is_pinned_host; };

    /**
     * @brief Get memory address of data of this Tensor
     *
     * @return address pointer
     *
     * @note this function will trigger memory alloc in tensor implement
     */
    void* get_memory_ptr() const;

    /**
     * @brief Get the memory with the offset describe in idx of this Tensor
     *
     * @param[in] idx indeces of tensor
     *
     * @return address pointer
     */
    void* get_memory_ptr(const std::vector<size_t>& idx) const;

    //! Get capacity of the Tenosr in bytes
    size_t get_tensor_total_size_in_byte() const;

    //! Check whether the memory of tensor is contigous
    bool is_continue_memory() const;

    /**
     * @brief set layout to this Tensor
     *
     * @param[in] layout layout that will set into this Tensor
     *
     * @note this will change the layout and reallocate memory of the tensor
     */
    void set_layout(const Layout& layout);

    /**
     * @brief reset layout with user alloced memory
     *
     * @param[in] prepared_data user prepared data pointer
     *
     * @param[in] data_length_in_byte size of this memory
     *
     * @note the memory will not be managed by the lite, later, the user should delete it
     */
    void reset(void* prepared_data, size_t data_length_in_byte);

    /**
     * @brief reset layout with user alloced memory and corresponding layout
     *
     * @param[in] prepared_data user prepared data pointer
     *
     * @param[in] layout desired layout
     *
     * @note the memory will not be managed by the lite, later, the user should delete it
     */
    void reset(void* prepared_data, const Layout& layout);

    /**
     * @brief reshape the tensor with new shape
     *
     * @param[in] shape target shape
     *
     * @note the data type will keep unchanged
     */
    void reshape(const std::vector<int>& shape);

    /**
     * @brief get a slice from the origin tensor
     *
     * @param[in] start start idx of each dim
     *
     * @param[in] end end idx of each dim
     *
     * @param[in] step step of each dim
     *
     * @return ref pointer of a new Tensor
     *
     * @note if tensor = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], start = {0, 0}, end = {2,
     * 2}, step = {1, 2}. Then result = [[1, 3], [4, 6], [7, 9]]
     */
    std::shared_ptr<Tensor> slice(
            const std::vector<size_t>& start, const std::vector<size_t>& end,
            const std::vector<size_t>& step = {});

    //! memset Tensor with zero
    void fill_zero();

    /**
     * @brief copy data from another tensor
     *
     * @param[in] src source tensor
     *
     * @note the best way for tensor copy is just set the dst device left layout empty.
     * Layout will be set the same as src when copying
     */
    void copy_from(const Tensor& src);

    //! share memory with other tensor
    void share_memory_with(const Tensor& src_tensor);

    //! update the menbers from the implement
    void update_from_implement();

public:
    friend class TensorHelper;

private:
    std::shared_ptr<TensorImplBase> m_tensor_impl;  ///< tensor implementation.
    bool m_is_pinned_host =
            false;  ///< flag whether the storage of the tensor is pinned, this is only
                    ///< used when the compnode is not in CPU.
    int m_device_id = 0;  ///< device id of this Tensor.
    Layout m_layout;      ///< layout of this Tensor.
    LiteDeviceType m_device_type =
            LiteDeviceType::LITE_CPU;  ///< devie type of this Tensor. should not change
                                       ///< after constructing.
};

/**
 * @class LiteAny
 *
 * @brief a class can hold any type data
 *
 * @note the visit type is valide will not be checked
 */
class LITE_API LiteAny {
public:
    /**
     * @enum Type
     *
     * @brief enum for data type
     */
    enum Type {
        STRING = 0,
        INT32 = 1,
        UINT32 = 2,
        UINT8 = 3,
        INT8 = 4,
        INT64 = 5,
        UINT64 = 6,
        BOOL = 7,
        VOID_PTR = 8,
        FLOAT = 9,
        NONE_SUPPORT = 10,
    };

    /**
     * @class HolderBase
     *
     * @brief Base class for holding any type of data
     */
    class HolderBase {
    public:
        /**
         * @brief virtual deconstructor
         */
        virtual ~HolderBase() = default;

        /**
         * @brief clone data
         *
         * @return a new ref pointer of the data
         *
         * @note pure virtual interface
         */
        virtual std::shared_ptr<HolderBase> clone() = 0;
    };

    /**
     * @class AnyHolder
     *
     * @brief template class that holds any type of data
     */
    template <class T>
    class AnyHolder : public HolderBase {
    public:
        /**
         * @brief default constructor
         */
        AnyHolder(const T value) : m_value(value) {}

        /**
         * @brief clone data of this holder
         *
         * @return a ref pointer of m_value
         */
        virtual std::shared_ptr<HolderBase> clone() override {
            return std::make_shared<AnyHolder>(m_value);
        }

    public:
        T m_value;  ///< value
    };

    /**
     * @brief default constructor
     */
    LiteAny() = default;

    /**
     * @brief constructor with value of any type
     *
     * @param[in] value data
     */
    template <class T>
    LiteAny(T value) : m_holder(new AnyHolder<T>(value)) {
        m_type = get_type<T>();
    }

    /**
     * @brief copy constructor
     *
     * @param[in] any data
     */
    LiteAny(const LiteAny& any) {
        m_holder = any.m_holder->clone();
        m_type = any.m_type;
    }

    /**
     * @brief assign operator overloading
     *
     * @param[in] any data
     */
    LiteAny& operator=(const LiteAny& any) {
        m_holder = any.m_holder->clone();
        m_type = any.m_type;
        return *this;
    }

    /**
     * @brief get data type of this hold
     *
     * @return type of data
     * - STRING
     * - INT32
     * - UINT32
     * - UINT8
     * - INT8
     * - INT64
     * - UINT64
     * - BOOL
     * - VOID_PTR
     * - FLOAT
     * - NONE_SUPPORT
     */
    template <class T>
    Type get_type() const;

    /**
     * @brief check whether type mismatch
     *
     * @param[in] expect expected type
     *
     * @param[in] get got type
     *
     * @note if type is miss matching, it will throw
     */
    void type_missmatch(size_t expect, size_t get) const;

    /**
     * @brief cast with type safty
     *
     * @return casted type
     *
     * @note if type is miss matching, it will throw
     */
    template <class T>
    T safe_cast() const {
        if (get_type<T>() != m_type) {
            type_missmatch(m_type, get_type<T>());
        }
        return static_cast<LiteAny::AnyHolder<T>*>(m_holder.get())->m_value;
    }

    /**
     * @brief check whether can cast to one kind of type
     *
     * @return successful or not
     * - true successful
     * - false failed
     */
    template <class T>
    bool try_cast() const {
        if (get_type<T>() == m_type) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief unsafe cast to void*
     *
     * @return pointer to hold data
     *
     * @note only check the storage type and the visit type length, so it's not safe
     */
    void* cast_void_ptr() const {
        return &static_cast<LiteAny::AnyHolder<char>*>(m_holder.get())->m_value;
    }

private:
    std::shared_ptr<HolderBase> m_holder;  ///< holder member
    Type m_type = NONE_SUPPORT;            ///< type member
};

/**
 * @class TensorUtils
 *
 * @brief provide special tensor tool functions
 */
class LITE_API TensorUtils {
public:

    /**
     * @brief concat all the input tensor to one on the specified dim.
     *
     * @param[in] tensors input tensors
     *
     * @param[in] dim specified dim
     *
     * @param[in] dst_device type of output tensor
     *
     * @param[in] dst_device_id id of output tensor
     *
     * @return concated tensor
     *
     * @note the result tensor reside in dst_device_id of dst_device, if dst_device is
     * LITE_DEVICE_DEFAULT, the device will get from the first tensor
     */
    static std::shared_ptr<Tensor> concat(
            const std::vector<Tensor>& tensors, int dim,
            LiteDeviceType dst_device = LiteDeviceType::LITE_DEVICE_DEFAULT,
            int dst_device_id = -1);
};
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
